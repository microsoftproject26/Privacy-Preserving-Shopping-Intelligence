"""The one prediction path every method uses, the prediction artifact, and alignment.

`predict_adapter` drives any model adapter that exposes `module`, `query(batch)`, `head_weight()` and `head_bias()`
(the protocol of the federated simulator's adapters) under an `EvalConfig`: for each batch it computes the query
once, q = adapter.query(batch) (+ personal_fn(batch) for a personalised model, exactly as the adapter's scores add
the personal vector), and ranks the RANKABLE rows against the full head in one score block of
score_block >= K, i.e. the model's logits op for op (ranking.block_scores). The model is put in eval mode under
no_grad and every RNG state is restored afterwards.

The evaluation stream is the manifest's rows in manifest order, in one of two forms, recorded as eval_stream:
ALL_ROWS (every manifest row) or E2E_ROWS (only the eval_mask rows: censored rows need no model input). Censored rows
always keep rank -1. Because a batch's composition can change the query bits, eval_stream is part of the comparison
key. Batching: consecutive rows in strictly ascending decision_id, each batch exactly eval_batch rows except an
unpadded tail (a short batch followed by another batch is refused). The runtime flags must equal the configured
values, and the device class is recorded in the artifact.

`_research_predict_adapter` exposes other score blocks for float-tolerance studies. Its artifacts carry
frozen = False and evaluate() refuses them: they can never become a result row.

Alignment: batches carry `decision_id`; a batch that carries `target_class` must agree with the manifest; the artifact
holds one exact int64 rank per manifest row (-1 for rows that are not RANKABLE), is bound to the manifest's ordered
hashes, and `align` refuses any mismatch. Per-row tie diagnostic: n_tied and eq_before per ranked row, so
pessimistic ranks = ranks - eq_before + n_tied.
"""
from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np
import torch

from .config import EvalConfig, check_runtime_flags, eval_device_class, resolve
from .errors import AlignmentError, EvalConfigError
from .manifest import EvalManifest
from .ranking import DEFAULT_CACHE_BYTES, rank_from_query, rank_from_scores

PATH_ADAPTER = "adapter_stream"
PATH_SCORES = "precomputed_scores"


@dataclass
class PredictionArtifact:
    decision_id: np.ndarray
    ranks: np.ndarray         # int64 per manifest row; -1 where the row is not RANKABLE
    n_tied: np.ndarray        # int64 per row; -1 where not ranked
    eq_before: np.ndarray     # int64 per row; -1 where not ranked
    header: dict = field(default_factory=dict)

    @property
    def pessimistic_ranks(self) -> np.ndarray:
        """Per-row pessimistic ranks (tie diagnostic); -1 where not ranked."""
        out = self.ranks.copy()
        m = self.ranks > 0
        out[m] = self.ranks[m] - self.eq_before[m] + self.n_tied[m]
        return out


def _new_artifact(manifest: EvalManifest, model_sha256: str, cfg: dict) -> PredictionArtifact:
    n = manifest.n
    minus = lambda: np.full(n, -1, dtype=np.int64)
    h = dict(manifest.hashes)
    h.update({"manifest_id": manifest.manifest_id, "model_sha256": str(model_sha256),
              "catalogue_K": manifest.catalogue_K,
              "candidate_order_hash": manifest.header.get("candidate_order_hash"), "eval_config": cfg})
    return PredictionArtifact(decision_id=manifest.decision_id.copy(), ranks=minus(), n_tied=minus(),
                              eq_before=minus(), header=h)


def stream_rows(manifest: EvalManifest) -> np.ndarray:
    """Manifest row indices of the evaluation stream: the E2E (eval_mask) rows, in manifest order."""
    return np.flatnonzero(manifest.eval_mask)


class _Cursor:
    """Consecutive-row bookkeeping for the two stream forms (ALL_ROWS / E2E_ROWS)."""

    def __init__(self, manifest: EvalManifest, eval_batch: int | None):
        self.m = manifest
        self.stream = stream_rows(manifest)
        self.eval_batch = eval_batch
        self.mode: str | None = None
        self.pos_all = 0
        self.pos_e2e = 0
        self.short_seen = False

    def take(self, batch: Mapping) -> np.ndarray:
        if "decision_id" not in batch:
            raise AlignmentError("every batch must carry decision_id")
        d = np.asarray(torch.as_tensor(batch["decision_id"]).cpu().numpy(), dtype=np.int64)
        n_b = int(d.shape[0])
        if n_b == 0:
            raise AlignmentError("empty batch")
        if self.eval_batch is not None:
            if self.short_seen:
                raise EvalConfigError(
                    f"a batch shorter than the configured eval_batch {self.eval_batch} was not the tail")
            if n_b > self.eval_batch:
                raise EvalConfigError(f"batch of {n_b} rows exceeds the configured eval_batch {self.eval_batch}")
            self.short_seen = n_b < self.eval_batch
        m = self.m
        rows = np.searchsorted(m.decision_id, d)
        if m.n == 0 or np.any(rows >= m.n) or not np.array_equal(m.decision_id[np.minimum(rows, m.n - 1)], d):
            raise AlignmentError("batch carries decision_ids that are not manifest rows")
        all_ok = (self.mode in (None, "ALL_ROWS")
                  and np.array_equal(rows, np.arange(self.pos_all, self.pos_all + n_b)))
        e2e_ok = (self.mode in (None, "E2E_ROWS") and self.pos_e2e + n_b <= self.stream.size
                  and np.array_equal(rows, self.stream[self.pos_e2e:self.pos_e2e + n_b]))
        if not (all_ok or e2e_ok):
            raise AlignmentError(f"batch rows are not the next consecutive manifest rows (ALL_ROWS at "
                                 f"{self.pos_all} or E2E_ROWS at {self.pos_e2e})")
        if all_ok and not e2e_ok:
            self.mode = "ALL_ROWS"
        elif e2e_ok and not all_ok:
            self.mode = "E2E_ROWS"
        self.pos_all += n_b
        self.pos_e2e += int(np.count_nonzero(m.eval_mask[rows]))
        if "target_class" in batch:
            t = np.asarray(torch.as_tensor(batch["target_class"]).cpu().numpy(), dtype=np.int64)
            if not np.array_equal(t, m.target_class[rows]):
                raise AlignmentError("batch target_class disagrees with the manifest")
        return rows

    def finish(self) -> str:
        if self.mode is None:           # no censored row was ever fed: complete as E2E if the stream is covered
            self.mode = ("E2E_ROWS" if self.pos_e2e == self.stream.size and self.pos_all == self.stream.size
                         else None)
            if self.pos_all == self.m.n:
                self.mode = self.mode or "ALL_ROWS"
        if self.mode == "ALL_ROWS" and self.pos_all == self.m.n:
            return "ALL_ROWS"
        if self.mode == "E2E_ROWS" and self.pos_e2e == self.stream.size:
            return "E2E_ROWS"
        raise AlignmentError(f"batches covered {self.pos_all} manifest rows / {self.pos_e2e} E2E rows; the "
                             f"manifest has {self.m.n} rows / {self.stream.size} E2E rows")


@torch.no_grad()
def _stream(adapter, batches: Iterable[Mapping], manifest: EvalManifest, art: PredictionArtifact, *,
            personal_fn, score_block: int, class_chunk: int, cache_bytes: int, eval_batch: int | None,
            device) -> PredictionArtifact:
    module = adapter.module
    was_training = module.training
    states = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
              torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    module.eval()
    rank_mask = manifest.rankable
    cur = _Cursor(manifest, eval_batch)
    try:
        W = adapter.head_weight()
        b = adapter.head_bias()
        if int(W.shape[0]) != manifest.catalogue_K:
            raise AlignmentError(f"head has {W.shape[0]} classes, manifest catalogue_K is {manifest.catalogue_K}")
        for batch in batches:
            brows = cur.take(batch)
            sel = np.flatnonzero(rank_mask[brows])
            if sel.size:
                dev = device if device is not None else W.device
                bt = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
                q = adapter.query(bt)
                if personal_fn is not None:
                    q = q + personal_fn(bt)
                idx = torch.as_tensor(sel, dtype=torch.long, device=q.device)
                tgt = torch.as_tensor(manifest.target_class[brows[sel]], dtype=torch.long, device=q.device)
                r = rank_from_query(q.index_select(0, idx), W, b, tgt, score_block=score_block,
                                    class_chunk=class_chunk, K_expected=manifest.catalogue_K,
                                    cache_bytes=cache_bytes)
                rows = brows[sel]
                art.ranks[rows] = r.ranks.cpu().numpy()
                art.n_tied[rows] = r.n_tied.cpu().numpy()
                art.eq_before[rows] = r.eq_before.cpu().numpy()
    finally:
        module.train(was_training)
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.set_rng_state(states[2])
        if states[3] is not None:
            torch.cuda.set_rng_state_all(states[3])
    art.header["eval_config"]["eval_stream"] = cur.finish()
    return art


def predict_adapter(adapter, batches: Iterable[Mapping], manifest: EvalManifest, *, model_sha256: str,
                    config: EvalConfig | None = None,
                    personal_fn: Callable[[Mapping], torch.Tensor] | None = None,
                    device: torch.device | None = None, score_block: int | None = None,
                    class_chunk: int | None = None, cache_bytes: int | None = None,
                    eval_batch: int | None = None) -> PredictionArtifact:
    """The prediction path for every trained model. `config` defaults to EvalConfig(); score_block / class_chunk /
    cache_bytes / eval_batch may be passed only if they EQUAL the configured values."""
    cfg = resolve(config, score_block=score_block, class_chunk=class_chunk, cache_bytes=cache_bytes,
                  eval_batch=eval_batch)
    dev = torch.device(device) if device is not None else adapter.head_weight().device
    check_runtime_flags(cfg, dev)
    if cfg.score_block < manifest.catalogue_K:
        raise EvalConfigError(f"configured score_block {cfg.score_block} < K {manifest.catalogue_K}: not one block")
    eval_cfg = cfg.eval_config(PATH_ADAPTER, eval_device_class(dev))
    art = _new_artifact(manifest, model_sha256, eval_cfg)
    return _stream(adapter, batches, manifest, art, personal_fn=personal_fn, score_block=cfg.score_block,
                   class_chunk=cfg.class_chunk, cache_bytes=cfg.cache_bytes, eval_batch=cfg.eval_batch,
                   device=device)


def _research_predict_adapter(adapter, batches: Iterable[Mapping], manifest: EvalManifest, *, model_sha256: str,
                              score_block: int, class_chunk: int, cache_bytes: int = DEFAULT_CACHE_BYTES,
                              personal_fn=None, device=None) -> PredictionArtifact:
    """NOT a result path: other score blocks for float-tolerance studies. evaluate() refuses its artifacts."""
    cfg = {"frozen": False, "path": "research_adapter_stream", "score_block": int(score_block),
           "class_chunk": int(class_chunk), "cache_bytes": int(cache_bytes)}
    art = _new_artifact(manifest, model_sha256, cfg)
    return _stream(adapter, batches, manifest, art, personal_fn=personal_fn, score_block=score_block,
                   class_chunk=class_chunk, cache_bytes=cache_bytes, eval_batch=None, device=device)


def predict_scores(score_batches: Iterable[tuple], manifest: EvalManifest, *, model_sha256: str,
                   config: EvalConfig | None = None, class_chunk: int | None = None) -> PredictionArtifact:
    """Precomputed full-catalogue score batches (decision_id [B], scores [B, K]); float or int64. For baselines;
    rows of trained models must come from predict_adapter (rows.py refuses this path for them)."""
    cfg = resolve(config, class_chunk=class_chunk)
    art = None
    rank_mask = manifest.rankable
    cur = _Cursor(manifest, cfg.eval_batch)
    for dids, scores in score_batches:
        if art is None:
            check_runtime_flags(cfg, scores.device)
            art = _new_artifact(manifest, model_sha256,
                                cfg.eval_config(PATH_SCORES, eval_device_class(scores.device)))
        brows = cur.take({"decision_id": dids})
        n_b = brows.size
        if int(scores.shape[0]) != n_b:
            raise AlignmentError("score rows differ from the batch decision_ids")
        sel = np.flatnonzero(rank_mask[brows])
        if sel.size:
            idx = torch.as_tensor(sel, dtype=torch.long, device=scores.device)
            tgt = torch.as_tensor(manifest.target_class[brows[sel]], dtype=torch.long, device=scores.device)
            r = rank_from_scores(scores.index_select(0, idx), tgt, class_chunk=cfg.class_chunk,
                                 K_expected=manifest.catalogue_K)
            rows = brows[sel]
            art.ranks[rows] = r.ranks.cpu().numpy()
            art.n_tied[rows] = r.n_tied.cpu().numpy()
            art.eq_before[rows] = r.eq_before.cpu().numpy()
    if art is None and cur.stream.size == 0:
        check_runtime_flags(cfg, torch.device("cpu"))
        art = _new_artifact(manifest, model_sha256,
                            cfg.eval_config(PATH_SCORES, eval_device_class(torch.device("cpu"))))
        cur.mode = "E2E_ROWS"
    if art is None:
        raise AlignmentError("no score batch was given")
    art.header["eval_config"]["eval_stream"] = cur.finish()
    return art


def align(manifest: EvalManifest, art: PredictionArtifact) -> None:
    """Refuse any artifact that is not exactly this manifest's rows, in order, with valid rank semantics."""
    if not np.array_equal(art.decision_id, manifest.decision_id):
        raise AlignmentError("artifact decision_ids differ from the manifest (order or membership)")
    for k, v in manifest.hashes.items():
        if art.header.get(k) != v:
            raise AlignmentError(f"artifact {k} {art.header.get(k)} != manifest {v}")
    if art.header.get("catalogue_K") != manifest.catalogue_K:
        raise AlignmentError("artifact catalogue_K differs from the manifest")
    r = np.asarray(art.ranks, dtype=np.int64)
    rk = manifest.rankable
    if r.shape != (manifest.n,):
        raise AlignmentError("artifact ranks must have one entry per manifest row")
    if bool(np.any(r[~rk] != -1)):
        raise AlignmentError("a non-RANKABLE row carries a rank (OOV and censored rows are never ranked)")
    if bool(np.any((r[rk] < 1) | (r[rk] > manifest.catalogue_K))):
        raise AlignmentError("a RANKABLE row has no exact rank in [1, K]")
