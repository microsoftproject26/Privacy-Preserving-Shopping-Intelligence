"""v4train.provider — the data protocols the central trainer programs against, and in-memory synthetic providers.

BatchProvider (training; DATA-04's `v4data.store` will implement it over M01 / M05):
    manifest_id     the registry's train manifest id of the run's role (e.g. HISTORICAL M01_COHORT_TRAIN / M05)
    split           "TRAIN" (anything else is refused)
    fingerprint     a sha256 identity of the provider's data (e.g. the manifest's ordered_decision_id_sha256 bound to
                    the Clean V2 transform-state sha256); recorded in every checkpoint and checked on resume
    n_decisions     N = the number of unique loss-eligible decisions (loss_mask AND target_class >= 0); EFE = exposures / N
    decision_ids()  int64 [N], the canonical order (strictly ascending decision_id)
    batch(pos)      a dict of CPU tensors for the canonical positions `pos`, in that order, in the Clean V2 layout
                    (the FEATURE_SCHEMA_CLEAN_V2 widths) with at least decision_id, target_class and loss_mask; every
                    row must be loss-eligible (the trainer refuses a batch that is not)
EvalProvider (evaluation; the same store over M02 / M06, or the full warm manifest for the tie-break):
    manifest        a v4eval EvalManifest (built through v4eval.manifest.build_eval_manifest, so the TEST gate ran)
    fingerprint     sha256 identity
    batches(bs)     the manifest's rows in manifest order — either every row (v4eval stream ALL_ROWS) or only its E2E
                    rows (E2E_ROWS; censored rows need no model input) — in batches of EXACTLY `bs` rows except an
                    unpadded tail (PREG1 T3: bs = the frozen eval_batch, 1,024), each with `decision_id` (and
                    optionally target_class, which v4eval checks against the manifest)
The order policy lives in the trainer (v4train.order); a provider only answers random-access requests.
Registry binding (TRAIN-01R2; checked by the trainer, deliberately NOT a Protocol member so isinstance checks of
existing providers are unchanged): both providers must also expose
    manifest_binding  {"manifest_id", "content_sha256", "header_sha256"} — the hashes of the manifest files the
                      provider verified, with the manifest registry's meaning (v4train.registry); the trainer compares
                      them, and the row count, with the registry entry of the run's role and refuses a mismatch.
No manifest id is hard-coded here any more: `expected_manifests` reads a manifest-role registry (by default the
HISTORICAL population's file, v4train/registries/REGISTRY_CLEAN_V2_HISTORICAL_262144.json).
"""
from __future__ import annotations

import hashlib
import json
from typing import Iterable, Iterator, Mapping, Optional, Protocol, runtime_checkable

import numpy as np
import torch

from v4eval.hashing import ordered_decision_id_sha256
from v4eval.manifest import EvalManifest, build_eval_manifest
from v4models.schema import InputWidths
from v4models.synthetic import synthetic_examples_v2, take

def expected_manifests(population: str, phase: str, registry=None, role: Optional[str] = None) -> tuple:
    """(train, eval) manifest ids of a role (explicit `role`, else derived from population and phase) in a
    manifest-role registry: a ManifestRegistry, a path, or None for the HISTORICAL population's registry file.
    Raises RegistryError for an unregistered role."""
    from .registry import HISTORICAL_REGISTRY_PATH, ManifestRegistry, load_manifest_registry, role_of
    if population == "FULL_TRAIN" and phase != "MAIN":
        raise ValueError("the full-TRAIN tie-break has only a main phase")
    reg = registry if isinstance(registry, ManifestRegistry) else \
        load_manifest_registry(registry if registry is not None else HISTORICAL_REGISTRY_PATH)
    return reg.role(role_of(population, phase, role))


def _sha_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False, default=str).encode("utf-8")).hexdigest()


@runtime_checkable
class BatchProvider(Protocol):
    manifest_id: str
    split: str
    fingerprint: str

    @property
    def n_decisions(self) -> int: ...
    def decision_ids(self) -> np.ndarray: ...
    def batch(self, positions: np.ndarray) -> Mapping[str, torch.Tensor]: ...


@runtime_checkable
class EvalProvider(Protocol):
    manifest: EvalManifest
    fingerprint: str

    def batches(self, batch_size: int) -> Iterable[Mapping]: ...


def _repeat_last_targets(ex: dict, K: int) -> torch.Tensor:
    """A learnable synthetic target: the class of the window's last real item (item token j + 3 <-> class j)."""
    last = ex["item_tokens"].gather(1, (ex["lengths"] - 1).clamp(min=0).view(-1, 1)).squeeze(1)
    return (last - 3).clamp(0, K - 1)


class SyntheticTrainProvider:
    """In-memory synthetic TRAIN decisions in the Clean V2 layout (tests only; no dataset row is read)."""

    split = "TRAIN"

    def __init__(self, N: int, K: int, widths: InputWidths, *, seed: int = 0, max_len: int = 8, min_len: int = 1,
                 manifest_id: str = "M01_COHORT_TRAIN", learnable: bool = True, vocab: Optional[dict] = None,
                 decision_id_start: int = 1000, decision_id_step: int = 3):
        g = torch.Generator().manual_seed(int(seed))
        ex = synthetic_examples_v2(int(N), int(K), widths, g, vocab=vocab, max_len=max_len, min_len=min_len,
                                   window=max_len)
        if learnable:
            ex["target_class"] = _repeat_last_targets(ex, K)
        ex["decision_id"] = torch.arange(decision_id_start, decision_id_start + decision_id_step * N,
                                         decision_id_step, dtype=torch.int64)
        self._ex = ex
        self.K = int(K)
        self.manifest_id = manifest_id
        self.access_log: list = []                 # decision ids served, in order (order-policy tests)
        self.fingerprint = hashlib.sha256(
            (manifest_id + ordered_decision_id_sha256(ex["decision_id"].numpy())
             + hashlib.sha256(ex["target_class"].numpy().tobytes()).hexdigest()).encode()).hexdigest()
        # synthetic stand-ins for the verified manifest file hashes (registry binding; TRAIN-01R2)
        self.manifest_binding = {
            "manifest_id": manifest_id,
            "content_sha256": _sha_json([ordered_decision_id_sha256(ex["decision_id"].numpy()),
                                         hashlib.sha256(ex["target_class"].numpy().tobytes()).hexdigest()]),
            "header_sha256": _sha_json({"manifest_id": manifest_id, "N": int(N), "K": int(K), "kind": "TRAIN"})}

    @property
    def n_decisions(self) -> int:
        return int(self._ex["decision_id"].shape[0])

    def decision_ids(self) -> np.ndarray:
        return self._ex["decision_id"].numpy().copy()

    def batch(self, positions) -> dict:
        idx = torch.as_tensor(np.asarray(positions, dtype=np.int64))
        b = take(self._ex, idx)
        self.access_log.extend(b["decision_id"].numpy().tolist())
        return b


class SyntheticEvalProvider:
    """In-memory synthetic VALIDATION decisions + a v4eval manifest (with OOV and censored rows; tests only)."""

    def __init__(self, n: int, K: int, widths: InputWidths, *, seed: int = 1, n_users: Optional[int] = None,
                 manifest_id: str = "M02_VALIDATION_WARM", max_len: int = 8, oov_every: int = 17,
                 censor_every: int = 13, header_extra: Optional[dict] = None, vocab: Optional[dict] = None):
        g = torch.Generator().manual_seed(int(seed))
        ex = synthetic_examples_v2(int(n), int(K), widths, g, vocab=vocab, max_len=max_len, window=max_len)
        tgt = _repeat_last_targets(ex, K).numpy().astype(np.int64)
        i = np.arange(n)
        censored = (i % censor_every) == censor_every - 1
        oov = ((i % oov_every) == oov_every - 1) & ~censored
        eval_mask = ~censored
        tgt = np.where(eval_mask & ~oov, tgt, -1)
        n_users = n_users or max(1, n // 3)
        users = (i * 7919) % n_users
        did = np.arange(50_000, 50_000 + 2 * n, 2, dtype=np.int64)
        cols = {"decision_id": did, "eval_user_key": users.astype(np.int64), "eval_mask": eval_mask,
                "target_class": tgt, "target_oov": oov,
                "censor_class": np.where(censored, "CEN_NO_NEXT", "NONE"),
                "v4_split": np.full(n, "VALIDATION")}
        header = {"manifest_id": manifest_id, "catalogue_K": int(K), "candidate_order_hash": "SYNTHETIC-CANDIDATES"}
        header.update(header_extra or {})
        self.manifest = build_eval_manifest(header, cols)
        ex["decision_id"] = torch.as_tensor(did)
        ex["target_class"] = torch.as_tensor(tgt)
        self._ex = ex
        self.fingerprint = hashlib.sha256((manifest_id + self.manifest.hashes["ordered_decision_id_sha256"])
                                          .encode()).hexdigest()
        self.manifest_binding = {
            "manifest_id": manifest_id,
            "content_sha256": _sha_json([self.manifest.hashes, hashlib.sha256(tgt.tobytes()).hexdigest()]),
            "header_sha256": _sha_json(header)}

    def batches(self, batch_size: int) -> Iterator[dict]:
        n = self.manifest.n
        for s in range(0, n, int(batch_size)):
            yield take(self._ex, torch.arange(s, min(n, s + int(batch_size))))


__all__ = ["BatchProvider", "EvalProvider", "SyntheticTrainProvider", "SyntheticEvalProvider", "expected_manifests"]
