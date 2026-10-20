"""Pilot-cache-backed stand-ins for `CompactStore` / `DecisionSet` /
`eval_full.evaluate_model` -- used ONLY by `test_train_full.py` (Sonnet B's
own test suite) until Opus swaps in the real modules (owners: Sonnet A
`full_store.py`, Sonnet C `eval_full.py`; see `FULL_INTERFACES.md`).

`train_full.py` never imports this module in its production code path (that
path lazily imports the REAL `full_store`/`eval_full`); it is only reached
when `--test-tiny` is passed.

Backing data: the pilot's existing `$W/cache/train` (1,000,000 rows, the
pilot TRAIN manifest, all `loss_mask=True`) and `$W/cache/vs` (112,365 rows,
the pilot VALIDATION_SELECT 10k-user cohort) window caches
(`window_cache.CacheReader`, frozen pilot module, imported not edited), plus
`eval_user_key` joined positionally from
`$W/artifacts/validation_select_cohort.parquet` (identical row order to
`cache/vs`, exactly as `train.py:eval_user_key_array()` already assumes).

Interface-fidelity notes (documented, not hidden -- see sonnetB_trainer.md):
  * `end_row`, `user_bucket`, `prior_view_count`, `prefix_position` are NOT
    reconstructable from the pilot window cache (it stores post-window
    tensors, not raw store row numbers or bucket ids). They are filled with
    harmless placeholders (`end_row`=cache row index, `user_bucket`=0,
    `prior_view_count`=0, `prefix_position`=length). Nothing in
    `train_full.py` or `fake_evaluate_model` reads them for anything other
    than structural completeness of the `DecisionSet`-shaped object; the
    real `full_store.py` populates them properly.
  * `fake_evaluate_model` calls the pilot's own `evaluator.evaluate()` on
    CPU (mirroring `train.py:run_vs_evaluation`), not the real (GPU,
    class-chunked) `eval_full.evaluate_model`. It reproduces the same
    contract-visible side effects (model.eval()+no_grad, RNG save/restore,
    `model.training` restore) and the same result shape
    (`result["end_to_end"]["macro"]["mrr@20"]` etc.), which is all
    `train_full.py`'s selection logic consumes.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from window_cache import CacheReader  # frozen pilot module, imported not edited
from evaluator import evaluate as pilot_evaluate  # frozen pilot module
import train as pilot_train  # frozen pilot module (build_named_kwargs, VERSION_HASH)

__all__ = ["FakeDecisionSet", "FakeStore", "fake_evaluate_model"]


# ---------------------------------------------------------------------------
# DecisionSet stand-in
# ---------------------------------------------------------------------------

@dataclass
class FakeDecisionSet:
    split: str
    kind: str
    decision_id: np.ndarray
    end_row: np.ndarray
    length: np.ndarray
    target_class: np.ndarray
    loss_mask: np.ndarray
    eval_mask: np.ndarray
    client_idx: np.ndarray
    user_bucket: np.ndarray
    eval_user_key: Optional[np.ndarray]
    prior_view_count: np.ndarray
    prefix_position: np.ndarray

    def __len__(self) -> int:
        return int(self.decision_id.shape[0])

    def take(self, positions: np.ndarray) -> "FakeDecisionSet":
        positions = np.asarray(positions, dtype=np.int64)

        def sub(a):
            return a[positions] if a is not None else None

        return FakeDecisionSet(
            split=self.split, kind=self.kind,
            decision_id=sub(self.decision_id), end_row=sub(self.end_row),
            length=sub(self.length), target_class=sub(self.target_class),
            loss_mask=sub(self.loss_mask), eval_mask=sub(self.eval_mask),
            client_idx=sub(self.client_idx), user_bucket=sub(self.user_bucket),
            eval_user_key=sub(self.eval_user_key),
            prior_view_count=sub(self.prior_view_count),
            prefix_position=sub(self.prefix_position),
        )

    def sha256(self) -> str:
        return hashlib.sha256(
            np.ascontiguousarray(self.decision_id).astype(np.int64).tobytes()
        ).hexdigest()

    def with_decision_id(self, decision_id: np.ndarray) -> "FakeDecisionSet":
        """Test-only helper: return a copy with `decision_id` overridden (used
        to inject a deliberate TRAIN/VS overlap for the leakage-assert test)."""
        return replace(self, decision_id=np.asarray(decision_id, dtype=np.int64))


F0_KEEP_KEYS = {
    "item_tokens", "attention_mask", "position_ids", "lengths", "last_valid_index",
    "target_class", "loss_mask", "eval_mask", "decision_id", "client_id",
}


def _load_eval_user_key(cohort_parquet: Path, idx: np.ndarray) -> np.ndarray:
    import pyarrow.parquet as pq
    t = pq.read_table(cohort_parquet, columns=["eval_user_key"])
    arr = t["eval_user_key"].to_numpy(zero_copy_only=False)
    return np.ascontiguousarray(arr[idx]).astype(np.int64)


def _decision_set_from_cache(split: str, kind: str, idx: np.ndarray, batch: dict,
                              eval_user_key: Optional[np.ndarray]) -> FakeDecisionSet:
    n = int(idx.shape[0])
    return FakeDecisionSet(
        split=split, kind=kind,
        decision_id=batch["decision_id"].numpy().astype(np.int64),
        end_row=idx.astype(np.int64),
        length=batch["lengths"].numpy().astype(np.int32),
        target_class=batch["target_class"].numpy().astype(np.int64),
        loss_mask=batch["loss_mask"].numpy().astype(bool),
        eval_mask=batch["eval_mask"].numpy().astype(bool),
        client_idx=batch["client_id"].numpy().astype(np.int64),
        user_bucket=np.zeros(n, dtype=np.int16),
        eval_user_key=eval_user_key,
        prior_view_count=np.zeros(n, dtype=np.int32),
        prefix_position=batch["lengths"].numpy().astype(np.int32),
    )


class FakeStore:
    """Pilot-cache-backed stand-in for `full_store.CompactStore`."""

    def __init__(self, train_reader: CacheReader, vs_reader: CacheReader,
                 ds_train: FakeDecisionSet, ds_vs: FakeDecisionSet,
                 channels: str, splits: tuple):
        self._train_reader = train_reader
        self._vs_reader = vs_reader
        self._ds = {("TRAIN", "loss"): ds_train, ("VALIDATION_SELECT", "eval"): ds_vs}
        self.channels = channels
        self.splits = splits
        self.n_rows = len(ds_train) + len(ds_vs)
        self.version_hash = pilot_train.VERSION_HASH
        self.context_length = 50

    @classmethod
    def build(cls, *, cache_train_dir: Path, cache_vs_dir: Path, cohort_parquet: Path,
              channels: str = "F0", n_train: int = 4096, n_vs: int = 2048) -> "FakeStore":
        train_reader = CacheReader(Path(cache_train_dir))
        vs_reader = CacheReader(Path(cache_vs_dir))
        n_train = min(n_train, train_reader.header["row_count"])
        n_vs = min(n_vs, vs_reader.header["row_count"])

        train_idx = np.arange(n_train, dtype=np.int64)
        vs_idx = np.arange(n_vs, dtype=np.int64)

        train_batch = train_reader.read(train_idx)
        ds_train = _decision_set_from_cache("TRAIN", "loss", train_idx, train_batch, None)

        vs_batch = vs_reader.read(vs_idx)
        eval_user_key = _load_eval_user_key(Path(cohort_parquet), vs_idx)
        ds_vs = _decision_set_from_cache("VALIDATION_SELECT", "eval", vs_idx, vs_batch, eval_user_key)

        return cls(train_reader, vs_reader, ds_train, ds_vs, channels, ("TRAIN", "VALIDATION_SELECT"))

    def decisions(self, split: str, kind: str) -> FakeDecisionSet:
        return self._ds[(split, kind)]

    def batch(self, ds: FakeDecisionSet, positions) -> dict:
        positions = np.asarray(positions, dtype=np.int64)
        reader = self._train_reader if ds.split == "TRAIN" else self._vs_reader
        raw = reader.read(positions)
        if self.channels == "F0":
            return {k: v for k, v in raw.items() if k in F0_KEEP_KEYS}
        return dict(raw)

    def decision_item_tokens(self, ds: FakeDecisionSet) -> np.ndarray:
        reader = self._train_reader if ds.split == "TRAIN" else self._vs_reader
        idx = np.arange(len(ds), dtype=np.int64)
        raw = reader.read(idx)
        lengths = raw["lengths"].numpy()
        item_tokens = raw["item_tokens"].numpy()
        return item_tokens[np.arange(len(ds)), lengths - 1].astype(np.int64)


# ---------------------------------------------------------------------------
# evaluate_model stand-in (CPU, via the pilot's own evaluator.evaluate)
# ---------------------------------------------------------------------------

def fake_evaluate_model(model, store: FakeStore, ds: FakeDecisionSet, device,
                         *, eval_batch: int = 1024, k: int = 20,
                         return_ranks: bool = False) -> dict:
    was_training = model.training
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    model.eval()
    n = len(ds)
    rows = {
        "target_class": torch.as_tensor(ds.target_class, dtype=torch.long),
        "eval_mask": torch.as_tensor(ds.eval_mask, dtype=torch.bool),
        "eval_user_key": torch.as_tensor(ds.eval_user_key, dtype=torch.long),
        "client_id": torch.as_tensor(ds.client_idx, dtype=torch.long),
    }

    def scores_fn():
        with torch.no_grad():
            for start in range(0, n, eval_batch):
                end = min(start + eval_batch, n)
                positions = np.arange(start, end, dtype=np.int64)
                batch = store.batch(ds, positions)
                batch_gpu = {kk: vv.to(device, non_blocking=True) for kk, vv in batch.items()}
                kw = pilot_train.build_named_kwargs(batch_gpu, model)
                logits = model(**kw)
                yield logits.detach().to(torch.float32).cpu()

    result = pilot_evaluate(scores_fn, rows, k=k, chunk_size=8192)

    if was_training:
        model.train()
    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)

    if return_ranks:
        result = dict(result)
        result["ranks"] = None  # not reconstructed by this fake
    return result
