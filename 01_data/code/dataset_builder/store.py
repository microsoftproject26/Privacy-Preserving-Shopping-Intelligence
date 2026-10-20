"""v4data.store (owner: Data Lead, DATA-04) - Clean V2 batches over the snapshot CompactStore.

* `CleanV2Store` wraps `snap_v1/reuse/full_store.CompactStore` (channels F4, byte-identical snapshot, unmodified).
  It never loads TEST, never writes anything to disk (no per-decision cache), and yields bounded batches: each
  batch is a gather of at most `batch_size` windows, re-encoded by `transform.clean_v2_view` on the fly.
* `ClientView` is the client-scoped reader of CLIENT_CONTRACT_V1 C5.1 / PKG/12 T23: constructed from the M01
  manifest and ONE client_key_hex, it can read only that client's contiguous decisions row range; any request
  outside it raises OwnershipError.

Import `ppsi_env` before this module (isolation, T43); the snapshot modules are imported from there.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional, Sequence

import numpy as np

from .common import dataset_root, refuse_test_split
from .transform import V1_VIEW_TAG, CleanV2Spec, build_spec, clean_v2_view, input_schema_sha256_of


class OwnershipError(PermissionError):
    """A client-scoped reader was asked for a row it does not own (T23)."""


def _runtime_input_schema_sha256() -> str:
    """The v1-rich layout the vendored runtime loader actually produces (not the schema's claim)."""
    from rees46_eda.config import EVENT_QUALITY_FLAGS, USER_CONTEXT_FEATURES, USER_CONTEXT_MASKS
    from rees46_eda.loaders import EVENT_NUMERIC_BLOCK
    return input_schema_sha256_of(EVENT_NUMERIC_BLOCK, EVENT_QUALITY_FLAGS, USER_CONTEXT_FEATURES,
                                  USER_CONTEXT_MASKS)


class CleanV2Store:
    def __init__(self, compact_store, spec: CleanV2Spec, state_sha256: Optional[str] = None):
        if getattr(compact_store, "channels", None) != "F4":
            raise ValueError("Clean V2 needs the rich (F4) CompactStore")
        for s in compact_store.splits:
            refuse_test_split(s)
        self.cs = compact_store
        self.spec = spec
        self.state_sha256 = state_sha256
        self.runtime_input_schema_sha256 = _runtime_input_schema_sha256()
        if self.runtime_input_schema_sha256 != spec.input_schema_sha256:
            raise ValueError("the vendored loader's v1 column layout differs from the frozen schema's")

    @classmethod
    def build(cls, *, splits: Sequence[str] = ("TRAIN",), buckets: Optional[Sequence[int]] = None,
              ds_root: Optional[Path] = None, state_sha256: Optional[str] = None, n_threads: int = 3,
              log=print) -> "CleanV2Store":
        for s in splits:
            refuse_test_split(s)
        from full_store import CompactStore  # snapshot copy, via ppsi_env
        cs = CompactStore.build(Path(ds_root) if ds_root else dataset_root(), splits=tuple(splits), channels="F4",
                                buckets=list(buckets) if buckets is not None else None, n_threads=n_threads, log=log)
        return cls(cs, build_spec(), state_sha256)

    # -- decisions ----------------------------------------------------------------------------------
    def decision_set(self, split: str, kind: str):
        """DecisionSet over the store's retained decisions of `split`, `kind` in {loss, eval}, WITHOUT the legacy
        client_idx-based eval_user_key (V4 keys come from the manifests; EVALUATION_CONTRACT_V4 E3.5)."""
        refuse_test_split(split)
        from full_store import DecisionSet
        arr = self.cs._split_decisions[split]
        mask = arr["loss_mask"] if kind == "loss" else arr["eval_mask"]
        idx = np.flatnonzero(mask)
        order = np.argsort(arr["decision_id"][idx], kind="stable")
        sel = idx[order]
        did = arr["decision_id"][sel].astype(np.int64)
        if did.size > 1 and not np.all(np.diff(did) > 0):
            raise AssertionError("decision_id not strictly ascending")
        return DecisionSet(split=split, kind=kind, decision_id=did, end_row=arr["global_row"][sel].astype(np.int64),
                           length=arr["length"][sel], target_class=arr["target_class"][sel].astype(np.int64),
                           loss_mask=arr["loss_mask"][sel].astype(bool), eval_mask=arr["eval_mask"][sel].astype(bool),
                           client_idx=arr["client_idx"][sel].astype(np.int64),
                           user_bucket=arr["user_bucket"][sel].astype(np.int16), eval_user_key=None,
                           prior_view_count=arr["prior_view_count"][sel], prefix_position=arr["prefix_position"][sel])

    @staticmethod
    def positions_for(ds, decision_ids: np.ndarray) -> np.ndarray:
        d = np.asarray(decision_ids, dtype=np.int64)
        pos = np.searchsorted(ds.decision_id, d)
        pos_c = np.clip(pos, 0, max(len(ds) - 1, 0))
        if len(ds) == 0 or not np.array_equal(ds.decision_id[pos_c], d):
            raise KeyError("some requested decision_id is not a retained decision of this store")
        return pos_c

    # -- batches ------------------------------------------------------------------------------------
    def v1_batch(self, ds, positions) -> dict:
        b = self.cs.batch(ds, positions)
        b["feature_view"] = V1_VIEW_TAG
        b["v1_input_schema_sha256"] = self.runtime_input_schema_sha256
        return b

    def batch(self, ds, positions) -> dict:
        return clean_v2_view(self.v1_batch(ds, positions), self.spec, encoded_with="V1_NORMALIZERS",
                             state_sha256=self.state_sha256)

    def iter_batches(self, ds, positions: Optional[np.ndarray] = None, batch_size: int = 256) -> Iterator[dict]:
        """Streaming, bounded: at most `batch_size` windows materialised at a time; nothing is cached or written."""
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        pos = np.arange(len(ds), dtype=np.int64) if positions is None else np.asarray(positions, dtype=np.int64)
        for start in range(0, pos.shape[0], batch_size):
            yield self.batch(ds, pos[start:start + batch_size])

    def client_view(self, clients, decisions, client_key_hex: str) -> "ClientView":
        return ClientView(self, clients, decisions, client_key_hex)


class ClientView:
    """Reads only one client's M01 rows (T23). `clients`/`decisions` are the M01 tables (pyarrow or dict of arrays)."""

    def __init__(self, store: Optional[CleanV2Store], clients, decisions, client_key_hex: str):
        keys = _col(clients, "client_key_hex")
        hits = np.flatnonzero(keys == client_key_hex)
        if hits.size != 1:
            raise OwnershipError(f"client key present {hits.size} times in the clients table (expected 1)")
        i = int(hits[0])
        self.client_key_hex = client_key_hex
        self.client_ord = int(_col(clients, "client_ord")[i])
        self.row_start = int(_col(clients, "row_start")[i])
        self.row_count = int(_col(clients, "row_count")[i])
        self._store = store
        # only this client's contiguous slice is ever taken from the decisions table
        part = _slice(decisions, self.row_start, self.row_count, ("decision_id", "client_ord"))
        self._decision_id = np.asarray(_col(part, "decision_id"), dtype=np.int64)
        own = np.asarray(_col(part, "client_ord"), dtype=np.int64)
        if own.size != self.row_count or not np.all(own == self.client_ord):
            raise OwnershipError("the client's declared row range contains another client's rows")
        self._owned = set(self._decision_id.tolist())

    def decision_ids(self) -> np.ndarray:
        return self._decision_id.copy()

    def check_owned(self, decision_ids) -> None:
        bad = [int(d) for d in np.asarray(decision_ids, dtype=np.int64).tolist() if int(d) not in self._owned]
        if bad:
            raise OwnershipError(f"client {self.client_ord} requested {len(bad)} decision(s) it does not own")

    def batches(self, ds, batch_size: int = 16, decision_ids=None) -> Iterator[dict]:
        ids = self._decision_id if decision_ids is None else np.asarray(decision_ids, dtype=np.int64)
        self.check_owned(ids)
        if self._store is None:
            raise RuntimeError("this ClientView has no store attached")
        pos = self._store.positions_for(ds, ids)
        yield from self._store.iter_batches(ds, pos, batch_size)


def _col(tbl, name):
    if isinstance(tbl, dict):
        return np.asarray(tbl[name])
    return tbl[name].to_numpy(zero_copy_only=False)


def _slice(tbl, start: int, count: int, cols: tuple):
    """Only rows [start, start+count) of the named columns of a decisions table (pyarrow zero-copy slice, or a
    dict of arrays)."""
    if count < 0 or start < 0:
        raise OwnershipError("negative row range")
    n = tbl.num_rows if hasattr(tbl, "num_rows") else len(np.asarray(tbl["decision_id"]))
    if start + count > n:
        raise OwnershipError("client row range runs past the decisions table")
    if isinstance(tbl, dict):
        return {k: np.asarray(tbl[k])[start:start + count] for k in cols}
    return tbl.select(list(cols)).slice(start, count)
