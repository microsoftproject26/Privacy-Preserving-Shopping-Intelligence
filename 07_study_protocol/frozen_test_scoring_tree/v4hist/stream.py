"""v4hist.stream (owner: Data/integration engineer, DATA-05) — HistoryStore, the streaming reference path under a
history policy.

A HistoryStore is the unmodified CleanV2Store (over the snapshot CompactStore) of the policy's store partitions, plus
the RowIndex proven aligned to it. Its decision sets are the streaming path's (VALIDATION: the two splits merged in
ascending decision_id, exactly v4int.cache.merged_decision_set), under the policy's window lengths
(v4hist.windows.policy_decision_set). Batches are CleanV2Store.batch -> CompactStore.batch -> clean_v2_view. Nothing
is cached or written. This is the reference the manifest-scoped cache (v4int.cache) is proven bitwise equal to.

A HistoryStore may also be built over MORE (later, non-TEST) partitions than the policy needs: its store must hold at
least store_splits(policy, decision splits). The proofs use this to show that a window does not depend on which
later partitions the store happens to hold.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence

import ppsi_env  # noqa: F401  (isolation first)

import numpy as np

from v4data.common import dataset_root
from v4data.store import CleanV2Store

from .policy import NON_TEST_ORDER, require_policy, store_splits
from .policy import USER_HISTORY_L50
from .windows import (RowIndex, add_session_features, align_to_store, build_row_index, policy_decision_set,
                      window_rows)


class HistoryStore:
    def __init__(self, store: CleanV2Store, index: RowIndex, policy, decision_splits: Sequence[str]):
        self.policy = require_policy(policy)
        self.decision_splits = tuple(decision_splits)
        have = set(store.cs.splits)
        need = set(store_splits(self.policy, self.decision_splits))
        if not need <= have <= set(NON_TEST_ORDER):
            raise ValueError(f"store partitions {sorted(have)} must include {sorted(need)} and be non-TEST")
        self.store = store
        self.index = index
        self.alignment = align_to_store(index, store.cs)

    @classmethod
    def build(cls, policy, decision_splits: Sequence[str], *, buckets: Sequence[int], ds_root: Optional[Path] = None,
              state_sha256: Optional[str] = None, n_threads: int = 3, splits: Optional[Sequence[str]] = None,
              log: Callable = print) -> "HistoryStore":
        """`splits` (optional) = the store partitions; default = store_splits(policy, decision_splits)."""
        p = require_policy(policy)
        ss = tuple(splits) if splits is not None else store_splits(p, decision_splits)
        root = Path(ds_root) if ds_root is not None else dataset_root()
        st = CleanV2Store.build(splits=ss, buckets=list(buckets), ds_root=root, state_sha256=state_sha256,
                                n_threads=n_threads, log=log)
        idx = build_row_index(root, st.cs.buckets, ss)
        return cls(st, idx, p, decision_splits)

    def raw_decision_set(self, kind: str):
        from v4int.cache import merged_decision_set          # the streaming path's own merge (INT-01)
        return merged_decision_set(self.store, self.decision_splits, kind)

    def decision_set(self, kind: str):
        return policy_decision_set(self.raw_decision_set(kind), self.index, self.policy)

    def positions_for(self, ds, decision_ids) -> np.ndarray:
        return self.store.positions_for(ds, decision_ids)

    def batch(self, ds, positions) -> dict:
        b = self.store.batch(ds, positions)
        if self.policy == USER_HISTORY_L50:                  # DECISION_LOG 47 + 48 (index coordinates)
            pos = np.asarray(positions, dtype=np.int64)
            add_session_features(b, self.index.session_id, ds.end_row[pos], ds.length[pos])
        return b

    def windows(self, ds, positions) -> tuple:
        pos = np.asarray(positions, dtype=np.int64)
        return window_rows(ds.end_row[pos], ds.length[pos])


__all__ = ["HistoryStore"]
