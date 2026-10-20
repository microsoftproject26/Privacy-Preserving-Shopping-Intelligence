"""v4hist — the input-history policy of every decision (task DATA-05; owner decision DECISION_LOG 35-36).

  policy    the REQUIRED history-policy enum (USER_HISTORY_L50 locked, SESSION_L50 the v1 diagnostic), import-free so
            the launcher can parse its arguments before torch / the snapshot are imported
  windows   the window resolver over the snapshot CompactStore rows: the per-row user index (proven chronological and
            aligned to the store), the policy window lengths, the policy DecisionSet, and the leakage / structure
            checks used by the proofs
  stream    HistoryStore, the streaming reference path under a policy (the cache == stream proofs)
  census    window-length statistics and the 5-100 pool membership (census_v2 definitions, read-only)

This package imports nothing at package import time (same rule as v4int).
"""
__version__ = "DATA-05.1"
