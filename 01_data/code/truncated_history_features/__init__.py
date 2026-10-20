"""v4trunc (DATA-07) — the E11.4 truncated-history feature builder and its providers (EVALUATION_CONTRACT_V4 §11).

  rules      the per-channel recompute table (locality class, rule, PARTIAL channels) — pure data
  events     per-user event tables read from the frozen v1 feature files (never TEST) + the v1 replay carry-in
  recompute  the v1 formulas of every LOCAL_HISTORY channel, applied to a virtual history (pure numpy)
  builder    H_h(q) + e_q -> v1 raw rows -> the UNMODIFIED production encode / batch path (Clean V2 + DATA-05 keys)
  panel      COLD_H / WARM_TRUNCATION panel manifests: verification, sub-panel evaluation manifests, support labels
  provider   Study A evaluation batches and Study B support batches over the builder
  golden     T41 golden replay (h = full history == production, bitwise) and the E11.7 controls

Import `ppsi_env` (or v4train) before this package: the snapshot CompactStore / loaders come from code/snap_v1.
"""
