# PREG1 amendment 06M — early warm-start preview on CALIB (S7 / S8 preview; descriptive only) (registration only)

- **Status:** REGISTERED 2026-09-30 ~08:15Z (server clock) by the Operations Director, on the owner's instruction ("federated starts
  from a random model; train the model first, as real systems do"). Before any FA_WARM / FA_WARM_FROZEN run exists or is read.
  θ_pre = PRE endpoint_6.0 (sha256 cf08588c…), init record `runs_pretrain/PRETRAIN_FOR_FL_SMALL_SASREC_D64_B2_s2026/
  PRETRAIN_FOR_FL_INIT.json` (sha256 c8d644a6…, write-once).
- **Why now:** 06K runs S7 / S8 at the S4 tuning point, about a day away. The warm start is the largest expected lever (research §1.2
  cause 5), and GPU capacity is free now. A preview at the current best point (S2 leader) shows the size of the effect early.

## M1. Preview runs (CALIB, seed 2026, full 6.0 EFE, same protocol as every screen)
- Tuning point = the S2 leader: server FedAdam η_s 0.01, τ 1e-3, m 1, central LR shape, fresh client AdamW, P 2 / B 16.
- FA_WARM (from θ_pre): warm factor f ∈ {1, 0.25} (the 06K S7 factors).
- FA_WARM_FROZEN (from θ_pre, item tables frozen, not uploaded): f ∈ {1, 4} (the 06K S8 factors).
- Run dirs `runs_ds3/WARM_PREVIEW/*`; code tree code_ds3r4.

## M2. Status of the values
- DESCRIPTIVE only. They never enter select-joint, select-warm, FL_TUNING_SELECTION or FL_WARM_SELECTION_S7|S8, and never change
  a registered rule. The registered S7 / S8 screens (06K) still run at the S4 point and alone decide the MAIN warm runs.
- They are reported in the ladder register as "S7/S8 preview at the S2 point", next to S2 FedAdam (cold start) — the same point, so
  the difference is the warm start alone.
- If the S2 file (after the τ edge) names a different winner, the preview stays at the point above and is labelled so.
