# PREG1 amendment 02R1 — supersedes §1 of PREG1_AMENDMENT_02 (P2 design). §2–§6 of 02 stand.

- **Status:** REGISTERED by the Operations Director on 2026-09-27, before any P2 launch. Authority: owner directive
  DECISION_LOG 60 (no legacy anchoring; model size in the search; 6 EFE; complete experiments on both families) and
  Fable's follow-up ruling `company/reviews/BEST_BACKBONE_FABLE_FOLLOWUP_SIZE.md` (DECISION_LOG 61).
- **Withdrawn from 02 §1:** the "as executed" grid centres, the 2-EFE rung, KEEP_REGISTERED and the +0.0010 adoption
  margin, and the "fewer deviations from the registered recipe" tie rule. PREG1_AMENDMENT_02.md is kept unchanged as
  the record.
- **Changes after the first P2 launch are refused** (a new amendment and a Fable call are needed).

## 1. Model variants (MODEL_VARIANT_REGISTRY, TRAIN-02)
GRU_D128 (current build), GRU_D256 (item embedding 256, input_proj→256, LayerNorm(256), nn.GRU(256, 256), readout→256,
tied head); SASREC_D256_B2 (current build), SASREC_D256_B3 (one more causal block), SASREC_D384_B2 (4 heads, head dim
96, ff ×4, untied, C7 recentering). θ0 per variant × seeds 2026 / 2027 in a NEW THETA0_VARIANTS.json; the existing
derived-schema REGISTRY.json and THETA0_REGISTRY.json stay byte-identical.

## 2. Design (identical procedure for both families; seed 2026; full 6.0 EFE; CALIB only)
- Data and settings: train CALIB_TRAIN_5_100_32768, evaluate CALIB_VALIDATION_5_100_32768 (MAIN_VALIDATION never read
  in P2); S*, USER_HISTORY_L50, STRICT, B 256, exposure_point end, frozen T3 evaluator, EVAL_ROWS_ONLY; client-macro
  MRR@20 (E2E); evaluations at 2 / 4 / 6 EFE; the endpoint read at 6.0 EFE only; weights-only checkpoints at 5.0 / 5.5
  / 6.0 for CKPT_AVG.
- Fixed: AdamW (0.9, 0.999), eps 1e-8, weight decay 1e-5 (ndim ≥ 2), clip 1.0, label smoothing 0, 6.0 EFE.
- Discrete factors (full factorial):
  - GRU: d {128, 256} × schedule {S0, S1} × dropout p {0.1, 0.2, 0.3} = 12 cells, one trial each (input p, readout p/2);
  - SASRec: size {D256_B2, D256_B3, D384_B2} × schedule {S0, S1} = 6 cells × 2 trials = 12; dropout p {0.1, 0.2, 0.3}
    rotated so that each level appears 4 times and every (size, p) pair at least once (single p).
- Schedules on the 6-EFE clock: S0 = ((0,0),(0.02,1),(2.0,1),(3.0,0.1),(5.5,0.1),(6.0,0.01)); S1 (WSD, linear) =
  ((0,0),(0.02,1),(4.2,1),(6.0,0.01)).
- LR: one 12-stratum Latin hypercube on log10(peak LR) over [log10 3e-4, log10 3e-3]; one value per stratum at the
  stratum midpoint; the stratum-to-trial permutation comes from numpy.random.default_rng(20260927) with the same
  procedure for both families. The HP registry records the drawn table and its sha256 before the first launch.
- Stage 2: the best trial (6.0 value) of EACH size level is re-run at seed 2027 (GRU 2 runs, SASRec 3 runs).
  Total 14 GRU + 15 SASRec training runs; all recorded in the HP ledger; no hidden trials.

## 3. Selection per family
- Tuned = the higher 2-seed mean at 6.0 EFE among the stage-2 configs. Tie |Δ| ≤ 0.0005 → the smaller model (fewer
  parameters), then the lower LR.
- Exclusions: a non-finite loss or metric, or a 6.0 value < 0.5× the family's best, is excluded; a size level with no
  valid 6.0 run is a Fable trigger.
- CKPT_AVG (the uniform average of the 5.0 / 5.5 / 6.0 checkpoints) is evaluated on the stage-2 configs and carried to P4
  iff it beats the endpoint by > 0.0005 on the 2-seed mean.
- Rows: role TUNING, label HP_SEARCH, refused by v4train.selection. The tuned recipes (variant, LR, dropout, schedule)
  are appended as amendment 02b before the P3 launch.

## 4. Preconditions and triggers
- Before P2: SYS-04c synthetic STRICT bench of GRU_D256, SASREC_D256_B3 and SASREC_D384_B2 (2 reps each). Trigger if a
  variant runs below 0.5× the assumed rate (GRU_D256 12 upd/s; SASREC_D384_B2 6.5 upd/s) or peaks above 8 GiB on the A16.
- Admissibility proxy (no registered cap): visit bytes ≤ 2 × 665 MB and per-client AdamW state ≤ 4 GiB. The winner's
  payload and a SYS-02b FL forecast are registered before ARCHITECTURE_LOCK; if the downstream core exceeds 1.5× the
  current forecast, Ops brings reduction options (methods, rounds, seeds) to Fable — never a smaller backbone.
- Cost (INFERENCE): ≈ 13 GPU-h, ≈ 9 h wall on two A16 (one family per GPU).
