# PREG1 amendment 02 — registration of the equal-effort tuning (P2), polish (P4) and baselines. PREG1_LOCK_VALUES.json unchanged.

- **Status:** REGISTERED by the Operations Director on 2026-09-27, per Fable's BEST_BACKBONE ruling
  (`company/reviews/BEST_BACKBONE_FABLE_POSITION.md`, APPROVE_WITH_AMENDMENTS, Q4) with the owner override
  (DECISION_LOG 57–58: LEGACY_REPLAY dropped). Written BEFORE the first P2 launch; any change after the first P2 launch
  is refused (a new amendment and a Fable call are needed).
- **Scope:** registration only. PREG1 T1 thresholds, T2 exposure point, T3 numerics, the 6.0-EFE CONTROLLED_BUDGET,
  LOCK_V2, REGISTRY_V2, S*, USER_HISTORY_L50 and STRICT are unchanged.
- **Tuned recipes** are appended later as amendment 02b (before the P3 contest launch).

## 1. P2 — equal-effort tuning (identical for GRU and SASRec)
- Data: train CALIB_TRAIN_5_100_32768 (N_EFE 646,067; 2,524 updates/EFE at B 256), evaluate CALIB_VALIDATION_5_100_32768
  only (phase CALIBRATION). MAIN_VALIDATION is never read in P2. S*, USER_HISTORY_L50, STRICT, B 256, exposure_point end,
  frozen T3 evaluator, EVAL_ROWS_ONLY, client-macro MRR@20 (E2E). Endpoint reads only (never best-of-marks).
- Factors:
  - F1 peak LR ∈ {0.5, 1.0, 2.0} × executed peak 1e-3 (source `g0/receipts_res02/EXECUTED_RECIPES.md`, sha256
    01ae1bd448ebfd206776f2df6b77142cf3bc31c0afa075e2201e4aa2fd0023b4, GRU line 16, SASRec line 56);
  - F2 dropout ∈ {as executed (GRU 0.20 input / 0.10 readout; SASRec 0.2), as executed + 0.10 on every dropout site};
  - F3 schedule on the 6-EFE clock ∈ {S0 = registered KNOTS ((0,0),(0.02,1),(2.0,1),(3.0,0.1),(5.5,0.1),(6.0,0.01));
    S1 = WSD ((0,0),(0.02,1),(4.2,1),(6.0,0.01))}. S0 ≡ S1 on [0, 2].
- Excluded (Fable Q2): weight decay changes, label smoothing, any shape change (dimensions, blocks, tying).
- Rungs:
  - R1: the 6 configs F1 × F2, 2.0 EFE, seed 2026. Advance the registered control (1.0×, as executed) plus the top 2 other
    configs by the 2.0-EFE value.
  - R2: the 3 advanced configs × F3 = 6 runs, 6.0 EFE, seed 2026; rank by the 6.0-EFE value.
  - R3: the top 2 of R2 at seed 2027, 6.0 EFE. CKPT_AVG (the uniform parameter average of the 5.0 / 5.5 / 6.0 checkpoints)
    is evaluated on both R3 configs at both seeds.
  - 14 training runs per family. Every run (including excluded ones) is recorded in the HP registry ledger; no hidden
    trials.
- Selection per family: tuned = the higher 2-seed mean at 6.0 EFE among the R3 configs; tie |Δ| ≤ 0.0005 → the config with
  fewer deviations from the registered recipe. ADOPT the tuned recipe only if its seed-2026 R2 value exceeds the registered
  control's seed-2026 R2 value (control = 1.0×, as executed, S0) by > 0.0010; otherwise KEEP_REGISTERED.
- CKPT_AVG is carried to P4 iff it beats the endpoint by > 0.0005 (2-seed mean, same config).
- Exclusions (G1 §D): a run with a non-finite loss or metric, or a value < 0.5× the best of its rung, is excluded; if both
  R3 configs are excluded → Fable trigger.
- Rows: role TUNING, run label HP_SEARCH, refused by v4train.selection; never used by G2.

## 2. P3 contest (unchanged rules; tuned recipes from 02b)
MAIN, seeds 2026 + 2027, 6.0 EFE, marks every 0.5 EFE, G2 mechanical per T1 (DECISIVE_STABLE only);
ARCHITECTURE_EQUIVALENT → HOLD (owner-pending GRU-on-cost); continuation 6 → 8 only on STILL_IMPROVING, its rows never
selected. Every MAIN_VALIDATION number is also reported on the 38,496 CALIB-disjoint MAIN_VALIDATION users (diagnostic,
never selection); overlap disclosed (12,715 of 51,211 = 24.8%).

## 3. P4 polish (evaluation only)
- C_BEST (per family and seed) = the higher MAIN_VALIDATION value among {6.0 endpoint, continuation endpoint if run,
  CKPT_AVG if carried}. Labelled composite; never the contest row, never the FL recipe.
- SYSTEM = the 2-seed log-probability ensemble of the family's contest runs. Never the backbone.

## 4. Baselines (label BASELINE; CPU; never selection)
Popularity (MAIN TRAIN counts), last-item repeat, session item-kNN on MAIN TRAIN events. Their own settings (kNN k and
sample size) are tuned on CALIB_VALIDATION during P2 with a registered grid in the baseline task; they read
MAIN_VALIDATION only from P3 on, and TEST only at ALL_METHODS_LOCK. Same catalogue A (K 158,486), same eval rows, the
frozen T3 ranking/metric path (per-row candidate scores densified with -inf for unscored items).
Resource rule (Fable amended ruling): low priority, <= 16 GiB RSS, cores left after the trainers; if it breaches 16 GiB
or lowers a trainer's step rate by > 5%, Ops stops it and runs it after P3 (no Fable call).

## 5. P5 claim
One TEST look at ALL_METHODS_LOCK: contest singles, C_BEST, SYSTEM, BASELINE; paired user bootstrap (1,000 resamples,
seed 20260926). No comparison with v1 models or published REES46 numbers is claimed (DL 57). The tuning ledger is
published with the result.

## 6. Tie-break wording (Fable G1 finding 3; registration only)
PREG1_LOCK_VALUES T1 names M02 / 262,144 users and the full-TRAIN tie-break (2,694,998 / 642,228). Under ADR-005 R2 D9
these read as: the MAIN eval role of LOCK_V2 (MAIN_VALIDATION_5_100_131072) and the TB band (r < 1,348,682). Thresholds
unchanged. No tie-break manifest exists; a tie-break launch is a Fable trigger.
