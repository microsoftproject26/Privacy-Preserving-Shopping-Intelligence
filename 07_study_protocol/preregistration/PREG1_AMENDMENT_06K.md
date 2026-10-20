# PREG1 amendment 06K — ladder stages S7 (warm start from a pretrained model, FA_WARM) and S8 (frozen item tables) (registration only)

- **Status:** REGISTERED 2026-09-29 by the Operations Director (Claude Opus 5.5, session eda-7b) — the chief-scientist session
  assigned 06K to Ops (DL 202) — on the owner's decisions ("do what you see right to get the strongest realistic FL"; FA_WARM
  approved). Research basis: `company/reviews/FL_STRONGEST_LADDER_RESEARCH_20260929.md` §2.6, §2.7, §3 (S7, S8). Registered BEFORE
  the pretraining run exists and before any FA_WARM / S8 value exists or is read. Builds on 06F–06J; nothing in them changes.

## 1. PRE — the pretraining model (deployment branch, Gboard-style)
- Data: `PRETRAIN_TRAIN_5_100_NONMAIN` (DATA-10) = the registered TIEBREAK band minus MAIN = ranks [131,072, 1,348,682),
  **1,217,610 users / 24,089,824 TRAIN decisions**, disjoint from MAIN, CALIB, U_NEW, FS and every panel (band hash 716beeb9…).
  Interpretation: users who opted in to share data (or public data of the same store); MAIN users' data never leaves their devices.
- Model/recipe: SASREC_D64_B2 with the registered SMALL recipe (SMALL_SASREC_D64_B2, peak 0.000711, S0, dropout), seed 2026,
  6.0 EFE via `v4int/pretrain_small.py` (unchanged launch_central); marks on UNEW_VAL_VALIDATION are DESCRIPTIVE (run label
  PRETRAIN_FOR_FL; refused by every selection path). **θ_pre = the endpoint_6.0 checkpoint** (no checkpoint selection; automatic
  continuation, if it triggers, does not change θ_pre) recorded in `PRETRAIN_FOR_FL_INIT.json` with its sha256.
- Co-scheduling: the central launcher runs PRE with `--allow-shared-gpu --coschedule-06g` (06G §5; flag recorded in the launch record).

## 2. S7 — FA_WARM
- FA exactly as the FL tuning selected by the ladder up to S4 (`contracts/fl/FL_TUNING_SELECTION.json`), except the initial global
  model = θ_pre instead of θ0. CALIB screen of the local-LR multiplier at warm start: m ∈ {m*, m*/4} (m* = the selected multiplier;
  a warm start usually needs a smaller step); rule = 06I §3 / 06J selection rule restricted to these two runs (tie → m*). MAIN: seed
  2026 and 2027 (the deployment-branch headline candidate, reported beside FA; K-contrast added: **K20 = R(FA_WARM / C_BIG)** and
  **K21 = FA_WARM − PRE** (paired Δ: the value of federated training on the private users over the pretrained model alone), 2 seeds,
  PRACTICAL_BEST, secondary).
- References always reported beside S7: **PRE-alone** (θ_pre evaluated on MAIN_VALIDATION by reeval, no training on MAIN users) and
  **FT_PRE** (θ_pre + the registered FT_C local fine-tune procedure with the SELECTION_FT_C recipe), DESCRIPTIVE.
- Never warm-start from C_FULL or C_BIG (they contain MAIN users' data centrally).

## 3. S8 — FA_WARM_FROZEN (realism branch)
- As S7, but the two item tables (input item embeddings and the output layer; ≈ 98 % of parameters) are FROZEN at θ_pre and NOT
  transmitted; only the remaining parameters (the sequence encoder, ≈ 0.4M) are trained locally and uploaded. Bytes per visit and
  the DP noise are recomputed for the uploaded parameter count (fl.comm). Disclosed: removes the input-table item-set leak of
  non-DP deltas; the frozen tables come from the opted-in cohort. CALIB screen m ∈ {m*, 4·m*} (fewer parameters tolerate larger
  steps); MAIN seed 2026 (2027 if time). DESCRIPTIVE unless it beats S7 on CALIB by ≥ 0.002, in which case it is also run with 2 seeds.
- A DP variant of S8 (DP8_FROZEN) is NICE tier, registered here with the same z/q/T/δ and S re-calibrated on the S8 trajectory.

## 4. Thesis reporting
The FL improvement ladder (company/registers/FL_IMPROVEMENT_LADDER.md) records S7/S8 as the deployment/realism branch after S6;
the main FL headline (K2) stays the pre-declared cold-start FA of the ladder's S4 selection; S7/S8 are reported as the "realistic
Edge deployment" rows with their own contrasts (K20, K21) and the PRE row always beside them.

## 5. Code: FL-12a (warm-start init from PRETRAIN_FOR_FL_INIT.json in launch_fl), FL-12b (frozen tables + bytes + DP noise dimension).
