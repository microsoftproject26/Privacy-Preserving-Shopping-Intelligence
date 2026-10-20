# G1 COMBINED GATE PACKET — S* lock + data lock v2 + G1 code readiness (Ops, 2026-09-27T11:20Z)

Facts only. Tags: PROJECT_MEASURED (M), INFERENCE (I), PLAN (P). Paths are relative to `.local_ai/ppsi_v4/`.
Budget is tight: one answer, APPROVE / APPROVE_WITH_CONDITIONS / FIX, with numbered items.

## A. Feature search result (M) — ask: confirm the S* lock
- Rule (ADR-005 R2 + R4, binding): accept iff for the WORSE family the seed-mean relative drop vs L0S is ≤ 1.0% AND
  its 95% UB ≤ 2.0%; FS cohort = top 32,768 keys; CONTROLLED_BUDGET 6 EFE client-macro MRR@20 (E2E); seeds 2026 + 2027.
- Round A (`runs_fs/stageB/DECISION_A.json`, sha 142a9915…): drop groups within noise for BOTH families:
  G3 weekday, G4 utc_hour sin/cos, G7 item age + flag, G11 prev-session gap + mask, G12a prev view/session counts +
  recent distinct main categories. G13 SESSION_INTENT kept. REQUIRED item/category/session_offset kept.
- `runs_fs/FS_FINAL.json` (sha 799590a4…, 04:04:19Z): action LOCK, S* = `L3S_NEAR_DUPLICATE_SESSION-minus-G11+G12a+G3+G4+G7`,
  25 information channels, schema self c53d47fe…, no fallback. Relative drop (negative = S* better), n_users 10,269:
  | family | point | 95% CI | per-seed points 2026 / 2027 |
  |---|---|---|---|
  | GRU | −0.90% | [−1.63, −0.15] | −1.32 / −0.48 |
  | SASRec (worse) | −0.13% | [−1.13, +0.86] | −0.17 / −0.09 |
  L3S itself (the pre-drop ladder rung): GRU −0.67% (UB +0.10%), SASRec −0.08% (UB +0.86%).
- S* input columns (24; the schema counts 25 information channels = these + the output catalogue): tokens item,
  category, brand, session_offset (REQUIRED), session_mode_main_category; numeric log1p_price, log1p_prev_raw_view_gap_s,
  log1p_segment_elapsed_s, prefix_representative_count, prefix_known_category_count, prefix_category_switch_count,
  current_known_category_run_length; flags price_missing, prev_gap_unavailable; user context recent_category/brand
  match share, recent_mean_log_price, current_minus_recent_mean_log_price, current_matches_session_mode,
  session_mode_share; 4 masks. session_mode vocab 17 in the model vs 16 values in the data (superset; DL 49).
- Kept groups with mixed or one-family effects (drop rule requires no effect for BOTH families), grouped permutation
  importance in absolute MRR@20 units, 95% CI: **G8** GRU −0.00051 [−0.00078, −0.00024] (permuting it HELPS GRU) vs
  SASRec +0.00098 [+0.00056, +0.00141]; G9 GRU +0.00007 (CI spans 0) vs SASRec +0.00044 (CI > 0); G12b GRU +0.00058
  (CI > 0) vs SASRec +0.00008 (CI spans 0). Owner (2026-09-27): the search is NOT reopened; G8 is recorded as a thesis
  finding; an optional diagnostic on the G2 winner only, after G2.
- Process incidents, both fail-closed, kept on disk: `_failed_attempt1` (run_fs did not pass the FS registry → manifest
  verification refused), `_failed_attempt2` (stage_b without CUDA_VISIBLE_DEVICES → θ0 registration refused). DL 49, 52.
- Recorded as ADR-005 Addendum 3.

## B. DATA_PROTOCOL_LOCK_V2 (M) — ask: approve as the production data lock
- `contracts/DATA_PROTOCOL_LOCK_V2.json`, written O_EXCL by `tools/write_data_protocol_lock_v2.py`; lock_self
  bc699a97dd2fab054e5a5c00f9661dd32b0677941d3f23f2e4b81751b3b7b330; file sha 905ee4f9…. v1 unmodified.
- Built from, and fail-closed against, server-measured hashes: DATA06_SERVER_RECEIPT (b73ac762…; overall_pass, 9/9
  checks, test_split_read false, tests 9 passed + 7 NC failed as intended, RSS peak 5.6 GiB), REGISTRY_V2 (self
  f6a08207…), FS_FINAL, DECISION_A/B, S* schema, derived-schema REGISTRY (self 06d1f136…), THETA0_REGISTRY.
- Content: data version CLEAN_V2_POP5100_V2; splits/labels, catalogue A (K 158,486) and PREG1 values carried from v1;
  history USER_HISTORY_L50; production schema S*; shared statistics = full-TRAIN TRANSFORM_STATE_CLEAN_V2 (D5, R2);
  15 manifests with content/header/population hashes; bands MAIN r<131,072, CALIB r<32,768, U_NEW_VAL/HOLDOUT
  32,768 each, FS top 32,768, TB r<1,348,682; no TEST manifest. Contracts unchanged since v1 (0 hash differences).
- Known tolerated deviation: DATA-06's inherited assertion `calib_prefix_is_first_65536` (CALIB is by key v2) (DL 50).
- NOTE: REGISTRY_V2.source_data_protocol_lock still names the v1 lock (da51d271…), because v2 did not exist when
  DATA-06 ran. Ops proposes: keep REGISTRY_V2 unchanged; LOCK_V2 binds REGISTRY_V2 by self-hash (one-way binding).

## C. G1 code readiness (M unless tagged) — ask: rule on R2, R4 and the INT-02 scope
- Accepted code (Ops Level-1 re-runs): TRAIN-01R2 105 passed / 25 NC (trainer da2aff7e; closes G1 residuals R1
  registry-driven manifest roles and R3 native SMOKE_NOT_EVIDENCE refusal); tests_fs 59 / 5 NC; DATA-05 (causality
  proofs exhaustive on the pool, 0 violations); DATA-06 9 / 7 NC; INT-01 50 / 8 NC; SYS-04 STRICT determinism bitwise
  on the A16 for both families (full K). OPS-07 run1_final (6/7 suites) is superseded as evidence.
- Open (`company/plans/G1_RESIDUALS.md`):
  - **R2** `eval_stream`: PREG1_AMENDMENT_01 item 5 proposes `EVAL_ROWS_ONLY` for every method/evaluation; the frozen
    eval config must carry it (fingerprint change → re-bind the T3 receipt). Ask: confirm EVAL_ROWS_ONLY.
  - **R4** comparison key for production rows. Current code (v4int.providers L390–410) binds data = DATA_PROTOCOL_LOCK
    self-hash (v1) and feature = FEATURE_SCHEMA_CLEAN_V2 self-hash. Ops proposes: data = LOCK_V2 self-hash, plus
    manifest_registry = REGISTRY_V2 self, feature = S* self c53d47fe…, history_policy = USER_HISTORY_L50,
    determinism = STRICT, eval_stream = EVAL_ROWS_ONLY.
  - **INT-02 (P, Sonnet)**: `v4int/launch_central.py` passes manifest_registry / data_version / schema_id (S*) and binds
    LOCK_V2; `v4int.verify` accepts LOCK_V2 + REGISTRY_V2; update `tests_int` test_resume_exact and test_smoke_selection
    for the TRAIN-01R2 keys; then **R5** = one complete OPS-07 server run of all suites (exit 0) as the G1 evidence.
- Proposed gate mechanics (P): Fable rules now; Ops writes `company/gates/G1_APPROVED.json` ONLY after INT-02 +
  R5 pass, citing this verdict and the R5 receipt; no second Fable call unless R5 fails or a code change goes beyond
  the listed scope.

## D. Next production step after G1 (P) — ask: confirm or amend
- LR calibration (ADR-002 grid, recomputed per R2 D9): {1.0, 0.5} × executed peak (1e-3) × {GRU, SASRec}, 2 EFE,
  seed 2026, CALIB (32,768 users, 646,067 decisions) under S*, eval on CALIB_VALIDATION; 4 runs as detached per-GPU
  queues. Selection of the peak is mechanical (best CALIB metric at 2 EFE per family).
- Contest: MAIN (131,072 / 2,587,776), S*, seeds 2026 + 2027, both families, 6 EFE (60,651 updates), STRICT;
  G2 ARCHITECTURE_LOCK mechanical per T1 (`contracts/PREG1_LOCK_VALUES.json`, v4train.selection); Fable only on a
  trigger or HOLD. (I) ≈ 6–8 GPU-h, ≈ 4 h wall; disk 23 GB free, guard floor 8 GiB.

## E. Still pending owner ratification ([PROXY]; not asked here)
Fast-search rule; D5 full-TRAIN statistics (Opus preferred cohort refit); unseen-user protocol and sizes;
"EQUIVALENT → GRU on cost".
