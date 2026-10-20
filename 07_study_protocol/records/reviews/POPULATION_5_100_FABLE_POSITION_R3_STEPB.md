# POPULATION_5_100 — Fable third output (a re-ruling plus the STEP B answer), verbatim

Ops note: produced after the FINAL ruling because messages queued to Fable were processed in order. Consolidated by Ops in DECISION_LOG 42 and the synthesis; not adopted wholesale.

---

ROUTING: claude-fable-5-1 (Research Director, read-only; owner-proxy per DECISION_LOG 37 — every ruling below is PROVISIONAL_PENDING_OWNER_RATIFICATION where marked ★)

VERDICT: **APPROVE_WITH_LOCKS.** The census (CENSUS-02) and feature statistics (FEATSTAT-01) are sufficient to freeze population, catalogue, statistics provenance and the feature-search procedure before any V4 outcome exists. No TEST outcome and no v1 level informed anything below; only dispersion (paired SD 0.111, ρ 0.858) was used. The quarantined run-1 and the Opus/matrix drafts were not opened. DATA-05 (USER_HISTORY_L50 builder, leakage proofs, window census, step cost) remains a G1 blocker.

## DECISION

**D1 — Pool and partition (LOCK).**
- POOL = stratum C: users with 5 ≤ n_eligible_TRAIN ≤ 100 under catalogue A, eligibility per CLIENT_CONTRACT C2.1 (TRAIN split, loss_mask, target_ts < TRAIN_END, target in the support≥5 catalogue). 1,446,986 users / 28,623,386 examples; population_sha256 `e2a082cd…fbc70`.
- selection_key_v2 = first 8 bytes, big-endian, of sha256(`PPSI_V4_POP_5_100_SEL_20260926|` + digest32), digest32 = the C1.2 client digest (salt `PPSI_V4_POPULATION_20260926` unchanged); rank = ascending key (uniqueness asserted). One key rule for every V4.1 manifest, panels included.
- Blocks by rank: **MAIN [0, 131,072)** hash `563e7663…28b3`; **CALIB [0, 32,768)** hash `9be38de1…1f02` (nested); **U_NEW_VAL [131,072, 196,608)**; **U_NEW_TEST [196,608, 262,144)**; **FS [262,144, 327,680)** (feature search); **TIEBREAK universe = POOL \ (U_NEW_VAL ∪ U_NEW_TEST ∪ FS)** = 1,250,378 users, MAIN included, examples counted at build. U_NEW users are never trained on by any arm, calibration, screen or tie-break.
- VALIDATION manifests: E2E decisions of each block's users in [11-16, 11-23); MAIN_VAL_WARM ≈ 40,764 users / 519,714 rows (MEASURED). TEST manifests only after ALL_METHODS_LOCK.

**D2 — N_MAIN = 131,072; N_CALIB = 32,768 (N/4) ★.**
- Precision: paired GRU−SASRec 95% half-width 0.00108 (prior) resolves every predeclared threshold (T1a 0.0020); within-family FL contrasts at ~1% relative are resolvable at 40,764 evaluable users. 262,144 buys √2 precision no rule needs, at 2× every downstream run: 144/404 GPU-h core (GRU/SASRec) versus the 88/263 envelope accepted at G0, before the unmeasured USER_HISTORY_L50 step-cost multiplier and before the owner's added work (screen, unseen groups, strata panels). 65,536 is rejected: 20,205 evaluable users, class coverage 94.9%, half-width 0.00153.
- Coverage: 97.6% of MAIN rankable VAL targets have an in-cohort positive; the n distribution (p50 13) is identical to the pool at every N (uniform hash).
- CALIB: 10,070 evaluable users / 127,530 rows; the same 1/4 ratio as the old lock; calibration is a 2–3-point choice and its cost at 32,768 is 4/12 GPU-h for the winner's μ and λ grids.

**D3 — Minimal-sufficient feature search (as reconciled in STEP B below; DECISION_LOG 39 amended).**
- Candidate set: the FEATSTAT nested ladder, L0 = Clean V2 (53) reference. L1/L2 removals are exact or near-constant on the pool and n ≥ 1 and carry no risk; L3 (near-duplicates) must be trained, not adopted.
- Cohort: FS (65,536 users, ≈20k evaluable, INFERENCE), USER_HISTORY_L50, catalogue A, statistics fitted on FS TRAIN per D5. Disjoint from MAIN/CALIB/U_NEW; excluded from the tie-break; rows labelled FEATURE_SEARCH_SCREEN and refused by architecture selection.
- Statistic and rule: seed-mean (2026, 2027) client-macro MRR@20 at CONTROLLED_BUDGET (6.0 EFE); r̂ = 100·(L0 − Lk)/L0 bootstrapped directly (E7.5, W shared, seed 20260926, 1000). **PASS iff, for the WORSE family, r̂ ≤ 1.0% AND UB95 ≤ 2.0%.** Identical columns for both families.
- Procedure: L0 and L3 trained; then grouped permutation importance (screen only); S* = L3 minus within-noise groups, trained and tested; fallbacks per STEP B; cap 10 schema evaluations. L* is hashed as feature_contract before the contest; the contest re-calibrates LR on CALIB under L*.
- Disclosure: the screen previews family levels on FS; selection remains mechanical under T1 on MAIN.

**D4 — Catalogue A (keep global support ≥ 5, K = 158,486).** B removes 9.8% of K to gain nothing the MAIN users use (0.136% of pool targets), makes 26,992 pool VAL targets OOV, moves 1,145 users across the C boundary (circular) and breaks class-space comparability. Strict-unseen: item-side statistics (catalogue, vocabularies, first_seen) are declared deployment-available bootstrap (the platform's item table predates any user) and do include U_NEW's TRAIN views — LIMITATION, disclosed, bounded (far below the 15,536-class effect of removing 3.36M users). T3 score_block stays K.

**D5 — Statistics population ★ (LOCK).** Rule: *item-side = bootstrap from full TRAIN; user-behaviour-side = learned from the training cohort only.*
- DEPLOYMENT_AVAILABLE_BOOTSTRAP: catalogue A, attribute vocabularies, recovery tables, first_seen_train (item age), rezero rule, declared constants (+3h clock, quantile levels).
- COHORT_SPECIFIC_LEARNED: every z-normalizer of the surviving schema (clip 0.001/0.999 + mean/std, M1/F-1 valid-only conventions unchanged), fitted on the training cohort's own TRAIN rows: MAIN for production, FS for the screen, TIEBREAK universe for the tie-break. Realised once per cohort as an accounted federated-statistics round (moments by secure sum, clip bounds by quantile sketch — standard practice, e.g. [FedPS](https://arxiv.org/html/2602.10870v1), [SecureFedYJ](https://arxiv.org/pdf/2210.01639)); the block is shipped identically to C/FA/FP/PF/CPF/LO and counted in FL communication. U_NEW never contributes (strict unseen by construction).
- POOLED_FROM_EXCLUDED_CLIENTS: none remains on the user side. Trade-off: comparability lost only on the 10 count channels (mean |Δz| ≤ 0.28), most of which the ladder removes.

**D6 — FL client work (LOCK).** B = 16, 2 passes per visit, 3 visits (6 EFE); steps per visit 2–14 (p50 2, 61% single-batch); B = 8 rejected (×1.68 client steps for one more proximal-active step). 64 clients/round, seeded permutation sweeps, fixed groups, full participation: 2,048 rounds per sweep, 6,144 to 6.0 EFE. n_consumed weighting for FA/FP/PF; AdamW reset per round; μ ∈ {0.001, 0.01, 0.1}, λ ∈ {0, 1e-4, 1e-3} recalibrated on CALIB at 2 EFE under L*. Retry: STRICT determinism means a failed visit reproduces; it is a run failure, never a silent skip — restart from the last round checkpoint after a fix, with a receipt; EFE counts consumed examples only (T2 P2 table unchanged).

**D7 — Unseen users (LOCK).** Literature: strong generalization ([Liang et al. 2018](https://arxiv.org/abs/1802.05814)), participation gap ([Yuan et al. ICLR 2022](https://arxiv.org/abs/2110.14216)), support/query reconstruction for new clients ([FedRecon](https://arxiv.org/abs/2102.03448); [Motley](https://arxiv.org/abs/2206.09262)).
- Groups: U_NEW_VAL and U_NEW_TEST, 65,536 pool users each (D1), user-disjoint from everything trained.
- Global timeline kept: query = E2E decision in VALIDATION (U_NEW_VAL) or RETRO_TEST (U_NEW_TEST, after ALL_METHODS_LOCK); context = the USER_HISTORY_L50 window, identical to MAIN; support (Study B) = the user's own labelled transitions strictly before the query (E11.3). No October origins; rolling-origin backtests OUT OF SCOPE (break the timeline and the seal; DL 34).
- Arms: C/FA/FP → CONTEXT_ONLY (no adaptation registered). PF/CPF → p_u = 0 row (NO_ADAPTATION) and p_u reconstructed from own support (ADAPTED). LO → its normal procedure from θ0 + bootstrap block; its participation gap is 0 by construction (anchor).
- Chosen on U_NEW_VAL, once, before any U_NEW_TEST row: only the unseen adaptation recipe within the registered grid (passes {1,2,4}, personal LR, λ). Nothing else.
- h analysis: two axes — h_ctx (visible prior events) and h_sup (support labels); grid {0,1,5,10,20,50} + ALL_AVAILABLE. **NO_HISTORY** (h_ctx = 0) and **NO_ADAPTATION** (h_ctx = all, h_sup = 0) are distinct rows. Query-count strata (1+/5+/10+) descriptive only, never a filter.

**D8 — Non-main panels (LOCK).** TRUE_COLD = A0 ∪ A1 VAL users (381,295; A0/A1 reported separately), SPARSE = B (155,059), HEAVY = D (36,373): CONTEXT_ONLY evaluation at the locked checkpoints (PRACTICAL_BEST and CONTROLLED_BUDGET), every arm, never in the matched block. Adaptive studies on the 16,384 smallest-key-v2 users of each; M04 (candidate A) and M04S semantics kept, rebuilt with key v2 and USER_HISTORY_L50; SPARSE Study B uses the user's 1–4 own TRAIN labels as support. FIXED_20/FIXED_50/ALL_AVAILABLE conventions unchanged.

**D9 — Impact matrix.**
SUPERSEDED: client eligibility; main cohort size; calibration cohort; feature schema (→ L*); feature normalizers (→ D5); validation population (manifests; the window is UNAFFECTED); cold-start definitions (SPARSE is not cold; h split per D7); thesis scope; history window (USER_HISTORY_L50; SESSION_L50 named ablation).
RESEARCH_REVIEW_REQUIRED → decided here: pooled bootstrap statistics (D5); architecture tie-break population (TIEBREAK universe; stage-1 ≈ 31 GPU-h, INFERENCE, within the 65 cap; rules unchanged); local batch (kept 16); device preprocessing and export inputs (re-specified after L*, DEV task).
RECOMPUTE_ONLY: architecture selection rule (T1 unchanged on MAIN_VAL_WARM); EFE budget (2,587,776 examples/EFE; 60,651 central updates to 6.0; 12 marks + endpoint); LR calibration; client sampling and rounds; FedProx/PFL grids; LO replay (131,072 clients); communication accounting (+1 statistics round); CIs/bootstrap (new W per user set).
UNAFFECTED: catalogue threshold; OOV policy; stopping rule C3; local passes; FedAvg weighting; PFL state design; CPF control; T2 "end"; T3 numerics; STRICT training; eval_stream EVAL_ROWS_ONLY.

**D10 — Compute and scope (INFERENCE).** Screen ≈ 10 GPU-h; contest 6.4; calibration 5–13; downstream core 72 (GRU) / 202 (SASRec); unseen and strata adaptation ≈ 10/30; tie-break if triggered ≤ 65/125. Total ≈ 105 (GRU) to 260 (SASRec) GPU-h, ≈ 2.5–5.5 days on two A16s, before DATA-05's step-cost multiplier. Limitations: claims hold for users with 5–100 eligible TRAIN examples on one platform over 46 days of Oct–Nov 2019; VALIDATION/TEST previously inspected (retrospective); FL is simulated with pooled item-side bootstrap; cohort statistics come from a simulated federated round; the unseen panel measures the participation gap within the same population; cold/sparse/heavy results are generalization checks; L* is a 2-seed non-inferiority screen, not equivalence; the precision prior is v1 VC-based and must be re-measured on the first v4 calibration.

## OPEN_DISAGREEMENT_RISKS
- N_MAIN 131,072 vs 262,144 ★: I traded √2 precision for halved wall clock and headroom; the owner's "largest useful" wording may favour 262,144.
- Cohort-fitted normalizers ★ lose comparability on the count channels; the alternative (full-TRAIN) is simpler but pooled from excluded clients.
- Catalogue A includes U_NEW views in support counts — a bootstrap-vs-strict-unseen tension I resolved by declaration and disclosure.
- The screen reveals GRU/SASRec levels on FS before the contest; mitigated by mechanical T1, not eliminated.
- B = 16 leaves FedProx one proximal-active step for 61% of clients — a property of the population, reported, not fixed.
- Unseen groups at 65,536 each raise Study-B cost; halving them is a legitimate owner call.

---

## STEP B — D3 reconciliation with DECISION_LOG 39 (binding delta)

ROUTING: claude-fable-5-1

VERDICT: **ACCEPT_WITH_AMENDMENTS.** The procedure is scientifically acceptable: L3 is *tested*, not adopted blind; S* from permutation importance is only a candidate generator and is confirmed by retraining under the same non-inferiority rule; 2 seeds on every accepted or reference schema; FS is disjoint and refused by selection. Speed is served without moving a decision onto outcomes.

LOCK_VALUES delta (D3 only):
- **P1 ACCEPT.** FS ranks [262,144, 327,680), 65,536 users, USER_HISTORY_L50, catalogue A, statistics fitted on FS TRAIN (D5).
- **P2 ACCEPT** unchanged: seed-mean CONTROLLED_BUDGET client-macro MRR@20; PASS iff WORSE family r̂ ≤ 1.0% AND UB95 ≤ 2.0%; 2 seeds for L0 and every accepted/reference schema. Amend: screen runs evaluate at 2.0, 4.0, 6.0 EFE only (descriptive); 6.0 carries the statistic.
- **P3 ACCEPT** executed peak 1e-3 for all rungs and both families, with the FS fallback triggered automatically if any L0 run is non-finite or its 6.0 value is below its 2.0 value by > 5% relative (a within-run test, not a family comparison).
- **P4 ACCEPT** (a)–(f), with three amendments: (b) noise is judged on the seed-mean drop with the seed-mean bootstrap CI; (c) a group is "within noise" only if ≤ 0.1% relative AND CI contains 0 for BOTH families and BOTH seeds' signs agree; (d) if S* fails, first test S* + the single most important removed group (max over families), then the skip-one ladder; (e) L1a/L1b definitions hashed before use. L* = the smallest passing schema; stop there.
- **P5 ACCEPT** conditional: host RAM check for 2 × ~7.1 GiB provider RSS under the 16 GiB guards; screen runs keep no weights — digests, endpoint ranks and receipts only (disk 13 GB, floor 8 GiB).
- Every screen row: FEATURE_SEARCH_SCREEN, refused by T1; L* hashed as feature_contract before the contest.
