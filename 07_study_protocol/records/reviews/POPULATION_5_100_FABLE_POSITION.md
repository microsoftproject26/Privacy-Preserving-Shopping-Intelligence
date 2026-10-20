# POPULATION_5_100 — Fable position (Fable 5.1, read-only; owner-proxy per DECISION_LOG 37; provisional, listed for ratification)

Inputs: packet; owner amendment §1–§15; DECISION_LOG 31, 33–37; CENSUS-02 and FEATSTAT-01 handoffs + `FEATSTAT_SUMMARY.md`; `POP5100_COST_MODEL.json`; `PREG1_LOCK_VALUES.json`; `DATA_PROTOCOL_LOCK.json`; EVALUATION/CLIENT contracts; G0/G0B/PREG1 syntheses. Not opened: the Opus position, the impact-matrix draft, `_superseded_run1/`. No model outcome used; the v1 prior enters as dispersion only.

ROUTING: claude-fable-5-1

VERDICT: **APPROVE_WITH_LOCKS.** The amended population + feature protocol is approved with the LOCK_VALUES below. No production training before: (i) DATA-05 leakage/equivalence receipts; (ii) v2 manifests and the I_META_RANDOM v2 hash are registered; (iii) the L0 regression guard passes; (iv) the D3 search registration is filed. Status OWNER_PROXY_FABLE.

## DECISION

**D1 Pool and partition.** Pool C = users with 5 ≤ n ≤ 100 eligible TRAIN examples under catalogue A (DATA-04 eligibility), TRAIN-only: 1,446,986 users / 28,623,386 examples, `e2a082cd…fbc70`. Order C by `selection_key_v2` = first 8 bytes big-endian of sha256(b"PPSI_V4_POP_5_100_SEL_20260926|" + digest32), digest32 = the C1.2 digest (salt `PPSI_V4_POPULATION_20260926` unchanged); keys unique. Contiguous, disjoint rank bands:

| band | ranks | role |
|---|---|---|
| MAIN | [0, 131,072) | contest + all downstream; `563e7663…28b3` |
| CALIB | [0, 32,768) | nested prefix; LR/μ/λ calibration; `9be38de1…1f02` |
| RESERVED_EXT | [131,072, 262,144) | unused; keeps a nested extension to 262,144 open for owner ratification without disturbing held-out bands |
| FS | [262,144, 327,680) | feature-search cohort |
| U_NEW_VAL | [327,680, 393,216) | unseen users; adaptation-recipe selection only |
| U_NEW_TEST | [393,216, 458,752) | unseen users; sealed |
| TIEBREAK | C \ (U_NEW_VAL ∪ U_NEW_TEST) | 1,315,914 users; T1 tie-break universe (owner §7) |

The same key orders B, D and cold users for D8. Hashes of FS, U_NEW_*, TIEBREAK and every panel are computed at build and registered before training.

**D2 N_MAIN = 131,072; N_CALIB = 32,768.** Precision: the paired GRU−SASRec 95% half-width is 0.00108 (prior), under the T1 threshold 0.0020 with margin; 262,144 buys 0.00076 at 2× cost, which no predeclared rule needs; the 0.0002 tolerance is unresolvable at any N. Coverage: 40,764 VAL users / 519,714 eval rows (515,636 rankable); 97.6% of rankable targets have an in-cohort positive; 116,211 classes trained. Representativeness: uniform hash; n p50 13, mean 19.74 vs pool 19.78. Cost: downstream core 72 / 202 GPU-h (GRU/SASRec) vs 144 / 404; 262,144 would exceed the old accepted SASRec scale (366) with no headroom for unmodelled items and the PENDING L50 GRU step cost. CALIB = N/4 (10,070 VAL users, 127,530 rows), the old ratio; N/8 (4,954 users) is too thin near 0.002.

**D3 Feature search** (confirms DECISION_LOG 36, amended).
- Ladder: FEATSTAT-01 rungs with L1 split — L1a (39): 6 dead flags, price_zero, brand_missing, main_category_missing, category_code_missing, local_weekday_sin/cos, is_weekend, recent_history_length (constants/exact duplicates); L1b (37): main_category_tokens, daypart_tokens (information-preserving but a different inductive bias — measured, not assumed). Then L2 35 → L3 30 → L4 28 → L5 18 → L6 16 → L7 13 → L8 10. L9 ID-only: descriptive anchor, run last, never selectable. Identical columns for both families; L0 = Clean V2 reference.
- Cohort: FS (65,536; disjoint from MAIN/CALIB/unseen); its VAL users M_FS_VAL (≈ 20.2k users / 255k rows expected, measured at build); USER_HISTORY_L50; statistics per D5 fitted on FS. Contest VALIDATION is never touched by feature selection.
- Recipe per rung: both families × seeds {2026, 2027}, 6.0 EFE, common schedule, exposure_point end, marks every 0.5 EFE; peak LR = the CALIB calibration ({1.0, 0.5}× executed, 2 EFE, seed 2026, on M_CALIB_VAL) run under L0 before the search; identical across rungs.
- Statistic: seed-mean client-macro MRR@20, E2E, CONTROLLED_BUDGET (6.0 endpoint); PRACTICAL_BEST reported only. r_f(k) = (M_f(L0) − M_f(Lk)) / M_f(L0), bootstrapped directly (E7.5) on M_FS_VAL's shared W.
- Rule: rung passes iff, for the WORSE family, point r̂ ≤ 1.0% AND 95% upper bound ≤ 2.0% (UB ≤ 1% would reject everything at ~0.0015 absolute half-width).
- Stop rule: ascending; at the first failing rung Lk, one skip: test L'(k+1) = L(k+1) ∪ G_k (G_k = the group removed at Lk); if it passes, continue L'(j) = L(j) ∪ G_k until the next failure, which ends the search; if L'(k+1) fails, the search ends. L* = smallest passing schema visited. Cap 10 rung evaluations. Registered before the first run.
- Budget: 3.2 GPU-h per rung → ≤ 32, expected ≈ 20.
- Guards: configurable-schema code at L0 must reproduce the frozen L0 model bitwise on a fixture; L5+ needs the SASRec user-branch decoupling (`sasrec_v2.py`) with tests. L* locked by ADR before the contest; the contest re-calibrates LR under L*.

**D4 Catalogue A** (K = 158,486, order unchanged). B drops 15,536 classes for 8–16 MB on models that are 97% item table, makes 26,992 pool VAL targets OOV (−0.46 pp), moves 1,145 users across the population boundary (circularity) and breaks comparability — a confound, not a gain. Strict-unseen implication: support counts include the unseen bands' TRAIN views; the catalogue is declared DEPLOYMENT_AVAILABLE_BOOTSTRAP (platform catalogue) and not refit; disclose the count of classes whose support would fall below 5 without those views (count only).

**D5 Shared statistics: option C — fitted on the training cohort's own TRAIN rows.**

| statistic | class | rule |
|---|---|---|
| 17 normalizers (14 v1-rule, M1×2, F-1 valid-only) | COHORT_SPECIFIC_LEARNED | refit on MAIN (contest, downstream), FS (search), TIEBREAK (tie-break); same 0.001/0.999 clip + valid-row mean/std |
| first_seen_train (item age) | COHORT_SPECIFIC_LEARNED | MIN over the training cohort's TRAIN rows; availability flips reported |
| catalogue K/support/order; attribute vocabularies; recovery tables | DEPLOYMENT_AVAILABLE_BOOTSTRAP (declared; pooled provenance disclosed) | unchanged |
| rezero map, clip levels, +3h clock, calendar vocabularies | DEPLOYMENT_AVAILABLE_BOOTSTRAP | unchanged |

One I_META_RANDOM v2 per training cohort, identical for C/FA/FP/PF/CPF/LO. For FA/FP/PF it is accounted as one federated-statistics round (secure sums for mean/std, a quantile sketch for clips, a min for first-seen; bytes in communication accounting); for LO a shipped block. Unseen bands and non-main strata receive MAIN's block and contribute to nothing fitted. Trade-off: price/time/age normalizers move ≤ 0.02 mean |Δz|, so comparability is lost only on count channels (mostly removed by L7); cohort first-seen weakens item age, which L6 tests anyway.

**D6 FL client work** (B = 16: visit steps p50 2, max 14, 60.6% single-batch, Gini 0.44).
- Local batch 16 kept: B = 32 leaves 83% of clients with the proximal term acting on one step; B = 8 costs 1.7×. With 2 passes every client takes ≥ 2 steps, so FedProx acts on step 2 for all.
- 2 local passes per visit (exact exposure: 2n examples), 3 visits = 6 EFE; never fixed steps.
- n_consumed weighting for FA and FP (max ratio 20:1 under the cap).
- 64 clients/round; seeded full-sweep permutations, fixed groups, 100% participation per sweep; N/128 = 1,024 rounds/EFE; 6,144 rounds to 6.0; continuation only per C3.
- EFE at MAIN: 2,587,776 examples; 224,606 client steps; 10,109 central updates/EFE (60,651 to 6.0); LO 1,347,636 steps over 6 passes.
- Retry: a failed visit re-executes from the round checkpoint with the same RNG; never skipped or double-counted; planned == actual n_consumed enforced (T2 P2); restarts logged.
- Grids (local LR {peak_C, 0.5×}, μ {0.001, 0.01, 0.1}, λ {0, 1e-4, 1e-3}), p dim = query dim, AdamW lifecycle: unchanged; recalibrated on CALIB at 2 EFE for the winner under L*.

**D7 Unseen-user protocol** (participation gap, Yuan et al. 2022; strong generalization, Liang et al. 2018).
- Groups: U_NEW_VAL, U_NEW_TEST, 65,536 each (≈ 20.2k VAL-active users each; eval-only cost). No query-count filter; strata 1+/5+/10+ descriptive only.
- Global timeline kept: support S(u) = all of u's observed events strictly before the query (TRAIN period and earlier VALIDATION events); query = every E2E VALIDATION decision, user-macro; RETRO_TEST sealed.
- Two labelled axes: history budget h ∈ {ALL, 0, 1, 5, 10, 20, 50} (h = 0 = query event only; E11.2 FIXED_20/FIXED_50/ALL_AVAILABLE; under L50 truncation applies to the window directly) and adaptation A ∈ {NONE, FEW_SHOT}. "No adaptation" = A NONE at h ALL (the participation-gap row); "no history" = h 0.
- Never-participated user: C, FA, FP — global model, A NONE only. PF, CPF — shared part + p_u = 0 at NONE; p_u reconstructed from own support at FEW_SHOT (FedRecon-style, Singhal et al. 2021). LO — per-seed init + block at NONE (by design), full local training on own support at FEW_SHOT.
- Chosen on U_NEW_VAL: only the FEW_SHOT recipe (passes {1,2,4}, locked personal LR/λ) on FIXED_20, once, before any U_NEW_TEST outcome. Nothing else is selected on unseen users.
- Strict unseen: absent from every trained parameter, fitted statistic (D5), calibration, search, contest and tie-break; present only in catalogue support (D4 disclosure).
- U_NEW_TEST VALIDATION rows: evaluated once after ARCHITECTURE_LOCK + recipe lock as pre-registered confirmation; RETRO_TEST after ALL_METHODS_LOCK.
- Rolling-origin October backtests: OUT_OF_SCOPE (break the global timeline; one retrain per origin).

**D8 Non-main panels** (v2 key; eval-only unless FEW_SHOT; never pooled with MAIN).
- TRUE COLD: COLD_PANEL_VAL = 381,295 n = 0 VAL users, reported A0 299,035 / A1 82,260; COLD_H = 16,384 smallest keys; COLD_S (recipe selection) = ranks 16,384–32,767.
- SPARSE: SPARSE_PANEL_VAL = 155,059 B users (1,236,410 rows); SPARSE_H = 16,384 smallest keys, FEW_SHOT support = own eligible TRAIN examples + earlier VALIDATION events.
- HEAVY: HEAVY_PANEL_VAL = 36,373 D users (1,370,490 rows), h ALL, A NONE; unseen, so the row mixes participation gap and activity shift (disclosed).
- WARM_TRUNCATION = 16,384 smallest-key MAIN E2E users (trained-on).
- One h grid, conventions, C_ref (E11.10) and registered recipe grid across all h panels; recipe on COLD_S for cold, on U_NEW_VAL FIXED_20 for unseen/sparse. E11.4 recompute rules and T41 re-proved under L50 (DATA-05).

**D9 Impact matrix.**
- SUPERSEDED: client eligibility; main cohort size; calibration cohort; selection key/salt; feature normalizers + first_seen; validation population (M02/M06 → MAIN/CALIB VAL; panels rebuilt); cold-start definitions (four strata); history window (SESSION_L50 → USER_HISTORY_L50, ablation only); thesis scope.
- RESEARCH_REVIEW_REQUIRED, resolved here: feature schema (D3; L* by ADR); pooled bootstrap block (D5); tie-break population (TIEBREAK; staged rule, thresholds, caps 65/125 unchanged; evaluated on TIEBREAK VAL users, MAIN VAL as subset); device preprocessing (L50 needs on-device 50-event history; recompute after L*).
- RECOMPUTE_ONLY: EFE quantities; LR calibration (twice: L0, L*); FedProx/PF calibration; rounds; LO steps; communication accounting (params after L* + the statistics round); export inputs after L*; bootstrap W per user set; OOV/censor counts per manifest.
- UNAFFECTED: catalogue threshold/order; OOV policy; T1 rule and triggers (equivalent-outcome fallback still pending ratification); T2 end; T3 numerics; STRICT training determinism and eval_stream EVAL_ROWS_ONLY (both confirmed here); C3; local batch, passes, FedAvg weighting; PFL p_u design; CPF; LO rule; bootstrap method; temporal boundaries; macro-by-user.

**D10 Compute and scope.** INFERENCE from the cost model (GRU L50 step cost PENDING; SASRec pad-50, unaffected): search ≤ 32 (≈ 20); LR calibrations ≈ 1; contest 6.4; tie-break if triggered ≈ 32 stage 1 / ≈ 65 total; winner FL/PF calibration ≈ 5 (GRU) / 16 (SASRec); downstream 72 / 202; panels ≤ 20. Totals ≈ 125–190 GPU-h (GRU) or ≈ 265–330 (SASRec): ≈ 3–4 or 5.5–7 days on two A16s, versus ≈ 10–11 days at 262,144. Scope: the matched causal claim holds for 5–100-example REES46 users (Oct–Nov 2019), a uniform 131,072 sample, one 46-day TRAIN / 7-day windows, previously inspected; cold, sparse, heavy and unseen rows are secondary generalization evidence; the feature decision transfers from 65,536 to 131,072 (disclosed); 1% is an equivalence tolerance (a passing rung may be ≤ 2% worse at the CI bound); FL is simulated without a privacy mechanism; the precision prior is v1/VC and is re-measured on the first V4 runs.

## LOCK_VALUES
- POPULATION: pool C (1,446,986 / 28,623,386; `e2a082cd…fbc70`); key v2 salt `PPSI_V4_POP_5_100_SEL_20260926` over the C1.2 digest; N_MAIN 131,072 (`563e7663…28b3`); N_CALIB 32,768 (`9be38de1…1f02`); RESERVED_EXT [131,072, 262,144); FS [262,144, 327,680); U_NEW_VAL [327,680, 393,216); U_NEW_TEST [393,216, 458,752); TIEBREAK = C minus unseen bands (1,315,914).
- SEARCH: L0/L1a/L1b/L2…L8 (+L9 anchor); FS; 2 seeds; 6 EFE; endpoint statistic; r̂ ≤ 1% and UB95 ≤ 2% on the WORSE family; skip-one stop; cap 10; LR from CALIB under L0.
- CATALOGUE: A, K 158,486.
- STATISTICS: I_META_RANDOM v2 fitted per training cohort; catalogue/vocab/constants bootstrap.
- FL: B 16; 2 passes; 3 visits; 64/round; full sweeps; n_consumed; 6,144 rounds; grids unchanged; CALIB 2 EFE.
- UNSEEN: 65,536 × 2; global timeline; h × A axes; recipe on U_NEW_VAL FIXED_20 only; backtests out of scope.
- PANELS: cold 381,295 (+16,384 h, +16,384 s); sparse 155,059 (+16,384 h); heavy 36,373; warm truncation 16,384.
- EFE at MAIN: 2,587,776 examples; 10,109 updates; 60,651 to 6.0.
- Unchanged: T1/T2/T3, STRICT, EVAL_ROWS_ONLY, C3, bootstrap 1000 @ 20260926.

## OPEN_DISAGREEMENT_RISKS
1. 131,072 vs 262,144: headroom and sufficiency over precision beyond any rule; RESERVED_EXT keeps the extension available.
2. Search at 65,536 × 2 seeds vs 131,072 × 1 seed: seed noise (v1 SASRec 0.00087) is comparable to the margin, so seeds win; size transfer disclosed.
3. Endpoint statistic for rungs while the contest uses PRACTICAL_BEST: removes max-selection noise; both reported.
4. Option C weakens item age and ends v1 comparability on count normalizers; the cleaner boundary is worth it.
5. Skip-one departs from strict nesting; without it one failing group (item age) would freeze ~5 removable channels.
6. Unseen bands hold ~9% of the pool at eval-only cost; the participation-gap contrast is unpaired (hw ≈ 0.003 at 20k users), a secondary claim.
7. Catalogue A means strict unseen excludes support counts; a third catalogue would reintroduce circularity.
8. GRU step cost under L50 unmeasured; a 2× rise still leaves the GRU path under the SASRec path.

Sources: [Yuan et al., ICLR 2022 (participation gap)](https://arxiv.org/abs/2110.14216); [Liang et al., WWW 2018 (strong generalization)](https://dl.acm.org/doi/10.1145/3178876.3186150); [Wu et al., Motley 2022](https://arxiv.org/abs/2206.09262); [Singhal et al., FedRecon, NeurIPS 2021](https://arxiv.org/abs/2102.03448); federated preprocessing statistics: [FedPS](https://arxiv.org/html/2602.10870v1), [SecureFedYJ](https://arxiv.org/pdf/2210.01639).
