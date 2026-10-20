# BEST_BACKBONE plan — Fable 5.1 ruling (verbatim; owner-proxy where applicable; received 2026-09-27)

Draft: `company/plans/BEST_BACKBONE_PLAN_DRAFT.md`. Saved verbatim by the Operations Director.
OWNER OVERRIDE (DECISION_LOG 57, 58): LEGACY_REPLAY is DROPPED everywhere below (Q5, P1, P5, claim (b), trigger 3,
the AMENDMENT_02 item). The owner's correction reached Fable after this ruling was produced; the rest stands.
Ops note (consistency with trigger 2): the non-neural baselines read MAIN_VALIDATION only from P3 on; during P2 their
own settings (kNN k / sample size) are tuned on CALIB_VALIDATION, the same equal-effort principle.

---

VERDICT: APPROVE_WITH_AMENDMENTS

Assumptions: SYS-04b/INT-01 step rates (17.51/9.19 upd/s STRICT) and G1_COST_UPDATE eval times (0.162/0.223 ms per row) hold; the V4 VALIDATION/TEST periods coincide with v1 VS/TEST (not verified by me — Ops states it in the P1 packet); no tuning manifest beyond LOCK_V2 exists and none is created (trigger 7). Verified locally: recipe.py, schedule.py, gru_v2.py:234–252 (128-d hard-coded), sasrec_v2.py (d_model/blocks are config), PREG1 T1 rules, LOCK_V2 manifests (MAIN 131,072/2,587,776; MAIN_VALIDATION 51,211 users/519,714 rows; CALIB 646,067; CALIB_VALIDATION 12,715/127,530), packet §D L71 (60,651 updates), EXECUTED_RECIPES, FINAL_SASREC_REPORT, SOTA_CLAIM_STATUS.

Q1 — CALIB acceptable. G1 §D and ADR-005 already bind calibration to CALIB; the nesting is symmetric across families; MAIN_VALIDATION stays unread until P3; the claim lives on sealed TEST. Conditions: every P2 read is CALIB_VALIDATION only (phase CALIBRATION); disclose the overlap (12,715 of 51,211 MAIN_VALIDATION users = 24.8%); report every P3 MAIN_VALIDATION number also on the 38,496 CALIB-disjoint users (diagnostic, never selection). A disjoint band needs new manifests → refused.

Q2 — Cut to LR × dropout × schedule. Drop weight decay (1e-5 vs 1e-3 at LR 1e-3 over 60k updates = cumulative shrink exp(−0.06), a no-op class; larger values act on the 158k-row table unpredictably), label smoothing (no evidence at full-K CE), and all shape changes: GRU 128-d is hard-coded (v4models edit + θ0 + client contract), and my F0-d256 advice was for 40M examples — at 2.59M MAIN examples the 20.3M-parameter item table is already ≫ data; SASRec d384/3 blocks likewise. gBCE is moot at full-K CE; tying stays as executed (GRU tied, SASRec untied). Cooldown shape linear (schedule.py is piecewise-linear; 1−sqrt is a refinement not worth new T2 pins). Checkpoint averaging: POLISH only, tested once (P2 rung 3).

Q3 — No 10 EFE. 6.0 is the locked CONTROLLED_BUDGET (T2 knots, T1 endpoint); FL must use the same registered 6-EFE recipe, so a longer central breaks central-vs-FL equality; CALIB→MAIN transfer of "optimal EFE" is unreliable at 4× less data; the registered continuation (6→8, STILL_IMPROVING only, rows never selected) is the longer branch. C_BEST allowed as a labelled composite (P4), never the contest row, never the FL recipe. Ops' P3 costs (10.7/18 GPU-h) are from the superseded 91,569-update cohort; 60,651 updates → ≈ 7.2 GPU-h (I).

Q4 — Approve. TRAIN-02 (Opus): HP registry (per-family registered tuples with sha256; RunConfig.validate accepts only registered values), role TUNING → rows HP_SEARCH refused by v4train.selection, knots and dropout as registered values (S0 default keeps the planned_lr_table hash), CKPT_AVG eval script; v4eval untouched (T3 stays BIND); Opus gate review + full server suite, no Fable call. PREG1_AMENDMENT_02 (registration only) before the first P2 launch: space, rungs, seeds, selection/tie/adoption rules, S1 knots, CKPT_AVG rule, LEGACY_REPLAY protocol, C_BEST rule, plus finding-3 TB wording. Tuned recipes appended (02b) before P3.

Q5 — Approve with conditions: (i) item-vocab identity check (K 158,486, v1 vocab hash ↔ catalogue A) before scoring; (ii) v1's own input pipeline, the frozen T3 evaluator, EVAL_ROWS_ONLY, family LEGACY_V1 in the comparison key; (iii) VALIDATION replay is descriptive (v1 epoch/seed/cooldown were chosen on v1 VS); TEST replay only at ALL_METHODS_LOCK; (iv) state the asymmetries: v1 = 40.2M examples, all users, session window; V4 = 2.59M examples, 131,072 users, USER_HISTORY_L50. It is a protocol-matched SYSTEM comparison. If V4 < legacy, that is data scale, not failure — record it so.

Q6 — G1 ruling stands; only §D LR-CAL is replaced by P2. INT-02 closed; G1_APPROVED.json as ruled. One Fable call (G1b) at "P2 table → tuned recipes registered → P3 launch".

FINAL PLAN (per family, identical; GPU-h are INFERENCE)
P0 G1 closes; TRAIN-02 merged; AMENDMENT_02 written.
P1 Baselines: LEGACY_REPLAY (Q5); popularity, last-item, session item-kNN on MAIN TRAIN events (CPU, label BASELINE). ≈ 1 GPU-h.
P2 Tuning on CALIB_TRAIN (2,524 updates/EFE), eval CALIB_VALIDATION, STRICT, S*, B 256, endpoint reads only.
 Space: F1 peak ∈ {0.5,1,2}×1e-3; F2 dropout ∈ {as-is (GRU 0.20/0.10, SASRec 0.2), +0.10}; F3 schedule on the 6-EFE clock ∈ {S0 registered knots; S1 WSD knots (0,0),(0.02,1),(4.2,1),(6.0,0.01)}. S0 ≡ S1 on [0,2], so rung 1 ignores F3.
 Rung 1: 6 configs (F1×F2), 2 EFE, seed 2026; advance the registered control (1e-3, as-is) plus the top 2 others by the 2.0 value. GRU 0.6 / SASRec 1.1.
 Rung 2: 3 × F3 = 6 runs, 6 EFE, seed 2026; rank by the 6.0 value. GRU 1.9 / SASRec 3.3.
 Rung 3: top 2, seed 2027, 6 EFE; CKPT_AVG (uniform average of the 5.0/5.5/6.0 checkpoints) evaluated on both. GRU 0.6 / SASRec 1.1.
 Selection: tuned = higher 2-seed mean at 6.0; tie |Δ| ≤ 0.0005 → fewer deviations from the registered recipe. Adopt only if its seed-2026 rung-2 value exceeds the control's by > 0.0010, else KEEP_REGISTERED. CKPT_AVG carried to P4 iff it beats the endpoint by > 0.0005 (2-seed mean). Exclusions per G1 §D (non-finite, < 0.5× best). 14 runs per family, all in the HP registry, no hidden trials. Total ≈ 8.6 GPU-h, ≈ 6 h wall.
P3 Contest: MAIN, tuned recipes, seeds 2026+2027, 6 EFE, STRICT, marks every 0.5; G2 mechanical per T1; continuation to 8.0 only if STILL_IMPROVING. ≈ 7.2 GPU-h (+ ≤ 2.5), ≈ 4.6 h wall.
P4 Polish: per family and seed, C_BEST = higher MAIN_VALIDATION value of {6.0 endpoint, continuation endpoint if run, CKPT_AVG if carried}; 2-seed log-prob ensemble as SYSTEM. Evaluation only, ≈ 1 GPU-h; winner first, loser if budget remains.
P5 ALL_METHODS_LOCK: one TEST look — contest singles, C_BEST, SYSTEM, LEGACY_REPLAY, BASELINE; paired user bootstrap (1000, seed 20260926).
P6 OPTIONAL (owner): tuned winner on the full 5–100 pool (28.6M examples, ≈ 671k updates: GRU ≈ 11, SASRec ≈ 21 GPU-h per seed) — the only equal-data comparison with v1. Outside this budget.
Total P0–P5 ≈ 18–21 GPU-h, ≈ 1–1.5 days wall.

FAIRNESS & CLAIMS
May claim: (a) family X > Y under identical rows, protocol, tuning space, rungs, seeds and selection rule (T1 DECISIVE_STABLE, TEST CI excludes 0); (b) V4 winner vs LEGACY_REPLAY on identical TEST rows as a protocol-matched system comparison, asymmetries stated; (c) vs the non-neural baselines. May not claim: SOTA or any comparison with published REES46 numbers; "better architecture than v1" (confounded by data scale and input contract); equal-data superiority without P6; any TEST number before ALL_METHODS_LOCK; SYSTEM/C_BEST rows against a single legacy model as model-vs-model. Publish the tuning ledger (trials, budget, rule) with the result.

TRIGGERS
1. Any P2 run non-finite, both rung-3 configs excluded, or the tuned recipe needs code beyond TRAIN-02 scope.
2. Any MAIN_VALIDATION read before P3, or any edit to LOCK_V2/REGISTRY_V2/FS_FINAL/S*/DATA-06/v4eval.
3. LEGACY_REPLAY vocab mismatch or fingerprint failure.
4. Contest: any T1 trigger, HOLD, ARCHITECTURE_EQUIVALENT, or a continuation launch.
5. Budget > 125% of the P2/P3 estimates, disk-guard breach, or a non-bitwise STRICT run.
6. Owner asks for P6 or for shape changes.

LITERATURE (checked)
- Shehzad & Jannach 2023, RecSys — rankings flip under tuning; equal configurations per model → P2's identical space/rungs.
- Hidasi & Czapp 2023, RecSys — third-party GRU4Rec is flawed/untuned; take GRU tuning seriously.
- Rendle, Zhang & Koren 2019, arXiv — badly run baselines invert conclusions; legacy replayed as-built, limitation stated.
- Ferrari Dacrema et al. 2019, RecSys — tuned simple baselines beat most neural methods → P1(b) mandatory.
- Latifi, Mauro & Jannach 2021, Inf. Sci. — session kNN remains hard to beat → kNN baseline.
- Klenitskiy & Vasilev 2023, RecSys — SASRec with full-softmax CE beats BERT4Rec; our objective is right.
- Petrov & Macdonald 2023, RecSys (gSASRec) — gBCE fixes negative-sampling overconfidence; moot at full-K.
- Hägele et al. 2024, NeurIPS — constant LR + cooldown ≈ cosine; cooldown benefit plateaus near 20% of steps; (1−sqrt) > linear; SWA helps but a gap to cooldown remains → S1, CKPT_AVG as polish only.
- Sanyal et al. 2023, WANT@NeurIPS (LAWA) — spaced-checkpoint averaging at high LR beats EMA/SWA in LM pretraining → cheap test, not a certainty.
- Li et al. 2018, JMLR (Hyperband) — successive halving for budgeted search → rungs.

---
## FABLE AMENDED RULING (received after the owner correction; supersedes the sections it names; verbatim summary of the changed sections)
- Q5 WITHDRAWN (owner): v1 GRU/SASRec are LESSONS only; no v1 checkpoint is scored on V4 rows; no v1 number in any V4
  table. P6 (full-pool equal-data run) DROPPED.
- P1 baselines (popularity, last-item repeat, session item-kNN) fitted on MAIN_TRAIN_5_100_131072 events only, CPU, in
  parallel with P2/P3. Resource rule: the baseline job runs at low priority with <= 16 GiB RSS on the cores left after the
  trainers' ~3 threads each; if it breaches 16 GiB or lowers a trainer's step rate by > 5% (log_every monitor), Ops stops it
  and runs it after P3 — no Fable call. Scoring: per-row candidate scores densified (-inf for unscored items) through the
  frozen T3 evaluator (E2E, EVAL_ROWS_ONLY); label BASELINE, refused by selection.
- P2/P3/P4 as delivered; P5 one TEST look: GRU and SASRec contest singles (2 seeds each), C_BEST, SYSTEM, BASELINE.
  Total ≈ 17–19 GPU-h, ≈ 1–1.5 days wall (INFERENCE).
- FAIRNESS: equal effort = same CALIB rows, 14-run ladder, space size, seeds, selection/tie/adoption rules, evaluator, all
  trials hashed in the HP registry; budget equality is in EFE and trial count, not GPU-h (SASRec ≈ 1.9× slower) — state it.
  May claim: (a) V4 GRU vs V4 SASRec on MAIN TEST with CI only if T1 gives DECISIVE_STABLE (else T1's verbatim wording);
  (b) each family vs the baselines on the same rows; (c) C_BEST and SYSTEM against each other only. May not claim: anything
  vs v1 numbers; SOTA or published REES46 comparisons; claims from CALIB/HP_SEARCH rows; TEST before ALL_METHODS_LOCK; that
  the tuned recipe generalises to FL.
- TRIGGERS (replace the delivered list): 1 P2 non-finite / both R3 excluded / code beyond TRAIN-02; 2 any MAIN_VALIDATION
  read before P3 or any edit to LOCK_V2/REGISTRY_V2/FS_FINAL/S*/DATA-06/v4eval; 3 contest T1 trigger, HOLD,
  ARCHITECTURE_EQUIVALENT or a continuation launch; 4 budget > 125%, disk-guard breach, non-bitwise STRICT run; 5 a baseline
  that cannot go through the frozen T3 evaluator, or the owner asking for shape changes or a longer clock.
- Ops reconciliation: "in parallel with P2/P3" is read together with trigger 2 — baselines read MAIN_VALIDATION only from
  P3 on; during P2 their settings are tuned on CALIB_VALIDATION (PREG1_AMENDMENT_02 §4).
