# BEST_BACKBONE plan — DRAFT for Fable (Ops, 2026-09-27)

Tags: M = PROJECT_MEASURED, I = INFERENCE, P = PLAN. Paths relative to `.local_ai/` unless marked V4 (= `ppsi_v4/`).

## 0. Owner directive (verbatim meaning, 2026-09-27)
"The GRU vs SASRec contest must take its time and produce the BEST result of both, using everything we learned, all
prior research and all prior experiments. I want to be stronger than the models previously built on the same data,
because I did high-quality work. Make a plan for this, you and Fable."
Replaces the 4-hour 'fixed recipe + 2-point LR calibration' contest (packet §D, DL 55) — not the locks.

## 1. What stays LOCKED (not reopened)
Population 5–100 and the bands; USER_HISTORY_L50; S* (24 input columns); catalogue A (K 158,486); full-TRAIN statistics;
DATA_PROTOCOL_LOCK_V2; T3 frozen eval numerics + EVAL_ROWS_ONLY; STRICT determinism; TEST sealed until
ALL_METHODS_LOCK; client-macro MRR@20 (E2E) as the headline metric; relative-only publication until ARCHITECTURE_LOCK.

## 2. The bar to beat (M, from earlier campaigns on the same REES46 v1 data, different population/window/protocol)
- GRU v1 F0: TEST macro MRR@20 0.168695 (v1 population, session-scoped window, 128-d tied scoring, 20.8M params of which
  20.3M item embeddings). SASRec v1 A1_RICH256_s2027 e5 (with cooldown): TEST 0.170475; best system S3 0.173224.
- These numbers are NOT comparable to V4 as they stand (different population, window, eval rows). A claim "stronger than
  the previous models" needs a protocol-matched comparison: the old checkpoints (kept on the server) scored on the SAME
  V4 decision rows (P1 below).
- Published REES46 numbers (e.g. Transformers4Rec-style session benchmarks) use other splits/metrics: cite only, no claim.

## 3. Lessons we already own (M unless tagged)
L1 SASRec v2 failed (VS 0.09–0.12); v3 fixed it (RICH features, schedule S_prime: warmup 2000 → constant → cosine to
   floor 1e-4). A cooldown phase added +0.0048 to +0.0049 (e3→e5) for both SASRec arms — the largest single gain we saw.
L2 GRU v1: F4 < F0 was confounded by the timing of the single LR step (plateau-triggered) → the schedule matters as much
   as the architecture; the current common schedule is v4train.schedule (warmup + constant + 0.01·peak end).
L3 Fable (GRU full-run review): GRU's bottleneck is the embedding/scoring dimension (128-d tied), not recurrent width;
   "pre-register F0-d256 instead" of h512.
L4 Seed-to-seed gaps are small (SASRec ≤ 0.0009) → 2 seeds resolve differences ≳ 0.002.
L5 Feature work: KEEP_CURRENT in v1; S* is −0.9% (GRU) / −0.13% (SASRec) vs L0S in V4 → features are settled.
L6 USER_HISTORY_L50: 87% of VAL rows now see TRAIN-period events (v1 saw only the session) → likely the largest V4 gain (I).
L7 Ensembles: SASRec A0+A1 (S2) gave ~0 over the best single; averaging epochs (S4a) hurt (−0.0018) → do not expect gains
   from naive epoch averaging; a 2-seed ensemble is a SYSTEM row, never the backbone.
Literature to apply (Fable to verify/extend): Rendle et al. 2019 and Dacrema et al. 2019 (equal tuning effort for every
   compared model); Hidasi & Czapp 2023 (GRU4Rec is badly served by untuned/third-party setups — tune GRU seriously);
   Klenitskiy & Vasilev 2023 and Petrov & Macdonald 2023 gSASRec (full-softmax CE — which we already use at full K — makes
   SASRec strong); Li et al. 2018 Hyperband / successive halving for budgeted tuning.

## 4. Proposed phases (P)
P0 G1 closes as ruled (INT-02 → R5 → G1_APPROVED). Unchanged.
P1 BASELINE BAR on the same V4 rows (MAIN_VALIDATION now, TEST only at ALL_METHODS_LOCK):
   (a) LEGACY_REPLAY: score the kept v1 checkpoints (GRU F0 best; SASRec A1_RICH256_s2027 e5) on the V4 MAIN decision rows
       with their OWN v1 input pipeline (session window, v1 features) — Opus task, evaluation only, ≈ 1 GPU-h (I);
   (b) non-neural: popularity, last-item repeat, session item-kNN (cheap CPU).
P2 EQUAL-BUDGET TUNING on CALIB (never MAIN_VALIDATION), identical trial count and rungs for both families,
   successive halving, all trials logged (no hidden trials):
   - space (shape-preserving first): peak LR {0.5, 1, 2}×1e-3; cooldown share of the budget {0, 20%, 30%} (cosine to
     0.01·peak); dropout {as-is, +0.1}; label smoothing {0, 0.1}; weight decay {1e-5, 1e-3};
   - shape changes (need θ0 registration + client-contract size check): GRU embedding/scoring d {128 → 256} (L3);
     SASRec d {256, 384} or blocks {2, 3};
   - rungs: 12 configs × 2 EFE → top 4 × 4 EFE → top 2 × 6–10 EFE (the budget question is decided here).
   - cost (I, from SYS-04b rates, CALIB ≈ ¼ MAIN): ≈ 12–15 GPU-h ≈ 7–8 h wall on 2 A16.
P3 CONTEST on MAIN with each family's tuned recipe, seeds 2026 + 2027, budget from P2, STRICT, marks every 0.5 EFE.
   Cost (I): at 6 EFE ≈ 10.7 GPU-h (≈ 6–7 h wall); at 10 EFE ≈ 18 GPU-h (≈ 11–12 h wall, SASRec GPU-bound).
   G2 mechanical per T1 (DECISIVE_STABLE); ARCHITECTURE_EQUIVALENT → HOLD (owner-pending GRU-on-cost).
P4 POLISH on the winner (and the loser, for the report): checkpoint/EMA averaging of the final cooldown marks (cheap; the
   v1 negative result was averaging across the LR schedule, not within a cooldown — test once on CALIB before use);
   2-seed ensemble as a SYSTEM row; optional G8 diagnostic (owner note).
P5 CLAIM at ALL_METHODS_LOCK: winner vs LEGACY_REPLAY vs non-neural baselines on identical MAIN TEST rows, paired user
   bootstrap. Before that, only relative VALIDATION statements.
Total (I): ≈ 1.5–2 days of wall clock instead of ≈ 4 h; LLM cost kept low with detached server queues.

## 5. Questions for Fable (decide; owner-proxy where needed)
Q1 Scope/fairness: is equal-budget tuning on CALIB (nested inside MAIN users, but different evaluation reads) acceptable,
   or must tuning use a band disjoint from MAIN (e.g. TB users, or the RESERVED_EXT band)?
Q2 Search space: approve / cut / add. Include shape changes (GRU d256, SASRec d384 or 3 blocks) or not? Any
   literature-backed item we miss (e.g. gSASRec negatives are moot at full-K CE; embedding tying; RMS norm; EMA)?
Q3 Budget: may the contest budget exceed 6 EFE (e.g. 10 EFE with cooldown)? If yes, does the downstream FL matrix
   (C/FA/FP/PF/CPF/LO) have to use the same budget (fairness; ≈ 1.7× FL cost) or is a separate C_BEST central reference
   allowed while FL uses the 6-EFE registered recipe?
Q4 Registration: the recipe is frozen (G0(b) C8; RunConfig refuses non-registered values). Approve a TUNING role +
   HP registry (TRAIN-02, Opus) and a PREG1_AMENDMENT_02 that registers the tuned recipes BEFORE the contest starts?
Q5 LEGACY_REPLAY: approve as the protocol-matched "stronger than before" comparison; any condition?
Q6 Does anything here change the G1 ruling (INT-02 scope stays closed; LR-CAL as ruled is replaced by P2)?
