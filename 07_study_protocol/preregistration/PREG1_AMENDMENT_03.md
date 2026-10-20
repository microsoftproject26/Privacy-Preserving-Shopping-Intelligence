# PREG1 amendment 03 — downstream BIG/SMALL design (registration only)

- **Status:** REGISTERED by the Operations Director on 2026-09-27, BEFORE any P2 value has been read (P2 is running; Ops
  has read only STATUS/step-rate fields). Authority: owner directive DECISION_LOG 83 (+ AskUserQuestion: FL arms at the
  SMALL size) and Fable's ruling `company/reviews/DOWNSTREAM_BIG_SMALL_FABLE_RULING.md` (APPROVE_WITH_AMENDMENTS).
- **Unchanged:** P2 (02R1/02R2), P3 and G2 (T1 on the P3 runs), all locks, TEST sealing, the staggered-P3 seal rules.
- **Changes after the first P2/P3 value is read are refused** (a new amendment and a Fable call are needed).

## 1. Sizes (one family, two registered sizes)
- **BIG** = the G2 winner's family and size. **C_BIG = the P3 winner runs themselves** (no re-run).
- **SMALL** = the winner's family with the model dimension d reduced until the per-visit dense payload (down + up, FP32)
  is ≤ **170 MB** (today's GRU_D128 is 168 MB); everything else identical to BIG (blocks, heads, ff, tying, dropout).
  Client optimizer state ≤ 4× the shared bytes follows.
  - GRU → **GRU_D128** (registered). If GRU wins at D128, BIG = SMALL and the size step is zero by construction (a
    legitimate finding).
  - SASRec → **SASREC_D64_B2** (≈ 20.4M parameters, ≈ 164 MB per visit, 4 heads of dim 16; parameter-matched to GRU_D128).
    Registered now for conditional use: variant entry, θ0 for seeds 2026/2027 (server CPU build, C7 recentering),
    inventory, and a SYS-04c bench (TRAIN-04).
- Optional (owner): a tighter 85 MB tier (D64 for both families). Not adopted unless the owner asks before any P2 read.

## 2. Downstream arms (the matrix keeps internal equality)
All six matrix arms — **C_SMALL, FA, FP, PF, CPF, LO** — run at SMALL with ONE recipe, one θ0 per seed, MAIN, 6.0 EFE,
the frozen T3 evaluator. No FL/LO arm at any other size. C_SMALL must exist before any FL result is read.

## 3. Recipe for SMALL
- If SMALL is a size in P2 (GRU_D128): its per-size stage-2 winner (2 seeds); no extra tuning.
- Otherwise (SASREC_D64_B2): a registered 4-LR mini-ladder on CALIB at the four SASRec midpoints of 02R2 §1, schedule and
  dropout inherited from the SASRec winner, seed 2026, full 6.0 EFE; the best re-run at seed 2027; the same selection rule
  as 02R1 §3. Disclosed asymmetry (it only guards against overstating the size cost).

## 4. Decomposition and acceptability
- C_BIG → C_SMALL = the cost of the smaller model; C_SMALL → FL_SMALL = the cost of federation; FL_SMALL vs LO_SMALL =
  the value of collaboration.
- **Acceptability threshold: PENDING the owner** (must be registered before the first FL result is read). Fable's proposal:
  relative drop of client-macro MRR@20 (seed mean, 2 seeds) ≤ 2% point AND UB95 ≤ 3% by paired user bootstrap, for
  C_BIG → FL_SMALL and C_BIG → C_SMALL separately, decided on MAIN TEST at ALL_METHODS_LOCK. The loss is measured and
  reported in every case; the threshold only makes the verdict pre-registered.

## 5. Cost (INFERENCE)
Matrix at ≈ 21M parameters ≈ 58 GPU-h whichever family wins (vs ≈ 174 for SASRec at D256); C_SMALL 2 seeds ≈ 2.6;
mini-ladder ≈ 1.5; θ0/bench ≈ 0.3.

## 6. Triggers (Fable)
SMALL variant fails the bench or θ0 determinism; a C_SMALL 6.0 value < 0.5× C_BIG; any cap change after a P2/P3 read;
the threshold unregistered at the first FL read; SYS-02b at SMALL > 125% of ≈ 62 GPU-h; all earlier triggers stand.
