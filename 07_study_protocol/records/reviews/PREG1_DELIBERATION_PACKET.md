# Pre-G1 deliberation packet for Fable. Questions and facts only. Owner-ordered, 2026-09-26.

You are Fable 5.1, the Research Director. You are read-only and your ruling is binding.
- **Independence:** form your own view. Do NOT open `company/reviews/PREG1_OPUS_POSITION.md`. A synthesis follows your
  answer.
- **Root:** `F:\Queens University\Microsoft Project\eda\.local_ai\ppsi_v4\`.

**The owner's instruction, verbatim in substance:**
- The 262,144-user population is the required matched downstream population, but it is not automatically equivalent
  to selecting the genuinely strongest centralized architecture from all TRAIN data. Predeclare, BEFORE any central
  result, an exact trigger for a full-TRAIN GRU-vs-SASRec tie-break if the matched-cohort result is small or
  seed-inconsistent. If the cohort result is decisively stable, skip the full-data tie-break to save compute.
- Review the LR discretization: central applies s(EFE after update), while FL applies s(EFE at round start), which
  makes the first FL round's LR 0. Do not silently change it; decide the scientifically correct matched
  interpretation, document it, and lock it before experiments.
- eval_score_block must be frozen identically for C/FA/FP/PF/CPF/LO before G1.

## Facts (PROJECT_MEASURED unless marked)
- **Selection rule so far** (your G0 D1/D8 and G0(b) C3): the per-family mean over seeds 2026 and 2027 of the
  PRACTICAL_BEST client-macro MRR@20 on M02 (the 62,368 warm-cohort VALIDATION users) of the 262,144 cohort.
- **Seed variation evidence:** in the v1 full-data SASRec campaign the two-seed gaps were e3 0.000083, e4 0.000869 and
  e5 0.000216 (living reference §5.6e). The cohort has 9.7% of the examples, so the cohort seed variation is unknown
  (INFERENCE: probably larger).
- **Cost** (`g0/COST_FORECAST.json`, INFERENCE from measured step costs): central 6 EFE on ALL costs 14.7 h (GRU) and
  37.4 h (SASRec) per seed on one A16, strict FP32. On the cohort it costs 1.4 and 3.6 h.
- **Full-data evaluation population:** 642,228 warm VALIDATION users, 8,430,391 rows (`g0/data_census/*`).
- **Code:**
  - Central: `code/v4train/schedule.py` L12 and L87–90: "The LR of an update is peak * s(EFE at the END of that
    update)".
  - FL: `code/fl/schedule.py` L15–21 and L61–74. FA/FP/PF use round_lr = peak × s(EFE at round start), so round 0 runs
    at LR 0 (θ1 = θ0) while its exposures still count; the warmup spans about 41 rounds. LO step t of T uses
    s(6·t/T), so every LO client's first step is at LR 0, and a median client (one step per pass) loses 1 of its 6
    updates.
  - s knots: (0, 0), (0.02, 1), (2.0, 1), (3.0, 0.1), (5.5, 0.1), (6.0, 0.01).
- **Evaluator** (`code/v4eval/ranking.py` L12–19, L33; handoff `company/handoffs/EVAL-02.md`):
  - chunked FP32 scoring is not bit-identical to a full matmul (≤ ≈ 5e-5, which can move near-tie ranks by 1);
  - `score_block ≥ K` is a single addmm, bit-identical to the logits path;
  - the current default is 32,768.

## Decisions requested
- **T1.** The exact predeclared trigger: statistics, thresholds, seed handling. Also the tie-break design (seeds,
  population, evaluation set, budget), its decision rule when it is itself inconclusive, and how its outcome relates to
  the downstream matched C (which stays the cohort run).
- **T2.** The locked exposure point for the LR schedule for central, FL (FA/FP/PF, including the personal LR) and LO,
  with its rationale and the tests that must pin it.
- **T3.** The frozen evaluation numeric configuration for all methods (score_block, eval batch, device and
  determinism), and how the ONNX device study relates to it.

## Return format (target ≤ 1,000 words)
ROUTING, VERDICT, a DECISION with LOCK_VALUES for each topic, and OPEN_DISAGREEMENT_RISKS. Read-only.
