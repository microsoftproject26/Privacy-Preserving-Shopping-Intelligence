# ADR-004 — Architecture-selection tie-break, LR exposure point, frozen evaluation numerics (pre-G1; owner-ordered)

## Decision
Lock the three sections of `contracts/PREG1_LOCK_VALUES.json`:
- **T1:**
  - the cohort contest selects the family;
  - a predeclared any-of trigger (|Δ̄| < 0.0020; seed-inconsistent or |δ_s| ≤ 0.0002; |Δ̄| < 2 × the maximum seed gap;
    the bootstrap CI contains 0) starts a **staged full-TRAIN tie-break**;
  - stage 1 is seed 2026, and it is decisive iff |Δ| ≥ 0.0040 and the CI excludes 0;
  - stage 2 is seed 2027, with the two-seed rule;
  - the budget caps are 65 and 125 GPU-h;
  - EQUIVALENT means HOLD and ask the owner. The owner-proxy default (GRU on cost) is pending ratification.
- **T2:** `exposure_point = "end"` for every unit of work, with blocking pins P1–P7.
- **T3:** a single scoring block of K = 158,486, eval_batch 1,024 in manifest order, deterministic A16 FP32 flags,
  FP64 metrics, new comparison-key fields, and an ONNX parity-only study.

## Question
Owner directives: DECISION_LOG item 18.
- Does cohort selection find the genuinely strongest central architecture?
- Is the LR schedule matched across the paradigms?
- Are the evaluation numerics identical across methods?

## Options considered
- T1:
  - no tie-break;
  - an unconditional full-data contest;
  - a triggered tie-break with 1 seed, 2 seeds, or staged seeds.
- T2: the exposure point at the start, midpoint or end of each unit.
- T3: score_block at the default 32,768 or equal to K; a fixed or free eval batch.

## Project evidence
- The code facts are verified by both sides: `v4train/schedule.py` L12 and L87–91; `fl/schedule.py` L14–29 and L61–107;
  `v4eval/ranking.py` L13–33.
- EVAL-02's measured effect of the block width is ≤ 5.3e-5, and it moves near-tie ranks by 1.
- The v1 full-data seed gaps are ≤ 0.000869 (SASRec only).
- The costs come from COST_FORECAST.

## Opus and Fable positions
- `reviews/PREG1_OPUS_POSITION.md`.
- `reviews/PREG1_FABLE_POSITION.md`, including the step-B ruling.

## Agreement and disagreement
- T2 and T3 were agreed independently.
- T1 disagreed only on the tie-break seeds (Opus 1, Fable 2). It was resolved by Fable accepting a staged design with a
  stricter stage-1 threshold of 0.0040. See `reviews/PREG1_SCIENTIFIC_SYNTHESIS.md`.

## Rationale
- T1 keeps the matched downstream contract while guarding against a cohort-scale artifact in the family choice, and it
  saves compute when the evidence is decisive.
- T2 removes the zero-LR units and matches central exactly.
- T3 removes the block dependence and fixes GEMM shapes for every method.

## Consequences
- The tie-break manifests are built only if the trigger fires.
- The owner must ratify the EQUIVALENT fallback, or HOLD applies.
- Evaluation uses 649 MB per batch.
- Tiny LO clients skip the warmup.

## Validation required (G1 blocking)
- FL-02 implements pins P1–P7.
- TRAIN-01 implements the explicit exposure-point key, the frozen eval config, and the T1/T2 selection logic with tests.
- EVAL-03 provides the frozen-config loader (Part A) and the T3 GPU receipt (Part B, after the final DATA-04 server
  build).

## Thesis relevance
- Methodology: the selection protocol and its robustness check.
- Experimental setup: the LR discretization and the evaluation numerics.
- Limitations: cohort-scale selection, and the seed-spread evidence being SASRec-only.
