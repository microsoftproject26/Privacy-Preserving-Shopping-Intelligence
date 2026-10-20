# Pre-G1 deliberations — Opus position. Independent memo, withheld from Fable until Fable answers.

The owner ordered these on 2026-09-26 (DECISION_LOG 18). Every threshold here must be frozen BEFORE any central
result exists.

## T1. CENTRAL_ARCHITECTURE_SELECTION_POPULATION — a full-TRAIN tie-break trigger
**Question.** The 262,144-user cohort is the required matched downstream population. Selecting the family on the
cohort is not the same as selecting the strongest centralized architecture on all of TRAIN. When should a full-TRAIN
GRU-vs-SASRec tie-break run?

**Proposal:**
- **Primary rule (unchanged).** Take the per-family mean over seeds 2026 and 2027 of the PRACTICAL_BEST client-macro
  MRR@20 on M02 (the cohort VALIDATION warm manifest). The higher mean wins.
- **Decisiveness test (frozen now).** Compute Δ_s = SASRec_s − GRU_s for each seed, and Δ̄ = the mean of the two.
  The cohort result is DECISIVE only if all three hold:
  1. sign(Δ_2026) = sign(Δ_2027) and both are non-zero (seed consistency);
  2. |Δ̄| ≥ τ, with τ = max(0.002, 2·σ̂), where σ̂ = the maximum over the two families of |MRR_2026 − MRR_2027|. This is
     a two-seed range, not an SD, and it is the same construction as the feature-study gate;
  3. the paired user bootstrap on M02 (1000 resamples, seed 20260926, shared weights, the seed-averaged difference)
     gives a 95% CI that excludes 0.
- **If DECISIVE:** the primary-rule winner is locked and the full-data tie-break is skipped.
- **If not DECISIVE: run the full-TRAIN tie-break:**
  - both families on ALL warm TRAIN users (40,180,608 examples; a new manifest M_FULL_TRAIN built with the same
    builder);
  - the same Clean V2 features, catalogue, schedule shape (6 EFE), executed optimizer values and strict FP32;
  - **seed 2026 only.** At full data the measured seed gaps were 0.0002–0.0009 (SASRec campaign), and the tie-break
    adds information about data scale, not replication;
  - evaluated on all warm VALIDATION users (642,228 users; 8,430,391 rows).
- **Cost** (INFERENCE, SYS-02): about 14.7 GPU-h (GRU) plus 37.4 GPU-h (SASRec), about 37 h wall with one family per
  GPU.
- **Tie-break decision rule:**
  - If |Δ_full| ≥ 0.002 and its paired bootstrap CI excludes 0, the full-data winner is locked, labelled
    SELECTED_BY_FULL_TRAIN_TIEBREAK.
  - Otherwise the cohort primary rule decides, labelled NOT_DECISIVE_COHORT_RULE.
- **Downstream is unaffected.** C for the downstream table is always the cohort central run of the winner (the
  matched reference). The full-data runs are architecture-selection evidence only, never a C_MATCHED denominator.

## T2. LR discretization
**Measured code facts:**
- Central: `v4train/schedule.py:12,87-90` gives an update the LR peak × s(EFE at the END of the update).
- FL: `fl/schedule.py:15-18,61-63` uses peak × s(EFE at round START), so round 0 has LR 0: θ1 = θ0, while its
  exposures still count.
- LO: `fl/schedule.py:19-21,66-74` gives step t the LR s(6t/T), so every LO client's first step has LR 0. A median
  client takes one step per pass, so it loses 1 of its 6 updates. That is material for LO.

**Proposal:** one convention for every unit of work (a central update, an FL round, an LO step): LR = peak × s(EFE at
the END of the unit's consumed exposures).
- FL round r: (exposures before r + Σ n_consumed planned for r) / N. The plan is deterministic, so the exposures are
  known before the round.
- LO step t of T: s(6·(t+1)/T).
- The PF personal LR has the same shape.

**Rationale:**
- It matches the central convention exactly, and it removes zero-LR units that consume budget without learning.
- EFE accounting is unchanged.

**Disclosure:**
- Granularity differs. The 0.02-EFE warmup spans about 305 central updates and about 41 FL rounds, but less than one
  LO step, so LO clients effectively skip warmup.
- The LO end-point convention gives a six-step client the sequence 1, 1, 0.1, 0.1, 0.1, 0.01.

**Alternative:** the midpoint. For FL it is negligibly different from the end-point, because a round is about
0.0005 EFE. For LO it also gives full LR on step 1. I rejected it only because it is a new convention not matched to
central.

**Tests:**
- round-0 LR = s(round-end EFE) > 0;
- the LO first-step LR = s(6/T);
- the central values are unchanged;
- one shared function is used by C, FL and LO.

## T3. eval_score_block freeze
Freeze one evaluation numeric configuration for C, FA, FP, PF, CPF and LO, and for every cold-start evaluation:
- `score_block = K` (158,486), a single block. It is bit-identical to the model's logits path
  (`v4eval/ranking.py:19`), so there is no block-dependent near-tie effect.
- `eval_batch = 1024`, fixed, because GEMM kernels can depend on the batch dimension.
- The same device class (A16), torch build and deterministic flags.
- Memory: 1024 × 158,486 × 4 B ≈ 649 MB per scoring block, which is fine on a 15 GiB A16.
- The ONNX device study uses its own numerics. It is a feasibility measurement, not a metric source. Any accuracy
  reported from ONNX is labelled as such and never mixed into the method tables.
