# G1 cost update (Operations Director, 2026-09-26T17:20Z)

## Measured (PROJECT_MEASURED)
- **Evaluation of the full M02 manifest** (EVAL-03 T3 GPU receipt: one A16, frozen T3 config, 808,610 eval rows):
  GRU 131 s and SASRec 180 s per evaluation. Peak allocated memory is 1.9 and 2.2 GB. The G0 estimate for SASRec was
  about 95 s, so the real cost is about 2×.
- **Evaluations per central run:** 12 marks at 0.5-EFE intervals up to 6.0, plus the endpoint, about 13. That is about
  28 min per run for GRU and about 39 min for SASRec.

## Revised central-contest estimate (INFERENCE)
Training uses SYS-01's synthetic step rates, and data loading is excluded.
- **GRU:** about 1.4 h of training plus 0.47 h of evaluation, ≈ 1.9 h per seed.
- **SASRec:** about 3.6 h of training plus 0.65 h of evaluation, ≈ 4.3 h per seed.
- **Contest, 2 families × 2 seeds:** ≈ 12.4 GPU-h, or about 6–7 h of wall-clock time with one family per GPU.
- **LR calibration** (2 peaks × 2 families × 2 EFE, evaluated on M06, a 4× smaller population): ≈ 3–4 GPU-h.
- **The data-loading rate on real batches is not yet measured.** It is the main uncertainty. The first central run
  records it, and the plan is updated after that.

## Downstream (unchanged order of magnitude)
The FL evaluations also follow the 0.5-EFE marks. SASRec adds ≈ 0.65 h per run for about 12 runs, i.e. ≈ 8 GPU-h
total. This is small against the FL training core (SYS-02).


## Update 2026-09-26T18:07Z: full-K step rates measured (SYS-04b)
- **PROJECT_MEASURED:** the real CentralTrainer on the A16 at K = 158,486 and B = 256, synthetic data, STRICT
  determinism. GRU runs at 17.51 updates/s and SASRec at 9.19 updates/s (2 reps each). Peak GPU memory is 0.77
  and 1.64 GiB.
- **INFERENCE:** a cohort run at 6 EFE is 91,569 updates.
  - Training takes GRU 1.45 h and SASRec 2.77 h. The G0 forecast was 1.4 and 3.6 h.
  - Adding about 13 evaluations (0.47 and 0.65 h) gives ≈ 1.9 h and ≈ 3.4 h per seed.
  - The contest (2 × 2) is ≈ 10.7 GPU-h. With one GRU seed and one SASRec seed per GPU, the wall
    clock is ≈ 5.3 h.
  - This excludes data loading. INT-01 measures whether real batches bound the step, which is still PENDING.

## Update 2026-09-26T19:23Z: real-data path measured (INT-01; smoke on the HISTORICAL population, infrastructure evidence only)
- **PROJECT_MEASURED:**
  - With the real cached provider the run is not data-bound: GRU runs at 0.986 and SASRec at 0.994 of the pure-GPU
    step, 20.9 and 10.0 updates/s at full K in STRICT mode.
  - The provider delivers about 91k rows/s and uses 202 bytes per cached row. Peak RSS is about 7.1 GiB per MAIN
    process.
  - The M06 eval (200,985 rows) takes 33 s for GRU and 46 s for SASRec.
- **INFERENCE:**
  - The per-run cost is linear in the cohort's examples: 6 × E / 256 updates, plus about 13 evaluations whose cost is
    linear in the VALIDATION rows.
  - Superseded-population example: the old cohort (3.9M examples) needed about 9.9 GPU-h for the contest.
  - For the 5–100 pool, rescale with CENSUS-02's exact examples for each candidate N: about 5.19M examples at
    N = 262,144, or 1.33× the old cohort.
