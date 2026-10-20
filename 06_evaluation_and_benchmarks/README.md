# 06_evaluation_and_benchmarks - evaluator, contrasts, public benchmarks

## Purpose

The measurement layer: the full-catalogue evaluator and the contrast statistics that produced
the confirmatory tables, the frozen scoring code records, and the six public benchmark datasets
that check whether the REES46 recipe transfers. Everything that turns trained checkpoints into
labelled numbers lives here.

## Experiments, in order

1. **Evaluator conventions (September 2026).** Ranking covers all 158,486 products with
   pessimistic ties and no sampled negatives; the primary metric is client-macro MRR@20 with
   out-of-catalogue targets counted as misses (end-to-end denominator); retention is
   100 x arm/reference; intervals are 95% paired user-bootstrap intervals (1,000 resamples);
   the secondary contrast family is Holm-adjusted and equivalence uses two one-sided tests.
   Descriptive conventions, frozen before use.
2. **Benchmark design and reproduction (1 October 2026).** Six public datasets in two waves -
   Amazon Beauty, Sports, Toys and MovieLens-1M byte-identical to the published release, Video
   Games and Baby Products from Amazon Reviews 2023 - under leave-one-out full-catalogue
   ranking with NDCG@10. Our copy reproduced three published baselines (MC, SeqRules, PCTM) in
   12 of 12 cells to four decimals. Descriptive.
3. **Confirmatory test evaluation (5-7 October 2026).** The contrast registry
   (`ANA_CONTRASTS_v4`, 33 items: two primary, eighteen secondary in one Holm family, thirteen
   tertiary) was scored once on REES46 TEST; the curated CSVs live in this stage's `results/`.
   Headlines (confirmatory): K20b 88.49% [87.73, 89.18]; K21 -0.000508 [-0.000933, -0.000037];
   K9c +0.002283 [0.001588, 0.003076]; and (confirmatory (follow-up)) Q1 -0.000081 [-0.000179,
   0.000024], EQUIVALENT within +/-0.0005. Absolute MRR@20 on TEST (seed 2026): `C_FULL`
   0.180388, `PRE` 0.160129, `FA_WARM_FROZEN` 0.159797, cold `FA` 0.126459, last-item 0.075569.
   Descriptive for the absolute values.
4. **Benchmark test looks (5-7 October 2026).** One look per dataset under the registered claim
   rule. Confirmatory for the Wave-1 sets, descriptive for the Amazon 2023 sets: the cloud
   reference `C_FULL` is the best arm on all six datasets; the best federated arm keeps 13% to
   77% of it under the matched-recipe protocol (13.20% on Toys, 77.06% on MovieLens-1M). On
   MovieLens-1M our `C_FULL` scores 0.1936, above the printed published eSASRec value of
   0.1739, so the retention ratios are measured against a reference that is not a straw man.
5. **Closure studies (12-17 October 2026).** Exploratory: the disjoint server-data study (a
   warm start beats cold federated learning on all four Wave-1 datasets at both server-band
   sizes, all eight intervals above zero; it beats a central model on the same users only at
   the 50% band); the capacity study (a larger device model shows no general benefit on TEST);
   and the training-free baselines on REES46 (a first-order Markov chain scores 0.1366
   client-macro MRR@20 on MAIN validation, a floor a cold federated model passes only after
   on-device fine-tuning).

## Code map

| Package | What it does | Origin |
|---|---|---|
| `code/ppsi/evaluation` | Evaluation harness of the public package: metrics, ranking, paired bootstrap, contrast statistics | Main study (public port) |
| `code/ppsi/benchmarks` | Definitions of the six benchmark datasets and their splits | Main study (public port) |
| `code/ppsi/benchrun` | The benchmark runner (matched-recipe and tuned protocols, one lock and one TEST look per dataset) | Main study (public port) |
| `code/v4eval` | The frozen research evaluator: metrics, ranking, manifests, hashing, bootstrap, device-class pinning | Main study |
| `code/v4bench` | Benchmark engine of the research tree: central and federated arms, recipes, selection, finals | Main study |
| `code/v4bench_w2` | Wave-2 (Amazon 2023) dataset conversion and verification | Main study |
| `code/experiment_extensions/baselines` | Closure-phase training-free baselines (Markov chain, sequential rules, PCTM) | Closure phase |
| `tools/tests_g2` | Tests of the contest decision tooling | Main study |
| `frozen_code/`, `archive/` | Records of the frozen scoring trees and archived run outputs (never edited) | Main study / closure phase |
| `scripts/prototype/{evaluation,experiments}`, `tests/prototype/{benchmarks,benchrun,evaluation,experiments}`, `tests/research/tests_{ana,bench,bench_w2,eval}` | Prototype scripts and tests, research suites | Prototype / main study |

## How to run

```bash
uv run python 06_evaluation_and_benchmarks/run.py --from-results   # rebuild every table and figure from saved results
uv run python 06_evaluation_and_benchmarks/run.py --mode auto      # per unit: skip if complete, resume if supported, else start
uv run python 06_evaluation_and_benchmarks/run.py --force          # redo from scratch (old outputs are moved aside first)
```

Scoring needs the checkpoints to score (Release assets or `PPSI_RUNS_ROOT`) and, for REES46,
the frozen dataset under `PPSI_DATA_ROOT`; the benchmark datasets are public and are converted
by `v4bench_w2` for the Amazon 2023 sets. The confirmatory scoring itself is a sealed,
one-time record: re-running a TEST evaluation is not part of the deliverable, and the frozen
scoring tree is kept in [`07_study_protocol`](../07_study_protocol/README.md).
`--from-results` rebuilds the tables and figures from the saved result files only.

## results/ and figures/

`results/benchmarks/` holds the benchmark tables (including `confirmatory_track_a.csv` and the
published-comparison tables); `results/rees46_confirmatory/` holds the confirmatory CSVs
(`confirmatory_phase1_contrasts.csv`, `confirmatory_phase2_contrasts.csv`, the arms table and
the follow-up records) with a README describing their provenance. Saved run records live under
`results/runs/main_study/` and `results/runs/closure_phase/`. `figures/` holds the rebuilt
retention, forest and benchmark figures. If a curated table and a primary result file ever
disagree, the primary file wins.

## Known limitations

Benchmark ratios are never pooled with REES46 retentions: leave-one-out splits share users
between training and test, and retention ratios there are not comparable in magnitude. The
Amazon 2023 sets are descriptive by rule. MovieLens-1M ran on the GPU class and the other
benchmarks on CPUs, so ratios compare only within a dataset. All benchmark values are
descriptive with respect to every REES46 claim. On the Amazon sets the cloud reference is not
state of the art against published non-neural baselines, which the report discloses.
