# 03_federated_learning - the federated simulator and the federated arms

## Purpose

The federated half of the study: the simulator (FedAdam server step, client sampling, clipping
and noise for the DP-labelled arms, 8-bit uploads), the strictly sequential federated recipe
tuning stages, every federated arm of the main study and the follow-up analysis, and the
exploratory tau-fixed re-runs that diagnosed the FedAdam scaling fault.

## Experiments, in order

1. **Prototype federated plumbing (August-September 2026).** Flower-based FedAvg with
   experiment contracts; a real-data smoke run reproduced an independent weighted average to
   within 2.384e-7, and the scoped T1 pilot kept 88.38% of a matched centralised pilot (0.112683
   against 0.099591 macro MRR on the category-change slice). Validation (pilot numbers are not
   comparable with main-study retentions). Descriptive for the plumbing checks.
2. **Federated recipe tuning stages (28 September - 1 October 2026), on CALIB.** Registered,
   strictly sequential screens, all descriptive: stage 0 FedAvg as registered 0.023473; local
   learning rate x8 (stage 1b) 0.028796; FedAdam server optimiser (stage 2) 0.034247; constant
   rate shape with client SGD (stage 3) 0.094936; one local pass (stage 4) 0.107562 - a
   cumulative factor of 4.58. Warm screens at the stage-4 point: all-weights warm training
   (stage 7) 0.170582; frozen item tables (stage 8) 0.170709, which became the warm recipe.
3. **MAIN federated runs (30 September - 3 October 2026).** Cold `FA` at two seeds, the warm
   arms `FA_WARM` and `FA_WARM_FROZEN`, fine-tuning arms, DP-labelled chains - all sealed with
   values unread before the single validation read.
4. **Single validation read (3 October 2026).** Validation (best-of-marks, optimistic):
   K20b 88.81 [88.27, 89.36]; K21 (warm frozen minus pretrained) -0.000433 [-0.000758,
   -0.000101]; cold line K2 86.07 [85.56, 86.62].
5. **Confirmatory test evaluation (5-7 October 2026).** Confirmatory: the realistic edge
   system `FA_WARM_FROZEN` keeps 88.49% [87.73, 89.18] of `C_FULL` (K20b); federated training
   on top of the pretrain changes MRR@20 by -0.000508 [-0.000933, -0.000037] (K21); the cold
   `FA` keeps 85.11% [84.30, 85.90] of `C_BIG` (K2) and 70.10% [69.19, 70.94] of `C_FULL`
   (K17); the pretrained start is worth +0.033119 [0.031774, 0.034516] over the cold federated
   model (K_INIT); 8-bit uploads retain 100.11% [99.85, 100.38] (K14). Confirmatory
   (follow-up): the tuned warm-start federated model minus the pretrain is -0.000081
   [-0.000179, 0.000024], EQUIVALENT within +/-0.0005 (Q1).
6. **Follow-up analysis runs (4-5 October 2026).** All 22 server-step grid cells fell within
   +/-0.00012 of the pretrain's CALIB score (0.171302), so the registered tie rule selected the
   smallest server step; the tuned warm-start federated model is close to the pretrain by
   construction. Descriptive.
7. **Tau-fixed re-runs, after the test evaluation.** Exploratory: the registered FedAdam
   stabiliser tau = 1e-3 was far larger than the update norms (~1e-7), and the tuned
   warm-start federated model moved only about 0.06% from the pretrain in 13,654 rounds. The
   tau-fixed warm run minus the pretrain is +0.00022 [-0.00011, 0.00057] (T2 look, B1), so the
   confirmatory answer is robust to the fault; the corrected cold arm is worse than the
   registered one (-0.00680 [-0.00778, -0.00587], D1). With tau fixed, federated arms reach
   75-92% of the cloud reference on the six public benchmarks (T1 look; reported with stage
   [`06`](../06_evaluation_and_benchmarks/README.md)).
8. **Server-data-size experiments.** Exploratory: with a 100,000-user pretrain, federated
   training adds +0.00743 [0.00644, 0.00838] (original recipe, v1) and +0.00911 [0.00810,
   0.01009] (tuned, tau-fixed, v2) over its own pretrain; the 100,000-user pretrain alone
   retains 73.79% [72.95, 74.66] of `C_FULL` against 88.77% for the full pretrain. The value of
   federation depends on how much data the server already has.

## Code map

| Package | What it does | Origin |
|---|---|---|
| `code/ppsi/federated` | Federated arms of the public package: client training, finetune, local-only, personalisation | Main study (public port) |
| `code/ppsi/fedsim` | The federated simulator: FedAdam and its tau, FedAvgM, Poisson sampling, clipping and DP noise, 8-bit stochastic-rounding uploads | Main study (public port) |
| `code/fl` | Research-tree federated engine: launchers, receipts, communication records, GPU receipts | Main study |
| `code/x_rees46` | Exploratory tau-fixed re-runs (`launch_fl_x*.py`, `reeval_x*.py`, `x_analysis.py`) and their register entries | Closure phase |
| `code/experiment_extensions/federated` | Closure-phase federated extensions (fine-tuning on moving models, more local passes, FedProx, personal head) | Closure phase |
| `scripts/prototype/federated`, `scripts/prototype/mvp` | Prototype federated scripts (smoke, scaling, pilots) | Prototype stage |
| `scripts/tau_fixed_lanes` | Shell launch scripts of the tau-fixed exploratory re-runs | Closure phase |
| `tests/prototype/{federated,fedsim,mvp}`, `tests/research/tests_fl{,_independent}` | CPU tests (including the FedAdam tau unit test) and research suites | Prototype / main study |

## How to run

```bash
uv run python 03_federated_learning/run.py --from-results   # rebuild this stage's tables and figures from saved results
uv run python 03_federated_learning/run.py --mode auto      # per training unit: skip if complete, resume if supported, else start
uv run python 03_federated_learning/run.py --force          # retrain from scratch (old outputs are moved aside first)
```

Federated training needs the frozen dataset under `PPSI_DATA_ROOT`, the server run trees under
`PPSI_RUNS_ROOT` and the GPU environment; on the original hardware a single federated run was
latency-bound (about 550 rounds per hour per job with ten jobs in parallel). The research code
is hash-pinned and is re-run from the flat tree recorded in
[`../07_study_protocol/`](../07_study_protocol/README.md). `--from-results` needs no GPU and no
data.

## results/ and figures/

`results/` holds the curated summaries of this stage, including
`results/exploratory_test_contrasts.csv` and
`results/exploratory_warm_tau_grid_calibration.csv`, and the saved run records under
`results/runs/main_study/` and `results/runs/closure_phase/`. `figures/` holds the rebuilt
tuning-ladder, retention and update-norm figures.

## Known limitations

The registered FedAdam stabiliser (tau = 1e-3) was badly scaled for the observed update sizes;
the code default keeps 1e-3 so the registered runs stay reproducible, and 1e-6 is recommended.
The tau-fixed re-runs changed several ingredients at once (tau with the server step, the
item-table policy or the budget), so no gain is attributed to tau alone. Frozen item tables
mean the warm arm never trains the (about 98% of) item-table parameters and scores zero for
targets outside the pretraining band. Without differential privacy an update reveals the viewed
items; "only weights leave the device" is data minimisation, not privacy.
