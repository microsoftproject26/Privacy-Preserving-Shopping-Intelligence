# Results

This page is the final report of the deployment study. The tables behind every number are in
[`results/rees46/`](../results/rees46/README.md) and [`results/benchmarks/`](../results/benchmarks/README.md). The
figures are drawn by [`notebooks/final_results.ipynb`](../notebooks/final_results.ipynb). The last section documents
the Quality Retention tool for the phase-1 run records.

## Research question

*How much shopping-recommendation quality can be retained when moving from cloud-based to on-device or federated
models?*

The study asks how much is retained, not whether a fixed threshold is met. A realistic edge deployment has three
options, each with a model small enough to run in the browser:

| Option | What leaves the device | Arms |
|---|---|---|
| **Cloud only** | raw events (training) and the query context (serving) | `C_BIG`, `C_FULL`, `C_SMALL` |
| **Federated** | model weights only, the encoder alone when item tables are frozen | `FA*`, `FP*`, `PF*`, `DP*` |
| **Small local model** | nothing after the download: a server-trained model, fine-tuned on the device | `PRE`, `FT_PRE`, `FT_*` |
| Device only (control) | nothing, and no shared model | `LO`, `LAST` |

## Setup

* **REES46.** Multi-category e-commerce events from October-November 2019. The task is next-item prediction over
  the full catalogue of 158,486 products.
  * 131,072 users form the federated population.
  * 1,217,610 other users form a server-side population. They are assumed to have consented to server training and
    train the shipped model `PRE`.
  * The full-data central model `C_FULL` sees both populations (1,348,682 users).
* **Models.** SASRec from `ppsi.seqrec`.
  * The central reference is 256 wide with 3 blocks (83.9 M parameters).
  * The device model is 64 wide with 2 blocks (20.7 M parameters, of which 263,822 are the encoder; the rest are
    item tables).
* **Federated simulation.** `ppsi.fedsim` runs FedAdam on the server, client SGD with momentum, 64 clients per round
  and 10 % dropout, for about 13,650 rounds.
  * DP-FedAvg uses user-level RDP accounting with Poisson sampling (q = 2^-7, 384 rounds, delta 5e-6).
  * 8-bit uploads, FedProx, personalised FedAvg, on-device fine-tuning and local-only training use the same code.
* **Public benchmarks.** Beauty, Sports, Toys and MovieLens-1M use the byte-identical eSASRec splits. Video Games
  and Baby Products use Amazon Reviews 2023. All are leave-one-out and run with the same recipe and code
  (`ppsi.benchrun`).
* **Metrics.**
  * REES46: client-macro MRR@20 over the full catalogue, with out-of-vocabulary targets counted as misses.
  * Benchmarks: full-catalogue NDCG@10.
  * Retention is `100 x arm / reference` on the same users and seeds.
  * Uncertainty: paired user bootstrap, 1,000 resamples, 95 % percentile intervals.
* **Preregistration and a single test look.**
  * The arms, contrasts, tiers, checkpoint rules and analysis script were registered before the results.
  * Every method was locked before the test split was read, and the test split was scored once.
  * The primary pair is `K20b` and `K21`. The 18 secondary contrasts are Holm-adjusted, and the tertiary ones are
    descriptive.
  * A phase-2 follow-up was registered after the phase-1 validation read and locked before its test values existed.
  * Everything done after the test look is labelled *exploratory* and reported beside the confirmatory numbers.

## Main results

### REES46 (confirmatory)

Retention against the full-data central model `C_FULL`. The 100 % reference is seed 2026 (MRR@20 0.1804), as in `K17`,
`K19` and `K20c`; the two-seed mean is 0.1801. `K20b` and `Q9` average the two seeds, each against its own `C_FULL`, and `FT_PRE` uses the two-seed arm means:

| Deployment option | Arm | Retention | Source |
|---|---|---|---|
| Cloud, full data | `C_FULL` | 100 | reference |
| Small local model: server pretrain, shipped | `PRE` | 88.8 | `K20c` |
| Small local model: pretrain plus on-device fine-tune | `FT_PRE` | 90.2 | arm values (descriptive) |
| Federated, warm start, frozen item tables | `FA_WARM_FROZEN` | **88.5** [87.7, 89.2] | `K20b` (primary) |
| Federated, phase-2 tuned recipe | `W_STAR` | 88.7 | `Q9` |
| Federated, random start | `FA` | 70.1 | `K17` |
| Device only | `LO` | 1.6 | `K19` |

* **The primary answer.**
  * The deployable federated system keeps **88.5 %** of the central model.
  * It is 0.000508 MRR@20 *below* the pretrained model it starts from (`K21`, CI -0.000933 to -0.000037).
  * The phase-2 tuned recipe is equivalent to the pretrained model within +-0.0005 (`Q1`).
  * On REES46, federated training on the private users adds nothing measurable over a strong server-side pretrain,
    while on-device fine-tuning adds a small, significant +0.0023 (`K9c`).
* **The random-start line** isolates the cost of federation.
  * It keeps 85.1 % of a central model on the same users (`K2`) and 89.9 % at matched model size (`K3`).
  * The smaller model itself costs 5.3 % (`K1`).
  * Collaboration matters: device-only training keeps 2.1 % (`K8`), and federated training beats the last-item
    heuristic by +0.051 (`K11b`).
* **Engineering costs are small.** 8-bit uploads (`K14`) and INT8 inference (`K16`) are within noise.
* **Multiplicity.** Holm rejected 15 of the 18 secondary contrasts in phase 1 and none of `Q2`-`Q8` in phase 2.

### Public benchmarks (confirmatory)

* With the recipe unchanged, the central model wins everywhere, and the best federated arm keeps 13-77 % of it.
* Against the published full-ranking numbers (Spotify's Table 1 on the same splits), our central SASRec is above
  eSASRec on MovieLens-1M (0.1951 against 0.1739) and below on the Amazon sets. With its best grid cell it is on par
  with eSASRec on Beauty and Toys.
* See [`results/benchmarks/README.md`](../results/benchmarks/README.md).

## The FedAdam tau finding

After the test look we found that FedAdam's adaptivity constant, `tau = 1e-3`, was larger than 99-100 % of the
per-round update values.

* In that regime FedAdam is server SGD with a step of `eta_s / tau` times a tiny update, so the federated models
  barely moved. The tuned warm REES46 model moved about 0.06 % from its start over 13,654 rounds.
* With `tau = 1e-6` the steps are 20-100 times larger.
* The mechanism, the measurements and the recommendation are in [`fedadam-tau.md`](fedadam-tau.md), with a unit
  test in `tests/fedsim/test_fedsim_fedadam_tau.py`.

Exploratory re-runs with the fix (after the confirmatory test look):

* **Benchmarks.** The fixed warm federated arm keeps 75-92 % of `C_FULL` on the test split, up from 13-77 %.
  The central device-size model keeps 90-110 %, so the remaining gap of 14-22 % is the cost of federation, not of
  model size.
* **REES46, strong pretrain.** The tau-fixed warm model is +0.00022 over `PRE`, and the CI includes 0. Larger server
  steps made the pretrained model worse. The confirmatory answer is therefore not a tau artefact.

## The value of federated learning depends on the server-side pretraining

Taken together, REES46, the 100k-user pretrain and the benchmarks give one picture.

* **Strong pretrain.** On REES46 the pretrain uses 1.2 M server users, and federated training adds about 0
  (`K21`, `Q1`, and the tau-fixed re-run).
* **Weak pretrain (exploratory, after the test look).** On REES46 with a pretrain of only 100,000 users, federated training adds +0.0074 with the
  original recipe and +0.0091 with the tau fix. Retention against `C_FULL` rises from 73.8 to 79.0, still below the
  1.2 M-user pretrain alone.
* **No useful pretrain (the retention figures are exploratory, after the test look).** On the benchmarks, the 15 % band pretrain is weak (9-54 % of `C_FULL`). There a correctly
  configured federated run adds most of the quality, at 75-92 % of `C_FULL`.
* **A real server-side data advantage (exploratory, after the test look).** On validation, with a server band disjoint from the federated clients, a
  warm start beats a cold federated run by 27-55 % on the Amazon sets when the band holds 50 % of the users, and by
  5-8 % at 15 %. It does not help on MovieLens-1M.

The more the server already knows, the less federated training adds.

## Privacy

* **Membership inference (descriptive).** The audit compares 10,000 members against 10,000 non-members, using a loss score
  calibrated by `PRE`. It is valid, because `PRE` scores at chance.
  * The federated models from the confirmatory run do not leak (AUC 0.50), but only because they hardly moved.
  * The tau-fixed cold federated model (exploratory, after the test look), which really learns from its users, leaks clearly: AUC 0.747, with a
    true-positive rate of 14 % at 1 % false positives.
  * Central fine-tuning of `PRE` on the users' raw events leaks too (AUC 0.626).
* **Differential privacy.** DP-FedAvg at epsilon 8 brings every audited model back to chance (AUC 0.497-0.504).
  * On the confirmatory test, the noise alone costs 1.5 % on the cold line (`K12`), 5.6 % at epsilon 1 (`K13`) and
    0 % on the warm line (`K12w`, `Q6`).
  * The whole DP chain (a shorter Poisson schedule with clipping and noise; exploratory, validation readings after the
    test look) is cheap from a pretrained start: 95 %
    of the non-private run on validation. From a cold start it is expensive: 29 %.
* **Secure aggregation (simulated, descriptive).**
  * Every decoded sum is bit-identical to the plain sum.
  * The upload overhead is under 1 % for the encoder update.
  * Of 27 cells per update size, one fails to decode: the sparse k = 16 graph at 30 % dropout.

Read together: federated learning that learns also leaks membership. DP at epsilon 8 closes the leak, and its cost
depends on the starting point.

## Device feasibility (descriptive)

* **Bytes.** With frozen item tables a client uploads only the encoder update: 263,822 values, about 1.06 MB per
  round in FP32 (0.26 MB with 8-bit uploads). Training the full model costs about 82.8 MB each way per visit.
* **Inference in the browser.** ONNX Runtime Web ran 68 runs on a laptop, a phone and a server CPU, with no failures.
  * After loading, a recommendation takes about 2-5 ms at the median.
  * Cold start (fetching the model, creating the session and the first inference) takes 0.7-0.9 s on the laptop.
    On the phone it takes 2.5 s with the 21 MB INT8 model and about 5 s with the 83 MB FP32 model.
  * INT8 costs no measurable quality (`K16`).

## Limitations

* **The registered recipe was not federated learning at its best.** The confirmatory federated numbers measure the
  registered recipe, whose `tau` froze training; the fix was found after the test look. The tau-fixed results are
  exploratory, and their recipes were chosen knowing the confirmatory outcome.
* **One pretraining seed.** The REES46 pretrain `PRE` has one seed, so contrasts against it carry no pretraining-seed
  variance. The benchmark warm arms include it.
* **`C_FULL` was not re-tuned at 1.35 M users.** Ratios against it may be biased upwards.
* **Some methods were inert by construction.** On the benchmarks 94-97 % of client visits were a single local
  step, so FedProx's proximal term had almost no effect; on REES46 too, `FP` matches `FA`. Frozen-table warm arms cannot rank items absent from the
  pretraining band (benchmarks).
* **Only part of the privacy analysis is formal.** Choosing the best checkpoint on the users' validation data is not
  covered by the DP accountant (the DP contrasts therefore use the fixed endpoint). Secure aggregation and the
  network are simulated, and bytes are counted rather than measured.
* **The benchmark comparisons have limits.** Retention on leave-one-out benchmarks is not comparable in magnitude
  with REES46. Published cells are single runs with a spread of about 0.003, and most federated-recommendation
  papers report sampled metrics.
* **Devices were measured for inference only.** Three devices measured inference; on-device training time and
  energy were not measured on phones.
* **One dataset, one period.** The REES46 results are from one retailer and one period. No later holdout month was
  evaluated.

## Reproducibility

* **Code.** `ppsi.seqrec` (models), `ppsi.fedsim` (federated simulation, DP accounting), `ppsi.evaluation.fullrank`
  (full-catalogue evaluation and the paired bootstrap), `ppsi.benchrun` (the benchmark arms) and `ppsi.ondevice`
  (ONNX export and the browser bench). Each package has its own document in `docs/` and its own test suite.
* **Determinism.** Strict FP32 and deterministic kernels. Seeds come from explicit parts. Resumed and retried runs
  reproduce bit for bit on the same device class.
* **Data.** REES46 access and checksums are in [`data-access.md`](data-access.md); the benchmark layout is in
  [`benchmark-datasets.md`](benchmark-datasets.md). No user-level data is in this repository.
* **Results.** The aggregate tables are in `results/`, and `notebooks/final_results.ipynb` redraws the figures from
  them with no data download.

## Phase 1 result registry (Quality Retention tool)

Phase 1 run records live locally under:

```text
artifacts/experiment-results/<run-id>.result.json
```

The `artifacts/` directory is ignored by Git. Each file must be an existing
`ExperimentResult v1`; this task does not define another result schema.

Validate all files and print a deterministic Quality Retention report:

```powershell
uv run --locked python scripts/experiments/results.py artifacts/experiment-results
```

Write the report to a local artifact instead:

```powershell
uv run --locked python scripts/experiments/results.py artifacts/experiment-results `
  --output artifacts/quality-retention.json
```

The command rejects invalid records, duplicate run IDs, duplicate metrics, and
ambiguous R1 baselines. A comparison uses the existing frozen compatibility
tuple plus the same seed, task, cohort, metric, direction, and unit.

QR is `comparison value / R1 value` only for approved `MAXIMIZE` metrics.
Log Loss, Brier Score, `MINIMIZE`, and `NEUTRAL` metrics remain absolute.
Missing R1 results and incompatible identities are reported explicitly without
inventing a comparison.
