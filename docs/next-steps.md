# Next steps for the team

This page is the starting point for the team taking over the deployment study. It covers where the project stands,
which branch holds what and in which order to merge, how to reproduce the main tables, and what is left. The
evidence behind every number (run records, locks, logs and audit files) is in the project archive handed over
separately; this repository holds the code, the aggregate result tables and the documentation.

## Where the project stands

**Research question.** How much shopping-recommendation quality survives the move from a cloud model to federated
or on-device models?

**Confirmatory results.** These come from one preregistered analysis with a single test look, and they are final.

* **REES46, primary result.** The deployable warm federated system keeps **88.5 %** of the full-data central model's
  MRR@20 (CI 87.7-89.2). This system is the server-pretrained model improved by federated training with frozen item
  tables.
* **REES46, no gain from federation.** That system is 0.0005 *below* the pretrained model it started from, so
  federated training added nothing over a strong server pretrain. The phase-2 tuned recipe is equivalent to the
  pretrain within +-0.0005.
* **REES46, random start.** Federated training from random weights keeps 85.1 % of a central model on the same users,
  and 89.9 % at the same model size. Device-only training keeps 2.1 %.
* **REES46, side costs.** On-device fine-tuning adds a small, significant gain. 8-bit uploads and INT8 inference cost
  nothing measurable. DP at epsilon 8 costs 1.5 % on the random-start line and nothing on the warm line.
* **Public benchmarks.** On the six datasets the central model wins everywhere. With the registered recipe, the best
  federated arm keeps 13-77 % of it.

**The tau finding.** After the test look we found that FedAdam's `tau = 1e-3` was larger than almost every per-round
update value, so the federated runs barely moved. With `tau = 1e-6` the server steps are 20-100 times larger. See
[`fedadam-tau.md`](fedadam-tau.md).

**Exploratory findings.** These come after the confirmatory look and are labelled as such everywhere.

* **Benchmarks with the fix.** The fixed federated recipe keeps 75-92 % of the central model on the test split, and a
  central model of the device size keeps 90-110 %. The remaining 14-22 % gap is the cost of federation.
* **REES46 with the fix.** Over the strong pretrain there is still no gain: +0.0002, CI including 0.
* **The value of federation depends on the server's data.**
  * With a pretrain on only 100,000 users, federated training adds +0.009 MRR@20.
  * On the benchmarks a warm start pays off when the server band is large (+27-55 % at 50 % of the users), but not on
    MovieLens-1M.
  * Federated training at d 256 is no better than at d 64.
* **Privacy.**
  * A federated model that really learns leaks membership (AUC 0.75 from a random start).
  * DP at epsilon 8 removes the leak. Its cost is small from a pretrained start (95 % of utility kept) and large from
    a random start (29 %).
  * Simulated secure aggregation adds under 1 % upload overhead.

The full report is [`results.md`](results.md), and the tables are in `results/`.

## Branches and merge order

**Already merged into `main`:**

| Branch | Pull request | What it holds |
|---|---|---|
| `fix/fedadam-tau` | 103 | recipe validation of `fl.server_tau`, a unit test of the tau regime, `docs/fedadam-tau.md` |
| `results/rees46-final` | 104 | `results/rees46/`: confirmatory contrasts (K1-K21, Q1-Q11), arm values, exploratory looks, privacy audit, DP cost, secure aggregation, device latency; the `.gitignore` rule that admits `results/**/*.csv` |
| `results/benchmarks-final` | 105 | `results/benchmarks/`: Track A / B test results, the published comparison, the exploratory fixed-recipe test, the disjoint-band and capacity validation studies |
| `docs/final-results-report` | 106 | `docs/results.md`, `notebooks/final_results.ipynb`, the README link |
| `feature/benchmark-runner` | 102 | the benchmark runner; its last commit only merges the published history and changes no file, so the branch can be deleted |

**Still to merge, in this order.** Each branch already contains the current `main`, and the branches were
trial-merged in this sequence without conflicts. Open a pull request for each, let CI run and get it reviewed.

| Order | Branch | What it holds |
|---|---|---|
| 1 | `feature/secure-aggregation` | `ppsi.fedsim.secagg` and `secagg_grid`, tests, `docs/federated-secure-aggregation.md` |
| 2 | `feature/membership-inference-audit` | `ppsi.privacy` (the audit and per-user losses), tests, `docs/membership-inference-audit.md` |
| 3 | `feature/contrast-inference` | `ppsi.evaluation.contrast_inference` (Holm, equivalence, seed statistics, harm shares), tests, docs |
| 4 | `feature/fixed-benchmark-recipe` | `pre_start` in the benchmark recipe, the study's recipes in `config/benchrun/`, tests, docs |
| 5 | `docs/rees46-deployment-protocol` | `docs/rees46-deployment-data.md` (task, splits, populations) and `docs/rees46-deployment-recipes.md` |
| 6 | `docs/next-steps` | this page |

Some documents link to files that arrive in a later branch of the list, so the links are complete only after all of
them are merged.

## Reproducing the main tables

* **From the committed tables (no data needed).** Run `notebooks/final_results.ipynb`, which reads `results/` and
  redraws the figures. Every number in `docs/results.md` is in one of the CSV files listed in
  `results/rees46/README.md` and `results/benchmarks/README.md`.
* **Benchmarks from scratch (public data, CPU or one GPU).**
  1. Prepare the datasets as described in [`benchmark-datasets.md`](benchmark-datasets.md).
  2. Run the arms with `python -m ppsi.benchrun.run`, using the study's recipes in `config/benchrun/`. For the
     confirmatory numbers use `registered_recipe.json`; for the exploratory numbers use the `fixed_recipe_*.json`
     files. [`benchmark-runner.md`](benchmark-runner.md) lists the commands, and the CPU class is deterministic.
  3. Score the holdout once with `python -m ppsi.benchrun.run test` after selection.
* **REES46.**
  1. Get the frozen dataset as described in [`data-access.md`](data-access.md).
  2. Read the task, splits and populations in [`rees46-deployment-data.md`](rees46-deployment-data.md) and the
     recipes in [`rees46-deployment-recipes.md`](rees46-deployment-recipes.md).
  3. The training code is in the repository: `ppsi.seqrec`, `ppsi.central`, `ppsi.fedsim`, `ppsi.ondevice`,
     `ppsi.evaluation`. The study-specific launchers that bound it to the server's data layout are in the archive.
* **Privacy and systems.**
  * `python -m ppsi.fedsim.secagg_grid` reproduces the secure-aggregation table.
  * `ppsi.privacy.membership_inference.audit` runs the membership audit on per-user losses.
  * `python -m ppsi.fedsim.accountant` gives the DP noise multipliers.
  * The browser latency bench is in [`on-device-benchmark.md`](on-device-benchmark.md).

## What remains

1. **Thesis write-up.**
   * Use `docs/results.md` as the backbone.
   * Keep confirmatory and exploratory numbers apart.
   * State the tau finding and its consequence: the confirmatory federated results measure the registered recipe.
   * Present "the value of federated learning depends on the server-side data" as the main exploratory message.
   * Disclose the limitations listed at the end of `docs/results.md`.
2. **Optional exploratory test looks.** The disjoint server-band study, the capacity study (federated training at
   d 256) and the DP re-runs of the tau-fixed models have validation results only. Each can get one exploratory test
   look. Write down the runs and checkpoints before scoring, score once, and label the result exploratory beside the
   confirmatory tables.
3. **Possible additions.**
   * The training-free REES46 baselines (last item, category / brand rule, item-embedding kNN) as a package module.
   * An on-device fine-tuning cost measurement on phones.
   * A later-month holdout.
4. **Repository clean-up.**
   * Delete merged and stale branches; there are many early `s1-*` / `s2-*` branches.
   * Update `README.md` around the final results.
   * Check that CI passes on `main` after the merges.
   * Consider moving the phase-1 documents under `docs/phase-1/`.
