# 05_privacy_and_security - DP mechanisms, leakage audits, secure-aggregation simulation

## Purpose

The privacy evaluation: the differential-privacy mechanism of the DP-labelled federated arms
(DP-FedAvg with Poisson sampling, clipping and Gaussian noise), the membership-inference audit,
the update-leakage audit, and the secure-aggregation simulation. The stage states plainly what
the evidence does not establish.

## Experiments, in order

1. **DP calibration (30 September 2026).** Poisson sampling with q = 2^-7, a fixed denominator
   of 1,024 clients, T = 384 rounds, delta = 5e-6; noise multipliers z = 0.5642 (nominal
   epsilon = 8) and z = 1.3094 (nominal epsilon = 1); clip norm S = 0.284363, the median
   client-delta norm over 20,625 recorded norms. A sampler/accountant mismatch was caught by
   review before any DP run (the first sampler would have given epsilon of about 1,388, not 8).
   Descriptive.
2. **DP utility cost in the confirmatory test evaluation (5-7 October 2026).** Confirmatory,
   at the fixed endpoint (controlled budget): nominal epsilon = 8 retains 98.52% [97.70, 99.38]
   of its non-noised twin (K12; per-seed 100.72 and 96.31, flagged seed-sensitive); nominal
   epsilon = 1 retains 94.38% [92.53, 96.00] (K13, one seed). On the warm line the DP contrasts
   sit within 0.15 points of 100 (K12w 100.00 [99.95, 100.07]; K13w 100.03 [99.92, 100.15]).
   Confirmatory (follow-up): Q6 100.00 [100.00, 100.00] and Q11 99.99 [99.98, 100.00]. These
   small costs are partly a symptom of a federated stage that barely moved; they are not a
   claim that DP is free.
3. **Membership-inference audit (4-5 October 2026, rerun at the closure).** 10,000 members
   against 10,000 non-members of the TRAIN window. On the confirmatory-run models, only the
   centrally fine-tuned model leaked (AUC 0.6255). Descriptive for those rows; exploratory for
   the tau-fixed models: the cold federated arm that really learns leaks (AUC 0.7466 [0.7400,
   0.7533], TPR at 1% FPR 0.1426, epsilon lower bound 2.112); nominal epsilon = 8 brings it to
   AUC 0.5019; the warm model from the 100,000-user pretrain shows no material leakage (0.5370
   [0.5295, 0.5453]). The audit's power ceiling (about 7.2) is below 8, so an epsilon = 8 run
   could never fail it.
4. **Closure DP8 test look (12-17 October 2026).** Exploratory, whole-recipe contrasts on TEST:
   the tau-fixed cold arm keeps 31.30% [30.39, 32.34] of its non-DP twin; the 100,000-user
   pretraining warm arm keeps 94.71% [94.10, 95.32]. A protection that looks free on a model
   that does not learn is not evidence that it is free in general.
5. **Update-leakage audit (technical closure).** Exploratory: a single client upload reveals
   the viewed item set almost exactly (item-set F1 of 1.0 against 0.0564 for a matched
   popularity baseline, an advantage of 94.36 percentage points); inside a 64-client aggregate
   the attack advantage is 2.625 pp (95% CI 2.312 to 2.953), above the fixed 2 pp threshold -
   a negative finding: secure aggregation is needed but not shown to be sufficient. A category
   probe found no material advantage, which is not proof of privacy.
6. **Secure-aggregation simulation.** Descriptive: 27 cells over full and sparse masking graphs
   and drop-outs of 0 to 0.3; the masked sum is bit-identical to the plain sum in every
   completed cell; upload overhead ratios 1.002 to 1.009 for the 263,822-value encoder update;
   one cell failed exactly when too few clients survived. A simulation of the arithmetic, not
   production cryptography. The canary design's ceiling (epsilon of about 5.213) is below 8, so
   it cannot reject epsilon = 8; the DP canary restart was stopped.

## Code map

| Package | What it does | Origin |
|---|---|---|
| `code/ppsi/privacy` | Secure-aggregation simulation and membership-inference audit of the public package | Main study (public port) |
| `code/e3_audit` -> `code/update_leakage_audit` (import name `e3_audit`) | Update-leakage audit: replay of the first 50 rounds, capture of aggregates, analysis, secure-aggregation features | Closure phase |
| `code/experiment_extensions/privacy` -> `code/closure_experiments/privacy` (import name `experiment_extensions.privacy`) | Closure-phase privacy extensions (amended replay cap, DP8 reruns) | Closure phase |
| `tests/prototype/privacy` | CPU tests of the privacy package | Main study |

The DP mechanism itself (clipping, noise, Poisson sampling, the accountant) lives in the
federated engine: [`../03_federated_learning/code/federated_core`](../03_federated_learning/README.md)
(import name `fl`) and
[`../03_federated_learning/code/ppsi/fedsim`](../03_federated_learning/README.md).

## How to run

```bash
uv run python 05_privacy_and_security/run.py --from-results   # rebuild this stage's tables and figures from saved results
uv run python 05_privacy_and_security/run.py --mode auto      # per unit: skip if complete, resume if supported, else start
uv run python 05_privacy_and_security/run.py --force          # redo from scratch (old outputs are moved aside first)
```

The audits replay federated rounds, so they need the frozen dataset under `PPSI_DATA_ROOT`, a
checkpoint to replay from (Release assets or `PPSI_RUNS_ROOT`) and, for the DP reruns, the GPU
environment. `--from-results` rebuilds the audit tables from the saved records without them.

## results/ and figures/

`results/` holds the curated privacy summaries - `privacy_dp_cost.csv`,
`privacy_membership_inference.csv`, `secure_aggregation.csv` - and the saved run records under
`results/runs/main_study/` and `results/runs/closure_phase/`. Per-user membership-inference
scores are deliberately excluded. `figures/` holds the rebuilt audit figures.

## Known limitations

No formal DP guarantee is established for the legacy DP runs: the implementation drew its noise
and Poisson streams from a disclosed public seed, so epsilon values are labels of the
configuration, and the arms are called DP-labelled, never private. Secure aggregation was
simulated only. Non-DP federated learning and local fine-tuning carry no formal privacy claim,
and an upload reveals the viewed items (the update-leakage audit measures it). The canary
audit cannot reject epsilon = 8 by construction. No production cryptography was implemented.
