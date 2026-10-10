# REES46 results

Final results of the deployment study on REES46 (multi-category e-commerce events, October-November 2019). Every
file here is an aggregate table: one row per arm, contrast or device cell. No user id and no per-user or
per-decision value is stored.

## Setup in one paragraph

The task is next-item recommendation over the full catalogue of 158,486 products. 131,072 users are the federated
population: their events stay on their devices, except in the central reference arms. 1,217,610 other users are a
server-side population, assumed to have opted in to server training; they train the pretrained model `PRE`. Every
learned arm is a SASRec model from `ppsi.seqrec`. The central reference is 256 wide with 3 blocks. The device model is
64 wide with 2 blocks (20.7 M parameters, most of them in the item tables). Federated arms run in `ppsi.fedsim`, with
FedAdam on the server, client SGD with momentum, one local pass, batch 16 and about 13,650 rounds of 64 clients.

## Metrics and statistics

* **Primary metric:** client-macro MRR@20 over the full catalogue. A user's reciprocal ranks are averaged over that
  user's decisions, then averaged over users. A target outside the vocabulary counts as a miss. NDCG@10 and HR@10
  are secondary.
* **Retention** `R(A / B) = 100 x A / B`. Contrasts are on the same users, with the same checkpoint rule and the same
  seed, averaged over seeds.
* **Uncertainty:** paired user bootstrap, 1,000 resamples, 95 % percentile intervals. Ratios are bootstrapped
  directly. `ci_vs_reference` says whether the interval lies above, below, or includes the reference (100 for a
  ratio, 0 for a difference).
* **Checkpoint rules:** `practical_best` is the best of 12 evaluation marks on the users' own validation data.
  `controlled_budget` is the fixed endpoint at 6 epochs of exposure. The DP contrasts use `controlled_budget`, because
  selecting a checkpoint on the users' data is not covered by the DP accountant.
* **Seeds:** 2026 and 2027, unless the `seeds` column says otherwise.

## Confirmatory test results (single test look)

The contrasts, their tiers and the analysis script were registered and locked before the test split was read, and
the test split was then scored exactly once. These numbers are final.

* **Test population:** 26,254 federated users with 229,742 test decisions.
* **Multiplicity:** the primary pair `K20b` and `K21` is read unadjusted. The 18 secondary contrasts form one
  Holm step-down family (`holm` column). Tertiary contrasts are unadjusted and descriptive.

| File | Content |
|---|---|
| `confirmatory_phase1_contrasts.csv` | the 33 registered contrasts `K1`-`K21` (and their warm-line variants), with point, CI, per-seed values and the Holm decision |
| `confirmatory_phase1_arms.csv` | MRR@20, NDCG@10 and HR@10 of every arm and baseline on the test split |
| `confirmatory_phase2_contrasts.csv` | the follow-up contrasts `Q1`-`Q9` and `Q11` |

**Headline.**

* The deployable warm system keeps 88.49 % of its matched central model (`K20b`, CI 87.73-89.18).
  This system is the pretrained model improved by federated training, with frozen item tables.
* That system is 0.000508 MRR@20 *below* the pretrained model it started from (`K21`, CI -0.000933 to -0.000037).
  On REES46, federated training on the private users adds nothing measurable over a strong server-side pretrain.
* Started from random weights, federated training keeps 85.11 % of a central model trained on the same users (`K2`)
  and 89.88 % at matched model size (`K3`). Device-only training keeps 2.1 % (`K8`).
* On-device fine-tuning adds +0.0023 MRR@20 over the pretrained model (`K9c`). 8-bit uploads (`K14`) and INT8
  inference (`K16`) cost nothing measurable.
* Differential privacy at epsilon 8 keeps 98.5 % on the random-start line (`K12`) and 100 % on the warm line (`K12w`).

**Phase 2** asked whether a tuned warm federated recipe (`W_STAR`) recovers a gain over the pretrained model. Its
hypotheses were written after the phase-1 validation read, its arms were selected on a separate calibration split,
and everything was locked before any test value existed. The registered reading of `Q1` is **equivalent**: the 90 %
interval of `W_STAR - PRE` lies inside the +-0.0005 margin. The Holm family `Q2`-`Q8` rejected nothing, and
`Q9 = R(W_STAR / C_FULL) = 88.72`.

Arm names in the tables:

| Arm | Meaning |
|---|---|
| `C_BIG` | central model (d 256) trained on the 131,072 federated users |
| `C_FULL` | central model (d 256) trained on 1,348,682 users (server users and federated users) |
| `C_SMALL` | central device-size model (d 64) trained on the federated users |
| `PRE` | device-size model pretrained on the 1,217,610 server users; the shipped warm start |
| `FA`, `FP`, `PF`, `FA_Q8` | federated training from random weights: FedAdam server, FedProx, personalised FedAvg, 8-bit uploads |
| `FA_WARM`, `FA_WARM_FROZEN` | federated training from `PRE`, all weights or with frozen item tables (only the 263,822 encoder values are sent) |
| `*_WARM_FROZEN` | the method on the warm, frozen-table line |
| `FA_1024`, `DP8`, `DP1` | Poisson-sampled training (about 1,024 clients per round, 384 rounds) with clipping, without noise and at epsilon 8 / 1 |
| `FT_*` | the named base model downloaded and fine-tuned on each device |
| `LO` | each device trains alone |
| `W_STAR` | phase 2: the tuned warm federated recipe (FedAdam server step 3e-6, frozen item tables, tau 1e-3), selected on the calibration split |
| `PRE_FT_CENTRAL`, `PRE_FT_CENTRAL_FROZEN` | phase 2: central fine-tuning of `PRE` on the federated users' raw events (the headroom), all weights or frozen item tables |
| `C_FULL_SMALL` | phase 2: the device-size model trained centrally on all 1.35 M users |
| `LAST`, `POP`, `SKNN`, `RULE`, `KNN_EMB_*` | training-free baselines |

## Exploratory re-runs (after the confirmatory test look)

> **Exploratory, after the confirmatory test look.** These runs were designed after the confirmatory results were
> known. Each exploratory test look locked its runs before scoring, and none of them replaces or changes a
> confirmatory number. Read them beside the confirmatory tables, never instead of them.

After the test look we found that FedAdam's adaptivity constant `tau = 1e-3` was larger than almost every per-round
update value, so the federated models barely moved (see [`docs/fedadam-tau.md`](../../docs/fedadam-tau.md)). The
re-runs use `tau = 1e-6`.

| File | Content |
|---|---|
| `exploratory_test_contrasts.csv` | three exploratory test looks, MRR@20, `practical_best` |
| `exploratory_warm_tau_grid_calibration.csv` | the warm, tau-fixed server-step grid on the calibration split |

The three looks in `exploratory_test_contrasts.csv`:

* **`tau_fixed_warm_and_cold`** re-runs both lines with `tau = 1e-6`.
  * Warm line, unfrozen tables (`FA_WARM_TAUFIX`): it is +0.00022 over `PRE`, with a CI from -0.00011 to 0.00057
    that includes 0, and it keeps 89.05 % of `C_FULL`. The confirmatory answer stands: with a strong pretrain,
    federated training adds nothing measurable. On the calibration split, larger server steps from the pretrained
    model made the model worse during training (`exploratory_warm_tau_grid_calibration.csv`).
  * Cold line (`FA_COLD_TAUFIX`): the re-tuned recipe is 0.0068 below the confirmatory cold model. Tau and the
    server step changed together, so this is not a pure tau effect.
* **`pb100_v1`** gives the server a weaker pretrain, `PRE100K`, trained on 100,000 users instead of 1.2 M. It uses the
  original recipe with `tau = 1e-3`, and one seed. Federated training from that pretrain adds +0.0074 (CI
  0.0064-0.0084). Retention against `C_FULL` rises from 73.8 to 77.9.
* **`pb100_v2_tau_fixed`** repeats that setting with `tau = 1e-6` and a tuned recipe, over two seeds. The gain over
  `PRE100K` grows to +0.0091 (CI 0.0081-0.0101), and retention reaches 79.0. That is still below the 1.2 M-user
  pretrain alone (-0.0179).

**Reading.** The value of federated training depends on how strong the server-side pretraining is. Over a strong
pretrain it adds about nothing. Over a weak pretrain, a correctly configured federated run closes part of the gap.

## Privacy audit and differential privacy

| File | Content |
|---|---|
| `privacy_membership_inference.csv` | user-level membership inference on final models |
| `privacy_dp_cost.csv` | utility cost of DP-FedAvg |
| `secure_aggregation.csv` | secure-aggregation simulation |

The audit, the secure-aggregation simulation and the device measurements below are descriptive. They report what
was measured and have no registered hypothesis or acceptance rule beyond the validity and pass checks stated here; the
rows labelled tau-fixed or 100k-pretrain were run after the test look and are exploratory as well.

**Membership inference.** The audit uses the training-window rows of 10,000 federated users (members) and 10,000
users the models never saw (non-members). Each user's score is their mean loss under the model minus their mean
loss under `PRE`. The threshold is set at 1 % false-positive rate on one half of the users and evaluated on the
other half. The audit is valid only if `PRE` itself scores at chance (AUC CI including 0.5), which it does.

* A model shows *material leakage* if the AUC CI lower bound is above 0.55 or the TPR at 1 % FPR is above 3 %.
* A DP model passes if the empirical epsilon lower bound does not exceed its accountant epsilon.
* The federated models from the confirmatory run (`FA_WARM_FROZEN`, `W_STAR`) show no material leakage, because they barely moved
  away from `PRE`.
* The central fine-tune `PRE_FT_CENTRAL` leaks (AUC 0.626).
* The tau-fixed cold federated model really learns from its users, and it leaks clearly (AUC 0.747, TPR 14 % at 1 %
  FPR). Its DP version at epsilon 8 passes (AUC 0.502).
* The tau-fixed warm model from the 100k pretrain leaks a little but stays below the threshold (AUC 0.537). Its DP
  version passes.

The rows of `PRE`, `FA_WARM_FROZEN`, `W_STAR` and `PRE_FT_CENTRAL` come from the first audit, of the models from the confirmatory run (a descriptive audit); the same rows in
the two later audits are identical and are listed once.

**DP utility cost.**

* *Confirmatory test, noise only.* Each DP run is compared with the twin that uses the same Poisson schedule and
  clipping but no noise: `K12`, `K13`, `K12w`, `K13w`, `Q6`, `Q11`. Noise alone costs 0-6 %.
* *Exploratory validation, the whole DP chain.* DP-FedAvg at epsilon 8 runs 384 Poisson rounds of about 1,024
  clients with clipping and noise; the non-private run uses the regular schedule. On the tau-fixed cold line it
  reaches only 29 % of the non-private run (two-seed mean validation MRR@20 0.0384 against 0.1306; the per-seed best
  values are 0.03804 and 0.03882 for DP8, and 0.13061 and 0.13067 for the non-private cold federated runs). On the
  warm line from the 100k pretrain it reaches 95 % (0.146 against 0.153). Both are validation readings made after
  the test look, so they are exploratory.
* So DP closes the leak, but its measured cost is large from a random start and small from a pretrained start.

**Secure aggregation (simulation).**

* The protocol is Bonawitz et al. (2017), semi-honest with four rounds, plus the SecAgg+ k-regular graph of Bell
  et al. (2020). The cryptography is simulated, and the message sizes are counted at their real values.
* The grid is 64 clients, dropout 0 / 10 / 30 %, a complete graph or k = 16 / 32 neighbours, and three seeds.
* Every decoded sum is bit-identical to the plain fixed-point sum. The upload overhead is under 1 % for the encoder
  update and about 0.01 % for the full model.
* One cell per update size fails to decode: k = 16 at 30 % dropout, seed 2028, where 8 shares are below the
  neighbourhood threshold of 9.

## Device feasibility

`device_latency.csv` holds in-browser inference of the phase-2 federated model, exported to ONNX and run with
ONNX Runtime Web: 68 runs on three devices, none failed. It reports latency, cold start and model size only, with
no accuracy.

* FP32 is 82.9 MB and INT8 is 21.2 MB.
* After loading, one recommendation takes about 2-5 ms at the median on a laptop and on a phone.
* Cold start (fetching the model file, creating the session and the first inference) takes 0.7-0.9 s on the laptop.
  On the phone it takes 2.5 s with INT8 and 4.9-5.2 s with FP32.
* INT8 costs nothing measurable in quality (`K16`).
