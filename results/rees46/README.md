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
