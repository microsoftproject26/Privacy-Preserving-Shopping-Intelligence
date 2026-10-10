# REES46 deployment study: training recipes

The hyperparameters of every arm in [`results.md`](results.md), and how they were chosen. The data and populations are
in [`rees46-deployment-data.md`](rees46-deployment-data.md). The same recipe, transferred to the public benchmarks, is
`config/benchrun/registered_recipe.json`.

## Central models

All central arms train with:

* the exact full-catalogue cross-entropy;
* AdamW (0.9, 0.999, eps 1e-8, weight decay 1e-5 on matrices) with gradient clip 1.0;
* the warm-up and decay schedule `S0` of `ppsi.central.schedule` on an exposure clock of 6 epochs of exposure (EFE);
* evaluation every 0.5 EFE.

`practical_best` is the best of those 12 marks, and `controlled_budget` is the 6-EFE endpoint.

| Arm | Model | Peak learning rate | Trained on |
|---|---|---|---|
| `C_BIG` | SASRec d 256, 3 blocks (83.9 M parameters) | 4.0e-4 | the 131,072 federated users |
| `C_FULL` | the same | 4.0e-4 (not re-tuned at this scale) | 1,348,682 users |
| `C_SMALL` | SASRec d 64, 2 blocks (20.7 M parameters) | 7.11e-4 | the federated users |
| `PRE` | the same as `C_SMALL` | 7.11e-4 | the 1,217,610 server users; the warm start is the 6-EFE endpoint, one seed |

## Federated recipe

The cold federated recipe was tuned on the calibration users in stages. Each stage changed one thing and kept the
change only if calibration MRR@20 improved:

| Stage | Change | Calibration MRR@20 |
|---|---|---|
| 0 | FedAvg, client AdamW, the registered local learning rate | 0.0235 |
| 1 | local learning rate x 8 (x 4 and x 16 were worse) | 0.0288 |
| 2 | FedAdam server optimiser (step 0.01, `tau` 1e-3) | 0.0342 |
| 3 | constant round learning rate and client SGD with momentum 0.9, learning rate 0.1 | 0.0949 |
| 4 | one local pass instead of two, batch 16 | 0.1076 |

**The final cold recipe (`FA`).**

* FedAdam server: step 0.01, beta 0.9 / 0.99, `tau` 1e-3, `v_0 = tau^2`, no bias correction.
* Clients: a fresh SGD optimiser with momentum 0.9 on every visit, learning rate 0.1, gradient clip 1.0, one local
  pass of batch 16.
* Participation: 64 clients per round with 10 % dropout, for 13,654 rounds, which is 6 epochs of exposure in
  expectation.

**Method variants.**

* `FP`: FedProx with mu 0.01.
* `PF`: a personal query vector on the device (lambda 1e-4, clip 1.0) that is never uploaded.
* `FA_Q8`: 8-bit stochastic-rounding uploads, with one scale per tensor.

**Warm start.** Two registered screens from `PRE` chose the client and server learning-rate factors:

| Arm | Client factor | Server factor | What is trained and sent |
|---|---|---|---|
| `FA_WARM` | 0.25 | 0.1 | all weights |
| `FA_WARM_FROZEN` | 1.0 | 0.1 | the encoder only (263,822 values); the three item tables stay frozen |

The frozen variant scored within 0.0005 of the unfrozen one on calibration, and it sends about 80 times fewer bytes
per round. It is the deployable warm system and the subject of the primary contrasts.

## Differential privacy

DP-FedAvg with user-level RDP accounting (`ppsi.fedsim.accountant`):

* **Sampling.** Each client joins a round independently with probability q = 2^-7, about 1,024 clients per round,
  for 384 rounds with 2 local passes. The fixed denominator is round(q x N).
* **Noise.** The noise multiplier is z = 0.5642 for epsilon 8 and z = 1.3094 for epsilon 1, at delta 5e-6.
* **Clip norm.** The clip norm is the median client-update L2 norm over the first 20 rounds of the unclipped twin
  (`S_CAL`): 0.2844 on the cold line and 0.2754 on the warm frozen line. It is recorded before any DP run.
* **Non-private twin.** `FA_1024` uses the same schedule and clipping without noise. The DP contrasts compare
  against it, so they isolate the cost of the noise.
* **Not covered.** Selecting the best checkpoint on the users' validation data is outside the accountant, so the DP
  contrasts use the fixed endpoint.

## On-device arms

* **Fine-tuning (`FT_C`, `FT_PRE`, `FT_FA*`).** The base model is downloaded and fine-tuned on each device: a fresh
  AdamW optimiser, 0.3 x the small model's peak learning rate, 4 passes over the user's own decisions, batch 16.
  This cell was selected on the unseen validation users, so no federated user's values were read.
* **Local only (`LO`).** Each device trains its own model from random weights for 4 passes, with no collaboration.
  It is an untuned control, not the ceiling of device-only learning.
* **INT8.** Dynamic INT8 quantisation of the ONNX export (`ppsi.ondevice`).

## Phase 2 and the exploratory re-runs

* **`W_STAR`, phase 2.** The warm frozen-table recipe re-tuned on calibration: the server step was cut to 3e-6, with
  `tau` 1e-3. It was locked before its test look.
* **Tau-fixed warm (exploratory).** `tau` 1e-6, trainable item tables at 0.1 x the client rate, and a server step of
  3e-6 chosen on calibration from {3e-6, 1e-5, 3e-5}. The smallest step won; larger steps made the pretrained model
  worse.
* **Tau-fixed cold (exploratory).** `tau` 1e-6, a server step of 1e-3 chosen from {1e-4, 3e-4, 1e-3, 1e-2}, and the
  client rate unchanged.
* **100k-user pretrain (exploratory).** The same `PRE` recipe on 100,000 server users only. The tau-fixed warm run
  from it uses a server step of 3e-5, a client factor of 0.1, item tables at 0.1 x the client rate, and `tau` 1e-6.

See [`fedadam-tau.md`](fedadam-tau.md) for why `tau` matters.
