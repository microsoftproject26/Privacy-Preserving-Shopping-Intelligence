# Federated and on-device deployment study

This note describes the second phase of the project: measuring how much recommendation quality survives when the
shopping model moves from the cloud to federated training or to the user's device. It explains what we compare,
why, and how the comparison is kept fair. Results will be added once the evaluation is locked.

## The question

> How much shopping recommendation quality can be retained when moving from cloud-based to on-device or federated models?

We answer it as if the recommender were shipped inside a browser (for example Microsoft Edge), with three ways to
deploy the same model family:

| Deployment | Where the data goes | Where the model runs |
|---|---|---|
| Cloud | browsing events are sent to the server; one large model is trained there | on the server |
| Federated | events never leave the device; the device trains locally and sends only model updates | on the device |
| On-device | a small model is shipped once and, optionally, adapted on the device with the user's own history | on the device |

The headline number is **retention**: the quality of a deployment divided by the quality of the cloud model, reported
with a confidence interval.

## Data and task

- Public REES46 multi-category store events, October to November 2019.
- Task: predict the next product a user views, ranked against the full catalogue (158,486 products), no sampled negatives.
- Users with 5 to 100 training examples; each user is one simulated device.
- Chronological split: training up to 15 November, validation 16–22 November, test 23–29 November. The test week is
  evaluated once, after every method is locked.
- Metric: per-user MRR@20 averaged over users (NDCG@10 and HR@10 reported as well).

## Models

The backbone was chosen by a pre-registered comparison between a GRU and a SASRec (self-attention) sequence model,
each tuned with the same budget. SASRec was selected. Two sizes are used:

- **Large** (cloud reference): SASRec, hidden size 256, three blocks.
- **Small** (federated and on-device): SASRec, hidden size 64, two blocks, about 21M parameters, sized so that one
  download plus one upload of the model stays under 170 MB.

Two cloud references are reported: a cloud model trained on the same users as the federated system, and a cloud
model trained on the full user pool (about ten times more users), which is closer to what a real service would have.

## What is compared

| Group | Methods |
|---|---|
| Cloud | large model, small model, full-pool model |
| Federated | FedAvg, FedProx, FedAvg with a personal on-device component, FedAvg with 8-bit uploads |
| Federated with differential privacy | DP-FedAvg at ε = 8 and ε = 1 (user-level, Poisson sampling, RDP accounting) |
| On-device | local model trained only on the user's history; cloud model fine-tuned on the device; federated model fine-tuned on the device |
| Simple baselines | popularity, last item, session kNN, kNN on cached item embeddings, category/brand rules |

Device realism is measured separately: model size, payload per visit, and single-thread latency of the exported
ONNX model (FP32 and INT8), including a run in the browser with ONNX Runtime Web.

## Making the federated system as strong as a real one

A federated system trained with the settings of the central model is a weak baseline. Before any final run we tune
it in stages on a calibration group of users, each stage building on the best setting of the previous one:

1. client learning rate (a widening grid);
2. server optimizer (FedAdam, FedAvg with momentum);
3. learning-rate shape over rounds and the client optimizer (SGD with momentum vs. Adam);
4. local work per visit (passes and batch size);
5. a longer training budget;
6. on-device personalisation on top of the federated model;
7. starting federated training from a model pretrained on other users who share their data;
8. freezing the large item tables so that only a small part of the model is sent each visit.

Every stage is written down before its results are read, and the untuned starting point is kept as a lower bound.

## How fairness is kept

- The plan, the selection rules and the list of confirmatory comparisons are fixed before the numbers they govern are seen.
- All methods share the same data version, features, evaluator and training budget unless a stage says otherwise.
- Confidence intervals come from a paired bootstrap over users (1,000 resamples).
- Training and evaluation are deterministic (FP32, fixed seeds, two seeds per method).
