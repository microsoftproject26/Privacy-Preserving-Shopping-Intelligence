# Practical Federated-Learning Implementation Notes

**Logical ID:** `FederatedImplementationNotes`  
**Artifact version:** `v1`  
**Task:** `S1-PR-04` / Issue `#17`  
**Canonical path:** `docs/papers-notes.md`  
**Source access date:** `2026-08-27`  
**Scope:** Translate the project's four required research sources into implementation decisions.  
**Non-goal:** This is not a general literature-review essay and does not freeze paper hyperparameters as project defaults.

---

## 1. How to read this document

Each source is split into two layers:

1. **Paper facts** — claims supported by the cited source.
2. **Project choices** — decisions for this repository, derived from the project scope and frozen contracts.

The decision labels are:

- **ADOPT** — use the idea or guardrail in the project.
- **DEFER** — retain it for a later approved task/regime; it is not a current Phase-1 requirement.
- **REJECT** — explicitly do not use it as a default or scientific claim.

A paper's experimental settings are reference parameters, not project defaults. Every final value must be explicit,
versioned, and justified by project evidence.

---

## 2. Executive decision summary

| Topic | Project decision | Phase boundary |
|---|---|---|
| FedAvg | Use standard FedAvg as the `R2A` federated reference. | **ADOPT for Phase 1** |
| Client aggregation weight | Keep the weight key and its semantics explicit. The final scientific weighting policy is owned by client-partition/sampling freeze work, not copied from a paper. | **ADOPT interface; DEFER final policy** |
| FedAdam | Preserve a clean FedOpt/FedAdam configuration path for `R2B`; do not promote `R2B` into the current two-week Phase-1 core without a separate approved decision. | **ADOPT later regime; DEFER Phase-1 execution** |
| Personalization layers | Use explicit shared/local parameter ownership as the design principle for `R3`. | **ADOPT contract principle; DEFER execution** |
| R5 relationship | Keep `R5` separate from FedPer-style R3: R5 starts from a trained R1 checkpoint and then adapts locally. | **ADOPT distinction** |
| T1 evaluation | Evaluate T1 exhaustively over the frozen training-observed category vocabulary. | **ADOPT** |
| T3 evaluation | Consume the approved, frozen T3 candidate protocol; do not use per-regime random negatives or hide retrieval misses. | **ADOPT guardrail; exact T3 semantics owned elsewhere** |
| Privacy wording | Describe data locality/reduced raw-data centralization, not formal privacy, unless DP/secure aggregation is actually implemented. | **ADOPT limitation** |
| Paper/library defaults | Never promote paper or Flower defaults to scientific defaults without recording and approval. | **REJECT** |

---

# 3. McMahan et al. — Federated Averaging

## 3.1 Source identity

- **Title:** Communication-Efficient Learning of Deep Networks from Decentralized Data
- **Authors:** H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Agüera y Arcas
- **Venue:** AISTATS 2017, PMLR 54:1273–1282
- **Stable identity:** `arXiv:1602.05629v4`
- **DOI:** `10.48550/arXiv.1602.05629`
- **Primary sources:** [PMLR][FAVG-PMLR], [arXiv v4][FAVG-ARXIV]

## 3.2 Paper facts

1. The work defines federated learning around keeping raw training data distributed on devices and learning a
   shared model from locally computed updates. `[FAVG-1]`
2. It presents FedAvg as iterative model averaging with multiple local updates before communication. `[FAVG-2]`
3. Unbalanced and non-IID client data are treated as defining characteristics of the setting, not exceptions.
   `[FAVG-3]`
4. Communication is treated as the principal systems constraint; the reported experiments reduced required
   communication rounds substantially relative to synchronized federated SGD. `[FAVG-4]`
5. The main reference controls are client participation fraction `C`, local epochs `E`, local minibatch size `B`,
   client learning rate, and number of rounds. `[FAVG-5]`
6. The standard example-weighted form gives larger aggregation weight to clients contributing more local training
   examples. `[FAVG-6]`

## 3.3 Assumptions and limitations relevant to this project

- The reference algorithm is synchronous and round based.
- Selected clients receive a common global model before local training.
- Clients may have different local dataset sizes and distributions.
- The paper is not a complete failure/retry, malicious-client, formal-DP, or secure-aggregation specification.
- The paper's datasets and model architectures differ from user-as-client, multi-task REES46 training.

## 3.4 Evidence-to-implementation table

| Topic | Paper evidence | Project choice | Status | Exact code/config consequence |
|---|---|---|---|---|
| Raw-data locality | `[FAVG-1]` | Preserve one canonical user as one logical client in federated regimes. | **ADOPT** | S1-PR-06 must enforce user isolation; no repartitioning to manufacture IID clients. |
| Standard algorithm | `[FAVG-2]` | `R2A` is standard FedAvg. | **ADOPT** | Use Flower's built-in `FedAvg` with only thin project-specific tracing/validation. Do not reimplement generic averaging. |
| Non-IID/unbalanced data | `[FAVG-3]` | Measure real heterogeneity instead of hiding it. | **ADOPT** | Persist client event/example distributions and per-round participation traces. |
| Communication trade-off | `[FAVG-4]` | Treat local work and communication as separate measured axes. | **ADOPT** | Version local epochs/steps and later join actual upload/download measurement; never substitute parameter-count estimates. |
| Participation | `[FAVG-5]` | Keep eligible pool, clients/round, replacement, retry, and failure semantics outside this note. | **DEFER** | S1-PR-06 freezes sampling/weighting policy and emits `ClientSamplingTrace v1`. |
| Local work | `[FAVG-5]` | Keep local epochs/steps, batch size, optimizer, and client LR explicit. | **ADOPT** | No hidden constants in the Flower adapter; values come from validated ExperimentConfig/trainer config. |
| Aggregation weight | `[FAVG-6]`, `[FLWR-FAVG]` | Use one explicit versioned weight field. Do not equate raw event count with training contribution. | **ADOPT interface; DEFER final semantics** | Flower `weighted_by_key` is configured explicitly. Log total local examples used and per-task contributor/example counts separately. |
| Paper hyperparameters | Experimental settings, not universal claims | Do not copy `C/E/B/lr` values. | **REJECT** | Project validation and approved runtime/protocol evidence choose final values. |
| Formal privacy claim | Outside the paper's guarantee | Do not state that plain FedAvg provides formal privacy. | **REJECT** | Approved wording is data locality/reduced raw-data centralization. |

## 3.5 Project parameter checklist for R2A

The eventual R2A ExperimentConfig must identify at least:

- strategy ID/version;
- client optimizer and learning rate;
- local epochs or local steps;
- local batch size;
- eligible-client manifest hash;
- sampler version and seed derivation;
- clients per round;
- aggregation weight key and definition;
- number of server rounds;
- evaluation cadence;
- timeout/failure/retry behavior;
- CommonInitialization identity;
- data/split/example/evaluator/environment identities.

This list is a configuration requirement, not a set of chosen numerical defaults.

---

# 4. Reddi et al. — Adaptive Federated Optimization

## 4.1 Source identity

- **Title:** Adaptive Federated Optimization
- **Authors:** Sashank Reddi, Zachary Charles, Manzil Zaheer, Zachary Garrett, Keith Rush, Jakub Konečný,
  Sanjiv Kumar, H. Brendan McMahan
- **Venue:** ICLR 2021
- **Stable identity:** `arXiv:2003.00295v5`
- **DOI:** `10.48550/arXiv.2003.00295`
- **Primary source:** [arXiv v5][FOPT-ARXIV]
- **Implementation reference:** [Flower FedAdam API][FLWR-FADAM]

## 4.2 Paper facts

1. FedOpt separates a **client optimizer** from a **server optimizer**. Clients perform local optimization, while
   the server applies an optimizer to the negative average model delta, treated as a pseudo-gradient. `[FOPT-1]`
2. FedAvg is a special case of this framework with SGD-like server behavior and server learning rate 1.
   `[FOPT-2]`
3. The paper introduces federated adaptive server optimizers including FedAdagrad, FedAdam, and FedYogi.
   `[FOPT-3]`
4. Keeping adaptive state on the server preserves cross-device compatibility and does not require clients to
   transmit optimizer state. `[FOPT-4]`
5. The relevant controls include client learning rate `eta_l`, server learning rate `eta`, `beta_1`, `beta_2`,
   `tau`, local steps, client sampling, and aggregation weighting. `[FOPT-5]`
6. The paper reports strong empirical results but does not establish that FedAdam must outperform FedAvg on every
   project or dataset. `[FOPT-6]`

## 4.3 Evidence-to-implementation table

| Topic | Paper evidence | Project choice | Status | Exact code/config consequence |
|---|---|---|---|---|
| Client/server split | `[FOPT-1]` | Make optimizer ownership explicit now. | **ADOPT** | Separate `client_optimizer` and `server_optimizer` records; never use one ambiguous `learning_rate`. |
| R2A relation | `[FOPT-2]` | Keep R2A as plain FedAvg. | **ADOPT** | Do not silently add server momentum/adaptivity to R2A. |
| FedAdam | `[FOPT-3]` | Keep FedAdam as the named `R2B` later regime. | **ADOPT later; DEFER Phase-1 execution** | Unified trainer/Flower adapter must support strategy injection without a second training core. |
| Adaptive state | `[FOPT-4]` | Server optimizer state is server-owned and checkpointed with the run. | **ADOPT** | Round checkpoint records server moments/state, round index, config hash, init hash, sampler/RNG state. |
| FedAdam controls | `[FOPT-5]`, `[FLWR-FADAM]` | Every control must be explicit; no unrecorded library defaults. | **ADOPT** | Record `eta`, `eta_l`, `beta_1`, `beta_2`, `tau`, weight key, and strategy version. |
| Controlled comparison | Framework logic | R2B should match R2A on client optimizer/local work unless a separate approved ablation changes them. | **ADOPT** | The compatibility/preflight layer rejects unintended drift in shared scientific fields. |
| FedYogi/FedAdagrad | `[FOPT-3]` | Do not add extra regimes in the current plan. | **DEFER** | They remain literature-supported future alternatives only. |
| Paper/Flower defaults | Reference settings | Do not freeze them automatically. | **REJECT** | A future R2B protocol records the chosen search space and validation-only selection. |
| Guaranteed superiority | `[FOPT-6]` | Do not claim FedAdam is inherently better. | **REJECT** | Report quality, convergence, communication-to-target, failures, and resources. |

## 4.4 Phase boundary

This note does **not** promote FedAdam into the current two-week Phase-1 core. It only freezes the engineering
interpretation required so that a later approved R2B experiment uses the same trainer, data, initialization, and
evaluator as R2A.

---

# 5. Arivazhagan et al. — Personalization Layers / FedPer

## 5.1 Source identity

- **Title:** Federated Learning with Personalization Layers
- **Authors:** Manoj Ghuhan Arivazhagan, Vinay Aggarwal, Aaditya Kumar Singh, Sunav Choudhary
- **Publication:** arXiv, 2019
- **Stable identity:** `arXiv:1912.00818v1`
- **DOI:** `10.48550/arXiv.1912.00818`
- **Primary source:** [arXiv v1][FPER-ARXIV]
- **Implementation cross-check:** [Flower FedPer baseline][FLWR-FPER]

## 5.2 Paper facts

1. FedPer addresses statistical heterogeneity through a base/shared component and client-specific personalization
   layers. `[FPER-1]`
2. Shared/base layers participate in federated learning while personalization layers remain local to each client.
   `[FPER-2]`
3. The paper evaluates deep feed-forward networks on non-identical CIFAR partitions and a personalized Flickr
   aesthetics dataset. `[FPER-3]`
4. The paper does not determine the correct layer split for a GRU-based, multi-task shopping model. `[FPER-4]`

## 5.3 Evidence-to-implementation table

| Topic | Paper evidence | Project choice | Status | Exact code/config consequence |
|---|---|---|---|---|
| Shared/local split | `[FPER-1]` | Use explicit parameter ownership as the R3 design principle. | **ADOPT contract principle** | Every trainable parameter is classified as shared/global or personal/local by a versioned ModelConfig. |
| Local state boundary | `[FPER-2]` | Local personalization state is never aggregated or sent to other clients. | **ADOPT** | Flower payload filtering and tests must prove local-only keys are absent. |
| Model-specific split | `[FPER-3]`, `[FPER-4]` | Do not copy the paper's layer placement. | **REJECT** | Adapter/head location and size freeze only after project architecture evidence. |
| Task heads | No project-specific support in the source | Do not automatically make T1/T2/T3 heads personal. | **REJECT** | Head ownership requires an explicit approved R3 ModelConfig. |
| Adapter architecture | Outside current evidence | Defer location, dimension, initialization, LR, steps, and storage policy. | **DEFER** | Later R3 work owns these choices. |
| Cold/new client | Personalization requires local state/data | Preserve an explicit global/unadapted fallback. | **ADOPT requirement** | R3 evaluation must distinguish before-adaptation and after-adaptation behavior where applicable. |
| R5 | FedPer is not R5 | Keep R5 as trained-R1 checkpoint plus local adaptation. | **ADOPT distinction** | R5 uses `PRETRAINED_R1_CHECKPOINT`, not the untrained common-init path used by R1–R4. |
| Privacy claim | Data locality is not a formal guarantee | Do not equate personalization with formal privacy. | **REJECT** | Keep the threat/privacy documentation explicit about leakage from shared updates. |

## 5.4 Minimum R3 interface requirements retained for later work

- parameter ownership enum;
- separate shared and local state serialization;
- shared-only upload/download;
- local checkpoint identity;
- cold/new-client fallback;
- local storage/model-size measurement;
- per-client/cohort evaluation;
- no change to frozen T1/T2/T3 evaluator semantics.

Personalization execution remains deferred and is not a Phase-1 requirement created by this note.

---

# 6. Krichene and Rendle — Sampled Ranking Metrics

## 6.1 Source identity

- **Title:** On Sampled Metrics for Item Recommendation
- **Authors:** Walid Krichene, Steffen Rendle
- **Venue:** KDD 2020, pages 1748–1757
- **DOI:** `10.1145/3394486.3403226`
- **Primary source:** [Google Research publication page][SMET-GOOGLE]
- **Proceedings identity:** [KDD 2020 proceedings][SMET-KDD]

## 6.2 Paper facts

1. Naive sampled ranking metrics are inconsistent with their exact versions: they need not preserve relative model
   ordering, even in expectation. `[SMET-1]`
2. As the sample gets smaller, differences between metrics diminish; with very small samples, metrics collapse
   toward AUC-like behavior. `[SMET-2]`
3. Corrected estimators can improve sampled-metric quality, but the paper's overall recommendation is to avoid
   sampling for metric calculation when possible. `[SMET-3]`

## 6.3 Evidence-to-implementation table

| Topic | Paper evidence | Project choice | Status | Exact code/config consequence |
|---|---|---|---|---|
| T1 candidate space | `[SMET-1]`, `[SMET-3]` | Score T1 exhaustively over the frozen training-observed category vocabulary. | **ADOPT** | The T1 evaluator rejects sampled-category evaluation. |
| Comparable regimes | `[SMET-1]` | All regimes consume the same frozen evaluation examples, vocabulary, and evaluator. | **ADOPT** | Their hashes remain part of ExperimentConfig/compatibility identity. |
| T3 retrieval/ranking | `[SMET-1]` | Consume the separately approved frozen T3 candidate protocol. | **ADOPT guardrail** | No per-run/per-regime random negative set; candidate manifest/protocol hash is recorded. |
| Retrieval misses | Evaluation reasoning | Do not inject a missed positive and hide retrieval failure. | **ADOPT** | Report candidate recall/no-positive/retrieval-miss behavior separately from conditional ranking metrics. |
| Corrected sampled estimators | `[SMET-3]` | Not needed for frozen T1; only consider if a later approved protocol truly requires sampling. | **DEFER** | Any use is diagnostic/versioned, not an unannounced headline metric. |
| Random-negative headline evaluation | `[SMET-1]` | Do not use it. | **REJECT** | Validator/CI must fail a sampled-candidate T1 configuration. |
| Changing negatives after seeing results | Scientific invalidity | Never tune evaluation membership using test outcomes. | **REJECT** | Evaluation manifests are frozen before final tests and shared across regimes. |

## 6.4 Important distinction

A frozen T3 candidate-retrieval protocol is a defined end-to-end task, not permission to call a random-negative
metric an exact full-catalog metric. The project must report both retrieval coverage and ranking quality under the
approved protocol.

---

# 7. Cross-source implementation decisions

## 7.1 Adopt in the current architecture and interfaces

1. One shared local train/eval core for centralized and Flower adapters.
2. Standard Flower FedAvg for R2A, with thin tracing and invariant checks only.
3. Explicit client optimizer, server strategy, local-work, sampling, and weighting identities.
4. Same data/model/seed/init/evaluator identities for controlled regime comparison.
5. Explicit shared/local parameter ownership for any personalized regime.
6. Exhaustive T1 evaluation and a frozen T3 candidate protocol.
7. Actual system/communication measurements rather than estimates.
8. Approved privacy wording: data locality and reduced raw-data centralization.

## 7.2 Defer unless later evidence and approval require them

- FedAdam execution as R2B during the later regime phase;
- FedProx as a client-drift fallback;
- SCAFFOLD, FedYogi, FedAdagrad, and FedAvgM;
- exact R3 adapter/head architecture;
- meta-learning/pFedMe/Ditto/FedRep-style alternative personalization methods;
- guided client selection such as FedFast/Oort;
- low-rank/submodel communication compression;
- differential privacy and secure aggregation;
- corrected sampled estimators.

## 7.3 Reject as defaults or claims

- copying paper hyperparameters;
- using unrecorded Flower defaults as scientific configuration;
- forcing IID clients;
- assuming FedAdam must outperform FedAvg;
- sending local-only personalization state to the server;
- changing task-head ownership without an approved ModelConfig;
- independently sampling negatives per regime;
- describing plain FL as anonymous or formally private;
- changing frozen data/task/evaluation protocols to match a paper.

---

# 8. Consequences for named project tasks

## S1-PR-05 — Unified Trainer

- one local train/eval core;
- thin centralized and Flower adapters;
- explicit client/server optimizer configuration;
- no separate incompatible R1/R2 code paths;
- Checkpoint v1 preserves server optimizer/RNG/config/init identity.

## S1-PR-06 — User-to-Client Partitioning and Sampling

- one user maps to one opaque client;
- deterministic, versioned sampling;
- exact weighting/failure/retry policy freezes here;
- do not introduce FedFast/Oort-style selection into the baseline without approval.

## S1-PR-07 — Real REES46 Flower Smoke

- standard FedAvg first;
- real client/example counts;
- structured selected/successful/contributing-client evidence;
- integration smoke is not hyperparameter tuning.

## S2 scale/profiling work

- separate client population, selected clients per round, and runtime concurrency;
- retain OOM/timeouts/failures as evidence;
- measure process-tree RAM, round timing, and actual payload bytes.

## R2B

- same shared scientific identities as R2A;
- explicit `eta`, `eta_l`, `beta_1`, `beta_2`, `tau`;
- validation-only hyperparameter selection;
- report convergence and communication-to-target, not quality alone.

## R3/R5

- R3: shared federated component plus explicitly local personal state.
- R5: trained R1 checkpoint plus local adaptation.
- Neither changes the frozen evaluator or task definitions.

---

# 9. Validation oracle for this document

The document is valid only if all conditions hold:

- all four required sources have stable identities and access dates;
- each source has paper facts, assumptions/parameters, and an evidence-to-implementation table;
- each source has ADOPT, DEFER, and/or REJECT decisions as applicable;
- paper facts are not presented as project decisions;
- no paper or library hyperparameter is promoted to a default;
- FedAdam and personalization are not promoted into the current Phase-1 core;
- T1 sampled-category headline evaluation is rejected;
- no new formal dependency edge is created from S1-PR-04;
- the artifact path and SHA-256 are recorded.

This documentation task has no standalone validator or generated evidence
pipeline. Reviewers validate the checklist above directly against this file
and the cited primary sources.

---

# 10. References

## Core papers

[FAVG-PMLR]: https://proceedings.mlr.press/v54/mcmahan17a.html
[FAVG-ARXIV]: https://arxiv.org/abs/1602.05629
[FOPT-ARXIV]: https://arxiv.org/abs/2003.00295
[FPER-ARXIV]: https://arxiv.org/abs/1912.00818
[SMET-GOOGLE]: https://research.google/pubs/on-sampled-metrics-for-item-recommendation/
[SMET-KDD]: https://www.kdd.org/kdd2020/proceedings/index.html

## Implementation references

[FLWR-FAVG]: https://flower.ai/docs/framework/ref-api/flwr.serverapp.strategy.FedAvg.html
[FLWR-FADAM]: https://flower.ai/docs/framework/ref-api/flwr.serverapp.strategy.FedAdam.html
[FLWR-FPER]: https://flower.ai/docs/baselines/fedper.html

## Claim map

- `[FAVG-1]`–`[FAVG-4]`: FedAvg PMLR abstract and arXiv v4.
- `[FAVG-5]`–`[FAVG-6]`: FedAvg paper algorithm/parameterization; Flower strategy API is an implementation cross-check.
- `[FOPT-1]`–`[FOPT-6]`: Adaptive Federated Optimization arXiv v5, Sections 1–3 and algorithms.
- `[FPER-1]`–`[FPER-4]`: FedPer arXiv v1 abstract/method scope; Flower baseline is an implementation cross-check.
- `[SMET-1]`–`[SMET-3]`: Google Research/KDD abstract and DOI record.

---

# 11. Final S1-PR-04 decision

The papers support the project's current structure but do not justify expanding the current two-week Phase-1
scientific scope.

- `R2A` remains standard FedAvg.
- `R2B` remains the later FedAdam regime, not a new Phase-1 requirement.
- `R3` retains explicit shared/local parameter ownership, with exact personalization architecture deferred.
- `R5` remains trained-R1 initialization plus local adaptation.
- T1 remains exhaustive; T3 consumes its separately frozen candidate protocol.
- Paper/library defaults remain non-authoritative.
- Formal privacy mechanisms remain future work unless implemented and measured.
