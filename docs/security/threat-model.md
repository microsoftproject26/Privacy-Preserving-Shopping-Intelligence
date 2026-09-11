# Phase 1 Threat Model

Scope: R1 centralized training, simulated R2A/FedAvg, research data, and artifacts.
This is a design document, not a vulnerability report.

## 1. Overview

This repository is an offline research project, not a deployed service. It converts local REES46
data, trains through one shared trainer core, and compares a centralized R1 path with an R2A
Flower/FedAvg path. The current executable R2A flow is a synthetic Ray simulation on one host, not
real remote-device federation (scripts/training_smoke.py:1-5, scripts/training_smoke.py:206-216).

| Component | What it handles | Evidence |
|---|---|---|
| Data conversion | Shopping events and stable user/session/item identifiers | scripts/data/convert_rees46.py:193-207 |
| R1 | All supplied batches inside one process | ppsi/training/centralized.py:35-68 |
| R2A client | Local batches, shared state, and per-client metrics | ppsi/training/flower.py:69-123 |
| R2A coordinator | Individual updates before FedAvg aggregation | scripts/training_smoke.py:74-109 |
| Checkpoints/results | Model state, optimizer state, metrics, and experiment identities | ppsi/training/checkpoint.py:110-153, ppsi/training/result.py:40-105 |

~~~mermaid
flowchart LR
    Data[Local REES46 data]
    R1[R1 centralized process]
    Client[R2A logical client]
    Server[Flower coordinator]
    Store[Local artifacts]

    Data -->|all selected batches| R1
    Data -. future per-user partition .-> Client
    Server -->|shared model state| Client
    Client -->|updated state and metrics| Server
    R1 --> Store
    Server --> Store
~~~

### Effective resources

| Workflow | Resource | Configuration/source | Effective value or location | Recipients | Enforcing control | Evidence or unknown |
|---|---|---|---|---|---|---|
| Conversion | Raw and processed histories | conversion.v1 config | Local dataset and artifact paths | Local operator and converter | Size, hash, and header checks | Host access controls are external; config/conversion.v1.json:4-23 |
| Git fixture | Real selected histories | Explicit ignore exception | fixtures/smoke_sample.parquet | Every repository recipient | Repository access and review | Namespacing is not anonymization; .gitignore:431-439 |
| R1 | All training batches | Caller-provided Phase1Batch stream | One centralized process | R1 host and operator | Batch/state validation | Confidentiality is host-owned; ppsi/training/centralized.py:35-68 |
| Simulated R2A | Batches, updates, and metrics | Local Flower/Ray simulation | One shared host | Client worker and coordinator | State validation | No physical isolation; scripts/training_smoke.py:206-216 |
| Checkpoints | Model, optimizer, and RNG state | Caller-selected path | Local/approved artifact storage | Authorized artifact readers | Hash and semantic checks | ACL, encryption, and retention are not frozen; ppsi/training/checkpoint.py:220-263 |

## 2. Threat Model, Trust Boundaries, and Assumptions

### Protected assets

- raw and processed shopping histories;
- stable user, session, and item identifiers;
- client batches, labels, model updates, and per-client metrics;
- checkpoints, experiment results, and scientific protocol identities;
- repository and CI credentials.

The tracked smoke Parquet is an intentional exception to the general Parquet ignore rule and
contains complete histories for selected real-data users (.gitignore:431-439,
fixtures/canonical_event_v1.json:39-47). Namespaced IDs retain their raw source values; they are
linkable identifiers, not anonymous values (fixtures/canonical_event_v1.json:32-37).

### Main boundaries

#### R1 centralized boundary

R1 intentionally gives one process every supplied batch. Anyone controlling that process or its
artifact storage can access those histories. R1 must not be described as privacy-preserving
(ppsi/training/centralized.py:35-68).

#### R2A client/coordinator boundary

The intended real-data invariant is one user per logical client with raw examples kept client-side.
That partitioning is future work; the current smoke uses synthetic batches
(docs/papers-notes.md:305-317).

Clients receive shared model state and return updated shared state plus counts, losses, IDs, and
digests. State validation protects keys, shapes, dtypes, and finiteness, not privacy
(ppsi/training/flower.py:41-123, ppsi/training/state.py:124-176).

The coordinator sees each update before aggregation. The shared Ray host may inspect client memory,
so the simulation does not prove physical device or process isolation.

#### Artifact boundary

Full data, generated Parquet, and checkpoints should stay outside normal Git history. Checkpoints
are loaded with unsafe PyTorch deserialization only under a trusted-internal-artifact contract and
after hash and semantic checks (ppsi/training/checkpoint.py:220-263).

#### Public-repository boundary

The repository is public. Every artifact it holds is readable by anyone, and this is a trust
boundary the rest of this document assumes rather than states.

What is deliberately outside it: the raw REES46 file, the generated Parquet, the cohort manifests,
the materialised task examples and every checkpoint. Those carry `user_id` and are named
`INTERNAL_DO_NOT_UPLOAD_*` or ignored by path.

What is inside it and worth naming plainly:

| Artifact | Contains |
|---|---|
| `fixtures/smoke_sample.parquet` | 23,109 events, 1,499 distinct `rees46:user:*` identifiers, 4,865 sessions, drawn from the real `2019-Oct.csv` |
| `fixtures/reference/item_catalog_v1.proposed.parquet` | 113,765 items with median TRAIN price, price band and popularity. No user data. |
| `fixtures/reference/t3_candidate_lists_v1.proposed.parquet` | Item-to-item candidate lists. No user data. |
| `docs/evidence/**` | Aggregate statistics and hashes. No user data. |

The smoke fixture is the exception that has to be stated rather than left implicit. It carries real
user identifiers and their behaviour in a public repository, while `README.md`, the
`client_id_from_user` contract and the ignore rules all say raw user identities stay out.

The mitigating fact is that REES46 October 2019 is itself a public dataset, so this republishes
rather than discloses. That is a reason the risk is low, not a reason the inconsistency is
acceptable: a project whose subject is data locality should not have to explain why its own
repository contains user trajectories. The options are to record it here as a deliberate exception,
or to replace the fixture with synthetic rows and re-freeze its hash. This document records it; the
decision belongs to the team.

### Privacy claims

| Claim | Phase 1 position |
|---|---|
| R2A reduces raw-data centralization | Supported as the intended design |
| Current simulation proves real device isolation | Not supported |
| Model updates cannot leak information | Not supported |
| Differential Privacy is implemented | No |
| Secure Aggregation is implemented | No |
| Namespaced or hashed IDs are anonymous | No |
| The repository contains no user identifiers | No, see the public-repository boundary |

The approved wording is data locality and reduced raw-data centralization. Differential Privacy,
Secure Aggregation, and formal anonymity remain future work (docs/papers-notes.md:258-290).

### Assumptions and open questions

- There is no source-backed public API, tenant model, or remote client enrollment path.
- `label_matures_at` carries the session end for T2 and T3, which is when their outcome becomes
  observable, but the decision event's own time for T1. A T1 label comes from a later target, so
  for T1 the field must not be used to decide whether a label has matured. The leakage validator
  requires the T1 target itself to lie inside the split instead. The naming should be reconciled
  by the lane that owns the builder (ppsi/temporal.py, `_require_maturation`).
- Filesystem permissions, encryption, backup, and artifact retention are outside current code.
- Remote R2A would need client authentication, transport security, admission rules, and logging
  policy before deployment.
- The regime catalog calls R2A independent models while its type and implementation use FedAvg;
  the operational definition is FedAvg and the stale description should be corrected
  (config/experiments/regime_catalog.v1.json:9-12, docs/experiment-contracts.md:44-50).

## 3. Attack Surface, Mitigations, and Attacker Stories

These are design hypotheses, not confirmed vulnerabilities.

| Priority | Scenario | Prerequisite and impact | Existing control | Practical mitigation |
|---|---|---|---|---|
| High | Raw or derived histories are committed or published | Contributor or artifact writer broadens access to behavioral data | Data patterns are ignored; PR review | Keep only approved small fixtures in Git and review data-bearing changes |
| High | Untrusted checkpoint reaches the loader | Attacker-controlled checkpoint can execute with training-host authority | Trusted-artifact contract plus hash/identity checks | Restrict artifact writers and never accept external checkpoints |
| High | Protocol or identity is changed silently | Invalid R1/R2A comparison and thesis results | Versioned contracts and Git review | Keep compatibility checks and require explicit review for protocol changes |
| Medium | Coordinator infers from individual updates or metrics | Real per-user clients exist | Raw batches are not in the intended message payload | Minimize stored client fields, use opaque IDs, and make no formal privacy claim |
| Medium | Malicious client poisons FedAvg | Remote/untrusted clients are introduced | Shape, dtype, key, and finiteness validation | Define admission, weighting, clipping, and failure rules before remote deployment |
| Medium | Shared Ray host exposes client state | Real data enters one simulation host | Host OS access controls | Treat simulation as one trust zone; use synthetic CI data |
| Medium | Secret appears in source or CI logs | Developer or workflow mishandles a credential | GitHub secret controls and review | Use secret scanning, push protection, least privilege, and immediate rotation |

## 4. Severity Calibration

- Critical: remote compromise exposing the full dataset or organization-wide credentials. No such
  deployed remote service currently exists.
- High: substantial behavioral-data publication, untrusted checkpoint execution, or a merged
  scientific-integrity break that invalidates core results.
- Medium: bounded per-client metadata leakage, poisoning of a research run, one-host simulation
  exposure, or a narrowly scoped credential leak.
- Low: synthetic-only disclosure, local availability impact requiring the same operator, or
  documentation drift with no effect on controls or results.

Impact is reduced when data is synthetic, the actor already owns the host, or an independent control
blocks the boundary crossing. Missing deployment evidence is uncertainty, not proof of safety or a
confirmed finding.
