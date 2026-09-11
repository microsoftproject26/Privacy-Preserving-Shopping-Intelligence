# MVP execution decision — requester-delegated scope

Status: PREDECLARED FOR NEW MVP EXECUTION; completion is NOT asserted by this document.
Basis: the requester delegated MVP engineering and scope decisions on 2026-09-11.
No additional named team approvals are inferred. MVP scope authorized by requester delegation.

The inspected baseline is `a72d987b21407a941b954f5a7c0a92ade5cfff5d`. We retain Eid's accepted
one-layer GRU, the original published TaskExamples, the frozen raw T1 evaluator, and the
existing baseline evidence. We do not rebuild the model or repeat architecture or lambda tuning.

The NEW experiment is one matched-exposure T1 centralized/FedAvg pilot, seed 13, 1,000
TRAIN-eligible clients, 20 rounds × 50 selected clients, all local TRAIN examples once per
selected round, full 438,185 validation decisions, identical common untrained state, last
round selection. The exact resolved policy and hashes are recorded before execution.

Central Adam state persists; client Adam resets per round. This is a declared training-regime
difference, not a claim of identical optimizer state. A low score is reported, not rescued by
changing the population, seed, model, rounds or loss.

Existing joint T1/T2 warm-started models support the local inference story only after
role, hash and spec verification. T3 is frozen retrieval; no learned three-head success is claimed.

The course report and presentation contain preliminary results. This scoped decision does not
close the original full multi-task R1×3, R2a, final deployment, or Phase-1 freeze tasks.
Historical approvals are retained; this new decision does not backdate evidence or create
signatures for other members.

## 1. What the resolved policy actually fixes

`config/mvp/execution.v1.json` is the single authority for every size, seed and lifecycle used
below. This document explains the choices; it does not restate them as a second source of truth.

| Decision | Value | Why it is fixed before any metric exists |
|---|---|---|
| Population | 1,000 clients with at least one observed T1 TRAIN decision | Chosen before scoring, so it can never be a rescue after a disappointing number |
| Eligibility | TRAIN presence only | VALIDATION support must not decide who participates |
| Schedule | 20 rounds × 50 clients, `client_sampler_v1`, seed 13 | Precomputed once and replayed identically by both regimes |
| Local work | every one of a selected client's frozen TRAIN rows, once | No per-client cap, no sampling with replacement |
| Evaluation | all 438,185 frozen VALIDATION decisions at rounds 0, 10 and 20 | A fixed cadence, not a search over checkpoints |
| Selection | the completed round-20 state | No score-based checkpoint picking and no seed search |

## 2. The three counts this pilot keeps apart

A federated report is easy to misread, so these are reported separately and never merged:

- **1,000** is the declared pilot population, the pool clients are drawn from.
- **50** is the number of participants per round, and also the number of Flower SuperNode
  execution slots. Execution slots are not study clients.
- **1,000 scheduled participations** (20 × 50) is a count of round-client pairs; the number of
  distinct clients that actually trained is measured separately and is smaller.

## 3. Matched exposure, declared lifecycle difference

Both regimes consume the identical `(round, client, ordered decision keys)` sequence, and the
run records a SHA-256 over it. The comparison stage refuses to publish a pair whose exposure
digests differ, so a quietly different data diet cannot be reported as a regime effect.

What is deliberately *not* matched is the optimizer lifecycle. The centralized replay keeps one
Adam state for the whole run; each FedAvg client resets its optimizer at every server round,
which is the existing `FlowerLocalAdapter` contract. That difference lives in each result's
`regime_config` and in the comparison's limitations, not hidden inside a shared identity field.

For the same reason, the centralized side is **not** an all-data upper bound. It replays exactly
the federated schedule, so it answers "what does the same data diet give without averaging",
which is the question this pair can actually answer.

## 4. Identity and hash conventions

Four different digests appear in this run and they are not interchangeable:

| Digest | Over what | Produced by |
|---|---|---|
| raw file SHA-256 | the exact bytes on disk | `raw_file_sha256` |
| canonical text SHA-256 | LF-normalized text | `ppsi.training.identity.file_sha256` |
| model-lane state digest | the tensor encoding used when weights are created | `common_initialization` |
| training-state codec digest | the packed shared-state encoding | `shared_state_digest` |

The common initialization is generated once, saved once, and loaded by both regimes. Its raw
file hash, its model-lane digest and its codec digest are all recorded under separate names. A
difference between two conventions is never reported as a failure, and no recorded hash is ever
edited by hand to make a check pass.

The Flower client reports `received_digest` and `updated_digest` in the convention the inherited
strategy compares against, which is a third encoding again. Substituting the codec digest there
would look exactly like a redistribution failure.

## 5. Data boundary

Preparation recovers the producer's complete 97,279-user example slice from every client value in
the frozen T2 TRAIN file, censored rows included, and asserts it is a subset of the 388,789 C1
members. No modulo is reapplied and no additional filter is introduced. The global canonical
order is built over the complete split before the pilot subset exists, because `decision_order`
is a position inside that complete split.

TEST is never opened, listed, hashed or reconstructed. The raw loader skips row groups that its
recorded statistics prove cannot intersect the requested window, and reports the rows it did
decode outside the window; that measured number is published rather than a literal zero. A
stronger physical no-read claim is not made.

## 6. What this MVP does not close

This is a scoped preliminary demonstrator. It is not the original multi-task MVP commitment,
not the final R1 denominator, not a full R2a reference, and not a Phase-1 freeze. No
differential privacy, secure aggregation, device isolation, ONNX export or TEST performance is
claimed. The quality-retention number produced here compares the FedAvg pilot against the
matched centralized pilot, and its scope label says so.
