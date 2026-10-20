# Unified Trainer Interface — S1-PR-05

**Logical contract:** `Phase1Batch v1` / `Checkpoint v1`
**Task:** S1-PR-05 / Issue #18
**Consumers:** S1-PR-07 (#20), S1-SE-08 (#28), S2-DS-01 (#30)

This document freezes the shared local training interface used by centralized
and Flower execution. It does not freeze the final GRU architecture, final
T1/T2/T3 objectives, final client sampling/weighting policy, FedAdam, or
personalization.

## 1. Design principle

Centralized and Flower adapters call the same `LocalTrainerCore`.

```text
Phase1Batch -> Phase1Model -> RawModelOutput -> Objective -> LocalTrainerCore
                                               /                 \
                                      CentralizedAdapter   FlowerLocalAdapter
```

Adapters own orchestration. They may not implement separate forward, masking,
loss, backward, clipping, optimizer, or scheduler semantics.

## 2. Data decision

S1-PR-05 uses deterministic synthetic fixtures only. It does not train on the
full REES46 dataset or the 1,499-user real smoke fixture. Real REES46
TaskExamples are consumed by S1-PR-07 after the trainer contract is frozen.

No notebook is part of this task. The executable smoke script and tested
library are the canonical examples; a notebook would duplicate behavior and
introduce hidden state.

## 3. Physical package path

Reusable code lives under the flat importable package:

```text
ppsi/training/
```

This avoids putting library code under `scripts/` and avoids changing the
current `[tool.uv] package = false` project setting. Tests must prove that
`ppsi.training` imports inside a real Ray worker and through the Flower smoke.

## 4. Phase1Batch v1

Every batch uses physical `B >= 1`, `L >= 1`, and `K >= 1`. Semantic history
length may be zero.

| Field | dtype | shape | rule |
|---|---|---|---|
| `history_categorical_ids[name]` | int64 | `[B,L]` | named channels from representation metadata |
| `history_continuous_features` | float32 | `[B,L,F_history]` | `F_history` may be 0 |
| `lengths` | int64 | `[B]` | `0 <= length <= L` |
| `history_mask` | bool | `[B,L]` | exactly `j < lengths[b]` |
| `query_categorical_ids[name]` | int64 | `[B]` | named channels from representation metadata |
| `query_continuous_features` | float32 | `[B,F_query]` | `F_query` may be 0 |
| `candidate_ids` | int64 | `[B,K]` | aligned with T3 score index |
| `candidate_categorical_ids[name]` | int64 | `[B,K]` | collection may be empty |
| `candidate_continuous_features` | float32 | `[B,K,F_candidate]` | `F_candidate` may be 0 |
| `candidate_mask` | bool | `[B,K]` | false means no candidate contribution |
| `t1_target` | int64 | `[B]` | ignored when `t1_present=false` |
| `t2_target` | float32 | `[B,1]` | ignored when `t2_present=false` |
| `t3_gains` | float32 | `[B,K]` | aligned with candidate mask |
| `t1_present` | bool | `[B]` | authoritative T1 supervision mask |
| `t2_present` | bool | `[B]` | authoritative T2 supervision mask |
| `t3_present` | bool | `[B]` | authoritative T3 supervision mask |

### 4.1 Categorical channels

The trainer does not hard-code final item/category/brand/event feature names.
Each categorical channel is independently `[B,L]`, `[B]`, or `[B,K]` and is
bound by versioned representation metadata.

Do not pack unrelated categorical values into a composite token.

### 4.2 Padding identity

Each categorical channel declares its own pad ID. Zero is preferred but not a
global requirement. Valid IDs must not collide with the channel's pad ID.
Masks remain the semantic authority.

### 4.3 Right padding

```text
history_mask[b,j] == (j < lengths[b])
```

Left padding, holes, negative lengths, and lengths greater than `L` are
invalid.

### 4.4 Empty history

The technically selected behavior, pending the recorded governance approval,
is **ZERO_HIDDEN**:

```text
lengths[b] == 0
=> final history representation[b] is exactly a zero vector
```

No learned BOS/EMPTY event is used.

The stub/GRU implementation may run an encoder on physical padded storage,
gather the last valid state using a clamped index, and apply an exact zero
override after history-only transformation/normalization.

### 4.5 Missing tasks

Missing labels are not negatives. Presence masks determine contribution.
Stored absent-target fillers are finite canonical values, but changing those
fillers must not change loss, gradients, or updates.

### 4.6 T3 structural rule

```text
t3_present[b] == true
=> candidate_mask[b].any() == true
```

All-candidate-masked is legal only when `t3_present=false`. T1 and T2 may still
contribute. The objective remains numerically defensive even if malformed
all-masked input bypasses validation.

This contract does not freeze final T3 positive/gain semantics, candidate
retrieval science, or ranking loss.

## 5. Two validation levels

### Runtime/contract validation

Used by `LocalTrainerCore`:

- exact dtype/rank/shape;
- B/L/K alignment;
- right-padding length/mask relationship;
- finite floating tensors;
- target/presence alignment;
- T3 structural validity.

It does not rely on filler values for semantics.

### Canonical producer validation

Used at data-builder/fixture boundaries:

- categorical pad IDs in padded positions;
- zero continuous padding;
- canonical finite absent-target fillers;
- canonical candidate padding.

The separation is required so poison/invariance tests can alter masked storage
while still exercising real trainer behavior.

## 6. RawModelOutput

```text
t1_logits [B,C_category] floating
t2_logit  [B,1]          floating
t3_scores [B,K]          floating
```

No sigmoid, softmax, thresholding, sorting, or candidate reordering occurs in
the raw model contract.

## 7. Model and state interfaces

`Phase1Model` is a minimal protocol:

- callable from `Phase1Batch` to `RawModelOutput`;
- exposes category count;
- exposes `SharedStateSpec`.

The model may not perform hidden filesystem, database, catalog, or service
feature lookup. Model inputs are explicit batch tensors plus model state.

`SharedStateSpec` records deterministic key order, parameter/buffer kind,
shape, dtype, and shared/local ownership. S1-PR-05 uses all safe floating model
state as shared. This is only a future seam for R3; personalization is not
implemented here.

## 8. Objective and mergeable statistics

The trainer executes an injected objective:

```text
Objective(batch, raw_output) -> ObjectiveResult
```

`ObjectiveResult` preserves:

- scalar total loss;
- per-task additive numerator;
- per-task denominator/support;
- support unit;
- processed/contributing example counts;
- contributing tasks;
- status and diagnostics.

The task includes a non-scientific `ContractSmokeObjective` only to exercise
all heads, masks, and gradients. It must not be used as the final T3 or
multi-task objective.

### No-contributing-task

If no task contributes:

```text
status = NO_CONTRIBUTING_TASK
backward = not called
optimizer.step = not called
scheduler.step = not called
support = 0
```

## 9. LocalTrainerCore

The shared core owns:

- runtime batch validation;
- device transfer;
- train/eval mode;
- raw forward;
- raw-output validation;
- objective invocation;
- finite loss/gradient/parameter checks;
- backward;
- optional explicit gradient clipping;
- optimizer step;
- scheduler step under `OPTIMIZER_STEP` semantics;
- mergeable local summary.

It does not know regime, client ID, server round, client sampling, or final
result registry.

## 10. Adapters

### CentralizedAdapter

Owns only DataLoader/batch iteration, epoch/cursor orchestration, validation
cadence, checkpoint cadence, and run-level result transitions.

### FlowerLocalAdapter

Receives an explicit `AggregationWeightPolicy`; it does not freeze the final
scientific client weighting policy. S1-PR-05 uses a clearly non-scientific
contributing-row policy only for the deterministic smoke, while S1-PR-06 owns
the final policy.

Owns only shared-state translation/validation, client loader resolution, calls
to the same local core, and scalar metric return. The full Flower simulation
reuses the proven S1-PR-02 lifecycle and its FedAvg oracle.

For R2A v1, client-local optimizer and scheduler state use
`RESET_EACH_SERVER_ROUND`: a completed server-round boundary resets local
optimization state before the next fit. Resuming an interrupted fit instead
restores its checkpoint state and cursor; FedAdam server state is separate.

## 11. CommonInitialization

The stub model is constructed under one of the allowed seeds `13`, `42`, or
`2026`. `ppsi/training/initialization.py` reuses the existing S1-PR-03 canonical
serializer and hashing primitives, verifies the state SHA and complete stub
shape/config identity, and loads the state only after those checks pass. Stub initialization is a `FIXTURE_PROOF`, not an
authoritative final GRU initialization.

Wrong seed, model identity, state hash, or config identity fails before the
first training update.

## 12. Checkpoint v1

Checkpoint v1 is a trusted internal PyTorch runtime checkpoint. Its byte SHA is
an integrity check. Scientific compatibility is validated by semantic hashes.

It stores:

- run ID and attempt;
- model/optimizer/scheduler state;
- optional null GradScaler state;
- `TrainingCursor` with outer round, local epoch, next batch index, optimizer step;
- best criterion;
- ExperimentConfig identity;
- CommonInitialization identity;
- input/data identity;
- shared trainer core identity;
- objective config identity;
- environment lock and git identity;
- deterministic loader contract;
- Python/NumPy/Torch CPU/CUDA RNG state.

### Exact replay

Global RNG alone is insufficient. The exact CPU replay path uses:

- deterministic epoch permutation derived from run seed, sampler version,
  epoch, dataset identity, and length;
- explicit `next_batch_index`;
- `num_workers=0` for the P0 path.

Resume reconstructs the same permutation and continues from the next batch.

### Atomic persistence

1. write a unique temp file in the destination directory;
2. flush and fsync;
3. close the file;
4. compute SHA-256;
5. replace the destination with `os.replace` and Windows retry handling;
6. register the returned SHA in ArtifactRef/ExperimentResult.

Resume verifies file hash, semantic identities, and model tensor compatibility
before applying state. Application is transactional: if optimizer, scheduler,
scaler, RNG, or progress restore fails, the prior runtime state is restored.
The normal load path also restores `LocalTrainerCore.optimizer_step_count` from
the checkpoint cursor.

## 13. ExperimentResult ownership

Local steps and Flower client fits return internal mergeable summaries.
`ExperimentResult v1` remains a run/attempt-level record created by centralized
or Flower server orchestration. No parallel run-result schema is introduced.

## 14. Shared trainer core identity

`shared_trainer_core_sha256` is the canonical hash of an ordered manifest of
behavior-defining files:

- `batch.py`
- `outputs.py`
- `protocol.py`
- `state.py`
- `identity.py`
- `core.py`

Scientific objective configuration is identified separately by
`objective_config_sha256`. Adapters, docs, tests, stub model, and unrelated
files do not define the shared local trainer semantics.

## 15. Data/notebook decision

- Deterministic synthetic batches: **used**.
- Full REES46 or 1,499-user fixture: **not used in #18**.
- Tracked notebook: **not created**.
- Human explanation: this document plus one reproducible smoke command.

## 16. Canonical validation commands

```powershell
uv run --locked ruff check ppsi/training scripts/training_smoke.py scripts/generate_training_artifact_manifest.py tests/training
uv run --locked ruff format --check ppsi/training scripts/training_smoke.py scripts/generate_training_artifact_manifest.py tests/training
uv run --locked python -m pytest tests/training -v
uv run --locked python -m pytest -q tests/federated/test_fl_synthetic_smoke.py tests/experiments tests/training
uv run --locked python scripts/training_smoke.py --config config/unified_trainer_smoke.v1.json --output docs/evidence/s1-pr-05/unified_trainer_smoke_summary.v1.json --source-git-sha <COMMIT_A_SHA>
uv run --locked python scripts/generate_training_artifact_manifest.py
uv run --locked python scripts/generate_training_artifact_manifest.py --verify
```

Generate the evidence and artifact manifest last. Do not edit hashed artifacts
after manifest generation. Text artifacts are hashed as UTF-8 with LF line
endings and no BOM; directory-tree hashing excludes interpreter and pytest
caches so evidence is not machine-specific.

## Flower aggregation weighting boundary

The Flower adapter requires an injected `AggregationWeightPolicy`. The bundled `ContributingRowsSmokeWeightPolicy` is non-scientific and exists only for the deterministic S1-PR-05 smoke. S1-PR-06 owns the final scientific client-update weighting policy.
