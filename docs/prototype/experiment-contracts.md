# Experiment Contracts

This document describes the frozen S1-PR-03 experiment contracts for
reproducibility across federated learning regimes.

Frozen by Ahmed/project-owner delegation (Issue #16). PR review required
before merge.

## Schema Identity Convention

Every versioned JSON record uses:

```json
{
  "schema": "<stable_schema_id>_v1",
  "version": "1"
}
```

`version` is always a **string** (not integer).

## Frozen Schema Set

The 13 frozen S1-PR-03 schema IDs:

| Schema ID | Purpose |
|---|---|
| `artifact_ref_v1` | Shared artifact reference |
| `model_config_v1` | Model architecture definition |
| `experiment_config_v1` | Runnable experiment configuration |
| `experiment_result_v1` | Experiment output record |
| `metric_record_v1` | Single metric measurement |
| `system_measurement_reference_set_v1` | System measurement references (not raw values) |
| `common_initialization_v1` | Shared untrained tensor state |
| `comparison_compatibility_v1` | R1/R2A compatibility tuple |
| `experiment_contract_freeze_v1` | Machine-readable contract decision record |
| `experiment_artifact_map_v1` | Artifact path mapping |
| `regime_catalog_v1` | Regime definitions |
| `experiment_contract_validation_summary_v1` | Validation evidence |
| `experiment_contract_artifact_manifest_v1` | SHA-verified artifact manifest |

Unknown fields and unknown schema/version values are rejected.

## Regimes

| Regime | Orchestration | Initialization |
|---|---|---|
| R1 | Centralized reference baseline | COMMON_INITIALIZATION |
| R2A | Standard FedAvg baseline | COMMON_INITIALIZATION |
| R2B | FedAdam server optimization | COMMON_INITIALIZATION |
| R3 | Personalized federated learning | COMMON_INITIALIZATION |
| R4 | Strict local-only training | COMMON_INITIALIZATION |
| R5 | Centralized-pretrained then local adaptation | PRETRAINED_R1_CHECKPOINT |

R1–R4 share the same seed-specific untrained CommonInitialization.
R5 starts from a trained R1 checkpoint and uses a separate initialization kind.
R1 is a controlled centralized reference, not a guaranteed performance upper
bound. R2A is standard federated averaging, not independent local training.

## Run-ID Grammar v1

```
run-v1__<regime>__<task-set>__<cohort-set>__s<seed>__cfg<12hex>__a<attempt>
```

Example: `run-v1__r2a__t1-t2-t3__c1-c2-c3__s13__cfg0123456789ab__a1`

Rules:
- Lowercase
- Exact `run-v1` prefix
- Tasks in canonical T1, T2, T3 order
- Cohorts in canonical C1, C2, C3 order
- Seeds: exactly 13, 42, or 2026
- Config prefix = first 12 chars of full config SHA-256
- **Attempt starts at 1** (not 0)
- Build/parse must round-trip

## Attempt and Resume Semantics

States: `PLANNED`, `RUNNING`, `INTERRUPTED`, `SUCCEEDED`, `FAILED`, `CANCELLED`

- Attempt numbering starts at 1
- `PLANNED`/`RUNNING`/`INTERRUPTED` may resume the same attempt when
  config/init/checkpoint identities match
- `SUCCEEDED`/`FAILED`/`CANCELLED` are **terminal and immutable**
- Rerunning after a terminal state requires `attempt + 1` and a new run ID
- A resumed result records `checkpoint_ref` and `resume_count`
- Invalid state transitions fail

## Initialization Union

For R1/R2A/R2B/R3/R4:
```json
{
  "kind": "COMMON_INITIALIZATION",
  "common_initialization_ref": { "...ArtifactRef..." }
}
```

For R5:
```json
{
  "kind": "PRETRAINED_R1_CHECKPOINT",
  "pretrained_checkpoint_ref": { "...ArtifactRef..." },
  "parent_run_id": "run-v1__r1__..."
}
```

R5 cannot carry `common_initialization_ref`.
R1–R4 cannot carry `pretrained_checkpoint_ref`.

## R1/R2A Compatibility Tuple v1

Included fields (exact match required for comparison):

- `tasks`, `training_cohort`, `evaluation_cohorts`, `seed`
- 13 SHA-256 fields: `source_dataset_sha256`, `canonical_data_contract_sha256`,
  `cohort_manifest_sha256`, `split_manifest_sha256`, `task_examples_manifest_sha256`,
  `evaluation_manifest_sha256`, `representation_sha256`, `model_config_sha256`,
  `objective_config_sha256`, `shared_trainer_core_sha256`, `common_initialization_sha256`,
  `evaluation_protocol_sha256`, `evaluator_sha256`, `environment_lock_sha256`
- `git_sha`

Excluded operational differences: `regime`, server aggregation strategy, FL
rounds/sampling/retry, orchestration duration, `attempt`/`run_id`, timestamps,
hardware, SystemMeasurement values, wall-clock runtime.

Any mismatch returns structured field-level reasons, not only False.

## SystemMeasurement Reference-Only Boundary

ExperimentResult contains `system_measurements` as a reference set only.
No raw `model_size_bytes`, `peak_ram_bytes`, `inference_latency_ms`,
`upload_bytes`, `download_bytes`, or `wire_size_bytes` fields are permitted.

## Seed Proof

Only fixture proofs (`artifact_kind = FIXTURE_PROOF`) are generated.
Seeds 13, 42, 2026 produce independently reproducible CommonInitialization
tensors. This PR does NOT create AUTHORITATIVE initializations.

## Canonical Commands

```bash
# Validate all contracts
uv run --locked python scripts/experiments/validate_experiment_contracts.py

# Run experiment tests
uv run --locked pytest tests/experiments -v
```
