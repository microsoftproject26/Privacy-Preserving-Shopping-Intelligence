# Canonical Artifact Registry

| Logical ID | Canonical path | Version | Producer | Consumers |
|---|---|---:|---|---|
| `TrainingEnvironmentPython` | `.python-version` | v1 | S1-PR-01 | S1-PR-02/03/04/05, S1-SE-02 |
| `TrainingEnvironmentSpec` | `pyproject.toml` | v1 | S1-PR-01 | S1-PR-02/03/04/05, S1-SE-02 |
| `TrainingEnvironmentLock` | `uv.lock` | v1 | S1-PR-01 | S1-PR-02/03/04/05, S1-SE-02 |
| `TrainingEnvironmentSmoke` | `scripts/env_smoke.py` | v1 | S1-PR-01 | Developers, CI |
| `TrainingEnvironmentDocs` | `docs/training-environment.md` | v1 | S1-PR-01 | Developers, CI |
| `TrainingEnvironmentEvidence` | `docs/evidence/s1-pr-01/environment_validation.v1.json` | v1 | S1-PR-01 | Review / reproducibility evidence |
| `FLSyntheticSmokeConfig` | `config/fl_synthetic_smoke.v1.json` | v1 | S1-PR-02 | S1-PR-05, CI |
| `FLSyntheticSmokeEntryPoint` | `scripts/federated/fl_synthetic_smoke.py` | v1 | S1-PR-02 | S1-PR-05, CI |
| `FLSyntheticSmokeSummary` | `docs/evidence/s1-pr-02/fl_synthetic_smoke_summary.v1.json` | v1 | S1-PR-02 | S1-PR-05, CI |
| `FLSyntheticSmokeLifecycleTest` | `tests/federated/test_fl_synthetic_smoke.py` | v1 | S1-PR-02 | CI |
| `FLSyntheticSmokeDocs` | `docs/federated-synthetic-smoke.md` | v1 | S1-PR-02 | Developers |
| `FLSyntheticSmokeArtifactManifest` | `docs/evidence/s1-pr-02/artifact_manifest.v1.json` | v1 | S1-PR-02 | S1-PR-05 |
| `ExperimentConfigSchema` | `config/experiments/schemas/experiment_config.v1.schema.json` | v1 | S1-PR-03 | Developers |
| `ModelConfigSchema` | `config/experiments/schemas/model_config.v1.schema.json` | v1 | S1-PR-03 | Developers |
| `ComparisonCompatibilitySchema` | `config/experiments/schemas/comparison_compatibility.v1.schema.json` | v1 | S1-PR-03 | Developers |
| `MetricRecordSchema` | `config/experiments/schemas/metric_record.v1.schema.json` | v1 | S1-PR-03 | Developers |
| `SystemMeasurementSchema` | `config/experiments/schemas/system_measurement_reference_set.v1.schema.json` | v1 | S1-PR-03 | Developers |
| `ExperimentResultSchema` | `config/experiments/schemas/experiment_result.v1.schema.json` | v1 | S1-PR-03 | Developers |
| `CommonInitializationSchema` | `config/experiments/schemas/common_initialization.v1.schema.json` | v1 | S1-PR-03 | Developers |
| `ExperimentArtifactMapSchema` | `config/experiments/schemas/experiment_artifact_map.v1.schema.json` | v1 | S1-PR-03 | Developers |
| `RegimeCatalogSchema` | `config/experiments/schemas/regime_catalog.v1.schema.json` | v1 | S1-PR-03 | Developers |
| `ContractValidationSummarySchema` | `config/experiments/schemas/experiment_contract_validation_summary.v1.schema.json` | v1 | S1-PR-03 | Developers |
| `ExperimentContractArtifactManifestSchema` | `config/experiments/schemas/experiment_contract_artifact_manifest.v1.schema.json` | v1 | S1-PR-03 | Developers |
| `ArtifactRefSchema` | `config/experiments/schemas/artifact_ref.v1.schema.json` | v1 | S1-PR-03 | Developers |
| `ExperimentContractFreezeSchema` | `config/experiments/schemas/experiment_contract_freeze.v1.schema.json` | v1 | S1-PR-03 | Developers |
| `ExperimentContractFreeze` | `config/experiments/contract_freeze.v1.json` | v1 | S1-PR-03 | S1-PR-05, Developers |
| `RegimeCatalog` | `config/experiments/regime_catalog.v1.json` | v1 | S1-PR-03 | S1-PR-05, Developers |
| `ExperimentArtifactMap` | `config/experiments/experiment_artifact_map.v1.json` | v1 | S1-PR-03 | Developers |
| `ExperimentContractsPrimitives` | `scripts/experiments/contracts.py` | v1 | S1-PR-03 | S1-PR-05, #18, #19, #27, #36, #51 |
| `ExperimentSchemaValidators` | `scripts/experiments/schemas.py` | v1 | S1-PR-03 | S1-PR-05, #18, #19, #27, #36, #51 |
| `ExperimentCompatibility` | `scripts/experiments/compatibility.py` | v1 | S1-PR-03 | #36, #51 |
| `ExperimentInitialization` | `scripts/experiments/initialization.py` | v1 | S1-PR-03 | S1-PR-05 |
| `ExperimentCommonInitFixtureProofs` | `fixtures/experiments/common_initialization/` | v1 | S1-PR-03 | Developers |
| `ExperimentContractValidator` | `scripts/experiments/validate_experiment_contracts.py` | v1 | S1-PR-03 | CI |
| `ExperimentContractsDocs` | `docs/experiment-contracts.md` | v1 | S1-PR-03 | Developers |
| `ExperimentContractsDecisionEntry` | `docs/decisions.md` (S1-PR-03 section) | v1 | S1-PR-03 | Developers |
| `ExperimentContractsEvidence` | `docs/evidence/s1-pr-03/` | v1 | S1-PR-03 | S1-PR-05 |
| `FederatedImplementationNotes` | `docs/papers-notes.md` | v1 | S1-PR-04 | Developers / research reference |
| `Phase1BatchV1` | `ppsi/training/batch.py` | v1 | S1-PR-05 | S1-PR-07, S1-SE-08, S2-DS-01 |
| `Phase1RawOutputV1` | `ppsi/training/outputs.py` | v1 | S1-PR-05 | S1-PR-07, S2-DS-01 |
| `UnifiedTrainerCore` | `ppsi/training/core.py` | v1 | S1-PR-05 | R1/R2 adapters, S1-PR-07, S2-DS-01 |
| `UnifiedTrainerCentralizedAdapter` | `ppsi/training/centralized.py` | v1 | S1-PR-05 | S2-DS-01 |
| `UnifiedTrainerFlowerAdapter` | `ppsi/training/flower.py` | v1 | S1-PR-05 | S1-PR-07 |
| `CheckpointV1` | `ppsi/training/checkpoint.py` | v1 | S1-PR-05 | S1-PR-07, S2-DS-01 |
| `UnifiedTrainerSmokeConfig` | `config/unified_trainer_smoke.v1.json` | v1 | S1-PR-05 | CI / developers |
| `UnifiedTrainerSmokeEntryPoint` | `scripts/training_smoke.py` | v1 | S1-PR-05 | CI / developers |
| `UnifiedTrainerSmokeSummary` | `docs/evidence/s1-pr-05/unified_trainer_smoke_summary.v1.json` | v1 | S1-PR-05 | Review / CI |
| `SharedTrainerCoreManifest` | `docs/evidence/s1-pr-05/shared_trainer_core_manifest.v1.json` | v1 | S1-PR-05 | ExperimentConfig compatibility |
| `UnifiedTrainerArtifactManifest` | `docs/evidence/s1-pr-05/artifact_manifest.v1.json` | v1 | S1-PR-05 | Review / reproducibility |
| `UnifiedTrainerDocs` | `docs/training-interface.md` | v1 | S1-PR-05 | Developers / downstream lanes |
| `UnifiedTrainerContractTests` | `tests/training/` | v1 | S1-PR-05 | CI |
| `ExperimentResultFiles` | `artifacts/experiment-results/*.result.json` | v1 | Experiment runs | S2-SE-06 |
| `ExperimentResultsTool` | `scripts/experiments/results.py` | v1 | S2-SE-06 | Developers / CI |
| `ClientPartitioningModule` | `ppsi/federated/clients.py` | v1 | S1-PR-06 | S1-PR-07, S2-PR-06 |
| `ClientSamplingModule` | `ppsi/federated/sampling.py` | v1 | S1-PR-06 | S1-PR-07, S2-PR-06 |
| `ClientManifestV1` | `data/protocol/INTERNAL_DO_NOT_UPLOAD_client_manifest_v1.parquet` | v1 | S1-PR-06 | S1-PR-07, S2-PR-06 |
| `ClientSamplingTraceV1` | `data/protocol/INTERNAL_DO_NOT_UPLOAD_client_sampling_trace_v1.jsonl` | v1 | S1-PR-06 | S1-PR-07, CI / audit |
| `ClientManifestBuildCLI` | `scripts/federated/build_client_manifest.py` | v1 | S1-PR-06 | Developers |
| `ClientPartitionEvidence` | `docs/evidence/s1-pr-06/client_partition_summary.v1.json` | v1 | S1-PR-06 | Review / reproducibility |
| `ClientPartitioningTests` | `tests/federated/test_client_partitioning.py`, `tests/federated/test_client_sampling.py` | v1 | S1-PR-06 | CI |
| `ClientPartitioningDocs` | `docs/federated-client-partitioning.md` | v1 | S1-PR-06 | Developers |
| `FLRealSmokeConfig` | `config/fl_real_smoke.v1.json` | v1 | S1-PR-07 | CI, Developers |
| `FLRealSmokeTaskExampleAdapter` | `ppsi/federated/task_examples.py` | v1 | S1-PR-07 | S1-PR-07, S2-PR-06 |
| `FLRealSmokeEntryPoint` | `scripts/federated/fl_real_smoke.py` | v1 | S1-PR-07 | CI, Developers |
| `FLRealSmokeNotebook` | `notebooks/S1_PR_07_Real_Flower_Smoke.ipynb` | v1 | S1-PR-07 | Developers, Review |
| `FLRealSmokeDerivedClientManifest` | `data/protocol/INTERNAL_DO_NOT_UPLOAD_fl_real_smoke_client_manifest_v1.parquet` | v1 | S1-PR-07 | S1-PR-07 internal |
| `FLRealSmokeSamplingTrace` | `data/protocol/INTERNAL_DO_NOT_UPLOAD_fl_real_smoke_sampling_trace_v1.jsonl` | v1 | S1-PR-07 | S1-PR-07 internal |
| `FLRealSmokeInputEvidence` | `docs/evidence/s1-pr-07/task_example_inputs.v1.json` | v1 | S1-PR-07 | Review / reproducibility |
| `FLRealSmokeInitializationEvidence` | `docs/evidence/s1-pr-07/fl_real_smoke_initialization.v1.json` | v1 | S1-PR-07 | Review / reproducibility |
| `FLRealSmokeResolvedExperimentConfig` | `docs/evidence/s1-pr-07/fl_real_smoke_experiment_config.v1.json` | v1 | S1-PR-07 | Review / reproducibility |
| `FLRealSmokeSummary` | `docs/evidence/s1-pr-07/fl_real_smoke_summary.v1.json` | v1 | S1-PR-07 | Review / reproducibility |
| `FLRealSmokeTests` | `tests/federated/test_real_task_examples.py`, `tests/federated/test_fl_real_smoke.py` | v1 | S1-PR-07 | CI |
| `FLRealSmokeDocs` | `docs/federated-real-smoke.md` | v1 | S1-PR-07 | Developers |

## Rules

- Each logical artifact has one canonical repository-relative location.
- Do not create parallel aliases for the same contract.
- `uv.lock` is generated by uv and must not be edited manually.
- `.venv/` is a local reconstruction of the environment and is not a repository artifact.
