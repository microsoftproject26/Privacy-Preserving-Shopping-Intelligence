# Flower Real Data FedAvg Smoke Test

`S1-PR-07` validates the end-to-end simulated federated training pipeline on real REES46 TaskExamples using Flower 1.33.0, PyTorch, and Ray.

**NON-SCIENTIFIC INTEGRATION SMOKE.** This smoke test proves pipeline connectivity and determinism from real frozen data to federated aggregation. It does not establish recommendation quality, baseline comparisons, QR, or official GRU performance.

## Pipeline Path

```text
real frozen T1 TaskExamples
→ real C1 users / S1-PR-06 opaque client IDs
→ deterministic task-eligible client sampling
→ Phase1Batch (canonical representation)
→ LocalTrainerCore (canonical local trainer)
→ FlowerLocalAdapter (Flower 1.33 Message API)
→ Flower / Ray FedAvg (RealDataTracingFedAvg)
→ structured public evidence & ExperimentResult v1
```

## Data Scope & Boundaries

- **Base C1 ClientManifest**: 388,789 accepted C1 users (from merged S1-PR-06 client partitioning).
- **Frozen T1 TaskExample artifacts**: Generated from the upstream deterministic 25% C1 measurement/example slice (`data/examples/INTERNAL_DO_NOT_UPLOAD_task_examples_t1_train_v1.proposed.parquet` and `data/examples/INTERNAL_DO_NOT_UPLOAD_task_examples_t1_v1.proposed.parquet`). No additional filtering or subsampling is performed by S1-PR-07.
- **T1 Smoke Eligible Clients**: Measured directly from actual T1 TRAIN TaskExamples (~95,990 clients with `t1_train_example_count > 0`).
- **Client Identity**: Reuses merged `ppsi.federated.clients.client_id_from_user` to map raw user IDs to opaque `client-v1-<sha256>` identifiers.
- **No Raw Events**: The raw event stream is not accessed; sequential histories are unavailable.
- **Zero-History Representation**: Uses physical `L=1` with semantic length 0 (`lengths=0`, `history_mask=False`). Continuous query features are derived deterministically from `decision_order`.
- **No TEST Access**: `data/sealed_test/` is completely untouched.

## Deterministic Client Sampling

- Seed: `13`
- Rounds: 3
- Clients per round: 4 (sampled without replacement within rounds from clients with `t1_train_example_count > 0`)
- Sampler: Merged `ppsi.federated.sampling.sample_clients`
- Traces: Precomputed and verified per round against Flower worker assignments.

## Materialized Smoke Slices

To avoid multi-worker concurrent I/O on 48 MB parquets:
- **Train slice**: Up to 32 rows per selected client (stable sort by `client_id`, `session`, `decision_order`).
- **Validation slice**: First 256 rows across valid validation clients.
- Slices contain only `client_id`, `session`, `decision_order`, and `label_code`. Raw user IDs and raw string labels are never stored.

## Canonical Command

```powershell
uv run --locked python scripts/federated/fl_real_smoke.py --config config/fl_real_smoke.v1.json
```

## Artifacts

### Internal (Ignored) Artifacts
- `data/protocol/INTERNAL_DO_NOT_UPLOAD_fl_real_smoke_client_manifest_v1.parquet` — Derived client manifest with T1 example counts and eligibility.
- `data/protocol/INTERNAL_DO_NOT_UPLOAD_fl_real_smoke_sampling_trace_v1.jsonl` — Full selection trace with opaque client IDs.
- `data/examples/INTERNAL_DO_NOT_UPLOAD_fl_real_smoke_t1_train_v1.parquet` — Materialized training slice.
- `data/examples/INTERNAL_DO_NOT_UPLOAD_fl_real_smoke_t1_validation_v1.parquet` — Materialized validation slice.

### Public-Safe Evidence Artifacts
- `docs/evidence/s1-pr-07/task_example_inputs.v1.json` — Input file SHA-256 hashes, row counts, vocabulary verification.
- `docs/evidence/s1-pr-07/fl_real_smoke_initialization.v1.json` — Initial model digest and architecture proof.
- `docs/evidence/s1-pr-07/fl_real_smoke_experiment_config.v1.json` — Resolved `ExperimentConfig` (R2A, T1, non-scientific).
- `docs/evidence/s1-pr-07/fl_real_smoke_summary.v1.json` — Complete summary with round digests, oracle verification, validation history, and reproducibility proof.
- `artifacts/experiment-results/*.result.json` — Validated `ExperimentResult` v1.
- `notebooks/S1_PR_07_Real_Flower_Smoke.ipynb` — Human-readable public-safe walkthrough.
