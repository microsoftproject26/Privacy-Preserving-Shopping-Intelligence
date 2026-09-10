# Federated Client Partitioning

S1-PR-06 implements user-to-client partitioning and reproducible client
sampling for simulated federated learning.

## Data scope

The downloaded data is already the official deterministic **25% user slice**
(`user_id % 4 == 1`). The C1 cohort is consumed as-is from the upstream
cohort manifest. **No additional 25% filter or random population sampling is
applied.**

## One user = one client

Each canonical `user_id` maps to exactly one opaque `client_id`. Users are
never split across clients and clients never merge different users.

## Client identity

Client IDs are:

- **deterministic** — same user always produces the same ID
- **seed-independent** — experiment seed does not affect the mapping
- **versioned** — the identity version is embedded in the ID
- **opaque** — derived via SHA-256; raw user IDs cannot be trivially recovered

Format: `client-v1-<64 lowercase hex chars>`

Algorithm:

```text
client_id = "client-v1-" + SHA256("ClientIdentity/v1\n" + str(user_id))
```

The mapping does **not** claim anonymity or differential privacy.

## ClientManifest v1

An internal parquet artifact with one row per C1 client, sorted by
`client_id`. Fields:

| Field | Description |
|---|---|
| `manifest_version` | `"client_manifest_v1"` |
| `client_id` | Opaque versioned client identifier |
| `cohort` | `"C1"` |
| `eligible` | `true` for all C1 users |
| `train_event_count` | Number of TRAIN-period events (null if raw events unavailable) |
| `validation_event_count` | Number of VALIDATION-period events (null if raw events unavailable) |
| `train_event_type_counts` | JSON string of event type → count for TRAIN |
| `validation_event_type_counts` | JSON string of event type → count for VALIDATION |
| `first_allowed_time` | Earliest event timestamp in TRAIN/VALIDATION (null if raw events unavailable) |
| `last_allowed_time` | Latest event timestamp in TRAIN/VALIDATION (null if raw events unavailable) |
| `event_statistics_status` | `"not_measured_raw_events_unavailable"` (or `"measured"` if raw events supplied) |
| `task_example_counts_status` | `"not_measured"` — deferred to downstream integration (S1-PR-07) |

### Event statistics policy

Real per-client raw event statistics are **not measured** in the current real evidence
because the full raw event parquet (`data/raw/processed_raw_parquet_v1.parquet`) is not
distributed locally and is not required for the S1-PR-06 MVP partitioning/sampling scope.
Event counts and timestamps in the real client manifest remain explicitly null with
`event_statistics_status = "not_measured_raw_events_unavailable"`.

The optional `_compute_event_stats(...)` pipeline is implemented and verified under
controlled unit-test fixtures for future enrichment: when raw events are supplied,
`excluded_sessions` is **required** to filter out boundary-crossing sessions via
`(user_id, user_session)`, and temporal boundaries are dynamically loaded from
`config/s1-ds-05-06.v1.json`.

The semantic manifest identity (`manifest_content_sha256`) is computed over the
canonical, sorted CSV text representation of the manifest DataFrame to ensure
determinism independent of physical serialization details (Parquet metadata,
row-group layout, IPC chunks).

The manifest is an internal artifact (`INTERNAL_DO_NOT_UPLOAD`) and must not
be committed to the repository.

## Client sampling

Each federated round selects clients using a deterministic sampler:

1. Sort eligible client IDs lexicographically
2. Derive a round seed: `SHA256(sampler_version | experiment_seed | round_index)`
3. Select `clients_per_round` IDs without replacement using `random.Random`
4. The same client may appear in different rounds

Invariants:

- Same pool + seed + round → identical selection
- `clients_per_round > pool_size` → explicit error (no silent shrinking)
- No Python `hash()` — uses SHA-256 exclusively

## Privacy boundary

Detailed real client membership stays in ignored internal data paths.
Public evidence contains only aggregate counts, distributions, and
selection digests — never raw user IDs.

The `data/sealed_test/` directory is never accessed.

## Downstream consumer

[#20 (S1-PR-07)](https://github.com/microsoftproject26/Privacy-Preserving-Shopping-Intelligence/issues/20)
consumes `ClientManifest v1` and `ClientSamplingTrace v1`.
