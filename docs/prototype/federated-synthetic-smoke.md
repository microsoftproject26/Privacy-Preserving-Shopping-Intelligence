# Flower Synthetic FedAvg Smoke Test

## Overview
This document specifies the canonical artifacts for the Flower Synthetic FedAvg Smoke Test (`S1-PR-02`), verifying the core federated learning lifecycle using Flower `1.33.0`, PyTorch, and Ray.

## Test Coverage
The suite comprises **26 unit and integration tests** ensuring the following invariants are strictly covered:
- >=2 logical clients participating in training and evaluation.
- >=3 FedAvg rounds executed.
- Finite losses and parameters verified throughout training.
- Non-zero global model changes confirming actual learning.
- Flower aggregation mathematically equals an independent weighted oracle.
- Redistribution proof (updates redistribute to all clients correctly).
- Two-run repeatability with exact determinism (Seed `13`).
- Empty client data gracefully handled without fatal crashes.
- Malformed update shapes detected and rejected.
- Client/Ray exceptions surface visibly through `SmokeValidationError` rather than silently failing.
- NaN/non-finite updates detected and rejected.
- No participant/no replies round failures handled gracefully.
- Config schema/version mismatch failures.
- Summary schema/version mismatch failures.
- Config hash mismatch failures.
- uv.lock hash mismatch failures.
- Artifact manifest tamper/hash mismatch failures.
- Downstream producer contracts structured correctly for S1-PR-05 consumption.

## Execution Requirements
Run the smoke test with:
```bash
uv run --locked python scripts/federated/fl_synthetic_smoke.py \
  --config config/fl_synthetic_smoke.v1.json \
  --output docs/evidence/s1-pr-02/fl_synthetic_smoke_summary.v1.json
```

## Hash Provenance
The lifecycle enforces strict provenance using `artifact_manifest.v1.json`, hashing all key deliverables post-execution to guarantee artifact immutability.
