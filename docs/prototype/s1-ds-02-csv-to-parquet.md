# S1-DS-02 — REES46 CSV to Parquet

Final implementation of the Phase 1 data conversion task. The code reads the
October 2019 REES46 CSV in bounded ordered batches and produces
`ProcessedRawParquet v1` without loading the full CSV into memory.

## Files included

| File | Purpose |
|---|---|
| `scripts/data/convert_rees46.py` | The complete readable conversion code; no local Python modules are imported. |
| `config/conversion.v1.json` | Versioned conversion, validation, compression, and output settings. |
| `config/dataset_manifest.v1.json` | Approved input filename, size, SHA-256, row count, schema, and provenance. |
| `requirements.txt` | Minimal runtime dependencies. |
| `evidence/` | Small machine-readable evidence from the completed full run. |
| `.gitignore` | Prevents raw CSV, ZIP, Parquet, output data, and Python cache from being committed. |

## Setup

```powershell
python -m pip install -r requirements.txt
```

The default config expects this folder and `Dataset/` to be siblings:

```text
parent/
├── Dataset/2019-Oct.csv/2019-Oct.csv
└── Repository/
```

Paths may be changed in `config/conversion.v1.json`; scientific validation
rules and the recorded raw-file identity must not be weakened.

## Commands

Run from this folder:

```powershell
# Fast synthetic checks: valid rows, all rejection codes, malformed CSV,
# deterministic repeatability, and Parquet read-back.
python scripts/data/convert_rees46.py self-test

# Validate filename, byte size, columns, and SHA-256 before conversion.
python scripts/data/convert_rees46.py verify-input --config config/conversion.v1.json

# Run the full bounded-memory conversion.
python scripts/data/convert_rees46.py convert --config config/conversion.v1.json

# Validate every output hash plus row/source-row reconciliation.
python scripts/data/convert_rees46.py validate-output --output artifacts/full-run
```

The converter refuses to overwrite an existing final or staging directory.
This protects previous evidence. Remove or archive an old local output
explicitly before intentionally starting a new run.

## Output contract

Accepted rows contain:

| Field | Type |
|---|---|
| `event_time` | `Datetime(us, UTC)` |
| `event_type` | `String` |
| `product_id` | `String` |
| `category_id` | `String` |
| `category_code` | nullable `String` |
| `brand` | nullable `String` |
| `price` | `Float64` |
| `user_id` | `String` |
| `user_session` | nullable `String` |
| `source` | `String` |
| `source_file` | `String` |
| `source_row_number` | `Int64` |

Input order is preserved. The deterministic tie key is
`(event_time, source_row_number)`, and `source_row_number` is one-based and
excludes the CSV header.

Validation assigns at most one primary reason to an invalid field-level row:

1. `MISSING_REQUIRED_ID`
2. `INVALID_TIMESTAMP`
3. `UNKNOWN_EVENT_TYPE`
4. `INVALID_PRICE`

Structural CSV failures are reported as `CSV_PARSE_ERROR`. The default config
does not publish final output if any rejected row exists; it keeps the staging
evidence for review instead of silently removing research data.

## Verified full-run result

| Measurement | Result |
|---|---:|
| Source rows | `42,448,764` |
| Accepted rows | `42,448,764` |
| Rejected rows | `0` |
| Parquet files | `1` |
| Parquet row groups | `85` |
| Parquet bytes | `803,389,955` |
| Parquet SHA-256 | `8905CE1C6918BF00C2F96DC946B1D3A6A31B3928DB2C80AB5BC5C318E9DF61C6` |
| Conversion time | `114.555 seconds` |
| Peak RSS | `5,447,294,976 bytes` |
| Row reconciliation | `PASS` |
| Artifact hashes/read-back | `PASS` |

Event counts:

- `view`: `40,779,399`
- `cart`: `926,516`
- `purchase`: `742,849`

The two null `user_session` values are preserved. Their later sessionization
treatment belongs to the cohort/session policy and is not decided by this
conversion task.

## Artifact policy

The 803 MB Parquet dataset is a local/generated artifact and is intentionally
excluded from this upload package and normal Git history. Its filename, size,
SHA-256, and row counts are recorded in `evidence/dataset_manifest.v1.json`.
Use an approved shared artifact store if the team needs to exchange the binary.
