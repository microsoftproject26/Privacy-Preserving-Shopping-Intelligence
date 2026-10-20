# REGISTRY_V2_TEST_EXT.json: the TEST registry extension (DATA-09)

Authority: PREG1_AMENDMENT_06D §2; EVALUATION_CONTRACT_V4 §10 (E10.1 to E10.3); MANIFEST_SPEC_V4 M07 / M08 and
`global_rules.test_gate`. Code: `code/v4data09/d9_registry.py` (the rule), `d9_finalize.py registry` (the writer) and
`code/v4int/test_roles.py` (the only reader).

## Status
- The file does **not** exist before the TEST look.
- It is written once, on the server, by `code/v4data09/job_test_final.sh`. That happens only after
  `company/gates/ALL_METHODS_LOCK.json` exists, the Operations Director has appended a TEST_EXPOSURE_REGISTER entry with
  `access_purpose = MANIFEST_BUILD` and `split = RETRO_TEST` binding its sha256, and `job_test_build.sh` has built the five
  RETRO_TEST manifests under that gate.
- The file is created with O_EXCL. An existing file is accepted only when its bytes are identical. It is never
  overwritten.

## What the file contains
The file is **REGISTRY_V2 plus additions only**:
- Every REGISTRY_V2 top-level key, role and manifest entry is canonical-JSON identical. `self_sha256` is recomputed.
- `extends` records REGISTRY_V2's path, file sha256 and self_sha256. DATA_PROTOCOL_LOCK_V2 must pin that REGISTRY_V2.
- It adds five roles. Every support (train) manifest is the unchanged REGISTRY_V2 entry.

| role | train (unchanged) | eval (new, RETRO_TEST) |
|---|---|---|
| `MAIN_TEST` | `MAIN_TRAIN_5_100_131072` | `MAIN_RETRO_TEST_5_100_131072` |
| `UNSEEN_HOLDOUT_TEST` | `UNEW_HOLDOUT_SUPPORT_32768` | `UNEW_HOLDOUT_RETRO_TEST_32768` |
| `TRUE_COLD_TEST` | none (eval only) | `TRUE_COLD_RETRO_TEST` |
| `SPARSE_PANEL_TEST` | `SPARSE_PANEL_16384_SUPPORT` | `SPARSE_PANEL_16384_RETRO_TEST` |
| `HEAVY_PANEL_TEST` | `HEAVY_PANEL_16384_SUPPORT` | `HEAVY_PANEL_16384_RETRO_TEST` |

- It adds the five RETRO_TEST manifest entries. They use the REGISTRY_V2 entry shape, plus `v4_split = RETRO_TEST` and
  the two MANIFEST_SPEC M07 `header_extra` values: `all_methods_lock_sha256` and `test_exposure_register_entry_id`.
- `extension_test_gate` holds the gate binding: the lock sha256, the register entry id and the entry's sha256. Every
  RETRO_TEST entry and header repeats the same two values.
- It also adds `extension_task`, `extension_code`, `extension_build_receipt` and `extension_note`. M09 (TEST cold-h) is
  not registered.

## How it is read
- Only `v4int.test_roles` reads the file: `resolve_role`, `verify_test_manifest`, and the in-process wrappers installed by
  `install`.
- Each read first calls `v4data09.d9_gate.check_test_gate`, and refuses if the gate is closed or differs from
  `extension_test_gate`.
- The unedited `v4train.registry` loader and `v4int.verify` still refuse every TEST id. That refusal is the intended
  default.
