# REV-DEV-FTPRE: independent read-only review of DEV-FTPRE (Claude Opus 5.5, 2026-09-30)

**Verdict: PASS-WITH-FIXES.** I found no BLOCKER and no MAJOR. The five findings are all MINOR: test-quality and
robustness items. None of them stops the server `--dry-run` or smoke run. Fix M1 and M2 before the DL row says the NCs
cover the launcher and reeval wiring.

**Scope and method:** read-only. The only files written are this review and the pytest basetemp under
`.local_ai/_pytest_tmp/REV-DEV-FTPRE/`. No code or contract was edited. No server, no install, no real data.

**How the diffs were taken:** against the pre-edit copies in `devftpre_backup/`. Their sha256 equal the handoff's
"before" column. All ten "after" sha256 match the handoff (MEASURED).

## Evidence (MEASURED unless marked)

### Frozen identities
| identity | value | note |
|---|---|---|
| `v4eval.evaluator_code_sha256()` | `61b64e61c1a2e4f5…d8962e` | unchanged |
| `v4trunc_code_sha256()` | `4d766d5b1eb5…` | unchanged |
| `device_code_sha256()` | `114b0baa…` | changed, as the handoff says |

`launch_fl.py` (a23d5b4f…) and `lr_screen.py` (e81667d3…) are not in the DEV-FTPRE diff.

### Spec digests: independent check
- I loaded the pre-edit `launch_device.py` (48eee710) side by side with the new one.
- I compared `spec_digest(device_spec(...))` for LO / FT_C / FT_FA × 7 roles (MAIN, UNSEEN_VAL, UNSEEN_HOLDOUT,
  SPARSE_PANEL, TRUE_COLD, CALIBRATION, MAIN_TEST) × {select, selection, neither}.
- Result: **0 mismatches**. Every refusal also has the same exception type and text.
- `grid_cells("FT_C")` is identical in both versions.
- `FT_PROCEDURE` maps FT_C→FT_C and FT_FA→FT_FA, so `make_runners` is unchanged for those arms
  (launch_device.py:270).
- The spec only gains FT_PRE-only keys, under `if arm in SELECTION_ARM` (launch_device.py:231-234).

### Test suites
Command: next_item_eda venv, `-B -p no:cacheprovider -q <suite> --basetemp _pytest_tmp/REV-DEV-FTPRE/<suite>`, with
TMP/TEMP set to the same directory.

| suite | result | vs the handoff |
|---|---|---|
| tests_int | exit 0, **293 passed / 90 xfailed** | +18 / +4 |
| tests_ana | exit 0, **160 passed / 53 xfailed** | equal |
| tests_trunc | exit 0, **52 passed / 15 xfailed** | equal |

- The tests_int difference is exactly `test_fl14_06l.py`. That file has 22 tests (18 passed + 4 NCs), was modified at
  01:19 local time, and belongs to FL-14. It was written after this packet.
- DEV-FTPRE's own tests_int count therefore equals the handoff's 275 / 86.

### Negative controls
All 9 new NCs and the rewritten `test_nc_ft_pre_base_unchecked` were run with `--runxfail`. Every one fails. For the
failure reasons:
- `test_nc_ft_pre_accepts_base`: preflight did not raise.
- `test_nc_ft_pre_main_trained_base`: load_pre_base did not raise.
- `test_nc_no_ft_restriction_removed`: run did not raise.
- `test_nc_persistence_without_evidence`: `'NO_CI' == 0`.
- `test_nc_rankable_per_user`: the RANKABLE mean differs by 0.0113.
- `test_nc_prefix_only_base_check` and `test_nc_ft_pre_base_unchecked`: the report was written (rc 0 ≠ 2).
- Two NCs are **tautological**; see M1. `test_nc_ft_c_from_theta_pre` fails with "`<lambda>` did not raise", and
  `test_nc_no_ft_rows_not_descriptive` fails with `'REEVAL' == 'DESCRIPTIVE'`, but that value comes from the test's own
  input.

## Checks against the brief

### 1. FT_PRE = θ_pre + the registered FT_C procedure / selection: holds

**The base is θ_pre only**
- `check_ft_pre_args` (launch_device.py:834-856) refuses `--base` / `--base-kind`, a missing `--init-from` and
  `--select`. It runs in preflight before the role check (:918).
- `load_pre_base` (:859-877) calls the unmodified FL-12 loader, `launch_fl.load_pretrain_init` (launch_fl.py:783-813):
  - The bytes must match the record's sha256 / byte count before `torch.load`.
  - It checks endpoint_6.0, run label PRETRAIN_FOR_FL, run id PRETRAIN_FOR_FL_* and PRETRAIN_TRAIN_5_100_NONMAIN.
  - It denies the MAIN / CALIB / FULL_TRAIN manifests and the C_BIG / C_SMALL / C_FULL / C_MATCHED markers.
- Then it re-asserts `check_init_identity` and `apply_init_state` (the strict load + state_digest), and re-hashes the
  broadcast state.
- `refuse_theta_pre_base` (:880-891) works in both directions: FT_C / FT_FA from a PRETRAIN_FOR_FL_* base refuse, and
  FT_PRE whose start is not the θ_pre identity refuses.

**The selection is the registered FT_C one**
- `FT_PRE_SELECTION` = `contracts/device/SELECTION_FT_C.json`: local sha256 `e47965d2…` = DL 231, cell FT_C_f0.3_P4,
  made on UNSEEN_VAL.
- A non-smoke run refuses any other file, at preflight (:855) and again at run time (:1068).
- `device_spec` maps the FT_C cell to `FT_PRE_f<frac>_P<passes>` and refuses UNSEEN_VAL and `--select` (:182-186,
  :210-224).
- The UNSEEN_VAL manifest binding check (`verify_selection_manifest`) still applies.

**The procedure is FT_C's**
- `FTRecipe(arm=FT_PROCEDURE[arm])` is FT_C for FT_PRE.
- The test shows FT_PRE's ranks and value are bitwise equal to FT_C's on the same start.

**Rows and LAUNCH record**
- Rows get method / method_code FT_PRE, `ft_procedure`, `procedure_selection_*` and the `base_*` labels.
- None of these fields collides with COMPARISON_KEY or with an explicit keyword in `device_rows`.
- LAUNCH `start` carries kind PRE, `init_from`, `init_record_sha256`, and the selection `file_sha256`.

### 2. Leakage: none found
- No new code reads MAIN-VALIDATION or TEST values.
  - The FT_PRE launcher reads the init record and the registered selection. The selection's `values_primary` are
    UNSEEN_VAL values, and nothing uses them.
  - launch_trunc reads only its own rank arrays.
  - reeval computes, and reads no result.
- The TEST gate in `preflight` is untouched and still runs before any GPU or data access.
- reeval `check_args` still checks the TEST gate first.
- FT_PRE is refused on UNSEEN_VAL, which keeps the selection panel clean.
- The no-FT rows are DESCRIPTIVE. The report builds `no_ft_unseen_val` with its own DESCRIPTIVE label
  (retention_report.py:3149-3171), and no selection path reads them.
- v4eval is unchanged (61b64e61).

### 3. TRUNC_PER_USER.npz: holds
**Format**
- The format matches `load_trunc_per_user` (retention_report.py:3601-3622): a 0-d `format`, sorted-unique int64 `u__`,
  and finite float64 `m__` of the same length.
- The key function is the same (launch_trunc.py `trunc_cell_key` = retention_report.py:3595), and the method-or-arm
  rule is the same.

**Values**
- The values come from `v4eval.metrics.per_user` over eval_mask (metrics.py:66-72). That is exactly what
  `population_block` averages.
- The mean is checked against the row value to 1e-9 before anything is written (launch_trunc.py:424-446).

**Determinism**
- The file is written sorted, stored, with a fixed date / system / attributes, and atomically (:454-472).
- Byte-identical on a rerun is MEASURED in the test.
- Byte-identical across hosts is INFERENCE: it assumes the same numpy `format.write_array` header.

**Recording**
- The sha256 is recorded in STATUS.json and RESULT_ROWS.json (:902, :1016).

**Without the file**
- The report falls back to NO_CI (`_trunc_ratio_ci` → NO_PER_USER_EVIDENCE → `_persistent_hit_or_status` NO_CI).
- This is MEASURED in `test_persistence_rule_uses_the_evidence`.

### 4. reeval no-FT references: holds
**Admission**
- `check_no_ft_args` (reeval.py:240-249) runs inside `check_args` (:237), before `_sha(args.ckpt)` or any other file
  read.
- UNSEEN_VAL admits only C_SMALL/central/SASREC_D64_B2 and FA/fl/SASREC_D64_B2.
- `check_no_ft_source` (:252-270) adds two source checks:
  - FA must have LAUNCH arm FA, no `init_from`, and (from `source_fl`) fl05 role MAIN and label PRODUCTION.
  - C_SMALL must not be a registered PRE / C_FULL source. The non-dry path also enforces the MAIN pair (:836).

**Labels and scope**
- Rows get comparison_status DESCRIPTIVE plus the `no_ft_reference*` labels (:939). row_status stays REEVAL.
- The record gets `no_ft_reference` (:977-978).
- Every other role gets `{}`, so it is unchanged.
- PRE / C_FULL / C_BIG on UNSEEN_VAL are now refused. This matches the index / report rule
  (retention_report.py:1212, ana_block_index.py:311) and ANA-04.

### 5. Tests
See the counts and NC results above, and findings M1 and M2.

### 6. Behaviour outside scope
- **FT_C / FT_FA**
  - An FT_C / FT_FA base from θ_pre is now refused. This is intended under 06L-C §C7.
  - `--init-from` is refused for every arm other than FT_PRE.
  - `device_code_sha256` changed, so the 06E §8 guard refuses a mixed-tree resume. In-flight LO / FT_C runs and
    old-tree Study B runs must finish on their own tree. The handoff discloses this.
- **ANA side**
  - The index now refuses the old layout, an FT_C summary inside an FT_PRE directory. This is consistent with the new
    launcher.
- **Line endings**
  - `launch_trunc.py` was CRLF and is now LF. This has no functional effect, and the file is not part of any code
    identity that a receipt or spec compares. INFO only.

## Findings

**M1 (MINOR): two NCs are tautological and prove nothing about the code.**
- **Where**
  - `code/tests_int/test_dev_ftpre.py:375-378`, `test_nc_ft_c_from_theta_pre`. It monkeypatches
    `D.refuse_theta_pre_base` to a no-op lambda, then calls that same patched function. It fails with "`<lambda>` did
    not raise".
  - `code/tests_int/test_dev_ftpre.py:388-393`, `test_nc_no_ft_rows_not_descriptive`. It patches `RV.no_ft_labels`,
    then passes `extra=RV.no_ft_labels(...)` itself into `relabel`. The test supplies the missing label, so the NC only
    tests its own input.
- **Failure scenario:** someone deletes `refuse_theta_pre_base(args.arm, start_rec)` from `launch_device.run`
  (launch_device.py:1110), or the `no_ft_labels` injection from `reeval.run` (reeval.py:939). Both NCs still "fail as
  intended", and every positive test still passes.
- **Minimal fix**
  - Make each mutant act on the wiring. For example, factor the run-time start check and the row / record label step
    into small helpers that `run()` calls, such as `start_record_for(arm, …)` and `reeval_row_extra(args, extra)`, and
    target those helpers in the NC.
  - Or drop the two NCs and disclose that the wiring is covered only by the server smoke run (see M2).

**M2 (MINOR): the run()-level FT_PRE / no-FT wiring has no CPU test, and `--dry-run` does not reach it.**
- **Where**
  - `launch_device.run`:
    - the θ_pre branch (launch_device.py:1102-1110);
    - the `ft_pre_row_labels` merge into the row overrides (:1145-1146);
    - the selection `file_sha256` in LAUNCH (:1165-1167).
  - `reeval.run`: the non-dry row labels and record (reeval.py:939, :977-978).
- **What the tests do instead**
  - All `RV.run` calls in the tests use `--dry-run`.
  - The FT_PRE row test passes `overrides={"base_source": "PRE"}` by hand (test_dev_ftpre.py `_row_fn`), not
    `ft_pre_row_labels`.
  - `launch_device --dry-run` returns right after preflight, so θ_pre is never loaded.
- **Failure scenario:** a wiring slip only shows up on the server, for example `base_*` labels missing from the rows,
  or `start_rec` lacking `init_record_sha256`. For the second case, the ANA index would then refuse FT_PRE only at
  freeze time.
- **Minimal fix**
  - Treat the FT_PRE smoke (`--smoke --smoke-clients 8`) as **mandatory** before production. Check its LAUNCH_0
    `start.kind == "PRE"`, `start.init_record_sha256`, and the rows' `base_*` / `ft_procedure` fields.
  - Run one `ana_block_index --dev` pass on the smoke directory copy, renamed and relabelled PRODUCTION only in a
    scratch copy, or rely on the synthetic index test.
  - Optionally, add a CPU test that drives the `run()` branches with the harness monkeypatched.

**M3 (MINOR): FT_PRE compares a sha256 of a file that is rewritten on every call.**
- **Where**
  - `v4int/pretrain_small.py:288-305`. `fl_init_record` rewrites `PRETRAIN_FOR_FL_INIT.json` each time it is called
    (:408 and the `--fl-init-only` path :439), with a fresh `utc`. It uses no O_EXCL and no idempotence check.
  - `tools/retention_report.py:1278-1301` (`ft_pre_vs_pre_problem`) and the index (ana_block_index.py:649-656) require
    FT_PRE's `start_init_record_sha256` to equal PRE's `pre_init.init_record_sha256`.
- **Failure scenario:** the record is regenerated between the FT_PRE launches and the PRE re-evaluation, for example by
  a `--fl-init-only` rerun or a resume leg that returns 0. The block is then refused as ARM_IDENTITY, even though θ_pre
  (`checkpoint_sha256`) is byte-identical. K9c would be blocked.
- **Severity:** it fails closed, so there is no wrong number, and that is why this is MINOR.
- **Minimal fix**
  - Operational: record in the DL that `PRETRAIN_FOR_FL_INIT.json` is frozen once written, and never rerun
    `--fl-init-only` after the first FT_PRE / FA_WARM / PRE-reeval consumer. Launch both FT_PRE seeds and the PRE
    reeval only after PRE is final.
  - Or in code, a later packet: make `fl_init_record` refuse to overwrite a record that differs only in `utc`.

**M4 (MINOR): the index copies TRUNC_PER_USER.npz without checking the recorded sha256.**
- **Where:** `tools/ana_block_index.py:603-605`. The file is copied if present. Its sha256 is not compared with the
  `per_user.sha256` that launch_trunc records in the run's RESULT_ROWS.json / STATUS.json (launch_trunc.py:902-907,
  :1016-1020).
- **Failure scenario:** a replaced or stale npz enters the block unnoticed. The report's 1e-9 check (every cell's mean
  must reproduce its row value, retention_report.py:3636-3639) catches almost every such case. A file that differs only
  in how per-user values are distributed while keeping the same means would still pass, and it would change the CI.
- **Minimal fix:** in `trunc()`, when the file is present, require
  `sha256(file) == RESULT_ROWS.json["per_user"]["sha256"]`. If the recorded value is absent, which is the case for
  pre-DEV-FTPRE runs, the file must be absent too.

**M5 (MINOR): the registered selection is trusted by path, not by a pinned sha256.**
- **Where:** `launch_device.py:813-831`. The "registered" file is whatever currently sits at
  `V4/contracts/device/SELECTION_FT_C.json`. The code has no constant for DL 231's `e47965d2…`.
- **Failure scenario:** the contracts copy on a deployed tree is replaced by another verified FT_C selection, for
  example a seed-2027 one. A real FT_PRE run would accept it. It would still be visible, because the LAUNCH record
  carries `file_sha256`.
- **Minimal fix:** add `FT_PRE_SELECTION_SHA256 = "e47965d2…"` and refuse on mismatch in a non-smoke run. This is one
  line plus a test.

## Not findings (checked and OK)
- **Report checks for FT_PRE**
  - `pre_init` is allowed only on PRE, and `start_init` / `start_init_record_sha256` only on FT_PRE
    (retention_report.py:1215-1224).
  - FT_PRE without a PRE entry refuses.
  - PRE has exactly one identity: its only admissible file is `endpoint_6.0.pt`, so PB and CB cannot disagree
    (reeval.py `CENTRAL_SOURCES["PRE"]`).
- **Study B per-user evidence**
  - It uses the sub-panel manifest `m` of the same `execute` call (launch_trunc.py:983-1015).
  - It keeps only the rows of `spec["selected_cell"]`.
  - It requires exactly one kept primary row.
- **Study A resume:** it recomputes from the saved ranks when the cell file is missing or its sha256 differs, with the
  1e-9 value check. This is safe.
- **Handoff items 4 and 5:** the FA_WARM `init_from` versus PRE `pre_init` cross-check and the EVENTS_TO_X
  `self_sha256` round-trip are correctly listed as not done, and neither is a regression.

The thesis_reference was not updated by this reviewer. It is out of scope for a read-only review, so Ops should record
this verdict.
