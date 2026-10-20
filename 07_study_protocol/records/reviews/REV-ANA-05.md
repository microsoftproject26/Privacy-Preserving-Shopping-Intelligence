# REV-ANA-05: independent read-only review of ANA-05

Reviewer: Claude Sonnet 5.5, 2026-09-30. Read-only: no code or contract edited, no server, no installs, no Codex.

**VERDICT: PASS-WITH-FIXES.** There are no blockers. There is one MAJOR item (a fail-open path for warm runs that arrive without a run directory) and four MINOR items. The MAJOR item needs an Ops decision or a one-line rule before the freeze. It is not a code defect in the handoff's stated scope.

## Verified (MEASURED by the reviewer)
- **File hashes.** All 11 changed or new files match the handoff's "after" sha256 values. The "before" copies in the session scratchpad `ana05_backup/` match the claimed "before" shas.
- **Frozen items.**
  - v4eval `evaluator_code_sha256()` = `61b64e61c1a2e4f5…8d962e`, unchanged.
  - `ANA_CONTRASTS_v3.json` = `a3d1456d…`, unchanged.
  - `labels.CONFIRMATORY` is still the 26 K (asserted in `test_ana05.py:113-117`).
- **`--claims-only` output.** I ran the old script (the `ana05_backup` copy) and the new script on `ANA_CONTRASTS_v3.json`. The two CLAIM_REGISTER.json files are byte-identical (`cmp`), and so are the two HEADLINE_TEMPLATE.md files. The hashes are CLAIM_REGISTER `9ce7103ab666…` and HEADLINE_TEMPLATE `2d6aa76cc475…`, as claimed.
- **Tests, `tests_ana`:** 181 passed, 63 xfailed (exit 0).
- **Tests, `tests_int/test_ana05_reeval.py`:** 6 passed, 3 xfailed.
- **`--runxfail -k test_nc_`, `tests_ana/test_ana05.py`:** 9 of 9 fail at an `AssertionError`.
- **`--runxfail -k test_nc_`, `tests_int/test_ana05_reeval.py`:** 3 of 3 fail at an `AssertionError`.
- **Not re-run:** `--runxfail` on the legacy suites, and the worker's old-versus-new whole-`RETENTION_RESULT` diff. My attempt at the diff failed on a tooling issue in my scratch tree, not on the code. That diff remains the worker's claim only (see finding 5).

## Item 1: C_FULL s2027 is DESCRIPTIVE only
**Status: OK.**
- **Registered seeds.** `reeval.py:141` keeps `seed 2026` and adds `descriptive_seeds (2027,)`. `reeval.py:149` gives PRE `()`.
- **Seed rule.** `source_seeds` (`:498-502`) is used only at `:364`, where it replaces `== d["seed"]`. Every other `want` field is unchanged: run-id prefix, run label, providers, key base, variant, recipe, arm label and kind. The checkpoint-file and rule-file checks are unchanged, and `check_source_binding` and `verify_source_registry` are untouched.
- **Seed agreement.** The record seed must equal `--seed` (`source_central` `:311`) and the checkpoint meta seed (`:612`). Grep finds no other seed literal in the source checks.
- **s2026 output.** `source_seed_labels` (`:509`) and `add_source_seed_record` (`:518`) return or add nothing for s2026. The diff of `reeval.py` removes only the old `--seed` check and replaces the two `rec` assignments. `test_c_full_s2026_record_and_labels_unchanged` asserts the exact old label set and no `seed` or `seed_status` key.
- **Row labels.** The s2027 labels (`comparison_status`, `source_seed_status`, `source_seed_rule`) are not comparison-key fields. `relabel` refuses any comparison-key field, and there is a test for this.
- **Report side.** The `check_arm_identity` addition (`retention_report.py:1540-1559`) requires both labels on s2027 rows and refuses a label anywhere else.
- **Index side.** The `check_central_source` addition (`ana_block_index.py:185-191`) requires `seed_status` exactly when the record seed is in `DESCRIPTIVE_SEEDS`.
- **K15 and the other confirmatory contrasts.** `REGISTERED_SEEDS` is unchanged (`retention_report.py:466`). K20b and K20c take their denominator from `dens.get(2026)`.
  - The s2027 pairs reach only the existing DESCRIPTIVE `extra_seeds` row, the retention table and `ana05.descriptive_seeds`.
  - K15 stays single-seed 2026 confirmatory.
- **Caveat.** No real s2027 record exists locally. The tests use a record derived from the real s2026 fixture, which the handoff states (INFERENCE, PENDING the server dry run).

## Item 2: FA_WARM and FA_WARM_FROZEN binding
**Status: OK for run folders.** One fail-open path remains for warm runs that arrive without a run directory (see finding 1).
- **Index, `--fl` (`ana_block_index.py:436`, `:770-795`).** `warm_selection_of` requires identical `warm_selection` in every `LAUNCH_*.json` and runs `RR.warm_selection_problem`. That function checks format `_2`, the arm's stage, not a dry run, the file name, hex `sha256` and `self_sha256`, and that the winner factors equal the arm_spec factors (absent = 1). Both the PB and CB entries carry the binding.
- **Directory naming.** Renaming a directory does not get around it. A directory not named `FA_WARM*` that carries `init_from` is refused by `check_warm_init` in both directions. `FA_WARM_X2`-style names are excluded, not admitted.
- **Report (`retention_report.py:1811-1825`).** `load_marks` applies the same check to every entry that has a run directory. `check_ana05` adds four rules:
  - at most one warm-selection sha per stage;
  - the CB entry equals the binding of its run's PB entry;
  - if the file is in records, `sha256`, `self_sha256`, winner factors and θ_pre must all match the binding;
  - `warm_selection_record` verifies the file with the `launch_fl` canon rule.
- **Field names.** They agree with `launch_fl.bind_warm_selection` (`:2222-2225`) and the 2-D file writer (`winner{warm_lr_factor, warm_server_lr_factor}`).
- **Preview and CALIB screens.**
  - The index `check_warm_preview` (`:770-780`) refuses a `WARM_PREVIEW` parent directory, and any warm run whose LAUNCH `fl05` role or arm_spec role is not MAIN.
  - The report refuses a non-MAIN arm_spec role (`:1811`).
  - `reeval.py:573` already refuses a non-MAIN `fl05` role.
  - I found no route from a preview or S7/S8 CALIB screen into an entry through `--fl`.
  - The new tests cover these cases (`test_warm_preview_run_refused`, `test_index_refuses_a_warm_preview_directory`, `test_warm_selection_refusals`).

## Item 3: FT_FA_WARM pairing and harmed share
**Status: OK.**
- **Index `--dev` (`:468-480`).** It requires a directory named for the same arm and a LAUNCH start that `ft_warm_base_problem` accepts: kind = the base arm, `base_kind fa_warm`, `best.pt` or `endpoint_6.0.pt`, hex shas, a non-θ_pre run id, `init_from` = θ_pre, and frozen tables exactly for FROZEN. This check runs before the FT_PRE θ_pre rule. The fields mirror `launch_device.load_warm_base` (`:998-1005`) and `refuse_theta_pre_base` (`:1018`).
- **Report `check_ana05`.** The base must be the block's base-arm MAIN entry of the same seed. Its run id and `arm_spec_sha256` must match the `start_base`, its `init_from` must equal θ_pre, and PRE's `pre_init` must agree. An entry with no base entry is refused. Rows must carry `base_source`, `base_run_id` and `base_checkpoint_sha256` equal to the start (`:1570-1578`). The matching `launch_device.ft_warm_row_labels` exists.
- **Paired Δ.** `production_stack` pairs (FT entry, base PB entry) with the same seed, PRACTICAL_BEST and a `best.pt` base only. An endpoint base is reported as BASE_NOT_BEST and not paired. It calls the existing `delta()`, which computes a − b with `du < 0` as the harmed share. The known-answer test computes Δ independently from the rank files.
- **Confirmatory set.** Nothing new enters it. The only additions touching confirmatory machinery are the L4 privacy row and `PERSONALISATION + PRODUCTION_STACK` in the pair-consistency check, and that check is not a contrast.

## Item 4: B5 rollout quality
**Status: OK, descriptive and correctly computed.**
- `rollout_quality` (`retention_report.py` diff, about `:3470-3530`) computes a per-mark macro MRR@20 through the same Engine, the same W and the same `_Marked` cache as the curves.
- AUC is the trapezoid over the marks, `auc_mean` = AUC / (last mark − first mark) = AUC / 5.5, and the CI comes from the per-resample weighted trapezoid on the same W.
- First mark ≥ reference uses `value >= ref_val` (the point rule). The CI rule uses the paired Δ lower bound > 0, and NOT_REACHED / NO_REFERENCE are handled. The PRE reference is the s2026 θ_pre entry. LAST is the seed-free anchor.
- The known-answer test (`test_ana05.py:152-175`) recomputes AUC and first mark from the raw rank files. The `>` negative control fails as it should.
- Disclosure: the marks are VALIDATION marks on the same users as the PRACTICAL_BEST selection, so they are optimistic.

## Item 5: No existing confirmatory computation changed
- `--claims-only` is byte-identical (above).
- v4eval and the contracts file hashes are unchanged.
- The report diff is purely additive. Changed existing lines are limited to:
  - the ARMS / DEVICE_ARMS / CALIB_TUNED_ARMS / DEPLOYMENT_ARMS tuples;
  - the L4 row text, which is unchanged apart from extra arm names;
  - the `PERSONALISATION` loop;
  - `meta_auto` bookkeeping;
  - the result and report assembly keys.
- I did not independently re-run the worker's old-versus-new whole-result comparison (finding 5).

## Findings

**1. MAJOR: warm binding fails open for entries without a run directory.**
- **Where.** `retention_report.py` `check_ana05` marks these entries `PARTLY_UNBOUND` rather than refusing them. The entries are FA_WARM or FA_WARM_FROZEN MAIN entries that arrive via `--reeval` (REEVAL rows) or are otherwise rows-only. The handoff discloses this as judgement call 1.
- **Why it matters.** On a VALIDATION block, a REEVAL-sourced FA_WARM MAIN entry would enter with no 2-D warm-selection check. `reeval.py` checks only the FL run's `fl05` role, not its `warm_selection`. The 06M_B B2 requirement that MAIN warm runs be bound to the 2-D selection is then enforced only for `--fl` run folders.
- **Why it is acceptable on TEST.** The entries are re-evaluations of locked checkpoints whose VALIDATION run folders were bound.
- **Fix, choose one.**
  - Make `PARTLY_UNBOUND` a refusal (exit 2) on `--split VALIDATION`, keeping it a status on TEST; or
  - require the stage's warm-selection file in `--records` for any block that contains a FA_WARM or FA_WARM_FROZEN MAIN entry.
- **If neither fix is taken.** Ops must record in the DL that PARTLY_UNBOUND is an accepted TEST-only status, and check the VALIDATION `--check-only` output for it.

**2. MINOR: the `path` check is skipped when the path is absent.**
- `warm_selection_problem` checks the file name only `if ws.get("path") is not None`. A hand-built binding with no `path` passes, as long as its hex shas and factors fit.
- Real `launch_fl` bindings always carry a path. The binding is also tied to the file through `--records` sha equality when the file is supplied.
- Require `path` to be present.

**3. MINOR: the warm-selection file in `--records` is optional.**
- Without it the stage reads `LAUNCH_BINDING_ONLY`.
- The freeze procedure (handoff "What the ANA freeze still needs", step 3) lists the S7/S8 files in `--records`.
- Make the report refuse, or at least flag loudly, `LAUNCH_BINDING_ONLY` on the final VALIDATION block. This is the same fix family as finding 1.

**4. MINOR: `check_warm_preview` runs before the `DONE_S_CAL` exclusion in `ana_block_index.py:375-376`.**
- A `DONE_S_CAL` run with a non-MAIN LAUNCH role and `init_from` would now be refused with the preview label instead of being excluded.
- S_CAL runs are cold, so this is not expected to occur.
- Confirm on the server `--check-only` that no S_CAL directory is refused unexpectedly.

**5. MINOR: the worker's whole-result equality is a claim, not a check I reproduced.**
- The handoff says that outside the new keys, `RETENTION_RESULT` is identical to the pre-edit script's output on the default and ANA-04 full blocks.
- Code reading agrees: the diff is additive, and the changed lines touch only bookkeeping.
- Re-run `ana05work/compare_old_new.py` from the project tree when the freeze tree (r6) is built.

## Non-findings and notes
- The base-checkpoint sha is bound through `run_id` and `arm_spec_sha256` (a run-level binding), not checked against a per-checkpoint sha in the index entries. This is acceptable because `launch_device.load_warm_base` verified it at launch.
- The index accepts the `WARM_SELECTION_S7` / `_S8` records by sha only and does not parse them. This is by design.

## Files
- Root: `F:\Queens University\Microsoft Project\eda\.local_ai\ppsi_v4\`
- Reviewed: `code/v4int/reeval.py`, `tools/retention_report.py`, `tools/ana_block_index.py`, `code/tests_int/test_ana05_reeval.py`, `code/tests_ana/test_ana05.py`, `code/tests_ana/anakit.py`.
- Test logs: `.local_ai\_pytest_tmp\REV-ANA-05\{ana,int,nc_ana,nc_int}.txt`.
