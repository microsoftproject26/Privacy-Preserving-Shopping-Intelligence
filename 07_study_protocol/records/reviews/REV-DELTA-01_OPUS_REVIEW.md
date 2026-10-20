# REV-DELTA-01 — independent READ-ONLY review of the deltas before code_ds3r3 (Claude Opus 5.5, 2026-09-29)

**Scope.** This review follows `company/tasks/REV-DELTA-01.md`. It covers five deltas plus FL-09:
- INT-06, with the REV-FL-01 items folded in (there is no FL-08 handoff; INT-06 did that work);
- DATA-09;
- ANA-02;
- DATA-07R1;
- DEV-04, with the DATA-08 minors;
- FL-09 (added by the caller).

**Rules kept.**
- Only this file was written.
- No server command was run and nothing was installed.
- Pytest wrote only under `.local_ai/_pytest_tmp/REV-DELTA-01/`. That includes an extracted copy of `runs_ds3r2_deploy/code_sync.tar` in `…/ds3r2/`, used for byte comparisons.
- No RETRO_TEST row, and no RESULT_ROWS of any `runs_*` directory, was opened.

**Evidence tags.**
- MEASURED: a file:line reference, or a command I ran.
- INFERENCE: reasoning.
- OPINION: judgement.

**The tree was moving during the review (MEASURED; read this first).**
- `company/handoffs/INT-07.md` and `FL-10.md` did not exist when I started (checked at the start).
- `INT-07.md` appeared at 15:24:29 local, while my suites were running. `FL-10.md` still does not exist.
- The FL-10 worker edited these files during the review:
  - the locked package `fl/`: `server.py` at 15:13:17, `runtime.py` at 15:16:14, and `client.py`, which changed during my run;
  - `v4int/launch_fl.py` at 15:18 and `v4int/lr_screen.py` at 15:19;
  - new tests `tests_fl/test_fl10_fedadam.py` and `tests_int/test_fl10_fedadam_launch.py`.
- INT-07's edits were already on disk when I started: `v4int/{reeval,launch_device,providers,cache,retro_store,dump_rows_retro}.py` and `v4base/run.py`.
- Other workers' pytest and run_all processes ran concurrently. I saw them in the process list: `tests_int/run_all.py`, `tests_hist/run_all.py` and two tests_int pytest runs.
- What this review covers:
  - INT-06 and FL-09 as they sit in the current files, INT-07 and FL-10 edits included;
  - INT-07 and FL-10 themselves are **not** reviewed. They need their own delta review (REV-DELTA-02) before code_ds3r3 is packed.

**File sha256s at review time (MEASURED):**

| file | sha256 |
|---|---|
| `tools/retention_report.py` | `1e93a819…` (= ANA-02 handoff) |
| `tools/ana_block_index.py` | `f6bb6f48…` (= handoff) |
| `v4int/launch_trunc.py` | `1f6304df…` (= DATA-07R1) |
| v4trunc code | `4d766d5b…` (= DATA-07R1) |
| `v4int/test_roles.py` | `b02391e2…` (= DATA-09) |
| `v4int/reeval.py` | `33a8a9ce…` (= INT-07 handoff) |
| `v4int/launch_device.py` | `6dd4a4e7…` (= INT-07) |
| `v4int/launch_fl.py` | `536bfe55…` at 15:1x, then rewritten by FL-10. **Not** FL-09's `ca93ceea…` |
| `v4int/lr_screen.py` | `e1441456…`, then rewritten by FL-10. **Not** FL-09's `651d1c84…` |
| `fl/server.py` | `93d4c863…` (ds3r2: `63ccf395…`) |

---------------------------------------------------------------------------------------------------------------------
## Binding facts (a) and (b)

**(a) v4eval is FROZEN. PASS (MEASURED).**
- `evaluator_code_sha256()` on the local tree is `61b64e61c1a2e4f5a6fecd42f0ad3b98de26562d13a07758573cc9496b8d962e`.
- The same function on the extracted ds3r2 tar gives the same value.
- Every `code/v4eval/*.py` compares byte-identical to ds3r2 (`cmp`), and there is no new .py file under v4eval.
- The guard test `tests_eval/test_device_pin.py:99` `test_evaluator_identity_is_the_deployed_a40_block_identity` PASSED in my tests_eval run (80 passed).
- **No v4eval change, so no BLOCKER.**

**(b) Locked packages (`tests_int/run_all.py` IMPORTED: v4train, v4data, v4eval, v4models, fl, v4sup, ppsi_env.py).**
Compared with the deployed code_ds3r2 SYNC_MANIFEST (MEASURED):

| package | result |
|---|---|
| v4train, v4data, v4data06, v4eval, v4models, v4sup, snap_v1, ppsi_env.py | byte-identical, no new files |
| **fl/** | **NOT identical:** `fl/server.py`, `fl/runtime.py`, `fl/client.py` |

- The fl changes are FL-10 edits in progress. FL-10's packet authorises them, but nothing has reviewed them yet.
- Before/after my own run (`run_all.imported_hashes()` snapshots): 102 files each side. The only differences were those same three fl files, edited by FL-10 while my suites ran.
- None of the five reviewed deltas nor FL-09 touched a locked package (INT-06: `fl/accountant.py` untouched, as its handoff says).
- Default-path check of the FL-10 diff (INFERENCE from the diff, plus one MEASURED digest):
  - `Server.apply` changes only when `server_opt is not None` (server.py diff at old line 113).
  - The `RunConfig.digest()` drop rule now includes `("server_opt", None)` (runtime.py:151).
  - `sgd_m` is used only when requested (client.py).
  - MEASURED: the default FA RunConfig digest is `69cab77f…` under both the ds3r2 tree and the current tree.
- → **MAJOR (process) F3**, see the findings.

---------------------------------------------------------------------------------------------------------------------
## 1. Checklist: every FIX item of REV-FL-01 and REV-ANA-01, plus the packet items

Status values: FIXED, PARTIAL, MISSING, ACCEPTED (not a code item), OPS (Ops action).

| item | status | evidence (MEASURED file:line / test) |
|---|---|---|
| **REV-FL-01 B1** PF resume uses a stale store | FIXED | `launch_fl.execute` `eval_fn` uses `run.store` (`evaluate_pf(mark, run.store)`); the end save uses `save_pf_store(..., run)`. Test `tests_int/test_int06_fl_device.py:61`: per-mark store digests of straight vs interrupt(3)+resume are equal and non-empty; PF_STORE.pt digest = the 6.0 mark digest. NC `:110` (stale store) fails as intended |
| **B2** best-mark PF store not persisted | FIXED | `save_pf_store` → `PF_STORE_best.pt` after every step whose `best_half_efe` changed. `pf_best_record` refuses a stale best store unless the current mark is the best. PB rows carry `pf_store_file/sha256/digest`; `checkpoint_id …|pf_store=<digest>`. Tests `:61` (theta_digest = best.pt θ; best store equal straight vs resumed) and `:99`. `reeval.check_pf_store` (reeval.py:237) pairs a store with θ by theta_digest; a legacy store pairs with endpoint_6.0 only |
| **M1** device rows REHEARSAL on pinned A40 | FIXED | `launch_device.assemble(..., device_pin_sha256)` (L480-485); `finalize` gets the pin via `pin_for_device` (L525-527); RUN_SUMMARY carries it (L555). Tests `test_int06_fl_device.py:183`, `:200` |
| **M2** FL resume device/pin/code continuity | FIXED | `execute` compares ARM_SPEC `device` (`device_record`) and `code_sha256` (`fl_code_sha256` = fl/*.py + launcher), and refuses a legacy spec. Tests `:116` and NC `:140`; `tests_train/test_int06_device_pin.py:95` (central resume under another pin) |
| **M3** device shards mixed classes | FIXED | `shard_device_record` / `check_shard_device` (L322-343), checked on resume (L392-398) and in `finalize` (L528-530). Test `:208` |
| **D13 / 06E §1** LO cell rule | FIXED (PARTIAL: see F7) | `--arm LO --role UNSEEN_VAL --select` writes SELECTION.json; the unselected LO cells get `LO_CELL_NOT_SELECTED` / DESCRIPTIVE (`device_rows` L611-613). Tests `:231`, `:254`, NC `:269`. An LO MAIN run **without** `--selection` is still allowed (`check_role` L148 applies to non-LO arms only). The index refuses it later (`ana_block_index.py:312-316`) |
| MINOR `load_lr_selection` bound to the registered file + `lr_peak_base` | **MISSING** | `load_lr_selection` checks format / self-sha / rule only; `resolve_lr_mult` records the path, with no path or sha binding and no `lr_peak_base` check. FL-09 widened this: see F2 |
| MINOR `load_selection` re-verifies the UNSEEN_VAL manifest sha / 12,792 users | **MISSING** | `launch_device.load_selection` (L577-586) checks format, self-sha and role only (F6) |
| D1 accountant disclosure | ACCEPTED | `fl/accountant.py` is byte-identical (sha pinned by test_fl06_wor); the text is superseded by 06E §9(a) |
| D8 / D9 docstrings (dp.py, participation.py) | ACCEPTED (not edited) | fl is locked; editing it would change fl_code_sha and resume identity. Register as text-only in 06E §9 (OPINION) |
| FL A40 determinism: 2 identical FA smokes | OPS | not a code item; no evidence in the handoffs |
| INT-06 P1: device_pin.py guard reverted | FIXED | v4eval = ds3r2 bytes (binding fact a) |
| INT-06 pin refusals (absent / wrong name / tampered self-sha) | FIXED | tests_eval `test_device_pin.py:21,26,37`, NCs `:155,:160`; tests_train `test_int06_device_pin.py:48,55`, NCs `:110,:115` |
| pinned rows MEASURED_PINNED_DEVICE | FIXED | `test_device_pin.py:126`; reeval `check_pinned_rows` (reeval.py:350) refuses any other status / pin / class |
| reeval reproduces the harness rows (central / FL θ / PF θ+store) | FIXED (CPU, components) | `test_int06_reeval.py:229,251,280` on the INT-01 real-row CPU environment: equality of value, rows, ranks and result; NCs `:303,:311`. The reference is `launch_fl.role_evaluator`, proven equal to predict_adapter→evaluate in `test_fl05_real_calib.py:49`. So the proof is transitive, and `run()` past `--dry-run` is not executed (CUDA). See F10 |
| reeval roles / `--registry-ext` / `test_gate_ok` | FIXED | `resolve_role` (L433-446), `install_ext` + `verify_generic_ext` (L377-431), `test_gate_ok` → `v4int.test_roles.test_gate_ok` (L107-113, wired by INT-07). Tests `:323,:334,:347,:68`, NC `:393` |
| **reeval output ↔ ana_block_index interface** | **BROKEN** | see F1 |
| **REV-ANA-01 A-F1** trunc fields / C_ref / dup keys / FULL value | FIXED | `trunc_curves` (retention_report.py:1946…); tests `test_ana02.py:417,436`, NC `:422` (fixture built with `launch_trunc.trunc_labels`) |
| A-F2 ONNX_CPU sub-block, K16 in-block, DEVICE_MIXED fallback | FIXED | `ONNX_ARMS` (L141); tests `:366,:389`, NC `:403`; reads the DEV-04 `retention[tag]` dict |
| A-F3 RULE_FREE only for baselines; `Book.pairs` → NOT_AVAILABLE | FIXED | `BASELINE_ARMS` (L139); tests `:218,:233,:249`, NC `:254` |
| A-F4 REGISTERED_SEEDS / INCOMPLETE_SEEDS | FIXED | L209; test `:195`, NC `:209` |
| A-F5 full key, intended differences, MEASURED_PINNED_DEVICE + one pin | FIXED | `KEY_FIELDS` / `INTENDED_DIFFERENCES` (L117-120); tests `:272,:283`, NC `:294` |
| A-F6 rank coverage; users_subset = DEV-04 set | FIXED | `check_rank_coverage` (L914); tests `:314,:319,:329`, NC `:339` |
| A-F7 arm identity | FIXED | `check_arm_identity` (L788-798), `TRAIN_MANIFEST_OF` (L146); test `:346`, NC `:359` |
| A-F8 TEST gate: lock + register chain, contrasts = lock, ext rule, role map, same-script VALIDATION_RESULT, hard refusal | FIXED | `check_test_gate` (L479-514), `verify_test_ext_registry` (L525-553), `test_context_loader` (L556…, refusal exit 2), contrasts sha == lock (L2637), VALIDATION_RESULT checks (L2109-2129); tests `:457,:466,:504,:521`, NC `:529` |
| A-F9 rule map / hierarchy / 06D texts / multiplicity / still_improving | FIXED | L201-209, L939, L1584-1611; tests `:71,:95,:122,:133,:157`, NCs `:106,:170` |
| A-F10 rule vs efe / checkpoint_id | FIXED (real-id risk, F9) | `check_rule_vs_checkpoint` (L801-823); test `:544` |
| A-F11 `--check-only` runs plan_pairs | FIXED | `plan_pairs` (L2217); test `:564` |
| A-F12 sanity trigger CB vs PB | FIXED | `sanity_trigger` (L2155) |
| A-F13 evidence integrity | FIXED | `REFUSED_ROW_STATUS` (L148); test `:572` |
| A-F14 g2_run sha + lock-registry pin | FIXED | `G2_RUN_SHA256` (L151) = sha256 of `tools/g2_run.py` (MEASURED `3893a49a…`); test `:587` |
| A-F15 n_decisions; W sha | FIXED | test `:594` |
| A-F16 strata source manifest | FIXED | test `:608` |
| A-F17 historical seed mean | FIXED | tests `:618,:629` |
| **B-F1** smoke T41 receipt; n_compared + panel binding | FIXED | `check_t41_receipt` (launch_trunc.py:214-260: smoke refusal L239-242, panel binding L249-254) |
| B-F2 panel re-derivation | FIXED | `panel.check_panel_rederived`, called from `load_panel_world`; NC `test_nc_later_query_panel_rederived` |
| B-F3 WARM row_status (+ Study B) | FIXED | tests_trunc (Study A + B label tests) |
| B-F4 COLUMN_RULES import-time assert | FIXED | `builder.check_rule_coverage`; NC `test_nc_rule_coverage_unsupported_future` |
| B-F5 weak NCs | FIXED | `test_t41_golden.py:75` precondition + `:81` NC |
| B-F6 run_all basetemp / TMP | FIXED | `test_ana02.py:635`; tests_trunc/run_all.py diff (2 lines) |
| B-F7 E2X C_ref = C_BIG Study A; OOV counts | FIXED | `events_to_x_table` / `run_e2x` (DATA-07R1 handoff; tests_trunc) |
| T41 bucket 0 with the 7 admitted cells | PASS (REPORTED file, verified) | `_pytest_tmp/DATA-07R1/T41_BUCKET0_REPLAY.json` sha `34413aff…` (= handoff), code `4d766d5b…` (= current v4trunc): VALIDATION 173,921/173,921 PASS, 7 admitted, 48 brand-OOV flagged; TRAIN 622,168 PASS, 0 admitted. I did not re-run the 400 s replay |
| **C-F1** ext band / manifest population | FIXED | `d8_registry.expected_population()` (L79-…); tests_data08 22 passed |
| C-F2 c_full launch args | FIXED | `c_full.check_launch_args` (c_full.py:84-92); every launch_central default of those flags is None/False (launch_central.py:105-136), so no false refusal |
| **D-F1** OOV in INT8 ranks | FIXED | `bench.ranks_from_session` skips `target_class < 0` and counts `n_oov_skipped` (bench.py:170-190); tests_dev |
| D-F2 `--first-users` in selection_key_v2 order | FIXED | `dump_rows.first_users_selection` (L71-100, `census_v2.core.selection_key_v2`); tests_dev `test_dump_rows.py` (v2 order, and the NC that it differs from euk<N) |
| D-F3 SRI / threads / provenance | FIXED (not run in a browser) | DEV-04 handoff; `test_browser_page.py`; `node --check` only |
| **DATA-09** gate / read / build / registry / shim | see §4 | |
| **FL-09** multipliers / EXT select / MAIN refusal | FIXED, with F2 | see §9 |

---------------------------------------------------------------------------------------------------------------------
## 2. Regressions of locked contract behaviour — none found (MEASURED)

- **A16 path digests.**
  - tests_train `test_int06_device_pin.py:84` (the A16 path records no pin field) passes.
  - v4train is byte-identical to ds3r2.
- **FA / FP / FA_Q8 arm-spec digests.**
  - FL-09 pins 96 (arm-spec, RunConfig) pairs: `test_fl09_lr_ext.py::test_old_grid_digests_byte_identical_to_pre_fl09_code`. The FL-05 26-pair BASELINE and the FL-07 tests pass (tests_int 194 passed).
  - My own check: the default FA RunConfig digest is `69cab77f…` in ds3r2 and in the current tree.
  - The one tests_fl failure (`test_fl03_dropout.py:207`) is FL-10's stale test reference, not a digest change. `_old_digest` pops 3 keys, and RunConfig now has a 4th default field, `server_opt`. The real `digest()` drops it (runtime.py:151). → F4.
- **T3 frozen config.** `tests_eval/test_frozen_config_t3.py` passes, and v4eval is identical.
- **v4data / v4data06 byte-identity.**
  - Both are identical to ds3r2 (binding fact b).
  - DATA-09's no-side-effect proof: `tests_data09/test_d9_no_side_effect.py`, fresh interpreters, 8 DATA-04 plus 15 DATA-06 manifests identical with and without v4data09 imported. It passes.
- **G2 runner.** `tools/g2_run.py` sha = `3893a49a…` = the ANA embedded freeze constant; mtime 09-28, unchanged.

---------------------------------------------------------------------------------------------------------------------
## 3. INT-06

**Bitwise reproduction (MEASURED, CPU):**
- Central `model`, FL `theta` and PF theta + store each reproduce value, rows, ranks and result exactly (`test_int06_reeval.py:229-300`), with the NCs failing as intended.
- The equivalence is proven through `role_evaluator`, and `role_evaluator` equals predict_adapter → evaluate on real CALIB rows (`test_fl05_real_calib.py:49`).
- INFERENCE: the deployed ds3r2 reeval also called predict_adapter → evaluate directly (ds3r2 reeval.py:145-150). So old and new reeval give the same numbers for the same checkpoint.
- Gap (F10): `run()` past `--dry-run` has no test. It needs CUDA.

**Roles.** MAIN, UNSEEN_HOLDOUT, SPARSE/HEAVY_PANEL, TRUE_COLD and UNSEEN_VAL use the launch_device key rule (`role_key_base` L327-334). Base roles cannot be redefined by an extension (`resolve_role` L437-438).

**`--registry-ext`.** Additions-only (`verify_generic_ext`), with a c_full-pattern wrapper. TEST names are gated.

**`test_gate_ok`.** Wired (INT-07). It is checked before any read in `check_args` (L146).

**Pin.** `pin_for_device` raises on a wrong GPU name, torch or CUDA version; a tampered self-sha is refused by `load_pin` (tests above). reeval refuses unless live class == pin class == `NVIDIA_A40-FP32-DET` (reeval.py:500-503).

**M1, M2, M3.** Fixed; see §1.

**B1 / B2 tests.** They show equal per-mark store digests, equal PF_STORE.pt digests and a persisted, θ-paired best-mark store (§1).

**Interface defect F1 (MAJOR, MEASURED).**
- `ana_block_index.reeval()` requires top-level `REEVAL_RECORD.result_rows_sha256` and `ranks_sha256` (ana_block_index.py:230-233).
- The rewritten reeval writes these hashes only inside `outputs{…}` (reeval.py:370-374, `write_outputs`). The test `test_int06_reeval.py:408` asserts `outputs`.
- The ANA fixture `tests_ana/layoutkit.py:53-59` hand-writes the old ds3r2 field names. That is why ANA passes and the defect stays hidden.
- Consequence: every re-evaluation produced by the new tree will be refused with `IndexRefused` (fail closed). That covers the panel roles (§6.7), the FL/PF checkpoints and the whole TEST look.
- The 8 ds3r2 records do carry both keys (ds3r2 reeval.py:180-185), so they are ingestible.

---------------------------------------------------------------------------------------------------------------------
## 4. DATA-09

**Gate.** `d9_gate.check_test_gate` (d9_gate.py:106-139) refuses unless all of these hold:
- the lock exists;
- the register exists and its chain verifies through the Ops tool's `verify`, loaded by path;
- an entry has `access_purpose MANIFEST_BUILD` and `split RETRO_TEST` with `all_methods_lock_sha256 == sha256(lock)`.

The first qualifying entry binds the gate.

**`revalidate`** (L142-149) re-runs the check on the record's own paths and requires the same (lock sha, entry id, entry sha).

**`read_with_test`** (d9_read.py:25-33) calls `revalidate` first.
- A forged `GateRecord` is useful only while the real gate is open.
- A non-GateRecord refuses.
- There is no module-level default.

**Only TEST-scanning function (MEASURED grep of v4data09, v4int, v4base, v4dev, v4hist).** Every split-filtered ledger or feature scan that includes TEST goes through `read_with_test`:
- d9_build.py:157, 220;
- INT-07's `retro_store.py:170`.

The other `pq.read_table` / `ParquetFile` reads open built manifest tables inside gated functions (d9_finalize, test_roles:259) or non-TEST files (v4base.identity). INT-07 is not reviewed.

**Entry points gated first.** `d9_build.main` / `D9Builder` / `run_build`; `d9_finalize.main` (L245, first); `test_roles.resolve_role` / `test_gate_ok` / `verify_test_manifest` / `test_eval_manifest` (L116, 142, 165, 305); the job scripts (rc 8). NCs: tests_data09 gate refusals, 28 NCs, all AssertionError.

**M07 / M08 / grace / censoring.**
- Grace targets:
  - `grace_matured` iff status = OBSERVED_MATURED_IN_GRACE;
  - target_ts in [TEST_END, GRACE_END);
  - no eval target at or after the grace end;
  - metadata NOT_APPLICABLE.
  - Build-time asserts: d9_build.py:330-362.
- Censor precedence: LEGACY > RIGHT_EDGE (BU, next ∈ {TEST, LABEL_GRACE}) > AMBIGUOUS > NO_NEXT. `eval_mask` = {IN_PARTITION, MATURED}. An unmatched row stops the build (L126-141, d9_common.py:95-100). This is EVALUATION_CONTRACT §2.2 with VALIDATION_END replaced by the end of LABEL_GRACE.
- M08: the literal rule, with a hard count check against the full-population counter (a judgement call, documented).
- All of this is tested against an independent synthetic reference (`test_d9_build.py`).

**VALIDATION manifests byte-identical with or without v4data09.** PASS (§2).

**Registry extension.** Byte-identical to REGISTRY_V2 entries (`verify_test_ext_registry`); O_EXCL writing. Tested in `test_d9_registry_shim.py`.

**No real RETRO_TEST row read.**
- The 5 real tests skip unless the build receipt exists and the real gate is open (`test_d9_real.py:1-5`). The gate is closed locally: no `company/gates/ALL_METHODS_LOCK.json` (MEASURED).
- The INT-07 kit reads real **TRAIN** rows only (INT-07 handoff; `test_d9_int07_gate.py` uses an absent root).

**OPINION MINOR (F12).** The register chain has no external anchor. `verify` re-derives hashes, so anyone who can edit the file can re-chain it. This is a process-integrity gate, not a security boundary. Disclose it. Optionally record the register head sha in each DL row.

---------------------------------------------------------------------------------------------------------------------
## 5. ANA-02

**A-F1 … A-F9 against the review's own acceptance criteria.** Every one is FIXED (see §1): the ONNX sub-block, REGISTERED_SEEDS, the full key plus pinned status, rank coverage, arm identity, the hard TEST-gate refusal, and the rule map / hierarchy / flag. Each MAJOR has a positive test and an NC. The 18 NCs all fail with AssertionError only.

**`ana_block_index.py` never reads a metric value.** OK:
- The builder parses only the allowed JSON names (`ALLOWED_JSON` L70).
- `test_index_builder_never_parses_a_rows_file` (sentinel non-JSON RESULT_ROWS) passes, and `test_nc_index_builder_reads_rows` fails as intended.

**Determinism.** OK: `test_index_builder_deterministic`, and the old `test_deterministic_rerun_byte_identical`.

**`--check-only`.** OK: value-free (`test_check_only_runs_pairing_and_the_block_denominators`).

**Open items:**
- F1 (the index versus the new reeval record).
- F9 (the real C_BIG PB checkpoint-id form).
- F11: `company/gates/ANA_CONTRASTS_v2.json`, written 15:12 local, registered 12:12Z, carries K1–K19 (06I §1) and says it supersedes ANA_CONTRASTS.json. Script `1e93a819…` accepts exactly K1–K16, so it cannot run on the registered contrast file. ANA-03 is already planned for this. Do not freeze ANA before ANA-03.

---------------------------------------------------------------------------------------------------------------------
## 6. DATA-07R1 and DEV-04

- B-F1 … B-F7, C-F1, C-F2 and D-F1 … D-F3 are verified in §1. B-F6 belongs to ANA-02 and is done.
- T41 bucket 0 still PASSES with the same 7 admitted cells (receipt verified, same v4trunc code sha).
- `--first-users` is in selection_key_v2 order, proven by tests_dev `test_dump_rows.py`:
  - the order test;
  - the NC that the order differs from euk<N;
  - the count equals N when N are available.
- ANA side: `test_users_subset_v2_is_the_dev04_set`.
- The v4trunc code sha has changed (`4d766d5b…`), so every server T41 receipt must be made with this tree. None existed before (DATA-07R1 handoff).

---------------------------------------------------------------------------------------------------------------------
## 7. Test runs (MEASURED)

**Setup.**
- venv python 3.13, torch CPU, `-B -p no:cacheprovider`, per-suite `--basetemp` and TMP/TEMP under `_pytest_tmp/REV-DELTA-01`.
- Receipts: `…/REV-DELTA-01/out/*.xml|txt`. NC column = the `--runxfail -k test_nc_` run.
- tests_data08 = the synthetic + shim files only.

| suite | main run | NC run (--runxfail) | NC exception types |
|---|---|---|---|
| tests_eval | 80 passed, 28 xfailed, exit 0 | 28/28 failed | AssertionError only |
| tests_train | 206 passed, 52 xfailed, **13 errors + 1 failure**, all in `test_train05.py` (environment, see F5); isolated re-run of test_train05.py with a separate TMP: **13 passed, 9 xfailed** | 60/60 failed | AssertionError only |
| tests_int | 194 passed, 51 xfailed, exit 0 | 55/55 failed (4 new FL-10 NCs had landed by then) | AssertionError only |
| tests_fl | 231 passed, 70 xfailed, **1 failure** (`test_fl03_dropout.py:207`, FL-10 stale reference, F4) | 70/70 failed | AssertionError only |
| tests_ana | 75 passed, 18 xfailed, exit 0 | 18/18 failed | AssertionError only |
| tests_trunc | 42 passed, 11 xfailed, exit 0 (bucket-0 data present) | 11/11 failed | AssertionError only |
| tests_data08 (synthetic + shim) | 22 passed, 14 xfailed, exit 0 | 14/14 failed | AssertionError only |
| tests_data09 | 73 passed, 28 xfailed, 5 skipped (real-only), exit 0 | 28 failed, 1 skipped (real-only NC) | AssertionError only |
| tests_dev | 73 passed, 5 skipped (onnxruntime absent), 13 xfailed, exit 0 | 13/13 failed, 2 skipped | AssertionError only |
| tests_base | 64 passed, 10 xfailed, exit 0 | 10/10 failed | AssertionError only |

**Snapshot and locked packages.**
- v4train, v4data, v4eval, v4models, v4sup and ppsi_env.py are unchanged, before versus after the run and versus ds3r2.
- fl/ was changed by FL-10 during the run (F3).
- No suite wrote into a locked tree.

---------------------------------------------------------------------------------------------------------------------
## 8. Are the code_ds3r2 server artefacts invalidated by any delta? (INFERENCE, with the exact reasons)

**The 8 A40 re-evaluations (C_BIG / C_SMALL × PB / CB × 2 seeds): VALID, no re-run needed.**
- Why:
  - The evaluator is byte-identical (61b64e61…).
  - v4train, v4models, v4data and v4sup are identical.
  - The ds3r2 reeval scored through predict_adapter → evaluate with the pin, so rows are MEASURED_PINNED_DEVICE.
  - Their REEVAL_RECORDs carry the `result_rows_sha256` / `ranks_sha256` / `eval_manifest_id` / `device_class` / `device_pin_sha256` fields that `ana_block_index` reads.
- Two caveats, neither a numerics problem:
  - (i) F9: their C_BIG PB `checkpoint_id` is `02B_SASREC_MAIN_s<S>:best` (`phase1_after_strict_r2.sh:40`). ANA A-F10 accepts C_BIG PB only for `…best@3.0` or |efe − 3.0| < 1e-9 (retention_report.py:815-817). v4train marks are exposure-exact (`schedule.mark_exposures` = ⌈3N⌉ = 3N, schedule.py:191-194), so efe is very likely exactly 3.0. This is not proven on the real rows; confirm with the VALIDATION `--check-only` before anything else.
  - (ii) The identity-label gaps INT-06 already noted (LOW).

**LR-screen runs x0.5 / x1 / x2: VALID.**
- `lr_screen.read_run` reads the same RUN_SUMMARY / RESULT_ROWS / ARM_SPEC fields.
- The FA default digests are unchanged, so `_strip_lr` spec equality with x4 / x8 / x16 made by a newer tree holds.
- Selection is not a resume, so the differing fl_code_sha is irrelevant.

**Smokes.** Not evidence; unaffected.

**S_CAL / FA / FP / FA_Q8 at m = 2.** Superseded by **registration** (06F §2), not by code. FA_1024 and DP: the same by 06F §1 (the extended m applies to them, and DP needs the re-run S_CAL). This is the registration's consequence, INFERENCE.
- Code consequence for every ds3r2 FL run: it can never resume under ds3r3. The legacy ARM_SPEC has no device/code record, and fl_code_sha differs (M2 / 06E §8). Finish it with code_ds3r2 or discard it.

**PF run from ds3r2 (if any).** B2: its PRACTICAL_BEST cannot be rebuilt for TEST, so re-run it from ds3r3 (as INT-06 says).

**Device-arm runs (LO / FT_C / FT_FA / SEL_*) started from ds3r2.**
- Their rows carry REHEARSAL (M1).
- Their shards lack the class/pin/code record, so a ds3r3 resume or finalize refuses them (M3).
- → Re-run from ds3r3. SELECTION files made by ds3r2 `--select` runs are valid as content (value-based rule unchanged), but their run dirs cannot be re-finalized.

**C_FULL.** Valid.
- launch_central and v4train are unchanged.
- The C-F2 refusals do not match its launch line (no `--allow-shared-gpu`; setup_lanes_v1.sh:75).
- Resume it with the tree that started it (OPINION: the safest rule).

**KNN_EMB baselines.** Valid as ds3r2 outputs (INFERENCE). `v4base/run.py` changed only through INT-07, which is not reviewed.

**T41 / trunc.** There are no ds3r2 receipts. Receipts made under ds3r2 would be refused (v4trunc sha changed).

**Must Ops re-run from code_ds3r3?**
- Not for the re-evaluations, the first LR screen or C_FULL.
- Yes for every FL MAIN arm (registration-driven), PF, and the device arms.

---------------------------------------------------------------------------------------------------------------------
## 9. FL-09 (extended LR multipliers, lr_screen EXT selection)

**What holds (MEASURED):**
- `LR_MULTS` = (0.5, 1, 2, 4, 8, 16, 32) with matching tags; `lr_mult_value` refuses anything else (NC).
- The arm-spec shape is unchanged; old-grid digests are byte-identical (test).
- `apply_rule_ext` (lr_screen.py:77-93): best; ties ≤ 0.0005 → smallest |log2 m|, then smaller m. Non-finite values are excluded and recorded; ≥ 3 runs, each m at most once; per-run verification as in FL-05; one manifest; one `lr_peak_base`.
- `--first-screen` must verify, and its runs must match by RESULT_ROWS sha and value.
- The output is O_EXCL, and only m + sha are printed.
- `load_lr_selection` re-applies the rule and checks screened = values + excluded.
- A MAIN run with a 06B file is refused once `LR_SELECTION_EXT.json` exists. Tests: `test_fl09_lr_ext.py`, 7 tests + 4 NCs.

**F2 (MAJOR, MEASURED).** The MAIN launch does not bind the selection to the registered file or rule:
- (a) Any self-consistent EXT file at any path is accepted. The handoff's own step 2 writes such a file (`<R>/LR_SELECTION_EXT_PREVIEW.json`), and it would pass as `--lr-selection` for a MAIN run even after the final file exists.
- (b) Neither the file nor the launcher requires the 06F screened set (⊇ {0.5, 1, 2, 4, 8, 16}, plus 32 iff 16 is best), `first_screen_file_sha256 != null`, or `lr_peak_base == SMALL peak 0.000711`. The REV-FL-01 MINOR on this is still open.
- (c) Before the EXT file exists, a MAIN run still accepts the 06B file, which contradicts 06F §1 ("every MAIN FL run uses the extended selection").
- FL-10's in-progress `--fl-tuning` repeats pattern (a): `load_fl_tuning` checks format and self-sha, with no path binding.

**Judgement calls.**
- "best" = the raw best for the 32 edge rule: agree (OPINION), and register it in the DL.
- A crashed run is refused rather than excluded: agree. Record it in the DL.

---------------------------------------------------------------------------------------------------------------------
## Findings table

| finding | severity | file:line | fix | who |
|---|---|---|---|---|
| F1 new REEVAL_RECORD lacks top-level `result_rows_sha256` / `ranks_sha256`, so `ana_block_index` refuses every ds3r3 re-evaluation (panels, FL/PF, TEST look); layoutkit fakes the old format | MAJOR (fails closed) | reeval.py:357-375; ana_block_index.py:230-233; tests_ana/layoutkit.py:53-59 | reeval writes both keys (keep `outputs`), or the index reads `outputs`; add one test that feeds a real `write_outputs` record to the builder | Sonnet (INT follow-up; before any ds3r3 reeval) |
| F2 MAIN FL launch accepts any self-consistent LR selection (EXT preview incl.); no lr_peak_base / screened-set / first-screen checks; 06B accepted before EXT exists; same for `--fl-tuning` | MAJOR | launch_fl.py `load_lr_selection`, `require_lr_selection`, `resolve_lr_mult`, `load_fl_tuning` | non-smoke MAIN: resolved path == `contracts/fl/LR_SELECTION_EXT.json` (or FL_TUNING_PATH) and its sha; require first_screen sha, screened ⊇ 06F set, the 32 rule, lr_peak_base == recipe peak; refuse the 06B format for MAIN outright | Opus/Sonnet (fold into FL-10) |
| F3 tree not quiescent; locked `fl/` differs from ds3r2 (server.py, runtime.py, client.py: FL-10, unreviewed); INT-07 handoff arrived mid-review | MAJOR (process) | fl/{server,runtime,client}.py | pack code_ds3r3 only after FL-10 + INT-07 handoffs and REV-DELTA-02; DL row re-baselining fl's identity; run every suite on the frozen snapshot | Ops |
| F4 tests_fl failure: `_old_digest` does not drop `server_opt` | MINOR | tests_fl/test_fl03_dropout.py:199-209 | pop `server_opt` in the reference (the real digest is unchanged, 69cab77f… both trees) | FL-10 worker |
| F5 test isolation: `p2kit.P2_ROOT` is a shared dir in the code tree and `cleanup()` rmtree's it, so concurrent runs break test_train05 (13 E + 1 F; isolated 13 passed / 9 xfailed) | MINOR | tests_train/p2kit.py:24,37-38; test_train05.py:36 | per-session subdir (keyed by basetemp) and clean only it; never run suites concurrently for receipts | Sonnet |
| F6 `load_selection` does not re-verify `eval_manifest_content_sha256` vs the registry UNSEEN_VAL manifest, nor the 12,792 users (REV-FL-01 MINOR open) | MINOR | launch_device.py:577-586 | compare with `rep.manifest_binding` of the UNSEEN_VAL eval manifest | Sonnet |
| F7 LO MAIN / panel run allowed without `--selection` (FL-08 asked REQUIRE); refused only later by the index | MINOR (compute waste) | launch_device.py:148 | refuse LO without `--select`/`--selection` on non-UNSEEN_VAL roles | Sonnet |
| F8 dp.py / participation.py docstrings not corrected | MINOR (ACCEPT) | fl/dp.py:5,27-28; fl/participation.py:9-12 | leave (locked); text in 06E §9 / thesis | Ops |
| F9 ANA A-F10 untested on the deployed C_BIG PB id form `…:best`; passes only if efe == 3.0 exactly | MINOR (risk) | retention_report.py:815-817; phase1_after_strict_r2.sh:40; tests_ana/anakit.py:114-119 | server VALIDATION `--check-only` first; optionally accept REEVAL `checkpoint_sha256` == ARCHITECTURE_LOCK c_big best.pt sha; future runs use `--checkpoint-id …:best@3.0` | ANA-03 / Ops |
| F10 reeval `run()` past `--dry-run` untested (CUDA); bitwise proof is transitive through role_evaluator | MINOR | reeval.py:464-667 | on the server: re-run ONE ds3r2 re-evaluation (C_BIG s2026 PB) with ds3r3 into a new dir and compare the ranks sha with the ds3r2 output | Ops |
| F11 ANA_CONTRASTS_v2 (K1–K19) supersedes the file ANA-02 accepts; script refuses K17–K19 | MINOR (planned) | retention_report.py `load_contrasts`; company/gates/ANA_CONTRASTS_v2.json | ANA-03, then freeze | Sonnet (ANA-03) |
| F12 register hash chain has no external anchor | MINOR (OPINION) | tools/test_exposure_register.py:40-52 | disclose; record the register head sha in each DL row | Ops |

**No BLOCKER.** v4eval is frozen and verified. No delta changes an A16, arm-spec, T3, v4data/v4data06 or G2 behaviour. Every open defect fails closed or is a guard gap.

**VERDICT: ACCEPT_WITH_FIXES.**
- The reviewed deltas (INT-06 + REV-FL-01, DATA-09, ANA-02, DATA-07R1, DEV-04, FL-09) are accepted. F1 and F2 must be fixed before any ds3r3 re-evaluation is indexed, and before the final LR selection or MAIN FL launch.
- code_ds3r3 must not be packed until FL-10 and INT-07 are handed off and delta-reviewed (F3).
- No code_ds3r2 artefact is invalidated by a code delta.
