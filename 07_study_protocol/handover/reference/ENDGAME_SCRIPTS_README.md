# REES46 end-game operations scripts (OPS-ENDGAME, 2026-10-02)

**Status (OPS-ENDGAME-2, 2026-10-02): FINAL r7 form, validated, NOT RUN.** Phase C (pack / deploy / traj_cmp7) RAN on 10-01 (DL 342). Every
TO-CONFIRM item is resolved with evidence in §5, and the server validation is in §6: bash -n, 28 value-free reeval dry runs
and 25 argparse-only parses, all against the deployed r7 CLIs. The remaining decisions and blockers are in §7. No real
reeval, index build, freeze, lock, `--final` or TEST command was run, and no MAIN / VALIDATION / TEST value was read.

**Authority:**
- `reviews/FABLE_ENDGAME_REVIEW_20261001.md`: the §1 checklist, §3 D1–D5 and the §5 concurrence.
- DL 315–321: D2 / D3 / D4 are in DL 317.
- `contracts/PREG1_AMENDMENT_06P.md`.
- MASTER_PLAN_V5 §9–§11.
- Handoffs ANA-05, FIX-REEVAL-06O, FL-16 / FL-16b, DATA-09, INT-07.
- Format models: `reeval_cmds_r5.sh`, `reeval_06o_r6.sh`, `pack_r6.sh`, `deploy_r6.tmpl.sh`, `traj_cmp6.sh` and the lanes in `lanes_r6w/`.

**Common rules (enforced in `_endgame_lib.sh`):**
- Every server script sources `$Q/_common_code_ds3r7.sh` and refuses unless `CR = code_ds3r7`.
- Each phase logs to `runs_ds3/queues/endgame_<phase>.log`. Each job also writes its own log to `queues/EG<phase>_<name>.log`.
- A script refuses to start unless its predecessor's output file exists (`need`). It also refuses when its own output already exists (`once`: outputs are never overwritten).
- F, G and H also need `--i-confirm` as the last argument.
- A script stops at the first job with rc ≠ 0 or any `REFUSED` line (`ejob`).
- No log is ever printed to the console. Do not `cat` job logs from F onwards, because they contain values.
- `$Q = $V/runs_ds3/queues`, `$R = $V/runs_ds3`, `$V = …/ppsi_v4`.

## 1. Order

| # | Script (where) | Predecessor output it requires | Writes |
|---|---|---|---|
| C1 (**RAN 10-01, DL 342**) | `pack_r7.sh` (LOCAL) | REV-ANA-06, DEV-04, BM-02 reviews PASS; v4 contrasts sha in a DL row | `deploy_archive/r7/{code_ds3r7.tar, tools_r7.tar, SYNC_MANIFEST.json, TOOLS_MANIFEST.json}` |
| C2 (**RAN, DL 342**) | `deploy_r7.tmpl.sh` → filled `deploy_r7.sh` (SERVER) | tars + manifests in `$R/deploy_r7/`; r6 tools on the server = r6 manifest | `$V/code_ds3r7` (a-w); `$V/tools/{ana_block_index, retention_report, test_exposure_register}.py` r7 (the r6 copies go to `tools/_archive_r6/`); `$Q/_common_code_ds3r7.sh`; `$R/deploy_r7/R7_DEPLOYED.ok` |
| C3 (**RAN, DL 342**) | `traj_cmp7.sh` (SERVER, CPU) | `R7_DEPLOYED.ok` | `$R/deploy_r7/tmp/*`; **`R7_VERIFIED.ok`** |
| B | `B_reeval_panels.sh DRY <g>` then `REAL <g>` | `R7_VERIFIED.ok`; for REAL, `ok/B_DRY.ok`; the 4 FL sources DONE_MAIN | `$Q/endgame/FL_CKPT_RECORD_B.json` (O_EXCL); `B_TRUE_COLD_ADMISSION.tsv`; `$R/REEVAL_A40/<ARM>_<ROLE>_s2026_<PB\|CB>`; `ok/B_REAL.ok` |
| B2 | `B2_reeval_cfull_s2027.sh DRY <g>` then `REAL <g>` | `R7_VERIFIED.ok`; for REAL, `ok/B2_DRY.ok` | `$R/REEVAL_A40/C_FULL_[<ROLE>_]s2027_{PB,CB}` (8 DESCRIPTIVE); `ok/B2_REAL.ok` |
| P (Ops decision) | `P_park_smoke_dirs.sh --i-confirm` (SERVER) | the FTW / FTP lanes done (no process names a smoke dir) | `$R/_PARKED_SMOKE/<root>/<name>` (a-w) + `PARK_MANIFEST.json`; `ok/P_PARK.ok` (§4.6) |
| D | `D_freeze.sh <v4 file sha f1769f0b…>` (SERVER) | `ok/B_REAL.ok` + `ok/B2_REAL.ok` + the A6 inventory (all runs DONE) + v4 contrasts on the server with the DL sha | `$R/ANA/freeze_<tag>/{ANA_SCRIPT_FREEZE.candidate.json, claims_v4/, SECTION9_COMMANDS.txt}`; `ok/D_<tag>.ok` |
| D-local | Ops (LOCAL) | the candidate fetched and reviewed | `company/gates/ANA_SCRIPT_FREEZE.json` (O_EXCL copy of the candidate bytes); DL row with its sha; uploaded to `$V/company/gates/` and sha-checked |
| E | `E_check_only.sh <freeze sha>` | `ok/D_<tag>.ok` + the uploaded freeze with the DL sha | `$R/ANA/block_val_<tag>/` (a-w); **`$R/ANA/E_CHECK_PASS_<tag>.json`**; `ok/E_<tag>.ok` |
| F | `F_validation_read.sh <freeze sha> --i-confirm` | `E_CHECK_PASS_<tag>.json` (same freeze, same block tree digest) | `$R/ANA/out_val_<tag>/RETENTION_RESULT.json` + REPORT.md + curves (**the first MAIN read**); `ok/F_<tag>.ok` |
| G | `G_lock.sh <freeze sha> --i-confirm` | `out_val_<tag>/RETENTION_RESULT.json` + `ok/F_<tag>.ok` | `$Q/endgame/FL_CKPT_RECORD_G.json`; `$R/ANA/lock_<tag>/{ALL_METHODS_LOCK.candidate.json, REGISTER_ENTRY_5.json}`; `ok/G_<tag>.ok` |
| G-local | Ops (LOCAL) | the candidate fetched | `company/gates/ALL_METHODS_LOCK.json` (O_EXCL, candidate bytes); `python tools/test_exposure_register.py append --entry REGISTER_ENTRY_5.json` (entry 5); DL row; lock + register uploaded to `$V/company/{gates,registers}/` and sha-checked |
| H | `H_test.sh <step> <g\|-> <freeze sha> <lock sha> --i-confirm` | per step: `gate` → `build` → `final` → `reeval_dry` → `reeval` → `dev`; `base` / `onnx` after `final`; `analysis` after reeval + dev + base + onnx | `data_test/`, `REGISTRY_V2_TEST_EXT.json`, `$R/TEST/{REEVAL,DEV,BASE,ONNX}`, `$R/ANA/{block_test,out_test}_<tag>`; `ok/H_<step>_<tag>.ok` |

**Why C comes before B (review G-11).** The r6 reeval does not know the warm labels. B's FA_WARM_FROZEN / DP8_FROZEN rows need the ANA-06 reeval in r7.

**What the lanes keep doing.** In-flight r6 lanes, including the queued laneN33_* and laneD3_* lanes, finish and resume on r6, because `fl_code_sha256` changes with FL-16b. Each LAUNCH records which tree a run used, and D copies that into the freeze.

**Starting a lane.**
- Upload with `cd .local_ai/sasrec_v1/tools && MSYS_NO_PATHCONV=1 python lobot.py put <local> /home/jovyan/…/ppsi_v4/runs_ds3/queues/endgame/<file>` (an absolute path; `@W` = sasrec_v1).
- Then make the files read-only and sha-check them against the local copies.
- Start a lane as before: `(nohup setsid bash $Q/endgame/B_reeval_panels.sh DRY 1 > /dev/null 2>&1 < /dev/null &)`.

## 2. Preconditions, per phase
- **C1** (met and RAN 10-01, DL 342; kept as the record)
  - ANA-06 is done and REV-ANA-06 PASS, with the D5 acceptance wording: with v3 loaded, cold-line values and `--claims-only` are byte-identical to r6; with v4 loaded, the diff is listed.
  - REV-FL-16b PASS (DL 309).
  - The DEV-04 and BM-02 reviews are done.
  - The `ANA_CONTRASTS_v4.json` file sha is registered by a DL row. Its self sha today is 45aa5b53…, its `draft` is false, and it is built by ANA-06.
  - `pack_r7.sh` refuses when: launch_fl ≠ c9fdc8c0…, test_fl16b ≠ 0dbdbe0f…, any v4eval file changed, any receipt differs from r6, or any file outside the change set {reeval, launch_fl, tests_int, tests_ana, v4dev, tests_dev, v4bench, tests_bench} changed.
  - On a refusal, delete the LOCAL `deploy_archive/r7` (it is not evidence) and repack after the fix.
- **C2** (met and RAN 10-01, DL 342)
  - `$V/code_ds3r7`, `_common_code_ds3r7.sh`, `tools/_archive_r6` and `tools/test_exposure_register.py` must not exist.
  - The server `tools/` files must equal the r6 TOOLS_MANIFEST.
  - The server `tools/g2_run.py` must equal the local copy: `__G2_SHA__` comes from the pack output, and retention_report hashes this file.
  - `evaluator_code_sha256` in r7 must be 61b64e61….
- **B**
  - FL_T/FA_s2026, N33_DP8_s2026, FA_WARM_FROZEN_s2026 and DP8_FROZEN_s2026 must be DONE_MAIN.
  - N33_DP8_s2026 is queued behind the warm DP chain (DL 306), so B REAL is gated by that run.
  - It runs with `--allow-shared-gpu` (DL 292).
- **B2**
  - The C_FULL s2027 sealed CHECKPOINTS.json exists. MEASURED: today `REEVAL_A40` has no C_FULL s2027 MAIN / panel dir (only the 06O UNSEEN_VAL previews).
- **D: all of D3 / A1–A6 are done.**
  - laneD3_FTFA_2026 / 2027 have run: SELECTION_FT_FA.json is registered and FT_FA ×2 are done.
  - laneD3_PANEL has run (FT_C / LO on UNSEEN_HOLDOUT).
  - The warm FT lanes have run (FT_FA_WARM(_FROZEN) ×2).
  - Every N33 run is DONE_MAIN.
  - K16 is settled: laneD3_INT8_r7 (DL 343) writes `BENCH_RESULT.json` into `$R/ONNX_R7` (or `$R/ONNX_R7_U16384` after the 8.2
    fallback); the index gets exactly that dir. Otherwise D lists K16 under `declared_not_run`; K16 is MUST tier, so that
    declaration needs an Ops / Fable DL decision before E (§7.2).
  - No smoke-named dir is left in REEVAL_A40 / FL_T / DEV (P ran; D refuses otherwise, §4.6).
  - Upload `company/gates/ANA_CONTRASTS_v4.json` to `$V/company/gates/`. MEASURED: the server `company/gates/` holds only the G1 / G2 files.
  - D refuses when any FL_T dir is not DONE_MAIN. The exceptions are the STOPPED_REALLOCATED tuple and S_CAL*.
  - D also refuses when any expected run (the §11.2 / §11.3 list) is missing.
- **E**
  - The freeze is registered (DL row) and uploaded.
  - E refuses when any frozen identity (tools, reeval, contrasts, code root, the §9 command text) differs from what is deployed.
- **F**: the owner has been told that this command reads MAIN values.
- **H**
  - The owner has been told before `gate`.
  - The lock and register are on the server: `company/registers/` does not exist there today.
  - `g0/data_census/{SPLIT_COUNTS, CLIENT_CENSUS}.json` is uploaded (absent on the server, MEASURED; §4.9). The DATA-09 jobs
    run from r7 copies (Q12); `$V/code` is no longer used.

## 3. What each step reads → computes → writes
- **B / B2.** Reads the checkpoint records:
  - central runs: `sealed/CHECKPOINTS.json`;
  - PRE: the registered fl_init sha (cf08588c…);
  - FL runs: `ckpt/<run_id>/events.jsonl` plus the DEV LAUNCH `start_base`.

  B / B2 compute REEVAL rows and rank files, then write REEVAL dirs. The B5 post-check reads only the REEVAL_RECORD fields role, arm_label, checkpoint_sha256, device_pin_sha256 and status.
- **D.** Reads statuses, LAUNCH code identities, file hashes, the v4 contrast fields (hierarchy, Holm family, seeds, rule map) and `LIB.files_sha256` (from `retention_report.bind(code_ds3r7)`). It runs `--claims-only` under v4, which needs no block and reads no value. It writes the candidate freeze, which contains:
  - the §9 command text and its sha;
  - the declared-not-run list (DL 317 D3, plus K16 / TRUNC if they are absent);
  - the exclusions by name;
  - the N33 mapping;
  - FA_WARM* = S8;
  - no seed cuts and an empty `--allow-single-seed` list (D2 / §5).
- **E.** Runs the block index (paths, status and hashes only) and then `--check-only`, which binds the block and runs every check without printing a value. Writes the block dir and the PASS record (block tree digest).
- **F.** The single §9 read on the same block → `out_val_<tag>`.
- **G.** Hashes the block entries' rows / ranks files and the checkpoints:
  - FL: sealed against events.jsonl;
  - central: CHECKPOINTS.json;
  - PF stores.

  From these it builds the TEST plan:
  - reevals on MAIN_TEST: C_BIG ×2, C_SMALL ×2, C_FULL s2026 (+ s2027 DESCRIPTIVE), PRE, cold FA ×2, N33 FP / PF / FA_Q8 / FA_1024 / DP8 ×2, N33 DP1 s2026, FA_WARM ×2, FA_WARM_FROZEN ×2, FP / PF / FA_Q8 / FA_1024 _WARM_FROZEN ×2, DP8_FROZEN ×2 and DP1_WARM_FROZEN s2026, each PB + CB;
  - TEST panels exactly for the VALIDATION panel dirs that exist (TRUE_COLD only where B admitted it);
  - device runs: LO / FT_C / FT_PRE / FT_FA / FT_FA_WARM_FROZEN / FT_FA_WARM ×2, ordered seed 2026 first, plus FT_C and LO on UNSEEN_HOLDOUT_TEST s2026 (K9b), with the locked selections and the same bases as the VALIDATION runs (checked against their LAUNCH `start_base`);
  - the six baselines: SKNN k 500 / sample 5000, as BASE_R4.

  G writes the lock candidate (v4 contrasts verbatim + `all_methods_lock`) and the entry-5 JSON: MANIFEST_BUILD / RETRO_TEST / NONE / NO_FEEDBACK / lock sha / r7 aggregate.
- **H.**
  - `gate`: register chain + entry 5 + d9_gate OPEN.
  - `build` / `final`: DATA-09 jobs, counts-only.
  - `reeval*` / `dev` / `base` / `onnx`: the lock plan with `--registry-ext REGISTRY_V2_TEST_EXT.json`. Each device base sha is compared with the lock record before launch.
  - `analysis`: the TEST index (records + `VALIDATION_RESULT=` the F result) and `retention_report --split TEST --contrasts LOCK --all-methods-lock LOCK --test-exposure-register … --registry-ext …`.
  - After H: register entry 6 (RESULT), a DL row and the thesis batch, all done by Ops.

## 4. Gaps found while preparing (MEASURED on the server; no value read) and what OPS-ENDGAME-2 did
1. **No C_FULL s2027 reeval (B2) existed.** `B2_reeval_cfull_s2027.sh` added (OPS-ENDGAME); its 8 dry runs pass on r7 (§6).
2. **FL runs record no checkpoint file sha256** — only `content_digest` per save in `ckpt/<run_id>/events.jsonl`. RESOLVED (Q1).
3. **Missing on the server:** `company/registers/`, `company/gates/ANA_CONTRASTS_v4.json`. `tools/test_exposure_register.py`
   is now deployed (7d59e828, DL 342). Ops uploads the contrasts file (f1769f0b) before D and the register before H.
4. **K16:** `$R/ONNX` (r6) holds rows.npz but no BENCH_RESULT (DL 320). The r7 run is `laneD3_INT8_r7` (DL 343), into
   `$R/ONNX_R7` or, after the 8.2 fallback, `$R/ONNX_R7_U16384`. Neither exists yet (MEASURED 10-02 11:30Z; the lane waits
   for the Wave-1 driver). The scripts now pick exactly that dir (`onnx_val_dir`), never `$R/ONNX`. `$R/TRUNC` is empty.
5. **DATA-09 TEST jobs hard-code `cd "$W/code"`** (the old tree). RESOLVED (Q12): H runs r7 copies.
6. **Smoke dirs refuse the WHOLE index build.** `ana_block_index` calls `check_smoke_name` on every `--reeval` / `--fl` /
   `--dev` dir before any exclusion (L320-323; called at L350, L418, L510). MEASURED 10-02: `$R/DEV` holds `FT_PRE_SMOKE_s2026`,
   `FT_FA_WARM_SMOKE_s2026/7` and `FT_FA_WARM_FROZEN_SMOKE_s2026/7`, so E would refuse. **New `P_park_smoke_dirs.sh`**
   (PREPARED, NOT RUN; an Ops decision) moves them into `$R/_PARKED_SMOKE/DEV/` with a write-once manifest (tree digest
   before = after) and makes them read-only. D now refuses while any smoke-named dir is left in an indexed root.
   SEL_* dirs, S_CAL*, the STOPPED dirs and the 06O preview dirs are excluded by the index itself and need nothing.
7. The review's register command (`append --purpose … --lock …`) is not the tool's CLI: it is `append --entry <json>`
   (MEASURED --help), and G writes that JSON.
8. **DEV LAUNCH key:** launch_device records the base as `start {path, file_sha256, …}` (MEASURED); the prepared scripts
   read `start_base` (always empty, so the base cross-checks silently did nothing). FIXED in `seal_fl_ckpts` and in G.
9. **`g0/data_census/` is absent on the server** (MEASURED). d9_finalize `g0` needs SPLIT_COUNTS.json and
   CLIENT_CENSUS.json (`v4data09.d9_common` L69-70). Ops uploads them from the local `g0/data_census/` (sha-checked) before
   H `gate`, which refuses without them.
10. The C scripts (`pack_r7.sh`, `deploy_r7.tmpl.sh`, `traj_cmp7.sh`) RAN on 10-01 (DL 342) and are left byte-unchanged as
    the record of what ran. Their in-file TO-CONFIRM notes (Q5, Q8) are resolved below.

## 5. The 20 TO-CONFIRM items — all resolved (evidence; r7 tools = ana_block_index 6e7043bb, retention_report 1d50f7f2,
##    reeval 833f0c98; local copies byte-equal to the server)
- **Q1 FL checkpoint sha — RESOLVED, implemented in `seal_fl_ckpts` (lib).** Per checkpoint, in order:
  1. The file is in `<run>/ckpt/<run_id>/`, and the payload `run_id` is that dir name.
  2. Its size equals the last `events.jsonl` entry with `saved` = the file name. The entry format is MEASURED as `{bytes,
     content_digest, disk, saved, utc}` (`fl.checkpoint.CheckpointManager._save`).
  3. `load_checkpoint` re-verifies `__content_digest__`, and that digest equals the entry's `content_digest`.
  4. Only then is `sha256(file)` recorded. It must equal every DEV LAUNCH `start.file_sha256` whose `start.path` names
     the file.
  5. reeval then re-hashes the file against `--expect-sha256` (`reeval.run` L969).

  MEASURED on the server (scratch record): FA_s2026 best fd68fe1a… (2 DEV cross-checks: FT_FA_s2026 and SEL_FT_FA, both
  equal), FA_WARM_FROZEN_s2026 best 6f1c3e92… (2 cross-checks), DP8_FROZEN_s2026 endpoint f875d929… (no device run uses
  it). The re-seal re-verified all 3.
- **Q2 TRUE_COLD admission — RESOLVED.** reeval refuses TRUE_COLD only for the registered central sources C_FULL / PRE
  (`SOURCE_ROLES`, reeval L172 / L420; the refusal text names the role). All 6 B sources are admitted: the dry runs gave
  rc 0 for C_BIG, C_SMALL, FA, FA_WARM_FROZEN and DP8_FROZEN on TRUE_COLD. The N33_DP8 dry run waits for that run. The
  B DRY tolerance rule stays as a guard.
- **Q3 checkpoint ids — RESOLVED.** Central ids equal the VALIDATION REEVAL_A40 records (MEASURED:
  `02B_SASREC_MAIN_s2026:best` / `:endpoint_6.0`, `SMALL_SASREC_D64_B2_MAIN_s<S>:best`,
  `PRETRAIN_FOR_FL_SMALL_SASREC_D64_B2_s2026:endpoint_6.0`, `C_FULL_02B_SASREC_s<S>:best`). FL ids are
  `<run_id>:best` / `<run_id>:endpoint_6.0`, with `<run_id>` = the `ckpt/<run_id>` dir (launch_fl's own CB rows use
  `<run_id>:endpoint_6.0`, L1641). retention_report A-F10 accepts PRACTICAL_BEST through the row efe (L1985-2015).
- **Q4 dir names — RESOLVED.** The index rule is `<head>_s<S>_<PB|CB>`, where head = arm or arm + `_…`, and seed / rule
  must equal the record (L359-363). So `DP8_<ROLE>_s2026_CB` (block arm DP8; the N33 dir is only the source) and the TEST
  names `<ARM>_s<S>_<TAG>` / `<ARM>_<ROLE>_s<S>_<TAG>` all fit.
- **Q5 traj_cmp7 dry-run JSON — RESOLVED by the run (DL 342).** The parser was fixed for the multi-line JSON, and the
  arm_spec / config digests were equal r6 = r7.
- **Q6 LR_SELECTION — RESOLVED: not passed.** It is optional in the report (reported only if present, L4686) and in the
  index (RECORD_KEYS). The S4 tuning file replaced it and is hashed into the freeze. The env toggle was removed, because a
  frozen line must not depend on an env variable.
- **Q7 lock schema — RESOLVED.** `load_contrasts_v4` compares every `v4_registry_content()` key and ignores any other key
  (L1072-1081). `content_sha256` is built from the registered keys only, so the lock (v4 verbatim + `all_methods_lock`)
  loads on TEST and matches VALIDATION_RESULT's `contrasts_content_sha256` (L5107). G now recomputes `self_sha256` with
  the utf8-compact rule, so `_self_ok` gives VERIFIED (L4639-4647), and it keeps the v4 self sha as
  `contrasts_v4_self_sha256_before_lock`. d9_gate and the report bind the lock by file sha (L5981; d9_gate L72).
- **Q8 v4bench in r7 — RESOLVED (DL 341 / 342).** It is excluded; benchmarks run from code_bench3.
- **Q9 device status — RESOLVED.** `launch_device` writes `DONE` (L659) and the index requires it (L523). D now requires
  `DONE` for every expected device run.
- **Q10 PRE sha key — RESOLVED.** `PRETRAIN_FOR_FL_INIT.json → fl_init.sha256` (with path / kind endpoint_6.0) is exactly
  what `reeval.check_pre_init_record` checks (L527-543).
- **Q11 PF store sha — RESOLVED.** launch_fl records each store in RUN_SUMMARY: `pf_store_best {file, sha256}` for PB /
  `PF_STORE_best.pt` and `pf_store.record {file, sha256}` for CB / `PF_STORE.pt` (L1631-1637). G checks the file hash
  against that record and passes it as `--expect-pf-store-sha256`.
- **Q12 TEST build tree — RESOLVED (DL 327 gap).** H `build` / `final` write `queues/endgame/job_test_{build,final}_r7.sh`.
  Each is the r7 file (its sha = the r7 SYNC_MANIFEST entry, bc2509dd… / 24c60cab…) with only `"$W/code` →
  `"$W/code_ds3r7` changed: 1 line in build, 3 in final, checked by `diff`. It is written once, read-only. The `gate` step
  runs d9_gate from `$CR`. MEASURED in scratch: both copies pass the source sha check, the line counts, "no old tree
  left" and bash -n. The `$V/code` sync precondition is dropped.
- **Q13 K16 on TEST — PARTLY RESOLVED; OPEN BLOCKER for the TEST ONNX row (§7).** The VALIDATION form is the
  laneD3_INT8_r7 one (DL 343). H now repeats it on TEST: same checkpoint, `--quiet`, and `--first-users 16384` iff
  VALIDATION used ONNX_R7_U16384 (from the freeze → lock). But the r7 bench needs the rows' `batch_view` for C_SMALL's
  derived view (DEV-04 `check_rows_view`). `v4int/dump_rows_retro.py` (INT-07, unchanged in r7) drops the non-array name
  fields (ArrayFieldsView) and has no `--ckpt` binding. H therefore stops, value-free, after the dump when `rows.meta.json`
  lacks `batch_view` / `feature_view_binding`.
- **Q14 TEST index flags — RESOLVED.** `--fl` on TEST refuses (index L414-415). H passes `--reeval / --dev / --base
  [--onnx]`, `--registry-ext`, and `--records` = the VALIDATION records + `VALIDATION_RESULT=<F result>` (KEY=file form).
  Defined once in `_endgame_ana_args.sh` (`ana_index_test` / `ana_report_test`).
- **Q15 EVENT_SIZE — RESOLVED: not passed.** No registered per-event size exists. The report labels the raw-event bytes
  level NOT_REGISTERED (L4054-4065). It is in the freeze's `declared_not_run`.
- **Q16 contest singles — RESOLVED.** 02B_SASREC = the C_BIG s2026 / s2027 MAIN_TEST reevals already in the lock plan.
  02B_GRU / GRU_D128 cannot be re-evaluated: r7 reeval `--variant` takes SASREC_D256_B3 / SASREC_D64_B2 only (MEASURED
  --help), and no GRU_D128 VALIDATION entry exists. It is declared not run in the freeze and the lock.
- **Q17 PRE PB on TEST — RESOLVED.** VALIDATION has `PRE_s2026_PB` and `_CB` (MEASURED), so TEST keeps both labels on the
  one θ_pre file (A-F10 NO_SELECTION_ARMS).
- **Q18 FA_WARM panel rows — RESOLVED.** They are not registered (DL 317) and are declared not run.
- **Q19 second freeze name — RESOLVED.** Tag v1 = `ANA_SCRIPT_FREEZE.json`; another tag = `ANA_SCRIPT_FREEZE_<tag>.json`
  (lib), with its own DL row. The lock name stays fixed, because `d9_gate` DEFAULT_LOCK = `company/gates/ALL_METHODS_LOCK.json`
  (L42). Only one lock ever exists, after the one VALIDATION read.
- **Q20 FL-16 tests on the server — RESOLVED by the run (DL 342).** 21 passed and 16 xfailed; the 1 failure is the known
  Windows pin (DL 245).
- **Registered facts wired in (DL rows):**
  - D refuses unless the contrasts sha = **f1769f0b…** (DL 336), and the server file = it.
  - D checks the record prefixes: S7 **d67a0a85**, S8 **8b79d305**, S_RECORD_WARM_FROZEN **2e47ba39**,
    SELECTION_FT_FA **a645edaf** (DL 351), S_RECORD 7ab91eb2, SELECTION_FT_C e47965d2, tuning 9bca186e, pin db744ca3.
  - **N33 / STOPPED:** D's inventory skips FP_s2026, FP_s2027 and FA_1024_s2026, plus any dir carrying the
    `STOPPED_REALLOCATED.json` marker. It also requires the marker in the three named dirs (all present, MEASURED). It
    requires all 11 N33 dirs (FP / PF / FA_Q8 / FA_1024 / DP8 × 2 + DP1 s2026) DONE_MAIN. `--allow-single-seed` stays empty.

## 6. Validation done on the server (OPS-ENDGAME-2, 2026-10-02 ~11:35Z; scratch `$R/deploy_r7/tmp/eg2/`; nothing real run)
- `bash -n`: **14 / 14** scripts OK (bash 5.2.21). The embedded Python heredocs compile.
- **reeval `--dry-run` is value-free (code):** `run()` returns the dry record before the out-dir check, `gpu_checks` and
  any data / evaluator (L980-991). The record holds identities only (sha, source run, meta, labels). `main` prints only
  that record or `REEVAL REFUSED` (L1325-1333). MEASURED: 0 `*_DRY` dirs were created.
- **B dry runs, 20 / 20 rc 0, 0 REFUSED:** C_BIG, C_SMALL, FA, FA_WARM_FROZEN (PB) and DP8_FROZEN (CB) × UNSEEN_HOLDOUT /
  SPARSE_PANEL / HEAVY_PANEL / TRUE_COLD. DP8_FROZEN carries `warm_line.block_arm DP8_FROZEN`.
  **Not yet dry-runnable:** N33_DP8_s2026 endpoint × 4 roles (run not DONE_MAIN; the N33 DP chain started 08:35Z,
  DL 352). B DRY dry-runs them when it runs.
- **B2 dry runs, 8 / 8 rc 0:** C_FULL s2027 × MAIN / 3 panels × PB / CB, each `central_source.seed_status DESCRIPTIVE`,
  and the B2 grep matches.
- **argparse-only parse** (each tool's own `parse_args`, stopped right after parsing; nothing executed): **25 / 25
  PARSE_OK.** Covered: the frozen §9 INDEX line, the CHECK and READ lines, D's `--claims-only`, the TEST INDEX and REPORT
  lines, register `append --entry` / `verify`, reeval MAIN_TEST central / pf, launch_device MAIN_TEST × 6 arms +
  UNSEEN_HOLDOUT_TEST `--resume`, v4base MAIN_TEST × 6 baselines, dump_rows_retro `--first-users`, and the v4dev.bench
  laneD3_INT8_r7 form.

## 7. Remaining blockers / decisions (none blocks B, B2 or D's preparation)
1. **Smoke dirs (§4.6):** Ops decides to run `P_park_smoke_dirs.sh --i-confirm` after the FTW / FTP lanes are done, and
   adds a DL row with the manifest sha. Without it, D refuses (on purpose) and E would refuse.
2. **K16 VALIDATION:** if laneD3_INT8_r7 has not written a BENCH_RESULT by D, K16 (MUST tier) is declared not run. That
   needs an Ops / Fable DL decision before E.
3. **K16 TEST (Q13):** dump_rows_retro needs an INT-07 / DEV-04 fix (keep `batch_view`, bind `--ckpt`) + review + an r8
   tree before H `onnx`. Otherwise the TEST ONNX row is declared not run in the lock. Decide before G.
4. **Uploads before D / H:** `company/gates/ANA_CONTRASTS_v4.json` (f1769f0b) before D; `company/registers/` with
   entry 5 and `g0/data_census/{SPLIT_COUNTS, CLIENT_CENSUS}.json` before H `gate`. Each sha-checked.
5. B REAL waits for N33_DP8_s2026 (and D for every N33 / FT_FA / FT_FA_WARM_FROZEN_s2026 run).
6. A Fable review of these final scripts before any phase runs (DL 354).

## 8. OPS-ENDGAME-3 (2026-10-03): the Fable gate fixes (reviews/FABLE_GATE_ENDGAME_SCRIPTS_20261003.md, GO-WITH-FIXES)
Details and shas are in `company/handoffs/OPS-ENDGAME-3.md`. In short:
- **FIX-1, r7b, DEPLOYED + VERIFIED.**
  - Scripts: `pack_r7b.sh` (built from the r7 tar + the 8 reviewed files), `deploy_r7b.tmpl.sh`, `traj_cmp7b.sh`.
  - `$V/code_ds3r7b` sits beside r7 (r7 untouched).
  - G records the r7b aggregate + the dump_rows_retro / bench shas in `test_plan` and `code_trees`.
  - H `onnx` verifies all 688 r7b files against the lock, then runs ONLY the dump + bench from r7b (`r7b_run`).
  - Every other step stays on r7 (the lib guard is unchanged).
- **FIX-2.** All 15 changed files are uploaded to `queues/endgame`, CRLF-stripped; server = local (sha256sum -c), read-only.
  bash -n passes 17 / 17. The old server H was 94d5edd3; it is now 11c04583.
- **FIX-3.**
  - D pins `k16_pass1_artifacts`: 8 files of ONNX_R7_U16384 (excluding LATENCY_PASS2 / PASS2_HOST_LOAD) +
    ONNX_R7/PASS1_HOST_LOAD.json. D sets `k16_latency` PASS2_PENDING, and refuses if ONNX_R7 ever holds a BENCH_RESULT.
  - G re-hashes those files, copies them, adds the LATENCY_PASS2 sha if present, and states the TEST latency rule.
- **FIX-4.** D requires `ok/P_PARK.ok` and `_PARKED_SMOKE/PARK_MANIFEST.json` = ec779fd4ea9dbe51…, and lists the 5 dirs
  under `parked_smoke`. G copies them. Path note: the manifest is in `_PARKED_SMOKE/`, not `_PARKED_SMOKE/DEV/`.
- **FIX-5.**
  - `metric_table.py` (561d758a…) is frozen in the freeze `scripts`, so freeze_check covers it.
  - It runs in F after the single read (same block, E's index sha) and in H `analysis` after the TEST report.
  - It copies the evaluator cells verbatim (HR / NDCG / MRR @ 1 / 5 / 10 / 20, micro + macro, E2E + RANKABLE; 48 per
    entry); there is no new computation.
  - The registration text is `REG_DESCRIPTIVE_METRIC_TABLE.md`. D refuses without `MT_REG_DL=<DL row>`.
