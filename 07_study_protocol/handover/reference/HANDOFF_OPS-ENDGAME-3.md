# OPS-ENDGAME-3 handoff: Fable gate FIX-1..FIX-5 on the endgame scripts (Claude Opus, 2026-10-03)

**Authority:** reviews/FABLE_GATE_ENDGAME_SCRIPTS_20261003.md (GO-WITH-FIXES); DL 358 / 359 (DEV-05, REV-DEV-05 PASS);
DL 378 / 380 (OPS-B13-P2 `--latency-only`, REV-OPS-B13-P2 PASS); DL 356 (parked smoke dirs); DL 368 / 369 / 371 / 372 (K16).

**Not done:** no real reeval, index build, freeze, lock, `--final` or TEST command; no MAIN / VALIDATION / TEST value
read. Server actions were limited to: the r7b deploy, the traj_cmp7b CPU checks (tiny runs, pytest, launch_fl dry runs),
the endgame-file upload, and value-free hash / parse checks.

## FIX-1: code_ds3r7b (MEASURED)
**What r7b contains:** r7 plus exactly 8 files, each asserted against its reviewed sha:

| file | sha256 |
|---|---|
| v4dev/dump_rows.py | 3961e230… |
| v4dev/bench.py | **9662cba5…** |
| v4int/dump_rows_retro.py | 83f5f2f7… |
| tests_int/test_int07_test_path.py | c9558aad… |
| tests_dev/test_dump_rows_retro_view.py (new) | 6efd04bd… |
| tests_dev/r6_ref/dump_rows_r6.py.txt (new) | 4696c4af… |
| tests_dev/r6_ref/dump_rows_retro_r6.py.txt (new) | 9853cc55… |
| tests_dev/test_bench_latency_only.py (new) | 71f79745… |

**bench.py 9662cba5 contains DEV-05 (verified):** all 37 lines DEV-05 added to the r7 file (r7 0b74e77a → DEV-05
8cb79e96) are present in 9662cba5. The only DEV-05 line changed is the `__all__` export list.

**Pack:**
- tar `code_ds3r7b.tar` **3b766594e2e28857a8b2b4e17c7257117e76805004887811e79c226dbe930876**.
- aggregate **1cb1e70631b5cbc903dc3c406af6d909c876a2985a6c2807ee56cc8de6abbb8d**.
- 688 files; base r7 aggregate d76b264e….
- SYNC_MANIFEST.json 9c5783da…; filled deploy_r7b.sh 6bc257ef….
- Unchanged from r7 (asserted): reeval 833f0c98, launch_fl c9fdc8c0, launch_device 1968ca95, and v4eval / fl / receipts.

**Deploy:**
- Done 2026-10-03T12:49:31Z to `$V/code_ds3r7b` (read-only), beside r7, with r7 checked unchanged.
- Per-file verify 688 / 688; evaluator 61b64e61.
- No new `_common` file; tools/ untouched.

**traj_cmp7b (QUEUE_DONE 12:51:56Z, R7B_VERIFIED.ok):**
- Tiny-run digests r7 == r7b for all 7 arms (DP 08383b6f, FA 807eadfb, FA_1024 878fc357, FA_Q8 651f0eb0, FP 825b27f8, …;
  the DL 342 values).
- test_fl16 / 16b in r7b: 21 passed, 16 xfailed, 1 failed. The failure is the known Windows pin
  (`test_existing_arm_digests_and_trajectories_unchanged`, DL 245 / 342), accepted.
- DRY_EQUAL for DP8 and DP8F.
- DEV-05 / P2 tests in the deployed r7b: 53 passed, 14 xfailed.

**G:**
- Verifies all r7b files against deploy_r7b/SYNC_MANIFEST, with base = the frozen r7.
- Adds `code_trees.code_ds3r7b`.
- Adds to `test_plan`: `onnx_tree`, `onnx_tree_aggregate_sha256`, `onnx_dump_rows_retro_sha256`, `onnx_bench_sha256`.

**H `onnx`:**
- Re-verifies all 688 files and the aggregate against the lock fields.
- Runs only the dump (with `--ckpt`) and the bench via `r7b_run` (cd + PYTHONPATH = r7b).
- The batch_view guard stays. All other H steps run on r7.
- MEASURED: the verification snippet passes on the real tree, and both r7b command forms parse.

## FIX-2: server files
- 15 files uploaded to `runs_ds3/queues/endgame/`, CRLF-stripped, and confirmed equal to the local copies with
  `sha256sum -c`; all are read-only.
- bash -n passes on 17 / 17.

| file | sha256 |
|---|---|
| H_test.sh | 11c04583… (was 94d5edd3) |
| D_freeze.sh | 2fa8d837… |
| F_validation_read.sh | 7f98b65a… |
| G_lock.sh | 372e9e51… |
| metric_table.py | 561d758a… |
| pack_r7b.sh | 607f2021… |
| deploy_r7b.tmpl.sh | c4631787… |
| traj_cmp7b.sh | 988cd5c6… |

The other 7 files are unchanged (B, B2, E, P, lib, args, freeze_check).

## FIX-3 / FIX-4 / FIX-5
- **FIX-3:**
  - D's `k16_pass1_artifacts` covers the 8 files in ONNX_R7_U16384 (incl. BENCH_RESULT, rows.meta, rows.npz, both
    models, ranks) + ONNX_R7/PASS1_HOST_LOAD.json. All present, MEASURED.
  - `k16_latency` = PASS2_PENDING (DL 368).
  - D refuses a BENCH_RESULT in ONNX_R7 (none, MEASURED).
  - G re-hashes, copies, and adds the LATENCY_PASS2 sha when it exists. The TEST latency rule is in `test_plan`.
- **FIX-4:**
  - D requires P_PARK.ok and PARK_MANIFEST ec779fd4ea9dbe51…, and lists the 5 parked dirs. G copies them.
  - Path note: the manifest is at `_PARKED_SMOKE/PARK_MANIFEST.json`, not under `DEV/`.
- **FIX-5:**
  - Registration text: `plans/ops_scripts/endgame/REG_DESCRIPTIVE_METRIC_TABLE.md` (paste as a DL row before D).
  - The extractor `metric_table.py` (561d758ae2431d2a659e42e310d439f446bbf179dcd6d83f8400c1f3d2e77b57) is synthetic-tested
    locally: 48 cells per entry; it refuses on re-run and on the wrong split.
  - D freezes it, requires `MT_REG_DL`, and records `descriptive_metric_table`.
  - F runs it after the single read; H `analysis` runs it after the TEST report. A failure stops only the table.
  - It copies values only: no CI or mean is computed (stated in the registration).

## For Ops before D
1. Register the FIX-5 DL row. Write a DL row with these shas.
2. Upload ANA_CONTRASTS_v4.json (f1769f0b) to company/gates.
3. Run D with `MT_REG_DL=<row>`, once N33_FA_Q8_s2026 is DONE_MAIN.

The thesis-reference capture is left to Ops.
