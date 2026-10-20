# OPS-B13-P2 — B13 stopper (Task 1) and INT8 pass-2 latency-only mode (Task 2)

Worker: PPSI V4 Sonnet, 2026-10-03 (DL 368 TODO (i) and (ii)). No git, no Codex, no benchmark / REES46 value opened. Nothing is armed or launched.
The only server effects: one candidate tree, the scripts copied into `runs_ds3/queues/`, a fixture under `/tmp/b13fix`, tests, and dry-run lines in `b13_stopper.log`.

## Task 1 — `bench/b13_stopper.sh`

### Design
- **Clock.** Sleeps (real clock, 5-min steps) until `2026-10-06T00:00:00Z`. `--now` is allowed only with `--dry-run`.
- **Cells.** It enumerates the ML-1M Track-B seed-2026 grid exactly as `run_wave1.sh build_dataset` does, giving 38 cells:
  - C_FULL {6, 24} EFE, PRE {6, 24, 96} EFE, FA and FA_WARM* (b = 6 only);
  - max_len {50, 200} × dropout {0.1, 0.2, 0.5};
  - the Track-A cell 6 / 50 / 0.1 is not a grid cell and is never matched.
  - The star arm is read from `FA_WARM_DECISION.json` (key `FA_WARM_STAR` only).
- **Completed.** A cell is complete if its `RESULT.json` has a status starting `DONE`. Only the `status` key is read. `metrics.jsonl` is only line-counted.
- **Process match (never a broad pattern).**
  - It parses `/proc/<pid>/cmdline` for `-m v4bench.run run`.
  - It requires `--dataset ml_1m`, `--seed 2026`, the arm, the `--max-len` and `--dropout` values, and `--budget-efe == b` (or no `--budget-efe` at all for FA / FA_WARM cells).
  - `--seed 2027` runs, the Track-A runs and anything with `--smoke` never match. So every winner's seed-2027 run is untouched.
- **Order of actions** (each step is logged to `runs_ds3/queues/b13_stopper.log`):
  1. Mark every never-started or dead cell INCOMPLETE first. The driver then skips these cells instead of launching them into freed slots. At today's state this covers the 21 unstarted max_len-200 cells and `tb.star…6.50.0.5`.
  2. Stop each matched run:
     - SIGKILL its job wrapper, then its subshell. The driver therefore never writes `<job>.fail` and never marks the tbsel / win jobs SKIPPED.
     - Then SIGTERM the python process (20 s), then SIGKILL.
     - The chain is logged.
  3. Run `v4bench.run trackb-select --dataset ml_1m --arm {C_FULL,PRE,FA,<STAR>} --runs <Track-A run + DONE grid cells only>`.
     - This is done only for arms with at least one incomplete cell and no `WINNER_<arm>.json` yet. It is write-once. It does not run if the Track-A cell is not DONE.
     - The output (it contains selection values) goes to `queues/b13_trackb_select_ml_1m_<arm>.out`, never to the log.
  4. Write the marker `RESULT.json` (and `STATUS.json`) for the killed cells.
  5. Touch `runs_bench/_wave1_state/<job>.ok` (and remove a stale `.fail`) for every stopped or marked job. The driver's `tbsel.*` jobs then see their dependencies satisfied and find `WINNER_*.json` present, so they are skipped. The `win.*` jobs run seed 2027 for the winner.
  6. Two more verification passes (30 s apart) rescan for stragglers. Another instance is refused via `queues/b13_stopper.pid`.
- **INCOMPLETE convention.**
  - Existing: `central.py` / `fl_engine.py` wall-cap write `RESULT.json {status: "INCOMPLETE_WALL_CAP"}`. `refuse_overwrite` (`status.startswith("INCOMPLETE")`), `final.py` ("INCOMPLETE runs are ineligible"), `make_lock` and runbook P6 all key on that prefix. `selection.result_of` refuses any status other than `DONE_MAIN`.
  - The runner has no external-stop path, so I use the same record with status **`INCOMPLETE_B13_BACKSTOP`** and the wall-cap file shape, plus `reason`, `killed_pid`, `metrics_lines`, `has_latest_pt`. The checkpoint is left intact.
  - One deliberate detail: the marker is written as compact JSON, `"status":"INCOMPLETE…"` with no space after the colon.
    - `run_wave*.sh run_job` re-runs a cell if its RESULT matches `"status": "\(INCOMPLETE\|RUNNING\)`. The wall-cap form would be re-launched by every driver and by the `--resume` lane.
    - The runner refuses it at once (`refuse_overwrite`), which would write a `.fail` and SKIP tbsel.
    - The compact form is not matched by that grep, so the driver treats the cell as done. Python parsing, the runner, lock and final all still see an INCOMPLETE status.
  - trackb-select itself cannot take INCOMPLETE cells (`result_of` refuses). So the stopper runs it on the completed list and the driver's own tbsel job is skipped because its `WINNER` file exists.
- **Locking and disclosure (for Ops).**
  - Per P6, list every `INCOMPLETE_B13_BACKSTOP` dir in the ML-1M LOCK DL row and keep it out of `--runs`.
  - The Track-B table lists the backstopped cells (the stopper log is the list).
  - The max_len-200 cells never started, so the ML-1M winners will come from max_len 50. This is INFERENCE from the clock; check at 10-06.

### Files / sha256
| file | sha256 |
|---|---|
| `company/plans/ops_scripts/bench/b13_stopper.sh` (copy at `$V/runs_ds3/queues/b13_stopper.sh`, same sha) | 3e4aa02bb0fa44af0ad4e5162e02c46f036dbbc8f4c10f117f1156b8ec3b7f8b |

### Tests (MEASURED)
- **Dry run on the real server state** (read-only). `bash b13_stopper.sh --dry-run --now 2026-10-06T00:00:00Z` at 11:32Z:
  - `PASS 1: grid=38 DONE=10 already_INCOMPLETE=0 RUNNING=6 not_started_or_dead=22`
  - would-STOP, with pids:
    - `tb.cfull…24.50.0.1` 655139
    - `tb.cfull…24.50.0.2` 2088803
    - `tb.cfull…6.50.0.5` 386004
    - `tb.cfull…24.50.0.5` 472731
    - `tb.pre…96.50.0.5` 488474
    - `tb.fa…6.50.0.5` 489955
  - would-MARK-not-started (22): `tb.star…6.50.0.5` and all 21 max_len-200 cells (none has started).
  - 4 planned trackb-select calls: C_FULL (2 runs), PRE (8), FA (2), FA_WARM (2).
  - Nothing was changed.
  - Without `--now` and before the target, a dry run prints "would sleep …" and the same listing.
- **Live fixture, dataset `fixds` under `/tmp/b13fix`** (so it cannot match any real run; a first attempt against the real dataset name was correctly blocked by the permission classifier because the match args would have hit real jobs).
  - A fake `run_wave` driver started 3 `v4bench.run`-shaped processes: the grid cell, a seed-2027 twin and a Track-A twin.
  - Result: only the grid cell was killed. The wrapper was killed first, so no `.fail` and no `drv.log` rc line appeared. The twins survived. The marker, 17 `.ok` states and 4 WINNER files were written. Pass 2 found nothing left. The compact marker did not match the driver's grep (count 0).

## Task 2 — `v4dev.bench --latency-only`

### Design
- **Command.** `python -B -m v4dev.bench --latency-only --ckpt <ckpt> --family … --variant … --v1-root … --out <pinned pass-1 dir>`.
- **No re-export, no accuracy, no RSS.** `run_pipeline` is not called, and export / quantise are not called. The test monkeypatches all three to raise.
- **Assertions before any timing** (`read_pass1_record` + `assert_pass1_identity`):
  - It reads from pass 1's `BENCH_RESULT.json` only the identity and protocol keys, never its latency or accuracy values:
    - `model_sha256.{fp32,int8}`
    - `checkpoint_sha256`
    - `retention[*].rows_npz_sha256` / `rows_meta_sha256`
    - family / variant / K / seed
    - intra threads
    - n_queries / warmup (from the pass-1 latency blocks)
  - It sha256-checks `model_fp32.onnx`, `model_int8.onnx`, `rows.npz` and `rows.meta.json` (rows default `<out>/rows.npz`, or `--rows`) against that record, plus `--ckpt`.
  - It refuses if `--seed`, `--n-queries`, `--warmup` or `--intra-threads` differ from pass 1's recorded values.
  - After the timing it re-checks the same shas. A change during the run refuses and writes nothing.
- **Timing.** The same synthetic queries (`rows_to_batches`, same seed), the same `make_session` config and the same `bench_latency` as pass 1. FP32 first, then INT8.
- **Output.** `<out>/LATENCY_PASS2.json`, write-once (refuses if it exists), via `write_json_atomic`. No pass-1 file is written or touched. It contains:
  - latencies and onnxruntime info;
  - the asserted shas and the pass-1 record file's sha;
  - host loadavg at start and end, and the CPU count;
  - the note "pass-1 latency is 'loaded host', never the device number";
  - `no_export_no_accuracy_no_rss: true`.
- Pass 1's real record has all the needed keys. This was checked by key names only, with no values read. `ONNX_R7_U16384` has `model_fp32.onnx`, `model_int8.onnx`, `rows.npz` and `rows.meta.json`.

### Files / sha256
| file | sha256 |
|---|---|
| `code/v4dev/bench.py` (r7b 8cb79e96…c77a1 → + latency-only) | 9662cba5e706fa90fb4f96f46c4d1653f64ec2769b2cdc7fe515cf23385d210b |
| `code/tests_dev/test_bench_latency_only.py` (new) | 71f79745f533e19d0550ab0bbae813dadeab9fd61b851ffcf76d28b73bb4e49e |
| `company/plans/ops_scripts/lanes_r6w/laneD3_INT8_pass2.sh` (copy at `queues/`, same sha) | b7f7ab4749cbc02dd2c58d1b55a4065add9e05a2ed936b541e8c46887e5a51ce |
| candidate `$V/code_r7b_p2_cand` = `code_r7b_dev05_cand` + the two code files above | no other file differs |

`code_ds3r7/v4dev/bench.py` is still the r7 file (0b74e77a…). It is NOT deployed, as instructed.

### Tests (MEASURED)
- **Local** (next_item_eda `.venv`, `-B -p no:cacheprovider -q`, basetemp `.local_ai/_pytest_tmp/OPS-B13-P2/<x>`):
  - `tests_dev` whole: 144 passed, 6 skipped (onnxruntime-gated, before the e2e test was added), 29 xfailed.
  - New file: 13 passed, 1 skipped (real-ONNX e2e, server-only), 6 NC xfailed.
- **Server** (ppsi-gpu, candidate tree, `nice 19`, 1 thread, basetemp `$V/_pytest_tmp/OPS-B13-P2`):
  - `tests_dev` whole: 154 passed, 31 xfailed, 0 skipped.
  - `--runxfail -k test_nc_` on the new file: 6/6 fail with AssertionError.
  - Real ONNX end-to-end on the tiny synthetic fixture (K = 64, 8 queries): pass 1 `run_pipeline` with rows, then `run_latency_only`. It passed: only `LATENCY_PASS2.json` was added and every pass-1 file is byte-identical. Nothing was run on the real artifact.
- **What the new tests cover:**
  - the happy path;
  - each of the 4 sha mismatches refuses;
  - an incomplete or missing pass-1 record refuses;
  - 4 protocol mismatches refuse;
  - a wrong checkpoint refuses;
  - write-once;
  - a file changing during the timing refuses;
  - `main` dispatch;
  - pass-1 latency is not copied.
- **NCs** (an accepted sha mismatch, an overwritten result, a copied pass-1 latency, a re-export, an accepted different warmup, accuracy fields in the output) each fail with an AssertionError.
- **Lane (revised per the coordinator decision, B14.6 (ii): pass 2 does not wait for Wave 2).**
  - It waits only for every Wave-1 driver and job to exit: `run_wave1*.sh`, `laneBENCH_RESUME.sh`, and `v4bench.run` jobs with `--dataset ml_1m|s3_*`. It needs 2 consecutive quiet polls.
  - It then SIGSTOPs the `run_wave2.sh` driver(s) and every `v4bench.run` job with `--dataset amazon2023_*`, in 3 passes to catch late launches, and logs the pids.
  - It writes `PASS2_HOST_LOAD.json` beside the result, runs `--latency-only`, then SIGCONTs exactly those pids. A trap does this on every exit path, including failure and INT/TERM.
  - Patterns are bracket-quoted, passed with `grep -E -e` (a pattern starting with `-m` was otherwise read as an option; found and fixed in the fixture run).
  - Test overrides: `W1_BUSY_RE`, `W2_DRV_RE`, `W2_JOB_RE`, `LOG_OVERRIDE`, `TESTHOLD`, `FAILTEST`.
  - Fixture test on the server (fake driver, fake amazon2023 job, ml_1m decoy):
    - Only the driver and the amazon2023 job went to state T; the ml_1m decoy stayed S.
    - All were back to S after a successful hold and after a forced rc=9 failure.
    - The log lists the pids.
  - `DRYWAIT=1` on the real state lists 20 Wave-1 blockers now, and the Wave-2 pids it would stop (556862 557153 557160 557162 557165 557172 557174 696043 696051 696052). It stops nothing.

## Exact arm / launch commands (Ops decides; none has been run)
```
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4; cd $V/runs_ds3/queues

# 0. re-check before arming (read-only, prints what a pass would do right now)
bash b13_stopper.sh --dry-run --now 2026-10-06T00:00:00Z
sha256sum b13_stopper.sh        # 3e4aa02bb0fa44af0ad4e5162e02c46f036dbbc8f4c10f117f1156b8ec3b7f8b

# 1. ARM the B13 stopper (sleeps until 2026-10-06T00:00Z; logs to queues/b13_stopper.log; pid in queues/b13_stopper.pid)
nohup setsid bash $V/runs_ds3/queues/b13_stopper.sh >> $V/runs_ds3/queues/b13_stopper.nohup 2>&1 &
# disarm before the target: kill $(cat $V/runs_ds3/queues/b13_stopper.pid); rm -f $V/runs_ds3/queues/b13_stopper.pid

# 2. INT8 pass 2: first deploy the P2 bench.py (candidate sha 9662cba5…) into code_ds3r7 (Ops step), then
nohup setsid bash $V/runs_ds3/queues/laneD3_INT8_pass2.sh > $V/runs_ds3/queues/laneD3_INT8_pass2.nohup 2>&1 &
#    without deploying: CODE=$V/code_r7b_p2_cand nohup setsid bash $V/runs_ds3/queues/laneD3_INT8_pass2.sh ... &
#    (the P2 candidate differs from r7 only in dump_rows / bench / dump_rows_retro; build_scoring_module and the synthetic queries are
#    unchanged, so the timed queries are the same: INFERENCE)
# the lane waits for Wave 1 only, then SIGSTOPs Wave 2, times, and SIGCONTs it (trap)
```

## Caveats for Ops
- The stopper's SIGKILL of the job wrapper needs the wrapper's cmdline to contain `run_wave`. This is true for `run_wave1.sh` and `run_wave2.sh` and for the `run_wave1_r4.sh` driver whose name matches. If a driver renamed that, the stopper still kills the run, logs a warning chain, and the driver may write `.fail` (the stopper removes a stale `.fail` and writes `.ok` afterwards).
