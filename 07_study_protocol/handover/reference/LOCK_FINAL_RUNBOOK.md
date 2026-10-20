# Wave-1 benchmark LOCK and the single TEST look: Ops runbook (OPS-ENDGAME-2, 2026-10-02)

**Status:** a written procedure only. Nothing here has been run: no `lock`, no `final`, and no benchmark result read.

**Authority:**
- `contracts/PREG1_AMENDMENT_07.md` (sha 29beb5e3…, DL 341): B4, B7 R6 / R7, B10 Q4 / Q5, B11. Later sections supersede.
- `company/handoffs/BM-02.md`, Revision 3.
- The runner `$V/code_bench3`: `v4bench/run.py` fca7987f…, `final.py` ace8e4cd…, `selection.py` 002a31a3…, `data.py`
  9b15752b…. These four were MEASURED equal to `BM-02.sha256.txt` on 10-02.
- `company/plans/ops_scripts/bench/run_wave1.sh`, which stops before `lock` and `final`.

**Labels:** MEASURED (read on the server 10-02, names / statuses / logs only), RULE (code or amendment text), PLAN.

---------------------------------------------------------------------------------------------------------------------------
## 0. Rules that never bend
1. **One LOCK and one TEST look per dataset.**
   - Order: every run finished, then `lock`, then the DL row, then `final`.
   - Nothing is run, added or re-selected after the look (B4; R7).
   - A Wave-2 addendum can never touch a Wave-1 lock.
2. **The LOCK is write-once** (`make_lock` refuses when `LOCK.json` exists).
   - A refused `lock` writes nothing. Fix the cause and re-run the same command.
   - A LOCK that has been written is never edited or deleted.
3. **Ops never reads a value before the look.**
   - Allowed: directory names; the `status` key of `RESULT.json`; the decision files' decision keys
     (`FA_WARM_STAR`, `third_seed_2028_needed_for`, `winner_run`, `winner_cell`).
   - Never opened: `TEST_*`, `test/`, the holdout file, metric fields.
   - After the look, values are read only for the write-up (B5 / R8 claim rules).
4. **Device class CPU-FP32** (R9):
   - 16 threads per process, fixed (`run.py` refuses any other count);
   - `nice -n 15`, CUDA hidden;
   - `final`: workers × 16 threads ≤ 96 (`run.py` refuses more).

## 1. Common environment (server)
```bash
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
PY=/home/jovyan/venvs/ppsi-gpu/bin/python
cd $V/code_bench3
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONDONTWRITEBYTECODE=1
RB="nice -n 15 $PY -B -m v4bench.run"            # = run_wave1.sh RUNNER (cd $CODE, no PYTHONPATH)
# code identity: every v4bench / tests_bench file = BM-02.sha256.txt (uploaded beside the tree, sha-checked)
sha256sum -c $V/code_bench3/../_BM02_sha256.txt              # = company/handoffs/BM-02.sha256.txt uploaded + sha-checked; all OK or STOP
```
The four datasets are `s3_beauty`, `s3_sports`, `s3_toys` and `ml_1m` (`common.DATASETS`; Wave 1).
Run dirs are `runs_bench/<D>/<ARM>[__B<b>_L<ml>_D<dr>]_s<seed>/`.

## 2. Preconditions, per dataset (all must hold before `lock`)
| # | Check (value-free) | How | Rule |
|---|---|---|---|
| P1 | The Wave-1 driver has exited and **no job of this dataset failed or was skipped** | `pgrep -f "run_wave1.sh"` is empty; `ls runs_bench/_wave1_state/*.<D>*.fail` is empty | run_wave1 marks a failed job's dependents SKIPPED |
| P2 | All **Track-A** runs are finished: C_FULL, PRE, FA, FA_WARM, FA_WARM_FROZEN, FT_<STAR>, FP / PF / FA_Q8 _<SUF>, FA_1024_<SUF>, DP8_<SUF> at s2026 + s2027, and S_CAL_<SUF>_s2026 | §3 lists every dir; each `RESULT.json` status is `DONE_MAIN` (S_CAL: `DONE_S_CAL`) | B4; R7; `final._run_record` |
| P3 | `FA_WARM_DECISION.json` exists, and the DP / phase-2 arms follow its variant | file present; `FA_WARM_STAR` key only. STAR = FA_WARM_FROZEN → SUF = WARM_FROZEN (DP8_WARM_FROZEN); STAR = FA_WARM → DP_WARM rows (B9 M1d) | B2; B9 M1d; strict LOCK |
| P4 | **The seed-3 decision is taken, logged, and done** | `SEED3_DECISION.json` exists, plus a DL row (R6: before the LOCK, never after a look). For each label in `third_seed_2028_needed_for`, its `_s2028` run is DONE (§2.1) | R6; `make_lock` requires 2028 for those labels |
| P5 | **Track B is finished.** `trackb/WINNER_{C_FULL,PRE,FA,<STAR>}.json` exist. Each winner that is not the Track-A cell has its `<cell>_s2027` run DONE | `runs_bench/<D>/.win2027_<arm>.ok` × 4, and the `_s2027` dirs | R5; B8 O1; `make_lock` refuses an unlisted winner |
| P6 | **No INCOMPLETE run is selected or locked** | §3 prints every dir whose status is not DONE_*. Each must be (a) not a `winner_run` and (b) left out of `--runs`. List them in the DL row as INCOMPLETE (B10 Q4: `INCOMPLETE_WALL_CAP` is reported, never lockable or selectable). No dir may still be RUNNING | B10 Q4; `_run_record` refuses INCOMPLETE |
| P7 | The S record and decision files exist | `dp/S_RECORD_<STAR>.json`; `MANIFEST.json`. All pinned by the LOCK (`_pinned_files`) | R4; M3 |
| P8 | **Wave-1 cold holdout count = 0** | See below | B10 Q5 |
| P9 | The pinned contracts and the r6 tree digest are the registered ones | `make_lock` strict runs `recipe.check_pins(required=True)` + `selection.tree_digest()`; a mismatch refuses | R9 |

**P8, the cold count, cannot be measured before the look without reading the holdout.** It is enforced in code, inside the
look and before anything is written:
- `final._final_eval → get_view` raises `TEST refused: N cold holdout items on Wave-1 <D>` when the count is non-zero.
- This happens before `commit_look`, so `TEST_LOOK.json` is not written and the attempt is logged as `error`.

What Ops verifies:
1. Before `final`: `final.py` = ace8e4cd… and `data.py` = 9b15752b… (P0 sha check), so this guard is the reviewed code.
2. BM-00 / BM-02 MEASURED 0 cold items on all four Wave-1 sets (BM-02 §2), and the splits are byte-identical with the
   published release (DL 302).
3. After `final`: read only the count key, `TEST_RESULTS.json → n_cold_holdout_items_as_misses` (must be 0, §6).

### 2.1 Third-seed runs (only when `third_seed_2028_needed_for` is non-empty; PLAN, from the BM-02 command forms)
```bash
$RB run --dataset <D> --arm C_FULL --seed 2028                                   # if C_FULL is listed
$RB run --dataset <D> --arm PRE --seed 2028                                      # if FA_WARM* is listed: same-seed PRE
$RB run --dataset <D> --arm <STAR> --seed 2028 --pre-run $V/runs_bench/<D>/PRE_s2028   # STAR = FA_WARM_STAR
```
After these, the reported value of that arm is the 3-seed mean, with every seed listed (R6). No other arm gets a third seed.

## 3. The run list (one file per dataset; value-free)
Reads only the `status` key of each `RESULT.json`. Its output is names, plus a one-word status for the dirs left out.
```bash
D=s3_beauty                      # then s3_sports, s3_toys, ml_1m
B=$V/runs_bench/$D; L=$V/runs_bench/_lock_lists/$D.runs; mkdir -p $(dirname $L); [ ! -e $L ] || { echo "exists: $L"; exit 1; }
for d in $B/*/; do n=$(basename $d); case $n in SMOKE|dp|trackb) continue;; esac
  [ -f $d/RESULT.json ] || { echo "NO_RESULT $n (running or never started: STOP)"; continue; }
  st=$($PY -c "import json,sys; print(json.load(open(sys.argv[1])).get('status'))" $d/RESULT.json)
  case $st in DONE_MAIN|DONE_S_CAL) echo $n >> $L.tmp;; *) echo "NOT_LOCKABLE $n $st";; esac
done; sort $L.tmp > $L && rm $L.tmp && chmod a-w $L; wc -l $L; sha256sum $L
```
- Any `NO_RESULT` line means STOP: a run is still running or never started.
- A `NOT_LOCKABLE` line is allowed only for an INCOMPLETE Track-B cell that is not a winner (P6).
- `make_lock` checks the rest itself:
  - both seeds per Track-A label (+ 2028 where triggered);
  - seed 2026 per grid cell, and 2027 for the winners;
  - S_CAL at 2026;
  - every `RESULT.json` checkpoint sha256 (and the PF stores);
  - the decisions present.

## 4. LOCK (one command per dataset; write-once)
```bash
$RB lock --dataset s3_beauty --runs $(cat $V/runs_bench/_lock_lists/s3_beauty.runs)
$RB lock --dataset s3_sports --runs $(cat $V/runs_bench/_lock_lists/s3_sports.runs)
$RB lock --dataset s3_toys   --runs $(cat $V/runs_bench/_lock_lists/s3_toys.runs)
$RB lock --dataset ml_1m     --runs $(cat $V/runs_bench/_lock_lists/ml_1m.runs)
```
- Expected stdout: `LOCK written: runs_bench/<D>/LOCK.json (<n> runs)`. A refusal prints `REFUSED: …` (exit 2) and writes nothing.
- Never pass `--not-strict` (it is for tests only).
- After each lock:
  1. `chmod a-w runs_bench/<D>/LOCK.json`.
  2. Record its sha256 and the run-list sha in a **DL row**: "benchmark LOCK <D>", with the counts, the INCOMPLETE dirs
     left out, and the FA_WARM* / seed-3 / Track-B decisions by name.
  3. Fetch a local read-only copy.
- The LOCK binds the following, and `final` re-verifies every one of them:
  - format BM02_LOCK_1 and the data fingerprint;
  - `code_sha256` and `code_tree_digest`;
  - the pins: S4 tuning 9bca186e, S7 d67a0a85, S8 8b79d305, SELECTION_FT_C;
  - the pinned decision / winner / S-record / manifest files;
  - each run's RESULT and checkpoint sha256s;
  - `self_sha256`.

## 5. What Ops checks immediately before `--final` (per dataset)
1. `LOCK.json` exists, is read-only, and its sha256 = the DL row. `TEST_LOOK.json` does **not** exist (a second look
   refuses: `open_test_gate`).
2. `TEST_ATTEMPTS.jsonl` is absent, or holds only `start` / `error` pairs, with no `end`. If a previous attempt failed:
   - its error type / message is in a DL row;
   - the LOCK file is byte-identical to the DL-registered one (§7).
3. Code identity: `sha256sum -c BM-02.sha256.txt` passes. `final` itself re-checks `code_tree_digest`, the pins and every
   pinned file against the LOCK.
4. Resources:
   - ≥ 20 GB free disk (rank files + TEST_*.json are written after the look; a crash after `commit_look` spends the look);
   - no other ≥ 16-thread benchmark job on the CPU;
   - **the REES46 K16 INT8 bench (laneD3_INT8_r7) is not running**, so its single-thread latency stays quiet: check
     `pgrep -f "v4dev.bench|v4dev.dump_rows"` is empty.
5. The owner has been told: "single TEST look of <D> now", one dataset at a time.
6. The dataset is Wave 1. The cold-count guard applies to `common.DATASETS` only (P8).

## 6. The single TEST look (one command per dataset)
```bash
$RB final --dataset s3_beauty --workers 6        # 6 workers x 16 threads = 96 (the cap); FT arms use the workers
$RB final --dataset s3_sports --workers 6
$RB final --dataset s3_toys   --workers 6
$RB final --dataset ml_1m     --workers 6
```
**Order of `final`** (RULE, `run.py` main + `final.final_eval` / `_final_eval`; M1 / B11):
1. `run.py` runs `verify_lock` before the data are loaded. A refusal here prints `REFUSED`, exit 2, and is **not** in
   `TEST_ATTEMPTS.jsonl`: Ops records it in the DL row. It then refuses `workers × threads > 96`.
2. `final_eval` appends `start` to `TEST_ATTEMPTS.jsonl`.
3. Open the gate: `verify_lock` again, and a refusal if `TEST_LOOK.json` exists.
4. **Preflight:** every locked checkpoint scores its first / middle / last validation rows.
5. Only then is the holdout read: the cold-count guard runs, and every run is scored in memory.
6. `TEST_LOOK.json` is written atomically: **the look is taken here**.
7. The per-run `test/TEST_<rule>.json` + ranks and `TEST_RESULTS.json` are written (O_EXCL).
8. Append `end`. On any exception: append `error`, and re-raise.

**stdout** is counts only: `TEST done: <n> runs, <n> holdout users; populations ALL / BAND / OTHERS <n>`. The expected
holdout users are 22,311 / 35,539 / 19,365 / 6,038 (B1 / R7).

**After the look:**
1. Read only these:
   - the count keys `n_test_users`, `n_cold_holdout_items_as_misses` (**must be 0**) and `n_holdout_rows_unknown_user`
     from `TEST_RESULTS.json`;
   - the sha256 of `TEST_LOOK.json` / `TEST_RESULTS.json`;
   - the `TEST_ATTEMPTS.jsonl` lines.
2. Write a DL row: "benchmark TEST look <D>", with those shas, the attempt count and the user counts.
3. Make `test/`, `TEST_*.json` and `TEST_RESULTS.json` read-only, and fetch them.
4. Only then are the values used, for the write-up: B5 comparisons; R8 claim rule (|Δ NDCG@10| > 0.005 and every seed on
   the same side, else "on par"); Track A and Track B never mixed.

## 7. The TEST_ATTEMPTS rule (B11; `final.final_eval`)
- **Every** `final` invocation appends to `runs_bench/<D>/TEST_ATTEMPTS.jsonl` (append-only):
  - `{"event": "start", utc, workers, threads}` first;
  - then either `{"event": "end", utc, n_runs, n_test_users}` or `{"event": "error", utc, type, message}`.
- **The single TEST look is the attempt whose `TEST_LOOK.json` was written.** After that file exists, every later
  `final` refuses at the gate. That refusal is still logged as `start` / `error`, and it is disclosed but changes nothing.
- **An attempt that ends in `error` before `TEST_LOOK.json` exists is a crashed attempt.** That covers a gate refusal,
  a preflight failure, the cold-count guard, and an OOM before the commit. A refusal in `run.py`'s first `verify_lock` /
  thread check comes before `start` and is not logged in the file; Ops logs it in the DL row the same way. Such an attempt:
  - is disclosed in a DL row (type + message, never a value);
  - **may be retried only with the identical LOCK** (same file, same sha256 as its DL row; `verify_lock` also re-checks
    every pin, the tree digest and each run's checkpoint shas);
  - is never retried with a new or edited LOCK, an added / removed run, or a different code tree.
- **An error after `TEST_LOOK.json` is written** (e.g. disk full while writing the result files) means the look **is
  spent**. No retry. The partial outputs and the error are disclosed, and Ops / Fable decide in a DL row how they are
  reported.
- The thesis lists, per dataset, the number of attempts and the reason for each error.

## 8. Blockers MEASURED on 10-02 ~11:40Z (no value read; job logs and dir names only)
1. **The seed-2027 warm DP chain refuses on s3_beauty and s3_sports.**
   - The jobs `fa1024.<D>.2027` and `dp8.<D>.2027` are FAILED.
   - The log says: `REFUSED: dp/S_RECORD_FA_WARM_FROZEN.json: S record init_digest '<θ_pre s2026>' != this run's '<θ_pre
     s2027>'`.
   - Cause: the DAG writes the S record once (S_CAL at seed 2026 from `PRE_s2026`), but runs the 2027 FA_1024 / DP8 from
     `PRE_s2027`. Meanwhile the runner requires a warm start from the same-seed PRE (BM-02 Revision 2 minors), and it
     requires the S record to have been built from that start.
   - Effect: the strict LOCK refuses on all four datasets, because FA_1024_<SUF> and DP8_<SUF> lack seed 2027. s3_toys and
     ml_1m will hit the same refusal.
   - **Needs an Ops + Fable decision (DL row) and a BM-02 fix before any `lock`.** The two candidate fixes:
     - (a) one S_CAL per seed (`S_CAL_<SUF>_s2027` from `PRE_s2027` → a per-seed S record);
     - (b) the REES46 rule, one θ_pre for both DP seeds (as REES46's warm DP chain uses PRE s2026 for both).
   - Registration point: amendment 07 R4 / B9 M1c. Cost: a short re-run of 2 jobs per dataset.
   - Until then, P1 / P2 fail for these jobs, and the lock is not attempted.
2. **Wave 1 is still running.** Driver alive; 116 jobs ok, 4 failed (the 4 above), as of 10-02 ~11:40Z. ml_1m has not
   started (MANIFEST + SMOKE only). P1 holds only after the driver exits.
3. s3_beauty already holds two Track-B winner reruns at s2027 (`C_FULL__B24_L50_D0.5_s2027`, `PRE__B96_L50_D0.5_s2027`)
   and all four WINNER files. The FA / FA_WARM_FROZEN winners have no `_s2027` cell, which means the Track-A cell won
   (PLAN reading of `win2027`). Confirm through `.win2027_*.ok` at P5.
