# Benchmark LOCK and the single TEST look (Wave 1 + Wave 2): Ops runbook (OPS-ENDGAME-2, 2026-10-02; revised for B12-B16, 2026-10-04)

**Status:** a written procedure only. Nothing here has been run: no `lock`, no `final`, and no benchmark result read.
**Revision 2026-10-04 (B12 / B14 / B15 / B16):** the previous text (kept as `LOCK_FINAL_RUNBOOK.pre_B16.md`) named the code tree
`code_bench3` for every dataset and `runs_bench` for ML-1M. That is wrong now. Sections changed by an addendum carry its tag.

**Authority:**
- `contracts/PREG1_AMENDMENT_07.md` (sha 29beb5e3…, DL 341): B4, B7 R6 / R7, B10 Q4 / Q5, B11. Later sections supersede.
- `company/handoffs/BM-02.md`, Revision 3.
- **Code trees (one per device class; a LOCK pins the tree that trained it):**
  - `$V/code_bench3`: the original BM-02 runner (`run.py` fca7987f…, `final.py` ace8e4cd…, `selection.py` 002a31a3…, `data.py`
    9b15752b…; MEASURED equal to `BM-02.sha256.txt` on 10-02). **Not used for any LOCK / final any more.** It trained the early S3
    jobs; the r6 modules are unchanged, so `code_tree_digest` is the same value in every tree (BM-03 section 2).
  - **[B12] `$V/code_bench4`** = code_bench3 + the B12 delta (REV-BM-02 delta 2, DL 361). CPU-FP32. **Used for the S3 LOCKs and
    finals, and for all of Wave 2.**
  - **[B16] `$V/code_bench5b`** = code_bench5 (BM-03, device plumbing) + the BM-05 delta (`final --device`). **Used for ML-1M only**
    (training, LOCK and final). code_bench4 / code_bench5 are untouched.
- `company/plans/ops_scripts/bench/run_wave1.sh`, which stops before `lock` and `final`.
- **[B12]** `contracts/PREG1_AMENDMENT_07_ADDENDUM_B12.md` (DL 357): S_CAL per PRE seed; tree `code_bench4`.
- **[B13 / B15]** B13 (ML-1M clock backstop, DL 368) was **rescinded by B15 (DL 401)**. `b13_stopper.sh` was disarmed and is never
  run. No cell is cut by the clock; a cell is INCOMPLETE only by a DL row with a mechanical cause (B15.2, B10 Q4).
- **[B14]** `..._ADDENDUM_B14.md` (DL 377; Fable `company/reviews/FABLE_DECISION_WAVE2_20261003.md`): Wave 2 = Track A only,
  CPU-FP32, `code_bench4`, driver `run_wave2.sh`.
- **[B16]** `..._ADDENDUM_B16.md` (sha 85ba5e18..., DL 433): ML-1M on the A40 GPU device class; tree `code_bench5b`; driver
  `run_ml1m_gpu.sh` (1a623877...); `final --device cuda`. Handoffs BM-03 / BM-04 / BM-05.

**Labels:** MEASURED (read on the server 10-02, names / statuses / logs only), RULE (code or amendment text), PLAN.

---------------------------------------------------------------------------------------------------------------------------
## 0. Rules that never bend
1. **One LOCK and one TEST look per dataset.**
   - Order: every run finished, then `lock`, then the DL row, then `final`.
   - Nothing is run, added or re-selected after the look (B4; R7).
   - A Wave-2 addendum can never touch a Wave-1 lock.
   - **[B16.3 / B16.4] One device class per dataset, from the first run to the look.** A mix is refused by the runner (resume,
     theta_pre, S record, FT base, LOCK, trackb-select, decide-warm, seed3, `final`). No device switch inside a dataset
     (B14.6 for Wave 2).
2. **The LOCK is write-once** (`make_lock` refuses when `LOCK.json` exists).
   - A refused `lock` writes nothing. Fix the cause and re-run the same command.
   - A LOCK that has been written is never edited or deleted.
3. **Ops never reads a value before the look.**
   - Allowed: directory names; the `status` key of `RESULT.json`; the decision files' decision keys
     (`FA_WARM_STAR`, `third_seed_2028_needed_for`, `winner_run`, `winner_cell`).
   - Never opened: `TEST_*`, `test/`, the holdout file, metric fields.
   - After the look, values are read only for the write-up (B5 / R8 claim rules).
4. **Device class, by dataset** (R9; B14.6; B16.1):
   | Dataset | Device class | Code tree | Runs root | Driver |
   |---|---|---|---|---|
   | `s3_beauty`, `s3_sports`, `s3_toys` | CPU-FP32 | `code_bench4` | `runs_bench` | `run_wave1.sh` (finished) |
   | `amazon2023_video_games`, `amazon2023_baby_products` (Wave 2, **[B14]**) | CPU-FP32 | `code_bench4` | `runs_bench` | `run_wave2.sh` |
   | `ml_1m` **[B16]** | **NVIDIA_A40-FP32-DET** (`--device cuda`, strict deterministic algorithms, CUBLAS_WORKSPACE_CONFIG, cuDNN deterministic, TF32 off, matmul precision "highest") | `code_bench5b` | **`runs_bench_gpu`** | `run_ml1m_gpu.sh` |
   - CPU-FP32: 16 threads per process, fixed (`run.py` refuses any other count); `nice -n 15`, CUDA hidden;
     `final`: workers × 16 threads ≤ 96 (`run.py` refuses more).
   - **ML-1M on GPU:** CUDA is not hidden; `--device cuda --gpu G` goes on every `run` / `final`; `--threads 16` as in BM-05.
   - **[B16.3] Sealed / superseded:** the 46 CPU ML-1M run dirs under `runs_bench/ml_1m` carry `SUPERSEDED_DEVICE.json`, are
     read-only, and the CPU decision records are SUPERSEDED_DEVICE. State file `_wave1_state/ML1M_SUPERSEDED_DEVICE.json` stops
     `--resume` from relaunching them. Never opened, never lockable, never a winner, never compared. **Never lock `ml_1m` in
     `runs_bench`.** A CPU fallback exists only by a DL row with a mechanical cause before any ML-1M result is opened.
5. **[Order; DL 442 / DL 445] Benchmark TEST looks are on hold.** The owner holds every TEST, including the S3, Wave-2 and ML-1M
   looks, until the **REES46 Phase-2 sitting**. Until the owner releases them:
   - No `final` is run for any benchmark dataset. Preparing everything up to a LOCK is allowed once its preconditions hold
     (a `lock` is not a look; it is write-once, so run it only when the run list is final).
   - `H_test.sh` (the REES46 TEST look) is permission-locked on the server (chmod a-rwx, content sha 11c04583...,
     `HOLD_TEST_LOOK.txt` written). It is restored with `chmod 555` **only at the sitting**, and every H step needs `--i-confirm`.
     A benchmark `final` is never started while an H step or a Phase-2 lane runs on the same GPU (B16.5; BM-05 section 11 E3),
     and H does not start while a benchmark `final` runs. `server_fill_watchdog.sh` (DL 463) pauses the ML-1M / Wave-2
     fillers once H runs.
   - The order inside the sitting is the owner's call at the time. No addendum makes a benchmark look a prerequisite of H.

## 1. Common environment (server)
```bash
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
PY=/home/jovyan/venvs/ppsi-gpu/bin/python
RB="nice -n 15 $PY -B -m v4bench.run"            # = the drivers' RUNNER (cd $CODE, no PYTHONPATH)
```
**CPU-FP32 block** (S3 datasets and Wave 2) **[B12 / B14]**:
```bash
cd $V/code_bench4
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONDONTWRITEBYTECODE=1
RUNS=$V/runs_bench
```
**GPU block** (ML-1M only) **[B16]**:
```bash
cd $V/code_bench5b
export PYTHONDONTWRITEBYTECODE=1                  # CUDA NOT hidden; --device cuda / --gpu go on every command
RUNS=$V/runs_bench_gpu
```
**Code identity** (check before each lock / final, per tree):
- `code_bench4`: the BM-02 file list changed only by the B12 delta (`final.py`, `fl_engine.py`, `recipe.py`; DL 361). The reference
  is the digest set of the B12 deploy row, not `BM-02.sha256.txt` (that file describes code_bench3 and will not match the three
  changed files). The LOCK's `code_sha256` / `code_tree_digest` is the binding check.
- `code_bench5b`: `runs_ds3/deploy_bench5/NEW_FILES.sha256` (11 files, code_bench5) and `runs_ds3/deploy_bench5b/DELTA.sha256`
  (BM-05 delta, tar 17fc7c90...); every file must match.

Datasets and run dirs:
- Wave 1: `s3_beauty`, `s3_sports`, `s3_toys` (CPU) and `ml_1m` (GPU).
- **[B14]** Wave 2: `amazon2023_video_games`, `amazon2023_baby_products` (CPU). The `bench_proc_w2` data shas are pinned in the
  LOCK. Beauty_and_Personal_Care is DROPPED ("converted, not run").
- Run dirs are `$RUNS/<D>/<ARM>[__B<b>_L<ml>_D<dr>]_s<seed>/`; for ML-1M `$RUNS` = `runs_bench_gpu`.

## 2. Preconditions, per dataset (all must hold before `lock`)
Read `runs_bench/` below as `$RUNS/` (`runs_bench_gpu` for ML-1M). The state dir is `_wave1_state` (S3 and Wave 2; `run_wave2.sh`
shares it) or `runs_bench_gpu/_gpu_state` (ML-1M).
| # | Check (value-free) | How | Rule |
|---|---|---|---|
| P1 | The driver of this dataset has exited and **no job of this dataset failed or was skipped** | S3: `pgrep -f "run_wave1"` empty. Wave 2: `pgrep -f run_wave2.sh` empty (one driver per dataset, DL 360). **[B16]** ML-1M: `pgrep -f run_ml1m_gpu.sh` empty. A driver that yielded (`$RUNS/_gpu_state/YIELD`) or was SIGSTOPped (`runs_bench_gpu/ML1M_PAUSED_PGID.txt`) has NOT finished: `rm` the YIELD file and `run_ml1m_gpu.sh --resume` / `kill -CONT` first. Then `ls $RUNS/_wave1_state/*.<D>*.fail` (S3, Wave 2) or `ls $RUNS/_gpu_state/*.fail` (ML-1M) is empty | the drivers mark a failed job's dependents SKIPPED |
| P2 | All **Track-A** runs are finished: C_FULL, PRE, FA, FA_WARM, FA_WARM_FROZEN, FT_<STAR>, FP / PF / FA_Q8 _<SUF>, FA_1024_<SUF>, DP8_<SUF> at s2026 + s2027, and **[B12]** S_CAL_<SUF>_s2026 **and** S_CAL_<SUF>_s2027 | §3 lists every dir; each `RESULT.json` status is `DONE_MAIN` (S_CAL: `DONE_S_CAL`) | B4; R7; `final._run_record` |
| P3 | `FA_WARM_DECISION.json` exists, and the DP / phase-2 arms follow its variant | file present; `FA_WARM_STAR` key only. STAR = FA_WARM_FROZEN → SUF = WARM_FROZEN (DP8_WARM_FROZEN); STAR = FA_WARM → DP_WARM rows (B9 M1d) | B2; B9 M1d; strict LOCK |
| P4 | **The seed-3 decision is taken, logged, and done** | `SEED3_DECISION.json` exists, plus a DL row (R6: before the LOCK, never after a look). For each label in `third_seed_2028_needed_for`, its `_s2028` run is DONE (§2.1) | R6; `make_lock` requires 2028 for those labels |
| P5 | **Track B is finished.** `trackb/WINNER_{C_FULL,PRE,FA,<STAR>}.json` exist. Each winner that is not the Track-A cell has its `<cell>_s2027` run DONE | `runs_bench/<D>/.win2027_<arm>.ok` × 4, and the `_s2027` dirs | R5; B8 O1; `make_lock` refuses an unlisted winner |
| P6 | **No INCOMPLETE run is selected or locked.** **[B15]** For ML-1M every Track-B cell must have a RESULT.json (B13 is rescinded: no cell is dropped by the clock; a mechanically failed cell is resumed with identical hashes first and becomes INCOMPLETE only by a DL row) | §3 prints every dir whose status is not DONE_*. Each must be (a) not a `winner_run` and (b) left out of `--runs`. List them in the DL row as INCOMPLETE (B10 Q4: `INCOMPLETE_WALL_CAP` is reported, never lockable or selectable). No dir may still be RUNNING | B10 Q4; `_run_record` refuses INCOMPLETE |
| P7 | The S records and decision files exist | **[B12]** `dp/S_RECORD_<STAR>_s2026.json` and `dp/S_RECORD_<STAR>_s2027.json` (a pre-B12 `S_RECORD_<STAR>.json` is accepted for s2026 only); `MANIFEST.json`. All pinned by the LOCK (`_pinned_files`) | R4; B12; M3 |
| P8 | **Wave-1 cold holdout count = 0** | See below | B10 Q5 |
| P9 | The pinned contracts and the r6 tree digest are the registered ones | `make_lock` strict runs `recipe.check_pins(required=True)` + `selection.tree_digest()`; a mismatch refuses | R9 |
| P10 | **[B16]** ML-1M only: every record under `runs_bench_gpu/ml_1m` has device class NVIDIA_A40-FP32-DET, no run comes from `runs_bench/ml_1m`, and the CPU set stays sealed | `make_lock` / the runner refuse a mix; Ops checks that the lock list is built from `runs_bench_gpu` (section 3) | B16.1 / B16.3 |
| P11 | **[B14]** Wave 2 only: Track A only. There is no `trackb/`, no `.win2027_*.ok`, no `WINNER_*` (P5 does not apply); both manifests record the cold-item counts (69 / 59) | file listing | B14.2 / B14.3 |

**P8, the cold count, cannot be measured before the look without reading the holdout.** **[B14.2]** It applies to the four Wave-1
datasets only; for Wave 2 the cold items (69 / 59) count as misses and are printed, never refused. It is enforced in code, inside the
look and before anything is written:
- `final._final_eval → get_view` raises `TEST refused: N cold holdout items on Wave-1 <D>` when the count is non-zero.
- This happens before `commit_look`, so `TEST_LOOK.json` is not written and the attempt is logged as `error`.

What Ops verifies:
1. Before `final`: the tree's `final.py` / `data.py` pass the section 1 code-identity check, so this guard is the reviewed code (`data.py` is unchanged from code_bench3 in all trees).
2. BM-00 / BM-02 MEASURED 0 cold items on all four Wave-1 sets (BM-02 §2), and the splits are byte-identical with the
   published release (DL 302).
3. After `final`: read only the count key, `TEST_RESULTS.json → n_cold_holdout_items_as_misses` (Wave 1: must be 0, section 6; Wave 2: it must equal the manifest's 69 / 59, disclosed).

### 2.1 Third-seed runs (only when `third_seed_2028_needed_for` is non-empty; PLAN, from the BM-02 command forms; B14.4 applies the R6 rule to Wave 2)
For ML-1M **[B16]** every command also carries `--device cuda --gpu G --runs-root $RUNS`, and `--pre-run` points into
`runs_bench_gpu` (the commands below are the CPU form; the GPU driver builds the same commands with those flags, see `job()` in
`run_ml1m_gpu.sh`).
```bash
$RB run --dataset <D> --arm C_FULL --seed 2028                                   # if C_FULL is listed
$RB run --dataset <D> --arm PRE --seed 2028                                      # if FA_WARM* is listed: same-seed PRE
$RB run --dataset <D> --arm <STAR> --seed 2028 --pre-run $RUNS/<D>/PRE_s2028   # STAR = FA_WARM_STAR
```
After these, the reported value of that arm is the 3-seed mean, with every seed listed (R6). No other arm gets a third seed.

## 3. The run list (one file per dataset; value-free; `$RUNS` per section 1)
Reads only the `status` key of each `RESULT.json`. Its output is names, plus a one-word status for the dirs left out.
```bash
D=s3_beauty    # then s3_sports, s3_toys; amazon2023_video_games, amazon2023_baby_products [B14]; ml_1m with RUNS=runs_bench_gpu [B16]
B=$RUNS/$D; L=$RUNS/_lock_lists/$D.runs; mkdir -p $(dirname $L); [ ! -e $L ] || { echo "exists: $L"; exit 1; }
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
Run each block from the matching `cd` / `RUNS` of section 1.
```bash
# CPU-FP32, code_bench4, runs_bench   (S3: B12 tree; Wave 2: B14)
$RB lock --dataset s3_beauty --runs $(cat $RUNS/_lock_lists/s3_beauty.runs)
$RB lock --dataset s3_sports --runs $(cat $RUNS/_lock_lists/s3_sports.runs)
$RB lock --dataset s3_toys   --runs $(cat $RUNS/_lock_lists/s3_toys.runs)
$RB lock --dataset amazon2023_video_games   --runs $(cat $RUNS/_lock_lists/amazon2023_video_games.runs)     # B14.7
$RB lock --dataset amazon2023_baby_products --runs $(cat $RUNS/_lock_lists/amazon2023_baby_products.runs)

# GPU, code_bench5b, runs_bench_gpu   (ML-1M only; B16.1 / B16.4)
cd $V/code_bench5b; RUNS=$V/runs_bench_gpu
$RB lock --dataset ml_1m --runs-root $RUNS --runs $(cat $RUNS/_lock_lists/ml_1m.runs)
```
- **[B16]** The ML-1M `lock` carries `--runs-root` like every non-`run` subcommand of `run_ml1m_gpu.sh`; the LOCK records the
  device class (`make_lock` writes `device_class`, BM-03). Without `--runs-root` it would read `runs_bench`, where the CPU set is
  sealed. The `lock` flags are not shown in any script: confirm with `$RB lock --help` on the server before the first use.
- **[B16.6] ML-1M LOCK disclosure.** The DL row and the LOCK notes record: the B16 addendum sha (85ba5e18...), the evidence shas
  (deploy_bench5, determinism digests), the sentence "device class switched before any ML-1M value was read, on schedule
  grounds; the CPU runs are sealed SUPERSEDED_DEVICE and were never opened", **[B15.4]** the sentence "B13 rescinded before its
  trigger on schedule grounds; no value read", and any INCOMPLETE cell.
- **[B14.7] Wave-2 LOCK** needs every Track-A run (and any third seed) with a RUN_SUMMARY. A stopped dataset has no LOCK and no
  look ("converted, not run"); no partial Track-A table is ever locked.
- Expected stdout: `LOCK written: runs_bench/<D>/LOCK.json (<n> runs)`. A refusal prints `REFUSED: …` (exit 2) and writes nothing.
- Never pass `--not-strict` (it is for tests only).
- After each lock:
  1. `chmod a-w runs_bench/<D>/LOCK.json`.
  2. Record its sha256 and the run-list sha in a **DL row**: "benchmark LOCK <D>", with the counts, the INCOMPLETE dirs
     left out, and the FA_WARM* / seed-3 / Track-B decisions by name.
  3. Fetch a local read-only copy.
- The LOCK binds the following, and `final` re-verifies every one of them:
  - format BM02_LOCK_1, the data fingerprint, and **[B16]** the device class;
  - `code_sha256` and `code_tree_digest`;
  - the pins: S4 tuning 9bca186e, S7 d67a0a85, S8 8b79d305, SELECTION_FT_C;
  - the pinned decision / winner / **per-seed S-record [B12]** / manifest files;
  - each run's RESULT and checkpoint sha256s;
  - `self_sha256`.

## 5. What Ops checks immediately before `--final` (per dataset)
1. `LOCK.json` exists, is read-only, and its sha256 = the DL row. `TEST_LOOK.json` does **not** exist (a second look
   refuses: `open_test_gate`).
2. `TEST_ATTEMPTS.jsonl` is absent, or holds only `start` / `error` pairs, with no `end`. If a previous attempt failed:
   - its error type / message is in a DL row;
   - the LOCK file is byte-identical to the DL-registered one (§7).
3. Code identity: the section 1 check for the dataset's tree passes. `final` itself re-checks `code_tree_digest`, the pins and every
   pinned file against the LOCK.
4. Resources:
   - ≥ 20 GB free disk (rank files + TEST_*.json are written after the look; a crash after `commit_look` spends the look);
   - no other ≥ 16-thread benchmark job on the CPU;
   - **the REES46 K16 INT8 bench (laneD3_INT8_r7) is not running**, so its single-thread latency stays quiet: check
     `pgrep -f "v4dev.bench|v4dev.dump_rows"` is empty.
5. The owner has been told: "single TEST look of <D> now", one dataset at a time. **Per DL 445 the benchmark TEST looks are on
   hold: the owner's release is quoted in a DL row first. Until then nothing in section 6 is run (section 0 rule 5).**
6. Wave 1: the cold-count guard applies (P8). Wave 2: it does not (B14.2).
7. **[B16.5] ML-1M:** no H step, Phase-2 lane (`laneP2*.sh`) or other GPU job on the GPU chosen with `--gpu` (BM-05 E3: its GPU
   test must not share a GPU with an H step either); the ML-1M driver has exited (not merely paused).

## 6. The single TEST look (one command per dataset)
**On hold (DL 445): do not run until the owner releases it (section 0 rule 5).**
```bash
# CPU-FP32, from code_bench4: S3 and Wave 2
$RB final --dataset s3_beauty --workers 6        # 6 workers x 16 threads = 96 (the cap); FT arms use the workers
$RB final --dataset s3_sports --workers 6
$RB final --dataset s3_toys   --workers 6
$RB final --dataset amazon2023_video_games   --workers 6     # B14.7 (CPU, as its LOCK)
$RB final --dataset amazon2023_baby_products --workers 6

# GPU, from code_bench5b: ML-1M only [B16.4; BM-05 section 5c]
$RB final --dataset ml_1m --runs-root $V/runs_bench_gpu --device cuda --gpu 1 --workers 1 --threads 16
```
- **[B16.4]** `--device cuda` is mandatory for ML-1M. Without it `final` REFUSES ("the LOCK's device class is
  NVIDIA_A40-FP32-DET, this look runs on CPU-FP32") and logs no attempt. A CPU-written LOCK (S3, Wave 2) can never be finalised
  with cuda. `--workers W` starts W fine-tune processes on the one visible GPU; BM-05 uses 1. `--gpu` is the A40 index that
  section 5 item 7 verified free (BM-05 used 1).
- code_bench5b with the default `--device cpu` gives the same bytes as code_bench4 (BM-05), but code_bench4 is the tree the S3
  and Wave-2 LOCKs pin.

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
holdout users are 22,311 / 35,539 / 19,365 / 6,038 for s3_beauty / s3_sports / s3_toys / ml_1m (B1 / R7), and **[B14.4]** 94,762 / 150,777 for Video_Games / Baby_Products (all users, cold-item users included as misses).

**After the look:**
1. Read only these:
   - the count keys `n_test_users`, `n_cold_holdout_items_as_misses` (**Wave 1: must be 0**; Wave 2: the manifest's 69 / 59, disclosed) and `n_holdout_rows_unknown_user`
     from `TEST_RESULTS.json`;
   - the sha256 of `TEST_LOOK.json` / `TEST_RESULTS.json`;
   - the `TEST_ATTEMPTS.jsonl` lines.
2. Write a DL row: "benchmark TEST look <D>", with those shas, the attempt count and the user counts (ML-1M: also the device class and the B16.6 sentence).
3. Make `test/`, `TEST_*.json` and `TEST_RESULTS.json` read-only, and fetch them.
4. Only then are the values used, for the write-up: B5 comparisons; R8 claim rule (|Δ NDCG@10| > 0.005 and every seed on
   the same side, else "on par"); Track A and Track B never mixed. **[B14.5]** Wave 2 is ours-vs-ours and DESCRIPTIVE: no published row, no "on par", never pooled with Wave 1 or REES46. **[B16.6]** The ML-1M device-class sentence also appears in the Track-B table and the cross-dataset tables.

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
- **[B16]** An ML-1M retry keeps the device class and the tree (`code_bench5b`, `--device cuda`); a CPU retry is never allowed.

## 8. History: the 10-02 blockers (RESOLVED; kept for the record)
Blocker 1 was fixed by **B12** (option (a), per-seed S_CAL; code_bench4); all three S3 datasets finished every job (DL 369).
ML-1M's CPU jobs were sealed SUPERSEDED_DEVICE by B16.3 (DL 433) and ML-1M was re-run from scratch on the GPU (driver
`run_ml1m_gpu.sh`, runs root `runs_bench_gpu`). The text below is the 10-02 MEASURED state, unchanged.

### 8.0 Blockers MEASURED on 10-02 ~11:40Z (no value read; job logs and dir names only)
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
