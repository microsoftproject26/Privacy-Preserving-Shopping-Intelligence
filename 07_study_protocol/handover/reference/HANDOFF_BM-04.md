# BM-04 - ML-1M GPU driver (run_ml1m_gpu.sh)

Status: script + tests done locally; nothing launched. MUST NOT run (--launch / --resume) before B16 is registered. Plan mode is safe.

Files
- NEW company/plans/ops_scripts/bench/run_ml1m_gpu.sh (copy of run_wave1.sh, which is untouched: mtime 10-02 14:48, test_wave1_script.py 5/5 pass unchanged)
- NEW code/tests_bench/test_ml1m_gpu_script.py (7 tests, all pass, ~2 min; test_wave1_script.py 5 passed)

## Diff vs run_wave1.sh (183 diff lines; summary)
- Header documents the GPU version (B16, to be registered, no run before registration).
- CODE default $V/code_bench5; RUNS=$V/runs_bench_gpu (env GPU_RUNS exists only so the refusal can be tested); LOGS=$RUNS/logs; STATE=$RUNS/_gpu_state; log files gpu_<job>.log.
- DATASETS fixed ml_1m (any other refused). Refusals (exit 2): RUNS == $V/runs_bench, CODE ending code_bench4, MAXSLOTS > 8 or < 2, --slots-per-gpu not 1..4, bad GPU id.
- Options added: --gpu N | --gpus "0 1", --slots-per-gpu N (default 4), --max-slots (default 4 x GPUs, max 8). Removed: the 6-slot/96-thread rule.
- job(): every `run` command gets `--device cuda --gpu G --runs-root $RUNS` (G = GPUS[jobindex mod nGPU], deterministic); win2027 jobs pass the same flags via RUNFLAGS (win2027 now calls `$RUNNER run $RUNFLAGS ...`); non-run subcommands (manifest, decide-warm, trackb-select, seed3) get only `--runs-root $RUNS` (no --device). run_wave1 passed no --runs-root and relied on the runner default runs_bench; the GPU driver always passes the fresh root.
- CUDA_VISIBLE_DEVICES="" export removed (the runner sets visibility itself: BM-03). nice 15, OMP/MKL 16 threads kept.
- Plan mode, write-once DONE-file skip, `--resume` semantics, per-job logs, failure -> dependents SKIPPED: unchanged.
- Note: FT (w=2, --workers 2) also receives --device cuda as specified; the runner decides what FT does on cuda (BM-05 covers `final`).

## Plan mode, ML-1M
73 jobs (manifest 1 + Track A + decide-warm + FT + phase 2 + S_CAL/FA_1024/DP8 + seed3 + Track B grid + 4 trackb-select + 4 win2027), total weight 75 (2 FT jobs weigh 2). Order is topological; stops before lock/final.

## Fable section 11 checklist -> where satisfied
| Item | Where / how | Test |
|---|---|---|
| DAG = registered ML-1M graph | same build_dataset as run_wave1 (Track A -> decide-warm -> seed3; Track B ml_1m grid max_len {50,200}, C_FULL budgets {6,24}; trackb-select; winners s2027; no lock/final) | test_plan_is_default... (73 jobs, topological, keys) |
| (1) Yield | `want_yield`: file $RUNS/_gpu_state/YIELD, or a process matching $YIELD_RE (default `[H]_test\.sh\|[l]aneP2...\.sh`; Ops sets the exact Phase-2 lane pattern via env). Checked before every launch; launches nothing new, running jobs finish, exit 0 "driver YIELDED", pending jobs not marked failed; `--resume` continues (the runner's identical-hash resume rule applies) | test_yield_file_at_start_and_between_jobs_then_resume (YIELD appears mid-run; all started jobs finish; resume completes with no job repeated); test_yield_to_a_running_h_or_lane_process |
| (2) Exactly one driver | atomic `mkdir $STATE/DRIVER.lock` + pid (stale pid taken over, released on exit); refuses if a run_wave1.sh covering ml_1m (no --datasets, or listing ml_1m) is running, or a CPU `v4bench.run run ... --dataset ml_1m` outside runs_bench_gpu is active | test_exactly_one_ml1m_driver |
| (3) Explicit GPU slots | --slots-per-gpu (<=4) enforced per GPU via used_gpu(), plus total <= --max-slots <= 8 | test_fake_launch... (per-GPU peak <= 2 with --slots-per-gpu 2; 5 refused) |
| (4) Write-once state only under runs_bench_gpu | all state/logs/runs under $RUNS; runs_bench refused; every command carries --runs-root $RUNS; DONE-file skip | test_fake_launch... (V holds only code_bench5 + runs_bench_gpu; second launch starts nothing); test_run_lines_carry... (no runs_bench path in any command) |
| (5) Plan mode on real state | plan never writes or locks, never refuses on running drivers; prints a read-only state line (ok/fail markers, YIELD present/absent) | test_fake_launch... (plan after a full run: same file list and mtimes) |
| (6) Value-free logs | the driver prints only job names, timestamps, rc, counts; it never reads or prints a metric file (job stdout goes to the runner's own log) | test_fake_launch... (no metric words or 0.dddd values in driver output/logs) |
| Resume | `--resume` = run --resume for interrupted runs; finished runs skipped | inherited; yield test covers resume |

Limitations: YIELD_RE default is a guess at the Phase-2 lane name (laneP2*.sh); Ops must set it (or just touch the YIELD file). The laneD3_INT8_pass2.sh W1_BUSY_RE needs `run_ml1m_gpu.sh` added (Fable E3; separate DL row, not done here). Non-run jobs consume a slot on their nominal GPU but use none. psl() falls back to `ps -ef` (Git Bash tests only); the server uses procps `ps -eo`.

## Tests (run locally with Git Bash, fake runner; no real data)
- tests_bench/test_ml1m_gpu_script.py: 7 passed (plan/DAG 73 jobs; device/gpu/root flags and non-run lines; refusals: other dataset, runs_bench root, code_bench4, slots 9, bad gpu; fake launch; yield x2; single driver).
- tests_bench/test_wave1_script.py: 5 passed (unchanged).

## Exact server launch (after B16 registered, Ops plan read, CPU ML-1M branch dead per E2)
```
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
bash $V/company/plans/ops_scripts/bench/run_ml1m_gpu.sh --plan --gpus "0 1"            # Ops reads this first
nohup bash $V/company/plans/ops_scripts/bench/run_ml1m_gpu.sh --launch --gpus "0 1" > $V/runs_bench_gpu_driver.out 2>&1 &
```
(`--gpus "0 1"` = 2 A40s, 8 slots; single GPU: `--gpu 0`, 4 slots. Stop politely: `touch $V/runs_bench_gpu/_gpu_state/YIELD`; continue: remove it and use `--resume` with the same options. Before the first launch run `mkdir -p $V/runs_bench_gpu` is not needed; the driver creates it, and the runner's own manifest step writes MANIFEST under it.)
