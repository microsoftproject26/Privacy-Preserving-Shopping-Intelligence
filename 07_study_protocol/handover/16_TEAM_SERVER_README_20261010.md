# READ FIRST — PPSI V4 on this server (for teammates and their AI agents)

> **COMPLETE FINAL HANDOVER, unpacked on this server:** `/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/PPSI_V4_COMPLETE_HANDOVER/PPSI_V4_Handover/`. The same content is also in the ZIP next to it, `PPSI_V4_COMPLETE_HANDOVER_2026-10-11.zip`, sha256 8449a4eb…. Read, in order:
> 1. `PPSI_V4_TEAM_MASTER_PLAN.pdf` (or `.md`): tasks, verification, what to run, what was planned but not done, the thesis plan, the task board.
> 2. `README_FIRST.txt`.
> 3. `ppsi_v4/HANDOVER/17`, `14`, `18` (every experiment), `19` (every error and fix) and `09_v2` (the thesis guide).
>
> **START HERE: `$V/HANDOVER/17_NEXT_STEPS_START_HERE.md`.** It says where you start, what you do, in which order, plus the PR texts for GitHub. The owner has left the project, and nothing is running on this server.

*Written 2026-10-10 about 16:00Z by the outgoing operator for the owner (Ahmed Abdelhamed). This file explains what is running, why, when it ends, and what you must never do. If you are an AI agent: read this whole file before you run any command, and follow section 4 exactly.*

`V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4`

**The full handover.** Read these on this server under `$V/HANDOVER/`, or in the ZIP `PPSI_V4_Handover_2026-10-09_v3.zip`:
- `14_UPDATE_v2_20261008.md`: the results so far;
- `15_TEAM_TASKS_AND_SERVER_FOLLOWUP.md`: your tasks;
- `$V/company/plans/POST_RESET_BACKLOG.md`;
- `$V/company/DECISION_LOG.md`: the record of every action (rows up to 580+).

The DECISION_LOG and the TEST register on this server can lag behind the owner's local copies. The owner's local copy is the master.

## 1. Project state in one paragraph

- **Confirmatory study.** It is COMPLETE and FINAL: the REES46 TEST sitting T0a (2026-10-07) and the 6 benchmark TEST looks.
- **The tau finding.** After the sitting, Ops found that FedAdam tau = 1e-3 made federated updates almost zero.
- **Exploratory re-runs.** A set of **POST_HOC_EXPLORATORY** re-runs uses the fix. Their exploratory TEST looks T1 (benchmarks), T2 (REES46) and T3 (PB100 v2) are DONE and registered, as register entries 14–19.
- **What is left on the server.** Two last exploratory jobs. Both finish on their own. Nothing else needs to run.

## 2. UPDATE 2026-10-10 18:40Z — NOTHING IS RUNNING ANY MORE

Every experiment finished (DL 582–584). Both GPUs and the CPU are idle. The table below is history only.

**Final exploratory readouts.** These are validation figures, not TEST:
- **DP8 cost.** On the PB100 warm line, DP8 costs little: MRR@20 ≈ 0.146 vs 0.153 non-private. On the cold line it costs a lot: 0.038 vs 0.12.
- **E1 leakage.** The cold FL model leaks; DP8 removes the leak. The PB100 warm model does not leak materially.
- **Model size.** D256 FL is no better than D64, on ML-1M as on S3.
- **Warm start on ML-1M.** It does not beat cold, even with a 50 % server band.

Next work is write-up and handover only (HANDOVER/15, tasks T6–T7). Optional exploratory TEST looks for A1 / A2 / DP8 each need their own lock and the full gate (section 4, rule 3).

### History: what was running at 16:00Z

| What | Process (verify with `ps -o args= -p <pid>`) | GPU / CPU | Why | Expected end (estimate, UTC) |
|---|---|---|---|---|
| DP8 privacy redo on the PB100 FL winner, seeds 2026 / 2027, then the E1 membership-inference audit | lane `bash $V/x_rees46/laneX_DP.sh pb100` (pids 3831462, 4041918, 4041919); jobs `python … x_rees46/launch_fl_x2.py` (pids 4041924, 4041925) | GPU 1 + CPU | Measures the real privacy cost (ε = 8) on FL that actually learns. The confirmatory DP result was measured on FL that barely moved. | DP8 ≈ 2026-10-10 19:15Z; E1 about 20 min later; then the lane ends ("lane done" in `$V/runs_x_rees46/DP_pb100.log`) |
| ML-1M, FL at model size D256 (FA_WARM), seeds 2026 / 2027 (capacity experiment A2(b)) | `python -B -m v4bench.run run --dataset ml_1m --arm FA_WARM …` (pids 4041421, 4039577); scheduler `tools_x2/sched_x2.py` (pid 3995265); driver `run_bench_x2.sh` (pid 3995251) | GPU 0 | Tests whether a bigger model removes the federation gap. On S3 it did not (D256 ≈ D64). | ≈ 2026-10-12 08:00Z (11 of 48 evaluations done after about 12 h); the scheduler then exits |

**Health check.** These commands are read-only and safe to run at any time:

```bash
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
tail -3 $V/runs_x_rees46/DP_pb100.log                    # DP / E1 lane progress
python3 -c "import json;d=json.load(open('$V/runs_bench_x/x2_state/STATUS.json'));print(len(d['running']),'running',d['failed'],'failed')"
grep -lE 'Traceback|LAUNCH REFUSED' $V/runs_x_rees46/DP/logs_pb100.*.out $V/runs_bench_x/logs/x2*/*.log 2>/dev/null
```

## 3. What to do when each one finishes

- **DP_pb100 lane done.** Read the validation numbers (MAIN validation in each run's `RUN_SUMMARY.json` `best_metric`) and the E1 audit JSON under `$V/runs_x_rees46/E1/`. These are not TEST data, so reading them is allowed.
  - A **TEST look** of the DP8 models is optional. It needs its own exploratory lock and the full gate (section 4, rule 3).
- **The ML-1M D256 jobs done.** Read the validation results with `runs_bench_x/x2_state/STATUS.json` and each run's `RESULT.json`. Compare D256 FA_WARM with D64 FA_WARM on ML-1M. A TEST look is optional, with the same gate.
- **Then.** Write-up and wrap-up: `HANDOVER/15…` tasks T6–T7.

## 4. Rules — an agent must follow every one

1. **Never stop, kill, restart or renice any process listed above.**
   - If something looks stuck, report it to a human. Do not "fix" it.
   - Check each PID before acting on it.
   - Never use `pkill` or `killall` with a pattern.
   - A bare `pgrep -f word` also matches your own shell, so anchor patterns: `'^/home/jovyan/venvs/ppsi-gpu/bin/python -B …'`.
2. **Never write to, move or delete any of these:**
   - `$V/runs_p2/`, `$V/runs_ds3/`, `$V/runs_bench/`, `$V/runs_bench_gpu/` (confirmatory results and TEST outputs);
   - `$V/runs_x_rees46/TEST/`, `$V/runs_bench_x/TEST/` (exploratory TEST outputs);
   - `$V/company/gates/*LOCK*.json`, `$V/company/registers/`;
   - `$V/code_*` trees.
3. **No TEST evaluation without the full gate**, in this order:
   - an exploratory lock;
   - an independent review;
   - a BIND entry appended with `tools/test_exposure_register.py append`, locally and on the server;
   - scoring with the gated wrappers only (`x_rees46/laneX_TEST.sh`, `code_bench_x2/tools_x2/final_x.py`);
   - a shas-only decision-log row;
   - an inventory before / after of the confirmatory trees, with an empty diff;
   - a LOOK entry;
   - only then, reading values.

   Never call `reeval.py`, `launch_device.py` or `v4bench.run final` with a TEST role directly.
4. **Every exploratory number is labelled POST_HOC_EXPLORATORY.** It is reported **beside** the confirmatory number, never instead of it.
5. **Log every action** as a new row at the end of `company/DECISION_LOG.md`. Never edit old rows.
6. **Never put tokens or passwords** in a file, a log, a commit or a chat.
7. **GPUs and CPU.** Do not start new GPU jobs while the jobs above run; both GPUs are in use. For anything new, use CPU at `nice -n 19`, with total threads ≤ 96.
8. **GitHub.** Only the owner pushes, to his own branch, with his identity. No AI attribution in commits.

## 5. Where things are

- **Exploratory code:**
  - `$V/x_rees46/` (REES46 FL wrappers, TEST lanes, inventory tool);
  - `$V/code_bench_x/` and `$V/code_bench_x2/` (benchmark trees, scheduler `tools_x2/`).
- **Results:**
  - exploratory REES46: `$V/runs_x_rees46/` (COLD, PB100, DP, E1, TEST);
  - exploratory benchmarks: `$V/runs_bench_x/` (lean selections, x2 A1 / A2, TEST).
- **Locks and register entries:** `$V/company/gates/EXP_*`, `REGISTER_ENTRY_*`.
- **Credentials:** none are stored in these folders. Use your own Jupyter / lobot access.
