# REV-P2-RUN1 - review of laneP2_RUN1.sh (read-only review, 2026-10-03)
VERDICT: PASS_WITH_MINORS

Tests: `cd code && python -m pytest tests_int/test_p2_run1_lane.py -q` -> 6 passed in 190 s (MEASURED here, Git Bash, fake interpreter; nothing run on the server, no GPU).

## Checks
1. Job list = registered cells (PASS). Script lines 81-85, 104-123, 133-137. WA core = {fedavg 0.3, 1.0; fedavgm 0.03, 0.1; fedadam 1e-4, 3e-4} x c {0.003, 0.01, 0.03} = 18; WA-ext fedadam {1e-5, 1e-4} at c 0.1 = 2; conditional incumbent fedadam 1e-3 / c 0.1 = 21st; C1 = {FULL, FROZEN} x peak LR {1e-5, 3e-5, 1e-4, 3e-4} = 8 (FROZEN gets --p2-freeze-item-tables); C2 = 1. 31 jobs with P2-T3. No missing or extra cell, no wrong value (compared with PREG2 4.2 / 4.3 / 5.2 and the test pins it to launch_fl.p2_wa_recipes / p2_run_tag / central_ft.c1_run_id). Recipe files: deterministic, write-once (set -C, cmp on rewrite, chmod a-w), loaded by the real load_p2_recipe in the test (lines 205-210).
2. Targets (PASS). Every --run-dir / --dry-run-out / --out is under runs_p2; CODE must end in code_p2r1 (line 54); runs_ds3 / runs_bench* refused (53); runs_ds3 appears only as the read-only --s8tc-search-root. Flags match P2-01_02 s7 and P2-05 s6.1 / 6.2. WA: --arm P2_W --role CALIBRATION --seed 2026 --p2-recipe; C1: --role CALIBRATION; C2: no --role (MAIN via c_full_small), no --smoke (S8). Cap 250 GPU-h noted (176-177): estimate 44-69.
3. P2-T3 first (PASS). All other jobs have soft-after p2t3 (99, 116, 127, 274). Receipt BIT_IDENTICAL -> incumbent REUSED; FAIL, rc 2, or no receipt -> incumbent runs (conservative; 148-155). Check tool is the registered tools/p2_t3_check.py, receipt O_EXCL by the tool.
4. Name / H guard / slots (PASS). Basename laneP2_RUN1.sh matches run_ml1m_gpu.sh YIELD_RE `[l]aneP2[A-Za-z0-9_]*\.sh`. H_test.sh running -> refusal exit 2 before anything is written (185); no new launches if one appears (226, 270), exit 3. Per-GPU slots 1-4 (default 2), static GPU round-robin.
5. S8 / resume / isolation (PASS). The driver prints only job name, GPU, rc, seconds; run_done uses grep -q on the public RUN_SUMMARY stub / STATUS (TRAIN-03: the stub holds status only, metrics are under sealed/). DONE_* skipped; started-but-unfinished gets the launcher's own --resume (244); failures write .fail markers and do not stop others; old .fail moved aside; pid lock.
6. Tests real (PASS_WITH_MINORS): they exercise the real script with a fake interpreter and cross-check registered cells against launch_fl / central_ft.

## Minors (none blocks launch)
- m1 (script 144, 256): C2 first leg writes a DONE_MAIN stub before the in-process C3 continuation; the STATUS RUNNING guard covers a crash mid-continuation only if STATUS stays RUNNING then. After a C2 crash, check STATUS before trusting "done".
- m2 (script 226-281): the yield logic stops NEW launches when H_test.sh appears but does not stop running Phase-2 jobs; S7 "never share with H" is therefore Ops' duty (do not start H while the lane runs; touch _lane_state/STOP and wait first).
- m3 (script 245-247): P2-T3 smoke relaunch after an interrupted smoke adds --resume to a --smoke run; not exercised on a real launcher (fake only). If it refuses, delete the P2T3/run dir (it is evidence-free) and relaunch.
- m4 (script 285, 284): `PENDING=("${NEXT[@]}")` with an empty array fails under set -u on bash < 4.4; the server bash is expected to be newer, check `bash --version`.
- m5 (tests line 286): `assert A and B or C` precedence makes the STOP assertion weak.
- m6: no GPU smoke of C1 (P2-05 6.1a) or firewall S0 (P2-01_02 s7) is in the lane; run them first as handoff Q2 says. No per-WA dry run (Q9).
- m7: C1 / C2 / WA need code_p2r1 chmod -R a-w, tools/p2_t3_check.py at $V/tools, fresh $V/runs_p2 (handoff Q1, Q4).

## Launch command (server)
```
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
L=$V/company/plans/ops_scripts/p2/laneP2_RUN1.sh        # basename must stay laneP2_RUN1.sh
ls $V/runs_p2 2>/dev/null; bash --version | head -1
bash $L --plan --gpus "0 1" --c2-early                   # read the 31-job plan
(nohup setsid bash $L --launch --gpus "0 1" --c2-early > $V/runs_p2_lane.out 2>&1 < /dev/null &)
```
Stop: `touch $V/runs_p2/_lane_state/STOP`; relaunch with the same command (finished runs skipped, started ones resume).
