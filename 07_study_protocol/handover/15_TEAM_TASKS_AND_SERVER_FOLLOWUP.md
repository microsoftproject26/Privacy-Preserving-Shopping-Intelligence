# 15 — TEAM TASKS and server follow-up (v2, 2026-10-09)

*For the colleagues taking over. Read `14_UPDATE_v2_20261008.md` first: it covers what happened and the results. This file
says what to do, in order, and how to watch the server. The source of truth is `company/DECISION_LOG.md`: append one row
for every action, and never edit old rows.*

> **ملخص بالعربية.**
> - **ما خلص:** النتائج التأكيدية كلها خلصت ونهائية، فلا تُعاد ولا تُعدَّل.
> - **ما تبقّى:** تجارب استكشافية بعد الاطلاع (بعضها شغال على السيرفر الآن)، واختبارات TEST استكشافية بشروط صارمة (القسم 3)، ثم الكتابة والتسليم.
> - **المتابعة:** افحص السيرفر مرتين يوميًا على الأقل بالأوامر في القسم 1.

---

## 1. Watching the server (daily, about 5 minutes)

**Access.** Use the lobot Jupyter of the group workspace (shared `jovyan` account). Root:
`V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4`.

**Health check.** Run in a terminal:

```bash
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
vmstat 2 2 | tail -1 ; df -h $V | tail -1
pgrep -fc '^/home/jovyan/venvs/ppsi-gpu/bin/python -B .*launch_fl_x'      # FL exploratory jobs
pgrep -fc '^/home/jovyan/venvs/ppsi-gpu/bin/python -B -m v4bench.run run' # benchmark jobs
tail -3 $V/runs_x_rees46/COLD/lane.log $V/runs_x_rees46/PB100/lane.log $V/runs_x_rees46/DP_*.log
grep -lE 'Traceback|LAUNCH REFUSED' $V/runs_x_rees46/*/logs/*.out $V/runs_bench_x/logs/cap/*.log 2>/dev/null
```

**Rules for processes.**
- Always use anchored `pgrep` patterns, as above. A bare word also matches your own shell command.
- Kill only by a verified PID: `ps -o args= -p <pid>` first. Never `pkill` by pattern.
- Never kill a running `v4bench.run final` (a TEST look).

**If a job failed**:
- read its log tail;
- fix it only in the exploratory code copies (`$V/x_rees46/`, `$V/code_bench_x/`), never in a confirmatory tree;
- resume with `--resume` when the engine supports it;
- log a DL row.

The recent fixes are worked examples: DL 548–551.

## 2. Jobs running at handover and what to do when each finishes

| # | Lane | ETA (estimate) | When it finishes |
|---|---|---|---|
| R1 | REES46 cold FA, tau-fixed (C1): `runs_x_rees46/COLD/`, MAIN s2026 + s2027 | about 10-09 12:00Z | `laneX_DP.sh cold` starts DP8 + E1 on its own. Then build the TEST lock (task T2). |
| R2 | PB100 grid S3 (2× / 4× rounds), then `/tmp/pb_after.sh` relaunches `laneX_PB.sh` → final selection → MAIN 2 seeds | S3 about 10-09; MAIN 10-10 to 10-12 | `laneX_PB2.sh` builds `runs_x_rees46/locks/EXP_PB100_TEST_LOCK_v2.candidate.json` and the VALIDATION proof. Then task T3. If `/tmp` was cleaned, rerun `laneX_PB.sh` by hand: finished steps are skipped by their `_state/*.ok` markers. |
| R3 | DP8 + E1 on the PB100 winner (`laneX_DP.sh pb100`) | after R2 | part of T3 |
| R4 | C_SMALL central D64 on 6 benchmark datasets × 2 seeds: `runs_bench_x/…C_SMALL__B24…`, logs `runs_bench_x/logs/cap/` | hours | Read the validation `RESULT.json` (tracker best_value) and compute FL / C_SMALL (task T5). |

## 3. Tasks, in order

- **T1 and T2: DONE 2026-10-09** (DL 566–570; register entries 14–17; results in 14_UPDATE_v2 §2b). The text below is kept for reference.
- **T1. Fixed-benchmark exploratory TEST.** Covers the 6 datasets and the FA / FA_WARM winners in `runs_bench_x/lean/SELECT_*.json`, 2 seeds each.
  - Build an exploratory lock in the style of `company/gates/EXP_PB100_TEST_LOCK.json`, plus BIND and LOOK register drafts.
  - Get an independent review (Fable, or the supervisor).
  - Append BIND → score → write the result file → append LOOK.
  - Compare against C_FULL and against the confirmatory TEST rows.
- **T2. REES46 C1 (cold, tau-fixed) and Part B (warm, tau-fixed) exploratory TEST.** Use the same procedure.
  - Builder: `x_rees46/x_lock_build.py`; entries: `x_rees46/make_entries.py`; scorer: `x_rees46/laneX_TEST.sh` (GPU 0).
  - Contrasts: X_FA − PRE, X_FA − P2_WSTAR, X_FA − confirmatory FA, R(· / C_FULL).
- **T3. PB100 v2 exploratory TEST**, of the grid winner and its DP8 / E1. Use the lock v2 candidate from R2.
  - Do **not** pass `--with-ft` unless the lock lists FT entries (Fable condition, DL 540).
- **T4. New experiment A1: PRE on disjoint users.**
  - On s3_beauty / sports / toys / ml_1m, split the users into a server band S and FL clients F, with no overlap. Use 2 band sizes, 15 % and 50 %.
  - PRE trains on S only; FA cold and FA_WARM train on F only; evaluate on F.
  - Recipe: tau 1e-6, B24, unfrozen; 2 seeds; validation first, TEST under T1-style gating.
  - This tests whether a real server-data advantage makes the warm start worth it (REES46 says yes: K_INIT +0.033).
- **T5. New experiment A2: capacity decomposition.**
  - (a) C_SMALL is running (R4).
  - (b) Add FL at D256_B3 (cold and warm) on the same datasets.
  - Report: total gap = size cost (C_SMALL vs C_FULL) + federation cost (FL D64 vs C_SMALL).
- **T6. Write-up.** List in `company/plans/POST_RESET_BACKLOG.md` §D. In short:
  - the confirmatory results as printed;
  - the tau finding with its MEASURED evidence;
  - "FL value depends on server data" (REES46 vs PB100 vs A1);
  - the disclosures and incidents;
  - the published comparison (Spotify Table 1, full ranking).
- **T7. Wrap-up.**
  - Final report; `.local_ai/thesis_reference` capture (DL 442 onward).
  - GitHub: owner's branch only, commit identity checked by `git var` (`AhmedAbdelhamed01 <114183798+AhmedAbdelhamed01@users.noreply.github.com>`), no AI attribution, never push to main.
  - Final ZIP.

## 4. Never

- **Never touch the confirmatory record.** Never edit, re-run or re-select anything confirmatory:
  - `runs_p2/TEST`, `runs_ds3/ANA`, `runs_ds3/TEST`;
  - `runs_bench*/…/LOCK.json` / `TEST_*`;
  - `company/gates/*LOCK*.json`;
  - the history of `company/registers/TEST_EXPOSURE_REGISTER.json` (append only, with its tool).
- **Never score on TEST without these, in order:**
  - an exploratory lock;
  - an independent review;
  - a BIND entry first.
- **Labels.** Every exploratory number is labelled POST_HOC_EXPLORATORY and reported **beside** the confirmatory one, never instead of it.
- **Secrets.** Never put a token or password in a file, a chat or a commit.

---

## 5. Update 2026-10-08T23:50Z (server clock): lanes A1 + A2(b) launched, T1 prepared (DL 558–559)

All of this is POST_HOC_EXPLORATORY, inner validation only. Nothing here touches a confirmatory artifact.

**What runs.** One CPU lane, "X2", 88 runs (nice 19, 4 threads each, no GPU).

- **A1, disjoint PRE (task T4): 64 runs** under `runs_bench_x/x2_A1/<ds>/`.
  - Datasets: s3_beauty / s3_sports / s3_toys / ml_1m. Splits DJ15 and DJ50. Seeds 2026 and 2027.
  - Per split and seed:
    - PRE on S (D64, B96, the winner PRE's dropout);
    - FA_WARM on F, from that PRE's PRACTICAL_BEST;
    - FA cold on F;
    - C_SMALL on F (central D64, B24, dropout 0.1).
  - Splits: `runs_bench_x/x2_A1/splits/<ds>__DJ15|DJ50.json` (write-once, self-hashed).
    - Rule: the permutation of the registered 15 % band; S = its first 15 % / 50 % of users; F = the rest.
    - So S_DJ15 equals the registered band exactly (checked), and DJ15 sits inside DJ50.
  - Every A1 run selects and reports on F's users. That is group `OTHERS`, because `BAND` = S in these runs.
  - The PRE rows also report PRE-alone on F (OTHERS): this is the warm-start reference.
- **A2(b), FL at D256_B3 (task T5b): 24 runs** under `runs_bench_x/x2_A2B/<ds>/`.
  - Same 4 datasets, 2 seeds, the registered 15 % band, all users as clients.
  - PRE D256 uses the 02B_SASREC central recipe, B96, the winner's dropout.
  - FA_WARM D256 starts from it; FA D256 is the cold arm.
  - Decomposition: C_FULL vs C_SMALL = size cost (central); FL D64 vs C_SMALL = federation cost at D64; FL D256 vs C_FULL = federation cost at D256.
- **The recipe (fixed).**
  - tau 1e-6, B24, unfrozen, 1 local pass.
  - Server / client LR = the dataset's validation winner (`runs_bench_x/lean/SELECT_<ds>.json`, sha256 recorded in the plan).
  - FA: s 1e-3 / c 0.1 everywhere.
  - FA_WARM:
    - beauty and toys s 1e-3 / c 0.3;
    - sports s 3e-4 / c 0.025;
    - ml_1m s 3e-3 / c 0.1.

**Code.**
- `$V/code_bench_x2` is a copy of code_bench_x. It adds `--user-subset <split> --subset-role S|F` and `--model-variant SASREC_D256_B3`.
  - Every spec carries `tag: POST_HOC_EXPLORATORY`, a `study` label and the split record.
  - It writes only under `runs_bench_x/x2_*`.
- Sums: `X2_SHA256SUMS` d3dbc130….
  - The live scheduler started from sched_x2.py 40f83af0. The file is now 7d0d3498 (a cosmetic STATUS-key fix); it takes effect on the next relaunch.
- Local mirror: `ppsi_v4/code_bench_x2/`. Plan: `PLAN_X2.json` d9f9d3ca… (copy in `results/X2_20261009/`).
- CPU smoke passed (SMOKE_NOT_EVIDENCE, `runs_bench_x/x2_smoke`). It covered:
  - roles and counts (n_clients = n_F);
  - warm from the same split;
  - refusals for a wrong role, another split's PRE, D256→D64, C_SMALL at D256, and a root outside x2.

**Watching it.**

```bash
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
cat $V/runs_bench_x/x2_state/STATUS.json                      # done / running / waiting / failed
tail -5 $V/runs_bench_x/x2_state/scheduler.log
grep -lE 'Traceback|REFUSED' $V/runs_bench_x/x2_state/logs/A*.log
pgrep -fa '^/home/jovyan/venvs/ppsi-gpu/bin/python -B tools_x2/sched_x2.py'          # the scheduler
pgrep -fc '^/home/jovyan/venvs/ppsi-gpu/bin/python -B -m v4bench.run run'          # all benchmark jobs
cd $V/code_bench_x2 && /home/jovyan/venvs/ppsi-gpu/bin/python -B tools_x2/report_x2.py   # validation table -> x2_state/SUMMARY_X2.json
```

**How to resume.**
- Rerun `setsid nohup bash $V/company/plans/ops_scripts/bench/run_bench_x2.sh --slots 24 --threads 4 > $V/runs_bench_x/x2_state/driver_nohup.out 2>&1 &`.
- What the relaunch does:
  - DONE runs are skipped;
  - partial runs continue with `--resume`;
  - live runs are not started twice.
- Every live runs_bench_x job holds one of the 24 slots: 24 × 4 = 96 threads, the owner's cap.
- A run that fails twice shows under `failed` in STATUS.json.
  - Fix it only in code_bench_x2, update X2_SHA256SUMS, then relaunch.
  - Log a DL row.

**ETAs (INFERENCE, server clock).** The estimates come from the measured POST_HOC_X timings; D256 ≈ 3.3× D64 per round (toys smoke).

| Part | ETA |
|---|---|
| S3 parts of A1 | ≈ 10-09 06:00–12:00Z |
| ML-1M A1 (PRE on DJ50 ≈ 11 h, then FA_WARM) | ≈ 10-09 16:00–20:00Z |
| A2(b) S3 | ≈ 10-09 20:00Z |
| A2(b) ML-1M (MEASURED smoke: 51.7 s per D256 round, 4.5× D64, so FA D256 ≈ 36–50 h; PRE D256 ≈ 11–15 h, then FA_WARM D256 ≈ 36–45 h) | FA ≈ 10-10 12:00Z – 10-11 02:00Z; FA_WARM ≈ 10-11 02:00–14:00Z (critical path; DL 560) |

Total ≈ 449 job-hours.

**T1 (fixed-benchmark exploratory TEST): prepared, NOT run.**
- Tool: `code_bench_x2/tools_x2/final_x.py` (sha 861f7a8e…). It scores with code_bench_x's unchanged v4bench (`final.score_run`, `_load_holdout_view`) and writes only `runs_bench_x/TEST/<ds>/`.
- Candidate lock: `company/gates/EXP_BENCH_X_TEST_LOCK.candidate.json` (66a808a0…, draft).
  - It covers FA_X / FA_WARM_X / C_SMALL_X × 2 seeds × 6 datasets, plus the sha256 of the confirmatory TEST rank files.
  - Still pending in it: ml_1m C_SMALL s2026 / s2027 (running).
- Register drafts: `company/gates/REGISTER_ENTRY_DRAFT_EXP_BENCH_X_TEST_LOCK_BIND.json` / `_LOOK.json`. The FILL fields are completed after the final lock exists.
- Dry run: `results/X2_20261009/final_x_dryrun2.log`. It was refused at the BIND check; no TEST dir was created; the register is unchanged (bdfb936b…).
- **Order, after ml_1m C_SMALL is DONE:**
  1. `final_x.py build-lock` (candidate again).
  2. Fable gate on the candidate + final_x.py.
  3. `final_x.py build-lock --final`, which writes `EXP_BENCH_X_TEST_LOCK.json` write-once.
  4. Fill the BIND draft (lock file sha, final_x sha), then `tools/test_exposure_register.py append --entry …` (local + server; same register sha).
  5. Run the scoring: `final_x.py score --dataset <ds>` for each of the 6 datasets (CPU, 16 threads, nice 19).
  6. Run the analysis: `final_x.py analyze` → `runs_bench_x/TEST/ANALYSIS_EXP_BENCH_X/X_RESULT.json`.
  7. Fill and append the LOOK entry, then add a DL row.
- **Owner's yes is required before step 4** (backlog rule).
