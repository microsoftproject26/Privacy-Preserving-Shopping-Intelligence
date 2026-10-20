# 07 — Server and operations

*Facts are MEASURED from DL rows and scripts unless labelled. `$V` = `/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4`,
`$R` = `$V/runs_ds3`, `$Q` = `$R/queues`. The detailed history of the code trees and deploy procedure (r2–r5) is in
`HANDOVER/reference/PPSI_V4_TEAM_HANDOVER_20260930.md` §8; the endgame scripts are documented in
`HANDOVER/reference/ENDGAME_SCRIPTS_README.md`.*

## 1. The machine

- **Lobot**, Queen's shared JupyterHub group server `group-digi2026-g8` at `lobot.cs.queensu.ca`. No SSH, no scp. Every
  teammate works as the same Linux user `jovyan`: home-directory contents do **not** identify a person; never touch other
  teammates' directories, tunnels or processes (e.g. `~/ahmed` belongs to a teammate, not the owner).
- Hardware (DL 158, 168, 303): 2 × NVIDIA A40 (46 GB each), 128 CPUs visible, large RAM (cgroup 384 GiB at DL 158; "924 GB
  free" reported at DL 303), disk 345 GB (≈ 268 GB free on 10-03, DL 376). torch 2.13.0+cu130. Environment lock archived in
  `deploy_archive/ENV_LOCK_A40.txt` and `ENV_PKGS_A40.txt`.
- Python for every job: `/home/jovyan/venvs/ppsi-gpu/bin/python`. **Never install packages** into it or locally (DL 118).
  The benchmark runner uses RecTools etc. from a separate site dir (DL 302).
- Frozen data: `/home/jovyan/shared_data/rees46_nextitem_v1` (read-only).

## 2. Access channel (each teammate needs their own)

- The owner drove the server from Windows with `.local_ai/sasrec_v1/tools/lobot.py` (a copy lives in `gru_v1/tools/`). It
  starts a python3 Jupyter kernel over the REST API, runs one command / script over the kernel websocket, then deletes the
  kernel. Subcommands: `cmd "<bash>"`, `sh <local.sh>`, `put <local> <remote>`, `get <remote> <local>`, `sync`. Run it with
  the local venv `next_item_eda/.venv/Scripts/python.exe` (needs `requests`, `websockets`).
- **Credentials:** `lobot.py` reads the token and base URL from `C:\Users\Ahmed\AppData\Local\Temp\lobot_upload\.tok` and
  `.base` (outside the project; MEASURED). A teammate must obtain **their own** token (e.g. `jupyter server list` in their own
  Lobot terminal), store it in their own local files and edit the path constant `T` in their copy. **Never share, print,
  commit or copy a token onto the server.** If you prefer, use the Jupyter web terminal directly; everything below is plain
  bash.
- Traps (from the owner's notes): Git Bash rewrites `/home/...` arguments — use `MSYS_NO_PATHCONV=1` or the `@W/` prefix;
  Windows writes CRLF — strip it on the server and run `bash -n` before use; launch long jobs detached as
  `(nohup setsid bash x.sh > log 2>&1 < /dev/null &)`, otherwise the kernel call blocks; a kernel call dies after its websocket
  closes. An inline `pkill` once matched the executor shell itself (DL 360, 368) — use exact pid lists.

## 3. Server layout (`$V`)

| Path | Content |
|---|---|
| `runs_p3/` | the 8 contest runs; **C_BIG checkpoints** `runs_p3/SASREC/02B_SASREC_s202{6,7}/sealed/checkpoints/{best.pt,endpoint_6.0.pt}`; G2 outputs `runs_p3/g2/` |
| `runs_ds/` | C_SMALL (`runs_ds/C_SMALL_s202{6,7}/sealed/checkpoints/`), the SMALL LR ladder, early baselines |
| `runs_pretrain/PRETRAIN_FOR_FL_SMALL_SASREC_D64_B2_s2026/` | PRE; θ_pre = `sealed/checkpoints/endpoint_6.0.pt`; init record `PRETRAIN_FOR_FL_INIT.json` (`5ed989f6…`) |
| `runs_ds3/` | the deployment study: `CFULL/` (C_FULL s2026 / s2027), `FL_T/` (every MAIN FL run incl. `N33_*`; the 3 STOPPED_REALLOCATED dirs), `FL/` (pre-S4 ladder, never indexed), `LRSCREEN*`, `S7S8_T/`, `WARM_PREVIEW/`, `DEV/` (device runs), `REEVAL_A40/` (central / panel reevaluations), `BASE_R4/` (baselines with rank files), `ONNX_R7/`, `ONNX_R7_U16384/` (K16), `ANA/` (freezes, blocks, `out_val_v3/`), `TEST/` (empty of scores; H outputs go here), `_PARKED_SMOKE/`, `queues/` |
| `runs_ds3/queues/` | every lane script and its `<lane>.log`; `_common_code_ds3r*.sh` (binds a lane to one tree); `endgame/` (D–H scripts, `ok/` markers, `FL_CKPT_RECORD_*`) |
| `runs_p2/` | the 2026-09-27 contest **P2 tuning** runs **and** the planned Phase-2 lane root (`WA/`, `C1/`, `C2/`, `P2T3/`, `recipes/`, `logs/`, `_lane_state/`) |
| `runs_bench/` | Wave 1 / Wave 2 benchmarks (CPU); `runs_bench/ml_1m` sealed SUPERSEDED_DEVICE; `_wave1_state/` |
| `runs_bench_gpu/` | ML-1M on the A40 (B16) |
| `bench_raw/`, `bench_proc/`, `bench_proc_w2/` | benchmark raw and processed data with SHA256SUMS |
| `contracts/`, `company/gates/`, `company/registers/`, `tools/` | server copies of the registered files and the frozen tools (some registered files exist **only** here: S7 / S8 selections, S_RECORD_WARM_FROZEN, SELECTION_FT_FA, STRATA_CUTPOINTS_MAIN) |
| `code_*` trees | immutable deployed code (next section) |

## 4. Frozen code trees

| Tree | Used for | Source |
|---|---|---|
| `code_p3`, `code_p3b` | contest runs | DL 91, 102 |
| `code_ds`, `code_ds2` | SMALL ladder, C_SMALL, early baselines | DL 133, 137 |
| `code_ds3r2` … `code_ds3r5` (+ `r2s`, `r2p`, `ds3s2`) | A40 receipts, LR screens, PRE, C_FULL, ladder, early warm runs | 09-30 handover §8.4 |
| `code_ds3r6` | the warm MAIN line, warm method set, DP chains, N33 re-runs (runs resume on their own tree) | DL 291 |
| `code_ds3r7` | endgame B / B2 / D / E / F / G / H (all but the ONNX step); evaluator `61b64e61…` | DL 342 |
| `code_ds3r7b` | = r7 + DEV-05 + `--latency-only`; used only by H `onnx` and K16 pass 2 | DL 385 |
| tools_r8, tools_r9 | the two analysis-tool fixes (retention_report `d0247370` → `9a3b2c2a`) | DL 392, 397 |
| `code_p2r1` | Phase 2 | DL 435 |
| `code_bench1` … `code_bench4` | benchmark runner revisions (bench4 = B12) | DL 332–360 |
| `code_bench5`, `code_bench5b` | ML-1M GPU runner (+ `final --device`) | DL 409, 423 |

Archives with SYNC_MANIFESTs and tar shas: local `deploy_archive/` (r3, s2, r4–r9, r7b, bench5, bench5b, p2r1). Scratch
candidate trees on the server (`code_r7c_dev04`, `code_r7b_dev05_cand`, `code_r7b_p2_cand`) are not production.

**Deploy pattern** (r3 … p2r1): freeze local code (all suites exit 0; NCs fail when their guard is removed) → pack with
`pack_r*.sh` (asserts the reviewed shas of every changed file) → upload → extract into a **new** dir beside the old one →
verify every file's sha against SYNC_MANIFEST → `chmod -R a-w` → targeted server tests (one known Windows-CPU digest pin
fails identically on every tree, DL 245 / 342) → tiny-run digest comparison of the 7 FL arms (`traj_cmp*.sh`; values DP
`08383b6f`, FA `807eadfb`, FA_1024 `878fc357`, FA_Q8 `651f0eb0`, FP `825b27f8`, PF `991bd6a4`, S_CAL `ff1b5150` identical
r3 → r7b) → a lane helper `_common_code_<tree>.sh` → DL row.

## 5. Lanes, queues and monitoring

- A **lane** is a detached bash script in `$Q/` started with `nohup setsid`; it runs one or more jobs in order, may wait for a
  file or a `QUEUE_DONE` line, logs to `$Q/<lane>.log` and writes `LANE_STOP` on a failure or refusal. Lanes only launch
  what is already registered; they never decide on a value.
- Lane scripts for the warm line, DP chains, N33 and D3: `company/plans/ops_scripts/lanes_r6w/`; endgame: `…/endgame/`;
  benchmarks: `…/bench/` (`run_wave1.sh`, `run_wave2.sh`, `run_ml1m_gpu.sh`, `b13_stopper.sh` (disarmed)); Phase 2:
  `…/p2/laneP2_RUN1.sh`.
- **Each check (value-free):** `grep -l LANE_STOP $Q/lane*.log` newer than the last check; `grep -h REFUSED` in new logs;
  `nvidia-smi`; `ps -eo pid,etime,args | grep -E "launch_|v4bench|lane"`; count `RUN_SUMMARY.json` files; `df -h $V`. Read
  only status / progress fields (`status`, `round`, `efe`), never metrics. Read-only status helpers: `ops_scripts/status_*.sh`,
  `watch_events3.sh`, `watch_stops.sh`.
- Never rewrite a running lane file. To change a plan: stop / withdraw the lane (record it: LANE_WITHDRAWN, DL row), then
  start a new lane.
- An old `tests_fl/run_all.py` pytest (pid 1073, alive for days at DL 399) was left to inspect after F — check it is not a
  teammate's process before touching it.

## 6. GPU / CPU use, device classes and determinism

- **Device classes** (never mixed inside one comparison block): `A16-FP32-DET` (the contest); **`NVIDIA_A40-FP32-DET`**
  (the whole deployment study; pin `contracts/eval_device/EVAL_DEVICE_PIN_A40.json` `db744ca3…`; A40 and A16 ranks of the same
  model differ on 12,635 / 14,063 rows, DL 163); `BASELINE_CPU` rank files (LAST etc.; a registered exception, DL 230); the
  ONNX-CPU block (K16; onnxruntime 1.24.1, single-thread); `CPU-FP32` (S3 and Wave-2 benchmarks, 16 threads fixed);
  `NVIDIA_A40-FP32-DET` again for ML-1M (B16); `SERVER_BROWSER` and the owner's laptop for Phase-2 device numbers.
- **Determinism:** STRICT FP32 (TF32 off, deterministic algorithms, `CUBLAS_WORKSPACE_CONFIG=:4096:8`); results do not depend
  on which jobs share a GPU (DL 82, 179), so jobs may share a card (`--allow-shared-gpu`, plus `--coschedule-06g` for central
  runs). Name the GPU explicitly: the GPU index is part of the RunConfig (DL 110).
- **Throughput (MEASURED, INFERENCE for durations):** an FL process is host-bound (≈ 1 core, 18–30 % of an A40); 6 FL jobs ≈
  950 rounds / h each vs 10 jobs ≈ 550 / h each — the cards are compute-saturated, more concurrency only splits time (DL 294).
  A MAIN warm FL run (13,654 rounds) took ≈ 24–30 h wall when shared; a CALIB screen ≈ 5–7 h with 8 sharing.
- **Precedence now:** H steps > Phase-2 lanes (`laneP2*.sh`) > the ML-1M GPU driver, which yields through
  `runs_bench_gpu/_gpu_state/YIELD` or when it sees an H / laneP2 process (B16.5). K16 pass 2 waits for every Wave-1 driver
  (incl. `run_ml1m_gpu.sh`) and drains Wave 2 with SIGSTOP / SIGCONT (DL 378–380).

## 7. Disk and retention

- Keep ≥ 10 GiB free (floor 8 GiB + the next atomic file). Never delete evidence to make room. Deletions only with the
  owner's approval and a verified path / size / sha256 manifest (DL 19, 44, 46). Probes and smokes delete their own weights
  and keep digests (DL 30).
- Moving instead of deleting: smoke dirs were parked read-only in `$R/_PARKED_SMOKE/` with a manifest (DL 356); an aborted
  freeze's leftovers in `$R/ANA/_aborted_freeze_v1_*` (DL 387).
