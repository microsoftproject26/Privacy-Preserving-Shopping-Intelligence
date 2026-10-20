# MASTER PLAN V5 — from now to thesis submission (literal, executable; ONE A40)
Written 2026-09-28T22:05Z by Claude Fable 5.1 (chief scientist + Operations Director). Supersedes REST_OF_PROJECT_PLAN.md, the
§5 plan of FABLE_FULL_REVIEW_AND_PLAN_20260928.md and §E.1 of FABLE_THESIS_RESEARCH_REVIEW_20260928.md wherever they differ.
Binding rules: `contracts/PREG1_AMENDMENT_06D.md` (tiers §8; every decision), amendments 03–06C, EVALUATION_CONTRACT_V4 §10.
**Every duration is INFERENCE until §3 measures it.** Nothing is read before §9 except the three CALIB values of the LR screen (§4).

Conventions (server): `V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4`;
`PY=/home/jovyan/venvs/ppsi-gpu/bin/python`; `CR=$V/code_ds3` (the frozen downstream tree of §1.6; NEVER `code/`, `code_ds`, `code_ds2`);
`R=$V/runs_ds3` (new run root; `runs_ds`/`runs_p3` are never written); `G` = OUR A40 index (owner names it; the other card is never
used). Every GPU job: `cd $CR; export PYTHONPATH=$CR PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
V4_DATASET_ROOT=/home/jovyan/shared_data/rees46_nextitem_v1; unset CUDA_VISIBLE_DEVICES` (the launchers export the device from `--gpu`).
Long jobs detached: `(nohup setsid $PY -B <cmd> > <log> 2>&1 < /dev/null &)`; queues write a final `^QUEUE_DONE rc=<n>` line.
Before EVERY launch: `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; ps -eo pid,pcpu,pmem,args | grep -v grep | grep -c python`
(teammate's card untouched; ≤ 8 of our cores; ≤ 180 GiB RSS in total). Local channel: `.local_ai/sasrec_v1/tools/lobot.py`
(`cmd`, `put`, `get`; Git Bash with `MSYS_NO_PATHCONV=1`; scripts LF; every `cmd` is a fresh kernel). Tell the owner before each long job.
Records after every step: a DECISION_LOG row; RESUME_HERE line; thesis reference batch at the end of each phase.

---
## 0. Owner inputs (blocking for §2 onward)
| # | Needed | Why |
|---|---|---|
| O1 | **Our A40 index (0 or 1)** | binding resource split; never touch the other card |
| O2 | Go-ahead for the first long GPU jobs (§3–§5) and for CPU jobs up to 8 cores | shared server |
| O3 | The thesis submission date | sets the cut point in §13 |
| O4 | (later) run the browser benchmark on the owner's PC (§8.5) | the proposal's Edge target |
| O5 | (end) approval for the GitHub release of the reproducibility package | no push otherwise |

---
## 1. Local work before the server (in progress; owner not needed)
| # | Step | Owner | Acceptance | Status 22:05Z |
|---|---|---|---|---|
| 1.1 | REV-FL-01 + REV-ANA-01 independent Opus reviews | Fable dispatches | review files under company/reviews; verdicts | RUNNING |
| 1.2 | INT-06: A40 device pin + `v4int/reeval.py` + A40 receipt paths (06D §1) | Opus | suites exit 0; NCs; handoff INT-06 | RUNNING |
| 1.3 | DATA-09: gated TEST builder + REGISTRY_V2_TEST_EXT + loader gate + shim (06D §2) | Opus | tests_data09 synthetic; no real TEST row read | RUNNING |
| 1.4 | Fix packets from 1.1 (FL-08 / ANA-02 / DATA-07R1 as needed) + ANA-02 (rule map K12/K13, hierarchy labels, still-improving flag, `tools/ana_block_index.py`, tests_trunc basetemp mkdir) | Sonnet/Opus | suites exit 0 | after 1.1 |
| 1.5 | INT-07: TEST-role wiring (`launch_device --role MAIN_TEST|UNSEEN_HOLDOUT_TEST|…`, `v4base.run --phase MAIN_TEST`, `dump_rows --role MAIN_TEST`, `reeval --registry-ext`) through the DATA-09 shim; every TEST role refused without the gate | Sonnet | NCs: TEST role without gate refused | after 1.2 + 1.3 |
| 1.6 | One independent Opus review of INT-06 + DATA-09 + INT-07 + ANA-02 deltas; fixes; full local suites (tests_eval, tests_train, tests_int, tests_fl, tests_ana, tests_trunc, tests_data08, tests_data09, tests_dev, tests_base) | Fable | all exit 0, NCs fail as intended, locked packages unchanged | after 1.4–1.5 |
| 1.7 | **Freeze ANA-01**: `company/gates/ANA_SCRIPT_FREEZE.json` (sha256 of tools/retention_report.py + tools/ana_block_index.py + the library files it imports; K1–K16 text; 06D §3/§4 rule map) — O_EXCL | Fable | file written; DL row | after 1.6 |
| 1.8 | **Pack code_ds3**: `python tools/code_sync.py pack --out runs_ds3_deploy --exclude code/.train03r1_tmp` (+ tmp dirs); record tar/gz sha256 in DL | Fable | PACK.json | after 1.7 |
INFERENCE: 1.1–1.8 ≈ 1–2 days of worker/reviewer time. Items 1.4–1.8 run while the server does §2–§3 if the owner answers O1 early
(the deploy can be repeated as code_ds4 for the TEST tools: a NEW tree, never an overwrite; hashes in DL).

---
## 2. Server deploy and verification (CPU only; short; after O1)
```
# local: upload the pack + code_sync.py
$PY_LOCAL lobot.py put runs_ds3_deploy/code_sync.tar.gz $V/runs_ds3/deploy/code_sync.tar.gz   # sha printed; compare with PACK.json
# server (one cmd):
cd $V/runs_ds3/deploy && echo "<gz sha>  code_sync.tar.gz" | sha256sum -c - && rm -rf x && mkdir x && tar -xzf code_sync.tar.gz -C x && cp -a x/code $V/code_ds3 && rm -rf x && ls $V/code_ds3 | wc -l
# suites on the server (CPU; sequential; TMP under $V/runs_ds3/deploy/tmp):
cd $V/code_ds3 && export PYTHONPATH=$V/code_ds3 PYTHONDONTWRITEBYTECODE=1 TMPDIR=$V/runs_ds3/deploy/tmp && for s in tests_eval tests_train tests_int tests_fl tests_ana tests_trunc tests_data08 tests_data09 tests_base; do ... run_all.py / pytest -q ...; done > $V/runs_ds3/deploy/suites.log
# integrity: torch unchanged; locks; checkpoints
$PY -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_name(1))"   # must be 2.13.0+cu130; STOP if not
sha256sum $V/contracts/ARCHITECTURE_LOCK.json $V/contracts/DATA_PROTOCOL_LOCK_V2.json $V/runs_p3/SASREC/02B_SASREC_s2026/sealed/checkpoints/best.pt $V/runs_p3/SASREC/02B_SASREC_s2027/sealed/checkpoints/best.pt $V/runs_ds/C_SMALL_s2026/sealed/checkpoints/best.pt $V/runs_ds/C_SMALL_s2027/sealed/checkpoints/best.pt
```
Acceptance: fd201013…, bc699a97…, df35157f… / a8bbfff9… match the lock; suites exit 0; DL row with the code_ds3 tree hash.
Also now (CPU, ≈ 10 min, ≤ 8 cores): DATA-08 jobs 1–2 (`code_ds3/v4data08/job_build.sh`, `job_final.sh`, W=$V, code root = code_ds3 —
edit the two `W`/`code` lines of the copies under $V/runs_ds3/cfull/ so they use `$V/code_ds3`) → `data_cfull/receipts/DATA08_SERVER_RECEIPT.json`
overall_pass; fetch `contracts/manifest_registry/REGISTRY_V2_FULL_EXT.json` locally (sha in DL).

---
## 3. Measurement gate on the A40 (GPU G; ≈ 1–3 h; every number here is MEASURED and re-plans §4–§7)
3.1 **T3 evaluator receipt** (INT-06's command; ≈ 15 min): `$PY -B tests_eval/gpu_t3_receipt.py --gpu $G --families SASREC ...` →
    `runs_ds3/receipts/T3_A40/` — PASS bitwise vs the A16 bundle, or the exact diff count recorded (a diff is NOT a stop: it is the
    reason the A40 rows are a separate class).
3.2 **STRICT determinism** (INT-06's bench command; 2 reps × {SASREC_D64_B2, SASREC_D256_B3}; ≈ 20 min) → rep-to-rep bitwise = PASS; a
    mismatch STOPS the plan (report to the owner: STRICT failed on the A40).
3.3 **Write the pin** (Ops, O_EXCL): `$PY -B -m v4eval.device_pin write --device-index $G --t3-receipt <path> --strict-receipt <path> --out $V/contracts/eval_device/EVAL_DEVICE_PIN_A40.json`;
    fetch it locally; sha in DL. **No non-smoke job before this file exists.**
3.4 **Re-evaluate C_BIG and C_SMALL on the A40** (8 evaluations, ≈ 3–8 min each INFERENCE; values NOT read):
```
for S in 2026 2027; do
 $PY -B v4int/reeval.py --ckpt $V/runs_p3/SASREC/02B_SASREC_s$S/sealed/checkpoints/best.pt --expect-sha256 <lock sha> --ckpt-kind central --variant SASREC_D256_B3 --recipe-id 02B_SASREC --seed $S --role MAIN --arm-label C_BIG --selection-rule PRACTICAL_BEST --checkpoint-id 02B_SASREC_MAIN_s$S:best@3.0 --gpu $G --out-dir $R/REEVAL_A40/C_BIG_s${S}_PB --history-policy USER_HISTORY_L50
 $PY -B v4int/reeval.py --ckpt $V/runs_p3/SASREC/02B_SASREC_s$S/sealed/checkpoints/endpoint_6.0.pt ... --selection-rule CONTROLLED_BUDGET --checkpoint-id 02B_SASREC_MAIN_s$S:endpoint_6.0 --out-dir $R/REEVAL_A40/C_BIG_s${S}_CB
 $PY -B v4int/reeval.py --ckpt $V/runs_ds/C_SMALL_s$S/sealed/checkpoints/best.pt --ckpt-kind central --variant SASREC_D64_B2 --recipe-id SMALL_SASREC_D64_B2 --seed $S --role MAIN --arm-label C_SMALL --selection-rule PRACTICAL_BEST --checkpoint-id SMALL_SASREC_D64_B2_MAIN_s$S:best --gpu $G --out-dir $R/REEVAL_A40/C_SMALL_s${S}_PB --history-policy USER_HISTORY_L50
 $PY -B v4int/reeval.py --ckpt $V/runs_ds/C_SMALL_s$S/sealed/checkpoints/endpoint_6.0.pt ... --selection-rule CONTROLLED_BUDGET --out-dir $R/REEVAL_A40/C_SMALL_s${S}_CB
done
```
    Record wall time per evaluation (= the cost of one reeval, used in §11).
3.5 **Smokes** (each ≈ 5–15 min; SMOKE_NOT_EVIDENCE; weights deleted; note GPU util % and memory from `nvidia-smi` sampled every 10 s,
    and `visits/s` from the smoke summary):
```
$PY -B v4int/launch_fl.py --arm FA --seed 2026 --gpu $G --run-dir $R/smoke/FA --history-policy USER_HISTORY_L50 --smoke --smoke-rounds 3 --lr-mult 1
$PY -B v4int/launch_fl.py --arm FA --role CALIBRATION --lr-mult 1 --seed 2026 --gpu $G --run-dir $R/smoke/FA_CALIB --history-policy USER_HISTORY_L50 --smoke --smoke-rounds 3
$PY -B v4int/launch_fl.py --arm PF --seed 2026 --gpu $G --run-dir $R/smoke/PF --history-policy USER_HISTORY_L50 --smoke --smoke-rounds 3 --lr-mult 1
$PY -B v4int/launch_fl.py --arm FA_Q8 --seed 2026 --gpu $G --run-dir $R/smoke/FA_Q8 --history-policy USER_HISTORY_L50 --smoke --smoke-rounds 3 --lr-mult 1
$PY -B v4int/launch_fl.py --arm S_CAL --seed 2026 --gpu $G --run-dir $R/smoke/S_CAL --history-policy USER_HISTORY_L50 --smoke --smoke-rounds 3 --lr-mult 1
$PY -B v4int/launch_fl.py --arm DP --epsilon 8 --seed 2026 --gpu $G --run-dir $R/smoke/DP8 --history-policy USER_HISTORY_L50 --smoke --smoke-rounds 3 --lr-mult 1 --clip-record $R/smoke/S_CAL/S_RECORD.json
$PY -B v4int/launch_device.py --arm LO --seed 2026 --gpu $G --run-dir $R/smoke/LO --history-policy USER_HISTORY_L50 --role MAIN --smoke --smoke-clients 8
$PY -B v4int/launch_device.py --arm FT_C --seed 2026 --gpu $G --run-dir $R/smoke/SEL_FT_C --history-policy USER_HISTORY_L50 --role UNSEEN_VAL --select --base $V/runs_ds/C_SMALL_s2026/sealed/checkpoints/best.pt --base-kind central --smoke --smoke-clients 8
bash $V/runs_ds3/cfull/job_cfull_dryrun.sh 2026 $G          # DATA-08 job 3 (CPU-heavy provider build + config hash; no training)
```
3.6 **Fill the cost table and choose the lane count** (formula; INFERENCE columns replaced by measurements):
    - FL run cost = visits_total / visits_per_s; visits_total: FA/FP/PF/FA_Q8 6,827 × 57.6 ≈ 393k; LR-screen 1,707 × 57.6 ≈ 98k;
      FA_1024/DP 384 × 1,024 ≈ 393k; S_CAL 20 × 1,024 ≈ 20k. Plus 12 evaluations × the §3.4 reeval time.
    - Device cost = clients × cells × s/client (LO 3 cells; FT 2 trajectories × 4 passes + 3 rankings). MAIN clients = 40,764 E2E
      users (+ up to 51,211 with rankable rows); SEL 12,792; UNSEEN_HOLDOUT 12,862; SPARSE/HEAVY 16,384 (HEAVY support ≈ 185 rows/user);
      TRUE_COLD 381,295 (no support: start state only ⇒ a ranking pass).
    - Lanes = floor(0.85 / max GPU utilisation of one FL job), capped at 3; memory check: lanes × peak < 40 GB. C_FULL always alone.
    - Pre-measurement INFERENCE (mid): FL job 6–9 h at 30–50 % utilisation ⇒ 2–3 lanes; LO/FT 4–10 h per seed on MAIN; reeval 1 min;
      C_FULL 6–11 h (+33 % with continuation); TEST look ≈ 1–1.5 days.

---
## 4. LR screen (06B §1; the ONLY read before §9; 3 lanes in parallel; ≈ 2–4 h wall INFERENCE)
```
for M in 0.5 1 2; do (nohup setsid $PY -B v4int/launch_fl.py --arm FA --role CALIBRATION --lr-mult $M --seed 2026 --gpu $G --run-dir $R/LRSCREEN/x$M --history-policy USER_HISTORY_L50 --allow-shared-gpu > $R/LRSCREEN/x$M.log 2>&1 < /dev/null &); done
# when all three RUN_SUMMARY.json say DONE_MAIN:
$PY -B v4int/lr_screen.py select --runs $R/LRSCREEN/x0.5 $R/LRSCREEN/x1 $R/LRSCREEN/x2 --out $V/contracts/fl/LR_SELECTION.json
```
Acceptance: prints only the selected m and the file sha; fetch locally; DL row with the three CALIB values (they are the registered
screen values, allowed to be read) and the selected m. Every later FL command carries `--lr-selection $V/contracts/fl/LR_SELECTION.json`.

---
## 5. Federated queue (lanes L1..Ln share GPU G with `--allow-shared-gpu`; order = 06D §8; resume = same command + `--resume`)
Lane scripts `$R/queues/lane_<k>.sh` (LF; each job line: `$PY -B ... ; echo "JOB <name> rc=$?" >> $R/queues/lane_<k>.log`; last line `echo QUEUE_DONE rc=0`).
```
FL="$PY -B v4int/launch_fl.py --gpu $G --history-policy USER_HISTORY_L50 --allow-shared-gpu --lr-selection $V/contracts/fl/LR_SELECTION.json"
L1: $FL --arm FA --seed 2026 --run-dir $R/FL/FA_s2026 ; $FL --arm PF --seed 2026 --run-dir $R/FL/PF_s2026 ; $FL --arm FA_Q8 --seed 2026 --run-dir $R/FL/FA_Q8_s2026
L2: $FL --arm FA --seed 2027 --run-dir $R/FL/FA_s2027 ; $FL --arm PF --seed 2027 --run-dir $R/FL/PF_s2027 ; $FL --arm FA_Q8 --seed 2027 --run-dir $R/FL/FA_Q8_s2027
L3: $FL --arm FP --seed 2026 --run-dir $R/FL/FP_s2026 ; $FL --arm FP --seed 2027 --run-dir $R/FL/FP_s2027
then (after L1's FA_s2026 exists, any free lane):
S_CAL: $FL --arm S_CAL --seed 2026 --run-dir $R/FL/S_CAL   → copy S_RECORD.json to $V/contracts/dp/S_RECORD.json (O_EXCL) + DL row BEFORE the first DP launch
DP lane: $FL --arm FA_1024 --seed 2026 --run-dir $R/FL/FA_1024_s2026 --clip-record $V/contracts/dp/S_RECORD.json ; ... --seed 2027 ;
         $FL --arm DP --epsilon 8 --seed 2026 --run-dir $R/FL/DP8_s2026 --clip-record ... ; ... --seed 2027 ;
         $FL --arm DP --epsilon 1 --seed 2026 --run-dir $R/FL/DP1_s2026 --clip-record ... ;
         [NICE: --epsilon 4 (write contracts/fl/DP_Z_EPS4.json from fl.accountant first + DL row); DP1 s2027; FA_X2 if flagged]
```
Acceptance per run: `RUN_SUMMARY.json` status DONE_MAIN (S_CAL: DONE_S_CAL), 12 marks, `RESULT_ROWS.json`, `ckpt/best.pt` + `endpoint_6.0.pt`;
`LAUNCH_0.json` arm_spec_sha256 equals the `--dry-run` digest recorded before launch. Values never read (only STATUS / rate fields).
INFERENCE wall: core FL 8 runs ≈ 16–24 h on 3 lanes; S_CAL ≈ 0.5 h; FA_1024/DP 5 runs ≈ 10–20 h on 2–3 lanes.

---
## 6. Device queue (shares the card; starts after §3.4; SEL before any MAIN/panel application)
```
DEV="$PY -B v4int/launch_device.py --gpu $G --history-policy USER_HISTORY_L50 --allow-shared-gpu"
6.1 SEL_FT_C : $DEV --arm FT_C --seed 2026 --role UNSEEN_VAL --select --base $V/runs_ds/C_SMALL_s2026/sealed/checkpoints/best.pt --base-kind central --run-dir $R/DEV/SEL_FT_C
              → copy SELECTION.json to $V/contracts/device/SELECTION_FT_C.json (O_EXCL) + DL row (the cell only; no metric printed)
6.2 LO MAIN  : for S in 2026 2027: $DEV --arm LO --seed $S --role MAIN --run-dir $R/DEV/LO_s$S --n-shards 64 [--shards a:b in 2 processes]
6.3 FT_C MAIN: for S: $DEV --arm FT_C --seed $S --role MAIN --run-dir $R/DEV/FT_C_s$S --base $V/runs_ds/C_SMALL_s$S/sealed/checkpoints/best.pt --base-kind central --selection $V/contracts/device/SELECTION_FT_C.json --n-shards 64
6.4 SEL_FT_FA (after FA_s2026 DONE): $DEV --arm FT_FA --seed 2026 --role UNSEEN_VAL --select --base $R/FL/FA_s2026/ckpt/best.pt --base-kind fl --run-dir $R/DEV/SEL_FT_FA → contracts/device/SELECTION_FT_FA.json + DL row
6.5 FT_FA MAIN: for S: $DEV --arm FT_FA --seed $S --role MAIN --run-dir $R/DEV/FT_FA_s$S --base $R/FL/FA_s$S/ckpt/best.pt --base-kind fl --selection $V/contracts/device/SELECTION_FT_FA.json --n-shards 64
6.6 Panels (seed 2026; 2027 if time): for ROLE in UNSEEN_HOLDOUT SPARSE_PANEL HEAVY_PANEL TRUE_COLD: LO / FT_C / FT_FA as 6.2/6.3/6.5 with --role $ROLE --run-dir $R/DEV/<arm>_<ROLE>_s2026
6.7 Study A observational panels (evaluation only, reeval): for ROLE in UNSEEN_HOLDOUT SPARSE_PANEL HEAVY_PANEL TRUE_COLD: reeval C_BIG (best), C_SMALL (best), FA (best), DP8 (endpoint) → $R/REEVAL_A40/<arm>_<ROLE>_s2026_*
```
Acceptance: `RESULT_ROWS.json` per run; SELECTION files O_EXCL before any application (the launcher refuses otherwise); causality
refusals never fire (a refusal = STOP and report). INFERENCE: 6.1 ≈ 1–3 h; 6.2/6.3/6.5 ≈ 4–10 h per seed each; panels ≈ 1–2× MAIN per role.

---
## 7. C_FULL seed 2026 (exclusive window; after DATA-08 receipts + the §3.5 dry run; INFERENCE 6–11 h, +33 % if continuation)
Pre-launch: `nvidia-smi` shows OUR card free of our jobs (pause the lanes between jobs: they resume with `--resume`);
```
mkdir -p $R/CFULL/C_FULL_02B_SASREC_s2026 && (nohup setsid $PY -B $CR/v4int/c_full.py --ext-registry $V/contracts/manifest_registry/REGISTRY_V2_FULL_EXT.json -- --family SASREC --seed 2026 --phase MAIN --recipe-id 02B_SASREC --gpu $G --run-dir $R/CFULL/C_FULL_02B_SASREC_s2026 --history-policy USER_HISTORY_L50 > $R/CFULL/s2026.log 2>&1 < /dev/null &)
```
Acceptance: `LAUNCH_0.json` config_sha256 == the dry-run hash; STATUS DONE_MAIN or DONE_CONTINUATION; sealed/checkpoints best.pt +
endpoint_6.0.pt; then reeval both on MAIN (role MAIN, arm C_FULL) into `$R/REEVAL_A40/C_FULL_s2026_{PB,CB}` — the sealed run's own
rows are A40 rows already; the reeval is the uniform path. Seed 2027 = NICE. Overnight is the natural window.

---
## 8. CPU and analysis jobs (parallel throughout; ≤ 8 cores, nice 15)
8.1 Baselines: `$PY -B -m v4base.run --baseline KNN_EMB --phase MAIN --allow-main-validation --item-table $V/runs_p3/SASREC/02B_SASREC_s2026/sealed/checkpoints/best.pt --family SASREC --table-role BIG --out $R/BASE/KNN_EMB_BIG` and `--item-table $V/runs_ds/C_SMALL_s2026/sealed/checkpoints/best.pt --table-role SMALL --out $R/BASE/KNN_EMB_SMALL` (RULE done in runs_ds/baselines; POP/LAST/SKNN in runs_p3/baselines_main).
8.2 ONNX: `$PY -B -m v4dev.dump_rows --role MAIN --history-policy USER_HISTORY_L50 --out $R/ONNX/rows.npz` → `$PY -B -m v4dev.bench --ckpt $V/runs_ds/C_SMALL_s2026/sealed/checkpoints/best.pt --family SASREC --variant SASREC_D64_B2 --v1-root <v1 mappings> --out $R/ONNX --rows $R/ONNX/rows.npz --manifest-n <rows.meta.json manifest_n> --tag onnx_int8` (if the single-thread bench exceeds 2 h: `--first-users 16384`, 06A §6). Then `$PY -B -m v4dev.browser_queries --rows $R/ONNX/rows.npz --ckpt ... --out $R/ONNX/queries.json`; fetch model_fp32.onnx, model_int8.onnx, queries.json for the owner (§8.5).
8.3 DATA-07: `$PY -B v4int/launch_trunc.py --study T41 --panel WARM_TRUNCATION --run-dir $R/TRUNC/T41_WT` and `--panel COLD_H --run-dir $R/TRUNC/T41_CH` (CPU, 20–40 min each) → PASS with the 7 admitted cells only; Study A (GPU, eval-only, shares): `--study A --arm C_BIG|C_SMALL|FA|DP --seed 2026 --gpu $G --run-dir $R/TRUNC/SA_<arm>_s2026 --ckpt <locked> --t41-receipt $R/TRUNC/T41_WT/T41_RECEIPT.json`; Study B (GPU): `--study B --arm FT_C --seed 2026 ... --selection $V/contracts/device/SELECTION_FT_C.json --base ... --t41-receipt $R/TRUNC/T41_CH/T41_RECEIPT.json`, LO likewise; E2X afterwards. SHOULD-medium: run when lanes are free.
8.4 Strata / bins: nothing to run (cut points registered; ANA-01 joins by client_key_hex).
8.5 Owner's PC: `cd code/v4dev/browser && python -m http.server 8000` → Edge → `http://localhost:8000/` → load model_fp32.onnx / model_int8.onnx / queries.json → Run → Download BROWSER_LATENCY.json → put it in `company/owner/` (label HISTORICAL_REFERENCE).

---
## 9. First read — ANA-01 on VALIDATION (all arms at once, after the ANA freeze)
```
$PY -B tools/ana_block_index.py --split VALIDATION --device-class NVIDIA_A40-FP32-DET --reeval $R/REEVAL_A40 --fl $R/FL --dev $R/DEV --base $R/BASE $V/runs_ds/baselines $V/runs_p3/baselines_main --onnx $R/ONNX --trunc $R/TRUNC --records contracts/strata/STRATA_CUTPOINTS_MAIN.json contracts/dp/S_RECORD.json contracts/device/SELECTION_*.json contracts/fl/LR_SELECTION.json code/tests_fl/receipts/DP_WOR_SENSITIVITY_FL06.json --out $R/ANA/block_val   # reads paths/status/hashes only, never a metric
$PY -B tools/retention_report.py --block $R/ANA/block_val --split VALIDATION --contrasts $V/company/gates/ANA_CONTRASTS.json --out $R/ANA/out_val_check --expect-script-sha256 <frozen> --check-only
$PY -B tools/retention_report.py --block $R/ANA/block_val --split VALIDATION --contrasts $V/company/gates/ANA_CONTRASTS.json --out $R/ANA/out_val --expect-script-sha256 <frozen>
```
Acceptance: `--check-only` exit 0 (comparison keys equal, rules paired, ranks reproduce rows); the report is the FIRST read; DL row;
fetch RETENTION_RESULT.json + REPORT.md + curves locally; thesis reference batch (Appendix C tables). If an arm is missing it is
declared "not run" (06D §8), never substituted.

---
## 10. ALL_METHODS_LOCK (O_EXCL; `company/gates/ALL_METHODS_LOCK.json`)
Content: every arm in the block index with `RESULT_ROWS` sha256, rank-file sha256, checkpoint sha256 (best + endpoint), the SELECTION /
S_RECORD / LR_SELECTION / DP_Z / pin file hashes, the ANA freeze sha, the contrast list (verbatim 06B §2 + 06D §3/§4 rule map), the
comparison key (A40 class), the code_ds3 (and code_ds4) tree hashes, the list of arms NOT run, the TEST manifests to build (06D §2), and
the "device arms re-run their locked procedure" statement. Then `tools/test_exposure_register.py append --purpose MANIFEST_BUILD
--lock company/gates/ALL_METHODS_LOCK.json ...` (entry 5) → DL row → upload the lock + register to the server. **After this: no
recipe, arm, selection or script changes.**

---
## 11. The single TEST look (≈ 1–1.5 days INFERENCE; owner informed before it starts)
```
11.1 gate + build (CPU): bash $V/runs_ds3/test/job_test_build.sh ; bash .../job_test_final.sh   → data_test/, contracts/manifest_registry/REGISTRY_V2_TEST_EXT.json (sha in DL); counts-only integrity check
11.2 checkpoints (GPU, short): reeval every locked checkpoint on --role MAIN_TEST --registry-ext <TEST_EXT>: C_BIG (PB, CB) ×2 seeds; C_FULL (PB, CB); C_SMALL (PB, CB) ×2; FA/FP/PF(+store)/FA_Q8/FA_1024/DP8 (PB, CB) ×2; DP1 (PB, CB); [DP4]; panels UNSEEN_HOLDOUT_TEST/TRUE_COLD_TEST/SPARSE/HEAVY for the §6.7 arms
11.3 device arms (GPU, the long part): LO / FT_C / FT_FA on --role MAIN_TEST × 2 seeds with the LOCKED selections (order: seed 2026 all three, then seed 2027); UNSEEN_HOLDOUT_TEST seed 2026
11.4 baselines (CPU): v4base.run --phase MAIN_TEST for POP LAST SKNN KNN_EMB(SMALL,BIG) RULE
11.5 ONNX (CPU): dump_rows --role MAIN_TEST → bench --rows (INT8 ranks on TEST rows; same fallback rule)
11.6 index + report: ana_block_index.py --split TEST ... --records ... VALIDATION_RESULT=$R/ANA/out_val/RETENTION_RESULT.json ; retention_report.py --split TEST --contrasts $V/company/gates/ALL_METHODS_LOCK.json --out $R/ANA/out_test --expect-script-sha256 <frozen>
11.7 register entry 6 (RESULT), DL row, fetch everything, thesis reference batch. No re-run except an interrupted computation with identical hashes (E10.3).
```

---
## 12. Documentation, thesis map, reproducibility package
12.1 After each phase: DL rows; RESUME_HERE state line; `thesis_reference/tools/add_source_batch.py --list <batch.json>` (new IDs from
S470), EN reference §14.16+, STATE.md, CHANGELOG.md, `tools/verify_reference.py` 0 errors. After the TEST look: the structured
registers (DECISION/EXPERIMENT/INCIDENT/CLAIM) in one batch (a Sonnet script in the style of `tools/update_registers_*.py`).
12.2 **Thesis chapters → artefacts** (structure = FABLE_THESIS_RESEARCH_REVIEW §E.2; every number printed by a script):
| Chapter / item | Artefact |
|---|---|
| 1 Intro: question, ladder, contributions, scope; **"What changed from the proposal and why"** (tasks, datasets, backbones, FedBN, AUC, metrics: one row each with the reason) | proposal PDF; 06D §9g; ARCHITECTURE_LOCK |
| 4 Data: Table 4.1 split counts; Fig 4.1 timeline (Black Friday marked); Table 4.2 strata; Table 4.3 S* channels by locality | DATA_PROTOCOL_LOCK_V2, REGISTRY_V2, CENSUS-02, FS_FINAL.json, EVALUATION_CONTRACT §11.4 |
| 5 Methods: Table 5.1 privacy ladder (arm × leaves-device train/infer × guarantee × bytes/visit); knobs table; DP mechanism + accountant + both ε bounds + the unaccounted list + the noise note; selection on UNSEEN_VAL; K1–K16 + hierarchy | 06B §3, 06 §2, 06C, 06D §3–§4, RUN_SUMMARY bytes, DP_Z_RECORD, DP_WOR_SENSITIVITY_FL06 |
| 6 Setup: Table 6.1 run inventory (run id, seed, GPU, hours, config sha), A16/A40 note, the pin | LAUNCH_*.json, RUN_SUMMARY, EVAL_DEVICE_PIN_A40 |
| 7.1 Contest: Table 7.1 (PRACTICAL_BEST + 6.0 rows with CIs); Fig 7.1 the 12-mark curves of the 8 runs | G2_RESULT.json, cond5 diagnostics, sealed RESULT_ROWS/metrics |
| 7.2 **Main retention table (TEST)**: arm / seeds / mean / R vs C_BIG [CI] / R vs C_SMALL / R vs C_FULL / NDCG@10 / HR@10 / bytes per client / ε / size; still-improving flag; device-class sensitivity (A16 vs A40 for C_BIG/C_SMALL) | out_test/RETENTION_RESULT.json + REPORT.md; REEVAL_A40 vs sealed A16 rows |
| 7.3 Curves: R vs cumulative bytes (marks); R vs ε (∞, 8, 1, [4]); R vs size (BIG→SMALL→INT8; FA→FA_Q8 bytes); quality vs rounds | out_test/curves/*.csv |
| 7.4 Personalisation: FT_C−C_SMALL, FT_FA−FA, PF−FA, harmed-client share | out_test (K9–K11, personalisation block) |
| 7.5 Persona/strata: 3×3 heat-map; diversity, loyalty, activity tables (descriptive) | out_test/curves/strata_* |
| 7.6 Cold start and unseen users: R vs support bin; events-to-90/95; UNSEEN_HOLDOUT participation gap; truncation Study A/B if run | out_test bins; TRUNC/SA_*, SB_*, E2X |
| 7.7 Device: ONNX FP32/INT8 size, latency, RSS, parity, INT8 retention (K16); browser numbers | ONNX/BENCH_RESULT.json, BROWSER_LATENCY.json |
| 7.8 Baselines table | out_test baseline rows |
| 8 Discussion; 9 Threats (internal: screened LR only, μ untuned, checkpoint-rule dependence, A16→A40, max-of-marks on VALIDATION; external: one retailer, 46 days, Black Friday, 5–100 users, cohort size; construct: MRR@20, privacy by construction, no attack, battery; statistical: 2 seeds, user bootstrap scope, 16 confirmatory contrasts, previously inspected TEST); 10 Conclusion / future work | 06A §9, 06B §7, 06D §9, CLIENT_CONTRACT C5.6 |
| Appendices: A S* dictionary; B pre-registration record (all locks/amendments with hashes, the TEST register); C VALIDATION tables; D DP appendix; E run inventory + hashes; F reproducibility package; G ethics/data statement; H incident summary | contracts/, company/gates/, out_val, DP receipts, code tree hashes |
12.3 Reproducibility package (owner approval before any push): the code trees' hashes (code_p3, code_p3b, code_ds, code_ds2, code_ds3[/4]),
contracts/, company/gates/, company/registers/, the thesis reference registers, a README with the exact commands of this plan and of the
handoffs, seeds, the frozen dataset hash 860e5a1c…, and how to rebuild data_v3/data_cfull/data_test from the frozen ledger.
12.4 Bibliography fixes (proposal refs [5], [6], [8], [12]) and the new citations (Charles et al. 2021; Andrew et al. 2021; Wang, Balle &
Kasiviswanathan 2019; McMahan et al. 2018; Mironov 2017; MTZ 2019) — all verified by web search in this or the previous review.

---
## 13. Critical path, cut list, what to do if time runs out
Critical path: §1 (1–2 d) → §2–§3 (½ d) → §4 (½ d) → §5 core FL + §6 device (3–4 d wall with 3 lanes) → §7 C_FULL (½ d, overlapped) →
§9 → §10 → §11 (1–1.5 d) → §12 (1 d). **INFERENCE total ≈ 8–10 days after O1/O2** for the MUST + SHOULD-high set; ≈ 5–6 days for MUST only.
Cut order (last-to-first; 06D §8): NICE items → Study B → Study A/T41 → panels seed 2027 → FA_Q8 s2027 → DP1 → HEAVY/SPARSE panels →
FP/PF s2027 → C_FULL → FT_FA → TEST device arms seed 2027 (K6–K8 become 1-seed on TEST, labelled) → FA_1024/DP8 (the privacy curve
collapses to the ladder without ε points; disclosed) → FP/PF s2026. Never cut: C_BIG/C_SMALL re-evaluation, LR screen, FA ×2, LO ×2,
FT_C ×2, INT8, the VALIDATION preflight, the lock, the TEST look. Every cut item is listed in the thesis as "registered, not run".

---
## 14. What the owner must do (checklist)
1. Reply with O1 (GPU index) and O2 (go). 2. Give the submission date (O3). 3. Run the browser benchmark when the ONNX files arrive
(§8.5). 4. Fix the bibliography entries in the thesis. 5. Decide on the GitHub release at the end (O5). 6. Keep the teammate informed
about our CPU/RAM use (≤ 8 cores, ≤ 180 GiB). 7. Never delete `runs_*` or `code_*` trees on the server.

---
# ADDENDUM V5.1 — 2026-09-29T08:50Z (Claude Fable 5.1): TWO A40s, measured rates, what Codex did overnight, corrected lanes
Supersedes §0 O1/O2 and the one-card scheduling of §3.6, §5–§7, §13. The 06D §8 tiers (MUST / SHOULD / NICE) and every
registration gate are unchanged. Execution is owned by the Ops session (peer "eda-7b"); this addendum is the science/plan side.

## A. What is DONE (MEASURED, DL 162–168; values unread)
| Item | State | Evidence |
|---|---|---|
| DATA-08 C_FULL manifest | built + final receipt PASS (1,348,682 users / 26,677,600 decisions); REGISTRY_V2_FULL_EXT fetched | data_cfull/receipts; DL 162 |
| T3 evaluator receipt on the A40 | COMPLETE, all checks pass (two invocations bitwise, pass-2 recompute bitwise, single block = logits); ranks differ from the A16 bundle in 12,635 (GRU) / 14,063 (SASREC) rows → a separate device class is REQUIRED (06D §1 confirmed) | runs_ds3/T3_A40_r2/out/T3_GPU_RECEIPT.json; company/receipts/phase1/ |
| STRICT determinism on the A40 (D64_B2 + D256_B3, 2 reps) | PASS 00:37:17Z | runs_ds3/strict_a40/strict.status |
| Device pin | written 00:37:27Z (self db744ca3…; canonical-JSON rule corrected in code_ds3r2 before the pin) | contracts/eval_device/EVAL_DEVICE_PIN_A40.json |
| C_BIG / C_SMALL re-evaluation on the A40 | 8 runs (PB + CB × seeds), rows + ranks written, ≈ 3 min each | runs_ds3/REEVAL_A40/ |
| LR screen (06B §1) | x0.5 DONE (2 h 39 m), x1 DONE (2 h 40 m), x2 RUNNING since 06:21Z (ETA ≈ 09:00Z INFERENCE) | runs_ds3/LRSCREEN/ |
| Smoke gate (FA r20, FA CALIB r20, PF r10, FA_Q8 r10, S_CAL r3, DP8 r3, LO c64, SEL_FT_C c64) | all rc 0 | runs_ds3/pre/gate_rest.log |
| INT-06 | PARTIAL by Codex (pin, trainer/evaluator/rows/launcher provenance, central-checkpoint reeval on MAIN, STRICT bench paths); FL/PF checkpoint reeval and panel roles missing → being completed by the Ops session's INT-06 fix agent; independent reviews REV-FL-01 / REV-ANA-01 and DATA-09 running | DL 164, 168 |
Codex deviations to keep in mind (DL 168): everything was run SERIALLY (the LR screen took ~8 h instead of ~3 h); the overnight
queue omitted FP, LO and FT_C (06D §8 MUST order); it was stopped before any production run started. No registered value was read.

## B. Measured cost model (replaces the INFERENCE column of §3.6)
- FL visit throughput: **≈ 10.2 visits/s per process** (LR-screen x0.5 / x1: 98,323 visits in 9,568 / 9,632 s), single process
  at GPU 19 % utilisation and 1 CPU core at 100 % → the FL loop is CPU-bound; INFERENCE: 5–8 processes per card saturate a card.
- FA-class MAIN run (393,235 visits) ≈ **10.7 h per process**; LR-screen run ≈ 2.7 h; S_CAL (20,480 visits) ≈ 35 min;
  FA_1024 / DP (393,216 visits) ≈ 10.7 h; plus 12 evaluations × ≈ 3 min (measured reeval time on MAIN_VALIDATION).
- Central re-evaluation: ≈ 3 min per checkpoint (A40).
- Device arms: not yet measured at scale (smoke LO c64 / SEL_FT_C c64 rc 0 — take s/client from runs_ds3/pre/smoke/*/SMOKE_SUMMARY.json
  before sizing the lanes); C_FULL: measure from its first 0.5-EFE mark (INFERENCE 6–11 h + 33 % if continuation).
- Totals (INFERENCE): 13 FL runs × 10.7 h = 139 process-hours → ≈ 1.2 days on 5 lanes of one card; device arms ≈ 1–2 days on the
  other card after C_FULL; TEST look ≈ 1–1.5 days. With both cards: ≈ 4–6 days to the TEST numbers after the reviewed tree lands.

## C. Corrected lane layout (both cards; registered constraints)
- **GPU 1 = C_FULL alone** until its RUN_SUMMARY.json exists: `launch_central.py:335-336` refuses `--allow-shared-gpu` for a non-smoke
  MAIN run and `gpu_checks` (L352-353, v4sup.gpu.gpu_ready) refuses any co-resident compute process (amendment 06 §4). Sequence on
  GPU 1: DATA-08 dry run → C_FULL s2026 → (then) the device lanes and panels.
- **GPU 0 = FL lanes (5) + device lanes (2–3) with `--allow-shared-gpu`** until C_FULL finishes; every FL lane waits for
  `contracts/fl/LR_SELECTION.json` (lane0 = lr_screen select after x2, DL row with the three CALIB values); the DP lane waits for
  `contracts/dp/S_RECORD.json` (Ops copies S_CAL's record O_EXCL + DL row); FT lanes wait for `contracts/device/SELECTION_FT_C.json`
  / `SELECTION_FT_FA.json` (O_EXCL + DL row after SEL runs).
- Order inside the lanes = 06D §8: FA ×2, LO ×2, FT_C ×2 first; then FP, PF, FA_Q8, S_CAL → FA_1024 → DP8 → DP1; FT_FA after FA s2026;
  panels seed 2026; NICE items last. Every lane stops on the first failure and resumes with the same command + `--resume`.
- Production runs only from the reviewed tree (code_ds3r3 per the Ops session) — never from code_ds3pre / r1 / r2.
- CPU jobs in parallel (128 cores available): KNN_EMB BIG/SMALL, dump_rows + ONNX bench, DATA-07 T41 (both panels), then Study A/B on
  a free GPU slot.

## D. Open scientific items for this phase (owner-visible)
1. Whether code_ds3r2's eight re-evaluation rows remain valid after INT-06's completion (row-label / hash rule changes) — if the
   fixed reeval.py changes any recorded field, re-run the 8 evaluations from the reviewed tree (≈ 25 min) and supersede the old dirs
   (never delete; DL row).
2. The LR screen ends with lr_screen.select — the ONLY read before §9; its three CALIB values go into the DL row.
3. ANA-02 (rule map K12/K13, hierarchy, still-improving flag, block-index builder) must land before the ANA freeze; INT-07 before the
   TEST look; DATA-09 code before ALL_METHODS_LOCK; all four get one independent review.

## E. Two-session operating split (2026-09-29T08:40–08:48Z; owner: "Fable has full authority; the two sessions collaborate")
- **eda-7b (Ops, Opus) = server operator:** lanes on both A40s, the registration gates (LR_SELECTION, S_RECORD, SELECTION_FT_C /
  FT_FA / LO) with their DL rows, deploy of code_ds3r3, RESUME_HERE. Launched 08:40Z from code_ds3r2 (allowed by REV-FL-01):
  lane0 LR select after the screen; GPU0 lane1 FA→FA_Q8 s2026, lane2 FA→FA_Q8 s2027, lane3 FP ×2, lane4 S_CAL→FA_1024 ×2,
  lane5 DP8 ×2→DP1 s2026; GPU1 lane6 C_FULL dry run→C_FULL s2026 (alone on the card); laneC KNN_EMB BIG→SMALL. Waiting for
  code_ds3r3: laneP PF ×2 (B1/B2), lane7 SEL_FT_C→LO s2026→FT_C s2026, lane8 SEL_LO→LO s2027→FT_C s2027, lane9 SEL_FT_FA→FT_FA ×2,
  laneT T41 WT/CH (after DATA-07R1). Workers: INT-06 (+FL-08 items), DATA-09, ANA-02.
- **eda-3b (Fable) = chief scientist:** amendments, packets, scientific acceptance, the deep review, the thesis reference, ONE
  independent delta review before code_ds3r3 (`company/tasks/REV-DELTA-01.md`). Workers: DATA-07R1, DEV-04.
- Rules: DL row = tail max + 1 at write time; no server launch/kill without telling the other; announce a task id before dispatch;
  one owner per file set (map in DL 172).
