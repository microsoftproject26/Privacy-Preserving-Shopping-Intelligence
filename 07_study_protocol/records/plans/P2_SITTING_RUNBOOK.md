# P2 sitting runbook: from "RUN-3 finished" to the single T0a TEST sitting and the Phase-2 report (P2-SITTING, 2026-10-04)

Status: PLAN. Nothing in this file has been run. Every duration is INFERENCE unless marked MEASURED.
Binding: contracts/PREG2.md (be0561ab…) §6.3 S1–S8, §9.5 S7, §13 items 4–6; reviews/FABLE_P2_SCRIPT_FREEZE_20261004.md
(FREEZE_WITH_EDITS; conditions i–iii); DL 467 (Part B reduced), DL 469 (E1), DL 471–478.
Scripts: `company/plans/ops_scripts/p2/p2_lock_plan_build.py` (the lock plan), `laneP2_SIT.sh` (the sitting),
`p2_lane_lib.sh` (scoring-lane change), `tools/p2_lock.py` (S1 free fields). Handoff: `company/handoffs/P2-SITTING.md`.

Rules that never bend: no Phase-2 MAIN VALIDATION value is opened by a human before both sitting analyses exist (S8);
nothing under `runs_p2/TEST`, `runs_ds3/TEST`, `runs_ds3/ANA/*test*` or `p2_out_test_*` is fetched or opened before
both RETENTION_RESULT files exist; Ops writes the DL row first and fetches second (S3); there is no second sitting on
RETRO_TEST (S4); every refusal is a STOP for review, never a bypass.

```bash
# server variables used below
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
L=$V/company/plans/ops_scripts/p2; PY=/home/jovyan/venvs/ppsi-gpu/bin/python; Q=$V/runs_ds3/queues
```

## 0. Timeline at a glance (INFERENCE)

| # | step | where | wall time | critical path |
|---|---|---|---|---|
| 1 | RUN-3 + C2 finished; deploy P2-SITTING files | server | 10 min | yes |
| 2 | build the lock plan, dry-run the lock, write the candidate lock | server | 5–10 min | yes |
| 3 | **Fable touch point 2: P2_METHODS_LOCK review** | local | 30–90 min | yes |
| 3a | in parallel: E1 audit (GPU), E4 SecAgg (CPU), device export / measurement prep | server | 30–60 min / 5 min / owner time | no |
| 4 | O_EXCL lock copy, register entries 7 + 8, DL row, upload, server checks | local + server | 15 min | yes |
| 5 | sitting: `laneP2_SIT.sh --launch` (12 steps) | server | ≈ 16–28 h | yes |
| 6 | DL row (shas only) → fetch → register look entries → thesis tables | local | 1 h | yes |
| 7 | benchmark LOCKs and their single TEST looks (LOCK_FINAL_RUNBOOK) | server | per dataset | after 5 |

Sitting step durations (INFERENCE; the reeval estimate is INT-07's 3–4 min per checkpoint, the device estimate RUN-3's
≈ 1 GPU-h per FT run):

| step | what | estimate |
|---|---|---|
| 1 preflight | lock / tree digest (724 files) / freeze / checkpoints present | 1–2 min |
| 2 register | chain verify + entry checks | < 1 min |
| 3 p2_dry | 24 reeval_p2 + 2 launch_device_p2 dry runs, CPU, one at a time | 5–15 min |
| 4 h_restore | chmod 555 + sha | seconds |
| 5 h_reeval_dry | 109 Phase-1 dry runs (H plan) | 15–40 min |
| 6 h_scoring | H reeval 109 × 3–4 min on one GPU ∥ H base (CPU) ∥ H onnx (CPU) | 5.5–7.5 h |
| 7 h_dev | 14 Phase-1 device runs, sequential on one GPU (frozen H_test.sh) | 8–16 h |
| 8 p2_scoring | 24 MAIN_TEST reevals + 2 FT_P2_WSTAR, 2 GPUs × 4 slots | 1–1.5 h |
| 9 h_analysis | TEST index + retention_report --split TEST + metric table | 20–60 min |
| 10 p2_index | P2 TEST block | 5–10 min |
| 11 p2_report | P2 analysis (1,000 resamples) | 10–30 min |

The Phase-1 steps 6–7 are the long pole. H_test.sh is frozen and runs its reevals and device runs on one GPU each, and
S3 forbids Phase-2 scoring before all Phase-1 scoring, so the second GPU idles during H. The watchdog (v3, 25ece596)
pauses ML-1M and Wave 2 while any `bash …H_test.sh` runs, and holds ML-1M on the GPUs that run reeval_p2 /
launch_device_p2 in step 8.

## 1. After RUN-3: check and deploy (critical path)

1. Value-free completion check (STATUS words only, no sealed/ file):
   ```bash
   bash $L/laneP2_RUN3.sh --plan | grep -c 'state: done'          # expect 14 (C1 x4, W* x2, S_CAL, FA1024 x2, DP8 x2, DP1, FT x2)
   grep -h '"status"' $V/runs_p2/C2/P2_C2_CFULL_SMALL_SASREC_D64_B2_s2026/STATUS.json   # DONE_CONTINUATION or DONE_MAIN
   ls $V/runs_p2/_chain_state/ | grep -E 'run3.ok|STOP'                                   # run3.ok, no STOP
   ps -eo args | grep -cE 'code_p2r[0-9]+/v4int/(launch_fl|central_ft|c_full_small|launch_device_p2)\.py'   # 1 (the grep)
   ```
   An arm that is not finished when the lock is built is declared not run (S4) with its own DL row — never silently.
2. One review of the P2-SITTING files (REV-P2-SITTING) before the deploy; then deploy them **after** the chain and
   RUN-3 have ended (the running lanes keep the old lib in memory, but do not swap files under a live lane):
   `tools/p2_lock.py`, `company/plans/ops_scripts/p2/{p2_lane_lib.sh, laneP2_SIT.sh, p2_lock_plan_build.py}`.
   Upload, compare sha256 with the handoff table, `chmod 555` the scripts and `chmod 444` p2_lock.py; keep the old
   p2_lock.py as `tools/_pre_p2r3/p2_lock.py.c3f7202b`. Never copy anything into code_p2r2 / code_p2r3 (read-only trees;
   code_p2r3's digest is what the lock binds).
3. Server smoke (seconds): `bash -n $L/laneP2_SIT.sh $L/p2_lane_lib.sh`; `bash $L/laneP2_RUN3.sh --plan | head -n 2`
   (unchanged plan); `bash $L/laneP2_SIT.sh --plan | grep -c '^step '` → 12. The tests (test_p2_sitting.py, 9; the
   regression suites) are the local evidence in the handoff.

## 2. The lock plan and the candidate lock (server; value-free)

```bash
mkdir -p $V/runs_p2/LOCK
$PY -B $L/p2_lock_plan_build.py plan --dry-run | tail -n 1                 # P2_LOCK_PLAN DRY RUN: 14 sealed (arm, seed), 2 declared not run, 4 selection files, code tree code_p2r3; nothing written
$PY -B $L/p2_lock_plan_build.py plan --out $V/runs_p2/LOCK/P2_LOCK_PLAN.json
cd $V && $PY -B tools/p2_lock.py lock --plan runs_p2/LOCK/P2_LOCK_PLAN.json --out runs_p2/LOCK/P2_METHODS_LOCK.candidate.json --dry-run
$PY -B tools/p2_lock.py lock --plan runs_p2/LOCK/P2_LOCK_PLAN.json --out runs_p2/LOCK/P2_METHODS_LOCK.candidate.json
$PY -B tools/p2_lock.py verify --lock runs_p2/LOCK/P2_METHODS_LOCK.candidate.json   # {"n_sealed": 26, "n_declared_not_run": 2, "n_phase1_reused": 10, ...}
sha256sum runs_p2/LOCK/P2_LOCK_PLAN.json runs_p2/LOCK/P2_METHODS_LOCK.candidate.json
```
What the plan contains (MEASURED from the builder code; the counts assume every RUN-3 arm and C2 finished):
- `sealed`: P2_WSTAR {2026, 2027}, P2_WSTAR_FA1024 {2026, 2027}, P2_WSTAR_DP8 {2026, 2027}, P2_WSTAR_DP1 {2026},
  P2_C1_FT_CENTRAL_FULL / _FROZEN {2026, 2027}, P2_C2_CFULL_SMALL {2026}, FT_P2_WSTAR {2026, 2027} with its MAIN device
  run → 26 sealed (arm, seed, rule) in the lock (two rules each; FT on its W* base).
- `declared_not_run`: P2_WSTAR_DPFTRL8 and P2_WSTAR_SECAGG (every seed) with the DL 467 wording; plus the arms of
  P2_WARM_SELECTION_FINAL `dropped.units` (DP1, C2) if any; plus only what Ops adds with
  `--declare-not-run ARM:SEED=REASON` (or `ARM:*=REASON`) and a DL row.
- `selection_files`: P2_WARM_SELECTION_FINAL (25aadc2e…), _A (06a75d65…), P2_C1_SELECTION (20a23295…), S_RECORD_P2 of
  runs_p2/MAIN/P2_WSTAR_S_CAL_SMALL_s2026.
- `code_tree`: code_p2r3 (the deployed scoring tree, DL 477). `phase1_reused`: p2_lock DEFAULT_PHASE1_REUSED (10).
- `s1_free_fields` → copied by p2_lock into the lock's top level: `e1_validity_reading` (PRE's RAW mean-CE AUC CI must
  include 0.5; DL 469 ceiling note), `part_b_not_run_analyses` = ["E2", "E3", "E5_LDP"], `part_b_not_run_reason`
  (DL 467). This meets the freeze ruling's condition iii.
Refusals (exit 2, nothing written): an unfinished or missing run that is not declared; a declared arm whose run is
finished; FT_P2_WSTAR whose W* is declared; S_RECORD_P2 missing while a DP arm is sealed; a p2_lock.py without the free
fields. FT_P2_WSTAR without its MAIN device run needs `--ft-without-main-run` (P2-DEVLOCK open question 5).

## 3. Fable touch point 2 — P2_METHODS_LOCK (PREG2 §13 item 4 (2))

Fetch (all value-free): `runs_p2/LOCK/P2_LOCK_PLAN.json`, `runs_p2/LOCK/P2_METHODS_LOCK.candidate.json`, the `verify`
line, the sha256 lines, and the output of `bash $L/laneP2_SIT.sh --plan` (the 12 steps and their commands; it lists the
26 scoring jobs only once the lock sits at company/gates/P2_METHODS_LOCK.json, and they are derived mechanically from
`sealed_arms`: one MAIN_TEST reeval per sealed (arm, seed, rule), one FT_P2_WSTAR per seed). Fable must see:
1. the candidate lock (file sha256, self_sha256) and the plan (sha256 = the lock's `plan_sha256`);
2. sealed_arms: every (arm, seed, rule), checkpoint sha256, run id, source kind; FT_P2_WSTAR base = the sealed W* best.pt;
3. declared_not_run with reasons (DL 467 wording; any §5.9 drop; any Ops declaration and its DL row);
4. the three free fields (`e1_validity_reading`, `part_b_not_run_analyses`, `part_b_not_run_reason`) — freeze condition iii;
5. selection_files_sha256 vs DL 466 / 478, S_RECORD_P2 sha; code_tree aggregate of code_p2r3 (DL 477 tar 43ae3bb7);
   phase1_reused (10 entries bound to ALL_METHODS_LOCK 04a3cd19…); preg2 / contrasts / bootstrap seed 20261004;
6. the two register entry JSONs of step 4 (before they are appended) and the sitting order of `laneP2_SIT.sh --plan`;
7. the p2_lock.py diff (c3f7202b → the P2-SITTING sha: the free fields only) and this runbook.
Verdict needed before step 4. Any edit → a new plan, a new candidate (O_EXCL: new file names), the same review.

### 3a. In parallel with the review (no TEST row; nothing here selects an arm)

E1 — user-level membership inference (PREG2 §7.1; DESCRIPTIVE; TRAIN-window rows only). One GPU, 30–60 min
(INFERENCE, P2-14a §4). Must end before step 5 starts (S7: no Phase-2 job beside an H step) — if it is not done by then,
leave it running only if its GPU is not the H GPU, and expect E1 to read NOT_RUN in the sitting report (it is reported
DESCRIPTIVE afterwards).
```bash
CT=$V/code_p2r3; M=$V/runs_p2/MAIN; C1=$V/runs_p2/C1/P2_C1_FT_CENTRAL_FULL_MAIN_s2026; mkdir -p $V/runs_p2/E1
PRE=$V/runs_pretrain/PRETRAIN_FOR_FL_SMALL_SASREC_D64_B2_s2026
FAWF=$V/runs_ds3/FL_T/FA_WARM_FROZEN_s2026; FAWF_CK=$(ls -d $FAWF/ckpt/*/ | sed 's#/$##')   # exactly one dir (as endgame fl_ckdir)
T="PRE=$PRE/sealed/checkpoints/endpoint_6.0.pt FA_WARM_FROZEN=$FAWF_CK/best.pt \
   P2_WSTAR=$M/P2_WSTAR_SMALL_s2026/ckpt/P2_WSTAR_SMALL_s2026/best.pt \
   P2_WSTAR_DP8=$M/P2_WSTAR_DP8_SMALL_s2026/ckpt/P2_WSTAR_DP8_SMALL_s2026/endpoint_6.0.pt \
   P2_WSTAR_DP1=$M/P2_WSTAR_DP1_SMALL_s2026/ckpt/P2_WSTAR_DP1_SMALL_s2026/endpoint_6.0.pt \
   P2_C1_FT_CENTRAL_FULL=$C1/sealed/checkpoints/best.pt"
cd $CT && sha256sum v4int/audit_e1_mia.py          # bd9d6938…
CUDA_VISIBLE_DEVICES=1 V4_DATASET_ROOT=/home/jovyan/shared_data/rees46_nextitem_v1 $PY -B v4int/audit_e1_mia.py --device cuda --targets $T --out $V/runs_p2/E1/E1_AUDIT_v1 --dry-run; echo rc=$?
(CUDA_VISIBLE_DEVICES=1 V4_DATASET_ROOT=/home/jovyan/shared_data/rees46_nextitem_v1 nohup setsid $PY -B v4int/audit_e1_mia.py --device cuda --targets $T --out $V/runs_p2/E1/E1_AUDIT_v1 > $V/runs_p2_e1.out 2>&1 < /dev/null &)
sha256sum $V/runs_p2/E1/E1_AUDIT_v1/*.json         # hashes only; the value files are opened after both RETENTION_RESULT files exist
```
The record for the sitting block is `E1_MIA=$V/runs_p2/E1/E1_AUDIT_v1/E1_AUDIT.json` (lane option `--record`).

E4 — secure-aggregation simulation (PREG2 §7.4; DESCRIPTIVE; CPU only, about 1.5–2 min per dimension, MEASURED locally
in P2-14b). `--clip` = S of S_RECORD_P2 (the S_CAL clip bound, a training constant, not a metric); `--update-dim` = W*'s
uploaded numel = the FINAL winner's `trained_params[0]` (p2_select: "the uploaded numel after the freeze", a CALIB
selection record field):
```bash
SREC=$V/runs_p2/MAIN/P2_WSTAR_S_CAL_SMALL_s2026/S_RECORD_P2.json; FIN=$V/contracts/fl/P2_WARM_SELECTION_FINAL.json
S=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['S'])" $SREC)
N=$($PY -c "import json,sys; d=json.load(open(sys.argv[1])); print(next(c for c in d['candidates'] if c['label']==d['winner'])['trained_params'][0])" $FIN)
mkdir -p $V/runs_p2/E4 && cd $V/code_p2r3 && sha256sum v4int/e4_secagg_sim.py fl/secagg.py     # 1a0c50e1… / 9086e59f…
nice -n 10 $PY -B v4int/e4_secagg_sim.py --out $V/runs_p2/E4/E4_SECAGG.json --clip "$S" --update-dim "$N" > $V/runs_p2_e4.out 2>&1; echo rc=$?
```
Record: `E4_SECAGG=$V/runs_p2/E4/E4_SECAGG.json`. Note N beside the registered text value 265,162 in the DL row.

Devices — §8 measurement (Part B, DESCRIPTIVE; owner devices, O8). It needs no TEST row. Tools: P2-14
(`code/v4dev/browser/p2_bench.*`, `serve_coi.py`, `tools/p2_edge_server.sh`) on ONNX FP32 / INT8 exports of PRE and
P2_WSTAR (+ C_SMALL). OPEN (handoff Q6): the export command for the W* FL payload is not established (v4dev.export takes
a v4train record or a bare state dict), and no tool builds the `DEVICES` record the P2 block reads — so DEVICES reads
NOT_RUN in the sitting report unless Ops decides otherwise before step 5. Never on an H GPU during step 5.

## 4. Register the lock (local, then server)

```bash
# local, in ppsi_v4/ (Windows shell or Git Bash), after the Fable verdict
cp -n <fetched>/P2_METHODS_LOCK.candidate.json company/gates/P2_METHODS_LOCK.json && chmod 444 company/gates/P2_METHODS_LOCK.json
sha256sum company/gates/P2_METHODS_LOCK.json                       # = the candidate sha (byte copy)
python -B tools/p2_lock.py verify --lock company/gates/P2_METHODS_LOCK.json
python -B company/plans/ops_scripts/p2/p2_lock_plan_build.py p1-entry > company/gates/REGISTER_ENTRY_7_v3.json
python -B tools/p2_lock.py register-entry --lock company/gates/P2_METHODS_LOCK.json > company/gates/REGISTER_ENTRY_8_P2.json
python -B tools/test_exposure_register.py append --entry company/gates/REGISTER_ENTRY_7_v3.json   # entry 7: LOCKED_EVALUATION, ALL_METHODS_LOCK 04a3cd19…
python -B tools/test_exposure_register.py append --entry company/gates/REGISTER_ENTRY_8_P2.json   # entry 8: LOCKED_EVALUATION, the P2 lock sha, names P2_METHODS_LOCK
python -B tools/test_exposure_register.py verify                   # {"entries": 8, "errors": []}
sha256sum company/registers/TEST_EXPOSURE_REGISTER.json
```
DL row (lock file + self sha, plan sha, register sha, entry ids 7 / 8 and their entry sha256, Fable verdict file). Then
upload `company/gates/P2_METHODS_LOCK.json` (444), the two entry files and the register; on the server:
`sha256sum` (= local), `cd $V && $PY -B tools/test_exposure_register.py verify`, `$PY -B tools/p2_lock.py verify`.
The lane only verifies the register; it never appends (the register of record is the local file, as entries 5–6).

## 5. The sitting (server)

Preconditions: steps 1–4 done; owner informed (DL row) before the first step; RUN-3 / C2 / E1 not running on the GPUs;
no other Phase-2 lane or chain alive; HOLD_TEST_LOOK.txt (DL 442) renamed with the DL row; `free -g` checked if
`--slots-per-gpu` is raised (≈ 7 GiB host RAM per RETRO_TEST provider, INFERENCE, INT-07).
```bash
LS=<P2_METHODS_LOCK sha256 of the DL row>; RS=<TEST_EXPOSURE_REGISTER sha256 of the DL row>
bash $L/laneP2_SIT.sh --plan --lock-sha $LS --register-sha $RS        # 12 steps pending; 26 scoring jobs with exact commands
(nohup setsid bash $L/laneP2_SIT.sh --launch --lock-sha $LS --register-sha $RS --gpus "0 1" --slots-per-gpu 4 \
   --record E1_MIA=$V/runs_p2/E1/E1_AUDIT_v1/E1_AUDIT.json --record E4_SECAGG=$V/runs_p2/E4/E4_SECAGG.json \
   --i-confirm > $V/runs_p2_sit.out 2>&1 < /dev/null &)
tail -n 5 $V/runs_p2_sit.out; ls $V/runs_p2/_sit_state/              # step markers / STOP (value-free)
```
Order the lane enforces (PREG2 §6.3; each step writes `runs_p2/_sit_state/<step>.ok`, a restart resumes at the first
missing marker; the H steps also skip when their own `$Q/endgame/ok/H_<step>_v3.ok` exists):
1 preflight → 2 register → 3 p2_dry → 4 h_restore (chmod 555, sha 11c04583…) → 5 H reeval_dry → 6 H reeval ∥ base ∥
onnx → 7 H dev (= all Phase-1 scoring) → 8 Phase-2 scoring (reeval_p2 code_p2r3 on MAIN_TEST for every sealed
(arm, seed, rule) + FT_P2_WSTAR via launch_device_p2, both GPUs × slots) → 9 H analysis (then H_test.sh chmod 000 again)
→ 10 p2_block_index --split TEST → 11 p2_report --split TEST (script sha from P2_SCRIPT_FREEZE, stdout to
`runs_p2/TEST/logs/p2_report_test_v1.log`) → 12 done.
Design choice to confirm in review: the P2 dry runs (step 3, CPU, nothing written, no TEST row scored) run before the
Phase-1 look, so a Phase-2 refusal stops the sitting before H is unlocked.
On STOP: `cat $V/runs_p2/_sit_state/STOP` (step, rc, what; value-free). Do not open logs under runs_p2/TEST or
runs_ds3/TEST before both RETENTION_RESULT files exist unless the reviewer rules it necessary for a refusal. Fix only
mechanical causes (E10.3: an interrupted computation resumes with identical hashes; an existing incomplete reeval dir
is a manual decision, as in H). Then `mv STOP STOP.reviewed_<n>` and relaunch the same command.

## 6. After the sitting

The lane ends with `SITTING COMPLETE` and the two RETENTION_RESULT sha256 values (`runs_ds3/ANA/out_test_v3/` and
`runs_p2/TEST/p2_out_test_v1/`). Then, in this order: DL row with the shas only → fetch `out_test_v3`, `p2_out_test_v1`
and their REPORT.md → register the look(s) (next free entry numbers; observations METRICS; no number in the
description) → thesis reference batch. Reading rules: §9.4 reading as printed (overlap sentence when `overlap` is true,
freeze ruling Q1); E1 with `e1_validity_reading` and the DL 469 ceiling note; Q12a/b and Q13 DECLARED_NOT_RUN; E2 / E3 /
LDP NOT_RUN. The Phase-2 VALIDATION analysis (POST_VALIDATION_EXPLORATORY) is allowed only now:
`p2_report.py --split VALIDATION … --p2-test-result runs_p2/TEST/p2_out_test_v1/RETENTION_RESULT.json` (S8).

## 7. Benchmark locks right after the sitting

Follow `company/plans/ops_scripts/bench/LOCK_FINAL_RUNBOOK.md` (B12–B16 version): per dataset §2 preconditions, §4
LOCK (S3 / Wave 2: CPU-FP32, code_bench4, runs_bench; ML-1M: A40-FP32-DET, code_bench5b, runs_bench_gpu), §5 checks,
§6 the single TEST look (ML-1M `final --device cuda`). Before the ML-1M LOCK: the B16.4 erratum (code_bench5 vs 5b,
DL 470) under amendments/, and `lock --help` checked on the server. Resume the ML-1M GPU driver when the sitting frees
the GPUs (`run_ml1m_gpu.sh --resume --gpus "0 1"`, YIELD file removed first).

## 8. Device measurement (W*) (REV-P2-SITTING, open question 5)

Part B, DESCRIPTIVE (PREG2 §8, owner decision O8). No TEST row is read anywhere in this section and no accuracy number is
produced: latency, cold start, memory and model bytes only. Do it BEFORE `laneP2_SIT.sh --launch` (CPU timing on the
server must not overlap H base / onnx or the scoring lane); export and quantize are CPU only, never on an H GPU.
Every flag below was read from the script's own argparse / option parser (`v4dev.export`, `v4dev.quantize`,
`v4dev.dump_rows`, `v4dev.browser_queries`, `tools/p2_edge_server.sh`, `serve_coi.py`, the `p2_bench.js` query string).

```bash
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4; PY=/home/jovyan/venvs/ppsi-gpu/bin/python
CT=$V/code_p2r3; D=$V/runs_p2/DEVICE_WSTAR; S=$D/site; ROOTV1=/home/jovyan/shared_data/rees46_nextitem_v1
WB=$V/runs_p2/MAIN/P2_WSTAR_SMALL_s2026/ckpt/P2_WSTAR_SMALL_s2026/best.pt
mkdir -p $D $S/models $S/vendor; ls $CT/v4dev/export.py $CT/v4dev/browser/p2_bench.html   # the scoring tree must carry v4dev (read-only use; nothing is written into it)
```

### 8.1 Export W* s2026 best.pt to fp32 ONNX and INT8 (CPU)
`v4dev.export --ckpt` takes a v4train `{meta, model}` record or a bare state dict. W* best.pt is an FL payload
(`__format__` FL, weights under `theta`), which `v4dev.checkpointio.load_state_dict` does not read. No established command
converts it (OPEN ITEM 1). The one-liner below uses only existing functions (`fl.checkpoint.load_checkpoint`, which
re-verifies the payload digest, then `torch.save` of `theta`); it is UNTESTED and needs a review before use. The export is
strict: if `theta` is not exactly the SASREC_D64_B2 module state (keys / shapes), `load_state_dict(strict=True)` raises and
you STOP (do not edit the model to make it fit).
```bash
sha256sum $WB          # must equal the sealed P2_WSTAR s2026 PRACTICAL_BEST checkpoint_sha256 of the P2 lock (candidate lock before step 4)
cd $CT && PYTHONPATH=$CT CUDA_VISIBLE_DEVICES="" $PY -B -c "import sys,torch; from fl.checkpoint import load_checkpoint; torch.save(dict(load_checkpoint(sys.argv[1])['theta']), sys.argv[2])" $WB $D/wstar_s2026_state.pt
cd $CT && PYTHONPATH=$CT CUDA_VISIBLE_DEVICES="" $PY -B -m v4dev.export --ckpt $D/wstar_s2026_state.pt --family SASREC --variant SASREC_D64_B2 --v1-root $ROOTV1 --out $D/onnx
cd $CT && PYTHONPATH=$CT CUDA_VISIBLE_DEVICES="" $PY -B -m v4dev.quantize --fp32 $D/onnx/model_fp32.onnx --out $D/onnx      # writes model_int8.onnx + QUANTIZE_MANIFEST.json
sha256sum $D/onnx/model_fp32.onnx $D/onnx/model_int8.onnx $D/onnx/EXPORT_MANIFEST.json     # into the DL row
```

### 8.2 queries.json from NON-TEST rows
Split: role MAIN = the VALIDATION eval manifest of the frozen REES46 v1 dataset (the role `laneD3_INT8_r7.sh` dumps, DL 343).
`v4dev.dump_rows` accepts only `v4train.recipe.ROLES` roles and is given no `--registry-ext`, so it cannot reach MAIN_TEST.
Model inputs only (token / feature arrays); no model output and no metric is computed or read (S8). `--first-users` keeps the
dump small (the page uses the first 1,000 rows). If Fable prefers CALIBRATION rows over MAIN rows, change `--role`. `--ckpt` is
omitted from `dump_rows` (a bare state dict carries no registered feature view); `browser_queries` narrows to the model's
inputs because it is given `--ckpt` and `--family`.
```bash
cd $CT && PYTHONPATH=$CT CUDA_VISIBLE_DEVICES="" V4_DATASET_ROOT=$ROOTV1 $PY -B -m v4dev.dump_rows --role MAIN --history-policy USER_HISTORY_L50 --first-users 2000 --out $D/rows/rows.npz
cd $CT && PYTHONPATH=$CT CUDA_VISIBLE_DEVICES="" $PY -B -m v4dev.browser_queries --rows $D/rows/rows.npz --out $S/queries.json --n-queries 1000 --ckpt $D/wstar_s2026_state.pt --family SASREC --variant SASREC_D64_B2 --v1-root $ROOTV1
```
(`--bench-result` is not used, so queries.json carries no `model_*_sha256`; the model shas are recorded from 8.1.)

### 8.3 Assemble the site
```bash
cp $CT/v4dev/browser/p2_bench.html $CT/v4dev/browser/p2_bench.js $CT/v4dev/browser/serve_coi.py $S/
cp $D/onnx/model_fp32.onnx $S/models/fp32.onnx; cp $D/onnx/model_int8.onnx $S/models/int8.onnx
# vendor/: ort.min.js + the ort-wasm* files of onnxruntime-web 1.24.1 (the version p2_bench.html pins). NO script in the repo
# builds this directory (OPEN ITEM 2): stage the files from the package dist (npm tarball), then record their hashes:
(cd $S && sha256sum vendor/*)
```

### 8.4 Server Edge versions (SERVER_BROWSER; never a consumer device)
```bash
bash $V/tools/p2_edge_server.sh --site $S --out $D/edge_out                 # plan mode (default): prints every launch, changes nothing
bash $V/tools/p2_edge_server.sh --site $S --out $D/edge_out --run --cpus 0-3 --port 8799     # 4 versions x 3 sessions x 4 combos; downloads the Edge .debs (network, no root)
cat $D/edge_out/results/MANIFEST.tsv; ls $D/edge_out/results/*.DEVICE_BENCH_P2.json | wc -l  # one JSON per (version, model, arm, session)
```
The script resolves `serve_coi.py` as `$V/tools/../code/v4dev/browser/serve_coi.py`, so `$V/code` must exist on the server.
It needs `packages.microsoft.com` (else stage the .debs by the transfer channel, P2-14 open question 2).

### 8.5 Laptop headless Edge 154 (owner's laptop D1; Ops runs it; Windows PowerShell)
Copy `$S` (zip) to `C:\p2bench`. The command composes flags that `p2_edge_server.sh` already uses (`--headless=new
--disable-gpu --no-first-run --no-default-browser-check --disable-background-networking --disable-component-update
--disable-sync --disable-extensions --user-data-dir --enable-logging=stderr --v=0`; the server's `--no-sandbox`,
`--disable-dev-shm-usage`, `--password-store=basic` are container-only and dropped) with the page's own query string
(`auto=1 sink=<tag> d1=1 label=DEVICE vendor=1`) and `serve_coi.py --results-dir`. It is NOT a tested script; the
README_P2_DEVICE.md manual (non-headless InPrivate) stays the owner-facing fallback; the page records `headless_hint`.
Check `edge://version` = 154.0.4258.53 first; AC power, Best performance, nothing else open. Rotate the order per session as in
README_P2_DEVICE.md (session 2 starts with int8 1t, session 3 with fp32 mt).
```powershell
cd C:\p2bench
Start-Process -WindowStyle Hidden python -ArgumentList 'serve_coi.py','--root','.','--port','8765','--results-dir','C:\p2bench\sink'
$edge = "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe"
function Run-One($model,$arm,$session) {
  $tag = "D1_${model}_${arm}_s${session}"; $thr = if ($arm -eq 'wasm-mt') {4} else {1}
  $url = "http://127.0.0.1:8765/p2_bench.html?auto=1&d1=1&label=DEVICE&arm=$arm&threads=$thr&model=$model&session=$session&sink=$tag&vendor=1&fp32=models/fp32.onnx&int8=models/int8.onnx&queries=queries.json&n=1000&warmup=20"
  $prof = "C:\p2bench\profiles\$tag"; New-Item -ItemType Directory -Force $prof | Out-Null
  $p = Start-Process $edge -PassThru -RedirectStandardError "C:\p2bench\$tag.console.log" -ArgumentList '--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check','--disable-background-networking','--disable-component-update','--disable-sync','--disable-extensions',"--user-data-dir=$prof",'--enable-logging=stderr','--v=0',$url
  $t=0; while (-not (Test-Path "C:\p2bench\sink\$tag.json") -and $t -lt 1800 -and -not $p.HasExited) { Start-Sleep 5; $t+=5 }
  Get-CimInstance Win32_Process -Filter "Name='msedge.exe'" | Where-Object { $_.CommandLine -like "*$prof*" } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
  Start-Sleep 120                                                        # cool-down between runs
}
Run-One fp32 wasm-1t 1; Run-One int8 wasm-1t 1; Run-One fp32 wasm-mt 1; Run-One int8 wasm-mt 1   # session 1; repeat with session 2 / 3, rotated
```
Results: `C:\p2bench\sink\D1_<model>_<arm>_s<session>.json` (schema `ppsi.v4.device_bench_p2.v1`). A run whose JSON is missing or
carries an error is FAILED and is repeated, never filled in. Peak memory is not measured by the page.

### 8.6 What p2_report expects, and what writes it (OPEN ITEMS)
- `p2_report` reads two optional DESCRIPTIVE records, keys `DEVICES` ("real-device measurements, §8") and `D1_EXT`
  ("D1-ext update-norm summaries of the P2 FL runs, §3.3"). They enter only through the lane option `--record DEVICES=<file>` /
  `--record D1_EXT=<file>` -> `p2_block_index --records` -> `records/<KEY>.json`. The report needs one strict-JSON OBJECT, prints
  its sha256, self-sha status and top-level scalars (REPORTED), and prints NOT_RUN when the key is absent. The file must exist
  at step 10; the `--record` option is given at launch, so decide before launch.
- DEVICES: NO tool writes this record. `p2_edge_server.sh` writes per-run `*.DEVICE_BENCH_P2.json`, `MANIFEST.tsv`, `ENV.txt`;
  the laptop run writes per-run JSONs; nothing aggregates them into one object. OPEN ITEM 3: a small reviewed stdlib script
  (per device: model sha256s, bytes, medians / ranges over sessions, run counts, failed runs, as top-level scalars) is needed,
  or DEVICES reads NOT_RUN.
- D1_EXT: the FL runs write `UPDATE_NORM_LOG.jsonl` (launch_fl, one line per round); NO tool summarises it into one `D1_EXT`
  object. OPEN ITEM 4: same, or NOT_RUN. (FL training, independent of 8.1 to 8.5.)
- OPEN ITEM 1 (W* payload to bare state dict) and OPEN ITEM 2 (vendor/ assembly) are stated above.
