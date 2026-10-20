# 08 — Remaining work: the ordered runbook

*State at DL 435 (2026-10-03). Every step: check its preconditions, run it, write a DL row (next free number = tail max + 1;
UTC from the server), update RESUME_HERE at milestones. "REVIEW" = an independent read-only review is required before the
step runs (file 03 §8). Commands are PLAN text taken from the cited scripts and handoffs; run `--help` / plan mode first and
confirm them against the deployed tree. `$V`, `$R`, `$Q` as in file 07. Copies of the cited handoffs are in
`HANDOVER/reference/HANDOFF_*.md`.*

## 0. Before you touch anything

1. Get your own server access (file 07 §2). Read files 00, 03, 13.
2. Read the DL tail (rows after 435 may exist) and `company/RESUME_HERE.md` top block.
3. Value-free server check (file 07 §5): lanes alive, `LANE_STOP` / `REFUSED`, GPU use, RUN_SUMMARY counts, disk. Expected
   running at DL 435: the ML-1M GPU driver (`run_ml1m_gpu.sh`, both A40s), the Wave-2 CPU driver (`run_wave2.sh`). Nothing of
   Phase 2 and nothing of H is running.
4. Agree with the owner who is the single operations writer (one DL writer at a time).

## 1. Phase-2 code completion (Part A) — all local, then one new tree

| # | Item | Who / REVIEW | Done when |
|---|---|---|---|
| 1.1 | **P2-07 + P2-13** `code/v4int/reeval_p2.py`: PRE anchor on CALIBRATION (label P2_ANCHOR_PRE_CALIB, reusing `check_pre_init_record`); scoring of P2_W / P2_C1 / P2_C2 / merged-LoRA checkpoints on MAIN and MAIN_TEST (through the unchanged DATA-09 gate); **refuses PRE / FA_WARM_FROZEN / FT_PRE / C_FULL on TEST roles** (S5). Tests `tests_int/test_p2_reeval_anchor.py`, `test_p2_reeval.py` | dispatched DL 434; REVIEW Sonnet (P2-07) / Sonnet (P2-13) | handoff + review PASS |
| 1.2 | **P2-RUN1 lane** `company/plans/ops_scripts/p2/laneP2_RUN1.sh` (`e0289882`) + `code/tests_int/test_p2_run1_lane.py` | REVIEW (none recorded yet); also resolve the `runs_p2` root issue (file 13 R1) | review PASS; plan mode read by Ops |
| 1.3 | **WB3 / LoRA wiring** in `launch_fl`: HOOK 2 `install_sparse` for upload `delta_rows` and HOOK 1 `apply_lora` + `freeze_all_but` (exact edits in `HANDOFF_P2-01_02.md` §6 and §6c; modules `fl/sparse_rows.py`, `fl/peft.py` exist) | REVIEW Opus (P2-04 class) | Phase-1 digest table unchanged; WB3 / WC dry runs rc 0 |
| 1.4 | **P2-08a** DP chain for W*: `p2_s_cal_spec`, `S_RECORD_P2.json` (checks = `check_s_content`), arms P2_WSTAR_FA1024 / _DP8 / _DP1 with T 384, registered z, passes 2 / batch 16 (PHASE2_IMPLEMENTATION_PLAN P2-08a) | not dispatched; Opus worker + REVIEW Opus | tests `test_p2_dp_launch.py` |
| 1.5 | **FT_P2_WSTAR launch path**: `launch_device.py` must accept a P2_W base (W* best.pt, same seed), with the registered FT_C procedure (SELECTION_FT_C) — no packet exists yet (gap) | new packet + REVIEW | dry run on a P2 smoke base |
| 1.6 | **P2-12** `tools/p2_report.py` + `tools/p2_block_index.py` (import retention_report `9a3b2c2a`, never edit it; Q1 reading grid; labels; refuse TEST without P2 lock + freeze + register entry; check the S5 row hashes) | dispatched DL 434; REVIEW Opus; Fable checks the freeze | tests `tests_ana/test_p2_report.py` |
| 1.7 | **P2-00 lock tool** `tools/p2_lock.py` (`write_methods_lock`, `write_script_freeze`; O_EXCL; refuses unsealed arms outside declared-not-run; freeze refuses retention_report ≠ `9a3b2c2a`) | REVIEW Sonnet | tests `tests_ana/test_p2_lock.py` |
| 1.8 | Pack and deploy a new tree (e.g. `code_p2r2`) beside `code_p2r1` with the deploy pattern of file 07 §4 (Phase-1 digest table, CALIB arm-spec digests, tiny runs identical) | Ops | DL row with tar / aggregate shas |

Items 1.1 and 1.2 are enough to start RUN-1 (WA, C1, C2 need no hook); 1.3 is needed before RUN-2; 1.4 and 1.5 before RUN-3;
1.6 and 1.7 before the lock. Part B (P2-08b DP-FTRL + LDP, P2-09 SecAgg, P2-10 audits) only if time allows; unfinished
Part-B arms are declared not run.

## 2. Phase-2 runs

**2.1 S0 firewall on the server (CPU)** — `HANDOFF_P2-01_02.md` §7 "S0": the listed P2 + FL test files in `$T/tests_fl` and
`$T/tests_int`, and dry runs old tree vs new tree (cold DP8, DP_WARM_FROZEN, FA_WARM_FROZEN MAIN) with equal
`arm_spec_sha256` / `fl_config_digest`.

**2.2 RUN-1 (GPU; Phase 2 has priority over the ML-1M driver, which yields to `laneP2*.sh`)**
```bash
bash $V/company/…/laneP2_RUN1.sh --plan                     # server copy under $Q; prints jobs, GPUs, dirs; writes nothing
(nohup setsid bash $Q/laneP2_RUN1.sh --launch --gpus "0 1" --slots-per-gpu 2 > $Q/laneP2_RUN1.nohup 2>&1 < /dev/null &)
```
Order inside the lane: P2-T3 (65-round smoke of the incumbent + `tools/p2_t3_check.py` → `P2T3_RECEIPT.json`, rc 0 =
BIT_IDENTICAL) → WA 18 + 2 (+ the incumbent only if not BIT_IDENTICAL) → C1 CALIB 8 → C2 MAIN 1 (`--c2-early` optional).
The lane never runs while an H step runs and prints no metric. Also run the anchor (P2-07 reeval_p2, CALIBRATION role, PRE).

**2.3 Selections (CPU, mechanical; outputs write-once, then O_EXCL copy to `contracts/fl/` + DL row)** — CLI of
`code/v4int/p2_select.py`:
```bash
python -B v4int/p2_select.py select --stage C1 --runs <8 C1 dirs (+ edge runs)> --anchor <PRE_CALIB reeval dir> --out contracts/fl/P2_C1_SELECTION.json
python -B v4int/p2_select.py select --stage WA --runs <21 WA dirs (+ ≤ 2 edge)> --anchor <dir> \
       [--incumbent-phase1 $R/S7S8_T/<S8Tc dir> --p2t3-receipt <receipt>] --out contracts/fl/P2_WARM_SELECTION_A.json
python -B v4int/p2_select.py futility --wa-selection <A file> --c1-selection <C1 file>
```
Exit 3 = `P2EdgeRunsMissing`: launch the ≤ 3 registered edge runs (§5.7) and select again.

**2.4 RUN-2** (unless FUTILITY_SKIP_WB_WC): WB1–WB3 and WC1–WC5 at A* on CALIB (needs 1.3); then
`p2_select select --stage FINAL … --out contracts/fl/P2_WARM_SELECTION_FINAL.json` (with `--dropped` / `--cap-plan` if the
250 GPU-h cap forces drops, in the registered order).

**2.5 RUN-3 MAIN** (needs 1.4, 1.5): `launch_fl --arm P2_W --role MAIN --p2-selection contracts/fl/P2_WARM_SELECTION_FINAL.json`
seeds 2026 / 2027; `central_ft.py` MAIN for C1 FULL / FROZEN × 2 seeds (binds `P2_C1_SELECTION.json`); S_CAL(W*) →
`S_RECORD_P2.json` (write-once) → FA1024 ×2 → DP8 ×2 → DP1 s2026; then FT_P2_WSTAR ×2 on the W* best.pt. **Do not open any
MAIN VALIDATION value of these runs (S8).**

**2.6 Lock and freeze (Fable touch points 2 and 3)**: `tools/p2_lock.py` → `company/gates/P2_METHODS_LOCK.json` (binds PREG2
sha, P2_CONTRASTS_v3 self `e98580564b61…`, the tree digest, the selection files, S_RECORD_P2, every P2 checkpoint / rows sha,
seed 20261004, the declared-not-run list incl. Q14 / TEST2 and unfinished Part B, and the hashes of the reused Phase-1 rows) →
`P2_SCRIPT_FREEZE.json` (p2_report / p2_block_index shas; retention_report `9a3b2c2a`) → Fable reviews both → O_EXCL copies,
DL rows, upload and sha-check on the server.

## 3. The single TEST sitting (T0a) — Phase 1 then Phase 2

**Preconditions (all):** §2.6 done; register entries appended with the unchanged tool (`python tools/test_exposure_register.py
append --entry <json>`, then `verify`), both uploaded and verified on the server:
- entry 7: `LOCKED_EVALUATION`, split RETRO_TEST, all_methods_lock_sha256 = `04a3cd196149e2acf2ab5bb004f58c997199aa52e32f3c11c9ae647255beed23`;
- entry 8: `LOCKED_EVALUATION`, all_methods_lock_sha256 = the P2_METHODS_LOCK sha, the P2 lock named in observations_description.
Owner informed (DL row) before the first step. No Phase-2 lane, ML-1M driver job or benchmark on the GPUs used by H.

**Phase-1 steps** (H_test.sh, already passed: `gate`, `build`, `final`; markers in `$Q/endgame/ok/`). Each command is launched
detached; each refuses unless its predecessor marker exists:
```bash
F=59c83c861ff104d65292a15658a38a927626a286f9e628a1d0814c2de01d8134      # ANA_SCRIPT_FREEZE_v3
L=04a3cd196149e2acf2ab5bb004f58c997199aa52e32f3c11c9ae647255beed23      # ALL_METHODS_LOCK
(ANA_TAG=v3 nohup setsid bash $Q/endgame/H_test.sh reeval_dry <gpu> $F $L --i-confirm > /dev/null 2>&1 < /dev/null &)
#   then: reeval <gpu>  → dev <gpu> (long)   ; base - and onnx - may run after final (CPU; onnx uses code_ds3r7b)
```
Do **not** run H `analysis` yet (S3 order).

**Phase-2 scoring:** reeval_p2 on MAIN_TEST for every arm in the P2 lock; FT_P2_WSTAR on MAIN_TEST (device procedure); the
reused PRE / FA_WARM_FROZEN / FT_PRE / C_FULL rows come from the Phase-1 outputs (hash-bound, never re-scored).

**Analyses:** `ANA_TAG=v3 … H_test.sh analysis - $F $L --i-confirm` (TEST index + `retention_report --split TEST` + the
descriptive metric table) → `p2_report` in TEST mode → only when both RETENTION_RESULT files exist: DL row (shas only) →
fetch → register the look(s) (next free entry numbers; observations METRICS; no numbers in the description) → thesis tables.
Nothing is re-run after the look except an interrupted computation with identical hashes (E10.3; H resume rules). The
README's "entry 6 (RESULT)" is outdated (file 03 §6).

REVIEW: Fable reviewed H in its current form (GO, DL 386), but the Phase-2 interleaving (entries 7 / 8, the S3 order, an H
guard against starting before P2_METHODS_LOCK) has **not** been reviewed as a procedure — get one review of the sitting
script before step 1.

## 4. ML-1M on the GPU (B16)

1. Let `run_ml1m_gpu.sh` finish (73 jobs; it yields to Phase-2 / H). Never start a second ML-1M driver; never touch
   `runs_bench/ml_1m` (sealed). Remove the YIELD file before a `--resume` (REV-BM-04).
2. LOCK + single `final` per `LOCK_FINAL_RUNBOOK.md` with the corrections of file 06 §6: code tree **code_bench5b**, runs root
   `runs_bench_gpu`, `final --device cuda --gpu <g>`; disclosure sentence of B16.6 in the LOCK. REVIEW the exact commands.

## 5. Wave 2 (Video_Games, Baby_Products)

Let `run_wave2.sh` finish (54 Track-A jobs, CPU, code_bench4). Then per dataset: R6 third-seed decision (DL row) → LOCK →
single `final` (B14.7). If the thesis schedule requires it, a dataset may be stopped **whole, before its LOCK**, by a DL row
("converted, not run"); never a partial lock.

## 6. S3 Beauty / Sports / Toys locks and finals

All Wave-1 jobs are done (DL 369). Per dataset: run list (value-free, runbook §3) → `lock` (write-once) → DL row → pre-final
checks (runbook §5) → `final --workers 6` (CPU, ≤ 96 threads) → counts-only read → DL row → write-up under the R8 claim rule.
Use **code_bench4** (B12-aware) after confirming its file shas; REVIEW the commands once (file 06 §6). Run finals only when
the CPU is quiet enough not to disturb K16 pass 2 (or after it).

## 7. K16 INT8 pass 2 (latency only)

Lane `company/plans/ops_scripts/lanes_r6w/laneD3_INT8_pass2.sh` (local sha `b7f7ab47…` at DL 379; not launched). Before use:
(a) fix the REV-OPS-B13-P2 minor — write `PASS2_HOST_LOAD.json` before the bench and take LOAD0 after the sleep (DL 380);
(b) add `run_ml1m_gpu.sh` to `W1_BUSY_RE` (B16.5; DL 430 pending); (c) a short re-review; then launch. It waits for every
Wave-1 driver, SIGSTOPs Wave 2 for the window, runs `v4dev.bench --latency-only` from `code_ds3r7b` on the pinned
`ONNX_R7_U16384` artifact (shas asserted before and after) and writes only `LATENCY_PASS2.json` + `PASS2_HOST_LOAD.json` —
never a BENCH_RESULT into `ONNX_R7` (would break the frozen §9 text). The thesis device latency is pass 2.

## 8. Device measurements (Phase 2, Part B; DESCRIPTIVE)

- Export ONNX FP32 / INT8 of PRE, P2_WSTAR (LoRA merged if a WC cell wins) and C_SMALL with `v4dev.export` / `quantize`,
  check parity (top-20 identity ≥ 0.98 on 50 queries), build one `p2bench_<pair>.zip` per model pair (contents in
  `HANDOVER/reference/README_P2_DEVICE.md`).
- **Owner's laptop (D1):** the owner runs `serve_coi.py` + `p2_bench.html` in Edge InPrivate, 4 runs per session (fp32 / int8 ×
  wasm-1t / wasm-mt threads 4), 3 sessions on separate days with rotated order; sends back `DEVICE_BENCH_P2.json` files.
- **Server Edge versions (SERVER_BROWSER):** `tools/p2_edge_server.sh --site <dir> --out <dir>` prints the plan; add `--run`
  to execute (versions 154 / 152 / 150 / 146 pinned, `.deb` sha checked against the Microsoft repo, no root, one browser
  launch per run, loopback-only result sink). Never on a GPU used by H.
- Report median over sessions and the range; thresholds (PREG2 §8) are design targets only.

## 9. Records and the thesis reference

1. Capture DL 354 → latest into `../thesis_reference/` (single writer): new sources with
   `python tools/add_source_batch.py --list <json>` (new IDs only; **next free id S603**), a new EN reference section (§14.24),
   append STATE.md and CHANGELOG.md (backups first), then `tools/verify_reference.py` and `tools/verify_ref11.py`. Keep
   MEASURED / PLAN / INFERENCE / PENDING separate. The structured registers (DECISION / EXPERIMENT / INCIDENT / CLAIM /
   RESULTS) have been PENDING since 2026-09-27.
2. After the sitting: thesis tables (file 09), the disclosure register, the reproducibility package (code-tree archives,
   locks, registers, receipts), a RESUME_HERE block, and an external timestamp of `contracts/` + `company/gates/` + the DL
   hashes (multi-expert review D5, SHOULD-high).
3. GitHub: only the owner pushes, to his own branch, under his identity (DL 435; separate task already in progress).
