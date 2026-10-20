# P2-05 handoff: controls C1 (central fine-tune of θ_pre) and C2 (C_FULL at the SMALL recipe) (Claude Opus 5.5, 2026-10-03)
Status: **code and CPU tests DONE (MEASURED, local only).** Nothing ran on the server. No GPU, no training run, no TEST data, no git, no Codex.
Authority:
- contracts/PREG2.md (REGISTERED, sha be0561ab…, DL 411) §4.2 (C1), §4.3 (C2), §5.6 / §5.7 (C1 screen), §6.3 S8, §9.5;
- company/plans/PHASE2_IMPLEMENTATION_PLAN.md P2-05;
- reviews/FABLE_GATE_ENDGAME_SCRIPTS_20261003.md §9.
C2: owner **O3 = YES (DL 415)**, so c_full_small.py is production-ready and its server command is in §6.2.
**No existing file was edited.** launch_central, v4train, launch_fl, fl/*, c_full.py, pretrain_small.py and small_recipe.py are byte-unchanged (§4). The shims wrap them in-process only.

## 1. Files (all new; sha256[:12])
| file | sha | lines | content |
|---|---|---|---|
| `code/v4int/central_ft.py` | f392121de269 | 803 | C1 shim (roles MAIN / CALIBRATION, `--p2-freeze-item-tables`, `--peak-lr`, `--init-from`, `--c1-selection` + `--c1-selection-sha256`, dry run); shared P2 label wrapper; `read_c1_run` (CALIB only) |
| `code/v4int/c_full_small.py` | e7dedb2daca5 | 361 | C2 shim, a parameterised copy of c_full.py |
| `code/tests_int/test_p2_central_ft.py` | fa2381851f71 | 526 | 15 tests + 2 NC |
| `code/tests_int/test_p2_c_full_small.py` | 157288e47c93 | 209 | 9 tests + 2 NC |

Deploy only in the new Phase-2 tree (code_p2r1). Phase-1 trees stay untouched. The two v4int files change no Phase-1 module's digest: the trainer / FL code shas hash v4train/*.py and fl/*.py only.

## 2. Design

### 2.1 C1: `v4int/central_ft.py`
- **Launcher.** It runs the **unedited** `launch_central` in the pattern of pretrain_small / c_full. The launcher arguments are fixed:
  - `--phase MAIN --recipe-id SMALL_SASREC_D64_B2 --family SASREC --history-policy USER_HISTORY_L50`;
  - `--tuned-recipes` defaults to the production SMALL recipe;
  - seeds: CALIBRATION 2026; MAIN 2026 / 2027.
- **Refused launcher options:** `--peak-lr`, `--train/eval-manifest`, `--role`, `--hp-id`, `--calibration-record`, `--continuation-budget`, `--continue-run`, and every lock / transform / theta0 / HP override. `--allow-shared-gpu` is accepted only together with `--coschedule-06g`.
- **Arms and run ids.** The run id is **derived**; any other `--run-id` is refused:
  - `P2_C1_FT_CENTRAL_<FULL|FROZEN>_<MAIN|CALIB>_lr<peak>_s<seed>` (example: `P2_C1_FT_CENTRAL_FROZEN_CALIB_lr3e-5_s2026`);
  - FROZEN ⇔ `--p2-freeze-item-tables`.
- **Peak LR.** The value must be in the registered grid {1e-5, 3e-5, 1e-4, 3e-4} ∪ edges {3e-6, 1e-3}.
  - **MAIN (non-smoke)** binds `contracts/fl/P2_C1_SELECTION.json` by its file sha256 (`--c1-selection-sha256`, required; hashed before and after the checks). The checks are:
    1. registered path only;
    2. format `PPSI_V4_P2_C1_SELECTION_1`;
    3. self_sha256 (= launch_fl._canon);
    4. stage C1;
    5. **re-verification by `v4int.p2_select.load_p2_selection`** (P2-06; refused if that module is absent);
    6. `variants.<FULL|FROZEN>.winner_peak_lr` == `--peak-lr`.

    This is the layout P2-06's `selection_c1` writes now.
  - **CALIBRATION** refuses a selection file.
- **Budget axis.** The trainer's own MAIN phase gives:
  - 6.0 EFE on the role's TRAIN decisions;
  - 12 marks 0.5…6.0;
  - best / endpoint weights;
  - PRACTICAL_BEST = the strictly-best mark (StopTracker); CONTROLLED_BUDGET = 6.0.

  **Mark 0 is never a candidate**: θ_pre is never evaluated as a mark (`eval_marks("MAIN")` starts at 0.5; tested).
- **CALIBRATION role (wrapper 3).** The trainer view has roles MAIN / CONTINUATION / CALIBRATION = REGISTRY_V2's CALIBRATION pair:
  - entries are copied byte-for-byte and self-hashed with v4train.registry's rule;
  - so the unedited trainer runs its 6-EFE MAIN phase on CALIB_TRAIN → CALIB_VALIDATION;
  - the providers still verify against REGISTRY_V2;
  - rows / key_base carry `manifest_role CALIBRATION`. The trainer-internal run_role stays MAIN; see Q2.
- **Peak (wrapper 1 + wrapper 5).**
  - `_recipe_values` returns the SMALL recipe with the shim's peak. The recipe's own peak is recorded as `recipe_peak_lr` in the LAUNCH record.
  - `RunConfig._validate_train02`, for P2_C1 run ids only, checks the recipe rule on a copy carrying the recipe's peak. The actual peak must then equal the shim's and be on the grid.
  - The config hash is the normal one: it includes run_id and peak_lr.
- **Labels (wrapper 4; shared with C2).**
  - `RunConfig.run_label` gives the arm label instead of PRODUCTION for P2 run ids. It is a derived property, so it is not in the config hash.
  - `CentralTrainer._row_labels` sets: method = the arm label; row_status / comparison_status `P2_CONTROL`; `c_matched_eligible = downstream_seed_allowed = False`.
  - The rows also carry the `p2_*` fields: role, peak, frozen keys, trained numel, init sha / digest, continuation rule.
  - v4train.selection refuses these labels at every stage (tested).
- **Continuation.** Disabled; fixed 6.0 EFE:
  - `trainer.auto_continue = False`;
  - `continue_run` is refused;
  - for a STILL_IMPROVING run, the `latest` resumable is deleted after RESULT_ROWS / RUN_SUMMARY. The summary is rewritten (`latest_kept False`, `p2_continuation DISABLED_FIXED_6.0_EFE`) and an EVENTS row `P2_C1_NO_CONTINUATION` is added.
- **Recorded:**
  - RUN_START.json["p2_c1"]: identity, init record, loaded digest, freeze record, trained / total numel, selection binding;
  - every checkpoint meta["p2_c1"];
  - RUN_SUMMARY["p2_c1"] + `p2_trainable_numel` (read by P2-06's `c1_run_adapter`);
  - key_base `initialization_block = P2_THETA_PRE|SASREC|checkpoint_sha256=…|state_digest=…`.
- **Resume.** It re-applies the init and refuses when RUN_START's p2_c1 identity differs (tested with another θ_pre).
- **S8 (PREG2 §6.3).**
  - MAIN-phase runs keep every value under `sealed/` (TRAIN-03).
  - The shims print and record no metric value.
  - `read_c1_run` (a reader for P2-06) checks RUN_START first and refuses any MAIN-role run, or any run whose eval manifest is not CALIB_VALIDATION, **before opening any value-bearing file**. A test pins this: only RUN_START.json is opened.

### 2.2 C2: `v4int/c_full_small.py` (copy of c_full.py, which is untouched)
- **Fixed launch:** recipe SMALL_SASREC_D64_B2 (SASREC_D64_B2, the registered θ0 of seed 2026), seed 2026, production SMALL recipe, on FULL_TRAIN_5_100_TIEBREAK → MAIN_VALIDATION_5_100_131072.
- **Registry and wrappers:** REGISTRY_V2_FULL_EXT.json, through c_full's wrappers 1–3 (its `make_verify_wrapper` is reused unchanged).
- **Run id:** prefix `P2_C2_CFULL_SMALL_`; default `P2_C2_CFULL_SMALL_SASREC_D64_B2_s2026`.
- **Labels:** run_label `P2_C2_CFULL_SMALL`, row_status P2_CONTROL, c_matched_eligible False. K20b keeps C_FULL.
- **Continuation:** PRE's automatic C3 continuation is kept (sealed contest-shaped run).
- **Refused:** `--peak-lr`, manifests, overrides, and `--allow-shared-gpu` without `--coschedule-06g`.
- **Record:** `P2_C2_SHIM_<k>.json`.
- **Dry run:** `check_plan` proves the trainer MAIN role = (TIEBREAK, MAIN_VALIDATION) and the D64 variant.

## 3. TO-CHECK resolved: does CentralTrainer accept an initial state?
**No.** `CentralTrainer.__init__` always builds theta0 from (variant, seed). It checks theta0 against `theta0_init_sha256` (THETA0_VARIANTS) and builds AdamW over the adapter's parameters. There is no init argument. The shim therefore wraps `__init__` for P2_C1 run ids, and right after construction, **before step 0**, it does the following:
1. It asserts `state.updates == 0` and an empty AdamW state.
2. It loads θ_pre with `launch_fl.load_pretrain_init`:
   - the record's checkpoint bytes are hashed before torch.load;
   - the checkpoint must be PRE's endpoint_6.0, label PRETRAIN_FOR_FL, PRETRAIN_TRAIN_5_100_NONMAIN;
   - any MAIN / CALIB / FULL_TRAIN-trained or C_BIG / C_SMALL / C_FULL / C_MATCHED checkpoint is refused;
   - the checkpoint must carry the **pinned** sha `cf08588c…` and digest `ee33345b…`.
3. It applies the state with `launch_fl.apply_init_state` (strict load; the broadcast state must hash to the digest).
4. FROZEN only:
   - `fl.runtime.freeze_item_tables` (the S8 freeze: requires_grad False, FIXED buffers, no recentering);
   - a fresh AdamW, same recipe values, over the trained parameters only;
   - the checkpoints still hold the full model.
5. It records the init sha in RUN_START.

RUN_START's top-level `init_sha256` stays the built theta0. That is the trainer's own identity field for resume. The real start is in `p2_c1`.

## 4. Tests (MEASURED, CPU, Windows; temp on F: because C: is full)
**New files:** `test_p2_central_ft.py` + `test_p2_c_full_small.py` gave **24 passed, 4 xfailed**. All 4 NCs FAIL with an AssertionError under `--runxfail`.

The C1 tests run on the **real CentralTrainer** with synthetic providers (tests_train/trainkit) under the CALIB manifest ids:
- θ_pre is loaded before step 0, and RUN_START carries the sha;
- non-PRE pins, a MAIN-trained init and a C_SMALL run id are refused;
- FROZEN: the 3 item tables are **bit-unchanged at update 50 and at 6.0 EFE (75 updates)** while blocks train; the optimizer holds the trained numel only;
- FULL: the tables move;
- PB at the first mark 0.5 when values fall (never mark 0);
- the latest of a STILL_IMPROVING run is deleted and continue_run is refused;
- rows carry P2_CONTROL / c_matched False / p2_* fields;
- `read_c1_run` works;
- a resume with another θ_pre is refused, and the same θ_pre resumes to DONE_MAIN;
- S8: the reader refuses a MAIN run unread.

C2 tests run on the real REGISTRY_V2_FULL_EXT / REGISTRY_V2 (read only): the view resolves MAIN to TIEBREAK, the dry run binds TIEBREAK + D64, c_full.py's sha is unchanged around install.

**Suites before / after.** "Before" = the same suite with the two new files excluded. Nothing existing imports them, so this equals the pre-task state.

| suite | before | after |
|---|---|---|
| tests_int | 366 passed, 138 xfailed, 10 failed (514) | 429 passed, 146 xfailed, 3 failed (578 = 514 + my 28 + other workers' new test_p2_launch / test_p2_noop / test_p2_select) |
| tests_train | 212 passed, 61 xfailed, 2 failed | 212 passed, 61 xfailed, 2 failed (the same 2) |
| tests_data08 (c_full) | 22 passed, 8 skipped, 14 xfailed | identical |
| tests_data10 (pretrain_small) | 34 passed, 19 skipped, 19 xfailed | identical |

The failures are environmental or belong to other packets. None is in a module P2-05 touches, and none is in my two files (all 28 passed inside the after run):
- **MemoryError (9 in tests_int before).** This 16 GB host had about 2.8 GB free with other workers' pytest processes running.
- **test_fl16 frozen resume (tests_int, before).** It was refused because the FL code sha changed **during** the run: the concurrent P2-02 worker is editing fl/*.py.
- **After, tests_int (3):**
  - `test_fl10_fedadam_launch::test_stage_s2_refusals_and_edges`: a Windows `PermissionError` on an atomic `best.pt` rename (a file lock; it passed before);
  - `test_p2_launch::test_p2_resume_main_rows_wb_and_pf` and `::test_nc_phase1_admits_p2_client_lr`: the concurrent P2-02 worker's new file, still in progress.

  The 10 "before" failures pass in the after run (memory was free).
- **tests_train `test_02b_loads_from_another_cwd` / `test_p2_work_dir_is_per_session` (before and after).** The basetemp lies under the V4 root (on F:, because C: has about 2 GB free).

Full pytest outputs: `ppsi_v4/_pytest_tmp/P2-05/logs/{before,after}_*.txt`.

Unchanged by sha (MEASURED):

| file | sha |
|---|---|
| c_full.py | ec31af91… |
| pretrain_small.py | 1f5630c6… (= the reviewed value) |
| launch_central.py | 2d56c222… |
| v4train/trainer.py | f721cbdc… |
| v4train/recipe.py | 936db145… |

All of these mtimes predate the task.

## 5. Needed from other packets (no edit made by me)
- **launch_fl (P2-02 worker):** keep `load_pretrain_init`, `apply_init_state`, `pretrain_identity`, `check_init_identity`, `_canon` and `FROZEN_ITEM_KEYS` byte-identical in behaviour (as FL-14 did). central_ft imports them.
- **P2-06:** keep `load_p2_selection(path, "C1")` and the C1 layout `variants.<V>.winner_peak_lr` / `run_label`. Its `c1_run_adapter` already matches my RUN_SUMMARY: run_label = arm label, `peak_lr`, `seed`, `p2_trainable_numel`, practical_best.

## 6. Server commands (PLAN; run from the reviewed code_p2r1 tree)
```
W=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
PY=/home/jovyan/venvs/ppsi-gpu/bin/python; CT=$W/code_p2r1; R=$W/runs_ds3/PH2      # R: Ops may choose another root
I=$W/runs_pretrain/PRETRAIN_FOR_FL_SMALL_SASREC_D64_B2_s2026/PRETRAIN_FOR_FL_INIT.json
cd $CT && unset CUDA_VISIBLE_DEVICES && export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  V4_DATASET_ROOT=/home/jovyan/shared_data/rees46_nextitem_v1 && mkdir -p $R/C1 $R/C2 $R/dry $R/SMOKE
L="--family SASREC --phase MAIN --recipe-id SMALL_SASREC_D64_B2 --history-policy USER_HISTORY_L50"
```

### 6.1 C1 CALIB screen (8 runs, PREG2 §4.2; seed 2026; 4 per A40, co-scheduled)
**(a) GPU smoke, one per variant** (CALIB values only; weights deleted by the smoke):
```
for V in FULL FROZEN; do F=; [ $V = FROZEN ] && F=--p2-freeze-item-tables
  $PY -B $CT/v4int/central_ft.py --role CALIBRATION --peak-lr 1e-4 $F --init-from $I -- $L --seed 2026 --gpu 0 \
    --run-dir $R/SMOKE/C1_$V --smoke --smoke-updates 300 --allow-shared-gpu --coschedule-06g || break; done
```
**(b) Dry run, then launch:**
```
G=(0 1 0 1 0 1 0 1); k=0
for V in FULL FROZEN; do F=; [ $V = FROZEN ] && F=--p2-freeze-item-tables
  for LR in 1e-5 3e-5 1e-4 3e-4; do
    RID=P2_C1_FT_CENTRAL_${V}_CALIB_lr${LR}_s2026; RD=$R/C1/$RID; mkdir -p $RD
    A="--role CALIBRATION --peak-lr $LR $F --init-from $I -- $L --seed 2026 --gpu ${G[$k]} --run-dir $RD --allow-shared-gpu --coschedule-06g"
    $PY -B $CT/v4int/central_ft.py --dry-run-out $R/dry/$RID.json $A || break 2
    (nohup setsid $PY -B $CT/v4int/central_ft.py $A > $RD/launch.log 2>&1 < /dev/null &)
    k=$((k+1)); done; done
```
- Edge runs (§5.7: 1e-5 → 3e-6, 3e-4 → 1e-3) use the same form, only if P2-06 names them.
- Then select: `$PY -B $CT/v4int/p2_select.py select --stage C1 --runs $R/C1/P2_C1_FT_CENTRAL_*_CALIB_* --anchor <P2_ANCHOR_PRE_CALIB dir> --out $W/contracts/fl/P2_C1_SELECTION.json`.

**(c) C1 MAIN (after the selection; 2 variants × seeds 2026 / 2027):**
```
SEL=$W/contracts/fl/P2_C1_SELECTION.json; SHA=$(sha256sum $SEL | cut -d' ' -f1); G=(0 1 1 0); k=0
for V in FULL FROZEN; do F=; [ $V = FROZEN ] && F=--p2-freeze-item-tables
  LR=$($PY -c "import json;print(json.load(open('$SEL'))['variants']['$V']['winner_peak_lr'])")
  for S in 2026 2027; do RD=$R/C1/P2_C1_FT_CENTRAL_${V}_MAIN_s$S; mkdir -p $RD
    A="--role MAIN --peak-lr $LR $F --init-from $I --c1-selection $SEL --c1-selection-sha256 $SHA -- $L --seed $S --gpu ${G[$k]} --run-dir $RD --allow-shared-gpu --coschedule-06g"
    $PY -B $CT/v4int/central_ft.py --dry-run-out $R/dry/C1_MAIN_${V}_s$S.json $A || break 2
    (nohup setsid $PY -B $CT/v4int/central_ft.py $A > $RD/launch.log 2>&1 < /dev/null &)
    k=$((k+1)); done; done
```
S8: nobody opens `$RD/sealed/` of a MAIN run before both sitting analyses exist. STATUS / EVENTS / launch.log carry no value.

### 6.2 C2 (O3 = YES, DL 415; 1 run, seed 2026; PRE's automatic C3 continuation; ≈ 12–16 GPU-h INFERENCE)
```
RID=P2_C2_CFULL_SMALL_SASREC_D64_B2_s2026; RD=$R/C2/$RID; mkdir -p $RD
A="-- $L --seed 2026 --gpu <G> --run-dir $RD"      # add --allow-shared-gpu --coschedule-06g only if the card is shared (06G §5; never beside an H step, S7)
$PY -B $CT/v4int/c_full_small.py --dry-run-out $R/dry/$RID.json $A \
 && (nohup setsid $PY -B $CT/v4int/c_full_small.py $A > $RD/launch.log 2>&1 < /dev/null &)
```
- The dry run must print `run_label P2_C2_CFULL_SMALL` and `effective_auto_continuation true`. `plan` must show TIEBREAK / MAIN_VALIDATION / SASREC_D64_B2.
- There is no C2 smoke. Since the REV-P2-05 fix, `--smoke` is refused (S8: it would write a MAIN_VALIDATION number). The code path is c_full's, already exercised by C_FULL / PRE.

## 7. Open questions / notes for review (Sonnet)
- **Q1.** CALIB runs use the trainer's MAIN phase (sealed/, best + endpoint weights about 83 MB each). The CALIBRATION phase would end at 2.0 EFE and keep no weights, which breaks the registered 6-EFE axis.
- **Q2.** In C1 CALIB runs:
  - RUN_START / checkpoint `manifest_registry.role` and `run_role` say MAIN (the trainer view role), with CALIB manifest ids;
  - rows / key_base say `manifest_role CALIBRATION`;
  - `p2_c1.identity.role` is authoritative.

  P2-13 (reeval shim) should key on `p2_c1`.
- **Q3.** C1 checkpoints' meta `init_sha256` is the built theta0 (trainer identity), not θ_pre. θ_pre is in `meta.p2_c1.init_checkpoint_sha256`.
- **Q4.** The edge LRs 3e-6 / 1e-3 are accepted at any time. P2-06's edge rule decides whether they run.
- **Q5.** C2 is seed 2026 only (PREG2 §9.5). The shim refuses 2027.
- **Q6.** `central_ft.read_c1_run` duplicates P2-06's own reader, with stricter identity checks. P2-06 may import it or keep its own.

## 8. REV-P2-05 fix (m1 / m2; PASS_WITH_MINORS, company/reviews/REV-P2-05.md)
**What changed.**
- **central_ft.py** now **refuses `--smoke` with `--role MAIN`** in `check_launch_args`, before `launch_central` runs or any data loads. The CALIBRATION smoke stays allowed.
- **c_full_small.py** refuses `--smoke` entirely, because its evaluation manifest is always MAIN_VALIDATION.

Both exist so that no MAIN_VALIDATION number can land in SMOKE_EVAL.json (PREG2 §6.3 S8). C2 status: O3 = NO (DL 424), so C2 is not run; the shim is kept consistent.

**New tests.** `test_main_smoke_refused_before_any_data_load` + NC `test_nc_main_smoke_accepted` (C1), and NC `test_nc_c2_smoke_accepted` (C2).

**Rerun of the two files (MEASURED):** 25 passed, 6 xfailed. All 6 NCs FAIL as intended under `--runxfail`.

**New shas (sha256[:12]):**

| file | sha |
|---|---|
| central_ft.py | **4d95d12cd55d** |
| c_full_small.py | c9960d4cde62 |
| test_p2_central_ft.py | 47b4abddfa7f |
| test_p2_c_full_small.py | 5bfd26a67ca7 |

These supersede the §1 shas.
