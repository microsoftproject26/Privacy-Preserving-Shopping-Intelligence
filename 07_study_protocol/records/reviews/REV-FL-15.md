# REV-FL-15 — independent read-only review of FL-15 (06M_B B2 2-D S7 / S8 screens, B4 FT_FA_WARM, R5 TRAIN_LOSS)

Reviewer: Claude Opus 5.5 (read-only agent), 2026-09-30. I edited no code or contract, and used no server, GPU, real data, pip, git
or Codex.
Authority read: 06K §1–§5, 06M M1–M2, 06M_B B1–B5, 06J §1–§6, 06L_E E1, DECISION_LOG row 261, handoff `company/handoffs/FL-15.md`.
Baseline: `deploy_archive/r4/code_ds3r4.tar`. The old files were read with `tar -xOf` into the session scratchpad only, never over
the tree. Old sha256 values = the handoff "before" values (MEASURED): lr_screen e81667d3, launch_fl 5652e4c7,
launch_device 0ae48c72, fl/runtime 443fdfc6, test_fl12_launch a50a6d01.
Reviewed files (sha256 MEASURED, all equal to the handoff "after" values): lr_screen 860ec751…, launch_fl 5facf3a7…,
launch_device 9106f67c…, fl/runtime e627e943…, test_fl12_launch 5adaae72…, test_fl15 bb39f723….

## VERDICT: **PASS-WITH-FIXES**

- No BLOCKER and no MAJOR.
- The registered rules are implemented as written:
  - the B2 grids, the 06J band + MIN_GAIN rule with the DL 261 tie order, and the excluded-incumbent refusal;
  - the MAIN binding of both factors;
  - the FT_FA_WARM base verification and the SELECTION_FT_C pin;
  - the TRAIN_LOSS z > 0 exclusion.
- There are four MINORs. None has to be fixed before the r5 tree is packed or the S7 / S8 screens are launched.
  - m1 is a disclosure to add to the DL.
  - m2 and m3 are cheap hardening. I recommend both before the **select-warm** read (m2) and before the first FT_FA / FT_FA_WARM
    launch (m3).
  - m4 is cosmetic.

## Findings

### BLOCKER
None.

### MAJOR
None.

### MINOR
**m1 — The in-band winner can itself gain less than 0.002 over (m*, η*). This is consistent with S2–S4, but it is not disclosed.**
- Where: `lr_screen.py:903-908` (`warm_decision`). The MIN_GAIN test uses the *best candidate* (`top`, :905). The winner is
  `band[0]` in tie order (:908). A simpler cell that sits within 0.0005 of the best can therefore win even when its own gain is
  in [0.0015, 0.002).
- MEASURED: `warm_decision("FA_WARM", "fedadam", {(1,1): 0.1000, (1,0.1): 0.1017, (0.25,1): 0.1021, (0.25,0.1): 0.09})` →
  winner `f=1|sf=0.1`, gain reported 0.0021 (the best candidate's), while the winner's own gain is 0.0017.
- This is the same semantics as `stage_decision` (`lr_screen.py:276-282`, FIX-DELTA-02, already reviewed). It also matches the
  literal 06J §4(ii) text ("a stage whose *best candidate* gains < 0.002"). **It is not a defect.**
- The risk is reporting only: the thesis ladder could state "S7 gained ≥ 0.002" for a winner that gained 0.0017.
- Minimal fix (no code needed):
  - In the S7 / S8 DL row (or the DL 261 addendum), state that `gain` / `best_candidate` refer to the best candidate. The winner is
    the tie-order head of the band, so its own gain can be as low as 0.0015.
  - Report the winner's own Δ over (m*, η*) in the ladder.
  - Optional: a unit case in `test_warm_decision_rule` that pins this example.

**m2 — select-warm accepts any verified tuning stage file (S2 / S3 / S4), not only the registered S4 FL_TUNING_SELECTION.**
- Where: `lr_screen.py:922-926`. `F.load_fl_tuning(tuning)` accepts every stage in `STAGES`, and no stage / registered-path check
  follows. This behaviour already existed (FL-12 used the same call).
- Failure scenario: select-warm is run with `--tuning` pointing at an S3 stage file, and 4 runs launched at that point are given.
  The runs pass (`r["tuning"] == tw`), and an O_EXCL `FL_WARM_SELECTION_S7.json` is written against the wrong incumbent.
  - The error surfaces only at the MAIN launch: `bind_warm_selection` compares the tuning sha with the bound registered file
    (`launch_fl.py:2175-2202`). The wasted artefact is write-once.
  - The MAIN guard makes this safe, so it is MINOR. Checklist item 1 ("bound to the registered FL_TUNING_SELECTION (S4)") is
    therefore enforced only downstream.
- Minimal fix: in `selection_warm`, after `load_fl_tuning`, refuse `tobj["stage"] != "S4"`. Optionally also refuse a path that is
  not `F.FL_TUNING_PATH` (by `_same_path`) for non-test use. Add one refusal test.

**m3 — FT_FA (`--base-kind fl`) still accepts an FA_WARM / FA_WARM_FROZEN MAIN checkpoint without a refusal.**
- Where:
  - `launch_device.py:1160-1170`: `load_base`, kind `fl`, checks only the loaded-state digest.
  - `launch_device.py:1015`: `refuse_theta_pre_base` refuses only the `PRETRAIN_FOR_FL_` prefix.
- Failure scenario: `--arm FT_FA --base <FA_WARM run>/ckpt/FL_FA_WARM_SMALL_s2026/best.pt --base-kind fl` runs, and its rows are
  labelled FT_FA. The result is rows labelled FT_FA (FT_FA recipe) but computed from a warm base, which contaminates the
  FT_FA contrast and the FT_FA_WARM − FA_WARM comparison.
  - The gap already existed. FL-15 makes it reachable, because FA_WARM MAIN checkpoints now have the exact FT_FA base layout and
    a dedicated arm exists for them.
- Minimal fix: in `refuse_theta_pre_base` (or `resolve_start`), refuse an `FT_FA` / `FT_C` start whose run id starts with
  `FL_FA_WARM`. A stronger version reads the base run's RUN_SUMMARY and refuses when `arm_spec.arm ∈ launch_fl.WARM_ARMS`. The
  refusal message should point to `--arm FT_FA_WARM[_FROZEN] --base-kind fa_warm`. Add one test.

**m4 — A MAIN warm `--dry-run` resolves the factors without the server-axis / tuning-sha binding.**
- Where: `launch_fl.py:2037-2038`. The dry-run branch calls `warm_selection_winner(load_warm_selection(...))` and skips
  `bind_warm_selection` (registered path, tuning sha, `check_warm_server_axis`).
- `arm_spec` still refuses a factor that is outside the resolved optimiser's grid. A foreign η* or a foreign tuning file, however,
  gives a dry-run spec that the real launch then refuses.
- It is cosmetic: a dry run launches nothing. It was pre-existing for the 1-D factor.
- Minimal fix (optional): call `check_warm_server_axis` against the resolved tuning in the dry-run branch as well.

## Checklist evidence

### 1. select-warm (B2)
- **4 registered runs.**
  - `lr_screen.py:929` requires `len(run_dirs) == len(warm_grid)`, which is 4. `lr_screen.py:934` requires the multiset of
    (local, server) cells to equal the grid, so a duplicate, missing or foreign cell refuses.
  - The CLI takes `--runs nargs="+"`. The 2-run 06K screen refuses (test_fl12 `test_select_warm_files_and_refusals`, NC
    `test_nc_2run_screen_accepted`).
- **Grids.**
  - `launch_fl.py:235`: `WARM_FACTORS` S7 (1, 0.25), S8 (1, 4).
  - `WARM_SERVER_FACTORS` fedadam (1, 0.1), fedavgm / fedavg (1, 0.25).
  - `warm_grid` is the cross product with (1, 1) first.
  - `warm_server_opt` (`launch_fl.py:758-768`) sets FedAdam `η*/10` (exact division, the other hyper-parameters unchanged),
    FedAvgM `FEDAVGM_LR/4` with β 0.9, and FedAvg `{fedavgm, lr 0.25, β 0}`.
  - This matches 06M_B B2 {m*, m*/4} / {m*, 4m*} × {η*, η*/10 | η*/4}.
- **Incumbent and rule.**
  - `warm_incumbent` = (1, 1). It refuses when (1, 1) is excluded or missing (DL 261(a)).
  - The band is `best − v <= 0.0005` on raw values (:903).
  - The best non-incumbent must gain ≥ `MIN_GAIN` 0.002 (:905-907). Otherwise the winner is the incumbent and the tie set is
    [incumbent].
  - Non-finite values are excluded and recorded in `excluded` (read_run + `selection_warm`), per 06J §4(v).
- **Tie order.** `warm_order_key` (:864-869) = (not the incumbent, number of changed axes, |log2 local|, |log2 server|, wf, sf).
  This is DL 261(a) "(m*, η*) → fewer changed axes → local closest to 1 → server closest to 1". With two values per axis,
  |log2| and "closest to 1" coincide. Tested in `test_warm_decision_rule`.
- **Same spec except the two axes.**
  - `_strip_lr` (:303-318) additionally drops `warm_server_lr_factor`. The cfg `server_opt` / `lr_peak_local` / solver LR were
    already dropped.
  - All four stripped specs must be equal (:941-945).
  - The dropped fields are re-verified per run:
    - `r["tuning"] == tw`;
    - `spec_tuning` rebuilds cfg `server_opt` through `warm_server_opt` (launch_fl :462-466);
    - `read_run` checks the local LR × wf and the row labels `fl_warm_server_lr_factor` / `fl_warm_server_lr`.
- **One eval manifest, one device record, the tuning file.**
  - One eval manifest (:946), one `lr_peak_base`, one device record `_one_device` (:950).
  - The file records the tuning sha / self-sha / stage / winner label, and the server axis η* is taken from the tuning winner.
  - The S4-only binding is enforced downstream only (m2).

### 2. MAIN warm launch
- **Both factors come from the 2-D file only.**
  - `resolve_spec` (:2036-2041): non-dry → `bind_warm_selection` → `wf, wsf = winner_factor, winner_server_lr_factor`.
  - `bind_warm_selection` requires all of:
    - the registered path;
    - `warm_selection_winner`, which refuses format `_1` (:2153-2160);
    - the arm;
    - tuning sha == the bound registered FL_TUNING sha;
    - the bound file re-hashed;
    - `check_warm_server_axis`: optimiser, η* and winner label = the bound winner (:2163-2172);
    - θ_pre identity == `--init-from`.
  - `load_warm_selection_2d` (:2099-2150) re-derives the grid, server_lrs, η* registration and the full `warm_decision`, and
    compares every decision key and the winner. A tampered file refuses (10 tamper cases in test_fl15).
- **CLI overrides refuse.**
  - `warm_server_preflight` (:1563-1583): with a non-smoke MAIN run, `--warm-server-lr-factor` refuses.
  - `--warm-selection` together with the factor refuses. `--warm-lr-factor` on MAIN refuses (pre-existing).
  - NCs: `test_nc_main_cli_server_factor`, `test_nc_1d_file_configures_main`, `test_nc_foreign_server_axis_binds`,
    `test_nc_main_run_ignores_selected_server_lr`.
- **FedAvg η/4 is exact.**
  - `fl/server.py:235`: `m = β·m_prev + Δ`. At β = 0 this is `0·m_prev + Δ = Δ` bitwise; m_prev is finite, and a −0.0 term adds
    exactly.
  - `:239`: `new = θ + m·0.25`, the lr ≠ 1 branch. So `θ + 0.25·(agg − θ)` exactly.
  - Tested bitwise over 3 rounds (`test_fedavg_server_factor_is_scaled_fedavg`). Resume is bitwise
    (`test_warm_server_factor_runs_exact_and_resume`).
  - FedAdam (η 0.01, f 0.1) is bitwise the η 0.001 run.
  - The row shows no `fl_server_opt` (the tuning point is FedAvg) plus `fl_warm_server_lr` 0.25. RUN_SUMMARY `server_opt` shows
    fedavgm β 0, as disclosed.

### 3. FT_FA_WARM(_FROZEN) (B4)
- `load_warm_base` (`launch_device.py:936-1007`) checks:
  - the layout `<run>/ckpt/<run_id>/{best,endpoint_6.0}.pt`;
  - DONE_MAIN;
  - run id equal across ARM_SPEC and RUN_SUMMARY;
  - the arm = base arm, in the spec, ARM_SPEC and RUN_SUMMARY;
  - role MAIN;
  - `spec_digest` = both recorded digests, and `ARM_SPEC.arm_spec == spec`;
  - `check_warm_spec` and `spec_tuning`;
  - `_main_trained` on the run id + init identity (:978), plus the init-manifest refusal inside `check_init_identity`;
  - not θ_pre (:981);
  - seed == `--seed` (:983);
  - the checkpoint is in `summary.checkpoints` (real `ckpt.files()` names; launch_fl :1418);
  - the optional sha pin;
  - payload `kind == "weights"`, run_id, and `config_digest == fl_run_config(spec, run_id, ·).digest()` (:994; `ckpt_every` is
    excluded from the digest);
  - the loaded-state digest.
- The tests use **real** `F.execute` outputs (the `warm_base` fixture), not synthetic records. The real-vs-synthetic gap that
  FIX-REEVAL found is therefore not present here.
- Procedure:
  - `FT_PROCEDURE` = FT_C and `SELECTION_ARM` = FT_C.
  - `check_ft_pre_selection_file(arm=…)` enforces the registered file and the DL 231 pin e47965d2 in preflight and again at
    run time (`run`).
  - The FT cell is bitwise the FT_C cell on the same start (`test_ft_fa_warm_cell_is_the_ft_c_procedure_on_the_base`).
  - `LOCK_CODE` "FA" affects only the ARCHITECTURE_LOCK allowed-method check (`assert_arm_allowed`), not the recipe.
- **Digests unchanged:**
  - `device_spec` adds `base_arm` only for the two new arms.
  - The pinned LO / FT_C / FT_FA digests (PRE_DEVFTPRE_DIGESTS), the 6 FT_PRE digests and the 1 FT_FA digest pass.
- Gap: FT_FA still accepts a warm base (m3).

### 4. TRAIN_LOSS.jsonl (R5)
- `round_loss_record` (`fl/runtime.py:202-214`) uses only the visits with steps > 0:
  - `mean_step_loss = Σloss_sum / Σsteps`;
  - `client_mean_loss = mean(loss_sum_u / steps_u)`.
- `VisitRecord.loss_sum` is the sum over the steps of the batch-mean CE of the final attempt, consistent with `steps`
  (`fl/client.py:205-236`, `fl/server.py:381-382`).
- Non-finite values → None and `finite` False. The file is written with `allow_nan=False`.
- The record is added to both step paths (:511, :596). No RNG, no extra forward pass, no state.
- **Resume trim:** `trim_train_loss(cursor)` on resume and `trim_train_loss(0)` on a fresh start (launch_fl :1314 / :1321). The
  resumed file is byte-identical to the straight run; a test injects a stale line to prove it.
- **z > 0:** `train_loss_applies` (:1120-1123) is False when `cfg.dp.noise_multiplier > 0`. In that case there is no file, no
  append and no RUN_SUMMARY key; S_CAL / FA_1024 (z = 0) do write the file. This matches DL 261(b).
- **Pins:** `PRE_FL10_TRAJ` holds for FA / DP / FA_1024 / FP / FA_Q8 in test_fl15, and for all 7 arms in the tests_fl / FL-10 pins.

### 5. Nothing else changed
- **AST function comparison against r4 (MEASURED):**
  - launch_fl changed only `spec_tuning`, `parse_args`, `arm_spec`, `warm_row_labels`, `check_warm_spec`, `execute`,
    `warm_preflight`, `default_run_id`, `resolve_spec`, `load_warm_selection` and `bind_warm_selection`, plus new functions.
    **`load_pretrain_init`, `apply_init_state` and `check_init_identity` are byte-identical.**
  - lr_screen changed only `_strip_lr`, `read_run`, `selection_warm`, `write_selection_warm` and `main`, plus 3 new functions.
    `stage_decision`, `selection_stage`, `selection`, `selection_ext` and `load_fl_tuning` (in launch_fl) are unchanged.
    `_NOT_DUMPED` keeps the 06B / 06F files byte-identical.
  - fl/runtime changed only `FLRun` (the `train_loss` key), plus `round_loss_record`.
- `v4eval.evaluator_code_sha256()` = 61b64e61c1a2e4f5a6fecd42f0ad3b98de26562d13a07758573cc9496b8d962e (MEASURED).
- The CLIP_LOG functions (`clip_round_record`, `clip_log_summary`, `trim_clip_log`) are unchanged, so the 06L_E rules are intact.
- **Tree against r4:** the only other differences are:
  - the FIX-REEVAL files (`reeval.py`, `test_gate_b_reeval.py`, `test_fix_reeval_real_records.py`, fixtures);
  - 3 `tests_fl/receipts/*.json` files. These are wall-clock receipts that tests_fl rewrites on every run; the handoff's run and
    my rerun both rewrote them.
  - Note for the r5 packing: take those 3 receipts from r4, or accept them as regenerated. They are not code.

### 6. Tests (next_item_eda venv, `-B -p no:cacheprovider -q`, basetemp in the session scratchpad; suites run one at a time)
| suite | result (MEASURED) | expectation |
|---|---|---|
| tests_int (full) | **327 passed, 110 xfailed**, exit 0, 375 s | DL 261: FL-15 319 / 106 + FIX-REEVAL = 327 / 110 ✓ |
| tests_fl (full, alone) | **248 passed, 75 xfailed**, exit 0, 175 s; no access violation | 248 / 75 ✓ |
| tests_ana (full) | **161 passed, 54 xfailed**, exit 0 | 161 / 54 ✓ |
| `--runxfail -k test_nc_`, 21 tests_int files that import launch_fl / lr_screen / launch_device / fl.runtime | **84 / 84 fail**; all AssertionError or a subclass (1 = `HistoryLeakError(AssertionError)`, test_int07, unchanged file) | handoff: 72 / 72 on 14 files (a subset) ✓ |
| `--runxfail -k test_nc_`, 12 tests_fl files that import fl.runtime | **39 / 39 fail**, all AssertionError | handoff: 11 / 11 on 4 files (a subset) ✓ |

- **All 13 FL-15 NCs fail on the intended assertion (junit messages read), so none is vacuous:**
  - 2-run screen → `selection_warm did not raise`;
  - `MIN_GAIN` 0 → the winner becomes (1, 0.1);
  - excluded incumbent → `warm_decision did not raise`;
  - resolve_spec drops wsf → `None == 0.1`;
  - trim removed → `rounds are not strictly increasing`;
  - DP gate removed → the file exists;
  - and so on for the rest.
- Weak spots, non-blocking:
  - `test_nc_warm_base_main_trained` mutates the shared launch_fl deny-lists. It therefore proves that "the MAIN checks refuse"
    but not specifically `load_warm_base`'s own `_main_trained` call (:978). That call is exercised positively by the C_BIG
    run-id case.
  - No test shows select-warm refusing a run launched at η*/10 through `--server-lr` instead of the factor. The code refuses it
    (`r["tuning"] != tw`). This also keeps the 06M / B1 preview runs out of select-warm, as 06M M2 requires.

## Recommendation to Ops
- Accept FL-15. Build `code_ds3r5` = r4 + FIX-REEVAL + FL-15. For the 3 regenerated tests_fl receipts, take the r4 copies or record
  them as regenerated.
- Before the S7 / S8 **select-warm** read: add the m1 disclosure to the DL, and apply m2 if convenient (a 2-line guard + test).
- Before the first FT_FA or FT_FA_WARM device launch: apply m3 (a 3-line guard + test). m3 changes `device_code_sha256`, so decide
  before packing r5, or pack it in r6 before any device launch.
- m4 is optional.
