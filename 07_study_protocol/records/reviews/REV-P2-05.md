# REV-P2-05: independent read-only review of P2-05 (C1 central_ft, C2 c_full_small)
VERDICT: PASS_WITH_MINORS

Evidence: `pytest tests_int/test_p2_central_ft.py tests_int/test_p2_c_full_small.py -q` = 24 passed, 4 xfailed (the 4 NCs). sha256 of c_full.py (ec31af91), pretrain_small.py (1f5630c6), launch_central.py (2d56c222), v4train/trainer.py (f721cbdc), v4train/recipe.py (936db145) match the handoff; c_full/launch_central/pretrain_small mtimes predate the task. No existing file edited. No values read.

## Checks
1. theta_pre init: PASS. central_ft.py:446-452 wraps CentralTrainer.__init__; apply_c1_init (367-430) runs right after construction, asserts updates==0, exposures==0, empty AdamW state (377), loads via launch_fl.load_pretrain_init, pins cf08588c / ee33345b checked (385-389, defaults 77-78), apply_init_state proves the digest (391). Trainer inspected (trainer.py:337-352): the only construction-time derived state is AdamW (lazy, empty state), no EMA, no scheduler object (LR comes from planned table), no cached copy of theta0 beyond built.init_sha256 (identity string only). Tied aliases follow load_state_dict. No stale state survives. Init recorded in RUN_START["p2_c1"] (454-458), meta, summary, key_base (427).
2. Frozen tables: PASS. freeze_item_tables (fl/runtime.py:191-197) sets requires_grad False, re-points adapter._shared to the trained subset, no-ops post_step; a fresh AdamW is built from adapter.param_groups -> _shared (central_ft.py:402-404), so frozen tensors are in neither the optimizer nor the weight-decay groups; clip uses shared_parameters (trainer.py:871). Tests assert bit-unchanged at update 50 and 6.0 EFE.
3. C1 MAIN: PASS. bind_c1_selection (248-293): registered path only, file sha binding, format/self_sha/stage, p2_select re-verification, winner_peak_lr == --peak-lr. Labels P2_CONTROL, c_matched_eligible False (297-299). continue_run refused (482), auto_continue False (406), latest deleted (487-500). 6.0 EFE from trainer MAIN phase; mark 0 never evaluated.
4. C2: PASS. c_full_small.py:41-44 REQUIRED recipe SMALL_SASREC_D64_B2, seed 2026; ids TIEBREAK -> MAIN_VALIDATION (51), check_plan (203-212) proves it in the dry run; tuned recipes pinned to the production SMALL recipe (112-114 region, check_launch_args); peak-lr refused; c_full.py only imported (make_verify_wrapper), untouched.
5. S8: PASS. read_c1_run (688-748) opens only RUN_START.json before refusing a MAIN-role/non-CALIB run (699-709); test pins it. Shims print no metric values.
6. Server commands: PASS. Run roots are new ($R/C1, $R/C2, $R/dry, $R/SMOKE under runs_ds3/PH2), run ids derived, dry-run precedes launch, selection sha passed for MAIN. `break 2` in the loops is fine.

## Minor findings
- m1 (central_ft.py:763-765, :189-223): `--smoke` is accepted with `--role MAIN`, skipping selection binding; a smoke MAIN run evaluates MAIN_VALIDATION into SMOKE_EVAL.json (S8 exposure if opened). Suggest refusing --smoke with --role MAIN. The handoff commands only smoke CALIB, so no current exposure.
- m2 (c_full_small.py, handoff 6.2): the optional C2 smoke likewise writes a MAIN_VALIDATION number; handoff already says read exit code only. Acceptable.
- m3 (central_ft.py:285): `run_label` None accepted in the selection variant; harmless given p2_select re-verification.
- m4: the 3 failing tests in the wider tests_int suite belong to other packets (P2-02 / Windows file lock), not these files.
