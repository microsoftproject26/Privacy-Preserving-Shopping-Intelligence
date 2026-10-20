# REV-DELTA-02 — independent READ-ONLY delta review before code_ds3r3 (Claude Opus 5.5, 2026-09-29)

**Scope.** `company/tasks/REV-DELTA-02.md` plus the caller's additions. Deltas reviewed: FIX-DELTA-01 (F1, F2, F5–F7), FL-10
(FedAdam, FedAvgM, FL_CONST, client SGD-m, S2/S3 select-joint), FL-12/13 (S4 passes/batch, S7 FA_WARM, S8 frozen tables), INT-07
(gated TEST path), ANA-03 (K17–K21, F1 index side, F9, charged bytes), DATA-10 (PRETRAIN manifest + pretrain_small shim), Ops'
edits (`v4int/launch_central.py --coschedule-06g`, `v4int/pretrain_small.py` shared-GPU rule, `tests_fl/test_fl03_dropout.py`
`_old_digest`, `tests_int/test_fl10_fedadam_launch.py:279` S4→S5). Extra questions: (a) S2 comparability, (b) DP arms keep P = 2.

**Rules kept.** Only this file was written. Pytest and the extracted `code_snap_s2.tar` (= tree code_ds3s2) live under
`.local_ai/_pytest_tmp/REV-DELTA-02/`. No server command, no install, no git, no Codex. No RETRO_TEST row, no RESULT_ROWS of any
`runs_*` directory, no MAIN value was opened. Two read-only in-memory probes imported `v4int.lr_screen` / `v4int.launch_fl` (no file
written). Evidence tags: MEASURED (file:line or a command I ran), INFERENCE, OPINION.

**Quiescent tree (MEASURED).** No pytest / worker process was running at the start (process list). sha256 of all 556
`code/ tools/ contracts/` .py/.json/.md/.sh files was taken before the first suite (18:02 local) and after the last one (18:26):
**identical**. No `_p2work` or `__pycache__` was left. One oddity: the directory entry `contracts/manifest_registry/` got mtime
18:08:00 during my tests_int run, with no file added, removed or changed (listing and hashes identical). Another Claude session was
alive (process list). Not attributable from here; no content effect.

**File sha256 at review time (MEASURED; each = its handoff).** `v4int/launch_fl.py` e5015be4…; `v4int/lr_screen.py` 08d92b4a…;
`fl/server.py` 8c94ee53…; `fl/runtime.py` 443fdfc6…; `fl/client.py` 8a41e6d6…; `fl/comm.py` 2f8bd172…; `fl/dp.py` 516301a9… (= ds3r2);
`v4int/reeval.py` 4744a48d…; `v4int/launch_device.py` 48eee710…; `v4int/launch_central.py` 2d56c222…; `v4int/pretrain_small.py`
26d8004a…; `tools/retention_report.py` 10bc046a…; `tools/ana_block_index.py` bfc3fc72…; `tests_fl/test_fl12_frozen.py` e49b6d81…;
`tests_int/test_fl12_launch.py` a4dc5a42…; `tests_fl/test_fl03_dropout.py` 20e2b318…; `tests_int/test_fl10_fedadam_launch.py` 12063b9b….

---------------------------------------------------------------------------------------------------------------------
## Binding facts

| fact | result | evidence (MEASURED) |
|---|---|---|
| v4eval FROZEN | PASS | `v4eval.evaluator_code_sha256()` = `61b64e61c1a2e4f5…` on the local tree; every v4eval/*.py `cmp`-identical to ds3r2; tests_eval 80 passed incl. `test_evaluator_identity_is_the_deployed_a40_block_identity` |
| locked packages byte-identical | PASS (fl re-baselined by design) | vs ds3r2: v4train, v4data, v4data06, v4eval, v4models, v4sup, snap_v1, v4hist, v4fs_data, census_v2, ppsi_env.py identical. fl differs in exactly client.py, comm.py, runtime.py, server.py (FL-10, FL-12b; all additive, OFF by default). v4data08/d8_registry.py = DATA-08 C-F1 (REV-DELTA-01) |
| every default fl RunConfig / arm-spec digest byte-identical | PASS | FL-10 224-pair pin + tiny-world trajectories, FL-05/07/09 BASELINE pins, `test_fl12_launch.py:115` default digests + trajectory, `test_fl03_dropout.py::test_defaults_bit_identical_digests_and_payload`: all pass in my run. `fa_rounds(n, 2)` = ceil(60N/1152) = ceil(30N/576), the old value (probe: 6,827 / 1,707) |
| FedAvgM β = 0 ≡ FedAvg bitwise | PASS | `test_fl10_ladder.py:107`; code path `server.py` FedAvgM.step returns `agg` itself at β = 0, η = 1 |
| FedAdam = hand-computed reference | PASS | `test_fl10_fedadam.py:89` (float64 hand loop, 3 rounds) and `:109` (independent numpy-float32 replica, bitwise); m_0 = 0, v_0 = τ², no bias correction; NCs `:241` (bias-corrected) and `:249` (v_0 = 0) fail as intended |
| bitwise resume of server moments | PASS | `test_fl10_fedadam.py:202`, `test_fl10_ladder.py:115`, S4 `test_fl12_launch.py:253`, warm `:337`, frozen `:348`, `test_fl12_frozen.py:172`; runtime refuses a FedAdam resume without moments and the reverse (runtime.py, FL-10 hunk) |
| no real RETRO_TEST row read | PASS | INT-07 tests use the DATA-09 synthetic population + real TRAIN rows only; `test_d9_real.py` 5 skipped (no gate locally); no `company/gates/ALL_METHODS_LOCK.json` |

---------------------------------------------------------------------------------------------------------------------
## 1. FIX-DELTA-01 checklist (REV-DELTA-01 findings)

| item | status | evidence |
|---|---|---|
| F1 reeval record ↔ index | FIXED (both sides) | `reeval.py:370-376` writes top-level `result_rows_sha256` / `ranks_sha256` + `outputs`; `ana_block_index.record_hashes` reads both forms, refuses disagreement / absence; `tests_ana/layoutkit.py` now writes through the real `write_outputs` (ANA-03) |
| F2 MAIN selection binding | FIXED as packeted, but see M2 | `launch_fl.py:1475-1516` `bind_main_selection`: path must resolve to the registered file, sha hashed before/after, `check_ext_for_main` (06B refused, first-screen sha = registered file with DL 181 prefix, screened ⊇ 06F set, x32 rule, lr_peak_base) and `check_tuning_for_main`; tests `test_fix_delta01_fl_binding.py` |
| F4 `_old_digest` | FIXED (Ops) | `test_fl03_dropout.py:199-205` pops `server_opt`, `lr_shape` (`pop(k, None)`); the assertion is still real-digest == old-rule digest, so the test keeps its power |
| F5 p2kit isolation | FIXED | `_p2work/<sha(basetemp)[:12]>`; tests_train 214 passed alone; residual same-suite sharing (conftest.work, tests_int/_p2work) noted by the handoff, unchanged |
| F6 selection manifest re-verify | FIXED | `launch_device.py:624-636` `verify_selection_manifest` (UNSEEN_VAL eval manifest through `providers._verify`, binding + registry entry + 12,792 users); stubbed locally, needs one server smoke (handoff) |
| F7 LO without --selection | FIXED | `launch_device.py:160-166` `check_lo_selection` (non-smoke LO off UNSEEN_VAL needs --selection; TEST roles included) |
| F9 C_BIG `…:best` id | FIXED | `retention_report.py:909-913` + `deployed_c_big_best` (sha = ARCHITECTURE_LOCK c_big best.pt per seed) |
| F3 process (quiescent tree, review before pack) | SATISFIED by this review | quiescence above |

---------------------------------------------------------------------------------------------------------------------
## 2. FL-10 (FedAdam, FedAvgM, FL_CONST, client SGD-m, S2/S3 selection)

- **FedAdam** (`server.py` FedAdam.step): Δ = new − θ_r, m = β1 m + (1−β1) Δ, v = β2 v + (1−β2) Δ², θ += η_s m / (√v + τ); one
  elementwise FP32 op at a time; moments committed only when every key is finite. Keys = `manifest.shared_keys`, so frozen S8 tables
  and buffers are never stepped. Every θ change goes through `Server.apply` (run_round, pool, DP noise-only release). Correct against
  Reddi et al. 2021 Alg. 2 (MEASURED by the tests above).
- **v_0 = τ²** departs from the text of 06G §1 ("m_0 = v_0 = 0"). It follows Alg. 2 ("v_{−1} ≥ τ²"). The departure appears only in
  DL 211/212 and the FL-10 handoff. OPINION: the choice is correct, but the amendment text must be corrected (erratum row / note on
  06G §1) before any S2 value is read. The S2 runs already use τ² (m9).
- **FedAvgM** is correct at the registered η = 1: θ' = agg + β m_prev, which is algebraically θ + m_r.
- **FL_CONST** (`runtime.fl_const_factor`, exact Fraction): 100x up to 1 %, then 1, then 91/10 − 9x over the last 10 %. It is
  continuous at x = 0.01 and x = 0.9, and the final factor is exactly 0.1 = 06I §3.
- **Client SGD-m** (`client.py`): fresh SGD per visit, momentum 0.9, dampening 0, no Nesterov, weight decay 0, clip kept. The default
  `LocalSolver` is unchanged (no new dataclass field).
- **Launcher.** Only registered values are accepted. Tuning keys are written only when non-default, and `spec_tuning` re-verifies them.
- **select-joint S2/S3** follows the 06J tie orders. Edge runs are accepted only if their condition holds, and two-run edges must be
  complete. The file is O_EXCL, self-sha'd, and prints the winner only.
- **Defects:** 06J §4(ii) is missing (B1), and edge runs are allowed but never required (M3). See §9.

---------------------------------------------------------------------------------------------------------------------
## 3. FL-12/13 (S4, S7 FA_WARM, S8 FA_WARM_FROZEN)

**S4 round count.** MEASURED by probe: `fa_rounds` gives 13,654 / 6,827 / 3,414 MAIN and 3,414 / 1,707 / 854 CALIB for P = 1 / 2 / 4.
- It equals ceil(6.0 N_dec / (64 × 0.9 × P × N_dec / N)), the 06J §3 formula. Batch leaves exposures unchanged.
- Round-count runs normalise marks and the LR schedule by the planned E_total (`runtime._half_units`, the lr_table). So the 12 marks
  and the "central" LR shape are identical in exposure terms at any P.
- One at a time is enforced (`tuning_config` refuses both at once). The spec records `exposure_budget`, and `spec_tuning` re-verifies
  it. A non-default P requires the registered N (`launch_fl.py` "S4 exposure budget assumes N").
- INFERENCE: with P = 4 the sweep covers only 1.67 visits per client. The realised planned_efe is therefore a random-partial-sweep sum
  near 6.0, not exactly 6.0. It is recorded in `lr_table.planned_efe`; Ops should quote it per S4 run (m7).

**S4 selection.** `lr_screen` requires P1, P4 and B8 at the S3 winner's point, allows no edges, and its core-point check sits in
`load_fl_tuning`.
- `load_fl_tuning` now requires `stage_history` = S1b..previous stage.
- The tie order B8 → P1 → P4 after the incumbent is the worker's reading of 06J "ties → incumbent". Register it before any S4 read (m9).

**S7 warm start** (`load_pretrain_init`, `pretrain_identity`, `check_init_identity`, `apply_init_state`):
- The checkpoint bytes are hashed before `torch.load(weights_only=True)`.
- The following must all hold: kind `endpoint_6.0`; run label, run id prefix and train manifest `PRETRAIN_TRAIN_5_100_NONMAIN`; family
  and variant as registered; no MAIN / CALIB / FULL manifest; no C_BIG / C_SMALL / C_FULL / C_MATCHED run id.
- The loaded broadcast state must hash to the checkpoint's own `state_digest`, which becomes the server's `init_sha256`.
- MEASURED interface check: v4train `_save_weights` writes `kind`, `meta.state_digest`, `meta.run_label`, `meta.model_variant` and
  `meta.manifest_registry.train_manifest` (trainer.py:628-636, 966-969). `pretrain_small.fl_init_record` writes `run_id`, `run_label`,
  `train_manifest` and `fl_init.{kind,path,sha256,bytes}` (pretrain_small.py:288-303). The two agree.
- Tested only on a synthetic v4train-shaped checkpoint, so one server `--dry-run` is required (handoff).

**S8 frozen tables** (`runtime.freeze_item_tables`):
- The three keys become FIXED buffers: no grad, not in the optimizer / clip / FedProx, not uploaded, not aggregated, and worker copies
  are verified bitwise at each visit.
- Recentering is off for the frozen output layer. That is correct: softmax is invariant to the common shift, and re-applying it would
  change the bits.
- The freeze sits in the manifest digest, so an unfrozen resume is refused. Pools are refused.
- DP clip and noise run over the uploaded keys (`test_fl12_frozen.py:154`).
- Bytes: per-visit bytes = uploaded parameters. The frozen tables are charged once per visited client.
- Reporting nit (m6): after the freeze the frozen tables are FIXED buffers, so `per_visit.bootstrap_bytes_once` (comm.py:76) now also
  contains them, next to `frozen_bytes_once`. No per-client total double counts (`per_client_incl_frozen_once` uses `frozen_bytes_once`
  only). But the record shows the same bytes twice, and a dropped-at-first-visit client is not charged the download.

**Warm LR factor.** S7 {1, ¼} and S8 {1, 4} multiply the tuning point's local LR. For client SGD this means lr/4 or 4·lr, which is the
reading of 06K's "m*/4" for SGD (register).
- A MAIN warm run takes the factor only from the registered `FL_WARM_SELECTION_S7|S8.json`, bound to the same tuning sha and θ_pre.
- OPINION: sound.

**ANA side of S7.** RUN_SUMMARY `arm` = `FA_WARM` (`execute` uses `spec["arm"]`), and the index accepts it when the directory is named
`FA_WARM_s<S>`. `FA_WARM_FROZEN` is not an ANA arm: `--fl` refuses it (fails closed). S8 stays outside the block, and 06K's
"≥ 0.002 over S7 → 2 seeds" rule is an Ops step with no code.

---------------------------------------------------------------------------------------------------------------------
## 4. INT-07 (gated TEST path)

**VALIDATION branch unchanged.** MEASURED diff vs ds3r2:
- `cache.build_subset_store(test_gate=None)` runs the old code path verbatim (refuse_test_split, store_splits, CleanV2Store).
- `providers.StoreEvalProvider` adds the TEST branch only when the verified header says `v4_split == RETRO_TEST`. A `test_gate` on a
  non-TEST manifest is refused.
- `comparison_key_base` gets its split from `_key_split`, which is VALIDATION unless the provider is gated RETRO_TEST (the gate is
  re-validated there). A RETRO_TEST header without a gate is refused.
- The fresh-interpreter no-side-effect proof (int07_probe) and the identity checks on the locked refusals pass (tests_int 239 passed).

**TEST store.** Every retro_store entry point re-validates the GateRecord. A kept decision that is not a TEST-partition decision
refuses the build (cache.py gated hunk).
- The window rule is history_splits extended by TEST, and it is causal (`check_windows` + NC).
- The frozen normalisers are reused with no refit.

**Consumers.** reeval, launch_device, v4base and dump_rows_retro behave as the handoff describes.
- launch_fl refuses every TEST role through ROLES = {MAIN, CALIBRATION} (pinned in tests_data09). No explicit refusal was added
  (acceptable).

**Open (unchanged from the handoff).**
- The real CUDA paths are server-only.
- The v4dev `dump_rows` crash on string batch fields (DEV owner) affects §8.2 ONNX `--role MAIN` (m12).

**Verdict:** ACCEPT.

---------------------------------------------------------------------------------------------------------------------
## 5. ANA-03

**Contrasts.**
- K1–K19 are verbatim from `ANA_CONTRASTS_v2.json`, and the file's self-sha verifies.
- The v1 file is accepted only with `--split VALIDATION --allow-v1-contrasts` and never for TEST (NC).
- The multiplicity line is generated from the loaded file.
- K20/K21 are DESCRIPTIVE until Ops registers `ANA_CONTRASTS_v3.json` (K1–K21) before ALL_METHODS_LOCK.

**Other items.**
- C_FULL pairing is the same as K15.
- PRE uses one seed and is shared as the denominator of both FA_WARM seeds (DL 206 (2)).
- F1 and F9 are fixed (§1).
- Every NC fails with AssertionError: 24/24 in my run.

**Defect M4.** DL 218 (Ops decision) makes the registered charged-bytes variant **downloads-only** (06I §7b). But
`retention_report.py:1020-1040` makes `bytes_per_client_charged_dropped` the up+down variant, and that variant is printed in the
retention table, in the privacy-ladder JSON and in the `cum_bytes_per_client_charged_dropped` curve (:1970-2002, 2721, 2779). The
download-only value sits only in the JSON as `…_download_only`. Swap them (or rename) before the ANA freeze.

---------------------------------------------------------------------------------------------------------------------
## 6. DATA-10 and the Ops edits

**DATA-10.** ACCEPT.
- Band = TIEBREAK − MAIN, with the band hash pinned as a hard check (`d10_common.py:68`, d10_build.py:87-92, 210-213).
- Every read goes through `read_non_test`.
- The registry extension keeps REGISTRY_V2 entries byte-identical.
- The shim labels rows PRETRAIN_FOR_FL, which every selection path refuses.
- tests_data10: 34 passed, 19 real-data skipped, 19 NCs failed (AssertionError).
- C3 auto-continuation to 8.0 EFE does not move θ_pre, which is endpoint_6.0 (checkpoint.py KINDS).

**`launch_central --coschedule-06g`.** Correct.
- Without `--allow-shared-gpu` it is refused. With it, a non-smoke central MAIN run may share a card, and `preflight.coschedule_06g` is
  recorded (launch record only, never in a config sha).
- STRICT FP32 numerics do not depend on co-scheduling (STAGGERED_P3 ruling), so θ_pre is unaffected (INFERENCE).
- m4: the flag lets ANY raw launch_central MAIN run share a card, while 06K §1 names PRE only. `c_full.py:91-92` still refuses the
  shared GPU for C_FULL.

**`pretrain_small`.** `--allow-shared-gpu` requires `--coschedule-06g`, and the reverse (pretrain_small.py:112-116). PRE was launched
with both flags, so neither refusal can have changed the PRE launch.
- I cannot see the server's code_ds3r2p bytes. Ops should record in a DL row the sha of ds3r2p's `pretrain_small.py` / `launch_central.py`
  against the local pre-FIX a5550479 / a4ded7da, so that θ_pre's provenance is closed (m4).

**Test fixes.**
- `test_fl03_dropout.py`: correct (§1).
- `test_fl10_fedadam_launch.py:279` now asserts that "S5" is refused. That is correct but weak. S4 is covered by `test_fl12_launch.py:199-252`.

---------------------------------------------------------------------------------------------------------------------
## 7. Question (a): are the S2 CALIB values from code_ds3s2 comparable with later stages? **YES** (INFERENCE from MEASURED diffs)

**What differs.** I extracted `code_snap_s2.tar` (launch_fl f2840329, fl/server 8c94ee53, both verified) and diffed all 463 .py files
against the current tree. Only five files differ, plus two new tests:
- `fl/runtime.py`: freeze helper, plus a frozen-keys check that is inert when nothing is frozen;
- `fl/comm.py`: `visit_bytes(…, frozen_keys=())`, where the default returns the same dict;
- `v4int/launch_fl.py`: S4 / warm / frozen options, all OFF by default;
- `v4int/lr_screen.py`: S4, select-warm, `read_run(arms=…)`;
- `tests_int/test_fl10_fedadam_launch.py`.

**Why the values stay comparable.**
1. Numerics are unchanged at the S2 points. The arm spec, RunConfig digest, LR table and marks are byte-identical at default passes and
   batch (pins pass), and every FL-12/13 branch is gated on a non-default key. An S2 configuration re-run on the new tree would give
   the same bits. The evaluator is identical.
2. Selection works across trees:
   - The new `read_run` accepts ds3s2 runs: the solver passes/batch 2/16 equal the tuning point, there are no warm keys, and row labels
     are compared only when present.
   - `_strip_lr` equality holds between ds3s2 and new-tree specs, because default keys are absent from both.
   - `_check_incumbent` compares only the RESULT_ROWS sha and the value.
   - An S2 file written by either tree has `stage_history` = [S1b], which satisfies the new chain check.
   - `load_fl_tuning` reads configs with `.get`, so missing passes/batch keys read as the defaults.
3. `fl_code_sha256` differs, which only means an S2 run must finish (or resume) on ds3s2 (06E §8).

**Condition.** Run the S2 `select-joint` with the FIXED lr_screen (B1 and M3 below), not the ds3s2 copy. Selection is not a resume,
so any tree may select.

**Minor gap.** Neither `read_run` nor `select-joint` checks that the runs share one device class / pin (ARM_SPEC `device`). Add the
check (m5).

---------------------------------------------------------------------------------------------------------------------
## 8. Question (b): DP arms (S_CAL, FA_1024, DP) keep 2 local passes whatever S4 selects — **SOUND, but the code does not implement it yet**

**Why it is sound** (INFERENCE from the registered mechanism):
- For the Poisson arms, EFE = T × q × P × V / N_dec = 384 × (1,024 / 131,072) × P = 3P. So P = 2 is the only pass count that keeps
  both the registered T = 384 (the accountant's, and so z(ε)) and the 6.0-EFE equal-exposure budget.
- Changing P would force either a new T, and with it new z values registered before the run, or a broken budget.
- K12 / K13 = R(DP / FA_1024) keep numerator and denominator at P = 2, and so does 06H DP_OPT_SENS. The privacy-cost contrasts are
  therefore not confounded by the pass count. No confirmatory contrast pairs FA (possibly P ≠ 2) with FA_1024 / DP (K1–K19 checked).
- The DP arms then run exactly at the S3 winner's point, where their LR was tuned (at P = 2): coherent.

**Costs to register and disclose:**
- It departs from 06I §2 / 06J §3 ("the S4 winner is FA for every FL arm"). DL 219 registers it before any S4 read, which is enough if
  it is also mirrored in the ladder register and the thesis text.
- The descriptive privacy-ladder figure FA → FA_1024 → DP8 would mix pass counts when S4 picks P ≠ 2.

**Batch.** Local batch does not enter the user-level DP-FedAvg accountant (clip per user delta, noise per round). Either choice is
therefore valid, but the registration and the code must agree:
- DL 219 says batch 16;
- the code gives the Poisson arms the S4 batch (`launch_fl.py:379-381` applies batch 8 to every arm; probe: S_CAL / FA_1024 / DP accept
  B8).

**Code gap (M1, MEASURED by probe).**
- `_apply_tuning` refuses P ≠ 2 for S_CAL / FA_1024 / DP (`launch_fl.py:382-386`).
- `bind_main_selection` forces every non-smoke MAIN run onto `--fl-tuning` once `FL_TUNING_SELECTION.json` exists. There is no override
  path.
- So if S4 selects P1 or P4, **no DP arm can be launched**.
- The S4 file also claims `applies_to` [..., "FA_1024", "DP", "S_CAL", ...] (`lr_screen.py:592`).

---------------------------------------------------------------------------------------------------------------------
## 9. Findings (BLOCKERS first)

**B1 (BLOCKER for every select-joint S2/S3/S4; not for the running screens) — 06J §4(ii) is not implemented.**
- 06J §4(ii): "a stage whose best candidate gains < 0.002 over its incumbent is recorded 'no measurable gain' and the incumbent carries on".
- `lr_screen.apply_rule_stage` (lr_screen.py:171-199) picks the best, with only the 0.0005 tie band. `load_fl_tuning` re-applies the
  same rule, so the check it provides is the same flawed rule.
- MEASURED probe:
  - S2 incumbent vs a FedAdam run +0.0012 → FedAdam wins;
  - S4 incumbent vs P1 +0.0013 → P1 wins.
  - The registered rule gives the incumbent in both cases.
- The S2 τ-edge condition (`_allowed_edges`, lr_screen.py:505-508) uses the same rule, so the edge decision can be wrong too.
- No test mentions it (grep "0.002" / "measurable": none).
- Fix:
  - In `apply_rule_stage` (every stage), after the tie set: if max(non-incumbent finite values) − incumbent value < 0.002, then
    winner = incumbent and tie set = [incumbent], and the file records `no_measurable_gain: true` and `gain`.
  - Use the same winner for the τ-edge condition.
  - `load_fl_tuning` must recompute it.
  - Add a positive test and an NC.
  - The S2 file must be written by the fixed tree.

**Major findings:**

| # | finding | severity | file:line | fix | who |
|---|---|---|---|---|---|
| M1 | Poisson arms cannot launch under an S4 file with P ≠ 2, and would silently take B8 against DL 219 "batch 16" (§8) | MAJOR (conditional on the S4 outcome; blocks DP / FA_1024 / S_CAL) | launch_fl.py:379-386, 1475-1516; lr_screen.py:592 | Arm-aware resolution. For Poisson arms the tuning file's winner applies with local_passes = 2 (and batch per ONE registered rule: 16 per DL 219, or the S4 batch; pick one and register it). Record `poisson_local_rule` + DL ref in spec, LAUNCH and rows. The S4 file's `applies_to` notes the exception. Add test + NC | Opus (FL follow-up, before S_CAL) |
| M2 | Non-smoke MAIN accepts `LR_SELECTION_EXT.json` today (the S1b point), and FL_TUNING at stage S2/S3. 06G §4 / 06I §3 HOLD / 06J §3 require the S4 file for every MAIN FL arm | MAJOR (registration conformance; guard gap) | launch_fl.py:1401-1409, 1449-1516 | Non-smoke MAIN: kind must be `fl_tuning` and stage `S4`. A registered descriptive ladder row (LADDER_S1a type) needs an explicit label flag recorded in LAUNCH | Opus |
| M3 | Registered edge runs are optional: select-joint writes the stage file when an edge condition is TRUE but its runs are absent (06J §1-§2 "→ two runs" / "→ one run") | MAJOR (a registered step can be skipped silently) | lr_screen.py:488-518, 551-557 | Refuse when a condition-true group is absent and list the runs to launch; `load_fl_tuning` re-checks | Opus |
| M4 | Charged-bytes primary column = up+down, while DL 218 registered downloads-only | MAJOR (blocks the ANA freeze) | retention_report.py:1020-1040, 1970-2002, 2721, 2779 | Swap the primary / DESCRIPTIVE variants (or rename them); test | Sonnet (ANA-03b) |

**Minor findings:**

| # | finding | severity | file:line | fix | who |
|---|---|---|---|---|---|
| m4 | `--coschedule-06g` open to any central MAIN run (06K §1 names PRE); server ds3r2p bytes of the shim / launcher unverified | MINOR | launch_central.py:340-347 | Restrict to the pretrain_small shim, or register the generalisation; DL row with the ds3r2p shas vs a5550479 / a4ded7da | Ops |
| m5 | select-joint / select-warm do not check one device class / pin across runs | MINOR | lr_screen.py `read_run`, `selection_stage` | Compare ARM_SPEC `device` across runs | Opus |
| m6 | Frozen bytes appear in both `bootstrap_bytes_once` and `frozen_bytes_once`; a first-visit dropout is not charged the table download | MINOR (reporting) | fl/comm.py:58-76; launch_fl.py `bytes_record` | Exclude frozen keys from bootstrap, or document; disclose the dropout case | Opus |
| m7 | S4 P1/P4 realised planned_efe is not exactly 6.0 (partial sweep) | MINOR (disclosure) | runtime lr_table | Quote `lr_table.planned_efe` per S4 run in the ladder register | Ops |
| m8 | ana_block_index accepts any PRODUCTION MAIN FL dir named `<ARM>_s<S>`; LADDER_S1a or a pre-tuning run could enter as K2's FA by a naming slip | MINOR (process guard) | ana_block_index.py:264-320 | Require LAUNCH `main_selection.kind == fl_tuning` with the registered sha for FL block entries | Sonnet |
| m9 | Registration records owed before the reads: 06G §1 v_0 = τ² erratum (before S2 values); S4 tie order B8 → P1 → P4; S7/S8 factor on client-SGD LR; F2's "x16 rule winner" reading (DL 193) | MINOR (records) | contracts/PREG1_AMENDMENT_06G.md §1; DL | One DL / erratum row each, before the corresponding read | Ops |
| m10 | Local tree lacks `contracts/fl/LR_SELECTION.json` (first-screen file, sha cdf6663d…), which `check_ext_for_main` requires | MINOR (Ops) | launch_fl.py:1424-1430 | Ensure code_ds3r3 / the server tree carries it | Ops |
| m11 | K20/K21 DESCRIPTIVE until `ANA_CONTRASTS_v3.json` is registered; PRE-alone reeval row form (PRETRAIN-manifest checkpoint) unverified | MINOR (planned) | retention_report.py | Register v3 before ALL_METHODS_LOCK; server `reeval --dry-run` of θ_pre on MAIN first | Ops |
| m12 | v4dev `dump_rows` fails on string batch fields (INT-07 finding) | MINOR (other owner) | v4dev/dump_rows.py | DEV owner fix before §8.2 | eda-3b |

---------------------------------------------------------------------------------------------------------------------
## 10. Running server artefacts — affected? (INFERENCE, with reasons)

**LADDER_S1a (code_ds3r2s):** numerics not affected. The FedAvg default path is bit-identical under FL-10/12/13, and v4eval is
unchanged. It cannot resume under a newer tree (fl_code_sha, 06E §8). It stays DESCRIPTIVE. m8 is about keeping it out of `--fl`.

**C_FULL ×2 (ds3r2, through c_full.py):** not affected. v4train and v4eval are identical, c_full.py is unchanged, and the shared GPU
is still refused there. Resume with its own tree.

**The 06F S1b screens:** not affected. Their selection is written, and the values come from an unchanged evaluator and FedAvg path.

**The S2 screens (ds3s2):** VALID and comparable (§7). Finish them on ds3s2 and select with the fixed lr_screen (B1, M3).

**PRE (ds3r2p):** not affected. Co-scheduling and the two added refusals do not change its numerics or its launch. θ_pre is usable
once `PRETRAIN_FOR_FL_INIT.json` exists and one FA_WARM `--dry-run` passes the identity checks.

**The 8 A40 re-evaluations:** valid (REV-DELTA-01). Their records carry the top-level hashes the index reads.

**Nothing needs to be re-run from code_ds3r3.**

---------------------------------------------------------------------------------------------------------------------
## 11. Test runs (MEASURED; venv python 3.13, torch CPU, `-B -p no:cacheprovider`, per-suite basetemp/TMP under `_pytest_tmp/REV-DELTA-02`, run ONCE sequentially on the quiescent tree, receipts `…/REV-DELTA-02/<suite>/{out,nc_out}.txt, *junit.xml`)

| suite | main run | NC run (`--runxfail -k test_nc_`) | NC exception types |
|---|---|---|---|
| tests_fl | 248 passed, 75 xfailed, exit 0 | 75/75 failed | AssertionError only |
| tests_int | 239 passed, 67 xfailed, exit 0 | 67/67 failed | AssertionError (1 = HistoryLeakError ⊂ AssertionError) |
| tests_eval | 80 passed, 28 xfailed, exit 0 | 28/28 failed | AssertionError only |
| tests_ana | 100 passed, 24 xfailed, exit 0 | 24/24 failed | AssertionError only |
| tests_data10 | 34 passed, 19 skipped (real data), 19 xfailed, exit 0 | 19 failed, 2 skipped | AssertionError only |
| tests_data09 | 73 passed, 5 skipped (real-only), 28 xfailed, exit 0 | 28 failed, 1 skipped | AssertionError only |
| tests_data08 (D8_TESTS_SYNTHETIC_ONLY=1) | 22 passed, 8 skipped, 14 xfailed, exit 0 | 14 failed, 2 skipped | AssertionError only |
| tests_dev | 73 passed, 5 skipped (onnxruntime absent), 13 xfailed, exit 0 | 13 failed, 2 skipped | AssertionError only |
| tests_base | 64 passed, 10 xfailed, exit 0 | 10/10 failed | AssertionError only |
| tests_trunc | 42 passed, 11 xfailed, exit 0 | 11/11 failed | AssertionError only |
| tests_train | 214 passed, 61 xfailed, exit 0 | 61/61 failed | AssertionError only |
| tests_hist | 62 passed, 1 skipped, 13 xfailed, exit 0 | 13/13 failed | AssertionError only |

Pre/post sha256 of 556 tree files are identical; locked packages are unchanged against ds3r2 (binding facts).

---------------------------------------------------------------------------------------------------------------------
**VERDICT: ACCEPT_WITH_FIXES.**
- FIX-DELTA-01, FL-10 (maths), FL-12/13, INT-07, ANA-03 (except M4), DATA-10 and the Ops edits are accepted. v4eval is frozen, every
  default digest is byte-identical, FedAdam / FedAvgM / resume are proven, and no running artefact is invalidated.
- **B1 must be fixed before the first select-joint (S2).**
- M1–M3 must be fixed before S_CAL / any MAIN FL launch.
- M4 must be fixed before the ANA freeze.
- code_ds3r3 may be packed once B1 and M1–M3 are fixed and their tests pass on a quiescent tree. A focused delta check of those
  fixes suffices; no full re-review is needed.
