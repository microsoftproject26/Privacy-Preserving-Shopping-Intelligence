# G1 COMBINED GATE — Fable 5.1 ruling (verbatim; owner-proxy where applicable; received 2026-09-27)

Packet: `company/reviews/G1_COMBINED_GATE_PACKET.md`. Saved verbatim by the Operations Director.

VERDICT: APPROVE_WITH_CONDITIONS

Assumption: Ops' PROJECT_MEASURED claims (suite counts, server-hash identity) are taken as true; I verified locally FS_FINAL.json, DATA_PROTOCOL_LOCK_V2.json, REGISTRY_V2 source field, providers.py L388–422, verify.py L395–408, PREG1 files and ADR-002/005.

A (S* lock): CONFIRM. FS_FINAL: all five dropped groups (G3, G4, G7, G11, G12a) are within_noise for BOTH families; worse family SASRec −0.13% (UB +0.86%) passes ≤1.0%/≤2.0%; GRU −0.90% (UB −0.15%). Addendum 3 is a correct mechanical application of R2+R4. G8 stays a recorded finding; not reopened.

B (DATA_PROTOCOL_LOCK_V2): APPROVE as the production data lock. One-way binding: YES. Keep REGISTRY_V2 byte-identical; LOCK_V2 → REGISTRY_V2 by self-hash is the only non-circular direction, and regenerating the registry would void the server-measured DATA-06 receipt. Conditions: B1 `v4int.verify` (L406–408 today requires registry.source == lock self, so it would REFUSE REGISTRY_V2 under LOCK_V2) accepts iff BOTH registry.source_data_protocol_lock.lock_self_sha256 == LOCK_V2.supersedes.lock_self_sha256 (da51d271…) AND registry.self_sha256 == LOCK_V2.population_and_manifests.manifest_registry.self_sha256 (f6a08207…); any other pairing refused; one negative control per clause. B2 DataPaths default lock_path → LOCK_V2; v1 verification is HISTORICAL only.

C (G1 code):
R2: eval_stream = "EVAL_ROWS_ONLY" (contract name in the frozen config and the key; v4eval's E2E_ROWS mapping stays internal, recorded in eval_stream_source; ALL_ROWS refused). T3 receipt: re-BIND only if evaluator_code_sha256 is unchanged; if the v4eval tree changes, T3 is re-RUN on the A16 (all bitwise items) inside R5.
R4 comparison key (closed list): data_protocol_hash = bc699a97… (LOCK_V2 self); data_version = CLEAN_V2_POP5100_V2; manifest_registry_self_sha256 = f6a08207…; feature_schema_self_sha256 = c53d47fe…; feature_contract_hash = input_contract_sha256(S* self, USER_HISTORY_L50, None when state == locked); history_policy = USER_HISTORY_L50; transform_state_sha256 = 2ff0ff4e…; split = VALIDATION; label_policy_hash = _fp(LOCK_V2.splits_and_labels); population_hash = training manifest header population_sha256 (MAIN 563e7663… / CALIB 9be38de1…); candidate_order_hash = eval manifest header; train_manifest_id; eval_manifest_id; determinism_mode = STRICT; eval_stream = EVAL_ROWS_ONLY; exposure_point = end; T3 additions (eval_score_block 158486, eval_batch 1024, eval_device_class A16-FP32-DET). Every value explicit (P1); no code defaults.
INT-02 scope (closed):
1. launch_central passes manifest_registry (REGISTRY_V2), data_version, schema_id S*; binds LOCK_V2.
2. verify: B1/B2 + negative controls.
3. providers.comparison_key_base: re-point reads to LOCK_V2 paths (features_and_transforms.production_schema.self_sha256; shared_statistics.transform_state_sha256 — the v1 paths schema_self_sha256 / transform_state_self_sha256 do not exist in v2 → KeyError today) and emit the R4 fields.
4. v4eval.frozen_config carries eval_stream; fingerprint recomputed; T3 per R2.
5. tests_int test_resume_exact / test_smoke_selection → TRAIN-01R2 keys; tests for items 2–3.
6. v4train.selection: only if a literal M02/v1 manifest id is an acceptance condition, replace it with the LOCK_V2 MAIN eval role; docstrings need no change; thresholds untouched.
7. Nothing else: no v4data/v4data06 (d6 code sha is lock-bound), model, or schedule edits.
G1_APPROVED.json: YES, Ops may write it WITHOUT a second Fable call iff INT-02 stayed within items 1–7 and R5 = one complete OPS-07 server run of ALL suites (incl. tests_int and the tests_train_gpu probe), exit 0, 0 failures, every NC failed as intended, per-suite passed counts ≥ the accepted counts. Fields: gate "G1"; ruling {path, sha256 of this verdict as saved}; conditions [B1, B2, R2, R4, INT-02 items 1–7] each with status + evidence path; r5_receipt {path, sha256, utc, exit_code, per-suite passed/NC}; code_tree_sha256 per package at R5; lock_v2_self; registry_v2_self; schema_self; fs_final_sha256; preg1_lock_values_sha256; preg1_amendment_01_sha256 (item 5 CONFIRMED); eval_fingerprint + t3 {rebind|rerun, receipt sha}; comparison_key_template (the R4 list); determinism_mode STRICT; eval_stream; proxy_pending_owner (packet §E); fable_second_call false; written O_EXCL with self-hash.

D (LR calibration + contest): CONFIRM with amendments. Grid {1.0, 0.5} × executed peak per family, the peak read from the registered v1 run config (sha cited), not the literal 1e-3. 2 EFE on CALIB_TRAIN_5_100_32768 (N_EFE 646,067) = the first 2 EFE of the common 6-EFE schedule on the 6-EFE clock; seed 2026; STRICT; S*; B 256; marks every 0.5 EFE on CALIB_VALIDATION_5_100_32768, E2E client-macro MRR@20. Selection per family: higher metric at the 2.0-EFE mark (endpoint read, not best-of-marks); both candidates finish 2.0 EFE before either number is read. Tie |Δ| ≤ 0.0002 → 1.0× (equal evidence = no change). Sanity: a candidate with a non-finite loss/metric or a 2.0-EFE metric < 0.5× the other is excluded; both excluded → trigger. Rows LR_CALIBRATION, population CALIB, refused by selection; MAIN_VALIDATION never read. Contest as stated (MAIN, S*, seeds 2026+2027, 6 EFE, STRICT, marks every 0.5 EFE, calibrated peak explicit per run config); G2 mechanical only on DECISIVE_STABLE.

TRIGGERS:
1. R5 fails, or any change outside INT-02 items 1–7.
2. Evaluator code changes and the T3 re-run is not bitwise.
3. Calibration: both candidates excluded, or a non-finite run.
4. Contest: any T1 trigger (T1a–d), HOLD, or ARCHITECTURE_EQUIVALENT (GRU-on-cost is owner-pending; default HOLD).
5. Any tie-break launch (finding 3 first).
6. Disk-guard breach, or a STRICT run that is not bitwise reproducible.
7. Any edit to REGISTRY_V2, LOCK_V2, FS_FINAL, the S* schema or DATA-06 code.

FINDINGS/RISKS:
1. verify.py L406–408 refuses REGISTRY_V2 under LOCK_V2 as written; B1 is mandatory.
2. comparison_key_base reads v1-only lock paths; KeyError under LOCK_V2 until item 3.
3. PREG1_LOCK_VALUES T1 still names M02/262,144/62,368 and the full-TRAIN tie-break (2,694,998 / 642,228); ADR-005 R2 D9 supersedes these to the MAIN eval role and the TB band; thresholds unaffected. Needs PREG1_AMENDMENT_02 (registration only) before any tie-break; not a G1 blocker.
4. G8: permuting helps GRU (both seeds, CI < 0) while costing SASRec; kept by rule; thesis finding only.
5. The executed peak 1e-3 is asserted, not cited; source it from a registered value.
6. Calibration resolution is coarse (12,715 eval users, 2-point grid); the tie rule guards against noise-driven LR flips.
7. Tolerated deviation calib_prefix_is_first_65536 must remain tolerated, never "fixed" in DATA-06 code.
