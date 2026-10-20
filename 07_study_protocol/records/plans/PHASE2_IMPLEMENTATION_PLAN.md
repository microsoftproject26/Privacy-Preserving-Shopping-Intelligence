# PHASE2_IMPLEMENTATION_PLAN — work packets for PREG2 under T0a (PLAN; nothing runs before PREG2 is registered)

- Companion of contracts/PREG2.md (CANDIDATE, pending Fable's delta re-review). "§" points there. Revised 2026-10-03 after
  the Fable gate company/reviews/FABLE_GATE_ENDGAME_SCRIPTS_20261003.md §9 (GO-WITH-EDITS): **O1 = T0a** (one TEST sitting
  for Phase 1 + Phase 2). **The T0b branch is removed** (Fable refused it). TEST2 is only an optional replication. **Every
  compute or time figure is INFERENCE** unless marked MEASURED.
- Ground rules: no edit to any frozen Phase-1 artefact. tools/retention_report.py (sha 9a3b2c2a; ANA_SCRIPT_FREEZE_v3
  59c83c86; ALL_METHODS_LOCK 04a3cd19), tools/ana_block_index.py, v4int/reeval.py and v4eval/* stay byte-identical;
  Phase-2 code imports or wraps them. Every new option is OFF by default; with it off every arm spec, RunConfig digest,
  row and trajectory is byte-identical (proved by tests). No Codex. D1 was run by Ops (result in DL 407) and is not
  repeated.
- Scope split (Fable E-S4): **Part A** = the critical path to the sitting: P2-00…P2-07, P2-08a, P2-12, P2-13 (C0, C1, C2
  if O3 = yes, WA–WC, W* ×2, FT_P2_WSTAR ×2, the DP chain; Q1–Q11). **Part B** = in parallel and never delaying the
  sitting: P2-08b, P2-09, P2-10, P2-14 (SecAgg, DP-FTRL, LDP, audits, devices). An unfinished Part-B arm is declared not
  run in P2_METHODS_LOCK.

## 1. Base cost units (INFERENCE, anchored on MEASURED wall times)
| unit | estimate | anchor |
|---|---|---|
| CALIB warm-FL run, frozen tables (3,414 rounds, passes 1) | 1.5–2.5 GPU-h; 5–7 h wall with 8 runs sharing 2 A40 | S7/S8 screens, 8 concurrent ≈ 5–7 h (DL 292) |
| the same with unfrozen tables (WB) | × 1.5–2 | dense 20.7 M-parameter client state / aggregation |
| MAIN warm-FL run (13,654 rounds) | 6–10 GPU-h; 24–30 h wall shared | FA_WARM_FROZEN s2026 ≈ 27 h wall, shared (DL 292 → 346) |
| MAIN DP Poisson run (T 384, m 1,024, passes 2) | 6–10 GPU-h | about the same client-row passes as a MAIN sweep run |
| per-user local FT on an FL model (P5-type) | 3–6 GPU-h | FT_FA_WARM_FROZEN ≈ 4.5 / 9.7 h wall shared (DL 346–362) |
| central D64 run, 6 EFE on MAIN | 1–1.5 GPU-h | C_SMALL 0.5 EFE per 4.5 min (DL 144) |
| central D64 on FULL_TRAIN (C2) | 12–16 GPU-h incl. a possible continuation | ≈ 10× C_SMALL's decisions |
| one checkpoint scored on one role | ≈ 0.2 GPU-h | reeval history |

## 2. Dependency order (T0a)
```
Fable delta re-review of contracts/PREG2.md (touch point 1) → DL row: PREG2 REGISTERED; P2_CONTRASTS_v1 final after O3
  ├─ Part A code (parallel): P2-01 · P2-02 · P2-05 · P2-06 · P2-07 · P2-08a · P2-12 · P2-13 → reviews → code_p2r1
  │     ├─ RUN-1: anchor PRE_CALIB + WA (20) + C1 CALIB screen (8) + C2 (if O3 = yes)
  │     │     └─ select A* (+ ≤ 3 edge runs) → futility check (§5.8, 0.001; caveat D-j)
  │     │           └─ RUN-2: WB (3) + WC (5)      [P2-03, P2-04 written during RUN-1]
  │     │                 └─ select W* → RUN-3 MAIN: W* ×2, C1 ×4, S_CAL(W*), FA1024 ×2, DP8 ×2, DP1
  │     │                       └─ FT_P2_WSTAR ×2 (needs W* best.pt)
  │     └─ Part B code (parallel, never blocks): P2-08b · P2-09 · P2-10 · P2-14
  │           └─ RUN-4: DPFTRL8, LDP ×2 (CALIB), SECAGG equivalence, canary audits ×3, upload capture, E1 MIA, devices
  ├─ P2-11 TEST2 data (only if O2 = yes; parallel; Fable approves the spec = touch point 5)
  └─ P2_METHODS_LOCK (touch point 2) + P2_SCRIPT_FREEZE (touch point 3), both registered and on the server
        └─ register: LOCKED_EVALUATION (Phase-1 lock) + LOCKED_EVALUATION (P2 lock sha)
              └─ THE SINGLE SITTING (S3 order): Phase-1 scoring (H plan as locked) → Phase-2 scoring → Phase-1 analysis
                 → Phase-2 analysis → both RETENTION_RESULT files exist → DL row → then fetch / open
                    └─ TEST2 replication scoring (same sitting if ready, else later under its own register)
```
There is no second RETRO_TEST sitting (S4). An arm not sealed at lock time is declared not run and can be scored only on
TEST2.

## 3. Work packets

### P2-00 Governance, locks, freeze (Part A)
- Do: Ops records Fable's delta re-review and registers contracts/PREG2.md (read-only, sha in a DL row).
  company/gates/P2_CONTRASTS_v1.json (skeleton written 2026-10-03; draft true; self-sha 18fa8b67…) goes final after O3
  (Q8 kept or removed) with its self-sha in a DL row.
- Lock tool tools/p2_lock.py (new):
  - `write_methods_lock` → company/gates/P2_METHODS_LOCK.json, O_EXCL. Contents (S1): PREG2 sha; P2_CONTRASTS_v1 self-sha;
    code-tree digest of code_p2r1; selection-file shas (P2_WARM_SELECTION_A / _FINAL, P2_C1_SELECTION, S_RECORD_P2);
    arms → checkpoint sha / rows sha / rule / seed; the declared-not-run list (06D §8 wording, incl. dropped and
    unfinished Part-B arms); bootstrap seed 20261004; the hash binding of the Phase-1 rows reused by Q1 / Q2 / Q4 / Q5 /
    Q7 / Q9 (PRE, FA_WARM_FROZEN, FT_PRE, C_FULL: S5). PRE / FA_WARM_FROZEN TEST rows are taken from the Phase-1 plan's
    output once and are never scored twice.
  - `write_script_freeze` → company/gates/P2_SCRIPT_FREEZE.json, O_EXCL (S2): shas of tools/p2_report.py and
    tools/p2_block_index.py, plus the retention_report sha they import (must equal 9a3b2c2a).
  - TEST access: tools/test_exposure_register.py **unchanged**. Two LOCKED_EVALUATION entries: one bound to
    ALL_METHODS_LOCK 04a3cd19, one whose all_methods_lock_sha256 field carries P2_METHODS_LOCK's sha. Both are appended
    before the first TEST reeval of any arm. TEST2: tools/test2_exposure_register.py (a copy with its own REG path and
    split enum TEST2_DEC2019).
- Tests: tests_ana/test_p2_lock.py (O_EXCL; self-sha; refuses when a bound code / selection / checkpoint sha differs;
  refuses a lock that lists an arm without a sealed checkpoint outside declared-not-run; the freeze refuses a
  retention_report sha ≠ 9a3b2c2a).
- Review: **Sonnet** (lock tool, Fable §9.4); Fable reviews the lock and the freeze contents (touch points 2–3). Compute: none.

### P2-01 D1-ext instrumentation and the no-op check (Part A)
- Code: fl/update_log.py (new: `group_of`, `round_update_record`); fl/server.py optional `Server.observer` (default None)
  called inside `Server.apply` with (θ_r, new_state, θ_new), with no tensor modified; fl/runtime.py `FLRun._step_rounds`
  puts the observer record under `update_norms` (monitoring only; not in RunConfig); v4int/launch_fl.py
  `update_norm_applies`, `append_update_norm_log`, `read_update_norm_log`, `trim_update_norm_log`,
  `update_norm_summary` (copies of the TRAIN_LOSS.jsonl functions at L1338–1388), called from `execute`.
  tools/p2_noop_check.py (criterion (a) on the trained subset; imports Ops' D1 script if it exists on the server,
  TO-VERIFY its path); v4int/p2_probe.py (criterion (b), labels unused).
- **Tests (Fable E-S3, registered):** tests_fl/test_p2_update_log.py: (1) an 8-round synthetic FLRun with the observer on
  vs off: θ trajectory, comm ledger and every checkpoint digest (latest / best / endpoint) bit-identical; (2) norms equal
  a hand computation; (3) a DP run with z > 0 logs post-noise server-side quantities only: no pre-noise aggregate norm,
  no per-client value (06L_E E1), asserted on the log's key set. tests_int/test_p2_noop.py (NO_OP / MOVED / DRIFT on
  synthetic checkpoints).
- Review: Opus worker, **Sonnet** review. Compute: CPU + ≈ 2 GPU-h probes.

### P2-02 Phase-2 recipe plumbing (WA grid, η_s = 0, client LR, item-LR multiplier) (Part A)
- Code: v4int/launch_fl.py arm family `P2_W` (method FA; PF for WC5), `--p2-recipe <json>` (smoke / CALIBRATION),
  `--p2-selection contracts/fl/P2_WARM_SELECTION_FINAL.json` for MAIN (sha-bound). New `p2_recipe_value`,
  `p2_server_opt`, `check_p2_spec`, `bind_p2_selection`, `p2_run_id`. A P2 branch in `arm_spec`, `resolve_spec`,
  `default_run_id`, `warm_preflight`. `check_init_identity` is unchanged (θ_pre only). The Phase-1 grids stay as they
  are. fl/client.py `LocalSolver.item_lr_mult` (None default), a third param group with `lr_mult`, and `_set_lr`
  honouring it. **Digest guard:** fl/runtime.py `RunConfig.digest` drops `solver.item_lr_mult` when None. The P2 client
  LR grid {0.001, 0.003, 0.01, 0.03, 0.1} is admitted for P2 arms only.
- Tests: tests_fl/test_p2_recipe.py (η_s = 0 → θ bit-identical over 3 rounds for FedAvg / FedAvgM / FedAdam; FedAvg 0.3
  exact; the item-LR multiplier; Phase-1 digests of FA, FA_WARM_FROZEN, DP_WARM_FROZEN and S_CAL equal committed
  references). tests_int/test_p2_launch.py (dry runs of every WA cell; MAIN refuses without the selection; Phase-1 arms
  refuse every P2 flag; **P2-T3** incumbent bit-identity vs S8Tc's first 64 rounds, else re-run).
- Review: Opus worker, **Opus** review. Compute: CPU; P2-T3 ≈ 0.1 GPU-h.

### P2-03 Delta-compressed item uploads, WB3 (Part A)
- Code: fl/sparse_rows.py (`encode`, `decode`, `payload_bytes_sparse`; `SparseChannel(VirtualChannel)`,
  `install_sparse`, modelled on fl/quant.py Q8Channel / install_q8). fl/aggregate.py unchanged.
- Tests: tests_fl/test_p2_sparse.py (round trip; bytes formula; untouched rows = θ; k = K equals dense).
- Review: Opus worker, **Sonnet** review. Compute: CPU.

### P2-04 Parameter-efficient variants, WC (Part A)
- Code: fl/peft.py (`LoRALinear`, `apply_lora` on qkv / out / ff1 / ff2 of both CausalBlocks, `merged_state`,
  `freeze_all_but`, written beside the unedited fl.runtime.freeze_item_tables). Checkpoints carry the merged
  base-architecture θ plus `p2_lora` for resume. TO-CHECK: whether CausalBlock reads `self.qkv.weight` directly. WC5 =
  the existing PF path with the P2 recipe.
- Tests: tests_fl/test_p2_peft.py (B = 0 identity; merged = wrapped forward ≤ 1e-6; only LoRA keys shared; bytes; resume).
- Review: Opus worker, **Opus** review. Compute: CPU + ≈ 0.1 GPU-h smoke.

### P2-05 Controls C1 / C2 (Part A)
- Code: v4int/central_ft.py (shim, pattern of pretrain_small.py: θ_pre start sha-checked; `--p2-freeze-item-tables`;
  roles MAIN / CALIBRATION; MAIN binds contracts/fl/P2_C1_SELECTION.json; labels P2_C1_FT_CENTRAL_FULL / _FROZEN,
  row_status P2_CONTROL; no continuation; TO-CHECK the trainer's initial-state hook). v4int/c_full_small.py
  (parameterised copy of c_full.py with SMALL_SASREC_D64_B2, run id prefix P2_C2_CFULL_SMALL_). **C2 is built and run
  only if owner O3 = yes**; otherwise only the shim's tests are kept.
- Tests: tests_int/test_p2_central_ft.py, tests_int/test_p2_c_full_small.py (as before; mark 0 is never a PB candidate).
- Review: Opus worker, **Sonnet** review. Compute: C1 screen 2; C1 MAIN 5; C2 14 per seed. **Decision: owner O3**
  (Fable recommends YES).

### P2-06 Selection tool (Part A)
- Code: v4int/p2_select.py (reuses `v4int.lr_screen.read_run` via an arms parameter; lr_screen unchanged).
  `p2_candidates`, `p2_order_key`, `p2_decide` with the **CALIB tie band 0.001** (Fable E-O4a), the anchor flag
  NO_MEASURABLE_CALIB_GAIN at **0.001**, `edge_extension` (≤ 3 runs, the registered step table), and `futility` at
  **0.001** (E-O4b), which writes FUTILITY_SKIP_WB_WC together with caveat D-j. CLI `select --stage WA|FINAL|C1` →
  write-once P2_WARM_SELECTION_A.json, P2_WARM_SELECTION_FINAL.json, P2_C1_SELECTION.json. Refuses MAIN / TEST
  manifests and missing runs.
- Tests: tests_int/test_p2_select.py (tie order at 0.001; edges and the single extension; EDGE_UNRESOLVED; futility at
  0.001 with the D-j text; non-finite exclusion; write-once; refusal of a MAIN / TEST manifest).
- Review: Opus worker, **Opus** review. Compute: CPU.

### P2-07 PRE anchor on CALIB (Part A)
- Code: v4int/reeval_p2.py (shim around the unchanged reeval.py): role CALIBRATION for the PRE checkpoint only (reuses
  `check_pre_init_record`), label P2_ANCHOR_PRE_CALIB.
- Tests: tests_int/test_p2_reeval_anchor.py. Review: **Sonnet**. Compute: 0.2 GPU-h.

### P2-08a DP-FedAvg chain for W* (Part A) — Fable E-O4d
- Code: launch_fl P2 equivalents of the FL-16 chain: `p2_s_cal_spec`, S_RECORD_P2.json (format
  PPSI_V4_DP_S_RECORD_P2_1, checks = `check_s_content`), arms P2_WSTAR_FA1024 / _DP8 / _DP1 with **T = 384, the
  registered z record, local passes 2 / batch 16 for FA1024 and DP alike**, and W*'s server / client recipe. No new
  accountant.
- Tests: tests_int/test_p2_dp_launch.py (refuses a z or T other than the registered one; refuses passes ≠ 2 / batch ≠ 16;
  S-record binding refuses a mismatch; FA1024 and DP8 specs differ only in z).
- Review: Opus worker, Opus review. Compute: S_CAL 0.5 + FA1024 16 + DP8 16 + DP1 8 = 40.5 GPU-h.

### P2-08b DP-FTRL and LDP (Part B)
- Code: fl/dp_ftrl.py (`TreeNoise`, `finalize_dp_ftrl`; `RunConfig.dp_ftrl` dropped from the digest when None);
  fl/accountant_ftrl.py (tree-aggregation zCDP, one participation per sweep; cross-checked with `dp_accounting` if
  installable, TO-VERIFY, else brute force); fl/ldp.py (`ldp_upload`, per-upload ε).
- Tests: tests_fl/test_p2_dpftrl.py, tests_fl/test_p2_ldp.py.
- Review: Opus worker, Opus review, **Fable concurrence on the accountant (touch point 4)**. Compute: DPFTRL8 8 (+2 CALIB
  check); LDP 2 × 2 → 14 GPU-h. DPFTRL8 s2027 only within the cap (first in the drop order).

### P2-09 Secure-aggregation simulation (Part B)
- Code: fl/secagg.py (`FixedPoint`, `pair_seed`, `SecAggSim`, Shamir over GF(2^61 − 1), `cost_model`;
  `SecAggChannel`). Tests: tests_fl/test_p2_secagg.py (bit-exact sum at p_drop 0 / 0.1 / 0.3; failure below t; cost
  formula; trajectory = plain fixed-point aggregation).
- Review: Opus worker, **Opus** review. Compute: CPU 2–4 h; CALIB equivalence ≈ 2 GPU-h; MAIN s2026 ≈ 8 GPU-h (drop
  order tier 1).

### P2-10 Privacy audit tools (Part B)
- Code: tools/p2_mia_audit.py (members 10,000 MAIN / non-members 10,000 UNEW_HOLDOUT, TRAIN rows only; **the PRE-null
  validity check runs first and must pass before any ε_LB is computed or written**, Fable E-S3; AUC, TPR@{0.1 %, 1 %},
  A / B threshold split, Clopper–Pearson ε_LB); v4int/canary_provider.py (`CanaryProvider`); launch_fl
  `--audit-canaries` and `--audit-capture-uploads` (CALIBRATION + P2_AUDIT only; MAIN refuses);
  tools/p2_canary_audit.py (exposure; one-run ε_LB); tools/p2_update_leakage.py.
- Tests: tests_ana/test_p2_audit.py (null → AUC ≈ 0.5; a failing PRE null blocks ε_LB output; Clopper–Pearson table;
  MAIN refuses the audit flags).
- Review: Opus worker, **Opus** review. Compute: ≈ 10.5 GPU-h.

### P2-11 TEST2 data (optional replication; only if owner O2 = yes)
- Ops TO-VERIFY list first (PREG2 §6.2 items 1–8, COUNTS_ONLY). Code: code/v4data11/ (`d11_ingest.py`,
  `d11_manifests.py`, `d11_registry.py`, `d11_gate.check_test2_gate`); v4int/test2_store.py (wraps retro_store);
  tools/test2_exposure_register.py. Tests: code/tests_data11/ (boundaries, causality, OOV, the gate refuses before
  P2_METHODS_LOCK).
- Review: Opus worker, **Opus** review; **Fable approves the manifest spec (touch point 5)**. Compute: CPU 2–6 h; 2–4
  days of engineering. Never on the critical path: if it is not ready at the sitting, TEST2 is scored later under its
  own register (label TEST2_NOT_AVAILABLE until then).

### P2-12 Phase-2 analysis (Part A)
- Code: tools/p2_report.py **imports** tools/retention_report.py (no edit): `Engine`, `Book`, `ratio`, `delta`,
  `holm_band`, `_seed_stats`, `k14_equivalence`, `load_index` / `load_block`, `parse_contrast` (with its own arm table
  extended by the P2 labels). tools/p2_block_index.py imports ana_block_index. Adds: the P2_CONTRASTS_v1 loader (same
  field set as ANA_CONTRASTS_v4), the §9.4 reading (SUCCESS / EQUIVALENT ±0.0005 TOST on the 90 % CI / FAILURE /
  INCONCLUSIVE) and the interpretation grid, the labels of §6.4 (CONFIRMATORY_P2 (PREREGISTERED_FOLLOWUP) with D-a
  verbatim; TERTIARY_P2; REPLICATED_FRESH / NOT_REPLICATED / TEST2_NOT_AVAILABLE), D-h printed beside Q1, Q10, the
  declared-not-run list, and the Part-B DESCRIPTIVE sections. TEST mode requires P2_METHODS_LOCK + P2_SCRIPT_FREEZE + the
  P2 LOCKED_EVALUATION entry, and that the reused Phase-1 rows (PRE, FA_WARM_FROZEN, FT_PRE, C_FULL) match the hashes
  bound in the P2 lock (S5). Output dir p2_out_test_*; S3: nothing is opened until both RETENTION_RESULT files exist.
- Tests: tests_ana/test_p2_report.py (Holm over Q2–Q7(+Q8); TOST ±0.0005; label per tier; refusal without the lock /
  freeze / register entry; refusal when a reused Phase-1 row hash differs; retention_report sha unchanged after import).
- Review: Opus worker, **Opus** review; **Fable checks P2_SCRIPT_FREEZE (touch point 3)**. Compute: CPU.

### P2-13 Scoring P2 checkpoints (Part A)
- Code: v4int/reeval_p2.py admits the P2 arm labels (FL payloads with LAUNCH / ARM_SPEC / p2_recipe checks mirroring
  `source_fl` / `check_warm_line`; P2_C1 / P2_C2 central checkpoints; merged-LoRA payloads) for MAIN, MAIN_TEST (through
  the unchanged DATA-09 gate) and TEST2 roles. It **refuses PRE / FA_WARM_FROZEN / FT_PRE / C_FULL on TEST roles**
  (S5: those rows come once from the Phase-1 plan).
- Tests: tests_int/test_p2_reeval.py (dry runs per source kind; TEST roles refused while the gate is closed or the P2
  register entry is missing; the S5 refusal).
- Review: Opus worker, **Sonnet** review. Compute: ≈ 8 GPU-h of scoring.

### P2-14 Real-device measurements (Part B)
- Code: code/v4dev/browser/bench.js + index.html (cold start, memory, 3-session bookkeeping, device identity,
  DEVICE_BENCH_P2.json); code/v4dev/browser/serve_coi.py (COOP / COEP); code/v4dev/android/ (README, termux_bench.py,
  TO-VERIFY); tools/p2_android_energy.sh. Tests: tests_dev/test_p2_device_page.py.
- Review: **Sonnet**. Compute: ≈ 0.5 GPU-h; owner time 1–2 days. **Decision: owner O8 (devices)**; thresholds DESCRIPTIVE.

### P2-15 Records
DL row for registration, every run wave, each selection, the lock, the freeze and the sitting. Thesis reference updated
after registration and after each selection (Fable §9.4). RESUME_HERE top block.

## 4. Compute summary (INFERENCE) and the cap
| item | part | GPU-h |
|---|---|---|
| WA core + WA-ext + edges (CALIB) | A | 46 |
| WB + WC (CALIB) | A | 20.5 |
| anchor, P2-T3, probes | A | 2.5 |
| C1 screen + MAIN | A | 7 |
| C2 (if O3 = yes) | A | 14 |
| MAIN W* ×2 | A | 16 |
| FT_P2_WSTAR ×2 | A | 9 |
| DP chain (S_CAL, FA1024 ×2, DP8 ×2, DP1) | A | 40.5 |
| scoring (MAIN / RETRO_TEST / TEST2) | A | 8 |
| DP-FTRL + LDP | B | 14 |
| SecAgg (CALIB; MAIN optional) | B | 2 (+8) |
| audits | B | 10.5 |
| device export / parity | B | 0.5 |
| **total** | | **≈ 190 (≈ 200 with the MAIN SecAgg run)** — cap 250 GPU-h |

Drop order if the projection exceeds 250 (Fable E-O4c): (1) tertiary-only arms DP1 → SECAGG on MAIN → DPFTRL8 s2027;
(2) exploration cells WC3/WC4 → WB2 → WC1/WC2 → WB3; (3) C2 last. Everything dropped is declared not run in
P2_METHODS_LOCK. The primary path is never dropped.

GPU precedence (S7): Phase-2 lanes never share a GPU with H steps. Phase 2 has priority over the ML-1M GPU move (Fable
§8): §8's timing gate includes Phase-2 demand (≈ 190 GPU-h on 2 A40 ≈ 4–6 days); if Phase 2 takes both A40s, ML-1M stays
on CPU (§8 option A). Ops writes one DL row before any BM-03 effort is spent.

Wall time (INFERENCE): Part-A code 3–4 days incl. reviews; RUN-1 ≈ 1.5 days; RUN-2 ≈ 1 day; RUN-3 ≈ 2–3 days; FT ≈ 1 day;
lock + freeze + sitting ≈ 1 day. **The Phase-1 TEST result therefore waits until ≈ 10-17…10-21** (Fable §9.1; the owner
prefers this).

## 5. Fable's five touch points (exactly these; Fable §9.4)
1. Delta re-review of contracts/PREG2.md before registration (text only, ≤ 30 min) — REQUIRED, next.
2. P2_METHODS_LOCK before the sitting.
3. P2_SCRIPT_FREEZE.
4. DP-FTRL accountant concurrence (P2-08b).
5. The TEST2 manifest spec, if O2 = yes (P2-11).
Everything else is Ops plus the listed reviewers.

## 6. Decisions
| id | decision | status |
|---|---|---|
| O1 | confirmation route | T0a (Fable) |
| O9 | second study on the RETRO_TEST manifests | YES under S1–S7 (Fable) |
| O4 | grids / ties / edges / futility / cap | accepted with ties and futility 0.001, TEST margin ±0.0005, cap 250, drop order by tier (Fable) |
| O7 | DP T / z | T = 384, registered z, passes 2 / batch 16 (Fable) |
| O2 | December 2019 data version (TEST2 replication) | PENDING — owner |
| O3 | C2 | PENDING — owner (Fable: YES) |
| O5 | second θ_pre seed | NO (Fable) |
| O6 | second dataset | NO (Fable) |
| O8 | devices | PENDING — owner; thresholds DESCRIPTIVE |

## 7. Risks and controls
- Any TEST value seen before both locks exist breaks T0a. Control: S2 / S3. p2_report and reeval_p2 refuse without the
  P2 lock, the freeze and the register entry. Ops writes the DL row before any fetch.
- No second sitting (S4): an arm late for the lock is declared not run (TEST2 only). Part B never blocks.
- Digest drift in Phase-1 code: byte-identity tests in every packet; deployment as a new tree code_p2r1.
- Frozen-tool edits: imports / wraps only; tests assert the shas are unchanged.
- Selection leakage: CALIB only; the tool refuses MAIN / TEST manifests.
- A small true effect (ceiling ≈ 0.001; MDE ≈ 0.00047): reading D-h registered; C1 / C2 make a null interpretable. No
  positive result is promised.
- GPU contention with ML-1M: S7 precedence and the DL row.
