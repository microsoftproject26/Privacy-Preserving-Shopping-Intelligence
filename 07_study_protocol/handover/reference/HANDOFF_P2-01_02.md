# P2-01 + P2-02 — D1-ext instrumentation, the no-op check, and the Phase-2 recipe plumbing (P2_W)

Worker: Opus (one worker, shared files). Date 2026-10-03. Local code + local unit tests only: no server, no git, no Codex,
no run beyond unit tests, no TEST data, no evaluation value read. contracts/ not edited.
Spec: PHASE2_IMPLEMENTATION_PLAN §P2-01 / §P2-02; contracts/PREG2.md (REGISTERED, sha be0561ab…, DL 411; §3.2, §3.3,
§5.1–§5.7) which supersedes the draft where they differ; Fable gate FABLE_GATE_ENDGAME_SCRIPTS_20261003 §9 (E-S3
bit-identity on / off, post-noise-only logging under DP, CALIB tie band / futility 0.001).
Status labels: every statement below is MEASURED on this Windows CPU host unless marked PLAN / INFERENCE / PENDING.

## 0. Result in one paragraph
Done. `P2_W` is a new launch_fl arm family driven by a P2 recipe (the exact `v4int.p2_select` recipe layout and label;
P2-06 now reads P2 runs with its own `read_p2_run`, R1–R12, verified by REV-P2-01_02 §1 (5)). Q8 now dispatches on the codec name. D1-ext writes UPDATE_NORM_LOG.jsonl for P2 runs only, through an observer hook on
`fl.server.Server` that is `None` by default. The item-LR multiplier is a `LocalSolver` **subclass** (`ItemLRSolver`), not
a field (deviation from the plan, see §3.1: a field would have changed every LO recipe digest). Firewall: 56 Phase-1
(arm spec, RunConfig) pairs, 13 explicit RunConfig pins and 21 production CALIB arm-spec digests are byte-identical
before / after; no Phase-1 grid, list, refusal or smoke limit changed; the Phase-1 test suites give the same results
before and after (§5). The P2-T3 incumbent cfg is S8Tc's cfg exactly; the server check is tools/p2_t3_check.py (§7), whose receipt
v4int.p2_select.check_p2t3_receipt accepts (REV-P2-01_02 R1; see §9 "REV fixes").

## 1. Archive (before any edit)
`code/_archive_p2_wave_a/` holds byte copies of the four edited files + `SHA256SUMS` (verified = the pre-edit tree hash):
```
c9fdc8c083e834f875b5d9c48d1144da0740137374fc5219c18a623ee8927032  v4int/launch_fl.py
8c94ee530ba699fe0b3f2490cc4e755b6f08846e4387814fc4486b5888918958  fl/server.py
e627e943d93abbdda68125ef4f3ebd72580f85bf9f3d5e2dfd2e02d8b90f436b  fl/runtime.py
8a41e6d6bfb385df17a2b24685b0bc255534b7366c2c42c246124fcb776f81af  fl/client.py
```

## 2. Files and diff summary
| file | sha256 (new) | change |
|---|---|---|
| code/fl/client.py | f869ac90… | + `ITEM_LR_KEYS`, `ItemLRSolver(LocalSolver)` (field `item_lr_mult`, validated finite > 0), `solver_from_dict`, `item_lr_groups`, `_make_optimizer_item_lr`; `make_optimizer` branches to it only when the solver carries a multiplier; `_set_lr` keeps a group's `lr_mult` ratio (`lr` unchanged for groups without the tag). LocalSolver itself unchanged. |
| code/fl/server.py | ccbcc5a1… | `Server.observer = None`; `Server.apply` calls `observer(theta_r, aggregate, theta_{r+1})` at its end (fresh broadcast clones; the aggregate as given = post-noise for DP). Observer None = the old code path. |
| code/fl/runtime.py | 412d3829… | `RunConfig.from_dict` uses `solver_from_dict`; `RunConfig.digest` drops a `None` `solver.item_lr_mult` (guard); `step()` / `_step_rounds()` add `update_norms` only when `server.observer` is set (`_update_norms`: observer record or `{"server_step": False}`; `n_clients` omitted when z > 0). |
| code/v4int/launch_fl.py | f4e56b80… (was c6192343… before §9) | P2 constants (outside ALL_ARMS), `is_p2_arm`, `--p2-recipe` / `--p2-selection`, P2 branches in `arm_spec`, `execute`, `preflight` (smoke ≤ 64 for P2 only), `fl05_preflight`, `resolve_spec`, `default_run_id`, `run`, `dry_run`; FA_Q8 installed only for the Q8 codec NAME (`is_q8_codec`, REV-P2-03_04 (a)); new functions listed in §3.2 (+ `p2_run_labels`, `is_q8_codec`); UPDATE_NORM_LOG functions (`update_norm_applies`, `update_norm_round_record`, `append_ / read_ / trim_update_norm_log`, `update_norm_summary`); docstring section P2-01 / P2-02. |
| code/fl/update_log.py (new) | 42449044… | `group_of`, `round_update_record`, `UpdateNormObserver`, `noop_round_record`; PREG2 §3.2 `param_noop_criterion` (a), `functional_rows` / `functional_noop_from_rows` / `functional_noop_criterion` (b), `classify_noop`; registered thresholds (1 %, 4,096, 0.95, top-20, 4,096 rows, seed 20261004). |
| tools/p2_noop_check.py (new) | 4074e4e8… | criterion (a) from checkpoints (θ_pre = PRE endpoint_6.0 `model`, run best.pt / endpoint_6.0.pt `theta`); T from RUN_SUMMARY `update_norm_log.trained_keys` (else float keys − ARM_SPEC frozen tables); frozen tables checked bit-identical; with `--probe` → NO_OP / MOVED / DRIFT / INCOMPLETE; O_EXCL output. |
| code/v4int/p2_probe.py (new) | c25bd351… | criterion (b) on the 4,096 seeded CALIB_TRAIN rows (StoreTrainProvider of the registered CALIBRATION train manifest; labels stripped before scoring; eval mode; two SMALL-recipe models θ_pre / θ); chunked per-row stats; self-sha256, O_EXCL. Server-only path NOT run here (PENDING); its pure parts are tested. |
| code/tests_fl/test_p2_update_log.py (new) | 111e99f6… | §4 |
| code/tests_fl/test_p2_recipe.py (new) | fb9ba7d7… | §4 |
| code/tests_int/test_p2_noop.py (new) | d183997b… | §4 |
| code/tests_int/test_p2_launch.py (new) | 9a68eaf1… (was 11b246f3…) | §4 |
| tools/p2_t3_check.py (new, §9) | 921320e9… | the P2-T3 comparison + O_EXCL receipt in p2_select's format |
| code/tests_int/test_p2_t3_check.py (new, §9) | fc626336… | tiny S8Tc + P2 incumbent smoke runs → BIT_IDENTICAL accepted by p2_select; FAIL cases; refusals |

## 3. Design notes
### 3.1 Item-LR multiplier: a subclass, not a LocalSolver field (deviation from the plan text)
The plan says `LocalSolver.item_lr_mult: Optional[float] = None` + a RunConfig digest guard. MEASURED risk: `fl.local_only.LORecipe.digest()`
hashes `asdict(self)` including its `solver: LocalSolver`, so a new LocalSolver field (even `None`) changes every Phase-1
LO recipe digest — a file this packet does not own. `ItemLRSolver(LocalSolver)` keeps LocalSolver's field set exactly
(test pins the list), so LO / FT / RunConfig digests of every existing config are untouched by construction. The RunConfig
guard is still there (drops a `None` key; test + NC). `dataclasses.replace(solver, lr=…)` keeps the subclass (tested).
Groups: the item tables (`ITEM_LR_KEYS` = `FROZEN_ITEM_KEYS`) move out of each `param_groups` group into a twin group with
the same options (so AdamW's ndim decay rule is kept) at `lr × m`, tag `lr_mult`. A multiplier with no trainable item
table (frozen tables, LoRA) is refused, never a silent no-op; LoRA cells never carry a multiplier (check_p2_recipe), so
the P2-03/04 note "tolerate absent item keys" is satisfied by construction.
### 3.2 The P2_W family (launch_fl)
- One arm `P2_W` (P2-06 reads `P2_ARMS = ("P2_W",)`); WC5 = `personal_head: true` → method PF (rows method_code PF;
  `METHOD_CODE["P2_W"] = "FA"` for the ARCHITECTURE_LOCK check). Not in ALL_ARMS (test_fl16 pins ALL_ARMS).
- Recipe = v4int.p2_select layout: `server_opt, server_lr, client_lr, items, item_lr_mult, upload, lora_rank,
  lora_lr_factor, personal_head`; label = `p2_select.p2_label` (tested equal on all 21 WA cells; `p2_select.wa_cells()`
  = `launch_fl.p2_wa_recipes()`). Spec stores exactly these keys in `p2_recipe` (+ `p2_label`), as p2_run_adapter expects.
- Admitted values (P2 only): server LR per point = WA core ∪ WA-ext ∪ incumbent ∪ the §5.7 steps (FedAvg {0.1, 0.3, 1, 2},
  FedAvgM {0.01, 0.03, 0.1, 0.3}, FedAdam {3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3}); client LR {0.001, 0.003, 0.01, 0.03,
  0.1}; item_lr_mult {0.1, 0.01, 0.001}. Cell membership (which pairs are run) is P2-06's registered cell sets.
- `p2_server_opt`: FedAvg η_s = 1 → no server_opt (plain FedAvg); η_s ≠ 1 → FedAvgM β 0 at LR η_s (the warm_server_opt
  path); FedAvgM β 0.9; FedAdam β 0.9 / 0.99, τ 1e-3, v0 τ².
- `p2_arm_spec`: PREG2 §5.1 fixed parts (`P2_FIXED`): SGD-m 0.9 fresh per visit, clip 1.0, wd 0, 1 pass, batch 16,
  FL_CONST, 64 clients, sweep, 10 % drop-out, 6.0 EFE → CALIB 3,414 / MAIN 13,654 rounds (`exposure_budget` recorded);
  θ_pre via `init_from` + the unchanged `check_init_identity`; frozen tables unless `items: unfrozen`. The cfg has the
  Phase-1 layout, so the incumbent recipe's cfg **is** S8Tc's cfg (test).
- `check_p2_spec`: a P2 spec must equal `p2_arm_spec` of its own recipe / seed / role / θ_pre (hand edits refused).
- `p2_preflight`: refuses every Phase-1 option (`P2_REFUSED_FLAGS`, 17 flags) and `--freeze-item-tables`; needs
  `--init-from`; CALIBRATION at seed 2026; exactly one of `--p2-recipe` (smoke / CALIBRATION / dry run) and
  `--p2-selection` (MAIN only); a non-smoke MAIN run without the registered selection is refused.
- `bind_p2_selection` (MAIN): path must resolve to `contracts/fl/P2_WARM_SELECTION_FINAL.json`; sha before / after;
  format `PPSI_V4_P2_WARM_SELECTION_FINAL_1`, self-sha (`_canon`), stage FINAL, `winner == p2_label(winner_recipe)`,
  then `v4int.p2_select.load_p2_selection(path, "FINAL")` re-derives the decision (fails closed if absent), and
  `common.init_from` (when recorded) = this run's θ_pre.
- Run ids: `P2_W_<TAG>_CALIB_s2026` (e.g. `P2_W_FEDADAM_eta0.001_c0.1_CALIB_s2026`), MAIN bound `P2_WSTAR_SMALL_s<seed>`.
- Rows: `study PREG2`, `row_status P2_ARM`, `comparison_status P2_FOLLOWUP` (MAIN), `p2_recipe_label`, `fl_*` recipe
  fields; CALIBRATION rows keep the FL_LR_SCREEN labels (fl_rows applies them last).
- Smoke limit 65 rounds for P2 only (P2-T3: 64 updates + the loss line of θ_64; §9 m2); Phase-1 stays 50 (test).
### 3.3 D1-ext
Observer attached in `execute` after the Server is built and before `FLRun.resume` (θ_pre = the server's verified init
state). Records: per group (`block<i>.qkv|out|ff1|ff2`, `layernorm`, `gates`, `item_tables`, else module name) and total over
T = the manifest shared keys: ‖Δ_agg‖, ‖step‖, ‖θ−θ_pre‖, ‖θ_pre‖, rel; `cos_prev`; `n_clients` (omitted z > 0); rounds
without a step → `{"server_step": false}`. float64 sums; non-finite → None + `finite: false`. `cos_prev` restarts after a
resume (`cos_prev_available: false`; listed in RUN_SUMMARY `cos_prev_unavailable_rounds`). Log trimmed on resume like
TRAIN_LOSS. RUN_SUMMARY `update_norm_log` carries `trained_keys` / `trained_numel` (read by p2_noop_check and by P2-06).
### 3.4 Tie band / futility 0.001
Recorded as `P2_CALIB_TIE_BAND` / `P2_FUTILITY` (Fable §9.2 E-O4a/b) for reference only; the rule is applied by
v4int.p2_select (P2-06), which already uses 0.001. PREG2.md §5.6 also sets NO_MEASURABLE_CALIB_GAIN to 0.001 (Fable to
confirm); p2_select now uses `ANCHOR_MIN_GAIN = 0.001` (Q3 resolved).

## 4. Tests (new) — what each proves
- tests_fl/test_p2_update_log.py (4 + 1 NC): observer ON vs OFF over 8-round FLRuns (FedAvg tied + drop-out, FedAdam
  frozen, FedAvgM untied, DP-FedAvg z 0.8 Poisson): per-round θ digests, ledger, step records and the content digest of
  every checkpoint (latest / best / endpoint) identical (E-S3); norms = hand computation (float64, rel 1e-12) incl. cos;
  DP: no `n_clients`, only server-side fields, the logged aggregate is the released θ (step == Δ_agg, θ_{r+1} == agg
  bitwise); no-step rounds; double step refused; group_of. NC: an observer that touches the weights breaks identity.
- tests_fl/test_p2_recipe.py (5 + 1 NC): η_s = 0 → θ bit-identical over 3 rounds for FedAvg (FedAvgM β0 lr 0) / FedAvgM /
  FedAdam (tied and frozen families); FedAvg 0.3 == θ + 0.3 Δ bitwise; item-LR groups / ratio / effect / refusals;
  13 Phase-1 RunConfig digests (FA, FP, PF, FA_Q8, FA_1024, DP8, S_CAL ×2, FA_WARM, FA_WARM_FROZEN MAIN + CALIB,
  DP_WARM_FROZEN; FA / S_CAL default = committed test_fl05 BASELINE 66770288 / 8fb83823) = pre-P2 MEASURED values;
  LocalSolver fields; digest guard. NC: a digest keeping `item_lr_mult: None` changes the pins.
- tests_int/test_p2_noop.py (5 + 1 NC): (a) thresholds incl. "small group may move"; the tool's NO_OP / MOVED(a) /
  MOVED(b) / DRIFT / DRIFT(=) / MOVED(better) / INCOMPLETE / MOVED(no probe) classifications; T from RUN_SUMMARY; frozen-key
  refusal; changed frozen table reported; CLI O_EXCL + refusals; (b) identical / near / far / shifted scores; probe rows
  pure + sorted + distinct; labels stripped; chunked == full; no RNG drawn. NC: total-only (a).
- tests_int/test_p2_launch.py (10 + 2 NC): firewall (56-pair table sha 0c89ec77…, grids, lists, method codes, 21 production
  digests); `dry_run` of the 21 WA cells + MAIN / WB / WC5; all refusals; MAIN binding (stub verifier; copy / tamper /
  stage / format / θ_pre / verifier refusal / verifier absent); P2-T3 local (cfg == S8Tc cfg, same RunConfig digest; tiny
  run through P2 path == Phase-1 path: θ digest, ledger, evals, TRAIN_LOSS bytes); UPDATE_NORM_LOG complete, P2 only;
  resume (interrupt at 5 → equal digest, log trimmed, cos restart); MAIN rows labels; WB (ItemLRSolver) and WC5 (PF)
  tiny runs DONE_MAIN.

## 5. Suite counts before / after (per-file runs; pytest -p no:cacheprovider, basetemp outside the tree)
Method: per-file runner (each test file in its own pytest process, junit parsed per test case), because this host crashes
the one-process suite (Windows access violation / MemoryError under the parallel workers' load; MEASURED, see below).
BEFORE = a byte copy of the pre-edit code tree (taken before the first edit, run in a sibling folder of `code/` so the
V4-relative paths resolve), restricted to the test files that existed before this packet. AFTER = the edited `code/`
(final launch_fl c6192343…), the same files + this packet's new files.

| suite | before (pre-edit tree) | after (edited tree), same files | case-by-case diff | new files (after) |
|---|---|---|---|---|
| tests_fl (27 files) | 239 passed, 75 xfailed, 9 skipped | 239 passed, 75 xfailed, 9 skipped | **none** | test_p2_update_log 4 + 1 xfail; test_p2_recipe 5 + 1 xfail |
| tests_int (40 files) | 375 passed, 139 xfailed | 375 passed, 139 xfailed | **none** | test_p2_noop 5 + 1 xfail; test_p2_launch 10 + 2 xfail |

- tests_int BEFORE, first pass: 372 passed, 138 xfailed, 2 failed (test_fl10_fedadam_launch::test_engine_determinism_rows_and_resume[fedadam]:
  MemoryError; test_int06_fl_device::test_nc_unselected_lo_cell_is_evidence: FileNotFound after a temp-dir clean-up of
  mine), test_fl05_real_calib crashed (access violation). Re-run of those 3 files on the same pre-edit copy: 2 passed;
  22 passed / 6 xfailed; 9 passed / 3 xfailed — all green, so the table uses the re-run (environmental, not code).
- One-process attempts (for the record, not evidence): tests_fl crashed (access violation, exit 139) in
  test_families_smoke / test_t32 both times; tests_int on C: temp gave 79 failed / 63 errors (C: had 0.1 GB free →
  DiskRefused / MemoryError), then 373 passed / 137 xfailed / 4 MemoryError errors with temp on F:.
- The parallel workers' new tests also pass against the edited tree: tests_fl test_p2_peft 10 + 1 xfail, test_p2_sparse
  16 + 1 xfail; tests_int test_p2_select 22 + 1 xfail, test_p2_central_ft 15 + 2 xfail, test_p2_c_full_small 9 + 2 xfail.
- Digest identity outside pytest: the 56-pair Phase-1 table recomputed after the last edit is byte-identical to the
  pre-edit table (sha 0c89ec77…); the 21 production CALIB arm-spec digests of contracts/fl/FL_TUNING_S2|S3|SELECTION.json
  reproduce (21 / 21) before and after.
- Tree hygiene: tests_fl/receipts (LO_DETERMINISM_1024*.json and T34_POOL_RUNS.json are rewritten by any tests_fl run) were
  restored byte-identically from a pre-run backup; tests_int/receipts unchanged; my temp folders removed. `code/.pytest_cache`
  (created 20:15 local) is not mine: every run of mine used `-p no:cacheprovider`.

Commands (Git Bash; T = a temp dir OUTSIDE the V4 tree):
```bash
cd code/tests_fl  && TMP=$T TEMP=$T python -B -m pytest -q -p no:cacheprovider -rfE --basetemp=$T/bt test_p2_update_log.py test_p2_recipe.py
cd code/tests_int && TMP=$T TEMP=$T python -B -m pytest -q -p no:cacheprovider -rfE --basetemp=$T/bt test_p2_noop.py test_p2_launch.py
# whole suites as measured: one pytest process per test file with --junitxml (scratchpad run_perfile.py)
```

## 6. Hook points for the later packets (one small edit each; NOT implemented)
| packet | where | the edit |
|---|---|---|
| P2-03 WB3 (`delta_rows`) | `P2_UPLOADS` + `check_p2_recipe` (admit `"delta_rows"`, unfrozen only) + `p2_arm_spec` (set `spec["upload_codec"] = {"name": "SPARSE_ROWS_WB3", "k_top": 3170}` so the ARM_SPEC sha carries it) + **HOOK 2 `p2_post_run(run, spec)`** (called after `FLRun(...)` and after `FLRun.resume(...)`, where install_q8 sits) | `sp = fl.sparse_rows.install_sparse(run, k_top=int(spec["upload_codec"]["k_top"]))`; return `{"channel": sp}`; in the round loop `run.channel is sp` + `sp.check_round(out_r["round"], out_r["survived"])`; publish `bytes.per_visit.upload_bytes` = ledger bytes_up / Σ visit_counts (checkpointed; §6c (b)), `sp.summary()` as audit only. Q8 already dispatches on the codec name (`is_q8_codec`, §6c (a)). |
| P2-04 WC1–WC4 (LoRA) | `check_p2_recipe` (admit `lora_rank` {1, 4, 16, 64} + `lora_lr_factor` {1, 10}, frozen only) + `p2_arm_spec` (solver lr = c × factor; `spec["lora"] = {"r", "alpha": r, "targets": ["qkv","out","ff1","ff2"], "seed"}`; **no** `frozen_item_tables`) + **HOOK 1 `p2_pre_server(spec, server_adapter, workers, init_sha256) -> (record, init_sha256)`** (called after the freeze block, before `Server(...)`) + HOOK 2 | HOOK 1: `apply_lora` + `freeze_all_but(server_adapter, lora_keys(...))`, `init_sha256 = lora_init_sha256(server_adapter, init_sha256)`, then each worker `apply_lora` + `freeze_all_but(..., source=server broadcast)`; HOOK 2: `install_peft(run)`; `bytes_record(..., frozen_keys=frozen_keys_of(server_adapter))`. The D1-ext observer then sees T = the LoRA keys automatically. |
| P2-08 DP for W* | `p2_arm_spec` (+ `cfg["dp"]`, Poisson plan m 1,024, T 384, passes 2 / batch 16 by DL 219 / PREG2 §5.5) and `p2_preflight` (admit `--epsilon` / `--clip-record` for the DP variants; currently in `P2_REFUSED_FLAGS`) | runtime already omits `n_clients` and logs post-noise quantities only for z > 0; TRAIN_LOSS / CLIP_LOG gates (`train_loss_applies`, `clip_log_applies`) read `cfg.dp` and work unchanged. |
Per handoffs/P2-03_04.md: new fl/*.py files change `fl_code_sha256`; this packet adds `fl/update_log.py` too. Deploy only
in the new P2 tree (code_p2r1); in-flight Phase-1 runs must keep resuming from their own tree (06E §8).

## 6b. For P2-06 (REV-P2-06 M2): the exact P2_W run format and how to read it
MEASURED from a tiny P2_W CALIB run of the final code:
- `ARM_SPEC.json` = `{arm: "P2_W", run_id, arm_spec_sha256, arm_spec, device, code_sha256}`.
- `arm_spec` keys: `arm, study ("PREG2"), plan {group_size 64, sampling "sweep"}, cfg, max_rounds, lr_mult (1.0),
  lr_peak_base (the SMALL peak), role, upload_codec (null), init_from (θ_pre identity = INIT_IDENTITY_KEYS), p2_recipe
  (exactly P2_RECIPE_KEYS = p2_select.RECIPE_KEYS; floats / None / bool), p2_label, p2_fixed, exposure_budget
  (n_clients, passes 1, rounds)`, plus `frozen_item_tables` iff `items == "frozen"`. There is **no** `knobs` and no
  top-level `server_opt / client_opt / local_passes`.
- `cfg` uses the Phase-1 RunConfig layout: `method` FA (PF iff personal_head), `seed`, `lr_peak_local` = c,
  `solver {lr: c, batch_size 16, passes 1, clip 1.0, weight_decay 0.0, fused null, optimizer "sgd_m"[, item_lr_mult m]}`,
  `n_shards 2, max_attempts 2, endpoint_efe 6.0, exposure_point "end", mu 0.0, pf (null | {lr: SMALL peak, lam 1e-4,
  clip 1.0}), dp null, dropout_p 0.1, endpoint_rounds (3,414 CALIB / 13,654 MAIN), lr_shape "fl_const"[, server_opt]`.
- Digest rule: `arm_spec_sha256 = launch_fl.spec_digest(arm_spec)` = sha256(json.dumps(digest_body(spec), sort_keys=True,
  default=str)). `digest_body` drops `lr_mult` (1.0) and therefore `lr_peak_base`, drops `role` when "MAIN", and drops
  `upload_codec` when null. Validity: `launch_fl.check_p2_spec(arm_spec)` (the spec must equal `p2_arm_spec` of its own
  recipe / seed / role / init_from / lr_peak_base).
- `RUN_SUMMARY.json` keys: `arm, arm_spec, arm_spec_sha256, method, run_id, rounds, done, phase, server_digest,
  store_digest, counters, ledger, evals, best_metric, best_half_efe, lr_table_digest, plan, server_opt (if any),
  participation, frozen_item_tables (the freeze record, if frozen), train_loss_log, update_norm_log {file, rounds_logged,
  rounds_expected, complete, missing_rounds, server_steps, cos_prev_unavailable_rounds, sha256, trained_keys,
  trained_numel, last_total_T, authority}, p2 {label, recipe, pre_server, post_run, authority}, wall_s_this_leg,
  bytes {per_visit {upload_bytes, shared_numel, download_bytes, …, frozen_* if frozen}, per_client, fleet, …}, lr_table,
  status, checkpoints`.
- Rows (RESULT_ROWS): fl_rows' own fields (method "P2_W", fl_lr_mult 1.0, arm_spec_sha256, …) + `launch_fl.p2_run_labels(spec)`
  (fl_init*, fl_frozen_item_tables, study, row_status P2_ARM, comparison_status P2_FOLLOWUP, p2_recipe_label, method_code,
  fl_server_opt, fl_server_lr, fl_client_opt, fl_client_lr, fl_lr_shape, fl_local_passes, p2_items[, p2_item_lr_mult]);
  then, for role CALIBRATION, the FL_LR_SCREEN relabel (role / manifest_role / run_role / population_scope CALIBRATION;
  run_label / row_status / comparison_status FL_LR_SCREEN). This is exactly
  `fl_rows(rows, ..., role=spec["role"], tuning_labels=p2_run_labels(spec))`.
- Reader (resolved: p2_select 7a05afcd has its own `read_p2_run`, R1–R12, checked field by field in REV-P2-01_02 §1 (5);
  the text below is the design it follows): p2_select's own `read_p2_run` that keeps
  lr_screen.read_run's structural checks, uses `F.check_p2_spec` instead of `F.spec_tuning` / `F.check_warm_spec`, and
  compares row labels with `F.p2_run_labels(spec)` + the CALIB relabel. A P2 branch inside `spec_tuning` would have to admit
  P2 values into the Phase-1 tuning grid, which the firewall forbids, so I did not add one.

## 6c. Wiring notes from the reviews (for P2-03, to apply with the hook)
- (a) REV-P2-03_04, **dispatch on the codec NAME:** done in this packet. `execute` installs FA_Q8 only when
  `upload_codec["name"] == "Q8_SR_PER_TENSOR"` (`is_q8_codec`), passes the codec to `bytes_record` only then, and refuses a
  Phase-1 arm with any other codec. WB3's `{"name": "SPARSE_ROWS_WB3", "k_top": 3170}` therefore never installs Q8; HOOK 2
  installs `install_sparse` for that name.
- (b) REV-P2-03_04, **WB3 bytes come from the checkpointed ledger**, not from `sp.summary()` (it restarts after a resume).
  The per-visit upload figure (the §5.6 tie-break scalar) = `run.ledger` bytes_up / Σ `run.counters.visit_counts`. HOOK 2's
  record and `bytes_record` should publish it as `bytes.per_visit.upload_bytes` (the field the p2_select adapter reads);
  keep `sp.summary()` only as a leg-local audit.

## 7. Server commands (PLAN; not run)
Variables (as in ops_scripts/endgame/traj_cmp7.sh): `V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4`,
`R=$V/runs_ds3`, `PY=/home/jovyan/venvs/ppsi-gpu/bin/python`, P2 tree `T=$V/code_p2r1` (new deploy beside r6 / r9; this
packet's four edited files + fl/update_log.py + v4int/p2_probe.py + tools/p2_noop_check.py + the P2-03/04/05/06 files),
`PRE=$V/runs_pretrain/PRETRAIN_FOR_FL_SMALL_SASREC_D64_B2_s2026`, `ENV="PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 V4_DATASET_ROOT=/home/jovyan/shared_data/rees46_nextitem_v1"`.

**S0 — firewall on the server, before any P2 run (CPU; model: traj_cmp7.sh).** (1) In `$T/tests_fl` and
`$T/tests_int`: `pytest -q -p no:cacheprovider --basetemp $R/P2/tmp/pt test_p2_update_log.py test_p2_recipe.py
test_fl12_frozen.py test_fl10_fedadam.py` and `test_p2_launch.py test_p2_noop.py test_fl05_launch.py test_fl16.py
test_fl16b.py test_fl15.py` (expect the one known Windows-pin failure of test_fl16 per DL 245 / 342 only). (2) Dry runs
old tree vs `$T`, equal `arm_spec_sha256` / `fl_config_digest`, rc 0: cold DP8 (`--fl-tuning $V/contracts/fl/FL_TUNING_SELECTION.json
--clip-record $V/contracts/dp/S_RECORD.json --epsilon 8`) and DP_WARM_FROZEN (+ `--init-from $PRE/PRETRAIN_FOR_FL_INIT.json
--freeze-item-tables --warm-selection $V/contracts/fl/FL_WARM_SELECTION_S8.json --clip-record $V/contracts/dp/S_RECORD_WARM_FROZEN.json`)
— exactly traj_cmp7.sh step (3) with `code_ds3r7` replaced by `code_p2r1`; add FA_WARM_FROZEN MAIN the same way.

**P2-T3 — incumbent reproduces S8Tc's first 64 rounds (GPU, same GPU class as S8Tc; 65 CALIB rounds + one smoke eval).**
```bash
mkdir -p $R/P2/P2T3
cat > $R/P2/P2T3/INCUMBENT_RECIPE.json <<'J'
{"format": "PPSI_V4_P2_RECIPE_1", "label": "fedadam|eta=0.001|c=0.1", "recipe": {"server_opt": "fedadam", "server_lr": 0.001, "client_lr": 0.1, "items": "frozen", "item_lr_mult": null, "upload": "dense", "lora_rank": null, "lora_lr_factor": null, "personal_head": false}}
J
# history policy and GPU class must be S8Tc's (read its LAUNCH_*.json first; the tool checks both)
cd $T && env $ENV $PY -B v4int/launch_fl.py --arm P2_W --role CALIBRATION --seed 2026 --gpu <g> --allow-shared-gpu \
  --history-policy USER_HISTORY_L50 --init-from $PRE/PRETRAIN_FOR_FL_INIT.json \
  --p2-recipe $R/P2/P2T3/INCUMBENT_RECIPE.json --smoke --smoke-rounds 65 --run-dir $R/P2/P2T3/run
env $ENV $PY -B $V/tools/p2_t3_check.py --code-root $T --p2-run-dir $R/P2/P2T3/run \
  --s8tc-search-root $R/S7S8_T --out $R/P2/P2T3/P2T3_RECEIPT.json      # rc 0 BIT_IDENTICAL, 1 FAIL, 2 refused (no file)
```
Reading: `verdict BIT_IDENTICAL` ⇒ the incumbent may be reused (`p2_select ... --incumbent-phase1 <S8Tc dir>
--p2t3-receipt $R/P2/P2T3/P2T3_RECEIPT.json`; p2_select.check_p2t3_receipt checks test / verdict / rounds 64 /
s8tc_run_id / result_rows_sha256 / run_summary_sha256); `FAIL` ⇒ re-run the incumbent as a WA cell (the receipt lists
which check failed). Evidence wording: **TRAIN_LOSS-exact** — 65 byte-equal lines 0..64, each a function of θ_r, cover
θ_0..θ_64 (all 64 updates) plus the configuration checks; θ_64 itself is not stored by S8Tc. The smoke ends with one
SMOKE_NOT_EVIDENCE evaluation on CALIB_VALIDATION (CALIB only; nobody reads it) and deletes the weights.

**P2-01 checks on real runs (after any P2 CALIB run is DONE_MAIN).**
```bash
cd $T && env $ENV $PY -B v4int/p2_probe.py --run-dir <P2 run dir> --theta-pre $PRE/sealed/checkpoints/endpoint_6.0.pt \
  --gpu <g> --history-policy USER_HISTORY_L50 --out <P2 run dir>/P2_PROBE.json          # ≈ 0.1 GPU-h (INFERENCE)
env $ENV $PY -B $V/tools/p2_noop_check.py --code-root $T --run-dir <P2 run dir> \
  --theta-pre $PRE/sealed/checkpoints/endpoint_6.0.pt --probe <P2 run dir>/P2_PROBE.json --out <P2 run dir>/P2_NOOP.json
```
(DRIFT needs `--calib-value <run PB value> --pre-calib <PRE_CALIB>` given by Ops from the selection record; the tool
reads no value.) First real use: a CPU-only dry check of p2_probe on one finished CALIB run is advised (it is untested on
real data; see Q1).

## 8. Open questions
- **Q1 (p2_probe on real data, PENDING):** the server path (two SMALL models, StoreTrainProvider of the CALIBRATION train
  manifest via `expected_manifests("COHORT", "CALIBRATION")`, label stripping) is untested here. Run it once on a finished
  CALIB run and check `probe_manifest` = CALIB_TRAIN_5_100_32768 before relying on (b).
- **Q2 (criterion (b) wording):** "mean |Δscore| of the top-1 item" is implemented as |s_θ(i*) − s_pre(i*)| with
  i* = argmax s_pre (recorded as `top1_definition`). Using θ's own top-1 gives a different number. The PREG2 text is
  ambiguous; Fable / Ops to confirm at P2_SCRIPT_FREEZE.
- **Q3 (anchor flag): RESOLVED** — p2_select 7a05afcd uses 0.001 (REV-P2-06 M1 fixed).
- **Q4 (LoRA and D1-ext / no-op; REV-P2-03_04 O2):** with LoRA, T = the A / B keys and A starts random, so `rel_pre_dist` and
  §3.2 (a) on T do not measure how far the merged model moved. Proposal: for WC runs P2-04 attaches the observer on the
  merged base weights (or p2_noop_check reads the merged θ that best.pt carries); needs a one-line PREG2 reading note.
- **Q5 (n_clients under DP):** for z > 0 the per-round cohort size is omitted from UPDATE_NORM_LOG (strict reading of 06L_E
  E1 / PREG2 §3.3: no per-client value; |C_r| of a Poisson cohort is outside the noise). The checkpointed participation log
  still has it (Phase-1 machinery). Fable to confirm whether §3.3's "number of clients aggregated" applies to DP runs.
- **Q6 (cos_prev after resume):** Δ_{r−1} is not checkpointed (that would change the checkpoint payload), so the first step
  after a resume has `cos_prev: null` (listed in RUN_SUMMARY). Descriptive only.
- **Q7 (η_s = 0 with an unfrozen recentred output layer):** the η_s = 0 bit-identity holds for frozen tables / tied families
  (tested). With an unfrozen SASRec output layer the server re-applies the recentring, which can change bits (fl.server
  docstring, ≤ 4.7e-10). C0 is θ_pre itself (evaluation, no training), so nothing depends on it.
- **Q8:** the path of Ops' D1 script (DL 407) on the server is unknown here; p2_noop_check is self-contained and does not
  import it.
- **Q9 (deploy):** new fl/update_log.py and the edits change `fl_code_sha256`; deploy only as the new P2 tree (code_p2r1).
  The P2 lock (P2_METHODS_LOCK) must bind the final launch_fl sha (f4e56b80… at this hand-off) — see §9 m3.

## 9. REV fixes (REV-P2-01_02 = PASS_WITH_MINORS; applied 2026-10-03)
| finding | fix | where |
|---|---|---|
| **R1** (required) P2-T3 receipt ≠ p2_select.check_p2t3_receipt | New **tools/p2_t3_check.py** (sha 921320e9d76b81d62e629cb99d373946cb62f17afd1543f2e7aec24a4c9fb416): finds the unique S8 f1/sf0.1 run (or takes `--s8tc-run-dir`), compares, writes the receipt **O_EXCL from Python only after the comparison completed** (a missing input → exit 2, no file). Receipt keys = p2_select's: `test "P2-T3"`, `verdict "BIT_IDENTICAL"` (only if every check holds, else `"FAIL"`), `rounds 64`, `s8tc_run_id` (ARM_SPEC run_id = the run id execute also writes into RESULT_ROWS), `result_rows_sha256` / `run_summary_sha256` (S8Tc file bytes hashed, never parsed) + the checks, `p2_smoke_server_digest`, TRAIN_LOSS shas, tool sha, `self_sha256`. p2_select.py not edited (7a05afcd). | tools/p2_t3_check.py; §7 |
| **m2 (a)** 64 lines miss update 64 | `P2_MAX_SMOKE_ROUNDS` 64 → **65** (P2-only constant); the tool compares **65** TRAIN_LOSS lines (θ_0..θ_64), requires a 65-round SMOKE_DONE_NOT_EVIDENCE, records the P2 smoke `server_digest`; evidence worded "TRAIN_LOSS-exact". | launch_fl `P2_MAX_SMOKE_ROUNDS`, preflight; tool |
| **m2 (b)** extra checks | the tool also compares p2_select's INCUMBENT_COMPARED (cfg, plan, max_rounds, frozen_item_tables, init_from, exposure_budget, role), `check_p2_spec`, the device record, clients, history policy, GPU name and the CALIB manifests. | tool |
| **m2 (c)** empty receipt on assert | no shell redirection any more; O_EXCL write in Python after the comparison. | tool |
| **m3** code-tree discipline | (process note) never overlay P2 files on code_ds3r7 / r7b; P2 runs cannot resume across P2 trees, so when P2-03 / P2-04 change fl/*.py or launch_fl, WA runs started on code_p2r1 must finish on code_p2r1 (06E §8 refuses a cross-tree resume); P2_METHODS_LOCK must list **every** P2 code tree by aggregate sha (as ALL_METHODS_LOCK `code_trees` does), not only launch_fl; tools/p2_noop_check.py (4074e4e8…) and tools/p2_t3_check.py (921320e9…) live outside the code tree — bind their shas separately. | this handoff |
| **m4** stale text | §0 / §6b now say p2_select has its own read_p2_run; Q3 marked resolved; shas updated. | this handoff |
| **m5** log can block completion | P2-only: `_update_norm_lines` skips and counts a torn line; `update_norm_summary` flags `strictly_increasing` / `unparsable_lines` (and `complete False`) instead of raising; `trim_update_norm_log` drops torn lines. Phase-1 TRAIN_LOSS / CLIP_LOG functions untouched. | launch_fl `update_norm_*`; test |

Observations O1–O6: no change (O5 agrees with Q5; O2 / O6 stay with P2_SCRIPT_FREEZE / P2-04).

**New shas:** v4int/launch_fl.py f4e56b8008e7f4a66d5a6a07cbc0b9d978c28d90d9f9596a3b166f5b01475734 (P2-only diff vs
c6192343: P2_MAX_SMOKE_ROUNDS 65, docstring, `_update_norm_lines`, non-raising `update_norm_summary`);
tools/p2_t3_check.py 921320e9…; tests_int/test_p2_launch.py 9a68eaf1… (65 / 66 smoke limit, + test_update_norm_log_never_blocks_completion);
tests_int/test_p2_t3_check.py fc626336…. Unchanged: fl/client.py f869ac90, fl/server.py ccbcc5a1, fl/runtime.py 412d3829,
fl/update_log.py 42449044, v4int/p2_probe.py c25bd351, tools/p2_noop_check.py 4074e4e8, test_p2_update_log 111e99f6,
test_p2_recipe fb9ba7d7, test_p2_noop d183997b; p2_select.py 7a05afcd (not edited).

**Re-runs after the fixes (MEASURED, per file, `-p no:cacheprovider`, temp outside the tree):** test_p2_update_log 4 + 1 xfail;
test_p2_recipe 5 + 1 xfail; test_p2_noop 5 + 1 xfail; test_p2_launch 11 + 2 xfail; test_p2_t3_check 2 + 1 xfail (the PASS
receipt is accepted by the real p2_select.check_p2t3_receipt; FAIL receipts refused); test_p2_select 25 + 4 xfail; and the
Phase-1 launch tests with the same counts as the §5 baseline: test_fl05_launch 9 + 2, test_fl12_launch 14 + 3,
test_fl15 22 + 16, test_fl16 14 + 7, test_fl16b 8 + 9 (all xfails are the NCs). The 56-pair Phase-1 table is still
byte-identical (sha 0c89ec77…); tests_fl / tests_int receipts unchanged.
