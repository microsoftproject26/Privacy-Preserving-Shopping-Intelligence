# REV-ANA-01: independent read-only review of ANA-01, DATA-07, DATA-08, DEV-02/03 (Claude Opus 5.5, 2026-09-29)

- **Scope.** This review follows `company/tasks/REV-ANA-01.md`. Nothing outside this file was edited. No server command was run. Pytest wrote only under `.local_ai/_pytest_tmp/REV-ANA-01/`. Two throw-away probe scripts ran from the session scratchpad, outside the V4 tree.
- **Reviewed file.** `tools/retention_report.py` sha256 = `61c1eafd6a3740455de867f01ba7816dcc5ac5faad99cae10ea1c2b0fb135849` (MEASURED; this equals the ANA-01 handoff).
- **Evidence tags.**
  - MEASURED means file:line, or a run I made.
  - INFERENCE means reasoning.
  - OPINION means judgement.
  - "Sub-review" marks sections B, C and D. Two independent Opus sub-reviewers produced them under this task. I spot-checked their MAJOR claims against the code (the cited lines are re-verified).
- **Classes.** Each finding is one of:
  - **FIX**: must change before the frozen use.
  - **DISCLOSE**: goes into the thesis or registration text.
  - **OK**: no action.
- **Owner of each FIX.** "ANA-02" means the ANA-02 worker implements it in `tools/retention_report.py` or the ANA tests. Other owners are named.

## FIX list (ANA-02 = implement in retention_report.py; others: owner named)
| id | sev | FIX (one line) | owner |
|---|---|---|---|
| A-F1 | MAJOR | trunc_curves reads the wrong field names for DATA-07 rows: the sub-panels merge and C_ref is always missing | ANA-02 |
| A-F2 | MAJOR | K16 cannot be computed under one block device class (the ONNX CPU rows); it needs an ONNX sub-block | ANA-02 (+DEV-04 bench FP32 ranks) |
| A-F3 | MAJOR | RULE_FREE is allowed for non-baseline arms; LO declared RULE_FREE makes K11 refuse the whole report (probe) | ANA-02 |
| A-F4 | MAJOR | Registered single-seed contrasts (K13, K15, K16 "seed 2026") silently become 2-seed means when seed 2027 exists; a missing second seed is also silent | ANA-02 |
| A-F5 | MAJOR | The comparison key covers only 3 fields plus the query hash; add the rest of the E9 key, the 06D §1 status and the pin checks | ANA-02 |
| A-F6 | MAJOR | Rank-coverage guard: every RANKABLE row of the evaluated view must have rank >= 1, or an unscored -1 deflates the metric (INT8 subset) | ANA-02 |
| A-F7 | MAJOR | Arm identity is taken from the index only; a C_FULL / C_BIG mix-up is undetectable. Check `train_manifest_id` / `method` / run id | ANA-02 |
| A-F8 | MAJOR | The TEST path has no ALL_METHODS_LOCK / TEST_EXPOSURE_REGISTER gate and no loader for the TEST extension registry | ANA-02 (+DATA-09) |
| A-F9 | MAJOR | Per-contrast confirmatory rule map, hierarchy and still-improving flag (06D §3-§5); exact locations in A.8 | ANA-02 |
| A-F10 | MINOR | Cross-check the declared rule against the rows' efe / checkpoint_id (CB = 6.0; C_BIG PB = best@3.0 per the lock) | ANA-02 |
| A-F11 | MINOR | `--check-only` must also run every pairing check and the full-population non-zero check (no value printed) | ANA-02 |
| A-F12 | MINOR | The sanity trigger uses C_SMALL at PB; amendment 04 §1 says the "C_SMALL 6.0 value" (CB) | ANA-02 |
| A-F13 | MINOR | Evidence integrity: refuse a RUN_SUMMARY status other than DONE_MAIN, and rows labelled SMOKE_NOT_EVIDENCE / FL_LR_SCREEN / CALIBRATION | ANA-02 |
| A-F14 | MINOR | Record and verify the tools/g2_run.py sha; replicate g2_run's lock-vs-REGISTRY_V2 check (g2_run.py:212-213) | ANA-02 |
| A-F15 | MINOR | Every contrast needs n_decisions (E7.4); check the rows' bootstrap_W_sha256 against the MAIN W when present (E7.2) | ANA-02 |
| A-F16 | MINOR | Check that the STRATA record's source_manifest_id equals roles.MAIN.train_manifest_id | ANA-02 |
| A-F17 | MINOR | historical(): R is computed against C_BIG seed 2026 only; use the seed mean, or name the seed | ANA-02 |
| B-F1 | MAJOR | check_t41_receipt accepts a SMOKE receipt (2 buckets) as the production gate | DATA-07 fix (Sonnet) |
| B-F2 | MAJOR | The registered panel is never re-derived, so an earliest-q_u error or an incomplete sub-panel passes | DATA-07 fix (Opus) |
| B-F3 | MAJOR | WARM_TRUNCATION rows carry row_status "COLD_START_STUDY_A", which breaks E11.1a labelling | DATA-07 fix (Sonnet) |
| B-F4 | MINOR | The builder never consults COLUMN_RULES; add an import-time rule/column equality assert (UNSUPPORTED_FUTURE refuses) | DATA-07 fix |
| B-F5 | MINOR | Two weak negative controls (a vacuous events-to-X NC; a merged-stream NC that can skip) | DATA-07 fix |
| B-F6 | MINOR | run_all.py: mkdir --basetemp (parents) and set TMP/TEMP (the task's F14) | ANA-02 (per its packet) |
| B-F7 | MINOR | events_to_x_table does not check that the C_ref rows come from arm C; no OOV-flag count per cell | DATA-07 fix |
| C-F1 | MINOR | verify_ext_registry does not check the TIEBREAK band / new manifest population fields | DATA-08 fix (Sonnet) |
| C-F2 | MINOR | c_full.check_launch_args does not refuse --transform-state, --allow-shared-gpu, --manifest-dir, ... | DATA-08 fix (Sonnet) |
| D-F1 | MAJOR | bench.ranks_from_int8_session raises on OOV rows (target -1): 4,078 E2E OOV rows in MAIN_VALIDATION | DEV fix (Sonnet) |
| D-F2 | MAJOR | `--first-users` = eval_user_key < N is digest (v1-key) order, not the v2 population-selection-key order of 06A §6 | Ops registration note OR DEV code |
| D-F3 | MINOR | Browser page: SRI/crossorigin; numThreads set after the first session; no ckpt/rows sha in queries.json | DEV fix (Sonnet) |

---
## A. retention_report.py

### A.1 Ratio construction and pairing — OK, with A-F3 / A-F4 / A-F10
- **Per-resample ratio — OK (MEASURED).**
  - R_b = 100·M_b/C_b is computed on ONE shared W per user set: `ratio()` L757-803 uses `eng.res(num, vkey)` and `eng.res(den, vkey)` on the same cached plan (`Engine.view` L663-673).
  - The plan is `v4train.bootstrap_ci._plan`, i.e. `make_plan(unique(euk[eval_mask]))` (bootstrap_ci.py:47-48).
  - `make_plan` sorts the keys and seeds W with default_rng(20260926), 1,000 resamples (v4eval/bootstrap.py:29-38, 60-67).
  - A zero C_b gives UNDEFINED with the count, and no resample is dropped (L777-782; E7.5).
  - Per E7.2 each distinct user set (subset, stratum or bin) gets its own W. That is correct and recorded as `W_sha256`.
- **2-seed mean — OK (MEASURED).** The point is the mean of the per-seed R, and the CI comes from the per-resample mean of the per-seed ratio vectors on the same W (L798-800). This is the 06A §7 reading.
  - DISCLOSE (OPINION): the thesis must say "mean of per-seed ratios", not "ratio of seed means"; amendment 04 §1 is ambiguous.
- **Can a PB numerator pair with a CB denominator? No.** Entries are keyed by (role, arm, rule) (`Book` L846-851), and `check_pairs` refuses a rule mismatch or a seed mismatch (L736-754). The NC is `test_nc_pairing_rule_and_seed`.
  - Residual hole A-F10: when the rows carry no `selection_rule`, the index is trusted (`rule_source = "INDEX"`, L469-470).
    - launch_device rows carry `selection_rule = None` (launch_device.py:537).
    - reeval rows carry whatever `--selection-rule` was passed (reeval.py:152).
    - A mislabelled best.pt / endpoint pair is therefore undetectable.
  - **FIX A-F10:** in `select_rows` (L452-479), when the rule is not RULE_FREE:
    - CONTROLLED_BUDGET requires every evidence row to have `efe == 6.0`, or a `checkpoint_id` ending `endpoint_6.0`;
    - PRACTICAL_BEST with a `run_dir` requires `efe == best_half_efe/2`;
    - C_BIG PRACTICAL_BEST requires `efe == 3.0`, or a checkpoint id `best@3.0` (06 §7 "best@3.0 per the lock");
    - a row without efe or checkpoint_id is refused unless it carries `selection_rule`.
- **A-F3 (MAJOR, MEASURED by probe): RULE_FREE for a non-baseline arm crashes the report.**
  - `load_index` accepts RULE_FREE with a seed for any arm (L414-422).
  - `Book.get` falls back to RULE_FREE entries (L855).
  - `check_pairs` then refuses a RULE_FREE denominator against a checkpoint-rule numerator (L741-742).
  - Probe: the anakit block with its 4 LO entries set to RULE_FREE (the natural choice, since device rows have no selection_rule) gave `rc 2 ANA REFUSED: FA|2026|PRACTICAL_BEST|MAIN / LO|2026|RULE_FREE|MAIN: a rule-free denominator cannot pair with a checkpoint rule`. That is K11, FA − LO.
  - **FIX:** in `load_index`, allow RULE_FREE only for `BASELINE_ARMS = (RULE, SKNN, LAST, POP, KNN_EMB_SMALL, KNN_EMB_BIG)`. Device arms (LO, FT_C, FT_FA, FT_KD) and C_SMALL_INT8/FP32 must be declared PRACTICAL_BEST. Refuse otherwise, with a clear message.
  - Also, in `Book.pairs`, return [] (so the result reads NOT_AVAILABLE) instead of letting a RULE_FREE denominator reach `check_pairs`.
  - Add a test covering LO under CONTROLLED_BUDGET: K8 and K11 read NOT_AVAILABLE, and the run does not refuse.
  - DISCLOSE: device arms, and therefore K6-K11 and K16, exist only under PRACTICAL_BEST.
- **A-F4 (MAJOR): registered seed sets.**
  - `Book.pairs` pairs every seed present (L861-872).
  - 06B §2 registers K13 "(seed 2026)", K15 "(seed 2026; …)" and K16 "(seed 2026)".
  - 06D §8 lists DP1 s2027 and C_FULL s2027 as NICE runs. If they run, K13 and K15 silently become 2-seed means, which is a different confirmatory number from the registered one. The reverse case is also silent: K2 with only one FA seed is reported with `n_seeds = 1` and no flag.
  - **FIX:** add `REGISTERED_SEEDS = {"K13": (2026,), "K15": (2026,), "K16": (2026,)}`, default (2026, 2027).
    - The CONFIRMATORY row uses exactly those seeds.
    - Any extra-seed mean is printed as DESCRIPTIVE.
    - A missing registered seed gives `status = INCOMPLETE_SEEDS` beside the value (no refusal, per the 06D §8 cut rule).

### A.2 Rank files, OOV, user set — OK, with A-F6
- **Rank-to-row agreement — OK (MEASURED).**
  - `check_rank_rows` (L496-509) recomputes macro-E2E MRR@20, NDCG@10 and HR@10 from the ranks through `v4eval.metrics.credits_from_ranks` + `population_block`, and refuses when `|d| > 1e-9`.
  - It runs for every non-historical ranked entry (L641-646) and for every FL mark (L559). The NC is `test_nc_rank_file_swapped`.
- **OOV / unrankable rows — OK (MEASURED).** A rank < 1 gets credit 0 through the same library function (metrics.py:31-40), so this is identical to v4eval.
- **User set — OK (MEASURED).** The E2E set (`euk[eval_mask]`) is used, not RANKABLE (bootstrap_ci.py:47-48; `user_stats` with `eval_mask`, L683).
- **A-F6 (MAJOR): no rank-coverage guard.**
  - `bench.ranks_from_int8_session` writes -1 for every unscored row (bench.py:173). A -1 on a RANKABLE row is scored as a miss.
  - If `users_subset` is omitted, or the subset differs from the dumped users, the INT8 value is deflated. The rows/ranks agreement check cannot catch this when both come from the same ranks.
  - **FIX:** in `load_block` after L644, and in `load_marks`, require `ranks[view.rankable] >= 1` for every ranked entry on its own view (full view or `users_subset`). Also require `users_subset` for C_SMALL_INT8 / C_SMALL_FP32 whenever any rankable row outside it has rank -1.

### A.3 Comparison key; baselines; INT8 subset — FIX A-F2, A-F5
- **What is checked — OK as far as it goes (MEASURED).**
  - `check_key` (L565-610) checks eval_device_class, split and evaluator_code_sha256 across all non-historical evidence and mark rows.
  - It checks query_order_hash per role, against the loaded manifest (the subset view hash for `users_subset`).
  - It checks eval_manifest_id when present.
  - `EvalManifest.hashes` is a property of `eval_mask`, so the subset hash differs from the full one (v4eval/manifest.py:66-69).
- **A-F5 (MAJOR): the key is too narrow for E9.1.**
  - `v4eval.rows.COMPARISON_KEY` (rows.py:26-29) also contains candidate_order_hash, label_policy_hash, data_protocol_hash, feature_contract_hash, precision_id, objective_id, eval_score_block, eval_batch and eval_stream. None of these are compared. A row from another catalogue order or label policy would silently enter a confirmatory ratio.
  - Since 06D §1, a pinned A40 row carries `status = MEASURED_PINNED_DEVICE` and `device_pin_sha256`. A pre-pin A40 row carries the SAME eval_device_class with `status = REHEARSAL_NON_FROZEN_DEVICE` (rows.py:113-114), so today it passes. The script's own note (L609-610) predates 06D.
  - **FIX:**
    - Add to `KEY_FIELDS` (L87): candidate_order_hash, label_policy_hash, data_protocol_hash, feature_contract_hash, precision_id, objective_id, eval_score_block and eval_stream. These must match across non-historical rows.
    - Record, not refuse, `INTENDED_DIFFERENCES = (population_hash, topology_hash, initialization_block, eval_batch)` per entry. eval_batch is included because of the launch_device per-client batching of 06A §9(f).
    - Refuse any non-historical row whose `status != "MEASURED_PINNED_DEVICE"` when the index class is `NVIDIA_A40-FP32-DET`, and require one `device_pin_sha256` across the block.
    - If a server `--check-only` then shows a legitimate difference, Ops moves that field to the intended list with a DECISION_LOG row. The key is never silently weakened.
- **Baselines — OK / DISCLOSE (MEASURED).**
  - CPU baselines differ in eval_device_class, so they must be `historical_reference: true`. They are then point-only (L647-649), excluded from `Book` (L848), and labelled HISTORICAL_REFERENCE (L1440-1449, report L1680-1683).
  - An unranked non-historical entry gets its value from its rows and `ci95 = NO_RANKS` (L694-701).
  - The tests give POP the A40 class (anakit.py:95, 252). The real CPU POP would have to be historical.
- **INT8 subset — OK for the denominator.** `_vkey_for` takes the numerator's subset (L875-877). A ranked C_SMALL / C_SMALL_FP32 denominator is then re-evaluated on the same users, and `check_pairs` refuses a denominator that sits on a different subset (L749-751).
- **A-F2 (MAJOR): K16 is structurally not computable.**
  - C_SMALL_INT8 (and any ONNX FP32) rows come from onnxruntime on CPU (bench.py:1-20), so their eval_device_class is not the A40 class. As non-historical entries they make `check_key` refuse (L596-598). As historical entries they drop out of `Book`, and K16 reads NOT_AVAILABLE.
  - The `CONTRAST_FALLBACK` C_SMALL_FP32 → C_SMALL (L123, L998-1000) would divide ONNX-CPU INT8 by PyTorch-A40 FP32. That mixes quantisation with runtime numerics, which contradicts the registered "R(INT8 / FP32)".
  - No producer writes `rows` for ONNX ranks: bench writes only a ranks file and BENCH_RESULT.json (bench.py:355-365), and `load_index` requires `rows` (L425-426).
  - **FIX (ANA-02):**
    - Add an ONNX sub-block: entries with `arm in (C_SMALL_FP32, C_SMALL_INT8)` may declare `"comparison_block": "ONNX_CPU"`.
    - These are exempt from the A40 device and status checks only. Among themselves they must share one runtime identity (onnxruntime version, intra_op_num_threads from BENCH_RESULT), the query hash of their subset, and the evaluator sha.
    - Allow `rows: null` for them, with the value computed from ranks and `BENCH_RESULT.json` given. The script verifies `retention.ranks_sha256` against the file.
    - K16 is CONFIRMATORY only as C_SMALL_INT8 / C_SMALL_FP32 inside that sub-block.
    - The fallback to A40 C_SMALL is printed as DESCRIPTIVE with the label `DEVICE_MIXED`.
  - **FIX (DEV):** `v4dev.bench` also writes FP32 ONNX ranks for the same rows (tag onnx_fp32), after D-F1.

### A.4 Contrast registry — OK
- `load_contrasts` (L926-944) refuses ids other than exactly K1-K16, duplicate ids, and any contrast or question text that differs from `REGISTERED_CONTRASTS` (L104-121).
- `test_embedded_registered_text_matches_the_contracts` ties that table to the 06B file.
- OPINION: `_norm_text` (L881-882) folds whitespace and the minus/en-dash variants, so the check is "verbatim up to whitespace and dash glyphs". That is acceptable. DISCLOSE it in the ANA freeze note.
- K9-K11 use the same W as the full MAIN view (L1006) and seed-s-with-seed-s pairing (`Book.pairs`).
- K15 with one C_FULL seed pairs with C_BIG seed 2026 only (the pairs loop is over the denominator seeds, L864). The test `test_single_seed_contrasts_and_subset_int8` covers this. See A-F4 for the s2027 case.

### A.5 Curves and bins — OK except A-F1
- **Bins — OK (MEASURED).**
  - `SUPPORT_BINS` = {0, 5-10, 11-20, 21-50, 51-100}, inclusive (L90, L1209-1213); the NC is `test_nc_support_bin_edges`.
  - n_support is the client's row count in the role's TRAIN-layout manifest, via the verified `v4base.identity` tables (L303-314). INFERENCE: this equals the pool count that defines "5-100", because DATA-08's arithmetic makes the pool examples the sum of TRAIN-layout decisions (fs_common.py:37-38). So MAIN users fall only in 5-100, and any user outside is counted as OUT_OF_REGISTERED_BINS and excluded from the rows (L1231).
  - Bin 0 comes from TRUE_COLD (L1223-1224, L1251).
  - events-to-X is the smallest bin with M ≥ X·C_ref on raw floats; there is no interpolation and "NOT_REACHED" is reported (L1258-1261). The NC is `test_nc_events_to_x_interpolates`.
  - The strata join is by client_key_hex (L1193-1202), and NO_TRAIN gets its own group (L1195-1198). See A-F16.
- **FL bytes — OK + DISCLOSE (MEASURED).**
  - `bytes.per_client.mean` = mean over clients of visits × (down + up) (launch_fl.py:332-336).
  - `visit_counts` counts survivors only (participation.py:245-268), so dropped clients are not charged, which is consistent with the ledger (launch_fl.py:340).
  - With a constant expected number of survivors per round (57.6, or 1,024 for DP / FA_1024), the fleet-mean cumulative bytes grow linearly in rounds. The prorated value `mean × round/rounds` (L1080-1081) is therefore exact up to drop-out noise.
  - DISCLOSE: it is a mean, not a per-client curve; `retry_bytes_down` is in `fleet` but not in `per_client`.
- **A-F1 (MAJOR, MEASURED by probe): the DATA-07 row fields are not the ones `trunc_curves` reads.**
  - DATA-07 rows carry `trunc_panel`, `trunc_panel_type`, `trunc_h` (None when full), `trunc_full_history`, `trunc_query_h` and `method` (launch_trunc.py:212-222, 238-242).
  - `trunc_curves` reads `panel|role`, `subpanel|sub_panel`, `h|trunc_h|history_budget` and `full_history` (L1282-1290).
  - Probe result: every row gets panel None and subpanel "". The FULL rows (h None) are dropped, so every R = NO_C_REF. FIXED_20 h=5 and FIXED_50 h=5 collide in one key, and the last row silently wins (M = 0.2 instead of two values).
  - **FIX:**
    - Read `trunc_panel`, `trunc_panel_type`, `trunc_h`, `trunc_full_history` and `trunc_query_h`.
    - Key the C_ref per (panel, panel_type, seed, query_h) for ALL_AVAILABLE and per (panel, panel_type, seed) for FIXED.
    - Refuse duplicate keys.
    - Require `method in (C_BIG,)` for C_ref (E11.10 "C method").
    - Apply the block key (eval_device_class, split, evaluator sha) to TRUNC rows.
    - Require each TRUNC_ROWS file to declare its rule (PRACTICAL_BEST), because DATA-07 rows have `selection_rule = None` (launch_trunc.py:236).
    - Print M's own FULL-history value beside each series (E11.10 "always reported alongside"; it is dropped today at L1294).
    - Add a known-answer test built from `launch_trunc.trunc_labels` output.

### A.6 Determinism and safety — OK, with A-F11
- **Determinism — OK (MEASURED).** Sorted JSON, no clock, relative path keys (L225-232, L1491). `test_deterministic_rerun_byte_identical` passes.
- **`--check-only` — OK (MEASURED).** It prints the key strings, counts and the script sha, never a metric (L1765-1769); `test_check_only_…` greps for numbers.
  - A-F11: pairing (L736-754) and the E8.3 full-population check (L726-733) run only in `analyse`. A check-only PASS can therefore be followed by a refusal of the real run. **FIX:** add a `plan_pairs(book, contrasts)` pass to the check-only path that runs `check_pairs` over every table/K pair and `require_nonzero_denominator` on the full views. It prints nothing numeric.
- **Output writes — OK (MEASURED).** `--out` must be absent or empty, and every file is written O_EXCL (L239-247, L1757-1759).
- **Exit codes — OK (MEASURED).** 0, 2 (refused), 5 (error) (L1778-1788). MINOR DISCLOSE: argparse usage errors also exit 2.

### A.7 TEST split — FIX A-F8
- **Today (MEASURED):**
  - `SPLIT_VALUES["TEST"]` accepts TEST / RETRO_TEST rows (L88).
  - `shift_indicator` uses `records.VALIDATION_RESULT`, with its self-sha verified and its split required to be VALIDATION (L1395-1401, L1405-1423).
  - The production loader goes through LOCK_V2 → REGISTRY_V2 → `verify_manifest_by_registry` (g2_run.py:174-197). REGISTRY_V2 has no TEST manifest, and `v4data.manifests.refuse_test_manifest` refuses any id containing "TEST" (manifests.py:98-105). A TEST run therefore ends in ERROR (KeyError / TestGateError, exit 5), not in a clean refusal.
- **What is missing:**
  - The script never checks ALL_METHODS_LOCK or the TEST_EXPOSURE_REGISTER (E10.1: "the builder and the loader both refuse otherwise").
  - It never checks that `--contrasts` IS the lock.
- It reads TEST values only when `--split TEST` is given and the TEST rows exist (OK: nothing is read early).
- **FIX (ANA-02; the loader part depends on DATA-09):**
  - (i) `--split TEST` requires `--all-methods-lock <path>` and `--test-exposure-register <path>`. The lock sha must be in the register, and `--contrasts` must equal the lock file (sha).
  - (ii) `--registry-ext <REGISTRY_V2_TEST_EXT.json>`: verify every REGISTRY_V2 entry is byte-identical, with REGISTRY_V2 pinned by LOCK_V2 (reuse the rule of DATA-08's `verify_ext_registry`, d8_registry.py:79-106). Load TEST manifests only through the DATA-09 gated reader, and require the register entry id in each TEST manifest header.
  - (iii) Role map for the index:
    - MAIN.eval = MAIN_RETRO_TEST_5_100_131072, with train_manifest_id unchanged (MAIN_TRAIN, for strata and bins);
    - UNSEEN_HOLDOUT.eval = UNEW_HOLDOUT_RETRO_TEST_32768, with train = UNEW_HOLDOUT_SUPPORT_32768;
    - TRUE_COLD.eval = TRUE_COLD_RETRO_TEST.
  - (iv) VALIDATION_RESULT is required for TEST. Its script sha, library shas and contrasts sha must equal this run's (the same frozen script).
  - Until DATA-09 lands, (ii) is a hard refusal with a clear message instead of exit 5.

### A.8 Smallest change: per-contrast rule, hierarchy, still-improving (06D §3-§5) — FIX A-F9
- **Constants (after L121).** Add:
  - `CONFIRMATORY_RULE = {"K12": "CONTROLLED_BUDGET", "K13": "CONTROLLED_BUDGET"}` (default PRACTICAL_BEST);
  - `HIERARCHY = {"K2": "primary", "K1": "secondary", "K3": "secondary", "K8": "secondary", "K12": "secondary"}` (default tertiary);
  - the 06D §3 and §4 decision sentences, embedded verbatim, extending `test_embedded_registered_text_matches_the_contracts` (test_retention_report.py:117);
  - `DISCLOSURE_06D_3 = "best-of-12 selection on the users' own validation data is not covered by the accountant"`.
- **`load_contrasts` (L926-944).** Keep the parsed outer object. Require `obj["confirmatory_rule_map"] == CONFIRMATORY_RULE` (refuse if it is missing or differs). Return the map, and put it in `build_result.labels` (L1504).
- **`confirmatory` (L992-1014).** Pairing needs no change: `Book.pairs(..., rule)` already uses one rule on both sides. After each `rec["by_rule"][rule] = …`:
  - set `["label"] = "CONFIRMATORY" if rule == CONFIRMATORY_RULE.get(id, "PRACTICAL_BEST") else "DESCRIPTIVE"`;
  - for K12 / K13 under PRACTICAL_BEST, add `["disclosure"] = DISCLOSURE_06D_3`;
  - set `rec["confirmatory_rule"]` and `rec["hierarchy"]`;
  - apply the A-F4 seed map here too.
- **`render_report`.**
  - L1569: add the multiplicity line: "16 confirmatory contrasts; unadjusted 95% paired user-bootstrap CIs (1,000 resamples); hierarchy K2 primary; K1, K3, K8, K12 secondary; rest tertiary (06D §4); claim vocabulary §7.6 for K1-K16 only".
  - L1574-1584: add the columns `hierarchy` and `label`, bold the CONFIRMATORY row, and print the disclosure.
- **Still-improving flag.**
  - In `load_marks` (L526-541), before the `ranks_rel` branch, always set `best = float(summ["best_half_efe"]) / 2` and `entry.meta_auto.update(best_mark=best, still_improving_at_6_0=best >= 5.5)`.
  - Add both keys to `_meta_row` (L949-950).
  - In `retention_table` (L973-977), add `rec["still_improving_at_6.0_any_seed"]` (the OR over the seeds' entries with marks).
  - Add a column in the report table (L1594-1603), labelled DESCRIPTIVE.
  - E12.1(6)'s definition, "best at 5.5 or 6.0", equals `>= 5.5`: OK.
- **Tests.** Cover the map applied; K12 PB labelled DESCRIPTIVE; a lock JSON without the map refused (NC); the hierarchy labels; the flag from an anakit run_dir with `best_half_efe = 11`.

### A.9 Other ANA findings (MINOR FIX unless stated)
- **A-F7 (MAJOR): arm identity.**
  - C_FULL rows inherit `run_label = PRODUCTION`, `row_status = CENTRAL_CONTEST` and `c_matched_eligible = True` (sub-review, trainer.py:574-576), and they carry `train_manifest_id` (trainer.py:543).
  - **FIX:** in `select_rows`, when the field is present, require:
    - `train_manifest_id == "FULL_TRAIN_5_100_TIEBREAK"` for C_FULL;
    - `train_manifest_id == "MAIN_TRAIN_5_100_131072"` for C_BIG and C_SMALL;
    - `method == arm` for FL rows and device rows (launch_fl.py:311).
- **A-F12:** `sanity_trigger` (L1426-1432) uses PB. Amendment 04 §1 says "C_SMALL 6.0 value < 0.5× C_BIG". Use C_SMALL CONTROLLED_BUDGET vs C_BIG PB (the headline denominator), and print both.
- **A-F13:** `load_marks` never checks `RUN_SUMMARY.status == "DONE_MAIN"` (launch_fl.py:490). Also refuse rows with `run_label == "SMOKE_NOT_EVIDENCE"` or `row_status in ("FL_LR_SCREEN",)` or `role == "CALIBRATION"` (launch_fl.py:316-320).
- **A-F14:** `_load_g2` (L280-284) executes tools/g2_run.py, but its sha is not in `library_files_sha256` (L269-276). Also, the ANA loader skips g2_run's `lock_manifest_registry_file_sha256` equality check (g2_run.py:212-213). Record the sha, verify it against ANA_SCRIPT_FREEZE, and replicate that check in `production_context_loader`.
- **A-F15:** E7.4 requires n_decisions per contrast. `ratio`/`delta` records carry n_users and W_sha256 only (`Engine.meta` L703-707); add `n_decisions` (the sum of N of the view). Also E7.2: when rows carry a non-null `bootstrap_W_sha256`, it must equal the MAIN full-view `W_sha256`.
- **A-F16:** `curve_strata` (L1186-1206) does not check `STRATA.source_manifest_id == roles.MAIN.train_manifest_id`. Note also that `contracts/strata/STRATA_CUTPOINTS_MAIN.json` does not exist yet (MEASURED, file not found). Ops must write it, O_EXCL, before the preflight (06 §6).
- **A-F17:** `historical()` L1443-1446 computes a RULE_FREE baseline's R against the first C_BIG seed only. Use the mean over seeds, or print the seed.
- **DISCLOSE:** only retained% is reported. E7.5's loss_b = 100 − retained_b is exact per resample, so its CI is the mirror interval.

## B. DATA-07 (v4trunc; sub-review, MAJORs spot-checked)
- **B.9 E11.4 — OK (MEASURED).**
  - LOCAL_HISTORY is recomputed over the listed rows only: the H_h(q) store-order predecessors plus e_q (builder.py:310-350, recompute.py:34-168).
  - The user context of non-query rows is zeroed.
  - SHARED columns are copied unchanged, identical across h (rules.py:43-54,69; test_e11_rules.py:76-82).
  - B-F4 MINOR FIX: there is no active UNSUPPORTED_FUTURE refusal. The builder hard-codes its column sets and never reads COLUMN_RULES, so the rules.py:3-4 docstring is false. Add an import-time equality assert between the rule classes and the builder's column sets, with any other rule raising.
- **C0 carry-in — OK + DISCLOSE (MEASURED/INFERENCE).**
  - C0(u) is the float64 cumsum of ln(1+price) over every preceding user's non-TEST rows of the replay stream (TRAIN, +VS or +VC) in the bucket, ordered (user_id, ts, decision_id), taken at u's first row (events.py:316-334).
  - C0 includes validation-period rows, which can lie AFTER decision_ts(q) in time.
  - It cancels algebraically: psum = cu[p] − cu[p−W] (recompute.py:117-118). It changes only float64 rounding (about 5e-8 absolute), i.e. at most the last float32 bit of recent_mean_log_price and current_minus_recent_mean_log_price.
  - It carries no usable information, and E11.7 deletion invariance holds (test_e11_rules.py:215-228).
  - **DISCLOSE verbatim:** "window price sums reproduce v1's single bucket-stream float64 cumsum; C0 cancels algebraically and affects only the last float32 bit of two price channels."
- **B.10 T41.**
  - Admission is restricted to `user_context` cells on builder-flagged rows whose differing columns all lie inside the flag's columns (golden.py:123-124,167-177): OK.
  - DISCLOSE:
    - Columns 5 and 6 are also admissible on `main_oov_collision` rows. "Only column 7" is a bucket-0 fact, not a property of the code.
    - The "173,921 / 7 admitted" figure has no receipt yet, so it is REPORTED.
  - `--require-exact-t41` is wired (launch_trunc.py:263 → 558 / 649 → 205-208; exit 2 at 765-767). OK. It blocks the whole panel.
  - **B-F1 MAJOR (MEASURED):**
    - `check_t41_receipt` (launch_trunc.py:185-209) never reads `smoke_buckets` / `label_smoke` (written at :449), so a 2-bucket SMOKE receipt gates production.
    - **FIX:** refuse `smoke_buckets or label_smoke` unless `--smoke`. Require `golden_replay.n_compared` to equal the panel's query count, and bind to `panel_record.content_sha256`.
  - NCs: all 9 are strict xfail with real assertions (MEASURED, all failed with AssertionError).
    - B-F5 MINOR: `test_nc_events_to_x_interpolates` is vacuous (test_studies.py:155-157); interpolation gives 8, not 7. `test_nc_golden_merged_stream` can `pytest.skip` (test_t41_golden.py:76).
    - Fix: assert the value `not in (5, 10)`; replace the skip with `assert rows.size > 0`.
  - Study B labels — OK:
    - support requires e_j ∈ H_h(q), target_class ≥ 0 and target_ts < decision_ts(q) strictly (panel.py:250-256);
    - support blocks never contain e_q (provider.py:75-81);
    - loss_mask = True is set only on support (provider.py:91);
    - every cell restarts every client from `start_state`, one query per user per cell (launch_device.py:379).
    - There is no leakage between queries.
- **B.11 Panels.**
  - One query per (type, h, user), n_permitted == h, FIXED queries identical across h, and FIXED_20 h ∈ {0,1,5,10,20} (panel.py:213-242): OK.
  - **B-F2 MAJOR (MEASURED):** `derive_panel` (panel.py:260) is used only by the test fixture (tests_trunc/conftest.py:61-62). A registered panel with a later-than-earliest q_u, or with missing eligible users, passes `load_panel_world` (launch_trunc.py:359). **FIX:** re-derive the panel from the loaded event table there, and refuse unless the set of (panel_type, h, client_key_hex, query_decision_id) is identical.
  - DISCLOSE:
    - At h = 50 the model sequence holds 49 prior events plus e_q (the L50 cap, builder.py:385), while the user context uses 50.
    - For a VC query, h counts VS events that its user context never replays (rules.py:107-108).
- **B-F3 MAJOR (MEASURED).** Study A rows set `row_status = "COLD_START_STUDY_A"` for WARM_TRUNCATION too (launch_trunc.py:241, while comparison_status is correct at :220). **FIX:** `row_status = labels["comparison_status"]`.
- **B-F7 MINOR.** `events_to_x_table` (launch_trunc.py:133-158) should refuse C_ref rows whose method is not the C arm, and should add the per-cell count of OOV-flagged queries.
- **B-F6 MINOR.** run_all.py does not create `--basetemp` or set TMP/TEMP.

## C. DATA-08 (C_FULL; sub-review)
- **Population — OK (MEASURED).**
  - TIEBREAK = ranks [0, 1,348,682) of the pool in ascending selection_key_v2 order (fs_common.py:47-49; d8_build.py:179-187; d8_common.py:88).
  - 1,348,682 + 3×32,768 = 1,446,986 = POOL_N.
  - The cover array is exactly 1, and the band is disjoint from U_NEW_VAL, U_NEW_TEST and FS by rank and by user (d8_common.py:90-99).
  - The TB hash, user count and decision count are hard checks (d8_build.py:202-207, 261-264).
  - TRAIN rows only, from the unedited M01 `_train_manifest` with its assertions (d8_build.py:253-274).
- **Registry extension — OK + C-F1 MINOR.** Every base key, role, manifest and band is canonical-JSON identical, and only the named additions are allowed (d8_registry.py:79-106). The content of `population_bands.TIEBREAK` and the new manifest's population fields are not checked. **FIX:** assert rank_range == BANDS["TIEBREAK"], the TB sha, n_users and n_decisions.
- **Wrappers — OK (MEASURED).**
  - They are installed only from `c_full.main` (c_full.py:299).
  - `check_launch_args` forces family, phase MAIN, recipe 02B_SASREC, COHORT, USER_HISTORY_L50 and the `C_FULL_` run id (c_full.py:77-94).
  - Wrapper 3 passes non-extension paths through unchanged (c_full.py:161-162).
  - The eval manifest entry, the evaluator and transform_state (no refit) are unchanged.
  - RunConfig differs only in run_id, view path, cuda id, train_manifest_id, population_hash and manifest_registry_self_sha256.
  - **C-F2 MINOR:** also refuse `--transform-state`, `--allow-shared-gpu` (06D §7: alone on the card), `--manifest-dir`, `--manifest-hashes`, `--header-list`, `--lock-path`, `--theta0-variants` and `--hp-registry`.
  - **DISCLOSE:** resume a C_FULL run only through c_full.py; plain launch_central rewrites the trainer view (launch_central.py:584).
- **Row identity.** C_FULL rows look like contest C rows (`c_matched_eligible=True`, trainer.py:574-576). `v4train.selection.downstream_c_rows` filters on family, row_status, rule and method only, so pooled C_FULL rows would pass as C. ANA side: A-F7. **Ops rule:** keep `runs_cfull/` out of every runs_p3 glob, and disclose that ANA overrides the trainer labels.
- **C-F3 DISCLOSE.** A C3 auto-continuation to 8.0 applies (06D §7, accepted). The 06 §5 figure 1,315,914 is corrected by 06D §7.

## D. DEV-02 / DEV-03 (sub-review, MAJORs spot-checked)
- **D-F2 MAJOR (registration; MEASURED): `--first-users 16384` is ASSUMED, not proven.**
  - eval_user_key = the dense rank by raw client digest (manifests.py:528-533, 667-670), i.e. the v1 selection key (common.py:271-275).
  - The 5-100 population and MAIN are ordered by selection_key_v2 = sha256(SEL_PREFIX_V2 + digest)[:8] (census_v2/core.py:60-63). That is a different permutation.
  - Both orders are outcome-free hash orders, so the subset is unbiased. But "population-selection-key order" (06A §6) literally means v2.
  - The rank covers MAIN_VALIDATION's 51,211 users, and censored-only users also get a key. So `euk < 16384` gives ≤ 16,384 users with E2E rows (INFERENCE: slightly fewer).
  - **FIX, one of:**
    - (a) Ops registers, before the dump: "06A §6 fallback users = eval_user_key < 16,384 (client-digest order); the E2E count is reported";
    - (b) DEV code: sort the E2E users by `selection_keys_v2(client_key_hex)` and keep 16,384.
  - The ANA `users_subset` must be exactly the dumped user set (A-F6 guards this).
- **D-F1 MAJOR (MEASURED).** dump_rows keeps OOV E2E rows (target -1). MAIN_VALIDATION has 519,714 E2E vs 515,636 rankable rows (REGISTRY_V2). `ranks_from_int8_session` calls `rank_from_scores`, which raises on target < 0 (bench.py:174-183; v4eval/ranking.py:56-62). **FIX:** skip `target_class < 0` rows (leave -1 = credit 0, as v4eval does), record `n_oov_skipped`, and add a test with OOV rows. The synthetic fixtures are all-rankable, which hides the problem.
- **DISCLOSE.** `np.savez` embeds zip timestamps, so `rows_npz_sha256` is not reproducible across runs. The sha still binds the file that was used; drop the "deterministic on disk" claim.
- **Browser — OK / D-F3 MINOR.**
  - OK: the version is pinned `onnxruntime-web@1.24.1` (index.html:15); the only other script is local bench.js; the parity logic is sound (a top-20 set on queries 0-49, the same stable tie order, reference = the PyTorch CPU wrapper).
  - **FIX:**
    - add SRI `integrity` + `crossorigin`, or vendor the files;
    - `ort.env.wasm.numThreads` is set after the first session (bench.js:153-155), so run each thread setting in a fresh page load or record the applied value;
    - put `ckpt_sha256` / `rows_npz_sha256` into queries.json and check them against the BENCH_RESULT model sha.
  - **DISCLOSE:** INT8 parity against the FP32 reference is quantisation drift; WebGPU INT8 may fall back to CPU kernels; the tag is not verified reachable. The browser output is HISTORICAL_REFERENCE (06B §6.6).

## E. Other issues that could make a thesis number wrong; deviations from registered text
| deviation | registered text | verdict |
|---|---|---|
| Seed mean of R = mean of per-seed ratios | 06A §7 "then the 2-seed mean" | ACCEPT + DISCLOSE (A.1) |
| K13 / K15 / K16 become 2-seed if s2027 exists | 06B §2 "(seed 2026)" | FIX A-F4 |
| K16 denominator falls back to A40 PyTorch C_SMALL | 06B §2 "R(INT8 / FP32) for C_SMALL" | FIX A-F2 |
| Sanity trigger at PB | 04 §1 "C_SMALL 6.0 value" | FIX A-F12 |
| Curves, strata and bins at PB only | 06A §2 C_ref = C_BIG PRACTICAL_BEST | ACCEPT |
| Device arms, INT8 and K6-K11 only under PB | 06B §2 "CONTROLLED_BUDGET reported" | ACCEPT + DISCLOSE (no CB device run exists) |
| Contrast text check folds whitespace and dashes | 06B §2 "verbatim" | ACCEPT + DISCLOSE |
| FL bytes at marks prorated | 06A §3 "Bytes = fl.comm.visit_bytes" | ACCEPT + DISCLOSE (the mean is exact up to drop-out noise) |
| INT8 subset in digest order, not v2 key | 06A §6 "population-selection-key order" | FIX D-F2 (registration note acceptable) |
| TRUNC C_ref not restricted to C / PB | E11.10 | FIX A-F1 |
| E11.10 M full-history value not printed | E11.10 "always reported alongside" | FIX A-F1 |
| WARM rows labelled COLD_START row_status | E11.1a "never … reported as cold start" | FIX B-F3 |

INFERENCE: once A-F1 through A-F9 are done, I see no remaining path by which a K1-K16 number could be computed wrong without a refusal. The biggest residual risk is operational: the ANA-02 index builder must set rules, subsets and blocks correctly. A-F3, A-F5, A-F7 and A-F10 are the defence in depth against that.

## Findings table
| finding | severity | file:line | fix | who |
|---|---|---|---|---|
| A-F1 trunc field names, sub-panel collision, no C_ref | MAJOR | retention_report.py:1274-1314; launch_trunc.py:212-222 | read trunc_* fields; refuse dup keys; C_ref = C_BIG; key check; print M FULL | Opus (ANA-02) |
| A-F2 K16 device-class impossibility; no rows producer | MAJOR | retention_report.py:87,123,425,565-610,998; bench.py:355-365 | ONNX_CPU sub-block; rows-free ranks entries + BENCH_RESULT sha; fallback = DESCRIPTIVE | Opus (ANA-02) + Sonnet (DEV bench FP32 ranks) |
| A-F3 RULE_FREE non-baseline crashes K11 | MAJOR | retention_report.py:414-422,741,855 | RULE_FREE only for baselines; Book.pairs → NOT_AVAILABLE | Sonnet (ANA-02) |
| A-F4 registered seed sets | MAJOR | retention_report.py:861-872,992-1014 | REGISTERED_SEEDS map; INCOMPLETE_SEEDS flag | Sonnet (ANA-02) |
| A-F5 narrow comparison key; pinned-status check | MAJOR | retention_report.py:87,565-610; rows.py:26-29,113-114 | full E9 key + intended-difference list; MEASURED_PINNED_DEVICE + one pin sha | Opus (ANA-02) |
| A-F6 rank coverage (-1 on rankable) | MAJOR | retention_report.py:641-646; bench.py:173 | assert ranks[view.rankable] >= 1 | Sonnet (ANA-02) |
| A-F7 arm identity (C_FULL vs C_BIG) | MAJOR | retention_report.py:452-479; trainer.py:543,574-576 | check train_manifest_id / method per arm | Sonnet (ANA-02) |
| A-F8 TEST gate + TEST registry loader | MAJOR | retention_report.py:317-345,1753-1775; manifests.py:98-105 | lock + register + ext registry + role map + same-script VALIDATION_RESULT | Opus (ANA-02, loader after DATA-09) |
| A-F9 06D §3-§5 rule map / hierarchy / flag | MAJOR | retention_report.py:121,926-944,992-1014,524-541,949,973,1504,1569-1603 | as A.8 | Opus (ANA-02) |
| A-F10 rule vs efe/checkpoint_id | MINOR | retention_report.py:452-479 | CB ⇒ efe 6.0; C_BIG PB ⇒ best@3.0; FL PB ⇒ best mark | Sonnet |
| A-F11 check-only skips pairing / E8.3 | MINOR | retention_report.py:726-754,1765 | plan_pairs in check-only | Sonnet |
| A-F12 sanity trigger rule | MINOR | retention_report.py:1426-1432 | C_SMALL CB vs C_BIG PB | Sonnet |
| A-F13 smoke / screen / non-DONE evidence | MINOR | retention_report.py:524-562 | refuse status ≠ DONE_MAIN, smoke/screen labels | Sonnet |
| A-F14 g2_run sha + lock-registry check | MINOR | retention_report.py:269-284,317-345 | record + verify; replicate g2_run.py:212-213 | Sonnet |
| A-F15 n_decisions; bootstrap_W_sha256 | MINOR | retention_report.py:703-707 | add n_decisions; compare W sha | Sonnet |
| A-F16 strata source manifest | MINOR | retention_report.py:1186-1206 | check source_manifest_id | Sonnet |
| A-F17 historical R seed | MINOR | retention_report.py:1443-1446 | seed mean or print seed | Sonnet |
| B-F1 smoke T41 receipt accepted | MAJOR | launch_trunc.py:185-209,449 | refuse smoke; n_compared + panel sha | Sonnet |
| B-F2 panel not re-derived | MAJOR | launch_trunc.py:359; panel.py:260 | derive + set-equality refusal | Opus |
| B-F3 WARM row_status | MAJOR | launch_trunc.py:241 | row_status = comparison_status | Sonnet |
| B-F4 COLUMN_RULES unused | MINOR | builder.py; rules.py:3-4; events.py:185-186 | import-time equality assert | Sonnet |
| B-F5 weak NCs | MINOR | test_studies.py:155-157; test_t41_golden.py:76 | stronger asserts | Sonnet |
| B-F6 run_all basetemp | MINOR | tests_trunc/run_all.py:84 | mkdir parents + TMP/TEMP | Sonnet (ANA-02) |
| B-F7 E2X C_ref arm; OOV count | MINOR | launch_trunc.py:133-158 | refuse non-C C_ref; add counts | Sonnet |
| C-F1 ext band content | MINOR | d8_registry.py:79-106 | assert band + manifest population fields | Sonnet |
| C-F2 launch args | MINOR | c_full.py:77-94 | refuse listed flags | Sonnet |
| D-F1 OOV crash in INT8 ranks | MAJOR | bench.py:166-184; ranking.py:56-62 | skip target < 0; n_oov_skipped; test | Sonnet |
| D-F2 first-users order | MAJOR | dump_rows.py:62-73; manifests.py:528-533,667-670 | registration note (a) or v2 sort (b) | Ops / Sonnet |
| D-F3 browser SRI / threads / sha tie | MINOR | index.html:15; bench.js:153-155 | as D | Sonnet |

**VERDICT: ACCEPT_WITH_FIXES.** Implement every MAJOR fix before ANA_SCRIPT_FREEZE, before any INT8 bench run with `--rows`, and before any Study A/B launch. MINORs go in the same pass where cheap. There are no BLOCKERs: nothing has been read, and every defect found either refuses loudly or is fixable before the first read.

## Tests run (MEASURED; TMP/TEMP and --basetemp under `_pytest_tmp/REV-ANA-01/`; venv python)
- `python -B -m pytest code/tests_ana -p no:cacheprovider -q --basetemp …/ana/bt1` gave **39 passed, 7 xfailed** (28.6 s).
- `… code/tests_ana -k test_nc_ --runxfail --basetemp …/ana/bt2` gave **7 failed** (all 7 NCs fail as intended), 39 deselected.
- `… code/tests_trunc -p no:cacheprovider -q -rs --basetemp …/trunc/bt` gave **35 passed, 9 xfailed, 0 skipped** (the real bucket-0 data is present) (sub-review).
- `… code/tests_trunc -k test_nc_ --runxfail` gave **9 failed with AssertionError**, 35 deselected (sub-review).
- `… code/tests_data08 -p no:cacheprovider -q --basetemp …/d8dev/bt1` gave **21 passed, 8 skipped (server-only real data), 12 xfailed** (sub-review).
- `… code/tests_dev/test_dump_rows.py code/tests_dev/test_browser_page.py --basetemp …/d8dev/bt2` gave **28 passed, 4 xfailed** (sub-review).
- Probes (scratchpad, synthetic, anakit):
  - LO set to RULE_FREE gave `rc 2 … rule-free denominator cannot pair` (A-F3);
  - `trunc_curves` on `launch_trunc.trunc_labels`-shaped rows gave panel None, subpanel "", C_ref None, and the FIXED_20/FIXED_50 h=5 collision (A-F1).
