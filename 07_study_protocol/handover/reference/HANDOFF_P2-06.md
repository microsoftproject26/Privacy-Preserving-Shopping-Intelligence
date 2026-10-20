# P2-06 — Phase-2 selection tool (v4int/p2_select.py) — handoff (revised after REV-P2-06)

- Worker: Claude (Opus), 2026-10-03. Local code and local CPU tests only. No server, no git, no runs, no TEST data, no Codex.
- Binding spec: contracts/PREG2.md, REGISTERED by DL 411.
  - sha256 `be0561abe5ceec450d3a75bef49853a3c480d3847f75559dd94e9e0545df18a8`, re-measured.
  - Sections used: §4.2, §5.1–§5.9, §6.3 S8, §11 D-j.
  - The review: company/reviews/REV-P2-06.md (verdict FAIL). The fixes are in §0.
- Files. Both are new and owned by this packet; no other file was edited.
  - `code/v4int/p2_select.py`: sha256 `7a05afcd…c792f` (after the delta fixes; REV fixes version was `166786f6…5839`)
  - `code/tests_int/test_p2_select.py`: sha256 `a4d35239…4cfd` (REV fixes version `43202a02…f402`)
- Files read but not edited:
  - `v4int/lr_screen.py`: `1a391b1e…f8fd3`, unchanged.
  - `v4int/launch_fl.py`: P2-01+02 is still editing it. It was `c9fdc8c0` at the first hand-off, `cd8d7216` at the review, and `c6192343…f1f2a` at this run. The tests use its live P2 functions.
  - `v4int/central_ft.py`: P2-05, `f392121d…d510`.
  - Every selection file records the launch_fl / central_ft / p2_select / lr_screen sha256 at the time it is written. The P2 lock must bind the final ones.

## 00. Delta fixes (REV-P2-06 delta review = PASS_WITH_MINORS; Ops checks the diff)
- New shas: p2_select.py `7a05afcd91337c2cfc1a7a124844c82f92712c71233142a07eb914e0c26c792f`; test_p2_select.py
  `a4d352394fdbeb16eabd38d1efdc1697b25188b6c0ce9717a4ab98036ba54cfd`. Unchanged: lr_screen `1a391b1e…f8fd3`, central_ft
  `4d95d12c…4077` (frozen, NOT edited), launch_fl `c6192343…f1f2a`.
- **N1**: R9 is now a set comparison like read_run: `READ_RUN_LABEL_KEYS` (lr_screen.py:366-369, verbatim; a test checks
  each key against the lr_screen source) + `p2_item_lr_mult`; a key of that set present in the row but not in the
  expected labels is refused. Test: fl_server_tau / fl_warm_lr_factor / fl_local_batch / fl_warm_server_lr_factor /
  p2_item_lr_mult each refused; NC `test_nc_r9_one_directional_accepts_extra_label`.
- **N2**: R12 requires `schedule_mark` to be finite (missing / null -> refused, as read_run's float() refuses) and
  `best_half_efe` an int. Test: missing / null / wrong refused; NC `test_nc_nan_schedule_mark_passes_tolerance_check`.
- **Open item O-A (diverged C1)**: handled in p2_select (`c1_nonfinite_record`), central_ft untouched. When
  `central_ft.read_c1_run` refuses, the run is recorded EXCLUDED with its reason iff its value-free RUN_START identity
  verifies (central_ft's own checks repeated: arm / variant = run id, role CALIBRATION, CALIB eval manifest, seed 2026)
  AND either (a) STATUS.json FAILED with "TrainingDiverged: …" and no DONE_MAIN summary, or (b) a DONE_MAIN summary with
  a non-finite practical_best value (strict JSON refused by central_ft). Any other refusal still refuses. The candidate
  carries `c1_nonfinite_path` {kind DIVERGED | NON_FINITE, the central_ft refusal text, STATUS sha}. Tests: a diverged
  cell is excluded with its reason in P2_C1_SELECTION.json and the file verifies; NON_FINITE path; DiskGuardError and
  MAIN-role failures refused; NC `test_nc_central_ft_alone_records_diverged_c1`.
- **Tests**: test_p2_select.py **25 passed + 4 xfailed** (3.8 s), no monkeypatch; with `--runxfail` all 4 NCs fail
  with AssertionError, as required.

## 0. REV fixes (REV-P2-06 → this revision)

| finding | fix |
|---|---|
| **M1** anchor flag 0.0005 | `ANCHOR_MIN_GAIN = 0.001`. NO_MEASURABLE_CALIB_GAIN iff CALIB_GAIN < 0.001, strict and exact (PREG2 §5.6, Fable §10). Tests: a 0.0006 gain is flagged; a gain of exactly 0.001 is not flagged; 0.001 − 1e-12 is flagged; 0.001 + 1e-12 is not flagged; the winner is carried in every case. |
| **M2** read_run refuses real P2 specs; the tests hid it with a monkeypatch | `read_p2_run` now re-implements lr_screen.read_run's checks **one by one (R1–R12, listed in the module docstring with read_run line refs)**. lr_screen stays unchanged. The two Phase-1-specific checks are replaced by their launch_fl P2 counterparts: `check_p2_spec` replaces spec_tuning, check_warm_spec and the lr_peak_local check; `warm_row_labels + p2_row_labels` with the `fl_rows` CALIB relabel replaces `row_labels(t)`. **The value-free ARM_SPEC.json is verified before any value-bearing file is opened (§6.3 S8)**: digest, P2 arm, role CALIBRATION, seed 2026, `check_p2_spec`. **The monkeypatch is removed.** Every P2 fixture is built with the real `F.p2_arm_spec` and real `F.fl_rows` rows, and read by the real reader. A test also proves that `lr_screen.read_run` still refuses these runs, which is why the dedicated reader exists. C1 reading is delegated to the P2-05 verifier `central_ft.read_c1_run` (m8). |
| **M3** a NaN in D1-ext blocked the file | Every run-derived payload is sanitised by `_finite_json`: NaN / ±inf are stored as "NaN" / "Infinity" / "-Infinity". A run with a non-finite **selection value** is excluded and recorded. A run with finite selection value but non-finite **diagnostics** (D1-ext, other marks) is FLAGGED: `nonfinite_fields` on the candidate and top-level `nonfinite_diagnostics`. It is still ranked, because D1-ext never selects (§3.3). `max_of_12_marks` is taken over finite marks. Test: one NaN-valued run whose D1-ext carries NaN, plus a finite run with D1-ext = inf and a NaN at mark 6.0. The file is written and verifies. |
| m1 corner run | The registered §5.7 no longer names a corner ("At most 3 extension runs per stage" only), so **the corner run is removed**. A WA-core winner now gets at most 2 extension cells (server axis at c*, client axis at η*). The ≤ 3 cap is still enforced. |
| m2 tertiary order | Drops now follow **one strict order**, as registered in §5.9: P2_WSTAR_DP1 → P2_WSTAR_SECAGG (MAIN) → P2_WSTAR_DPFTRL8 s2027 → WC3/WC4 → WB2 → WC1/WC2 → WB3 → C2. `check_dropped` accepts only a prefix. Part-B arms unfinished at lock time are declared not run under §2.1 in the P2 lock, not by this tool. |
| m3 cap plan vs check_dropped | `cap_drop_plan` requires the savings of every unit (0 if not planned) and drops units strictly in order while the total exceeds 250. Its output always passes `check_dropped` (tested over several projections). |
| m4 drops unbound | FINAL `--dropped` now requires `--cap-plan` (format `PPSI_V4_P2_CAP_PLAN_1`: projected_gpu_h + savings), and the drops must equal the plan's. The plan is recorded in the file, including its sha. |
| **m5** float boundaries | The tie band (≤ 0.001, inclusive), futility (< 0.001, strict) and the anchor flag (< 0.001, strict) are now computed in exact decimal arithmetic (Fraction of each value's shortest repr). Boundary tests sit at exactly 0.001 and at 0.001 ± 1e-12. Example: 0.1716 vs 0.1706 is tied, although float gives a gap of 0.0010000000000000009. |
| m6 authority text | The tool cites contracts/PREG2.md (REGISTERED, sha, DL 411) and no longer cites the draft or "GATE §9". The futility caveat is D-j, verbatim. |
| m7 incumbent guard | The P2-T3 receipt must be a JSON object with: `test` "P2-T3", `verdict` "BIT_IDENTICAL", `rounds` 64, `s8tc_run_id` = the run read, and `result_rows_sha256` / `run_summary_sha256` equal to that run's. The test checks that `"{}"` is refused. The Phase-1 incumbent must also equal `F.p2_arm_spec` of the incumbent recipe (built at the P2 runs' θ_pre / seed / role / peak) on cfg (whole), plan, max_rounds, frozen_item_tables, init_from, exposure_budget and role. Tests cover another θ_pre, passes 2 and server factor 1, and all are refused. |
| m8 S8 order | FL: ARM_SPEC is checked first (above). C1: delegated to `central_ft.read_c1_run`, which checks the RUN_START identity first. |
| m9 extra UNRESOLVED readings | The registered §5.7 text is followed literally: **EDGE_UNRESOLVED iff the union winner is a registered extension cell** ("again on the new edge"). A non-finite extension cell (new or existing) and a winner that moved to another base run are now recorded only as edge `notes`; they never change the status. |
| m10 verifier | `load_p2_selection` now also re-derives `calib_gain`, the anchor flag, `flags`, and the registered required set equal to the base candidates. For FINAL it also re-derives the futility decision and the drop prefix. Malformed files are refused (P2SelectionRefused), not crashed on. |
| m11 | Exit codes: `LaunchRefused` / `CentralFTRefused` → exit 2, edge runs missing → exit 3. FINAL under futility with the Phase-1 incumbent as A* records the WA file's `common` (θ_pre). `--runs` may then be empty (CLI tested). §5a now holds the after-runs. |
| review §4 | Cross-module test: `P.p2_label == F.p2_label` over every WA, edge and FINAL cell; `RECIPE_KEYS == F.P2_RECIPE_KEYS`; `P2_ARMS == F.P2_ARMS`; every WA and edge cell passes `F.check_p2_recipe`. `P2_COMMON_FIELDS`: `knobs` removed; added study, cfg.lr_shape, solver weight_decay, lr_peak_base, exposure_budget and p2_fixed. The adapter cross-checks `RUN_SUMMARY.p2.{label, recipe}` and `spec.p2_label`. |

## 1. Design (current)
- **Pure core:**
  - `p2_recipe` / `p2_label` (= launch_fl's)
  - `p2_candidates`
  - `p2_order_key`
  - `p2_decide(stage, runs, anchor)` (§5.6)
  - `edge_extension(stage, winner)` (the §5.7 table, no corner)
  - `stage_outcome` (§5.6 + §5.7)
  - `futility` (§5.8)
  - `check_dropped` / `cap_drop_plan` / `read_cap_plan` (§5.9)
- **Readers:**
  - `read_p2_run` (R1–R12) → **`p2_run_adapter` = the single alignment point** for P2 recipe, bytes and D1-ext fields;
  - `read_phase1_incumbent` (unedited lr_screen.read_run + receipt + cfg identity);
  - `read_c1_run` (central_ft delegate);
  - `read_anchor` → `anchor_adapter`.
- **Writers:** O_EXCL via `write_exclusive`; `self_sha256 = launch_fl._canon`; file text `json.dumps(indent=1, sort_keys=True)`.
- **Verifier:** `load_p2_selection`. launch_fl `bind_p2_selection` and central_ft `bind_c1_selection` call it.
- **CLI:**
  - `select --stage WA|FINAL|C1`
  - `futility` (the decision Ops needs before RUN-2)
  - Output: only labels, flags and the sha are printed.
  - Exit codes: 0 written, 2 refused, 3 edge runs missing.
- **Order of use:**
  1. anchor
  2. C1
  3. WA (exit 3 until the edge runs exist)
  4. futility
  5. RUN-2 if CONTINUE
  6. FINAL (with `--cap-plan` if anything is dropped)

## 2. Fields the tool depends on (check the P2-01+02 / P2-03 / P2-04 / P2-05 / P2-07 handoffs against this list)
- **P2 run (launch_fl, P2-02):**
  - ARM_SPEC.json `{arm, arm_spec_sha256, arm_spec, device}`.
  - `arm_spec`:
    - arm `P2_W`, role, `cfg.seed` (R1–R3);
    - `check_p2_spec(arm_spec)` must pass;
    - `p2_recipe` = exactly `P2_RECIPE_KEYS`; `p2_label`;
    - `cfg.solver.lr` = the client LR applied;
    - P2_COMMON_FIELDS: role, study, cfg.seed, plan, max_rounds, cfg.endpoint_rounds, cfg.dropout_p, cfg.endpoint_efe, cfg.lr_shape, cfg.solver.{passes, batch_size, clip, weight_decay}, lr_peak_base, exposure_budget, p2_fixed, init_from.
  - RUN_SUMMARY: `status`, `arm_spec`, `arm_spec_sha256`, `evals` [[u, ·, value] × 12], `best_half_efe`, `bytes.per_visit.{upload_bytes, shared_numel}`, `update_norm_log`, `p2.{label, recipe}`.
  - RESULT_ROWS: `run_id` (the receipt's s8tc_run_id for S8), and exactly one PRACTICAL_BEST E2E macro mrr@20 row produced by `fl_rows(role CALIBRATION, lr_mult 1, tuning_labels = warm_row_labels + p2_row_labels)`, carrying `eval_manifest_id`, `split`, `value`, `value_status` and `schedule_mark`.
- **P2-03 (WB3):** `bytes.per_visit.upload_bytes` must carry the per-visit upload payload (a mean, if it varies), and `check_p2_recipe` must admit `upload: delta_rows`.
- **P2-04 (LoRA):** `cfg.solver.lr` = c × lora_lr_factor; `shared_numel` = LoRA parameters only; `check_p2_recipe` must admit lora_rank / lora_lr_factor.
- **P2-05 (C1):** `central_ft.read_c1_run` output: run_dir, run_id, arm, variant, role, peak_lr, seed, trained_numel, finite, value, mark, result_rows_sha256, run_start_sha256.
- **P2-07 (anchor):** not written yet. Assumed: `<out>/RESULT_ROWS.json {role, rows}` with one `method P2_ANCHOR_PRE_CALIB` E2E macro mrr@20 row on CALIB, plus `REEVAL_RECORD.json`. That is `anchor_adapter`, one edit.
- **P2-T3 receipt (P2-02):** the keys listed in §0, m7.

## 3. Rule implemented vs PREG2 (REGISTERED)
| item | registered | implemented |
|---|---|---|
| metric / mark 0 | §5.6, §5.1 | R6 + R12 (FL marks 1..12), central_ft (C1 mark ≥ 0.5) |
| candidates | §5.2–§5.4, §4.2 | WA 21; FINAL A* + WB1–3 + WC1–5 minus drops ({A*} under futility); C1 2 × 4. Missing required runs and unregistered runs are refused |
| tie band | 0.001 | ≤ 0.001, exact, inclusive |
| preference | bytes → trained params → server step → client LR → label; C1: then the smaller LR | `p2_order_key`, exact rationals |
| anchor flag | < 0.001; W* still carried | < 0.001, exact, strict |
| non-finite | excluded and recorded | selection value → excluded; diagnostics → flagged, never blocking |
| edge | §5.7 table; ≤ 3 runs; re-applied; UNRESOLVED when again on the new edge | table verbatim, no corner; status literal; other facts in `notes` |
| futility | both gains < 0.001; D-j | exact, strict; D-j verbatim |
| cap | 250; DP1 → SECAGG → DPFTRL8 s2027 → WC3/WC4 → WB2 → WC1/WC2 → WB3 → C2 | strict prefix, bound to `--cap-plan` |
| CALIB only | F3, §6.3 S8 | CALIB_VALIDATION_5_100_32768 allowlist. Role CALIBRATION is checked before any value file is opened (FL: ARM_SPEC; C1: RUN_START) |

## 4. Tests and counts (CPU; `--basetemp` on F:, because C: has about 2.4 GB free and the FL checkpoint disk guard refuses temp files there)
- **`tests_int/test_p2_select.py` (revised, no monkeypatch): 22 passed + 1 xfailed** (5.1 s).
  - With `--runxfail`, the negative control fails with an AssertionError, as required.
  - Coverage:
    - registered sets and launch_fl alignment;
    - exact server step;
    - preference order;
    - tie band boundary (exact, ±1e-12);
    - anchor flag at 0.001 with boundaries;
    - non-finite value rules;
    - step table without a corner;
    - extension statuses and notes;
    - an existing extension cell, and a moved-winner note;
    - FINAL WB2 / LoRA edges (pure records: WB3 and LoRA runs cannot be launched until P2-03 / P2-04);
    - futility boundaries;
    - the strict drop order, and a cap plan that is always a prefix;
    - a real P2 run read without monkeypatch (lr_screen.read_run refuses the same run);
    - WA write-once, tamper refusal, and flag re-derivation;
    - refusals: Phase-1 label, missing run, MAIN / TEST manifest, MAIN role, hand-edited spec, foreign anchor;
    - M3: a NaN run and NaN diagnostics;
    - WA edge with the CLI exit codes (3 / 0 / 2), with no value printed;
    - Phase-1 incumbent: receipt fields, cfg, θ_pre;
    - C1 per variant: edge resolved and unresolved, NaN exclusion, mark 0, MAIN / TEST / role refused;
    - FINAL with drops bound to a cap plan;
    - FINAL under futility with the no-gain flag;
    - FINAL with the Phase-1 incumbent as A* under futility (CLI with no P2 run).
- §5a below holds the lr_screen-importing suites.

### 5a. After-runs (lr_screen-importing tests_int suites: fl05, fl09, fl10, fl12, fl14, fl15, fl16, fix_delta01_fl_binding, fix_delta02)
- **Before (first hand-off, one session):** 116 passed, 49 xfailed, 12 errors. The errors were MemoryErrors from host memory pressure in test_fl15's real-FL fixture.
- **After the first hand-off:** a single session aborted (`INTERNALERROR MemoryError`; the host's commit charge is close to its limit with C: full). The same suites were then run **file by file** against launch_fl at that time: **121 passed, 54 xfailed, 0 errors**, plus test_p2_select (old) 17 passed + 1 xfailed.
- **After this revision (file by file, launch_fl `c6192343…f1f2a`): 122 passed, 55 xfailed, 0 errors.** Per file:

| suite | passed | xfailed |
|---|---|---|
| fl05 | 4 | 1 |
| fl09 | 7 | 4 |
| fl10 | 22 | 6 |
| fl12 | 14 | 3 |
| fl14 | 20 | 6 |
| fl15 | 22 | 16 |
| fl16 | 14 | 7 |
| fix_delta01_fl_binding | 6 | 3 |
| fix_delta02 | 13 | 9 |

  - test_p2_select: 22 passed + 1 xfailed.
  - Since the previous run, fl05 has gained one test (P2-01+02's launch_fl work). lr_screen is byte-unchanged.

## 5. Open questions (for the delta review / Ops)
- **Q2.** The item-LR multiplier is not in the §5.6 tie order. WB2 (×0.01) tied with its ×0.001 extension is decided by the lexical label, so the extension wins → EDGE_UNRESOLVED (tested). Should (4) include the item-table LR?
- **Q3.** "Fewer trained parameters" for WC5 is the ordinal (uploaded numel, private-head flag).
- **Q5.** (3) compares FedAdam η_s numerically with FedAvg / FedAvgM steps (literal, as registered). FedAdam therefore wins every cross-optimiser tie (review O1).
- **Q7.** The WA incumbent and WA-ext 1e-4 have no registered step, so they get no extension.
- **Q8.** Futility uses the raw best values (edge runs and incumbent included).
- **Q10 (new).** A diverged **C1** run never reaches DONE_MAIN, because the central trainer raises TrainingDiverged. The selection then refuses that cell as missing, so the C1 file can be blocked. Possible handling: an Ops ruling (re-run, or a declared-diverged record). A C1 row with value NaN and a null `practical_best.value` is excluded (tested). central_ft's strict JSON refuses a NaN inside RUN_SUMMARY.
- **Q11 (new).** Edge notes such as "extension non-finite" have no status effect under the literal §5.7. Should Ops / Fable want them disclosed beside the Q results, the P2 report should print `edge.notes`.
