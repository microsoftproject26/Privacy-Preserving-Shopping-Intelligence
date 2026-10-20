# TEST_EVIDENCE_B.md -- Sonnet B independent tests, SASREC_FEATURE_OPT_V1

Local only: no server, no `tools/lobot.py`, no network, no git, no pip. Every write went through
`FO_ROOT_OVERRIDE` / `FO_CAMPAIGN_ROOT` pytest tmp dirs (via the `set_fo_root` / `set_fo_campaign_root`
fixtures in `conftest_fo.py`); nothing was written into the real study folders. No code under test
was modified.

## Files

- `code/tests/conftest_fo.py` -- shared fixtures (sys.path/env setup, `set_fo_root`,
  `set_fo_campaign_root`, tiny rich model builder, `SyntheticStore` builder).
- `code/tests/test_fo_env.py` -- fo_env.py (isolation, path guards). 12 tests.
- `code/tests/test_fo_feature_view.py` -- feature_view.py (views, probes, lean fold, M1, calendar,
  D3, no-label leakage, MEASURED_AFFINE_SHIFT cross-check). 29 tests.
- `code/tests/test_fo_models.py` -- fo_models.py (parameter_mapping, D3 zero-init, strict-load
  view refusal). 6 tests.
- `code/tests/test_fo_eval.py` -- fo_eval.py (OOV zero-credit denominator, write_once /
  write_json_once). 8 tests.
- `code/tests/test_fo_stream_fit.py` -- stream_fit.py (fit-on-VS trap). 2 tests.
- `code/tests/test_fo_screen.py` -- screen.py (manifest, schedule, train_arm, main() label_smoothing
  guard). 16 tests.
- `code/tests/test_fo_pin_baseline.py` -- pin_baseline.py (choose, noise_reference, candidates). 8 tests.
- `code/tests/test_fo_finalize.py` -- finalize_fo.py (hygiene_parity, gate 2(b) end-to-end on
  fabricated inputs). 13 tests. **NEW (Fable Review C request, 2026-09-25)**.
- `code/tests/test_fo_probe_runner.py` -- probe_runner.py (gate_2a, encoding_check /
  ENCODING_MIN_ROWS). 10 tests. **NEW (Fable Review C request)**.
- `code/tests/test_fo_queue.py` -- fo_queue.py (gpu_busy fail-closed, ready() gating). 10 tests.
  **NEW (Fable Review C request)**.

## Result

**UPDATE (2026-09-25, round 1):** the lead fixed both defects reported below (`fo_env.study_path`
now uses `Path.is_relative_to`; `fo_env.write_json_once` serialises before creating the file, and
`fo_eval.write_once` / `stream_fit.py` / `pin_baseline.py` / `finalize_fo.py` now use it). Per the
coordinator's instruction, the two defect-documenting tests were converted in place into
regression tests (see "Before / after" below) rather than kept failing.

**UPDATE (2026-09-25, round 2):** Fable Review C requested additional independent tests after the
lead's other Review C fixes (`finalize_fo.hygiene_parity` gating, `probe_runner.ENCODING_MIN_ROWS`
+ idempotent `gate_2a()`, `fo_queue.gpu_busy` fail-closed behavior, `screen.py`'s
label_smoothing guard, `feature_view.MEASURED_AFFINE_SHIFT` cross-check). Three new files were
added (`test_fo_finalize.py`, `test_fo_probe_runner.py`, `test_fo_queue.py`) and two existing files
were extended (`test_fo_feature_view.py`, `test_fo_screen.py`). No defects were found in this round;
every new behavior matched its specification, including all requested negative controls.

Current: `pytest code/tests/test_fo_*.py -q` -> **114 tests collected, 114 passed, exit code 0**.
Combined with the pre-existing `test_profile_train_sample.py`:
`pytest code/tests/test_fo_*.py code/tests/test_profile_train_sample.py -q` -> **125 tests, 125
passed, exit code 0**.

History, for the record:
- Round 0 (initial submission): 75 tests collected, 73 passed, 2 failed (deliberately: they
  documented the two defects below), exit code 1.
- Round 1 (after lead's fixes, defect tests converted to regression tests): 77 tests / 88 combined,
  all passed, exit code 0.
- Round 2 (this update, Fable Review C's additional tests): 114 tests / 125 combined, all passed,
  exit code 0.

## Before / after: the two converted tests

1. `code/tests/test_fo_env.py` -- `test_study_path_DEFECT_sibling_name_prefix_escapes_the_guard`
   (asserted the escape was let through) was replaced by
   `test_study_path_rejects_sibling_name_prefix` (asserts `PermissionError` is now raised for the
   sibling-prefix path, with two negative controls: a genuine in-root path, and an in-root path
   whose final component contains the substring ".." without being a real ".." segment, are both
   still accepted).
2. `code/tests/test_fo_eval.py` -- `test_write_once_raises_on_nan` and
   `test_write_once_DEFECT_nan_rejection_leaves_a_stray_empty_file` (the latter asserted the stray
   file existed) were replaced by
   `test_write_json_once_nan_leaves_no_file_and_retry_succeeds` (parametrized over both
   `fo_eval.write_once` and `fo_env.write_json_once`: NaN raises `ValueError`, no file is left
   behind, and a retry with valid data at the same path now succeeds) plus a negative control
   `test_write_json_once_negative_control_overwrite_still_refused` (an overwrite attempt on an
   existing valid file still raises `FileExistsError` and leaves the original content untouched,
   for both entry points).

Both converted tests carry a dated docstring note: "2026-09-25: converted from defect-documenting
test after lead fix".

## Round 2 (Fable Review C additional tests, 2026-09-25)

| # | Item | Test file | Status |
|---|------|-----------|--------|
| 1 | `finalize_fo.main()` end-to-end on fabricated inputs: gate 2(b) threshold (both sigma_hat branches), CI lower>0, R@20 guard, at-most-one finalist (larger delta wins), decision KEEP_CURRENT/PAIR_REQUIRED, cohort decision_id-mismatch raises | test_fo_finalize.py | pass (7 tests) |
| 2 | `finalize_fo.hygiene_parity`: PARITY_OK baseline + one negative control per condition (label, counts, rotation residual, max\|dlogit\|, rank-identical), missing LEAN_PARITY.json raises | test_fo_finalize.py | pass (6 tests) |
| 3 | `probe_runner.gate_2a`: P1 excluded even if positive, each of the 3 sub-conditions individually required, negative deltas never candidates, largest-delta selection + negative control (all fail -> no candidate) | test_fo_probe_runner.py | pass (7 tests) |
| 4 | `probe_runner.encoding_check` / `ENCODING_MIN_ROWS`: tiny random store -> "unverified" + ok False; hand-built internally-consistent fixture with floor lowered -> ok True; single injected violation -> ok False | test_fo_probe_runner.py | pass (3 tests) |
| 5 | `fo_queue.gpu_busy` fail-closed (nonzero rc / exception / unparsable line / unmapped bus -> every GPU busy) + negative control (normal mapping -> only the right GPU busy); `fo_queue.ready()` (FINAL_STAGE_DONE absent / campaign alive / disk below floor+peak -> not ready) + negative control (everything satisfied -> ready, and the requested GPU itself busy -> not ready) | test_fo_queue.py | pass (10 tests) |
| 6 | `screen.main()`'s label_smoothing guard raises RuntimeError before training (heavy deps monkeypatched: fake TRAIN/VS store, real tiny cohort parquet, sentinel marker past the guard) + negative control (label_smoothing=0.0 passes through to the marker, not the guard) | test_fo_screen.py | pass (2 tests) |
| 7 | `feature_view.fold_lean_state_dict` raises when `MEASURED_AFFINE_SHIFT` disagrees with the clip-inferred shift (monkeypatched) + negative control (real dict does not raise) | test_fo_feature_view.py | pass (2 tests) |

Round-2 negative controls, with observed numbers:
- gate 2(b) threshold: sigma_hat=0.0005 -> thr=0.002 (GATE_B_MIN dominates); sigma_hat=0.003 ->
  thr=0.006 (2*sigma_hat dominates); both verified via `fin["gate_2b_threshold"]`.
- "just below threshold" control: delta=0.0015 < thr=0.002 -> `delta_ge_threshold` False, the
  other two gate_2b conditions True -> decision KEEP_CURRENT (isolates the one failing condition).
- "above threshold but fails R@20 guard" control: delta=0.02 >= thr, dr=-0.006 < -0.001 ->
  `recall_guard` False only -> decision KEEP_CURRENT.
- two passing arms (deltas 0.03 and 0.05): finalist = the 0.05 arm.
- cohort decision_id mismatch (one arm's ranks reordered): raises `RuntimeError: ... cohort order
  mismatch`.
- hygiene_parity: each of the 5 negative controls breaks exactly one `checks[...]` condition to
  False while every other condition stays True (asserted explicitly), confirming isolation.
- gate_2a: probe with delta=0.0009 (just under 0.001) excluded; CI lower=-0.001 excluded; weekend
  slice=-0.001 excluded; picks 0.03 over 0.01/0.02 while excluding P1 at 0.999.
- encoding_check: tiny synthetic VS split (n_vs=64) leaves every one of the 10 states unverified
  under the real ENCODING_MIN_ROWS=100 floor; the hand-built fixture (floor lowered to 1) reaches
  count>=1 on all 10 states with zero violations -> ok True; a single flipped `price_zero` flag
  with the same lowered floor -> ok False, violation message names `price_missing_events`.
- fo_queue.gpu_busy: all four fail-closed scenarios return `{"0": [...], "1": [...]}` marking both
  GPUs busy; the normal-mapping control narrows busy-ness to exactly GPU 0 (`busy["1"] == []`).
- fo_queue.ready: each of the three false conditions produces a distinct reason string
  ("FINAL_STAGE_DONE absent", "campaign processes alive", "disk free ... < floor"); the
  everything-satisfied control returns `(True, "ok")`, and requesting the one busy GPU flips it
  back to False with a "compute processes" reason.
- screen.main() guard: label_smoothing=0.1 -> `RuntimeError` mentioning "label_smoothing" raised
  before `fm.load_weights_file` is ever called; label_smoothing=0.0 -> the sentinel
  `_PastGuardMarker` fires instead, proving the guard specifically did not trip.
- fold_lean_state_dict: MEASURED_AFFINE_SHIFT tampered by +999.0 -> `ValueError` mentioning
  "measured shift"; the real (untampered) dict -> no exception.

No defects were found in this round.

## Package-list item coverage (task packet numbering)

| # | Item | Test file | Status |
|---|------|-----------|--------|
| 2 | Identity view / lean fold parity + neg. control | test_fo_feature_view.py | pass (max\|dlogit\|=1.9e-7 vs neg-control >1e-2) |
| 3 | Column order vs FEATURE_GROUPS.json + swap neg. control | test_fo_feature_view.py | pass |
| 4 | Poisoned PAD (all views + all probes), D3=0 at PAD | test_fo_feature_view.py | pass |
| 5/7 | M1 re-encoding (available/unavailable/z0-boundary/no-double-log) | test_fo_feature_view.py | pass |
| 6 | Fit-on-VS trap + neg. control (VS extreme values never move TRAIN fit) | test_fo_stream_fit.py | pass |
| 8 | Degenerate std raises | test_fo_feature_view.py | pass |
| 9 | Calendar (sin²+cos²=1, local=utc+3h, weekday/is_weekend 7×5, utc rollback) | test_fo_feature_view.py | pass |
| 10/16 | Probe closure (P4 last-position-only, P2 leaves category/has_history, dup flags move together) | test_fo_feature_view.py | pass |
| 11/12/14/15 | D3 (future-invariance, boundaries, OOV/MISSING never count, prior excludes current, hand-built example) | test_fo_feature_view.py | pass |
| 13 | No labels reach `inner.forward` (kwarg spy) | test_fo_feature_view.py | pass |
| 17 | parameter_mapping (d3/lean), D3 zero-init == control at init, strict-load view refusal | test_fo_models.py | pass |
| 18/19 | screen: manifest determinism/uniqueness/range/sha256@full-scale, n_updates, lr_at, train_arm coverage, CONTROL≈C1_D3 loss@u1 | test_fo_screen.py | pass |
| 20 | fo_eval.metrics OOV rank=-1 counted in denominator (hand computation) | test_fo_eval.py | pass |
| 22/24 | refuse_split, study_path, write_once, assert_isolated (+neg. control) | test_fo_env.py, test_fo_eval.py | pass (+2 defects, see below) |
| pin_baseline | choose() tie rule, noise_reference (+fallback), candidates() from fabricated summaries | test_fo_pin_baseline.py | pass |

## Negative controls (every critical test), with observed numbers

- Lean fold parity: real relation max\|dlogit\| = **1.937e-07**; negative control (+1.0 on
  `log1p_prefix_elapsed_s`) max\|dlogit\| = **0.2377** (assert > 1e-2).
- Column-order swap (swap `log1p_price` / `utc_hour_sin` columns): max\|dlogit\| after swap,
  observed > 1e-2 in the run (assert threshold 1e-2); breaks lean parity as required.
- M1 available-row-at-z0: confirmed re-encoded value != 0.0 (not accidentally zeroed).
- `m1_states_from_fit` degenerate std: raises `ValueError` as required.
- `stream_fit` fit-on-VS trap: TRAIN valid-only mean/std identical to 1e-12 across two runs where
  only the VS rows' value changed (5.0/6.0/7.0/100×3 -> valid_mean=6.0 both runs); the "all rows"
  statistic (includes clipped-garbage unavailable rows) moved by more than 1.0 away from the
  valid-only mean, proving valid-only really excludes them; the VS-only drift statistic itself DID
  move (by >1.0) between the two runs, proving the perturbation was actually applied.
- D3 zero-init vs CONTROL at init: max\|dlogit\| = **1.19e-07**; negative control (perturbing the
  D3 weight columns by +0.5) makes CONTROL vs C1_D3 first-update loss diverge by > 1e-3.
- `screen.manifest` sha256 at full scale (n_train=40,180,608) matches
  `config/candidate_registry.json["manifest_sha256"]` exactly; negative control (MANIFEST_SEED+1)
  produces a different array (`np.array_equal` is False).
- `choose()` tie rule: negative control swaps which seed has the earlier epoch and confirms the
  winner follows epoch order, not a hardcoded seed preference.
- `noise_reference` fallback: sigma_hat == 0.002, fallback_used == True when s2027 absent; with
  s2027 present, sigma_hat = max(|0.550-0.548|, |0.560-0.551|) = **0.009** (hand-computed, matches).
- `candidates()`: negative controls confirm a non-`COMPLETE` run and a missing weights file are
  both excluded from the candidate list.
- `assert_isolated`: negative control injects a fake `paths` module with a foreign `__file__` ->
  raises `RuntimeError: isolation breach`.
- `verify_snapshot`: negative control drifts a tracked file's content after hashing -> raises
  `RuntimeError: snapshot drift`.
- `fo_eval.metrics` OOV-in-denominator: hand-computed macro_mrr@20 = **0.3** (population of 4,
  including the unrankable OOV row with zero credit); negative control that instead excludes the
  OOV row from the population gives **0.4**, a materially different number, proving the hand
  computation is sensitive to the inclusion rule.
- `write_once` / `write_json_once`: negative control confirms an overwrite attempt on an existing
  file still raises `FileExistsError` and leaves the original content unchanged, for both
  `fo_eval.write_once` and `fo_env.write_json_once`.
- `study_path` sibling-prefix escape: now raises `PermissionError` (regression test); negative
  controls confirm a genuine in-root path, and an in-root path whose final component contains the
  substring ".." without a real ".." segment, are both still accepted.

## Defects found (2) -- FIXED by the lead on 2026-09-25

1. **`code/fo_env.py:74-77`** (`study_path`) -- the escape guard used to be a bare string
   `str(p).startswith(str(FO_ROOT.resolve()))` with no path-separator boundary check, letting a
   sibling directory whose *name* merely started with the root directory's name (e.g. root
   `.../tmp_root`, sibling `.../tmp_root_evil`) through as if it were inside `FO_ROOT`.
   **Fix**: `study_path` now checks `p.is_relative_to(FO_ROOT.resolve())` (component-wise).
   Regression test: `code/tests/test_fo_env.py::test_study_path_rejects_sibling_name_prefix`.

2. **`code/fo_eval.py:92-94`** (`write_once`) -- used to open the destination with
   `open(path, "x", ...)` (create, exclusive) *before* `json.dump(..., allow_nan=False)` validated
   the object, so a NaN-rejected write left a stray empty file behind that then permanently
   blocked any retry at the same path with `FileExistsError`.
   **Fix**: new `fo_env.write_json_once` serialises with `json.dumps(obj, allow_nan=False)` FIRST
   and only then opens the file exclusively; `fo_eval.write_once` now delegates to it, as do
   `stream_fit.py`, `pin_baseline.py` and `finalize_fo.py`.
   Regression test: `code/tests/test_fo_eval.py::test_write_json_once_nan_leaves_no_file_and_retry_succeeds`.

No other defects were found; all other required behaviors matched their specification exactly
(within float-rounding tolerances noted above).
