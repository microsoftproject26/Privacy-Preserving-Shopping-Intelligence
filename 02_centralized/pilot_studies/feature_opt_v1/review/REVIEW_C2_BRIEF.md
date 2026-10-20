# Review C, correction-only follow-up (≤ 600 output tokens)

Your pass 1 is saved verbatim in
`F:\Queens University\Microsoft Project\eda\.local_ai\feature_opt_v1\review\FABLE_CODE_REVIEW.md`
(FIX_REQUIRED). Check ONLY that each finding is fixed in the changed files. The hashes are in
`...\feature_opt_v1\review\REVIEW_C2_HASHES.json`, where `changed_since_pass1` lists them. Every
other file is unchanged from the pass-1 hashes you accepted.

## Fixes by finding (all files under `...\feature_opt_v1\code\`)
1. **HIGH, finalize parity claim.** `finalize_fo.py`:
   - `hygiene_parity()` requires a FULL_POPULATION label, every DUPLICATES_FULL count = 0, a
     rotation residual ≤ 1e-6, max|Δlogit| ≤ 1e-3 and rank-identical ≥ 0.999;
   - a missing LEAN_PARITY.json raises;
   - `TRANSFORM_STATE.status` is PARITY_FAILED_NOT_ADOPTED and `SELECTED_FEATURES.equivalence` is
     None unless PARITY_OK.
2. **HIGH, vacuous encoding check.** `probe_runner.py`:
   - `ENCODING_MIN_ROWS = 100`, and `unverified` lists each state below the floor; `ok` requires no
     violations AND no unverified states;
   - the check now runs over ALL VALIDATION_SELECT eval decisions, not just the cohort, so rarer
     states reach the floor;
   - otherwise it aborts with rc 5.
3. **MEDIUM, label smoothing.** `screen.py` raises before training if the registry's
   label_smoothing ≠ 0.0.
4. **MEDIUM, GPU check fail-open.** `fo_queue.gpu_busy()` is fail-closed: an rc ≠ 0, an exception,
   an unparsable line or an unmapped bus marks every GPU busy.
5. **MEDIUM, retry hazard.** `probe_runner.main` is idempotent per stage:
   - the encoding check is reused if present;
   - the probes are skipped only if GROUP_PROBE_RESULTS exists, in which case the baseline ranks
     are reloaded and verified by sha;
   - C2_DECISION is written if missing;
   - lean parity runs whenever LEAN_PARITY.json is missing;
   - M1_FIT.json is required up front.
6. **MEDIUM, missing finalize tests.** New `tests\test_fo_finalize.py` (13 tests), plus
   `test_fo_probe_runner.py` (10), `test_fo_queue.py` (10) and additions to `test_fo_screen.py` /
   `test_fo_feature_view.py`. Result: 125 tests, 125 pass, exit 0. Evidence and negative controls
   are in `...\review\TEST_EVIDENCE_B.md`.
7. **LOW.** `fo_env.assert_isolated` uses `is_relative_to`.
8. **LOW.** `feature_view.fold_lean_state_dict` cross-checks the clip-inferred shift against
   `MEASURED_AFFINE_SHIFT` and raises on a mismatch.
9. **Choice 4.** The D3 ratio definition is recorded in `config\candidate_registry.json`
   (`definition_clarified_2026-09-25_review_C`).

Also: `finalize_fo` now hashes the normalizers file via `feature_view.normalizers_path()`, the same
resolution the transform uses.

## Output
```
ROUTING: ...
VERDICT: APPROVE_FOR_ISOLATED_EXPERIMENTS | FIX_REQUIRED
remaining blockers (file:line) or "none"
APPROVED_HASHES: "REVIEW_C2_HASHES.json files" if approved
```
