# REV-ANA-08 (independent, read-only; spec FABLE_GATE_ENDGAME_SCRIPTS_20261003 section 6)

VERDICT: PASS_WITH_MINORS    Scope (diff confined to s_record_warm + S_CAL_ROUNDS + docstring + tests): YES

Checked: diff vs tools/_archive_r8, launch_fl.check_s_content (L709-732), KNOBS s_cal_rounds=20 (launch_fl L222), tests_ana/test_ana06.py. Ran `pytest tests_ana/test_ana06.py -q`: 22 passed, 19 xfailed.

1. Mirror: PASS. retention_report.py s_record_warm (~L4826-4835): S bool/non-number/non-finite/<=0; n_norms bool/non-int/<=0, no equality with rounds; rule str startswith "median"; rounds is list == list(range(S_CAL_ROUNDS)) and no bool/non-int items (True==1 and 0.0==0 are caught by the item check). Label is already forced to S_CALIBRATION by the retained seed/label check, so the single-branch rounds rule matches. All prior checks (format, self_sha256, arm/variant, stage, seed/label, init_from, frozen tables, init_sha256) untouched in the diff.
2. Crash paths: none for realistic input (rounds non-list handled before comparison; isfinite only reached after isinstance number check; NaN/inf refused). Only edge: int S > ~1e308 makes math.isfinite raise OverflowError (identical to launch_fl; see deviation c).
3. Deviations: (a) separate keys S/n_norms/rule/rounds_0based, message prefix unchanged: acceptable. (b) NC xfail(strict) convention plus a separate real-tool refusal test: acceptable, and better. (c) no float() in isfinite: acceptable, matches launch_fl exactly; minor (OverflowError on absurd int S) not worth deviating.
4. Scope: YES. Diff is the constant, docstring, s_record_warm body only.
5. Tests: test_s_cal_rounds_pinned_to_launch_fl (test_ana06.py L295-297) really pins R.S_CAL_ROUNDS to F.KNOBS. test_warm_s_content_mirror_refused (L303-307) runs the real tool on 7 bad records (short, shifted, bool round, n 0/-5/True, mean rule) and expects the m5 refusal. NC (L310-325) uses a lax loader restoring the good field + re-signing, so a missing check would make check_refuses fail -> xfail is meaningful. Fixture n=20625 / rounds 0..19 (L87-88) is n != len(rounds).

Minors (non-blocking): (i) no test for S bool/NaN/<=0 beyond the old S=0 NC; (ii) no test for non-list rounds or float items (0.0); logic is correct by inspection. (iii) S_CAL_ROUNDS is a pinned literal, guarded only by the test (allowed by spec).
