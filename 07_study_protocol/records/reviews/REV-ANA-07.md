# REV-ANA-07 (independent, read-only; 2026-10-03)

VERDICT: PASS_WITH_MINORS
Diff exceeds §5 scope: NO (hunks: constant L316, load_calib_context L2629-2658, report consumer L5554-5558, tests only).

Targeted run: `pytest tests_ana/test_ana04.py -k calib` -> 5 passed, 1 xfailed (the pre-existing mutant NC). Full suite not re-run.

## Checks
1. Logic matches §5 / g2_run: calib_hex from all CALIB manifest users (retention_report.py:2632); manifest counts on all MAIN manifest users (2634-2637), enforced == CALIB_REGISTERED_COUNTS (2649); subset rule (2645) as before; E2E pins 40764/10070 (316, 2652); disjoint view = MAIN E2E users not in CALIB manifest hex (2639), keys at 2655; disjoint_keys meaning unchanged (E2E MAIN users outside CALIB manifest users - this is the amendment's view). OK.
2. Crash paths: hex_by_euk coverage is checked for every manifest eval_user_key at load (L1284 MAIN/CALIB; production loader L1454), so manifest-only users cannot KeyError. Empty arrays: m_users empty -> bool empty array fine; c_e2e empty handled (2642). TEST split: subset and registered checks are VALIDATION-only as before; counts still computed, no crash.
3. Consumers: only L3766 (passes dict through) and report L5554-5557 read it; both handle the new shape. metric_table.py, tools/g2_run.py, tests_g2, endgame ops_scripts: no consumer. runs_p3/g2/cond5_diagnostics.py uses g2 mapping only.
4. Scope confined: yes.
5. NCs exercise refusals: wrong manifest (test_ana04.py:489-494, needle "manifest user counts"), wrong E2E pins (497-502, "E2E user counts", manifest passes first so the E2E branch is the one hit), missing CALIB user (505-508, hits subset check). Real.

## Minors (non-blocking)
- retention_report.py:2641-2643: counts_e2e["calib_users"] is the count of distinct CALIB E2E hexes, not mapped into MAIN; fine as the DATA-06 pin, but not subset-verified. Cosmetic.
- :2645 vs :2648: the subset check runs before the registered-count check, so on real data a non-subset gives the subset message; no consequence.
- :2645 compares len(calib_hex) (set of hex) not the CALIB key count; duplicate hexes across CALIB keys would collapse silently (g2 enforces 1:1). Unchanged behaviour, optional.
- NC at test_ana04.py:497 asserts the substring only; fine.
