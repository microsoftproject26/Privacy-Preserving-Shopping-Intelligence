# ANA-08 handoff (FABLE_GATE_ENDGAME_SCRIPTS_20261003.md §6: second E refusal)

- Archive: tools/_archive_r8/retention_report.py (byte-identical, sha256 d0247370f58ead3654bd0024211ae3718918b858f9bba358666f8415e2157035)
- New tools/retention_report.py sha256: 9a3b2c2a918ef2baacc5299426941f1eb94c366f5bb6aa7c29b4bfdbca0afe58
- Edited: s_record_warm S-content checks (now mirror launch_fl.check_s_content), new constant S_CAL_ROUNDS = 20 (not imported from launch_fl; pinned by test), docstring line; code/tests_ana/test_ana06.py (fixture n=20625 / rounds 0..19; pin test; positive refusal test over 7 bad records; 7 parametrized NCs with a lax loader).
- Test command: `cd code && python -m pytest tests_ana -q -x` -> `217 passed, 91 xfailed in 871.24s` (no failures).
- Claims-only: archived copy run from a scratch copy of tools/ (archive swapped in) vs new tools copy, `--claims-only --contrasts company/gates/ANA_CONTRASTS_v4.json --code-root code`: `diff -r` IDENTICAL. CLAIM_REGISTER.json fb67243286835e213848ec4868af061e54fb9ab7b834c91b8ebcfc2784aaaf23, HEADLINE_TEMPLATE.md 04ea667880bb256fbd5c2726c1be75e38a90f57fe37a8a011ff36b76dd8d3c39 (both).
- Deviations: (a) refusal-message keys are now separate (S, n_norms, rule, rounds_0based) instead of the old combined "n_norms / rounds_0based"; message prefix "REV-FL-16 m5" unchanged. (b) NC tests follow the repo xfail(strict) convention (a lax loader must make the refusal check fail), so a separate non-xfail test asserts the real tool refuses the 7 bad records. (c) S check uses math.isfinite(S) without float() (as launch_fl); only affects huge ints (OverflowError path), none in practice.

## Unified diff (tools)
```diff
--- tools/_archive_r8/retention_report.py	2026-10-03 17:33:54.652310600 +0300
+++ tools/retention_report.py	2026-10-03 17:34:14.100378000 +0300
@@ -405,6 +405,7 @@
 POISSON_BASE = ("FA_1024", "DP8", "DP1")                  # the 06C Poisson arms (cfg.dp set; DP: z > 0)
 S_FORMAT_WARM = "PPSI_V4_DP_S_RECORD_WARM_1"              # = launch_fl.S_FORMAT_WARM (FL-16)
 S_RECORD_WARM_KEY = {"FA_WARM": "S_RECORD_WARM", "FA_WARM_FROZEN": "S_RECORD_WARM_FROZEN"}   # records key per variant
+S_CAL_ROUNDS = 20                                                                # = launch_fl.KNOBS["s_cal_rounds"]
 S_CAL_WARM_ARM = {"FA_WARM": "S_CAL_WARM", "FA_WARM_FROZEN": "S_CAL_WARM_FROZEN"}            # = launch_fl.S_CAL_WARM_ARM
 STAR_TIE = 0.0005                                         # 06N N2: "ties within 0.0005 -> S8"
 STAR_RULE = ("06N N2: FA_WARM* = the variant (S7 = FA_WARM full model, S8 = FA_WARM_FROZEN frozen tables) whose "
@@ -4810,7 +4811,7 @@
     """ANA-06 (FL-16 S_CAL_WARM*; REV-FL-16 m5 on the ANA side): a registered warm S record (contracts/dp/
     S_RECORD_WARM[_FROZEN].json) — format PPSI_V4_DP_S_RECORD_WARM_1, self_sha256 (the launch_fl canon rule), written by
     the variant's S_CAL arm at seed 2026 as S_CALIBRATION, theta_pre init_from, frozen keys exactly for S8, and an S
-    CONTENT check: S finite > 0, n_norms an integer > 0 and = the number of rounds_0based."""
+    CONTENT check mirroring launch_fl.check_s_content: S finite real > 0, n_norms an int > 0, a median rule, rounds_0based = 0..19."""
     bad = {}
     if obj.get("format") != S_FORMAT_WARM:
         bad["format"] = obj.get("format")
@@ -4822,13 +4823,16 @@
         bad["stage"] = obj.get("stage")
     if obj.get("seed") != 2026 or obj.get("label") != "S_CALIBRATION":
         bad["seed / label"] = (obj.get("seed"), obj.get("label"))
-    S, n = obj.get("S"), obj.get("n_norms")
-    if not (isinstance(S, (int, float)) and not isinstance(S, bool) and math.isfinite(float(S)) and float(S) > 0):
+    S, n, rounds, rule = obj.get("S"), obj.get("n_norms"), obj.get("rounds_0based"), obj.get("rule")
+    if isinstance(S, bool) or not isinstance(S, (int, float)) or not math.isfinite(S) or S <= 0:
         bad["S"] = S
-    rounds = obj.get("rounds_0based")
-    if not (isinstance(n, int) and not isinstance(n, bool) and n > 0) or \
-            (isinstance(rounds, list) and len(rounds) != n) or not isinstance(rounds, list):
-        bad["n_norms / rounds_0based"] = (n, None if not isinstance(rounds, list) else len(rounds))
+    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
+        bad["n_norms"] = n
+    if not isinstance(rule, str) or not rule.startswith("median"):
+        bad["rule"] = rule
+    if not isinstance(rounds, list) or rounds != list(range(S_CAL_ROUNDS)) or \
+            any(isinstance(r, bool) or not isinstance(r, int) for r in rounds):
+        bad["rounds_0based"] = rounds
     why = init_from_problem(obj.get("init_from"))
     if why:
         bad["init_from"] = why
```
