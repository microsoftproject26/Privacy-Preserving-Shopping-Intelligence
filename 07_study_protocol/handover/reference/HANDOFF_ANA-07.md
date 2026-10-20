# ANA-07 handoff (implements FABLE_GATE_ENDGAME_SCRIPTS_20261003.md §5 items 1-4)

- Archive: tools/_archive_r7/retention_report.py (byte-identical, sha256 1d50f7f2b75b0750ee6f91910c9d9cbd665462e5fdc5c8c7b04870ea1074de81)
- New tools/retention_report.py sha256: d0247370f58ead3654bd0024211ae3718918b858f9bba358666f8415e2157035
- Tests edited: code/tests_ana/test_ana04.py (check_calib known answer for both dicts; +1 pass-case test; 3 NCs: wrong manifest counts, wrong E2E pins, CALIB user missing from MAIN). Registered constants monkeypatched to the synthetic 48/16/32 (E2E 48/16).
- Test command: `cd code && python -m pytest tests_ana -q -x`  -> `215 passed, 84 xfailed in 836.78s` (no failures).
- Claims-only: `python -B <tools copy> --claims-only --contrasts company/gates/ANA_CONTRASTS_v4.json --code-root code --out <scratch>` run for the archived copy (in a scratch copy of tools/ with the archive swapped in, since it imports sibling modules) and for the new copy: `diff -r` IDENTICAL. CLAIM_REGISTER.json sha256 fb67243286835e213848ec4868af061e54fb9ab7b834c91b8ebcfc2784aaaf23, HEADLINE_TEMPLATE.md 04ea667880bb256fbd5c2726c1be75e38a90f57fe37a8a011ff36b76dd8d3c39 (both runs).
- Deviations from §5: none in substance. Notes: (a) order of checks: subset rule (VALIDATION, as before, now on manifest users) runs before the registered-count refusals; manifest and E2E refusals both under `enforce_registered_counts and split == VALIDATION`. (b) the registered-claim text string at ~L735 ("06L-C C14 ... counts must be the registered 51,211 / ...") was left untouched because it feeds the claims output that must stay byte-identical; it remains true. (c) counts_e2e["calib_users"] = distinct hex of CALIB E2E users (CALIB-member E2E users); disjoint E2E = MAIN E2E users whose hex is not a CALIB manifest user. (d) REPORT.md section 11 line now reads "N CALIB-disjoint users, of whom M have ≥ 1 eval row (...)", followed by the full counts JSON {"manifest","e2e"}. Only consumers of the old counts dict: calib_disjoint() passes it through (unchanged) and the REPORT.md writer (updated).

## Unified diff (archive vs new)
```diff
--- tools/_archive_r7/retention_report.py	2026-10-03 17:09:09.193600700 +0300
+++ tools/retention_report.py	2026-10-03 17:09:09.229177700 +0300
@@ -313,6 +313,7 @@
 NO_FT_PAIRS = (("FT_C", "C_SMALL"), ("FT_FA", "FA"))               # the SELECTION value of the FT arm vs its base
 CALIB_EVAL_ID = "CALIB_VALIDATION_5_100_32768"                     # G2 condition 5 / amendment 05 (CALIB-disjoint)
 CALIB_REGISTERED_COUNTS = {"main_users": 51211, "calib_users": 12715, "disjoint_users": 38496}   # amendment 02 §2
+CALIB_E2E_REGISTERED = {"main_users": 40764, "calib_users": 10070}   # DATA-06 L46 / L48: users with >= 1 eval row
 CALIB_TUNED_ARMS = ("C_SMALL", "FA", "FP", "PF", "CPF", "FA_1024", "DP8", "DP4", "DP1", "FA_Q8", "FA_WARM",
                     "FA_WARM_FROZEN", "DP8_FROZEN", "FT_FA",       # 06L-C §C14: every arm tuned on CALIB users
                     "FT_FA_WARM", "FT_FA_WARM_FROZEN",             # ANA-05: their FL bases were screened on CALIB
@@ -2628,17 +2629,29 @@
     if c.hex_by_euk is None:
         raise AnaRefused("CALIBRATION context without its users table (client_key_hex): MAIN \\ CALIB is undefined")
     cm = c.manifest
-    calib_hex = {str(c.hex_by_euk[u]) for u in np.unique(cm.eval_user_key[cm.eval_mask]).tolist()}
-    users = _euks(main)
+    calib_hex = {str(c.hex_by_euk[u]) for u in np.unique(cm.eval_user_key).tolist()}      # ALL CALIB manifest users
+    mm = main.manifest
+    m_users = np.unique(mm.eval_user_key)                                                  # ALL MAIN manifest users
+    m_in = np.asarray([str(main.hex_by_euk[u]) in calib_hex for u in m_users.tolist()], dtype=bool)
+    counts_manifest = {"main_users": int(m_users.size), "calib_users": int(m_in.sum()),
+                       "disjoint_users": int(m_users.size - m_in.sum())}
+    users = _euks(main)                                                                    # MAIN E2E users (>= 1 eval row)
     in_calib = np.asarray([str(main.hex_by_euk[u]) in calib_hex for u in users.tolist()], dtype=bool)
-    counts = {"main_users": int(users.size), "calib_users": int(in_calib.sum()),
-              "disjoint_users": int(users.size - in_calib.sum())}
-    if c.record.get("enforce_registered_counts") and split == "VALIDATION" and counts != CALIB_REGISTERED_COUNTS:
-        raise AnaRefused(f"MAIN / CALIB / disjoint user counts {counts} are not the registered {CALIB_REGISTERED_COUNTS} "
-                         "(amendment 02 §2; G2 condition 5)")
-    if split == "VALIDATION" and len(calib_hex) != counts["calib_users"]:       # at TEST: the same people, other rows
-        raise AnaRefused(f"{len(calib_hex)} CALIB users but {counts['calib_users']} of them among the MAIN E2E users: "
-                         "CALIB is not a subset of MAIN")
+    c_e2e = np.unique(cm.eval_user_key[cm.eval_mask])
+    counts_e2e = {"main_users": int(users.size), "calib_users": int(np.unique(
+                      [str(c.hex_by_euk[u]) for u in c_e2e.tolist()]).size if c_e2e.size else 0),
+                  "disjoint_users": int(users.size - in_calib.sum())}
+    counts = {"manifest": counts_manifest, "e2e": counts_e2e}
+    if split == "VALIDATION" and len(calib_hex) != counts_manifest["calib_users"]:   # at TEST: the same people, other rows
+        raise AnaRefused(f"{len(calib_hex)} CALIB manifest users but {counts_manifest['calib_users']} of them among the "
+                         "MAIN manifest users: CALIB is not a subset of MAIN")
+    if c.record.get("enforce_registered_counts") and split == "VALIDATION":
+        if counts_manifest != CALIB_REGISTERED_COUNTS:
+            raise AnaRefused(f"MAIN / CALIB / disjoint manifest user counts {counts_manifest} are not the registered "
+                             f"{CALIB_REGISTERED_COUNTS} (amendment 02 §2; G2 condition 5)")
+        if {k: counts_e2e[k] for k in CALIB_E2E_REGISTERED} != CALIB_E2E_REGISTERED:
+            raise AnaRefused(f"MAIN / CALIB E2E user counts {counts_e2e} are not the registered {CALIB_E2E_REGISTERED} "
+                             "(DATA-06)")
     keys = users[~in_calib].astype(np.int64)
     return {"status": "COMPUTED", "counts": counts, "calib_eval_manifest_id": cm.manifest_id,
             "disjoint_keys": keys, "mapping": "CALIB_VALIDATION users -> MAIN by client_key_hex",
@@ -5538,7 +5551,11 @@
     if cd.get("status") != "COMPUTED":
         L += [f"- {cd.get('status')}: {cd.get('reason')}", ""]
     else:
-        L += [f"- counts {json.dumps(cd['counts'], sort_keys=True)}; {cd['mapping']}; {cd['note']}.", ""]
+        _cm, _ce = cd["counts"]["manifest"], cd["counts"]["e2e"]
+        L += [f"- {_cm['disjoint_users']:,} CALIB-disjoint users, of whom {_ce['disjoint_users']:,} have ≥ 1 eval row "
+              f"(manifest users: MAIN {_cm['main_users']:,}, CALIB {_cm['calib_users']:,}; E2E users: MAIN "
+              f"{_ce['main_users']:,}, CALIB {_ce['calib_users']:,}); counts {json.dumps(cd['counts'], sort_keys=True)}; "
+              f"{cd['mapping']}; {cd['note']}.", ""]
         rows = [[a["arm"], _stat_cell(a["R_vs_C_BIG_disjoint"]), _stat_cell(a["R_vs_C_BIG_full_MAIN"])]
                 for a in cd["arms"]]
         L += _table(["arm", "R% vs C_BIG on MAIN \\ CALIB [CI]", "R% vs C_BIG on MAIN [CI]"], rows) + [""]
```
