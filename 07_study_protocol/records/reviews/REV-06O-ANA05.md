# REV-06O-ANA05 - independent read-only review (Claude Sonnet, 2026-10-01)

Scope: (A) ANA-05 fixes F1-F5 (DL 280) in tools/retention_report.py + tools/ana_block_index.py; (B) FIX-REEVAL-06O (PREG1_AMENDMENT_06O).
Hashes verified: reeval 67bb26f3, retention_report 6e6ac09a, ana_block_index 0067fc65, test_fix_reeval_06o 2e4d88a6,
test_preview_06o dfdca1bf, test_dev_ftpre 0c38056e. Pre-copies in fix06o_backup match 615f1c60 / b586eecd / d6fce3cb / f49b1cb6.
No result values read. Tests re-run (basetemp .local_ai/_pytest_tmp/REV-06O): test_fix_reeval_06o + test_preview_06o + test_dev_ftpre
31 passed / 14 xfailed; `--runxfail -k test_nc_` on the two 06O files: 8/8 NC fail at an assertion.

## Verdict: PASS (no BLOCKER, no MAJOR; 3 MINOR)

## (B) 06O checks
- Never a contrast/selection/stop/claim: retention_report.py:1303 (load_index refuses C_FULL/PRE on UNSEEN_VAL), :1538-1553
  (check_arm_identity refuses any row with preview_reference, under any arm; called at :1704 and :1904); ana_block_index.py:347-354
  excludes labelled dirs, refuses an unlabelled C_FULL/PRE UNSEEN_VAL record (old no-FT refusal, :355+) and a preview_reference on any
  other arm/role. No other consumer of UNSEEN_VAL rows in code/tools.
- MAIN/TEST untouched: admission is gated on role == UNSEEN_VAL (reeval.py:275-283, :291-300, is_preview_reference); test gate and
  TEST roles run first in check_args (:244); MAIN/C_FULL own-eval branches unchanged. Part A/B comparison: only UNSEEN_VAL C_FULL/PRE
  central rows differ (compare_06o.txt: 26 differ, all inside scope).
- Source checks still apply: admission only relaxes check_no_ft_args; classify_central_source (claims by arm label -> full field,
  seed, file-vs-rule check), verify_source_registry/check_source_binding, sha, meta, PRE init record all run; check_no_ft_source
  re-asserts central_source == arm (:291-297). check_args:264 still refuses fl/pf kinds for C_FULL/PRE.
- Not widenable: PREVIEW_SOURCES is a fixed tuple, requires ckpt_kind central and the source's own variant; everything else falls to
  the byte-identical no-FT refusal. C_SMALL/FA paths untouched (pinned by tests).
- test_dev_ftpre change (line 381): PRE with a C_SMALL central run now passes the arg gate and is refused by classify
  ("not the registered PRE"); still a refusal, justified.
- NCs non-vacuous (8/8 fail under --runxfail; each mutates one guard).

## (A) F1-F5 (current code, DL 280)
- F1: check_ana05 retention_report.py:4396-4433 binds every warm entry on VALIDATION (run folder / same-seed run / warm_run_problem
  :4316), TEST keys against VALIDATION bound_runs (:4414-4423); index reeval_warm_binding ana_block_index.py:800-826 sha-checks LAUNCH files. OK.
- F2: :1436 path file name required (None -> refuses). OK.  F3: require_warm_record :4340, called :4398. OK.
- F4: ana_block_index.py:393-396 S_CAL exclusion precedes check_warm_preview. OK.  F5: evidence file only (compare_fix.txt); not re-run.
- Caveat: the pre-fix hashes stated in the handoff (77682cc1 / 0fcc7ad0) are not in the scratchpad (ana05_backup holds c699b3bf /
  780b3ef9, an earlier state), so I reviewed F1-F4 in the current code rather than as a diff.

## Findings
MINOR
1. retention_report.py:4400-4405 - a REEVAL entry sharing (role, seed) with a run-folder entry is bound only by identical
   warm_selection; its own rows (fl_run_id/arm_spec_sha256) are not compared to that run folder's id. Matches the DL 280 spec (b), but leaves a gap.
2. ana_block_index.py:347 - the preview exclusion runs before the eval_manifest_id / split checks, so a mislabelled preview dir is
   listed as excluded instead of refused (harmless; it never enters).
3. reeval.py:277-283 - UNSEEN_VAL disjointness from the C_FULL band is taken from registration (06K section 1), not checked in code.
