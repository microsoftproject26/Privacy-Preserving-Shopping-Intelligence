# REV-ANA-06 — independent read-only review (Claude Opus 5.5, 2026-10-02)

**Verdict: PASS-WITH-FIXES** (0 BLOCKER, 0 MAJOR, 4 MINOR). No MAIN / VALIDATION / TEST value read; synthetic fixtures only;
no git, no Codex, no code edited. Scratch: session scratchpad `r6/` (r6 tars extracted with tarfile), `rev_ana06_d5.py`,
`rev_p5.py`; outputs under `_pytest_tmp/REV-ANA-06/`.

## Hashes (MEASURED)
retention_report 864bcf05, ana_block_index 5a8d5d29, reeval 833f0c98, ANA_CONTRASTS_v4 f1769f0b, test_ana06 1e3ca517,
test_ana06_reeval e879d15d, anakit 2ab1055b, test_ana04 f2784d42 = the handoff. r6 extracts = 6e6ac09a / 0067fc65 / 67bb26f3
(the stated "before").

## D5 acceptance (MEASURED by my own script, old = r6 extract, new = tree)
- v3 reports (anakit v2 block, ANA-04 full v3 block, ANA-05 warm block): RETENTION_RESULT differs only in `/ana06`,
  `/script/sha256`, `/self_sha256`. REPORT.md is line-identical outside the new section 12c, apart from the two sha lines.
- `--claims-only` v3: CLAIM_REGISTER (9ce7103a) and HEADLINE (2d6aa76c) are byte-identical old / new.
- v4 claims vs v3: the changed ids are K2 (tier primary→secondary, forbidden), K20, K20b and K21 (contrast resolved to
  FA_WARM_FROZEN; K20b / K21 primary). New: K_INIT, P5, K4w, K5w, K12w, K13w, K14w (secondary) and the 6 descriptive rows.
  Nothing is dropped. Headline lines 0, 6 and 10 change (K2 ablation + K_INIT; the primary pair). Exactly the D5 set.
- v4 report on the 06N block:
  - 33 confirmatory contrasts. K20b and K21 are CONFIRMATORY and NOT_IN_HOLM_FAMILY, and their points equal the v3 `*` rows.
  - Holm family = 18; tertiary band = 13.
  - K12w / K13w are CB, K13w is [2026], K14w has TOST.
  - Cold K values and the retention tables are identical between v3 and v4. Only the Holm fields differ (family 26 → 18 / 13).
  - P5 (ANA-05 warm block, best.pt bases): COMPUTED, CONFIRMATORY, seeds 2026 / 2027.

## Tests (MEASURED, next_item_eda venv, basetemp `_pytest_tmp/REV-ANA-06/`)
- tests_ana (full suite): **208 passed, 83 xfailed**, which equals the handoff.
- tests_int: only test_ana06_reeval and test_fix_reeval_06o were run, **16 passed, 7 xfailed**. I did not re-run the full
  tests_int suite because of the time cap.
- NCs run with `--runxfail`: test_ana06 11 / 11 and test_ana06_reeval 2 / 2 fail on an assertion (junit checked).

## Spec checks — all OK
- **06P P1–P7 + ERRATUM 1.** The v4 file holds the text verbatim, and its 06P sha 75f8dbad matches DL 331 M2. Holm family =
  the 18 erratum ids. Tertiary = the 13 listed. registered_seeds = v3 + K13w [2026]. The two P7 dependencies are present.
  post_validation_rule is unchanged. A draft file is refused without the flag and on TEST.
- **Warm labels.** The 12 labels are mapped (DP_WARM_FROZEN ε8 = DP8_FROZEN); S_CAL_WARM* are excluded.
- **N33.** `N33_*` dirs are indexed as the cold arm, and a non-N33 arm such as N33_FA refuses.
- **Stopped dirs.** They are excluded by name and by marker.
- **06O.** The 06O preview rows stay excluded.
- **MINOR 1** is fixed (`run_rows_problem`).
- **Single seed.** No `--allow-single-seed` is needed.
- **NCs.** The NCs of test_ana06 (11) and test_ana06_reeval (2) fail on an assertion (junit checked); none is vacuous.
- **Tertiary "Holm band beside".** 06L_C C13 lists the Holm band among the "Statistics additions (DESCRIPTIVE unless stated)".
  So a DESCRIPTIVE band with no decision is consistent with C13. The within-tertiary family (m = 13) is a choice that C13 does
  not fix, so it stays on the DL registration list.

## Findings
MINOR
1. `tools/ana_block_index.py:134-136`: `SINGLE_SEED_REGISTERED` still holds DP8_FROZEN (06L-C C17), but 06N N3.2, D4 and K12w
   make it 2 seeds. A missing DP8_FROZEN s2027 is therefore not refused at the index. The report does flag it INCOMPLETE_SEEDS.
2. `tools/ana_block_index.py:347` `reeval()`: there is no STOPPED_REALLOCATED source check. A REEVAL of FL_T/FP_s2026,
   FP_s2027 or FA_1024_s2026 would still index as cold-arm panel evidence, which breaks DL 317 "never evidence". The record
   carries `source_run`, so the check is possible. None is scheduled (D3).
3. `tools/retention_report.py:4143`: the v4 claim register keeps the flag "EQUIVALENT: K14 only", but K14w also gets TOST.
4. `tools/retention_report.py:3300`: the P5 best.pt-base filter has no known-answer test and no NC. I verified the positive
   case by hand only.

Ops note: handoff item 7 (six pytest processes killed by command-line match) — ask BM-02 to re-run its last tests_bench.
