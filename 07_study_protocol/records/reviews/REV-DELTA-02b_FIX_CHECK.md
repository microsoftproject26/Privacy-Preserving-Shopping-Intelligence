# REV-DELTA-02b — focused check of FIX-DELTA-02 (Claude Opus 5.5, 2026-09-29) — VERDICT: PASS

**Scope and rules.** This file is the only thing written in the tree. Test and probe outputs are under
`.local_ai/_pytest_tmp/REV-DELTA-02b/`. No server command, install, git, or Codex was used. No RETRO_TEST row or MAIN RESULT_ROWS
was read. The probe was in memory only: `-B`, with a script in the session scratchpad. Evidence tags: MEASURED (a command or diff I
ran), INFERENCE.

## Identity of the reviewed bytes (MEASURED)
- sha256 of the 8 files equals the handoff:
  - lr_screen b1e243f4…
  - launch_fl 1ef778d2…
  - retention_report da394e6d…
  - test_fix_delta02 fee7945c…
  - test_fl10 b2f93f0f…
  - test_fl12 a50a6d01…
  - test_fix_delta01 35fe039d…
  - test_ana03 ad7bcae5…
- The pre-fix copies in `_pytest_tmp/FIX-DELTA-02/orig/` have REV-DELTA-02's shas (e5015be4 / 08d92b4a / 10bc046a / …). The diffs
  below are against those copies.
- Tree hash vs REV-DELTA-02 `post_hashes.txt` covers the code/tools/contracts .py/.json/.md/.sh files. The **only** differences are
  those 8 files. The extra lines in my list are test-written `receipts/*.json`, which were outside REV-DELTA-02's hash set.
  - So no locked package changed: fl/, v4train, v4eval, v4data*, v4models, v4sup and ppsi_env are the same as at REV-DELTA-02.
  - No contracts/ file changed.
- `v4eval.evaluator_code_sha256()` = `61b64e61c1a2e4f5a6fecd42f0ad3b98de26562d13a07758573cc9496b8d962e`.
- The tree was identical before and after my test runs (receipts excluded). No `__pycache__` was left.

## Findings: closure status
| # | rule | status | evidence (MEASURED) |
|---|---|---|---|
| B1 | 06J §4(ii) | CLOSED | `stage_decision`: the 0.0005 band stays. The best finite non-incumbent candidate replaces the incumbent only if gain ≥ `MIN_GAIN` 0.002. Otherwise the winner is the incumbent, the tie set is [incumbent], and `no_measurable_gain` is set (also when no candidate is left: gain None). Exactly one incumbent is required. `apply_rule_stage` delegates to it. The same function drives the S2 τ-edge condition (`edge_groups`) and `launch_fl.load_fl_tuning`. `load_fl_tuning` recomputes and compares gain / best_candidate / no_measurable_gain / tie_band_set / min_gain. The eta edge still reads the raw best base FedAdam run, as the 06J §1 wording says. |
| M3 | 06J §1–§2 edge runs | CLOSED | `check_stage_runs`: one incumbent, every required label, and every condition-true edge group complete. A group with no runs raises `EdgeRunsMissing`, which names each run with its `launch_fl --role CALIBRATION` flags. A partial group, extra runs, or runs whose condition is false are refused. The τ edge is evaluated only after a condition-true eta edge is complete. `load_fl_tuning` re-derives `required` (`stage_required`), checks that the S2 incumbent is FedAvg at m*, and checks `edges`. |
| M1 | DL 219 (P = 2, B = 16) | CLOSED | `POISSON_ARMS` = FA_1024 / DP / S_CAL. `poisson_tuning` takes only the 7 optimizer / LR / shape fields and sets P 2 / B 16. When it overrides the selection, `poisson_local_rule` goes into the spec digest, is re-verified by `spec_tuning` / `check_poisson_rule`, and is written to the rows (`DL219_P2_B16`) and to LAUNCH. A default-local winner leaves the digest plain (test). An explicit B8 is refused for the Poisson arms. The S4 file carries `applies_to_exception`. |
| M2 | 06G §4, 06I §3 HOLD, 06J §3 | CLOSED | `registered_selection()` returns only FL_TUNING_SELECTION.json. `bind_main_selection` refuses `--lr-selection` for non-smoke MAIN. `check_tuning_for_main` refuses any stage ≠ S4. The binding records `stage`. Smoke / CALIBRATION are unchanged. The old EXT-binds-MAIN test was replaced by `test_ext_is_never_a_main_input`, and the 06F content checks are still exercised through `check_ext_for_main`. |
| M4 | 06I §7b, DL 218 | CLOSED | `bytes_per_client_charged_dropped` = mean + dropped × download / N. This is the registered value in the primary column, the ladder JSON and the second curve series. The up+down value is `_up_down`, labelled DESCRIPTIVE in the table column, the ladder JSON and the third series. A record without `download_bytes` gives `NO_BYTES_RECORD`. Table widths check out: header 18 = per-seed row 18 = mean row 18. `_download_only` is gone. |
| m5 | device | CLOSED | `_one_device` in select-joint / select-warm. The file records `device`. It fails closed when the records differ, and that includes a missing record. |
| m6 | bytes note | CLOSED (by documentation, as REV-DELTA-02 allowed) | `bytes_record` adds `frozen_bootstrap_note`. fl/comm.py is unchanged. |

## B1 synthetic probe (MEASURED; values from the original review)
- S2: incumbent FedAvg m=8 0.1000, FedAdam η 0.01 +0.0012 → winner **fedavg**, tie ['fedavg|m=8'], gain 0.0012,
  no_measurable_gain True.
- S4: incumbent 0.1000, P1 +0.0013 → winner **passes 2 (incumbent)**, gain 0.0013, no_measurable_gain True.
- Controls: +0.0025 → FedAdam wins (S2) and P1 wins (S4). `apply_rule_stage` agrees. S2 τ-edge with a +0.0012 FedAdam best →
  no τ edge required: `([], None)`.

## Tests (MEASURED; venv py3.13, `-B -p no:cacheprovider`, TMP/TEMP/--basetemp under `_pytest_tmp/REV-DELTA-02b/<suite>`, run once each, sequentially)
- tests_int: **253 passed, 76 xfailed, exit 0**
- tests_fl: **248 passed, 75 xfailed, exit 0**. The FL-10 224-pair pin, the trajectory pin, the FL-12 default pins and the
  test_fl03 default digests are all included and pass, so the default digests are byte-identical.
- tests_ana: **101 passed, 25 xfailed, exit 0**
- The counts equal the FIX-DELTA-02 handoff.
- NC run (`--runxfail -k test_nc_`) on test_fix_delta02.py + test_ana03.py: 16/16 failed, and every failure site is an
  AssertionError. This covers the 9 new NCs and the M4 NC.

## Record-only notes (not blocking)
- **n1 (boundary):** "gain exactly 0.002 wins" does not hold reliably in IEEE floats.
  - Probe: 0.102 − 0.1 = 0.0019999999999999879 → no gain; 0.0308 − 0.0288 = 0.0020000000000000018 → gain.
  - Raw RESULT_ROWS values are full-precision floats, so an exact decimal tie is practically impossible (INFERENCE).
  - Register the reading "raw float `gain < 0.002`" with the m9 records, or round both values to a fixed precision before
    subtracting. Do this before the S2 read.
- **n2:** with `--fl-tuning`, the explicit tuning flags (`--local-passes` etc.) are not consulted, because the tuning point comes
  from the file (`resolve_tuning`). This predates FIX-DELTA-02 and is not a regression.
- The Ops items m4, m7–m12 and the DL 219 mirror stay open as the handoff lists them.

**VERDICT: PASS.**
- B1, M1, M2, M3, M4, m5 and m6 are closed against 06J §4, 06I §3, 06G §4, 06I §7b and DL 218–219.
- Nothing else changed: only the 8 declared files differ, v4eval is 61b64e61…, the locked packages are untouched, and the default
  digests pass.
- code_ds3r3 may be packed from this tree.
- The S2 select-joint must run with this lr_screen (b1e243f4…), and n1 should be recorded before the S2 values are read.
