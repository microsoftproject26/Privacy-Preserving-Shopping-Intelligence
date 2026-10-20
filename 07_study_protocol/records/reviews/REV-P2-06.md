# REV-P2-06 — independent review of the Phase-2 selection tool (v4int/p2_select.py)

- Reviewer: Claude (Opus), 2026-10-03. READ-ONLY on the project (this file is the only project write). No server, no run,
  no value read (synthetic fixtures only), no Codex. Two probe scripts ran in the session scratchpad only.
- Spec (binding): contracts/PREG2.md REGISTERED (sha256 be0561abe5ceec450d3a75bef49853a3c480d3847f75559dd94e9e0545df18a8,
  re-measured) §4.2, §5.2–§5.9, §6.3 S8, F3; FABLE_GATE_ENDGAME_SCRIPTS_20261003 §9–§10.
- Reviewed (MEASURED sha256): code/v4int/p2_select.py 1d83e62a…77ab0 and code/tests_int/test_p2_select.py 92e67d5a…5b41
  (both = handoff). lr_screen.py 1a391b1e…f8fd3 (mtime 09-30) = handoff, unchanged.
- Test run: `cd code && python -m pytest tests_int/test_p2_select.py -q` → **17 passed, 1 xfailed** (12.5 s). With
  `--runxfail` the negative control `test_nc_argmax_ignores_tie_preference` FAILS, as required.

## VERDICT: **FAIL** (fix, then a delta review)

The core decision logic (cell sets, tie order, exact server step, step table, futility, write-once, self-sha) is sound
and well tested. Three MAJOR defects block use: (M1) the anchor-flag threshold does not match the REGISTERED text;
(M2) the tool refuses every real P2 run that the launch_fl being written now produces, and the tests hide this with a
monkeypatch; (M3) one diverged run blocks the whole selection file. All three are small, well-defined fixes.

## launch_fl.py — not changed by this packet (MEASURED)
- launch_fl.py is now cd8d7216…a0f364 (mtime 2026-10-03 19:30), not the handoff's c9fdc8c0. The P2-01+02 worker changed
  it: `code/_archive_p2_wave_a/` (created 19:22:48, after p2_select's last write at 19:21:28) holds c9fdc8c0 with its
  SHA256SUMS, the diff header names "P2-01 / P2-02", and DL 414 records P2-01+02 resumed in parallel. P2-06 did not
  edit launch_fl. Note: the selection file records the launch_fl sha at write time, so the P2 lock must bind the final sha.

## MAJOR findings
**M1 — The anchor-flag threshold does not match the registered text.** REGISTERED §5.6 (PREG2.md:243-244):
"NO_MEASURABLE_CALIB_GAIN iff CALIB_GAIN < 0.001". Fable §10 confirmed this, and PHASE2_IMPLEMENTATION_PLAN:130 says the same.
The tool uses 0.0005: p2_select.py:82 (`ANCHOR_MIN_GAIN = 0.0005`), :26-28, :150 (GATE_EDITS "anchor_flag"), :459.
Tests pin the wrong value: test_p2_select.py:191 asserts 0.0005, and :245 expects a 0.0006 gain to be not flagged. Under
0.001 that gain is flagged. The comment at :582 needs the same fix. The flag is printed beside every Q result.
Fix: set 0.001, update the texts, and add boundary tests (see m5).

**M2 — `p2_run_adapter` is not the single alignment point: every real P2 run is refused (MEASURED by probe).**
`read_p2_run` → `lr_screen.read_run(arms=P2_ARMS)` (p2_select.py:661). read_run calls `F.spec_tuning` and
`F.check_warm_spec` (lr_screen.py:341-345) and compares row labels with `row_labels(t) + warm_row_labels` (lr_screen.py:365-372).
I built specs with the **current** `launch_fl.p2_arm_spec` (launch_fl.py:2799) and checked them with `check_p2_spec`
(fedadam 1e-4 / c 0.1 and fedavg 0.3 / c 0.01). Rows were written by `fl_rows` with `warm_row_labels + p2_row_labels`,
exactly as `run` writes them. Results:
- `spec_tuning` refuses: "solver passes / batch 1 / 16 != the spec's 2 / 16". The P2 spec has passes 1 (PREG2 §5.1) and
  no top-level server_opt / client_opt.
- `check_warm_spec` refuses: "P2_W: warm-start keys in a cold-start arm spec".
- So `read_p2_run` raises SelectionRefused (exit 2) for both runs.
- The row-label check would also fail: `p2_row_labels` (launch_fl.py:2855) always writes fl_server_opt / fl_client_* /
  fl_local_passes, while `row_labels(t)` writes non-default dimensions only.

The P2-02 code uses its own `check_p2_spec` / `p2_row_labels`. It adds no P2 branch to spec_tuning or check_warm_spec, so
the integration requirement in handoff §4.1 has not been met. The tests hide this:
- an autouse monkeypatch of `check_warm_spec` (test_p2_select.py:39-50);
- synthetic specs carrying Phase-1 defaults (passes 2, weight_decay 1e-5, `knobs`; :58-71), not the registered passes 1.

The failure is loud: it is fail-closed, not a silent mis-read. Fix (recommended):
- Give p2_select its own `read_p2_run` that keeps read_run's structural checks: DONE_MAIN; digest = RUN_SUMMARY =
  ARM_SPEC; role CALIBRATION / seed 2026; 12 marks; exactly one PRACTICAL_BEST E2E macro mrr@20 row equal to the
  evaluation at its best mark; run_label FL_LR_SCREEN and fl_lr_mult 1; non-finite → excluded.
- Replace spec_tuning / check_warm_spec with `F.check_p2_spec`.
- Replace the label check with `warm_row_labels(spec) + p2_row_labels(spec)`, with the CALIB relabel applied.
- Add a test that builds runs with `F.p2_arm_spec` and **no monkeypatch**.

Do not add P2 branches to the Phase-1 functions spec_tuning / check_warm_spec (F6; ANA_SCRIPT_FREEZE_v3 records
launch_fl_sha256). lr_screen stays unchanged either way.

**M3 — A NaN in a recorded D1-ext summary blocks the whole file (MEASURED by probe).** In the probe, one WA cell has
value NaN, as expected for a diverged run, and RUN_SUMMARY `update_norm_log.last_total_T = NaN`. `write_selection("WA")`
then raises "the selection object cannot be canonicalised (… nan)". The cause: `_finish` uses `_canon` with
`allow_nan=False` (p2_select.py:903-909) and copies `d1ext` verbatim (:630, :638).

launch_fl's `update_norm_summary` (launch_fl.py:1504-1522) copies `last_total_T` from the log. `_json_safe` keeps NaN.
§5.6 and §9.5 require non-finite runs to be "excluded and recorded", and this defect makes that impossible.

Fix: store a sanitised copy of d1ext (non-finite values → the strings "NaN" / "Infinity" / "-Infinity", plus a flag), or
store the file sha + finite fields only. Add a test with a NaN-valued run whose D1-ext carries NaN.

## MINOR findings (fix before use unless Ops records a ruling)
- **m1 — Corner cell vs the REGISTERED §5.7 text.** The draft said "At most 3 extension runs per stage (two axes plus their
  corner)" (PREG2_FOLLOWUP_DRAFT.md:217). The REGISTERED text (PREG2.md:251-257) drops the parenthesis. It keeps "at the
  winner's other coordinates" and "At most 3". The tool still **requires** the corner run (p2_select.py:378-379, :30-31,
  EDGE_RULE_TEXT :140-143; exit 3 until it exists). "At most 3" only makes sense with a corner, so the reading can be
  defended. It is still not literal text, and it costs one run and can set EDGE_UNRESOLVED. **Ruling required** (Ops DL
  row, or Fable at P2_METHODS_LOCK) before the WA file is written.
- **m2 — Tertiary drop tier treated as an unordered set.** p2_select.py:116 (`ordered=False`) and :540-560 accept
  `["SECAGG_MAIN"]` or `["DPFTRL_S2027"]` alone, and test :358 pins this. REGISTERED §5.9 (PREG2.md:267) orders the tier:
  "P2_WSTAR_DP1 → P2_WSTAR_SECAGG on MAIN → P2_WSTAR_DPFTRL8 s2027". Part-B arms that are unfinished at lock time are
  declared not run under §2.1, not under the cap. Record those separately (for example a distinct "not run (Part B)"
  list), and keep the cap order strict.
- **m3 — `cap_drop_plan` and `check_dropped` disagree.** The plan skips units with zero or absent savings
  (p2_select.py:569-575). For `cap_drop_plan(300, sav)` it drops C2 before WC1/WC2 and WB3, which breaks "C2 last", and
  `check_dropped` refuses that exact plan (MEASURED). Test :368-370 pins the inconsistent plan. Require savings for every
  unit and stop at the first unit that cannot be dropped. Assert that the plan passes `check_dropped`.
- **m4 — Drops are not bound to the cap projection.** FINAL accepts any valid prefix (p2_select.py:1037, :1058) with no
  evidence that the projected total exceeds 250 GPU-h ("If the projected total exceeds it", §5.9). Recommendation: FINAL
  takes the `cap_drop_plan` input (projected GPU-h + savings), records it, and requires drops == plan.
- **m5 — Exact-0.001 boundaries are decided by float rounding (MEASURED).**
  - The band is `best - value <= 0.001` (:453): 0.1716 vs 0.1706 gives 0.0010000000000000009 (NOT tied), while 0.1720 vs
    0.1710 gives 0.000999…73 (tied).
  - Futility `< 0.001` (:528) flips the same way.
  - Use exact decimal arithmetic, as the tool already does for server steps (:192-194): `Fraction(repr(a)) -
    Fraction(repr(b))` against `Fraction(1, 1000)`.
  - Add the missing boundary tests at exactly 0.001: tie inclusive → tied; futility strict → CONTINUE; anchor strict →
    not flagged.
  - Today's tests use only 0.0006 / 0.0011 (:234-238) and 0.0007 / 0.0009 / 0.0011 (:346-350).
- **m6 — The authority text cites the draft.** p2_select.py:1-57, :146-153 (AUTHORITY, GATE_EDITS) and FUTILITY_DISCLOSURE
  (:144) cite PREG2_FOLLOWUP_DRAFT + "GATE §9". Cite contracts/PREG2.md (REGISTERED, be0561ab…, DL 411) and name the
  futility caveat D-j. The section numbers themselves match the REGISTERED file.
- **m7 — Incumbent reuse guard is weak.** The P2-T3 receipt is checked for existence and sha only (p2_select.py:674-676,
  :697), and the tests accept `"{}"` (test :610). §5.2 says "reused only if" P2-T3 is bit-identical. Require a minimal
  receipt: test P2-T3, verdict BIT_IDENTICAL, 64 rounds, S8Tc run_id and RESULT_ROWS / RUN_SUMMARY sha equal to the run
  being read. Also, the Phase-1 incumbent is compared with the P2 runs on init_from and seed only (:868-874). Also check
  passes, batch, max_rounds, dropout and plan.
- **m8 — S8 "refused unread" order.**
  - p2_select's `read_c1_run` parses RUN_SUMMARY / RESULT_ROWS before the manifest check and never checks role
    (:741-753).
  - P2-05 already ships an S8-strict verifier, `central_ft.read_c1_run` (central_ft.py:688-747). It checks the RUN_START
    identity (role CALIBRATION + CALIB manifest) before any value-bearing file is opened. Delegate to it, so there is one
    C1 verifier.
  - P2 runs have the same pattern through read_run. No value is printed or written, so S8's human rule holds, but the
    stricter order is available.
- **m9 — Edge status reads beyond the text.** UNRESOLVED on (b) a non-finite extension or (c) "winner moved to a base run
  whose extension was not run" (p2_select.py:504-509). Both are conservative interpretations. §5.7 names only "again on
  the new edge". Record them in a DL interpretation note. They are also inconsistent: when an extension cell equals an
  **existing** base run that is non-finite, (b) does not fire, because :496/:504 look at `extra` only.
- **m10 — The verifier can be hardened.** `load_p2_selection` re-derives winner, tie_set and edge status (:995-1005) but
  not `flags`, `calib_gain`, the registered `required` set, or FINAL futility / drop consistency. `bind_p2_selection`
  (launch_fl.py:2901-2940) reads `flags`.
- **m11 — Handoff gaps.** Handoff §5a "After-run" was never filled (P2-06.md:130-131). Its launch_fl sha is stale (see
  above). Exceptions outside SelectionRefused exit 1 with a traceback instead of 2, for example the LaunchRefused from
  `lr_mult_value` outside read_run's try (lr_screen.py:339). That is still fail-closed.

## Rule-by-rule check against REGISTERED PREG2 (MEASURED unless marked)
| rule | registered | implemented | status |
|---|---|---|---|
| metric / mark 0 | CALIB_VALIDATION E2E macro MRR@20 at PB, marks 0.5–6.0 | read_run 12 marks; C1 mark ∈ [0.5, 6.0] (:763) | OK |
| WA candidates | 18 core + WA-ext 2 at c 0.1 + incumbent | `wa_cells` :279-292 | OK |
| FINAL candidates | {A*} ∪ WB1-3 ∪ WC1-5 at A*'s recipe | `final_cells` :301-314 | OK |
| C1 | {1e-5,3e-5,1e-4,3e-4} × {FULL,FROZEN}, within variant | :317-323, :439-440 | OK |
| tie band | 0.001 | `TIE_CALIB` :80, :453 | OK (boundary: m5) |
| preference 1–5 | bytes → trained params → server step → client LR → label | `p2_order_key` :267-275; exact rationals | OK |
| C1 preference | "bytes do not differ; then the smaller LR" | (0, numel equal in variant, 0, peak LR, label) | OK |
| anchor flag | < **0.001**; W* still carried | 0.0005 | **M1** |
| non-finite | excluded and recorded | read_run + `p2_decide` :442-447 | OK (D1-ext: M3) |
| edge steps | §5.7 table incl. C1 3e-6 / 1e-3 | `EDGE_STEPS` :104-113, values verbatim | OK |
| ≤ 3 runs, one extension, re-applied, EDGE_UNRESOLVED | §5.7 | :353-401, :464-513 | OK (corner m1; extra readings m9) |
| futility | both gains < 0.001; skip WB/WC; D-j | `futility` :516-532 | OK (boundary m5) |
| cap / drop | 250; tertiary DP1→SECAGG→DPFTRL → WC3/4→WB2→WC1/2→WB3 → C2 last | :83, :116-119 | tertiary order **m2**; plan **m3** |
| CALIB only (F3, S8) | MAIN / TEST refused (test) | allowlist on eval_manifest_id :645-648, :753, :793; role/seed :lr_screen 335 | OK (order m8) |
| outputs | write-once A / FINAL / C1, self-sha, values, marks, bytes, D1-ext | :1086-1107, :122-124 | OK |

## (3) Write-once and self_sha256 style (MEASURED)
- `write_exclusive` uses `os.O_CREAT | os.O_EXCL` (v4models/variants.py:148-159), and there is a prior exists-check
  (p2_select.py:1090). Tested via the API and the CLI (exit 2 on rewrite).
- The self_sha256 is `launch_fl._canon`: compact `sort_keys`, ASCII-escaped, `allow_nan=False`. The file text is
  `json.dumps(indent=1, sort_keys=True)`, ASCII.
- That style is **identical** to the 5 existing contracts/fl/*.json files, all re-verified here (`_canon` matches, indent-1,
  ASCII). It is accepted by retention_report `_self_ok` (`_canon_compact`) and equals `central_ft.canonical_self_sha256`.
- Missing required runs are refused (`_partition` :834-836). Missing edge runs are refused with exit 3, and the message
  names them.

## (4) What must be aligned with the concurrent launch_fl (P2-01/02) and siblings
1. **The read path (M2).** This is the blocking item, and it lives outside `p2_run_adapter`.
2. Already aligned:
   - `P2_ARMS = ("P2_W",)`;
   - `RECIPE_KEYS` = `F.P2_RECIPE_KEYS`;
   - `p2_label` byte-identical in both modules;
   - FINAL format, stage, winner, winner_recipe and common.init_from as read by `bind_p2_selection`.

   Add a cross-module test: `P.p2_label == F.p2_label` over every registered and edge cell, and
   `P.RECIPE_KEYS == F.P2_RECIPE_KEYS`.
3. `cfg.solver.lr` must equal the client LR applied. launch_fl writes c. For WC1/WC2, P2-04 must write c × 10, or the
   adapter must change.
4. `bytes.per_visit.upload_bytes` / `shared_numel` exist (`bytes_record` → `fl.comm.visit_bytes`; shared_numel = the
   uploaded numel after the freeze).
   - WB3 (P2-03): the payload varies per visit. The `p2_post_run` hook must publish a per-visit figure there, or the
     adapter must read it from `RUN_SUMMARY.p2.post_run`.
   - LoRA (P2-04): the manifest must count LoRA parameters only.
5. The `update_norm_log` key matches `update_norm_summary` (NaN: M3).
6. Cross-check `RUN_SUMMARY.p2.{label, recipe}` (launch_fl run path) against the arm spec inside the adapter.
7. `P2_COMMON_FIELDS` (p2_select.py:160-162):
   - `("knobs",)` does not exist in real P2 specs. It is None for every run, so the check is vacuous.
   - Add `study`, `p2_fixed`, `lr_peak_base` (the WC5 PF LR), `exposure_budget` and `upload_codec`.
8. Under futility with a Phase-1 incumbent as A*, FINAL has no P2 run, so `common` is None. `bind_p2_selection` then
   skips the θ_pre check (`init_from_checked` False). Record the incumbent's init_from in that case.
9. C1: the trainer and central_ft write `run_label`, `peak_lr`, `seed`, `p2_trainable_numel` and `practical_best`;
   `device_pin_sha256` appears only when the run is pinned. Prefer delegating to `central_ft.read_c1_run` (m8). The
   central_ft C1 binder reads `format`, `stage` "C1", `variants.*.winner_peak_lr` and `run_label`, which p2_select writes
   (MEASURED).
10. Anchor (P2-07): nothing emits `P2_ANCHOR_PRE_CALIB` yet, so `anchor_adapter` is still an assumption.

## (5) Test coverage
Covered:
- registered sets;
- the full preference order;
- the exact server step;
- 0.0006 tied / 0.0011 not;
- the anchor flag, with the winner carried;
- non-finite exclusion;
- the whole step table;
- the extension cases: missing (exit 3), resolved, unresolved (wins / non-finite / moved), unregistered extra refused,
  existing-cell coincidence;
- futility on both conditions;
- tier drops;
- write-once (API + CLI);
- self-sha + tamper re-derivation;
- refusals: Phase-1 label, MAIN / TEST manifest, MAIN role, missing run, foreign anchor;
- incumbent with receipt;
- C1 per variant;
- FINAL drops and futility;
- the negative control.

Missing:
- **exact-0.001 boundary tests** for the tie band, futility and the anchor flag (m5);
- the anchor flag at the registered 0.001 (M1);
- a real-format P2 run with no monkeypatch (M2);
- a NaN D1-ext (M3);
- a C1 extension that wins (UNRESOLVED);
- a FINAL LoRA edge through files;
- a TEST manifest on C1 / anchor (the code path is shared, so this is low risk).

## Observations (INFERENCE; registered behaviour, for disclosure only)
- O1. Preference (3) compares FedAdam η_s with FedAvg / FedAvgM steps numerically, in different units. FedAdam therefore
  wins every cross-optimiser tie (handoff Q5, implemented literally).
- O2. Smaller steps and fewer bytes win ties. A tied **downward** extension therefore always wins and gives
  EDGE_UNRESOLVED: client 0.001, FedAvg 0.1, FedAvgM 0.01, FedAdam 3e-5 / 3e-6, item-LR 0.001, LoRA r1, C1 3e-6. A tied
  upward extension loses and gives EDGE_RESOLVED. This asymmetry follows from the registered rule.
- O3. Futility uses the raw best, including edge runs and the incumbent (handoff Q8). This errs toward running WB / WC,
  which is acceptable.

## Required for PASS (delta review)
1. Fix M1, M2 and M3, with tests.
2. Fix m2, m3 and m5, with the exact-boundary tests.
3. Get a DL ruling on m1 (corner) and m9 (the extra UNRESOLVED readings).
4. Update the authority text (m6).
5. Re-run tests_int/test_p2_select.py and the lr_screen-importing suites.

---

## Delta review (2026-10-03, after the worker's "REV fixes", company/handoffs/P2-06.md §0)

- Same reviewer, READ-ONLY. Only this file was written in the project. Probes ran in the scratchpad only. No server, no
  value read, no Codex.
- Files (MEASURED sha256):
  - p2_select.py 166786f6…275839 and test_p2_select.py 43202a02…bf402 (both = the handoff);
  - lr_screen.py 1a391b1e…f8fd3, **unchanged**;
  - launch_fl.py c6192343…f1f2a (P2-01+02's live edits, not this packet);
  - central_ft.py f392121d…d510 (P2-05);
  - PREG2.md be0561ab…, REGISTERED.
- Tests:
  - `python -m pytest tests_int/test_p2_select.py -q` → **22 passed, 1 xfailed** (4.6 s).
  - With `--runxfail`, the negative control FAILS, as required.
  - The sibling binder tests that call the stricter verifier (test_p2_launch / test_p2_central_ft `-k "select or bind"`)
    → **3 passed**.
  - The lr_screen-importing suites (122 passed / 55 xfailed) are **REPORTED** by the worker and were not re-run here: that
    needs pytest temp space on F: inside the project, or on C:, where 6.9 GB is below the 8 GiB FL disk floor. lr_screen is
    byte-unchanged (MEASURED).

### NEW VERDICT: **PASS_WITH_MINORS**
Every MAJOR is fixed and was checked against the REGISTERED text. Every earlier minor is fixed or resolved by the literal
registered reading. Two small faithfulness gaps remain in the R1–R12 re-implementation (N1, N2). Both accept rows that
lr_screen.read_run refuses. Neither can be reached with rows that `launch_fl.fl_rows` writes. Fix both before the first real
selection file is written. Ops can verify the fix by diff plus two tests; no further review is needed.

### Earlier findings: re-verified
| finding | evidence (file:line) | status |
|---|---|---|
| **M1** anchor 0.001 | `ANCHOR_MIN_GAIN = 0.001` :95; exact strict `gain < 1/1000` :491-494 = PREG2.md:243-244; tests :208, :260-267 (0.0006 flagged; exactly 0.001 not flagged; ±1e-12 both ways; winner carried) | FIXED |
| **M2** real P2 runs | `read_p2_run` :683-756 implements R1–R12 (mapping below). Real launch_fl `ARM_SPEC.json` carries `arm_spec` (launch_fl.py:1662-1664), so R1 matches the real format. **No reader monkeypatch remains**: grep finds "monkeypatch" only in a test name (test :430); conftest's only autouse fixture sets strict numerics. Fixtures use the real `F.p2_arm_spec` / `F.fl_rows` / `warm_row_labels` / `p2_row_labels` (test :51-86). Test :430-436 proves read_p2_run accepts the real run and lr_screen.read_run refuses it. C1 → `central_ft.read_c1_run` :811-836 | FIXED (N1 / N2 below) |
| **M3** NaN blocks file | `_finite_json` :210-228 applied to D1-ext and max-of-marks :669-676; NaN marks flagged :732-738; non-finite value → excluded (R10). Test :496-516: NaN run excluded with D1-ext "NaN"; a finite run with D1-ext inf + NaN mark is flagged, ranked, and the file is written and verifies | FIXED |
| m1 corner | no corner :404-422 (registered §5.7 literal); max 2 cells per WA winner, test :289-298 | RESOLVED (literal) |
| m2 tertiary order | one strict `DROP_ORDER` :129 = PREG2.md:266-269; prefix-only :566-575; test :396-402 | FIXED |
| m3 plan vs check | every unit's savings required; drops in strict order :578-599; prefix property tested :405-407 | FIXED |
| m4 cap binding | `--dropped` needs `--cap-plan`, and drops must equal the plan :1156-1160; plan recorded :1200; test :613-644 | FIXED |
| m5 float boundaries | exact `Fraction(repr)` :201-207; band inclusive :488; futility strict :557-558; tests at exactly 0.001 and ±1e-12 :250-257, :380-386 | FIXED |
| m6 authority | REGISTERED sha / DL 411 :1-3, :92, :159-160; D-j verbatim :157-158 | FIXED |
| m7 incumbent | receipt test / verdict / rounds / run_id / both shas :759-777 (`"{}"` refused, test :556-559); cfg / plan / rounds / θ_pre equal to `p2_arm_spec` of the incumbent :926-954; test :545-575 | FIXED |
| m8 S8 order | FL: R1–R3 use ARM_SPEC.json only, before RUN_SUMMARY / RESULT_ROWS :688-703 (MAIN role refused unread, test :484-486); C1: central_ft RUN_START-first | FIXED |
| m9 extra UNRESOLVED | status literal (`EDGE_UNRESOLVED` iff the union winner is an extension cell) :542; facts kept in `notes` :536-541 | RESOLVED (literal; see O-B) |
| m10 verifier | re-derives calib_gain / flag / flags / registered required set; FINAL futility and drop prefix :1061-1123; tamper tests :452-468, :639-644 | FIXED |
| m11 | LaunchRefused / CentralFTRefused → exit 2 :1276; futility + Phase-1 A* records the WA `common` :1181-1182 (CLI test :689-695); handoff §5a filled | FIXED |
| review §4 alignment | cross-module test (labels of all WA / edge / FINAL cells = launch_fl's; RECIPE_KEYS, P2_ARMS; every cell passes `check_p2_recipe`) :195-201; `P2_COMMON_FIELDS` :167-170 (knobs removed); adapter checks RUN_SUMMARY.p2 :657-659 | FIXED |

### R1–R12 vs lr_screen.read_run (lr_screen.py:320-397), MEASURED by reading both and by probe
| read_run check | P2 counterpart | faithful? |
|---|---|---|
| three files readable :324-329 | R1 (ARM_SPEC) :688, R4 :703-704 | yes |
| status DONE_MAIN :332-333 | R4 :705-706 | yes |
| arm ∈ arms, role CALIBRATION, seed 2026 :334-336 | R2 :696-698 (taken from ARM_SPEC, before any value file) | yes (stricter order) |
| digest = RUN_SUMMARY = ARM_SPEC, side.arm :337-338 | R1 :692-695 + R5 :707-708 | yes |
| lr_mult / lr_peak_base / spec_tuning / check_warm_spec / lr_peak_local = solver LR :339-351 | R3 `check_p2_spec` :699-702 (spec == `p2_arm_spec` of its recipe: lr_mult 1, θ_pre identity, solver LR = c, frozen tables) + adapter solver-LR check :653-656 | yes |
| 12 marks :352-354 | R6 :710-712 | yes |
| exactly one PB E2E macro mrr@20 row :355-359 | R7 :713-717, + CALIB manifest :719-721 | yes (stricter) |
| role / run_label / method / digest / fl_lr_mult :361-364 | R8 :722-726 | yes |
| tuning labels: the **set** of listed fl_* keys in the row == want :365-372 | R9 :727-731: each expected key equal (one-directional) | **N1** |
| non-finite → excluded :376-386 | R10 :739-742 | yes |
| value_status OK :387-388 | R11 :744-745 | yes |
| value = eval at best mark; schedule_mark = best/2 :389-392 | R12 :746-751 | **N2** |

### New MINOR findings (fix before the first real selection file; Ops verifies by diff + tests)
- **N1 — R9 checks only one direction (MEASURED).** A PB row that carries an extra Phase-1 label is **accepted** by
  `read_p2_run` (probe: `fl_server_tau`, `fl_warm_lr_factor`, `fl_local_batch`, `fl_warm_server_lr_factor`). read_run's
  set comparison refuses the same row (probe on a Phase-1 row). Fix (p2_select.py:727-731): also refuse any key of
  read_run's listed label set (lr_screen.py:366-369) that is present in the row but absent from `want_rl`. Add one test.
- **N2 — R12 schedule_mark bypass (MEASURED).** `_value_or_nan` turns a missing or null `schedule_mark` into NaN, and
  `abs(NaN − x) > 1e-12` is False. Both rows are **accepted** in the probe; a wrong value of 1.0 is refused. read_run
  (lr_screen.py:391) raises on a missing mark. Fix (p2_select.py:749): require a finite number. Add one test.

### Open items for Ops (non-blocking for this code; record by DL row)
- **O-A (worker Q10): a diverged C1 run.** It never reaches DONE_MAIN (TrainingDiverged), and central_ft's strict JSON
  refuses a NaN in RUN_SUMMARY. The C1 file therefore cannot record that cell as "excluded and recorded" (§4.2 → §5.6;
  §9.5): it is refused as a missing run. Either P2-05 provides a value-free DIVERGED record that read_c1_run returns as
  `finite False`, or Ops rules before the C1 file is written if any C1 run diverges.
- **O-B: one DL interpretation row** for the literal §5.7 readings:
  - no corner run;
  - EDGE_UNRESOLVED only when the union winner is an extension cell;
  - a non-finite extension, or a union winner that moved to a base run on an un-extended edge, is recorded in `edge.notes`
    only and stays EDGE_RESOLVED.

  Also have tools/p2_report.py print `edge.notes` beside the Q results (worker Q11).
- **O-C: zero-savings units.** `cap_drop_plan` lists a unit with savings 0 as dropped while the total exceeds the cap
  (:589-596). A unit that has already run must never be declared not run. Build the projection before those units run.
- **O-D: C1 device pin.** C1 candidates carry `device: None` (:826), because the central_ft record exposes no device pin.
  `_same_eval` therefore no longer compares C1 device pins (central_ft verifies run identity).
- **Selection-file sha binding.** Every selection file records the launch_fl / central_ft / p2_select / lr_screen sha256
  at write time (:876-879). The P2 lock must bind the final ones (launch_fl is still changing under P2-01+02).
