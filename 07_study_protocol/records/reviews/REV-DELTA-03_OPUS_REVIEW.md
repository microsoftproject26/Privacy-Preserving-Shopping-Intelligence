# REV-DELTA-03: independent review of GATE-B (B1–B4) and ANA-04 (06L-C analysis items)

- **Reviewer:** Claude Opus 5.5, read-only (the only write is this file), 2026-09-29.
- **Scope:** the deltas in `company/handoffs/GATE-B.md` and `company/handoffs/ANA-04.md`. They were checked against:
  - `contracts/PREG1_AMENDMENT_06L.md`, `06L_C.md`, `06K.md`, `06I.md`, `06E.md`, `06D.md`;
  - `EVALUATION_CONTRACT_V4.md` §7–§9;
  - `company/reviews/MULTI_EXPERT_REVIEW_20260929/00_SYNTHESIS.md` §4, Gates B and C.
- **What was not touched:** no server, no install, no git, no Codex. No RETRO_TEST data and no MAIN RESULT_ROWS were
  read. Only synthetic test blocks were used.

## VERDICT: **ACCEPT_WITH_FIXES**. No BLOCKER.

The statistics are correct where it matters for a confirmatory number:
- the paired user bootstrap on one shared W;
- ratios bootstrapped directly, including R\* and the seed mean of the per-seed ratios;
- no dropped resamples;
- the seed pairing rules;
- the K14 TOST.

The v3 contrast registry matches the amendments. The five new IDs are acceptable (see §3). B1 refuses every tested way
of passing a MAIN-trained checkpoint off as PRE or C_FULL. v4eval and the default digests are unchanged.

Five MAJOR items (M1–M5) should be fixed before Ops freezes the ANA script (`ANA_SCRIPT_FREEZE`). They are small code or
DL-row items and need no rule change. M1 is the most important: it closes an arm-identity gap in the analysis itself.

## 0. BLOCKERS
None.

## 1. Integrity checks (MEASURED)
| check | result |
|---|---|
| sha256 of every file in both handoff tables | all 18 match (retention_report 41f2ab49…, ana_block_index 6584a029…, run.py 400d20a4…, ANA_CONTRASTS_v3 a3d1456d…, reeval 7a86b07d…, dump_rows 4696c4af…, launch_trunc de009063…, and the test files) |
| `evaluator_code_sha256()` (venv) | `61b64e61c1a2e4f5a6fecd42f0ad3b98de26562d13a07758573cc9496b8d962e`, unchanged |
| `launch_trunc.v4trunc_code_sha256()` | `4d766d5b1eb5…`, unchanged (the T41 receipts stay valid) |
| launch_fl / lr_screen / dump_rows_retro | 1ef778d2… / b1e243f4… / 9853cc55…, unchanged. v4eval/\* and v4trunc/\* mtimes are before both deltas |
| ANA_CONTRASTS_v3 self_sha256 | verifies under the UTF-8 compact rule (`ensure_ascii=False`), not under the ASCII rule. State the rule in the DL row |
| K1–K19 rows of v3 vs v2 | texts identical. Only `population` / `source` were added |

## 2. Statistics (§7–§8, 06L-C C3, C13–C15)
- **Paired bootstrap and shared W.**
  - `Engine` builds one W per evaluated view (`BC._plan`). Every arm, seed and contrast on that view reuses it (E7.2).
  - Subsets (the ONNX users, the strata, MAIN \ CALIB) get their own W. The CI is the [2.5, 97.5] percentile. OK.
- **Ratios bootstrapped directly (E7.5).**
  - `ratio()` computes 100·M_b/C_b for each resample, and the seed mean is the mean over seeds of the per-resample
    ratios (06E §9(e)).
  - If any resample has C_b = 0, the CI is UNDEFINED with a count, and no resample is dropped.
  - Marginal CI endpoints are never divided. OK.
- **Headroom R\*** = 100·(arm − LAST)/(C_ref − LAST), computed per resample on the same W.
  - If C_ref ≤ LAST at the point or in any resample, the result is UNDEFINED with a count (no dropping).
  - The C_ref pairs with the arm's seed. C_FULL s2026 is shared for the deployment arms. OK.
- **Harmed share.** It is computed from per-user macro differences (the seed mean of M_u − C_u) over the paired user
  set. The same rule is used for ratios, which is equivalent to M_u/C_u < 1. Harmed, improved and tied shares and the
  quantiles are reported, plus a same-arm s2027 − s2026 null. OK.
- **K14.** TOST at α = 0.05 is implemented as the 90% percentile CI of the seed-mean resamples lying inside [99, 101].
  That is ±1 % relative, registered before any FA_Q8 read. The 95% CI is shown beside it. OK.
- **Seed pairing.** `check_pairs` / `Book.pairs` handle all the cases correctly:
  - same seed, same rule;
  - PRE = one θ_pre (s2026) shared by FA_WARM and FT_PRE seeds (06K, DL 206(2));
  - K20b = both FA_WARM seeds over C_FULL s2026, scoped to that contrast only (06L-C C7);
  - LAST = a seed-free anchor that pairs with any rule and seed;
  - the same role and the same user subset are required;
  - a seed may not appear twice.
- **Registered seeds** match the amendments:
  - K13, K15, K16, K17, K18, K19, K20c and K9b use {2026};
  - every other contrast uses {2026, 2027};
  - a missing registered seed gives INCOMPLETE_SEEDS, and extra seeds go to a DESCRIPTIVE `extra_seeds` row;
  - K12/K13 are CONTROLLED_BUDGET with the 06D §3 disclosure.
- **Holm band.** Family = the registered count (26), including missing K. This is conservative and correct (judgement
  call 5). Step-down levels are α/(m−j+1), and the adjusted p is monotone. It is labelled DESCRIPTIVE. The band has a
  presentation defect (M3).
- **SEED_SENSITIVE and the user+seed interval.** These are descriptive heuristics, and the rule is printed. Acceptable.
- **Repeat / novel stratum.** REPEAT = rankable ∧ LAST tie count 0. This is correct for a recency scorer: a history item
  gets a unique score, and every non-history item falls into the bottom tie. The strata are row-level, each on its own W.
  OK.

## 3. ANA_CONTRASTS_v3 vs the amendments
- **The 26 IDs** are K1–K21, K9b, K9c, K11b, K20b and K20c. S8−S7 is a registered DESCRIPTIVE contrast.
- **Hierarchy.**
  - K2 is primary.
  - Secondary = 06D §4 {K1, K3, K8, K12} + K17 (06I) + K20, K21 (06K) + K11b (C3) + K9b, K9c, K20b, K20c (C7).
  - This matches exactly, and the word "co-headline" is removed (C4).
- **Algebraic dependencies.** Every relation checks out algebraically: K3, K9, K10, K11, K17–K19, K20b and K20c, as
  exact per-seed point identities. K11b = K11 + (LO − LAST) is also correct, but see m6.
- **Wording the agent chose itself.** The IDs K20b and K20c, and the question texts of K9b, K9c, K11b, K20b and K20c,
  are not in 06L-C, which registers formulas only. This is **acceptable**, for four reasons:
  1. The formulas, populations and seeds follow 06L-C C3/C7 exactly.
  2. The texts are the adopted review wording: review 10 R1(c) is reproduced verbatim for K11b, and R2(b)/(c) for K9b and
     K9c. DL 206(1) is the source for K20b and K20c.
  3. No text adds a claim.
  4. The file becomes the registration once the DL row records its sha, which is what 06L-C C4 asks for.
  The DL row should explicitly ratify three readings the agent made:
  - (a) the two IDs and the five texts;
  - (b) **K9b = seed 2026 only**, which is review 10 R2(b) and the panel convention of 06D §8. 06E §2's default would
    otherwise be two seeds;
  - (c) **S8−S7 as DESCRIPTIVE**. 06L-C C4 lists "(C8)" inside the v3 registry without saying "confirmatory", so this is
    a reading, and the family count of 26 depends on it.

## 4. Comparison-key blocking
- **A40 main block.** One eval_device_class, one device_pin_sha256, and status MEASURED_PINNED_DEVICE on every
  non-historical row. KEY_FIELDS = E9.1 / 06E §4 exactly. The query hash and eval_manifest_id are checked per role.
  Unchanged. OK.
- **C_FULL / PRE keys (B1).** Neither the `central_source` labels nor the `source_*` labels are key fields.
  - C_FULL on MAIN keeps its source key (train FULL_TRAIN; retention_report A-F7 requires it).
  - C_FULL on MAIN_TEST takes the gated key with the source train side.
  - PRE on MAIN and every source on a panel take the role key, with the train side verified key-only (server only; the
    handoff discloses this).
  - OK, but see M1.
- **ONNX_CPU sub-block.** Unchanged: one runtime identity and one user view. K16 is confirmatory only there, and the A40
  fallback is DEVICE_MIXED DESCRIPTIVE. OK.
- **BASELINE_CPU sub-block (new).** Every E9.1 field is checked against the main block's values, except
  eval_device_class, precision_id and objective_id. The query hash and eval_manifest_id are checked per role. The sha
  chain RANKS_RECORD → rows / ranks / ties is checked in the index. `evaluate_with_ranks` matches
  `scoring.run_baseline_evaluation` line for line.
  - Pairing LAST (CPU) with A40 rows as `BASELINE_CPU_EXACT_RANKS` is sound. LAST's integer recency scores do not depend
    on the device, and 06L-C C3 asks for exactly this paired Δ(arm − LAST).
  - However, it is a de-facto exception to 06E §4 / E9.1, and K11b is confirmatory on it. Ops should record the
    exception in a DL row (see M4).
- **Minor.** If the block has no main-block entries, `check_baseline_block` checks nothing, because it skips any field
  whose main value is None (m3).

## 5. B1 reeval sources
- **A run claimed by a source must match every field, or it is refused.** A run is claimed by its train manifest,
  run-id prefix, run label or arm label. It must then match:
  - the run-id prefix and the run label;
  - the train and eval providers and the key-base manifests;
  - the variant, recipe and seed;
  - `--ckpt-kind central`;
  - for PRE, the file `endpoint_6.0.pt`.
- **Further checks.**
  - The extension must be additions-only and bind the pair, and the key base's registry self sha must equal the
    extension's.
  - TRUE_COLD is refused.
  - A source arm label on an fl/pf checkpoint is refused.
  - Every way of passing a MAIN-trained checkpoint as PRE is refused: C_SMALL with `--arm-label PRE` fails on the run-id
    prefix, the run label and the train manifest; C_BIG as C_FULL is tested.
- **MAIN pair unchanged.** The MAIN-pair dry-run record is byte-identical to the pre-B1 record.
- **Gaps:** m1 (C_FULL checkpoint files) and m2 (the PRE meta mark).

## 6. B2 / B3 / B4
- **B2.** `numeric_array_field` requires an array with a row axis and a numeric or bool dtype. The field set is the same
  as the retro dump's (tested). `dropped_non_array_fields` is written only when a field was dropped. OK. The owner of
  §8.2 is still unassigned (Ops).
- **B3.**
  - S8 bytes = `per_client_incl_frozen_once.mean`, with the invariant 0 ≤ incl − excl ≤ frozen_bytes_once.
  - The charged-dropped variants add the once component.
  - The curve charges the once component as a step from the first mark and prorates only the training traffic. Early
    marks therefore overstate S8's bytes, which is conservative against S8. The rule is labelled INFERENCE. Disclose it.
  - The index refuses a frozen run under any other arm, and refuses S8 without its record. OK.
- **B4.**
  - `NO_C_REF` before any hit; `NOT_REACHED` only when every h had a C_ref.
  - Study B keeps only the selected cell, exactly one per (panel_type, h); a FULL cell is refused.
  - A/B mixes are refused, and the C_ref must be C_BIG Study A rows of the same seed and panel.
  - No IndexError path remains. OK.

## 7. MAJOR (fix before the ANA freeze; small)
- **M1: PRE (and C_FULL on panels) identity in the analysis.** `PRE_TRAIN_MANIFESTS` accepts MAIN_TRAIN_5_100_131072,
  because PRE on MAIN uses the role key. `check_arm_identity` also accepts method "C" for PRE. As a result, a C_SMALL
  endpoint_6.0 row set (method C, train MAIN_TRAIN) in a PRE-named entry passes the analysis. So does a REEVAL_RECORD
  with arm_label PRE written by the pre-B1 reeval of code_ds3r3, which had no source check.
  - The index does not require `REEVAL_RECORD.central_source.source == arm`.
  - Fix, in `ana_block_index.reeval` and/or `check_arm_identity`: for arm ∈ {PRE, C_FULL}, require the record's
    `central_source.source == arm` and the rows' `central_source == arm`.
  - For PRE, additionally require `source_train_manifest_id == PRETRAIN_TRAIN_5_100_NONMAIN`.
  - Refuse method "C" for PRE.
  - Add a negative control: C_SMALL rows relabelled PRE must be refused.
  - This is the analysis-side half of "no MAIN-trained checkpoint accepted as PRE".
- **M2: FA_WARM identity.** The index remaps a run whose `RUN_SUMMARY.arm == "FA"` to FA_WARM purely by the directory
  name `FA_WARM_s<S>`. The real launcher writes `arm = "FA_WARM"` / `"FA_WARM_FROZEN"`, with `arm_spec.init_from`.
  - Consequence: a cold-start FA run copied into an `FA_WARM_s2026/` directory becomes FA_WARM, which feeds K20, K20b,
    K21 and S8−S7.
  - Fix: FA_WARM, FA_WARM_FROZEN and DP8_FROZEN require `arm_spec.init_from` (a PRETRAIN_FOR_FL identity); every other FL
    arm must not carry it. This is the same two-way rule B3 already applies to `frozen_item_tables`.
  - Note: DP8_FROZEN has no launch path in launch_fl yet (freezing is limited to FA_WARM_FROZEN). The analysis admission
    is ahead of the launcher.
- **M3: Holm band vs Holm decision.** After the first non-rejection (`stop`), later ranks still get a band at their own
  level α/(m−j+1). A tail contrast can therefore show a band that excludes the reference while `holm_reject` is False.
  - Example: m = 26, rank 25 with p = 0.02 ≤ 0.025 after an earlier failure.
  - Fix: after `stop`, give every non-rejected K the band at the level of the first non-rejection, α/(m−r) with r the
    number of rejections (the Strassburger–Bretz compatible bound). Alternatively, print only `p_holm` / `holm_reject`
    beside the Bonferroni band.
  - Add a test that the band and the decision agree.
- **M4: DL row for the analysis exceptions.** One row should ratify:
  - (a) the BASELINE_CPU pairing (LAST × A40, `BASELINE_CPU_EXACT_RANKS`) as the 06L-C C3 exception to 06E §4 / E9.1,
    with K11b confirmatory on it;
  - (b) the §3 readings (K20b/K20c IDs and texts, K9b seed 2026, S8−S7 DESCRIPTIVE, family 26);
  - (c) the v3 sha a3d1456d…, self-sha under the UTF-8 compact rule.
  - DL 229 records the v3 sha, but "subject to REV-DELTA-03" must be closed by this row.
- **M5: TEST must not run on a superseded registry.** `load_contrasts` still accepts format v2 (K1–K19) on
  `--split TEST`. On TEST, K9b, K9c, K11b, K20b and K20c would then silently become DESCRIPTIVE, and the Holm family would
  drop to 19.
  - Since 06L-C C4, the registry is v3.
  - Fix: refuse any format other than PPSI_V4_ANA_CONTRASTS_3 on TEST. v2 may stay a VALIDATION dry check only.

## 8. MINOR
- **m1.** `CENTRAL_SOURCES["C_FULL"].checkpoint_files = None`, so any file under checkpoints/ is accepted, for example
  `endpoint_8.0.pt` after the 06D §7 continuation, or an intermediate best@. The handoff says "best.pt or
  endpoint_6.0.pt". Restrict C_FULL to those two files and bind the rule to the file (PB ↔ best.pt, CB ↔
  endpoint_6.0.pt).
- **m2.** For PRE, check that the checkpoint meta `efe` / `mark` = 6.0, not only the file name. Optionally tie
  `--expect-sha256` to `PRETRAIN_FOR_FL_INIT.json fl_init.sha256` in code rather than by operator procedure. The same
  applies to FT_PRE's base: the index checks only the LAUNCH start run-id prefix.
- **m3.** `check_baseline_block` checks nothing when the block has no main-block entries. Also check the baseline rows'
  key fields against each other.
- **m4.** The claim register uses "not resolved" instead of the contract's E7.6 vocabulary "no detectable difference".
  Align it, or print both.
- **m5.** K20c's forbidden entry "'…needs no federated training' unless K21 includes 0" implicitly allows an
  absence-of-evidence claim, which contradicts the general rule. Make it unconditional unless a registered equivalence
  test exists.
- **m6.** Listing K11b under `algebraic_dependencies` together with the multiplicity sentence ("…are not independent
  evidence") mislabels K11b. LO − LAST is not a registered K, so K11b carries new evidence. Annotate it as "shares LO with
  K11".
- **m7.** Disclose three things:
  - DP8_FROZEN's "z = 0 twin = S8" (06L-C C17, followed literally) differs from DP8_FROZEN in cohort and round count
    (1,024 / 384 vs the FA plan). R(DP8_FROZEN / S8) therefore mixes DP with that change.
  - The S8 bytes step charged at the first mark (§6).
  - The raw-event bytes level uses TRAIN decisions as the event count.
- **m8.** Still open, not in this delta: C10 (matched-h parity M(h)/C_BIG(h) and the persistence + CI parity rule) has no
  code in retention_report / launch_trunc. It must exist before any "events to parity" sentence. The GATE-B EVENTS_TO_X
  `self_sha256` is computed before the JSON round trip (pre-existing, disclosed). The receipts in
  `tests_int/receipts/TEST_RUN.json` and `tests_trunc/receipts` still carry the old shas; re-run `run_all.py` when the
  tree is deployed.

## 9. Tests (MEASURED; venv python, `-B -p no:cacheprovider`, TMP/TEMP/--basetemp `.local_ai/_pytest_tmp/REV-DELTA-03/<suite>`, run once each, sequentially)
| suite | exit | result | handoff claim |
|---|---|---|---|
| tests_ana | 0 | 142 passed, 43 xfailed (403 s) | 142 / 43, matches |
| tests_int | 0 | 262 passed, 78 xfailed (289 s) | 262 / 78, matches |
| tests_base | 0 | 64 passed, 10 xfailed | 64 / 10, matches |
| tests_dev | 0 | 77 passed, 5 skipped, 14 xfailed | 77 / 5 / 14, matches |
| tests_trunc | 0 | 48 passed, 13 xfailed | 48 / 13, matches |

Outputs: `_pytest_tmp/REV-DELTA-03/<suite>/out.txt` and `SUMMARY.txt`. The negative controls were not re-run with
`--runxfail`. This review relies on the handoffs' record of that run, together with a reading of the NC mutants in
test_ana04, test_gate_b and test_gate_b_reeval. No test checks the gaps in M1, M2, M3 or M5.

## 10. Required before the freeze (order)
1. M1 and M2 (index / identity), then M3 (Holm band) and M5 (TEST refuses v2), each with a negative control.
2. M4 DL row.
3. m1 and m2 in reeval before the first real C_FULL / PRE reeval. The dry runs in GATE-B.md are fine to run now.
4. A short fix check (Opus, read-only) on those deltas. Then Ops freezes the ANA script together with the
   `--claims-only` output.
