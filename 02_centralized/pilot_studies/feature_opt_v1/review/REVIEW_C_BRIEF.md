# Review C — actual code before any study training (SASREC_FEATURE_OPT_V1)

You are the owner's delegated senior reviewer. The owner said: "have Fable review everything and do
what is best without coming back to me". Your verdict is binding and executed by the lead (Opus).
- Read-only. Do not execute, edit or delegate.
- The first line of your answer must be `ROUTING: <exact model id from your system prompt>`.
- Keep the answer to ≤ 1,600 tokens.

## Your earlier decision (binding plan)
`F:\Queens University\Microsoft Project\eda\.local_ai\feature_opt_v1\review\AUTO_DECISION_SCOPE.md`

## Frozen protocol and registry (review for consistency with the code)
- `...\feature_opt_v1\FEATURE_EXPERIMENT_PROTOCOL.md`
- `...\feature_opt_v1\config\candidate_registry.json`
- `...\feature_opt_v1\config\BASELINE_RULE.md`

## Code to review (`F:\Queens University\Microsoft Project\eda\.local_ai\feature_opt_v1\code\`)
Read in this order:
1. `fo_env.py`: isolation (it imports only the byte-identical snapshot `code\snap_sasrec`, whose
   hashes are in `snap_sasrec\SNAPSHOT_MANIFEST.json`), plus the path and split guards.
2. `feature_view.py`: the 8 probes, the lean view + `fold_lean_state_dict`, the M1 re-encoding,
   and D3.
3. `fo_models.py`: the wrapper, strict loads and the parameter mapping.
4. `fo_eval.py`, which wraps the audited snapshot evaluator. Reference only:
   `snap_sasrec\reuse\eval_full.py` lines 131–175, 264–275 and 362–446.
5. `probe_runner.py`: the encoding check, scoring, gate 2(a) and lean parity.
6. `screen.py`: the manifest, the schedule and `train_arm`. It mirrors the campaign `train_step` in
   `snap_sasrec\train_sasrec.py` lines 294–309 and 681–710.
7. `stream_fit.py`, `pin_baseline.py`, `finalize_fo.py` (gate 2(b) and the deliverables),
   `fo_queue.py` (the detached supervisor), `protect_snapshot.py`, `protect_diff.py`,
   `run_bounded.py`.

## Independent test evidence (Sonnet B, cannot edit the code)
`...\feature_opt_v1\review\TEST_EVIDENCE_B.md`, plus the tests in `code\tests\test_fo_*.py`.
- The first run was 75 tests, 73 passed.
- The 2 failures were real defects found by Sonnet B:
  1. a `study_path` string-prefix escape;
  2. `write_once` leaving a stray file on a NaN rejection.
- Both are fixed (`fo_env.write_json_once`, `Path.is_relative_to`). The tests are being converted
  to regressions; the final count is in the evidence file.
- Lead's smoke (`scratch\smoke_fv.py`):
  - lean fold parity max|Δlogit| is 1.9e-7, and a broken-relation negative control gives 0.24;
  - D3 at init is exactly equal to CONTROL.
- On the real server data, the parquet schema contains every column the code reads (74 columns).
- The screen manifest sha is identical under numpy 2.4.1 (local) and 2.4.6 (server).

## Lead interpretation choices you must confirm or overrule
1. **M1 refit.** Valid-only mean/std with the RECORDED clip bounds kept, so the fold is exact.
   Valid-only clip quantiles are reported, not applied.
2. **Probe positions.**
   - P4 touches only the current (last) event, plus `current_minus_recent` via the loader rule.
   - P5 and P6 set "unavailable" at every real position.
   - P2 leaves `category_tokens` (the fine category) untouched, because `category_missing` is
     constant 0.
3. **Gate 2(a)** excludes P1, the OOD anchor.
4. **C1_D3.**
   - Its 5 channels are appended to the event numeric block, i.e. `numflag_proj` input columns
     19–23, zero-initialised.
   - The rotation fold and the D3 path are independent.
   - The D3 ratio denominator counts every position ≤ t, including OOV ones.
5. **C2** is the probe transform applied as a permanent adapter in both training and evaluation.
6. **The screen** uses the campaign training store (TRAIN+VS). VS is used only for the cohort
   evaluation.
7. **The pair is NOT implemented** yet. It would need a correction-only review if gate 2(b)
   produces a finalist.
8. **`fo_queue`:**
   - a GPU counts as busy if it has ANY compute process;
   - the queue waits for the campaign's FINAL_STAGE_DONE with no campaign process alive;
   - disk must stay ≥ 8 GiB + the stage peak;
   - it starts only if `review/FEATURE_CODE_REVIEW_PASS.json` exists. The lead writes that file
     from your APPROVE, with the shas.

## Output
```
ROUTING: ...
VERDICT: APPROVE_FOR_ISOLATED_EXPERIMENTS | FIX_REQUIRED | BLOCKED
FINDINGS: critical/high/medium with file:line (or "none")
CHECKS: one line each —
  - active paths never written
  - TRAIN-only fit
  - no future information
  - no unqualified OOV identity comparison
  - no structural-mask ablation
  - no absent value made to look known
  - no double log/scale
  - matched control/exposure/schedule
  - checkpoint mapping strict
  - single-decision CE
  - exact evaluation
  - warm-start labelled exploratory / pair from scratch
CHOICES: rulings on items 1–8
APPROVED_HASHES: "as listed in REVIEW_C_HASHES.json" if approved, else the files that must change
```
Hashes of the reviewed files are in `...\feature_opt_v1\review\REVIEW_C_HASHES.json`.
