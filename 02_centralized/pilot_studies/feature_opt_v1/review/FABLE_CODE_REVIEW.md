# FABLE_CODE_REVIEW — Review C, pass 1 (owner-delegated), 2026-09-25

- **Runtime routing** (from the transcript): ['claude-fable-5-1']
- **Agent:** sasrec-fable-reviewer (Read/Grep/Glob only)
- **Brief:** review/REVIEW_C_BRIEF.md
- **Reviewed hashes:** review/REVIEW_C_HASHES.json

## Verbatim

ROUTING: claude-fable-5-1
VERDICT: FIX_REQUIRED

FINDINGS:
- HIGH `code\finalize_fo.py:91-114,117` — TRANSFORM_STATE/SELECTED_FEATURES assert "reproduce the incumbent's predictions to float rounding" unconditionally: no threshold on LEAN_PARITY (`max_abs_logit_diff`, `ranks_identical_fraction`), and `lean=None` (LEAN_PARITY.json absent) still writes the claim [evidence]. Also nothing gates the fold's assumed relations: `audit/DUPLICATES_FULL.json` mismatch counts and `local_hour_rotation_max_residual` are never read [evidence]. Fix: require LEAN_PARITY.json; write the hygiene deliverable only if all DUPLICATES_FULL counts are 0, rotation residual ≤ 1e-6, max|Δlogit| ≤ 1e-3 and rank-identical ≥ 0.999 (thresholds [inference]); otherwise emit `PARITY_FAILED`, no equivalence claim.
- HIGH `code\probe_runner.py:54-57,81` — `check()` passes vacuously when `n == 0`; `ok = not bad` ignores counts. The protocol's "every injected encoding is compared with real rows; any mismatch aborts" is not met if the cohort has no row in a state (e.g. `cur_price_missing_supported`) [evidence]. Fix: `ok` requires every count > 0 (floor ≥ 100 [inference]); otherwise abort with `UNVERIFIED_ENCODING`.
- MEDIUM `code\screen.py:97` — `label_smoothing=0.0` hard-coded; the campaign uses `entry["label_smoothing"]` (`snap_sasrec\train_sasrec.py:391,697`) [evidence]. Fix: assert `float(ch["registry_entry"]["label_smoothing"]) == 0.0` before training.
- MEDIUM `code\fo_queue.py:64-74,84` — `gpu_busy` fails open: a compute app whose bus id is not in the `--query-gpu` map is keyed under the raw id and never matches `str(g)`; an `nvidia-smi` failure yields "free" [evidence]. Fix: non-zero rc or any unmapped app → all GPUs busy.
- MEDIUM `code\probe_runner.py:90-92,109` — retry hazard: attempt 2 hits `FileExistsError` on `PROBE_ENCODING_CHECK.json`; if the run dies after `GROUP_PROBE_RESULTS.json` is written, the early return skips lean parity forever and finalize silently proceeds with `lean=None` [evidence]. Fix: skip the encoding write if present; run the lean-parity block whenever LEAN_PARITY.json is missing.
- MEDIUM tests — no test of `finalize_fo` gate 2(b) (threshold, CI, R@20 guard, single finalist) with fabricated results and negative controls [evidence: no `test_fo_finalize*.py` in `code\tests`].
- LOW `code\fo_env.py:51` — `assert_isolated` uses the same string-prefix idiom that was a defect in `study_path`; harmless today (live path is not a prefix) [inference]; use `is_relative_to`.
- LOW `code\feature_view.py:299-301` — the affine shift is inferred from clip bounds, never cross-checked against the measured shifts (`stream_fit.AFFINE` 0.0/1.0) [evidence]; add the assertion.

CHECKS:
- active paths never written: PASS [evidence] — all writes go through `study_path` / `F/`; snapshot `paths.py:19-20,39-41` puts snapshot `reuse`/`vendor` on sys.path; campaign root `S` is unused by the study; dataset read-only.
- TRAIN-only fit: PASS [evidence] — `stream_fit.py:74-80` fits TRAIN valid rows with recorded clips; VS drift-only; D3 no fit; manifest from TRAIN; P8 cell TRAIN-modal.
- no future information: PASS [evidence] — `d3_features` `tril(-1)`/cumsum causal, tested (`test_d3_future_never_changes_past`); probes touch context only; P4 uses `lengths-1`.
- no unqualified OOV identity comparison: PASS [evidence] — `known = real & tok>=3` on both sides of `eq` (`feature_view.py:348-351,359`), tested.
- no structural-mask ablation: PASS [evidence] — `attention_mask`/`lengths` never modified; P1 toggles `side_enabled` only, labelled OOD.
- no absent value made to look known: PASS [evidence] — every probe sets the flag/mask with the loader's re-zero value; M1 repair keeps mask 0 with 0.0. Caveat: the HIGH encoding-check vacuity above.
- no double log/scale: PASS [evidence] — `enc_event` clips+z once on raw; sin/cos unscaled; M1 inverse `x*old_std+old_mean` stays in log1p space (`to_lean`, tested).
- matched control/exposure/schedule: PASS [evidence] — same pinned weights, fresh AdamW, same manifest (sha checked at full scale), `lr_at` warmup/cosine, `train_arm` mirrors `train_step` (loss_mask, n_total, clip, set_lr, step, recenter); seed 2026 after build. Caveat: smoothing assertion (MEDIUM).
- checkpoint mapping strict: PASS [evidence] — `strict=True`, `_resize_inputs` then load, `parameter_mapping`, refusal test.
- single-decision CE: PASS [evidence] — `ce_sum` with `loss_mask`, one-target check.
- exact evaluation: PASS [evidence] — snapshot `evaluate_model` under `strict_eval`, exhaustive K, OOV zero credit in denominator (tested); VS-only store either yields identical windows or raises (`full_store.py:260`).
- warm-start labelled exploratory / pair from scratch: PASS [evidence] — labels in `screen.py:173`, registry `pair.otherwise`; pair not implemented.

CHOICES:
1. M1 refit: CONFIRM. Recorded clips kept → affine on available rows; fold `a·x' + (c−z0)·mask + z0` is algebraically exact (`feature_view.py:320-327`) [evidence]. Report, do not apply, valid-only quantiles.
2. Probe positions: CONFIRM all three. P4 last-event only with `current_minus_recent` via the supported rule; P5/P6 at every real position; P2 leaves `category_tokens` (MISSING row untrained) [evidence].
3. Gate 2(a) excludes P1: CONFIRM (`probe_runner.py:141`).
4. C1_D3: CONFIRM. Columns 19-23 zero-init, `_side_residual` concat order matches `d3_state_dict`; fold and D3 independent; ratio denominator `t+1` incl. OOV is bounded and causal — record this definition in the registry text [inference].
5. C2 as permanent adapter in train+eval: CONFIRM (`FOModel.forward`, `build_arm`).
6. Screen on the training store, VS cohort-only: CONFIRM (`build_training_store`, TRAIN loss decisions).
7. Pair not implemented: CONFIRM; correction-only review required before any pair training.
8. fo_queue: CONFIRM the rules as written, subject to the fail-closed GPU fix. Pass file must be written only after the fixed files are re-hashed.

APPROVED_HASHES: not approved as a set. Files that must change: `code/finalize_fo.py`, `code/probe_runner.py`, `code/screen.py`, `code/fo_queue.py` (plus optional `code/fo_env.py`, `code/feature_view.py` low items, and a new `code/tests/test_fo_finalize.py`). All other entries in `F:\Queens University\Microsoft Project\eda\.local_ai\feature_opt_v1\review\REVIEW_C_HASHES.json` are acceptable as listed; re-submit the changed files' hashes for a correction-only check before `FEATURE_CODE_REVIEW_PASS.json` is written.


---

# Review C, pass 2 (correction-only), 2026-09-25T11:43:32Z

- **Runtime routing:** ['claude-fable-5-1']
- **Brief:** review/REVIEW_C2_BRIEF.md
- **Hashes:** review/REVIEW_C2_HASHES.json

ROUTING: claude-fable-5-1
VERDICT: APPROVE_FOR_ISOLATED_EXPERIMENTS
remaining blockers: none

Per-finding verification (all paths under `F:\Queens University\Microsoft Project\eda\.local_ai\feature_opt_v1\`):

1. HIGH parity claim — FIXED [evidence]. `code\finalize_fo.py:35-46` gates FULL_POPULATION label, all non-residual counts == 0 (and non-empty), residual <= 1e-6, max|dlogit| <= 1e-3, rank-identical >= 0.999; `:53-54` raises on missing LEAN_PARITY.json; `:124` status `PARITY_FAILED_NOT_ADOPTED` and `:134-135` `equivalence=None` unless PARITY_OK; `:112` hashes `fv.normalizers_path()`.
2. HIGH vacuous encoding check — FIXED [evidence]. `code\probe_runner.py:33` floor 100; `:82-84` `unverified` list, `ok = not bad and not unverified`; `:122` runs on `store.decisions("VALIDATION_SELECT","eval")`; `:125` rc 5.
3. MEDIUM label smoothing — FIXED [evidence]. `code\screen.py:162-163` raises before `load_weights_file`/training.
4. MEDIUM GPU fail-open — FIXED [evidence]. `code\fo_queue.py:67-89`: exception, rc != 0, unparsable query line, empty map, malformed app line or unmapped bus all return every-GPU-busy. Note [inference]: `ready()` only knows GPUs "0"/"1"; the plan only requests 0/1 (`:162-171`), so no gap today.
5. MEDIUM retry hazard — FIXED [evidence]. `code\probe_runner.py:100-102` M1_FIT required up front; `:119-123` encoding check reused; `:136-141` probes skipped only if GROUP_PROBE_RESULTS exists, baseline ranks reloaded and sha + decision_id verified; `:163-164` C2 written if missing; `:166` lean parity whenever LEAN_PARITY.json missing. Residual, non-blocking [inference]: a crash inside stage 2 after `BASELINE.npz`/`P*.npz` are written but before GROUP_PROBE_RESULTS would make attempt 2 fail loudly if `E.save_ranks` is write-once (fo_eval.py unchanged, not re-read); that is fail-closed, not silent, so acceptable.
6. MEDIUM finalize tests — FIXED [evidence]. `code\tests\test_fo_finalize.py`: 6 `hygiene_parity` tests each isolating exactly one failing check (`_assert_only_this_check_false`), 7 end-to-end tests covering both sigma_hat threshold branches, CI, R@20 guard, larger-delta finalist, decision_id mismatch raise, missing LEAN_PARITY raise. `test_fo_probe_runner.py` (10): gate_2a sub-conditions + P1 exclusion; encoding_check unverified on tiny store, ok=True on consistent fixture with floor 1, single violation -> ok False. `test_fo_queue.py` (10): four fail-closed cases + normal-mapping control; `ready()` gating. `test_fo_screen.py:238-247` guard raises / 0.0 passes to sentinel. `test_fo_feature_view.py:445-464` shift cross-check + control. "125 pass" is the lead's claim in `review\TEST_EVIDENCE_B.md:46-49`; I did not execute anything [evidence of claim only].
7. LOW — FIXED [evidence] `code\fo_env.py:51` `is_relative_to`.
8. LOW — FIXED [evidence] `code\feature_view.py:229,306-307`.
9. Choice 4 — RECORDED [evidence] `config\candidate_registry.json:31`.

Caveat [inference]: `hygiene_parity` assumes `DUPLICATES_FULL.json.checks[split]` is a flat {name: int} map plus the residual key; a differing real schema would raise (fail-closed), not pass.

Scope: approval covers isolated experiments only; the pair stage remains unimplemented and needs its own correction-only review. I did not recompute file hashes.

APPROVED_HASHES: "REVIEW_C2_HASHES.json files" (`F:\Queens University\Microsoft Project\eda\.local_ai\feature_opt_v1\review\REVIEW_C2_HASHES.json`)
