# Sonnet C report -- eval_full.py / baselines_full.py / lock_and_eval.py / test_eval_full.py

## Files delivered (all under `code/`, uploaded to `$W/code/`)
- `eval_full.py` -- `rank_targets`, `aggregate`, `evaluate_model`, `per_user_metric`, `paired_bootstrap`.
- `baselines_full.py` -- POP-full/TRANS-full fit + scoring, three swappable readers (CompactStore/parquet/pilot cache), regression gate, `full-vs` CLI.
- `lock_and_eval.py` -- `lock` / `confirm` / `test` subcommands, claim/lock exclusivity, `--dry-run-split` mechanics harness.
- `test_eval_full.py` -- pytest suite (25 tests).
- This report (also uploaded to `$W/agent_reports/sonnetC_eval.md`).

## Commands run and exit codes

```
CUDA_VISIBLE_DEVICES=1 ~/venvs/ppsi-gpu/bin/python -m pytest code/test_eval_full.py -v
  -> 25 passed in 52.0s                                          (exit 0)

CUDA_VISIBLE_DEVICES=1 ~/venvs/ppsi-gpu/bin/python code/baselines_full.py regression-gate
  -> REGRESSION GATE PASS max_abs_diff=1.490e-08                 (exit 0)

CUDA_VISIBLE_DEVICES=1 ~/venvs/ppsi-gpu/bin/python code/baselines_full.py full-vs \
    --eval-batch-pop 16384 --eval-batch-trans 2048 --class-chunk 32768 \
    --out $W/tests_full/C/baselines_full_vs.json
  -> wrote baselines_full_vs.json; total_elapsed_s=1314.7         (exit 0)

lock_and_eval.py mechanics (via a throwaway verification script,
$W/tests_full/C/verify_lock_mechanics.py -- not one of the 4 owned files,
kept on the server for inspection; large scratch copies it made were
cleaned up):
  lock (fresh)                                    -> rc=0 (pass)
  lock (2nd invocation, same --locks-dir)          -> rc=1 (correctly refused)
  lock on synthetic equal-metric run dirs          -> rc=0, selected_arm=F0 (tie -> F0, correct)
  confirm --dry-run-split (stale pre-seeded claim, no --rerun-after-crash) -> rc=1 (correctly refused)
  confirm --dry-run-split --rerun-after-crash "..." -> rc=0 (incident recorded, proceeded)
  confirm --dry-run-split (again, VC_DONE.json now exists) -> rc=1 (correctly refused)
  lock + tampered MODEL_LOCK.json checkpoint sha256 -> confirm rc=1 (correctly refused)
  lock -> confirm -> test (full dry-run chain)     -> rc=0, rc=0
  test --dry-run-split (again, TEST_DONE.json exists) -> rc=1 (correctly refused)
  -> ALL MECHANICS CHECKS COMPLETED                               (exit 0 overall)

Real full_store.CompactStore integration smoke test (Sonnet A's full_store.py
landed mid-task; via $W/tests_full/C/verify_real_store_integration.py, also
throwaway, not one of the 4 owned files) -- a 2-bucket CompactStore
(splits=(TRAIN, VALIDATION_SELECT), channels=F4, buckets=[0,1]):
  CompactStore.build                              -> OK, 3.0s, 1,983,124 rows
  store.decisions("VALIDATION_SELECT","eval") + evaluate_model(pilot F0 ckpt)
                                                    -> OK, n=274,442, user_count=24,872,
                                                       9,166-9,284 rows/s (consistent with the
                                                       10k-cohort measurement above)
  baselines_full.fit_from_store(store, TRAIN loss ds) -> OK, n_fit_rows=1,251,146
  run_pop_baseline(fit, real ds)                   -> OK
  run_trans_baseline(fit, real ds, last_item=...)  -> OK (see "bug found and fixed" below)
  -> ALL REAL-STORE INTEGRATION CHECKS PASSED                     (exit 0)

Frozen pilot tests re-run for sanity (not edited by us): test_evaluator.py,
test_model.py, test_objective.py -> 18 passed, 10 xfailed (unchanged).
```

## Key numbers

### Parity gate (iii) -- pilot 10k VS cohort, eval_batch=512, tolerance 1e-6
- F0 (`T-M_F0_item_only`, update 7814) and F4 (`T-M_F4_rich_side_information`, update 7814):
  end-to-end micro/macro MRR@20 and Recall@20, `n`, and `user_count` all matched the recorded
  pilot `eval_full` record **exactly within 1e-6** (both arms). Two `evaluate_model` calls per
  arm were bit-identical (`ranks` array equality).
- Eval throughput measured directly (`evaluate_model`'s own timing, F0 arm, 10k cohort,
  112,365 rows, eval_batch=512): **9,267 rows/s**, elapsed 12.1s, **peak GPU memory 1.40 GB**
  (well under the ~6 GB guideline).
- **Critical finding (root-caused, not papered over):** this only reproduces within 1e-6 if
  TF32 is disabled (`torch.backends.cuda.matmul.allow_tf32 = False`,
  `torch.backends.cudnn.allow_tf32 = False`) and cuDNN determinism is set -- i.e. exactly
  `train.py:set_determinism()`'s flags. Without it, the discrepancy was ~1e-6 to 9e-6 (fails
  the frozen tolerance) -- large enough to flip a handful of near-tied ranks in the GRU's own
  logits (a matmul-heavy op). `evaluate_model` itself does **not** set these flags (they are
  process-global; ownership belongs to whichever entrypoint calls it first). Two call sites
  needed the fix themselves: `test_eval_full.py` (sets them at import time) and
  `lock_and_eval.py` (a standalone entrypoint, unlike a trainer process that would already have
  set them) -- `lock_and_eval.py`'s own `_configure_threads()` now sets them, verified to close
  the gap from 3.1e-6 to **exactly 0.0** on the same dry-run cohort evaluation.
  **Action for Sonnet B / Opus**: confirm `train_full.py` sets these same flags (it should, via
  `set_determinism()`, before its first `evaluate_model` call) -- otherwise every epoch-end VS
  eval during training is silently ~1e-5-noisy relative to a TF32-disabled process, which could
  matter for the +0.001 Δ-vs-best plateau/patience/best-checkpoint logic (sec. 5-6) on
  borderline evaluations.

### Baselines regression gate (protocol sec. 10.10)
Pilot TRAIN manifest fit -> 10k VS cohort: POP and TRANS end-to-end/closed micro+macro
Recall@20/MRR@20/NDCG@20 and `user_count` all matched `artifacts/BASELINES.json` with
**max_abs_diff = 1.49e-08** (limit 1e-6, identical `n`/`user_count`).

### Full-scale population counts (gate 3), read independently via `baselines_full.py`'s reader (b)
- TRAIN `loss_mask==true` rows: **40,180,608** -- matches `FULL_RUN_PROTOCOL.md` sec. 2 exactly.
- VALIDATION_SELECT `eval_mask==true` rows: **8,826,488** over **790,459** distinct eval users --
  matches protocol sec. 6 exactly. (Independently confirms gate 3's two headline counts a second
  way, via direct Parquet reads rather than the pilot's own cached artifacts.)

### Full-TRAIN fit stats (POP/TRANS-full)
`max_pop=199,677`, `max_trans=52,414`, **`max_composite_score=10,466,122,369`** (>> 2^24 =
16,777,216 -- confirms the pilot's float32-exactness assumption does **not** hold at full
scale, exactly as anticipated; int64 scoring/ranking throughout is not optional here),
`n_trans_pairs_fit=40,135,437`, `n_trans_nonzero_src_tgt=12,281,909`.

### Full-VS POP-full / TRANS-full (fit on all 40,180,608 TRAIN loss rows, evaluated on all 8,826,488 VS eval rows)

| | POP-full | TRANS-full |
|---|---|---|
| end_to_end n / user_count | 8,826,488 / 790,459 | 8,826,488 / 790,459 |
| closed n | 8,776,943 | 8,776,943 |
| micro MRR@20 / Recall@20 | 0.011690 / 0.048525 | 0.140909 / 0.409993 |
| **macro MRR@20** / Recall@20 | **0.021579** / 0.085257 | **0.161638** / 0.445292 |
| warm cohort macro MRR@20 | 0.020075 | 0.161629 |
| unseen cohort macro MRR@20 | 0.026071 | 0.161664 |
| elapsed / throughput | 452.1 s / 19,522 rows/s | 798.0 s / 11,060 rows/s |

OOV target rows (end-to-end minus closed): 8,826,488 - 8,776,943 = **49,545** (rate 0.561%).
Full `slices` diagnostic block (prior_views 0/1/2-4/5-9/10+, prefix_len 1/2/3-4/5-9/10-19/20-49/50+)
was computed for both POP and TRANS and saved in `baselines_full_vs.json`.

Total wall-clock for the full run (model_ready load + TRAIN fit + VS eval-row read + POP + TRANS):
**1314.7 s (~21.9 min)**, modestly over the "~15 min" soft target in the brief -- see "Unresolved
issues" below for why, and the memory-safety tradeoff that drove the batch-size choice.

Result file: `$W/tests_full/C/baselines_full_vs.json` (aggregates + slices only, no per-row dumps).
This is a **reference number for Opus; it is NOT TEST** (fit/eval both restricted to
TRAIN/VALIDATION_SELECT, never VC/TEST).

## Interface assumptions / deviations from FULL_INTERFACES.md (please confirm)

1. `evaluate_model` gained two additive, defaulted keyword-only params beyond the documented
   signature: `class_chunk` (rank-chunk width, default 32768) and `prefetch` (background
   host-batch prefetch, default True). Both are backward compatible with a caller that only
   passes the documented kwargs (`model, store, ds, device, eval_batch, k, return_ranks`).
2. `DecisionSet`/`CompactStore` did not exist when this task started (Sonnet A in progress) --
   `test_eval_full.py` and `lock_and_eval.py` each define their own small, clearly-labelled
   stand-in `DecisionSet`/store (backed by the pilot's frozen `window_cache.CacheReader`) for
   that reason, documented in each file's module docstring as a stand-in, not a test of
   `CompactStore` itself. **`full_store.py` landed partway through this task**, so before
   finishing I additionally wrote a throwaway integration smoke test
   (`$W/tests_full/C/verify_real_store_integration.py`) against the REAL `CompactStore`, built
   small (2 buckets, `splits=("TRAIN","VALIDATION_SELECT")`, `channels="F4"`, ~2M rows, 3s build):
   `store.decisions(...)`, `evaluate_model(...)` with the real pilot F0 checkpoint,
   `baselines_full.fit_from_store(...)`, and `run_pop_baseline`/`run_trans_baseline(...)` all ran
   correctly end-to-end (see command log above) -- `CompactStore`'s public API matches
   `FULL_INTERFACES.md` exactly (verified via `inspect.signature`/`dataclasses.fields` before
   writing the smoke test), and `eval_full.py`/`baselines_full.py` needed **zero changes** to
   work against it. **One real bug WAS found and fixed** this way (see below). The non-dry-run
   `lock_and_eval.py` `confirm`/`test` CLI commands themselves are still unexercised end-to-end
   (that needs a completed `train_full.py` run + a full-scale store build, out of scope here),
   but their two building blocks (`evaluate_model` and the `baselines_full` reader/scorer
   functions) are now verified against the real store, not just against documented signatures.
3. `lock_and_eval.py`'s `lock` needs to decide whether a run's `summary.json` reports a
   "completed" run. `train_full.py` (Sonnet B) does not exist yet, so its actual `summary.json`
   schema is unknown. Implemented defensively (`_is_completed()`): accept an explicit `"status"`
   field if present, else fall back to the pilot's own schema (`best_metric` + `update_number`
   both present). **Please confirm/adjust `_is_completed()` once `train_full.py`'s real schema
   exists** -- this is the one place in this deliverable where key names are a guess rather than
   a spec citation.
4. `aggregate()`'s field naming follows `DecisionSet` (`client_idx`), not the pilot's ad hoc
   `evaluator.evaluate()` "rows" adapter (`client_id`). `test_eval_full.py` explicitly renames
   the T8/T9 fixtures' `client_id` -> `client_idx` when feeding them to `aggregate()`. Not a
   deviation, just flagging the naming seam between the two contracts.
5. `eval_user_key` for the FULL VALIDATION_SELECT population is computed via
   `pyarrow.compute.index_in` against the sorted-distinct-user_id set, rather than
   `build_manifests.py`'s numpy-string-boxing approach (whose own docstring says that approach is
   too slow over the full ~8-9M-row population -- it was only ever applied there to the much
   smaller 10k-user cohort). Same definition (dense position in the sorted distinct user_id list
   over the split's eval_mask rows), a faster implementation; verified correct by reproducing the
   known distinct-user count (790,459, matching `PILOT_MANIFEST.json`'s
   `distinct_eval_users_total`).

## Unresolved issues / findings for Opus

1. **TF32 must be explicitly disabled by every process that evaluates the GRU model**, or
   parity/lock numbers drift by ~1e-6 to 1e-5. Fixed in `test_eval_full.py` and
   `lock_and_eval.py`; **`evaluate_model` itself does NOT set this** (by design -- it's a
   process-global flag, and `train_full.py` should own it once, via `set_determinism()`, per
   protocol sec. 4/13). Please make sure `train_full.py` sets it before its first eval call.
2. **GPU memory ceiling for `baselines_full.py`'s batched ranking.** `rank_targets`'s
   class-chunked comparisons must materialise a real `[eval_batch, class_chunk]` buffer per
   K-sub-chunk (a broadcast/expanded input does not stay zero-cost through a comparison op) --
   `eval_batch=65536` with `class_chunk=32768` tried to allocate 16 GiB and **OOM'd** on GPU 1's
   ~14.6 GB usable capacity (smaller headroom than assumed). Fixed by using `eval_batch=16384`
   (POP) / `2048` (TRANS) with `class_chunk=32768`, both empirically safe and used for the actual
   full-VS run above. This is purely a `baselines_full.py` systems knob (result-invariant, per
   protocol sec. 13); documented in the module (`score_and_rank_pop`'s docstring) so it isn't
   rediscovered the hard way. `evaluate_model`'s own default
   (`eval_batch=1024`, `class_chunk=32768` -> 33.5M elements/chunk) is far below the danger zone.
3. **Host RAM** during the full-VS run briefly reached ~16.3 GB RSS (slightly over the ≤16 GiB
   soft guideline) while the 40.18M-row TRAIN join was materialised alongside the already-loaded
   75M-row `model_ready` table. The shared box has a 64 GiB cgroup limit and ~900+ GiB free at
   the time, so there was no actual contention/OOM, but flagging for awareness if this is re-run
   concurrently with memory-hungry work from A/B. Cheap follow-up not done here for time: drop
   the `model_ready` table's now-redundant columns after each join, or read it bucket-by-bucket.
4. Full-VS run wall-clock (~21.9 min) modestly exceeds the "~15 min" soft target, driven by the
   memory-safe (smaller) batch sizes from finding 2 above; throughput was 19,522 rows/s (POP) and
   11,060 rows/s (TRANS). A faster run is likely possible with a smaller `class_chunk` (e.g. 8192)
   paired with a larger `eval_batch`, trading more Python-loop iterations for smaller peak memory
   per iteration -- not attempted here for time, but the knob is documented and CLI-exposed
   (`--class-chunk`).
5. **Bug found and fixed via the real-store integration smoke test**: `baselines_full.
   run_trans_baseline(fit, eval_rows, device, ...)` originally always read
   `eval_rows.last_item` internally. That works for `EvalRows` (readers b/c, which bundle
   `last_item` as a field), but a real `DecisionSet` (reader a) has no `last_item` field at all
   (by design -- FULL_INTERFACES.md's `DecisionSet` doesn't carry it; the last-item token comes
   from `store.decision_item_tokens(ds)` instead). Calling `run_trans_baseline` with a real
   `DecisionSet` raised `AttributeError: 'DecisionSet' object has no attribute 'last_item'`.
   Fixed by adding an optional `last_item: np.ndarray | None = None` parameter to
   `run_trans_baseline` (falls back to `eval_rows.last_item` when not given, so existing
   `EvalRows`-based call sites -- the CLI, the regression gate -- are unaffected); `lock_and_eval.
   py`'s `_evaluate_four_models` now computes `store.decision_item_tokens(ds)` itself in the
   non-dry-run branch and passes it through. Re-verified after the fix: full real-store
   integration smoke test passes end-to-end, and the full `test_eval_full.py` suite (25 tests)
   and the baselines regression gate (max_abs_diff unchanged at 1.49e-08) both still pass.
6. `lock_and_eval.py`'s non-dry-run `confirm`/`test` CLI commands themselves (real `CompactStore`,
   real VC/TEST) are still **not run end-to-end** -- that needs a completed `train_full.py` run
   (to produce real `summary.json`/checkpoints for `lock`) plus a full-scale store build, both out
   of scope for this task (and VC/TEST must never be loaded in Wave 1 regardless). What IS now
   verified: the real `CompactStore`'s API surface, `evaluate_model` against it, and
   `baselines_full`'s fit/score functions against it (finding 5 above) -- i.e. every building
   block `confirm`/`test` calls has been exercised against the real store, just not through the
   CLI's own claim/lock-file plumbing (which was instead verified via the dry-run harness, see
   above). Also still verified only via `--dry-run-split`: lock success + refusal on a 2nd
   invocation, the tie -> F0 rule, claim exclusivity + `--rerun-after-crash` recovery, a 2nd claim
   refused once `*_DONE.json` exists, checkpoint sha256 mismatch refusal, and the full
   `lock -> confirm -> test` chain producing real bootstrap output.
7. `test_evaluator.py`/`test_model.py`/`test_objective.py` (frozen pilot suite, not owned by us)
   re-run as a sanity check: 18 passed, 10 xfailed, unchanged from before this task.

## Never changed evaluation semantics to make something pass
No tolerance, denominator, macro-grouping, tie rule, or OOV-credit rule was altered to make a
test pass. Both real discrepancies found (the ~1e-6-to-9e-6 parity-gate drift, and the OOM) were
root-caused to (1) an ambient TF32/determinism setting and (2) a batch-size/memory tuning issue,
respectively -- neither was a semantics bug, and both were fixed at the actual cause rather than
by loosening the frozen 1e-6 tolerance or the evaluation definitions.
