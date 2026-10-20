# Sonnet A — data path & storage (Wave 1) — result report

Scope: `code/full_store.py` (`CompactStore`/`DecisionSet`), `code/test_full_store.py` (launch
gates), `code/verify_frozen.py`, `code/disk_guard.py`, `code/bench_store.py`. No edits to any
vendored/frozen module (`vendor/rees46_eda/*`, `window_cache.py`, `index_utils.py`,
`build_manifests.py`, `model.py`, `objective.py`, `evaluator.py`, `train.py`). No writes inside
`/home/jovyan/shared_data/`. No git commands. No teammate files touched.

**All acceptance gates PASS.** `overall_pass: true` in `tests_full/A/gates_A.json`.

## Files delivered

| File | Purpose |
|---|---|
| `code/full_store.py` | `CompactStore` / `DecisionSet` per `FULL_INTERFACES.md` §1 |
| `code/test_full_store.py` | All launch gates (i), (ii), negative controls, T11, invariants, normaliser check |
| `code/verify_frozen.py` | sha256 re-check of all 409 `FILES.json`-listed files |
| `code/disk_guard.py` | `free_bytes`, `check_floor`, CLI |
| `code/bench_store.py` | Build time / RSS / batch-latency measurements |
| `agent_reports/sonnetA_data_path.md` | This report (also at `@W/agent_reports/`) |

Outputs on the server (not duplicated locally, per the "no disk cache" / scratch-dir rules):
`tests_full/A/gates_A.json`, `tests_full/A/verify_frozen.json`, `tests_full/A/bench_store.json`.
Logs: `logs/A_test_full_store.log`, `logs/A_verify_frozen.log`, `logs/A_bench_store.log`,
`logs/A_bench_seq.log`, `logs/A_edge_scan_full.log`.

## `full_store.py` design summary

`CompactStore.build()` streams each `model_ready/user_bucket=<b>/features.parquet` bucket ONCE
via pyarrow column projection (7 columns for `channels="F0"`, ~63 for `"F4"` — never the full
~80-column file), reproduces `BucketStore`'s exact row order
(`np.lexsort((position_in_segment, segment_index, user_id))`, applied to the `splits`-filtered
subset — proved equivalent to filtering after a full-bucket lexsort, since lexsort is stable and
a retained row's sort key never depends on a dropped row), and computes every channel value with
the SAME calls the vendored `collate()` uses (`Normalizers.apply`, `MISSING_FLAG_OF`,
`USER_MISSING_FLAG_OF`, imported not copied). Decision metadata is joined to feature rows
per-bucket via sort+`searchsorted` (no Python dict, no per-decision loop), and `end_row`/`length`
are derived from the same per-bucket join so that no permanent per-row `position_in_segment` or
`decision_id` array needs to survive past the bucket that produced it — only the compact channel
arrays (indexed by global row) and the small per-(split) decision-level arrays (only rows with
`loss_mask|eval_mask` true) are retained. `eval_user_key` is computed lazily, on first
`decisions(split, "eval")` call, via the same pyarrow `unique`+`sort_indices`+`searchsorted`
construction as `build_manifests.build_vs_cohort`, scanning the FULL split (all 64 buckets,
regardless of the store's own `buckets=` restriction) — the raw `user_id` arrays used for that
lookup go out of scope (and are never logged) as soon as the integer keys are built.
`user_context_masks` is stored `uint8` only after an explicit prepass proves every value is
exactly `0.0`/`1.0` (confirmed true; `float32` is the coded fallback, untested since the frozen
data never triggers it).

No interface deviations from `FULL_INTERFACES.md` §1 — every field, method signature and dtype
matches as specified (`DecisionSet.take`, `.sha256()`, `CompactStore.decisions/batch/
decision_item_tokens/build`, `EXPECTED_VERSION_HASH`, `L=50`).

## Key numbers

| Measurement | Value | Expected / target |
|---|---|---|
| F4 store rows (TRAIN+VALIDATION_SELECT) | **63,690,404** | 63,690,404 ✓ |
| F4 build time (n_threads=3, all 64 buckets) | **160.5 s** | — |
| F4 build time (n_threads=1, sequential) | 208.4 s | — |
| F4 settled RSS after build | **17,478 MB (17.07 GiB)** | ≤ 12 GiB target — **exceeded** |
| F4 peak RSS during build (`ru_maxrss`, n_threads=3) | **27,600 MB (26.95 GiB)** | — |
| F4 peak RSS during build (n_threads=1) | 25,915 MB (25.31 GiB) | — |
| F0 store rows | 63,690,404 | same row set as F4 (channels only differ) |
| F0 build time | **51.1 s** | — |
| F0 RSS delta over baseline | **1,987 MB (1.94 GiB)** | ≤ 1.5 GiB target — **modestly exceeded** |
| `batch()` B=256 random TRAIN, warm (F4) | **1.44 ms** | — |
| `batch()` B=1024 sequential VS (F4) | **5.79 ms** | — |
| `decision_item_tokens` over all 40.18M TRAIN loss decisions (F4) | **0.52 s** | — |
| TRAIN loss decisions | **40,180,608** | 40,180,608 ✓ |
| VS eval decisions | **8,826,488** | 8,826,488 ✓ |
| VS distinct eval users | **790,459** | 790,459 ✓ |
| `catalogue.product_idx == arange(158486)+3` | **True** | ✓ |
| `verify_frozen.py`: files verified / missing / mismatched | **409 / 0 / 0** | 409/0/0 ✓ |
| `verify_frozen.py`: total bytes / elapsed | 9,155,066,376 / 7.69 s | — |
| Live free disk at time of run | 23.68 GiB free | ≥ 10 GiB target, ≥ 8 GiB floor ✓ |

## Gate-by-gate results (`tests_full/A/gates_A.json`)

- **class_map**: pass (158,486 rows, `product_idx == arange+3` exact).
- **counts**: pass — all four counts exact matches (table above); `store.decisions("VALIDATION_CONFIRM"/"TEST", ...)` both raise `ValueError` (never loaded).
- **gate (i) F4** (byte-exact vs pilot `window_cache` cache): pass. All **1,000,000** TRAIN-manifest rows and all **112,365** VS-cohort rows compared, all **19** `CacheReader`/`collate()` arrays each, in 50,000-row chunks, `np.array_equal` on `uint32`-viewed floats (catches −0.0/NaN). Zero mismatches. `eval_user_key` mismatches vs the cohort parquet: **0**.
- **gate (i) F0 subset**: pass. 20,000 random TRAIN + 20,000 random VS decisions, 10 F0 keys (`item_tokens, attention_mask, position_ids, lengths, last_valid_index, target_class, loss_mask, eval_mask, decision_id, client_id`), zero mismatches.
- **gate (ii)** (byte-exact vs fresh `DecisionWindowLoader.collate()`): pass. **4,206** decisions compared (≥4,096 required) across **10** buckets (≥8 required) — 4,200 random (seeded) + edge cases. Found and verified: position 1, 49, 50, >50; first event of a segment; a window containing a `loss_mask=false` context row; an unseen-user (`client_idx==2`) decision. Zero mismatches on any of the 19 arrays.
  - **Two named edge cases are exhaustively verified ABSENT from the frozen dataset**, not a test gap: "last event of a segment" as a servable (`loss_mask|eval_mask` true) decision, and "a segment/window crossing TRAIN→VALIDATION_SELECT". Searched across **all 64** `model_ready` buckets (not just the 10 used for the random sample) via a vectorised full-bucket scan; found zero instances of either. Reasoning, not just absence: (1) this dataset's target rule is "the next event within the same `(user_id, segment_index)` group" — the exact grouping key `BucketStore`/`collate()` itself uses — so a segment's last event structurally can never carry a valid target, for ANY segment, not only ones ending near a split boundary; (2) no `(user_id, segment_index)` group was found containing rows from both `TRAIN` and `VALIDATION_SELECT`, so no window any store could ever build crosses that fence. Both are treated as allowed misses (`edge_cases_unexpectedly_missing: []`); everything else in the requested edge-case list was found and passed.
- **negative controls**: pass. A window shifted by one row (`positions+1`) is detected as different from the correct window (`shift_by_one_detected: true`); a perturbed normaliser constant (`log1p_price` mean +5.0, injected via a monkey-patched `Normalizers.load` into a fresh 2-bucket store) is detected as a numeric-array mismatch against the correctly-normalised store (`perturbed_normaliser_detected: true`).
- **T11 cold-user fixture**: pass. Smallest 8 VS `decision_id`s with `client_idx==2` and smallest 8 with `client_idx>=3` both found (exactly 8 each, directly from `decisions_index`, independent of the store). An unseen user (`client_idx==2`) with 4 VS eval decisions was found via `eval_user_key` grouping (never raw `user_id`) whose `has_history_mask` sequence is `[0.0, 1.0, 1.0, 1.0]` — 0 at the first decision, 1 from the second on. The vacuous `post_freeze_smoke.py:101` pattern (`bool(np.any(client_id==OOV)) or True`) is shown to unconditionally return `True` on a deliberately broken input (a `client_id` array with **no** OOV entries at all — simulating exactly the bug the check exists to catch), while the real check (`bool(np.any(client_id==OOV))`) correctly returns `False` on that same broken input. `post_freeze_smoke.py` itself was not touched.
- **feature invariants** (TRAIN sample, n=100,000): all six hold with **zero violations**: `prefix_item_count == position_in_segment`; `prefix_representative_count == position_in_segment - 1`; `log1p_prefix_elapsed_s == log1p_segment_elapsed_s`; `is_weekend == (local_weekday_idx ∈ {8,9})`; daypart consistent with `local_hour`; `price_missing ⇒ log1p_price == 0` (225 price-missing rows in the sample, all zero).
- **normaliser check** (full TRAIN population, all 64 buckets, 19 normalised features): pass after correcting the recompute population (see below) — **max relative deviation 3.34e-14** (floating-point-exact), threshold 1e-2.

### A real methodology bug this gate caught (in my own test, not in the frozen data)

My first `gate_normalizer_check` pass recomputed mean/std over **all** `split=='TRAIN'` rows and
got a **0.358 max relative deviation** (`recent_mean_log_price` std off by 36%), which would have
been a hard fail. Investigating rather than accepting or silently "fixing" it: `collate()` already
tells you these two user-context features (`recent_mean_log_price`,
`current_minus_recent_mean_log_price`) are forced to `0.0` whenever their support mask
(`recent_price_supported`) is `0` — a placeholder, not a real observation. Recomputing restricted
to `recent_price_supported==1` brought both features to **~1e-15** relative deviation
(floating-point exact). The same pattern held for the three `MISSING_FLAG_OF`-gated event numerics
(`log1p_price`, `log1p_prev_raw_view_gap_s`, `log1p_item_observed_age_s`): unrestricted, they showed
0.18%–6.9% deviation; restricted to their own "not missing/conflict" mask, all three also landed at
~1e-14. All 19 normalised features now match to floating-point precision. This is evidence the
frozen `normalizers.json` was fit on the semantically-valid subset per feature (not a bug in the
frozen file) — `gate_normalizer_check` now applies the same per-feature exclusion collate() itself
uses (`MISSING_FLAG_OF` / `USER_MISSING_FLAG_OF`) before recomputing.

## `bench_store.py` results

See table above for the headline numbers; full JSON at `tests_full/A/bench_store.json`. Both
`batch()` calls and `decision_item_tokens()` are sub-second even over the full 40.18M-decision
TRAIN set, confirming the store's read path is not a training bottleneck at the target microbatch
sizes (256 train / 1024 eval per `FULL_RUN_PROTOCOL.md` §4/§6).

## Commands run (server, via `tools/lobot.py`), abbreviated

| # | Command | Exit |
|---|---|---|
| 1 | 2-bucket, 10-bucket smoke builds + `decisions()`/`batch()`/`decision_item_tokens()` sanity | 0 |
| 2 | `_find_edge_cases` isolated run, 20 then 64 buckets, to characterise `crossing_train_vs`/`seg_last` | 0 |
| 3 | `verify_frozen.py --out tests_full/A/verify_frozen.json` (detached) | 0, `all_ok: true` |
| 4 | `test_full_store.py --out tests_full/A/gates_A.json` (detached, all 64 buckets, F4+F0) | 0 after normaliser-population fix (see above); 1 on the first pass (normaliser gate only) |
| 5 | `patch_normalizer.py` (recompute + rewrite just `normalizer_check` into the existing `gates_A.json` after the fix, to avoid a second ~10+ minute full rebuild) | 0 |
| 6 | `bench_store.py --out tests_full/A/bench_store.json` (detached) | 0 |
| 7 | `bench_seq.py` (ad hoc, n_threads=1 F4 build, to characterise the RSS-vs-threads tradeoff) | 0 |
| 8 | `disk_guard.py $W --planned-bytes 3221225472` | 0, `pass_floor: true, pass_target: true` |

Long-running steps were launched with `setsid nohup ... > log 2>&1 &` and polled with a remote
`while ! grep -q DONE_MARKER ...; sleep N; done` loop, per the environment's kernel-timeout
guidance (the launch call itself reliably raises a client-side websocket timeout even though the
job starts correctly and detaches — confirmed via a separate `ps aux` check every time before
relying on the poll).

## Unresolved concerns / honest limitations

1. **Peak RSS during `CompactStore.build(channels="F4")` exceeds the environment's 20 GiB process
   budget**: measured `ru_maxrss` **27.6 GiB** at `n_threads=3`, **25.9 GiB** at `n_threads=1` (only
   a ~6% reduction — dropping thread concurrency is NOT the dominant cause). The settled RSS after
   build (17.5 GiB) also exceeds the design target of ≤12 GiB stated in `FULL_RUN_DISK_BUDGET.md`
   and this task's brief. No OOM occurred (host has 918 GiB "available" per `free -h`; the pod-level
   64 GiB shared limit was not hit), and every correctness gate still passed, so this is a resource
   discipline finding, not a functional failure. Likely cause (consistent with the T-H task's
   documented finding for the pilot's `window_cache.py`): materialising `user_id` and other
   string/object pyarrow columns as NumPy object arrays of boxed Python strings, once per bucket
   across 64 buckets, combined with CPython's/glibc's allocator not returning freed arena pages to
   the OS between buckets — a characteristic of the allocator, not a logical leak (this store never
   holds more than one bucket's raw columns at a time per thread, by construction; the RSS growth is
   a monotonically-climbing high-water mark, not sustained active use). I did not attempt a
   risk-bearing rewrite (e.g. replacing `user_id` string handling with `pyarrow.dictionary_encode`
   codes) this late against a fully green gate suite; flagging this for Opus/whoever tunes §13's
   systems addendum: if the 20 GiB figure is a hard cgroup enforcement rather than a soft target,
   `full_store.py` will need that follow-up optimisation before the real training/eval processes
   build their own stores.
2. **F0's measured RSS delta (1.94 GiB) modestly exceeds its 1.5 GiB target.** Root cause understood
   precisely (not a mystery): F0 still retains the full per-(split) decision-level accumulator
   (`decision_id, end_row, length, target_class, loss_mask, eval_mask, client_idx, user_bucket,
   prior_view_count, prefix_position` — ~46 bytes × ~49M retained TRAIN-loss+VS-eval decisions ≈
   2.25 GiB theoretical, close to measured) for ALL loss/eval decisions regardless of channel set,
   since `DecisionSet`'s reporting fields (`prior_view_count`, `prefix_position`, etc.) are part of
   the shared contract, not an F4-only extra. The `item_tokens` array itself is only 254 MB as
   expected.
3. **`fit_scope` in `normalizers.json` is just the string `"TRAIN"`**, with no explicit statement of
   the missing/unsupported-row exclusion described above; I inferred and empirically confirmed the
   exact exclusion rule from `collate()`'s own `MISSING_FLAG_OF`/`USER_MISSING_FLAG_OF` logic rather
   than from any written spec. Flagged in case a documentation update is wanted for the next
   consumer of `normalizers.json`.
4. Per FULL_INTERFACES.md, `channels="F0"` batches never include `category_tokens` /
   `event_numeric_features` / etc. — verified structurally (only the 10 documented keys are present
   in `batch()`'s F0 branch) and empirically (gate (i) F0 subset only ever requests those 10 keys).
5. No VALIDATION_CONFIRM or TEST row or decision was loaded anywhere in this work (grep-checked:
   every `CompactStore.build` call in `full_store.py`, `test_full_store.py`, and `bench_store.py`
   uses only `splits=("TRAIN", "VALIDATION_SELECT")` or a strict subset of it); `store.decisions()`
   raises `ValueError` for both forbidden splits by construction (never special-cased — it simply
   checks membership in `self.splits`).
6. No raw `user_id` value is ever written to disk or logged anywhere in these five files —
   `eval_user_key`/`prefix_position`/etc. integers are the only per-user information that persists,
   and the raw `user_id` arrays used to build `eval_user_key` go out of scope inside
   `_compute_eval_user_key` as soon as the integer keys are computed.

## Acceptance

All required gates pass, including the corrected normaliser check. The two RSS figures above are
reported as measured, honest, evidence-backed findings against the stated targets — not silently
absorbed and not "fixed" by changing what the store computes — since the task instructions say to
report rather than force a pass when addressing them would mean either accepting build-time risk
this late or reinterpreting the target. No interface deviation from `FULL_INTERFACES.md`.
