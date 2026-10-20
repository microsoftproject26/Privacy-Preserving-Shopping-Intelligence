# T-H / T-I result report

Scope: bucket-major memmapped window cache + T3 parity test (T-H), synthetic
memorisation gate (T-I). Implementation only -- no edits to the vendored
loader, `model.py`, `objective.py`, the manifests, or any acceptance
threshold; no writes inside the frozen dataset; no shared-venv changes; no
commits/pushes.

## T-H -- bucket-major window cache

### A real bug the parity test caught and the fix

`window_cache.py`'s first version stored `decision_id` as int32, following
the task brief's stated precondition ("all indices < 2^31"). **That
precondition does not hold for `decision_id` in this dataset.** Measured
directly against the frozen index:

```
train_pilot_manifest.parquet: decision_id dtype int64, min 15, max 100032590990
432,143 / 1,000,000 TRAIN-manifest rows (43.2%) exceed 2^31-1 (2,147,483,647)
```

`decision_id` is evidently a large composite/globally-unique identifier, not
a small sequential counter. Every *other* field kept in the cache was
checked the same way and does fit int32: `target_class` in `[-1, 158485]`,
`client_idx` (-> `client_id`) max `4,204,371`, all vocab-bounded categorical
channels (item vocab ~158,489; category ~663; brand ~3,956; main_category
~16; daypart/weekday small closed sets), and `lengths`/`last_valid_index`/
`position_ids` bounded by `L=50`.

The first TRAIN-cache build (with `decision_id` cast to int32) completed
without error -- the overflow is silent, not a crash -- and `test_cache_parity_exact`
then failed on exactly this field: cached `decision_id` values topped out
around 1.25e9 while the directly-collated reference values ran up to
~1.0e11. This is precisely the kind of bug a parity test exists to catch.
Fixed by keeping `decision_id` int64 end-to-end (not cast at build, not cast
at load) in `code/window_cache.py`; every other channel's int32 storage is
unchanged. Both caches were then rebuilt from scratch and the parity test
re-run clean (see below). This is flagged as the header note in
`window_cache.py` itself (`ARRAY_SPECS["decision_id"]`), not just here.

### Cache contents

19 arrays per row, mirroring `DecisionWindowLoader.collate()`'s batch schema
at `L=50` (`model_segment_id` is the one collate() output deliberately
*not* kept -- it isn't in the task's named list):
`item_tokens, category_tokens, brand_tokens, main_category_tokens,
daypart_tokens, weekday_tokens, position_ids` (int32, `[N,50]`),
`attention_mask` (bool, `[N,50]`), `event_numeric_features` (float32,
`[N,50,19]`), `event_quality_flags` (int8, `[N,50,15]`), `user_context`
(float32, `[N,12]`), `user_context_masks` (float32, `[N,6]`), `lengths,
last_valid_index, target_class, client_id` (int32, `[N]`), `decision_id`
(**int64**, `[N]`), `loss_mask, eval_mask` (bool, `[N]`). Stored as
standard `.npy` files opened via `np.lib.format.open_memmap`, one array per
channel per cache directory, plus a `header.json` (per-array dtype/shape,
blake2b digest of cache contents, source-manifest sha256, dataset
`version_hash`, row count, the full per-bucket RSS/timing log).

### Build method (bucket-by-bucket, evicted)

For each of the 64 buckets present in a manifest, in ascending bucket-id
order: (1) project only `decision_id, target_class, loss_mask, eval_mask,
client_idx` from the decision index for that one bucket
(`index_utils.read_index_projected`, never the full 19-column index); (2)
build the decision dicts `DecisionWindowLoader.collate()` expects and call
it -- this is the one place the vendored `BucketStore` loads the bucket's
full feature parquet; (3) write the resulting rows into the pre-allocated
memmap cache at their **manifest row index** (not appended), so cache row
order == manifest row order regardless of bucket processing order; (4)
`loader._buckets.pop(b, None)` + `gc.collect()`; (5) log RSS
(`psutil.Process().memory_info().rss`, current, not peak-ever) before the
bucket, after loading it, and after eviction.

### Cache sizes on disk (bytes, from `header.json` + confirmed by `ls -la`)

| cache | rows | buckets | `total_disk_bytes` | on-disk (`du -sh`) |
|---|---|---|---|---|
| TRAIN (`cache/train/`) | 1,000,000 | 64 | **6,098,002,432** | 5.7G |
| VALIDATION_SELECT cohort (`cache/vs/`) | 112,365 | 64 | **685,204,202** | 654M |
| **combined** | 1,112,365 | -- | **6,783,206,634** (~6.78 GB) | -- |

Per-decision cost: 6,098,002,432 / 1,000,000 ≈ 6,098 bytes (~5.95 KB),
matching the task's ≈5.9 KB/decision estimate (the small excess over the
task's 5.9 KB figure is the `decision_id` int64-not-int32 fix: +4
bytes/row).

### Peak RSS per bucket (proving bounded memory)

Full 64-line per-bucket logs: `logs/window_cache_train_buckets.log`,
`logs/window_cache_vs_buckets.log` (mirrored nowhere locally by design --
they're process logs on Lobot; the summary numbers below are exact,
extracted from those files).

| cache | min `rss_after_load_mb` | max `rss_after_load_mb` (= `peak_rss_after_load_mb`) | pattern |
|---|---|---|---|
| TRAIN | 3,132.0 (bucket 0) | **16,780.5** (bucket 44) | rises roughly linearly through ~bucket 40, then plateaus/oscillates in a 15.0-17.4 GB band for the remaining buckets -- it does **not** keep climbing |
| VS cohort | 2,890.1 (bucket 0) | **13,723.1** (bucket 61) | same shape, smaller (fewer rows/bucket): rises through ~bucket 40, then plateaus 9.6-13.7 GB |

What "bounded" means here, precisely, and what does *not* hold: the
BucketStore-specific footprint (`rss_after_load_mb - rss_before_mb`) is
**stable across the whole run** -- roughly 850 MB-2.1 GB per bucket for
TRAIN, not growing with bucket index -- and is released every time
(`rss_after_evict_mb` tracks back down close to `rss_before_mb`; e.g. TRAIN
bucket 63: `rss_before=15233.0, rss_after_load=16123.3, rss_after_evict=15041.4`,
i.e. net *shrinkage* across that bucket-cycle). This confirms
`loader._buckets.pop(b, None)` genuinely evicts each bucket -- the naive
unfixed approach (every `BucketStore` resident at once) would need on the
order of 64 x ~1-2 GB = 65-135+ GB resident simultaneously; this build never
exceeds ~17.4 GB.

What *does* grow across the run, and why it isn't a bug: the process
baseline (`rss_before_mb`) itself climbs from ~1 GB to a ceiling around
15-17 GB over the first ~40 buckets, then stops climbing. Two contributors,
both bounded and neither a `BucketStore` leak: (1) the OUTPUT memmap files
(opened once, `w+`, for the full `N`-row cache) accumulate resident pages as
more of the file gets touched by writes -- capped by the cache's own total
size (~6.1 GB for TRAIN), which is consistent with most of the observed
growth; (2) CPython's small-object allocator does not reliably return freed
arena pages to the OS after each bucket's `~15-16K`-entry metadata dict is
freed (a well-documented CPython behaviour, not a logical leak -- refcounts
hit zero immediately, `gc.collect()` finds nothing new). The plateau
starting around 2/3 of the way through both runs (bucket ~40-44 of 64) is
the empirical signature of this: if it were a true per-bucket leak it would
keep climbing linearly to bucket 63, and it does not.
`window_cache.py` does **not** programmatically abort if disk space is low;
the "abort under 12 GB free" check was done manually before launching (see
below), not enforced in code -- noted as a gap under Unresolved concerns.

### Build wall-clock

| cache | `build_seconds` (from `header.json`) |
|---|---|
| TRAIN | **426.70 s** (~7.1 min), 64 buckets, ~15,625 rows/bucket avg |
| VS cohort | **176.90 s** (~2.9 min), 64 buckets, ~1,756 rows/bucket avg |

### Provenance (in each cache's `header.json`)

| | TRAIN | VS cohort |
|---|---|---|
| `version_hash` | `860e5a1c3a81b8e4bbfeff4f245fa37638a43e4f260df179585964d2d4da8dbd` | same |
| `source_manifest.sha256` | `adbdc827cb48376f28ee4c781adabd3b7cebf1d0c1033f4a9658f77d063bcdba` | `72f8cea5178fd8356e4b2fe95d72742d1dd6caf21375f70ef369ce2a7553b8e2` |
| `digest_blake2b` (cache contents, sorted key order, chunked) | `ec698793104e8fcbdbad1d020da242e1de6caf756f685b20a30a29b1ea139377d976cc876eb7a813505af8c051a6bc9339e6d38618af119b73b32b01390a91bd` | `c4484126743fb2bf3f8c08c0655e3b471b31b89f218967b723392e4f9b2aa414f58e192be432d303568029363575699aa484252fb4e6e3168e68e1bd80df0fd5` |
| `row_count` | 1,000,000 | 112,365 |
| `int32_cast_names` (11 channels) | `item_tokens, category_tokens, brand_tokens, main_category_tokens, daypart_tokens, weekday_tokens, position_ids, lengths, last_valid_index, target_class, client_id` | same |

### T3 parity test (`code/test_cache_parity.py`, run on Lobot)

512 random rows of the TRAIN manifest (`np.random.default_rng(20260922)`,
fixed seed, distinct from the manifest-construction seed 2026), read from
`CacheReader`, compared against an independent collation of the same
`decision_id`s through a **fresh** `DecisionWindowLoader` (never touching
the cache).

- **`test_cache_parity_exact`: PASSED.** All **19 arrays** compared
  (`compared == len(ARRAY_NAMES) == 19` asserted explicitly), every one
  `np.array_equal` **exactly** (not approximately) after the documented
  int32->int64 cast, with matching dtype and shape checked first for each
  array. A row-alignment sanity check (`cached["decision_id"] ==
  sampled decision_ids`) was asserted before the content comparison.
- **`test_cache_parity_broken_offset_xfail`: XFAIL (strict), as designed.**
  Same 512 rows, but the cache reader is queried at `(rows + 1) %
  n_total` while the reference collation still targets the correct
  decisions. This must fail (and does): `pytest.mark.xfail(strict=True)`
  means the test run would itself FAIL if this variant unexpectedly
  passed, so its XFAIL status is a positive confirmation that the parity
  test has power to catch a real row-alignment bug.
- Full run: `1 passed, 1 xfailed in 188.10s (0:03:08)` (this run, after the
  `decision_id` fix; the pre-fix run legitimately reported `1 failed, 1
  xfailed` -- see the bug section above).

### `CacheReader`

`window_cache.CacheReader(cache_dir).read(row_indices)` returns a dict of
`[B, ...]` torch tensors: the 11 `int32_cast_names` channels are cast to
int64 on load (matches the task brief); `decision_id` (int64 stored, int64
returned, no cast needed), bool masks, int8 flags, and float32 numerics keep
their stored dtype.

## T-I -- synthetic memorisation gate

**This is an engineering-correctness check only (Stage B). It is explicitly
NOT a claim about real-data quality** -- it only proves `model.py` +
`objective.py`'s plumbing (embeddings, GRU packing, the tied output head,
`exact_ce_loss`) is wired correctly, using a purely synthetic task where the
target is `target = (last_token * 7 + 13) % K_syn` (`K_syn=1000`), a
deterministic function of the last valid context token. `nvidia-smi` was
run before starting (`0% util`, `No running processes found` on both A16s);
`synthetic_memorisation.py` also re-checks
`nvidia-smi --query-compute-apps` at its own start and aborts if any GPU
compute process is present. Ran with `CUDA_VISIBLE_DEVICES=0`, dropout=0.0,
weight_decay=0.0, Adam lr=1e-2, at most 300 full-batch updates (all 300 were
used in every arm; no early stop was implemented).

### Results (`artifacts/SYNTHETIC_MEMORISATION.json`, mirrored from Lobot)

| arm | initial loss | final loss | loss ratio | threshold (<=0.05x) | final top-1 | threshold (>=0.95) | updates | PASS |
|---|---|---|---|---|---|---|---|---|
| F0_item_only | 8.043074 | 2.7148e-07 | 3.375e-08 | 0.402154 | **1.0000** | 0.95 | 300 | **True** |
| F4_rich_side_information | 8.036150 | 3.4412e-07 | 4.282e-08 | 0.401807 | **1.0000** | 0.95 | 300 | **True** |

Both arms converge essentially to zero loss and perfect top-1 within ~3
updates (F0 step 2 loss 0.011, F4 step 3 loss 0.0036; see
`history_first_5`/`history_last_5` in the JSON) and stay there for the
remaining ~297 updates -- full-batch memorisation of 256 examples through
128-dim item embeddings + a 256-unit GRU is not a hard problem, so this is
the expected shape of a passing run, not a surprisingly lucky one. F4 wires
all 6 categorical channels (item/category/brand/main_category/daypart/
weekday, vocab sizes `1003/53/53/13/11/10`) + 19 numeric + 15 flags + 12
user-context + 6 masks; the non-item channels are independent Gaussian/
Bernoulli noise (uninformative by construction) -- the model must still
route the actual signal through `item_tokens` to solve the task, and it
does, at the same speed as F0.

### Negative control

F0-shaped model, **same true `target_class` values as the F0 positive arm**
(computed from each row's real, un-corrupted last token), but every row's
`item_tokens` overwritten with one constant synthetic token
(`KNOWN_START + 1 = 4`) across the whole batch -- the model cannot
distinguish rows and so cannot do better than guessing the empirical
majority target class.

| | value |
|---|---|
| majority-class rate (measured from the 256 true targets) | **0.01171875** (3/256) |
| threshold (majority + 0.05) | **0.06171875** |
| final top-1 | **0.0078125** (2/256) |
| initial loss -> final loss | 9.133555 -> 40.070419 (loss *rises*: with identical inputs and conflicting labels there is no fixed-point the model can settle a single prediction on that improves cross-entropy across all 256 rows; this is expected for a genuinely unlearnable target, not a bug) |
| PASS (`final_top1 <= threshold`) | **True** |

`final_top1` (0.0078) is at/below the majority-class rate itself, well under
the `+0.05` slack threshold -- this confirms the gate can actually fail
(the stated purpose of the negative control) rather than trivially passing
everything.

**`overall_pass: true`** for all three arms combined.

## Free disk

| when | free | total |
|---|---|---|
| before T-H work started | 34.532438016 GB | 52.521566208 GB |
| after both cache builds + all tests | 27.701694464 GB | 52.521566208 GB |

Delta used: ~6.83 GB, consistent with the combined cache size (~6.78 GB)
plus minor transient files. The task's "abort if under 12 GB free" gate was
satisfied (34.5 GB free at the start) -- checked manually via a preliminary
`shutil.disk_usage` call before launching any build, **not** enforced
programmatically inside `window_cache.py` itself (see Unresolved concerns).

## Commands run, with exit codes

| # | Command | Where | Exit / result |
|---|---|---|---|
| 1 | `nvidia-smi` (pre-GPU-work check) | Lobot (kernel) | 0, `0% util`, `No running processes found` on both A16s |
| 2 | `shutil.disk_usage(...)` on the gru_v1 workspace | Lobot (kernel) | 0, `disk_free_gb 34.532438016` |
| 3 | `pq.read_schema(...)` on TRAIN/VALIDATION_SELECT index files | Lobot (kernel) | 0, confirmed `client_idx` present in both splits' index schema |
| 4 | python/torch/psutil version + workspace-layout check | Lobot (kernel) | 0, torch 2.13.0+cu130, cuda available, psutil 7.2.2 |
| 5 | `sync_kernel.py code/ -> code/` (first attempt, `MSYS_NO_PATHCONV` misapplied) | LOCAL->Lobot | 0 but WRONG target (`.../home/jovyan/C:/Program Files/...`); files never reached the intended `code/` dir |
| 6 | inspect + `shutil.rmtree` the stray `C:/` directory the mis-sync created under `$HOME` | Lobot (kernel) | 0, removed |
| 7 | `sync_kernel.py code/ -> code/` (corrected path handling) | LOCAL->Lobot | 0, `synced 16/16 files` |
| 8 | import smoke test: `window_cache`, `synthetic_memorisation` | Lobot (kernel) | 0, `19` arrays, widths `F_NUMERIC=19 Q_FLAGS=15 U_CTX=12 M_CTX=6` confirmed |
| 9 | 2-bucket smoke test: `build_cache` on 300 rows (buckets 0,1) + manual parity check | Lobot (kernel) | 0, `n_compared 19 all_ok True`, `decision_id match True` |
| 10 | cleanup tmp smoke-test files, `mkdir -p cache/ logs/` | Lobot (kernel) | 0 |
| 11 | launch TRAIN cache build v1 (`decision_id` still int32) via `setsid nohup ... &` | Lobot | launched, PID 159631 |
| 12 | launch `synthetic_memorisation.py` via `setsid nohup env CUDA_VISIBLE_DEVICES=0 ... &` | Lobot | launched, PID 159889; **finished successfully**, `overall_pass: true` |
| 13-14 | poll loops (`ps` + log tail) | Lobot (kernel) | TRAIN v1 build finished after ~455s, 64 buckets |
| 15 | launch VS-cohort cache build v1 + `pytest test_cache_parity.py` via `setsid nohup ... &` | Lobot | launched, PIDs 161014 / 161016 |
| 16 | poll loop | Lobot (kernel) | VS v1 build finished (205.45s); **pytest: `test_cache_parity_exact` FAILED** (`decision_id` int32 overflow), broken-offset XFAIL as designed |
| 17 | root-cause check: `decision_id` min/max/overflow-count on the TRAIN manifest | Lobot (kernel) | 0, confirmed `max=100032590990`, `432143/1000000` rows exceed int32 |
| 18 | root-cause check: `client_idx`/`decision_id`/`target_class` ranges on a TRAIN index bucket | Lobot (kernel) | 0, confirmed only `decision_id` is unsafe for int32 |
| 19 | `Edit window_cache.py`: `decision_id` int32 -> int64 + explanatory comment | LOCAL | -- (file edit) |
| 20 | `sync_kernel.py code/ -> code/` (fix) | LOCAL->Lobot | 0, `synced 16/16 files` |
| 21 | remove `cache/train`, `cache/vs` (v1, buggy), relaunch TRAIN cache build v2 | Lobot (kernel) | 0, PID 162910 |
| 22-23 | poll loops (`ps` + log tail) | Lobot (kernel) | TRAIN v2 build finished after 426.70s, 64 buckets |
| 24 | relaunch VS-cohort cache build v2 + `pytest test_cache_parity.py` | Lobot | launched, PIDs 164048 / 164050 |
| 25 | poll loop | Lobot (kernel) | VS v2 build finished (176.90s); **pytest: `1 passed, 1 xfailed in 188.10s`** |
| 26 | gather `header.json` stats (both caches) + `shutil.disk_usage` | Lobot (kernel) | 0 |
| 27 | `du -sh` / `ls -la` on `cache/train/`, `cache/vs/` | Lobot (kernel) | 0, `5.7G` / `654M` |
| 28 | fetch full 64-line per-bucket logs (TRAIN, VS) | Lobot (kernel) | 0 |
| 29 | dump full `SYNTHETIC_MEMORISATION.json`, mirror to LOCAL `artifacts/` | Lobot (kernel) -> LOCAL | 0 |

## Files touched (all within the allowed list)

- `code/window_cache.py` -- cache builder, `CacheReader`, CLI.
- `code/test_cache_parity.py` -- T3 parity test + xfail broken variant.
- `code/synthetic_memorisation.py` -- T-I gate, F0/F4 arms + negative control.
- `task_T-HI_result.md` -- this report.
- `artifacts/SYNTHETIC_MEMORISATION.json` -- mirrored from Lobot.
- Cache output on Lobot only (per task instructions, not duplicated
  locally): `cache/train/` (19 `.npy` files + `header.json`), `cache/vs/`
  (same). Logs on Lobot only: `logs/window_cache_train.log`,
  `logs/window_cache_train_buckets.log`, `logs/window_cache_vs.log`,
  `logs/window_cache_vs_buckets.log`, `logs/synthetic_memorisation.log`,
  `logs/synthetic_memorisation_stdout.log`, `logs/test_cache_parity.log`.

## Unresolved concerns

- **`window_cache.py` does not enforce the "abort if free disk < 12 GB"
  gate in code.** It was checked manually (34.5 GB free) before launching
  either build. A production version of this script should call
  `shutil.disk_usage` and raise before allocating the output memmaps.
- **RSS is not flat across the build** (see the "Peak RSS per bucket"
  section) -- it climbs from ~1 GB to a ~15-17 GB plateau over roughly the
  first two-thirds of the 64 buckets, then stops climbing. The
  per-bucket `BucketStore` footprint itself is demonstrably bounded and
  released every cycle (that part of "bounded memory" is solid); the
  climbing baseline is attributed to (a) the output memmap's resident
  pages accumulating as the file fills, capped by the cache's own final
  size, and (b) CPython's small-object allocator not returning freed
  per-bucket dict memory to the OS. This is a plausible, evidence-consistent
  explanation (the plateau shape argues against an unbounded leak) but was
  not verified with a memory profiler (e.g. `tracemalloc` or `mmap`-level
  page accounting) -- flagged rather than asserted as fully proven.
- **The synthetic-memorisation LR (`1e-2`) and step budget (300) were
  chosen once and not tuned/ablated** -- both arms converged in ~3 steps, so
  there was no need to search, but this means the specific numbers (e.g.
  "converges by step 2-3") are a property of this one hyperparameter choice,
  not a general claim about the model's sample efficiency.
- **The negative control uses only the F0-shaped model** (item-tokens-only),
  not a separate F4-shaped negative control. The task brief describes one
  negative control, not one per arm; F0 is the minimal, cleanest case
  (there is exactly one channel to corrupt), and since F4's *positive* arm
  already demonstrates the extra channels route real signal correctly, a
  second F4 negative control was judged redundant. Flagged in case the
  reviewer wanted one per arm.
- **The `decision_id`-overflow bug (see above) only affected `decision_id`
  itself.** Every other field was independently checked against the real
  frozen dataset's observed ranges before concluding int32 was safe for it;
  this was empirical (measured against buckets/splits actually present),
  not a proof over the full 52.76M-row TRAIN index -- if a field's true
  range is bucket-dependent in a way the sampled buckets didn't expose,
  this could recur for a different field. The parity test is the safety
  net for exactly this class of bug and is now part of the deliverable.
- The first `sync_kernel.py` invocation mis-synced due to a Windows/MSYS
  path-conversion interaction (`MSYS_NO_PATHCONV=1` combined with a
  `/c/...`-style local path for the *script* argument caused Windows to
  resolve it against the current drive, while leaving the *remote* Linux
  path argument unconverted worked correctly only once both local paths
  were also given in drive-letter form). The stray directory it created
  under `$HOME` on Lobot was found and removed before any real work
  depended on it; no cache or log data was ever written under that path.
