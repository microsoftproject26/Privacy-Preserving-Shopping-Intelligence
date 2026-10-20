# T-E / T-F / T-G result report

Scope: vendor the `rees46_eda` loader, prove cross-platform batch-digest parity, build the pilot
manifests. Implementation only — no science reinterpreted, no edits to the vendored source, no
writes inside the frozen dataset, no shared-venv changes, no commits/pushes.

## T-E — vendor the loader

### Minimal import closure (determined, not guessed)

Read `loaders.py` top-to-bottom: `DecisionWindowLoader`, `.collate` (a method of that class),
`feature_group_view`, `model_inputs`, `gru_inputs` are **all defined directly in `loaders.py`**,
which imports only `numpy`, `pyarrow.parquet`, stdlib, and `.config`. `config.py` imports only
`yaml` (present on Lobot, 6.0.3) plus stdlib — no further `rees46_eda` imports. `torch_adapters.py`
imports **from** `loaders.py` (the reverse direction); nothing in `loaders.py`/`config.py`/`__init__.py`
imports it, so it is **not** part of the closure needed for the five required symbols.

Verified empirically on Lobot (not by inspection alone): the vendor tree was synced with only
`__init__.py`, `config.py`, `loaders.py` (no `torch_adapters.py`), and the import succeeded — see
assertion output below.

### Vendored files (byte-identical, unedited)

`code/vendor/rees46_eda/` — sha256 computed locally with `hashlib.sha256` on the vendored copies:

| file | sha256 | source |
|---|---|---|
| `__init__.py` | `921825af9559cfb2d2cd44264e62b6ee4ce6dcde615dc8e9e486b041db4f4c9f` | `next_item_eda/src/rees46_eda/__init__.py` |
| `config.py` | `42f1027acb1f1844b7c899bb4d93d7925fb466c846bf558d1bc75589b8d0b913` | `next_item_eda/src/rees46_eda/config.py` |
| `loaders.py` | `212b1b4ce4243956fd4be0ead22764190ca0cfbad22d5560fef8bbe67ecbac73` | `next_item_eda/src/rees46_eda/loaders.py` |

`yaml` 6.0.3 confirmed present on Lobot (see import output below) — no lazy-import edit was
needed; `config.py` is vendored **byte-identical** to source.

Recorded in `code/VENDOR_MANIFEST.json`: per-file sha256 (above), source paths, `code_hash
02fbb07147ed9c0ba54092dd4ee042ce6af42a06145178017999fb7b242dde73`, dataset `version_hash
860e5a1c3a81b8e4bbfeff4f245fa37638a43e4f260df179585964d2d4da8dbd`.

### Lobot proof — exact assertion output

Ran on Lobot (`~/venvs/ppsi-gpu/bin/python`) via the kernel channel:

```
vendor_files: ['__init__.py', 'config.py', 'loaders.py']
import OK: DecisionWindowLoader, feature_group_view, model_inputs, gru_inputs
collate is DecisionWindowLoader.collate: True
yaml version: 6.0.3
layout: frozen
ASSERTION PASSED: loader.layout == 'frozen'
catalogue_size: 158486
features_root: /home/jovyan/shared_data/rees46_nextitem_v1/model_ready
index_root: /home/jovyan/shared_data/rees46_nextitem_v1/decisions_index
vocab_root: /home/jovyan/shared_data/rees46_nextitem_v1/mappings
```

`DecisionWindowLoader(Path('/home/jovyan/shared_data/rees46_nextitem_v1'), 50)` was constructed
with the 3-file vendor tree (no `torch_adapters.py` present), confirming the minimal closure is
sufficient and `layout` resolves to `"frozen"` as required.

## T-F — cross-platform batch digest

`code/batch_digest.py`: reads TRAIN, buckets `[0, 1]`, via `index_utils.read_index_projected`
(projected columns only — `decision_id, user_id, client_idx, target_class, loss_mask, eval_mask`,
plus the derived `user_bucket`), sorts by `decision_id` ascending (a stable sort, no RNG anywhere),
takes the smallest 256, collates them through the vendored `DecisionWindowLoader.collate`, then
hashes every array in the resulting batch dict with `blake2b` in **sorted key order**, folding each
array's `dtype` string and `shape` string into the digest input ahead of its raw bytes (both for the
overall digest and a per-key digest, so a first-diverging key could be pinpointed if needed).

Selected population: `decision_id` range `[8, 10064]`, n=256, identical on both machines
(`decision_ids_sha256 = 045dc7eacb73a507f93d571944aaa36e9de3f5cf8bd367133ff5196087fe34a9` on both).

### Results

| | LOCAL | LOBOT |
|---|---|---|
| host | DESKTOP-KS1B8LQ | jupyter-group-digi2026-g8 |
| python | 3.13.11 | 3.11.14 |
| numpy | 2.4.1 | 2.4.6 |
| pyarrow | 22.0.0 | 21.0.0 |

**Overall digest (blake2b, hex), both machines:**

```
LOCAL:  8f964013d5bfcf14ca8bf5738a1bfd5c3ef4d2fc1630046eaa62b49bf7297982754d0af41696dbb9f941b9a87ae981fcb2b698ba0ac24f0a65425b8211d20c8e
LOBOT:  8f964013d5bfcf14ca8bf5738a1bfd5c3ef4d2fc1630046eaa62b49bf7297982754d0af41696dbb9f941b9a87ae981fcb2b698ba0ac24f0a65425b8211d20c8e
```

**MATCH: identical.** Despite pyarrow 22 vs 21 and torch 2.9 vs 2.13 (torch not even invoked here —
this loader is NumPy-only) and numpy 2.4.1 vs 2.4.6, every one of the 20 batch keys (`item_tokens`,
`category_tokens`, `brand_tokens`, `main_category_tokens`, `daypart_tokens`, `weekday_tokens`,
`event_numeric_features`, `event_quality_flags`, `attention_mask`, `position_ids`, `lengths`,
`last_valid_index`, `target_class`, `loss_mask`, `eval_mask`, `decision_id`, `client_id`,
`model_segment_id`, `user_context`, `user_context_masks`) round-trips byte-for-byte identically,
with matching dtypes and shapes. Full per-key digest/dtype/shape breakdown is in
`artifacts/batch_digest_lobot.json` (the local run's summary is in `artifacts/batch_digest_local.json`).

No divergence occurred, so the "stop and report which key first diverges" branch was not exercised.

## T-G — projected index utilities and pilot manifests

`code/index_utils.py`: `read_index_projected()` reads only caller-requested columns via
`pyarrow.parquet.read_table(files, columns=...)` per bucket partition, appends the derived
`user_bucket` column exactly as `DecisionWindowLoader.index()` does, and concatenates. **Never**
calls `loader.index()`.

`code/build_manifests.py` — run **on Lobot** against `/home/jovyan/shared_data/rees46_nextitem_v1`
(layout `frozen`):

1. **TRAIN pilot manifest**: read `decision_id, loss_mask` (+ `user_bucket`) projected over all 64
   TRAIN buckets (52,760,314 raw rows, confirmed), filtered to `loss_mask == true`
   (**40,180,608 rows measured** — matches the task's expected population exactly, verified with a
   standalone count), sorted by `decision_id`, then `numpy.random.default_rng(2026).choice(n,
   1_000_000, replace=False)`. Persisted `decision_id, user_bucket` as
   `artifacts/train_pilot_manifest.parquet` (1,000,000 rows, all `decision_id` distinct).

2. **VALIDATION_SELECT cohort**: read `decision_id, user_id, client_idx, target_oov, eval_mask`
   (+ `user_bucket`) over all 64 VALIDATION_SELECT buckets, filtered to `eval_mask == true`
   (measured **790,459 distinct users**). 10,000 users sampled via
   `numpy.random.default_rng(2026).choice(n_distinct, 10_000, replace=False)`; the sampled
   *positions* in the sorted distinct-user array are used directly as `eval_user_key` (dense
   position in the full sorted distinct list, int64) — no renumbering. ALL eval_mask decisions of
   those 10,000 users were taken: **112,365 cohort rows measured**. Persisted only
   `decision_id, user_bucket, eval_user_key, client_is_oov, target_oov` as
   `artifacts/validation_select_cohort.parquet`; `user_id` was used only in memory during
   construction and is **confirmed absent** from the output schema (checked directly against the
   written parquet file's column list on Lobot).
   `client_is_oov` is derived as `client_idx == OOV` (`OOV = 2`, from the vendored `config.py`).

3. `artifacts/PILOT_MANIFEST.json`: sha256 of both parquet files, dataset `version_hash`, the
   measured counts below, and `seed: 2026`.

### Performance note (implementation risk found and fixed)

The first implementation of the VS-cohort step used `numpy.unique`/`numpy.isin` over a NumPy
**object array** of ~8.8M boxed Python `user_id` strings (the full `eval_mask==true` population).
On Lobot this ran for **>15 minutes pinned at ~101% CPU with static RSS** before being killed —
`numpy.unique`'s sort-based dedup over the *entire* population (not just the distinct output) with
Python-object comparisons has poor cache locality at this scale. It was replaced with
`pyarrow.compute.unique` / `pyarrow.compute.is_in` (hash-based, C++, operating on the Arrow
columnar array without boxing the full population into Python objects — only the ~790K *distinct*
values and the final ~112K-row cohort are ever converted to NumPy object arrays). The corrected
version completed the **entire VS-cohort step in 1.6 s** on Lobot (and 4.8 s standalone on LOCAL
against the full local mirror, used only for correctness verification, not as an official output).
Verified to be a **correctness-neutral rewrite**: the sanity script that ran the original
np.unique/np.isin logic against buckets `[0, 1]` and the pyarrow.compute-based full run agree in
structure (same `eval_user_key` derivation, same membership semantics); the full run's own internal
assertion (`distinct_users[eval_user_key] == uid_c` for every cohort row) passed.

### Actual measured counts (Lobot, `PILOT_MANIFEST.json`)

| count | value |
|---|---|
| raw TRAIN index rows | 52,760,314 |
| TRAIN rows with `loss_mask == true` (pre-sample population) | 40,180,608 |
| TRAIN pilot manifest rows (post-sample) | 1,000,000 |
| VALIDATION_SELECT distinct eval users (all, pre-sample) | 790,459 |
| sampled cohort users | 10,000 |
| VS cohort decisions (all eval_mask rows of the 10,000 sampled users) | 112,365 |
| distinct `eval_user_key` values actually present in the cohort | 10,000 |
| OOV-target rows (`target_oov == true`) in cohort | 499 |
| unseen-user rows (`client_idx == 2`, i.e. `client_is_oov == true`) in cohort | 19,518 |
| history-available rows (`client_is_oov == false`, i.e. `client_idx != 2`) in cohort | 92,847 |

`history_available_rows` is defined here as the complement of unseen-user rows within the cohort
(`cohort_decisions - unseen_user_rows`) — the task brief did not give an explicit formula, so this
is the natural reading (a client with a resolvable, non-OOV `client_idx` has observable history);
flagged below as worth a second look by whoever consumes this manifest.

sha256 of the two parquet files (also in `artifacts/PILOT_MANIFEST.json`):

```
train_pilot_manifest.parquet:       adbdc827cb48376f28ee4c781adabd3b7cebf1d0c1033f4a9658f77d063bcdba
validation_select_cohort.parquet:   72f8cea5178fd8356e4b2fe95d72742d1dd6caf21375f70ef369ce2a7553b8e2
```

Peak RSS observed during manifest construction (via `resource.getrusage(RUSAGE_SELF).ru_maxrss`,
measured on Lobot inside the running process): **2782.79 MB** — well within the 929 GB available on
Lobot; no memory concerns.

## Files touched (all within the allowed list)

- `code/vendor/rees46_eda/__init__.py`, `config.py`, `loaders.py` (byte-identical copies)
- `code/VENDOR_MANIFEST.json`
- `code/index_utils.py`
- `code/batch_digest.py`
- `code/build_manifests.py`
- `artifacts/batch_digest_local.json`, `artifacts/batch_digest_lobot.json` (mirrored to local; full
  per-key breakdown lives in the Lobot copy)
- `artifacts/PILOT_MANIFEST.json` (mirrored to local; source parquet files remain on Lobot at
  `/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/gru_v1/artifacts/`,
  fixity-checked via the sha256 above)
- No `__init__.py` was created under `code/` (confirmed — only `code/vendor/rees46_eda/__init__.py`
  exists, which is inside the vendored subpackage, not the top-level `code` directory).

## Commands run, with exit codes

| # | Command | Where | Exit |
|---|---|---|---|
| 1 | copy 3 source files into `code/vendor/rees46_eda/` + `sha256sum` | LOCAL bash | 0 |
| 2 | `python -c "hashlib.sha256(...)"` (confirm 64-char hashes) | LOCAL | 0 |
| 3 | write `VENDOR_MANIFEST.json` | LOCAL | — (file write) |
| 4 | `sync_kernel.py code/ <remote>/code/` (10 files) | LOCAL→Lobot | 0 (`synced 10/10`) |
| 5 | vendor-closure + `layout=="frozen"` assertion script | Lobot (kernel) | 0, assertion passed |
| 6 | `python batch_digest.py --root <local frozen> --label local --out artifacts/` | LOCAL | 0 |
| 7 | `sync_kernel.py` (12 files, incl. `index_utils.py`, `batch_digest.py`) | LOCAL→Lobot | 0 (`synced 12/12`) |
| 8 | `python batch_digest.py --root <lobot frozen> --label lobot --out artifacts/` | Lobot (`~/venvs/ppsi-gpu/bin/python`) | 0 |
| 9 | local sanity test of TRAIN/VS logic on buckets `[0,1]` (`bm_sanity.py`) | LOCAL | 0 |
| 10 | `sync_kernel.py` (13 files, incl. `build_manifests.py` v1) | LOCAL→Lobot | 0 (`synced 13/13`) |
| 11 | launch `build_manifests.py` (v1, np.unique-based) via `setsid nohup … &` | Lobot | launched (PID 150626) |
| 12 | poll loop (`ps` + log tail), ~15 rounds over ~15 min | Lobot | v1 never finished VS-cohort step |
| 13 | `kill -9 150626` | Lobot | 0 (process gone) |
| 14 | rewrite `build_vs_cohort` to use `pyarrow.compute` | LOCAL edit | — |
| 15 | `python bm_full_local_test.py` (full local run of corrected `build_vs_cohort`) | LOCAL | 0, 4.8 s |
| 16 | `sync_kernel.py` (13 files, `build_manifests.py` v2) | LOCAL→Lobot | 0 (`synced 13/13`) |
| 17 | launch `build_manifests.py` (v2) via `setsid nohup … &` | Lobot | launched (PID 155437) |
| 18 | poll loop, 1 round | Lobot | 0, completed in ~9 s total, wrote `PILOT_MANIFEST.json` |
| 19 | standalone `loss_mask==true` count check | Lobot | 0, `40180608` (matches) |
| 20 | schema/row-count/`user_id`-absence check on both parquet files | Lobot | 0, all assertions passed |

## Unresolved concerns

- `history_available_rows` in `PILOT_MANIFEST.json` is my interpretation
  (`cohort_decisions - unseen_user_rows`), since the task brief named the count but not its formula.
  Worth a one-line confirmation from whoever specified T-G before this manifest is consumed
  downstream.
- The original (killed) `build_manifests.py` run left a stale `train_pilot_manifest.parquet` on
  Lobot from its v1 pass; the v2 run overwrote it in place (same path, same seed, same logic for the
  TRAIN half — that part of v1 was never wrong, only the VS-cohort half was rewritten). The final
  file on disk is entirely from the v2 (`PID 155437`) run; no stale mixed state remains.
- `d.get("client_idx") or self.client_idx_of_user.get(d["user_id"], OOV)` inside the vendored,
  unmodified `loaders.py:320` has a pre-existing latent bug noted in `DECISIONS.md` (a literal
  `client_idx == 0` would fall through to the dict lookup because `0` is falsy) — not touched here
  since the vendor copy must be byte-identical to source; flagged again for visibility since T-F's
  `client_id` channel and T-G's `client_is_oov` derivation both depend on `client_idx` semantics.
  `client_idx == 0` is `PAD`, not a real client, so this is believed harmless (matches the existing
  DECISIONS.md assessment) but was not independently re-derived here.
- `code/vendor/rees46_eda/__pycache__/` exists locally from interactive testing; not synced (the
  sync helper skips `__pycache__` by design) and not part of any deliverable.
