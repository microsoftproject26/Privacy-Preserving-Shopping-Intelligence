# Data contract — shared GRU / SASRec next-item batch

Run `run_20260920_234844` · feature set `rees46_next_item_shared_v4_localtime` · default context length **L = 50** ·
catalogue **K = 158486** rankable classes.

This document describes how to load a real batch **without training anything**.

## 1. Load a real small batch

```python
import sys; sys.path.insert(0, "src")
from pathlib import Path
from rees46_eda.config import PATHS
from rees46_eda.loaders import DecisionWindowLoader, gru_inputs, sasrec_inputs

run_dir = PATHS.run_dir(Path("state/RUN_ID").read_text().strip())
loader  = DecisionWindowLoader(run_dir, context_length=50)

index = loader.index("TRAIN", buckets=[0])       # lightweight decision index
batch = loader.batch_from_index(index, rows=range(32))

print(batch["item_tokens"].shape)                # (32, 50)
print(batch["event_numeric_features"].shape)     # (32, 50, 19)
print(batch["user_context"].shape)               # (32, 12)

gru    = gru_inputs(batch)                       # packs by `lengths`, zero initial state
sasrec = sasrec_inputs(batch)                    # key-padding + causal mask, query at last valid
```

PyTorch tensors (data only — no model, no optimizer, no training loop):

```python
from rees46_eda.torch_adapters import DecisionDataset
ds = DecisionDataset(run_dir, "TRAIN", context_length=50, buckets=[0])
tensors = ds.collate([ds[i] for i in range(32)])   # use as a DataLoader collate_fn
```

## 2. Batch schema

| key | dtype | shape | meaning |
|---|---|---|---|
| `item_tokens` | int64 | `[B, 50]` | context item tokens; PAD=0 on padded positions |
| `category_tokens` | int64 | `[B, 50]` | resolved category id token |
| `brand_tokens` | int64 | `[B, 50]` | resolved brand token |
| `main_category_tokens` | int64 | `[B, 50]` | resolved main-category root token |
| `event_numeric_features` | float32 | `[B, 50, 19]` | normalized numeric block, fixed documented order |
| `event_quality_flags` | int8 | `[B, 50, 15]` | missing / recovered / conflict indicators |
| `user_context` | float32 | `[B, 12]` | decision-time user context |
| `user_context_masks` | float32 | `[B, 6]` | availability/support masks for the above |
| `attention_mask` | bool | `[B, 50]` | True on valid positions |
| `position_ids` | int64 | `[B, 50]` | 1..length on valid positions, 0 on padding |
| `lengths` | int64 | `[B]` | valid prefix length, always >= 1 |
| `last_valid_index` | int64 | `[B]` | `lengths - 1` |
| `target_class` | int64 | `[B]` | compact class in `0..158486-1`, or **-1** when no rankable target |
| `loss_mask` | bool | `[B]` | TRAIN decisions that may contribute to the loss |
| `eval_mask` | bool | `[B]` | future-partition decisions that may be scored |
| `decision_id`, `client_id`, `model_segment_id` | int64 | `[B]` | identifiers, **not predictors** |

Numeric feature order in `event_numeric_features`:

```
[
  "log1p_price",
  "utc_hour_sin",
  "utc_hour_cos",
  "utc_weekday_sin",
  "utc_weekday_cos",
  "local_hour_sin",
  "local_hour_cos",
  "local_weekday_sin",
  "local_weekday_cos",
  "is_weekend",
  "log1p_prev_raw_view_gap_s",
  "log1p_segment_elapsed_s",
  "log1p_item_observed_age_s",
  "prefix_representative_count",
  "prefix_item_count",
  "prefix_known_category_count",
  "prefix_category_switch_count",
  "current_known_category_run_length",
  "log1p_prefix_elapsed_s"
]
```

Quality-flag order in `event_quality_flags`:

```
[
  "price_missing",
  "price_zero",
  "price_conflict",
  "brand_missing",
  "brand_recovered",
  "brand_conflict",
  "category_missing",
  "category_recovered",
  "category_conflict",
  "category_code_missing",
  "category_code_recovered",
  "main_category_missing",
  "prev_gap_unavailable",
  "item_age_unavailable",
  "simultaneous_block"
]
```

User-context order and mask order:

```
features: ["log1p_prev_view_count", "log1p_prev_session_count", "recent_history_length", "has_history", "recent_distinct_items", "recent_distinct_known_main_categories", "recent_category_match_share", "recent_brand_match_share", "recent_mean_log_price", "current_minus_recent_mean_log_price", "log1p_time_since_prev_user_event_s", "log1p_time_since_prev_session_last_event_s"]
masks   : ["has_history_mask", "recent_category_match_supported", "recent_brand_match_supported", "recent_price_supported", "prev_user_event_available", "prev_session_available"]
```

## 3. Padding and masking

- **Right padding, always.** Valid positions occupy `[0, length)`; everything after is padding.
- `position_ids` are `1..length` then `0`. A model that adds a learned positional embedding
  should reserve index 0 for padding.
- `attention_mask` is True exactly on valid positions. SASRec receives
  `key_padding_mask = ~attention_mask` plus a lower-triangular causal mask.
- No supervised context is ever all padding: `lengths >= 1` is asserted.
- The target is **never** part of the input window.

## 4. Output classes and translation

- Output space is `0..K-1` over **real catalogue items only**.
- `vocabularies/catalogue.parquet` gives the RAW STRING `product_id`, its `output_class`
  and its `product_idx`, so the inverse mapping class -> input index -> raw string id is
  explicit and lossless.
- `target_class = -1` means *no rankable target*. Those rows must be excluded from the ranking
  denominator, not scored as a miss.
- An observed target outside the catalogue keeps its true raw identity in the ledger
  (`target_product_id`, `target_oov = true`). In TRAIN its `loss_mask` is false; in future
  partitions it stays end-to-end eligible and scores **zero**.
- `PAD`, `MISSING` and `OOV` are **not** output classes and can never be a correct answer.

## 4d. الفرق بين تخطيط النسخة المجمّدة ومسارات المُحمِّل

النسخة المجمّدة تُعيد تسمية المجلدات لتكون مقروءة، والمُحمِّل يفتح الأسماء
الأصلية. من غير الجدول ده، النسخة المجمّدة **لا تُفتح** بالمُحمِّل المذكور في
`model_ready/FEATURE_GROUPS.json` نفسه:

| في النسخة المجمّدة | ما يفتحه `DecisionWindowLoader` |
|---|---|
| `mappings/` | `vocabularies/` |
| `decisions/` | `decision_ledgers_annotated/` |
| `decisions_index/` | `indices/decisions/` |
| `model_ready/` | `features/` |

للتشغيل من النسخة المجمّدة: أعد تسمية المجلدات إلى أسماء المُحمِّل، أو مرّر
مجلد الرن الأصلي `artifacts/<run_id>/`. النسخة المجمّدة هي **سجل التجميد
وإثبات السلامة**، ومجلد الرن هو ما يقرأه الكود.

## 4c. Ablation feature groups (F0..F4)

كل مجموعة **عرض (view) على نفس الـbatch**، مش استخراج منفصل. يعني كل المجموعات
ترى **نفس القرارات ونفس النافذة ونفس الأهداف** — فالمقارنة بينها تقيس **محتوى
المعلومة** وحده، مش اختلافًا في البيانات. مفاتيح الإشراف (`target_class`,
`loss_mask`, `eval_mask`, `decision_id`) موجودة في كل مجموعة: مجموعة بلا هدف
ليست ablation بل مهمة مختلفة.

| المجموعة | قنوات categorical | خصائص عددية | سياق المستخدم |
|---|---|---|---|
| `F0_item_only` | 1 (`item_tokens`) | لا شيء | لا |
| `F1_core_categorical` | 4 (+ كاتوجري، براند، فئة رئيسية) | لا شيء | لا |
| `F2_time_context` | 6 (+ daypart، weekday) | 13 | لا |
| `F3_user_session_context` | 6 | 18 (+ خصائص بادئة الجلسة) | **نعم** |
| `F4_rich_side_information` | 6 | كل القنوات (19) | **نعم** |

الاستعمال: `rees46_eda.loaders.feature_group_view(batch, "F2_time_context")`.
التعريف المجمّد يُصدَّر مع الداتا في `model_ready/FEATURE_GROUPS.json`.

> **هذه ليست توصية بأي مجموعة.** تحديد المجموعة الفائزة سؤال تدريب، والبتّ فيه
> أثناء المعالجة المسبقة كان سيصبح ضبطًا للداتا على درجات النموذج — وهو بالضبط
> ما يمنعه البروتوكول.

## 4b. Identifier representation (canonical identity is a STRING)

| identifier | canonical / audit / intermediate / ledger artifacts | model-facing artifacts |
|---|---|---|
| user | `user_id` **STRING** (trimmed source token) | `client_idx` contiguous int |
| product | `product_id` **STRING** | `product_idx` contiguous int (= input item token) |
| category | `category_id` **STRING** | `category_idx` contiguous int |
| brand | `brand_resolved` **STRING** | `brand_idx` contiguous int |
| source session | `user_session_raw` **STRING** | not modelled |
| derived session | `derived_session_id` **STRING** | not modelled |

* No identifier is ever stored as FLOAT or DOUBLE at any stage. An earlier build
  routed identifiers through `HUGEINT`, which DuckDB serializes to Parquet as
  `DOUBLE`; `category_id` values are ~2.05e18, above 2^53, so they were being
  rounded. Identifiers are now the raw source string end to end.
* Leading zeros survive, no `.0` suffix is stripped, and nothing depends on an
  identifier being numeric.
* The integer indices are **deterministic and TRAIN-fitted**, assigned by sorted
  order of the raw string, with `PAD=0 / MISSING=1 / OOV=2` reserved and known
  values from 3 upward. A value first seen in a future partition is `OOV`.
* Mapping files preserve `raw_string_id -> integer_index`:
  `vocabularies/catalogue.parquet` (product_id -> product_idx / output_class),
  `vocabularies/clients.parquet` (user_id -> client_idx),
  `vocabularies/vocab_category_id.parquet`, `vocab_brand.parquet`,
  `vocab_main_category.parquet`, `vocab_category_code.parquet` (value -> idx).
* `decision_id` is **not** a source identifier: it is the derived surrogate
  `source_file_id * 1e11 + source_row_number`, provably within int64 because the
  largest source file has 67,501,979 records.
* Stage `03b_roundtrip` proves, on full data, that every raw identifier read
  back out of Parquet is the same STRING as in the source CSV, and fails the run
  if any identifier column is stored as a floating-point type.

## 5. Events, history pointers and lazy materialization

- Complete safe sequences live in `artifacts/run_20260920_234844/features/user_bucket=*/features.parquet`,
  ordered by `(user_id, segment_index, position_in_segment)`.
- A window is resolved by that **logical key**, never by physical file order, so repartitioning
  cannot change a batch.
- The decision index in `artifacts/run_20260920_234844/indices/decisions/` is lightweight; arrays are
  materialized lazily at collate time. Nothing stores one row per prefix.

## 6. The controlled-context GRU requirement

For a controlled bounded-window comparison, the GRU **must not** carry hidden state from before
the shared L-window. `gru_inputs()` returns `initial_state = "zeros"` for exactly this reason.
A full-session stateful GRU is a **differently labelled history-policy experiment**, not this
contract, and must be reported as such.

## 7. Consumer warning — this is a per-decision ledger

> **Do not copy the original SASRec sampler's "last window per user" behaviour onto this
> ledger.** That sampler emits one training window per user. This dataset emits **one supervised
> decision per retained representative**. Applying the original behaviour would silently discard
> the large majority of supervised decisions and produce numbers that are not comparable with
> anything reported here.

Both primary loaders must retain **all** eligible decision ids and identical true targets; that
equality is asserted in `scripts/verify_training_ready.py`.

## 8. Negative sampling

`loaders.negative_sampler(seed, decision_id, epoch, n, catalogue_size, positive_class)` derives
candidates from a digest of (seed, decision id, epoch, sample index). It excludes the current
positive and, by construction, every special token. It does **not** exclude the user's future
history. Main evaluation uses the **full frozen catalogue**, allows non-consecutive revisits, and
does not filter previously seen items.
