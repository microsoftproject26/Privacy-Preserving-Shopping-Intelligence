# Open questions — cannot be answered from code alone, need a bounded data sample

Each item: the question, why code cannot settle it, and the exact measurement
`code/profile_train_sample.py` performs. All measurements are labelled SAMPLE (a declared
deterministic subset of buckets), never claimed as exact population statistics.

## Q1. Exact encoded vector of a cold user (`has_history=0`) in `user_context` / `user_context_masks`

- Why code can't settle it: the zero-fill/normalization chain is defined
  (`reuse/vendor/rees46_eda/loaders.py:60-63,303-312`; `reuse/full_store.py:296-313`), but the
  EXACT floating-point vector a cold user actually receives depends on which of the 12
  `user_context` entries are gated by `has_history_mask` vs. their own more specific support flag
  (`recent_category_match_supported`, `recent_brand_match_supported`, `recent_price_supported`,
  `prev_user_event_available`, `prev_session_available`) — and whether ALL of them are exactly 0
  for a genuinely cold user, or whether some entries (e.g. `recent_category_match_share`, which has
  its OWN support flag rather than `has_history_mask`) could theoretically read non-zero if the
  upstream population definitions of "cold" ever diverge between masks.
- Measurement: for a bounded TRAIN sample, select all rows where `user_context[3]` (has_history)
  == 0, report the full 12-value `user_context` vector and 6-value `user_context_masks` vector
  (mean/min/max/n_distinct over that subgroup — should collapse to a single constant vector if the
  implementation is exactly as documented).

## Q2. Encoded value of category-derived counts/shares when the category is unknown: raw 0 -> `z(0)` or 0-after-normalization?

- Why code can't settle it: `prefix_known_category_count`, `prefix_category_switch_count`,
  `current_known_category_run_length` are **not** in `MISSING_FLAG_OF`/`USER_MISSING_FLAG_OF`
  (`reuse/vendor/rees46_eda/loaders.py:55-63`) — i.e. they are NOT force-zeroed by any flag when
  the category is unknown; per `FEATURE_DICTIONARY.md:45-47` a "0" for these features is a **true
  count** (e.g. "0 when the current category is unknown" is the correct semantic value, not a
  missing sentinel), so it goes through the ordinary `clip+z-score` path and becomes `z(0) =
  (0-mean)/std`, a NON-zero encoded float. This is a real behavioural difference from the
  documented "numeric missing maps to zero after normalization" convention
  (`FEATURE_DICTIONARY.md:73-74`), because these are not "missing", they are genuinely zero. Code
  alone establishes the RULE but not the observed frequency/impact.
- Measurement: for a bounded TRAIN sample, report the encoded (z-scored) value distribution of
  these three features restricted to rows where the relevant category-unknown condition holds
  (`main_category_tokens` in {MISSING=1, OOV=2}), confirming they cluster at `z(0)` (not literal
  0.0) as the formula predicts, and report what fraction of all rows this affects.

## Q3. The `current_minus_recent_mean_log_price` masking discrepancy (see REDUNDANCY_FINDINGS)

- Why code can't settle it: upstream (`next_item_eda/src/rees46_eda/userctx.py:220-222`, NOT
  byte-identity-verified against the frozen build) zeroes this feature when `both = price_sup &
  price_ok` is false — i.e. when EITHER the recent window OR the CURRENT event's own price is
  invalid. Downstream re-zeroing at collate/batch time
  (`reuse/vendor/rees46_eda/loaders.py:60-63,309-312`; `reuse/full_store.py:296-307`) keys ONLY on
  `recent_price_supported`, not on the current event's `price_missing`/`price_zero`/`price_conflict`
  flags. If `recent_price_supported=1` but the current price is itself invalid, the RAW value
  stored in the parquet feature table is already 0.0 (per the upstream gate), so the downstream
  code does not corrupt it further — but if any future rebuild ever recomputed this column at
  batch time instead of reading the pre-materialized parquet value, the two gates would disagree.
  Code alone cannot show whether this row population is empty (upstream and downstream always
  agree in practice) or non-empty (a real edge case) in the CURRENT frozen data.
- Measurement: for a bounded TRAIN+VALIDATION_SELECT sample, count rows where
  `recent_price_supported==1` AND (`price_missing==1` OR `price_zero==1` OR `price_conflict==1`)
  on the SAME event, and report the encoded `current_minus_recent_mean_log_price` value for those
  rows (expect exactly the z-scored image of 0.0 if upstream materialization is what actually
  reaches the model; flag any row where it is not).

## Q4. The grain/population each scaler was fitted on: event rows, representatives, or decisions?

- Why code can't settle it: `normalizers.json` states `"fit_scope": "TRAIN"` and "fitted on VALID
  rows only (missing sentinels excluded from the fit)" but does not distinguish between (a) every
  retained EVENT row (`features.parquet`, one row per representative-view, `~52.76M` TRAIN
  decisions is the DECISION count per `splits.json`, not the row count), (b) the decision-level
  population after the per-decision context window is applied, or (c) some other resampling. The
  dictionary and normalizers.json describe the RULE ("log first, clip in transformed space,
  mean/std on clipped TRAIN values") but not literally which table's row count the mean/std divide
  by. `protocol.resolved.yaml:119` says catalogue fit_scope is `full_valid_train_population`
  (representative-view support), which may or may not be the same population the numeric scalers
  were fit on.
- Measurement: recompute the sample mean/std of a subset of numeric features (e.g. `log1p_price`,
  `prefix_item_count`) over the declared TRAIN buckets' VALID rows at both the per-event-row grain
  and the per-decision grain, and compare against the frozen `normalizers.json` values (mean
  5.04455/std 1.20851 for `log1p_price`) to see which population the sample statistic converges
  toward. This cannot prove exact equality to the true (full-population) fit, but a large
  divergence at one grain and a close match at the other is informative.

## Q5. Whether raw product identity is available read-only for OOV-safe repeat features

- Why code can't settle it: `item_tokens` maps unseen-in-TRAIN products to `OOV=2`
  (`FEATURE_DICTIONARY.md:13-19`), which collapses ALL out-of-vocabulary products to the same
  token — so any feature computed downstream from `item_tokens` alone (none currently exist in the
  A1 rich channel set, but this affects future feature design) cannot distinguish "the same OOV
  product repeated" from "two different OOV products in a row." The raw string `product_id` is
  preserved end-to-end in canonical/decision-ledger artifacts
  (`DATA_CONTRACT.md:177-207`), so in principle a read-only join against
  `vocabularies/catalogue.parquet` / the decision ledger could recover raw identity for OOV rows,
  but nothing in the current A1 batch pipeline (`full_store.py`, `loaders.py`) does this join.
- Measurement: `profile_train_sample.py` reports, for the declared bucket sample, the fraction of
  context-window positions whose `item_tokens == OOV(2)`, and whether the RAW `product_id` string
  is recoverable at all from the columns actually projected by `_needed_feature_columns` (F4) —
  expected answer is NO (raw product_id is not among the projected columns), which should be
  reported explicitly rather than assumed.

## Q6. How often OOV tokens occur in context (frequency, not just existence)

- Why code can't settle it: `catalogue_size=158486`, `client_vocabulary_size=4204379` and the
  44.890% future-client-OOV rate are documented at the CLIENT level
  (`TRAINING_READINESS.json` warnings), and `TARGET_OOV_ON_TEST` gives the OOV rate for the
  TARGET on TEST (3.383%), but there is no published rate for **item-token OOV inside the TRAIN
  context window itself** (i.e. how often a position in `item_tokens`/`category_tokens`/
  `brand_tokens`/`main_category_tokens` is OOV=2 rather than a known token), which is exactly what
  the rich arm's embeddings see at train time.
- Measurement: for the declared bucket sample (TRAIN split), report OOV rate (and separately
  MISSING rate) per categorical channel (`item_tokens`, `category_tokens`, `brand_tokens`,
  `main_category_tokens`, `daypart_tokens`, `weekday_tokens`) over VALID (non-PAD) positions only.

## Q7 (bonus, surfaced by this audit). Is `category_missing` ever true on TRAIN given `residual_missing_rates.category_id == 0.0`?

- `TRAINING_READINESS.json` reports TRAIN `category_id` residual missing rate as exactly 0.0 and
  `RECOVERY_ZERO_YIELD:category_id — nothing was missing, so the enabled family had nothing to
  resolve` — implying `category_missing` should be constant-0 on TRAIN. Code alone (the flag's
  existence in `EVENT_QUALITY_FLAGS`) does not prove it is actually always 0 in the materialized
  parquet.
- Measurement: `profile_train_sample.py`'s standard per-feature profile includes a `constant` flag
  for every quality flag; this question is answered as a side effect of the main profiling pass
  (no separate query needed), but is called out here because a non-constant result would directly
  contradict a documented readiness-gate number.
