# Measured audit findings (bounded SAMPLE) — 2026-09-25T10:36Z

## Scope and sources
- **Sample:** 3 of the 64 user buckets (0, 17, 42) of `model_ready/user_bucket=*/features.parquet`.
  - TRAIN rows: 2,460,225.
  - VALIDATION_SELECT rows: 506,612.
  - VC/TEST never read.
- **Label:** SAMPLE_NOT_POPULATION. These are not exact population statistics.
- **Runs:**
  - Profiler run on one core with a hard RSS guard: 36.5 s, peak 537 MiB (the package target is
    512 MiB; exceeded by 25 MiB for about 36 s, under the 600 MiB guard).
  - Duplicate check: 8 s, peak 396 MiB.
- **Result files:**
  - `audit/TRAIN_PROFILE_SAMPLE.json` (produced by `code/profile_train_sample.py`, sha ae0fd075…);
  - `audit/MASK_DUP_TIME_CHECK.json` (produced by `code/check_mask_dups.py`).
- **What the model receives:** the actual model input is built by the reused store
  (`evidence/snap/reuse/full_store.py:270-313`, `vendor/rees46_eda/loaders.py:54-63,290-312`).
  - Event numerics are re-zeroed only for `log1p_price` (price missing or conflict),
    `log1p_prev_raw_view_gap_s` and `log1p_item_observed_age_s`.
  - User context is re-zeroed only for `recent_mean_log_price` and
    `current_minus_recent_mean_log_price` (by `recent_price_supported`).

## M1 — fit-population defect in two user time features (PROJECT_MEASURED)
The dictionary and normalizers.json state that scalers are "fitted on VALID rows only (missing
sentinels excluded)" and that "numeric missing maps to zero after normalization". Two user-context
time features do not follow this:

| feature | fitted mean/std (normalizers.json) | all-rows sample mean/std | valid-only sample mean/std | unavailable rows (TRAIN sample) | model input when unavailable |
|---|---|---|---|---|---|
| `log1p_time_since_prev_session_last_event_s` | 1.8615 / 4.3549 | 1.8650 / 4.3591 | **11.820 / 1.663** | 2,072,042 / 2,460,225 (84.2%) | z(0) = **−0.4275** (not re-zeroed) |
| `log1p_time_since_prev_user_event_s` | 4.9503 / 3.3790 | 4.9525 / 3.3821 | 5.382 / 3.181 | 196,432 (8.0%, the cold users) | z(0) = **−1.4650** (not re-zeroed) |
| control: `log1p_prev_raw_view_gap_s` | 5.1736 / 3.2604 | 4.764 / 3.430 | **5.177 / 3.263** | 196,432 | re-zeroed to 0 (correct) |

- **Cause:** the fitted parameters match the all-rows population. The unavailable raw zeros were
  therefore included in the fit.
- **Why it happens:** neither feature is in `USER_MISSING_FLAG_OF` (`loaders.py:59-62`), so neither
  is re-zeroed.
- **Consequence for `prev_session`:**
  - valid values are encoded as (x − 1.86)/4.35, a valid-value spread of 0.38 instead of 1;
  - the encoded valid range is about [1.49, 2.97];
  - unavailable rows sit at −0.43;
  - the availability mask is present (`prev_session_available`, `prev_user_event_available`), so
    the information is not lost.
- **Classification:** a transform/policy defect, not leakage.
- **Expected impact on ranking:** HYPOTHESIS, small. The user branch MLP sees the mask.

## M2 — exact duplicates, measured on the whole sample with zero exceptions (TRAIN and VS)
Nine duplicated channels:
- `log1p_prefix_elapsed_s` == `log1p_segment_elapsed_s` (numeric, same fitted params).
- `prefix_item_count` == `prefix_representative_count` + 1 (numeric, same std, so identical after
  the z-score).
- `local_hour_sin/cos` is an exact fixed +3h rotation of `utc_hour_sin/cos` (max residual 6.6e-8).
  This is a linear function of the other pair.
- `local_weekday_sin/cos` equals `utc_weekday_sin/cos` on 97.4% of rows, and is exactly one weekday
  step on the rest. Not a linear duplicate.
- flag `price_zero` == flag `price_missing`.
- flag `main_category_missing` == flag `category_code_missing`.
- user value `has_history` == mask `has_history_mask` == mask `prev_user_event_available`.
- mask `recent_price_supported` differs from `has_history_mask` on 204 TRAIN rows (not a duplicate).

## M3 — dead inputs
Six quality flags are constant 0 in both the TRAIN and VS samples:
- `price_conflict`, `brand_conflict`;
- `category_missing`, `category_recovered`, `category_conflict`;
- `category_code_recovered`.

A zero input gets zero gradient, so its weight column stays at Xavier init. It has no effect while
the input stays 0.

## M4 — missingness and OOV rates (TRAIN sample)
- **Brand:** MISSING 15.7%.
- **Main category:** MISSING 34.0% (it is derived from category_code). `category_idx` is always known.
- **Product in context:** OOV only 0.13%. The OOV-collision risk for a repeat feature (D3) is small.
- **Price:** missing/zero 0.23% in TRAIN, 0.54% in VS.
- **Cold users** (has_history = 0): 8.0% in TRAIN, 3.3% in VS. Their encoded vector is constant.

## M5 — TRAIN → VS drift (expected temporal drift, reported for probe interpretation)
- **Calendar:** VS is a short window. `is_weekend` has mean 0.756 in VS versus 0.251 in TRAIN, and
  the weekday sin/cos means shift strongly.
- **User history:** VS users have more history. `recent_history_length` median is 34 in VS versus
  15 in TRAIN, and `log1p_prev_view_count` is higher.
- **Item age:** `log1p_item_observed_age_s` mean is 15.03 in VS versus 14.17 in TRAIN, because VS
  uses the frozen TRAIN first-seen.
- **Session position:** prefix counts are higher in VS (median `prefix_item_count` 5 versus 4).
- **Price:** no drift (median `log1p_price` 5.09 in both).

## Not established
- Population-exact statistics.
- VC/TEST behaviour of the constant flags (not read, by rule).
- The byte identity of local `next_item_eda/src/rees46_eda/userctx.py` with the frozen build
  (NOT_VERIFIED; `loaders.py` and `config.py` are verified identical).
