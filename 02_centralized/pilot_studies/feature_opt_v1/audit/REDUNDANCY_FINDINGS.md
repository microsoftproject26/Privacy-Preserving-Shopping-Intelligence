# Redundancy findings — SASRec A1 (rich_frozen_channels) rich arm

Scope: exact or algebraic duplicates, and deterministic derivations, found IN CODE (formulas +
fitted params) among the channels that actually reach the A1 model. Each finding states whether
it is proven from code alone or needs a bounded data check (`code/profile_train_sample.py`).

## 1. `price_zero` is an exact duplicate of `price_missing` in this dataset (PROVEN ON FULL DATA)

- Formula: `price==0` is defined INVALID for the numeric channel (protocol-owner decision,
  deviating from the frozen MASTER_PROMPT text) — `price_zero=1` and `price_missing=1` are set
  together whenever the source recorded a literal zero.
  `evidence/snap/dataset/PROTOCOL_DEVIATIONS.md:42-61`.
- Proof (already measured on the full dataset, not just in code): "of the feature rows carrying
  either flag, 198,060 of 198,060 carry both, and no other combination exists (`price_conflict`
  never fires)." `evidence/snap/dataset/PROTOCOL_DEVIATIONS.md:69-76`.
- Consequence: `price_zero` carries **no information beyond `price_missing`** in this run. The
  document itself is explicit that this is an empirical fact of this dataset, not a structural
  guarantee — "worth keeping for audit and for future data where the two could diverge... must
  not be reported as an additional informative signal here."
- Where both reach the model: both flags are part of `event_quality_flags` (indices 0 and 1),
  concatenated into `numflag_proj` alongside `log1p_price`
  (`sasrec_v1/code/sasrec_model.py:53,167,251-256`).
- Break condition: none observed in this data (0 negative/nonfinite/unparseable prices were
  exercised — `PROTOCOL_DEVIATIONS.md:78-84`); would only diverge on new data where `price_conflict`
  actually fires.
- **Needs a data check?** No — already proven on full data by the pipeline's own audit.

## 2. `log1p_prefix_elapsed_s` is an exact duplicate of `log1p_segment_elapsed_s`

- Formula: `log1p_prefix_elapsed_s` = ln(1+seconds since segment start); FEATURE_DICTIONARY states
  it is "identical to segment elapsed because prefix state resets at the model-segment boundary"
  (`evidence/snap/dataset/FEATURE_DICTIONARY.md:41,48`).
- Fitted-parameter proof: `evidence/snap/dataset/normalizers.json` — both features have
  `clip_lo=0.0`, `clip_hi=9.561067581176758`, `mean=4.548892813156198`,
  `std=2.8300548438973165` (segment) vs `std=2.8300548438974196` (prefix) — equal to float64
  rounding noise (1e-13 relative), i.e. fit on the same underlying values.
- Where both reach the model: both are entries 11 and 18 of `event_numeric_features`
  (`reuse/vendor/rees46_eda/config.py:353-371,424-429`), both z-scored independently via the
  **same** `Normalizers.apply` call (`reuse/vendor/rees46_eda/loaders.py:75-82`) and both fed
  through `numflag_proj` (`sasrec_v1/code/sasrec_model.py:167,253-254`) — the rich arm literally
  doubles this one signal's weight in the 34-wide numeric+flags input.
- Break condition: the two features would diverge only if a future feature-set version changed
  where "prefix" resets relative to "segment" (currently declared identical by construction, not
  observed to diverge in TRAIN).
- **Needs a data check?** Recommended but low priority — `profile_train_sample.py` includes an
  exact-equality check (`log1p_segment_elapsed_s == log1p_prefix_elapsed_s` on VALID rows) to
  confirm no post-hoc micro-divergence (e.g. float32 rounding at different code paths).

## 3. `prefix_item_count` is an exact affine duplicate of `prefix_representative_count`

- Formula: `prefix_representative_count = position_in_segment - 1`;
  `prefix_item_count = position_in_segment` (through the current event) —
  `evidence/snap/dataset/FEATURE_DICTIONARY.md:43-44`. So
  `prefix_item_count = prefix_representative_count + 1` **exactly**, for every row.
- Fitted-parameter proof: `normalizers.json` — `prefix_representative_count`
  `{clip_lo:0, clip_hi:140, mean:6.716780248881764, std:11.908258422307725}` vs
  `prefix_item_count` `{clip_lo:1, clip_hi:141, mean:7.716780248881764, std:11.90825842230758}` —
  clip bounds and mean shifted by exactly +1, std unchanged to 1e-12. This is the fingerprint of
  an affine (`y = x + 1`) relationship surviving the shared clip/z-score pipeline.
- Where both reach the model: entries 13 and 14 of `event_numeric_features`, both through
  `numflag_proj` (same fusion point as finding 2).
- Break condition: none within TRAIN's clip range; would only diverge if one side clipped and the
  other did not, which cannot happen since the clip bounds are also offset by exactly 1 and both
  cover the full observed range (0..140 / 1..141).
- **Needs a data check?** Recommended — `profile_train_sample.py` checks
  `prefix_item_count - prefix_representative_count == 1` on VALID rows exactly (pre-normalization
  raw values are not directly recoverable from the z-scored tensor alone without inverting the
  known `mean`/`std`, so the check inverts the transform using `normalizers.json`'s own fitted
  parameters and confirms the raw difference is exactly 1).

## 4. `local_hour_sin/cos` is an exact fixed-phase rotation of `utc_hour_sin/cos`

- Formula: local time = UTC + 3h, a **fixed** offset with **no DST transition** inside the
  2019-10-01..2019-11-30 window (`reuse/vendor/rees46_eda/config.py:229-238,253-255`,
  `CONTEXTUAL_TZ_OFFSET_HOURS = 3`). Since local_hour = (utc_hour + 3) mod 24 for every row with
  no exception, `local_hour_sin = sin(theta + phi)`, `local_hour_cos = cos(theta + phi)` for a
  **constant** `phi = 2*pi*3/24`, i.e. an exact 2x2 rotation matrix applied to
  `(utc_hour_sin, utc_hour_cos)`:
  ```
  local_hour_sin = utc_hour_sin*cos(phi) + utc_hour_cos*sin(phi)
  local_hour_cos = utc_hour_cos*cos(phi) - utc_hour_sin*sin(phi)
  ```
- Where both reach the model: entries 1,2 (utc) and 5,6 (local) of `event_numeric_features`, both
  through the same `numflag_proj` (rich arm carries BOTH the UTC and local-time versions of the
  same underlying diurnal cycle — 4 of the 19 numeric channels, plus `daypart_tokens`, encode one
  signal three times over).
- Break condition: **none** — the offset is provably constant across the entire frozen window
  (`config.py:229-233`: "Russia has observed permanent standard time since October 2014... no DST
  transition inside the data window, so a fixed +3h shift is exact here"). This is a code-provable
  exact duplicate, not merely likely.
- **Needs a data check?** No — the rotation is a closed-form identity given a provably-constant
  offset; `profile_train_sample.py` still includes it as a cheap corroborating check since it is
  nearly free once the numeric block is loaded.

## 5. `local_weekday_sin/cos` is IDENTICAL to `utc_weekday_sin/cos` on non-boundary rows, and a one-weekday-step rotation on boundary rows

- Same +3h basis as finding 4, but `local_weekday` is computed from `isodow(local_ts)`
  (`reuse/vendor/rees46_eda/config.py:277-279`, `FEATURE_DICTIONARY.md:33` — "weekday taken AFTER
  the shift, so a late-night UTC Friday is a local Saturday"). Weekday only advances when the
  local CALENDAR DATE differs from the UTC calendar date, i.e. only when the +3h shift crosses
  local midnight. For the ~21/24 of a day where `utc_hour + 3 < 24` (i.e. UTC hour < 21), the
  local date equals the UTC date, so `local_weekday_idx == utc_weekday_idx` **exactly** — this is
  a STRAIGHT exact duplicate on those rows, not merely a rotation of the hour-cycle's `phi` (an
  earlier draft of this finding incorrectly described it as an hour-phase rotation; corrected here
  after re-deriving it directly and cross-checking with `profile_train_sample.py`'s
  `check_weekday_rotation`). For UTC hours **21:00–23:59:59**, the +3h shift crosses local
  midnight and `local_weekday = utc_weekday + 1 (mod 7)` — a discontinuous rotation by exactly
  `2*pi/7` in the weekday sin/cos plane (a completely different phase than local_hour's `2*pi*3/24`).
- Where both reach the model: entries 3,4 (utc) and 7,8 (local) of `event_numeric_features`.
- Break condition: **exactly the UTC-hour-21-to-24 rows** (documented, not hypothetical — this is
  the same day-boundary logic already described in `FEATURE_DICTIONARY.md:33`). This is the
  packet's "sin/cos pair that is an exact rotation... mark whether equality can break" case:
  identity holds on ~87.5% of rows and is replaced by a one-step (2*pi/7) rotation on the
  remaining ~12.5% by construction.
- **Needs a data check?** Yes — `profile_train_sample.py::check_weekday_rotation` reports, per
  bucket, how many VALID rows satisfy the identity hypothesis, how many satisfy the one-day-
  forward hypothesis, and how many satisfy NEITHER (`n_unexplained`, expected to be exactly 0 if
  this finding's physics is right); this also sanity-checks that the ~12.5% boundary estimate
  (3/24 hours) is what is actually observed.

## 6. `is_weekend` / `daypart_tokens` / `weekday_tokens` are lossy (many-to-one) derivations of the same local-time signal already carried by `local_hour_sin/cos` + `local_weekday_sin/cos`

- `is_weekend = 1[isodow(local_ts) >= 6]` — a coarsening of `local_weekday_sin/cos` /
  `weekday_tokens` (`FEATURE_DICTIONARY.md:34`).
- `daypart_tokens` is a 5-band coarsening of `local_hour_sin/cos` (`config.py:242-248`).
- `weekday_tokens` is the token form of the same weekday identity as `local_weekday_sin/cos`.
- These are **not exact duplicates** (many raw values map to the same coarse bucket, so the map is
  not invertible), but they are **deterministic, zero-degree-of-freedom functions** of information
  already present elsewhere in the rich arm's numeric+categorical channels. Counted in
  `FEATURE_DEPENDENCIES.json`'s CALENDAR `total_closure`, not in the exact-duplicate list above.
- **Needs a data check?** No — deterministic by formula; not a candidate for a numeric equality
  test (different value spaces: continuous sin/cos vs. discrete token vs. binary indicator).

## Not found to be redundant (checked and ruled out)

- `category_missing` vs `category_code_missing`: cover DIFFERENT fields (`category_id_resolved`
  vs. the full `category_code_resolved`); TRAIN residual missing rates differ sharply (0.0% vs.
  33.983%, `evidence/snap/dataset/TRAINING_READINESS.json`), so they are not measuring the same
  thing despite the naming similarity.
- `current_minus_recent_mean_log_price` vs. `log1p_price`/`recent_mean_log_price`: this IS a
  direct algebraic function of the other two (`current - recent_mean`) when both are available,
  but it is listed as a **derivation with a masking discrepancy**, not a redundancy, in
  `OPEN_QUESTIONS.md` — the concern is a possible encoding inconsistency at the missing-value
  boundary, not that the channel is informationless.
