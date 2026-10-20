# Feature dictionary — `rees46_next_item_shared_v4_localtime`

Generated 2026-09-21T14:44:48Z from the executed run.

Both backbones receive an **identical** feature representation. A feature marked
`IDENTIFIER ONLY` is carried for provenance and is not a default predictor.

## Special categorical tokens (identical for every categorical field)

| token | value | meaning |
|---|---|---|
| `PAD` | 0 | padded position; never a real event |
| `MISSING` | 1 | value genuinely unobserved. **Never a valid model event for `product_idx`** |
| `OOV` | 2 | observed but outside the frozen TRAIN vocabulary |
| known values | >= 3 | sorted deterministic assignment |

Output classes are a **separate** compact `0..K-1` space over REAL catalogue items only,
with an explicit inverse mapping back to input item tokens and raw product ids. PAD, MISSING
and OOV are therefore structurally incapable of being a correct class.

## Features

| feature | type | units | formula | availability time | fitting scope | null / OOV policy | clipping & scaling | role | model compatibility |
|---|---|---|---|---|---|---|---|---|---|
| `product_idx (batch key `item_tokens`)` | int64 categorical | contiguous index | catalogue lookup of the representative's raw product_id | decision time (the observed event itself) | TRAIN representatives, support >= 5 | not in catalogue -> OOV=2; PAD=0 on padded positions; MISSING=1 never valid | none | INPUT | GRU + SASRec |
| `category_token` | int64 categorical | token id | vocabulary lookup of category_id_resolved | as-of the event (TRAIN) / frozen end-of-TRAIN snapshot (future) | TRAIN resolved values | NULL -> MISSING=1; unseen -> OOV=2 | none | INPUT | GRU + SASRec |
| `brand_token` | int64 categorical | token id | vocabulary lookup of brand_resolved | as-of the event / frozen snapshot | TRAIN resolved values | NULL -> MISSING=1; unseen -> OOV=2 | none | INPUT | GRU + SASRec |
| `main_category_token` | int64 categorical | token id | first dot-delimited root of category_code_resolved | as-of the event / frozen snapshot | TRAIN resolved values | NULL -> MISSING=1; unseen -> OOV=2 | none | INPUT | GRU + SASRec |
| `category_code_token` | int64 categorical | token id | vocabulary lookup of the FULL category_code_resolved | as-of the event / frozen snapshot | TRAIN resolved values | NULL -> MISSING=1; unseen -> OOV=2 | none | OPTIONAL SELECTOR (stored, not a required primary input) | GRU + SASRec |
| `log1p_price` | float32 | log(1 + currency units) | ln(1 + price) for a finite AND STRICTLY POSITIVE observed block price (price == 0 is INVALID for this channel, not a free item) | decision time | TRAIN | missing/negative/conflicting -> numeric missing + flag | clip TRAIN 0.1%/99.9% in log space, then z-score | INPUT | GRU + SASRec |
| `utc_hour_sin / utc_hour_cos` | float32 | dimensionless | sin/cos(2*pi*(hour + minute/60 + second/3600)/24) | decision time | n/a (closed form) | never missing | NEVER clipped or scaled | INPUT | GRU + SASRec |
| `local_hour_sin / local_hour_cos` | float32 | dimensionless | sin/cos(2*pi*(local_hour + minute/60 + second/3600)/24) where local = UTC + 3h - a CONTEXTUAL clock hypothesis about the audience, NOT confirmed store geography; canonical time stays UTC everywhere | decision time | n/a (closed form; NOT fitted) | never missing - a whole-hour shift cannot fail | NEVER clipped or scaled | INPUT (F2, F3, F4) | GRU + SASRec |
| `local_weekday_sin / local_weekday_cos` | float32 | dimensionless | sin/cos(2*pi*(isodow(local_ts)-1)/7) - weekday taken AFTER the shift, so a late-night UTC Friday is a local Saturday | decision time | n/a (closed form; NOT fitted) | never missing | NEVER clipped or scaled | INPUT (F2, F3, F4) | GRU + SASRec |
| `is_weekend` | float32 binary 0/1 | indicator | 1 when isodow(local_ts) >= 6 | decision time | n/a (closed form) | never missing | NEVER clipped or scaled | INPUT (F2, F3, F4) | GRU + SASRec |
| `local_hour` | int32 0..23 | local hour | extract(hour FROM UTC + 3h) | decision time | n/a | never missing | none - AUDIT/EDA column, not a model channel | AUDIT ONLY | not a model input |
| `local_daypart` | string | band name | frozen v1 bands on the LOCAL hour: NIGHT 00-06, MORNING 06-12, AFTERNOON 12-17, EVENING 17-21, LATE_EVENING 21-24 | decision time | n/a (declared, not fitted) | structurally cannot be missing - the bands tile 0..23 with no gap | none - readable companion to daypart_idx | AUDIT ONLY | not a model input |
| `daypart_idx (batch key `daypart_tokens`)` | int64 categorical | contiguous index | NIGHT=3, MORNING=4, AFTERNOON=5, EVENING=6, LATE_EVENING=7 | decision time | declared closed vocabulary (every value occurs in TRAIN) | structurally cannot be PAD/MISSING/OOV on a real event; PAD=0 only on padded positions | none | INPUT (F2, F3, F4) | GRU + SASRec |
| `local_weekday_idx (batch key `weekday_tokens`)` | int64 categorical | contiguous index | Monday=3 .. Sunday=9, computed AFTER the timezone shift | decision time | declared closed vocabulary | structurally cannot be PAD/MISSING/OOV on a real event | none | INPUT (F2, F3, F4) | GRU + SASRec |
| `utc_weekday_sin / utc_weekday_cos` | float32 | dimensionless | sin/cos(2*pi*(isodow-1)/7) | decision time | n/a (closed form) | never missing | NEVER clipped or scaled | INPUT | GRU + SASRec |
| `log1p_prev_raw_view_gap_s` | float32 | log(1 + seconds) | ln(1 + seconds since the user's previous VALID raw view block, pre-compression) | uses only strictly earlier timestamps | TRAIN | no previous view -> 0 after normalization + prev_gap_unavailable | clip + z-score in log space | INPUT | GRU + SASRec |
| `log1p_segment_elapsed_s` | float32 | log(1 + seconds) | ln(1 + seconds since the model segment's first block) | decision time | TRAIN | never missing (0 at segment start) | clip + z-score | INPUT | GRU + SASRec |
| `log1p_item_observed_age_s` | float32 | log(1 + seconds) | ln(1 + seconds since the item's first EARLIER TRAIN occurrence; frozen TRAIN first-seen for future partitions) | strictly earlier occurrences only | TRAIN | first occurrence / new future item -> 0 + item_age_unavailable | clip + z-score | INPUT | GRU + SASRec |
| `prefix_representative_count` | float32 | count | position_in_segment - 1 | decision time | TRAIN | never missing | clip + z-score | INPUT | GRU + SASRec |
| `prefix_item_count` | float32 | count | position_in_segment (through the current event) | decision time | TRAIN | never missing | clip + z-score | INPUT | GRU + SASRec |
| `prefix_known_category_count` | float32 | count | distinct known main categories in the segment prefix | decision time | TRAIN | unknown categories are not counted | clip + z-score | INPUT | GRU + SASRec |
| `prefix_category_switch_count` | float32 | count | running count of CHANGED transitions in the segment prefix | decision time | TRAIN | UNKNOWN transitions are not counted as switches | clip + z-score | INPUT | GRU + SASRec |
| `current_known_category_run_length` | float32 | count | length of the current consecutive run of the same KNOWN main category | decision time | TRAIN | 0 when the current category is unknown | clip + z-score | INPUT | GRU + SASRec |
| `log1p_prefix_elapsed_s` | float32 | log(1 + seconds) | ln(1 + seconds since the segment start; identical to segment elapsed because prefix state resets at the model-segment boundary) | decision time | TRAIN | never missing | clip + z-score | INPUT | GRU + SASRec |
| `log1p_prev_view_count` | float32 | log(1 + count) | ln(1 + number of the user's previous available representatives in this replay stream) | strictly earlier events of the same user | TRAIN | 0 for a cold user + has_history=0 | clip + z-score | INPUT | GRU + SASRec |
| `log1p_prev_session_count` | float32 | log(1 + count) | ln(1 + distinct behavioural sessions observed so far) | strictly earlier events | TRAIN | 0 for a cold user | clip + z-score | INPUT | GRU + SASRec |
| `recent_history_length` | float32 | count | min(previous events, 100) | strictly earlier events | TRAIN | 0 for a cold user | clip + z-score | INPUT | GRU + SASRec |
| `has_history` | float32 (0/1) | indicator | 1 when the user has any previous event | strictly earlier events | n/a | 0 for a cold user | NEVER clipped or scaled | INPUT + MASK | GRU + SASRec |
| `recent_distinct_items` | float32 | count | exact distinct product ids in the 100 most recent PREVIOUS representatives | strictly earlier events | TRAIN | 0 for a cold user | clip + z-score | INPUT | GRU + SASRec |
| `recent_distinct_known_main_categories` | float32 | count | exact distinct KNOWN main categories in the recent-100 window | strictly earlier events | TRAIN | unknown categories excluded; 0 when none | clip + z-score | INPUT | GRU + SASRec |
| `recent_category_match_share` | float32 in [0,1] | share | share of KNOWN recent main categories equal to the current KNOWN main category | strictly earlier events | n/a (already bounded) | 0 + recent_category_match_supported=0 when the current or all recent categories are unknown; two unknowns NEVER match | NEVER clipped or scaled | INPUT | GRU + SASRec |
| `recent_brand_match_share` | float32 in [0,1] | share | share of KNOWN recent brands equal to the current KNOWN brand | strictly earlier events | n/a | 0 + recent_brand_match_supported=0 when unsupported | NEVER clipped or scaled | INPUT | GRU + SASRec |
| `recent_mean_log_price` | float32 | log(1 + currency units) | mean log1p(price) over valid prices in the recent-100 window | strictly earlier events | TRAIN | 0 after normalization + recent_price_supported=0 | clip + z-score | INPUT | GRU + SASRec |
| `current_minus_recent_mean_log_price` | float32 | log units | log1p(current price) - recent_mean_log_price when both are available | current event + strictly earlier events | TRAIN | 0 after normalization when either side is unavailable | clip + z-score | INPUT | GRU + SASRec |
| `log1p_time_since_prev_user_event_s` | float32 | log(1 + seconds) | ln(1 + seconds since the user's previous representative | strictly earlier events | TRAIN | 0 + prev_user_event_available=0 for a cold user | clip + z-score | INPUT | GRU + SASRec |
| `log1p_time_since_prev_session_last_event_s` | float32 | log(1 + seconds) | at a NEW behavioural session, ln(1 + seconds since the previous session's last available event) | strictly earlier events | TRAIN | 0 + prev_session_available=0 otherwise | clip + z-score | INPUT | GRU + SASRec |
| `event_quality_flags[Q]` | int8 (0/1) | indicator | per-event missing / recovered / conflict / availability indicators | decision time | n/a | these ARE the missing indicators | NEVER clipped or scaled | INPUT | GRU + SASRec |
| `decision_id / client_id / model_segment_id` | int64 | identifier | stable identifiers carried through the batch; client_id is the PSEUDONYMOUS client_idx, never the raw user id | n/a | client_idx is fitted on TRAIN clients ONLY | a client absent from TRAIN resolves to the reserved OOV=2, never a valid-looking index; ~45% of future-partition clients are in that state | none | IDENTIFIER ONLY - not a default predictor | neither backbone consumes these as features |
| `user_id / product_id / category_id (artifacts)` | STRING | identifier | the trimmed SOURCE STRING; canonical identity in every canonical, audit, intermediate and decision-ledger artifact. Never float, never rounded | n/a | n/a | NULL only when the source token is missing | none | PROVENANCE - never a model input | not consumed by either backbone |
| `target_class` | int64 | compact class id | catalogue output class of the observed next different product, -1 when none is rankable | STRICTLY AFTER the decision time | TRAIN catalogue | -1 when unobserved/immature; OOV targets keep their raw identity and score zero | none | LABEL | GRU + SASRec |

## Normalization policy

- Fit scope: **TRAIN only**. Clip bounds are the TRAIN 0.1% / 99.9% quantiles of the
  **already log-transformed** value (log first, clip in transformed space), then mean/std
  are computed on the clipped TRAIN values.
- Sine/cosine components and boolean flags are **never** clipped or scaled.
- A constant feature maps to zero with scale 1.
- Numeric missing maps to **zero after normalization**, always accompanied by its missing
  indicator — never to `(0 - mean) / std`.
- Unclipped raw values are preserved in the artifacts; clipping rates are exported to
  `reports/tables/06_numeric_clipping_rates.csv`.
- Every tensor value is finite; this is asserted on the full data.

### Fitted parameters actually used

| feature | clip_lo | clip_hi | mean | std | constant |
|---|---|---|---|---|---|
| `current_known_category_run_length` | 0 | 75 | 3.36355 | 6.37329 | no |
| `current_minus_recent_mean_log_price` | -4.09069 | 3.68137 | 0.0020607 | 0.993332 | no |
| `log1p_item_observed_age_s` | 8.02976 | 15.1914 | 14.2241 | 1.01729 | no |
| `log1p_prefix_elapsed_s` | 0 | 9.56107 | 4.54889 | 2.83005 | no |
| `log1p_prev_raw_view_gap_s` | 0.693147 | 14.8914 | 5.17361 | 3.26035 | no |
| `log1p_prev_session_count` | 0 | 4.49981 | 1.46028 | 0.878214 | no |
| `log1p_prev_view_count` | 0 | 7.23346 | 2.71272 | 1.4865 | no |
| `log1p_price` | 1.15688 | 7.84912 | 5.04455 | 1.20851 | no |
| `log1p_segment_elapsed_s` | 0 | 9.56107 | 4.54889 | 2.83005 | no |
| `log1p_time_since_prev_session_last_event_s` | 0 | 14.8779 | 1.86154 | 4.35493 | no |
| `log1p_time_since_prev_user_event_s` | 0 | 14.8779 | 4.95025 | 3.37901 | no |
| `prefix_category_switch_count` | 0 | 7 | 0.149452 | 0.578631 | no |
| `prefix_item_count` | 1 | 141 | 7.71678 | 11.9083 | no |
| `prefix_known_category_count` | 0 | 5 | 0.906572 | 0.686465 | no |
| `prefix_representative_count` | 0 | 140 | 6.71678 | 11.9083 | no |
| `recent_distinct_items` | 0 | 98 | 21.4812 | 23.5508 | no |
| `recent_distinct_known_main_categories` | 0 | 8 | 1.6579 | 1.37781 | no |
| `recent_history_length` | 0 | 100 | 28.8617 | 32.2929 | no |
| `recent_mean_log_price` | 1.90528 | 7.46742 | 5.03827 | 0.882684 | no |

## Deliberately NOT enabled

| capability | status | reason |
|---|---|---|
| learned user-ID embedding | not fitted | not required by the main contract; the dense id is available for a separately named personalization experiment |
| session UUID embedding | excluded | a UUID is an opaque key, not a learnable property |
| relative category price / per-user median price | optional, off | not required for initial readiness; would need its own causality and resource test |
| purchase / cart derived signals | excluded | this is a view-only next-item task |
| future repeat span, final run length, last-repeat timestamp | audit only | using them would leak the future into the first representative |
