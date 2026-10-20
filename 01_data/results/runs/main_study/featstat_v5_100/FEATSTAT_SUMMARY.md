# FEATSTAT-01 — feature statistics and normalizer provenance on the 5–100 MAIN pool (TRAIN only)

- Evidence: PROJECT_MEASURED by `code/featstat_v1` on the server; receipt stage `FINAL`, `date -u` 2026-09-26T20:06:13Z → 2026-09-26T20:46:25Z, wall 2412.3 s, peak RSS 12133 MiB (guard 16,384).
- Pool (5 ≤ n ≤ 100): 1,446,986 users / 28,623,386 supervised examples; comparison (n ≥ 1): 2,694,998 / 40,180,608. All 9 census counts reproduced: True.
- Clean V2 rows via v4data: 32,768 real CleanV2Store windows compared byte for byte, mismatches 0; encode_rows max |diff| 4.77e-07; RC1–RC7 on every decision row: 0 violations. Inputs sha256 = FILES.json: 132 files, mismatches 0.
- Grain: one row per supervised TRAIN example (its own event = last_valid_index). The JSON also holds a window-exposure grain computed under the v1 SESSION_L50 window; the owner has since approved USER_HISTORY_L50 (DECISION_LOG 35-36), so those window numbers are SUPERSEDED, descriptive only, and not used by any rule here; the window statistics of record are DATA-05's. VALIDATION/TEST not read.

## 1. Near-constant and dead channels on the pool (mode share)

| channel | pool mode share | full mode share | pool prevalence / default share | status change |
|---|---|---|---|---|
| price_missing | 0.997791 | 0.997493 | 0.22092% | — |
| price_zero | 0.997791 | 0.997493 | 0.22092% | — |
| price_conflict | 1.000000 | 1.000000 | 0.00000% | — |
| brand_recovered | 0.999981 | 0.999976 | 0.00190% | — |
| brand_conflict | 1.000000 | 1.000000 | 0.00000% | — |
| category_missing | 1.000000 | 1.000000 | 0.00000% | — |
| category_recovered | 1.000000 | 1.000000 | 0.00000% | — |
| category_conflict | 1.000000 | 1.000000 | 0.00000% | — |
| category_code_recovered | 1.000000 | 1.000000 | 0.00000% | — |
| item_age_unavailable | 0.996841 | 0.996423 | 0.31592% | — |
| simultaneous_block | 0.999974 | 0.999965 | 0.00263% | — |

## 2. Exact functional dependencies on the pool (0 violations; informative determinants only)

| X → Y | violations pool | violations full | note |
|---|---|---|---|
| utc_weekday_sin → utc_weekday_cos | 0 | 0 |  |
| weekday_tokens ↔ local_weekday_sin | 0 | 0 | one-to-one |
| local_weekday_sin → local_weekday_cos | 0 | 0 |  |
| weekday_tokens → local_weekday_cos | 0 | 0 |  |
| local_weekday_sin → is_weekend | 0 | 0 |  |
| weekday_tokens → is_weekend | 0 | 0 |  |
| log1p_prefix_elapsed_s ↔ log1p_segment_elapsed_s | 0 | 0 | one-to-one |
| prefix_item_count ↔ prefix_representative_count | 0 | 0 | one-to-one |
| log1p_price → price_missing | 0 | 0 |  |
| price_zero ↔ price_missing | 0 | 0 | one-to-one |
| log1p_price → price_zero | 0 | 0 |  |
| brand_tokens → brand_missing | 0 | 0 |  |
| current_known_category_run_length → category_code_missing | 0 | 0 |  |
| main_category_missing ↔ category_code_missing | 0 | 0 | one-to-one |
| category_tokens → category_code_missing | 0 | 0 |  |
| main_category_tokens → category_code_missing | 0 | 0 |  |
| category_code_idx → category_code_missing | 0 | 0 |  |
| current_known_category_run_length → main_category_missing | 0 | 0 |  |
| category_tokens → main_category_missing | 0 | 0 |  |
| main_category_tokens → main_category_missing | 0 | 0 |  |
| category_code_idx → main_category_missing | 0 | 0 |  |
| log1p_prev_raw_view_gap_s → prev_gap_unavailable | 0 | 0 |  |
| log1p_prev_view_count → prev_gap_unavailable | 0 | 0 |  |
| log1p_prev_session_count → prev_gap_unavailable | 0 | 0 |  |
| recent_history_length → prev_gap_unavailable | 0 | 0 |  |
| has_history ↔ prev_gap_unavailable | 0 | 0 | one-to-one |
| recent_distinct_items → prev_gap_unavailable | 0 | 0 |  |
| log1p_time_since_prev_user_event_s → prev_gap_unavailable | 0 | 0 |  |
| has_history_mask ↔ prev_gap_unavailable | 0 | 0 | one-to-one |
| prev_user_event_available ↔ prev_gap_unavailable | 0 | 0 | one-to-one |
| log1p_item_observed_age_s → item_age_unavailable | 0 | 0 |  |
| log1p_prev_view_count → recent_history_length | 0 | 0 |  |
| log1p_prev_raw_view_gap_s → has_history | 0 | 0 |  |
| log1p_prev_view_count → has_history | 0 | 0 |  |
| log1p_prev_session_count → has_history | 0 | 0 |  |
| recent_history_length → has_history | 0 | 0 |  |
| recent_distinct_items → has_history | 0 | 0 |  |
| log1p_time_since_prev_user_event_s → has_history | 0 | 0 |  |
| has_history_mask ↔ has_history | 0 | 0 | one-to-one |
| prev_user_event_available ↔ has_history | 0 | 0 | one-to-one |
| recent_mean_log_price → recent_price_supported | 0 | 0 |  |
| log1p_time_since_prev_session_last_event_s → prev_session_available | 0 | 0 |  |
| log1p_prev_raw_view_gap_s → has_history_mask | 0 | 0 |  |
| log1p_prev_view_count → has_history_mask | 0 | 0 |  |
| log1p_prev_session_count → has_history_mask | 0 | 0 |  |
| recent_history_length → has_history_mask | 0 | 0 |  |
| recent_distinct_items → has_history_mask | 0 | 0 |  |
| log1p_time_since_prev_user_event_s → has_history_mask | 0 | 0 |  |
| prev_user_event_available ↔ has_history_mask | 0 | 0 | one-to-one |
| log1p_prev_raw_view_gap_s → prev_user_event_available | 0 | 0 |  |
| log1p_prev_view_count → prev_user_event_available | 0 | 0 |  |
| log1p_prev_session_count → prev_user_event_available | 0 | 0 |  |
| recent_history_length → prev_user_event_available | 0 | 0 |  |
| recent_distinct_items → prev_user_event_available | 0 | 0 |  |
| log1p_time_since_prev_user_event_s → prev_user_event_available | 0 | 0 |  |
| category_tokens → main_category_tokens | 0 | 0 |  |
| category_code_idx → main_category_tokens | 0 | 0 |  |
| category_tokens → category_code_idx | 0 | 0 |  |

Multi-column determinants:

- (utc_hour_sin, utc_hour_cos) → local_hour_sin: pool 0 (0.0000%), full 0
- (utc_hour_sin, utc_hour_cos) → local_hour_cos: pool 0 (0.0000%), full 0
- (utc_hour_sin, utc_hour_cos) → daypart_tokens: pool 0 (0.0000%), full 0
- (utc_hour_sin, utc_hour_cos, utc_weekday_sin, utc_weekday_cos) → local_weekday_sin: pool 0 (0.0000%), full 0
- (utc_hour_sin, utc_hour_cos, utc_weekday_sin, utc_weekday_cos) → local_weekday_cos: pool 0 (0.0000%), full 0
- (utc_hour_sin, utc_hour_cos, utc_weekday_sin, utc_weekday_cos) → is_weekend: pool 0 (0.0000%), full 0
- (utc_hour_sin, utc_hour_cos, utc_weekday_sin, utc_weekday_cos) → weekday_tokens: pool 0 (0.0000%), full 0
- (utc_hour_sin, utc_hour_cos, utc_weekday_sin, utc_weekday_cos) → daypart_tokens: pool 0 (0.0000%), full 0
- (log1p_price, recent_mean_log_price) → current_minus_recent_mean_log_price: pool 81,914 (0.2862%), full 133,244
- (log1p_price, recent_mean_log_price, price_missing, recent_price_supported) → current_minus_recent_mean_log_price: pool 81,914 (0.2862%), full 133,244
- (item_tokens) → category_tokens: pool 25,454 (0.0889%), full 43,479
- (item_tokens) → brand_tokens: pool 115,027 (0.4019%), full 175,985
- (item_tokens) → main_category_tokens: pool 9,617 (0.0336%), full 16,127
- (category_tokens) → main_category_tokens: pool 0 (0.0000%), full 0

## 3. High-similarity clusters (single linkage; tensor order)

- **POOL pearson_ge_0.95:** {utc_weekday_sin, local_weekday_sin} min|r| 0.9895; {utc_weekday_cos, local_weekday_cos} min|r| 0.9912; {log1p_prev_raw_view_gap_s, log1p_time_since_prev_user_event_s} min|r| 0.9839; {log1p_segment_elapsed_s, log1p_prefix_elapsed_s} min|r| 1.0000; {prefix_representative_count, prefix_item_count} min|r| 1.0000; {recent_history_length, recent_distinct_items} min|r| 0.9556
- **POOL pearson_ge_0.99:** {utc_weekday_cos, local_weekday_cos} min|r| 0.9912; {log1p_segment_elapsed_s, log1p_prefix_elapsed_s} min|r| 1.0000; {prefix_representative_count, prefix_item_count} min|r| 1.0000
- **POOL spearman_ge_0.95:** {utc_weekday_sin, local_weekday_sin} min|r| 0.9891; {utc_weekday_cos, local_weekday_cos} min|r| 0.9907; {log1p_segment_elapsed_s, log1p_prefix_elapsed_s} min|r| 1.0000; {prefix_representative_count, prefix_item_count} min|r| 1.0000; {log1p_prev_view_count, recent_history_length, recent_distinct_items} min|r| 0.9796
- **POOL spearman_ge_0.99:** {utc_weekday_cos, local_weekday_cos} min|r| 0.9907; {log1p_segment_elapsed_s, log1p_prefix_elapsed_s} min|r| 1.0000; {prefix_representative_count, prefix_item_count} min|r| 1.0000; {log1p_prev_view_count, recent_history_length} min|r| 1.0000
- **POOL phi_ge_0.95_binary:** {price_missing, price_zero} min|r| 1.0000; {category_code_missing, main_category_missing} min|r| 1.0000; {prev_gap_unavailable, has_history, recent_price_supported, has_history_mask, prev_user_event_available} min|r| 0.9991
- **POOL pearson_ge_0.95_kept_only:** {utc_weekday_sin, local_weekday_sin} min|r| 0.9895; {utc_weekday_cos, local_weekday_cos} min|r| 0.9912; {log1p_prev_raw_view_gap_s, log1p_time_since_prev_user_event_s} min|r| 0.9839; {recent_history_length, recent_distinct_items} min|r| 0.9556
- **POOL spearman_ge_0.95_kept_only:** {utc_weekday_sin, local_weekday_sin} min|r| 0.9891; {utc_weekday_cos, local_weekday_cos} min|r| 0.9907; {log1p_prev_view_count, recent_history_length, recent_distinct_items} min|r| 0.9796
- **FULL pearson_ge_0.95:** {utc_weekday_sin, local_weekday_sin} min|r| 0.9886; {utc_weekday_cos, local_weekday_cos} min|r| 0.9908; {log1p_prev_raw_view_gap_s, log1p_time_since_prev_user_event_s} min|r| 0.9833; {log1p_segment_elapsed_s, log1p_prefix_elapsed_s} min|r| 1.0000; {prefix_representative_count, prefix_item_count} min|r| 1.0000; {recent_history_length, recent_distinct_items} min|r| 0.9579
- **FULL pearson_ge_0.99:** {utc_weekday_cos, local_weekday_cos} min|r| 0.9908; {log1p_segment_elapsed_s, log1p_prefix_elapsed_s} min|r| 1.0000; {prefix_representative_count, prefix_item_count} min|r| 1.0000
- **FULL spearman_ge_0.95:** {utc_weekday_sin, local_weekday_sin} min|r| 0.9883; {utc_weekday_cos, local_weekday_cos} min|r| 0.9903; {log1p_segment_elapsed_s, log1p_prefix_elapsed_s} min|r| 1.0000; {prefix_representative_count, prefix_item_count} min|r| 1.0000; {log1p_prev_view_count, recent_history_length, recent_distinct_items} min|r| 0.9839
- **FULL spearman_ge_0.99:** {utc_weekday_cos, local_weekday_cos} min|r| 0.9903; {log1p_segment_elapsed_s, log1p_prefix_elapsed_s} min|r| 1.0000; {prefix_representative_count, prefix_item_count} min|r| 1.0000; {log1p_prev_view_count, recent_history_length} min|r| 0.9992
- **FULL phi_ge_0.95_binary:** {price_missing, price_zero} min|r| 1.0000; {category_code_missing, main_category_missing} min|r| 1.0000; {prev_gap_unavailable, has_history, recent_price_supported, has_history_mask, prev_user_event_available} min|r| 0.9993
- **FULL pearson_ge_0.95_kept_only:** {utc_weekday_sin, local_weekday_sin} min|r| 0.9886; {utc_weekday_cos, local_weekday_cos} min|r| 0.9908; {log1p_prev_raw_view_gap_s, log1p_time_since_prev_user_event_s} min|r| 0.9833; {recent_history_length, recent_distinct_items} min|r| 0.9579
- **FULL spearman_ge_0.95_kept_only:** {utc_weekday_sin, local_weekday_sin} min|r| 0.9883; {utc_weekday_cos, local_weekday_cos} min|r| 0.9903; {log1p_prev_view_count, recent_history_length, recent_distinct_items} min|r| 0.9839

## 4. Missingness (mask-based) and availability overlap

| channel | availability indicator | pool unavailable | full unavailable | pool V2 exact-zero share |
|---|---|---|---|---|
| log1p_price | price_available | 0.2209% | 0.2507% | 0.2209% |
| log1p_prev_raw_view_gap_s | prev_gap_available | 3.9983% | 5.3371% | 3.9983% |
| log1p_item_observed_age_s | item_age_available | 0.3159% | 0.3577% | 0.3159% |
| recent_category_match_share | category_match_supported | 37.6365% | 39.2115% | 41.1183% |
| recent_brand_match_share | brand_match_supported | 19.7266% | 21.7163% | 46.6068% |
| recent_mean_log_price | recent_price_supported | 4.0055% | 5.3445% | 4.0055% |
| current_minus_recent_mean_log_price | f1_price_diff_available | 4.2194% | 5.5869% | 4.2194% |
| log1p_time_since_prev_user_event_s | history_available | 3.9983% | 5.3371% | 3.9983% |
| log1p_time_since_prev_session_last_event_s | prev_session_available | 86.5283% | 87.2689% | 86.5283% |
| log1p_prev_view_count | history_available | 3.9983% | 5.3371% | 0.0000% |
| log1p_prev_session_count | history_available | 3.9983% | 5.3371% | 0.0000% |
| recent_history_length | history_available | 3.9983% | 5.3371% | 0.0000% |
| recent_distinct_items | history_available | 3.9983% | 5.3371% | 0.0000% |
| recent_distinct_known_main_categories | history_available | 3.9983% | 5.3371% | 0.0000% |
| item_tokens | item_known | 0.0934% | 0.1124% | n/a |
| category_tokens | category_known | 0.0000% | 0.0000% | n/a |
| brand_tokens | brand_known | 15.7222% | 16.5253% | n/a |
| main_category_tokens | main_category_known | 33.9625% | 34.6708% | n/a |

## 5. Feature-group prevalence (share of pool / full examples)

Value availability = any value channel of the group carries an observed (not unknown) value; DERIVED from the measured indicator co-occurrence counts (inclusion-exclusion). Non-default = any member differs from its default/unknown encoding (tokens: known), MEASURED.

| group | pool value available | pool non-default | full value available | full non-default |
|---|---|---|---|---|
| ITEM_ID | 99.907% | 99.907% | 99.888% | 99.888% |
| ITEM_METADATA_TOKENS | 100.000% | 100.000% | 100.000% | 100.000% |
| CALENDAR | 100.000% | 100.000% | 100.000% | 100.000% |
| PRICE_EVENT | 99.779% | 100.000% | 99.749% | 100.000% |
| VIEW_GAP_EVENT | 96.002% | 100.000% | 94.663% | 100.000% |
| ITEM_AGE_POOLED_FIRST_SEEN | 99.684% | 100.000% | 99.642% | 100.000% |
| SEGMENT_PREFIX | 100.000% | 94.439% | 100.000% | 94.220% |
| METADATA_QUALITY_FLAGS | n/a (indicators only) | 40.174% | n/a | 41.087% |
| SIMULTANEOUS_BLOCK | n/a (indicators only) | 0.003% | n/a | 0.004% |
| USER_HISTORY_COUNTS | 96.002% | 96.002% | 94.663% | 94.663% |
| USER_MATCH | 86.266% | 86.266% | 84.436% | 84.436% |
| USER_PRICE | 95.995% | 95.995% | 94.655% | 94.655% |
| USER_TIME_SINCE | 96.002% | 96.002% | 94.663% | 94.663% |
| USER_CONTEXT_ALL | 96.002% | 96.002% | 94.663% | 94.663% |

## 6. Transfer check (pool vs n ≥ 1)

- Channels whose status changes: none
- Max |Δ Pearson| 0.1450 at ['prefix_representative_count', 'log1p_prev_session_count']; max |Δ Spearman| 0.1628 at ['prefix_representative_count', 'log1p_prev_session_count']; pairs with |Δ| > 0.05: Pearson 53, Spearman 43.
- Ordered pairs whose exact-FD status differs: 0
- Largest mean shifts (pool − full, in full SD): recent_history_length -0.329, recent_distinct_items -0.314, log1p_prev_view_count -0.207, log1p_prev_session_count -0.193, prefix_representative_count -0.140, prefix_item_count -0.140, recent_distinct_known_main_categories -0.125, current_known_category_run_length -0.063

## 7. Normalizer and shared-statistic provenance (full TRAIN vs 5–100 pool refit)

| channel | full clip [lo, hi] / mean / std | pool refit clip [lo, hi] / mean / std | max / mean |Δz| on pool examples |
|---|---|---|---|
| log1p_price | [1.1569, 7.8491] / 5.04455 / 1.20851 | [1.2, 7.8491] / 5.05592 / 1.19882 | 0.03521 / 0.01025 |
| log1p_prev_raw_view_gap_s | [0.69315, 14.891] / 5.17361 / 3.26035 | [0.69315, 14.839] / 5.16437 / 3.23849 | 0.02287 / 0.00384 |
| log1p_segment_elapsed_s | [0, 9.5611] / 4.54889 / 2.83005 | [0, 9.1868] / 4.69533 / 2.65339 | 0.1622 / 0.0545 |
| log1p_item_observed_age_s | [8.0298, 15.191] / 14.2241 / 1.01729 | [8.0787, 15.192] / 14.2304 / 1.01528 | 0.02995 / 0.006177 |
| prefix_representative_count | [0, 140] / 6.71678 / 11.9083 | [0, 61] / 5.56712 / 7.48473 | 2.848 / 0.2376 |
| prefix_known_category_count | [0, 5] / 0.906572 / 0.686465 | [0, 5] / 0.900505 / 0.658595 | 0.2616 / 0.03144 |
| prefix_category_switch_count | [0, 7] / 0.149452 / 0.578631 | [0, 5] / 0.133623 / 0.495333 | 2.015 / 0.04945 |
| current_known_category_run_length | [0, 75] / 3.36355 / 6.37329 | [0, 45] / 3.13956 / 4.95074 | 2.785 / 0.139 |
| log1p_prev_view_count | [0, 7.2335] / 2.71272 / 1.4865 | [0, 4.7449] / 2.59769 / 1.08771 | 0.607 / 0.2325 |
| log1p_prev_session_count | [0, 4.4998] / 1.46028 / 0.878214 | [0, 3.6376] / 1.38178 / 0.693049 | 0.7757 / 0.1728 |
| recent_history_length | [0, 100] / 28.8617 / 32.2929 | [0, 100] / 20.9298 / 20.8166 | 1.596 / 0.2802 |
| recent_distinct_items | [0, 98] / 21.4812 / 23.5508 | [0, 83] / 16.0272 / 15.1773 | 1.801 / 0.2808 |
| recent_distinct_known_main_categories | [0, 8] / 1.6579 / 1.37781 | [0, 7] / 1.5863 / 1.20261 | 0.6244 / 0.09368 |
| recent_mean_log_price | [1.9053, 7.4674] / 5.03827 / 0.882684 | [1.9559, 7.463] / 5.05368 / 0.878935 | 0.03243 / 0.01751 |
| current_minus_recent_mean_log_price | [-4.0907, 3.6814] / 0.00206569 / 0.994534 | [-4.0907, 3.6814] / 0.00159393 / 0.984842 | 0.04002 / 0.006975 |
| log1p_time_since_prev_user_event_s | [0, 14.878] / 5.37889 / 3.17817 | [0, 14.878] / 5.37003 / 3.15756 | 0.02232 / 0.003789 |
| log1p_time_since_prev_session_last_event_s | [0, 14.878] / 11.8169 / 1.65802 | [0, 14.878] / 11.841 / 1.62161 | 0.064 / 0.02013 |

- Catalogue: K_A 158,486, K_B (pool-only support ≥ 5) 142,950, A∖B 15,536; pool examples whose current item leaves B 0.1192% (SESSION_L50 window positions, superseded: 0.1451%); pool targets whose class leaves B 38,799 (0.1356%); support = catalogue.representative_views mismatches 0.
- First-seen proxy validation: availability agreement 99.9999%, exact value agreement 99.9757%.
  - normalizer_only_pool: max |Δz| 0.0299, mean |Δz| 0.0062
  - fully_pool_first_seen_and_normalizer: max |Δz| 7.2543, mean |Δz| 0.0221
  - pool_first_seen_full_normalizer: max |Δz| 7.0282, mean |Δz| 0.0178
  - availability flips (available under full first-seen, unavailable under pool first-seen): 52,473 of 28,623,386

## 8. Proposed compact-candidate ladder (statistics only; nested; identical for both families)

- Rungs nested: True; judging rule: owner delegation (DECISION_LOG 35-36 via Ops): the smallest schema whose MRR@20 drop is <= 1% relative on the WORSE of GRU and SASRec, identical columns for both families; owner amendment section 7.
| rung | channels | tokens / num / flags / uctx / masks | GRU params (Δ) | SASRec params (Δ) | removed here |
|---|---|---|---|---|---|
| L0_CLEAN_V2 | 53 | 6 / 15 / 15 / 12 / 4 | 20,963,806 (+0) | 83,094,112 (+0) | — |
| L1_EXACT | 37 | 4 / 12 / 5 / 11 / 4 | 20,961,630 (-2,176) | 83,090,400 (-3,712) | price_conflict, brand_conflict, category_missing, category_recovered, category_conflict, category_code_recovered, price_zero, brand_missing, main_category_missing, category_code_missing, main_category_tokens, local_weekday_sin, local_weekday_cos, is_weekend, recent_history_length, daypart_tokens |
| L2_RARE | 35 | 4 / 12 / 3 / 11 / 4 | 20,961,566 (-2,240) | 83,090,336 (-3,776) | simultaneous_block, brand_recovered |
| L3_NEAR_DUPLICATE | 30 | 4 / 10 / 3 / 8 / 4 | 20,961,310 (-2,496) | 83,090,080 (-4,032) | utc_weekday_sin, utc_weekday_cos, log1p_time_since_prev_user_event_s, recent_distinct_items, has_history |
| L4_SPARSE_AVAILABILITY | 28 | 4 / 10 / 3 / 7 / 3 | 20,961,182 (-2,624) | 83,089,952 (-4,160) | log1p_time_since_prev_session_last_event_s, prev_session_available |
| L5_NO_USER_CONTEXT_BLOCK | 18 | 4 / 10 / 3 / 0 / 0 | 20,952,286 (-11,520) | 83,072,095 (-22,017) | log1p_prev_view_count, log1p_prev_session_count, recent_distinct_known_main_categories, recent_category_match_share, recent_brand_match_share, recent_mean_log_price, current_minus_recent_mean_log_price, recent_category_match_supported, recent_brand_match_supported, recent_price_supported |
| L6_NO_POOLED_POPULARITY | 16 | 4 / 9 / 2 / 0 / 0 | 20,952,222 (-11,584) | 83,072,031 (-22,081) | log1p_item_observed_age_s, item_age_unavailable |
| L7_NO_DERIVABLE_SEGMENT_SUMMARIES | 13 | 4 / 6 / 2 / 0 / 0 | 20,952,126 (-11,680) | 83,071,935 (-22,177) | prefix_known_category_count, prefix_category_switch_count, current_known_category_run_length |
| L8_ID_PLUS_MINIMAL_CONTEXT | 10 | 2 / 5 / 2 / 0 / 0 | 20,820,854 (-142,952) | 82,936,055 (-158,057) | brand_tokens, weekday_tokens, log1p_segment_elapsed_s |
| L9_ID_ONLY_ANCHOR | 2 | 1 / 0 / 0 / 0 / 0 | 20,791,190 (-172,616) | 82,897,430 (-196,682) | category_tokens, log1p_price, utc_hour_sin, utc_hour_cos, log1p_prev_raw_view_gap_s, prefix_representative_count, price_missing, prev_gap_unavailable |

### USER_HISTORY_L50 column (hypotheses for the experiments, not drop decisions)

| channel | would a cross-session 50-event window carry it directly? | basis |
|---|---|---|
| log1p_prev_view_count | MOSTLY_CARRIED | the window length is min(prior events + 1, 50), so the prior-event count is readable from the window exactly below 49 prior events and saturates above; pool recent_history_length q0.75 = 30, q0.95 = 66 |
| log1p_prev_session_count | PARTIALLY_CARRIED | sessions (model segments) inside the window are visible through prefix_representative_count = 0 at each segment start; sessions older than the 50 events are not (pool q0.95 = 12 sessions) |
| recent_history_length | MOSTLY_CARRIED | min(prior events, 100): identical to the window length below 50; already an exact function of log1p_prev_view_count (removed at L1) |
| has_history | CARRIED_DIRECTLY | window length >= 2 exactly when the user has a prior event; already exactly NOT prev_gap_unavailable on every TRAIN row (identity ledger E9 = 0) and log1p_prev_view_count > 0 (E6 = 0) |
| recent_distinct_items | MOSTLY_CARRIED | distinct products among the <= 100 most recent events: countable from the window's item tokens for the last 49 events (distinct OOV items collapse to token 2); pool q0.95 = 48 distinct items |
| recent_distinct_known_main_categories | MOSTLY_CARRIED | countable from the window's category tokens (category determines main category exactly on TRAIN) |
| recent_category_match_share | MOSTLY_CARRIED | share of recent known main categories equal to the current one: computable from the window's category tokens over the last 49 events (the channel uses up to 100) |
| recent_brand_match_share | MOSTLY_CARRIED | as recent_category_match_share, with the brand tokens |
| recent_mean_log_price | MOSTLY_CARRIED | mean of the valid log prices of the recent events: the window carries log1p_price and price_missing per event (the channel averages up to 100 events) |
| current_minus_recent_mean_log_price | MOSTLY_CARRIED | the current log1p_price minus the window's own mean price (clip and F-1 gate aside) |
| log1p_time_since_prev_user_event_s | CARRIED_DIRECTLY | the current event's log1p_prev_raw_view_gap_s already carries it (pool Pearson 0.9839); the window extent does not change this |
| log1p_time_since_prev_session_last_event_s | CARRIED_DIRECTLY | non-zero only at a session's first event, where it is the time since the previous event: the current event's view gap together with the segment-start marker |
| recent_category_match_supported | MOSTLY_CARRIED | a known current category plus >= 1 known category in the window |
| recent_brand_match_supported | MOSTLY_CARRIED | as recent_category_match_supported, for brand |
| recent_price_supported | MOSTLY_CARRIED | >= 1 valid price among the window's prior events |
| prev_session_available | CARRIED_DIRECTLY | segment start at the current event (prefix_representative_count = 0) with a prior event in the window |
| log1p_segment_elapsed_s | NOT_CARRIED_DIRECTLY | a session-relative clock; recoverable only by summing the window's per-event gaps back to the last segment start; under USER_HISTORY_L50 the window position no longer restarts at the session start, so session-relative channels become the model's only session-boundary signal |
| prefix_representative_count | NOT_CARRIED_GAINS_ROLE | under SESSION_L50 it equalled the window position for segments <= 50 (position_ids = rep + 1); under USER_HISTORY_L50 it is the session-boundary marker (0 at a segment start) inside a cross-session window |
| prefix_known_category_count | PARTIALLY_CARRIED | a function of the category tokens after the last segment start in the window; exact when the segment prefix fits in the window (pool prefix_representative_count q0.99 = 38) |
| prefix_category_switch_count | PARTIALLY_CARRIED | as prefix_known_category_count (transitions of main category) |
| current_known_category_run_length | PARTIALLY_CARRIED | the run of the same known main category ending at the current event, readable from the window when the run is < 50 (pool q0.99 = 25) |
