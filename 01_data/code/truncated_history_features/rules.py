"""v4trunc.rules — the E11.4 recompute table of every stored v1 column the Clean V2 batch is built from (pure data).

EVALUATION_CONTRACT_V4 §11.4 (Clean V2 table, schema 2.1.0) + LOCALITY_MAP. The builder never decides a rule at run
time: its hard-coded column sets are asserted EQUAL to the rule classes of this table at import
(v4trunc.builder.check_rule_coverage, REV-ANA-01 B-F4). A stored v1 column that is absent from this table, or carries a
rule the builder cannot serve (e.g. UNSUPPORTED_FUTURE), refuses the import (fail-closed).

Rules
  KEEP_EVENT          LOCAL_EVENT: taken unchanged from the permitted event row.
  SHARED_FROZEN       SHARED_CATALOGUE / SHARED_TRAIN_STATISTIC: the frozen end-of-TRAIN value, identical across h.
  RECOMPUTE_SEGMENT   LOCAL_HISTORY per-event, session(segment)-prefix: recomputed over the PERMITTED events of the
                      model segment only (v1 pipeline6.event_feature_sql, restricted to H_h(q) + e_q).
  PARTIAL_ORIGIN      LOCAL_HISTORY per-event, recomputed with the segment origin = the earliest PERMITTED event of the
                      segment (contract §11.4 PARTIAL: log1p_segment_elapsed_s; the dropped duplicate
                      log1p_prefix_elapsed_s follows it, RC1). Disclosed as a truncated-origin value.
  PREV_GAP_RULE       the contract's LOCAL_EVENT exception: the earliest permitted event's previous raw view lies outside
                      the budget, so prev_gap_unavailable = 1 there and log1p_prev_raw_view_gap_s = 0 (re-zeroed);
                      every other permitted event keeps its stored (partition-scoped) value.
  RECOMPUTE_USER      LOCAL_HISTORY decision-level user context: the v1 recent-100 replay (userctx.user_context_features)
                      run over the permitted events that the v1 replay stream of the query's partition contains
                      (TRAIN for TRAIN, TRAIN + VALIDATION_SELECT for VS, TRAIN + VALIDATION_CONFIRM for VC:
                      pipeline6.STREAMS), then e_q. Window sums use the v1 bucket-stream carry-in (see recompute).
  PARTIAL_OOV_IDENTITY  RECOMPUTE_USER, but the frozen v1 files carry only the TRAIN-fitted token of a brand / main
                      category, not its string: two different out-of-vocabulary values share token OOV (2), so a match
                      share can differ from v1 exactly when the query's value AND a window value are OOV. MEASURED
                      rare (brand: 8 of 1,038,917 bucket-0 rows); the affected rows are flagged per query.
  DECISION_META       not a feature: decision id / client idx / targets / masks (DecisionSet metadata, unchanged).
"""
from __future__ import annotations

LOCAL_EVENT, LOCAL_HISTORY = "LOCAL_EVENT", "LOCAL_HISTORY"
SHARED_CATALOGUE, SHARED_TRAIN = "SHARED_CATALOGUE", "SHARED_TRAIN_STATISTIC"

KEEP_EVENT = "KEEP_EVENT"
SHARED_FROZEN = "SHARED_FROZEN"
RECOMPUTE_SEGMENT = "RECOMPUTE_SEGMENT"
PARTIAL_ORIGIN = "PARTIAL_ORIGIN"
PREV_GAP_RULE = "PREV_GAP_RULE"
RECOMPUTE_USER = "RECOMPUTE_USER"
PARTIAL_OOV_IDENTITY = "PARTIAL_OOV_IDENTITY"

# stored column -> (locality, rule)
COLUMN_RULES = {
    # tokens
    "product_idx": (SHARED_TRAIN, SHARED_FROZEN), "category_idx": (SHARED_CATALOGUE, SHARED_FROZEN),
    "brand_idx": (SHARED_CATALOGUE, SHARED_FROZEN), "main_category_idx": (SHARED_CATALOGUE, SHARED_FROZEN),
    "daypart_idx": (LOCAL_EVENT, KEEP_EVENT), "local_weekday_idx": (LOCAL_EVENT, KEEP_EVENT),
    # event numeric (EVENT_NUMERIC_FEATURES)
    "log1p_price": (LOCAL_EVENT, KEEP_EVENT), "utc_hour_sin": (LOCAL_EVENT, KEEP_EVENT),
    "utc_hour_cos": (LOCAL_EVENT, KEEP_EVENT), "utc_weekday_sin": (LOCAL_EVENT, KEEP_EVENT),
    "utc_weekday_cos": (LOCAL_EVENT, KEEP_EVENT), "local_hour_sin": (LOCAL_EVENT, KEEP_EVENT),
    "local_hour_cos": (LOCAL_EVENT, KEEP_EVENT), "local_weekday_sin": (LOCAL_EVENT, KEEP_EVENT),
    "local_weekday_cos": (LOCAL_EVENT, KEEP_EVENT), "is_weekend": (LOCAL_EVENT, KEEP_EVENT),
    "log1p_prev_raw_view_gap_s": (LOCAL_HISTORY, PREV_GAP_RULE),
    "log1p_segment_elapsed_s": (LOCAL_HISTORY, PARTIAL_ORIGIN),
    "log1p_item_observed_age_s": (SHARED_TRAIN, SHARED_FROZEN),
    "prefix_representative_count": (LOCAL_HISTORY, RECOMPUTE_SEGMENT),
    # session prefix (SESSION_PREFIX_FEATURES)
    "prefix_item_count": (LOCAL_HISTORY, RECOMPUTE_SEGMENT),
    "prefix_known_category_count": (LOCAL_HISTORY, RECOMPUTE_SEGMENT),
    "prefix_category_switch_count": (LOCAL_HISTORY, RECOMPUTE_SEGMENT),
    "current_known_category_run_length": (LOCAL_HISTORY, RECOMPUTE_SEGMENT),
    "log1p_prefix_elapsed_s": (LOCAL_HISTORY, PARTIAL_ORIGIN),
    # flags
    "price_missing": (LOCAL_EVENT, KEEP_EVENT), "price_zero": (LOCAL_EVENT, KEEP_EVENT),
    "price_conflict": (LOCAL_EVENT, KEEP_EVENT), "brand_missing": (LOCAL_EVENT, KEEP_EVENT),
    "brand_recovered": (LOCAL_EVENT, KEEP_EVENT), "brand_conflict": (LOCAL_EVENT, KEEP_EVENT),
    "category_missing": (LOCAL_EVENT, KEEP_EVENT), "category_recovered": (LOCAL_EVENT, KEEP_EVENT),
    "category_conflict": (LOCAL_EVENT, KEEP_EVENT), "category_code_missing": (LOCAL_EVENT, KEEP_EVENT),
    "category_code_recovered": (LOCAL_EVENT, KEEP_EVENT), "main_category_missing": (LOCAL_EVENT, KEEP_EVENT),
    "prev_gap_unavailable": (LOCAL_EVENT, PREV_GAP_RULE), "item_age_unavailable": (SHARED_TRAIN, SHARED_FROZEN),
    "simultaneous_block": (LOCAL_EVENT, KEEP_EVENT),
    # user context (decision level)
    "log1p_prev_view_count": (LOCAL_HISTORY, RECOMPUTE_USER), "log1p_prev_session_count": (LOCAL_HISTORY, RECOMPUTE_USER),
    "recent_history_length": (LOCAL_HISTORY, RECOMPUTE_USER), "has_history": (LOCAL_HISTORY, RECOMPUTE_USER),
    "recent_distinct_items": (LOCAL_HISTORY, RECOMPUTE_USER),
    "recent_distinct_known_main_categories": (LOCAL_HISTORY, PARTIAL_OOV_IDENTITY),
    "recent_category_match_share": (LOCAL_HISTORY, PARTIAL_OOV_IDENTITY),
    "recent_brand_match_share": (LOCAL_HISTORY, PARTIAL_OOV_IDENTITY),
    "recent_mean_log_price": (LOCAL_HISTORY, RECOMPUTE_USER),
    "current_minus_recent_mean_log_price": (LOCAL_HISTORY, RECOMPUTE_USER),
    "log1p_time_since_prev_user_event_s": (LOCAL_HISTORY, RECOMPUTE_USER),
    "log1p_time_since_prev_session_last_event_s": (LOCAL_HISTORY, RECOMPUTE_USER),
    "has_history_mask": (LOCAL_HISTORY, RECOMPUTE_USER),
    "recent_category_match_supported": (LOCAL_HISTORY, RECOMPUTE_USER),
    "recent_brand_match_supported": (LOCAL_HISTORY, RECOMPUTE_USER),
    "recent_price_supported": (LOCAL_HISTORY, RECOMPUTE_USER),
    "prev_user_event_available": (LOCAL_HISTORY, RECOMPUTE_USER),
    "prev_session_available": (LOCAL_HISTORY, RECOMPUTE_USER),
}

# the channels with a stated non-exact rule (reported in every receipt / launch record)
PARTIAL_CHANNELS = {
    "log1p_segment_elapsed_s": "contract §11.4 PARTIAL (LOCALITY_MAP L60): origin = earliest permitted event of the "
                               "segment; a truncated-origin value whenever the segment starts before H_h(q)",
    "log1p_prefix_elapsed_s": "dropped v1 duplicate of log1p_segment_elapsed_s (RC1); same truncated origin",
    "log1p_prev_raw_view_gap_s": "contract §11.4 LOCAL_EVENT exception: earliest permitted event -> unavailable, 0",
    "prev_gap_unavailable": "contract §11.4 LOCAL_EVENT exception: 1 at the earliest permitted event",
    "recent_brand_match_share": "OOV identity (DATA-07 finding): distinct out-of-vocabulary brands share token 2 in "
                                "the frozen v1 files; exact unless the query brand and a window brand are both OOV",
    "recent_category_match_share": "OOV identity: as brand, for main categories (0 OOV rows MEASURED in bucket 0)",
    "recent_distinct_known_main_categories": "OOV identity: distinct OOV main categories count once",
}

# DATA-05 USER_HISTORY_L50 keys: in-window functions of the served window -> recomputed by construction
SESSION_KEYS_RULE = "IN_WINDOW (v4hist.windows.add_session_features over the truncated window)"

# v1 replay streams (EDA pipeline6.STREAMS): the partitions each decision partition's user context replays
STREAMS = {"TRAIN": ("TRAIN",), "VALIDATION_SELECT": ("TRAIN", "VALIDATION_SELECT"),
           "VALIDATION_CONFIRM": ("TRAIN", "VALIDATION_CONFIRM")}
USER_CONTEXT_WINDOW = 100          # protocol.yaml user_context_window_previous_representatives (V1/protocol.resolved:160)
BUDGETS = (0, 1, 5, 10, 20, 50)    # E11.1
FIXED_BUDGETS = {"FIXED_50": (0, 1, 5, 10, 20, 50), "FIXED_20": (0, 1, 5, 10, 20)}
SUBPANELS = ("FIXED_50", "FIXED_20", "ALL_AVAILABLE")


def rule_of(column: str) -> tuple:
    try:
        return COLUMN_RULES[column]
    except KeyError:
        raise KeyError(f"stored column {column!r} has no E11.4 rule (v4trunc.rules): refusing to serve it") from None


def rules_record() -> dict:
    """The table as recorded in receipts / launch files."""
    by_rule: dict = {}
    for c, (loc, r) in COLUMN_RULES.items():
        by_rule.setdefault(r, []).append(c)
    return {"columns": {c: {"locality": loc, "rule": r} for c, (loc, r) in COLUMN_RULES.items()},
            "by_rule": {k: sorted(v) for k, v in by_rule.items()}, "partial_channels": dict(PARTIAL_CHANNELS),
            "session_keys": SESSION_KEYS_RULE, "streams": {k: list(v) for k, v in STREAMS.items()},
            "user_context_window": USER_CONTEXT_WINDOW}
