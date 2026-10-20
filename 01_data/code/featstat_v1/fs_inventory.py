"""featstat_v1.fs_inventory - static model-input inventory of Clean V2 (no dataset rows are read).

Sources: contracts FEATURE_SCHEMA_CLEAN_V2.json (via v4data.common.load_schema, self-hash verified), the model
families' embedding sizes (v4models gru_v2 / sasrec_v2 constants, quoted) and VOCAB_SIZES of snap_v1/reuse/train.py.
Also the per-family parameter formulas used by the compact-candidate ladder (verified against the MODEL-01
inventories: GRU 20,963,806 and SASRec 83,094,112 parameters at the Clean V2 widths).
"""
from __future__ import annotations

from typing import Mapping, Sequence

VOCAB_SIZES = {"item_tokens": 158489, "category_tokens": 663, "brand_tokens": 3956, "main_category_tokens": 16,
               "daypart_tokens": 8, "weekday_tokens": 10}                      # snap_v1/reuse/train.py:79-86
K = 158486
SIDE_TOKENS = ("category_tokens", "brand_tokens", "main_category_tokens", "daypart_tokens", "weekday_tokens")
GRU_EMBED_DIM = {"item_tokens": 128, "category_tokens": 32, "brand_tokens": 32, "main_category_tokens": 8,
                 "daypart_tokens": 4, "weekday_tokens": 4}                    # v4models/gru_v2.py:149-156
SASREC_SIDE_DIMS = {"category_tokens": 32, "brand_tokens": 32, "main_category_tokens": 8, "daypart_tokens": 4,
                    "weekday_tokens": 4}                                       # v4models/sasrec_v2.py:56-59
SASREC_D, SASREC_BLOCKS, SASREC_FF, SASREC_MAXLEN = 256, 2, 4, 50

# semantic feature groups (FEATSTAT-01 definition; used for group prevalence and the ladder)
GROUPS = {
    "ITEM_ID": ["item_tokens"],
    "ITEM_METADATA_TOKENS": ["category_tokens", "brand_tokens", "main_category_tokens"],
    "CALENDAR": ["utc_hour_sin", "utc_hour_cos", "utc_weekday_sin", "utc_weekday_cos", "local_weekday_sin",
                 "local_weekday_cos", "is_weekend", "daypart_tokens", "weekday_tokens"],
    "PRICE_EVENT": ["log1p_price", "price_missing", "price_zero", "price_conflict"],
    "VIEW_GAP_EVENT": ["log1p_prev_raw_view_gap_s", "prev_gap_unavailable"],
    "ITEM_AGE_POOLED_FIRST_SEEN": ["log1p_item_observed_age_s", "item_age_unavailable"],
    "SEGMENT_PREFIX": ["log1p_segment_elapsed_s", "prefix_representative_count", "prefix_known_category_count",
                       "prefix_category_switch_count", "current_known_category_run_length"],
    "METADATA_QUALITY_FLAGS": ["brand_missing", "brand_recovered", "brand_conflict", "category_missing",
                               "category_recovered", "category_conflict", "category_code_missing",
                               "category_code_recovered", "main_category_missing"],
    "SIMULTANEOUS_BLOCK": ["simultaneous_block"],
    "USER_HISTORY_COUNTS": ["log1p_prev_view_count", "log1p_prev_session_count", "recent_history_length",
                            "has_history", "recent_distinct_items", "recent_distinct_known_main_categories"],
    "USER_MATCH": ["recent_category_match_share", "recent_brand_match_share", "recent_category_match_supported",
                   "recent_brand_match_supported"],
    "USER_PRICE": ["recent_mean_log_price", "current_minus_recent_mean_log_price", "recent_price_supported"],
    "USER_TIME_SINCE": ["log1p_time_since_prev_user_event_s", "log1p_time_since_prev_session_last_event_s",
                        "prev_session_available"],
}
GROUPS["USER_CONTEXT_ALL"] = (GROUPS["USER_HISTORY_COUNTS"] + GROUPS["USER_MATCH"] + GROUPS["USER_PRICE"]
                              + GROUPS["USER_TIME_SINCE"])


def group_of(name: str) -> str:
    for g, mem in GROUPS.items():
        if g != "USER_CONTEXT_ALL" and name in mem:
            return g
    return "DROPPED_REDUNDANT"


def build_inventory(schema: Mapping) -> dict:
    ch = schema["channels"]
    lay = schema["tensor_layout_v2"]
    rows = []
    for c in ch:
        blk = c["block"]
        if c["name"] == "output_catalogue":
            role = "REQUIRED (task definition: next-item ranking over the K-class catalogue)"
        elif c["name"] == "item_tokens":
            role = "REQUIRED (task definition: the item-ID sequence)"
        elif c["decision"] == "DROP_REDUNDANT":
            role = "DROPPED by Clean V2 (exact redundancy; reported for completeness)"
        else:
            role = "SIDE_INFORMATION (candidate for removal)"
        emb = None
        if blk == "categorical":
            nm = c["name"]
            emb = {"vocab_size": VOCAB_SIZES[nm], "gru_embed_dim": GRU_EMBED_DIM[nm],
                   "gru_params": VOCAB_SIZES[nm] * GRU_EMBED_DIM[nm],
                   "sasrec_embed_dim": SASREC_D if nm == "item_tokens" else SASREC_SIDE_DIMS[nm],
                   "sasrec_params": VOCAB_SIZES[nm] * (SASREC_D if nm == "item_tokens" else SASREC_SIDE_DIMS[nm])}
        rows.append({"n": c["n"], "name": c["name"], "block": blk, "group": group_of(c["name"]),
                     "dtype": c["dtype"], "source_column": c["stored_column"], "transform_state_v2": c["state_v2"],
                     "decision": c["decision"], "v1_slot": c["v1_slot"], "v2_slot": c["v2_slot"],
                     "locality": c["locality"], "pooled_other_users": c["pooled_other_users"], "role": role,
                     "embedding": emb})
    return {
        "source": "contracts/FEATURE_SCHEMA_CLEAN_V2.json 2.1.0 (self-hash verified) + v4models embed dims + "
                  "snap_v1/reuse/train.py VOCAB_SIZES; no dataset rows",
        "blocks": {
            "item_id_sequence": {"key": "item_tokens", "shape": "[B,50] int64", "required": True},
            "categorical_side_embeddings": {"keys": list(SIDE_TOKENS), "shape": "[B,50] int64 each",
                                            "required": False},
            "event_numeric_features": {"shape": "[B,50,15] float32", "names": lay["event_numeric_features"]["names"],
                                       "required": False},
            "event_quality_flags": {"shape": "[B,50,15] int8", "names": lay["event_quality_flags"]["names"],
                                    "required": False},
            "user_context": {"shape": "[B,12] float32", "names": lay["user_context"]["names"], "required": False},
            "user_context_masks": {"shape": "[B,4] float32", "names": lay["user_context_masks"]["names"],
                                   "required": False},
            "structural_keys": {"keys": ["attention_mask", "position_ids", "lengths", "last_valid_index"],
                                "required": True, "note": "window structure, not information channels"},
            "output_catalogue": {"K": K, "required": True},
            "dropped_exact_redundancy": [c["name"] for c in ch if c["decision"] == "DROP_REDUNDANT"],
        },
        "channels": rows,
        "groups": GROUPS,
    }


# ------------------------------------------------------------------------------ parameter formulas
def _lin(i: int, o: int) -> int:
    return i * o + o


def gru_params(tokens: Sequence[str], nf: int, uc: int, um: int, *, k: int = K) -> dict:
    """ContextGRU (v4models/gru_v2.py) parameter count for a channel set. user context is late-fused only when
    uc + um > 0; numeric_flags_proj exists only when nf > 0."""
    p = {}
    width = 0
    for t in tokens:
        p[t] = VOCAB_SIZES[t] * GRU_EMBED_DIM[t]
        width += GRU_EMBED_DIM[t]
    if nf > 0:
        p["numeric_flags_proj"] = _lin(nf, 32)
        width += 32
    p["input_proj"] = _lin(width, 128)
    p["input_norm"] = 256
    p["gru"] = 3 * (128 * 256 + 256 * 256 + 2 * 256)
    if uc + um > 0:
        p["user_ctx_proj"] = _lin(uc + um, 64)
        p["readout_proj"] = _lin(256 + 64, 128)
    else:
        p["readout_proj"] = _lin(256, 128)
    p["output_bias"] = k
    p["total"] = sum(p.values())
    p["concat_width"] = width
    return p


def sasrec_params(tokens: Sequence[str], nf: int, uc: int, um: int, *, k: int = K) -> dict:
    """SASRecCE A1_RICH256 (v4models/sasrec_v2.py) parameter count; the side branch exists when any side token or
    nf > 0, the user branch when uc + um > 0 (both need a code change when they differ from the Clean V2 set:
    SIDE_CHANNELS is a module constant and `rich` couples all branches)."""
    d = SASREC_D
    p = {"item_embed": VOCAB_SIZES["item_tokens"] * d, "pos_embed": SASREC_MAXLEN * d,
         "blocks": SASREC_BLOCKS * (2 * 2 * d + _lin(d, 3 * d) + _lin(d, d) + _lin(d, SASREC_FF * d)
                                    + _lin(SASREC_FF * d, d)),
         "final_norm": 2 * d, "input_norm": 2 * d, "output_embed": k * d, "output_bias": k}
    side = [t for t in tokens if t != "item_tokens"]
    if side or nf > 0:
        sw = 0
        se = 0
        for t in side:
            se += VOCAB_SIZES[t] * SASREC_SIDE_DIMS[t]
            sw += SASREC_SIDE_DIMS[t]
        if side:
            p["side_embeds"] = se
        if nf > 0:
            p["numflag_proj"] = _lin(nf, 32)
            sw += 32
        p["side_proj"] = _lin(sw, d)
        p["side_norm"] = 2 * d
        p["side_gate"] = 1
    if uc + um > 0:
        p["user_mlp"] = _lin(uc + um, 64)
        p["user_proj"] = _lin(64, d)
        p["user_norm"] = 2 * d
        p["user_gate"] = 1
    p["total"] = sum(p.values())
    return p


def check_formulas() -> dict:
    """The Clean V2 widths must reproduce the MODEL-01 inventories exactly."""
    toks = ["item_tokens"] + list(SIDE_TOKENS)
    g = gru_params(toks, 30, 12, 4)["total"]
    s = sasrec_params(toks, 30, 12, 4)["total"]
    return {"gru_total": g, "gru_expected": 20963806, "sasrec_total": s, "sasrec_expected": 83094112,
            "ok": g == 20963806 and s == 83094112}
