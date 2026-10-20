"""featstat_v1.fs_ladder - the PROPOSED compact-candidate ladder (statistics only; not a decision) from the
FEATSTAT-01 JSON outputs. Every removal carries an evidence rule that is re-checked against the measured pool AND
comparison statistics; a removal whose rule fails on the pool is NOT applied (the channel stays) and is reported.
Identical channel lists for both families (no family favouritism); widths and parameter counts per family.

Usage: python -B fs_ladder.py <featstat_v5_100 dir>   -> writes FEATSTAT_LADDER.json in that dir.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from featstat_v1 import fs_inventory as INV  # noqa: E402

V2_NUM = ["log1p_price", "utc_hour_sin", "utc_hour_cos", "utc_weekday_sin", "utc_weekday_cos", "local_weekday_sin",
          "local_weekday_cos", "is_weekend", "log1p_prev_raw_view_gap_s", "log1p_segment_elapsed_s",
          "log1p_item_observed_age_s", "prefix_representative_count", "prefix_known_category_count",
          "prefix_category_switch_count", "current_known_category_run_length"]
V2_FLAG = ["price_missing", "price_zero", "price_conflict", "brand_missing", "brand_recovered", "brand_conflict",
           "category_missing", "category_recovered", "category_conflict", "category_code_missing",
           "category_code_recovered", "main_category_missing", "prev_gap_unavailable", "item_age_unavailable",
           "simultaneous_block"]
V2_UCTX = ["log1p_prev_view_count", "log1p_prev_session_count", "recent_history_length", "has_history",
           "recent_distinct_items", "recent_distinct_known_main_categories", "recent_category_match_share",
           "recent_brand_match_share", "recent_mean_log_price", "current_minus_recent_mean_log_price",
           "log1p_time_since_prev_user_event_s", "log1p_time_since_prev_session_last_event_s"]
V2_MASK = ["recent_category_match_supported", "recent_brand_match_supported", "recent_price_supported",
           "prev_session_available"]
TOKENS = ["item_tokens", "category_tokens", "brand_tokens", "main_category_tokens", "daypart_tokens", "weekday_tokens"]
ALL = TOKENS + V2_NUM + V2_FLAG + V2_UCTX + V2_MASK
DEAD = ["price_conflict", "brand_conflict", "category_missing", "category_recovered", "category_conflict",
        "category_code_recovered"]
# governing flag/mask of each gated channel (a governing indicator is kept while its channel is kept)
GOVERNS = {"price_missing": ["log1p_price"], "prev_gap_unavailable": ["log1p_prev_raw_view_gap_s"],
           "item_age_unavailable": ["log1p_item_observed_age_s"],
           "recent_price_supported": ["recent_mean_log_price", "current_minus_recent_mean_log_price"],
           "has_history": ["log1p_time_since_prev_user_event_s"],
           "prev_session_available": ["log1p_time_since_prev_session_last_event_s"],
           "recent_category_match_supported": ["recent_category_match_share"],
           "recent_brand_match_supported": ["recent_brand_match_share"]}


# USER_HISTORY_L50 (owner-approved window: the last 50 events of the user's whole prior stream, across sessions and
# partition fences, strictly causal). Per channel: would the window carry the same information directly?
# HYPOTHESES for the experiments, not drop decisions; window statistics are DATA-05's.
L50_HYPOTHESES = {
    "log1p_prev_view_count": {
        "hypothesis": "MOSTLY_CARRIED",
        "basis": "the window length is min(prior events + 1, 50), so the prior-event count is readable from the window "
                 "exactly below 49 prior events and saturates above; pool recent_history_length q0.75 = 30, q0.95 = 66"},
    "log1p_prev_session_count": {
        "hypothesis": "PARTIALLY_CARRIED",
        "basis": "sessions (model segments) inside the window are visible through prefix_representative_count = 0 at "
                 "each segment start; sessions older than the 50 events are not (pool q0.95 = 12 sessions)"},
    "recent_history_length": {
        "hypothesis": "MOSTLY_CARRIED",
        "basis": "min(prior events, 100): identical to the window length below 50; already an exact function of "
                 "log1p_prev_view_count (removed at L1)"},
    "has_history": {
        "hypothesis": "CARRIED_DIRECTLY",
        "basis": "window length >= 2 exactly when the user has a prior event; already exactly NOT prev_gap_unavailable "
                 "on every TRAIN row (identity ledger E9 = 0) and log1p_prev_view_count > 0 (E6 = 0)"},
    "recent_distinct_items": {
        "hypothesis": "MOSTLY_CARRIED",
        "basis": "distinct products among the <= 100 most recent events: countable from the window's item tokens for "
                 "the last 49 events (distinct OOV items collapse to token 2); pool q0.95 = 48 distinct items"},
    "recent_distinct_known_main_categories": {
        "hypothesis": "MOSTLY_CARRIED",
        "basis": "countable from the window's category tokens (category determines main category exactly on TRAIN)"},
    "recent_category_match_share": {
        "hypothesis": "MOSTLY_CARRIED",
        "basis": "share of recent known main categories equal to the current one: computable from the window's "
                 "category tokens over the last 49 events (the channel uses up to 100)"},
    "recent_brand_match_share": {
        "hypothesis": "MOSTLY_CARRIED", "basis": "as recent_category_match_share, with the brand tokens"},
    "recent_mean_log_price": {
        "hypothesis": "MOSTLY_CARRIED",
        "basis": "mean of the valid log prices of the recent events: the window carries log1p_price and price_missing "
                 "per event (the channel averages up to 100 events)"},
    "current_minus_recent_mean_log_price": {
        "hypothesis": "MOSTLY_CARRIED",
        "basis": "the current log1p_price minus the window's own mean price (clip and F-1 gate aside)"},
    "log1p_time_since_prev_user_event_s": {
        "hypothesis": "CARRIED_DIRECTLY",
        "basis": "the current event's log1p_prev_raw_view_gap_s already carries it (pool Pearson 0.9839); the window "
                 "extent does not change this"},
    "log1p_time_since_prev_session_last_event_s": {
        "hypothesis": "CARRIED_DIRECTLY",
        "basis": "non-zero only at a session's first event, where it is the time since the previous event: the "
                 "current event's view gap together with the segment-start marker"},
    "recent_category_match_supported": {
        "hypothesis": "MOSTLY_CARRIED", "basis": "a known current category plus >= 1 known category in the window"},
    "recent_brand_match_supported": {
        "hypothesis": "MOSTLY_CARRIED", "basis": "as recent_category_match_supported, for brand"},
    "recent_price_supported": {
        "hypothesis": "MOSTLY_CARRIED", "basis": ">= 1 valid price among the window's prior events"},
    "prev_session_available": {
        "hypothesis": "CARRIED_DIRECTLY",
        "basis": "segment start at the current event (prefix_representative_count = 0) with a prior event in the "
                 "window"},
    "log1p_segment_elapsed_s": {
        "hypothesis": "NOT_CARRIED_DIRECTLY",
        "basis": "a session-relative clock; recoverable only by summing the window's per-event gaps back to the last "
                 "segment start; under USER_HISTORY_L50 the window position no longer restarts at the session start, "
                 "so session-relative channels become the model's only session-boundary signal"},
    "prefix_representative_count": {
        "hypothesis": "NOT_CARRIED_GAINS_ROLE",
        "basis": "under SESSION_L50 it equalled the window position for segments <= 50 (position_ids = rep + 1); "
                 "under USER_HISTORY_L50 it is the session-boundary marker (0 at a segment start) inside a "
                 "cross-session window"},
    "prefix_known_category_count": {
        "hypothesis": "PARTIALLY_CARRIED",
        "basis": "a function of the category tokens after the last segment start in the window; exact when the "
                 "segment prefix fits in the window (pool prefix_representative_count q0.99 = 38)"},
    "prefix_category_switch_count": {
        "hypothesis": "PARTIALLY_CARRIED", "basis": "as prefix_known_category_count (transitions of main category)"},
    "current_known_category_run_length": {
        "hypothesis": "PARTIALLY_CARRIED",
        "basis": "the run of the same known main category ending at the current event, readable from the window "
                 "when the run is < 50 (pool q0.99 = 25)"},
}


class Ev:
    def __init__(self, d: Path):
        self.s = json.loads((d / "FEATSTAT_STATS.json").read_text(encoding="utf-8"))
        self.f = json.loads((d / "FEATSTAT_FD.json").read_text(encoding="utf-8"))
        self.ix = {n: i for i, n in enumerate(self.f["names"])}
        self.cc = {n: i for i, n in enumerate(self.s["correlation"]["pearson"]["POOL"]["names"])}

    def viol(self, x, y, pop):
        return self.f["violations"][pop][self.ix[x]][self.ix[y]]

    def r(self, kind, x, y, pop):
        m = self.s["correlation"][kind][pop]["matrix"]
        return m[self.cc[x]][self.cc[y]]

    def st(self, n, pop):
        return self.s["channel_stats"][n][pop]

    def win_prev(self, n, pop):
        return self.s["window_exposure_grain"][pop].get(n, {}).get("prevalence")


def rule_constant(ev, n):
    ok = {p: ev.st(n, p)["distinct"] == 1 for p in ("POOL", "FULL")}
    return ok["POOL"], {"rule": "constant on the pool (distinct = 1)", "pool_distinct": ev.st(n, "POOL")["distinct"],
                        "full_distinct": ev.st(n, "FULL")["distinct"], "transfers": ok["POOL"] == ok["FULL"]}


def rule_fd(ev, by, n):
    v = {p: ev.viol(by, n, p) for p in ("POOL", "FULL")}
    return v["POOL"] == 0, {"rule": f"exactly determined by kept channel {by} on the pool (0 violations)",
                            "violations_pool": v["POOL"], "violations_full": v["FULL"],
                            "transfers": (v["POOL"] == 0) == (v["FULL"] == 0)}


def rule_rare(ev, n, thr=1e-3):
    pp, pf = ev.st(n, "POOL").get("prevalence"), ev.st(n, "FULL").get("prevalence")
    ok = pp is not None and pp < thr
    return ok, {"rule": f"non-governing flag with prevalence < {thr:g} on the pool's supervised examples",
                "prevalence_pool": pp, "prevalence_full": pf, "transfers": (pf is not None and pf < thr) == ok}


def rule_multi(ev, det, n):
    rows = {r["population"]: r["violations"] for r in ev.f["multi_column"]
            if r["determinant"] == list(det) and r["dependent"] == n}
    return rows.get("POOL") == 0, {"rule": f"exactly determined by kept channels ({', '.join(det)}) on the pool",
                                   "violations_pool": rows.get("POOL"), "violations_full": rows.get("FULL"),
                                   "transfers": (rows.get("POOL") == 0) == (rows.get("FULL") == 0)}


def rule_corr(ev, keeper, n, thr=0.95):
    vals = {(k, p): abs(ev.r(k, keeper, n, p)) for k in ("pearson", "spearman") for p in ("POOL", "FULL")}
    ok = vals[("pearson", "POOL")] >= thr or vals[("spearman", "POOL")] >= thr
    okf = vals[("pearson", "FULL")] >= thr or vals[("spearman", "FULL")] >= thr
    return ok, {"rule": f"|rho| >= {thr} (Pearson or Spearman) with kept channel {keeper} on the pool",
                "pearson_pool": vals[("pearson", "POOL")], "spearman_pool": vals[("spearman", "POOL")],
                "pearson_full": vals[("pearson", "FULL")], "spearman_full": vals[("spearman", "FULL")],
                "transfers": ok == okf}


def rule_avail(ev, n, indicator, thr):
    ind = ev.s["missingness"]["availability_overlap"]
    ap = ind["POOL"]["indicator_available_share"][indicator]
    af = ind["FULL"]["indicator_available_share"][indicator]
    return ap < thr, {"rule": f"available on < {thr:g} of pool examples ({indicator})", "available_pool": ap,
                      "available_full": af, "transfers": (ap < thr) == (af < thr)}


def widths(ch):
    return {"tokens": [t for t in TOKENS if t in ch], "numeric": [c for c in V2_NUM if c in ch],
            "flags": [c for c in V2_FLAG if c in ch], "user_context": [c for c in V2_UCTX if c in ch],
            "user_masks": [c for c in V2_MASK if c in ch]}


def build(d: Path) -> dict:
    ev = Ev(d)
    rungs = []
    cur = list(ALL)
    base_w = widths(cur)
    base_g = INV.gru_params(base_w["tokens"], 30, 12, 4)["total"]
    base_s = INV.sasrec_params(base_w["tokens"], 30, 12, 4)["total"]

    def add(rid, title, rationale, steps, extra_note=None):
        nonlocal cur
        applied, rejected = [], []
        for n, fn in steps:
            if n not in cur:
                continue
            ok, evid = fn(ev, n)
            (applied if ok else rejected).append({"channel": n, "evidence": evid})
            if ok:
                cur = [c for c in cur if c != n]
        w = widths(cur)
        nf = len(w["numeric"]) + len(w["flags"])
        uc, um = len(w["user_context"]), len(w["user_masks"])
        g = INV.gru_params(w["tokens"], nf, uc, um)
        s = INV.sasrec_params(w["tokens"], nf, uc, um)
        code = []
        if set(w["tokens"]) != set(TOKENS):
            code.append("SASRec SIDE_CHANNELS is a module constant (v4models/sasrec_v2.py:56): a configurable side-token "
                        "list is needed; GRU takes the categorical list from its spec")
        if uc + um == 0:
            code.append("both families: user-context branch absent (GRU spec user_context=False; SASRec `rich` "
                        "couples the side and user branches: code change)")
        if nf == 0 and len(w["tokens"]) == 1:
            code.append("item-only topology (historical GRU F0 / SASRec A0)")
        rungs.append({
            "id": rid, "title": title, "rationale": rationale, "removed_here": applied, "removal_rejected": rejected,
            "channels": w, "n_information_channels": sum(len(v) for v in w.values()) + 1,
            "n_information_channels_note": "tokens + numeric + flags + user context + masks + the output catalogue",
            "input_widths": {"tokens": len(w["tokens"]), "numeric_dim": len(w["numeric"]), "flags_dim": len(w["flags"]),
                             "user_context_dim": uc, "user_mask_dim": um},
            "gru": {"params_total": g["total"], "delta_vs_clean_v2": g["total"] - base_g, "concat_width": g["concat_width"],
                    "numeric_flags_proj_in": nf, "user_ctx_proj_in": uc + um},
            "sasrec": {"params_total": s["total"], "delta_vs_clean_v2": s["total"] - base_s, "numflag_proj_in": nf,
                       "user_mlp_in": uc + um},
            "code_changes_needed": code, "note": extra_note,
            "user_history_l50_column": {c: L50_HYPOTHESES[c]["hypothesis"]
                                        for c in [x["channel"] for x in applied] + sum(w.values(), [])
                                        if c in L50_HYPOTHESES}})

    add("L0_CLEAN_V2", "Current Clean V2 (53 channels)", "reference: the locked Clean V2 schema", [])
    add("L1_EXACT", "Drop constant and exactly-determined channels",
        "information-preserving on the pool: every removed channel is constant or an exact function of channels that "
        "stay (0 FD violations on the 28.6M pool examples, and 0 on the n>=1 population); governing flags/masks are "
        "kept while their channel is kept",
        [(n, rule_constant) for n in DEAD]
        + [("price_zero", lambda e, n: rule_fd(e, "price_missing", n)),
           ("brand_missing", lambda e, n: rule_fd(e, "brand_tokens", n)),
           ("main_category_missing", lambda e, n: rule_fd(e, "category_tokens", n)),
           ("category_code_missing", lambda e, n: rule_fd(e, "category_tokens", n)),
           ("main_category_tokens", lambda e, n: rule_fd(e, "category_tokens", n)),
           ("local_weekday_sin", lambda e, n: rule_fd(e, "weekday_tokens", n)),
           ("local_weekday_cos", lambda e, n: rule_fd(e, "weekday_tokens", n)),
           ("is_weekend", lambda e, n: rule_fd(e, "weekday_tokens", n)),
           ("recent_history_length", lambda e, n: rule_fd(e, "log1p_prev_view_count", n)),
           ("daypart_tokens", lambda e, n: rule_multi(e, ["utc_hour_sin", "utc_hour_cos"], n))],
        "exact on the pool; several identities cross paths (token -> flag, token -> token, numeric -> token) or are "
        "non-affine (recent_history_length = min(expm1(x),100)): exact information, not an affine capacity identity")
    add("L2_RARE", "Drop near-constant non-governing flags",
        "pool prevalence below 1e-3 on the supervised examples; neither flag gates a numeric channel",
        [("simultaneous_block", rule_rare), ("brand_recovered", rule_rare)])
    add("L3_NEAR_DUPLICATE", "Drop near-duplicates (|rho| >= 0.95 on the pool)",
        "one member per high-similarity cluster; the event-level member is kept over the decision-level one; "
        "has_history goes with the last channel it gates (it is exactly log1p_prev_view_count > 0 and exactly "
        "NOT prev_gap_unavailable on every TRAIN row)",
        [("utc_weekday_sin", lambda e, n: rule_corr(e, "local_weekday_sin", n)),
         ("utc_weekday_cos", lambda e, n: rule_corr(e, "local_weekday_cos", n)),
         ("log1p_time_since_prev_user_event_s", lambda e, n: rule_corr(e, "log1p_prev_raw_view_gap_s", n)),
         ("recent_distinct_items", lambda e, n: rule_corr(e, "log1p_prev_view_count", n)),
         ("has_history", lambda e, n: rule_fd(e, "log1p_prev_view_count", n))],
        "utc_weekday_* is compared with local_weekday_* (dropped at L1 as an exact function of weekday_tokens, which "
        "stays); with the constant +3h clock, utc weekday is also a function of (weekday_tokens, utc hour): INFERENCE "
        "from the contract section 4.3 identity, not separately measured")
    add("L4_SPARSE_AVAILABILITY", "Drop value channels available on < 50% of pool examples (with their mask)",
        "availability: the channel carries a value on a minority of pool examples; its mask goes with it",
        [("log1p_time_since_prev_session_last_event_s",
          lambda e, n: rule_avail(e, n, "prev_session_available", 0.5)),
         ("prev_session_available", lambda e, n: (True, {"rule": "mask of the removed channel"}))])
    add("L5_NO_USER_CONTEXT_BLOCK", "Drop the remaining decision-level user-context block",
        "(i) USER_HISTORY_L50 hypothesis: the cross-session 50-event window carries most of this block directly "
        "(see user_history_l50_hypotheses; on the pool the prior-event count is < 50 for between 75% and 95% of the "
        "examples: recent_history_length q0.75 = 30, q0.95 = 66); (ii) its count normalizers are the most "
        "population-sensitive statistics (pool-refit mean |dz| 0.17-0.28); removes the late-fusion user branch in "
        "both families",
        [(n, lambda e, n: (True, {"rule": "user-context block"})) for n in V2_UCTX + V2_MASK],
        "a hypothesis for the experiments, not a drop decision")
    add("L6_NO_POOLED_POPULARITY", "Drop the pooled first-seen (popularity/recency) channel pair",
        "deployability/provenance: log1p_item_observed_age_s and item_age_unavailable are anchored to the pooled "
        "first sighting of the item by any TRAIN user (SHARED_TRAIN_STATISTIC); a pool-only recomputation changes "
        "34% of the pool values (FEATSTAT_PROVENANCE first_seen_item_age)",
        [("log1p_item_observed_age_s", lambda e, n: (True, {"rule": "provenance (pooled first-seen)"})),
         ("item_age_unavailable", lambda e, n: (True, {"rule": "governing flag of the removed age channel"}))],
        "PRIOR (full population, SASRec incumbent reliance, not retraining): item age was the most relied-upon side "
        "feature (P6 -0.024654, about 13% relative of the 0.187 cohort metric); the highest-risk single removal, "
        "placed after the lower-risk rungs")
    add("L7_NO_DERIVABLE_SEGMENT_SUMMARIES", "Drop segment-prefix summaries derivable from the window",
        "USER_HISTORY_L50 hypothesis: prefix_known_category_count, prefix_category_switch_count and "
        "current_known_category_run_length are functions of the window's category tokens after the last segment "
        "start (marked by prefix_representative_count = 0), exact when the segment prefix fits in the window "
        "(pool prefix_representative_count q0.99 = 38 < 50)",
        [(n, lambda e, n: (True, {"rule": "derivable segment summary (hypothesis)"}))
         for n in ("prefix_known_category_count", "prefix_category_switch_count",
                   "current_known_category_run_length")])
    keep8 = {"item_tokens", "category_tokens", "log1p_price", "price_missing", "log1p_prev_raw_view_gap_s",
             "prev_gap_unavailable", "utc_hour_sin", "utc_hour_cos", "prefix_representative_count"}
    add("L8_ID_PLUS_MINIMAL_CONTEXT", "Item ID + minimal local event context",
        "one representative per LOCAL_EVENT/LOCAL_HISTORY event group with >= 90% pool availability (price, view gap, "
        "time of day, position in segment = the session-boundary marker inside a cross-session window) plus the "
        "catalogue category token (100% known on the pool; it exactly determines main_category and category_code)",
        [(n, lambda e, n: (True, {"rule": "not in the minimal set"})) for n in list(cur_all()) if n not in keep8])
    add("L9_ID_ONLY_ANCHOR", "Item ID only (reference anchor, not a proposal)",
        "the historical GRU F0 / SASRec A0 topology; bounds the ladder from below",
        [(n, lambda e, n: (True, {"rule": "anchor"})) for n in list(cur_all()) if n != "item_tokens"])
    nested = all(set(sum(rungs[k + 1]["channels"].values(), [])) <= set(sum(rungs[k]["channels"].values(), []))
                 for k in range(len(rungs) - 1))
    return {"task": "FEATSTAT-01", "status": "PROPOSED (statistics only; not a decision; no model outcome)",
            "judging_rule_reminder": "owner delegation (DECISION_LOG 35-36 via Ops): the smallest schema whose MRR@20 "
                                     "drop is <= 1% relative on the WORSE of GRU and SASRec, identical columns for "
                                     "both families; owner amendment section 7",
            "rungs_nested": nested, "window_policy": "USER_HISTORY_L50 (owner-approved; replaces SESSION_L50); "
                                                     "window statistics belong to DATA-05",
            "baseline_params": {"gru": base_g, "sasrec": base_s}, "rungs": rungs,
            "user_history_l50_hypotheses": L50_HYPOTHESES}


def cur_all():
    return ALL


if __name__ == "__main__":
    d = Path(sys.argv[1])
    out = build(d)
    (d / "FEATSTAT_LADDER.json").write_text(json.dumps(out, indent=1, allow_nan=False), encoding="utf-8")
    for r in out["rungs"]:
        print(r["id"], r["input_widths"], "channels", r["n_information_channels"], "gru", r["gru"]["params_total"],
              "sasrec", r["sasrec"]["params_total"], "rejected", [x["channel"] for x in r["removal_rejected"]])
