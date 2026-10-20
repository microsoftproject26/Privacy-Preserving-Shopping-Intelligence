"""v4trunc.recompute — the v1 formulas of every LOCAL_HISTORY channel, over a virtual history (pure numpy).

A virtual history is a list of event-table rows in store (chronological) order: H_h(q) followed by e_q (the query is
the LAST row). Nothing outside the list is read, except the per-user v1 stream carry-in C0 (a function of other
users' rows; see v4trunc.events).

Segment-prefix channels — EDA pipeline6.event_feature_sql, restricted to the listed rows:
  position_in_segment' = 1..k over the listed rows of each stored segment (row_number over (user, segment) by time);
  prefix_representative_count = position' - 1; prefix_item_count = position';
  prefix_known_category_count = number of distinct KNOWN main categories among the segment's listed rows so far;
  prefix_category_switch_count = cumulative CHANGED transitions (lag inside the listed segment rows: the first listed
      row of a segment has no previous category -> UNKNOWN, never CHANGED);
  current_known_category_run_length = length of the current run of one known main category (0 when unknown);
  log1p_segment_elapsed_s = log1p_prefix_elapsed_s = ln(1 + max(block_ts - origin, 0) seconds), origin = the first
      LISTED row of the segment (contract PARTIAL: a truncated origin when the segment starts before H_h(q)).
User context — EDA userctx.user_context_features for ONE row (the query), on the v1 replay stream restricted to the
listed rows (TRAIN / VS / VC stream rule of the query's partition), ordered by (epoch seconds, decision_id), W = 100
previous stream rows; window price sums as a difference of a cumulative sum that starts at C0 (the v1 arithmetic),
then the stored-column conversions of pipeline6 (float32; ln(1 + x) of the count/duration columns).
Every value is returned in the stored v1 column dtype (float32 / bool), ready for the unmodified encoder.
"""
from __future__ import annotations

import math
from typing import Mapping, Optional

import numpy as np

from .rules import USER_CONTEXT_WINDOW

F32 = np.float32


def segment_channels(segment: np.ndarray, ts_us: np.ndarray, main_code: np.ndarray) -> dict:
    """Segment-prefix channels of the listed rows (store order). main_code: identity code, -1 = unknown."""
    seg = np.asarray(segment, dtype=np.int64)
    ts = np.asarray(ts_us, dtype=np.int64)
    mc = np.asarray(main_code, dtype=np.int64)
    n = seg.size
    pos = np.zeros(n, np.int64)
    known_cnt = np.zeros(n, np.int64)
    sw_cnt = np.zeros(n, np.int64)
    run = np.zeros(n, np.int64)
    elapsed = np.zeros(n, np.float32)
    cur_seg, origin, seen, prev, k, cnt_sw, run_len, run_cat = None, 0, set(), None, 0, 0, 0, None
    for i in range(n):
        if cur_seg is None or seg[i] != cur_seg:
            cur_seg, origin, seen, prev, k, cnt_sw, run_len, run_cat = int(seg[i]), int(ts[i]), set(), None, 0, 0, 0, None
        k += 1
        m = int(mc[i]) if mc[i] >= 0 else None
        if m is not None and prev is not None and m != prev:
            cnt_sw += 1
        if m is not None:
            seen.add(m)
            if run_cat == m:
                run_len += 1
            else:
                run_len, run_cat = 1, m
        else:
            run_len, run_cat = 0, None
        pos[i], known_cnt[i], sw_cnt[i], run[i] = k, len(seen), cnt_sw, run_len
        elapsed[i] = F32(math.log(1 + max((int(ts[i]) - origin) // 1_000_000, 0)))
        prev = m
    return {"position_in_segment": pos,
            "prefix_representative_count": (pos - 1).astype(np.float32),
            "prefix_item_count": pos.astype(np.float32),
            "prefix_known_category_count": known_cnt.astype(np.float32),
            "prefix_category_switch_count": sw_cnt.astype(np.float32),
            "current_known_category_run_length": run.astype(np.float32),
            "log1p_segment_elapsed_s": elapsed.copy(),
            "log1p_prefix_elapsed_s": elapsed.copy()}


def _ln1p32(x: float) -> np.float32:
    """float32(ln(1 + float32(x))): the pipeline6 conversion of a float32 replay count / duration column."""
    return F32(math.log(1.0 + float(F32(x))))


def user_context_query(*, ts_s: np.ndarray, decision_id: np.ndarray, item: np.ndarray, main_code: np.ndarray,
                       brand_code: np.ndarray, log_price: np.ndarray, session: np.ndarray, q: int, carry_in: float,
                       window: int = USER_CONTEXT_WINDOW) -> dict:
    """The v1 user context of stream row `q` (index into the given arrays, which hold exactly the stream rows of the
    virtual history in any order): stored v1 columns (USER_CONTEXT_FEATURES + USER_CONTEXT_MASKS) plus audit fields."""
    ts_s = np.asarray(ts_s, dtype=np.int64)
    did = np.asarray(decision_id, dtype=np.int64)
    order = np.lexsort((did, ts_s))                     # v1 stream order inside one user: ts, then decision_id
    k = int(np.flatnonzero(order == int(q))[0])          # the query's position in the stream = # previous rows
    prior = order[:k]
    pu = k
    W = int(window)
    win = prior[max(0, k - W):k]
    ses = np.asarray(session, dtype=np.int64)[order[:k + 1]]
    fos = np.ones(k + 1, dtype=bool)
    fos[1:] = ses[1:] != ses[:-1]
    sess_ord = int(fos.sum()) - 1
    prev_session_count = 0 if pu == 0 else (sess_ord if fos[k] else sess_ord + 1)
    it = np.asarray(item, dtype=np.int64)
    mc = np.asarray(main_code, dtype=np.int64)
    bc = np.asarray(brand_code, dtype=np.int64)
    lp = np.asarray(log_price, dtype=np.float64)
    distinct_items = int(np.unique(it[win]).size)
    mw = mc[win]
    distinct_cats = int(np.unique(mw[mw >= 0]).size)

    def share(code: np.ndarray):
        cw = code[win]
        known_q = code[q] >= 0
        denom = float(np.count_nonzero(cw >= 0))
        num = float(np.count_nonzero((cw >= 0) & (cw == code[q]))) if known_q else 0.0
        sup = bool(known_q and denom > 0)
        return (num / max(denom, 1.0) if sup else 0.0), sup

    cat_share, cat_sup = share(mc)
    brand_share, brand_sup = share(bc)
    ok = np.isfinite(lp)
    pv = np.where(ok, lp, 0.0)
    cu = np.cumsum(np.concatenate(([float(carry_in)], pv[prior])), dtype=np.float64)   # cu[j] = C0 + v_0 + .. v_{j-1}
    psum = float(cu[pu] - cu[max(pu - W, 0)])
    pcnt = float(np.count_nonzero(ok[win]))
    price_sup = pcnt > 0
    mean_lp = psum / max(pcnt, 1.0) if price_sup else 0.0
    cmr = float(lp[q] - mean_lp) if (price_sup and ok[q]) else 0.0
    avail = pu > 0
    dt = float(ts_s[q] - ts_s[prior[-1]]) if avail else 0.0
    tspu = max(dt, 0.0)
    sess_avail = bool(avail and fos[k])
    tsps = max(dt, 0.0) if sess_avail else 0.0
    hh = F32(1.0 if avail else 0.0)
    out = {
        "log1p_prev_view_count": _ln1p32(pu),
        "log1p_prev_session_count": _ln1p32(prev_session_count),
        "recent_history_length": F32(min(pu, W)),
        "has_history": hh,
        "recent_distinct_items": F32(distinct_items),
        "recent_distinct_known_main_categories": F32(distinct_cats),
        "recent_category_match_share": F32(cat_share),
        "recent_brand_match_share": F32(brand_share),
        "recent_mean_log_price": F32(mean_lp),
        "current_minus_recent_mean_log_price": F32(cmr),
        "log1p_time_since_prev_user_event_s": F32(math.log(1.0 + max(float(F32(tspu)), 0.0))),
        "log1p_time_since_prev_session_last_event_s": F32(math.log(1.0 + max(float(F32(tsps)), 0.0))),
        "has_history_mask": hh,
        "recent_category_match_supported": F32(1.0 if cat_sup else 0.0),
        "recent_brand_match_supported": F32(1.0 if brand_sup else 0.0),
        "recent_price_supported": F32(1.0 if price_sup else 0.0),
        "prev_user_event_available": hh,
        "prev_session_available": F32(1.0 if sess_avail else 0.0),
    }
    audit = {"stream_prior_rows": int(pu), "window_rows": int(win.size),
             "brand_oov_collision": bool(bc[q] == 2 and np.any(bc[win] == 2)),
             "main_oov_collision": bool((mc[q] == 2 and np.any(mw == 2)) or np.count_nonzero(mw == 2) >= 2)}
    return {"columns": out, "audit": audit}


def stream_rows(split_code: np.ndarray, q_split_code: int, streams: Mapping[str, tuple], split_names: tuple) -> np.ndarray:
    """Boolean mask of the listed rows that belong to the v1 replay stream of a decision in partition q_split_code."""
    allowed = [split_names.index(s) for s in streams[split_names[int(q_split_code)]]]
    return np.isin(np.asarray(split_code), allowed)


def first_permitted_prev_gap(prev_gap_unavailable: np.ndarray, gap: np.ndarray) -> tuple:
    """The contract's LOCAL_EVENT exception: the earliest listed event's previous raw view is outside the budget."""
    f = np.asarray(prev_gap_unavailable).copy()
    g = np.asarray(gap).copy()
    if f.size:
        f[0] = True
        g[0] = F32(0.0)
    return f, g


def sequential_prefix(values: np.ndarray, carry_in: float = 0.0) -> np.ndarray:
    """cu[j] = carry_in + v_0 + ... + v_{j-1}, sequential float64 (numpy add.accumulate is sequential)."""
    return np.cumsum(np.concatenate(([float(carry_in)], np.asarray(values, dtype=np.float64))), dtype=np.float64)


__all__ = ["segment_channels", "user_context_query", "stream_rows", "first_permitted_prev_gap", "sequential_prefix"]
