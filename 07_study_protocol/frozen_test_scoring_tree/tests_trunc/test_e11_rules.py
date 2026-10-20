"""E11.4 recompute rules per locality class, E11.5 frozen normalization and the E11.7 controls.

  * budget semantics (h counts strictly earlier events; h = 0 -> the query only; window length min(50, h + 1));
  * LOCAL_EVENT and SHARED columns are the stored values of the permitted rows for every h;
  * LOCAL_HISTORY per-event channels are recomputed over the permitted rows (segment prefix, truncated origin) and
    equal the stored values whenever the permitted rows start at a segment start;
  * the PREV_GAP rule at the earliest permitted event;
  * the query's user context = an independent naive v1 replay over the permitted rows of its stream (carry-in);
  * the v1 stream rule (a VC query never replays VS events), the carry-in is a function of other users only;
  * E11.7: deletion invariance (+ NC: a builder that uses one extra event), the leak control (+ NC).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from trunckit import (ExtraPriorEventBuilder, FullHistoryUctxBuilder, assert_raises, make_builder, nc,
                      synthetic_table, val_query_rows)
from v4trunc import events as E
from v4trunc.builder import FULL, SEGMENT_COLUMNS, UCTX_COLUMNS, BudgetError, TruncatedHistoryBuilder
from v4trunc.golden import batches_equal, deletion_invariance
from v4trunc.rules import COLUMN_RULES, KEEP_EVENT, SHARED_FROZEN, STREAMS, USER_CONTEXT_WINDOW

SPL = ("TRAIN", "VALIDATION_SELECT", "VALIDATION_CONFIRM")


def _pairs(table, n, seed=1, truncated=True):
    """Deterministic (q_row, h) pairs over VALIDATION queries with h in the registered budgets (h < avail if
    truncated)."""
    rng = np.random.default_rng(seed)
    qs = val_query_rows(table)
    out = []
    for q in rng.permutation(qs).tolist():
        avail = q - int(table.user_start[int(table.row_user[q])])
        hs = [h for h in (0, 1, 5, 10, 20, 50) if (h < avail if truncated else h <= avail)]
        if hs:
            out.append((q, int(rng.choice(hs))))
        if len(out) >= n:
            break
    return out


def _syn_builder(t):
    class NoEncode(TruncatedHistoryBuilder):
        def __init__(self, table):
            self.table = table
    return NoEncode(t)


# ------------------------------------------------------------------------------------------------ budget semantics
def test_budget_semantics_synthetic():
    ev = [("TRAIN", 1000 + 10 * i, 0, 0, i, 3, 3, 5.0, 1, False, True, 1010 + 10 * i) for i in range(6)]
    t = synthetic_table([ev])
    b = _syn_builder(t)
    q = 5
    for h in range(0, 6):
        rows = b.listed_rows(q, h)
        assert rows.size == h + 1 and rows[-1] == q and np.all(np.diff(rows) == 1)
    assert b.listed_rows(q, FULL).size == 6
    assert_raises(BudgetError, b.listed_rows, q, 6)
    assert_raises(BudgetError, b.listed_rows, q, -1)


def test_window_length_and_query_last(builder, sample_table):
    for q, h in _pairs(sample_table, 60, truncated=False):
        bt = builder.batch([(q, h)])
        assert int(bt["lengths"][0]) == min(50, h + 1)
        assert int(bt["decision_id"][0]) == int(sample_table.decision_id[q])
        assert int(bt["last_valid_index"][0]) == min(50, h + 1) - 1
        assert int(bt["item_tokens"][0, min(50, h + 1) - 1]) == int(sample_table.raw["product_idx"][q])


# ------------------------------------------------------------------------------------------------ locality classes
def test_local_event_and_shared_columns_unchanged(builder, sample_table):
    t = sample_table
    keep_cols = [c for c, (_, r) in COLUMN_RULES.items() if r in (KEEP_EVENT, SHARED_FROZEN)]
    for q, h in _pairs(t, 120):
        blk = builder.block(q, h)
        for c in keep_cols:
            assert np.array_equal(np.asarray(blk.raw[c]), np.asarray(t.raw[c][blk.rows])), c


def test_segment_prefix_recomputed_over_permitted_rows_synthetic():
    # one segment of 6 events with main categories 3,3,4,1(unknown),4,4 ; query = last
    mains = [3, 3, 4, 1, 4, 4]
    ev = [("TRAIN", 1000 + 60 * i, 0, 0, i, mains[i], 3, 5.0, 1, False, True, 1060 + 60 * i) for i in range(6)]
    t = synthetic_table([ev])
    b = _syn_builder(t)
    rows = b.listed_rows(5, 2)                                  # rows 3, 4, 5: mains 1(unknown), 4, 4
    from v4trunc.recompute import segment_channels
    s = segment_channels(t.segment[rows], t.ts_us[rows], t.main_code[rows])
    assert s["position_in_segment"].tolist() == [1, 2, 3]
    assert s["prefix_representative_count"].tolist() == [0.0, 1.0, 2.0]
    assert s["prefix_known_category_count"].tolist() == [0.0, 1.0, 1.0]
    assert s["prefix_category_switch_count"].tolist() == [0.0, 0.0, 0.0]
    assert s["current_known_category_run_length"].tolist() == [0.0, 1.0, 2.0]
    assert s["log1p_segment_elapsed_s"].tolist() == [np.float32(0.0), np.float32(math.log(61)),
                                                    np.float32(math.log(121))]          # truncated origin (PARTIAL)
    s6 = segment_channels(t.segment, t.ts_us, t.main_code)                              # the whole segment
    assert s6["prefix_category_switch_count"].tolist() == [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    assert s6["prefix_known_category_count"].tolist() == [1.0, 1.0, 2.0, 2.0, 2.0, 2.0]


def test_segment_channels_equal_stored_when_permitted_rows_start_a_segment(builder, sample_table):
    t = sample_table
    n = 0
    for q, h in _pairs(t, 400, seed=3):
        blk = builder.block(q, h)
        if int(t.position[blk.rows[0]]) != 1:
            continue
        n += 1
        for c in SEGMENT_COLUMNS:
            assert np.array_equal(blk.raw[c], t.raw[c][blk.rows]), c
        assert not blk.audit["segment_origin_truncated"]
    assert n > 10


def test_prev_gap_rule(builder, sample_table):
    t = sample_table
    for q, h in _pairs(sample_table, 120, seed=4):
        blk = builder.block(q, h)
        assert bool(blk.raw["prev_gap_unavailable"][0]) and float(blk.raw["log1p_prev_raw_view_gap_s"][0]) == 0.0
        assert np.array_equal(blk.raw["prev_gap_unavailable"][1:], t.raw["prev_gap_unavailable"][blk.rows[1:]])
        assert np.array_equal(blk.raw["log1p_prev_raw_view_gap_s"][1:],
                              t.raw["log1p_prev_raw_view_gap_s"][blk.rows[1:]])


def test_non_query_user_context_zeroed(builder, sample_table):
    for q, h in _pairs(sample_table, 40, seed=5):
        blk = builder.block(q, h)
        for c in UCTX_COLUMNS:
            assert np.all(blk.raw[c][:-1] == 0)


# ------------------------------------------------------------------------------------------------ user context
def _naive_uctx(t, rows, W=USER_CONTEXT_WINDOW):
    """An independent per-row statement of the v1 user-context definitions over the listed rows."""
    q = int(rows[-1])
    qs = SPL[int(t.split_code[q])]
    allowed = {SPL.index(s) for s in STREAMS[qs]}
    sr = [int(r) for r in rows if int(t.split_code[r]) in allowed]
    sr.sort(key=lambda r: (int(t.ts_us[r] // 1_000_000), int(t.decision_id[r])))
    k = sr.index(q)
    prior = sr[:k]
    win = prior[-W:] if W > 0 else []
    ses = [int(t.session[r]) for r in sr[:k + 1]]
    fos = [True] + [ses[i] != ses[i - 1] for i in range(1, len(ses))]
    psc = 0 if k == 0 else (sum(fos) - 1 if fos[k] else sum(fos))
    cu = [float(t.carry[qs][int(t.row_user[q])])]
    for r in prior:
        v = t.log_price[r]
        cu.append(cu[-1] + (float(v) if np.isfinite(v) else 0.0))
    psum = cu[k] - cu[max(k - W, 0)]
    pcnt = float(sum(1 for r in win if np.isfinite(t.log_price[r])))
    mean = psum / max(pcnt, 1.0) if pcnt > 0 else 0.0

    def share(code):
        kn = [r for r in win if code[r] >= 0]
        if code[q] >= 0 and kn:
            return sum(1 for r in kn if code[r] == code[q]) / len(kn), 1.0
        return 0.0, 0.0
    cs, csup = share(t.main_code)
    bs, bsup = share(t.brand_code)
    dt = float(int(t.ts_us[q] // 1_000_000) - int(t.ts_us[prior[-1]] // 1_000_000)) if k else 0.0
    f = np.float32
    return {"log1p_prev_view_count": f(math.log(1 + k)), "log1p_prev_session_count": f(math.log(1 + psc)),
            "recent_history_length": f(min(k, W)), "has_history": f(1.0 if k else 0.0),
            "recent_distinct_items": f(len({int(t.item[r]) for r in win})),
            "recent_distinct_known_main_categories": f(len({int(t.main_code[r]) for r in win if t.main_code[r] >= 0})),
            "recent_category_match_share": f(cs), "recent_brand_match_share": f(bs),
            "recent_mean_log_price": f(mean),
            "current_minus_recent_mean_log_price": f((t.log_price[q] - mean) if (pcnt > 0 and np.isfinite(t.log_price[q])) else 0.0),
            "log1p_time_since_prev_user_event_s": f(math.log(1 + max(dt, 0.0))),
            "log1p_time_since_prev_session_last_event_s": f(math.log(1 + (max(dt, 0.0) if (k and fos[k]) else 0.0))),
            "has_history_mask": f(1.0 if k else 0.0), "recent_category_match_supported": f(csup),
            "recent_brand_match_supported": f(bsup), "recent_price_supported": f(1.0 if pcnt > 0 else 0.0),
            "prev_user_event_available": f(1.0 if k else 0.0), "prev_session_available": f(1.0 if (k and fos[k]) else 0.0)}


def test_user_context_equals_naive_reference(builder, sample_table):
    t = sample_table
    for q, h in _pairs(t, 300, seed=6, truncated=False) + [(q, FULL) for q, _ in _pairs(t, 60, seed=7)]:
        blk = builder.block(q, h)
        want = _naive_uctx(t, blk.rows)
        for c, v in want.items():
            assert np.float32(blk.raw[c][-1]).tobytes() == np.float32(v).tobytes(), (c, q, h)


def test_h0_has_no_history_synthetic():
    ev = [("TRAIN", 1000 + 10 * i, 0, 0, i, 3, 3, 5.0, 1, False, True, 1010 + 10 * i) for i in range(4)]
    t = synthetic_table([ev])
    b = _syn_builder(t)
    uc = b.user_context(b.listed_rows(3, 0))["columns"]
    for c in ("has_history", "recent_history_length", "log1p_prev_view_count", "recent_price_supported",
              "recent_category_match_supported", "recent_brand_match_supported", "prev_user_event_available",
              "recent_mean_log_price", "recent_distinct_items"):
        assert float(uc[c]) == 0.0, c


def test_vc_query_never_replays_vs_events_synthetic():
    ev = [("TRAIN", 1000, 0, 0, 1, 3, 3, 5.0, 1, False, True, 1010), ("TRAIN", 1010, 0, 0, 2, 3, 3, 5.0, 1, False, True, 5000),
          ("VALIDATION_SELECT", 5000, 1, 1, 3, 3, 3, 5.0, 1, True, False, 5010),
          ("VALIDATION_SELECT", 5010, 1, 1, 4, 3, 3, 5.0, 1, True, False, 9000),
          ("VALIDATION_CONFIRM", 9000, 2, 2, 5, 3, 3, 5.0, 1, True, False, 9010)]
    t = synthetic_table([ev])
    b = _syn_builder(t)
    uc = b.user_context(b.listed_rows(4, 4))["columns"]
    assert uc["log1p_prev_view_count"] == np.float32(math.log(3))            # the 2 TRAIN rows only
    uc2 = b.user_context(b.listed_rows(4, 2))["columns"]
    assert float(uc2["has_history"]) == 0.0                                  # H_2 = the two VS events -> none replayed


def test_carry_in_is_a_function_of_other_users_only():
    uid = np.array(["10", "10", "11", "11", "12"], dtype=object)
    ts = np.array([1, 2, 1, 2, 1], dtype=np.int64)
    did = np.arange(5, dtype=np.int64)
    sc = np.zeros(5, dtype=np.int8)
    lp = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
    c0 = E.stream_carry_in(uid, ts, did, sc, lp)["TRAIN"]
    assert c0["10"] == 0.0 and c0["11"] == 1.1 + 2.2 and c0["12"] == (1.1 + 2.2) + 3.3 + 4.4
    lp2 = lp.copy()
    lp2[2:4] = [9.0, 9.0]                                                    # user 11's own rows
    assert E.stream_carry_in(uid, ts, did, sc, lp2)["TRAIN"]["11"] == c0["11"]
    lp3 = lp.copy()
    lp3[0] = 7.0                                                             # a preceding user's row
    assert E.stream_carry_in(uid, ts, did, sc, lp3)["TRAIN"]["11"] != c0["11"]


# ------------------------------------------------------------------------------------------------ E11.7
def _mk(t):
    return make_builder(t)


def test_e11_7_deletion_invariance(builder, sample_table):
    rec = deletion_invariance(builder, _pairs(sample_table, 80, seed=8), _mk)
    assert rec["n_checked"] == 80 and rec["n_changed"] == 0, rec


@nc("a builder that uses one extra (pre-window) event must change when the pre-window events are deleted")
def test_nc_deletion_invariance_extra_prior_event(sample_table):
    leaky = make_builder(sample_table, ExtraPriorEventBuilder)
    rec = deletion_invariance(leaky, _pairs(sample_table, 40, seed=9), lambda t: make_builder(t, ExtraPriorEventBuilder))
    assert rec["n_changed"] == 0, rec


def _leak_differs(builder, leaky, pairs):
    return [not batches_equal(builder.batch([p]), leaky.batch([p])) for p in pairs]


def test_e11_7_leaking_full_history_user_context_is_detected(builder, sample_table):
    pairs = [(q, h) for q, h in _pairs(sample_table, 80, seed=10) if h < 20]
    d = _leak_differs(builder, make_builder(sample_table, FullHistoryUctxBuilder), pairs)
    assert len(d) > 20 and np.mean(d) > 0.9


@nc("the full-history user_context builder must NOT reproduce the E11.4 truncated features")
def test_nc_leaking_builder_equals_e11_4(builder, sample_table):
    pairs = [(q, h) for q, h in _pairs(sample_table, 40, seed=11) if h < 20]
    assert not any(_leak_differs(builder, make_builder(sample_table, FullHistoryUctxBuilder), pairs))


def test_session_keys_come_from_the_truncated_window(builder, sample_table):
    from v4hist.windows import session_offset_tokens
    t = sample_table
    for q, h in _pairs(t, 40, seed=12):
        blk = builder.block(q, h)
        bt = builder.batch_of_blocks([blk])
        n = min(50, blk.n)
        tok = session_offset_tokens(blk.session, np.array([blk.n - 1]), np.array([n]))
        assert np.array_equal(bt["session_offset_tokens"].numpy(), tok)
        assert int((bt["session_offset_tokens"][0] > 0).sum()) == n


def test_frozen_normalizers_shared_across_h(builder, sample_table):
    """E11.5: the same normalizer object and Clean V2 state serve every h (no per-h refit)."""
    q, _ = _pairs(sample_table, 1, seed=13)[0]
    avail = q - int(sample_table.user_start[int(sample_table.row_user[q])])
    for h in [x for x in (0, 1, 5) if x <= avail]:
        bt = builder.batch([(q, h)])
        assert bt["clean_v2_state_sha256"] == builder.state_sha256 and builder.renorm is None



# ------------------------------------------------------------------------------------------------ B-F4 rule coverage
def _coverage_ok(rules) -> bool:
    from v4trunc.builder import RuleCoverageError, check_rule_coverage
    try:
        check_rule_coverage(rules)
    except RuleCoverageError:
        return False
    return True


def test_rule_coverage_equals_builder_column_sets():
    """REV-ANA-01 B-F4: the builder's hard-coded column sets EQUAL the COLUMN_RULES rule classes (asserted at import);
    an UNSUPPORTED_FUTURE / unknown rule, a column moved to another class, a missing or an extra column all refuse."""
    from v4trunc.builder import PREV_GAP_COLUMNS, RULE_COVERAGE, RuleCoverageError, check_rule_coverage
    from v4trunc.rules import LOCAL_HISTORY, RECOMPUTE_USER
    assert set(RULE_COVERAGE["segment"]) == set(SEGMENT_COLUMNS)
    assert set(RULE_COVERAGE["user_context"]) == set(UCTX_COLUMNS)
    assert set(RULE_COVERAGE["prev_gap"]) == set(PREV_GAP_COLUMNS)
    assert set(RULE_COVERAGE["unchanged"]) == set(E.RAW_COLUMNS) - set(SEGMENT_COLUMNS) - set(UCTX_COLUMNS) \
        - set(PREV_GAP_COLUMNS)
    assert all(COLUMN_RULES[c][1] in (KEEP_EVENT, SHARED_FROZEN) for c in RULE_COVERAGE["unchanged"])
    assert _coverage_ok(dict(COLUMN_RULES))
    bad = [dict(COLUMN_RULES, recent_brand_match_share=(LOCAL_HISTORY, "UNSUPPORTED_FUTURE")),
           dict(COLUMN_RULES, log1p_price=(LOCAL_HISTORY, RECOMPUTE_USER)),               # served unchanged
           dict(COLUMN_RULES, prefix_item_count=("LOCAL_EVENT", KEEP_EVENT)),             # recomputed by the builder
           {c: v for c, v in COLUMN_RULES.items() if c != "brand_idx"},
           dict(COLUMN_RULES, not_a_stored_column=("LOCAL_EVENT", KEEP_EVENT))]
    for rules in bad:
        assert not _coverage_ok(rules)
    assert_raises(RuleCoverageError, check_rule_coverage,
                  dict(COLUMN_RULES, has_history=(LOCAL_HISTORY, "UNSUPPORTED_FUTURE")))


@nc("a rule table with an UNSUPPORTED_FUTURE column must be refused by the builder coverage check")
def test_nc_rule_coverage_unsupported_future():
    from v4trunc.rules import LOCAL_HISTORY
    assert _coverage_ok(dict(COLUMN_RULES, recent_brand_match_share=(LOCAL_HISTORY, "UNSUPPORTED_FUTURE")))
