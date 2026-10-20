"""`session_offset_tokens` (DECISION_LOG 47, owner): for window position j of decision d, token = min(session(d) -
session(j), 7) + 1 on valid positions, 0 on PAD (vocabulary 9), where session is the v1 BEHAVIOURAL session
(`session_index`: a new session iff the gap between consecutive raw blocks > 3600 s; never reset at a fence).

Proofs: the stored session id is the v1 behavioural session (fence-straddling sessions stay one session; every
disagreement with the partition-scoped stored raw-view gap is a fence or a barrier-block segment break); offset 0
exactly on the current-session positions; non-decreasing going back; causal; a fence-straddling synthetic session gets
one offset; a user with no earlier session gets all tokens 1; served batches carry the key under USER_HISTORY_L50 only
(cache == stream bitwise is covered by test_determinism). Negative controls: offsets computed from model segments
(fence-cut) and a future-event leak are both caught.
"""
from __future__ import annotations

import numpy as np
import pytest

from histkit import S, nc, raw_rows, v1_sessions
from v4data.common import TRAIN_END_US, VALIDATION_SPLITS
from v4hist.windows import (SESSION_OFFSET_CAP, SESSION_OFFSET_KEY, SESSION_OFFSET_VOCAB, RowIndex,
                            check_session_offsets, session_offset_tokens, window_rows)

NON_TEST = ("TRAIN",) + tuple(VALIDATION_SPLITS)
H = 3600 * S


def _sets(hist_train, hist_val):
    return (("TRAIN", hist_train, hist_train.decision_set("loss")),
            ("VALIDATION", hist_val, hist_val.decision_set("eval")))


def test_constants():
    assert (SESSION_OFFSET_KEY, SESSION_OFFSET_CAP, SESSION_OFFSET_VOCAB) == ("session_offset_tokens", 7, 9)


def test_real_session_ids_are_the_v1_behavioural_sessions(hist_val):
    ix = hist_val.index
    oc = ix.order_checks
    assert oc["session_decreasing"] == 0 and oc["session_starts_inside_segment"] == 0
    assert oc["fence_rows_same_session"] >= 1                     # sessions straddling a fence exist, one id each
    raw = raw_rows(0, NON_TEST, ["log1p_prev_raw_view_gap_s", "prev_gap_unavailable"])
    assert np.array_equal(raw["decision_id"].astype(np.int64), ix.decision_id)
    g = np.expm1(raw["log1p_prev_raw_view_gap_s"].astype(np.float64))
    unav = raw["prev_gap_unavailable"].astype(bool)
    uo, sid = ix.user_ord.astype(np.int64), ix.session_id.astype(np.int64)
    same_u = np.zeros(ix.n_rows, dtype=bool)
    same_u[1:] = uo[1:] == uo[:-1]
    new_s = np.ones(ix.n_rows, dtype=bool)
    new_s[1:] = ~same_u[1:] | (sid[1:] != sid[:-1])
    fence = np.zeros(ix.n_rows, dtype=bool)
    fence[1:] = same_u[1:] & (ix.split_code[1:] != ix.split_code[:-1])
    seg_start = ix.position == 1
    # a new session (not a user's first row) always shows a stored gap > 3600 s or an unavailable previous view
    inner = new_s & same_u
    assert int(inner.sum()) > 100_000 and bool(np.all((g[inner] > 3600.5) | unav[inner]))
    # the stored gap is partition-scoped and skips barrier blocks: where it says > 3600 s but the session continues,
    # the row is always a fence or a barrier-block model-segment break
    odd = same_u & ~new_s & (g > 3600.5)
    assert bool(np.all(fence[odd] | seg_start[odd]))


def test_tokens_exhaustive_on_real_decisions(hist_train, hist_val):
    for name, hs, ds in _sets(hist_train, hist_val):
        ix = hs.index
        rows = ix.rows_of(ds.decision_id)
        tok = session_offset_tokens(ix.session_id, ds.end_row, ds.length)
        idx, valid = window_rows(ds.end_row, ds.length)
        rep = check_session_offsets(ix, rows, idx, valid, tok)
        assert rep["n_decisions"] == len(ds) and sum(rep["token_counts_1_to_8"]) == rep["n_positions"]
        assert rep["decisions_with_an_earlier_session"] > 0.4 * len(ds), name
        assert rep["token_counts_1_to_8"][-1] > 0, name                          # the cap bucket is reached
        both = valid[:, 1:] & valid[:, :-1]
        assert np.all((tok[:, :-1] >= tok[:, 1:])[both])                        # non-decreasing going back


def test_no_earlier_session_means_every_token_is_one(hist_val):
    ds = hist_val.decision_set("eval")
    ix = hist_val.index
    uo = ix.user_ord.astype(np.int64)
    first = np.searchsorted(uo, uo, side="left")
    r = ds.end_row
    cand = ix.session_id[r] == ix.session_id[first[r]]
    assert int(cand.sum()) > 1_000
    tok = session_offset_tokens(ix.session_id, r[cand], ds.length[cand])
    _, valid = window_rows(r[cand], ds.length[cand])
    assert np.all(tok[valid] == 1) and np.all(tok[~valid] == 0)


def test_served_batches_carry_the_key_under_user_history_only(hist_val, providers_user, providers_session):
    ds = hist_val.decision_set("eval")
    pos = np.arange(0, 512)
    b = hist_val.batch(ds, pos)
    t = b[SESSION_OFFSET_KEY]
    assert t.dtype.is_floating_point is False and tuple(t.shape) == (512, 50) and str(t.dtype) == "torch.int64"
    assert np.array_equal(t.numpy(), session_offset_tokens(hist_val.index.session_id, ds.end_row[pos], ds.length[pos]))
    assert np.array_equal(t.numpy() > 0, b["attention_mask"].numpy())
    tu, eu = providers_user
    ts, es = providers_session
    assert SESSION_OFFSET_KEY in tu.batch(np.arange(64)) and SESSION_OFFSET_KEY in next(iter(eu.batches(1024)))
    assert SESSION_OFFSET_KEY not in ts.batch(np.arange(64)) and SESSION_OFFSET_KEY not in next(iter(es.batches(1024)))
    assert tu.describe()["history"]["session_offset_tokens"]["vocab"] == 9
    assert ts.describe()["history"]["session_offset_tokens"] is None


# ------------------------------------------------------------------------------------------------ synthetic
def _straddle_fixture():
    """One user: two TRAIN events 20 and 10 minutes before 2019-11-16 00:00, two VALIDATION_SELECT events 5 and 15
    minutes after (a new model segment at the fence, the SAME behavioural session), then a later VS session (2 h gap)."""
    ts = np.array([TRAIN_END_US - 20 * 60 * S, TRAIN_END_US - 10 * 60 * S, TRAIN_END_US + 5 * 60 * S,
                   TRAIN_END_US + 15 * 60 * S, TRAIN_END_US + 15 * 60 * S + 2 * H], dtype=np.int64)
    uo = np.zeros(5, dtype=np.int64)
    ix = RowIndex.from_arrays(user_ord=uo, position=[1, 2, 1, 2, 1], ts_us=ts, split_code=[0, 0, 1, 1, 1],
                              session_id=v1_sessions(uo, ts))
    return ix


def test_a_fence_straddling_session_gets_one_offset():
    ix = _straddle_fixture()
    assert ix.session_id.tolist() == [1, 1, 1, 1, 2] and ix.order_checks["fence_rows_same_session"] == 1
    rows = np.array([3, 4], dtype=np.int64)
    ln = np.minimum(50, ix.user_rank[rows] + 1)
    tok = session_offset_tokens(ix.session_id, rows, ln)
    assert tok[0, :4].tolist() == [1, 1, 1, 1]                     # one session across the fence: offset 0 throughout
    assert tok[1, :5].tolist() == [2, 2, 2, 2, 1]                  # the next session: the straddling one is offset 1
    idx, valid = window_rows(rows, ln)
    check_session_offsets(ix, rows, idx, valid, tok)


def test_synthetic_exhaustive():
    from histkit import synthetic_index
    ix = synthetic_index(seed=11, n_users=300)
    r = np.arange(ix.n_rows, dtype=np.int64)
    ln = np.minimum(50, ix.user_rank.astype(np.int64) + 1)
    idx, valid = window_rows(r, ln)
    rep = check_session_offsets(ix, r, idx, valid, session_offset_tokens(ix.session_id, r, ln))
    assert rep["decisions_with_an_earlier_session"] > 0


# ------------------------------------------------------------------------------------------------ negative controls
def _segment_tokens(ix, rows, ln):
    """The deliberately wrong builder: offsets counted in MODEL SEGMENTS (cut at every fence and barrier)."""
    seg = np.cumsum(ix.position == 1).astype(np.int64)
    return session_offset_tokens(seg, rows, ln)


@nc("session offsets computed from model segments (fence-cut) must fail the behavioural-session proof")
def test_nc_offsets_from_model_segments_are_caught():
    ix = _straddle_fixture()
    rows = np.array([3, 4], dtype=np.int64)
    ln = np.minimum(50, ix.user_rank[rows] + 1)
    idx, valid = window_rows(rows, ln)
    check_session_offsets(ix, rows, idx, valid, _segment_tokens(ix, rows, ln))


@nc("session offsets anchored at a later (future) session must fail the proof")
def test_nc_future_event_leak_is_caught(hist_val):
    ix = hist_val.index
    ds = hist_val.decision_set("eval")
    uo = ix.user_ord.astype(np.int64)
    last = np.searchsorted(uo, uo, side="right") - 1
    r = ds.end_row
    ln = ds.length
    idx, valid = window_rows(r, ln)
    ic = np.where(valid, idx, 0)
    sid = ix.session_id.astype(np.int64)
    leak = np.where(valid, np.minimum(sid[last[r]][:, None] - sid[ic], SESSION_OFFSET_CAP) + 1, 0).astype(np.int64)
    check_session_offsets(ix, ix.rows_of(ds.decision_id), idx, valid, leak)
