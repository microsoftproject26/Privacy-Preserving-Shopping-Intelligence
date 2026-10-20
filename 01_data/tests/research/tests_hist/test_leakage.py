"""Leakage proofs. Exhaustive on a synthetic fixture (every row a decision, both policies, against a naive reference)
and on every real bucket-0 TRAIN loss and VALIDATION eval decision: every window row is the decision's user, at or
before the decision in the user's stream, has an event timestamp <= the decision's and < the end of the decision's
partition, and no row of a later partition (so no TEST row) appears. The served bytes equal the checked rows.
Negative controls: a window including the next event, a window crossing into another user, a window reading a TEST
row for a VALIDATION decision."""
from __future__ import annotations

import numpy as np
import pytest

from histkit import POLICIES, nc, naive_window, synthetic_index
from v4hist.policy import CONTEXT_LENGTH as L, SESSION_L50, USER_HISTORY_L50
from v4hist.windows import TEST_CODE, check_served_batch, check_windows, policy_lengths, window_rows


@pytest.fixture(scope="module")
def syn():
    return synthetic_index(seed=20260926, n_users=400)


def _all_rows(ix):
    return np.arange(ix.n_rows, dtype=np.int64)


def test_synthetic_fixture_is_rich(syn):
    """The fixture exercises every case: several segments and partitions per user, sessions > L, users > L events."""
    assert syn.n_rows > 20_000 and syn.n_users > 300
    assert int((syn.position > L).sum()) > 0 and int((syn.user_rank >= L).sum()) > 1_000
    assert set(np.unique(syn.split_code).tolist()) == {0, 1, 2, 3}
    assert int((syn.position == 1).sum()) > 2 * syn.n_users


@pytest.mark.parametrize("policy", POLICIES)
def test_synthetic_exhaustive_every_row_both_policies(syn, policy):
    r = _all_rows(syn)
    idx, valid = window_rows(r, policy_lengths(syn, r, policy))
    rep = check_windows(syn, r, idx, valid, policy=policy)
    assert rep["n_decisions"] == syn.n_rows
    for k in range(syn.n_rows):                                       # the naive definition, one decision at a time
        assert idx[k][valid[k]].tolist() == naive_window(syn, k, policy), k
    if policy == USER_HISTORY_L50:
        assert rep["rows_from_earlier_partitions"] > 0 and rep["rows_from_earlier_segments"] > 0
        tst = syn.split_code == TEST_CODE                             # synthetic TEST decisions see earlier partitions
        assert np.any(syn.split_code[idx[tst][valid[tst]]] < TEST_CODE)
    else:
        assert rep["rows_from_earlier_partitions"] == 0 and rep["rows_from_earlier_segments"] == 0


def _real_sets(hist_train, hist_val):
    return (("TRAIN", hist_train, hist_train.decision_set("loss"), hist_train.raw_decision_set("loss")),
            ("VALIDATION", hist_val, hist_val.decision_set("eval"), hist_val.raw_decision_set("eval")))


def test_real_bucket0_exhaustive_every_decision_both_policies(hist_train, hist_val):
    tot = {}
    for name, hs, du, dsess in _real_sets(hist_train, hist_val):
        rows = hs.index.rows_of(du.decision_id)                        # from the decision identity, not the window
        assert np.array_equal(rows, du.end_row) and np.array_equal(rows, dsess.end_row)
        for pol, ds in ((USER_HISTORY_L50, du), (SESSION_L50, dsess)):
            idx, valid = window_rows(ds.end_row, ds.length)
            rep = check_windows(hs.index, rows, idx, valid, policy=pol)
            tot[(name, pol)] = rep
    assert tot[("TRAIN", USER_HISTORY_L50)]["n_decisions"] > 600_000
    assert tot[("VALIDATION", USER_HISTORY_L50)]["n_decisions"] > 170_000
    v = tot[("VALIDATION", USER_HISTORY_L50)]
    assert v["decisions_with_earlier_partition_rows"] > 0.5 * v["n_decisions"]       # the policy is not vacuous
    assert tot[("VALIDATION", SESSION_L50)]["rows_from_earlier_partitions"] == 0


def test_served_batches_equal_the_checked_rows(hist_train, hist_val):
    rng = np.random.default_rng(5)
    for _name, hs, du, dsess in _real_sets(hist_train, hist_val):
        for ds in (du, dsess):
            for _ in range(8):
                pos = rng.choice(len(ds), 256, replace=False)
                idx, valid = window_rows(ds.end_row[pos], ds.length[pos])
                check_served_batch(hs.batch(ds, pos), hs.store.cs, hs.store.spec, idx, valid)


def test_validation_windows_never_hold_a_later_partition_row(hist_val):
    """VALIDATION_SELECT decisions never see a VALIDATION_CONFIRM row (a later partition held by the same store)."""
    du = hist_val.decision_set("eval")
    ix = hist_val.index
    vs = ix.split_code[du.end_row] == 1
    idx, valid = window_rows(du.end_row[vs], du.length[vs])
    assert int(vs.sum()) > 100_000 and np.all(ix.split_code[idx[valid]] <= 1)
    assert int(((ix.split_code == 2)).sum()) > 0                      # the store does hold VALIDATION_CONFIRM rows


# ------------------------------------------------------------------------------------------ negative controls
@nc("a window that includes the next event (the resolver shifted by one row) must fail the causality check")
def test_nc_window_includes_the_next_event(hist_val):
    du = hist_val.decision_set("eval")
    ix = hist_val.index
    nxt = np.minimum(du.end_row + 1, ix.n_rows - 1)
    keep = ((du.end_row + 1 < ix.n_rows) & (ix.user_ord[nxt] == ix.user_ord[du.end_row])
            & (ix.split_code[nxt] == ix.split_code[du.end_row]))         # the next event of the same user / partition
    rows = du.end_row[keep]
    idx, valid = window_rows(rows + 1, du.length[keep])               # the deliberately wrong resolver
    check_windows(ix, rows, idx, valid, policy=USER_HISTORY_L50)


@nc("a window without the user_start clamp (lo = r - 49) must fail the same-user check")
def test_nc_window_crosses_into_another_user(hist_train):
    du = hist_train.decision_set("loss")
    ix = hist_train.index
    ln = np.minimum(L, du.end_row + 1).astype(np.int16)                # no max(user_start, .) clamp
    idx, valid = window_rows(du.end_row, ln)
    check_windows(ix, du.end_row, idx, valid, policy=USER_HISTORY_L50)


def _val_decisions_of_users_with_test_rows(syn):
    val = np.flatnonzero((syn.split_code == 1) | (syn.split_code == 2))
    uo = syn.user_ord.astype(np.int64)
    last_of_user = np.searchsorted(uo, uo, side="right") - 1
    r = val[syn.split_code[last_of_user[val]] == TEST_CODE]
    return r, last_of_user[r]


def test_synthetic_has_validation_decisions_of_users_with_test_rows(syn):
    r, end = _val_decisions_of_users_with_test_rows(syn)
    assert r.size > 1_000 and np.all(syn.split_code[end] == TEST_CODE)
    idx, valid = window_rows(r, policy_lengths(syn, r, USER_HISTORY_L50))      # the correct resolver: no TEST row
    check_windows(syn, r, idx, valid, policy=USER_HISTORY_L50)
    assert not np.any(syn.split_code[idx[valid]] == TEST_CODE)


@nc("a VALIDATION decision's window that reads a TEST-partition row (synthetic fixture) must fail")
def test_nc_validation_window_reads_a_test_row(syn):
    """The deliberately wrong resolver takes the user's LATEST L events, whatever the decision time."""
    r, end = _val_decisions_of_users_with_test_rows(syn)
    ln = np.minimum(L, syn.user_rank[end].astype(np.int64) + 1)
    idx, valid = window_rows(end, ln)
    check_windows(syn, r, idx, valid, policy=USER_HISTORY_L50)
