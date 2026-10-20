"""Session-mode CANDIDATE features (DECISION_LOG 48, owner), USER_HISTORY_L50 only:
session_mode_main_category_tokens int64 [B, 50], current_matches_session_mode float32 [B],
current_matches_session_mode_available float32 [B], session_mode_share float32 [B].

Proofs against a naive per-position reference (Python loops, written independently of the vectorised code): the tie
rule (most recent occurrence); a PAST session's token is its full in-window session mode; a CURRENT-session token is the
causal prefix mode, and at the decision position it equals the mode computed directly over that prefix; match / mask /
share follow the definition; everything is exact on real bucket-0 windows (TRAIN and VALIDATION) and on random synthetic
windows; the served batches carry the keys (identical builder for TRAIN and eval: test_identical_builder_for_train_and_eval
compares every key bitwise). Negative control: a non-causal variant (a later event of the current session changes an
earlier position's mode) is caught.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from histkit import nc
from v4hist.windows import (MAIN_CATEGORY_VOCAB, MAIN_UNKNOWN, MATCH_KEY, MATCH_MASK_KEY, SESSION_MODE_KEY, SHARE_KEY,
                            USER_ONLY_KEYS, session_mode_features, window_rows)

B = 256


def _mode(items):
    """items: [(position, category)] of KNOWN categories; the most frequent, ties -> the latest occurrence."""
    if not items:
        return MAIN_UNKNOWN
    c = Counter(cat for _, cat in items)
    top = max(c.values())
    lastpos = {}
    for k, cat in items:
        lastpos[cat] = k
    return max((cat for cat in c if c[cat] == top), key=lambda cat: lastpos[cat])


def naive(m, v, sp):
    """The reference definition for one window."""
    L = len(m)
    n = int(np.sum(v))
    cur = sp[n - 1]
    out = [0] * L
    for j in range(n):
        run = [k for k in range(n) if sp[k] == sp[j]]
        use = [k for k in run if k <= j] if sp[j] == cur else run
        out[j] = _mode([(k, int(m[k])) for k in use if m[k] >= 3])
    pref = [k for k in range(n) if sp[k] == cur]
    known = [(k, int(m[k])) for k in pref if m[k] >= 3]
    pm = _mode(known)
    cc = int(m[n - 1])
    avail = cc >= 3 and pm >= 3
    match = float(avail and cc == pm)
    share = (sum(1 for _, cat in known if cat == pm) / len(known)) if avail else 0.0
    return out, match, float(avail), share


def check_against_naive(m, v, sp, feats, max_rows=None):
    n = m.shape[0] if max_rows is None else min(max_rows, m.shape[0])
    for b in range(n):
        out, match, avail, share = naive(m[b], v[b], sp[b])
        assert feats[SESSION_MODE_KEY][b].tolist() == out, b
        assert feats[MATCH_KEY][b] == np.float32(match) and feats[MATCH_MASK_KEY][b] == np.float32(avail), b
        assert abs(float(feats[SHARE_KEY][b]) - share) <= 1e-6, b


def test_keys_and_dtypes():
    assert USER_ONLY_KEYS == ("session_offset_tokens", "session_mode_main_category_tokens",
                              "current_matches_session_mode", "current_matches_session_mode_available",
                              "session_mode_share")
    m = np.array([[5, 7, 5, 7, 9, 1, 0, 0]])
    f = session_mode_features(m, m > 0, np.array([[1, 1, 1, 1, 2, 2, 0, 0]]))
    assert f[SESSION_MODE_KEY].dtype == np.int64 and f[SESSION_MODE_KEY].shape == (1, 8)
    assert all(f[k].dtype == np.float32 and f[k].shape == (1,) for k in (MATCH_KEY, MATCH_MASK_KEY, SHARE_KEY))
    assert MAIN_CATEGORY_VOCAB == 16 and MAIN_UNKNOWN == 1


def test_tie_rule_and_prefix_causality():
    m = np.array([[5, 7, 5, 7, 5, 7, 7, 0]])
    v = m > 0
    sp = np.array([[1, 1, 1, 1, 1, 1, 1, 0]])                      # one (current) session
    f = session_mode_features(m, v, sp)
    # prefix modes: [5] 5; [5,7] tie -> 7 (latest); [5,7,5] 5; tie -> 7; 5 (3:2); tie 3:3 -> 7; 7 (4:3)
    assert f[SESSION_MODE_KEY][0].tolist() == [5, 7, 5, 7, 5, 7, 7, 0]
    assert f[MATCH_KEY][0] == 1.0 and abs(float(f[SHARE_KEY][0]) - 4 / 7) < 1e-7
    past = session_mode_features(np.array([[5, 7, 5, 7, 9]]), np.ones((1, 5), bool), np.array([[1, 1, 1, 1, 2]]))
    assert past[SESSION_MODE_KEY][0].tolist() == [7, 7, 7, 7, 9]    # a past session: its FULL mode (tie -> latest)
    unk = session_mode_features(np.array([[1, 1, 4]]), np.ones((1, 3), bool), np.array([[1, 1, 1]]))
    assert unk[SESSION_MODE_KEY][0].tolist() == [1, 1, 4] and unk[MATCH_MASK_KEY][0] == 1.0


def test_random_synthetic_windows_equal_the_naive_reference():
    rng = np.random.default_rng(48)
    for _ in range(40):
        Bn, L = 64, 50
        lens = rng.integers(1, L + 1, size=Bn)
        v = np.arange(L)[None, :] < lens[:, None]
        m = np.where(v, rng.choice([1, 3, 4, 5, 6, 7], size=(Bn, L), p=[0.2, 0.3, 0.2, 0.15, 0.1, 0.05]), 0)
        starts = rng.random((Bn, L)) < 0.15
        sp = np.where(v, np.cumsum(starts, axis=1) + 1, 0)
        check_against_naive(m, v, sp, session_mode_features(m, v, sp))


def _real(hs, kind, pos):
    ds = hs.decision_set(kind)
    b = hs.batch(ds, pos)
    idx, valid = window_rows(ds.end_row[pos], ds.length[pos])
    sp = hs.index.session_id.astype(np.int64)[np.where(valid, idx, 0)]
    return b, b["main_category_tokens"].numpy(), valid, sp


def test_real_windows_equal_the_naive_reference(hist_train, hist_val):
    rng = np.random.default_rng(7)
    for hs, kind in ((hist_train, "loss"), (hist_val, "eval")):
        n = len(hs.decision_set(kind))
        for _ in range(6):
            pos = rng.choice(n, B, replace=False)
            b, m, v, sp = _real(hs, kind, pos)
            feats = {k: b[k].numpy() for k in (SESSION_MODE_KEY, MATCH_KEY, MATCH_MASK_KEY, SHARE_KEY)}
            check_against_naive(m, v, sp, feats)
            assert np.array_equal(feats[SESSION_MODE_KEY] > 0, v)


def test_served_batches_carry_the_keys(providers_user, providers_session):
    tu, eu = providers_user
    ts, _ = providers_session
    b = tu.batch(np.arange(128))
    for k in USER_ONLY_KEYS:
        assert k in b
    assert str(b[SESSION_MODE_KEY].dtype) == "torch.int64" and tuple(b[SESSION_MODE_KEY].shape) == (128, 50)
    assert all(str(b[k].dtype) == "torch.float32" and tuple(b[k].shape) == (128,) for k in (MATCH_KEY, MATCH_MASK_KEY,
                                                                                            SHARE_KEY))
    assert not any(k in ts.batch(np.arange(8)) for k in USER_ONLY_KEYS)
    assert eu.describe()["history"]["session_mode_features"]["main_category_vocab"] == 16


def test_distribution_on_real_validation(hist_val):
    ds = hist_val.decision_set("eval")
    pos = np.arange(len(ds))
    b, m, v, sp = _real(hist_val, "eval", pos)
    av = b[MATCH_MASK_KEY].numpy() > 0
    assert 0.3 < float(av.mean()) <= 1.0
    share = b[SHARE_KEY].numpy()[av]
    assert np.all((share > 0) & (share <= 1.0))
    unk = float(np.mean(b[SESSION_MODE_KEY].numpy()[v] == MAIN_UNKNOWN))
    print(f"DATA05_BUCKET0_VAL_MODE: n={len(ds)} available={av.mean():.4f} match_rate="
          f"{b[MATCH_KEY].numpy()[av].mean():.4f} share_median={np.median(share):.3f} unknown_positions={unk:.4f}")


def _noncausal(m, v, sp):
    """The deliberately wrong builder: CURRENT-session positions get the FULL current-session mode (a later event of
    the session changes an earlier position's value)."""
    f = session_mode_features(m, v, sp)
    L = m.shape[1]
    lens = v.sum(axis=1)
    cur = sp[np.arange(m.shape[0]), lens - 1]
    tok = f[SESSION_MODE_KEY].copy()
    for b in range(m.shape[0]):
        is_cur = v[b] & (sp[b] == cur[b])
        tok[b, is_cur] = tok[b, lens[b] - 1]                     # the whole current session's mode, everywhere
    f[SESSION_MODE_KEY] = tok
    return f


@nc("a session mode that lets a later event of the current session change an earlier position is caught")
def test_nc_future_event_inside_the_current_session_is_caught():
    m = np.array([[5, 7, 7, 0]])
    v = m > 0
    sp = np.array([[1, 1, 1, 0]])
    check_against_naive(m, v, sp, _noncausal(m, v, sp))           # position 0 would read 7 (from the future)
