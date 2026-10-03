"""Sampled ranking metrics against brute-force enumeration and their sampling schemes (CPU, synthetic scores)."""
import itertools
import math

import numpy as np
import pytest

from ppsi.benchmarks import sampled_metrics as sm


# ------------------------------------------------------------------------------------------------ sampled metrics
def _brute_expected(rank, n_cand, N, ks):
    """Enumerate every size-N subset of the other n_cand-1 items (r-1 better, rest worse)."""
    M = n_cand - 1
    items = [1] * (rank - 1) + [0] * (M - rank + 1)                     # 1 = better than the target
    hr = {k: 0.0 for k in ks}; nd = {k: 0.0 for k in ks}; mrr = 0.0; cnt = 0
    for sub in itertools.combinations(range(M), N):
        sr = 1 + sum(items[i] for i in sub)
        cnt += 1
        mrr += 1.0 / sr
        for k in ks:
            if sr <= k:
                hr[k] += 1; nd[k] += 1 / math.log2(sr + 1)
    return {**{f"HR@{k}": hr[k] / cnt for k in ks}, **{f"NDCG@{k}": nd[k] / cnt for k in ks}, "MRR": mrr / cnt}


def test_expected_matches_brute_force_enumeration():
    ks = (1, 2, 3, 5)
    for rank, ncand, N in [(1, 9, 3), (4, 9, 3), (9, 9, 3), (3, 7, 6), (2, 12, 4), (5, 8, 7)]:
        got = sm.expected_sampled_metrics([rank], [ncand], N, ks)
        want = _brute_expected(rank, ncand, N, ks)
        for k_, v in want.items():
            assert got[k_] == pytest.approx(v, abs=1e-12), (rank, ncand, N, k_)


def test_expected_averages_over_users_and_clips_n():
    r, nc = np.array([1, 4, 9]), np.array([9, 9, 9])
    got = sm.expected_sampled_metrics(r, nc, 100, (5,))                  # N > M -> all items used, rank unchanged
    assert got["HR@5"] == pytest.approx(np.mean([1, 1, 0]))
    want = np.mean([1 / math.log2(rr + 1) if rr <= 5 else 0 for rr in r])
    assert got["NDCG@5"] == pytest.approx(want)


def _fixture(U=6, K=30, seed=0):
    rng = np.random.default_rng(seed)
    scores = rng.normal(size=(U, K))
    target = rng.integers(0, K, U)
    seen = [rng.choice([c for c in range(K) if c != target[u]], 5, replace=False) for u in range(U)]
    return scores, target, seen


def test_full_ranks_vs_sort_with_seen_filter_and_tie_rule():
    scores, target, seen = _fixture()
    scores[0, :] = 1.0                                                   # all tied: rank = 1 + #unseen cols below target
    r, nc = sm.full_ranks(scores, target, seen)
    for u in range(len(target)):
        cand = [c for c in range(scores.shape[1]) if c == target[u] or c not in set(seen[u])]
        order = sorted(cand, key=lambda c: (-scores[u, c], c))
        assert r[u] == order.index(int(target[u])) + 1 and nc[u] == len(cand)


def test_sampled_uniform_matches_brute_force_and_is_seeded():
    scores, target, seen = _fixture()
    a = sm.evaluate(scores, target, seen, scheme="uniform", n_neg=7, ks=(3, 5), seed=11)
    b = sm.evaluate(scores, target, seen, scheme="uniform", n_neg=7, ks=(3, 5), seed=11)
    c = sm.evaluate(scores, target, seen, scheme="uniform", n_neg=7, ks=(3, 5), seed=12)
    assert a == b and a["label"] == sm.LABEL and a["metrics"] != c["metrics"]
    neg = sm.draw_uniform(scores.shape[1], target, seen, 7, 11)
    ranks = []
    for u in range(len(target)):
        assert len(set(neg[u])) == 7 and target[u] not in neg[u] and not (set(neg[u]) & set(seen[u]))
        ranks.append(1 + sum(1 for n in neg[u] if (scores[u, n], -n) > (scores[u, target[u]], -target[u])))
    want = sm.metrics_from_ranks(np.array(ranks), (3, 5))
    assert a["metrics"] == want


def test_uniform_mean_converges_to_expected_value():
    rng = np.random.default_rng(3)
    U, K, N = 4000, 60, 9
    scores = rng.normal(size=(U, K)); target = rng.integers(0, K, U)
    full, nc = sm.full_ranks(scores, target, None)
    exp = sm.expected_sampled_metrics(full, nc, N, (5, 10))
    emp = sm.evaluate(scores, target, None, scheme="uniform", n_neg=N, ks=(5, 10), seed=1)["metrics"]
    for k in exp:
        assert emp[k] == pytest.approx(exp[k], abs=0.02)


def test_popularity_scheme_excludes_seen_target_and_follows_counts():
    scores, target, seen = _fixture(U=300, K=40)
    counts = np.arange(1, 41, dtype=float) ** 2
    neg = sm.draw_popularity(counts, target, seen, 5, seed=5)
    assert all(len(set(n)) == 5 and t not in n and not (set(n) & set(s)) for n, t, s in zip(neg, target, seen))
    assert neg.mean() > 25                                               # skewed to high-count (high-index) items
    out = sm.evaluate(scores, target, seen, scheme="popularity", n_neg=5, ks=(3,), seed=5, counts=counts)
    assert out["label"] == sm.LABEL


def test_fixed_scheme_replaces_seen_and_keeps_rest():
    catalog = np.array([10, 20, 30, 40, 50, 60, 70, 80])
    fixed = sm.fixed_list_columns([20, 40, 999, 60], catalog)            # 999 not in catalogue -> -1
    assert fixed.tolist() == [1, 3, -1, 5]
    scores = np.random.default_rng(0).normal(size=(3, 8)); target = np.array([0, 2, 7])
    seen = [np.array([1]), np.array([], dtype=int), np.array([3, 5])]
    neg = sm.draw_fixed(fixed, 8, target, seen, seed=9)
    assert neg.shape == (3, 4)
    assert neg[1].tolist()[:2] == [1, 3] and neg[1][3] == 5              # unseen fixed ids stay
    assert neg[0][0] != 1 and neg[0][0] not in {1}                       # seen id replaced by an unseen one
    assert all(int(x) not in set(seen[2].tolist()) for x in neg[2])
    assert (neg >= 0).all() and (neg < 8).all()
    out = sm.evaluate(scores, target, seen, scheme="fixed", ks=(1, 2), seed=9, fixed_cols=fixed)
    assert out["n_neg"] == 4 and out["label"] == sm.LABEL


def test_mbht_fixed_lists_have_100_ids():
    assert len(sm.MBHT_FIXED_100["tmall_beh"]) == 100 and len(sm.MBHT_FIXED_100["ijcai_beh"]) == 100
    assert len(set(sm.MBHT_FIXED_100["tmall_beh"])) == 100


def test_label_on_every_output():
    out = sm.evaluate_expected([2, 3], [10, 10], n_neg=4, ks=(2,))
    assert out["label"] == "SECONDARY / for comparison with sampled-metric papers only"
