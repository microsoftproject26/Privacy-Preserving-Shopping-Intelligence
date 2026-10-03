"""Metric tests: brute-force checks of the rank rule (ties, seen-item filter) and the metric definitions."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from ppsi.benchrun import metrics as M


def _ranks_bruteforce(scores, target, seen):
    B, K = scores.shape
    out = []
    for b in range(B):
        t = int(target[b])
        if t in seen[b]:
            out.append(0)
            continue
        cand = [j for j in range(K) if j not in seen[b]]
        order = sorted(cand, key=lambda j: (-float(scores[b, j]), j))
        out.append(order.index(t) + 1)
    return np.asarray(out)


def _fixture(seed=0, B=60, K=40):
    g = torch.Generator().manual_seed(seed)
    scores = torch.randn(B, K, generator=g)
    scores[:5, :6] = 1.0                                 # ties: smaller class id first
    rng = np.random.default_rng(seed)
    target = rng.integers(0, K, size=B)
    seen = [set(rng.choice(K, size=int(rng.integers(0, 12)), replace=False).tolist()) for _ in range(B)]
    rows = np.repeat(np.arange(B), [len(s) for s in seen])
    items = np.concatenate([sorted(s) for s in seen]) if rows.size else np.zeros(0, dtype=np.int64)
    return scores, target, seen, rows, items


def test_ranks_match_bruteforce_with_ties_and_seen_filter():
    scores, target, seen, rows, items = _fixture()
    got = M.ranks_from_scores(scores.clone(), target, rows, items)
    assert np.array_equal(got, _ranks_bruteforce(scores.numpy(), target, seen))


def test_unreachable_and_cold_targets_are_rank_zero():
    s = torch.randn(3, 10)
    got = M.ranks_from_scores(s.clone(), [2, 4, 5], [0, 1], [2, 4])
    assert got[0] == 0 and got[1] == 0 and got[2] >= 1
    assert M.ranks_from_scores(torch.randn(2, 5), [-1, 3], [], [])[0] == 0          # a cold holdout item: a miss


def test_metrics_against_a_direct_definition():
    scores, target, _seen, rows, items = _fixture(seed=3, B=80, K=60)
    ranks = M.ranks_from_scores(scores.clone(), target, rows, items)
    mine = M.metrics_from_ranks(ranks)
    for k in (10, 20):
        hits = [(1 <= r <= k) for r in ranks]
        assert mine[f"hr@{k}"] == pytest.approx(np.mean(hits), abs=1e-12)
        assert mine[f"ndcg@{k}"] == pytest.approx(np.mean([1 / np.log2(r + 1) if h else 0.0
                                                           for r, h in zip(ranks, hits, strict=True)]), abs=1e-12)


def test_mrr_at_20_definition():
    r = np.asarray([1, 2, 20, 21, 0, 5])
    m = M.metrics_from_ranks(r)
    assert m["mrr@20"] == pytest.approx((1 + 0.5 + 0.05 + 0 + 0 + 0.2) / 6)
    assert m["n_unreachable"] == 1 and m["hr@10"] == pytest.approx(3 / 6)
    assert M.metrics_from_ranks(np.zeros(0, dtype=np.int64))["ndcg@10"] is None


def test_grouped_metrics_partition():
    r = np.asarray([1, 3, 0, 12, 2, 40])
    users = np.arange(6)
    band = np.asarray([True, False, True, False, False, True])
    g = M.grouped_metrics(r, users, band)
    assert g["BAND"]["n_users"] == 3 and g["OTHERS"]["n_users"] == 3 and g["ALL"]["n_users"] == 6
    assert g["ALL"]["ndcg@10"] == pytest.approx((3 * g["BAND"]["ndcg@10"] + 3 * g["OTHERS"]["ndcg@10"]) / 6)


def test_nc_no_seen_filter_changes_the_result():
    """Negative control: dropping the seen-item filter must change the metric (a filter that does nothing is caught)."""
    scores, target, seen, rows, items = _fixture(seed=5)
    scores[np.arange(len(seen)), [max(s) if s else 0 for s in seen]] += 5.0     # a seen item scored highest
    with_f = M.metrics_from_ranks(M.ranks_from_scores(scores.clone(), target, rows, items))
    no_f = M.metrics_from_ranks(M.ranks_from_scores(scores.clone(), target, np.zeros(0, dtype=np.int64),
                                                    np.zeros(0, dtype=np.int64)))
    assert with_f["ndcg@10"] != no_f["ndcg@10"]


def test_tie_break_prefers_the_smaller_id():
    s = torch.zeros(1, 5)
    assert M.ranks_from_scores(s.clone(), [3], [], [])[0] == 4      # ties: smaller ids first -> id 3 is 4th
    assert M.ranks_from_scores(s.clone(), [0], [], [])[0] == 1


def test_nonfinite_scores_refused():
    s = torch.zeros(2, 4)
    s[0, 1] = float("nan")
    with pytest.raises(Exception, match="non-finite"):
        M.ranks_from_scores(s, [0, 1], [], [])
