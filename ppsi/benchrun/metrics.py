"""Full-catalogue ranking with seen-item filtering, and the benchmark metrics.

Protocol of the reference benchmark ({full_catalogue: True, filter_viewed: True}): every catalogue item is a
candidate except the user's seen items (the items of the input prefix); no sampled negatives. Rank of the target
(1-based) = 1 + #(unfiltered items with a strictly higher score) + #(unfiltered items with an equal score and a smaller
external item id), i.e. descending score, ascending external id (the class index order IS the ascending external id
order). A target that is itself a seen item can never be recommended: rank 0 (= miss, still counted in the
denominator, as the reference counts a holdout user whose item is absent from the recommendations).
Per user (one target): HR@k = [rank <= k]; NDCG@k = 1 / log2(rank + 1) if rank <= k (ideal DCG = 1 with one target);
MRR@20 = 1 / rank if rank <= 20. Means over users (float64).
"""
from __future__ import annotations

import math

import numpy as np

from .common import GROUPS, METRICS, BenchRefused


def ranks_from_scores(scores, targets, seen_rows, seen_items) -> np.ndarray:
    """scores [B, K] float32 torch tensor (consumed: seen cells are overwritten), targets [B] -> ranks int64 [B]."""
    import torch
    if scores.ndim != 2:
        raise BenchRefused("scores must be [B, K]")
    if not bool(torch.isfinite(scores).all()):
        raise BenchRefused("non-finite scores")
    B, K = scores.shape
    t = torch.as_tensor(np.asarray(targets, dtype=np.int64))
    cold = t < 0                                   # a holdout item outside the catalogue: a MISS, rank 0
    t = torch.where(cold, torch.zeros_like(t), t)
    if B and int(t.max()) >= K:
        raise BenchRefused("target outside the catalogue")
    if len(seen_rows):
        scores[torch.as_tensor(np.asarray(seen_rows, dtype=np.int64)),
               torch.as_tensor(np.asarray(seen_items, dtype=np.int64))] = float("-inf")
    st = scores.gather(1, t.view(-1, 1))
    reach = torch.isfinite(st.view(-1))
    gt = (scores > st).sum(1)
    ar = torch.arange(K).view(1, -1)
    eq_before = ((scores == st) & (ar < t.view(-1, 1))).sum(1)
    r = (1 + gt + eq_before).to(torch.int64)
    r = torch.where(reach & ~cold, r, torch.zeros_like(r))
    return r.numpy().astype(np.int64)


def metrics_from_ranks(ranks) -> dict:
    r = np.asarray(ranks, dtype=np.int64)
    n = int(r.size)
    out = {"n_users": n, "n_unreachable": int((r == 0).sum())}
    if n == 0:
        return dict(out, **{m: None for m in METRICS})
    rf = r.astype(np.float64)
    for k in (10, 20):
        hit = (r >= 1) & (r <= k)
        out[f"hr@{k}"] = float(hit.mean())
        out[f"ndcg@{k}"] = float(np.where(hit, 1.0 / np.log2(np.where(hit, rf, 1.0) + 1.0), 0.0).mean())
    hit20 = (r >= 1) & (r <= 20)
    out["mrr@20"] = float(np.where(hit20, 1.0 / np.where(hit20, rf, 1.0), 0.0).mean())
    return out


def grouped_metrics(ranks, users, band) -> dict:
    """{ALL, BAND (the 15 % pretraining users), OTHERS (the 85 %)} -> metrics."""
    r = np.asarray(ranks, dtype=np.int64)
    inb = np.asarray(band, dtype=bool)[np.asarray(users, dtype=np.int64)]
    out = {"ALL": metrics_from_ranks(r), "BAND": metrics_from_ranks(r[inb]), "OTHERS": metrics_from_ranks(r[~inb])}
    assert tuple(out) == GROUPS
    return out


def per_user_ndcg(rank: int, k: int) -> float:
    return 1.0 / math.log2(rank + 1) if 1 <= rank <= k else 0.0
