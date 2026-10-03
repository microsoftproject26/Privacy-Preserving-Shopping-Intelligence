"""Sampled ranking metrics. EVERY output carries LABEL.

Inputs are in item-COLUMN space: scores [U, K] (one row of full-catalogue scores per user), target [U] (column of the
held-out item), seen [U] (list of arrays of columns the user has already consumed; candidates exclude them but never
the target). Ties: a candidate beats the target when score is higher, or equal with a smaller column index (the
ascending-ID tie rule), so ranks are deterministic.

Schemes
  uniform     N negatives drawn uniformly without replacement from the unseen non-target items (seeded per user).
  popularity  N negatives drawn without replacement with probability proportional to counts**power over the same pool.
  fixed       MBHT: a fixed list of 100 popular item ids (the lists hard-coded in MBHT mbht.py customized_sort_predict
              for tmall_beh / ijcai_beh are in MBHT_FIXED_100); any id the user has seen is replaced by a random
              unseen item (MBHT: random.randint(1, n_items); here a uniform draw from the catalogue, so the item has
              a score; the replacement is seeded). MBHT does not exclude the target from the list; neither do we.
  expected    no sampling: the exact expectation over uniform negatives from the FULL rank (Krichene & Rendle 2020,
              KDD, 'On sampled metrics for item recommendation'): with M = candidates - 1 other items, r - 1 of them
              better than the target and N negatives, the number X of better negatives is hypergeometric(M, r-1, N),
              HR@k = P(X <= k-1), NDCG@k = sum_{j<k} P(X=j) / log2(j+2), MRR = sum_j P(X=j) / (j+1).
"""
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Sequence

import numpy as np

LABEL = "SECONDARY / for comparison with sampled-metric papers only"

# MBHT repo, recbole/model/sequential_recommender/mbht.py customized_sort_predict (100 ids each, RecBole item ids).
MBHT_FIXED_100 = {
    "ijcai_beh": [73, 3050, 22557, 5950, 4391, 6845, 1800, 2261, 13801, 2953, 4164, 32090, 3333, 44733, 7380, 790, 1845, 2886, 2366, 21161, 6512, 1689, 337, 3963, 3108, 715, 169, 2558, 6623, 888, 6708, 3585, 501, 308, 9884, 1405, 5494, 6609, 7433, 25101, 3580, 145, 3462, 5340, 1131, 6681, 7776, 8678, 52852, 19229, 4160, 33753, 4356, 920, 15312, 43106, 16669, 1850, 2855, 43807, 15, 8719, 89, 3220, 36, 2442, 9299, 8189, 701, 300, 526, 4564, 516, 1184, 178, 2834, 16455, 9392, 22037, 344, 15879, 3374, 2984, 3581, 11479, 6927, 779, 5298, 10195, 39739, 663, 9137, 24722, 7004, 7412, 89534, 2670, 100, 6112, 1355],
    "tmall_beh": [2544, 7010, 4193, 32270, 22086, 7768, 647, 7968, 26512, 4575, 63971, 2121, 7857, 5134, 416, 1858, 34198, 2146, 778, 12583, 13899, 7652, 4552, 14410, 1272, 21417, 2985, 5358, 36621, 10337, 13065, 1235, 3410, 14180, 5083, 5089, 4240, 10863, 3397, 4818, 58422, 8353, 14315, 14465, 30129, 4752, 5853, 1312, 3890, 6409, 7664, 1025, 16740, 14185, 4535, 670, 17071, 12579, 1469, 853, 775, 12039, 3853, 4307, 5729, 271, 13319, 1548, 449, 2771, 4727, 903, 594, 28184, 126, 27306, 20603, 40630, 907, 5118, 3472, 7012, 10055, 1363, 9086, 5806, 8204, 41711, 10174, 12900, 4435, 35877, 8679, 10369, 2865, 14830, 175, 4434, 11444, 701],
}


# ------------------------------------------------------------------------------------------------ ranks / metrics
def _beats(neg_scores, neg_cols, t_score, t_col):
    return (neg_scores > t_score) | ((neg_scores == t_score) & (neg_cols < t_col))


def full_ranks(scores: np.ndarray, target: np.ndarray, seen: Sequence | None = None):
    """1-based rank of the target among its candidates (seen items other than the target removed) and the number of
    candidates (including the target) per user."""
    scores = np.asarray(scores)
    U, K = scores.shape
    cols = np.arange(K)
    ranks = np.empty(U, np.int64)
    ncand = np.empty(U, np.int64)
    for u in range(U):
        t = int(target[u])
        ok = np.ones(K, bool)
        if seen is not None and len(seen[u]):
            ok[np.asarray(seen[u], np.int64)] = False
        ok[t] = True
        better = _beats(scores[u], cols, scores[u, t], t) & ok
        better[t] = False
        ranks[u] = 1 + int(better.sum())
        ncand[u] = int(ok.sum())
    return ranks, ncand


def metrics_from_ranks(ranks: np.ndarray, ks: Iterable[int]) -> dict[str, float]:
    r = np.asarray(ranks, np.float64)
    out = {}
    for k in ks:
        hit = r <= k
        out[f"HR@{k}"] = float(hit.mean())
        out[f"NDCG@{k}"] = float(np.where(hit, 1.0 / np.log2(r + 1.0), 0.0).mean())
    out["MRR"] = float((1.0 / r).mean())
    return out


# ------------------------------------------------------------------------------------------------ expected (exact)
def _lgamma_table(n: int) -> np.ndarray:
    return np.array([0.0] + [math.lgamma(x) for x in range(1, n + 2)], dtype=np.float64)   # lg[x] = log Gamma(x); lg[0] unused


def hypergeom_pmf(M, s, N, j, lg):
    """P(X=j), X ~ Hypergeometric(population M, successes s, draws N); arrays broadcast; invalid j -> 0."""
    M, s, N, j = np.broadcast_arrays(*(np.asarray(a, np.int64) for a in (M, s, N, j)))
    valid = (j >= 0) & (j <= s) & (j <= N) & (N - j <= M - s)
    jj = np.where(valid, j, 0)
    ss = np.where(valid, s, 0)
    nn = np.where(valid, N, 0)
    mm = np.where(valid, M, 0)
    # a = max(0, ...) keeps indices valid for masked entries
    def lc(n, k):
        n = np.maximum(n, 0)
        k = np.clip(k, 0, n)
        return lg[n + 1] - lg[k + 1] - lg[n - k + 1]
    logp = lc(ss, jj) + lc(mm - ss, nn - jj) - lc(mm, nn)
    return np.where(valid, np.exp(logp), 0.0)


def expected_sampled_metrics(full_rank, n_candidates, n_neg: int, ks: Iterable[int]) -> dict[str, float]:
    """Exact expected sampled HR / NDCG @k and MRR for uniform negatives, from FULL ranks (Krichene & Rendle 2020).
    n_candidates may be a scalar or per-user (candidates INCLUDING the target)."""
    r = np.asarray(full_rank, np.int64)
    nc = np.broadcast_to(np.asarray(n_candidates, np.int64), r.shape)
    M = nc - 1
    s = r - 1
    if (s > M).any() or (r < 1).any():
        raise ValueError("rank outside 1..n_candidates")
    N = np.minimum(int(n_neg), M)
    lg = _lgamma_table(int(M.max()) + 2)
    ks = list(ks)
    jmax = int(min(max(ks + [N.max() + 1]), N.max() + 1))                # X ranges over 0..N
    J = np.arange(jmax)[None, :]
    P = hypergeom_pmf(M[:, None], s[:, None], N[:, None], J, lg)        # [U, jmax]
    out = {}
    for k in ks:
        kk = min(k, jmax)
        out[f"HR@{k}"] = float(P[:, :kk].sum(1).mean())
        out[f"NDCG@{k}"] = float((P[:, :kk] / np.log2(J[:, :kk] + 2.0)).sum(1).mean())
    out["MRR"] = float((P / (J + 1.0)).sum(1).mean())
    return out


# ------------------------------------------------------------------------------------------------ negative drawing
def _rng(seed: int, u: int):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(seed), int(u)])))


def _pool(K, t, seen_u):
    ok = np.ones(K, bool)
    if seen_u is not None and len(seen_u):
        ok[np.asarray(seen_u, np.int64)] = False
    ok[t] = False
    return np.flatnonzero(ok)


def draw_uniform(K: int, target, seen, n_neg: int, seed: int) -> np.ndarray:
    out = np.empty((len(target), n_neg), np.int64)
    for u, t in enumerate(target):
        pool = _pool(K, int(t), None if seen is None else seen[u])
        if pool.size < n_neg:
            raise ValueError(f"user {u}: only {pool.size} candidate negatives")
        out[u] = _rng(seed, u).choice(pool, n_neg, replace=False)
    return out


def draw_popularity(counts: np.ndarray, target, seen, n_neg: int, seed: int, power: float = 1.0) -> np.ndarray:
    counts = np.asarray(counts, np.float64)
    K = counts.size
    out = np.empty((len(target), n_neg), np.int64)
    for u, t in enumerate(target):
        pool = _pool(K, int(t), None if seen is None else seen[u])
        p = counts[pool] ** power
        if (p > 0).sum() < n_neg:
            raise ValueError(f"user {u}: fewer than {n_neg} items with positive popularity")
        out[u] = _rng(seed, u).choice(pool, n_neg, replace=False, p=p / p.sum())
    return out


def draw_fixed(fixed_cols: np.ndarray, K: int, target, seen, seed: int) -> np.ndarray:
    """MBHT scheme in column space; fixed_cols = the fixed list mapped to columns (-1 = not in the catalogue)."""
    fixed_cols = np.asarray(fixed_cols, np.int64)
    out = np.empty((len(target), fixed_cols.size), np.int64)
    for u in range(len(target)):
        rng = _rng(seed, u)
        sset = set() if seen is None else {int(x) for x in seen[u]}
        row = fixed_cols.copy()
        for n in range(row.size):
            if row[n] < 0 or int(row[n]) in sset:
                while True:
                    c = int(rng.integers(0, K))
                    if c not in sset:
                        break
                row[n] = c
        out[u] = row
    return out


def fixed_list_columns(fixed_ids: Sequence[int], catalog: np.ndarray) -> np.ndarray:
    catalog = np.asarray(catalog)
    idx = np.searchsorted(catalog, fixed_ids)
    ok = (idx < catalog.size) & (catalog[np.minimum(idx, catalog.size - 1)] == np.asarray(fixed_ids))
    return np.where(ok, idx, -1).astype(np.int64)


def sampled_ranks(scores: np.ndarray, target: np.ndarray, negatives: np.ndarray) -> np.ndarray:
    """1-based rank of the target among itself + its negatives (duplicates among negatives count as listed)."""
    U = len(target)
    out = np.empty(U, np.int64)
    for u in range(U):
        t = int(target[u])
        nc = negatives[u]
        out[u] = 1 + int(_beats(scores[u, nc], nc, scores[u, t], t).sum())
    return out


# ------------------------------------------------------------------------------------------------ entry points
def evaluate(scores, target, seen=None, *, scheme: str, n_neg: int = 100, ks=(5, 10), seed: int = 2026,
             counts=None, power: float = 1.0, fixed_cols=None) -> dict:
    scores = np.asarray(scores)
    K = scores.shape[1]
    target = np.asarray(target, np.int64)
    if scheme == "uniform":
        neg = draw_uniform(K, target, seen, n_neg, seed)
    elif scheme == "popularity":
        neg = draw_popularity(counts, target, seen, n_neg, seed, power)
    elif scheme == "fixed":
        neg = draw_fixed(fixed_cols, K, target, seen, seed)
    else:
        raise ValueError(f"unknown scheme {scheme}")
    m = metrics_from_ranks(sampled_ranks(scores, target, neg), ks)
    return {"label": LABEL, "scheme": scheme, "n_neg": int(neg.shape[1]), "seed": seed, "n_users": len(target),
            "metrics": m}


def evaluate_expected(full_rank, n_candidates, *, n_neg: int = 100, ks=(5, 10)) -> dict:
    return {"label": LABEL, "scheme": "expected_uniform_hypergeometric", "n_neg": int(n_neg),
            "n_users": len(np.atleast_1d(full_rank)),
            "metrics": expected_sampled_metrics(full_rank, n_candidates, n_neg, ks)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=LABEL)
    ap.add_argument("--ranks", required=True, help=".npz with 'rank' (1-based full rank) and 'n_candidates'")
    ap.add_argument("--n-neg", type=int, default=100)
    ap.add_argument("--ks", type=int, nargs="*", default=[5, 10])
    a = ap.parse_args(argv)
    z = np.load(a.ranks)
    print(json.dumps(evaluate_expected(z["rank"], z["n_candidates"], n_neg=a.n_neg, ks=a.ks), indent=1))


if __name__ == "__main__":
    main()
