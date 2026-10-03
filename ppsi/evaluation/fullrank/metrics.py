"""Per-decision credits, micro / user-macro aggregation in float64, coverage and the coverage identities.

Credits, from int64 ranks; rank -1 (an OOV row) gives 0 for every metric:
    HR@k = Recall@k = 1[r <= k]    RR@k = 1[r <= k] / r    NDCG@k = 1[r <= k] / log2(r + 1)
One target per decision: Recall@k == HR@k, reported once as hr@k; at k = 1 all three coincide.
Aggregation: micro = mean over the population's rows; user-macro = mean over users (eval_user_key) of each user's
mean. Every sum and mean is float64 (float32 accumulation of the macro is measurably wrong at scale). An empty
population is UNDEFINED(EMPTY_POPULATION) for every metric, never NaN and never 0.
Identities: micro E2E = coverage x micro RANKABLE; macro E2E = mean_u(cov_u x m_u). The invalid factorisation
mean_u(cov_u) x macro RANKABLE is computed only so that tests can show it differs.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .values import EMPTY_POPULATION, Undefined

K_GRID: tuple[int, ...] = (1, 5, 10, 20)
FAMILIES: tuple[str, ...] = ("mrr", "hr", "ndcg")
PRIMARY = ("macro", "mrr@20")
PROMINENT = (("macro", "ndcg@10"), ("macro", "hr@10"))
IDENTITY_TOL = 1e-9


def metric_names(ks: Iterable[int] = K_GRID) -> tuple[str, ...]:
    return tuple(f"{f}@{k}" for f in FAMILIES for k in ks)


def credits_from_ranks(ranks: np.ndarray, ks: Iterable[int] = K_GRID) -> dict[str, np.ndarray]:
    r = np.asarray(ranks, dtype=np.int64)
    rf = r.astype(np.float64)
    out = {}
    for k in ks:
        hit = (r >= 1) & (r <= k)
        out[f"hr@{k}"] = hit.astype(np.float64)
        out[f"mrr@{k}"] = np.where(hit, 1.0 / np.where(hit, rf, 1.0), 0.0)
        out[f"ndcg@{k}"] = np.where(hit, 1.0 / np.log2(np.where(hit, rf, 1.0) + 1.0), 0.0)
    return out


def _group(users: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keys, inv = np.unique(users, return_inverse=True)
    return keys, inv


def population_block(credits: dict[str, np.ndarray], mask: np.ndarray, users: np.ndarray) -> dict:
    idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    n = int(idx.size)
    if n == 0:
        und = Undefined(EMPTY_POPULATION)
        return {"n_decisions": 0, "n_users": 0,
                "micro": {k: und for k in credits}, "macro": {k: und for k in credits}}
    keys, inv = _group(users[idx])
    cnt = np.bincount(inv, minlength=keys.size).astype(np.float64)
    micro, macro = {}, {}
    for name, c in credits.items():
        sub = c[idx]
        micro[name] = float(sub.sum(dtype=np.float64) / n)
        per_user = np.bincount(inv, weights=sub, minlength=keys.size) / cnt
        macro[name] = float(per_user.mean(dtype=np.float64))
    return {"n_decisions": n, "n_users": int(keys.size), "micro": micro, "macro": macro}


def per_user(credit: np.ndarray, mask: np.ndarray, users: np.ndarray
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(sorted user keys, per-user credit sum, per-user row count) over the masked rows, float64."""
    idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    keys, inv = _group(users[idx])
    s = np.bincount(inv, weights=np.asarray(credit, dtype=np.float64)[idx], minlength=keys.size)
    c = np.bincount(inv, minlength=keys.size).astype(np.float64)
    return keys, s, c


def coverage(e2e_mask: np.ndarray, rankable_mask: np.ndarray):
    n_e = int(np.count_nonzero(e2e_mask))
    if n_e == 0:
        return Undefined(EMPTY_POPULATION)
    return float(np.count_nonzero(np.asarray(rankable_mask) & np.asarray(e2e_mask)) / n_e)


def identity_terms(credit: np.ndarray, e2e_mask: np.ndarray, rankable_mask: np.ndarray,
                   users: np.ndarray) -> dict:
    """The terms of both identities for one metric. Undefined when the E2E population is empty."""
    e2e_mask = np.asarray(e2e_mask, dtype=bool)
    rankable_mask = np.asarray(rankable_mask, dtype=bool) & e2e_mask
    if not e2e_mask.any():
        u = Undefined(EMPTY_POPULATION)
        return {"micro_e2e": u, "coverage": u, "micro_rankable": u, "macro_e2e": u,
                "macro_identity_rhs": u, "mean_cov": u, "macro_rankable": u, "invalid_factorization": u}
    c = np.asarray(credit, dtype=np.float64)
    ke, se, ne = per_user(c, e2e_mask, users)
    kr, sr, nr = per_user(c, rankable_mask, users)
    pos = np.searchsorted(ke, kr)
    s_r = np.zeros_like(se)
    n_r = np.zeros_like(ne)
    s_r[pos], n_r[pos] = sr, nr
    cov_u = n_r / ne
    m_u = np.divide(s_r, n_r, out=np.zeros_like(s_r), where=n_r > 0)      # term is 0 when R_u is empty
    n_e_rows, n_r_rows = float(ne.sum()), float(nr.sum())
    micro_e2e = float(se.sum() / n_e_rows)
    cov = n_r_rows / n_e_rows
    out = {"micro_e2e": micro_e2e, "coverage": cov,
           "macro_e2e": float((se / ne).mean()), "macro_identity_rhs": float((cov_u * m_u).mean()),
           "mean_cov": float(cov_u.mean())}
    if n_r_rows == 0:
        out["micro_rankable"] = Undefined(EMPTY_POPULATION)
        out["macro_rankable"] = Undefined(EMPTY_POPULATION)
        out["invalid_factorization"] = Undefined(EMPTY_POPULATION)
    else:
        out["micro_rankable"] = float(sr.sum() / n_r_rows)
        out["macro_rankable"] = float((sr / nr).mean())
        out["invalid_factorization"] = out["mean_cov"] * out["macro_rankable"]
    return out


def identity_errors(terms: dict) -> dict:
    """Absolute errors of the two valid identities; 0.0 when both sides are exactly 0 (an all-OOV population)."""
    if isinstance(terms["micro_e2e"], Undefined):
        return {"micro": terms["micro_e2e"], "macro": terms["micro_e2e"]}
    micro_rhs = (0.0 if isinstance(terms["micro_rankable"], Undefined)
                 else terms["coverage"] * terms["micro_rankable"])
    return {"micro": abs(terms["micro_e2e"] - micro_rhs),
            "macro": abs(terms["macro_e2e"] - terms["macro_identity_rhs"])}
