"""featstat_v1.fs_core - pure, unit-tested statistics kernels used by fs_run (FEATSTAT-01).

Every function here is deterministic and data-agnostic: value/count tables, exact linear quantiles (the
numpy "linear" / DuckDB quantile_cont rule), clipped moments, mid-ranks, exact functional-dependency violation
counts, Goodman-Kruskal lambda, Cramer's V (plain and Bergsma bias-corrected), union-find clusters and a strict
JSON sanitizer. No I/O, no dataset access.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------------------- population
POOL_LO, POOL_HI = 5, 100          # owner amendment section 1: 5 <= n_eligible_TRAIN_examples <= 100, inclusive


def stratum_of(n: np.ndarray) -> np.ndarray:
    """0 = A1 (TRAIN events, n = 0), 1 = B sparse (1-4), 2 = C MAIN (5-100 inclusive), 3 = D heavy (>100)."""
    n = np.asarray(n)
    out = np.zeros(n.shape, dtype=np.int8)
    out[(n >= 1) & (n <= 4)] = 1
    out[(n >= POOL_LO) & (n <= POOL_HI)] = 2
    out[n > POOL_HI] = 3
    return out


# ------------------------------------------------------------------------------------------- tables
def compact_uint(n_distinct: int):
    """Smallest unsigned dtype that can hold codes 0..n_distinct-1."""
    if n_distinct <= 1 << 8:
        return np.uint8
    if n_distinct <= 1 << 16:
        return np.uint16
    if n_distinct <= 1 << 32:
        return np.uint32
    return np.uint64


def local_codes(values: np.ndarray):
    """(sorted distinct values, compact codes) of a 1-D array (np.unique; -0.0 and +0.0 are one value)."""
    u, inv = np.unique(values, return_inverse=True)
    return u, inv.reshape(-1).astype(compact_uint(len(u)), copy=False)


def value_counts(values: np.ndarray):
    """(sorted distinct values, int64 counts)."""
    u, c = np.unique(values, return_counts=True)
    return u, c.astype(np.int64)


def merge_tables(vals: Sequence[np.ndarray], cnts: Sequence[np.ndarray]):
    """Merge several (sorted-or-not distinct values, counts) tables into one sorted table with summed counts."""
    vals = [v for v in vals if v.size]
    cnts = [c for c in cnts if c.size]
    if not vals:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.int64)
    v = np.concatenate(vals)
    c = np.concatenate(cnts).astype(np.int64, copy=False)
    order = np.argsort(v)
    v = v[order]
    c = c[order]
    starts = np.flatnonzero(np.r_[True, v[1:] != v[:-1]])
    return v[starts], np.add.reduceat(c, starts)


class TableAccumulator:
    """Accumulates (values, counts) pieces and merges them every `every` pieces to bound memory."""

    def __init__(self, every: int = 16):
        self.vals: list = []
        self.cnts: list = []
        self.every = every
        self._merged = False

    def add(self, v: np.ndarray, c: np.ndarray) -> None:
        if v.size:
            self.vals.append(v)
            self.cnts.append(_i32(c))
            self._merged = False
        if len(self.vals) >= self.every:
            v2, c2 = merge_tables(self.vals, self.cnts)
            self.vals, self.cnts = [v2], [_i32(c2)]

    def result(self):
        if self._merged and len(self.vals) == 1:
            return self.vals[0], self.cnts[0]
        v, c = merge_tables(self.vals, self.cnts)
        self.vals, self.cnts = [v], [_i32(c)]
        self._merged = True
        return self.vals[0], self.cnts[0]


def _i32(c: np.ndarray) -> np.ndarray:
    if c.size and int(c.max()) >= 2 ** 31:
        raise OverflowError("count exceeds int32")
    return c.astype(np.int32)


# ---------------------------------------------------------------------------------------- quantiles
def quantile_linear(vals: np.ndarray, cnts: np.ndarray, q: float) -> Optional[float]:
    """Exact linear-interpolation quantile of the multiset {vals[i] x cnts[i]} (numpy method='linear',
    DuckDB quantile_cont): h = (N-1) q, x[floor h] + (h - floor h) (x[floor h + 1] - x[floor h])."""
    cnts = np.asarray(cnts, dtype=np.int64)
    n = int(cnts.sum())
    if n == 0:
        return None
    h = (n - 1) * float(q)
    lo = int(math.floor(h))
    frac = h - lo
    cum = np.cumsum(cnts)
    i_lo = int(np.searchsorted(cum, lo, side="right"))
    x_lo = float(vals[i_lo])
    if frac == 0.0 or lo + 1 >= n:
        return x_lo
    i_hi = int(np.searchsorted(cum, lo + 1, side="right"))
    return x_lo + frac * (float(vals[i_hi]) - x_lo)


def quantiles_table(vals, cnts, qs=(0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)) -> dict:
    return {f"q{q:g}": quantile_linear(vals, cnts, q) for q in qs}


def clipped_moments(vals: np.ndarray, cnts: np.ndarray, lo: float, hi: float) -> dict:
    """Mean and population std (ddof 0) of clip(x, lo, hi) over the multiset, plus clip rates."""
    w = np.asarray(cnts, dtype=np.float64)
    n = float(w.sum())
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "clipped_low_rate": None, "clipped_high_rate": None}
    x = np.asarray(vals, dtype=np.float64)
    c = np.clip(x, lo, hi)
    mean = float(np.dot(w, c) / n)
    var = float(np.dot(w, (c - mean) ** 2) / n)
    return {"n": int(n), "mean": mean, "std": math.sqrt(max(var, 0.0)),
            "clipped_low_rate": float(w[x < lo].sum() / n), "clipped_high_rate": float(w[x > hi].sum() / n)}


def table_moments(vals: np.ndarray, cnts: np.ndarray) -> dict:
    w = np.asarray(cnts, dtype=np.float64)
    n = float(w.sum())
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "var": None}
    x = np.asarray(vals, dtype=np.float64)
    mean = float(np.dot(w, x) / n)
    var = float(np.dot(w, (x - mean) ** 2) / n)
    return {"n": int(n), "mean": mean, "var": var, "std": math.sqrt(max(var, 0.0))}


def midranks(cnts: np.ndarray) -> np.ndarray:
    """1-based average (mid) rank of every code, from per-code counts in code (= value) order."""
    c = np.asarray(cnts, dtype=np.float64)
    return np.cumsum(c) - (c - 1.0) / 2.0


# ---------------------------------------------------------------------------- functional dependency
def fd_from_dense(t: np.ndarray) -> tuple[int, int]:
    """Violations of X->Y and Y->X from a dense contingency table t[x, y]: N - sum_x max_y n_xy (and transposed).
    A violation count is the minimum number of rows that must change for the dependency to hold exactly."""
    n = int(t.sum())
    if n == 0:
        return 0, 0
    return n - int(t.max(axis=1).sum()), n - int(t.max(axis=0).sum())


def fd_from_pairs(px: np.ndarray, py: np.ndarray, cnt: np.ndarray) -> tuple[int, int]:
    """Same as fd_from_dense from sparse pair lists; px must be sorted ascending (pairs sorted by (x, y))."""
    cnt = np.asarray(cnt, dtype=np.int64)
    n = int(cnt.sum())
    if n == 0 or cnt.size == 0:
        return 0, 0
    st = np.flatnonzero(np.r_[True, px[1:] != px[:-1]])
    smx = int(np.maximum.reduceat(cnt, st).sum())
    o = np.argsort(py)
    pys = py[o]
    cs = cnt[o]
    st2 = np.flatnonzero(np.r_[True, pys[1:] != pys[:-1]])
    smy = int(np.maximum.reduceat(cs, st2).sum())
    return n - smx, n - smy


def gk_lambda(violations: int, n: int, max_marginal_y: int) -> Optional[float]:
    """Goodman-Kruskal lambda(Y|X) = (sum_x max_y n_xy - max_y n_y) / (N - max_y n_y); 1 = exact FD, 0 = X gives
    no reduction of prediction error over the mode of Y; None when Y is constant (undefined, trivially determined)."""
    den = n - max_marginal_y
    if den <= 0:
        return None
    return ((n - violations) - max_marginal_y) / den


def cramers_v(px, py, cnt, nx: np.ndarray, ny: np.ndarray) -> dict:
    """Cramer's V and the Bergsma (2013) bias-corrected V from sparse pair counts and the marginals (by code)."""
    cnt = np.asarray(cnt, dtype=np.float64)
    m = cnt > 0
    n = float(cnt.sum())
    r = int(np.count_nonzero(nx))
    k = int(np.count_nonzero(ny))
    if n <= 1 or r < 2 or k < 2:
        return {"chi2": None, "V": None, "V_bias_corrected": None, "r": r, "k": k}
    e = nx[px[m]].astype(np.float64) * ny[py[m]].astype(np.float64)
    s = float(np.sum(cnt[m] ** 2 / e))
    chi2 = n * (s - 1.0)
    v = math.sqrt(max(chi2, 0.0) / (n * (min(r, k) - 1)))
    phi2 = chi2 / n
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1)
    kc = k - (k - 1) ** 2 / (n - 1)
    den = min(kc - 1, rc - 1)
    vc = math.sqrt(phi2c / den) if den > 0 else None
    return {"chi2": chi2, "V": v, "V_bias_corrected": vc, "r": r, "k": k}


def fd_binary_any(b: np.ndarray, cy: np.ndarray, dy: int, counts_y: np.ndarray) -> tuple[int, int, np.ndarray]:
    """B (0/1 codes) vs Y (codes): returns (violations B->Y, violations Y->B, n1_y) using one weighted bincount."""
    n1 = np.rint(np.bincount(cy, weights=b.astype(np.float64), minlength=dy)).astype(np.int64)
    n0 = counts_y.astype(np.int64) - n1
    n = int(counts_y.sum())
    v_by = n - int(n0.max(initial=0) + n1.max(initial=0))
    v_yb = int(np.minimum(n0, n1).sum())
    return v_by, v_yb, n1


# ------------------------------------------------------------------------------------------ clusters
def clusters(names: Sequence[str], mat: np.ndarray, thr: float, *, absolute: bool = True) -> list[dict]:
    """Deterministic single-linkage clusters: an undirected edge i-j when |r_ij| >= thr (r finite); clusters are the
    connected components of size >= 2, members listed in the given (tensor) order, clusters ordered by their first
    member. Each cluster reports min/max |r| over its member pairs and whether it is a clique at thr."""
    n = len(names)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    a = np.abs(mat) if absolute else mat
    for i in range(n):
        for j in range(i + 1, n):
            v = a[i, j]
            if np.isfinite(v) and v >= thr:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
    groups: dict = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    out = []
    for root in sorted(groups):
        mem = groups[root]
        if len(mem) < 2:
            continue
        vals = [float(a[i, j]) for x, i in enumerate(mem) for j in mem[x + 1:]]
        fin = [v for v in vals if np.isfinite(v)]
        out.append({"members": [names[i] for i in mem], "size": len(mem),
                    "min_abs_r": min(fin) if fin else None, "max_abs_r": max(fin) if fin else None,
                    "clique_at_threshold": bool(fin and len(fin) == len(vals) and min(fin) >= thr)})
    return out


# ----------------------------------------------------------------------------------------------- json
def jsafe(o):
    """Strict-JSON sanitizer: numpy scalars/arrays to Python, NaN/inf to None, tuples to lists."""
    if isinstance(o, dict):
        return {str(k): jsafe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsafe(v) for v in o]
    if isinstance(o, np.ndarray):
        return [jsafe(v) for v in o.tolist()]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def corr_from_comoment(c: np.ndarray) -> np.ndarray:
    """Correlation matrix from a centred co-moment matrix; NaN where a variance is 0 (constant column)."""
    d = np.sqrt(np.clip(np.diag(c), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = c / np.outer(d, d)
    r[~np.isfinite(r)] = np.nan
    np.fill_diagonal(r, np.where(d > 0, 1.0, np.nan))
    return np.clip(r, -1.0, 1.0, where=np.isfinite(r), out=r)


def f32(x: float) -> float:
    return float(np.float32(x))


def iter_chunks(n: int, size: int) -> Iterable[slice]:
    for s in range(0, n, size):
        yield slice(s, min(n, s + size))
