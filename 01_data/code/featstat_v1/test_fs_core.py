"""Synthetic unit tests for featstat_v1.fs_core (brute force vs kernel). Run: python -B test_fs_core.py
Exit 0 and a JSON line {"tests_passed": n, "tests_failed": 0} on success. No dataset access."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from featstat_v1 import fs_core as F  # noqa: E402

RESULTS = []


def case(fn):
    try:
        fn()
        RESULTS.append((fn.__name__, True, ""))
    except Exception as e:  # noqa: BLE001
        RESULTS.append((fn.__name__, False, repr(e)))
    return fn


def brute_fd(x, y):
    n = len(x)
    s = 0
    for xv in np.unique(x):
        _, c = np.unique(y[x == xv], return_counts=True)
        s += c.max()
    return n - s


@case
def t_strata_boundaries():
    n = np.array([0, 1, 4, 5, 100, 101, 11628])
    assert F.stratum_of(n).tolist() == [0, 1, 1, 2, 2, 3, 3]


@case
def t_quantile_matches_numpy():
    rng = np.random.default_rng(1)
    x = np.round(rng.normal(size=5001), 2).astype(np.float32)
    v, c = F.value_counts(x)
    for q in (0.0, 0.001, 0.25, 0.5, 0.999, 1.0):
        a = F.quantile_linear(v, c, q)
        b = float(np.percentile(x.astype(np.float64), q * 100, method="linear"))
        assert abs(a - b) <= 1e-12 * max(1.0, abs(b)), (q, a, b)


@case
def t_merge_tables_equals_global_unique():
    rng = np.random.default_rng(2)
    parts = [rng.integers(0, 50, size=rng.integers(1, 300)).astype(np.float32) for _ in range(9)]
    acc = F.TableAccumulator(every=3)
    for p in parts:
        acc.add(*F.value_counts(p))
    v, c = acc.result()
    v2, c2 = F.value_counts(np.concatenate(parts))
    assert np.array_equal(v, v2) and np.array_equal(c, c2)


@case
def t_signed_zero_is_one_value():
    v, c = F.value_counts(np.array([0.0, -0.0, 1.0], dtype=np.float32))
    assert len(v) == 2 and c.tolist() == [2, 1]


@case
def t_clipped_moments():
    x = np.array([0, 1, 2, 3, 10, 10], dtype=np.float32)
    v, c = F.value_counts(x)
    m = F.clipped_moments(v, c, 1.0, 3.0)
    cl = np.clip(x.astype(np.float64), 1, 3)
    assert abs(m["mean"] - cl.mean()) < 1e-15 and abs(m["std"] - cl.std()) < 1e-15
    assert abs(m["clipped_low_rate"] - 1 / 6) < 1e-15 and abs(m["clipped_high_rate"] - 2 / 6) < 1e-15


@case
def t_midranks_spearman():
    rng = np.random.default_rng(3)
    x = rng.integers(0, 7, size=2000)
    y = x + rng.integers(0, 3, size=2000)
    ux, cx = np.unique(x, return_inverse=True)
    uy, cy = np.unique(y, return_inverse=True)
    rx = F.midranks(np.bincount(cx))[cx]
    ry = F.midranks(np.bincount(cy))[cy]
    rho = np.corrcoef(rx, ry)[0, 1]
    # brute force average ranks
    def avg_rank(a):
        order = np.argsort(a, kind="stable")
        r = np.empty(len(a))
        r[order] = np.arange(1, len(a) + 1)
        for v in np.unique(a):
            r[a == v] = r[a == v].mean()
        return r
    rho2 = np.corrcoef(avg_rank(x), avg_rank(y))[0, 1]
    assert abs(rho - rho2) < 1e-12


@case
def t_fd_dense_and_pairs_match_brute():
    rng = np.random.default_rng(4)
    x = rng.integers(0, 6, size=3000)
    y = (x * 2 + (rng.random(3000) < 0.05)).astype(np.int64)  # nearly determined
    y = np.unique(y, return_inverse=True)[1]
    dx, dy = x.max() + 1, y.max() + 1
    t = np.bincount(x * dy + y, minlength=dx * dy).reshape(dx, dy)
    vxy, vyx = F.fd_from_dense(t)
    assert vxy == brute_fd(x, y) and vyx == brute_fd(y, x)
    key = x.astype(np.int64) * dy + y
    uk, cnt = np.unique(key, return_counts=True)
    p = F.fd_from_pairs(uk // dy, uk % dy, cnt)
    assert p == (vxy, vyx)


@case
def t_fd_exact_duplicate_zero():
    x = np.arange(100) % 7
    y = (x + 3) % 7
    t = np.bincount(x * 7 + y, minlength=49).reshape(7, 7)
    assert F.fd_from_dense(t) == (0, 0)


@case
def t_fd_binary_any():
    rng = np.random.default_rng(5)
    b = (rng.random(4000) < 0.3).astype(np.int64)
    y = rng.integers(0, 9, size=4000)
    y[b == 1] = 8  # b=1 -> y=8 (so b->y has violations only in b=0 group)
    vby, vyb, _ = F.fd_binary_any(b, y, 9, np.bincount(y, minlength=9))
    assert vby == brute_fd(b, y) and vyb == brute_fd(y, b)


@case
def t_gk_lambda():
    assert F.gk_lambda(0, 100, 60) == 1.0
    assert F.gk_lambda(40, 100, 60) == 0.0
    assert F.gk_lambda(0, 100, 100) is None


@case
def t_cramers_v_perfect_and_independent():
    x = np.repeat(np.arange(4), 250)
    y = x.copy()
    key = x * 4 + y
    uk, cnt = np.unique(key, return_counts=True)
    r = F.cramers_v(uk // 4, uk % 4, cnt, np.bincount(x), np.bincount(y))
    assert abs(r["V"] - 1.0) < 1e-12
    x2 = np.tile(np.arange(4), 400)
    y2 = np.repeat(np.arange(4), 400)
    uk, cnt = np.unique(x2 * 4 + y2, return_counts=True)
    r2 = F.cramers_v(uk // 4, uk % 4, cnt, np.bincount(x2), np.bincount(y2))
    assert abs(r2["V"]) < 1e-12 and r2["V_bias_corrected"] == 0.0


@case
def t_clusters_deterministic():
    names = ["a", "b", "c", "d"]
    m = np.array([[1, .96, .1, .2], [.96, 1, .97, 0], [.1, .97, 1, 0], [.2, 0, 0, 1]])
    cl = F.clusters(names, m, 0.95)
    assert cl == [{"members": ["a", "b", "c"], "size": 3, "min_abs_r": 0.1, "max_abs_r": 0.97,
                   "clique_at_threshold": False}]


@case
def t_corr_from_comoment_constant_column():
    x = np.c_[np.arange(10.0), np.ones(10), -np.arange(10.0)]
    xc = x - x.mean(0)
    r = F.corr_from_comoment(xc.T @ xc)
    assert abs(r[0, 2] + 1) < 1e-12 and np.isnan(r[0, 1]) and np.isnan(r[1, 1])


@case
def t_jsafe_strict():
    s = json.dumps(F.jsafe({"a": np.float32(1.5), "b": float("nan"), "c": np.arange(2), "d": (np.int64(3),)}),
                   allow_nan=False)
    assert s == '{"a": 1.5, "b": null, "c": [0, 1], "d": [3]}'


@case
def t_local_codes_compact():
    u, c = F.local_codes(np.array([3.0, 1.0, 3.0], dtype=np.float32))
    assert u.tolist() == [1.0, 3.0] and c.tolist() == [1, 0, 1] and c.dtype == np.uint8


if __name__ == "__main__":
    fails = [r for r in RESULTS if not r[1]]
    for name, ok, err in RESULTS:
        print(("PASS " if ok else "FAIL ") + name + ("" if ok else "  " + err))
    print(json.dumps({"tests_passed": len(RESULTS) - len(fails), "tests_failed": len(fails),
                      "names": [r[0] for r in RESULTS]}))
    sys.exit(1 if fails else 0)
