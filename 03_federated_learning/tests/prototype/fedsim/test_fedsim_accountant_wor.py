"""RDP of the subsampled Gaussian under fixed-size sampling WITHOUT replacement (Wang, Balle & Kasiviswanathan
2019, AISTATS, Theorem 9; accountant_wor.py) and the eps = 4 z record.

Proves: eps_WOR is non-increasing in z and non-decreasing in T (both sensitivity factors); gamma = 1 returns exactly the
unsubsampled Gaussian composition; the independently reproduced value is a BRUTE-FORCE numeric check at tiny cases (the
paper's figures are log-scale plots, not readable to a verifiable digit): for 1-D data in {0, 1/2, 1} (replace-one
sensitivity 1), every replace-one pair of datasets of size n, the exact WOR-subsample mixture of Gaussians, D_alpha by
fine-grid quadrature in both directions — max over pairs lies between the paper's Proposition 11 lower bound (computed
by accountant.py's independent Poisson code, which is the same expression) and our Theorem 9 bound; Theorem 9 is
within the paper's stated additive slack of Proposition 11; the sensitivity report round-trips through JSON with a
valid self-sha256.
Negative controls: gamma squared (too small a bound) breaks the brute-force check; Theorem 9 without the min breaks the
gamma = 1 limit; ignoring the replace-one sensitivity factor; a tampered report passing its self-hash.
"""
from __future__ import annotations

import itertools
import json
import math

import numpy as np
import pytest
from fedsim_testkit import nc

from ppsi.fedsim import accountant as A
from ppsi.fedsim import accountant_wor as W

M, N, T, DELTA = 1024, 131072, 384, 5e-6
GAMMA = M / N
ZS = (0.3, 0.5642, 0.8, 1.0, 1.3094, 2.0, 4.0)


# ------------------------------------------------------------------------------------------------ monotonicity
@pytest.mark.parametrize("c", [1.0, 2.0])
def test_monotone_in_z(c):
    eps = [W.epsilon_wor(M, N, T, z, DELTA, sensitivity_factor=c) for z in ZS]
    assert all(a >= b for a, b in itertools.pairwise(eps)), eps
    assert eps[0] > eps[-1]


@pytest.mark.parametrize("c", [1.0, 2.0])
def test_monotone_in_T(c):
    eps = [W.epsilon_wor(M, N, t, 1.3094, DELTA, sensitivity_factor=c) for t in (1, 10, 96, 384, 1000)]
    assert all(a <= b for a, b in itertools.pairwise(eps)), eps
    assert eps[0] < eps[-1]


def test_rdp_monotone_in_gamma_and_below_unsubsampled():
    for sigma in (0.5, 1.0, 2.0):
        for a in (2, 3, 8, 32, 256):
            vals = [W.rdp_wor(g, sigma, a) for g in (1e-4, GAMMA, 0.05, 0.3, 0.9, 1.0)]
            assert all(x <= y for x, y in itertools.pairwise(vals))
            assert vals[-1] == a / (2 * sigma * sigma)


# ------------------------------------------------------------------------------------------------ gamma = 1 limit
@pytest.mark.parametrize("z", [0.5642, 1.3094, 3.0])
def test_gamma_one_equals_unsubsampled_gaussian_composition(z):
    orders = list(W.WOR_ORDERS)
    expect = A.rdp_to_epsilon(orders, [T * a / (2 * z * z) for a in orders], DELTA)
    got = W.epsilon_wor_detail(4096, 4096, T, z, DELTA)
    assert (got["epsilon"], got["best_order"]) == expect
    assert got["epsilon"] == A.rdp_to_epsilon(orders, A.compute_rdp(1.0, z, T, orders), DELTA)[0]
    for a in (2, 17, 256):                               # continuity: just below 1 the min already picks the Gaussian
        assert W.rdp_wor(1 - 1e-12, z, a) == pytest.approx(a / (2 * z * z), rel=1e-12)


# ------------------------------------------------------------------------------------------------ brute force
def _log_mix(x, means, sigma):
    comp = -(x[None, :] - np.asarray(means, dtype=np.float64)[:, None]) ** 2 / (2 * sigma * sigma)
    return np.logaddexp.reduce(comp, axis=0) - math.log(len(means)) - math.log(sigma * math.sqrt(2 * math.pi))


def _renyi(logp, logq, x, alpha):
    f = alpha * logp + (1 - alpha) * logq
    mx = f.max()
    return (mx + math.log(np.trapezoid(np.exp(f - mx), x))) / (alpha - 1)


def brute_max_wor_divergence(n, m, sigma, alpha, values=(0.0, 0.5, 1.0)):
    """max over replace-one pairs (X, X') in values^n of D_alpha(M o subsample(X) || M o subsample(X')), both directions,
    M = sum of the m-subsample + N(0, sigma^2); subsample = uniform over all m-subsets (Definition 2)."""
    x = np.linspace(-(alpha - 1) * m - 14 * sigma, alpha * m + 14 * sigma, 60001)
    best = -math.inf
    for X in itertools.combinations_with_replacement(values, n):     # datasets up to permutation
        lp = _log_mix(x, [sum(s) for s in itertools.combinations(X, m)], sigma)
        for i, v in itertools.product(range(n), values):
            if v == X[i] or (i > 0 and X[i] == X[i - 1]):
                continue
            Xp = X[:i] + (v,) + X[i + 1:]
            lq = _log_mix(x, [sum(s) for s in itertools.combinations(Xp, m)], sigma)
            best = max(best, _renyi(lp, lq, x, alpha), _renyi(lq, lp, x, alpha))
    return best


BRUTE = [(4, 2, 0.5), (4, 2, 1.0), (4, 2, 2.0), (5, 1, 1.0), (6, 2, 0.7), (6, 2, 1.5)]


@pytest.mark.parametrize("n,m,sigma", BRUTE)
def test_bruteforce_tiny_case_between_prop11_and_thm9(n, m, sigma):
    g = m / n
    for alpha in (2, 3, 4, 6):
        brute = brute_max_wor_divergence(n, m, sigma, alpha)
        upper = W.rdp_wor(g, sigma, alpha)
        lower = A.rdp_sgm(g, sigma, alpha)                # Proposition 11 (Gaussian) == integer Poisson expansion
        assert lower - 1e-7 <= brute <= upper + 1e-9, (n, m, sigma, alpha, lower, brute, upper)


def test_prop11_equals_accountant_integer_poisson_and_thm9_slack():
    """Prop. 11 written out from the paper vs accountant._log_a_int; Theorem 9 <= Prop. 11 + a/(a-1) log(1/(1-g))
    + log 2/(a-1) (the paper's stated additive gap)."""
    for g in (1e-3, GAMMA, 0.05, 0.3):
        for sigma in (0.4, 1.0, 1.3094, 3.0):
            for a in (2, 3, 7, 20, 64):
                s = [math.log1p(a * g / (1 - g))]
                for j in range(2, a + 1):
                    s.append(W._log_binom(a, j) + j * math.log(g / (1 - g)) + (j - 1) * j / (2 * sigma * sigma))
                prop11 = a / (a - 1) * math.log1p(-g) + W._logsumexp(s) / (a - 1)
                assert prop11 == pytest.approx(A.rdp_sgm(g, sigma, a), rel=1e-10, abs=1e-15)
                thm9 = W.rdp_wor_thm9(g, sigma, a)
                assert prop11 <= thm9 + 1e-15
                assert thm9 <= prop11 - a / (a - 1) * math.log1p(-g) + math.log(2) / (a - 1) + 1e-12


def test_invalid_inputs():
    for bad in ((GAMMA, 1.0, 2.5), (GAMMA, 1.0, 1), (1.5, 1.0, 2), (GAMMA, -1.0, 2)):
        with pytest.raises(W.WorAccountantError):
            W.rdp_wor(*bad)
    assert W.rdp_wor(0.0, 1.0, 5) == 0.0 and W.rdp_wor(GAMMA, 0.0, 5) == math.inf


# ------------------------------------------------------------------------------------------------ study values
def test_report_values_and_roundtrip(tmp_path):
    rec = W.sensitivity_report()
    by = {r["z"]: r for r in rec["values"]}
    assert set(by) == {0.5642, 1.3094}
    for z, ref in ((0.5642, 7.99825), (1.3094, 0.99985)):
        r = by[z]
        assert round(r["eps_registered_poisson_add_remove"], 5) == ref == r["eps_registered_reference"]
        assert r["eps_registered_poisson_add_remove"] == A.epsilon(M, N, T, z, DELTA)
        # WOR c = 1 can never beat the Poisson integer-order value (Prop. 11 floor == Poisson RDP at q = gamma)
        assert r["eps_rdp_floor_prop11_c1_info"] == r["eps_poisson_on_integer_orders_2_256_info"]
        assert r["eps_rdp_floor_prop11_c1_info"] <= r["eps_wor_replace_one_c1_literal"]
        assert r["eps_rdp_floor_prop11_c2_info"] <= r["eps_wor_replace_one_c2_mechanism_faithful"]
        assert r["eps_wor_replace_one_c1_literal"] < r["eps_wor_replace_one_c2_mechanism_faithful"]
        assert r["eps_wor_replace_one_c2_mechanism_faithful"] == W.epsilon_wor(M, N, T, z / 2, DELTA)
    assert rec["eps4_achieved_epsilon"] <= 4.0 and rec["eps4_z"] >= rec["eps4_z_bisection"]
    assert rec["eps4_z"] == math.ceil(A.z_for_epsilon(M, N, T, DELTA, 4.0) * 1e4) / 1e4
    assert rec["eps4_achieved_epsilon"] == A.epsilon(M, N, T, rec["eps4_z"], DELTA)
    assert 0.5642 < rec["eps4_z"] < 1.3094
    assert rec["accountant_py_sha256"] == W.accountant_sha256()
    assert rec["report_sha256"] == W.self_sha256(rec)
    p = tmp_path / "dp_wor_sensitivity.json"
    p.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    back = json.loads(p.read_text(encoding="utf-8"))
    assert back == rec and W.self_sha256(back) == back["report_sha256"]


# ------------------------------------------------------------------------------------------------ negative controls
@nc("sampling ratio applied twice (gamma^2 in place of gamma): the bound undercuts the brute-force divergence")
def test_nc_gamma_squared_breaks_bruteforce():
    n, m, sigma, alpha = 4, 2, 1.0, 3
    brute = brute_max_wor_divergence(n, m, sigma, alpha)
    assert brute <= W.rdp_wor((m / n) ** 2, sigma, alpha) + 1e-9


@nc("Theorem 9 used without the min with the unsubsampled RDP: gamma = 1 no longer equals the plain Gaussian")
def test_nc_thm9_without_min_gamma_one():
    z, orders = 1.3094, list(W.WOR_ORDERS)
    raw = [T * W.rdp_wor_thm9(1.0, z, a) for a in orders]
    assert A.rdp_to_epsilon(orders, raw, DELTA)[0] == pytest.approx(
        A.rdp_to_epsilon(orders, [T * a / (2 * z * z) for a in orders], DELTA)[0], rel=1e-9)


@nc("replace-one sensitivity factor ignored (2 S treated as S)")
def test_nc_sensitivity_factor_ignored():
    assert W.epsilon_wor(M, N, T, 1.3094, DELTA, sensitivity_factor=2.0) == pytest.approx(
        W.epsilon_wor(M, N, T, 1.3094, DELTA, sensitivity_factor=1.0), rel=0.05)


@nc("a tampered report value still passes the self-sha256 check")
def test_nc_tampered_report():
    rec = W.sensitivity_report()
    rec["values"][0]["eps_wor_replace_one_c2_mechanism_faithful"] = 1.0
    assert W.self_sha256(rec) == rec["report_sha256"]
