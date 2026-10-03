"""The RDP accountant (Poisson-subsampled Gaussian; Mironov, Talwar & Zhang 2019) and the z record.

Proves: the fractional-order log A_alpha equals a direct numerical integration of its defining expectation (scipy
quad, log-space integrand) at the study's operating points and elsewhere; the fractional series equals the exact
integer expansion at integer orders; q = 1 is the plain Gaussian mechanism; epsilon is monotone (down in z and
delta, up in T and m); z_for_epsilon is the safe-side bisection root; the improved conversion is never looser than
the standard one; one published reference value is reproduced (see test_reference_abadi_2016 for its provenance);
the record for eps in {8, 1}, delta 5e-6, m 1,024, N 131,072, T 384 is consistent and serialisable.
Negative controls: composition not multiplied by T; subsampling ignored.
"""
from __future__ import annotations

import itertools
import json
import math

import numpy as np
import pytest
from fedsim_testkit import nc

from ppsi.fedsim import accountant as A

Q, N, M, T, DELTA = 1024 / 131072, 131072, 1024, 384, 5e-6


def log_a_numeric(q: float, sigma: float, alpha: float) -> float:
    """log E_{x~N(0,s^2)}[((1-q) + q exp((2x-1)/(2 s^2)))^alpha] by quad, shifted by the closed form for stability."""
    integrate = pytest.importorskip("scipy.integrate")   # skip (not fail) where scipy is absent

    ref = A._log_a_frac(q, sigma, alpha) if not float(alpha).is_integer() else A._log_a_int(q, sigma, int(alpha))

    def f(x):
        logpdf = -x * x / (2 * sigma * sigma) - math.log(sigma * math.sqrt(2 * math.pi))
        inner = np.logaddexp(math.log1p(-q), math.log(q) + (2 * x - 1) / (2 * sigma * sigma))
        return math.exp(logpdf + alpha * inner - ref)
    lo, hi = -40 * sigma, alpha + 40 * sigma
    pts = [0.0, 0.5, alpha * 0.5, float(alpha)]
    v, _ = integrate.quad(f, lo, hi, points=sorted({p for p in pts if lo < p < hi}), limit=800, epsabs=0,
                          epsrel=1e-12)
    return ref + math.log(v)


CASES = [(Q, 0.5642, 2.9), (Q, 1.3094, 15.5), (Q, 1.3094, 16.0), (0.01, 1.0, 1.5), (0.01, 4.0, 20.7),
         (0.05, 0.8, 3.7), (0.2, 2.0, 5.25), (0.001, 0.7, 1.1), (Q, 0.9, 7.3)]


@pytest.mark.parametrize("q,sigma,alpha", CASES)
def test_frac_rdp_matches_numerical_integration(q, sigma, alpha):
    got = A._log_a_frac(q, sigma, alpha)
    num = log_a_numeric(q, sigma, alpha)
    assert got == pytest.approx(num, rel=1e-7, abs=1e-12)


@pytest.mark.parametrize("q,sigma", [(Q, 0.5642), (Q, 1.3094), (0.01, 1.0), (0.1, 2.0)])
def test_frac_series_equals_integer_expansion(q, sigma):
    for a in (2, 3, 5, 8, 16, 32):
        assert A._log_a_frac(q, sigma, float(a)) == pytest.approx(A._log_a_int(q, sigma, a), rel=1e-9, abs=1e-14)


def test_plain_gaussian_and_edges():
    for a in (1.5, 2, 10, 64):
        assert A.rdp_sgm(1.0, 1.7, a) == pytest.approx(a / (2 * 1.7 ** 2))
    assert A.rdp_sgm(0.0, 1.0, 4) == 0.0
    assert math.isinf(A.rdp_sgm(Q, 0.0, 4))
    assert A.rdp_sgm(Q, 1.0, 4) < A.rdp_sgm(1.0, 1.0, 4)            # amplification by subsampling
    with pytest.raises(A.AccountantError):
        A.rdp_sgm(Q, 1.0, 1.0)


def test_monotonicity():
    zs = [0.5, 0.6, 0.8, 1.0, 1.3, 2.0, 4.0]
    e = [A.epsilon(M, N, T, z, DELTA) for z in zs]
    assert all(a > b for a, b in itertools.pairwise(e)), "epsilon must fall as z grows"
    assert A.epsilon(M, N, 100, 1.0, DELTA) < A.epsilon(M, N, 384, 1.0, DELTA) < A.epsilon(M, N, 1000, 1.0, DELTA)
    assert A.epsilon(256, N, T, 1.0, DELTA) < A.epsilon(1024, N, T, 1.0, DELTA) < A.epsilon(4096, N, T, 1.0, DELTA)
    assert A.epsilon(M, N, T, 1.0, 1e-3) < A.epsilon(M, N, T, 1.0, DELTA) < A.epsilon(M, N, T, 1.0, 1e-9)
    for a in (1.5, 2.5, 7.3, 30.0):                                    # RDP grows with the order
        assert A.rdp_sgm(Q, 1.0, a) < A.rdp_sgm(Q, 1.0, a + 0.5)


def test_improved_conversion_never_looser():
    for z in (0.56, 0.8, 1.31, 3.0):
        assert A.epsilon(M, N, T, z, DELTA, conversion="improved") <= A.epsilon(M, N, T, z, DELTA)


def test_z_for_epsilon_bisection_safe_side():
    for eps in (8.0, 1.0, 3.0):
        z = A.z_for_epsilon(M, N, T, DELTA, eps)
        assert A.epsilon(M, N, T, z, DELTA) <= eps
        assert A.epsilon(M, N, T, z * (1 - 1e-6), DELTA) > eps


def test_reference_abadi_2016():
    """Abadi et al. 2016 (CCS, "Deep Learning with Differential Privacy", Section 3.1): for q = 0.01, sigma = 4,
    delta = 1e-5 and T = 10,000 steps the moments accountant gives epsilon ~ 1.26 (vs ~ 9.34 by strong composition).
    Their accountant used integer moments lambda <= 32 with the tail bound, which equals the standard RDP conversion
    at orders lambda + 1 = 2..33; the RDP of the Poisson-subsampled Gaussian is the same quantity, so the value must
    be reproduced to the 2 decimals quoted."""
    orders = list(range(2, 34))
    eps, _ = A.rdp_to_epsilon(orders, A.compute_rdp(0.01, 4.0, 10000, orders), 1e-5)
    assert round(eps, 2) == 1.26
    assert A.epsilon(600, 60000, 10000, 4.0, 1e-5) <= eps + 1e-12        # the full order grid is never looser


def test_registration_record(tmp_path):
    rec = A.registration_record((8.0, 1.0), m=M, N=N, T=T, delta=DELTA)
    by = {r["epsilon_target"]: r for r in rec["records"]}
    for e, r in by.items():
        assert r["achieved_epsilon"] <= e and r["z"] >= r["z_bisection"]
        assert (r["m"], r["N"], r["T"], r["delta"]) == (M, N, T, DELTA)
        assert A.epsilon(M, N, T, r["z"], DELTA) == r["achieved_epsilon"]
    assert by[1.0]["z"] > by[8.0]["z"]
    p = tmp_path / "dp_z_record.json"
    p.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    assert json.loads(p.read_text(encoding="utf-8")) == rec


# ------------------------------------------------------------------------------------------------ negative controls
@nc("composition over rounds ignored (one round's RDP used as the total)")
def test_nc_composition_not_multiplied():
    orders = A.DEFAULT_ORDERS
    bad = A.rdp_to_epsilon(orders, A.compute_rdp(Q, 1.0, 1, orders), DELTA)[0]
    assert bad == pytest.approx(A.epsilon(M, N, T, 1.0, DELTA), rel=0.05)


@nc("subsampling ignored (q = 1): no amplification")
def test_nc_subsampling_ignored():
    orders = A.DEFAULT_ORDERS
    bad = A.rdp_to_epsilon(orders, A.compute_rdp(1.0, 1.3094, T, orders), DELTA)[0]
    assert bad == pytest.approx(A.epsilon(M, N, T, 1.3094, DELTA), rel=0.05)
