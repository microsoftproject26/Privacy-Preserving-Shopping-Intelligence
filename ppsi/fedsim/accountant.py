"""RDP accountant for DP-FedAvg.

Mechanism accounted (dp.py): per round, each sampled client's model delta is flat-L2-clipped to S, the clipped deltas
are SUMMED, N(0, z^2 S^2) is added per coordinate to the sum, and the result is divided by the FIXED cohort size m.
The sum has L2 sensitivity S under add/remove-one-client adjacency, so one round is the Gaussian mechanism with noise
multiplier sigma = z, applied to a subsample of rate q = m / N.

RDP of the subsampled Gaussian mechanism (SGM), Poisson subsampling at rate q (Mironov, Talwar & Zhang 2019,
"Renyi Differential Privacy of the Sampled Gaussian Mechanism", arXiv:1908.10530): for order alpha > 1,
    RDP(alpha) = log(A_alpha) / (alpha - 1),   A_alpha = E_{x ~ N(0, s^2)} [ ((1 - q) + q * exp((2x - 1) / (2 s^2)))^alpha ]
  * integer alpha: the exact binomial expansion A_alpha = sum_k C(alpha, k) (1-q)^(alpha-k) q^k exp((k^2 - k) / (2 s^2));
  * fractional alpha: the two-sided series of MTZ 2019 Section 3.3 (as in TensorFlow Privacy's rdp_accountant), whose
    terms use erfc around z0 = s^2 log(1/q - 1) + 1/2; validated in the tests against direct numerical integration of
    the defining expectation;
  * q = 1: the plain Gaussian mechanism, RDP(alpha) = alpha / (2 s^2); z = 0: infinite; q = 0: 0.
Composition over T rounds: RDP adds (Mironov 2017, "Renyi Differential Privacy", CSF, Prop. 1): T * RDP(alpha).
Conversion to (epsilon, delta), minimised over the order grid:
  * "standard" (Mironov 2017, Prop. 3):            eps = RDP(alpha) + log(1/delta) / (alpha - 1)
  * "improved" (Balle, Barthe, Gaboardi, Hsu & Sato 2020, "Hypothesis testing interpretations and Renyi DP",
    AISTATS, Thm. 21; the form used by TensorFlow Privacy):
                                                   eps = RDP(alpha) + log((alpha - 1)/alpha) - (log delta + log alpha)/(alpha - 1)
  The choice used by `registration_record` is "standard" (the more conservative of the two: a larger z for the
  same epsilon); the improved value at the same z is reported alongside for information only.

Disclosures (for the write-up):
  * Sampling. With participation sampling="uniform" the simulator samples a FIXED-SIZE cohort of m clients
    uniformly WITHOUT replacement per round (independent across rounds). That case is accounted here with the
    Poisson-subsampling bound at q = m / N. This is the common practice (e.g. McMahan et al. 2018, DP-FedAvg), not a
    proof for fixed-size sampling; a fixed-size sample under replace-one adjacency has its own (different) bounds
    (accountant_wor.py reports them). With Poisson participation the sampling is the accounted mechanism.
  * Drop-out only removes sampled clients (their contribution is a zero delta; the denominator stays m), so it can
    only reduce participation; it is not credited in the accounting (no privacy amplification is claimed for it).
  * The clip norm S is chosen from non-private FA_1024 statistics (a rule fixed in advance); that choice is excluded
    from the accounting. No secure aggregation; the simulator's server sees clipped deltas.
  * delta = 5e-6 < 1 / N (N = 131,072).
"""
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Sequence

DEFAULT_ORDERS = tuple([round(1.0 + x / 10.0, 10) for x in range(1, 100)] + list(range(11, 65))
                       + [80, 96, 128, 192, 256, 384, 512, 1024])
CONVERSIONS = ("standard", "improved")
REGISTERED_CONVERSION = "standard"


class AccountantError(ValueError):
    pass


# ------------------------------------------------------------------------------------------------ log-space helpers
def _log_add(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    hi, lo = (a, b) if a >= b else (b, a)
    return hi + math.log1p(math.exp(lo - hi))


def _log_sub(a: float, b: float) -> float:
    """log(exp(a) - exp(b)) for a >= b."""
    if b == -math.inf:
        return a
    if a < b:
        raise AccountantError("log_sub: negative result")
    if a == b:
        return -math.inf
    return a + math.log(-math.expm1(b - a))


def _log_erfc(x: float) -> float:
    """log(erfc(x)), stable for large positive x (asymptotic series beyond x = 25)."""
    if x < 25.0:
        return math.log(math.erfc(x))
    x2 = x * x
    s, term = 1.0, 1.0
    for n in range(1, 8):                               # (-1)^n (2n-1)!! / (2 x^2)^n
        term *= -(2 * n - 1) / (2.0 * x2)
        s += term
    return -x2 - math.log(x) - 0.5 * math.log(math.pi) + math.log(s)


def _is_int(alpha: float) -> bool:
    return float(alpha).is_integer()


# ------------------------------------------------------------------------------------------------ log A_alpha
def _log_a_int(q: float, sigma: float, alpha: int) -> float:
    """log A_alpha for integer alpha >= 1 (exact binomial expansion)."""
    alpha = int(alpha)
    log_a = -math.inf
    lq, l1q = math.log(q), math.log1p(-q)
    for k in range(alpha + 1):
        log_coef = math.lgamma(alpha + 1) - math.lgamma(k + 1) - math.lgamma(alpha - k + 1)
        s = log_coef + k * lq + (alpha - k) * l1q + (k * k - k) / (2.0 * sigma * sigma)
        log_a = _log_add(log_a, s)
    return log_a


def _log_a_frac(q: float, sigma: float, alpha: float) -> float:
    """log A_alpha for real alpha > 1 (MTZ 2019 Section 3.3 series; binomial coefficients by the exact recurrence)."""
    log_a0, log_a1 = -math.inf, -math.inf
    z0 = sigma * sigma * math.log(1.0 / q - 1.0) + 0.5
    lq, l1q = math.log(q), math.log1p(-q)
    log_coef, sign = 0.0, 1.0                            # C(alpha, 0) = 1
    i = 0
    while True:
        j = alpha - i
        log_t0 = log_coef + i * lq + j * l1q
        log_t1 = log_coef + j * lq + i * l1q
        log_e0 = math.log(0.5) + _log_erfc((i - z0) / (math.sqrt(2.0) * sigma))
        log_e1 = math.log(0.5) + _log_erfc((z0 - j) / (math.sqrt(2.0) * sigma))
        log_s0 = log_t0 + (i * i - i) / (2.0 * sigma * sigma) + log_e0
        log_s1 = log_t1 + (j * j - j) / (2.0 * sigma * sigma) + log_e1
        if sign > 0:
            log_a0 = _log_add(log_a0, log_s0)
            log_a1 = _log_add(log_a1, log_s1)
        else:
            log_a0 = _log_sub(log_a0, log_s0)
            log_a1 = _log_sub(log_a1, log_s1)
        i += 1
        if max(log_s0, log_s1) < -30.0 and i > alpha:
            break
        if i > 100000:
            raise AccountantError("fractional-order series did not converge")
        c = alpha - (i - 1)                              # C(alpha, i) = C(alpha, i-1) * (alpha - i + 1) / i
        if c == 0.0:
            break                                        # (only for integer alpha, where the series terminates)
        log_coef += math.log(abs(c)) - math.log(i)
        if c < 0:
            sign = -sign
    return _log_add(log_a0, log_a1)


def rdp_sgm(q: float, noise_multiplier: float, alpha: float) -> float:
    """One-step RDP of the Poisson-subsampled Gaussian mechanism at order alpha."""
    if not 0.0 <= q <= 1.0:
        raise AccountantError(f"sampling rate q={q} outside [0, 1]")
    if alpha <= 1.0:
        raise AccountantError("orders must be > 1")
    if noise_multiplier < 0:
        raise AccountantError("noise multiplier must be >= 0")
    if q == 0.0:
        return 0.0
    if noise_multiplier == 0.0:
        return math.inf
    if q == 1.0:
        return alpha / (2.0 * noise_multiplier ** 2)
    if math.isinf(noise_multiplier):
        return 0.0
    la = _log_a_int(q, noise_multiplier, int(alpha)) if _is_int(alpha) else _log_a_frac(q, noise_multiplier, alpha)
    return la / (alpha - 1.0)


def compute_rdp(q: float, noise_multiplier: float, steps: int, orders: Sequence[float] = DEFAULT_ORDERS) -> list:
    """RDP after `steps` compositions (Mironov 2017 Prop. 1), one value per order."""
    if steps < 0 or int(steps) != steps:
        raise AccountantError("steps must be a non-negative integer")
    return [int(steps) * rdp_sgm(q, noise_multiplier, a) for a in orders]


# ------------------------------------------------------------------------------------------------ conversion
def _eps_one(alpha: float, rdp: float, delta: float, conversion: str) -> float:
    if math.isinf(rdp):
        return math.inf
    if conversion == "standard":
        return rdp + math.log(1.0 / delta) / (alpha - 1.0)
    if conversion == "improved":
        if alpha <= 1.01:                                # numerically unusable there (as in TF Privacy)
            return math.inf
        return max(0.0, rdp + math.log1p(-1.0 / alpha) - (math.log(delta) + math.log(alpha)) / (alpha - 1.0))
    raise AccountantError(f"conversion must be one of {CONVERSIONS}")


def rdp_to_epsilon(orders: Sequence[float], rdp: Sequence[float], delta: float,
                   conversion: str = REGISTERED_CONVERSION) -> tuple:
    """(epsilon, best order) = min over the order grid."""
    if not 0.0 < delta < 1.0:
        raise AccountantError("delta must be in (0, 1)")
    best = (math.inf, None)
    for a, r in zip(orders, rdp):
        e = _eps_one(a, r, delta, conversion)
        if e < best[0]:
            best = (e, a)
    return best


def _check_cohort(m: int, N: int, T: int) -> float:
    if not (isinstance(m, int) and isinstance(N, int) and isinstance(T, int)):
        raise AccountantError("m, N and T must be integers")
    if not 1 <= m <= N or T < 0:
        raise AccountantError(f"need 1 <= m <= N and T >= 0 (m={m}, N={N}, T={T})")
    return m / N


def epsilon(m: int, N: int, T: int, z: float, delta: float, *, orders: Sequence[float] = DEFAULT_ORDERS,
            conversion: str = REGISTERED_CONVERSION) -> float:
    """epsilon of T rounds of DP-FedAvg with cohort m of N (accounted as Poisson rate q = m/N), multiplier z."""
    q = _check_cohort(m, N, T)
    return rdp_to_epsilon(orders, compute_rdp(q, z, T, orders), delta, conversion)[0]


def epsilon_detail(m: int, N: int, T: int, z: float, delta: float, *, orders: Sequence[float] = DEFAULT_ORDERS,
                   conversion: str = REGISTERED_CONVERSION) -> dict:
    q = _check_cohort(m, N, T)
    e, a = rdp_to_epsilon(orders, compute_rdp(q, z, T, orders), delta, conversion)
    return {"epsilon": e, "best_order": a, "q": q}


def z_for_epsilon(m: int, N: int, T: int, delta: float, eps: float, *, orders: Sequence[float] = DEFAULT_ORDERS,
                  conversion: str = REGISTERED_CONVERSION, rel_tol: float = 1e-9) -> float:
    """The smallest noise multiplier z (to rel_tol, returned from the SAFE side: epsilon(z) <= eps) by bisection.
    epsilon is non-increasing in z, so bisection is valid."""
    if eps <= 0:
        raise AccountantError("target epsilon must be > 0")
    f = lambda z: epsilon(m, N, T, z, delta, orders=orders, conversion=conversion)
    lo, hi = 0.0, 1.0
    while f(hi) > eps:
        lo, hi = hi, hi * 2.0
        if hi > 1e6:
            raise AccountantError("no z <= 1e6 reaches the target epsilon")
    while hi - lo > rel_tol * hi:
        mid = 0.5 * (lo + hi)
        if f(mid) > eps:
            lo = mid
        else:
            hi = mid
    return hi


def registration_record(eps_targets: Iterable[float] = (8.0, 1.0), *, m: int = 1024, N: int = 131072, T: int = 384,
                        delta: float = 5e-6, orders: Sequence[float] = DEFAULT_ORDERS, round_up_to: int = 4) -> dict:
    """The JSON record written before the first DP run: one entry per epsilon target. `z` is rounded UP to
    `round_up_to` decimals (so the registered z is on the safe side) and the achieved epsilon is recomputed at it."""
    out = []
    for e in eps_targets:
        z_exact = z_for_epsilon(m, N, T, delta, float(e), orders=orders)
        scale = 10 ** round_up_to
        z = math.ceil(z_exact * scale) / scale
        std = epsilon_detail(m, N, T, z, delta, orders=orders, conversion="standard")
        imp = epsilon_detail(m, N, T, z, delta, orders=orders, conversion="improved")
        out.append({"m": m, "N": N, "T": T, "delta": delta, "epsilon_target": float(e), "z": z,
                    "z_bisection": z_exact, "achieved_epsilon": std["epsilon"], "best_order": std["best_order"],
                    "conversion": "standard (Mironov 2017 Prop. 3)",
                    "achieved_epsilon_improved_conversion_info_only": imp["epsilon"], "q": m / N})
    return {"accountant": "RDP, Poisson-subsampled Gaussian (Mironov, Talwar & Zhang 2019); composition over T; "
                          "standard RDP->(eps, delta) conversion (Mironov 2017)",
            "sampling_disclosure": "fixed-size uniform sampling without replacement per round, accounted with the "
                                   "Poisson bound at q = m/N (common practice; conservative-by-convention, not "
                                   "proven for fixed-size sampling); drop-out only reduces participation and is "
                                   "not credited",
            "orders": list(orders), "records": out}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Noise multipliers for DP-FedAvg at target epsilons")
    ap.add_argument("--out", default=None)
    ap.add_argument("--m", type=int, default=1024)
    ap.add_argument("--N", type=int, default=131072)
    ap.add_argument("--T", type=int, default=384)
    ap.add_argument("--delta", type=float, default=5e-6)
    ap.add_argument("--eps", type=float, nargs="+", default=[8.0, 1.0])
    a = ap.parse_args(argv)
    rec = registration_record(a.eps, m=a.m, N=a.N, T=a.T, delta=a.delta)
    txt = json.dumps(rec, indent=1)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(txt)
    print(json.dumps(rec["records"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
