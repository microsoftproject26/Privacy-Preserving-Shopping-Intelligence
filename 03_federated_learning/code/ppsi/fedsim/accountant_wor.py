"""RDP of the subsampled Gaussian under fixed-size sampling WITHOUT replacement (WOR), as a sensitivity analysis.
The main accountant (accountant.py) stays the source of the noise multiplier z; this module only reports a
sensitivity number beside it.

Source (checked against the arXiv v2 PDF, arXiv:1808.00087v2, and the PMLR PDF; same numbering in both):
  Yu-Xiang Wang, Borja Balle, Shiva Prasad Kasiviswanathan, "Subsampled Renyi Differential Privacy and Analytical
  Moments Accountant", AISTATS 2019, PMLR 89:1226-1235.
  * Definition 2 (Subsample): given a dataset X of n points, `subsample` selects a random sample from the uniform
    distribution over all subsets of X of size m; gamma := m / n.
  * Adjacency (Section 2, Definition 1): X, X' in X^n are neighbouring if X' is obtained from X by REPLACING one data
    point by another arbitrary data point (d(X, X') <= 1). Datasets have the same size n: replace-one ("substitution").
  * Theorem 9 (RDP for Subsampled Mechanisms): let M take an input from X^m, m <= n. For all INTEGERS alpha >= 2, if M
    obeys (alpha, eps(alpha))-RDP, then M o subsample obeys (alpha, eps'(alpha))-RDP where
        eps'(alpha) <= 1/(alpha-1) * log( 1 + gamma^2 C(alpha,2) min{ 4(e^{eps(2)} - 1), e^{eps(2)} min{2, (e^{eps(inf)} - 1)^2} }
                                            + sum_{j=3}^{alpha} gamma^j C(alpha,j) e^{(j-1) eps(j)} min{2, (e^{eps(inf)} - 1)^j} ).
    For the Gaussian mechanism eps(inf) = inf, so every min{2, (e^{eps(inf)} - 1)^j} = 2 (the paper, Section 3.1,
    "the term can be simplified as min{4(e^{eps(2)} - 1), 2 e^{eps(2)}}").
  * Gaussian RDP (paper Section 4; Mironov 2017 Table II): eps(alpha) = alpha / (2 sigma^2) for sensitivity 1 under the
    paper's (replace-one) adjacency, sigma = noise std / sensitivity.
  * Proposition 11 (lower bound; used in the tests): for integer alpha >= 1, eps'(alpha) >= alpha/(alpha-1) log(1-gamma)
    + 1/(alpha-1) log(1 + alpha gamma/(1-gamma) + sum_{j>=2} C(alpha,j) (gamma/(1-gamma))^j e^{(j-1) eps(j)}). For the
    Gaussian this equals, term by term, the integer-order Poisson expression of accountant._log_a_int at q = gamma.

What is computed:
  rdp_wor(gamma, sigma, alpha) = min( Theorem-9 bound, alpha / (2 sigma^2) ).
  The min with the unsubsampled RDP is valid (not in Theorem 9 itself): with a coupling of the two uniform m-subsets that
  differ in at most the replaced point, M o subsample(X) and M o subsample(X') are equal-weight mixtures of pairs of
  outputs on neighbouring m-sets, and (P, Q) -> integral P^alpha Q^(1-alpha) is jointly convex for alpha > 1, so
  eps'(alpha) <= eps(alpha). It makes gamma = 1 return exactly the unsubsampled Gaussian (Theorem 9 alone does not:
  at gamma = 1 its j = alpha term carries the factor 2).
  Integer orders only (Theorem 9 is for integer alpha): WOR_ORDERS = 2..256. All sums in log space.
  Composition over T rounds (Mironov 2017 Prop. 1) and the RDP -> (eps, delta) conversion reuse accountant.py
  (compute by T * rdp, then accountant.rdp_to_epsilon with the "standard" conversion, Mironov 2017 Prop. 3).

Sensitivity and adjacency (for the write-up):
  The main accountant uses ADD/REMOVE-one-client adjacency, under which the clipped-delta SUM has L2 sensitivity S,
  so sigma = z. Theorem 9 is a REPLACE-one result. Under replace-one, one client's clipped delta u -> u' changes the sum
  by ||u - u'|| <= 2 S, so the replace-one sensitivity is 2 S and the faithful Gaussian RDP is alpha / (2 (z/2)^2):
  `sensitivity_factor` c (sigma = z / c). The report gives BOTH
    c = 1  — the literal specification eps(alpha) = alpha / (2 z^2) (valid for a mechanism whose replace-one
             sensitivity is S, i.e. NOT for dp.py as implemented); equals the c = 2 value at 2 z;
    c = 2  — the mechanism-faithful replace-one bound for dp.py (sum of clipped deltas, noise z S).
  Why c = 2 cannot be avoided even for "zero-out" (client -> null record contributing 0): Theorem 9's proof compares
  outputs on m-subsets that contain the target point against ones that contain another real point instead, so it needs
  eps(j) over all replace-one pairs of the subsample domain, whose sensitivity is 2 S.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path

from . import accountant as A

WOR_ORDERS = tuple(range(2, 257))
REGISTERED = {"m": 1024, "N": 131072, "T": 384, "delta": 5e-6}
REGISTERED_Z = {8.0: 0.5642, 1.0: 1.3094}
REGISTERED_EPS_REFERENCE = {0.5642: 7.99825, 1.3094: 0.99985}   # accountant.py epsilon at these z (5 dp)
ACCOUNTANT_PATH = Path(__file__).resolve().parent / "accountant.py"
CITATION = ("Wang, Balle & Kasiviswanathan 2019, 'Subsampled Renyi Differential Privacy and Analytical Moments "
            "Accountant', AISTATS 2019, PMLR 89:1226-1235 (arXiv:1808.00087v2), Theorem 9 (upper bound, integer "
            "alpha >= 2, sampling without replacement of m of n points, gamma = m/n, replace-one adjacency); "
            "Proposition 11 (lower bound)")


class WorAccountantError(ValueError):
    pass


def _log_binom(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _logsumexp(xs) -> float:
    xs = [x for x in xs if x != -math.inf]
    if not xs:
        return -math.inf
    hi = max(xs)
    if hi == math.inf:
        return math.inf
    return hi + math.log(sum(math.exp(x - hi) for x in xs))


def gaussian_rdp(alpha: float, sigma: float) -> float:
    """RDP of the Gaussian mechanism, sensitivity 1, noise std sigma: alpha / (2 sigma^2)."""
    if sigma == 0.0:
        return math.inf
    return alpha / (2.0 * sigma * sigma)


def log_thm9_bracket(gamma: float, sigma: float, alpha: int) -> float:
    """log of the bracket inside Theorem 9 for the Gaussian base mechanism (eps(inf) = inf -> every min{2, .} = 2)."""
    alpha = int(alpha)
    lg = math.log(gamma)
    e2 = gaussian_rdp(2, sigma)
    # log min{4 (e^{eps2} - 1), 2 e^{eps2}}; log(e^x - 1) = x + log1p(-e^{-x}) (stable for all x > 0)
    log_c2 = min(math.log(4.0) + e2 + math.log1p(-math.exp(-e2)), math.log(2.0) + e2)
    terms = [0.0, 2 * lg + _log_binom(alpha, 2) + log_c2]
    for j in range(3, alpha + 1):
        terms.append(j * lg + _log_binom(alpha, j) + (j - 1) * gaussian_rdp(j, sigma) + math.log(2.0))
    return _logsumexp(terms)


def rdp_wor_thm9(gamma: float, sigma: float, alpha: int) -> float:
    """Theorem 9 alone (no min with the unsubsampled RDP)."""
    return log_thm9_bracket(gamma, sigma, alpha) / (int(alpha) - 1)


def rdp_wor(gamma: float, sigma: float, alpha: int) -> float:
    """One-step RDP bound of the WOR-subsampled Gaussian: min(Theorem 9, alpha / (2 sigma^2))."""
    if not float(alpha).is_integer() or alpha < 2:
        raise WorAccountantError("Theorem 9 holds for integer orders alpha >= 2 only")
    if not 0.0 <= gamma <= 1.0:
        raise WorAccountantError(f"sampling ratio gamma={gamma} outside [0, 1]")
    if sigma < 0:
        raise WorAccountantError("noise multiplier must be >= 0")
    alpha = int(alpha)
    if gamma == 0.0 or math.isinf(sigma):
        return 0.0
    if sigma == 0.0:
        return math.inf
    base = gaussian_rdp(alpha, sigma)
    if gamma == 1.0:
        return base
    return min(rdp_wor_thm9(gamma, sigma, alpha), base)


def compute_rdp_wor(gamma: float, sigma: float, steps: int, orders: Sequence[int] = WOR_ORDERS) -> list:
    if steps < 0 or int(steps) != steps:
        raise WorAccountantError("steps must be a non-negative integer")
    return [int(steps) * rdp_wor(gamma, sigma, a) for a in orders]


def epsilon_wor_detail(m: int, N: int, T: int, z: float, delta: float, *, sensitivity_factor: float = 1.0,
                       orders: Sequence[int] = WOR_ORDERS) -> dict:
    """(eps, best order) of T rounds, cohort m of N drawn WITHOUT replacement, noise multiplier z relative to S;
    sensitivity_factor c = replace-one L2 sensitivity / S (2 for dp.py's clipped sum), so sigma = z / c."""
    gamma = A._check_cohort(m, N, T)
    if sensitivity_factor <= 0:
        raise WorAccountantError("sensitivity_factor must be > 0")
    sigma = z / sensitivity_factor
    e, a = A.rdp_to_epsilon(list(orders), compute_rdp_wor(gamma, sigma, T, orders), delta, A.REGISTERED_CONVERSION)
    return {"epsilon": e, "best_order": a, "gamma": gamma, "sigma_effective": sigma,
            "sensitivity_factor": sensitivity_factor}


def epsilon_wor(m: int, N: int, T: int, z: float, delta: float, *, sensitivity_factor: float = 1.0,
                orders: Sequence[int] = WOR_ORDERS) -> float:
    return epsilon_wor_detail(m, N, T, z, delta, sensitivity_factor=sensitivity_factor, orders=orders)["epsilon"]


# ------------------------------------------------------------------------------------------------ report
def canonical_json_bytes(obj) -> bytes:
    """Canonical JSON bytes (sorted keys, ASCII, no NaN) used for self-hashes."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def self_sha256(rec: dict, field: str = "report_sha256") -> str:
    return hashlib.sha256(canonical_json_bytes({k: v for k, v in rec.items() if k != field})).hexdigest()


def accountant_sha256() -> str:
    return hashlib.sha256(ACCOUNTANT_PATH.read_bytes()).hexdigest()


ADJACENCY_STATEMENT = (
    "The two numbers use DIFFERENT adjacencies. Main accountant (accountant.py): Poisson subsampling bound, ADD/REMOVE one "
    "client, clipped-sum sensitivity S, sigma = z. WOR (this module): Theorem 9 of Wang-Balle-Kasiviswanathan 2019, "
    "REPLACE-one client, datasets of fixed size N. Precise relations: (1) a replace-one pair is two add/remove steps "
    "(remove x, add x'), so an add/remove (eps, delta) guarantee implies replace-one (2 eps, (1 + e^eps) delta) by group "
    "privacy of size 2 — the factor 2 runs from add/remove TO replace-one, not the other way; (2) replace-one does NOT "
    "imply add/remove in general (datasets of different size are never replace-one neighbours, and fixed-size WOR sampling "
    "is not defined at N +/- 1); (3) under the zero-out convention (a client is 'removed' by replacing it with a null "
    "record that contributes a zero delta, N fixed — how drop-out is simulated), zero-out pairs ARE replace-one pairs "
    "over the domain extended by the null record, so a replace-one eps bounds the zero-out loss with NO factor (group "
    "size 1); for the Poisson-subsampled sum, zero-out and add/remove give identical output distributions, so the "
    "main eps is also a zero-out eps. Hence the like-for-like comparison with the main eps is the "
    "mechanism-faithful WOR value (sensitivity_factor 2: replace-one sensitivity of the clipped sum is 2 S); the c = 1 "
    "value (literal eps(alpha) = alpha/(2 z^2)) is NOT a valid bound for dp.py and equals the c = 2 value at "
    "noise 2 z. Neither number changes the chosen z.")


def sensitivity_report(z_values: Sequence[float] = (0.5642, 1.3094), *, m: int = REGISTERED["m"], N: int = REGISTERED["N"],
            T: int = REGISTERED["T"], delta: float = REGISTERED["delta"]) -> dict:
    rows = []
    for z in z_values:
        pois = A.epsilon_detail(m, N, T, z, delta)
        pois_int = A.rdp_to_epsilon(list(WOR_ORDERS), A.compute_rdp(m / N, z, T, WOR_ORDERS), delta)
        w1 = epsilon_wor_detail(m, N, T, z, delta, sensitivity_factor=1.0)
        w2 = epsilon_wor_detail(m, N, T, z, delta, sensitivity_factor=2.0)
        # Proposition 11 floor: the eps any RDP analysis on these orders gets from the paper's LOWER bound (= Poisson
        # integer-order RDP at q = gamma, sigma = z / c). Shows how much of the WOR number is Theorem-9 slack.
        floor = {c: A.rdp_to_epsilon(list(WOR_ORDERS), A.compute_rdp(m / N, z / c, T, WOR_ORDERS), delta)[0]
                 for c in (1, 2)}
        rows.append({
            "z": z,
            "eps_registered_poisson_add_remove": pois["epsilon"], "best_order_registered": pois["best_order"],
            "eps_registered_reference": REGISTERED_EPS_REFERENCE.get(z),
            "eps_poisson_on_integer_orders_2_256_info": pois_int[0],
            "eps_wor_replace_one_c1_literal": w1["epsilon"], "best_order_wor_c1": w1["best_order"],
            "eps_wor_replace_one_c2_mechanism_faithful": w2["epsilon"], "best_order_wor_c2": w2["best_order"],
            "eps_rdp_floor_prop11_c1_info": floor[1], "eps_rdp_floor_prop11_c2_info": floor[2],
        })
    r4 = A.registration_record((4.0,), m=m, N=N, T=T, delta=delta)["records"][0]
    rec = {
        "analysis": "WOR sensitivity (analysis only)",
        "source": CITATION,
        "theorem_9_gaussian_form": "eps'(alpha) <= 1/(alpha-1) log(1 + gamma^2 C(alpha,2) min{4(e^{eps(2)}-1), "
                                   "2 e^{eps(2)}} + sum_{j=3}^{alpha} 2 gamma^j C(alpha,j) e^{(j-1) eps(j)}), "
                                   "eps(j) = j/(2 sigma^2); implemented as min(that, alpha/(2 sigma^2))",
        "parameters": {"m": m, "N": N, "T": T, "delta": delta, "gamma": m / N},
        "orders_wor": [WOR_ORDERS[0], WOR_ORDERS[-1]],
        "conversion": "standard (Mironov 2017 Prop. 3), identical to accountant.rdp_to_epsilon",
        "composition": "T * RDP (Mironov 2017 Prop. 1); per-round cohorts independent",
        "values": rows,
        "eps4_z": r4["z"], "eps4_achieved_epsilon": r4["achieved_epsilon"], "eps4_z_bisection": r4["z_bisection"],
        "eps4_best_order": r4["best_order"],
        "eps4_note": "z for eps = 4 from the main accountant (accountant.z_for_epsilon, rounded UP to 4 dp, "
                     "achieved eps recomputed at the rounded z; standard conversion; Poisson add/remove)",
        "adjacency_statement": ADJACENCY_STATEMENT,
        "floor_note": "eps_rdp_floor_prop11_* use the paper's Proposition 11 LOWER bound on the WOR RDP in place of "
                      "Theorem 9 (same orders/conversion): no RDP-based WOR analysis on this order grid can go below "
                      "it. It is not a lower bound on the true (eps, delta) (RDP->(eps, delta) conversion is lossy); "
                      "info only.",
        "registered_z_unchanged": True,
        "accountant_py_sha256": accountant_sha256(),
    }
    rec["report_sha256"] = self_sha256(rec)
    return rec


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="WOR sampling sensitivity report for DP-FedAvg")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    rec = sensitivity_report()
    txt = json.dumps(rec, indent=1)
    if a.out:
        Path(a.out).write_text(txt, encoding="utf-8")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
