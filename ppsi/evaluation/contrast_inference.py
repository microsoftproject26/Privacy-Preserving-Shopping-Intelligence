"""Decision rules for a family of bootstrapped contrasts.

`ppsi.evaluation.fullrank.bootstrap` gives each contrast a point value and its paired-bootstrap resamples. This
module turns those into the readings a preregistered analysis needs:

* `seed_mean_resamples` - the per-resample mean over seeds of a delta or a retention ratio (seed s against seed s on
  the same resample matrix), the distribution behind a seed-mean contrast.
* `holm_step_down` - Holm's step-down over a family of contrasts, on normal intervals from the bootstrap standard
  error. Each contrast gets z, its two-sided p, its Holm level and adjusted p, the decision, and a band that excludes
  the reference exactly when the contrast is rejected (after the first non-rejection every later band is at
  alpha / (m - r), r = the rejections so far). A Bonferroni band is given beside it.
* `equivalence` - two one-sided tests at alpha: equivalent iff the (1 - 2 alpha) percentile interval lies inside the
  margin.
* `seed_stats` - per-seed points, the seed spread, a seed-sensitivity flag (spread larger than half the width of the
  seed-mean interval) and a user-plus-seed interval.
* `harm_shares` - the share of users the first method harms, helps or ties, from per-user deltas.

The family size passed to Holm is the registered size: a contrast that could not be computed still counts, so a
missing result never makes the others easier to reject.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from statistics import NormalDist

import numpy as np

from .fullrank.bootstrap import ResamplePlan, UserStats, resampled

_ND = NormalDist()


def seed_mean_resamples(plan: ResamplePlan, a_by_seed: Mapping[int, UserStats], b_by_seed: Mapping[int, UserStats],
                        statistic: str, kind: str = "delta") -> np.ndarray:
    """Per resample: mean over seeds of (a_s - b_s) for kind "delta", or of 100 a_s / b_s for kind "retained"."""
    if sorted(a_by_seed) != sorted(b_by_seed):
        raise ValueError("a seed-mean contrast needs the same seeds for both methods")
    if kind not in ("delta", "retained"):
        raise ValueError("kind must be 'delta' or 'retained'")
    per = []
    for s in sorted(a_by_seed):
        ra, rb = resampled(plan, a_by_seed[s], statistic), resampled(plan, b_by_seed[s], statistic)
        if kind == "retained" and np.any(rb == 0):
            raise ValueError(f"seed {s}: a zero denominator in some resample; the ratio is undefined")
        per.append(ra - rb if kind == "delta" else 100.0 * ra / rb)
    return np.mean(per, axis=0)


def holm_step_down(contrasts: Sequence[Mapping], *, alpha: float = 0.05, family_size: int | None = None) -> dict:
    """Holm over `contrasts`, each {"id", "point", "resamples", "reference"} (reference 0 for a delta, 100 for a
    retention ratio). Contrasts whose resamples are None are reported as not computed but stay in the family size."""
    m = int(family_size) if family_size is not None else len(contrasts)
    if m < len(contrasts):
        raise ValueError("the family size cannot be smaller than the number of contrasts")
    out, comp = {}, []
    for c in contrasts:
        if c.get("resamples") is None:
            out[c["id"]] = {"status": "NOT_COMPUTED"}
            continue
        se = float(np.std(np.asarray(c["resamples"], dtype=np.float64), ddof=1))
        ref, pt = float(c["reference"]), float(c["point"])
        if se > 0:
            z = (pt - ref) / se
            p = 2.0 * (1.0 - _ND.cdf(abs(z)))
        else:
            z, p = None, (0.0 if pt != ref else 1.0)
        comp.append((p, len(comp), c["id"], pt, se, z, ref))
    comp.sort(key=lambda x: (x[0], x[1]))
    running, stop, n_rej = 0.0, False, 0
    zb = _ND.inv_cdf(1.0 - alpha / (2.0 * m))
    for j, (p, _order, cid, pt, se, z, ref) in enumerate(comp, start=1):
        level = alpha / (m - j + 1)
        running = max(running, min(1.0, (m - j + 1) * p))
        reject = (not stop) and p <= level
        n_rej += int(reject)
        stop = stop or not reject
        band_level = level if reject else alpha / (m - n_rej)
        zq = _ND.inv_cdf(1.0 - band_level / 2.0)
        out[cid] = {"status": "COMPUTED", "se_bootstrap": se, "z": z, "p": p, "rank": j, "level": level,
                    "p_holm": running, "reject": bool(reject), "reference": ref,
                    "holm_band": [pt - zq * se, pt + zq * se], "bonferroni_band": [pt - zb * se, pt + zb * se]}
    return {"family_size": m, "alpha": alpha, "n_computed": len(comp), "n_rejected": n_rej, "contrasts": out}


def equivalence(resamples, margin: Sequence[float], *, alpha: float = 0.05) -> dict:
    """TOST at `alpha`: equivalent iff the (1 - 2 alpha) percentile interval lies inside [margin_low, margin_high]."""
    lo_m, hi_m = (float(x) for x in margin)
    if not lo_m < hi_m:
        raise ValueError("the margin must be an interval [low, high] with low < high")
    lo, hi = (float(x) for x in np.percentile(np.asarray(resamples, dtype=np.float64),
                                              [100.0 * alpha, 100.0 * (1.0 - alpha)]))
    return {"margin": [lo_m, hi_m], "ci": [lo, hi], "ci_level": 1.0 - 2.0 * alpha,
            "equivalent": bool(lo_m <= lo and hi <= hi_m)}


def seed_stats(per_seed_points: Mapping[int, float], seed_mean_resamples_: np.ndarray | None, point: float,
               ci95: Sequence[float] | None) -> dict:
    """Seed spread and sensitivity beside the user-bootstrap interval of a seed-mean contrast."""
    pts = [float(v) for v in per_seed_points.values()]
    out = {"per_seed_points": {str(k): float(v) for k, v in per_seed_points.items()}}
    if len(pts) < 2:
        return dict(out, status="SINGLE_SEED", seed_sensitive=None)
    spread = max(pts) - min(pts)
    out.update(status="COMPUTED", seed_spread=spread)
    if ci95 is None or seed_mean_resamples_ is None:
        return dict(out, seed_sensitive=None)
    half = (float(ci95[1]) - float(ci95[0])) / 2.0
    s2 = float(np.var(np.asarray(pts), ddof=1))
    vb = float(np.var(np.asarray(seed_mean_resamples_, dtype=np.float64), ddof=1))
    se = math.sqrt(vb + s2 / len(pts))
    out.update(user_ci_half_width=half, seed_sensitive=bool(spread > half), seed_variance=s2,
               user_seed_interval=[float(point) - 1.96 * se, float(point) + 1.96 * se])
    return out


def harm_shares(per_user_delta) -> dict:
    """Shares of users with delta < 0 (harmed), > 0 (improved) and = 0 (tied), plus delta quantiles."""
    du = np.asarray(per_user_delta, dtype=np.float64)
    if du.size == 0:
        raise ValueError("no users")
    q = np.percentile(du, [10, 25, 50, 75, 90])
    k = max(1, math.ceil(0.1 * du.size))
    return {"harmed_share": float(np.mean(du < 0)), "improved_share": float(np.mean(du > 0)),
            "tied_share": float(np.mean(du == 0)),
            "per_user_delta_quantiles": {f"P{p}": float(v) for p, v in zip((10, 25, 50, 75, 90), q, strict=True)},
            "worst_decile_mean_delta": float(np.mean(np.sort(du)[:k]))}
