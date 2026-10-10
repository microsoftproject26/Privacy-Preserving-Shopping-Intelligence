"""User-level membership-inference audit of a trained model (numpy only).

The question: can an attacker who sees a user's events and the final model tell whether that user's data was used in
training? The audit compares *members* (users whose training-window events were used) with *non-members* (users of
the same population who were never trained on), scoring each user by the loss the model gives their events.

* Score. `calibrated_scores(target, reference)` = minus (mean loss under the audited model - mean loss under a
  reference model that saw neither group). A large positive score says "the model fits this user unusually well".
  The raw score, minus the target loss alone, is reported beside it.
* AUC. ROC AUC of members above non-members (Mann-Whitney U, ties counted half) with a user-bootstrap percentile CI;
  members and non-members are resampled separately.
* Threshold transfer. Users are split into two halves. The threshold is set on half A's non-members at a target
  false-positive rate (the (k + 1)-th largest score, k = floor(FPR x n)) and evaluated on half B, with exact
  Clopper-Pearson bounds on the true- and false-positive rates.
* Empirical epsilon lower bound. eps_LB = ln(TPR_lower / FPR_upper), clipped at 0, from the half-B bounds at the
  threshold FPR. A DP model whose eps_LB exceeds its accountant epsilon is a failed audit.
* Validity. The reference model must not separate the two groups: its raw-loss AUC CI must include 0.5. Otherwise the
  groups differ in some other way and no leakage claim is made (the eps fields are withheld).
* Verdicts. A non-DP model shows MATERIAL_LEAKAGE iff the AUC CI lower bound is above 0.55 or the TPR at 1 % FPR
  lower bound is above 3 %; a DP model PASSes iff eps_LB <= its accountant epsilon.

Everything is deterministic given the seeds. The functions take per-user mean losses; computing those losses is the
caller's job (one forward pass per user over their training-window events, no gradient).
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from fractions import Fraction

import numpy as np

N_BOOT = 2000
CI_QUANTILES = (0.025, 0.975)
CP_ALPHA = 0.05
THRESHOLD_FPR = Fraction(1, 100)
FPR_LEVELS = (Fraction(1, 1000), Fraction(1, 100))
AUC_MATERIAL = 0.55
TPR1_MATERIAL = 0.03


class AuditError(ValueError):
    pass


# ------------------------------------------------------------------------------------------------ scores
def calibrated_scores(target_loss, reference_loss) -> np.ndarray:
    """Per user: reference loss - target loss (higher = the target model fits the user better than the reference)."""
    t, r = np.asarray(target_loss, dtype=np.float64), np.asarray(reference_loss, dtype=np.float64)
    if t.shape != r.shape:
        raise AuditError("target and reference losses must cover the same users")
    return r - t


def raw_scores(target_loss) -> np.ndarray:
    return -np.asarray(target_loss, dtype=np.float64)


# ------------------------------------------------------------------------------------------------ AUC
def average_ranks(x) -> np.ndarray:
    """1-based ranks with ties at their average rank (stable sort)."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    starts = np.flatnonzero(np.concatenate(([True], xs[1:] != xs[:-1]))) if n else np.zeros(0, dtype=np.int64)
    ends = np.concatenate((starts[1:], [n])) if n else starts
    r = np.empty(n, dtype=np.float64)
    r[order] = np.repeat((starts + ends - 1) / 2.0 + 1.0, ends - starts)
    return r


def auc(pos, neg) -> float:
    """P(member score > non-member score) + P(tie) / 2."""
    pos, neg = np.asarray(pos, dtype=np.float64), np.asarray(neg, dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        raise AuditError("AUC needs at least one member and one non-member")
    r = average_ranks(np.concatenate((pos, neg)))
    u = float(r[: pos.size].sum()) - pos.size * (pos.size + 1) / 2.0
    return u / (pos.size * neg.size)


def bootstrap_auc(pos, neg, *, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """AUC with a percentile CI; per replicate the members, then the non-members, are resampled with replacement from
    one generator, so every model audited with the same seed sees the same replicates."""
    pos, neg = np.asarray(pos, dtype=np.float64), np.asarray(neg, dtype=np.float64)
    rng = np.random.default_rng(seed)
    reps = np.empty(int(n_boot), dtype=np.float64)
    for b in range(int(n_boot)):
        ip = rng.integers(0, pos.size, size=pos.size)
        ineg = rng.integers(0, neg.size, size=neg.size)
        reps[b] = auc(pos[ip], neg[ineg])
    lo, hi = (float(v) for v in np.quantile(reps, CI_QUANTILES, method="linear"))
    return {"auc": auc(pos, neg), "ci95": [lo, hi], "n_boot": int(n_boot), "seed": int(seed),
            "n_members": int(pos.size), "n_nonmembers": int(neg.size)}


# ------------------------------------------------------------------------------------------------ exact binomial bounds
def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction of the regularized incomplete beta function (modified Lentz)."""
    tiny, eps = 1e-300, 1e-15
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 100_000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            return h
    raise AuditError(f"incomplete beta did not converge (a {a}, b {b})")


def betainc(a: float, b: float, x: float) -> float:
    """I_x(a, b), the regularized incomplete beta function (a, b > 0)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    bt = math.exp(lbt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def beta_ppf(q: float, a: float, b: float) -> float:
    """The q-quantile of Beta(a, b), by bisection on I_x(a, b)."""
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if betainc(a, b, mid) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def clopper_pearson(k: int, n: int, alpha: float = CP_ALPHA) -> list[float]:
    """Exact two-sided (1 - alpha) bounds of a binomial proportion k / n."""
    k, n = int(k), int(n)
    if n < 1 or not 0 <= k <= n:
        raise AuditError(f"Clopper-Pearson needs 0 <= k <= n and n >= 1 (k {k}, n {n})")
    lo = 0.0 if k == 0 else beta_ppf(alpha / 2.0, k, n - k + 1)
    hi = 1.0 if k == n else beta_ppf(1.0 - alpha / 2.0, k + 1, n - k)
    return [lo, hi]


# ------------------------------------------------------------------------------------------------ threshold transfer
def threshold_at_fpr(neg, fpr: Fraction) -> float:
    """The (k + 1)-th largest non-member score, k = floor(fpr x n): at most k non-members score above it. A user is
    called a member iff its score is strictly above the threshold."""
    neg = np.sort(np.asarray(neg, dtype=np.float64))[::-1]
    if neg.size == 0:
        raise AuditError("a threshold needs at least one non-member")
    k = math.floor(Fraction(fpr) * neg.size)
    if k >= neg.size:
        raise AuditError(f"FPR {fpr} admits every non-member: no threshold")
    return float(neg[k])


def eps_lower_bound(tpr_lo: float, fpr_hi: float) -> dict:
    """ln(TPR_lower / FPR_upper), clipped at 0 (raw is None when the bound is vacuous)."""
    raw = math.log(tpr_lo / fpr_hi) if tpr_lo > 0.0 and fpr_hi > 0.0 else None
    return {"eps_lb_raw": raw, "eps_lb": max(0.0, raw) if raw is not None else 0.0}


def transfer(pos_a, neg_a, pos_b, neg_b, fpr: Fraction, alpha: float = CP_ALPHA) -> dict:
    """Threshold from half A's non-members at `fpr`, evaluated on half B."""
    tau = threshold_at_fpr(neg_a, fpr)
    pos_b, neg_b = np.asarray(pos_b, dtype=np.float64), np.asarray(neg_b, dtype=np.float64)
    tp, fp = int(np.count_nonzero(pos_b > tau)), int(np.count_nonzero(neg_b > tau))
    tpr_ci, fpr_ci = clopper_pearson(tp, pos_b.size, alpha), clopper_pearson(fp, neg_b.size, alpha)
    out = {"fpr_target": str(fpr), "threshold": tau, "tp_b": tp, "n_members_b": int(pos_b.size), "fp_b": fp,
           "n_nonmembers_b": int(neg_b.size), "tpr_b": tp / pos_b.size, "fpr_b": fp / neg_b.size,
           "tpr_cp95": tpr_ci, "fpr_cp95": fpr_ci}
    out.update(eps_lower_bound(tpr_ci[0], fpr_ci[1]))
    return out


# ------------------------------------------------------------------------------------------------ the audit
def split_half(n: int, rng: np.random.Generator) -> np.ndarray:
    """A boolean half-A mask with exactly floor(n / 2) users in A (a seeded permutation)."""
    a = np.zeros(int(n), dtype=bool)
    a[rng.permutation(int(n))[: int(n) // 2]] = True
    return a


def score_metrics(members, nonmembers, half_a_members, half_a_nonmembers, *, n_boot: int = N_BOOT,
                  seed: int = 0) -> dict:
    """AUC with its CI on all users, the A -> B transfer at every FPR level, and eps_LB at THRESHOLD_FPR."""
    m, n = np.asarray(members, dtype=np.float64), np.asarray(nonmembers, dtype=np.float64)
    if not (np.all(np.isfinite(m)) and np.all(np.isfinite(n))):
        raise AuditError("a user score is not finite")
    am, an = np.asarray(half_a_members, dtype=bool), np.asarray(half_a_nonmembers, dtype=bool)
    if am.shape != m.shape or an.shape != n.shape:
        raise AuditError("the half-A masks must match the score arrays")
    levels = {str(f): transfer(m[am], n[an], m[~am], n[~an], f) for f in FPR_LEVELS}
    thr = levels[str(THRESHOLD_FPR)]
    return {"auc": bootstrap_auc(m, n, n_boot=n_boot, seed=seed), "tpr_at_fpr": levels,
            "eps_lb": thr["eps_lb"], "eps_lb_raw": thr["eps_lb_raw"], "eps_lb_threshold_fpr": str(THRESHOLD_FPR)}


def validity(reference_raw: Mapping) -> dict:
    """The reference model's raw-score AUC CI must include 0.5."""
    lo, hi = reference_raw["auc"]["ci95"]
    ok = lo <= 0.5 <= hi
    return {"auc_ci95": [lo, hi], "includes_0_5": bool(ok), "status": "VALID" if ok else "INVALID"}


def verdict(metrics: Mapping, accountant_epsilon: float | None = None) -> dict:
    """MATERIAL_LEAKAGE / NO_MATERIAL_LEAKAGE for a non-DP model (accountant_epsilon None); PASS / FAIL for a DP one."""
    if accountant_epsilon is not None:
        ok = metrics["eps_lb"] <= float(accountant_epsilon)
        return {"status": "PASS" if ok else "FAIL", "accountant_epsilon": float(accountant_epsilon),
                "eps_lb": metrics["eps_lb"]}
    a_lo = metrics["auc"]["ci95"][0]
    t_lo = metrics["tpr_at_fpr"][str(Fraction(1, 100))]["tpr_cp95"][0]
    leak = a_lo > AUC_MATERIAL or t_lo > TPR1_MATERIAL
    return {"status": "MATERIAL_LEAKAGE" if leak else "NO_MATERIAL_LEAKAGE",
            "auc_ci_lower_above_0_55": bool(a_lo > AUC_MATERIAL), "tpr1_lower_above_0_03": bool(t_lo > TPR1_MATERIAL)}


def audit(reference_loss: tuple, targets: Mapping[str, tuple], *, dp_epsilon: Mapping[str, float] | None = None,
          split_seed: int = 0, n_boot: int = N_BOOT, boot_seed: int = 0) -> dict:
    """Audit several models on one member / non-member selection.

    reference_loss = (member losses, non-member losses) of the reference model; targets[name] = the same pair for each
    audited model (per-user mean losses, users in the same order everywhere). dp_epsilon[name] marks a DP model.
    The audit first checks validity on the reference model; if it is invalid, no eps_LB is reported.
    """
    dp_epsilon = dict(dp_epsilon or {})
    ref_m, ref_n = (np.asarray(x, dtype=np.float64) for x in reference_loss)
    rng = np.random.default_rng(split_seed)
    half_m, half_n = split_half(ref_m.size, rng), split_half(ref_n.size, rng)
    ref_raw = score_metrics(raw_scores(ref_m), raw_scores(ref_n), half_m, half_n, n_boot=n_boot, seed=boot_seed)
    valid = validity(ref_raw)
    out = {"validity": valid, "targets": {}}
    for name, (t_m, t_n) in targets.items():
        cal = score_metrics(calibrated_scores(t_m, ref_m), calibrated_scores(t_n, ref_n), half_m, half_n,
                            n_boot=n_boot, seed=boot_seed)
        raw = score_metrics(raw_scores(t_m), raw_scores(t_n), half_m, half_n, n_boot=n_boot, seed=boot_seed)
        entry = {"calibrated": cal, "raw": raw}
        if valid["status"] == "VALID":
            entry["verdict"] = verdict(cal, dp_epsilon.get(name))
        else:
            for m in (cal, raw):
                for key in [k for k in m if k.startswith("eps_lb")]:
                    del m[key]
            entry["verdict"] = {"status": "NOT_ASSESSED (invalid audit)"}
        out["targets"][name] = entry
    return out


def as_rows(result: Mapping, names: Sequence[str] | None = None) -> list[dict]:
    """One summary row per audited model (calibrated score)."""
    rows = []
    for name in names or list(result["targets"]):
        c = result["targets"][name]["calibrated"]
        tp = c["tpr_at_fpr"][str(THRESHOLD_FPR)]
        rows.append({"model": name, "auc": c["auc"]["auc"], "auc_ci95_low": c["auc"]["ci95"][0],
                     "auc_ci95_high": c["auc"]["ci95"][1], "tpr_at_1pct_fpr": tp["tpr_b"],
                     "tpr_ci95_low": tp["tpr_cp95"][0], "epsilon_lower_bound": c.get("eps_lb"),
                     "verdict": result["targets"][name]["verdict"]["status"]})
    return rows
