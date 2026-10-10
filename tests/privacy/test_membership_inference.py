"""The membership-inference audit (ppsi.privacy.membership_inference) on synthetic per-user losses: AUC against brute
force, exact binomial bounds against closed forms, the threshold rule, the epsilon bound, validity and verdicts."""
from __future__ import annotations

import itertools
import math
from fractions import Fraction

import numpy as np
import pytest

from ppsi.privacy import membership_inference as mi


def brute_auc(pos, neg):
    wins = sum((p > n) + 0.5 * (p == n) for p, n in itertools.product(pos, neg))
    return wins / (len(pos) * len(neg))


def test_auc_matches_brute_force_with_ties():
    g = np.random.default_rng(1)
    pos, neg = g.integers(0, 6, 40).astype(float), g.integers(0, 6, 35).astype(float)
    assert mi.auc(pos, neg) == pytest.approx(brute_auc(pos, neg), abs=1e-12)
    assert mi.auc([2.0, 3.0], [0.0, 1.0]) == 1.0 and mi.auc([0.0], [1.0]) == 0.0 and mi.auc([1.0], [1.0]) == 0.5
    with pytest.raises(mi.AuditError):
        mi.auc([], [1.0])


def test_bootstrap_ci_brackets_the_point_and_is_seeded():
    g = np.random.default_rng(2)
    pos, neg = g.normal(0.3, 1, 300), g.normal(0, 1, 300)
    a, b = mi.bootstrap_auc(pos, neg, n_boot=200, seed=7), mi.bootstrap_auc(pos, neg, n_boot=200, seed=7)
    assert a == b and a["ci95"][0] < a["auc"] < a["ci95"][1]


def test_clopper_pearson_closed_forms():
    lo, hi = mi.clopper_pearson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(1 - 0.025 ** (1 / 10), abs=1e-9)
    lo, hi = mi.clopper_pearson(10, 10)
    assert hi == 1.0 and lo == pytest.approx(0.025 ** (1 / 10), abs=1e-9)
    lo, hi = mi.clopper_pearson(5, 10)
    assert lo == pytest.approx(0.187086, abs=1e-6) and hi == pytest.approx(0.812914, abs=1e-6)
    with pytest.raises(mi.AuditError):
        mi.clopper_pearson(11, 10)


def test_threshold_admits_at_most_k_false_positives():
    g = np.random.default_rng(3)
    neg = g.normal(size=1000)
    for fpr in (Fraction(1, 1000), Fraction(1, 100), Fraction(1, 10)):
        tau = mi.threshold_at_fpr(neg, fpr)
        assert np.count_nonzero(neg > tau) <= math.floor(fpr * neg.size)
    with pytest.raises(mi.AuditError):
        mi.threshold_at_fpr([0.0], Fraction(1, 1))


def test_epsilon_lower_bound():
    assert mi.eps_lower_bound(0.0, 0.01) == {"eps_lb_raw": None, "eps_lb": 0.0}
    r = mi.eps_lower_bound(0.1, 0.01)
    assert r["eps_lb"] == pytest.approx(math.log(10))
    assert mi.eps_lower_bound(0.005, 0.01)["eps_lb"] == 0.0


def losses(n, shift, seed):
    """Per-user mean losses: the reference sees neither group; the target fits members better by `shift`."""
    g = np.random.default_rng(seed)
    ref_m, ref_n = g.normal(5, 0.5, n), g.normal(5, 0.5, n)
    tgt_m = ref_m - shift - g.normal(0, 0.05, n)
    tgt_n = ref_n - g.normal(0, 0.05, n)
    return (ref_m, ref_n), (tgt_m, tgt_n)


def test_a_model_that_memorises_members_leaks():
    ref, tgt = losses(2000, 0.2, 4)
    out = mi.audit(ref, {"memorising": tgt}, n_boot=100)
    assert out["validity"]["status"] == "VALID"
    t = out["targets"]["memorising"]
    assert t["calibrated"]["auc"]["auc"] > 0.95 and t["verdict"]["status"] == "MATERIAL_LEAKAGE"
    assert t["calibrated"]["eps_lb"] > 1.0


def test_a_model_that_does_not_fit_members_better_does_not_leak():
    ref, tgt = losses(2000, 0.0, 5)
    out = mi.audit(ref, {"generalising": tgt}, n_boot=100)
    t = out["targets"]["generalising"]
    lo, hi = t["calibrated"]["auc"]["ci95"]
    assert lo <= 0.5 <= hi + 0.02
    assert lo <= 0.55 and t["verdict"]["status"] == "NO_MATERIAL_LEAKAGE"
    rows = mi.as_rows(out)
    assert rows[0]["model"] == "generalising" and rows[0]["verdict"] == "NO_MATERIAL_LEAKAGE"


def test_dp_verdict_compares_with_the_accountant_epsilon():
    ref, leaky = losses(2000, 0.2, 6)
    _, clean = losses(2000, 0.0, 6)
    out = mi.audit(ref, {"dp_clean": clean, "dp_leaky": leaky}, dp_epsilon={"dp_clean": 8.0, "dp_leaky": 1.0},
                   n_boot=50)
    assert out["targets"]["dp_clean"]["verdict"]["status"] == "PASS"
    assert out["targets"]["dp_leaky"]["verdict"]["status"] == "FAIL"


def test_an_invalid_reference_withholds_every_epsilon():
    g = np.random.default_rng(8)
    ref = (g.normal(4, 0.5, 1000), g.normal(5, 0.5, 1000))     # the groups already differ under the reference
    tgt = (ref[0] - 0.1, ref[1])
    out = mi.audit(ref, {"m": tgt}, n_boot=50)
    assert out["validity"]["status"] == "INVALID"
    t = out["targets"]["m"]
    assert "eps_lb" not in t["calibrated"] and t["verdict"]["status"].startswith("NOT_ASSESSED")


def test_half_split_and_input_checks():
    a = mi.split_half(11, np.random.default_rng(0))
    assert a.sum() == 5 and a.dtype == bool
    with pytest.raises(mi.AuditError):
        mi.calibrated_scores([1.0, 2.0], [1.0])
    with pytest.raises(mi.AuditError):
        mi.score_metrics([np.nan], [1.0], [True], [True])
