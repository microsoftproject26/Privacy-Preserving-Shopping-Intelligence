"""Decision rules for bootstrapped contrasts (ppsi.evaluation.contrast_inference): Holm step-down against a hand
computation, the band / decision agreement, the registered family size, TOST, seed statistics and harm shares."""
from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pytest

from ppsi.evaluation import contrast_inference as inf
from ppsi.evaluation.fullrank import bootstrap as B

ND = NormalDist()


def normal_resamples(se, n=4000, seed=0):
    g = np.random.default_rng(seed)
    x = g.standard_normal(n)
    return (x - x.mean()) / x.std(ddof=1) * se          # exactly the requested sample SD


def test_holm_matches_a_hand_computation():
    se = 1.0
    family = [{"id": "a", "point": 3.5, "resamples": normal_resamples(se, seed=1), "reference": 0.0},
              {"id": "b", "point": 2.3, "resamples": normal_resamples(se, seed=2), "reference": 0.0},
              {"id": "c", "point": 0.5, "resamples": normal_resamples(se, seed=3), "reference": 0.0}]
    out = inf.holm_step_down(family, alpha=0.05)
    p = {k: 2 * (1 - ND.cdf(abs(v["point"]))) for k, v in zip("abc", family, strict=True)}
    assert out["contrasts"]["a"]["p"] == pytest.approx(p["a"])
    assert out["contrasts"]["a"]["reject"] and out["contrasts"]["a"]["level"] == pytest.approx(0.05 / 3)
    assert out["contrasts"]["b"]["reject"] == (p["b"] <= 0.05 / 2)
    assert not out["contrasts"]["c"]["reject"]
    assert out["contrasts"]["b"]["p_holm"] == pytest.approx(max(3 * p["a"], 2 * p["b"]))
    assert out["n_rejected"] == 2


def test_band_excludes_the_reference_exactly_when_rejected():
    g = np.random.default_rng(4)
    family = [{"id": f"k{i}", "point": float(pt), "resamples": normal_resamples(1.0, seed=i), "reference": 0.0}
              for i, pt in enumerate(g.uniform(-4, 4, 12))]
    out = inf.holm_step_down(family)
    for c in out["contrasts"].values():
        lo, hi = c["holm_band"]
        assert (not lo <= 0.0 <= hi) == c["reject"]


def test_missing_contrasts_keep_the_family_size():
    family = [{"id": "a", "point": 2.2, "resamples": normal_resamples(1.0), "reference": 0.0},
              {"id": "gone", "point": None, "resamples": None, "reference": 0.0}]
    out = inf.holm_step_down(family)
    assert out["family_size"] == 2 and out["contrasts"]["gone"]["status"] == "NOT_COMPUTED"
    assert out["contrasts"]["a"]["level"] == pytest.approx(0.025)      # not 0.05: the missing one still counts
    with pytest.raises(ValueError):
        inf.holm_step_down(family, family_size=1)


def test_retention_ratios_use_the_100_reference():
    out = inf.holm_step_down([{"id": "r", "point": 98.0, "resamples": normal_resamples(0.5), "reference": 100.0}])
    assert out["contrasts"]["r"]["z"] == pytest.approx(-4.0) and out["contrasts"]["r"]["reject"]


def test_equivalence_is_the_90_percent_interval_inside_the_margin():
    tight = 0.0001 * np.random.default_rng(5).standard_normal(1000)
    assert inf.equivalence(tight, [-0.0005, 0.0005])["equivalent"]
    wide = 0.001 * np.random.default_rng(5).standard_normal(1000)
    r = inf.equivalence(wide, [-0.0005, 0.0005])
    assert not r["equivalent"] and r["ci_level"] == pytest.approx(0.9)
    with pytest.raises(ValueError):
        inf.equivalence(tight, [1.0, -1.0])


def test_seed_stats_flags_a_spread_larger_than_the_user_interval():
    res = normal_resamples(0.1)
    s = inf.seed_stats({2026: 1.0, 2027: 1.5}, res, 1.25, [1.05, 1.45])
    assert s["seed_spread"] == pytest.approx(0.5) and s["seed_sensitive"]
    assert s["user_seed_interval"][0] < 1.25 < s["user_seed_interval"][1]
    assert inf.seed_stats({2026: 1.0}, res, 1.0, [0.9, 1.1])["status"] == "SINGLE_SEED"


def test_harm_shares():
    h = inf.harm_shares([-1.0, 0.0, 0.0, 2.0, 3.0])
    assert (h["harmed_share"], h["tied_share"], h["improved_share"]) == (0.2, 0.4, 0.4)
    assert h["worst_decile_mean_delta"] == -1.0


def test_seed_mean_resamples_on_a_real_plan():
    rng = np.random.default_rng(6)
    n = 50
    users = np.arange(n)
    plan = B.make_plan(users.tolist(), n_resamples=200)

    def stats(level):
        credit = np.clip(rng.normal(level, 0.1, n), 0, 1)
        return B.user_stats(credit, np.ones(n, bool), np.ones(n, bool), users, "x")
    a = {2026: stats(0.6), 2027: stats(0.6)}
    b = {2026: stats(0.5), 2027: stats(0.5)}
    d = inf.seed_mean_resamples(plan, a, b, "macro_e2e")
    r = inf.seed_mean_resamples(plan, a, b, "macro_e2e", kind="retained")
    assert d.shape == (200,) and d.mean() > 0 and r.mean() > 100
    ref = B.seed_contrast(plan, a, b, "macro_e2e", "a-b")["ci95"]
    assert B.ci95(d) == pytest.approx(ref)
    with pytest.raises(ValueError):
        inf.seed_mean_resamples(plan, a, {2026: b[2026]}, "macro_e2e")
