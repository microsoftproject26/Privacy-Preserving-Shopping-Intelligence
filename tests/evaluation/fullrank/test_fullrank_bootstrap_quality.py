"""The paired user bootstrap and retained quality.

Proves: W is the numpy default_rng draw with seed 20260926, hash-identified and regenerable; every method, seed and
contrast shares one W, and a paired contrast equals a direct per-resample loop; every registered contrast is reported
whatever its sign; per-seed values, paired per-seed deltas and the seed-mean contrast use the same W; ratios are
bootstrapped directly (not divided CI endpoints); a zero central value in any resample makes the ratio CI UNDEFINED
with the count; statistics over different user sets are refused. Retained quality is unclamped (may exceed 100%),
and a zero central denominator is UNDEFINED, never 0.
Does not prove: coverage of the percentile interval, or independence of users (a known limitation).
"""
from __future__ import annotations

import numpy as np
import pytest
from fullrank_testkit import artifact_from_scores, assert_raises, nc, random_manifest, scores_for
from fullrank_testkit import evaluate_default as evaluate

from ppsi.evaluation.fullrank import bootstrap as bs
from ppsi.evaluation.fullrank.errors import PairingError
from ppsi.evaluation.fullrank.metrics import credits_from_ranks
from ppsi.evaluation.fullrank.quality import delta, relative_loss_percent, retained_percent
from ppsi.evaluation.fullrank.values import Undefined

SEED = 20260926


def _stats(m, art, label, metric="mrr@20"):
    c = credits_from_ranks(art.ranks)[metric]
    return bs.user_stats(c, m.eval_mask, m.rankable, m.eval_user_key, label=label)


@pytest.fixture(scope="module")
def setup():
    m = random_manifest(n_users=80, seed=21, oov=0.15)
    arts = {name: artifact_from_scores(m, scores_for(m, s)) for name, s in (("A", 1), ("B", 2), ("C", 3))}
    stats = {k: _stats(m, a, k) for k, a in arts.items()}
    plan = bs.make_plan(stats["A"].user_keys)
    return m, arts, stats, plan


# ------------------------------------------------------------------------------------------------ checks
def check_paired(contrast_fn, plan, a, b):
    out = contrast_fn(plan, a, b)
    ra, rb = bs.resampled(plan, a, "macro_e2e"), bs.resampled(plan, b, "macro_e2e")
    ref = bs.ci95(ra - rb)
    assert out["ci95"][0] == pytest.approx(ref[0], abs=1e-12) and out["ci95"][1] == pytest.approx(ref[1], abs=1e-12)
    assert out["W_sha256"] == plan.W_sha256


def check_all_reported(run_fn, plan, stats):
    reg = [("A_minus_B", "delta", "A", "B", "macro_e2e"), ("B_minus_A", "delta", "B", "A", "macro_e2e"),
           ("A_minus_A", "delta", "A", "A", "macro_e2e"), ("A_retained_vs_C", "retained", "A", "C", "macro_e2e")]
    out = run_fn(plan, stats, reg)
    assert [o["name"] for o in out] == [r[0] for r in reg]
    signs = {o["name"]: np.sign(o["point"]) for o in out if o["kind"] == "delta"}
    assert signs["A_minus_B"] == -signs["B_minus_A"] and signs["A_minus_A"] == 0


def check_ratio_direct(ratio_fn, plan, m_stats, c_stats):
    out = ratio_fn(plan, m_stats, c_stats)
    rm, rc = bs.resampled(plan, m_stats, "macro_e2e"), bs.resampled(plan, c_stats, "macro_e2e")
    ref = bs.ci95(100.0 * rm / rc)
    assert out["ci95"][0] == pytest.approx(ref[0], abs=1e-9) and out["ci95"][1] == pytest.approx(ref[1], abs=1e-9)


def check_retained_rules(ret_fn):
    assert ret_fn(0.30, 0.20) == pytest.approx(150.0)                     # may exceed 100%
    assert ret_fn(0.10, 0.20) == pytest.approx(50.0)
    assert isinstance(ret_fn(0.10, 0.0), Undefined)                       # zero denominator


# ------------------------------------------------------------------------------------------------ variants
def default_contrast(plan, a, b):
    return bs.contrast(plan, a, b, "macro_e2e", "a-b")


def unpaired_contrast(plan, a, b):
    """Bug: an independent resample draw per method."""
    pb = bs.make_plan(plan.user_keys, seed=plan.seed + 1)
    ra, rb = bs.resampled(plan, a, "macro_e2e"), bs.resampled(pb, b, "macro_e2e")
    return {"ci95": bs.ci95(ra - rb), "W_sha256": plan.W_sha256}


def positives_only(plan, stats, reg):
    return [o for o in bs.run_registry(plan, stats, reg) if o["kind"] != "delta" or o["point"] > 0]


def default_ratio(plan, m, c):
    return bs.ratio(plan, m, c, "macro_e2e", "retained")


def endpoint_ratio(plan, m, c):
    """Bug: dividing two marginal CI endpoints."""
    lm, hm = bs.ci95(bs.resampled(plan, m, "macro_e2e"))
    lc, hc = bs.ci95(bs.resampled(plan, c, "macro_e2e"))
    return {"ci95": [100.0 * lm / hc, 100.0 * hm / lc]}


def clamped_retained(M, C):
    return min(100.0, 100.0 * M / C) if C else 0.0


def zero_for_undefined(M, C):
    return 0.0 if C == 0 else 100.0 * M / C


# ------------------------------------------------------------------------------------------------ tests
def test_W_is_the_default_rng_draw_with_the_seed():
    W = bs.make_W(37, seed=SEED, n_resamples=5)
    rng = np.random.default_rng(SEED)
    for r in range(5):
        assert np.array_equal(W[r], rng.integers(0, 37, size=37))
    p = bs.make_plan(np.arange(37))
    bs.verify_plan(p)
    assert p.n_resamples == 1000 and p.seed == SEED and p.W.dtype == np.int32
    assert_raises(PairingError, bs.verify_plan, p, "0" * 64)


def test_paired_shared_W(setup):
    _, _, stats, plan = setup
    check_paired(default_contrast, plan, stats["A"], stats["B"])


@nc("independent resample draws per method (not paired)")
def test_nc_unpaired(setup):
    _, _, stats, plan = setup
    check_paired(unpaired_contrast, plan, stats["A"], stats["B"])


def test_paired_contrast_equals_a_direct_loop(setup):
    _, _, stats, plan = setup
    a, b = stats["A"], stats["B"]
    ma, mb = a.S / a.N, b.S / b.N
    diffs = np.array([ma[w].mean() - mb[w].mean() for w in plan.W])
    mine = bs.contrast(plan, a, b, "macro_e2e", "A-B")
    assert mine["point"] == pytest.approx(ma.mean() - mb.mean(), abs=1e-12)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    assert mine["ci95"][0] == pytest.approx(lo, abs=1e-12)
    assert mine["ci95"][1] == pytest.approx(hi, abs=1e-12)


def test_all_contrasts_reported(setup):
    _, _, stats, plan = setup
    check_all_reported(bs.run_registry, plan, stats)


@nc("contrasts filtered by sign (only positive deltas reported)")
def test_nc_sign_filtered(setup):
    _, _, stats, plan = setup
    check_all_reported(positives_only, plan, stats)


def test_ratio_bootstrapped_directly(setup):
    _, _, stats, plan = setup
    check_ratio_direct(default_ratio, plan, stats["A"], stats["C"])
    loss = bs.ratio(plan, stats["A"], stats["C"], "macro_e2e", "loss", kind="loss")
    ret = bs.ratio(plan, stats["A"], stats["C"], "macro_e2e", "ret")
    assert loss["point"] == pytest.approx(100.0 - ret["point"], abs=1e-9)


@nc("ratio CI from dividing marginal CI endpoints")
def test_nc_endpoint_ratio(setup):
    _, _, stats, plan = setup
    check_ratio_direct(endpoint_ratio, plan, stats["A"], stats["C"])


def test_zero_central_in_resamples_is_undefined():
    keys = np.arange(40)
    N = np.ones(40)
    S_c = np.zeros(40)
    S_c[0] = 1.0                                          # central scores on one user only: many resamples miss it
    c = bs.UserStats(keys, S_c, N, S_c.copy(), N.copy(), "C")
    mstat = bs.UserStats(keys, np.full(40, 0.5), N, np.full(40, 0.5), N.copy(), "M")
    plan = bs.make_plan(keys)
    out = bs.ratio(plan, mstat, c, "macro_e2e", "M/C")
    assert isinstance(out["ci95"], Undefined) and out["n_resamples_zero_denominator"] > 0
    zero = bs.UserStats(keys, np.zeros(40), N, np.zeros(40), N.copy(), "Z")
    assert isinstance(bs.ratio(plan, mstat, zero, "macro_e2e", "M/Z")["point"], Undefined)


def test_per_seed_and_seed_mean(setup):
    _, _, stats, plan = setup
    a_by = {2026: stats["A"], 2027: stats["C"]}
    b_by = {2026: stats["B"], 2027: stats["B"]}
    out = bs.seed_contrast(plan, a_by, b_by, "macro_e2e", "fam")
    assert [p["name"] for p in out["per_seed"]] == ["fam[seed=2026]", "fam[seed=2027]"]
    assert out["point"] == pytest.approx(np.mean([p["point"] for p in out["per_seed"]]), abs=1e-15)
    diffs = np.mean([bs.resampled(plan, a_by[s], "macro_e2e") - bs.resampled(plan, b_by[s], "macro_e2e")
                     for s in (2026, 2027)], axis=0)
    assert out["ci95"] == pytest.approx(bs.ci95(diffs), abs=1e-15)
    assert all(p["W_sha256"] == plan.W_sha256 for p in out["per_seed"])


def test_pairing_refused_on_different_user_sets(setup):
    _, _, stats, _ = setup
    other = bs.make_plan(np.arange(5))
    assert_raises(PairingError, bs.contrast, other, stats["A"], stats["B"], "macro_e2e", "x")


def test_rankable_statistics_use_the_same_draw(setup):
    _, _, stats, plan = setup
    a = bs.resampled(plan, stats["A"], "macro_rankable")
    assert a.shape == (1000,) and np.isfinite(a).all()
    assert bs.point(stats["A"], "macro_rankable") == pytest.approx(
        float((stats["A"].SR[stats["A"].NR > 0] / stats["A"].NR[stats["A"].NR > 0]).mean()))


def test_bootstrap_point_matches_evaluate(setup):
    m, arts, stats, _ = setup
    res = evaluate(m, arts["A"])
    assert bs.point(stats["A"], "macro_e2e") == pytest.approx(res["E2E"]["macro"]["mrr@20"], abs=1e-15)
    assert bs.point(stats["A"], "micro_e2e") == pytest.approx(res["E2E"]["micro"]["mrr@20"], abs=1e-15)
    assert bs.point(stats["A"], "macro_rankable") == pytest.approx(res["RANKABLE"]["macro"]["mrr@20"], abs=1e-15)


def test_retained_quality():
    check_retained_rules(retained_percent)
    assert relative_loss_percent(0.30, 0.20) == pytest.approx(-50.0)        # negative loss is valid
    assert isinstance(relative_loss_percent(0.1, 0.0), Undefined)
    assert delta(0.1, 0.0) == pytest.approx(0.1)                             # the delta stays defined
    assert_raises(ValueError, retained_percent, -0.1, 0.2)
    assert_raises(ValueError, retained_percent, float("nan"), 0.2)


@nc("retained quality clamped at 100%")
def test_nc_clamped():
    check_retained_rules(clamped_retained)


@nc("zero central denominator reported as 0 instead of UNDEFINED")
def test_nc_zero_as_zero():
    check_retained_rules(zero_for_undefined)
