"""Reports look up actual schema keys; a missing bootstrap is MISSING_EVIDENCE, never a fabricated verdict.

Proves: every lookup path is validated against the result schema (a typo raises); a known but absent block reads as
MISSING_EVIDENCE; a claim from a missing or UNDEFINED CI is MISSING_EVIDENCE / UNDEFINED; the claim vocabulary
follows the CI sign rule; results serialize to strict JSON (no NaN; UNDEFINED is explicit).
Does not prove: that a human report uses these helpers.
"""
from __future__ import annotations

import json

import pytest
from fullrank_testkit import artifact_from_scores, assert_raises, nc, random_manifest, scores_for
from fullrank_testkit import evaluate_default as evaluate

from ppsi.evaluation.fullrank.errors import SchemaKeyError
from ppsi.evaluation.fullrank.report import claim, claim_for, lookup
from ppsi.evaluation.fullrank.values import MISSING_EVIDENCE, Undefined, dumps


@pytest.fixture(scope="module")
def result():
    m = random_manifest(n_users=20, seed=31)
    return evaluate(m, artifact_from_scores(m, scores_for(m, 1)))


def check_missing_bootstrap(claim_fn, res):
    assert claim_fn(res, "A_minus_B") == MISSING_EVIDENCE


def naive_claim(res, name):
    """Bug: a missing bootstrap silently reads as a zero-width CI."""
    ci = res.get("bootstrap", {}).get(name, {}).get("ci95", [0.0, 0.0])
    return "A_EXCEEDS_B" if ci[0] > 0 else ("B_EXCEEDS_A" if ci[1] < 0 else "NO_DETECTABLE_DIFFERENCE")


def test_known_keys_and_typos(result):
    assert lookup(result, ("E2E", "macro", "mrr@20")) == result["E2E"]["macro"]["mrr@20"]
    assert lookup(result, ("strata", "cold_stratum", "WARM", "E2E", "n_users")) >= 0
    assert_raises(SchemaKeyError, lookup, result, ("E2E", "macro", "mrr@25"))
    assert_raises(SchemaKeyError, lookup, result, ("E2E", "macr", "mrr@20"))
    assert_raises(SchemaKeyError, lookup, result, ("bootstrp", "x"))


def test_missing_bootstrap_is_missing_evidence(result):
    assert lookup(result, ("bootstrap", "A_minus_B")) == MISSING_EVIDENCE
    check_missing_bootstrap(claim_for, result)


@nc("a report fabricates NO_DETECTABLE_DIFFERENCE from a missing bootstrap")
def test_nc_fabricated_claim(result):
    check_missing_bootstrap(naive_claim, result)


def test_attached_bootstrap_is_read_by_schema(result):
    import copy

    import numpy as np

    from ppsi.evaluation.fullrank import bootstrap as bs
    keys = np.arange(30)
    N = np.ones(30)
    hi = bs.UserStats(keys, np.full(30, 0.6), N, np.full(30, 0.6), N.copy(), "hi")
    lo = bs.UserStats(keys, np.full(30, 0.2), N, np.full(30, 0.2), N.copy(), "lo")
    plan = bs.make_plan(keys)
    res = bs.attach(copy.deepcopy(result), bs.run_registry(plan, {"hi": hi, "lo": lo},
                                                           [("hi_minus_lo", "delta", "hi", "lo", "macro_e2e")]))
    assert claim_for(res, "hi_minus_lo") == "A_EXCEEDS_B"
    assert lookup(res, ("bootstrap", "hi_minus_lo", "W_sha256")) == plan.W_sha256
    assert claim_for(res, "not_registered") == MISSING_EVIDENCE


def test_claim_vocabulary():
    assert claim({"ci95": [0.001, 0.02]}) == "A_EXCEEDS_B"
    assert claim({"ci95": [-0.02, -0.001]}) == "B_EXCEEDS_A"
    assert claim({"ci95": [-0.01, 0.01]}) == "NO_DETECTABLE_DIFFERENCE"
    assert claim({"ci95": Undefined("ZERO_DENOMINATOR_IN_RESAMPLES")}) == "UNDEFINED"
    assert claim(None) == MISSING_EVIDENCE and claim({}) == MISSING_EVIDENCE


def test_strict_json(result):
    s = dumps(result)
    json.loads(s, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    assert_raises(ValueError, dumps, {"x": float("nan")})
    assert_raises(ValueError, dumps, {"x": float("inf")})
