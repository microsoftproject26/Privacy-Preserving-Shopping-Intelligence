"""The population rules: OOV zero credit, identical E2E / RANKABLE sets across methods, censoring, manifest-only
strata, UNDEFINED empty populations, FP64 aggregation, the coverage identities and the macro counterexample.

Proves: OOV rows stay in the E2E denominator with 0 credit; two methods on one manifest are scored on the identical
E2E and RANKABLE row sets (ordered hashes equal); censored rows never enter a denominator; strata come from manifest
fields and client_idx is refused (a client-index warm cut differs); an empty population is UNDEFINED (a naive mean
returns NaN); the macro is exact to 1e-12 (float32 accumulation is not); micro E2E = coverage x micro RANKABLE and
macro E2E = mean_u(cov_u x m_u) hold, and two small hand-computed fixtures give the exact values, where the invalid
factorization differs.
Does not prove: anything about real cohort counts.
"""
from __future__ import annotations

import numpy as np
import pytest
from fullrank_testkit import (
    artifact_from_scores,
    assert_raises,
    manifest_from_rows,
    nc,
    random_manifest,
    scores_for,
)
from fullrank_testkit import evaluate_default as evaluate

from ppsi.evaluation.fullrank.errors import ManifestError, StrataError
from ppsi.evaluation.fullrank.manifest import stratum_masks
from ppsi.evaluation.fullrank.metrics import credits_from_ranks, identity_terms, population_block
from ppsi.evaluation.fullrank.values import Undefined, dumps


# ------------------------------------------------------------------------------------------------ fixtures
def fixture_a():
    # user a: one rankable row at rank 1; user b: one rankable row at rank 2 and one OOV row
    rows = [(0, "R", 3), (1, "R", 5), (1, "O", 0)]
    ranks = np.array([1, 2, -1])
    return rows, ranks


def fixture_b():
    rows, ranks = fixture_a()
    return rows + [(2, "O", 0)], np.append(ranks, -1)


def _terms(rows, ranks):
    m = manifest_from_rows(rows)
    c = credits_from_ranks(ranks, (20,))["mrr@20"]
    return identity_terms(c, m.eval_mask, m.rankable, m.eval_user_key)


# ------------------------------------------------------------------------------------------------ checks
def check_identity_values(macro_fn):
    ta = _terms(*fixture_a())
    assert ta["micro_e2e"] == pytest.approx(0.5, abs=1e-12)
    assert ta["coverage"] * ta["micro_rankable"] == pytest.approx(0.5, abs=1e-12)
    assert macro_fn(ta) == pytest.approx(0.625, abs=1e-12)
    assert ta["invalid_factorization"] == pytest.approx(0.5625, abs=1e-12)
    tb = _terms(*fixture_b())
    assert macro_fn(tb) == pytest.approx(1.25 / 3, abs=1e-12)
    assert tb["invalid_factorization"] == pytest.approx(0.375, abs=1e-12)
    assert tb["micro_e2e"] == pytest.approx(0.375, abs=1e-12)


def check_oov_zero_credit(e2e_block_fn):
    m = manifest_from_rows([(0, "R", 2), (0, "O", 0), (1, "O", 0), (1, "R", 4)])
    ranks = np.array([1, -1, -1, 1])
    blk = e2e_block_fn(m, ranks)
    assert blk["n_decisions"] == 4 and blk["n_users"] == 2
    assert blk["micro"]["mrr@20"] == pytest.approx(0.5, abs=1e-12)
    assert blk["macro"]["mrr@20"] == pytest.approx(0.5, abs=1e-12)


def check_same_populations(results):
    h0 = results[0]["hashes"]
    for r in results[1:]:
        assert r["hashes"] == h0
        assert r["E2E"]["n_decisions"] == results[0]["E2E"]["n_decisions"]
        assert r["RANKABLE"]["n_decisions"] == results[0]["RANKABLE"]["n_decisions"]


def check_manifest_strata(stratum_fn, m, client_idx):
    masks = stratum_fn(m, client_idx)
    cold = m.fields["cold_stratum"] != "WARM"
    assert np.array_equal(masks["WARM"], ~cold)


def check_empty_is_undefined(block_fn):
    m = manifest_from_rows([(0, "C", 0), (1, "C", 0)])
    blk = block_fn(m, np.array([-1, -1]))
    assert blk["n_decisions"] == 0
    assert all(isinstance(v, Undefined) for v in blk["micro"].values())
    assert all(isinstance(v, Undefined) for v in blk["macro"].values())


def check_fp64_macro(macro_fn):
    users = np.repeat(np.arange(3000), 1)
    credit = np.tile(np.array([1.0 / 3.0, 1.0 / 7.0, 1.0 / 11.0]), 1000)
    exact = (1.0 / 3.0 + 1.0 / 7.0 + 1.0 / 11.0) / 3.0
    assert abs(macro_fn(credit, users) - exact) < 1e-12


# ------------------------------------------------------------------------------------------------ variants
def default_e2e(m, ranks):
    return population_block(credits_from_ranks(ranks), m.eval_mask, m.eval_user_key)


def e2e_dropping_oov(m, ranks):
    return population_block(credits_from_ranks(ranks), m.rankable, m.eval_user_key)


def naive_empty_block(m, ranks):
    """Bug: plain means, so an empty population becomes NaN."""
    c = credits_from_ranks(ranks, (20,))
    idx = np.flatnonzero(m.eval_mask)
    with np.errstate(invalid="ignore", divide="ignore"):
        micro = {k: float(np.sum(v[idx]) / np.float64(idx.size)) for k, v in c.items()}
    return {"n_decisions": int(idx.size), "micro": micro, "macro": dict(micro)}


def default_macro(credit, users):
    return population_block({"x": credit}, np.ones(credit.size, bool), users)["macro"]["x"]


def float32_macro(credit, users):
    keys, inv = np.unique(users, return_inverse=True)
    s = np.zeros(keys.size, np.float32)
    np.add.at(s, inv, credit.astype(np.float32))
    c = np.bincount(inv).astype(np.float32)
    per = (s / c).astype(np.float32)
    acc = np.float32(0.0)
    for v in per:
        acc = np.float32(acc + v)
    return float(acc / np.float32(keys.size))


def default_strata(m, _client_idx):
    return stratum_masks(m, "cold_stratum")


def client_index_warm_cut(m, client_idx):
    return {"WARM": client_idx >= 3}


# ------------------------------------------------------------------------------------------------ tests
def test_identity_fixture_values():
    check_identity_values(lambda t: t["macro_identity_rhs"])
    ta = _terms(*fixture_a())
    assert ta["macro_e2e"] == pytest.approx(0.625, abs=1e-12)


@nc("the invalid factorization mean(cov) x macro_rankable used as the E2E macro")
def test_nc_invalid_factorization():
    check_identity_values(lambda t: t["invalid_factorization"])


def test_identities_hold_on_random_results():
    for seed in range(4):
        m = random_manifest(n_users=60, seed=seed, oov=0.2)
        res = evaluate(m, artifact_from_scores(m, scores_for(m, seed)))
        for name, err in res["identity_checks"].items():
            assert err["micro"] <= 1e-12 and err["macro"] <= 1e-12, name


def test_oov_zero_credit():
    check_oov_zero_credit(default_e2e)


@nc("OOV rows dropped from the E2E denominator")
def test_nc_oov_dropped():
    check_oov_zero_credit(e2e_dropping_oov)


def test_identical_populations_across_methods():
    m = random_manifest(n_users=30, seed=2)
    res = [evaluate(m, artifact_from_scores(m, scores_for(m, s))) for s in (10, 11, 12)]
    check_same_populations(res)


@nc("one method scored on a per-client catalogue (a different rankable set)")
def test_nc_different_rankable_set():
    rows_a = [(0, "R", 1), (0, "R", 2), (1, "R", 3), (1, "R", 4)]
    rows_b = [(0, "R", 1), (0, "O", 0), (1, "R", 3), (1, "R", 4)]     # a class missing from B's catalogue
    ma, mb = manifest_from_rows(rows_a), manifest_from_rows(rows_b)
    check_same_populations([evaluate(ma, artifact_from_scores(ma, scores_for(ma))),
                            evaluate(mb, artifact_from_scores(mb, scores_for(mb)))])


def test_censored_rows_never_in_a_denominator():
    m = manifest_from_rows([(0, "R", 1), (0, "C", 0), (1, "C", 0), (1, "O", 0)])
    res = evaluate(m, artifact_from_scores(m, scores_for(m)))
    assert res["E2E"]["n_decisions"] == 2 and res["RANKABLE"]["n_decisions"] == 1
    assert res["censoring"]["n_censored"] == 2 and res["censoring"]["counts"]["CEN_NO_NEXT"] == 2
    assert res["censoring"]["censored_share"] == pytest.approx(0.5)


def test_strata_from_manifest_fields_only():
    m = random_manifest(n_users=20, seed=4)
    masks = stratum_masks(m, "cold_stratum")
    assert set(masks) <= {"WARM", "COLD_NO_TRAIN_EVENTS", "COLD_TRAIN_EVENTS_NO_LABEL"}
    assert_raises(StrataError, stratum_masks, m, "client_idx")
    assert_raises(StrataError, stratum_masks, m, "panel_label")          # allowed field, absent from this manifest
    rows = [(0, "R", 1), (1, "R", 2)]
    assert_raises(ManifestError, manifest_from_rows, rows, extra={"client_idx": np.array([5, 2])})
    res = evaluate(m, artifact_from_scores(m, scores_for(m)))
    assert set(res["strata"]) == {"cold_stratum", "subperiod"}


def _client_cut_fixture():
    rows = [(0, "R", 1), (1, "R", 2), (2, "R", 3)]
    extra = {"cold_stratum": np.array(["WARM", "COLD_TRAIN_EVENTS_NO_LABEL", "COLD_NO_TRAIN_EVENTS"])}
    client_idx = np.array([7, 5, 2])      # user 1 has TRAIN events (idx >= 3) but no eligible example
    return manifest_from_rows(rows, extra=extra), client_idx


def test_strata_manifest_cut():
    m, cidx = _client_cut_fixture()
    check_manifest_strata(default_strata, m, cidx)


@nc("a warm cut client_idx >= 3 counts a COLD_TRAIN_EVENTS_NO_LABEL user as warm")
def test_nc_client_idx_warm_cut():
    m, cidx = _client_cut_fixture()
    check_manifest_strata(client_index_warm_cut, m, cidx)


def test_empty_population_is_undefined():
    check_empty_is_undefined(default_e2e)
    m = manifest_from_rows([(0, "C", 0), (1, "C", 0)])
    res = evaluate(m, artifact_from_scores(m, scores_for(m)))
    assert isinstance(res["coverage"], Undefined)
    s = dumps(res)                                                     # strict JSON, no NaN
    assert '"status": "UNDEFINED"' in s and "NaN" not in s


@nc("a naive population block returns NaN for an empty population")
def test_nc_naive_empty_population_nan():
    check_empty_is_undefined(naive_empty_block)


def test_fp64_macro():
    check_fp64_macro(default_macro)


@nc("float32 macro accumulation")
def test_nc_float32_macro():
    check_fp64_macro(float32_macro)


def test_grid_and_primary_present():
    m = random_manifest(n_users=25, seed=6)
    res = evaluate(m, artifact_from_scores(m, scores_for(m)))
    for den in ("E2E", "RANKABLE"):
        for agg in ("micro", "macro"):
            assert set(res[den][agg]) == {f"{f}@{k}" for f in ("mrr", "hr", "ndcg") for k in (1, 5, 10, 20)}
    blk = res["E2E"]["macro"]
    assert blk["mrr@1"] == blk["hr@1"] == pytest.approx(blk["ndcg@1"], abs=1e-15)   # k = 1 coincide
