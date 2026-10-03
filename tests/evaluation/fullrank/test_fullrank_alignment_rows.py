"""Alignment by decision_id and ordered manifest hash, manifest validation, and result rows with the full
comparison key.

Proves: predictions are refused when reordered, shortened, bound to another manifest, carrying a rank on an OOV /
censored row or disagreeing on a batch target; forbidden columns, unordered ids and inconsistent masks / censor
classes / header hashes are refused; a result row without topology, objective or precision is refused; rows with
mismatched keys are HISTORICAL_REFERENCE. All inputs are synthetic.
"""
from __future__ import annotations

import numpy as np
import torch
from fullrank_testkit import (
    artifact_from_scores,
    assert_raises,
    columns,
    config,
    manifest_from_rows,
    nc,
    random_manifest,
    random_rows,
    scores_for,
)
from fullrank_testkit import evaluate_default as evaluate

from ppsi.evaluation.fullrank import evaluator_code_sha256
from ppsi.evaluation.fullrank.errors import (
    AlignmentError,
    EvalConfigError,
    ManifestError,
    RowKeyError,
)
from ppsi.evaluation.fullrank.manifest import build_eval_manifest
from ppsi.evaluation.fullrank.predict import align, predict_scores
from ppsi.evaluation.fullrank.rows import (
    COMPARISON_KEY,
    check_block,
    comparison_status,
    make_row,
    rows_from_result,
)

ROWS = [(0, "R", 1), (0, "O", 0), (1, "R", 2), (1, "C", 0)]


# ------------------------------------------------------------------------------------------------ checks
def check_alignment(align_fn):
    m = manifest_from_rows(ROWS)
    art = artifact_from_scores(m, scores_for(m))
    align_fn(m, art)
    rev = type(art)(art.decision_id[::-1].copy(), art.ranks[::-1].copy(), art.n_tied, art.eq_before, dict(art.header))
    assert_raises(AlignmentError, align_fn, m, rev)
    other = manifest_from_rows(ROWS + [(2, "R", 3)])
    assert_raises(AlignmentError, align_fn, other, art)
    bad = type(art)(art.decision_id.copy(), art.ranks.copy(), art.n_tied, art.eq_before, dict(art.header))
    bad.ranks[1] = 3                                                       # a rank on the OOV row
    assert_raises(AlignmentError, align_fn, m, bad)
    h = dict(art.header)
    h["eval_ordered_decision_id_sha256"] = "f" * 64
    assert_raises(AlignmentError, align_fn, m, type(art)(art.decision_id, art.ranks, art.n_tied, art.eq_before, h))


def check_row_key(row_fn):
    key = {f: f"x-{f}" for f in COMPARISON_KEY}
    key.update(aggregation="macro", target_denominator="E2E", metric="mrr", k=20)
    row_fn(key)
    for f in ("topology_hash", "objective_id", "precision_id", "evaluator_code_sha256"):
        k2 = dict(key)
        k2[f] = None
        assert_raises(RowKeyError, row_fn, k2)


# ------------------------------------------------------------------------------------------------ variants
def length_only_align(m, art):
    if len(art.ranks) != m.n:
        raise AlignmentError("length")


def default_row(key):
    return make_row(key, method="C", seed=2026, value=0.1, n_decisions=10, n_users=3, oov_count=1, coverage=0.9)


def keyless_row(key):
    return dict(key, value=0.1)


# ------------------------------------------------------------------------------------------------ tests
def test_alignment():
    check_alignment(align)


@nc("alignment checked by length only")
def test_nc_length_only_alignment():
    check_alignment(length_only_align)


def test_batch_alignment_refusals():
    m = manifest_from_rows(ROWS)
    s = scores_for(m)
    rows = np.flatnonzero(m.eval_mask)                                # the evaluation stream (E2E rows)
    wrong_ids = [(torch.as_tensor(m.decision_id[rows][::-1].copy()), s[rows])]
    assert_raises(AlignmentError, predict_scores, wrong_ids, m, config=config(), model_sha256="x")
    short = [(torch.as_tensor(m.decision_id[rows[:2]]), s[rows[:2]])]   # a non-tail batch shorter than eval_batch
    assert_raises((AlignmentError, EvalConfigError), predict_scores, short, m, config=config(), model_sha256="x")
    two_short = [short[0], (torch.as_tensor(m.decision_id[rows[2:]]), s[rows[2:]])]   # a short non-tail batch
    assert_raises(EvalConfigError, predict_scores, two_short, m, config=config(), model_sha256="x")


def test_manifest_validation_refusals():
    cols = columns(ROWS)
    h = {"manifest_id": "SYNTH", "split": "VALIDATION", "catalogue_K": 24}
    build_eval_manifest(h, cols)
    for mutate, why in (
        (lambda c: c.update(user_id=np.array(["u"] * 4)), "raw id column"),
        (lambda c: c.update(decision_id=c["decision_id"][::-1].copy()), "descending ids"),
        (lambda c: c.update(censor_class=np.array(["NONE", "NONE", "NONE", "NONE"])), "censored row marked NONE"),
        (lambda c: c.update(target_oov=np.array([False, False, False, False])), "OOV row not flagged"),
        (lambda c: c.update(target_class=np.array([1, 0, 2, -1])), "OOV row with a class"),
        (lambda c: c.update(censor_class=np.array(["NONE", "NONE", "NONE", "CEN_WHATEVER"])), "unknown censor class"),
        (lambda c: c.update(cold_stratum=np.array(["WARM", "WARM", "HOT", "WARM"])), "unregistered stratum value"),
    ):
        c = {k: v.copy() for k, v in cols.items()}
        mutate(c)
        assert_raises(ManifestError, build_eval_manifest, h, c)
    m = build_eval_manifest(h, cols)
    assert_raises(ManifestError, build_eval_manifest,
                  dict(h, rankable_ordered_decision_id_sha256="0" * 64), cols)
    build_eval_manifest(dict(h, **m.hashes), cols)                         # the correct hashes are accepted


def test_row_comparison_key():
    check_row_key(default_row)


@nc("a row builder that does not require topology / objective / precision")
def test_nc_keyless_row():
    check_row_key(keyless_row)


def test_rows_from_result_and_block_rules():
    m = random_manifest(n_users=15, seed=41)
    res = evaluate(m, artifact_from_scores(m, scores_for(m)), evaluator_code_sha256=evaluator_code_sha256())
    base = {f: f"x-{f}" for f in COMPARISON_KEY if f not in ("metric", "k", "aggregation", "target_denominator")}
    assert_raises(RowKeyError, rows_from_result, res, base, method="FA", seed=2026)   # models: adapter path only
    rows = rows_from_result(res, base, method="POP", seed=2026, selection_rule="PRACTICAL_BEST", efe=6.0)
    assert len(rows) == 2 * 2 * 12
    assert all(r["query_order_hash"] == res["hashes"]["eval_ordered_decision_id_sha256"] for r in rows)
    assert all(r["topology_hash"] and r["objective_id"] and r["precision_id"] for r in rows)
    assert all(r["previously_seen_evaluation_period"] is True for r in rows)
    check_block(rows)
    assert all(r["eval_score_block"] == config().score_block and r["eval_batch"] == 1024 for r in rows)
    assert all(r["eval_device_class"] == "CPU-FP32-DET" and r["status"] == "MEASURED" for r in rows)
    assert all(r["n_target_tied"] == res["tie_diagnostic"]["n_target_tied"] and "value_pessimistic" in r for r in rows)
    other = dict(rows[0], precision_id="TF32")
    assert comparison_status(rows[0], other) == ("HISTORICAL_REFERENCE", ["precision_id"])
    other_blk = dict(rows[0], eval_score_block=4096)
    assert comparison_status(rows[0], other_blk) == ("HISTORICAL_REFERENCE", ["eval_score_block"])
    assert_raises(RowKeyError, check_block, [rows[0], other])
    assert_raises(RowKeyError, make_row, dict(rows[0]), method="C", seed=1, value=0.1, n_decisions=1, n_users=1,
                  oov_count=0, coverage=1.0, selection_rule="BEST_ON_TEST")


def test_undefined_value_row():
    rows = random_rows(n_users=3, seed=1, cens=1.0, oov=0.0)             # every row censored: empty populations
    m = manifest_from_rows(rows)
    res = evaluate(m, artifact_from_scores(m, scores_for(m)))
    base = {f: f"x-{f}" for f in COMPARISON_KEY if f not in ("metric", "k", "aggregation", "target_denominator")}
    out = rows_from_result(res, base, method="POP", seed=2026)
    assert all(r["value"] is None and r["value_status"] == "UNDEFINED:EMPTY_POPULATION" for r in out)
