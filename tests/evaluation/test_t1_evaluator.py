"""Tests for T1 Evaluation Harness (S2-PR-07 / #31)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import torch

from ppsi.evaluation.t1 import (
    _REQUIRED_TASK_EXAMPLE_COLUMNS,
    evaluate_t1_ranks,
    history_bucket,
    metric_records_from_t1_summary,
    preflight_t1_task_examples,
    rank_targets_from_rankings,
    rank_targets_from_scores,
    validate_t1_dense_labels,
    validate_t1_evaluator_config,
)
from scripts.experiments.schemas import validate_metric_record

_CONFIG_PATH = Path("config/evaluation/t1_evaluator.v1.json")
_FIXTURE_PATH = Path("fixtures/evaluation/t1_hand_worked.v1.json")


@pytest.fixture
def canonical_config() -> dict[str, Any]:
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def hand_worked_fixture() -> dict[str, Any]:
    with open(_FIXTURE_PATH, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 1 & 2: Config validation
# ---------------------------------------------------------------------------


def test_01_config_exact_schema_version_category_count(canonical_config: dict[str, Any]) -> None:
    validate_t1_evaluator_config(canonical_config)
    assert canonical_config["category_count"] == 588
    assert canonical_config["schema"] == "t1_evaluator_config_v1"
    assert canonical_config["version"] == "1"


def test_02_unknown_config_field_fails(canonical_config: dict[str, Any]) -> None:
    bad_config = copy.deepcopy(canonical_config)
    bad_config["unapproved_extra_field"] = 123
    with pytest.raises(ValueError, match="Unknown top-level config fields"):
        validate_t1_evaluator_config(bad_config)


# ---------------------------------------------------------------------------
# 3 - 6: Dense label validation
# ---------------------------------------------------------------------------


def test_03_dense_label_float_to_int() -> None:
    assert validate_t1_dense_labels([60.0, 0.0, 587.0]) == [60, 0, 587]


def test_04_non_integral_label_fails() -> None:
    with pytest.raises(ValueError, match="not an integer code"):
        validate_t1_dense_labels([60.5])


def test_05_negative_or_out_of_range_label_fails() -> None:
    with pytest.raises(ValueError, match="out of bounds"):
        validate_t1_dense_labels([-1])
    with pytest.raises(ValueError, match="out of bounds"):
        validate_t1_dense_labels([588])


def test_06_nan_or_inf_label_fails() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        validate_t1_dense_labels([float("nan")])
    with pytest.raises(ValueError, match="non-finite"):
        validate_t1_dense_labels([float("inf")])
    with pytest.raises(TypeError, match="boolean"):
        validate_t1_dense_labels([True])  # booleans are not valid codes
    with pytest.raises(ValueError, match="null/None"):
        validate_t1_dense_labels([None])


# ---------------------------------------------------------------------------
# 7 - 14: Score ranking and tie-breaking
# ---------------------------------------------------------------------------


def test_07_score_shape_wrong_width_fails() -> None:
    bad_scores = torch.zeros(2, 100)  # should be 588
    with pytest.raises(ValueError, match="does not match category_count 588"):
        rank_targets_from_scores(bad_scores, [0, 1], category_count=588)


def test_08_nan_or_inf_score_fails() -> None:
    scores = torch.zeros(1, 588)
    scores[0, 10] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        rank_targets_from_scores(scores, [0], category_count=588)


def test_09_score_rank_1() -> None:
    scores = torch.zeros(1, 6)
    scores[0, 3] = 10.0  # highest score
    rank = rank_targets_from_scores(scores, [3], category_count=6)
    assert rank.item() == 1


def test_10_score_rank_4() -> None:
    # 3 categories strictly higher than target
    scores = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0, 0.0]])
    # target is index 3 (score 2.0), categories 0, 1, 2 have scores 5.0, 4.0, 3.0 > 2.0
    rank = rank_targets_from_scores(scores, [3], category_count=6)
    assert rank.item() == 4


def test_11_rank_greater_than_20_gives_zero_mrr20() -> None:
    # rank 25 -> MRR@20 contribution must be exactly 0
    summary = evaluate_t1_ranks(
        ranks=[25],
        category_changed=[True],
        client_ids=["client-1"],
        mrr_cutoff=20,
    )
    assert summary.slices["overall"]["mrr_at_20_micro"] == 0.0
    assert summary.slices["overall"]["accuracy_at_1_micro"] == 0.0


def test_12_tie_with_lower_code_ahead() -> None:
    # scores: [0, 0, 1, 1, 0, 0], target_code = 3 (score 1.0).
    # code 2 has score 1.0 (equal), but code 2 < 3, so code 2 wins and is ahead (rank 1), target is rank 2.
    scores = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0]])
    rank = rank_targets_from_scores(scores, [3], category_count=6)
    assert rank.item() == 2


def test_13_tie_with_higher_code_behind() -> None:
    # scores: [0, 0, 1, 1, 0, 0], target_code = 2 (score 1.0).
    # code 3 has score 1.0 (equal), but code 3 > 2, so target code 2 is ahead (rank 1).
    scores = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0]])
    rank = rank_targets_from_scores(scores, [2], category_count=6)
    assert rank.item() == 1


def test_14_all_equal_scores_rank_is_target_code_plus_one() -> None:
    # all scores 1.0, target 4 -> codes 0, 1, 2, 3 have equal score and lower code -> 4 ahead -> rank 5.
    scores = torch.ones(1, 6)
    rank = rank_targets_from_scores(scores, [4], category_count=6)
    assert rank.item() == 5


# ---------------------------------------------------------------------------
# 15 - 18: Ranked-list adapter semantics
# ---------------------------------------------------------------------------


def test_15_ranking_permutation_accepted() -> None:
    rankings = torch.tensor([[2, 0, 1, 3]])
    ranks = rank_targets_from_rankings(rankings, [1], category_count=4)
    # code 1 is at index 2 -> 1-based rank 3
    assert ranks.item() == 3


def test_16_duplicate_ranking_rejected() -> None:
    bad_rankings = torch.tensor([[0, 1, 1, 3]])
    with pytest.raises(ValueError, match="strict permutation"):
        rank_targets_from_rankings(bad_rankings, [1], category_count=4)


def test_17_missing_or_partial_ranking_rejected() -> None:
    bad_rankings = torch.tensor([[0, 1, 2]])  # width 3, expected 4
    with pytest.raises(ValueError, match="does not match category_count 4"):
        rank_targets_from_rankings(bad_rankings, [1], category_count=4)


def test_18_out_of_range_ranking_rejected() -> None:
    bad_rankings = torch.tensor([[0, 1, 2, 4]])  # 4 is out of bounds for C=4
    with pytest.raises(ValueError, match="strict permutation"):
        rank_targets_from_rankings(bad_rankings, [1], category_count=4)


# ---------------------------------------------------------------------------
# 19 - 24: Hand-worked oracle and slice aggregations
# ---------------------------------------------------------------------------


def test_19_20_21_fixture_exact_mrr_and_macro_differs_from_micro(
    hand_worked_fixture: dict[str, Any],
) -> None:
    rows = hand_worked_fixture["rows"]
    ranks = [r["target_rank"] for r in rows]
    cat_changed = [r["category_changed"] for r in rows]
    client_ids = [r["client_id"] for r in rows]
    hist_counts = [r["train_history_count"] for r in rows]

    summary = evaluate_t1_ranks(
        ranks=ranks,
        category_changed=cat_changed,
        client_ids=client_ids,
        train_history_counts=hist_counts,
        mrr_cutoff=20,
    )

    exp_overall = hand_worked_fixture["expected"]["overall"]
    exp_next_distinct = hand_worked_fixture["expected"]["next_distinct"]

    # Overall slice assertions
    act_overall = summary.slices["overall"]
    assert act_overall["decision_count"] == exp_overall["decision_count"]
    assert act_overall["client_count"] == exp_overall["client_count"]
    assert (
        pytest.approx(act_overall["mrr_at_20_micro"], abs=1e-12) == exp_overall["mrr_at_20_micro"]
    )
    assert (
        pytest.approx(act_overall["mrr_at_20_macro"], abs=1e-12) == exp_overall["mrr_at_20_macro"]
    )
    assert (
        pytest.approx(act_overall["accuracy_at_1_micro"], abs=1e-12)
        == exp_overall["accuracy_at_1_micro"]
    )
    assert (
        pytest.approx(act_overall["accuracy_at_1_macro"], abs=1e-12)
        == exp_overall["accuracy_at_1_macro"]
    )

    # Next distinct slice assertions
    act_nd = summary.slices["next_distinct"]
    assert act_nd["decision_count"] == exp_next_distinct["decision_count"]
    assert act_nd["client_count"] == exp_next_distinct["client_count"]
    assert (
        pytest.approx(act_nd["mrr_at_20_micro"], abs=1e-12) == exp_next_distinct["mrr_at_20_micro"]
    )
    assert (
        pytest.approx(act_nd["mrr_at_20_macro"], abs=1e-12) == exp_next_distinct["mrr_at_20_macro"]
    )
    assert (
        pytest.approx(act_nd["accuracy_at_1_micro"], abs=1e-12)
        == exp_next_distinct["accuracy_at_1_micro"]
    )
    assert (
        pytest.approx(act_nd["accuracy_at_1_macro"], abs=1e-12)
        == exp_next_distinct["accuracy_at_1_macro"]
    )

    # Macro != Micro proof
    assert act_overall["mrr_at_20_macro"] != act_overall["mrr_at_20_micro"]
    assert act_nd["mrr_at_20_macro"] != act_nd["mrr_at_20_micro"]


def test_22_23_slice_filtering(hand_worked_fixture: dict[str, Any]) -> None:
    rows = hand_worked_fixture["rows"]
    summary = evaluate_t1_ranks(
        ranks=[r["target_rank"] for r in rows],
        category_changed=[r["category_changed"] for r in rows],
        client_ids=[r["client_id"] for r in rows],
    )
    # Overall contains 7 decisions across 4 clients
    assert summary.slices["overall"]["decision_count"] == 7
    assert summary.slices["overall"]["client_count"] == 4

    # next_distinct contains 5 decisions across 3 clients (fixture-C has only false)
    assert summary.slices["next_distinct"]["decision_count"] == 5
    assert summary.slices["next_distinct"]["client_count"] == 3


def test_24_empty_slice_is_zero_support() -> None:
    summary = evaluate_t1_ranks(
        ranks=[1, 2],
        category_changed=[False, False],  # next_distinct is empty!
        client_ids=["c1", "c2"],
    )
    nd = summary.slices["next_distinct"]
    assert nd["status"] == "ZERO_SUPPORT"
    assert nd["decision_count"] == 0
    assert nd["client_count"] == 0
    assert nd["mrr_at_20_micro"] is None
    assert nd["mrr_at_20_macro"] is None


# ---------------------------------------------------------------------------
# 25 - 27: History buckets
# ---------------------------------------------------------------------------


def test_25_history_bucket_boundaries() -> None:
    assert history_bucket(0) == "BELOW_10_RETAINED_C1"
    assert history_bucket(9) == "BELOW_10_RETAINED_C1"
    assert history_bucket(10) == "10_19"
    assert history_bucket(19) == "10_19"
    assert history_bucket(20) == "20_49"
    assert history_bucket(49) == "20_49"
    assert history_bucket(50) == "50_99"
    assert history_bucket(99) == "50_99"
    assert history_bucket(100) == "100_plus"
    assert history_bucket(1000) == "100_plus"
    assert history_bucket(None) is None
    with pytest.raises(ValueError, match="non-negative"):
        history_bucket(-1)


def test_26_inconsistent_history_count_for_client_fails() -> None:
    with pytest.raises(ValueError, match="Inconsistent train_history_count"):
        evaluate_t1_ranks(
            ranks=[1, 2],
            category_changed=[True, True],
            client_ids=["client-A", "client-A"],
            train_history_counts=[15, 25],  # inconsistent!
        )


def test_27_missing_history_produces_not_available() -> None:
    summary = evaluate_t1_ranks(
        ranks=[1, 2],
        category_changed=[True, True],
        client_ids=["c1", "c2"],
        train_history_counts=None,
    )
    assert summary.history_stratification_status == "NOT_AVAILABLE"
    assert summary.history_buckets is None


# ---------------------------------------------------------------------------
# 28 - 30: Metric records conversion
# ---------------------------------------------------------------------------


def test_28_29_30_metric_records_valid_and_support_convention(
    hand_worked_fixture: dict[str, Any],
) -> None:
    rows = hand_worked_fixture["rows"]
    summary = evaluate_t1_ranks(
        ranks=[r["target_rank"] for r in rows],
        category_changed=[r["category_changed"] for r in rows],
        client_ids=[r["client_id"] for r in rows],
    )
    records = metric_records_from_t1_summary(summary)

    # Must emit exactly 8 records (4 for next_distinct, 4 for overall)
    assert len(records) == 8
    for rec in records:
        validate_metric_record(rec)
        assert rec["task"] == "T1"
        assert rec["cohort"] == "C1"
        assert rec["direction"] == "MAXIMIZE"
        assert rec["unit"] == "FRACTION"
        # Macro support must equal client_count, micro support must equal decision_count
        if ".macro" in rec["metric_id"]:
            if "next_distinct" in rec["metric_id"]:
                assert rec["support"] == 3
            else:
                assert rec["support"] == 4
        elif ".micro" in rec["metric_id"]:
            if "next_distinct" in rec["metric_id"]:
                assert rec["support"] == 5
            else:
                assert rec["support"] == 7

        # No per-row / client identity leakage in records
        rec_str = json.dumps(rec)
        assert "fixture-" not in rec_str
        assert "client-" not in rec_str


# ---------------------------------------------------------------------------
# 31: Real TaskExample preflight rejects nulls using toy parquet
# ---------------------------------------------------------------------------


def test_31_preflight_helper_rejects_nulls_in_required_fields(tmp_path: Path) -> None:
    base_data = {
        "client": ["c1", "c2"],
        "session": ["s1", "s2"],
        "decision_order": [0, 1],
        "current_category": [10, 20],
        "label_value": [30.0, 40.0],
        "category_changed": [True, False],
        "task_mask": [True, True],
        "status": ["OBSERVED", "OBSERVED"],
        "cohort": ["C1", "C1"],
        "split": ["VALIDATION", "VALIDATION"],
    }
    # Valid toy file
    valid_file = tmp_path / "valid.parquet"
    pl.DataFrame(base_data).write_parquet(valid_file)
    res = preflight_t1_task_examples(valid_file, category_count=588)
    assert res["row_count"] == 2
    assert res["unique_client_count"] == 2
    assert res["all_rows_valid_contract"] is True

    # Test null rejection on every required column
    for col in _REQUIRED_TASK_EXAMPLE_COLUMNS:
        bad_data = copy.deepcopy(base_data)
        bad_data[col] = [None, bad_data[col][1]]  # inject null
        bad_file = tmp_path / f"null_{col}.parquet"
        pl.DataFrame(bad_data).write_parquet(bad_file)
        with pytest.raises(ValueError, match=f"Column '{col}' contains 1 null values"):
            preflight_t1_task_examples(bad_file, category_count=588)


# ---------------------------------------------------------------------------
# 32: Byte-stable repeated evidence on toy inputs
# ---------------------------------------------------------------------------


def test_32_byte_stable_repeated_evaluation_evidence() -> None:
    ranks = [1, 4, 2, 20]
    cat_changed = [True, True, True, False]
    client_ids = ["c1", "c1", "c2", "c3"]

    s1 = evaluate_t1_ranks(ranks, cat_changed, client_ids)
    s2 = evaluate_t1_ranks(ranks, cat_changed, client_ids)

    d1 = json.dumps(s1.to_dict(), sort_keys=True)
    d2 = json.dumps(s2.to_dict(), sort_keys=True)
    assert d1 == d2
