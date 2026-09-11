"""Explicit isolated-pandas checks, NOT auto-collected by the project pytest suite.

Run this exact file using the documented pinned preprocessing test environment.
No private files, raw-data downloads or TEST rows are used.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "baseline_preparation", ROOT / "scripts/baselines/prepare_baseline_views.py"
)
prep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prep)


def events():
    return pd.DataFrame(
        {
            "session": [10, 10, 20, 20],
            "client": [1, 1, 2, 2],
            "item": [100, 100, 200, 201],
            "category_code": [3, 3, 4, 5],
            "event_code": [0, 1, 0, 2],
            "event_time": pd.to_datetime(
                [
                    "2019-10-01T01:00:00Z",
                    "2019-10-01T01:00:00Z",
                    "2019-10-01T02:00:00Z",
                    "2019-10-01T02:01:00Z",
                ]
            ),
            "source_row_number": [100, 101, 200, 201],
            "price": [20.0, 20.0, 30.0, 40.0],
        }
    )


def example(task="t2"):
    frame = pd.DataFrame(
        {
            "client": [2],
            "session": [20],
            "decision_order": [2],
            "item": [200],
            "category": [4],
            "current_category": [4],
            "label_value": [0.0],
            "task_mask": [True],
            "status": ["OBSERVED"],
            "cohort": ["C1"],
            "split": ["TRAIN"],
        }
    )
    return frame


def test_upstream_session_hash_golden_vectors():
    frame = pd.DataFrame(
        {
            "user_id": [101, 101, 102],
            "user_session": ["abc", None, None],
            "source_row_number": [1, 2, 3],
        }
    )
    assert prep.session_keys(frame, "__NULL_SINGLETON__").tolist() == [
        5535697119745933737,
        11303999540078230511,
        5515013581191373333,
    ]


def test_null_sessions_are_distinct_provenance_singletons():
    frame = pd.DataFrame(
        {"user_id": [101, 101], "user_session": [None, None], "source_row_number": [2, 3]}
    )
    assert prep.session_keys(frame, "__NULL_SINGLETON__").nunique() == 2


def test_sessions_crossing_batches_are_assembled_before_global_order():
    frame = events()
    out = prep.canonicalize([frame.iloc[[3, 0]], frame.iloc[[2, 1]]], set())
    assert out["source_row_number"].tolist() == [100, 101, 200, 201]
    assert out["order"].tolist() == [0, 1, 2, 3]
    assert prep.bind_examples(out, example(), "t2").tolist() == [2]


def test_per_session_cumcount_cannot_replace_global_decision_order():
    frame = prep.canonicalize([events()], set())
    bad = example()
    bad["decision_order"] = [0]
    with pytest.raises(ValueError, match="binding"):
        prep.bind_examples(frame, bad, "t2")


def test_excluded_sessions_consumed_not_recomputed():
    frame = prep.canonicalize([events()], {10})
    assert frame["session"].tolist() == [20, 20]
    assert frame["order"].tolist() == [0, 1]


def test_same_timestamp_orders_by_source_row_number():
    frame = prep.canonicalize([events().iloc[::-1]], set())
    assert frame["source_row_number"].iloc[:2].tolist() == [100, 101]


def test_duplicate_provenance_rejected():
    frame = events()
    with pytest.raises(ValueError, match="duplicate raw provenance"):
        prep.canonicalize([frame, frame.iloc[[0]]], set())


def test_prefix_features_do_not_include_current_anchor():
    frame = prep.canonicalize([events()], set())
    features = prep.t2_predictors(frame)
    assert features.loc[0, "log_prior_events"] == 0
    assert features.loc[0, "log_prior_views"] == 0
    assert features.loc[1, "log_prior_views"] == pytest.approx(np.log(2))
    assert features.loc[1, "log_prior_carts"] == 0
    assert features.loc[1, "log_same_item_prior_events"] == pytest.approx(np.log(2))


def test_future_change_cannot_change_past_features():
    frame = prep.canonicalize([events()], set())
    before = prep.t2_predictors(frame).iloc[2].to_numpy()
    frame.loc[3, "price"] = 999999.0
    frame.loc[3, "event_code"] = 1
    frame.loc[3, "item"] = 999
    after = prep.t2_predictors(frame).iloc[2].to_numpy()
    np.testing.assert_array_equal(before, after)


def test_observed_binary_labels_and_censored_nulls():
    frame = example()
    prep.validate_examples(frame, "t2", "TRAIN")
    frame["task_mask"] = False
    frame["status"] = "CENSORED"
    frame["label_value"] = np.nan
    prep.validate_examples(frame, "t2", "TRAIN")
    frame["label_value"] = 0.0
    with pytest.raises(ValueError, match="censored labels"):
        prep.validate_examples(frame, "t2", "TRAIN")


@pytest.mark.parametrize("label", [None, np.inf, 0.5, 2, -1])
def test_invalid_t2_observed_labels_fail(label):
    frame = example()
    frame["label_value"] = label
    with pytest.raises(ValueError, match="invalid observed label"):
        prep.validate_examples(frame, "t2", "TRAIN")


def test_null_required_fields_do_not_pass_boolean_comparisons():
    frame = example()
    frame["split"] = None
    with pytest.raises(ValueError, match="null"):
        prep.validate_examples(frame, "t2", "TRAIN")


def test_integer_category_ids_above_float_precision_are_not_rounded():
    raw = pd.Series([2053013554969444627, 2053013554969444628], dtype="int64")
    mapping = {str(raw.iloc[0]): 0, str(raw.iloc[1]): 1}
    assert prep.dense_categories(raw, mapping).tolist() == [0, 1]
    raw_str = pd.Series([str(raw.iloc[0]), str(raw.iloc[1])])
    assert prep.dense_categories(raw_str, mapping).tolist() == [0, 1]
    with pytest.raises(ValueError, match="integers"):
        prep.dense_categories(raw.astype(float), mapping)
