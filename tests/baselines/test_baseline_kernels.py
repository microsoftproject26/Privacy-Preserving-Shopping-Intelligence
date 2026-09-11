"""Independent numerical/lineage checks; no private data and no model-quality claims."""

import copy

import numpy as np
import pytest
from scipy import sparse

from ppsi.baselines.classical import FEATURES, feature_matrix, make_pipeline, positive_probabilities
from ppsi.baselines.session_knn import SessionKNN
from ppsi.baselines.t2_metrics import evaluate_t2


def knn():
    return SessionKNN(category_count=4, candidate_limit=20).fit(
        ["A", "B", "C"],
        [10, 20, 30],
        [[1, 2], [2, 3], [4]],
        [[0, 1], [1, 2], [3]],
        [4, 3, 2, 1],
        split="TRAIN",
    )


def test_hand_worked_cosine_and_category_votes():
    scores, diag = knn().score_many_k([1, 2], [1, 2])
    np.testing.assert_allclose(scores[1], [1, 1, 0, 0])
    np.testing.assert_allclose(scores[2], [1, 1.5, 0.5, 0])
    assert diag == {"backoff": False, "candidate_count": 2, "neighbor_count": 2}


def test_neighbor_ties_newest_then_stable_id():
    scores, _ = knn().score_many_k([2], [1])
    np.testing.assert_allclose(scores[1], [0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0])
    model = SessionKNN(category_count=2).fit(
        ["B", "A"], [1, 1], [[2], [2]], [[1], [0]], [1, 1], split="TRAIN"
    )
    assert model.score_many_k([2], [1])[0][1].tolist() == [1, 0]


def test_binary_item_representation_deduplicates_query_and_sessions():
    model = knn()
    a = model.score_many_k([1, 2], [2])[0][2]
    b = model.score_many_k([1, 1, 2, 2], [2])[0][2]
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("query", [[], [999]])
def test_empty_or_no_overlap_uses_training_popularity(query):
    scores, diag = knn().score_many_k(query, [5])
    np.testing.assert_allclose(scores[5], [0.4, 0.3, 0.2, 0.1])
    assert diag["backoff"] is True


def test_unseen_item_remains_in_query_norm_not_silently_removed():
    scores, _ = knn().score_many_k([1, 999], [1])
    np.testing.assert_allclose(scores[1], [0.5, 0.5, 0, 0])


def test_fewer_than_k_is_valid():
    scores, diag = knn().score_many_k([4], [100])
    np.testing.assert_array_equal(scores[100], [0, 0, 0, 1])
    assert diag["neighbor_count"] == 1


@pytest.mark.parametrize("split", ["VALIDATION", "TEST"])
def test_nontraining_index_refused(split):
    with pytest.raises(ValueError, match="TRAIN only"):
        SessionKNN().fit(["A"], [1], [[1]], [[0]], np.ones(588), split=split)


@pytest.mark.parametrize("bad", [[True], [1.5], [-1]])
def test_query_id_coercions_rejected(bad):
    with pytest.raises(ValueError, match="integer IDs"):
        knn().score_many_k(bad, [1])


def test_index_not_mutated_by_queries():
    model = knn()
    before = (model.items_.data.copy(), model.items_.indices.copy(), copy.deepcopy(model.postings_))
    for query in ([1, 3], [999], [], [2, 4]):
        model.score_many_k(query, [1, 2])
    np.testing.assert_array_equal(before[0], model.items_.data)
    np.testing.assert_array_equal(before[1], model.items_.indices)
    for key in before[2]:
        np.testing.assert_array_equal(before[2][key], model.postings_[key])


def test_recent_candidate_limit_matches_brute_force_oracle():
    rng = np.random.default_rng(13)
    ids = [f"s{i:03d}" for i in range(50)]
    times = rng.integers(0, 10, 50).tolist()
    items = [rng.choice(30, size=6, replace=False).tolist() for _ in ids]
    cats = [rng.choice(8, size=3, replace=False).tolist() for _ in ids]
    model = SessionKNN(category_count=8, candidate_limit=9).fit(
        ids, times, items, cats, np.ones(8), split="TRAIN"
    )
    for _ in range(25):
        q = set(rng.choice(40, size=4, replace=False).tolist())
        candidates = sorted(
            [i for i in range(50) if q.intersection(items[i])], key=lambda i: (-times[i], ids[i])
        )[:9]
        similarities = {
            i: len(q.intersection(items[i])) / np.sqrt(len(q) * len(set(items[i])))
            for i in candidates
        }
        selected = sorted(candidates, key=lambda i: (-similarities[i], -times[i], ids[i]))[:5]
        expected = np.zeros(8)
        for i in selected:
            for c in set(cats[i]):
                expected[c] += similarities[i]
        actual, _ = model.score_many_k(sorted(q), [5])
        np.testing.assert_allclose(actual[5], expected, atol=1e-14)


def columns(n=240):
    rng = np.random.default_rng(13)
    result = {name: rng.uniform(0, 2, n) for name in FEATURES}
    result["category_code"] = np.arange(n) % 5
    result["weekend"] = np.arange(n) % 2
    return result


@pytest.mark.parametrize(
    "field", ["label_value", "label_matures_at", "session_end", "category_changed", "client"]
)
def test_future_or_identity_predictors_cannot_reach_estimator(field):
    values = columns()
    values[field] = np.zeros(240)
    with pytest.raises(ValueError, match="whitelist"):
        feature_matrix(values)


@pytest.mark.parametrize(
    "model,params",
    [("logistic_regression", {"C": 1}), ("lightgbm", {"num_leaves": 15, "reg_lambda": 10})],
)
def test_models_fit_real_pipelines_and_repeat_on_toy_data(model, params):
    x = feature_matrix(columns())
    y = (x[:, 1] > 1).astype(int)
    first = make_pipeline(model, params, threads=1)
    second = make_pipeline(model, params, threads=1)
    first.fit(x, y)
    second.fit(x, y)
    assert sparse.issparse(first.named_steps["preprocess"].transform(x))
    np.testing.assert_allclose(
        positive_probabilities(first, x), positive_probabilities(second, x), atol=1e-12
    )


def test_validation_cannot_fit_transformers_unseen_and_all_missing_supported():
    values = columns()
    values["log_query_price"][:] = np.nan
    x = feature_matrix(values)
    model = make_pipeline("logistic_regression", {"C": 1})
    y = (x[:, 2] > 1).astype(int)
    model.fit(x, y)
    scaler = model.named_steps["preprocess"].named_transformers_["numeric"].named_steps["scaler"]
    before = scaler.mean_.copy()
    validation = x[:10].copy()
    validation[:, 0] = -1
    validation[:, 2] = 1e8
    p = positive_probabilities(model, validation)
    assert np.isfinite(p).all()
    np.testing.assert_array_equal(scaler.mean_, before)


def test_t2_ap_manual_oracle_and_client_companion():
    # Rankings are [positive, negative, positive, negative]: AP = (1 + 2/3)/2.
    out = evaluate_t2([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.6], [1, 1, 2, 2], [0, 1, 2, 3])
    assert out["pr_auc_micro"] == pytest.approx(5 / 6)
    assert out["macro_pr_auc"] is None
    assert out["client_recall_at_3_macro"] == 1


def test_t2_top3_uses_client_vote_not_purchase_weight():
    y = [1, 1, 1, 1, 0, 1]
    out = evaluate_t2(y, [0.9, 0.8, 0.7, 0.6, 0.5, 0.4], [1, 1, 1, 1, 1, 2], range(6))
    assert out["client_recall_at_3_macro"] == pytest.approx((3 / 4 + 1) / 2)
    assert out["client_recall_at_3_client_unit_mean"] == out["client_recall_at_3_macro"]


def test_t2_ties_follow_frozen_row_order_not_current_input_order():
    y, p, ids, rows = map(np.asarray, ([0, 0, 0, 1], [0.5] * 4, [1] * 4, [0, 1, 2, 3]))
    first = evaluate_t2(y, p, ids, rows)
    take = [3, 2, 1, 0]
    second = evaluate_t2(y[take], p[take], ids[take], rows[take])
    assert first == second
    assert first["client_recall_at_3_macro"] == 0
    assert first["pr_auc_micro"] == 0.25


def test_t2_no_positives_is_not_fabricated_zero_score():
    out = evaluate_t2([0, 0], [0.1, 0.2], [1, 2], [0, 1])
    assert out["pr_auc_micro"] is None
    assert out["client_recall_at_3_macro"] is None


@pytest.mark.parametrize("probabilities", [[0.1, np.nan], [np.inf, 0.1], [-0.1, 0.1]])
def test_t2_bad_probabilities_fail(probabilities):
    with pytest.raises(ValueError):
        evaluate_t2([0, 1], probabilities, [1, 1], [0, 1])
