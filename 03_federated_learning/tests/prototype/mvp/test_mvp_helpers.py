"""Run on the repository after installing the supplied helpers; no substitute for real data integration."""

import math

import numpy as np
import pytest
import torch

from ppsi.baselines import t1_simple as actual_counts
from ppsi.federated import mvp_support as actual_support
from ppsi.training.t1_mvp_objective import masked_t1_sum


def model(counts):
    return counts.T1CountBaselines.fit(
        np.array([0, 0, 1, 1, 1]), np.array([1, 2, 0, 1, 2]), split="TRAIN", category_count=4
    )


def test_hand_worked_counts(counts):
    m = model(counts)
    assert m.target_counts.tolist() == [1, 2, 2, 0]
    assert m.transition_counts.tolist() == [[0, 1, 1, 0], [1, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    assert m.train_decisions == 5


@pytest.mark.parametrize("split", ["VALIDATION", "TEST", "train", "", None])
def test_train_only(counts, split):
    with pytest.raises(ValueError):
        counts.T1CountBaselines.fit(np.array([0]), np.array([1]), split=split, category_count=4)


@pytest.mark.parametrize(
    "current,targets",
    [
        ([0.0], [1]),
        ([True], [1]),
        ([0], [-1]),
        ([0], [4]),
        ([-1], [1]),
        ([4], [1]),
        ([0, 1], [1]),
        ([], []),
        ([[0]], [1]),
        (["0"], [1]),
    ],
)
def test_invalid_fit(counts, current, targets):
    with pytest.raises(ValueError):
        counts.T1CountBaselines.fit(
            np.array(current), np.array(targets), split="TRAIN", category_count=4
        )


@pytest.mark.parametrize("variant", ["popularity", "markov", "last_category"])
def test_score_shape_and_finiteness(counts, variant):
    scores = model(counts).scores(variant, np.array([-1, 0, 2, 4]))
    assert scores.shape == (4, 4)
    assert np.isfinite(scores).all()


def test_unknown_markov_backoff_zero(counts):
    assert not model(counts).scores("markov", np.array([-1, 2, 4])).any()


def test_last_category_one_hot(counts):
    assert model(counts).scores("last_category", np.array([0, 1, -1])).tolist() == [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
    ]


def test_last_category_slice_ties_not_force_zero(counts):
    scores = model(counts).scores("last_category", np.array([0]))[0]
    rank = list(np.lexsort((np.arange(4), -scores))).index(1) + 1
    assert rank == 2  # A distinct target can receive credit under raw full-universe ties.


def test_count_identity_order_independent(counts):
    x, y = np.array([0, 0, 1, 1, 1]), np.array([1, 2, 0, 1, 2])
    a = counts.T1CountBaselines.fit(x, y, split="TRAIN", category_count=4)
    b = counts.T1CountBaselines.fit(x[::-1], y[::-1], split="TRAIN", category_count=4)
    assert a.content_sha256() == b.content_sha256()
    with pytest.raises(ValueError):
        a.target_counts[0] = 99


def test_unknown_variant(counts):
    with pytest.raises(ValueError):
        model(counts).scores("suppressed", np.array([0]))


def test_hash_conventions(support, tmp_path):
    a, b, c = (tmp_path / n for n in ("a.py", "b.py", "c.py"))
    a.write_bytes(b"line\nnext\n")
    b.write_bytes(b"\xef\xbb\xbfline\r\nnext\r\n")
    c.write_bytes(b"line \nnext\n")
    assert support.canonical_text_sha256(a) == support.canonical_text_sha256(b)
    assert support.raw_file_sha256(a) != support.raw_file_sha256(b)
    assert support.canonical_text_sha256(a) != support.canonical_text_sha256(c)


@pytest.mark.parametrize("ids", [[], ["a", "a"], ["123"], [""], [1]])
def test_unique_client_guard(support, ids):
    with pytest.raises(ValueError):
        support.require_unique_clients(ids)


def test_client_order_is_complete_and_stable(support):
    rows = np.arange(120)
    a = support.client_epoch_order(rows, seed=13, server_round=1, client_id="opaque-a")
    b = support.client_epoch_order(rows[::-1], seed=13, server_round=1, client_id="opaque-a")
    c = support.client_epoch_order(rows, seed=13, server_round=2, client_id="opaque-a")
    assert np.array_equal(a, b)
    assert np.array_equal(np.sort(a), rows)
    assert not np.array_equal(a, c)


@pytest.mark.parametrize("rows", [[], [1, 1], [-1, 1], [1.0, 2.0]])
def test_bad_local_rows_rejected(support, rows):
    with pytest.raises(ValueError):
        support.client_epoch_order(np.asarray(rows), seed=13, server_round=1, client_id="opaque-a")


def test_exposure_digest_checks_order(support):
    a = [(1, "opaque-a", ["session:10", "session:11"])]
    b = [(1, "opaque-a", ["session:11", "session:10"])]
    assert support.exposure_digest(a) != support.exposure_digest(b)
    with pytest.raises(ValueError):
        support.exposure_digest(a + a)


@pytest.mark.parametrize("received", [["a"], ["a", "a"], ["a", "c"], []])
def test_no_partial_or_duplicate_reply(support, received):
    with pytest.raises(ValueError):
        support.require_complete_replies(["a", "b"], received)


def test_order_of_replies_not_identity(support):
    support.require_complete_replies(["a", "b"], ["b", "a"])


@pytest.mark.parametrize("bad", [None, True, "0.3", float("nan"), float("inf")])
def test_invalid_metric(support, bad):
    with pytest.raises(ValueError):
        support.require_finite_metric(bad)


def test_masked_ce_golden_and_gradient(loss_kernel):
    logits = torch.zeros(3, 2, requires_grad=True)
    targets = torch.tensor([0, -999, 1], dtype=torch.int64)
    present = torch.tensor([True, False, True])
    numerator, count = loss_kernel(logits, targets, present)
    assert count == 2
    assert abs(numerator.item() - 2 * math.log(2)) < 1e-6
    (numerator / count).backward()
    assert torch.equal(logits.grad[1], torch.zeros(2))
    assert torch.isfinite(logits.grad).all()


def test_absent_target_not_read(loss_kernel):
    numerator, support = loss_kernel(
        torch.zeros(2, 4), torch.tensor([-88, 99]), torch.tensor([False, False])
    )
    assert numerator is None and support == 0


@pytest.mark.parametrize("target", [-1, 4])
def test_present_target_out_of_range(loss_kernel, target):
    with pytest.raises(ValueError):
        loss_kernel(torch.zeros(1, 4), torch.tensor([target]), torch.tensor([True]))


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_logits_fail(loss_kernel, bad):
    with pytest.raises(ValueError):
        loss_kernel(torch.tensor([[bad, 1.0]]), torch.tensor([1]), torch.tensor([True]))


def test_nonboolean_mask_fails(loss_kernel):
    with pytest.raises(ValueError):
        loss_kernel(torch.zeros(1, 2), torch.tensor([0]), torch.tensor([1]))


def test_model_rng_independent_of_worker_and_distinct_by_client(support):
    a = support.client_rng_seed(seed=13, server_round=1, client_id="opaque-a")
    b = support.client_rng_seed(seed=13, server_round=1, client_id="opaque-a")
    c = support.client_rng_seed(seed=13, server_round=1, client_id="opaque-b")
    assert a == b and a != c and 0 <= a < 2**63


def test_direct_counts_reject_float_support(counts):
    with pytest.raises(ValueError):
        counts.T1CountBaselines(np.array([1, 0]), np.array([[1, 0], [0, 0]]), 1.0)


def identity():
    keys = [
        "model_sha256",
        "batch_spec_sha256",
        "common_init_sha256",
        "data_manifest_sha256",
        "evaluation_membership_sha256",
        "evaluator_sha256",
        "exposure_sha256",
    ]
    record = {k: "a" * 64 for k in keys}
    record.update(
        seed=13,
        metric_id="t1.next_distinct.mrr_at_20.macro",
        score_convention="RAW_NO_SUPPRESSION",
        validation_decisions=10,
        value=0.2,
    )
    return record


def test_metric_delta_requires_comparable_evidence(support):
    a, b = identity(), identity()
    b["value"] = 0.3
    assert abs(support.metric_delta(a, b) - 0.1) < 1e-12
    b["exposure_sha256"] = "b" * 64
    with pytest.raises(ValueError):
        support.metric_delta(a, b)


@pytest.mark.parametrize(
    "key,value",
    [
        ("common_init_sha256", None),
        ("model_sha256", "unknown"),
        ("seed", True),
        ("validation_decisions", 0),
        ("score_convention", "SUPPRESSED"),
    ],
)
def test_no_placeholder_comparison_identity(support, key, value):
    a, b = identity(), identity()
    a[key] = b[key] = value
    with pytest.raises(ValueError):
        support.require_comparison_identity(a, b)


@pytest.fixture(scope="module")
def counts():
    return actual_counts


@pytest.fixture(scope="module")
def support():
    return actual_support


@pytest.fixture(scope="module")
def loss_kernel():
    return masked_t1_sum
