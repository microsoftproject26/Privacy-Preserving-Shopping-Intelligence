"""The scale-aware float32 aggregation oracle: what it must accept and what it must reject.

The bound exists because `mvp-t1-001` failed a fixed 1e-6 tolerance on pure float32
accumulation noise. These tests hold it to both halves of that job: realistic noise at
fifty clients passes, and anything that is not float32 arithmetic still fails.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ppsi.federated.mvp_oracle import (
    ABSOLUTE_FLOOR,
    FLOAT32_EPS,
    ORACLE_POLICY_ID,
    AggregationOracleError,
    AggregationStructureError,
    allowed_abs_error,
    check_aggregation,
    policy_gamma,
    standard_gamma,
    weighted_abs_scale,
)


def _float32_aggregate(tensors, weights):
    """Exactly what Flower 1.33 does: float32 in-place accumulation of scaled tensors."""
    total = float(sum(weights))
    accumulated = None
    for tensor, weight in zip(tensors, weights, strict=True):
        scaled = tensor.numpy() * (weight / total)
        accumulated = scaled.copy() if accumulated is None else accumulated + scaled
    return torch.from_numpy(accumulated)


def _float64_oracle(tensors, weights):
    """The project's independent reference: float64 accumulation, cast back to float32."""
    total = float(sum(weights))
    accumulated = torch.zeros_like(tensors[0], dtype=torch.float64)
    for tensor, weight in zip(tensors, weights, strict=True):
        accumulated += tensor.to(torch.float64) * weight
    return (accumulated / total).to(tensors[0].dtype)


def _updates(tensors, weights, name="candidate_embeddings.weight"):
    return [({name: tensor}, int(weight)) for tensor, weight in zip(tensors, weights, strict=True)]


# --- A: realistic fifty-client accumulation noise must pass --------------------------


def test_identical_tensors_over_fifty_weighted_clients_pass(monkeypatch):
    """The exact shape of the mvp-t1-001 failure: an untrained head, fifty clients.

    Every client returns the same tensor, so the true weighted average is that tensor and
    any difference is arithmetic. A fixed 1e-6 tolerance rejected this; the derived bound
    must accept it.
    """
    torch.manual_seed(13)
    shared = (torch.randn(590, 8) * 1.2).to(torch.float32)
    weights = [1 + (index * 7) % 134 for index in range(50)]
    tensors = [shared.clone() for _ in weights]

    flower = _float32_aggregate(tensors, weights)
    oracle = _float64_oracle(tensors, weights)
    observed = float(torch.abs(flower - oracle).max())
    assert observed > 0.0, "this fixture is pointless if the arithmetic is exact"

    diagnostics = check_aggregation(
        updates=_updates(tensors, weights),
        oracle_state={"candidate_embeddings.weight": oracle},
        flower_state={"candidate_embeddings.weight": flower},
        server_round=1,
    )
    assert diagnostics["oracle_pass"] is True
    assert diagnostics["contributing_client_count"] == 50
    assert diagnostics["worst_ratio"] < 1.0
    assert diagnostics["oracle_policy_id"] == ORACLE_POLICY_ID


def test_a_realistic_mixture_of_trained_tensors_passes():
    """Clients that actually differ, aggregated the way Flower aggregates them."""
    torch.manual_seed(29)
    weights = [8 + index for index in range(50)]
    base = torch.randn(256, 64)
    tensors = [(base + torch.randn(256, 64) * 0.01).to(torch.float32) for _ in weights]
    flower = _float32_aggregate(tensors, weights)
    oracle = _float64_oracle(tensors, weights)
    diagnostics = check_aggregation(
        updates=_updates(tensors, weights, name="encoder.weight"),
        oracle_state={"encoder.weight": oracle},
        flower_state={"encoder.weight": flower},
        server_round=4,
    )
    assert diagnostics["oracle_pass"] is True
    assert diagnostics["passes_tight_diagnostic_bound"] is True


# --- B: a corrupted aggregate must fail ----------------------------------------------


def test_a_deliberately_corrupted_aggregate_fails():
    torch.manual_seed(13)
    weights = [10] * 50
    shared = torch.randn(64, 16).to(torch.float32)
    tensors = [shared.clone() for _ in weights]
    oracle = _float64_oracle(tensors, weights)
    corrupted = _float32_aggregate(tensors, weights).clone()
    corrupted[0, 0] += 1e-3
    with pytest.raises(AggregationOracleError, match="beyond float32 accumulation"):
        check_aggregation(
            updates=_updates(tensors, weights, name="encoder.weight"),
            oracle_state={"encoder.weight": oracle},
            flower_state={"encoder.weight": corrupted},
            server_round=2,
        )


def test_dropping_one_client_from_the_average_fails():
    """A real aggregation defect: the aggregate ignores one contributor."""
    torch.manual_seed(5)
    weights = [5 + index for index in range(50)]
    tensors = [torch.randn(32, 8).to(torch.float32) for _ in weights]
    oracle = _float64_oracle(tensors, weights)
    wrong = _float32_aggregate(tensors[:-1], weights[:-1])
    with pytest.raises(AggregationOracleError):
        check_aggregation(
            updates=_updates(tensors, weights, name="encoder.weight"),
            oracle_state={"encoder.weight": oracle},
            flower_state={"encoder.weight": wrong},
            server_round=3,
        )


# --- C: non-finite values must fail --------------------------------------------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_aggregate_fails(bad):
    weights = [4, 6]
    tensors = [torch.ones(3, 3), torch.ones(3, 3)]
    oracle = _float64_oracle(tensors, weights)
    broken = oracle.clone()
    broken[0, 0] = bad
    with pytest.raises(AggregationStructureError, match="not finite"):
        check_aggregation(
            updates=_updates(tensors, weights, name="encoder.weight"),
            oracle_state={"encoder.weight": oracle},
            flower_state={"encoder.weight": broken},
            server_round=1,
        )


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_client_tensor_fails(bad):
    weights = [4, 6]
    tensors = [torch.ones(3, 3), torch.ones(3, 3)]
    oracle = _float64_oracle(tensors, weights)
    poisoned = [tensors[0].clone(), tensors[1].clone()]
    poisoned[1][2, 2] = bad
    with pytest.raises(AggregationStructureError, match="not finite"):
        check_aggregation(
            updates=_updates(poisoned, weights, name="encoder.weight"),
            oracle_state={"encoder.weight": oracle},
            flower_state={"encoder.weight": oracle},
            server_round=1,
        )


def test_a_nan_is_never_excused_by_comparing_it_against_a_tolerance():
    """`NaN > tolerance` is False, so a naive comparison would pass it. This must not."""
    weights = [1, 1]
    tensors = [torch.ones(2, 2), torch.ones(2, 2)]
    oracle = _float64_oracle(tensors, weights)
    nan_aggregate = torch.full((2, 2), float("nan"))
    assert not bool(torch.abs(nan_aggregate - oracle).max() > 1e-6)
    with pytest.raises(AggregationStructureError):
        check_aggregation(
            updates=_updates(tensors, weights, name="encoder.weight"),
            oracle_state={"encoder.weight": oracle},
            flower_state={"encoder.weight": nan_aggregate},
            server_round=1,
        )


# --- D: key, shape and dtype mismatches must fail ------------------------------------


def test_a_missing_key_in_the_aggregate_fails():
    weights = [1, 1]
    tensors = [torch.ones(2, 2), torch.ones(2, 2)]
    oracle = {"a.weight": torch.ones(2, 2), "b.weight": torch.ones(2, 2)}
    with pytest.raises(AggregationStructureError, match="same parameters"):
        check_aggregation(
            updates=[({"a.weight": t, "b.weight": t}, w) for t, w in zip(tensors, weights)],
            oracle_state=oracle,
            flower_state={"a.weight": torch.ones(2, 2)},
            server_round=1,
        )


def test_a_client_with_a_different_key_set_fails():
    oracle = {"a.weight": torch.ones(2, 2)}
    with pytest.raises(AggregationStructureError, match="different key set"):
        check_aggregation(
            updates=[({"a.weight": torch.ones(2, 2)}, 1), ({"z.weight": torch.ones(2, 2)}, 1)],
            oracle_state=oracle,
            flower_state=oracle,
            server_round=1,
        )


def test_a_shape_mismatch_fails():
    oracle = {"a.weight": torch.ones(2, 2)}
    with pytest.raises(AggregationStructureError, match="shape mismatch"):
        check_aggregation(
            updates=[({"a.weight": torch.ones(2, 2)}, 1)],
            oracle_state=oracle,
            flower_state={"a.weight": torch.ones(3, 3)},
            server_round=1,
        )


def test_a_client_shape_mismatch_fails():
    oracle = {"a.weight": torch.ones(2, 2)}
    with pytest.raises(AggregationStructureError, match="wrong shape"):
        check_aggregation(
            updates=[({"a.weight": torch.ones(2, 2)}, 1), ({"a.weight": torch.ones(4, 4)}, 1)],
            oracle_state=oracle,
            flower_state=oracle,
            server_round=1,
        )


def test_a_dtype_mismatch_fails():
    oracle = {"a.weight": torch.ones(2, 2, dtype=torch.float32)}
    with pytest.raises(AggregationStructureError, match="dtype mismatch"):
        check_aggregation(
            updates=[({"a.weight": torch.ones(2, 2)}, 1)],
            oracle_state=oracle,
            flower_state={"a.weight": torch.ones(2, 2, dtype=torch.float64)},
            server_round=1,
        )


def test_a_client_dtype_mismatch_fails():
    oracle = {"a.weight": torch.ones(2, 2, dtype=torch.float32)}
    with pytest.raises(AggregationStructureError, match="wrong dtype"):
        check_aggregation(
            updates=[
                ({"a.weight": torch.ones(2, 2)}, 1),
                ({"a.weight": torch.ones(2, 2, dtype=torch.float64)}, 1),
            ],
            oracle_state=oracle,
            flower_state=oracle,
            server_round=1,
        )


@pytest.mark.parametrize("weight", [0, -3, True, 2.5])
def test_a_non_positive_or_non_integer_weight_fails(weight):
    oracle = {"a.weight": torch.ones(2, 2)}
    with pytest.raises(AggregationStructureError):
        check_aggregation(
            updates=[({"a.weight": torch.ones(2, 2)}, weight)],
            oracle_state=oracle,
            flower_state=oracle,
            server_round=1,
        )


def test_no_updates_at_all_fails():
    with pytest.raises(AggregationStructureError, match="no client updates"):
        check_aggregation(updates=[], oracle_state={}, flower_state={}, server_round=1)


# --- E: a difference materially larger than the bound must fail ----------------------


def test_a_difference_an_order_of_magnitude_past_the_bound_fails():
    weights = [20] * 50
    shared = torch.full((10, 10), 2.0)
    tensors = [shared.clone() for _ in weights]
    oracle = _float64_oracle(tensors, weights)
    allowance = allowed_abs_error(50, weighted_abs_scale(tensors, [1 / 50] * 50))
    drifted = oracle.clone()
    drifted[5, 5] += allowance * 10
    with pytest.raises(AggregationOracleError):
        check_aggregation(
            updates=_updates(tensors, weights, name="encoder.weight"),
            oracle_state={"encoder.weight": oracle},
            flower_state={"encoder.weight": drifted},
            server_round=1,
        )


def test_a_difference_just_inside_the_bound_passes_and_just_outside_fails():
    weights = [1] * 8
    shared = torch.full((4, 4), 1.0)
    tensors = [shared.clone() for _ in weights]
    oracle = _float64_oracle(tensors, weights)
    allowance = allowed_abs_error(8, weighted_abs_scale(tensors, [1 / 8] * 8))

    inside = oracle.clone()
    inside[0, 0] = float(oracle[0, 0]) + allowance * 0.9
    assert check_aggregation(
        updates=_updates(tensors, weights, name="w"),
        oracle_state={"w": oracle},
        flower_state={"w": inside},
        server_round=1,
    )["oracle_pass"]

    outside = oracle.clone()
    outside[0, 0] = float(oracle[0, 0]) + allowance * 4.0
    with pytest.raises(AggregationOracleError):
        check_aggregation(
            updates=_updates(tensors, weights, name="w"),
            oracle_state={"w": oracle},
            flower_state={"w": outside},
            server_round=1,
        )


# --- F: the bound stays tight at small scale -----------------------------------------


def test_two_client_small_tensors_stay_tightly_bounded():
    """The policy must not become a blanket loose tolerance at small n."""
    weights = [3, 5]
    tensors = [torch.full((4, 4), 0.5), torch.full((4, 4), 0.5)]
    scale = weighted_abs_scale(tensors, [3 / 8, 5 / 8])
    allowance = allowed_abs_error(2, scale)
    # Six roundings of a half-magnitude tensor: well under a ten-millionth.
    assert allowance < 5e-7
    oracle = _float64_oracle(tensors, weights)
    nudged = oracle.clone()
    nudged[1, 1] = float(oracle[1, 1]) + 1e-6
    with pytest.raises(AggregationOracleError):
        check_aggregation(
            updates=_updates(tensors, weights, name="w"),
            oracle_state={"w": oracle},
            flower_state={"w": nudged},
            server_round=1,
        )


def test_the_bound_grows_with_clients_and_with_magnitude():
    small = allowed_abs_error(2, 1.0)
    many = allowed_abs_error(50, 1.0)
    large = allowed_abs_error(2, 100.0)
    assert small < many
    assert small < large
    # Both factors are linear in the bound, so neither can be a hidden constant.
    assert many / small == pytest.approx(policy_gamma(50) / policy_gamma(2), rel=1e-9)
    assert large / small == pytest.approx(100.0, rel=1e-9)


def test_the_floor_only_applies_to_a_vanishing_scale():
    assert allowed_abs_error(50, 0.0) == ABSOLUTE_FLOOR
    assert allowed_abs_error(50, 1.0) > ABSOLUTE_FLOOR


def test_gamma_matches_its_declared_formula():
    for clients in (1, 2, 50, 1000):
        terms = (2 * clients + 2) * FLOAT32_EPS
        assert policy_gamma(clients) == pytest.approx(terms / (1 - terms), rel=1e-12)
    # The published diagnostic bound is genuinely tighter than the authorised one.
    for clients in (2, 50, 500):
        assert standard_gamma(clients) < policy_gamma(clients)


def test_gamma_rejects_nonsense_inputs():
    with pytest.raises(TypeError):
        policy_gamma(True)
    with pytest.raises(TypeError):
        policy_gamma(2.0)
    with pytest.raises(ValueError):
        policy_gamma(0)


def test_the_bound_uses_no_observed_error_or_quality_signal():
    """The allowance depends only on n, the weights and the tensor magnitudes."""
    weights = [7, 11, 13]
    tensors = [torch.full((3, 3), 2.0) for _ in weights]
    alphas = [w / sum(weights) for w in weights]
    scale = weighted_abs_scale(tensors, alphas)
    assert scale == pytest.approx(2.0, rel=1e-12)
    assert allowed_abs_error(3, scale) == pytest.approx(policy_gamma(3) * 2.0, rel=1e-12)


def test_diagnostics_report_every_required_field():
    weights = [4, 6]
    tensors = [torch.ones(2, 2), torch.ones(2, 2) * 2]
    oracle = _float64_oracle(tensors, weights)
    flower = _float32_aggregate(tensors, weights)
    diagnostics = check_aggregation(
        updates=_updates(tensors, weights, name="w"),
        oracle_state={"w": oracle},
        flower_state={"w": flower},
        server_round=7,
    )
    for field in (
        "server_round",
        "contributing_client_count",
        "total_aggregation_weight",
        "max_abs_diff",
        "allowed_abs_error",
        "worst_parameter",
        "worst_actual_abs_diff",
        "worst_allowed_abs_error",
        "worst_ratio",
        "oracle_policy_id",
        "float_dtype",
        "machine_epsilon",
        "gamma",
        "oracle_pass",
    ):
        assert field in diagnostics, field
    assert diagnostics["server_round"] == 7
    assert diagnostics["total_aggregation_weight"] == 10.0
    assert diagnostics["machine_epsilon"] == float(np.finfo(np.float32).eps)
    assert "client_id" not in str(diagnostics)
