"""A scale-aware float32 validity bound for the MVP's FedAvg aggregation oracle.

Why this module exists. The `mvp-t1-001` attempt failed its aggregation check at a fixed
absolute tolerance of 1e-6 on a parameter that no client had trained: T1 never reads the
candidate head, so all fifty clients returned that tensor unchanged and its exact weighted
average was the tensor itself. The whole observed difference was float32 accumulation
noise. A fixed absolute tolerance cannot tell that apart from a real disagreement, because
the noise floor of the operation grows with the number of contributing clients and with the
magnitude of the tensor being averaged.

So the bound here is derived, not chosen. It is a function of float32 machine epsilon, the
number of contributing clients, the aggregation weights and the tensor magnitudes. No
observed error and no model-quality number takes part in computing it.

**This is a numerical-validity policy only.** It relaxes nothing structural: a missing
reply, a duplicate client, a non-positive weight, a key, shape or dtype mismatch, a
non-finite value, or a difference genuinely outside the derived bound all still fail.

Derivation
----------
Flower 1.33 aggregates in float32. For every parameter element it computes

    acc = fl( sum_i  fl( alpha_i * x_ie ) ),    alpha_i = w_i / W

accumulating in place across the contributing replies. Each product costs one rounding and
the running sum costs one rounding per addition, so the standard forward-error result for a
multiply-accumulate of ``n`` terms (Higham, *Accuracy and Stability of Numerical
Algorithms*, Theorem 3.1) gives

    | fl(sum_i alpha_i x_ie) - sum_i alpha_i x_ie |  <=  gamma_n * sum_i alpha_i |x_ie|

with ``gamma_n = n*u / (1 - n*u)`` and unit roundoff ``u = eps/2``.

The independent oracle accumulates the same sum in float64 and casts the result back to
float32, which adds at most ``u * |S|`` and is itself bounded by the same scale term.

The authorised policy for this run counts each rounding at the full ``eps`` rather than at
``u``, and allows two further roundings for the weight computation and the final cast:

    gamma = ((2n + 2) * eps) / (1 - (2n + 2) * eps)

That is the bound this module enforces. The tighter standard bound above is computed
alongside it and published as a diagnostic, so a reader can see how much of the allowance
each round actually used and confirm the policy has not become a blanket loose tolerance.

The scale term is per tensor, taking each client's largest magnitude:

    weighted_abs_scale = sum_i alpha_i * max|x_i|

which upper-bounds ``sum_i alpha_i |x_ie|`` for every element ``e`` of that tensor.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

ORACLE_POLICY_ID = "SCALE_AWARE_FLOAT32_FORWARD_ERROR_V1"

# The smallest allowance any tensor gets, so an all-zero parameter still has a bound that
# is not exactly zero. It is a floor, never a safety factor on a nonzero bound.
ABSOLUTE_FLOOR = 1e-8

FLOAT32_EPS = float(np.finfo(np.float32).eps)


class AggregationOracleError(ValueError):
    """The Flower aggregate is not the weighted average, beyond float32 arithmetic."""


class AggregationStructureError(ValueError):
    """The replies or the aggregate are malformed; no numerical bound applies."""


def policy_gamma(contributing_clients: int, *, eps: float = FLOAT32_EPS) -> float:
    """The authorised relative accumulation allowance for ``n`` contributing clients."""
    if isinstance(contributing_clients, bool) or not isinstance(contributing_clients, int):
        raise TypeError("contributing_clients must be an integer")
    if contributing_clients < 1:
        raise ValueError("a bound needs at least one contributing client")
    terms = (2 * contributing_clients + 2) * eps
    if terms >= 1.0:
        raise ValueError("too many contributing clients for a meaningful float32 bound")
    return terms / (1.0 - terms)


def standard_gamma(contributing_clients: int, *, eps: float = FLOAT32_EPS) -> float:
    """The tighter textbook multiply-accumulate bound, published as a diagnostic.

    ``gamma_n = n*u / (1 - n*u)`` with unit roundoff ``u = eps/2``, plus the one rounding
    the float64 oracle spends casting its result back to float32.
    """
    unit_roundoff = eps / 2.0
    terms = contributing_clients * unit_roundoff
    if terms >= 1.0:
        raise ValueError("too many contributing clients for a meaningful float32 bound")
    return terms / (1.0 - terms) + unit_roundoff


def weighted_abs_scale(tensors: Sequence[torch.Tensor], alphas: Sequence[float]) -> float:
    """``sum_i alpha_i * max|x_i|`` for one parameter across the contributing clients."""
    if len(tensors) != len(alphas) or not tensors:
        raise AggregationStructureError("a scale needs one weight per contributing tensor")
    total = 0.0
    for tensor, alpha in zip(tensors, alphas, strict=True):
        largest = float(torch.abs(tensor.detach().to(torch.float64)).max().item())
        if not math.isfinite(largest):
            raise AggregationStructureError("a client tensor is not finite")
        total += alpha * largest
    return total


def allowed_abs_error(
    contributing_clients: int, scale: float, *, eps: float = FLOAT32_EPS
) -> float:
    """The largest difference float32 accumulation alone can explain for this tensor."""
    if not math.isfinite(scale) or scale < 0:
        raise AggregationStructureError("the tensor scale must be finite and non-negative")
    return max(ABSOLUTE_FLOOR, policy_gamma(contributing_clients, eps=eps) * scale)


@dataclass(frozen=True, slots=True)
class TensorOracleResult:
    """What the bound allowed for one parameter, and what was actually observed."""

    name: str
    actual_abs_diff: float
    allowed_abs_error: float
    weighted_abs_scale: float
    tight_allowed_abs_error: float

    @property
    def ratio(self) -> float:
        return self.actual_abs_diff / self.allowed_abs_error

    @property
    def passed(self) -> bool:
        return self.actual_abs_diff <= self.allowed_abs_error


def _alphas(weights: Sequence[int]) -> tuple[list[float], float]:
    if not weights:
        raise AggregationStructureError("no contributing client weights")
    for weight in weights:
        if isinstance(weight, bool) or not isinstance(weight, (int, np.integer)):
            raise AggregationStructureError("an aggregation weight is not an integer")
        if weight <= 0:
            raise AggregationStructureError("an aggregation weight is not positive")
    total = float(sum(int(weight) for weight in weights))
    return [int(weight) / total for weight in weights], total


def check_aggregation(
    *,
    updates: Sequence[tuple[Mapping[str, torch.Tensor], int]],
    oracle_state: Mapping[str, torch.Tensor],
    flower_state: Mapping[str, torch.Tensor],
    server_round: int,
) -> dict[str, Any]:
    """Validate the real Flower aggregate against the float64 oracle, tensor by tensor.

    Structure is checked before any number is compared: a key, shape or dtype mismatch, a
    non-positive weight or a non-finite value is a structural failure, not something a
    floating-point allowance may excuse.
    """
    if not updates:
        raise AggregationStructureError("no client updates to aggregate")
    weights = [weight for _, weight in updates]
    alphas, total_weight = _alphas(weights)
    contributing = len(updates)

    reference_keys = sorted(oracle_state)
    if sorted(flower_state) != reference_keys:
        raise AggregationStructureError(
            "the Flower aggregate and the oracle do not describe the same parameters"
        )
    for state, _ in updates:
        if sorted(state) != reference_keys:
            raise AggregationStructureError("a client reply carries a different key set")

    results: list[TensorOracleResult] = []
    for name in reference_keys:
        oracle_tensor = oracle_state[name]
        flower_tensor = flower_state[name]
        if tuple(flower_tensor.shape) != tuple(oracle_tensor.shape):
            raise AggregationStructureError(f"shape mismatch for {name}")
        if flower_tensor.dtype != oracle_tensor.dtype:
            raise AggregationStructureError(f"dtype mismatch for {name}")
        if not bool(torch.isfinite(flower_tensor).all()):
            raise AggregationStructureError(f"the Flower aggregate of {name} is not finite")
        if not bool(torch.isfinite(oracle_tensor).all()):
            raise AggregationStructureError(f"the oracle aggregate of {name} is not finite")

        client_tensors = []
        for state, _ in updates:
            tensor = state[name]
            if tuple(tensor.shape) != tuple(oracle_tensor.shape):
                raise AggregationStructureError(f"a client tensor has the wrong shape: {name}")
            if tensor.dtype != oracle_tensor.dtype:
                raise AggregationStructureError(f"a client tensor has the wrong dtype: {name}")
            if not bool(torch.isfinite(tensor).all()):
                raise AggregationStructureError(f"a client tensor is not finite: {name}")
            client_tensors.append(tensor)

        scale = weighted_abs_scale(client_tensors, alphas)
        difference = float(
            torch.abs(
                flower_tensor.detach().to(torch.float64) - oracle_tensor.detach().to(torch.float64)
            )
            .max()
            .item()
        )
        if not math.isfinite(difference):
            raise AggregationStructureError(f"the aggregation difference for {name} is not finite")
        results.append(
            TensorOracleResult(
                name=name,
                actual_abs_diff=difference,
                allowed_abs_error=allowed_abs_error(contributing, scale),
                weighted_abs_scale=scale,
                tight_allowed_abs_error=max(ABSOLUTE_FLOOR, standard_gamma(contributing) * scale),
            )
        )

    worst = max(results, key=lambda item: item.ratio)
    failures = [item for item in results if not item.passed]
    diagnostics = {
        "schema": "mvp_aggregation_oracle_v1",
        "version": "1",
        "server_round": int(server_round),
        "oracle_policy_id": ORACLE_POLICY_ID,
        "float_dtype": "float32",
        "machine_epsilon": FLOAT32_EPS,
        "contributing_client_count": contributing,
        "total_aggregation_weight": total_weight,
        "gamma": policy_gamma(contributing),
        "tight_gamma_diagnostic": standard_gamma(contributing),
        "parameters_checked": len(results),
        "max_abs_diff": max(item.actual_abs_diff for item in results),
        "allowed_abs_error": max(item.allowed_abs_error for item in results),
        "worst_parameter": worst.name,
        "worst_actual_abs_diff": worst.actual_abs_diff,
        "worst_allowed_abs_error": worst.allowed_abs_error,
        "worst_weighted_abs_scale": worst.weighted_abs_scale,
        "worst_ratio": worst.ratio,
        "worst_tight_allowed_abs_error": worst.tight_allowed_abs_error,
        "worst_tight_ratio": worst.actual_abs_diff / worst.tight_allowed_abs_error,
        "passes_tight_diagnostic_bound": all(
            item.actual_abs_diff <= item.tight_allowed_abs_error for item in results
        ),
        "oracle_pass": not failures,
        "note": (
            "Bound derived from float32 epsilon, contributing client count, aggregation "
            "weights and tensor magnitudes. No observed error or quality metric takes part."
        ),
    }
    if failures:
        worst_failure = max(failures, key=lambda item: item.ratio)
        raise AggregationOracleError(
            f"round {server_round}: {len(failures)} parameter(s) differ beyond float32 "
            f"accumulation. Worst is {worst_failure.name} with "
            f"{worst_failure.actual_abs_diff:.6e} against an allowance of "
            f"{worst_failure.allowed_abs_error:.6e} "
            f"(ratio {worst_failure.ratio:.3f}, {contributing} contributing clients)."
        )
    return diagnostics
