"""Conservative tree-restart zCDP accounting, without sampling amplification.

This accounts for a fixed public roster, at most one contribution per user per
sweep, and independent Gaussian noise on binary-tree nodes. It does not account
for earlier clipping/recipe selection or unnoised side outputs. Implementation
and run-level independent concurrence remain separate from these calculations.

Reference: tensorflow/privacy, analysis/tree_aggregation_accountant.py,
compute_rdp_tree_restart; Kairouz et al. (2021), Appendix D.
"""
from __future__ import annotations

import math
from numbers import Integral, Real


def integer(value, label):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be an integer")
    return int(value)


def finite(value, label, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a real number")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{label} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


def tree_depth(n_steps):
    """Max exposed dyadic nodes per leaf; powers of two include their root."""
    steps = integer(n_steps, "n_steps")
    if steps < 0:
        raise ValueError("n_steps must be nonnegative")
    return steps.bit_length()


def prefix_intervals(n_steps):
    """Disjoint canonical dyadic intervals forming the prefix [0,n_steps)."""
    tree_depth(n_steps)
    remaining, start, result = int(n_steps), 0, []
    while remaining:
        size = 1 << (remaining.bit_length() - 1)
        result.append((start, start + size))
        start += size
        remaining -= size
    return tuple(result)


def restart_depth_sum(steps_per_sweep):
    if isinstance(steps_per_sweep, Integral) and not isinstance(steps_per_sweep, bool):
        steps_per_sweep = (steps_per_sweep,)
    try:
        steps = tuple(steps_per_sweep)
    except TypeError as exc:
        raise ValueError("steps_per_sweep must be an integer or a nonempty iterable") from exc
    if not steps:
        raise ValueError("steps_per_sweep must be nonempty")
    return sum(tree_depth(s) for s in steps)


def adjacency_factor(adjacency):
    if adjacency == "add_remove_zero_slot":
        return 1
    if adjacency == "replace_content":
        return 2
    raise ValueError("adjacency must be add_remove_zero_slot or replace_content")


def restart_zcdp(noise_multiplier, steps_per_sweep, *, adjacency="add_remove_zero_slot"):
    """Noise multiplier is sigma/S, not sigma/(replacement sensitivity)."""
    z = finite(noise_multiplier, "noise_multiplier")
    depth = restart_depth_sum(steps_per_sweep)
    factor = adjacency_factor(adjacency)
    if depth == 0:
        return 0.0
    if z == 0:
        return math.inf
    scaled = (factor / z) * math.sqrt(depth / 2)
    return scaled * scaled  # overflow gives conservative +inf, never a finite underestimate


def epsilon_from_zcdp(rho, delta):
    if isinstance(rho, bool) or not isinstance(rho, Real) or math.isnan(float(rho)) or rho < 0:
        raise ValueError("rho must be nonnegative")
    delta = finite(delta, "delta", positive=True)
    if delta >= 1:
        raise ValueError("delta must be in (0,1)")
    rho = float(rho)
    return rho + 2 * math.sqrt(rho) * math.sqrt(-math.log(delta))


def rho_for_epsilon(epsilon, delta):
    eps = finite(epsilon, "epsilon")
    epsilon_from_zcdp(0, delta)
    log_inverse_delta = -math.log(float(delta))
    # Rationalized form avoids catastrophic cancellation for small epsilon.
    return (eps / (math.sqrt(log_inverse_delta + eps) + math.sqrt(log_inverse_delta))) ** 2


def noise_multiplier_for_epsilon(epsilon, delta, steps_per_sweep, *, adjacency="add_remove_zero_slot"):
    eps = finite(epsilon, "epsilon", positive=True)
    rho = rho_for_epsilon(eps, delta)
    depth = restart_depth_sum(steps_per_sweep)
    factor = adjacency_factor(adjacency)
    if depth == 0:
        return 0.0
    if rho == 0:
        return math.inf  # representable epsilon is too small for finite noise
    return factor * math.sqrt(depth / (2 * rho))


def rdp_at_order(rho, order):
    if isinstance(order, bool) or not isinstance(order, Real) or math.isnan(float(order)) or order <= 1:
        raise ValueError("RDP order must be greater than one")
    if isinstance(rho, bool) or not isinstance(rho, Real) or math.isnan(float(rho)) or rho < 0:
        raise ValueError("rho must be nonnegative")
    return 0.0 if rho == 0 else float(order) * float(rho)


def describe_budget(noise_multiplier, steps_per_sweep, delta, *, adjacency="add_remove_zero_slot"):
    """Public, input-independent calculation; contains no success certification."""
    steps = ((integer(steps_per_sweep, "steps"),) if isinstance(steps_per_sweep, Integral)
             else tuple(steps_per_sweep))
    rho = restart_zcdp(noise_multiplier, steps, adjacency=adjacency)
    return {"steps_per_sweep": list(steps), "depths": [tree_depth(s) for s in steps],
            "maximum_exposed_nodes_per_user": restart_depth_sum(steps),
            "noise_multiplier_sigma_over_S": float(noise_multiplier), "adjacency": adjacency,
            "rho": rho, "epsilon": epsilon_from_zcdp(rho, delta), "delta": float(delta),
            "amplification_used": False, "privacy_certified": False,
            "scope": "FIXED_CONFIGURATION_MODEL_PREFIX_MECHANISM_ONLY",
            "prior_private_tuning_and_side_outputs_accounted": False}
