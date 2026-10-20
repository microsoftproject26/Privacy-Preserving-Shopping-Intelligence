"""Independent small-tree geometry and conservative accounting controls."""
import math
import numpy as np
import pytest

from . import dpftrl_statistics as S


@pytest.mark.parametrize("rounds", [0, 1, 2, 3, 4, 5, 8, 13, 16, 32, 512, 2048])
def test_depth_matches_bruteforce_exposed_nodes(rounds):
    # One fresh node ends at each released prefix; its size is the low set bit.
    nodes = [(t - (t & -t), t) for t in range(1, rounds + 1)]
    multiplicities = [sum(a <= leaf < b for a, b in nodes) for leaf in range(rounds)]
    assert max(multiplicities, default=0) == S.tree_depth(rounds)
    for prefix in range(rounds + 1):
        intervals = S.prefix_intervals(prefix)
        covered = [leaf for a, b in intervals for leaf in range(a, b)]
        assert covered == list(range(prefix))
        assert all(a % (b - a) == 0 and (b - a) & (b - a - 1) == 0 for a, b in intervals)


def test_exact_power_of_two_has_extra_root_level():
    assert S.tree_depth(511) == 9
    assert S.tree_depth(512) == 10
    assert S.tree_depth(2048) == 12


def test_unequal_restarted_sweeps_match_gaussian_rdp_reference_equation():
    steps, z = [0, 1, 3, 8, 13], 2.7
    count = sum(math.floor(math.log2(s)) + 1 for s in steps if s)
    rho = S.restart_zcdp(z, steps)
    for alpha in [1.01, 2, 10, 64]:
        assert S.rdp_at_order(rho, alpha) == pytest.approx(alpha * count / (2 * z ** 2), rel=1e-14)
    assert S.restart_zcdp(z, steps, adjacency="replace_content") == 4 * rho
    assert S.restart_zcdp(2 * z, steps, adjacency="replace_content") == rho


def test_registered_fixed6_examples_roundtrip_conservative_zcdp():
    for rounds, expected_z in [(512, 5.469581806114534), (2048, 5.991626670657556)]:
        z = S.noise_multiplier_for_epsilon(8, 5e-6, [rounds] * 6)
        assert z == pytest.approx(expected_z, rel=1e-14)
        epsilon = S.epsilon_from_zcdp(S.restart_zcdp(z, [rounds] * 6), 5e-6)
        assert epsilon == pytest.approx(8, abs=2e-14)
        info = S.describe_budget(z, [rounds] * 6, 5e-6)
        assert info["privacy_certified"] is False and info["amplification_used"] is False
        assert info["prior_private_tuning_and_side_outputs_accounted"] is False


def test_stable_small_epsilon_inversion_and_empty_or_noiseless_mechanism():
    rho = S.rho_for_epsilon(1e-12, 5e-6)
    assert rho > 0
    assert S.epsilon_from_zcdp(rho, 5e-6) == pytest.approx(1e-12, rel=1e-14)
    assert S.restart_zcdp(0, [0, 0]) == 0
    assert S.restart_zcdp(0, [1]) == math.inf
    assert S.restart_zcdp(1e-300, [512]) == math.inf
    assert S.epsilon_from_zcdp(math.inf, 5e-6) == math.inf
    assert S.noise_multiplier_for_epsilon(8, 5e-6, [0]) == 0
    assert S.rdp_at_order(0, math.inf) == 0


@pytest.mark.parametrize("value", [-1, 0.5, True, math.inf, np.bool_(True)])
def test_bad_tree_counts_refused(value):
    with pytest.raises(ValueError):
        S.tree_depth(value)


@pytest.mark.parametrize("call", [
    lambda: S.restart_zcdp(-1, [1]), lambda: S.restart_zcdp(math.nan, [1]),
    lambda: S.restart_zcdp(1, []), lambda: S.restart_zcdp(1, [2.5]),
    lambda: S.restart_zcdp(1, [1], adjacency="poisson"),
    lambda: S.epsilon_from_zcdp(1, 0), lambda: S.epsilon_from_zcdp(1, 1),
    lambda: S.epsilon_from_zcdp(-1, .01), lambda: S.epsilon_from_zcdp(math.nan, .01),
    lambda: S.noise_multiplier_for_epsilon(0, .01, [1]),
    lambda: S.rdp_at_order(1, 1), lambda: S.rdp_at_order(-1, 2),
])
def test_invalid_accountant_inputs_refused(call):
    with pytest.raises(ValueError):
        call()
