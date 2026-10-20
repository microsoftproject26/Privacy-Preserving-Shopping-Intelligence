"""Tree covariance, anchored FTRL, zero slots, aliases and refusal controls."""
import math
import pytest
import ppsi_env  # noqa: F401 — copied environment isolation
import torch
from . import dpftrl_tree as T


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(1)


def tree(*, z=0, seed=907, epoch=0, shape=(2,)):
    return T.GaussianPrefixTree(torch.zeros(shape, dtype=torch.float64),
                              clip_norm=1, noise_multiplier=z, noise_seed=seed, epoch=epoch)


def ftrl(*, z=0, orders=None, post=None, group=2, theta=None):
    return T.ClientUpdateFTRL(torch.zeros(2, dtype=torch.float64) if theta is None else theta,
        public_roster=["a", "b", "c"], sweep_orders=orders or [("a", "b", "c")] * 6,
        clip_norm=5, noise_multiplier=z, noise_seed=907, eta=.4, group_size=group,
        post_recenter=post)


def test_zero_noise_exact_prefix_all_dyadic_carries_and_aliases():
    tr = tree(shape=(2, 3))
    total = torch.zeros((2, 3), dtype=torch.float64)
    for t in range(1, 34):
        leaf = torch.arange(6, dtype=torch.float64).reshape(2, 3) + t
        total += leaf
        actual = tr.append(leaf)
        leaf.fill_(math.nan)
        torch.testing.assert_close(actual, total, rtol=0, atol=0)
        actual.fill_(math.nan)
        torch.testing.assert_close(tr.current(), total, rtol=0, atol=0)
    assert tr.count == 33


def test_hand_node_noise_and_canonical_node_reuse(monkeypatch):
    tr = tree()
    monkeypatch.setattr(tr, "_noise", lambda start, stop: torch.full((2,), 10. * start + stop, dtype=torch.float64))
    a = tr.append(torch.tensor([1., 1.]))  # [0,1] noise1
    b = tr.append(torch.tensor([2., 2.]))  # [0,2] noise2
    held = tr._nodes[1]
    c = tr.append(torch.tensor([3., 3.]))  # [0,2] + [2,3], noise2+23
    assert tr._nodes[1] is held
    torch.testing.assert_close(a, torch.full((2,), 2., dtype=torch.float64), rtol=0, atol=0)
    torch.testing.assert_close(b, torch.full((2,), 5., dtype=torch.float64), rtol=0, atol=0)
    torch.testing.assert_close(c, torch.full((2,), 31., dtype=torch.float64), rtol=0, atol=0)
    d = tr.append(torch.tensor([4., 4.]))  # fresh root[0,4], noise4
    torch.testing.assert_close(d, torch.full((2,), 14., dtype=torch.float64), rtol=0, atol=0)


def test_prefix_noise_covariance_and_restart_independence():
    # Thousands of independent coordinates provide a low-cost fixed-seed control.
    size = 16384
    tr = tree(z=1, shape=(size,))
    zero = torch.zeros(size, dtype=torch.float64)
    x = [tr.append(zero) for _ in range(4)]
    covariance = torch.stack(x).cov(correction=0)
    expected = torch.tensor([[1., 0., 0., 0.], [0., 1., 1., 0.],
                             [0., 1., 2., 0.], [0., 0., 0., 1.]], dtype=torch.float64)
    torch.testing.assert_close(covariance, expected, rtol=0, atol=.055)
    restarted = tree(z=1, epoch=1, shape=(size,)).append(zero)
    assert abs(torch.stack([x[0], restarted]).cov(correction=0)[0, 1].item()) < .04


def test_local_generators_leave_global_rng_unchanged_and_repeat_exactly():
    before = torch.get_rng_state().clone()
    a, b = tree(z=1), tree(z=1)
    for _ in range(9):
        torch.testing.assert_close(a.append(torch.zeros(2)), b.append(torch.zeros(2)), rtol=0, atol=0)
    assert torch.equal(before, torch.get_rng_state())
    public = a.public_geometry()
    assert "seed" not in str(public) and "raw" not in str(public)


def test_flat_clip_preserves_direction_input_and_extreme_finite_values():
    original = torch.tensor([3., 4.], dtype=torch.float64)
    clipped = T.clip_l2(original, 2.5)
    torch.testing.assert_close(clipped, torch.tensor([1.5, 2.], dtype=torch.float64), rtol=0, atol=0)
    assert torch.equal(original, torch.tensor([3., 4.], dtype=torch.float64))
    huge = T.clip_l2(torch.tensor([1e308, -1e308], dtype=torch.float64), 1)
    assert torch.isfinite(huge).all() and torch.linalg.vector_norm(huge) <= 1 + 1e-14
    torch.testing.assert_close(huge[0], -huge[1], rtol=0, atol=0)


@pytest.mark.parametrize("rounds", [1, 3, 4, 5, 8, 13])
def test_each_leaf_add_remove_and_replacement_node_sensitivity(rounds):
    from .dpftrl_statistics import tree_depth
    for user_leaf in range(rounds):
        for replacement in [False, True]:
            a, b = tree(z=1), tree(z=1)
            squared = 0.
            differing = 0
            for leaf in range(rounds):
                other = torch.tensor([.125, -.25], dtype=torch.float64)
                left = torch.tensor([1., 0.], dtype=torch.float64) if leaf == user_leaf else other
                right = (torch.tensor([-1. if replacement else 0., 0.], dtype=torch.float64)
                         if leaf == user_leaf else other)
                a.append(left)
                b.append(right)
                # Every call releases exactly one fresh dyadic node. Shared
                # node noises cancel under this coupling, exposing sensitivity.
                new_a = next(n for n in a._nodes.values() if n.stop == leaf + 1)
                new_b = next(n for n in b._nodes.values() if n.stop == leaf + 1)
                difference = new_a.raw - new_b.raw
                norm = torch.linalg.vector_norm(difference).item()
                assert norm <= (2 if replacement else 1)
                squared += norm ** 2
                differing += int(norm > 0)
            multiplier = 4 if replacement else 1
            assert squared == multiplier * differing
            assert squared <= multiplier * tree_depth(rounds)


def test_absent_user_keeps_public_roster_and_partial_group_denominator64():
    options = dict(public_roster=["a", "b", "c"], sweep_orders=[("a", "b", "c")] * 6,
                   clip_norm=1, noise_multiplier=0, noise_seed=17, eta=1, group_size=64)
    original = torch.tensor([2., -1.], dtype=torch.float32)
    a, b = [T.ClientUpdateFTRL(original, **options) for _ in range(2)]
    original.fill_(math.nan)
    left = a.step({"a": torch.tensor([1., 0.])})
    right = b.step({})
    torch.testing.assert_close(left - right, torch.tensor([1. / 64, 0.]), rtol=0, atol=0)
    assert a.public_status() == b.public_status()
    assert a.public_status()["planned_denominator"] == 64
    a.begin_sweep()
    b.begin_sweep()
    assert a.next_group() == b.next_group() == ("a", "b", "c")
    left.fill_(math.nan)
    assert torch.isfinite(a.weights()).all()


def test_anchored_ftrl_unequal_tail_zeros_and_explicit_sweep_restart():
    run = ftrl(theta=torch.tensor([2., -1.], dtype=torch.float64))
    assert run.next_group() == ("a", "b")
    first = run.step({"a": torch.tensor([3., 4.]), "b": None})
    torch.testing.assert_close(first, torch.tensor([2.6, -.2], dtype=torch.float64), rtol=0, atol=1e-15)
    assert run.next_group() == ("c",)
    # Last group has one slot but denominator stays2, with no survivor scaling.
    end = run.step({"c": torch.tensor([0., 5.])})
    torch.testing.assert_close(end, torch.tensor([2.6, .8], dtype=torch.float64), rtol=0, atol=1e-15)
    run.begin_sweep()
    torch.testing.assert_close(run.step({}), end, rtol=0, atol=0)
    status = run.public_status()
    assert status["sweep_index"] == 1 and status["planned_denominator"] == 2
    assert status["privacy_certified"] is False and "EFE is not inferred" in status["budget_basis"]


def test_noop_round_still_advances_private_tree_noise():
    run = ftrl(z=1)
    previous = run.weights()
    result = run.step({})
    assert run._tree.count == 1 and not torch.equal(previous, result)
    assert run.next_group() == ("c",)


def test_one_public_visit_and_no_extra_sweeps():
    run = ftrl(orders=[("a", "b", "c")])
    with pytest.raises(ValueError, match="incomplete"):
        run.begin_sweep()
    run.step({})
    with pytest.raises(ValueError, match="unplanned"):
        run.step({"a": torch.zeros(2)})
    run.step({})
    with pytest.raises(ValueError, match="complete"):
        run.step({})
    with pytest.raises(ValueError, match="already"):
        run.begin_sweep()


def test_uploader_order_does_not_change_float64_aggregation():
    options = dict(public_roster=["a", "b", "c"], sweep_orders=[("a", "b", "c")],
                   clip_norm=1e20, noise_multiplier=0, noise_seed=1, eta=1, group_size=3)
    a, b = [T.ClientUpdateFTRL(torch.zeros(1, dtype=torch.float64), **options) for _ in range(2)]
    values = {"a": torch.tensor([1e16], dtype=torch.float64), "b": torch.tensor([1.], dtype=torch.float64),
              "c": torch.tensor([-1e16], dtype=torch.float64)}
    torch.testing.assert_close(a.step(list(values.items())), b.step(list(reversed(list(values.items())))), rtol=0, atol=0)


def test_post_recenter_is_applied_after_every_step_and_transactional_on_failure():
    centered = ftrl(post=lambda x: x - x.mean())
    result = centered.step({"a": torch.tensor([3., 4.])})
    assert result.mean().item() == pytest.approx(0, abs=1e-16)
    broken = ftrl(post=lambda x: x[:1])
    with pytest.raises(ValueError, match="shape"):
        broken.step({"a": torch.ones(2)})
    assert broken._tree.count == 0 and broken.public_status()["rounds_completed_in_sweep"] == 0
    assert broken.next_group() == ("a", "b")


@pytest.mark.parametrize("updates", [[("a", torch.ones(2)), ("a", torch.ones(2))],
                                      {"x": torch.ones(2)}, {"a": torch.ones(3)},
                                      {"a": torch.tensor([math.nan, 0.])}, {"a": torch.ones(2, dtype=torch.int64)}])
def test_bad_updates_refuse_before_round_commit(updates):
    run = ftrl()
    with pytest.raises(ValueError):
        run.step(updates)
    assert run._tree.count == 0 and run.next_group() == ("a", "b")


@pytest.mark.parametrize("kwargs", [{"clip_norm": 0}, {"noise_multiplier": -1}, {"noise_multiplier": math.inf},
                                     {"noise_seed": True}, {"noise_seed": -1}, {"epoch": -1}])
def test_bad_tree_parameters_refused(kwargs):
    arguments = dict(clip_norm=1, noise_multiplier=1, noise_seed=1, epoch=0)
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        T.GaussianPrefixTree(torch.zeros(2), **arguments)


@pytest.mark.parametrize("orders", [[("a", "a", "c")], [("a", "b")], [("a", "b", "x")], []])
def test_invalid_public_sweep_permutations_refused(orders):
    with pytest.raises(ValueError):
        T.ClientUpdateFTRL(torch.zeros(2), public_roster=["a", "b", "c"], sweep_orders=orders,
                           clip_norm=1, noise_multiplier=1, noise_seed=1, eta=1)
