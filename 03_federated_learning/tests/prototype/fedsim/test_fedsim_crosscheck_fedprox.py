"""FedProx: the proximal gradient at round start (must be zero), after k steps, and its sign.

The client loss is loss_u = mean_CE_u + (mu/2)||theta_shared - stopgrad(theta_round_start)||^2; at initialization
theta = theta_round_start the proximal gradient is 0, so a single local SGD step cannot demonstrate FedProx's benefit.

Everything here is derived from that definition plus basic calculus (d/dtheta (mu/2)||theta-a||^2 = mu*(theta-a)).
"""
from __future__ import annotations

import numpy as np
import torch
from fedsim_crosscheck_kit import fresh_adapter, one_client

from ppsi.fedsim.client import LocalSolver, _add_prox_grad_, batch_ce, client_update, prox_penalty
from ppsi.fedsim.numerics import state_digest


# ---------------------------------------------------------------------------------------- B1: sign, by hand
def test_prox_grad_hand_computed_pull_toward_anchor_case1():
    # p=[1,2], existing CE grad=[0.1,0.2], anchor=[0.5,0.5], mu=0.1
    # expected added term = mu*(p-a) = 0.1*[0.5,1.5] = [0.05,0.15]; new grad = [0.15,0.35]
    p = torch.tensor([1.0, 2.0], requires_grad=True)
    p.grad = torch.tensor([0.1, 0.2])
    a = torch.tensor([0.5, 0.5])
    _add_prox_grad_([p], [a], mu=0.1)
    assert torch.allclose(p.grad, torch.tensor([0.15, 0.35]), atol=1e-7), "formula, hand toy 1"


def test_prox_grad_hand_computed_pull_toward_anchor_case2_negative_sign():
    # p=[0.0] below its anchor a=[1.0]: mu*(p-a) = 1*(0-1) = -1 (a NEGATIVE addition), which during a
    # minimizing step (p -= lr*grad) pushes p UP toward the anchor -- the opposite-sign case from test 1.
    p = torch.tensor([0.0], requires_grad=True)
    p.grad = torch.tensor([0.0])
    a = torch.tensor([1.0])
    _add_prox_grad_([p], [a], mu=1.0)
    assert torch.allclose(p.grad, torch.tensor([-1.0])), "formula, hand toy 2: sign flips with (p-a)"


def test_prox_penalty_scalar_hand_computed():
    # (mu/2)*sum((p-a)^2) = 0.05*((0.5)^2+(1.5)^2) = 0.05*2.5 = 0.125
    p = torch.tensor([1.0, 2.0])
    a = torch.tensor([0.5, 0.5])
    val = prox_penalty([p], [a], mu=0.1)
    assert torch.allclose(val, torch.tensor(0.125), atol=1e-7), "penalty scalar, hand toy"


# ---------------------------------------------------------------------------------------- B4: autograd cross-check
def test_prox_grad_matches_autograd_of_the_documented_scalar():
    p = torch.tensor([2.0, -1.0, 0.5], requires_grad=True)
    a = torch.tensor([1.0, 1.0, 1.0])
    mu = 0.37
    p.grad = None
    prox_penalty([p], [a], mu).backward()
    ref = p.grad.detach().clone()
    p.grad = torch.zeros_like(p)
    _add_prox_grad_([p], [a], mu)
    shortcut = p.grad.detach().clone()
    assert torch.allclose(ref, shortcut, atol=1e-6), (
        "the exact-gradient shortcut must equal autograd of (mu/2)||theta-stopgrad(theta_r)||^2")


# ---------------------------------------------------------------------------------------- B2: zero at round start
def test_prox_grad_is_exactly_zero_at_round_start():
    adapter = fresh_adapter(K=6, d=4, tied=True, seed=1)
    theta_r = adapter.broadcast_state(clone=True)
    adapter.load_state_(theta_r)                       # worker == theta_r exactly, bit for bit
    shared = adapter.shared_parameters()
    anchors = [theta_r[k] for k in adapter.manifest.shared_keys]
    rng = np.random.default_rng(0)
    from ppsi.fedsim.synthetic import client_examples
    batch = client_examples(6, 6, 5, rng)
    for p in shared:
        p.grad = None
    logits = adapter.scores(batch)
    loss = batch_ce(logits, batch["target_class"])
    loss.backward()
    before = [p.grad.detach().clone() for p in shared]
    _add_prox_grad_(shared, anchors, mu=0.37)            # theta == theta_r here, so this must add exactly 0
    after = [p.grad.detach().clone() for p in shared]
    for b, af in zip(before, after):
        assert torch.equal(b, af), (
            "'at initialization theta=theta_round_start the proximal gradient is 0' -- bitwise")


# ---------------------------------------------------------------------------------------- B3: first step identical,
#                                                                                             multi-step differs
def test_single_step_visit_is_bitwise_identical_regardless_of_mu():
    # n <= batch_size and passes=1 => exactly ONE optimizer step. Since the proximal gradient is exactly
    # zero at round start (see test above), the update for that single step cannot depend on mu at all.
    client = one_client("solo", n=10, K=8, L=5, seed=2)
    solver = LocalSolver(lr=1e-3, passes=1, batch_size=16)
    digests = []
    for mu in (0.0, 0.1, 5.0, 100.0):
        adapter = fresh_adapter(K=8, d=4, tied=True, seed=9)
        theta_r = adapter.broadcast_state(clone=True)
        res = client_update(adapter, theta_r, client, solver, round_idx=0, seed=42, mu=mu)
        digests.append(state_digest(res.upload))
    assert len(set(digests)) == 1, (
        "'a single local SGD step cannot demonstrate FedProx's benefit' -- bitwise across mu")


def test_multi_step_visit_differs_with_mu():
    # n > batch_size with passes=1 (or passes>1) gives >1 step, so after step 1 theta has moved away from
    # theta_r and a mu>0 proximal term becomes genuinely nonzero at step 2+, changing the trajectory.
    client = one_client("solo2", n=40, K=8, L=5, seed=3)
    solver = LocalSolver(lr=1e-2, passes=1, batch_size=16)   # 3 steps: 16, 16, 8 (tail)

    def run(mu):
        adapter = fresh_adapter(K=8, d=4, tied=True, seed=9)
        theta_r = adapter.broadcast_state(clone=True)
        res = client_update(adapter, theta_r, client, solver, round_idx=0, seed=42, mu=mu)
        return state_digest(res.upload)

    d_fa = run(0.0)
    d_fp = run(5.0)
    assert d_fa != d_fp, "'mu>0 after multiple steps affects update'"


def test_fp_pulls_closer_to_anchor_than_fa_after_multiple_steps():
    # A softer, magnitude-level check of the same claim: with a large mu the shared parameters must end up
    # closer (L2) to theta_r than the mu=0 (FA) run, given the same steps/seed/anchor.
    client = one_client("solo3", n=48, K=8, L=5, seed=4)
    solver = LocalSolver(lr=5e-2, passes=1, batch_size=16)   # 3 steps

    def dist(mu):
        adapter = fresh_adapter(K=8, d=4, tied=True, seed=11)
        theta_r = adapter.broadcast_state(clone=True)
        res = client_update(adapter, theta_r, client, solver, round_idx=0, seed=7, mu=mu)
        d2 = 0.0
        for k in adapter.manifest.shared_keys:
            d2 += float(((res.upload[k] - theta_r[k]) ** 2).sum())
        return d2 ** 0.5

    d_fa = dist(0.0)
    d_fp = dist(50.0)
    assert d_fp < d_fa, "the proximal penalty must pull the update closer to theta_r than plain FA"
