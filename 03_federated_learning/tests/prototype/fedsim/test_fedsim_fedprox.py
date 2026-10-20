"""FedProx: mu = 0 equals FA exactly under the same RNG; mu > 0 is inert at the first step (theta = theta_r) and
changes the update from the second step; the proximal gradient is +mu (theta - theta_r) with the anchor theta_r.

Proves: the prox term's value, sign, anchor (the broadcast theta_r of THIS round, not theta0 and not the moving
iterate), its once-per-step placement, and that FP shares FA's RNG stream and local solver.
Does not prove: that any mu helps accuracy.
"""
from __future__ import annotations

import torch
from fedsim_testkit import clients, nc, one_client, server_and_worker, solver, sq_dist
from torch import nn

import ppsi.fedsim.client as flc
from ppsi.fedsim.client import _add_prox_grad_, client_update, prox_penalty
from ppsi.fedsim.numerics import max_abs_diff, states_equal
from ppsi.fedsim.server import run_round

SEED = 2026
ORIG_PROX = flc._add_prox_grad_


def _fa(srv, w, cs, sol, r):
    return run_round(srv, w, cs, sol, round_idx=r, seed=SEED)


def _fp0(srv, w, cs, sol, r):
    return run_round(srv, w, cs, sol, round_idx=r, seed=SEED, mu=0.0)


def _fp0_own_seed(srv, w, cs, sol, r):          # WRONG: FP draws a different RNG stream than FA
    return run_round(srv, w, cs, sol, round_idx=r, seed=SEED + 1, mu=0.0)


def check_mu0_equals_fa(fp_fn):
    cs = clients(8, seed=31)
    sol = solver()
    a, wa = server_and_worker(seed=3)
    b, wb = server_and_worker(seed=3)
    for r in range(2):
        _fa(a, wa, cs, sol, r)
        fp_fn(b, wb, cs, sol, r)
    assert states_equal(a.adapter.broadcast_state(), b.adapter.broadcast_state()), "FP(mu=0) != FA bitwise"


def _round1_state():
    """theta_r for round 1 (after one FA round), theta0, and a worker."""
    srv, w = server_and_worker(seed=5)
    theta0 = srv.broadcast()
    _fa(srv, w, clients(6, seed=41), solver(), 0)
    return srv.broadcast(), theta0, w[0]


def check_first_step_inert_then_active():
    theta_r, _, w = _round1_state()
    sol = solver(passes=1)
    c1, c3 = one_client("user-16", 16, seed=1), one_client("user-48", 48, seed=2)     # 1 and 3 local steps
    fa1 = client_update(w, theta_r, c1, sol, round_idx=1, seed=SEED, clone_upload=True)
    fp1 = client_update(w, theta_r, c1, sol, round_idx=1, seed=SEED, mu=0.1, clone_upload=True)
    assert fa1.steps == 1 and fp1.steps == 1
    assert states_equal(fa1.upload, fp1.upload), "at theta = theta_r the prox gradient must be exactly 0"
    fa3 = client_update(w, theta_r, c3, sol, round_idx=1, seed=SEED, clone_upload=True)
    fp3 = client_update(w, theta_r, c3, sol, round_idx=1, seed=SEED, mu=0.1, clone_upload=True)
    assert fa3.steps == 3
    d = max_abs_diff(fa3.upload, fp3.upload)
    assert d > 1e-6, f"mu > 0 must change the update after >= 2 steps (max diff {d:.3g})"


def check_prox_gradient(add_fn):
    torch.manual_seed(0)
    anchor = [torch.randn(5, 3), torch.randn(4)]
    delta = [torch.randn(5, 3), torch.randn(4)]
    params = [nn.Parameter(a + d) for a, d in zip(anchor, delta)]
    mu = 0.3
    for p in params:
        p.grad = torch.zeros_like(p)
    add_fn(params, anchor, mu)
    for p, d in zip(params, delta):
        assert torch.allclose(p.grad, mu * d, rtol=1e-6, atol=1e-7), "prox gradient must be +mu (theta - theta_r)"
    ref = [nn.Parameter(a + d) for a, d in zip(anchor, delta)]
    prox_penalty(ref, anchor, mu).backward()
    for p, r in zip(params, ref):
        assert torch.allclose(p.grad, r.grad, rtol=1e-6, atol=1e-7), "gradient form != autograd of the scalar penalty"
    before = sum(float(((p.detach() - a) ** 2).sum()) for p, a in zip(params, anchor))
    after = sum(float(((p.detach() - 0.5 * p.grad - a) ** 2).sum()) for p, a in zip(params, anchor))
    assert after < before, "a descent step on the prox term must move theta toward the anchor"


def check_pull_toward_anchor():
    theta_r, _, w = _round1_state()
    sol = solver(passes=2)
    c = one_client("user-48", 48, seed=2)
    keys = w.manifest.shared_keys
    fa = client_update(w, theta_r, c, sol, round_idx=1, seed=SEED, clone_upload=True)
    fp = client_update(w, theta_r, c, sol, round_idx=1, seed=SEED, mu=2.0, clone_upload=True)
    assert sq_dist(fp.upload, theta_r, keys) < sq_dist(fa.upload, theta_r, keys), "FP must stay closer to theta_r"


# ------------------------------------------------------------------------------------------------ tests
def test_mu0_equals_fa_exactly():
    check_mu0_equals_fa(_fp0)


def test_mu0_prox_path_is_bitwise_noop():
    g = torch.randn(7)
    p = nn.Parameter(torch.randn(7)); p.grad = g.clone()
    _add_prox_grad_([p], [torch.randn(7)], 0.0)
    assert torch.equal(p.grad, g)


def test_first_step_inert_then_active():
    check_first_step_inert_then_active()


def test_prox_gradient_sign_value_anchor():
    check_prox_gradient(_add_prox_grad_)


def test_prox_pulls_toward_round_anchor():
    check_pull_toward_anchor()


def test_anchor_aliasing_refused():
    srv, _ = server_and_worker(seed=5)
    live = {k: v for k, v in srv.adapter.broadcast_state().items()}          # the server's own tensors
    try:
        client_update(srv.adapter, live, one_client("user-x", 20), solver(), round_idx=0, seed=1, mu=0.1)
    except RuntimeError as e:
        assert "aliases" in str(e)
    else:
        raise AssertionError("an anchor aliasing the worker parameters must be refused")


@nc("FP draws its own RNG stream (seed differs from FA)")
def test_nc_mu0_different_rng():
    check_mu0_equals_fa(_fp0_own_seed)


@nc("anchor re-set to the moving iterate every step (prox never acts)")
def test_nc_moving_anchor(monkeypatch):
    monkeypatch.setattr(flc, "_add_prox_grad_",
                        lambda params, anchors, mu: ORIG_PROX(params, [p.detach().clone() for p in params], mu))
    check_first_step_inert_then_active()


@nc("anchor = theta0 instead of this round's theta_r")
def test_nc_theta0_anchor(monkeypatch):
    _, theta0, w = _round1_state()
    a0 = [theta0[k] for k in w.manifest.shared_keys]
    monkeypatch.setattr(flc, "_add_prox_grad_", lambda params, anchors, mu: ORIG_PROX(params, a0, mu))
    check_first_step_inert_then_active()


def _wrong_sign(params, anchors, mu):
    ORIG_PROX(params, anchors, -mu)


@nc("proximal gradient with the wrong sign (unit check)")
def test_nc_wrong_sign_unit():
    check_prox_gradient(_wrong_sign)


@nc("proximal gradient with the wrong sign (end to end)")
def test_nc_wrong_sign_end_to_end(monkeypatch):
    monkeypatch.setattr(flc, "_add_prox_grad_", _wrong_sign)
    check_pull_toward_anchor()
