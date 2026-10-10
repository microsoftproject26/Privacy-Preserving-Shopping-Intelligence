"""How the FedAdam adaptivity constant tau shapes the server step when the pseudo-gradients are tiny.

FedAdam (server.py) steps theta += eta_s m / (sqrt(v) + tau). Once the moments settle on a constant pseudo-gradient d,
m -> d and sqrt(v) -> |d|, so the per-coordinate step is eta_s d / (|d| + tau):
  * |d| << tau: the step is eta_s d / tau, linear in d. FedAdam is then plain server SGD with learning rate eta_s / tau,
    and with tau = 1e-3 and d around 1e-7 the step is about 1e-4 eta_s, four orders of magnitude below Adam's
    scale-free step eta_s;
  * |d| >> tau: the step is about eta_s whatever the scale of d (Adam-like normalisation).
The tests drive the optimiser directly with a fixed theta_r = 0, so the step is the returned state and no float32
cancellation hides a tiny update. 2,000 rounds make the v_0 = tau^2 start negligible (0.99^2000 < 2e-9).
"""
from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch

from ppsi.fedsim.server import FEDADAM_DEFAULTS, ServerOptConfig, make_server_opt

ETA = 0.01


def steady_step(tau: float, deltas, rounds: int = 2000, lr: float = ETA) -> torch.Tensor:
    """The server step / eta_s after `rounds` rounds with the same pseudo-gradient `deltas` every round."""
    d = torch.tensor(deltas, dtype=torch.float32)
    theta = OrderedDict(w=torch.zeros_like(d))
    opt = make_server_opt(ServerOptConfig("fedadam", lr=lr, tau=tau), SimpleNamespace(shared_keys=("w",)), theta)
    for _ in range(rounds):
        out = opt.step(theta, OrderedDict(w=d.clone()))
    assert opt.steps == rounds
    return out["w"].double() / lr


def test_default_tau_is_unchanged():
    assert FEDADAM_DEFAULTS["tau"] == 1e-3 and ServerOptConfig("fedadam", lr=ETA).tau == 1e-3
    assert ServerOptConfig("fedadam", lr=ETA, tau=1e-6).tau == 1e-6


def test_tau_1e3_with_tiny_updates_is_plain_sgd_with_a_tiny_step():
    tau = 1e-3
    s = steady_step(tau, [1e-7, 2e-7, -1e-7, 5e-8])
    assert s.tolist() == pytest.approx([1e-7 / tau, 2e-7 / tau, -1e-7 / tau, 5e-8 / tau], rel=1e-3)   # update / tau
    assert float(s[1] / s[0]) == pytest.approx(2.0, rel=1e-3)            # linear in the update: SGD, not Adam
    assert float(s[3] / s[0]) == pytest.approx(0.5, rel=1e-3)
    assert float(s.abs().max()) < 3e-4                                    # vs the Adam-like step of 1.0 x eta_s


def test_tau_1e6_restores_adam_like_normalisation():
    small = steady_step(1e-6, [1e-7])
    big = steady_step(1e-3, [1e-7])
    assert float(small[0]) == pytest.approx(1e-7 / (1e-7 + 1e-6), rel=1e-3)
    assert float(small[0] / big[0]) > 500                                 # the same update moves theta ~900x more
    scales = [1e-5, 1e-4, 1e-3]
    adam_like = steady_step(1e-6, scales)
    sgd_like = steady_step(1e-3, scales)
    assert all(0.9 <= float(x) <= 1.0 for x in adam_like)                 # about eta_s whatever the update's scale
    assert float(adam_like.max() / adam_like.min()) < 1.15
    assert float(sgd_like.max() / sgd_like.min()) > 40                    # tau 1e-3: the step still follows |d|


def test_first_round_is_damped_by_the_tau_squared_start():
    tau, d = 1e-3, 1e-7
    s = float(steady_step(tau, [d], rounds=1)[0])
    expected = 0.1 * d / ((0.99 * tau * tau + 0.01 * d * d) ** 0.5 + tau)  # m_1 = 0.1 d, v_1 = 0.99 tau^2 + 0.01 d^2
    assert s == pytest.approx(expected, rel=1e-4)
    assert s == pytest.approx(0.05 * d / tau, rel=5e-3)
