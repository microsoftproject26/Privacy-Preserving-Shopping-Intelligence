"""FedAvgM, the FL_CONST LR shape and client SGD with momentum 0.9 in the simulator (CPU, synthetic toy model).

Proves: FedAvgM (m = beta m + Delta, theta += eta m; beta 0.9, eta 1) equals a hand-computed 3-round reference; FedAvgM at
beta = 0 equals FedAvg BIT FOR BIT over a whole FLRun; FedAvgM resume restores the momentum bitwise; FL_CONST =
100 x (warm-up 1 %), 1, 91/10 - 9 x (cool-down 10 %, final exactly 0.1) at x = (r + 1) / T, the planned table uses it
and the run's LRs follow it, a central run's table digest is unchanged, FL_CONST needs a round-count endpoint; client
SGD-m0.9 (fresh per visit, no weight decay) equals the hand-computed heavy-ball update and a whole FLRun with it resumes
bitwise; the default LocalSolver / RunConfig digests are unchanged. NCs: FedAvgM without the momentum term matches the
reference; FL_CONST ending at 0.01 x peak (the central endpoint) matches the defined shape; client SGD without momentum
matches the reference.
"""
from __future__ import annotations

import dataclasses
from collections import OrderedDict
from dataclasses import replace

import numpy as np
import torch
from fedsim_testkit import assert_raises, clients, nc, solver, tiny

from ppsi.fedsim.checkpoint import CheckpointManager
from ppsi.fedsim.client import SGD_M, LocalSolver, make_optimizer, valid_rows
from ppsi.fedsim.numerics import state_digest
from ppsi.fedsim.participation import ParticipationPlan
from ppsi.fedsim.runtime import FLRun, LRTableError, RunConfig, fl_const_factor
from ppsi.fedsim.server import Server, ServerOptConfig


def toy(seed=1):
    return tiny(seed, recenter=False)


def _step_with(srv, deltas_by_key):
    cur = srv.broadcast()
    srv.apply(OrderedDict((k, (cur[k] + deltas_by_key[k]) if k in deltas_by_key else cur[k]) for k in cur))


def ref_fedavgm(theta0, deltas, beta=0.9, lr=1.0, momentum=True):
    th, m = list(theta0), [0.0] * len(theta0)
    for d in deltas:
        for i, di in enumerate(d):
            m[i] = (beta * m[i] if momentum else 0.0) + di
            th[i] = th[i] + lr * m[i]
    return th


# ------------------------------------------------------------------------------------------------ FedAvgM
def test_fedavgm_config():
    c = ServerOptConfig("fedavgm", lr=1.0)
    assert (c.beta1, c.beta2, c.tau, c.v_init) == (0.9, None, None, None)
    assert_raises(ValueError, ServerOptConfig, "fedavgm", 1.0, 0.9, 0.99)
    assert_raises(ValueError, ServerOptConfig, "fedavgm", 1.0, tau=1e-3)
    assert_raises(ValueError, ServerOptConfig, "fedavgm", 1.0, beta1=1.0)
    srv = Server(toy())
    opt = srv.attach_server_opt(c)
    assert type(opt).__name__ == "FedAvgM" and opt.MOMENTS == ("m",) and set(opt.state_dict()) == {
        "cfg", "steps", "keys", "m"}


def test_fedavgm_equals_hand_computed_three_round_reference():
    srv = Server(toy())
    srv.attach_server_opt(ServerOptConfig("fedavgm", lr=1.0))
    k = srv.manifest.shared_keys[0]
    shape = srv.broadcast()[k].shape
    theta0 = srv.broadcast()[k].flatten()[:3].double().tolist()
    fixed = [[0.02, -0.5, 1e-3], [-0.01, 0.3, 2e-3], [0.05, -0.2, -1e-3]]
    for d in fixed:
        full = torch.zeros(shape).flatten()
        full[:3] = torch.tensor(d)
        _step_with(srv, {k: full.view(shape)})
    got = srv.broadcast()[k].flatten()[:3].double().tolist()
    want = ref_fedavgm(theta0, [[float(np.float32(x)) for x in d] for d in fixed])
    assert all(abs(g - w) <= 1e-6 * max(1.0, abs(w)) for g, w in zip(got, want)), (got, want)
    assert srv.server_opt.steps == 3


# ------------------------------------------------------------------------------------------------ runtime helpers
def _cfg(**kw):
    base = {"run_id": "fedavgm", "method": "FA", "seed": 5, "lr_peak_local": 0.05, "solver": solver(), "exposure_point": "end",
                "dropout_p": 0.1, "endpoint_rounds": 10, "n_shards": 2, "ckpt_every": 3}
    base.update(kw)
    return RunConfig(**base)


def _run(tmp_path, cfg, name, resume=False):
    cl = {c.key: c for c in clients(8)}
    plan = ParticipationPlan.build(sorted(cl), seed=cfg.seed, manifest_hash="m", group_size=3, sampling="sweep")
    a = toy()
    srv = Server(a, init_sha256=state_digest(a.broadcast_state()))
    n_dec = sum(int(valid_rows(c.examples).numel()) for c in cl.values())
    return (FLRun.resume if resume else FLRun)(cfg, plan, srv, cl.__getitem__, n_dec, workers=[toy(9)],
                                               ckpt=CheckpointManager(tmp_path / name, cfg.run_id))


def _straight_vs_resumed(tmp_path, cfg, tag):
    straight = _run(tmp_path, cfg, tag + "s").run()
    part = _run(tmp_path, cfg, tag + "p")
    part.run(6)
    res = _run(tmp_path, cfg, tag + "p", resume=True)
    assert res.state.cursor == 6
    res.run()
    return straight.summary(), res.summary()


def test_fedavgm_beta0_equals_fedavg_bitwise(tmp_path):
    a = _run(tmp_path, _cfg(), "avg").run().summary()
    b = _run(tmp_path, _cfg(server_opt=ServerOptConfig("fedavgm", lr=1.0, beta1=0.0)), "m0").run().summary()
    assert a["server_digest"] == b["server_digest"] and a["evals"] == b["evals"]
    c = _run(tmp_path, _cfg(server_opt=ServerOptConfig("fedavgm", lr=1.0)), "m9").run().summary()
    assert c["server_digest"] != a["server_digest"]


def test_fedavgm_resume_bitwise(tmp_path):
    a, b = _straight_vs_resumed(tmp_path, _cfg(server_opt=ServerOptConfig("fedavgm", lr=1.0)), "fm")
    assert a["server_digest"] == b["server_digest"] and a["server_opt"] == b["server_opt"]


# ------------------------------------------------------------------------------------------------ FL_CONST
def test_fl_const_factor_shape():
    T = 1707
    f = [fl_const_factor(r, T) for r in range(T)]
    assert f[0] == 100 / 1707 and f[16] == 100 * 17 / 1707 and f[17] == 1.0          # warm-up: x <= 1/100
    assert all(v == 1.0 for v in f[17:1536]) and f[1535] == 1.0                          # constant (x <= 9/10)
    assert abs(f[1536] - (9.1 - 9 * 1537 / 1707)) < 1e-15 and f[-1] == 0.1               # cool-down, final exact
    assert all(f[i] >= f[i + 1] for i in range(1536, T - 1)) and all(v > 0 for v in f)
    assert_raises(LRTableError, fl_const_factor, T, T)
    assert [fl_const_factor(r, 10) for r in range(10)] == [1.0] * 9 + [0.1]


def test_fl_const_table_run_and_resume(tmp_path):
    cfg = _cfg(lr_shape="fl_const")
    assert _cfg().digest() == RunConfig.from_dict(dict(dataclasses.asdict(_cfg()), solver=_cfg().solver)).digest()
    assert cfg.digest() != _cfg().digest()
    r = _run(tmp_path, cfg, "t")
    rows = r.lr_table["rows"]
    assert [x[4] for x in rows] == [0.05 * fl_const_factor(i, 10) for i in range(10)]
    assert r.lr_table["lr_shape"] == "fl_const" and "lr_shape" not in _run(tmp_path, _cfg(), "c").lr_table
    r.run()
    assert [lr for _, lr in r.lr_log] == [x[4] for x in rows]
    a, b = _straight_vs_resumed(tmp_path, cfg, "fc")
    assert a["server_digest"] == b["server_digest"] and a["lr_table_digest"] == b["lr_table_digest"]
    assert_raises(ValueError, RunConfig, run_id="x", method="FA", seed=1, lr_peak_local=0.1, solver=solver(),
                  exposure_point="end", lr_shape="fl_const")                        # needs endpoint_rounds
    assert_raises(ValueError, _cfg, lr_shape="cosine")


# ------------------------------------------------------------------------------------------------ client SGD-m
def test_sgd_m_optimizer_equals_heavy_ball_reference():
    a = toy()
    sol = LocalSolver(lr=0.1, optimizer=SGD_M)
    opt = make_optimizer(a, sol)
    assert isinstance(opt, torch.optim.SGD)
    assert all(g["momentum"] == 0.9 and g["weight_decay"] == 0.0 and g["dampening"] == 0.0 and not g["nesterov"]
               for g in opt.param_groups)
    params = [p for g in opt.param_groups for p in g["params"]]
    p0 = [p.detach().clone() for p in params]
    g1 = [torch.full_like(p, 0.5) for p in params]
    g2 = [torch.full_like(p, -0.25) for p in params]
    for g in (g1, g2):
        for p, gi in zip(params, g):
            p.grad = gi.clone()
        opt.step()
    for p, a0, x, y in zip(params, p0, g1, g2):
        want = a0 - 0.1 * x - 0.1 * (0.9 * x + y)                                     # buf1 = g1; buf2 = 0.9 g1 + g2
        assert torch.allclose(p.detach(), want, rtol=0, atol=1e-7)
    assert LocalSolver(lr=0.1) == LocalSolver(lr=0.1, optimizer="adamw")
    assert_raises(ValueError, LocalSolver, lr=0.1, optimizer="sgd_nesterov")


def test_sgd_m_run_differs_and_resumes_bitwise(tmp_path):
    sgd = replace(solver(), optimizer=SGD_M, weight_decay=0.0, fused=None)
    cfg = _cfg(solver=sgd, lr_peak_local=0.1, lr_shape="fl_const")
    a, b = _straight_vs_resumed(tmp_path, cfg, "sg")
    assert a["server_digest"] == b["server_digest"]
    assert a["server_digest"] != _run(tmp_path, _cfg(lr_peak_local=0.1, lr_shape="fl_const"), "aw").run().summary()[
        "server_digest"]


# ------------------------------------------------------------------------------------------------ negative controls
@nc("FedAvgM without the momentum term (plain FedAvg) matches the FedAvgM reference")
def test_nc_fedavgm_without_momentum():
    th0, ds = [0.1, -0.2], [[0.02, -0.5], [-0.01, 0.3], [0.05, -0.2]]
    a, b = ref_fedavgm(th0, ds), ref_fedavgm(th0, ds, momentum=False)
    assert all(abs(x - y) <= 1e-9 for x, y in zip(a, b))


@nc("an FL_CONST ending at the central endpoint 0.01 x peak matches the defined shape (final 0.1)")
def test_nc_fl_const_wrong_endpoint():
    assert fl_const_factor(9, 10) == 0.01


@nc("client SGD without momentum matches the heavy-ball reference")
def test_nc_sgd_without_momentum():
    a = toy()
    opt = make_optimizer(a, LocalSolver(lr=0.1, optimizer="sgd"))                     # the momentum-0 fixture
    params = [p for g in opt.param_groups for p in g["params"]]
    p0 = [p.detach().clone() for p in params]
    for gv in (0.5, -0.25):
        for p in params:
            p.grad = torch.full_like(p, gv)
        opt.step()
    assert all(torch.allclose(p.detach(), a0 - 0.1 * 0.5 - 0.1 * (0.9 * 0.5 - 0.25), rtol=0, atol=1e-7)
               for p, a0 in zip(params, p0))
