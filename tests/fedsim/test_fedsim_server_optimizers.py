"""The FedAdam server optimiser of server.py / runtime.py (CPU, synthetic toy model).

Proves: FedAdam (Reddi et al. 2021 ICLR Alg. 2: m = b1 m + (1-b1) D, v = b2 v + (1-b2) D^2, theta += eta m / (sqrt v + tau),
no bias correction; m_0 = 0, v_0 = tau^2 as in Alg. 2 line 1) equals a hand-computed 3-round reference (float64
arithmetic to 1e-6 relative, and bit for bit against an independent numpy-float32 replica); eta_s = 0 leaves theta
unchanged (the moments still move); the pseudo-gradient of a DP round is the noisy fixed-denominator mean that FedAvg
would apply; a non-finite step leaves theta and the moments unchanged; RunConfig digest unchanged with server_opt None
and the checkpoint payload unchanged for FedAvg; FLRun resume restores the moments bitwise (straight == interrupted +
resumed) and refuses a checkpoint without / with unexpected moments. NCs: bias-corrected Adam, v_0 = 0.
"""
from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np
import torch
from fedsim_testkit import assert_raises, clients, nc, solver, tiny

from ppsi.fedsim.checkpoint import CheckpointError, CheckpointManager
from ppsi.fedsim.dp import DPConfig
from ppsi.fedsim.participation import ParticipationPlan
from ppsi.fedsim.runtime import FLRun, RunConfig
from ppsi.fedsim.server import (
    FEDADAM_DEFAULTS,
    V_INIT,
    Server,
    ServerOptConfig,
    ServerOptError,
    run_round,
)

B1, B2, TAU = 0.9, 0.99, 1e-3


def toy(seed=1):
    return tiny(seed, recenter=False)


def raw_server(seed=1):
    return Server(toy(seed))


def ref_fedadam_f64(theta0, deltas, lr, v0=TAU * TAU, bias_correct=False):
    """Hand reference in float64, element by element (plain Python floats)."""
    th, m, v, out = list(theta0), [0.0] * len(theta0), [v0] * len(theta0), []
    for t, d in enumerate(deltas, 1):
        for i, di in enumerate(d):
            m[i] = B1 * m[i] + (1 - B1) * di
            v[i] = B2 * v[i] + (1 - B2) * di * di
            mh, vh = (m[i] / (1 - B1 ** t), v[i] / (1 - B2 ** t)) if bias_correct else (m[i], v[i])
            th[i] = th[i] + lr * mh / (math.sqrt(vh) + TAU)
        out.append(list(th))
    return out


def ref_fedadam_np32(theta0, deltas_new, lr):
    """Independent numpy float32 replica of the op order (deltas formed as new - theta in float32)."""
    f = np.float32
    th = np.asarray(theta0, dtype=f)
    m, v = np.zeros_like(th), np.full_like(th, f(TAU * TAU))
    for new in deltas_new:
        d = (np.asarray(new(th), dtype=f) - th).astype(f)
        m = (m * f(B1) + d * f(1.0 - B1)).astype(f)
        v = (v * f(B2) + (d * d) * f(1.0 - B2)).astype(f)
        th = (th + (m / (np.sqrt(v) + f(TAU))) * f(lr)).astype(f)
    return th


def _step_with(srv, deltas_by_key):
    cur = srv.broadcast()
    new = OrderedDict((k, (cur[k] + deltas_by_key[k]) if k in deltas_by_key else cur[k]) for k in cur)
    srv.apply(new)


# ------------------------------------------------------------------------------------------------ math
def test_config_defaults_and_validation():
    c = ServerOptConfig("fedadam", lr=0.1)
    assert (c.beta1, c.beta2, c.tau, c.v_init) == (0.9, 0.99, 1e-3, "tau_squared") == (
        FEDADAM_DEFAULTS["beta1"], FEDADAM_DEFAULTS["beta2"], FEDADAM_DEFAULTS["tau"], V_INIT)
    for bad in ({"name": "fedyogi", "lr": 0.1}, {"name": "fedadam", "lr": -1.0}, {"name": "fedadam", "lr": float("nan")},
                {"name": "fedadam", "lr": 0.1, "beta1": 1.0}, {"name": "fedadam", "lr": 0.1, "tau": 0.0},
                {"name": "fedadam", "lr": 0.1, "v_init": "zero"}):
        assert_raises(ValueError, ServerOptConfig, **bad)
    srv = raw_server()
    opt = srv.attach_server_opt(c)
    k = srv.manifest.shared_keys[0]
    assert torch.equal(opt.m[k], torch.zeros_like(opt.m[k]))
    assert torch.equal(opt.v[k], torch.full_like(opt.v[k], float(np.float32(1e-6))))         # v_0 = tau^2
    assert srv.attach_server_opt(c) is opt
    assert_raises(ServerOptError, srv.attach_server_opt, ServerOptConfig("fedadam", lr=0.01))


def test_fedadam_equals_hand_computed_three_round_reference():
    srv = raw_server()
    srv.attach_server_opt(ServerOptConfig("fedadam", lr=0.1))
    k = srv.manifest.shared_keys[0]
    theta0 = srv.broadcast()[k].flatten()[:4].double().tolist()
    fixed = [[0.02, -0.5, 1e-4, 0.0], [-0.01, 0.3, 2e-4, 0.0], [0.05, -0.2, -1e-4, 3.0]]
    shape = srv.broadcast()[k].shape
    for d in fixed:
        full = torch.zeros(shape, dtype=torch.float32).flatten()
        full[:4] = torch.tensor(d, dtype=torch.float32)
        _step_with(srv, {k: full.view(shape)})
    got = srv.broadcast()[k].flatten()[:4].double().tolist()
    want = ref_fedadam_f64(theta0, [[float(np.float32(x)) for x in d] for d in fixed], 0.1)[-1]
    for g, w in zip(got, want):
        assert abs(g - w) <= 1e-6 * max(1.0, abs(w)), (got, want)
    # the untouched coordinates (Delta = 0 every round, m = 0) do not move at all
    assert torch.equal(srv.broadcast()[k].flatten()[4:], Server(toy()).broadcast()[k].flatten()[4:])
    assert srv.server_opt.steps == 3


def test_fedadam_bitwise_equals_numpy_float32_replica():
    srv = raw_server(3)
    srv.attach_server_opt(ServerOptConfig("fedadam", lr=0.01))
    k = srv.manifest.shared_keys[0]
    th0 = srv.broadcast()[k].numpy().copy()
    rng = np.random.default_rng(0)
    ds = [rng.normal(0, s, th0.shape).astype(np.float32) for s in (1e-2, 1e-4, 1.0)]
    for d in ds:
        _step_with(srv, {k: torch.from_numpy(d)})
    rep = ref_fedadam_np32(th0, [lambda th, d=d: (th + d).astype(np.float32) for d in ds], 0.01)
    assert np.array_equal(srv.broadcast()[k].numpy(), rep)


def test_zero_server_lr_leaves_theta_unchanged():
    srv = raw_server()
    srv.attach_server_opt(ServerOptConfig("fedadam", lr=0.0))
    before = srv.broadcast()
    for s in (0.3, -1.0):
        _step_with(srv, {k: torch.full_like(before[k], s) for k in srv.manifest.shared_keys})
    after = srv.broadcast()
    assert all(torch.equal(before[k], after[k]) for k in before)
    k = srv.manifest.shared_keys[0]
    assert not torch.equal(srv.server_opt.m[k], torch.zeros_like(before[k]))              # the moments did move


def test_non_finite_step_refused_state_unchanged():
    srv = raw_server()
    srv.attach_server_opt(ServerOptConfig("fedadam", lr=0.1))
    _step_with(srv, {k: torch.full_like(srv.broadcast()[k], 0.01) for k in srv.manifest.shared_keys})
    th, dg = srv.broadcast(), srv.server_opt.digest()
    keys = srv.manifest.shared_keys
    bad = {k: torch.full_like(th[k], 0.01) for k in keys}
    bad[keys[-1]] = torch.full_like(th[keys[-1]], float("inf"))
    assert_raises(ServerOptError, _step_with, srv, bad)
    assert srv.server_opt.digest() == dg and all(torch.equal(th[k], srv.broadcast()[k]) for k in th)
    assert srv.server_opt.steps == 1


def test_dp_fedadam_pseudo_gradient_is_the_noisy_mean():
    cl = clients(6)
    dp = DPConfig(0.5, 0.7, len(cl))
    kw = {"round_idx": 0, "seed": 11, "dp": dp, "n_sampled": len(cl)}
    a = raw_server()
    th0 = a.broadcast()
    run_round(a, [toy(7)], cl, solver(), **kw)                                            # FedAvg (server LR 1)
    delta = {k: torch.sub(a.broadcast()[k], th0[k]) for k in a.manifest.shared_keys}
    b = raw_server()
    b.attach_server_opt(ServerOptConfig("fedadam", lr=0.1))
    run_round(b, [toy(7)], cl, solver(), **kw)
    for k in b.manifest.shared_keys:
        assert torch.equal(b.server_opt.m[k], torch.add(torch.mul(torch.zeros_like(delta[k]), B1),
                                                        torch.mul(delta[k], 1.0 - B1)))
    c = raw_server()                                                                       # the noise is in it
    run_round(c, [toy(7)], cl, solver(), **dict(kw, dp=DPConfig(0.5, 0.0, len(cl))))
    assert any(not torch.equal(c.broadcast()[k], a.broadcast()[k]) for k in a.manifest.shared_keys)


# ------------------------------------------------------------------------------------------------ runtime
def _cfg(server_opt=None, **kw):
    base = {"run_id": "fedadam", "method": "FA", "seed": 5, "lr_peak_local": 0.05, "solver": solver(), "exposure_point": "end",
                "dropout_p": 0.1, "endpoint_rounds": 10, "n_shards": 2, "ckpt_every": 3, "server_opt": server_opt}
    base.update(kw)
    return RunConfig(**base)


def _run(tmp_path, cfg, name, resume=False):
    cl = {c.key: c for c in clients(8)}
    plan = ParticipationPlan.build(sorted(cl), seed=cfg.seed, manifest_hash="m", group_size=3, sampling="sweep")
    adapter = toy()
    from ppsi.fedsim.numerics import state_digest
    srv = Server(adapter, init_sha256=state_digest(adapter.broadcast_state()))
    ck = CheckpointManager(tmp_path / name, cfg.run_id)
    from ppsi.fedsim.client import valid_rows
    n_dec = sum(int(valid_rows(c.examples).numel()) for c in cl.values())
    args = (cfg, plan, srv, cl.__getitem__, n_dec)
    kw = {"workers": [toy(9)], "ckpt": ck}
    return (FLRun.resume if resume else FLRun)(*args, **kw)


def test_runconfig_digest_and_payload_unchanged_when_off(tmp_path):
    off = _cfg()
    d = RunConfig.from_dict({**{k: getattr(off, k) for k in off.__dataclass_fields__}})
    assert d.digest() == off.digest()
    on = _cfg(ServerOptConfig("fedadam", lr=0.1))
    assert on.digest() != off.digest() != _cfg(ServerOptConfig("fedadam", lr=0.01)).digest()
    import dataclasses
    rt = RunConfig.from_dict(dict(dataclasses.asdict(on), solver=on.solver))
    assert rt == on and rt.digest() == on.digest()
    r = _run(tmp_path, off, "off")
    r.run(2)
    assert "server_opt" not in r.state_dict() and "server_opt" not in r.summary() and r.server.server_opt is None


def test_runtime_resume_restores_moments_bitwise(tmp_path):
    cfg = _cfg(ServerOptConfig("fedadam", lr=0.1))
    straight = _run(tmp_path, cfg, "s").run()
    part = _run(tmp_path, cfg, "p")
    part.run(6)                                                   # latest.pt at round 6 (ckpt_every 3)
    res = _run(tmp_path, cfg, "p", resume=True)
    assert res.state.cursor == 6 and res.server.server_opt.steps == part.server.server_opt.steps
    assert res.server.server_opt.digest() == part.server.server_opt.digest()
    res.run()
    a, b = straight.summary(), res.summary()
    assert a["server_digest"] == b["server_digest"] and a["server_opt"] == b["server_opt"]
    fedavg = _run(tmp_path, _cfg(), "f").run()
    assert fedavg.summary()["server_digest"] != a["server_digest"]
    # refusals: moments missing / unexpected
    d = part.state_dict()
    fresh = _run(tmp_path, cfg, "x")
    d2 = dict(d)
    d2.pop("server_opt")
    assert_raises(CheckpointError, fresh.load_state_dict, d2)
    off = _run(tmp_path, _cfg(), "y")
    d3 = dict(off.state_dict(), server_opt=d["server_opt"])
    assert_raises(CheckpointError, off.load_state_dict, d3)
    bad = dict(d, server_opt=dict(d["server_opt"], cfg=dict(d["server_opt"]["cfg"], lr=0.01)))
    assert_raises(ServerOptError, _run(tmp_path, cfg, "z").load_state_dict, bad)


def test_flrun_refuses_server_with_foreign_optimizer(tmp_path):
    cfg = _cfg()
    cl = {c.key: c for c in clients(8)}
    plan = ParticipationPlan.build(sorted(cl), seed=5, manifest_hash="m", group_size=3, sampling="sweep")
    from ppsi.fedsim.numerics import state_digest
    a = toy()
    srv = Server(a, init_sha256=state_digest(a.broadcast_state()))
    srv.attach_server_opt(ServerOptConfig("fedadam", lr=0.1))
    assert_raises(ValueError, FLRun, cfg, plan, srv, cl.__getitem__, 1000, workers=[toy(9)])


# ------------------------------------------------------------------------------------------------ negative controls
@nc("Adam WITH bias correction matches the FedAdam reference (it must not: Alg. 2 has none)")
def test_nc_bias_corrected_adam_matches_reference():
    th0, ds = [0.1, -0.2], [[0.02, -0.5], [-0.01, 0.3], [0.05, -0.2]]
    a = ref_fedadam_f64(th0, ds, 0.1)[-1]
    b = ref_fedadam_f64(th0, ds, 0.1, bias_correct=True)[-1]
    assert all(abs(x - y) <= 1e-6 for x, y in zip(a, b))


@nc("v_0 = 0 gives the same server step as v_0 = tau^2")
def test_nc_v0_zero_matches_reference():
    srv = raw_server()
    srv.attach_server_opt(ServerOptConfig("fedadam", lr=0.1))
    k = srv.manifest.shared_keys[0]
    th0 = srv.broadcast()[k].flatten()[:2].double().tolist()
    d = [[1e-4, -2e-4]]
    full = torch.zeros(srv.broadcast()[k].shape).flatten()
    full[:2] = torch.tensor(d[0])
    _step_with(srv, {k: full.view(srv.broadcast()[k].shape)})
    got = srv.broadcast()[k].flatten()[:2].double().tolist()
    mutant = ref_fedadam_f64(th0, [[float(np.float32(x)) for x in d[0]]], 0.1, v0=0.0)[-1]
    assert all(abs(g - w) <= 1e-6 * max(1.0, abs(w)) for g, w in zip(got, mutant))
