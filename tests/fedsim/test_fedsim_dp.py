"""DP-FedAvg: cohort m, flat L2 clipping of client deltas, Gaussian noise on the SUM, fixed
denominator m, dedicated (seed, round) noise generator, the S rule helper, FLRun integration and resume.

Proves (tiny synthetic family, CPU): theta_{r+1} = theta_r + (sum_u clip(delta_u) + z S xi_r) / m bit for bit against
an independent reference built from single client visits; the recorded pre-clip norms are the flat L2 norms; dropped
clients contribute zero and never change the denominator; the noise is reproducible from (seed, round) alone and is
N(0, (zS)^2) per coordinate on the sum (empirical check); z = 0 with clipping = the FA_1024 control; the process-pool
driver gives the same bits as the in-process driver; a DP run resumes bitwise; configs that would break the
accounting are refused. Negative controls: denominator = survivors, noise on the mean, unclipped deltas.
"""
from __future__ import annotations

import statistics
from collections import OrderedDict
from dataclasses import replace

import pytest
import torch
from fedsim_testkit import clients, nc, solver, tiny

from ppsi.fedsim.checkpoint import CheckpointManager
from ppsi.fedsim.client import client_update
from ppsi.fedsim.dp import DPConfig, clip_factor, flat_delta_norm, median_clip_norm, noise_like
from ppsi.fedsim.numerics import state_digest
from ppsi.fedsim.participation import ParticipationPlan
from ppsi.fedsim.pool import ProcessPool, WorkerSpec
from ppsi.fedsim.runtime import FLRun, RunConfig
from ppsi.fedsim.server import Server, run_round

SEED = 2026
PEAK = 0.05
CS = clients(20, seed=404, invalid_frac=0.1)
BY_KEY = {c.key: c for c in CS}
N_DEC = sum(int((c.examples["target_class"] >= 0).sum()) for c in CS)
M = 8


def theta0():
    a = tiny(1, tied=False)
    a.server_post_aggregate()
    return a.broadcast_state(clone=True), state_digest(a.broadcast_state())


THETA0, INIT = theta0()


def fresh_server():
    a = tiny(1, tied=False)
    a.load_state_(THETA0)
    return Server(a, init_sha256=INIT)


def uplan():
    return ParticipationPlan.build(list(BY_KEY), seed=SEED, manifest_hash="dp-tests", group_size=M,
                                   sampling="uniform")


def pplan():
    """Every DP-configured FLRun needs a Poisson plan with group_size == m."""
    return ParticipationPlan.build(list(BY_KEY), seed=SEED, manifest_hash="dp-tests", group_size=M,
                                   sampling="poisson")


def reference_round(theta_r, keys, lr, dp: DPConfig, round_idx, *, noise_on="sum", denominator=None, clip=True):
    """Independent reference: single visits from theta_r, flat clip, sum, noise, / m (or deliberate variants)."""
    w = tiny(55, tied=False)
    shared = w.manifest.shared_keys
    total = OrderedDict((k, torch.zeros_like(theta_r[k])) for k in shared)
    norms = []
    for k_ in keys:
        res = client_update(w, theta_r, BY_KEY[k_], replace(solver(), lr=lr), round_idx=round_idx, seed=SEED,
                            clone_upload=True)
        if res.n_consumed == 0:
            continue
        n = flat_delta_norm(res.upload, theta_r, shared)
        norms.append(n)
        f = clip_factor(n, dp.clip_norm) if clip else 1.0
        for k in shared:
            total[k].add_(res.upload[k] - theta_r[k], alpha=f)
    m = dp.denominator if denominator is None else denominator
    noise = noise_like(shared, theta_r, seed=SEED, round_idx=round_idx, std=dp.noise_std) if dp.noise_std else None
    new = OrderedDict()
    for k in shared:
        if noise_on == "sum":
            t = total[k] + noise[k] if noise is not None else total[k]
            new[k] = theta_r[k] + t / float(m)
        else:                                               # variant: noise added to the MEAN
            new[k] = theta_r[k] + total[k] / float(m) + (noise[k] if noise is not None else 0.0)
    for k in w.manifest.buffer_keys:
        new[k] = theta_r[k]
    post = tiny(1, tied=False)                              # the server's post-aggregate step (recentering)
    post.load_state_(new)
    post.server_post_aggregate()
    return post.extract_shared(clone=True), norms


def one_round(dp, keys, n_shards=1, round_idx=0, lr=PEAK):
    srv = fresh_server()
    theta_r = srv.broadcast()
    rep = run_round(srv, [tiny(7, tied=False)], [BY_KEY[k] for k in keys], replace(solver(), lr=lr),
                    round_idx=round_idx, seed=SEED, n_shards=n_shards, dp=dp, n_sampled=M)
    return srv, theta_r, rep


def close_state(a, b, keys, atol=0.0):
    return all(torch.allclose(a[k], b[k], rtol=0, atol=atol) if atol else torch.equal(a[k], b[k]) for k in keys)


# ------------------------------------------------------------------------------------------------ mechanism
def _median_norm(keys):
    _, _, rep = one_round(DPConfig(None, 0.0, M), keys)
    return statistics.median(rep.dp["norms"])


def test_clip_sum_fixed_denominator_matches_reference():
    keys = uplan().round_keys(0)
    S = _median_norm(keys)                                     # about half the clients get clipped
    dp = DPConfig(S, 0.7, M)
    srv, theta_r, rep = one_round(dp, keys)
    ref, norms = reference_round(theta_r, keys, PEAK, dp, 0)
    shared = srv.manifest.shared_keys
    got = srv.adapter.extract_shared()
    assert close_state(got, ref, shared, atol=1e-6)
    _, _, cal = one_round(DPConfig(None, 0.0, M), keys)
    assert cal.dp["norms"] == pytest.approx(norms, rel=1e-6)
    assert 0 < sum(1 for n in norms if n > S) < len(norms)
    assert rep.dp["denominator"] == M and rep.dp["noise_std"] == pytest.approx(0.7 * S)


def test_no_side_outputs_in_private_rounds():
    keys = uplan().round_keys(0)
    _, _, priv = one_round(DPConfig(0.4, 1.0, M), keys)
    assert not {"norms", "norm_keys", "n_clipped"} & set(priv.dp)
    _, _, ctrl = one_round(DPConfig(0.4, 0.0, M), keys)
    assert "n_clipped" in ctrl.dp and "norms" not in ctrl.dp
    _, _, cal = one_round(DPConfig(None, 0.0, M), keys)
    assert len(cal.dp["norms"]) == len(cal.dp["norm_keys"]) > 0


def test_single_shard_bitwise_vs_reference():
    """1 shard, the same client order: the DP driver equals the reference math bit for bit (noise included)."""
    keys = uplan().round_keys(1)
    dp = DPConfig(_median_norm(keys), 0.9, M)
    srv, theta_r, _rep = one_round(dp, keys, round_idx=1)
    ref, _ = reference_round(theta_r, keys, PEAK, dp, 1)
    assert close_state(srv.adapter.extract_shared(), ref, srv.manifest.shared_keys)


def test_recorded_norms_are_flat_l2():
    keys = uplan().round_keys(2)
    _, theta_r, rep = one_round(DPConfig(None, 0.0, M), keys, round_idx=2)
    w = tiny(55, tied=False)
    res = client_update(w, theta_r, BY_KEY[rep.dp["norm_keys"][0]], solver(lr=PEAK), round_idx=2, seed=SEED,
                        clone_upload=True)
    d = torch.cat([(res.upload[k] - theta_r[k]).double().flatten() for k in w.manifest.shared_keys])
    assert rep.dp["norms"][0] == pytest.approx(float(d.norm()), rel=1e-12)


def test_dropped_contribute_zero_denominator_fixed():
    keys = uplan().round_keys(3)
    surv = keys[:5]                                            # 3 of 8 dropped
    dp = DPConfig(0.8 * _median_norm(keys), 0.0, M)
    srv, theta_r, rep = one_round(dp, surv, round_idx=3)
    ref, _ = reference_round(theta_r, surv, PEAK, dp, 3)     # sum over survivors / fixed m = 8
    assert close_state(srv.adapter.extract_shared(), ref, srv.manifest.shared_keys, atol=1e-6)
    assert rep.dp["n_sampled"] == M and rep.dp["n_survived"] == 5


def test_noise_reproducible_and_seeded_by_round():
    tmpl = {"a": torch.zeros(1000), "b": torch.zeros(10, 30)}
    x = noise_like(("a", "b"), tmpl, seed=SEED, round_idx=4, std=2.0)
    y = noise_like(("a", "b"), tmpl, seed=SEED, round_idx=4, std=2.0)
    z = noise_like(("a", "b"), tmpl, seed=SEED, round_idx=5, std=2.0)
    assert all(torch.equal(x[k], y[k]) for k in x) and not torch.equal(x["a"], z["a"])
    torch.manual_seed(123)                                     # the global RNG must not matter
    y2 = noise_like(("a", "b"), tmpl, seed=SEED, round_idx=4, std=2.0)
    assert all(torch.equal(x[k], y2[k]) for k in x)


def test_noise_is_on_the_sum_with_std_zS():
    keys = uplan().round_keys(0)
    S, z = 0.3, 1.5
    a, _theta_r, _ = one_round(DPConfig(S, z, M), keys)
    b, _, _ = one_round(DPConfig(S, 0.0, M), keys)
    diff = torch.cat([(a.adapter.extract_shared()[k] - b.adapter.extract_shared()[k]).flatten()
                      for k in a.manifest.shared_keys])
    # the tiny untied family recenters W_out/b after aggregation, so compare the std, not the bits
    assert float((diff * M).std()) == pytest.approx(z * S, rel=0.1)


def test_fa1024_control_is_z0_with_clipping():
    keys = uplan().round_keys(0)
    S = 0.5 * _median_norm(keys)
    srv, theta_r, rep = one_round(DPConfig(S, 0.0, M), keys)
    ref, _ = reference_round(theta_r, keys, PEAK, DPConfig(S, 0.0, M), 0)
    assert close_state(srv.adapter.extract_shared(), ref, srv.manifest.shared_keys, atol=1e-6)
    assert rep.dp["noise_std"] == 0.0 and rep.dp["n_clipped"] > 0


def test_pool_matches_in_process_bits():
    keys = uplan().round_keys(0)
    dp = DPConfig(0.4, 1.0, M)
    a, _, _ = one_round(dp, keys, n_shards=2)
    srv = fresh_server()
    spec = WorkerSpec("ppsi.fedsim.synthetic:make_tiny_adapter", (("K", 24), ("d", 8), ("tied", False), ("seed", 7)))
    with ProcessPool(spec, n_workers=2) as pool:
        pool.run_round(srv, [BY_KEY[k] for k in keys], solver(lr=PEAK), round_idx=0, seed=SEED, n_shards=2, dp=dp,
                       n_sampled=M)
    assert srv.digest() == a.digest()


def test_median_clip_norm_rule():
    log = [[r, [float(r + i) for i in range(4)]] for r in range(25)]
    out = median_clip_norm(log)
    assert out["S"] == statistics.median([float(r + i) for r in range(20) for i in range(4)])
    assert out["n_norms"] == 80
    with pytest.raises(ValueError):
        median_clip_norm(log[:10])


def test_config_refusals():
    with pytest.raises(ValueError):
        DPConfig(None, 1.0, M)
    with pytest.raises(ValueError):
        DPConfig(-1.0, 1.0, M)
    with pytest.raises(ValueError):
        RunConfig(run_id="x", method="FP", mu=0.01, seed=1, lr_peak_local=0.1, solver=solver(),
                  exposure_point="end", dp=DPConfig(1.0, 1.0, M), endpoint_rounds=3)
    with pytest.raises(ValueError):                          # DP needs the round-count endpoint
        RunConfig(run_id="x", method="FA", seed=1, lr_peak_local=0.1, solver=solver(), exposure_point="end",
                  dp=DPConfig(1.0, 1.0, M))
    sweep = ParticipationPlan.build(list(BY_KEY), seed=SEED, manifest_hash="dp-tests", group_size=M)
    c = RunConfig(run_id="x", method="FA", seed=SEED, lr_peak_local=PEAK, solver=solver(), exposure_point="end",
                  dp=DPConfig(1.0, 1.0, M), endpoint_rounds=5)
    with pytest.raises(ValueError):                          # noise with sweep sampling breaks the accounting
        FLRun(c, sweep, fresh_server(), BY_KEY.__getitem__, N_DEC, workers=[tiny(7, tied=False)])
    for z, S in ((0.0, 1.0), (0.0, None)):                 # FA_1024 control and S calibration: sweep refused too
        with pytest.raises(ValueError):
            FLRun(replace(c, dp=DPConfig(S, z, M)), sweep, fresh_server(), BY_KEY.__getitem__, N_DEC,
                  workers=[tiny(7, tied=False)])
    with pytest.raises(ValueError):                          # fixed-size uniform sampling refused too
        FLRun(c, uplan(), fresh_server(), BY_KEY.__getitem__, N_DEC, workers=[tiny(7, tied=False)])
    for m in (4, 16):                                        # group_size must equal the fixed denominator m
        with pytest.raises(ValueError):
            FLRun(replace(c, dp=DPConfig(1.0, 0.0, m)), pplan(), fresh_server(), BY_KEY.__getitem__, N_DEC,
                  workers=[tiny(7, tied=False)])
    with pytest.raises(ValueError):                          # Q2: no drop-out in FA_1024 / DP runs
        replace(c, dropout_p=0.1)


# ------------------------------------------------------------------------------------------------ FLRun integration
def dp_cfg(**kw):
    base = {"run_id": "dp", "method": "FA", "seed": SEED, "lr_peak_local": PEAK, "solver": solver(), "n_shards": 2,
                "exposure_point": "end", "dropout_p": 0.0, "dp": DPConfig(0.3, 1.2, M), "endpoint_rounds": 10}
    base.update(kw)
    return RunConfig(**base)


def new_run(c, ckpt=None, resume=False):
    args = (c, pplan(), fresh_server(), BY_KEY.__getitem__, N_DEC)
    kw = {"workers": [tiny(7, tied=False)], "ckpt": ckpt}
    return FLRun.resume(*args, **kw) if resume else FLRun(*args, **kw)


def test_flrun_dp_rounds_and_log():
    run = new_run(dp_cfg()).run()
    assert run.state.done and run.state.cursor == 10
    assert run.dp_norm_log == [], "a z > 0 run must not log per-client norms"
    assert all(x["n_survived"] == x["n_sampled"] and not x["dropped"] for x in run.participation_log)
    assert run.summary()["dp"]["noise_multiplier"] == 1.2
    assert "round_count_state" in run.state_dict() and run.state_dict()["round_count_state"]["dp_norm_log"] == []
    cal = new_run(dp_cfg(dp=DPConfig(None, 0.0, M))).run()      # the S-calibration pass
    assert [r for r, _ in cal.dp_norm_log] == list(range(10))
    assert all(len(n) <= x["n_survived"] for (_, n), x in zip(cal.dp_norm_log, cal.participation_log))
    assert median_clip_norm(cal.dp_norm_log, rounds=range(5))["n_norms"] > 0
    ctrl = new_run(dp_cfg(dp=DPConfig(0.3, 0.0, M))).run()      # FA_1024 control: no norm log either
    assert ctrl.dp_norm_log == []


def test_flrun_dp_resume_bitwise(tmp_path):
    ref = new_run(dp_cfg()).run()
    first = new_run(dp_cfg(ckpt_every=4), CheckpointManager(tmp_path, "dp"))
    first.run(max_rounds=6)
    second = new_run(dp_cfg(ckpt_every=4), CheckpointManager(tmp_path, "dp"), resume=True)
    assert second.state.cursor == 4
    second.run()
    assert second.server.digest() == ref.server.digest()
    assert second.dp_norm_log == ref.dp_norm_log and second.participation_log == ref.participation_log


# ------------------------------------------------------------------------------------------------ negative controls
@nc("dividing by the survivor count instead of the fixed m")
def test_nc_denominator_survivors():
    keys = uplan().round_keys(3)
    surv = keys[:5]
    dp = DPConfig(0.3, 0.0, M)
    srv, theta_r, _ = one_round(dp, surv, round_idx=3)
    bad, _ = reference_round(theta_r, surv, PEAK, dp, 3, denominator=len(surv))
    assert close_state(srv.adapter.extract_shared(), bad, srv.manifest.shared_keys, atol=1e-6)


@nc("noise added to the mean (std zS) instead of the sum (std zS / m after division)")
def test_nc_noise_on_mean():
    keys = uplan().round_keys(0)
    dp = DPConfig(0.3, 1.0, M)
    srv, theta_r, _ = one_round(dp, keys)
    bad, _ = reference_round(theta_r, keys, PEAK, dp, 0, noise_on="mean")
    shared = srv.manifest.shared_keys
    assert close_state(bad, srv.adapter.extract_shared(), shared, atol=1e-5)


@nc("unclipped deltas")
def test_nc_unclipped_deltas():
    keys = uplan().round_keys(0)
    S = 0.5 * _median_norm(keys)
    dp = DPConfig(S, 0.0, M)
    srv, theta_r, _ = one_round(dp, keys)
    bad, _ = reference_round(theta_r, keys, PEAK, dp, 0, clip=False)
    assert close_state(srv.adapter.extract_shared(), bad, srv.manifest.shared_keys, atol=1e-6)
