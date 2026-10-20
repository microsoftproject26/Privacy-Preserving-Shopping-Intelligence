"""quant.py, the FA_Q8 upload codec, on the synthetic FL fixtures (CPU).

Proves: per-tensor symmetric int8 with scale = max|delta| / 127 (all-zero tensor -> scale 0, q 0); stochastic rounding
seeded by derive_seed(seed, "q8", round, client_key) is bitwise deterministic and key/round dependent; it is unbiased
(the mean of many roundings -> delta) while round-to-nearest is not (negative control); per-element error <= scale;
upload bytes = sum(numel) + 4 x n_tensors (aliases excluded); a Q8 round through server.run_round equals the FedAvg
(aggregate.py, same shards) of the dequantised uploads bitwise; an FLRun with the Q8 channel records the Q8 bytes on its
ledger, resumes exactly after re-installing the channel, and differs from plain FA; the pool path and non-FA configs are
refused. Negative controls: deterministic rounding is biased; a seed without the client key; Q8 == FA.
"""
from __future__ import annotations

import math
from collections import OrderedDict

import torch
from fedsim_testkit import assert_raises, clients, nc, solver, tiny

import ppsi.fedsim.quant as Q
from ppsi.fedsim.aggregate import aggregate_uploads
from ppsi.fedsim.checkpoint import CheckpointManager
from ppsi.fedsim.client import client_update
from ppsi.fedsim.dp import DPConfig
from ppsi.fedsim.numerics import state_digest
from ppsi.fedsim.participation import ParticipationPlan
from ppsi.fedsim.runtime import FLRun, RunConfig
from ppsi.fedsim.server import Server, run_round

SEED = 2026
CS = clients(12, seed=505, invalid_frac=0.1)
BY_KEY = {c.key: c for c in CS}
N_DEC = sum(int((c.examples["target_class"] >= 0).sum()) for c in CS)


def _delta(n=257, seed=0, scale=1e-3):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, generator=g) * scale


def _gen(seed):
    return torch.Generator().manual_seed(seed)


# ------------------------------------------------------------------------------------------------ the quantiser
def test_q8_same_seed_bitwise_and_seed_parts_matter():
    a = tiny(1)
    theta = a.broadcast_state(clone=True)
    up = OrderedDict((k, theta[k] + _delta(theta[k].numel(), i).reshape(theta[k].shape))
                     for i, k in enumerate(a.manifest.shared_keys))
    keys = a.manifest.shared_keys
    p1 = Q.encode(up, theta, keys, seed=SEED, round_idx=3, key="user-00001")
    p2 = Q.encode(up, theta, keys, seed=SEED, round_idx=3, key="user-00001")
    for k in keys:
        assert torch.equal(p1[k][0], p2[k][0]) and torch.equal(p1[k][1], p2[k][1])
        assert p1[k][0].dtype == torch.int8 and p1[k][1].dtype == torch.float32 and p1[k][1].dim() == 0
    for other in ({"round_idx": 4, "key": "user-00001"}, {"round_idx": 3, "key": "user-00002"}):
        p3 = Q.encode(up, theta, keys, seed=SEED, **other)
        assert any(not torch.equal(p1[k][0], p3[k][0]) for k in keys)
    assert Q.visit_seed_q8(SEED, 3, "user-00001") != Q.visit_seed_q8(SEED + 1, 3, "user-00001")


def test_q8_scale_and_per_element_error_le_scale():
    for seed, d in enumerate((_delta(1000, 1), _delta(64, 2, 5.0), torch.tensor([3.0, -3.0, 0.0, 1e-9]),
                              torch.tensor([-2.5]), _delta(4096, 3, 1e-7))):
        q, s = Q.quantize_tensor(d, _gen(seed))
        assert float(s) == float(d.abs().max() / 127)
        assert int(q.abs().max()) <= 127
        err = (Q.dequantize_tensor(q, s) - d).abs()
        assert float(err.max()) <= float(s) * (1 + 1e-6), (float(err.max()), float(s))


def test_q8_all_zero_and_empty_tensors():
    for d in (torch.zeros(10), torch.zeros(3, 4), torch.zeros(0)):
        q, s = Q.quantize_tensor(d, _gen(0))
        assert float(s) == 0.0 and q.dtype == torch.int8 and tuple(q.shape) == tuple(d.shape)
        assert not q.any() and torch.equal(Q.dequantize_tensor(q, s), torch.zeros_like(d))
    assert_raises(Q.Q8Error, Q.quantize_tensor, torch.zeros(3, dtype=torch.float64), _gen(0))


def _mean_of_roundings(d, n, stochastic=True):
    acc = torch.zeros_like(d, dtype=torch.float64)
    s = None
    for i in range(n):
        q, s = Q.quantize_tensor(d, _gen(10_000 + i) if stochastic else None, stochastic=stochastic)
        acc += Q.dequantize_tensor(q, s).double()
    return acc / n, float(s)


def _assert_unbiased(stochastic: bool):
    d = _delta(200, 7)
    n = 3000
    mean, s = _mean_of_roundings(d, n, stochastic)
    tol = 5 * s / (2 * math.sqrt(n))                        # 5 sigma of the mean (per-element var <= s^2 / 4)
    dev = float((mean - d.double()).abs().max())
    assert dev <= tol, (dev, tol)


def test_q8_stochastic_rounding_is_unbiased():
    _assert_unbiased(True)


def test_q8_bytes_formula_aliases_excluded():
    for tied in (True, False):
        a = tiny(1, tied=tied)
        m = a.manifest
        want = sum(m.entries[k].numel for k in m.shared_keys) + 4 * len(m.shared_keys)
        assert Q.q8_payload_bytes(m) == want and Q.q8_upload_bytes(m) == want + 8    # + n_consumed header
        theta = a.broadcast_state(clone=True)
        up = OrderedDict((k, theta[k] + 1e-3) for k in m.shared_keys)
        assert Q.payload_bytes_q8(Q.encode(up, theta, m.shared_keys, seed=1, round_idx=0, key="u")) == want
        vb = Q.q8_visit_bytes(m)
        assert vb["upload_bytes"] == want + 8 and vb["download_bytes"] == m.shared_bytes
        assert vb["send_plus_receive_bytes"] == m.shared_bytes + want + 8
    assert tiny(1, tied=True).manifest.alias_keys                              # the tied fixture has an alias


# ------------------------------------------------------------------------------------------------ one round
def _registered_server(tied=False):
    a = tiny(1, tied=tied)
    a.server_post_aggregate()
    th, init = a.broadcast_state(clone=True), state_digest(a.broadcast_state())
    return Server(a, init_sha256=init), th, init


def test_q8_round_equals_fedavg_of_dequantised_uploads_bitwise():
    server, th, init = _registered_server()
    workers = [tiny(101, tied=False)]
    sol = solver()
    cl = CS[:5]
    ch = Q.Q8Channel(server.manifest, seed=SEED, theta_fn=lambda: server.adapter.broadcast_state())
    rep = run_round(server, workers, cl, sol, round_idx=7, seed=SEED, n_shards=2, channel=ch)
    ch.check_round(7, [c.key for c in cl])
    # manual: the same client updates from theta_r, encoded / decoded with the same seeds, FedAvg over the same shards
    ref_w = tiny(101, tied=False)
    uploads = []
    for c in cl:
        res = client_update(ref_w, th, c, sol, round_idx=7, seed=SEED, clone_upload=True)
        pay = Q.encode(res.upload, th, server.manifest.shared_keys, seed=SEED, round_idx=7, key=c.key)
        uploads.append((c.key, Q.decode(pay, th), res.n_consumed))
    a2 = tiny(1, tied=False)
    a2.load_state_(th)
    s2 = Server(a2, init_sha256=init)
    s2.apply(aggregate_uploads(s2.manifest, th, uploads, n_shards=2))
    assert server.digest() == s2.digest() == rep.state_digest
    assert rep.bytes_up == len(cl) * Q.q8_upload_bytes(server.manifest)
    assert rep.bytes_down == len(cl) * server.manifest.shared_bytes


def test_q8_check_round_detects_a_skipped_decode_and_pool_refused():
    server, _, _ = _registered_server()
    ch = Q.Q8Channel(server.manifest, seed=SEED, theta_fn=lambda: server.adapter.broadcast_state())
    ch.download(0, "a")
    assert_raises(Q.Q8Error, ch.check_round, 0, ["a"])
    assert_raises(Q.Q8Error, ch.upload_counted, 0, "a")
    assert_raises(Q.Q8Error, ch.upload, 1, "a", server.adapter.extract_shared(clone=True))   # no round-1 download


# ------------------------------------------------------------------------------------------------ FLRun integration
def _plan(group=4):
    return ParticipationPlan.build(list(BY_KEY), seed=SEED, manifest_hash="q8-tests", group_size=group,
                                   sampling="sweep")


def _cfg(**kw):
    base = {"run_id": "q8", "method": "FA", "seed": SEED, "lr_peak_local": 0.05, "solver": solver(), "n_shards": 2,
                "exposure_point": "end", "dropout_p": 0.1, "endpoint_rounds": 10}
    base.update(kw)
    return RunConfig(**base)


def _flrun(q8=True, ckpt=None, resume=False, **kw):
    server, _, _ = _registered_server()
    args = (_cfg(**kw), _plan(), server, BY_KEY.__getitem__, N_DEC)
    kwa = {"workers": [tiny(101, tied=False)], "ckpt": ckpt}
    run = FLRun.resume(*args, **kwa) if resume else FLRun(*args, **kwa)
    ch = Q.install_q8(run, seed=SEED) if q8 else None
    return run, ch


def _steps(run, ch, n):
    for _ in range(n):
        out = run.step()
        if ch is not None:
            assert run.channel is ch
            ch.check_round(out["round"], out["survived"])


def test_flrun_q8_ledger_bytes_and_differs_from_fa():
    run, ch = _flrun()
    _steps(run, ch, 10)
    surv = sum(x["n_survived"] for x in run.participation_log)
    _msgs, down, up, _retry = run.ledger.snapshot()
    assert up == surv * Q.q8_upload_bytes(run.server.manifest) and down == surv * run.server.manifest.shared_bytes
    fa, _ = _flrun(q8=False)
    _steps(fa, None, 10)
    assert fa.server.digest() != run.server.digest()
    again, ch2 = _flrun()
    _steps(again, ch2, 10)
    assert again.server.digest() == run.server.digest()                       # determinism (bitwise)


def test_flrun_q8_resume_is_exact(tmp_path):
    straight, ch = _flrun()
    _steps(straight, ch, 10)
    part, chp = _flrun(ckpt=CheckpointManager(tmp_path / "ck", "q8", retain_weights=False))
    _steps(part, chp, 4)
    part.ckpt.save_latest(part.state_dict())
    res, chr_ = _flrun(ckpt=CheckpointManager(tmp_path / "ck", "q8", retain_weights=False), resume=True)
    assert res.state.cursor == 4
    _steps(res, chr_, 6)
    assert res.server.digest() == straight.server.digest()
    assert res.ledger.snapshot() == straight.ledger.snapshot()


def test_install_q8_refuses_non_fa_configs():
    for kw in ({"method": "FP", "mu": 0.01},):
        run, _ = _flrun(q8=False, **kw)
        assert_raises(Q.Q8Error, Q.install_q8, run, seed=SEED)
    server, _, _ = _registered_server()
    plan = ParticipationPlan.build(list(BY_KEY), seed=SEED, manifest_hash="q8-tests", group_size=4,
                                   sampling="poisson")                  # DP runs are Poisson
    dp = FLRun(_cfg(dropout_p=0.0, dp=DPConfig(1.0, 0.0, 4)), plan, server, BY_KEY.__getitem__, N_DEC,
               workers=[tiny(101, tied=False)])
    assert_raises(Q.Q8Error, Q.install_q8, dp, seed=SEED)


# ------------------------------------------------------------------------------------------------ negative controls
@nc("deterministic (round-to-nearest) rounding is biased: the mean of many roundings does not approach delta")
def test_nc_q8_deterministic_rounding_is_biased():
    _assert_unbiased(False)


@nc("a rounding seed without the client key gives two clients the same rounding noise")
def test_nc_q8_seed_without_client_key():
    d = _delta(500, 9)
    g1 = torch.Generator().manual_seed(Q.derive_seed(SEED, "q8", 3))            # MUTANT: key dropped
    g2 = torch.Generator().manual_seed(Q.derive_seed(SEED, "q8", 3))
    q1, _ = Q.quantize_tensor(d, g1)
    q2, _ = Q.quantize_tensor(d, g2)
    assert not torch.equal(q1, q2), "clients must draw independent rounding noise"


@nc("the Q8 run is not FA: quantised uploads change the trajectory")
def test_nc_q8_equals_fa():
    run, ch = _flrun()
    _steps(run, ch, 3)
    fa, _ = _flrun(q8=False)
    _steps(fa, None, 3)
    assert fa.server.digest() == run.server.digest()
