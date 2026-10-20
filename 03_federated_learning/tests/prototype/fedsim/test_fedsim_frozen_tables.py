"""Frozen item tables in the simulator (CPU, the tiny untied SASRec-like family).

Proves: freeze_item_tables turns item_embed.weight / output_embed / output_bias into FIXED buffers (requires_grad False,
out of the shared set, the optimizer and the upload), idempotently, and refuses a tied table, half of the recentering
pair, an unknown key or another key set; the frozen tables stay BITWISE identical over a whole FLRun (FedAvg, FedAvgM and
DP-FedAvg) while the encoder moves; the ledger's per-visit bytes = the uploaded (shared) parameters (down = shared bytes,
up = shared bytes + 8), visit_bytes reports the frozen tables once per client; the aggregators refuse an upload that
carries a frozen key and a worker whose frozen copy differs; the DP clip / noise dimension is the uploaded parameter
count; a frozen run resumes bitwise (and never into an unfrozen run); pools and mismatched workers are refused.
NCs: an unfrozen run keeps the tables bitwise; a frozen output layer keeps its bits with the recentering hook left on.
"""
from __future__ import annotations

from collections import OrderedDict

import torch
from fedsim_testkit import assert_raises, clients, nc, solver, tiny

from ppsi.fedsim.adapter import BufferRule, ManifestError, Role
from ppsi.fedsim.aggregate import AggregationError, ShardAccumulator
from ppsi.fedsim.checkpoint import CheckpointError, CheckpointManager
from ppsi.fedsim.client import valid_rows
from ppsi.fedsim.comm import UPLOAD_HEADER_BYTES, visit_bytes
from ppsi.fedsim.dp import DPConfig, noise_like
from ppsi.fedsim.numerics import state_digest
from ppsi.fedsim.participation import ParticipationPlan
from ppsi.fedsim.runtime import (
    FROZEN_ITEM_KEYS,
    FLRun,
    FreezeError,
    RunConfig,
    freeze_item_tables,
    frozen_keys_of,
)
from ppsi.fedsim.server import RoundAborted, Server, ServerOptConfig, run_round

FK = FROZEN_ITEM_KEYS


def untied(seed=1):
    return tiny(seed, tied=False)                     # recentering output table, like SASRec v3


def frozen_pair(seed=1, wseed=9):
    s, w = untied(seed), untied(wseed)
    freeze_item_tables(s, FK)
    freeze_item_tables(w, FK, source=s.broadcast_state())
    return s, w


def tables(adapter):
    st = adapter.broadcast_state(clone=True)
    return OrderedDict((k, st[k]) for k in FK)


def _cfg(**kw):
    base = {"run_id": "frozen", "method": "FA", "seed": 5, "lr_peak_local": 0.05, "solver": solver(), "exposure_point": "end",
                "dropout_p": 0.1, "endpoint_rounds": 10, "n_shards": 2, "ckpt_every": 3}
    base.update(kw)
    return RunConfig(**base)


def _run(tmp_path, cfg, name, resume=False, frozen=True, plan_kw=None):
    cl = {c.key: c for c in clients(8)}
    pk = {"group_size": 3, "sampling": "sweep"} if plan_kw is None else plan_kw
    plan = ParticipationPlan.build(sorted(cl), seed=cfg.seed, manifest_hash="m", **pk)
    if frozen:
        s, w = frozen_pair()
    else:
        s, w = untied(), untied(9)
    srv = Server(s, init_sha256=state_digest(s.broadcast_state()))
    n_dec = sum(int(valid_rows(c.examples).numel()) for c in cl.values())
    ck = CheckpointManager(tmp_path / name, cfg.run_id)
    return (FLRun.resume if resume else FLRun)(cfg, plan, srv, cl.__getitem__, n_dec, workers=[w], ckpt=ck)


# ------------------------------------------------------------------------------------------------ the freeze
def test_freeze_roles_grad_and_refusals():
    a = untied()
    before = state_digest(a.broadcast_state())
    n_shared = a.manifest.shared_numel
    rec = freeze_item_tables(a, FK)
    m = a.manifest
    assert frozen_keys_of(a) == FK and all(m.entries[k].role == Role.BUFFER and m.entries[k].rule == BufferRule.FIXED
                                           for k in FK)
    assert not set(FK) & set(m.shared_keys) and set(a.extract_shared()) == set(m.shared_keys)
    named = dict(a.module.named_parameters())
    assert all(not named[k].requires_grad for k in FK)
    assert all(p.requires_grad for p in a.shared_parameters())
    groups = a.param_groups(1e-5)
    ids = {id(p) for g in groups for p in g["params"]}
    assert not any(id(named[k]) in ids for k in FK)
    assert rec["shared_numel"] + rec["frozen_numel"] == n_shared == m.shared_numel + rec["frozen_numel"]
    assert state_digest(a.broadcast_state()) == before            # the same state (keys and bits)
    assert freeze_item_tables(a, FK)["manifest_digest"] == rec["manifest_digest"]   # idempotent
    assert_raises(FreezeError, freeze_item_tables, a, ("item_embed.weight",))
    assert_raises(FreezeError, freeze_item_tables, untied(), ("output_embed",))       # half of the recentering pair
    assert_raises(FreezeError, freeze_item_tables, untied(), ("nope.weight",))
    assert_raises(FreezeError, freeze_item_tables, tiny(1), ("item_embed.weight",))   # tied (alias) table
    a.post_step()
    a.server_post_aggregate()
    assert state_digest(a.broadcast_state()) == before            # no recentering of the frozen output layer


def test_bootstrap_copies_the_frozen_tables_once():
    s, w = frozen_pair(1, 9)
    assert all(torch.equal(tables(s)[k], tables(w)[k]) for k in FK)
    enc_s, enc_w = s.extract_shared(), w.extract_shared()
    assert any(not torch.equal(enc_s[k], enc_w[k]) for k in enc_s)   # only the tables were copied


# ------------------------------------------------------------------------------------------------ runs
def test_frozen_tables_bitwise_identical_and_encoder_moves(tmp_path):
    for name, kw in (("fedavg", {}), ("fedavgm", {"server_opt": ServerOptConfig("fedavgm", lr=1.0)})):
        r = _run(tmp_path, _cfg(**kw), name)
        t0 = tables(r.server.adapter)
        e0 = r.server.adapter.extract_shared(clone=True)
        r.run()
        assert r.state.done and all(torch.equal(t0[k], tables(r.server.adapter)[k]) for k in FK), name
        assert all(torch.equal(t0[k], tables(r.workers[0])[k]) for k in FK)
        e1 = r.server.adapter.extract_shared()
        assert any(not torch.equal(e0[k], e1[k]) for k in e0), name
        assert r.summary()["participation"]["survived"] > 0


def test_bytes_per_visit_are_the_uploaded_count(tmp_path):
    r = _run(tmp_path, _cfg(), "b")
    r.run()
    man = r.server.manifest
    snap = r.ledger.snapshot()
    n_vis = sum(r.counters.visit_counts)
    assert snap[1] == n_vis * man.shared_bytes and snap[2] == n_vis * (man.shared_bytes + UPLOAD_HEADER_BYTES)
    vb = visit_bytes(man, frozen_keys=FK)
    full = untied().manifest
    frozen_bytes = sum(full.entries[k].nbytes for k in FK)
    assert vb["download_bytes"] == man.shared_bytes == full.shared_bytes - frozen_bytes
    assert vb["frozen_bytes_once"] == frozen_bytes and vb["uploaded_numel"] == man.shared_numel
    assert visit_bytes(full) == {k: v for k, v in visit_bytes(full).items()}     # unchanged record without frozen
    assert "frozen_bytes_once" not in visit_bytes(man)
    assert_raises(ValueError, visit_bytes, full, frozen_keys=FK)                 # not frozen in that manifest


def test_aggregation_ignores_and_refuses_frozen_params():
    s, w = frozen_pair()
    theta = s.broadcast_state(clone=True)
    acc = ShardAccumulator.empty(0, s.manifest, theta)
    up = dict(w.extract_shared(clone=True))
    up["output_embed"] = theta["output_embed"] + 1.0              # a client that uploads a frozen table
    assert_raises(AggregationError, acc.add, 0, "u", up, 5)
    w2 = untied(9)
    freeze_item_tables(w2, FK, source=s.broadcast_state())
    with torch.no_grad():
        dict(w2.module.named_parameters())["item_embed.weight"][3, 0] += 1.0     # a stale / tampered local copy
    assert_raises(ManifestError, w2.load_state_, theta)
    cl = clients(4)
    assert_raises(RoundAborted, run_round, Server(s), [w2], cl, solver(), round_idx=0, seed=5)
    rep = run_round(Server(s), [w], cl, solver(), round_idx=0, seed=5)     # the server keeps its own tables
    assert rep.weight > 0 and all(torch.equal(theta[k], s.broadcast_state()[k]) for k in FK)


def test_dp_dimension_is_the_uploaded_count(tmp_path):
    s, w = frozen_pair()
    before = tables(s)
    srv = Server(s)
    cl = clients(4)
    dp = DPConfig(0.5, 0.8, len(cl))
    rep = run_round(srv, [w], cl, solver(), round_idx=0, seed=5, dp=dp)
    assert rep.dp["noise_std"] > 0 and all(torch.equal(before[k], tables(s)[k]) for k in FK)
    nz = noise_like(s.manifest.shared_keys, s.broadcast_state(), seed=5, round_idx=0, std=1.0)
    assert set(nz) == set(s.manifest.shared_keys) and sum(t.numel() for t in nz.values()) == s.manifest.shared_numel
    assert not set(FK) & set(nz)
    cfg = _cfg(dropout_p=0.0, endpoint_rounds=6, dp=DPConfig(0.5, 0.8, 3))
    r = _run(tmp_path, cfg, "dp", plan_kw={"group_size": 3, "sampling": "poisson"})
    t0 = tables(r.server.adapter)
    r.run()
    assert all(torch.equal(t0[k], tables(r.server.adapter)[k]) for k in FK)


def test_frozen_resume_bitwise_and_refusals(tmp_path):
    cfg = _cfg()
    straight = _run(tmp_path, cfg, "s").run()
    part = _run(tmp_path, cfg, "p")
    part.run(6)
    res = _run(tmp_path, cfg, "p", resume=True)
    assert res.state.cursor == 6
    res.run()
    assert straight.summary()["server_digest"] == res.summary()["server_digest"]
    assert straight.ledger.snapshot() == res.ledger.snapshot()
    unfrozen = _run(tmp_path, cfg, "u", frozen=False)
    assert_raises(CheckpointError, unfrozen.load_state_dict, part.state_dict())   # manifest digest differs
    assert unfrozen.run().summary()["server_digest"] != straight.summary()["server_digest"]


def test_flrun_refuses_pool_and_mismatched_workers(tmp_path):
    cl = {c.key: c for c in clients(8)}
    plan = ParticipationPlan.build(sorted(cl), seed=5, manifest_hash="m", group_size=3, sampling="sweep")
    s, _w = frozen_pair()
    srv = Server(s, init_sha256=state_digest(s.broadcast_state()))
    assert_raises(ValueError, FLRun, _cfg(), plan, srv, cl.__getitem__, 1000, workers=[untied(9)])
    assert_raises(ValueError, FLRun, _cfg(), plan, srv, cl.__getitem__, 1000, pool=object())


# ------------------------------------------------------------------------------------------------ negative controls
@nc("without the freeze the item tables move (the key property 'tables bitwise identical' must fail)")
def test_nc_unfrozen_tables_bitwise_identical(tmp_path):
    r = _run(tmp_path, _cfg(), "nc", frozen=False)
    t0 = tables(r.server.adapter)
    r.run()
    assert all(torch.equal(t0[k], tables(r.server.adapter)[k]) for k in FK)


@nc("a frozen output layer whose recentering hook is left on changes bits at every server step")
def test_nc_recentering_left_on_keeps_bits(tmp_path):
    s, w = frozen_pair()
    srv = Server(s, init_sha256=state_digest(s.broadcast_state()))
    s.post_step = type(s).post_step.__get__(s)
    s.server_post_aggregate = type(s).server_post_aggregate.__get__(s)
    t0 = tables(s)
    run_round(srv, [w], clients(4), solver(), round_idx=0, seed=5)
    assert all(torch.equal(t0[k], tables(s)[k]) for k in FK)
