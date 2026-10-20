"""T34 — deterministic logical-order aggregation across 2 concurrent worker processes; exact retry accounting.

Proves: FA / FP / PF rounds run on 2 concurrent spawned worker processes (distinct PIDs, arbitrary completion order)
are bit-identical to the in-process single-worker driver with the same shard count; shard sums are combined in
shard-index order whatever the arrival order; a failed visit retried with the identical seed gives the no-failure
bits, counts n_consumed once and reports the wasted steps and retry bytes; exhausted retries abort the round with the
server state and every p_u unchanged.
Tolerance: exact, given the fixed intra-op thread count (1) in every process (CPU reductions may depend on it).
"""
from __future__ import annotations

import itertools
import json
from collections import OrderedDict
from pathlib import Path

import pytest
import torch

import fl.server as fls
from fl.adapter import build_manifest
from fl.aggregate import ShardAccumulator, combine_shards
from fl.client import FaultPlan, PFConfig
from fl.personal import PersonalStore
from fl.pool import ProcessPool, WorkerSpec
from fl.server import RoundAborted, Server, run_round
from fl.synthetic import make_tiny_adapter
from fl_testkit import assert_raises, clients, nc, solver, tiny

SEED = 2026
KW = (("K", 24), ("d", 8), ("tied", True), ("dropout", 0.1), ("seed", 500))
SPEC = WorkerSpec("fl.synthetic:make_tiny_adapter", KW, threads=1)
PF = PFConfig(lr=0.05, lam=1e-4)
CS = clients(10, seed=91, sizes=[90, 80, 70, 60, 50, 2, 3, 4, 5, 6])     # shard 0 heavy: arrives last
METHODS = {"FA": dict(mu=0.0, pf=None), "FP": dict(mu=0.05, pf=None), "PF": dict(mu=0.0, pf=PF)}
RECEIPTS = Path(__file__).resolve().parent / "receipts"
ORIG_UPDATE = fls.client_update


@pytest.fixture(scope="module")
def pool():
    p = ProcessPool(SPEC, n_workers=2)
    yield p
    p.close()


def _server():
    return Server(tiny(1, K=24, d=8, tied=True, dropout=0.1))


def run_inprocess(method, n_shards, rounds=2, fault=None):
    srv = _server()
    worker = make_tiny_adapter(**dict(KW))
    store = PersonalStore(srv.adapter.query_dim)
    reps = [run_round(srv, [worker], CS, solver(), round_idx=r, seed=SEED, store=store, n_shards=n_shards,
                      fault=fault, **METHODS[method]) for r in range(rounds)]
    return srv.digest(), store.digest(), reps


def run_pool(pool, method, n_shards, rounds=2, fault=None):
    srv = _server()
    store = PersonalStore(srv.adapter.query_dim)
    reps = [pool.run_round(srv, CS, solver(), round_idx=r, seed=SEED, store=store, n_shards=n_shards, fault=fault,
                           **METHODS[method]) for r in range(rounds)]
    return srv.digest(), store.digest(), reps


def check_retry_identical(runner):
    ref = runner(None)
    got = runner(FaultPlan(frozenset({(CS[1].key, 0, 3), (CS[7].key, 0, 1)})))
    assert got[0] == ref[0], "a retried visit must reproduce the no-failure server state bitwise"
    assert got[1] == ref[1], "a retried visit must reproduce the no-failure private states bitwise"
    rv = {v.key: v for v in got[2][0].visits}
    nv = {v.key: v for v in ref[2][0].visits}
    assert rv[CS[1].key].attempts == 2 and rv[CS[1].key].wasted_steps == 3
    assert rv[CS[7].key].attempts == 2 and rv[CS[7].key].wasted_steps == 1
    assert all(rv[k].n_consumed == nv[k].n_consumed for k in nv), "n_consumed counted once"
    assert got[2][0].weight == ref[2][0].weight
    assert got[2][0].retry_bytes_down == 2 * _server().manifest.shared_bytes


# ------------------------------------------------------------------------------------------------ tests
@pytest.mark.parametrize("method", ["FA", "FP", "PF"])
def test_t34_two_concurrent_workers_equal_inprocess(pool, method):
    out = {}
    for n_shards in (2, 4):
        a = run_inprocess(method, n_shards)
        b = run_pool(pool, method, n_shards)
        assert a[0] == b[0], f"{method} n_shards={n_shards}: pool server state != in-process"
        assert a[1] == b[1], f"{method} n_shards={n_shards}: pool private states != in-process"
        out[n_shards] = a[0]
    assert len(pool.pids_seen) == 2, f"expected 2 concurrent worker processes, saw {pool.pids_seen}"
    RECEIPTS.mkdir(exist_ok=True)
    path = RECEIPTS / "T34_POOL_RUNS.json"
    rec = json.loads(path.read_text()) if path.exists() else {}
    orders = pool.completion_orders[-4:]
    rec[method] = {"digest_by_n_shards": out, "equal_to_inprocess": True, "worker_pids": len(pool.pids_seen),
                   "completion_orders_last_4_rounds": orders,
                   "rounds_arriving_out_of_shard_order": sum(1 for o in orders if o != sorted(o))}
    path.write_text(json.dumps(rec, indent=1))


def test_t34_combine_is_arrival_order_invariant():
    check_combine(lambda shards: combine_shards(shards).total["x"])


def test_t34_retry_identical_inprocess():
    check_retry_identical(lambda f: run_inprocess("PF", 2, rounds=1, fault=f))


def test_t34_retry_identical_pool(pool):
    check_retry_identical(lambda f: run_pool(pool, "PF", 2, rounds=1, fault=f))


@pytest.mark.parametrize("where", ["inprocess", "pool"])
def test_t34_exhausted_retries_abort_round(pool, where):
    srv = _server()
    store = PersonalStore(srv.adapter.query_dim)
    kw = dict(seed=SEED, store=store, n_shards=2, pf=PF)
    w = [make_tiny_adapter(**dict(KW))]
    run_round(srv, w, CS, solver(), round_idx=0, **kw)
    before = (srv.digest(), store.digest(), srv.round)
    fault = FaultPlan(frozenset({(CS[6].key, 0, 1), (CS[6].key, 1, 1)}))
    if where == "inprocess":
        assert_raises(RoundAborted, run_round, srv, w, CS, solver(), round_idx=1, fault=fault, **kw)
    else:
        assert_raises(RoundAborted, pool.run_round, srv, CS, solver(), round_idx=1, fault=fault, **kw)
    assert (srv.digest(), store.digest(), srv.round) == before, "an aborted round must change nothing"


# ------------------------------------------------------------------------------------------------ negative controls
class _M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.x = torch.nn.Parameter(torch.zeros(1))


def check_combine(combine_fn):
    man = build_manifest(_M())
    tmpl = OrderedDict(x=torch.zeros(1))
    vals = [1e8, 1.0, -1e8, 3.0]
    shards = []
    for i, v in enumerate(vals):
        s = ShardAccumulator.empty(i, man, tmpl)
        s.add(i, f"u{i}", {"x": torch.tensor([v])}, 1)
        shards.append(s)
    results = {tuple(float(t) for t in combine_fn(list(p))) for p in itertools.permutations(shards)}
    assert len(results) == 1, f"combined result depends on arrival order: {sorted(results)}"


def _arrival_order_sum(shards):
    tot = shards[0].acc["x"].clone()
    for s in shards[1:]:
        tot.add_(s.acc["x"])
    return tot


@nc("shard sums added in arrival order")
def test_nc_t34_arrival_order_sum():
    check_combine(_arrival_order_sum)


@nc("retry re-seeded with the attempt number")
def test_nc_t34_retry_new_rng(monkeypatch):
    monkeypatch.setattr(fls, "client_update",
                        lambda *a, **kw: ORIG_UPDATE(*a, **{**kw, "seed": kw["seed"] + kw.get("attempt", 0)}))
    check_retry_identical(lambda f: run_inprocess("PF", 2, rounds=1, fault=f))


@nc("retry continues from the failed attempt's worker state instead of theta_r")
def test_nc_t34_retry_no_reload(monkeypatch):
    def cont(adapter, theta_r, *a, **kw):
        if kw.get("attempt", 0) > 0:
            theta_r = adapter.broadcast_state(clone=True)
        return ORIG_UPDATE(adapter, theta_r, *a, **kw)
    monkeypatch.setattr(fls, "client_update", cont)
    check_retry_identical(lambda f: run_inprocess("PF", 2, rounds=1, fault=f))
