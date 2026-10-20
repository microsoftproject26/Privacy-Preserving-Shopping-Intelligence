"""No private p_u or its optimizer state in any server payload; two clients on one worker never swap state.

Proves: (1) every upload is exactly the shared key set, and the raw bytes of each client's p_u and AdamW moments
appear nowhere in the serialized upload (a canary search that also catches a leak hidden under a shared key);
(2) a PF run with ONE reused worker is bit-identical (server state and every p_u / moment) to the same run with one
dedicated worker per client. Does not prove: resistance to inference from the shared update itself (not claimed).
"""
from __future__ import annotations

import io

import torch
from fedsim_testkit import assert_raises, clients, nc, server_and_worker, solver, tiny

import ppsi.fedsim.server as fls
from ppsi.fedsim.aggregate import AggregationError
from ppsi.fedsim.client import PFConfig
from ppsi.fedsim.comm import VirtualChannel
from ppsi.fedsim.numerics import states_equal, tensor_bytes
from ppsi.fedsim.personal import PersonalStore
from ppsi.fedsim.server import run_round

SEED = 2026
PF = PFConfig(lr=0.05, lam=1e-4)
ORIG_VISIT = fls.visit_with_retry
ORIG_UPDATE = fls.client_update


def check_no_private_in_payload(rounds=2):
    srv, w = server_and_worker(seed=12)
    cs = clients(5, seed=61, sizes=[18, 7, 26, 3, 40])
    store = PersonalStore(srv.adapter.query_dim)
    captured = []
    orig_add = fls.ShardAccumulator.add

    def spy(self, position, key, upload, n):
        buf = io.BytesIO()
        torch.save({k: v.detach().clone() for k, v in upload.items()}, buf)
        captured.append((key, set(upload), buf.getvalue()))
        return orig_add(self, position, key, upload, n)

    fls.ShardAccumulator.add = spy
    try:
        for r in range(rounds):
            before = len(captured)
            run_round(srv, w, cs, solver(), round_idx=r, seed=SEED, pf=PF, store=store)
            for key, ks, blob in captured[before:]:
                assert ks == set(srv.manifest.shared_keys), f"upload of {key} carries non-shared keys"
                st = store.get(key)
                if st is None:
                    continue
                private_tensors = [("p_u", st.p)] + [(f"adam.{n}", t) for n, t in st.opt_state.items() if t.numel() > 1]
                for name, t in private_tensors:
                    assert tensor_bytes(t) not in blob, f"private {name} of {key} found in its server payload"
    finally:
        fls.ShardAccumulator.add = orig_add


def check_no_swap_on_one_worker():
    cs = clients(4, seed=62, sizes=[14, 25, 6, 31])
    one_srv, one_w = server_and_worker(seed=13)
    ded_srv, _ = server_and_worker(seed=13)
    ded_w = [tiny(200 + i) for i in range(len(cs))]
    s1, s2 = PersonalStore(one_srv.adapter.query_dim), PersonalStore(ded_srv.adapter.query_dim)
    for r in range(3):
        run_round(one_srv, one_w, cs, solver(), round_idx=r, seed=SEED, pf=PF, store=s1, n_shards=len(cs))
        run_round(ded_srv, ded_w, cs, solver(), round_idx=r, seed=SEED, pf=PF, store=s2, n_shards=len(cs))
    assert states_equal(one_srv.adapter.broadcast_state(), ded_srv.adapter.broadcast_state()), "server state differs"
    assert s1.keys() == s2.keys() == sorted(c.key for c in cs)
    assert s1.digest() == s2.digest(), "a reused worker must restore each client's own p_u and moments"


# ------------------------------------------------------------------------------------------------ wrong variants
def _leaky_update(*a, **kw):
    """WRONG: hides the client's p_u inside a shared tensor of the upload (key set unchanged)."""
    res = ORIG_UPDATE(*a, **kw)
    if res.personal is not None and res.n_valid > 0:
        up = dict(res.upload)
        up["enc.bias"] = res.personal.p.detach().clone()
        res.upload = type(res.upload)(up)
    return res


def _caching_visit(adapter, theta_r, client, solver_, **kw):
    """WRONG: the worker keeps the last private state it saw instead of fetching the client's own by key."""
    cached = getattr(adapter, "_last_personal", None)
    if cached is not None:
        kw["get_personal"] = lambda _k: cached.clone()
    res = ORIG_VISIT(adapter, theta_r, client, solver_, **kw)
    if res.personal is not None:
        adapter._last_personal = res.personal
    return res


# ------------------------------------------------------------------------------------------------ tests
def test_no_private_state_in_payload():
    check_no_private_in_payload()


def test_extra_private_key_refused_by_channel_and_aggregator():
    srv, w = server_and_worker(seed=12)
    up = dict(w[0].extract_shared(clone=True))
    up["pf.p_u"] = torch.zeros(srv.adapter.query_dim)
    assert_raises(ValueError, VirtualChannel(srv.manifest).upload, 0, "user-x", up)
    from ppsi.fedsim.aggregate import ShardAccumulator
    acc = ShardAccumulator.empty(0, srv.manifest, srv.broadcast())
    assert_raises(AggregationError, acc.add, 0, "user-x", up, 3)


def test_two_clients_one_worker_no_swap():
    check_no_swap_on_one_worker()


@nc("p_u hidden in a shared tensor of the upload")
def test_nc_leaky_payload(monkeypatch):
    monkeypatch.setattr(fls, "client_update", _leaky_update)
    check_no_private_in_payload()


@nc("worker caches the last client's private state")
def test_nc_worker_state_swap(monkeypatch):
    monkeypatch.setattr(fls, "visit_with_retry", _caching_visit)
    check_no_swap_on_one_worker()
