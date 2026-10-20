"""SASRec-style output-table recentering under FL: the server recenters at init and after aggregation, the FedProx
anchor is the broadcast (recentered) theta_r, and scores at a fixed query are invariant (per-query constant shift).

Runs on the tiny untied model (a deliberate +0.3 common mode; raw, non-registered state, so the server recenters it
once at init) and on the Clean V2 SASRec (small K), whose REGISTERED theta0 is already recentered and is therefore
started with Server(..., init_sha256=...) and never recentered again (Ops lock decision on MODEL-01 risk 1).
Proves: gauge consistency of theta, anchor and aggregate; loss/ranking invariance at FIXED queries.
Does not prove: trajectory equivalence with an un-recentered run (a separate guarantee, package 06 §4).
"""
from __future__ import annotations

import pytest
import torch

import fl.client as flc
import fl.server as fls
from fl.client import client_update, make_optimizer
from fl.numerics import states_equal
from fl.server import Server, run_round
from fl_testkit import clients, nc, one_client, solver, tiny

SEED = 2026
ORIG_PROX = flc._add_prox_grad_


def _sasrec_built(seed=2026):
    from fl.families import build_v2
    return build_v2("SASREC", seed=seed, K=64)


def _sasrec(seed=2026):
    return _sasrec_built(seed).adapter


def _sasrec_server():
    bm = _sasrec_built()
    return Server(bm.adapter, init_sha256=bm.init_sha256)


def _tiny_server(recenter_raw_init=True):
    return Server(tiny(1, tied=False), recenter_raw_init=recenter_raw_init)


def _sasrec_clients():
    from fl.families import synthetic_examples_v2
    from fl.client import ClientData
    g = torch.Generator().manual_seed(4)
    return [ClientData("user-s0", synthetic_examples_v2(6, 64, g)),
            ClientData("user-s1", synthetic_examples_v2(20, 64, g))]


def common_mode(adapter) -> float:
    W, b = adapter.head_weight().detach(), adapter.head_bias().detach()
    return max(float(W.mean(0).abs().max()), abs(float(b.mean())))


def check_gauge(make_server, make_worker, cs, rounds=2):
    srv, w = make_server(), [make_worker()]
    assert common_mode(srv.adapter) < 1e-6, f"broadcast theta0 not recentered (common mode {common_mode(srv.adapter):.3g})"
    for r in range(rounds):
        captured = {}
        orig_apply = Server.apply

        def spy(self, new_state):
            captured["agg"] = {k: v.clone() for k, v in new_state.items()}
            orig_apply(self, new_state)
        Server.apply = spy
        try:
            run_round(srv, w, cs, solver(lr=0.01), round_idx=r, seed=SEED, mu=0.01)
        finally:
            Server.apply = orig_apply
        ref_adapter = make_worker()
        ref_adapter.load_state_(captured["agg"])
        ref_adapter.server_post_aggregate()
        assert states_equal(ref_adapter.broadcast_state(), srv.adapter.broadcast_state()), \
            "server state must be recenter(aggregate)"
        assert common_mode(srv.adapter) < 1e-6


def check_anchor(make_server, make_worker, cs, prox=None):
    srv, w = make_server(), [make_worker()]
    for r in range(2):
        theta_r = srv.broadcast()
        seen = []

        def spy(params, anchors, mu):
            if not seen:
                seen.append([a.clone() for a in anchors])
            (prox or ORIG_PROX)(params, anchors, mu)
        flc._add_prox_grad_ = spy
        try:
            one = one_client("user-one", 16, seed=5) if cs is None else cs[0]
            sol1 = solver(lr=0.01, passes=1, batch_size=10_000)                 # exactly one local step
            fa = client_update(w[0], theta_r, one, sol1, round_idx=r, seed=SEED, clone_upload=True)
            fp = client_update(w[0], theta_r, one, sol1, round_idx=r, seed=SEED, mu=0.5, clone_upload=True)
        finally:
            flc._add_prox_grad_ = ORIG_PROX
        assert all(torch.equal(a, theta_r[k]) for a, k in zip(seen[0], w[0].manifest.shared_keys)), \
            "the proximal anchor must be the broadcast theta_r"
        assert states_equal(fa.upload, fp.upload), "at theta = theta_r (same gauge) the first FP step must equal FA"
        run_round(srv, w, clients(4, seed=101) if cs is None else cs, solver(lr=0.01), round_idx=r, seed=SEED, mu=0.5)


def check_invariance(adapter, recenter):
    torch.manual_seed(0)
    with torch.no_grad():
        adapter.head_weight().add_(torch.randn(adapter.query_dim) * 0.5)    # inject a visible common mode
        adapter.head_bias().add_(1.0)
    q = torch.randn(5, adapter.query_dim)
    tgt = torch.randint(0, adapter.K, (5,))
    with torch.no_grad():
        before = adapter.logits(q)
        recenter(adapter)
        after = adapter.logits(q)
    shift = after - before
    assert float((shift - shift.mean(1, keepdim=True)).abs().max()) < 1e-4, "recentering must be a per-query shift"
    assert torch.allclose(torch.log_softmax(before, 1), torch.log_softmax(after, 1), atol=1e-4)
    assert torch.equal(before.topk(20).indices, after.topk(20).indices), "top-20 must be unchanged"
    ce0 = torch.nn.functional.cross_entropy(before, tgt, reduction="none")
    ce1 = torch.nn.functional.cross_entropy(after, tgt, reduction="none")
    assert torch.allclose(ce0, ce1, atol=1e-4)
    assert common_mode(adapter) < 1e-5


def _correct_recenter(adapter):
    adapter.post_step()


@torch.no_grad()
def _wrong_recenter(adapter):                     # WRONG: removes each row's mean over d (not the common mode)
    W = adapter.head_weight()
    W.sub_(W.mean(1, keepdim=True))
    b = adapter.head_bias()
    b.sub_(b.mean())


# ------------------------------------------------------------------------------------------------ tests
def test_recenter_server_gauge_tiny():
    check_gauge(_tiny_server, lambda: tiny(2, tied=False), clients(5, seed=102))


def test_recenter_server_gauge_sasrec():
    check_gauge(_sasrec_server, lambda: _sasrec(7), _sasrec_clients(), rounds=1)


def test_recenter_anchor_is_broadcast_theta_r_tiny():
    check_anchor(_tiny_server, lambda: tiny(2, tied=False), None)


def test_recenter_anchor_is_broadcast_theta_r_sasrec():
    check_anchor(_sasrec_server, lambda: _sasrec(7), _sasrec_clients())


@pytest.mark.parametrize("which", ["tiny", "sasrec"])
def test_recenter_scores_invariant_at_fixed_query(which):
    check_invariance(tiny(3, tied=False) if which == "tiny" else _sasrec(), _correct_recenter)


def test_recenter_leaves_optimizer_moments_alone():
    a = tiny(3, tied=False)
    opt = make_optimizer(a, solver(lr=0.01))
    b = one_client("user-m", 16).examples
    loss = torch.nn.functional.cross_entropy(a.scores(b), b["target_class"])
    loss.backward()
    opt.step()
    st = {k: v.clone() for k, v in opt.state[a.module.output_embed].items()}
    a.post_step()
    assert all(torch.equal(st[k], opt.state[a.module.output_embed][k]) for k in st)
    assert common_mode(a) < 1e-6


@nc("server does not recenter a raw theta0 before the first broadcast")
def test_nc_recenter_no_init_recenter():
    check_gauge(lambda: _tiny_server(recenter_raw_init=False), lambda: tiny(2, tied=False), clients(5, seed=102))


@nc("server skips recentering after aggregation")
def test_nc_recenter_skip_after_aggregation(monkeypatch):
    def apply_without_recenter(self, new_state):
        self.adapter.load_state_(new_state)
        self.round += 1
    monkeypatch.setattr(fls.Server, "apply", apply_without_recenter)
    check_gauge(_tiny_server, lambda: tiny(2, tied=False), clients(5, seed=102))


@nc("proximal anchor = raw (un-recentered) theta0")
def test_nc_recenter_raw_theta0_anchor():
    raw = tiny(1, tied=False)                                                # same init, never recentered
    a0 = [raw.broadcast_state()[k] for k in raw.manifest.shared_keys]
    check_anchor(_tiny_server, lambda: tiny(2, tied=False), None,
                 prox=lambda params, anchors, mu: ORIG_PROX(params, a0, mu))


@nc("recentering removes the per-row mean over d instead of the common mode")
def test_nc_recenter_wrong_axis():
    check_invariance(tiny(3, tied=False), _wrong_recenter)
