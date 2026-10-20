"""PF private vector: p = 0 recovers the shared forward; p_u and its AdamW moments persist across rounds
under its private key; a new (or empty-history) user starts at p = 0; keys are unique and never a shared OOV token;
CPF uses the same forward and the matched regularizer; p has the query dimension.

Proves: the PF forward / state lifecycle mechanics. Does not prove: that personalization helps accuracy.
"""
from __future__ import annotations

import pytest
import torch
from fedsim_testkit import assert_raises, clients, nc, one_client, server_and_worker, solver, tiny

import ppsi.fedsim.client as flc
from ppsi.fedsim.client import PFConfig, client_update
from ppsi.fedsim.numerics import states_equal
from ppsi.fedsim.personal import (
    PersonalStore,
    PrivateKeyError,
    assert_unique_keys,
    cpf_loss,
    pf_step_loss,
    validate_private_key,
)
from ppsi.fedsim.server import run_round

SEED = 2026
PF = PFConfig(lr=0.05, lam=1e-4)


def _eval_batch(adapter, n=9, seed=0):
    return one_client("user-eval", n, seed=seed).examples


def check_p0_forward(adapter, init_p):
    adapter.module.eval()
    b = _eval_batch(adapter)
    with torch.no_grad():
        ref = adapter.scores(b)
        got = adapter.scores(b, init_p(adapter))
    assert torch.equal(ref, got), "p = 0 must reproduce the shared-model scores bitwise"


def _store_p0(adapter):
    return PersonalStore(adapter.query_dim).p("user-new")


def _random_p0(adapter):                                   # WRONG: non-zero initialization
    return 1e-3 * torch.randn(adapter.query_dim)


def check_persistence(rounds=3, forgetful=False):
    srv, w = server_and_worker(seed=7)
    cs = clients(4, seed=51, sizes=[20, 9, 33, 5])
    store = PersonalStore(srv.adapter.query_dim)
    steps = {c.key: 0 for c in cs}
    for r in range(rounds):
        committed = {c.key: store.get(c.key) for c in cs}
        theta = srv.broadcast()
        rep = run_round(srv, w, cs, solver(), round_idx=r, seed=SEED, pf=PF, store=store)
        for v in rep.visits:
            steps[v.key] += v.steps
        # replay one client's visit from what was committed BEFORE the round: must equal what was committed after
        c = cs[r % len(cs)]
        again = client_update(tiny(99), theta, c, solver(), round_idx=r, seed=SEED, pf=PF, personal=committed[c.key])
        after = store.get(c.key)
        assert torch.equal(again.personal.p, after.p), "committed p_u must be the continuation of the stored p_u"
        if forgetful:
            store._s.clear()                                 # WRONG: private state dropped between rounds
    for c in cs:
        st = store.get(c.key)
        assert st is not None and float(st.p.abs().sum()) > 0, "p_u must move and persist"
        assert int(st.opt_state["step"]) == steps[c.key], "personal AdamW moments must persist across rounds"
        assert st.visits == rounds


def check_new_user_zero(key_fn):
    srv, w = server_and_worker(seed=8)
    store = PersonalStore(srv.adapter.query_dim)
    trained = one_client(key_fn("user-A"), 30, seed=3)
    run_round(srv, w, [trained], solver(), round_idx=0, seed=SEED, pf=PF, store=store)
    assert float(store.p(key_fn("user-A")).abs().sum()) > 0
    assert torch.equal(store.p(key_fn("user-B")), torch.zeros(srv.adapter.query_dim)), "a new user must start at 0"


# ------------------------------------------------------------------------------------------------ tests
@pytest.mark.parametrize("tied", [True, False])
def test_p0_equals_no_personal_forward_tiny(tied):
    check_p0_forward(tiny(1, tied=tied), _store_p0)


def test_pf_zero_lr_zero_lambda_equals_fa():
    cs = clients(6, seed=52)
    a, wa = server_and_worker(seed=9)
    b, wb = server_and_worker(seed=9)
    store = PersonalStore(a.adapter.query_dim)
    for r in range(2):
        run_round(a, wa, cs, solver(), round_idx=r, seed=SEED)
        run_round(b, wb, cs, solver(), round_idx=r, seed=SEED, pf=PFConfig(lr=0.0, lam=0.0), store=store)
    assert states_equal(a.adapter.broadcast_state(), b.adapter.broadcast_state())
    assert all(float(store.p(k).abs().sum()) == 0.0 for k in store.keys())  # noqa: SIM118  (a store, not a dict)


def test_cpf_same_forward_and_regularizer():
    a = tiny(2)
    a.module.eval()
    b = _eval_batch(a, n=6)
    P = torch.randn(3, a.query_dim)
    lam = 1e-3
    with torch.no_grad():
        one_user = cpf_loss(a, b, P, torch.full((6,), 1, dtype=torch.long), lam)
        pf = pf_step_loss(a, b, P[1], lam)
        assert torch.allclose(one_user, pf, rtol=0, atol=1e-7), (one_user, pf)
        mixed = cpf_loss(a, b, P, torch.tensor([0, 0, 2, 2, 2, 0]), lam)
        ce = torch.nn.functional.cross_entropy(a.scores(b, P[[0, 0, 2, 2, 2, 0]]), b["target_class"])
        reg = 0.5 * lam * (P[0].pow(2).sum() + P[2].pow(2).sum())            # each touched user once
        assert torch.allclose(mixed, ce + reg, rtol=0, atol=1e-6)


def test_p_persists_across_rounds():
    check_persistence()


def test_new_user_starts_at_zero_with_unique_identity():
    check_new_user_zero(lambda u: u)
    for bad in (2, 0, "2", "OOV", "<oov>", "", "  ", None):
        assert_raises(PrivateKeyError, validate_private_key, bad)
    assert_raises(PrivateKeyError, assert_unique_keys, ["user-a", "user-b", "user-a"])
    srv, w = server_and_worker(seed=8)
    dup = [one_client("user-a", 5), one_client("user-a", 7, seed=1)]
    assert_raises(PrivateKeyError, run_round, srv, w, dup, solver(), round_idx=0, seed=1, pf=PF,
                  store=PersonalStore(srv.adapter.query_dim))


def test_empty_history_client_stays_zero():
    srv, w = server_and_worker(seed=10)
    store = PersonalStore(srv.adapter.query_dim)
    cs = [one_client("user-empty", 6, invalid_frac=1.0), one_client("user-full", 12)]
    rep = run_round(srv, w, cs, solver(), round_idx=0, seed=SEED, pf=PF, store=store)
    assert rep.n_noop == 1 and "user-empty" not in store and "user-full" in store
    assert torch.equal(store.p("user-empty"), torch.zeros(srv.adapter.query_dim))


def test_store_returns_clones():
    store = PersonalStore(4)
    store.commit("user-a", flc.PersonalState(torch.ones(4), {"step": torch.tensor(1.0)}))
    got = store.get("user-a")
    got.p.add_(5.0)
    assert torch.equal(store.p("user-a"), torch.ones(4)), "a worker must never mutate committed state in place"


@nc("p initialized non-zero")
def test_nc_nonzero_init():
    check_p0_forward(tiny(1), _random_p0)


@nc("private state forgotten between rounds")
def test_nc_forgetful_store():
    check_persistence(forgetful=True)


@nc("private state keyed by a shared bucket (OOV-like) instead of the user")
def test_nc_bucket_key():
    check_new_user_zero(lambda u: "shared-bucket")
