"""PF: p persists across rounds, never in the payload, keyed by a private key (never OOV), a new user starts at
p=0, and worker reuse swaps no state.

Derived from the definition q_u = q_shared + p_u (p_u initializes at 0; p_u = 0 must recover the shared-model
predictions; keyed by a collision-free private user key, never by a shared OOV token), the separate p clip of 1.0
(which makes PF(lr_p=0, lambda=0) == FA), and the personal.py / client.py contracts.
"""
from __future__ import annotations

import pytest
import torch
from fedsim_crosscheck_kit import fresh_adapter, one_client

from ppsi.fedsim.client import ClientData, LocalSolver, PFConfig, client_update
from ppsi.fedsim.comm import VirtualChannel
from ppsi.fedsim.numerics import state_digest
from ppsi.fedsim.personal import (
    PersonalStore,
    PrivateKeyError,
    assert_unique_keys,
    validate_private_key,
)
from ppsi.fedsim.server import Server, run_round


# ---------------------------------------------------------------------------------------- C1: p=0 == no personal
def test_p_zero_forward_equals_no_personal_forward_bitwise():
    adapter = fresh_adapter(K=8, d=6, tied=True, seed=3)
    client = one_client("u", n=6, K=8, L=5, seed=1)
    batch = client.examples
    scores_no_p = adapter.scores(batch)
    scores_zero_p = adapter.scores(batch, torch.zeros(adapter.query_dim))
    assert torch.equal(scores_no_p, scores_zero_p), "'p_u=0 must recover shared-model predictions'"


# ---------------------------------------------------------------------------------------- C2: new user is p=0
def test_new_user_starts_at_p_zero():
    store = PersonalStore(query_dim=6)
    assert store.get("brand-new-user") is None, "a never-committed key has no state"
    assert torch.equal(store.p("brand-new-user"), torch.zeros(6)), "p_u 'initializes 0'"


# ---------------------------------------------------------------------------------------- C3: persistence across rounds
def test_p_persists_and_optimizer_step_count_accumulates_across_rounds():
    adapter = fresh_adapter(K=8, d=6, tied=True, seed=5)
    solver = LocalSolver(lr=1e-3, passes=2, batch_size=16)          # n=6 < 16 -> 1 step/pass -> 2 steps/round
    pf = PFConfig(lr=0.1, lam=1e-4, clip=1.0)
    client = one_client("persisting-user", n=6, K=8, L=5, seed=2, pref=None)

    personal = None
    ps = []
    for r in range(3):
        theta_r = adapter.broadcast_state(clone=True)
        res = client_update(adapter, theta_r, client, solver, round_idx=r, seed=99, pf=pf, personal=personal)
        adapter.load_state_(theta_r)                     # undo the shared-side move: isolate the p trajectory
        personal = res.personal
        ps.append(personal.p.detach().clone())
        # "Personal optimizer state persists per client in PF" -- the AdamW step counter must
        # accumulate (2 steps/round), never reset, across rounds.
        step_tensor = personal.opt_state["step"]
        expected_step = 2 * (r + 1)
        assert float(step_tensor) == expected_step, f"round {r}: AdamW step count must accumulate"

    assert not torch.equal(ps[0], torch.zeros(6)), "p_u must move away from 0 once trained"
    assert not torch.equal(ps[0], ps[1]), "round 2 must continue from round 1's p, not reset it"
    assert not torch.equal(ps[1], ps[2]), "round 3 must continue from round 2's p, not reset it"


def test_forgetting_personal_state_gives_a_different_result_than_persisting_it():
    # Negative-control-style contrast: if the caller (wrongly) never threads `personal` through, every round
    # restarts from p=0 and the AdamW step counter never accumulates -- a different, and wrong, trajectory.
    adapter_persist = fresh_adapter(K=8, d=6, tied=True, seed=5)
    adapter_forget = fresh_adapter(K=8, d=6, tied=True, seed=5)
    solver = LocalSolver(lr=1e-3, passes=2, batch_size=16)
    pf = PFConfig(lr=0.1, lam=1e-4, clip=1.0)
    client = one_client("forget-vs-persist", n=6, K=8, L=5, seed=2)

    personal = None
    for r in range(3):
        theta_r = adapter_persist.broadcast_state(clone=True)
        res = client_update(adapter_persist, theta_r, client, solver, round_idx=r, seed=99, pf=pf, personal=personal)
        adapter_persist.load_state_(theta_r)
        personal = res.personal
    p_persisted_final = personal.p.detach().clone()

    p_forgot_final = None
    for r in range(3):
        theta_r = adapter_forget.broadcast_state(clone=True)
        res = client_update(adapter_forget, theta_r, client, solver, round_idx=r, seed=99, pf=pf, personal=None)
        adapter_forget.load_state_(theta_r)
        p_forgot_final = res.personal.p.detach().clone()

    assert not torch.equal(p_persisted_final, p_forgot_final), (
        "persisting p_u across rounds must give a different (and correct) result than "
        "re-initializing it to 0 every round")


# ---------------------------------------------------------------------------------------- C4: never in the payload
def test_p_never_appears_in_the_upload_or_channel():
    adapter = fresh_adapter(K=8, d=6, tied=True, seed=6)
    solver = LocalSolver(lr=1e-3, passes=1, batch_size=16)
    pf = PFConfig(lr=0.1, lam=1e-4)
    client = one_client("payload-check", n=6, K=8, L=5, seed=1)
    theta_r = adapter.broadcast_state(clone=True)
    res = client_update(adapter, theta_r, client, solver, round_idx=0, seed=1, pf=pf, personal=None)
    assert set(res.upload.keys()) == set(adapter.manifest.shared_keys), (
        "the upload key set is exactly the shared set -- no private p_u key")
    ch = VirtualChannel(adapter.manifest)
    with pytest.raises(ValueError):
        ch.upload(0, "payload-check", {**res.upload, "p": torch.zeros(6)})
    # comm.py: "an upload of {key} is not exactly the shared key set"


# ---------------------------------------------------------------------------------------- C5: OOV / reserved keys
@pytest.mark.parametrize("bad_key", ["", "0", "1", "2", "pad", "PAD", " Pad ", "missing", "MISSING", "oov", "OOV",
                                     "<pad>", "<oov>", "<missing>", "none", "null", "  "])
def test_reserved_and_oov_tokens_are_refused_as_private_keys(bad_key):
    with pytest.raises(PrivateKeyError):
        validate_private_key(bad_key)
    # "never by shared OOV token" / personal.py reserved-token contract


def test_non_string_keys_are_refused():
    with pytest.raises(PrivateKeyError):
        validate_private_key(2)
    with pytest.raises(PrivateKeyError):
        validate_private_key(None)


def test_ordinary_user_key_is_accepted():
    validate_private_key("user-00042")   # must not raise


# ---------------------------------------------------------------------------------------- C6: duplicate keys refused
def test_duplicate_client_keys_refused_at_key_validation():
    with pytest.raises(PrivateKeyError):
        assert_unique_keys(["a", "b", "a"])


def test_duplicate_client_keys_refused_before_any_round_work_happens():
    adapter = fresh_adapter(K=8, d=6, tied=True, seed=8)
    server = Server(adapter)
    solver = LocalSolver(lr=1e-3, passes=1, batch_size=16)
    dup = one_client("same-key", n=4, K=8, L=5, seed=1)
    dup2 = ClientData("same-key", one_client("x", n=4, K=8, L=5, seed=2).examples)
    with pytest.raises(PrivateKeyError):
        run_round(server, [adapter], [dup, dup2], solver, round_idx=0, seed=1)
    assert server.round == 0, "a rejected round must leave the server state untouched"


# ---------------------------------------------------------------------------------------- C7: worker reuse swaps no state
def test_worker_reuse_across_two_clients_swaps_no_state():
    solver = LocalSolver(lr=1e-3, passes=2, batch_size=16)
    pf = PFConfig(lr=0.1, lam=1e-4, clip=1.0)
    clientA = one_client("client-A", n=5, K=8, L=5, seed=1)
    clientB = one_client("client-B", n=9, K=8, L=5, seed=2)

    # Reference: each client alone, on its own fresh adapter, starting from the SAME theta_r.
    ref_theta_r = fresh_adapter(K=8, d=6, tied=True, seed=4).broadcast_state(clone=True)
    ref_a_adapter = fresh_adapter(K=8, d=6, tied=True, seed=4)
    ref_b_adapter = fresh_adapter(K=8, d=6, tied=True, seed=4)
    ref_a = client_update(ref_a_adapter, ref_theta_r, clientA, solver, round_idx=0, seed=55, pf=pf, personal=None)
    ref_b = client_update(ref_b_adapter, ref_theta_r, clientB, solver, round_idx=0, seed=55, pf=pf, personal=None)

    # Now run BOTH clients in one round through a single reused worker (n_shards=1 -> one worker, sequential).
    worker = fresh_adapter(K=8, d=6, tied=True, seed=4)
    server = Server(fresh_adapter(K=8, d=6, tied=True, seed=4))
    server.adapter.load_state_(ref_theta_r)                # server theta == ref_theta_r
    store = PersonalStore(query_dim=6)
    report = run_round(server, [worker], [clientA, clientB], solver, round_idx=0, seed=55, pf=pf, store=store,
                       n_shards=1)

    assert store.get("client-A").p is not None
    assert torch.equal(store.get("client-A").p, ref_a.personal.p), (
        "two clients sharing one worker must not swap personal state (client-A)")
    assert torch.equal(store.get("client-B").p, ref_b.personal.p), (
        "two clients sharing one worker must not swap personal state (client-B)")
    va = next(v for v in report.visits if v.key == "client-A")
    vb = next(v for v in report.visits if v.key == "client-B")
    assert va.n_consumed == ref_a.n_consumed and vb.n_consumed == ref_b.n_consumed


def test_worker_reuse_aggregate_matches_independently_computed_fedavg():
    # A stronger cross-check than digests alone: independently re-derive the FedAvg aggregate from each
    # client's SOLO upload (computed on a private, never-reused adapter) via `aggregate_uploads`, and compare
    # it, bit for bit, against what `run_round` produced by reusing ONE worker for both clients. If the worker
    # had swapped or bled state between clients, the two uploads would no longer match their solo counterparts
    # and this aggregate would differ.
    from ppsi.fedsim.aggregate import aggregate_uploads

    solver = LocalSolver(lr=1e-3, passes=1, batch_size=16)
    clientA = one_client("solo-A", n=5, K=8, L=5, seed=1)
    clientB = one_client("solo-B", n=6, K=8, L=5, seed=2)
    theta_r = fresh_adapter(K=8, d=6, tied=True, seed=4).broadcast_state(clone=True)

    ref_a = client_update(fresh_adapter(K=8, d=6, tied=True, seed=4), theta_r, clientA, solver, round_idx=0, seed=3)
    ref_b = client_update(fresh_adapter(K=8, d=6, tied=True, seed=4), theta_r, clientB, solver, round_idx=0, seed=3)
    manifest = fresh_adapter(K=8, d=6, tied=True, seed=4).manifest
    expected = aggregate_uploads(manifest, theta_r,
                                 [("solo-A", ref_a.upload, ref_a.n_consumed), ("solo-B", ref_b.upload, ref_b.n_consumed)],
                                 n_shards=1)
    expected_digest = state_digest(expected)

    worker = fresh_adapter(K=8, d=6, tied=True, seed=4)
    server = Server(fresh_adapter(K=8, d=6, tied=True, seed=4))
    server.adapter.load_state_(theta_r)
    report = run_round(server, [worker], [clientA, clientB], solver, round_idx=0, seed=3, n_shards=1)

    assert report.state_digest == expected_digest, (
        "reusing one worker for both clients must give the same aggregate as independently "
        "computed solo uploads -- any state swap would change this bit-for-bit result")
