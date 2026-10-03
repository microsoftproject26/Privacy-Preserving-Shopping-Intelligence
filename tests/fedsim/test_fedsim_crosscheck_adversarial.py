"""Adversarial client-size edge cases: a 1-example client, a client whose example count is exactly 16 (the batch
size), a 0-example client, and duplicate client keys in a round.

Hand-computed from the aggregation rule (n_consumed = passes * n_valid; tails and no-op clients counted explicitly)
and the client.py contract, independently of the main simulator tests.
"""
from __future__ import annotations

import pytest
from fedsim_crosscheck_kit import fresh_adapter, one_client

from ppsi.fedsim.client import LocalSolver, PFConfig, client_update
from ppsi.fedsim.numerics import states_equal
from ppsi.fedsim.personal import PrivateKeyError
from ppsi.fedsim.server import Server, run_round


def _run(n, passes=2, batch_size=16, invalid_frac=0.0, seed=1):
    adapter = fresh_adapter(K=8, d=4, tied=True, seed=0)
    theta_r = adapter.broadcast_state(clone=True)
    client = one_client(f"n{n}", n=n, K=8, L=5, seed=seed, invalid_frac=invalid_frac)
    solver = LocalSolver(lr=1e-3, passes=passes, batch_size=batch_size)
    res = client_update(adapter, theta_r, client, solver, round_idx=0, seed=3)
    return adapter, theta_r, res


def test_client_with_exactly_one_example():
    _, _, res = _run(n=1, passes=2, batch_size=16)
    assert res.n_valid == 1
    assert res.steps_per_pass == 1, "ceil(1/16) = 1 (single, undersized batch)"
    assert res.steps == 2, "passes(2) * 1 = 2"
    assert res.n_consumed == 2, "n_consumed = passes * n_valid = 2*1 = 2"


def test_client_with_exactly_the_batch_size_sixteen_examples():
    # A boundary case: n == batch_size exactly divides -> NO tail batch at all.
    _, _, res = _run(n=16, passes=2, batch_size=16)
    assert res.n_valid == 16
    assert res.steps_per_pass == 1, "ceil(16/16) = 1: exactly one full batch, no tail batch (boundary)"
    assert res.steps == 2
    assert res.n_consumed == 32, "n_consumed = passes * n_valid = 2*16 = 32"


def test_client_with_zero_valid_examples_is_a_recorded_noop():
    adapter, theta_r, res = _run(n=8, passes=2, batch_size=16, invalid_frac=1.0)   # every row invalid
    assert res.n_valid == 0
    assert res.n_consumed == 0
    assert res.steps == 0
    assert res.steps_per_pass == 0
    # a no-op client's upload must be theta_r itself, bit for bit (no update happened)
    assert states_equal(res.upload, {k: theta_r[k] for k in adapter.manifest.shared_keys})


def test_zero_example_client_is_still_counted_in_a_round_but_carries_no_weight():
    solver = LocalSolver(lr=1e-3, passes=2, batch_size=16)
    noop_client = one_client("noop", n=6, K=8, L=5, seed=1, invalid_frac=1.0)
    real_client = one_client("real", n=6, K=8, L=5, seed=2)
    server = Server(fresh_adapter(K=8, d=4, tied=True, seed=0))
    report = run_round(server, [fresh_adapter(K=8, d=4, tied=True, seed=0)], [noop_client, real_client], solver,
                       round_idx=0, seed=3, n_shards=1)
    assert report.n_noop == 1, "the no-op client is counted"
    assert report.weight == real_client.examples["target_class"].shape[0] * 2, (
        "a no-op client (n=0) carries no weight -- only the real client's n_consumed counts")


def test_duplicate_client_keys_in_a_round_refused_with_pf_too():
    adapter = fresh_adapter(K=8, d=4, tied=True, seed=0)
    server = Server(adapter)
    solver = LocalSolver(lr=1e-3, passes=1, batch_size=16)
    pf = PFConfig(lr=0.1, lam=1e-4)
    from ppsi.fedsim.client import ClientData
    a = one_client("dup", n=4, K=8, L=5, seed=1)
    b = ClientData("dup", one_client("x", n=4, K=8, L=5, seed=2).examples)
    from ppsi.fedsim.personal import PersonalStore
    store = PersonalStore(query_dim=adapter.query_dim)
    with pytest.raises(PrivateKeyError):
        run_round(server, [fresh_adapter(K=8, d=4, tied=True, seed=0)], [a, b], solver, round_idx=0, seed=1, pf=pf,
                 store=store)
    assert len(store) == 0
