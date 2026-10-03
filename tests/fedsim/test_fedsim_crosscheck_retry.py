"""The retry path (identical RNG, no double consumption) and deterministic aggregation order.

A failed client is retried with identical initialization / RNG or the whole round aborts; it is never silently
dropped for speed; combine_shards sums in shard-index order. Derived from those rules and the client.py / server.py /
aggregate.py contracts.

The real two-process exercise is in test_fedsim_pool.py; this file proves the SAME order-invariance property
analytically at the aggregate.py level (arrival order reversed).
"""
from __future__ import annotations

import pytest
import torch
from fedsim_crosscheck_kit import fresh_adapter, one_client

from ppsi.fedsim.adapter import Entry, ParamManifest, Role
from ppsi.fedsim.aggregate import AggregationError, ShardAccumulator, combine_shards
from ppsi.fedsim.client import FaultPlan, LocalSolver
from ppsi.fedsim.numerics import state_digest
from ppsi.fedsim.personal import PersonalStore
from ppsi.fedsim.server import RoundAborted, Server, run_round, visit_with_retry


# ---------------------------------------------------------------------------------------- retry: identical RNG
def test_retry_after_injected_failure_is_bitwise_identical_to_a_clean_run():
    adapter_clean = fresh_adapter(K=8, d=5, tied=True, seed=2)
    adapter_retry = fresh_adapter(K=8, d=5, tied=True, seed=2)
    theta_r = fresh_adapter(K=8, d=5, tied=True, seed=2).broadcast_state(clone=True)
    client = one_client("flaky", n=20, K=8, L=5, seed=1)     # bs=16 -> 2 steps/pass, passes=2 -> 4 steps
    solver = LocalSolver(lr=1e-3, passes=2, batch_size=16)

    baseline = visit_with_retry(adapter_clean, theta_r, client, solver, round_idx=0, seed=7, mu=0.0, pf=None,
                                get_personal=lambda k: None, fault=None, max_attempts=2)

    # attempt 0 fails right after its FIRST optimizer step (step==1); attempt 1 has no matching fault.
    fault = FaultPlan(faults=frozenset({("flaky", 0, 1)}))
    retried = visit_with_retry(adapter_retry, theta_r, client, solver, round_idx=0, seed=7, mu=0.0, pf=None,
                               get_personal=lambda k: None, fault=fault, max_attempts=2)

    assert retried.attempts == 2, "the round retries a failed client, does not drop it"
    assert retried.wasted_steps == 1, "the failed attempt's single completed step is reported, never aggregated"
    assert state_digest(retried.upload) == state_digest(baseline.upload), (
        "'identical initialization/RNG' -- attempt 2 must reproduce the SAME trajectory bitwise")
    assert retried.n_consumed == baseline.n_consumed, "no double consumption: n_consumed unaffected by the retry"


def test_retry_exhausted_aborts_the_round():
    adapter = fresh_adapter(K=8, d=5, tied=True, seed=2)
    theta_r = adapter.broadcast_state(clone=True)
    client = one_client("always-fails", n=20, K=8, L=5, seed=1)
    solver = LocalSolver(lr=1e-3, passes=2, batch_size=16)
    fault = FaultPlan(faults=frozenset({("always-fails", 0, 1), ("always-fails", 1, 1)}))
    with pytest.raises(RoundAborted):
        visit_with_retry(adapter, theta_r, client, solver, round_idx=0, seed=7, mu=0.0, pf=None,
                         get_personal=lambda k: None, fault=fault, max_attempts=2)


def test_round_level_retry_gives_the_same_final_state_and_weight_as_a_clean_round():
    # No double consumption at the ROUND level: report.weight and the final aggregate must be unaffected by
    # a mid-round retry (the failed attempt's partial work is never added to the accumulator).
    clientA = one_client("A", n=20, K=8, L=5, seed=1)
    clientB = one_client("B", n=9, K=8, L=5, seed=2)
    solver = LocalSolver(lr=1e-3, passes=2, batch_size=16)

    server_clean = Server(fresh_adapter(K=8, d=5, tied=True, seed=9))
    report_clean = run_round(server_clean, [fresh_adapter(K=8, d=5, tied=True, seed=9)], [clientA, clientB],
                             solver, round_idx=0, seed=11, n_shards=1)

    server_retry = Server(fresh_adapter(K=8, d=5, tied=True, seed=9))
    fault = FaultPlan(faults=frozenset({("A", 0, 1)}))
    report_retry = run_round(server_retry, [fresh_adapter(K=8, d=5, tied=True, seed=9)], [clientA, clientB],
                             solver, round_idx=0, seed=11, n_shards=1, fault=fault, max_attempts=2)

    assert report_retry.weight == report_clean.weight, "retry must not double-count n_consumed"
    assert report_retry.state_digest == report_clean.state_digest, (
        "a retried round must reach the identical final state as the clean round")


def test_aborted_round_leaves_server_and_pf_store_unchanged():
    client = one_client("dies-both-attempts", n=20, K=8, L=5, seed=1)
    solver = LocalSolver(lr=1e-3, passes=2, batch_size=16)
    from ppsi.fedsim.client import PFConfig
    pf = PFConfig(lr=0.1, lam=1e-4)
    adapter = fresh_adapter(K=8, d=5, tied=True, seed=9)
    server = Server(adapter)
    store = PersonalStore(query_dim=adapter.query_dim)
    fault = FaultPlan(faults=frozenset({("dies-both-attempts", 0, 1), ("dies-both-attempts", 1, 1)}))
    with pytest.raises(RoundAborted):
        run_round(server, [fresh_adapter(K=8, d=5, tied=True, seed=9)], [client], solver, round_idx=0, seed=3,
                 pf=pf, store=store, fault=fault, max_attempts=2)
    assert server.round == 0, "an aborted round leaves the server state unchanged"
    assert len(store) == 0, "an aborted round must never commit any p_u (server.py: 'every p_u are unchanged')"


# ---------------------------------------------------------------------------------------- aggregation order
def _shard_toy(shard_index, position, key, weight, value):
    manifest = ParamManifest({"w": Entry("w", Role.SHARED, "torch.float32", (2,), 2, 4)})
    acc = ShardAccumulator.empty(shard_index, manifest, {"w": torch.zeros(2)})
    acc.add(position, key, {"w": torch.tensor(value, dtype=torch.float32)}, weight)
    return acc


def test_combine_shards_is_invariant_to_arrival_order():
    # shard0: 1 client at global position 0, n=3, w=[6.0, 0.0];
    # shard1: 1 client at global position 1, n=5, w=[2.0, 8.0] (positions as `plan_shards(2, 2)` would assign).
    # hand sum: total = 3*[6,0] + 5*[2,8] = [18,0]+[10,40] = [28,40]; weight = 8
    shard0 = _shard_toy(0, 0, "c0", 3, [6.0, 0.0])
    shard1 = _shard_toy(1, 1, "c1", 5, [2.0, 8.0])

    forward = combine_shards([shard0, shard1])
    reversed_arrival = combine_shards([shard1, shard0])

    assert torch.equal(forward.total["w"], torch.tensor([28.0, 40.0])), "hand-computed shard sum"
    assert torch.equal(forward.total["w"], reversed_arrival.total["w"]), (
        "aggregate.py: combine_shards 'add[s] shard sums in shard-index order (never arrival order)'")
    assert forward.weight == reversed_arrival.weight == 8
    assert forward.added == [(0, "c0", 3), (1, "c1", 5)]
    assert reversed_arrival.added == [(0, "c0", 3), (1, "c1", 5)], "logical order recorded regardless of arrival"


def test_combine_shards_invariant_with_four_shards_reversed_and_shuffled():
    shards = [_shard_toy(i, i, f"c{i}", n, [float(n), float(-n)]) for i, n in enumerate([1, 2, 3, 4])]
    # hand sum: sum n*[n,-n] for n in 1..4 = [1,-1]+[4,-4]+[9,-9]+[16,-16] = [30,-30]; weight=1+2+3+4=10
    expected = torch.tensor([30.0, -30.0])
    import random
    order_a = list(shards)
    order_b = list(reversed(shards))
    rnd = random.Random(0)
    order_c = list(shards)
    rnd.shuffle(order_c)
    results = [combine_shards(o).total["w"] for o in (order_a, order_b, order_c)]
    for r in results:
        assert torch.equal(r, expected), "hand-computed 4-shard sum, order-invariant"


def test_shard_accumulator_refuses_duplicate_or_repeated_client_within_one_shard():
    manifest = ParamManifest({"w": Entry("w", Role.SHARED, "torch.float32", (1,), 1, 4)})
    acc = ShardAccumulator.empty(0, manifest, {"w": torch.zeros(1)})
    acc.add(0, "a", {"w": torch.ones(1)}, 1)
    with pytest.raises(AggregationError):
        acc.add(0, "a", {"w": torch.ones(1)}, 1)   # same position added twice: order violation
