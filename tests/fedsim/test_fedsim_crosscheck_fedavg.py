"""FedAvg weighted aggregation: hand-computed toy exact values; empty/zero-weight refusal.

All expected numbers below are computed BY HAND from the formula theta_(r+1) = sum_u n_u * theta_(r,u) / sum_u n_u,
using a separate toy client set (unequal n, a tail batch, and a no-op client), independent of the main tests.
"""
from __future__ import annotations

import pytest
import torch
from fedsim_crosscheck_kit import flat_manifest, fresh_adapter, one_client

from ppsi.fedsim.adapter import BufferRule
from ppsi.fedsim.aggregate import (
    AggregationError,
    EmptyRoundError,
    ShardAccumulator,
    aggregate_uploads,
    combine_shards,
    finalize,
    plan_shards,
)
from ppsi.fedsim.client import LocalSolver, client_update


# ---------------------------------------------------------------------------------------- toy 1
# 4 clients, single shared vector "w" of dim 2. n = [2, 5, 1, 8] (unequal, includes a singleton).
# Hand computation:
#   dim0: 2*1 + 5*3 + 1*10 + 8*0  = 2 + 15 + 10 + 0  = 27 ; /16 = 1.6875
#   dim1: 2*2 + 5*(-1) + 1*10 + 8*4 = 4 - 5 + 10 + 32 = 41 ; /16 = 2.5625
def _toy_uploads():
    vals = [(2, [1.0, 2.0]), (5, [3.0, -1.0]), (1, [10.0, 10.0]), (8, [0.0, 4.0])]
    return [(f"c{i}", {"w": torch.tensor(v, dtype=torch.float32)}, n) for i, (n, v) in enumerate(vals)]


EXPECTED_W = torch.tensor([1.6875, 2.5625], dtype=torch.float32)


def _toy_manifest_and_server_state():
    manifest = flat_manifest({"w": (2,)},
                             {"class_map": ((3,), "torch.int64", BufferRule.FIXED),
                              "scale": ((1,), "torch.float32", BufferRule.SERVER_COPY)})
    server_state = {"w": torch.zeros(2), "class_map": torch.tensor([5, 6, 7], dtype=torch.int64),
                    "scale": torch.tensor([9.0])}
    return manifest, server_state


@pytest.mark.parametrize("n_shards", [1, 2, 3, 4])
def test_fedavg_weighted_average_hand_toy(n_shards):
    manifest, server_state = _toy_manifest_and_server_state()
    uploads = _toy_uploads()
    new_state = aggregate_uploads(manifest, server_state, uploads, n_shards=n_shards)
    # theta_(r+1) = sum n_u theta_u / sum n_u -- exact for these clean values, any n_shards.
    assert torch.equal(new_state["w"], EXPECTED_W), "hand-computed weighted mean"
    # "integer maps and class orders are not averaged" / FIXED and SERVER_COPY keep the server's value.
    assert torch.equal(new_state["class_map"], server_state["class_map"]), "buffer passthrough (FIXED)"
    assert torch.equal(new_state["scale"], server_state["scale"]), "buffer passthrough (SERVER_COPY)"


def test_fedavg_bitexact_across_shard_counts():
    manifest, server_state = _toy_manifest_and_server_state()
    uploads = _toy_uploads()
    results = [aggregate_uploads(manifest, server_state, uploads, n_shards=s)["w"] for s in (1, 2, 3, 4)]
    for r in results[1:]:
        # aggregate.py: "Any assignment of shards to 1, 2 or more workers ... gives the same bits."
        assert torch.equal(results[0], r), "aggregate.py contract: shard count must not change the bits"


def test_fedavg_weight_and_addition_order_recorded():
    manifest, server_state = _toy_manifest_and_server_state()
    uploads = _toy_uploads()
    plan = plan_shards(len(uploads), 2)
    shards = []
    for s, positions in enumerate(plan):
        acc = ShardAccumulator.empty(s, manifest, server_state)
        for pos in positions:
            key, up, n = uploads[pos]
            acc.add(pos, key, up, n)
        shards.append(acc)
    agg = combine_shards(shards)
    assert agg.weight == 16, "sum_u n_consumed,u must equal the hand-summed total"
    assert [a[0] for a in agg.added] == [0, 1, 2, 3], "logical order preserved across shards"
    assert agg.n_noop == 0


# ---------------------------------------------------------------------------------------- toy 2: tail batch
# A real client visit through client_update: n_valid=17, batch_size=16 -> ceil(17/16)=2 batches/pass
# (16 then a tail batch of 1), passes=2 -> steps=4, n_consumed = passes * n_valid = 34.
def test_tail_batch_step_and_weight_accounting():
    adapter = fresh_adapter(K=10, d=4, tied=True, seed=7)
    theta_r = adapter.broadcast_state(clone=True)
    client = one_client("tail-client", n=17, K=10, L=5, seed=1)
    solver = LocalSolver(lr=1e-3, passes=2, batch_size=16)
    res = client_update(adapter, theta_r, client, solver, round_idx=0, seed=123)
    assert res.n_valid == 17
    assert res.steps_per_pass == 2, "ceil(17/16) = 2 batches per pass (tail-batch rule)"
    assert res.steps == 4, "passes(2) * steps_per_pass(2) = 4"
    assert res.n_consumed == 34, "n_consumed = passes * n_valid = 2*17 = 34, not padded to batch size"


# ---------------------------------------------------------------------------------------- refusals
def test_refuses_empty_round_no_clients():
    manifest, server_state = _toy_manifest_and_server_state()
    with pytest.raises(EmptyRoundError):
        aggregate_uploads(manifest, server_state, uploads=[], n_shards=2)
    # "empty rounds ... are refused"


def test_refuses_zero_weight_round_all_noop():
    manifest, server_state = _toy_manifest_and_server_state()
    uploads = [("noop-a", {"w": torch.zeros(2)}, 0), ("noop-b", {"w": torch.ones(2)}, 0)]
    with pytest.raises(EmptyRoundError):
        aggregate_uploads(manifest, server_state, uploads, n_shards=1)
    # "a round with no clients or zero total weight is refused"


def test_noop_client_counted_but_no_weight():
    manifest, server_state = _toy_manifest_and_server_state()
    uploads = _toy_uploads() + [("noop", {"w": torch.tensor([999.0, -999.0])}, 0)]
    new_state = aggregate_uploads(manifest, server_state, uploads, n_shards=1)
    # the no-op client's huge, arbitrary values must NOT move the average ("carries no weight")
    assert torch.equal(new_state["w"], EXPECTED_W), "a no-op client (n=0) is counted but carries no weight"


def test_refuses_malformed_upload_key_set():
    manifest, server_state = _toy_manifest_and_server_state()
    acc = ShardAccumulator.empty(0, manifest, server_state)
    with pytest.raises(AggregationError):
        acc.add(0, "bad", {"w": torch.zeros(2), "extra": torch.zeros(1)}, 3)
    # aggregate.py: "an upload whose key set is not exactly the shared set is refused"


def test_refuses_out_of_order_positions_within_a_shard():
    manifest, server_state = _toy_manifest_and_server_state()
    acc = ShardAccumulator.empty(0, manifest, server_state)
    acc.add(3, "a", {"w": torch.ones(2)}, 1)
    with pytest.raises(AggregationError):
        acc.add(2, "b", {"w": torch.ones(2)}, 1)          # position must strictly increase
    with pytest.raises(AggregationError):
        acc.add(3, "c", {"w": torch.ones(2)}, 1)          # a repeated position is also an order violation


def test_refuses_shard_index_gap():
    manifest, server_state = _toy_manifest_and_server_state()
    acc0 = ShardAccumulator.empty(0, manifest, server_state)
    acc2 = ShardAccumulator.empty(2, manifest, server_state)   # index 1 missing
    acc0.add(0, "a", {"w": torch.ones(2)}, 1)
    acc2.add(1, "b", {"w": torch.ones(2)}, 1)
    with pytest.raises(AggregationError):
        combine_shards([acc0, acc2])
    # aggregate.py combine_shards: "shard set must be exactly 0..N-1"


def test_refuses_non_finite_aggregate():
    manifest, server_state = _toy_manifest_and_server_state()
    uploads = [("a", {"w": torch.tensor([float("nan"), 1.0])}, 1)]
    with pytest.raises(AggregationError):
        aggregate_uploads(manifest, server_state, uploads, n_shards=1)
    # aggregate.py finalize: "a non-finite aggregate is refused"


def test_finalize_rejects_planned_client_mismatch():
    manifest, server_state = _toy_manifest_and_server_state()
    uploads = _toy_uploads()
    plan = plan_shards(len(uploads), 1)
    acc = ShardAccumulator.empty(0, manifest, server_state)
    for pos in plan[0]:
        key, up, n = uploads[pos]
        acc.add(pos, key, up, n)
    agg = combine_shards([acc])
    with pytest.raises(AggregationError):
        finalize(agg, manifest, server_state, expected_keys=["not", "the", "planned", "clients"])
