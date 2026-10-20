"""FedAvg weighted aggregation: exact toy values, buffer rules, empty / zero-weight / malformed round rejection.

Proves: theta_{r+1} = sum n_u theta_u / sum n_u with n = n_consumed, exactly, for every shard count; integer maps and
server buffers are never averaged; aliases are not aggregated; empty rounds, zero-weight rounds, extra / missing
upload keys, order violations, dropped clients and non-finite aggregates are refused.
Does not prove: anything about optimization quality (see the pooled-SGD and FedProx tests).
"""
from __future__ import annotations

import functools
import operator
from collections import OrderedDict

import pytest
import torch
from fedsim_testkit import Toy, ToyAdapter, assert_raises, nc, one_client, server_and_worker, solver

from ppsi.fedsim.adapter import BufferRule, ManifestError, Role, build_manifest
from ppsi.fedsim.aggregate import (
    AggregationError,
    EmptyRoundError,
    ShardAccumulator,
    aggregate_uploads,
    combine_shards,
    finalize,
    plan_shards,
)
from ppsi.fedsim.server import run_round


def _toy():
    a = ToyAdapter(Toy())
    with torch.no_grad():
        a.module.w.copy_(torch.tensor([10.0, 20.0]))
        a.module.emb.weight.copy_(torch.arange(8, dtype=torch.float32).view(4, 2))
    return a


def _uploads(a, ws, ns):
    base = a.extract_shared(clone=True)
    out = []
    for i, (w, n) in enumerate(zip(ws, ns)):
        up = OrderedDict((k, v.clone()) for k, v in base.items())
        up["w"] = torch.tensor(w, dtype=torch.float32)
        up["emb.weight"] = torch.full((4, 2), float(i + 1))
        out.append((f"u{i}", up, n))
    return out


# ------------------------------------------------------------------------------------------------ checks
def check_exact_weighted(agg_fn):
    a = _toy()
    server_state = a.broadcast_state(clone=True)
    ups = _uploads(a, [[1, 2], [3, 4], [5, 6]], [1, 2, 1])
    new = agg_fn(a.manifest, server_state, ups)
    assert torch.equal(new["w"], torch.tensor([3.0, 4.0])), new["w"]                  # (1+6+5)/4, (2+8+6)/4
    assert torch.equal(new["emb.weight"], torch.full((4, 2), (1 * 1 + 2 * 2 + 1 * 3) / 4))
    ups = _uploads(a, [[1, 2], [3, 4], [100, 100]], [3, 1, 0])                        # a counted no-op client
    new = agg_fn(a.manifest, server_state, ups)
    assert torch.equal(new["w"], torch.tensor([1.5, 2.5])), new["w"]                  # (3+3)/4, (6+4)/4


def check_buffers_follow_rules(agg_fn):
    a = _toy()
    server_state = a.broadcast_state(clone=True)
    new = agg_fn(a.manifest, server_state, _uploads(a, [[1, 2], [3, 4]], [1, 3]))
    assert new["imap"].dtype == torch.long and torch.equal(new["imap"], torch.tensor([3, 4, 5]))
    assert new["scale"].dtype == torch.float32 and float(new["scale"]) == 2.0
    assert "head.weight" not in new                                                    # alias follows emb.weight
    assert set(new) == set(a.manifest.shared_keys) | set(a.manifest.buffer_keys)


def check_rejects_empty_and_zero(agg_fn):
    a = _toy()
    server_state = a.broadcast_state(clone=True)
    assert_raises(EmptyRoundError, agg_fn, a.manifest, server_state, [])
    assert_raises(EmptyRoundError, agg_fn, a.manifest, server_state, _uploads(a, [[1, 2], [3, 4]], [0, 0]))


# ------------------------------------------------------------------------------------------------ wrong variants
def unweighted_mean(manifest, server_state, uploads):
    return aggregate_uploads(manifest, server_state, [(k, u, 1) for k, u, _ in uploads])


def naive_state_average(manifest, server_state, uploads):
    """Averages every state entry a client could send, int maps included (cast to float)."""
    if not uploads:
        return OrderedDict(server_state)
    out = OrderedDict()
    W = sum(n for _, _, n in uploads) or 1
    for k in list(server_state) + ["head.weight"]:
        vals = [(u.get(k, server_state.get(k, u.get("emb.weight"))).to(torch.float32), n) for _, u, n in uploads]
        out[k] = sum(v * n for v, n in vals) / W
    return out


def silent_accept(manifest, server_state, uploads):
    if not uploads or sum(n for _, _, n in uploads) == 0:
        return OrderedDict(server_state)
    return aggregate_uploads(manifest, server_state, uploads)


# ------------------------------------------------------------------------------------------------ tests
@pytest.mark.parametrize("n_shards", [1, 2, 3, 5])
def test_exact_weighted_values(n_shards):
    check_exact_weighted(lambda m, s, u: aggregate_uploads(m, s, u, n_shards=n_shards))


def test_buffers_follow_rules():
    check_buffers_follow_rules(aggregate_uploads)


def test_rejects_empty_and_zero_weight():
    check_rejects_empty_and_zero(aggregate_uploads)


def test_round_driver_rejects_empty_and_zero_weight():
    srv, workers = server_and_worker()
    before = srv.digest()
    with pytest.raises(EmptyRoundError):
        run_round(srv, workers, [], solver(), round_idx=0, seed=1)
    zero = [one_client("user-a", 5, invalid_frac=1.0), one_client("user-b", 3, invalid_frac=1.0)]
    with pytest.raises(EmptyRoundError):
        run_round(srv, workers, zero, solver(), round_idx=0, seed=1)
    assert srv.digest() == before and srv.round == 0


def test_malformed_uploads_refused():
    a = _toy()
    s = a.broadcast_state(clone=True)
    ups = _uploads(a, [[1, 2]], [1])
    leak = OrderedDict(ups[0][1]); leak["pf.p_u"] = torch.zeros(2)
    with pytest.raises(AggregationError):
        aggregate_uploads(a.manifest, s, [("u0", leak, 1)])
    missing = OrderedDict(ups[0][1]); missing.pop("w")
    with pytest.raises(AggregationError):
        aggregate_uploads(a.manifest, s, [("u0", missing, 1)])
    nan = OrderedDict(ups[0][1]); nan["w"] = torch.tensor([float("nan"), 0.0])
    with pytest.raises(AggregationError):
        aggregate_uploads(a.manifest, s, [("u0", nan, 1)])
    acc = ShardAccumulator.empty(0, a.manifest, s)
    acc.add(3, "u3", ups[0][1], 1)
    with pytest.raises(AggregationError):
        acc.add(2, "u2", ups[0][1], 1)                                                 # logical order violated
    with pytest.raises(AggregationError):                                              # a planned client dropped
        finalize(combine_shards([acc]), a.manifest, s, expected_keys=["u3", "u4"])
    with pytest.raises(AggregationError):                                              # a shard missing
        combine_shards([ShardAccumulator.empty(1, a.manifest, s)])


def test_manifest_roles_and_refusals():
    a = _toy()
    e = a.manifest.entries
    assert e["w"].role == Role.SHARED and e["emb.weight"].role == Role.SHARED
    assert e["head.weight"].role == Role.ALIAS and e["head.weight"].alias_of == "emb.weight"
    assert e["imap"].rule == BufferRule.FIXED and e["scale"].rule == BufferRule.SERVER_COPY
    assert a.manifest.nonpersistent_buffers == ("tmp",)
    with pytest.raises(ManifestError):                          # float buffer without an explicit rule
        build_manifest(Toy())
    with pytest.raises(ManifestError):                          # integer maps can only be FIXED
        build_manifest(Toy(), buffer_rules={"scale": BufferRule.SERVER_COPY, "imap": BufferRule.SERVER_COPY})


def test_plan_shards_contiguous():
    assert plan_shards(5, 2) == [[0, 1, 2], [3, 4]]
    assert plan_shards(2, 3) == [[0], [1], []]
    assert functools.reduce(operator.iadd, plan_shards(64, 2), []) == list(range(64))


@nc("unweighted mean instead of n_consumed weights")
def test_nc_unweighted_mean():
    check_exact_weighted(unweighted_mean)


@nc("averaging every state entry (integer maps included)")
def test_nc_int_maps_averaged():
    check_buffers_follow_rules(naive_state_average)


@nc("empty / zero-weight round silently accepted")
def test_nc_silent_empty_round():
    check_rejects_empty_and_zero(silent_accept)
