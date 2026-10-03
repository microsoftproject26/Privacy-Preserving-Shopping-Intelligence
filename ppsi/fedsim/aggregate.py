"""Deterministic, streaming, n_consumed-weighted aggregation of the shared parameters.

    theta_{r+1}[k] = ( sum_u n_consumed,u * theta_{r,u}[k] ) / ( sum_u n_consumed,u )     for every SHARED key k

Determinism across workers. FP32 addition is not associative, so the reduction order is part of the protocol:
  * the round's clients are in a fixed LOGICAL order (the seeded participation plan);
  * `plan_shards` cuts that order into `n_shards` contiguous blocks (a protocol constant, independent of how many
    physical workers exist);
  * each shard is accumulated sequentially in logical order into its own FP32 `ShardAccumulator` (streaming: one
    running sum per shard, client states are never stored);
  * `combine_shards` adds the shard sums in shard-index order, whatever order they arrived in.
Any assignment of shards to 1, 2 or more workers, and any completion order, therefore gives the same bits.

Rules: aliases are not accumulated (they follow the canonical key); FIXED buffers (integer maps, class orders) and
SERVER_COPY buffers keep the server's value and are never averaged; a client with n_consumed = 0 is a counted no-op;
an upload whose key set is not exactly the shared set is refused (this is also the server-side guard that no private
state is uploaded); a round with no clients or zero total weight is refused; a non-finite aggregate is refused;
every planned client must be added exactly once (retries are never double-counted).
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor

from .adapter import BufferRule, ParamManifest


class EmptyRoundError(ValueError):
    pass


class AggregationError(RuntimeError):
    pass


def plan_shards(n_clients: int, n_shards: int) -> list:
    """Contiguous blocks of logical positions [0, n_clients): shard s gets positions [start_s, end_s)."""
    if n_shards < 1:
        raise ValueError("n_shards must be >= 1")
    base, extra = divmod(n_clients, n_shards)
    out, start = [], 0
    for s in range(n_shards):
        size = base + (1 if s < extra else 0)
        out.append(list(range(start, start + size)))
        start += size
    return out


@dataclass
class ShardAccumulator:
    shard_index: int
    keys: tuple
    acc: OrderedDict[str, Tensor]
    weight: int = 0
    added: list = field(default_factory=list)      # (position, client key, n_consumed) in addition order
    n_noop: int = 0

    @classmethod
    def empty(cls, shard_index: int, manifest: ParamManifest, template: Mapping[str, Tensor]) -> ShardAccumulator:
        keys = manifest.shared_keys
        acc = OrderedDict((k, torch.zeros(template[k].shape, dtype=torch.float32, device=template[k].device))
                          for k in keys)
        return cls(shard_index, keys, acc)

    def add(self, position: int, key: str, upload: Mapping[str, Tensor], n_consumed: int) -> None:
        if self.added and position <= self.added[-1][0]:
            raise AggregationError(f"logical order violated in shard {self.shard_index}: {position} after "
                                   f"{self.added[-1][0]}")
        n = int(n_consumed)
        if n < 0 or n != n_consumed:
            raise AggregationError(f"invalid n_consumed {n_consumed!r} for client {key}")
        if set(upload.keys()) != set(self.keys):
            extra = sorted(set(upload) - set(self.keys))
            missing = sorted(set(self.keys) - set(upload))
            raise AggregationError(f"upload of {key} is not exactly the shared set (extra={extra}, missing={missing})")
        self.added.append((int(position), key, n))
        if n == 0:
            self.n_noop += 1
            return
        for k in self.keys:
            t = upload[k]
            if t.dtype != torch.float32:
                raise AggregationError(f"upload {k} of {key} is {t.dtype}, expected float32")
            self.acc[k].add_(t, alpha=float(n))
        self.weight += n


@dataclass
class RoundAggregate:
    total: OrderedDict[str, Tensor]
    weight: int
    added: list
    n_noop: int


def combine_shards(shards: Sequence[ShardAccumulator]) -> RoundAggregate:
    """Add shard sums in shard-index order (never arrival order)."""
    if not shards:
        raise EmptyRoundError("no shards")
    ordered = sorted(shards, key=lambda s: s.shard_index)
    idx = [s.shard_index for s in ordered]
    if idx != list(range(len(ordered))):
        raise AggregationError(f"shard set must be exactly 0..{len(ordered) - 1}, got {idx}")
    total = OrderedDict((k, v.clone()) for k, v in ordered[0].acc.items())
    for s in ordered[1:]:
        for k in total:
            total[k].add_(s.acc[k])
    added = [a for s in ordered for a in s.added]
    return RoundAggregate(total, sum(s.weight for s in ordered), added, sum(s.n_noop for s in ordered))


def finalize(agg: RoundAggregate, manifest: ParamManifest, server_state: Mapping[str, Tensor], *,
             expected_keys: Sequence[str] | None = None) -> OrderedDict[str, Tensor]:
    """theta_{r+1}: shared = total / W; FIXED and SERVER_COPY buffers = the server's own values."""
    if not agg.added:
        raise EmptyRoundError("empty round: no client was aggregated")
    if agg.weight <= 0:
        raise EmptyRoundError(f"zero total weight over {len(agg.added)} client(s): round rejected")
    if expected_keys is not None:
        got = [a[1] for a in agg.added]
        if got != list(expected_keys):
            raise AggregationError(f"aggregated clients {got} != planned clients {list(expected_keys)}")
    new = OrderedDict()
    W = float(agg.weight)
    for k in manifest.shared_keys:
        v = torch.div(agg.total[k], W)
        if not bool(torch.isfinite(v).all()):
            raise AggregationError(f"non-finite aggregate for {k}")
        new[k] = v
    for k in manifest.buffer_keys:
        rule = manifest.entries[k].rule
        if rule not in (BufferRule.FIXED, BufferRule.SERVER_COPY):
            raise AggregationError(f"buffer {k} has unsupported rule {rule}")
        new[k] = server_state[k]
    return new


def aggregate_uploads(manifest: ParamManifest, server_state: Mapping[str, Tensor], uploads: Sequence,
                      n_shards: int = 1) -> OrderedDict[str, Tensor]:
    """Convenience: aggregate an in-memory list of (key, upload, n_consumed) in logical order."""
    plan = plan_shards(len(uploads), n_shards)
    shards = []
    for s, positions in enumerate(plan):
        acc = ShardAccumulator.empty(s, manifest, server_state)
        for pos in positions:
            key, up, n = uploads[pos]
            acc.add(pos, key, up, n)
        shards.append(acc)
    return finalize(combine_shards(shards), manifest, server_state, expected_keys=[u[0] for u in uploads])
