"""DP-FedAvg aggregation: flat L2 clipping of client deltas, Gaussian noise on the SUM, fixed denominator
(McMahan, Ramage, Talwar & Zhang 2018, ICLR, "Learning Differentially Private Recurrent Language Models"; the privacy
accountant is in accountant.py).

One round r with the sampled cohort C_r (|C_r| = m, the FIXED denominator), survivors S_r ⊆ C_r (drop-out), and the
broadcast theta_r:
    delta_u    = theta_{r,u} - theta_r                      over ALL shared tensors (aliases once), FP32
    norm_u     = || concat_k delta_u[k] ||_2                 (FP32 delta, reduced in float64, recorded)
    clipped_u  = delta_u * min(1, S / norm_u)                (flat clipping; factor exactly 1.0 when norm_u <= S)
    Sigma      = sum_{u in S_r} clipped_u                    (UNWEIGHTED; dropped clients contribute a zero delta)
    theta_{r+1}[k] = theta_r[k] + (Sigma[k] + z * S * xi_r[k]) / m          (server LR 1; m never = |S_r|)
  xi_r ~ N(0, I) is drawn on the CPU from a dedicated torch.Generator seeded with derive_seed(seed, "dp_noise", r),
  key by key in manifest order, then moved to the aggregate's device: the noise bits are therefore identical on any
  device and for any worker / shard layout (device-independent reproducibility of the noise; the client-side sums
  keep the existing FP32 determinism story: fixed shard plan, logical order, shard-index reduction).
  Buffers follow aggregate.py (FIXED / SERVER_COPY keep the server's value). A client with n_consumed = 0 is a
  counted no-op (its delta is zero). z = 0 with clipping on is the FA_1024 control (FedAvg at m = 1,024, clipped). `clip_norm=None` (only with
  z = 0) is the S-calibration mode: nothing is clipped, the pre-clip norms are returned (info["norms"]).
Side outputs: per-client norms ONLY in the calibration mode; the clipped count only in z = 0 clipped runs (FA_1024);
a z > 0 round returns no data-dependent statistic beyond theta_{r+1}.
Post-processing (SASRec recentering in Server.apply) does not affect the DP guarantee.
Poisson participation (guards only — the formula above is unchanged): under Poisson participation the cohort
size |C_r| ~ Binomial(N, m / N) may EXCEED m or be 0, so finalize_dp no longer bounds n_sampled by m (the denominator
is m whatever |C_r| is), and `allow_empty=True` (used by runtime.py for a 0-member Poisson round of a z > 0 run)
releases the noise-only update theta_r + z S xi_r / m.

Clip-norm rule (fixed before any DP run): S = median client-delta norm over FA_1024 seed-2026 rounds 1-20
(= round_idx 0..19); see `median_clip_norm`, which reads the per-round norm log that runtime.py records for every
DP-configured run.
"""
from __future__ import annotations

import math
import statistics
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor

from .adapter import BufferRule, ParamManifest
from .aggregate import AggregationError, EmptyRoundError
from .numerics import derive_seed


@dataclass(frozen=True)
class DPConfig:
    clip_norm: float | None          # S (flat L2 over all shared params); None = calibration (z must be 0)
    noise_multiplier: float = 0.0       # z; the noise std on the SUM is z * S
    denominator: int = 1024             # the FIXED cohort size m (never the survivor count)

    def __post_init__(self):
        if self.clip_norm is None:
            if self.noise_multiplier != 0.0:
                raise ValueError("noise needs a clip norm (clip_norm=None is the z = 0 calibration mode only)")
        elif not (self.clip_norm > 0 and math.isfinite(self.clip_norm)):
            raise ValueError("clip_norm must be a finite positive float")
        if not (self.noise_multiplier >= 0 and math.isfinite(self.noise_multiplier)):
            raise ValueError("noise_multiplier must be finite and >= 0")
        if int(self.denominator) != self.denominator or self.denominator < 1:
            raise ValueError("denominator must be a positive integer")

    @property
    def noise_std(self) -> float:
        return 0.0 if self.clip_norm is None else float(self.noise_multiplier) * float(self.clip_norm)


def flat_delta_norm(upload: Mapping[str, Tensor], theta_r: Mapping[str, Tensor], keys: Sequence[str]) -> float:
    """|| concat_k (upload[k] - theta_r[k]) ||_2: the FP32 delta that is summed, squared and reduced in float64."""
    s = 0.0
    for k in keys:
        d = torch.sub(upload[k].detach(), theta_r[k].detach().to(upload[k].device)).to(torch.float64)
        s += float((d * d).sum())
    return math.sqrt(s)


def clip_factor(norm: float, clip_norm: float | None) -> float:
    if clip_norm is None or norm <= clip_norm:
        return 1.0
    return float(clip_norm) / norm


@dataclass
class DPShardAccumulator:
    """Streaming sum of CLIPPED deltas for one shard, in logical order (same protocol as aggregate.py)."""
    shard_index: int
    keys: tuple
    acc: OrderedDict[str, Tensor]
    clip_norm: float | None = None
    weight: int = 0                                  # sum of n_consumed (reporting only; the mean is unweighted)
    added: list = field(default_factory=list)        # (position, key, n_consumed)
    n_noop: int = 0
    norms: list = field(default_factory=list)        # (position, key, pre-clip norm, factor) for n_consumed > 0

    @classmethod
    def empty(cls, shard_index: int, manifest: ParamManifest, template: Mapping[str, Tensor],
              dp: DPConfig) -> DPShardAccumulator:
        keys = manifest.shared_keys
        acc = OrderedDict((k, torch.zeros(template[k].shape, dtype=torch.float32, device=template[k].device))
                          for k in keys)
        return cls(shard_index, keys, acc, dp.clip_norm)

    def add(self, position: int, key: str, upload: Mapping[str, Tensor], n_consumed: int,
            theta_r: Mapping[str, Tensor]) -> None:
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
        for k in self.keys:
            if upload[k].dtype != torch.float32:
                raise AggregationError(f"upload {k} of {key} is {upload[k].dtype}, expected float32")
        self.added.append((int(position), key, n))
        if n == 0:
            self.n_noop += 1
            return
        norm = flat_delta_norm(upload, theta_r, self.keys)
        if not math.isfinite(norm):
            raise AggregationError(f"non-finite delta norm for client {key}")
        f = clip_factor(norm, self.clip_norm)
        for k in self.keys:
            d = torch.sub(upload[k], theta_r[k].to(upload[k].device))
            self.acc[k].add_(d.to(self.acc[k].device), alpha=f)
        self.norms.append((int(position), key, norm, f))
        self.weight += n


@dataclass
class DPRoundAggregate:
    total: OrderedDict[str, Tensor]
    weight: int
    added: list
    n_noop: int
    norms: list


def combine_dp_shards(shards: Sequence[DPShardAccumulator]) -> DPRoundAggregate:
    if not shards:
        raise EmptyRoundError("no shards")
    ordered = sorted(shards, key=lambda s: s.shard_index)
    idx = [s.shard_index for s in ordered]
    if idx != list(range(len(ordered))):
        raise AggregationError(f"shard set must be exactly 0..{len(ordered) - 1}, got {idx}")
    total = OrderedDict((k, v.clone()) for k, v in ordered[0].acc.items())
    for s in ordered[1:]:
        for k in total:
            total[k].add_(s.acc[k].to(total[k].device))
    return DPRoundAggregate(total, sum(s.weight for s in ordered), [a for s in ordered for a in s.added],
                            sum(s.n_noop for s in ordered), [x for s in ordered for x in s.norms])


def noise_like(keys: Sequence[str], template: Mapping[str, Tensor], *, seed: int, round_idx: int,
               std: float) -> OrderedDict[str, Tensor]:
    """std * N(0, I) per shared key (manifest order), from the dedicated CPU generator of (seed, round)."""
    g = torch.Generator(device="cpu").manual_seed(derive_seed(int(seed), "dp_noise", int(round_idx)))
    out = OrderedDict()
    for k in keys:
        xi = torch.randn(tuple(template[k].shape), generator=g, dtype=torch.float32)
        out[k] = torch.mul(xi, float(std))
    return out


def finalize_dp(agg: DPRoundAggregate, manifest: ParamManifest, server_state: Mapping[str, Tensor], dp: DPConfig, *,
                seed: int, round_idx: int, n_sampled: int,
                expected_keys: Sequence[str] | None = None, allow_empty: bool = False) -> tuple:
    """(theta_{r+1}, info). `n_sampled` = |C_r| (any size under Poisson sampling; the denominator is always m);
    dropped clients are simply absent (zero delta). `allow_empty`: a 0-member round releases the noise-only update."""
    if not agg.added and not allow_empty:
        raise EmptyRoundError("empty DP round: no client was aggregated (a 0-survivor round is a runtime no-op)")
    if expected_keys is not None:
        got = [a[1] for a in agg.added]
        if got != list(expected_keys):
            raise AggregationError(f"aggregated clients {got} != planned clients {list(expected_keys)}")
    m = int(dp.denominator)
    if not len(agg.added) <= int(n_sampled):
        raise AggregationError(f"survivors {len(agg.added)} <= sampled {n_sampled} violated")
    std = dp.noise_std
    noise = noise_like(manifest.shared_keys, agg.total, seed=seed, round_idx=round_idx, std=std) if std > 0 else None
    new = OrderedDict()
    for k in manifest.shared_keys:
        tot = agg.total[k]
        if noise is not None:
            tot = torch.add(tot, noise[k].to(tot.device))
        v = torch.add(server_state[k].to(tot.device), torch.div(tot, float(m)))
        if not bool(torch.isfinite(v).all()):
            raise AggregationError(f"non-finite DP aggregate for {k}")
        new[k] = v
    for k in manifest.buffer_keys:
        rule = manifest.entries[k].rule
        if rule not in (BufferRule.FIXED, BufferRule.SERVER_COPY):
            raise AggregationError(f"buffer {k} has unsupported rule {rule}")
        new[k] = server_state[k]
    info = {"denominator": m, "n_sampled": int(n_sampled), "n_survived": len(agg.added), "n_noop": agg.n_noop,
            "clip_norm": dp.clip_norm, "noise_multiplier": float(dp.noise_multiplier), "noise_std": std,
            "noise_seed_stream": ["dp_noise", int(round_idx)]}
    if dp.clip_norm is None:                         # S calibration (non-private by design): per-client norms
        info["norms"] = [x[2] for x in agg.norms]
        info["norm_keys"] = [x[1] for x in agg.norms]
    elif dp.noise_multiplier == 0.0:                 # FA_1024 control (non-private): the clipped count only
        info["n_clipped"] = sum(1 for x in agg.norms if x[3] < 1.0)
    # z > 0: no data-dependent side output at all (norms / clipped count are neither returned nor logged)
    return new, info


def median_clip_norm(norm_log: Iterable, rounds: Iterable[int] = range(20)) -> dict:
    """The clip-norm rule: the median pre-clip client-delta norm over the given 0-based rounds (default rounds
    1-20 = round_idx 0..19) of the FA_1024 seed-2026 run. `norm_log` = FLRun.dp_norm_log ([round, [norms]] rows).
    Clients with n_consumed = 0 are not in the log (no local training, delta 0). The log exists only for the
    S-calibration pass: the exact FA_1024 config (seed 2026, Poisson q = m / N with m = 1,024, T 384, dropout 0)
    with DPConfig(None, 0.0, 1024), rounds 1-20."""
    want = {int(r) for r in rounds}
    vals, seen = [], set()
    for r, norms in norm_log:
        if int(r) in want:
            seen.add(int(r))
            vals.extend(float(x) for x in norms)
    missing = sorted(want - seen)
    if missing:
        raise ValueError(f"norm log lacks rounds {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if not vals:
        raise ValueError("no client norms in the requested rounds")
    return {"S": statistics.median(vals), "n_norms": len(vals), "rounds_0based": sorted(want),
            "rule": "median client-delta L2 norm over FA_1024 seed-2026 rounds 1-20"}
