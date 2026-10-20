"""CPU binary-tree prefix noise and quadratic FTRL on clipped client deltas.

Math and state primitives only: no model/data loader, training launcher or TEST
route. All model vectors are the shared trainable vector chosen by the caller.
FP64 private sums avoid unnecessary accumulation error; outputs retain theta's
dtype. The Gaussian model assumes ideal independent randomness. The keyed local
Torch generators are reproducible implementation streams, not a cryptographic
or finite-precision privacy proof. Real runs must keep their noise seed private.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math

import torch

from .dpftrl_statistics import finite, integer, prefix_intervals


def _validate_tensor(value, shape=None):
    if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.dtype not in (torch.float32, torch.float64):
        raise ValueError("CPU float32/float64 tensor required")
    if value.numel() == 0 or (shape is not None and tuple(value.shape) != tuple(shape)):
        raise ValueError("nonempty tensor of the exact fixed shape required")
    if not torch.isfinite(value).all().item():
        raise ValueError("tensor must be finite")


def checked_tensor(value, shape=None):
    _validate_tensor(value, shape)
    return value.detach().to(dtype=torch.float64).clone()


def clip_l2(delta, clip_norm):
    """Flat L2 clipping without publishing a norm or clipping indicator."""
    bound = finite(clip_norm, "clip_norm", positive=True)
    result = checked_tensor(delta)
    scale = float(torch.max(torch.abs(result)).item())
    if scale:
        norm_scaled = float(torch.linalg.vector_norm(result / scale).item())
        # This ordering stays finite even when the unscaled norm would overflow.
        factor = min(1.0, (bound / scale) / norm_scaled)
        result.mul_(factor)
    return result


def _node_seed(noise_seed, epoch, start, stop):
    key = json.dumps(["DPFTRL_GAUSSIAN_NODE_1", noise_seed, epoch, start, stop], separators=(",", ":"))
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "little")


@dataclass(frozen=True)
class _Node:
    start: int
    stop: int
    raw: torch.Tensor
    noise: torch.Tensor


class GaussianPrefixTree:
    """Noised prefix sums. Raw sums and private seed have no public getter."""
    def __init__(self, reference, *, clip_norm, noise_multiplier, noise_seed, epoch=0):
        self._reference = checked_tensor(reference)
        self._shape = tuple(reference.shape)
        self._clip = finite(clip_norm, "clip_norm", positive=True)
        self._z = finite(noise_multiplier, "noise_multiplier")
        self._std = self._clip * self._z
        if not math.isfinite(self._std):
            raise ValueError("Gaussian node standard deviation must be finite")
        self._seed = integer(noise_seed, "private noise_seed")
        self._epoch = integer(epoch, "epoch")
        if not 0 <= self._seed < 2 ** 64 or self._epoch < 0:
            raise ValueError("private seed must be uint64 and epoch nonnegative")
        self._nodes, self._count = {}, 0

    @property
    def count(self):
        return self._count

    def _noise(self, start, stop):
        if self._std == 0:
            return torch.zeros(self._shape, dtype=torch.float64)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_node_seed(self._seed, self._epoch, start, stop))
        noise = torch.randn(self._shape, dtype=torch.float64, generator=generator) * self._std
        if not torch.isfinite(noise).all().item():
            raise ValueError("Gaussian node noise is nonfinite")
        return noise

    def _prepare_append(self, leaf):
        raw = checked_tensor(leaf, self._shape)
        nodes = dict(self._nodes)
        start, stop, level = self._count, self._count + 1, 0
        while level in nodes:
            previous = nodes.pop(level)
            if previous.stop != start:
                raise ValueError("invalid dyadic carry state")
            raw = previous.raw + raw
            start, level = previous.start, level + 1
        if not torch.isfinite(raw).all().item():
            raise ValueError("private sum overflow")
        nodes[level] = _Node(start, stop, raw, self._noise(start, stop))
        ordered = sorted(nodes.values(), key=lambda node: node.start)
        if tuple((n.start, n.stop) for n in ordered) != prefix_intervals(stop):
            raise ValueError("tree intervals do not form the exact public prefix")
        noised = torch.zeros(self._shape, dtype=torch.float64)
        for node in ordered:
            noised = noised + node.raw + node.noise
        if not torch.isfinite(noised).all().item():
            raise ValueError("noised prefix overflow")
        return nodes, stop, noised

    def _commit(self, prepared):
        self._nodes, self._count, _ = prepared

    def append(self, leaf):
        prepared = self._prepare_append(leaf)
        self._commit(prepared)
        return prepared[2].clone()

    def current(self):
        result = torch.zeros(self._shape, dtype=torch.float64)
        for node in sorted(self._nodes.values(), key=lambda node: node.start):
            result = result + node.raw + node.noise
        return result.clone()

    def public_geometry(self):
        return {"epoch": self._epoch, "released_prefixes": self._count,
                "active_intervals": [list(x) for x in prefix_intervals(self._count)],
                "node_noise_std": self._std, "accumulation_dtype": "float64",
                "noise_rng": "private CPU generator keyed by epoch and node interval"}


def _roster(keys):
    if isinstance(keys, (str, bytes)):
        raise ValueError("public roster must be a sequence of unique nonempty strings")
    keys = tuple(keys)
    if not keys or any(not isinstance(k, str) or not k for k in keys) or len(keys) != len(set(keys)):
        raise ValueError("public roster must contain unique nonempty strings")
    return keys


class ClientUpdateFTRL:
    """Fixed declared sweeps, zero slots and a fixed planned denominator.

The caller supplies a public permutation per sweep. Every planned slot is
visited exactly once, including omitted/None updates. No weights or counters
derived from private client data appear in the public status. Optional
post_recenter must be deterministic, data-independent postprocessing.
    """
    def __init__(self, theta_start, *, public_roster, sweep_orders, clip_norm,
                 noise_multiplier, noise_seed, eta, group_size=64, post_recenter=None):
        self._theta = checked_tensor(theta_start).to(dtype=theta_start.dtype)
        self._shape, self._dtype = tuple(theta_start.shape), theta_start.dtype
        self._roster = _roster(public_roster)
        self._group = integer(group_size, "planned group_size")
        if self._group < 1:
            raise ValueError("planned group_size must be positive")
        self._orders = tuple(_roster(order) for order in sweep_orders)
        if not self._orders or any(len(o) != len(self._roster) or set(o) != set(self._roster) for o in self._orders):
            raise ValueError("every fixed sweep must be an exact permutation of the public roster")
        self._clip = finite(clip_norm, "clip_norm", positive=True)
        self._z = finite(noise_multiplier, "noise_multiplier")
        self._eta = finite(eta, "eta", positive=True)
        self._seed = integer(noise_seed, "private noise_seed")
        self._post = post_recenter
        if post_recenter is not None and not callable(post_recenter):
            raise ValueError("post_recenter must be callable or None")
        self._sweep, self._round, self._seen, self._anchor, self._tree = -1, 0, set(), None, None
        self.begin_sweep()

    @property
    def rounds_per_sweep(self):
        return (len(self._roster) + self._group - 1) // self._group

    def begin_sweep(self):
        if self._sweep >= 0 and self._round != self.rounds_per_sweep:
            raise ValueError("cannot restart an incomplete public sweep")
        if self._sweep + 1 >= len(self._orders):
            raise ValueError("all predeclared sweeps already started")
        if self._sweep >= 0 and self._seen != set(self._roster):
            raise ValueError("public sweep did not cover every slot exactly once")
        tree = GaussianPrefixTree(self._theta, clip_norm=self._clip, noise_multiplier=self._z,
                                  noise_seed=self._seed, epoch=self._sweep + 1)
        self._sweep += 1
        self._round, self._seen = 0, set()
        self._anchor, self._tree = checked_tensor(self._theta), tree

    def next_group(self):
        if self._round >= self.rounds_per_sweep:
            raise ValueError("public sweep complete; explicit restart required")
        first = self._round * self._group
        return self._orders[self._sweep][first:first + self._group]

    def step(self, updates):
        planned = self.next_group()
        pairs = list(updates.items()) if isinstance(updates, Mapping) else list(updates)
        keys = []
        for pair in pairs:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError("updates must contain (public key, delta or None) pairs")
            key, value = pair
            if not isinstance(key, str) or key not in planned or key in keys or key in self._seen:
                raise ValueError("duplicate or unplanned client update")
            keys.append(key)
            if value is not None:
                _validate_tensor(value, self._shape)
        # Canonical planned-key order, independent of uploader arrival order.
        values = dict(pairs)
        total = torch.zeros(self._shape, dtype=torch.float64)
        for key in planned:
            value = values.get(key)
            if value is not None:
                total = total + clip_l2(value, self._clip)
        prepared = self._tree._prepare_append(total)
        theta = (self._anchor + (self._eta / self._group) * prepared[2]).to(dtype=self._dtype)
        checked_tensor(theta, self._shape)
        if self._post is not None:
            theta = self._post(theta.clone())
            checked_tensor(theta, self._shape)
            if theta.dtype != self._dtype:
                raise ValueError("postprocessing changed the fixed model dtype")
        # All input and postprocessing checks precede committing this round.
        self._tree._commit(prepared)
        self._theta = theta.detach().clone()
        self._seen.update(planned)
        self._round += 1
        return self.weights()

    def weights(self):
        return self._theta.clone()

    def public_status(self):
        return {"sweep_index": self._sweep, "rounds_completed_in_sweep": self._round,
                "rounds_per_sweep": self.rounds_per_sweep, "predeclared_sweeps": len(self._orders),
                "planned_denominator": self._group, "sampling": "public_permutation_sweep",
                "absent_slots": "zero contribution; planned slot still visited",
                "budget_basis": "fixed public sweeps; actual EFE is not inferred",
                "accumulation_dtype": "float64", "model_dtype": str(self._dtype),
                "server_optimizer": "quadratic_prefix_FTRL", "momentum": 0,
                "privacy_certified": False}
