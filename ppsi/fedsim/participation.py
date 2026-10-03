"""The seeded participation plan and the exposure / coverage counters.

Plan (`ParticipationPlan`), deterministic from (seed, cohort manifest hash, group size, the cohort's key set):
  * keys are sorted (the caller's order is irrelevant) and must be unique non-empty strings;
  * sweep s is a permutation WITHOUT replacement of all N keys, from numpy PCG64 seeded with
    derive_seed(seed, "participation", manifest_hash, group_size, s) (BLAKE2b; never Python's salted hash);
  * each sweep is cut into fixed consecutive groups of `group_size` (64); round r is group r % R of sweep r // R, with
    R = ceil(N / group_size) rounds per sweep;
  * PARTIAL LAST GROUP (policy SMALLER_FINAL_GROUP, disclosed in `describe()`): when N % group_size != 0, the last
    round of every sweep has N % group_size clients. Nobody is dropped, repeated or borrowed from the next sweep, so
    every sweep visits every client exactly once. (The study cohorts N = 262,144 and 65,536 are multiples of 64: no
    partial group occurs there.)
  * the logical client order inside a round is the permutation order (it fixes the aggregation order).
  * option `sampling="uniform"` (default "sweep" = everything above): round r is an INDEPENDENT
    uniform sample WITHOUT replacement of min(group_size, N) keys, numpy PCG64 seeded with
    derive_seed(seed, "participation_uniform", manifest_hash, group_size, r); the logical order is the draw order.
    Visits per client are random (mean = rounds x m / N). ExposureCounters' "sweeps" are consecutive blocks of
    ceil(N / m) rounds in this mode. (NOT the sampling the DP accountant assumes; refused for DP.)
  * option `sampling="poisson"` (FA_1024, the clip-norm calibration and every DP run): each of the N clients
    joins round r INDEPENDENTLY with probability q = min(group_size, N) / N, from numpy PCG64 seeded with
    derive_seed(seed, "poisson", r) (no RNG state, so a resumed run re-derives the same
    cohort). Draw: one bounded integer u_i ~ Uniform{0, .., N-1} per client in sorted-key order, member iff
    u_i < group_size — an EXACT inclusion probability group_size / N (numpy's bounded integers are unbiased; no float
    threshold). The cohort size |C_r| ~ Binomial(N, q) varies and may be 0 (an empty round is valid); the logical
    order is the sorted-key order. This is the Poisson subsampling of the RDP accountant (accountant.py,
    add/remove adjacency); the DP aggregation keeps the FIXED denominator m = group_size (dp.py).

Drop-out (`dropout_split`): the sampled cohort of round r loses each client independently with probability p,
from numpy PCG64 seeded with derive_seed(seed, "client_dropout", r) — one uniform draw per logical position (u < p =
dropped). It depends only on (run seed, round index) and the plan, so FA / FP / PF with the same plan drop the same
clients, and a resumed run re-derives the same outcome (no RNG state to checkpoint). p = 0 draws nothing.

Counters (`ExposureCounters`), updated from each round's ACTUAL visit records (n_consumed, steps, attempts):
  exposures (loss-contributing examples incl. repeated passes and tails), EFE = exposures / N_decisions (N = unique
  eligible TRAIN decisions of the comparison union, given by the caller), visits, no-op visits, optimizer steps,
  retries and wasted steps, the local-steps distribution, per-client visit counts and per-sweep coverage (distinct
  clients visited / N). Exact integer arithmetic; `half_efe_units()` = floor(2 * exposures / N) drives the
  every-0.5-EFE evaluation without float comparisons.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from .numerics import derive_seed

POLICY = "SMALLER_FINAL_GROUP"


class PlanError(ValueError):
    pass


def keys_digest(keys: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(keys)).encode("utf-8")).hexdigest()


@lru_cache(maxsize=4)
def _sweep_perm(seed: int, manifest_hash: str, group_size: int, sweep: int, n: int) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(derive_seed(int(seed), "participation", manifest_hash,
                                                         int(group_size), int(sweep))))
    p = rng.permutation(n)
    p.setflags(write=False)
    return p


SAMPLINGS = ("sweep", "uniform", "poisson")


@lru_cache(maxsize=8)
def _uniform_round(seed: int, manifest_hash: str, group_size: int, round_idx: int, n: int) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(derive_seed(int(seed), "participation_uniform", manifest_hash,
                                                         int(group_size), int(round_idx))))
    p = rng.choice(n, size=min(int(group_size), n), replace=False, shuffle=True)
    p.setflags(write=False)
    return p


@lru_cache(maxsize=8)
def _poisson_round(seed: int, group_size: int, round_idx: int, n: int) -> np.ndarray:
    """Sorted positions of the round's Poisson cohort: u_i ~ Uniform{0..n-1} (exact), member iff u_i < group_size."""
    rng = np.random.Generator(np.random.PCG64(derive_seed(int(seed), "poisson", int(round_idx))))
    u = rng.integers(0, n, size=n, dtype=np.int64)
    p = np.flatnonzero(u < int(group_size)).astype(np.int64)
    p.setflags(write=False)
    return p


def dropout_split(keys: Sequence[str], *, seed: int, round_idx: int, p: float) -> tuple:
    """(survivors, dropped) of the round's sampled keys, both in logical order (see the module docstring)."""
    ks = list(keys)
    if not 0.0 <= float(p) < 1.0:
        raise PlanError("dropout p must be in [0, 1)")
    if p == 0.0:
        return ks, []
    rng = np.random.Generator(np.random.PCG64(derive_seed(int(seed), "client_dropout", int(round_idx))))
    u = rng.random(len(ks))
    drop = u < float(p)
    return [k for k, d in zip(ks, drop) if not d], [k for k, d in zip(ks, drop) if d]


@dataclass(frozen=True)
class ParticipationPlan:
    keys: tuple
    seed: int
    manifest_hash: str
    group_size: int = 64
    sampling: str = "sweep"

    @classmethod
    def build(cls, keys: Sequence[str], *, seed: int, manifest_hash: str, group_size: int = 64,
              expected_keys_digest: str | None = None, sampling: str = "sweep") -> ParticipationPlan:
        ks = list(keys)
        if not ks:
            raise PlanError("empty cohort")
        if any(not isinstance(k, str) or not k for k in ks):
            raise PlanError("client keys must be non-empty strings")
        if len(set(ks)) != len(ks):
            raise PlanError("duplicate client keys in the cohort")
        if not isinstance(manifest_hash, str) or not manifest_hash:
            raise PlanError("a cohort manifest hash is required")
        if group_size < 1:
            raise PlanError("group_size must be >= 1")
        if expected_keys_digest is not None and keys_digest(ks) != expected_keys_digest:
            raise PlanError("cohort keys do not match the expected keys digest")
        if sampling not in SAMPLINGS:
            raise PlanError(f"sampling must be one of {SAMPLINGS}")
        return cls(tuple(sorted(ks)), int(seed), manifest_hash, int(group_size), sampling)

    @property
    def n_clients(self) -> int:
        return len(self.keys)

    @property
    def rounds_per_sweep(self) -> int:
        return -(-self.n_clients // self.group_size)

    @property
    def partial_group_size(self) -> int:
        return self.n_clients % self.group_size

    def sweep_order(self, sweep: int) -> np.ndarray:
        if sweep < 0:
            raise PlanError("negative sweep")
        return _sweep_perm(self.seed, self.manifest_hash, self.group_size, int(sweep), self.n_clients)

    def locate(self, round_idx: int) -> tuple:
        return divmod(int(round_idx), self.rounds_per_sweep)

    @property
    def inclusion_probability(self) -> float:
        """Poisson plans: q = min(group_size, N) / N (the accountant's sampling ratio)."""
        return min(self.group_size, self.n_clients) / self.n_clients

    def round_positions(self, round_idx: int) -> np.ndarray:
        """Indices into `keys` (sorted) of the clients of round r, in logical order."""
        if self.sampling == "poisson":
            if round_idx < 0:
                raise PlanError("negative round")
            return _poisson_round(self.seed, self.group_size, int(round_idx), self.n_clients)
        if self.sampling == "uniform":
            if round_idx < 0:
                raise PlanError("negative round")
            return _uniform_round(self.seed, self.manifest_hash, self.group_size, int(round_idx), self.n_clients)
        sweep, g = self.locate(round_idx)
        return self.sweep_order(sweep)[g * self.group_size:(g + 1) * self.group_size]

    def round_keys(self, round_idx: int) -> list:
        return [self.keys[i] for i in self.round_positions(round_idx)]

    def describe(self) -> dict:
        if self.sampling == "poisson":                   # the sweep / uniform descriptions are unchanged
            return {"n_clients": self.n_clients, "group_size": self.group_size, "sampling_mode": "poisson",
                    "inclusion_probability": self.inclusion_probability,
                    "inclusion_rule": "u_i ~ Uniform{0..N-1} (PCG64, derive_seed(seed, 'poisson', r)); member iff "
                                      "u_i < group_size",
                    "rounds_per_sweep": self.rounds_per_sweep, "seed": self.seed,
                    "manifest_hash": self.manifest_hash, "keys_digest": keys_digest(self.keys),
                    "sampling": ("independent Poisson participation per client per round (q = min(group_size, N) / "
                                 "N); variable cohort size (may be 0); logical order = sorted keys; 'sweeps' are "
                                 "blocks of rounds_per_sweep rounds")}
        if self.sampling == "uniform":                   # the sweep description (and digest) is unchanged
            return {"n_clients": self.n_clients, "group_size": self.group_size, "sampling_mode": "uniform",
                    "rounds_per_sweep": self.rounds_per_sweep, "seed": self.seed,
                    "manifest_hash": self.manifest_hash, "keys_digest": keys_digest(self.keys),
                    "sampling": ("independent uniform sample without replacement of min(group_size, N) clients per "
                                 "round (PCG64 per round); 'sweeps' are blocks of rounds_per_sweep rounds")}
        return {"n_clients": self.n_clients, "group_size": self.group_size, "rounds_per_sweep": self.rounds_per_sweep,
                "partial_group_size": self.partial_group_size, "partial_group_policy": POLICY,
                "partial_group_note": ("none: N is a multiple of the group size" if self.partial_group_size == 0 else
                                       f"the last round of every sweep has {self.partial_group_size} clients"),
                "seed": self.seed, "manifest_hash": self.manifest_hash, "keys_digest": keys_digest(self.keys),
                "sampling": "seeded permutation sweeps without replacement (PCG64), fixed consecutive groups"}

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.describe(), sort_keys=True).encode()).hexdigest()


@dataclass
class ExposureCounters:
    n_decisions_total: int
    n_clients: int
    rounds_per_sweep: int
    exposures: int = 0
    rounds: int = 0
    visits: int = 0
    noop_visits: int = 0
    steps: int = 0
    retries: int = 0
    wasted_steps: int = 0
    steps_hist: dict = field(default_factory=dict)
    visit_counts: np.ndarray = None
    seen_this_sweep: np.ndarray = None
    coverage_by_sweep: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.n_decisions_total < 1:
            raise PlanError("n_decisions_total must be >= 1")
        if self.visit_counts is None:
            self.visit_counts = np.zeros(self.n_clients, dtype=np.int32)
        if self.seen_this_sweep is None:
            self.seen_this_sweep = np.zeros(self.n_clients, dtype=bool)

    @classmethod
    def for_plan(cls, plan: ParticipationPlan, n_decisions_total: int) -> ExposureCounters:
        return cls(int(n_decisions_total), plan.n_clients, plan.rounds_per_sweep)

    @property
    def efe(self) -> float:
        return self.exposures / self.n_decisions_total

    def half_efe_units(self) -> int:
        return (2 * self.exposures) // self.n_decisions_total

    def reached_efe(self, efe: float) -> bool:
        """exposures >= efe * N, exactly (efe given as a multiple of 0.5 or any float)."""
        num, den = float(efe).as_integer_ratio()
        return self.exposures * den >= num * self.n_decisions_total

    def update(self, plan: ParticipationPlan, round_idx: int, visits: Sequence,
               expected_keys: Sequence[str] | None = None) -> None:
        """`expected_keys` (drop-out): the round's SURVIVORS, which must be a subset of the planned keys;
        default = all planned keys (unchanged behaviour)."""
        sweep, g = plan.locate(round_idx)
        if g == 0:
            self.seen_this_sweep[:] = False
        pos = {k: i for i, k in zip(plan.round_positions(round_idx), plan.round_keys(round_idx))}
        if expected_keys is not None:
            if not set(expected_keys) <= set(pos):
                raise PlanError(f"round {round_idx}: survivors are not a subset of the plan")
            pos = {k: pos[k] for k in expected_keys}
        if sorted(v.key for v in visits) != sorted(pos):
            raise PlanError(f"round {round_idx}: visited clients differ from the plan")
        for v in visits:
            i = pos[v.key]
            self.exposures += int(v.n_consumed)
            self.visits += 1
            self.noop_visits += int(v.n_consumed == 0)
            self.steps += int(v.steps)
            self.retries += int(v.attempts) - 1
            self.wasted_steps += int(v.wasted_steps)
            self.steps_hist[str(int(v.steps))] = self.steps_hist.get(str(int(v.steps)), 0) + 1
            self.visit_counts[i] += 1
            self.seen_this_sweep[i] = True
        self.rounds += 1
        self.coverage_by_sweep[str(sweep)] = int(self.seen_this_sweep.sum())

    def summary(self) -> dict:
        return {"exposures": self.exposures, "efe": self.efe, "rounds": self.rounds, "visits": self.visits,
                "noop_visits": self.noop_visits, "steps": self.steps, "retries": self.retries,
                "wasted_steps": self.wasted_steps, "steps_hist": dict(sorted(self.steps_hist.items(),
                                                                           key=lambda kv: int(kv[0]))),
                "coverage_by_sweep": {s: {"clients": c, "fraction": c / self.n_clients}
                                      for s, c in self.coverage_by_sweep.items()},
                "min_visits": int(self.visit_counts.min()), "max_visits": int(self.visit_counts.max())}

    def state_dict(self) -> dict:
        import torch
        return {"n_decisions_total": self.n_decisions_total, "n_clients": self.n_clients,
                "rounds_per_sweep": self.rounds_per_sweep, "exposures": self.exposures, "rounds": self.rounds,
                "visits": self.visits, "noop_visits": self.noop_visits, "steps": self.steps, "retries": self.retries,
                "wasted_steps": self.wasted_steps, "steps_hist": dict(self.steps_hist),
                "coverage_by_sweep": dict(self.coverage_by_sweep),
                "visit_counts": torch.from_numpy(self.visit_counts.copy()),
                "seen_this_sweep": torch.from_numpy(self.seen_this_sweep.copy())}

    @classmethod
    def from_state_dict(cls, d: dict) -> ExposureCounters:
        c = cls(int(d["n_decisions_total"]), int(d["n_clients"]), int(d["rounds_per_sweep"]))
        for k in ("exposures", "rounds", "visits", "noop_visits", "steps", "retries", "wasted_steps"):
            setattr(c, k, int(d[k]))
        c.steps_hist = {str(k): int(v) for k, v in d["steps_hist"].items()}
        c.coverage_by_sweep = {str(k): int(v) for k, v in d["coverage_by_sweep"].items()}
        c.visit_counts = d["visit_counts"].numpy().astype(np.int32).copy()
        c.seen_this_sweep = d["seen_this_sweep"].numpy().astype(bool).copy()
        return c
