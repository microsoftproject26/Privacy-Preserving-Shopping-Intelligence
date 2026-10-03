"""The paired user bootstrap.

* Unit = user (eval_user_key) of one E2E user set; 1000 resamples; seed 20260926.
* ONE resample matrix W per user set: users sorted ascending; rng = numpy.random.default_rng(seed);
  for r in range(1000): W[r] = rng.integers(0, n_users, size=n_users). W is identified by its sha256 (int32
  little-endian row-major); every method, seed and contrast on the user set uses the same W, and every statistic is
  computed from W's multiplicities, so pairing is exact.
* Statistics per resample: macro = mean over drawn users of m_u; micro (cluster form) = sum S_u / sum N_u;
  RANKABLE statistics use the same draw, skipping drawn users with an empty rankable set. CI = 95% percentile.
* Every registered contrast is reported whatever its sign. Ratios are bootstrapped directly per resample; a zero
  central value in any resample makes the ratio CI UNDEFINED, with the count, and no resample is dropped.
* Per-seed values, paired per-seed deltas and the seed-mean contrast on the same W.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from .errors import PairingError
from .hashing import matrix_sha256_int32, user_set_sha256
from .metrics import per_user
from .quality import relative_loss_percent, retained_percent
from .values import ZERO_DENOMINATOR_IN_RESAMPLES, Undefined

N_RESAMPLES = 1000
SEED = 20260926
STATISTICS = ("macro_e2e", "micro_e2e", "macro_rankable", "micro_rankable")
_BLOCK = 64


def make_W(n_users: int, *, seed: int = SEED, n_resamples: int = N_RESAMPLES) -> np.ndarray:
    if n_users < 1:
        raise PairingError("a bootstrap needs at least one user")
    rng = np.random.default_rng(int(seed))
    W = np.empty((int(n_resamples), int(n_users)), dtype=np.int32)
    for r in range(int(n_resamples)):
        W[r] = rng.integers(0, n_users, size=n_users)
    return W


@dataclass(frozen=True)
class ResamplePlan:
    user_keys: np.ndarray          # sorted int64 eval_user_key
    W: np.ndarray                  # int32 [R, n]
    W_sha256: str
    seed: int
    n_resamples: int
    user_set_sha256: str | None = None

    @property
    def n_users(self) -> int:
        return int(self.user_keys.shape[0])

    def counts(self) -> Iterable[np.ndarray]:
        """Multiplicity matrices [b, n] (float64), in resample order, block by block."""
        for a in range(0, self.n_resamples, _BLOCK):
            blk = self.W[a:a + _BLOCK]
            C = np.zeros((blk.shape[0], self.n_users), dtype=np.float64)
            for i, row in enumerate(blk):
                C[i] = np.bincount(row, minlength=self.n_users)
            yield C


def make_plan(user_keys: Sequence[int], *, seed: int = SEED, n_resamples: int = N_RESAMPLES,
              client_key_hex: Sequence[str] | None = None) -> ResamplePlan:
    keys = np.asarray(user_keys, dtype=np.int64)
    if keys.ndim != 1 or np.unique(keys).size != keys.size:
        raise PairingError("user keys must be unique")
    keys = np.sort(keys)
    W = make_W(keys.size, seed=seed, n_resamples=n_resamples)
    ush = user_set_sha256(client_key_hex) if client_key_hex is not None else None
    return ResamplePlan(keys, W, matrix_sha256_int32(W), int(seed), int(n_resamples), ush)


def verify_plan(plan: ResamplePlan, registered_W_sha256: str | None = None) -> None:
    """Regenerate W and check its hash (the hash, not the numpy version, defines W)."""
    W2 = make_W(plan.n_users, seed=plan.seed, n_resamples=plan.n_resamples)
    h = matrix_sha256_int32(W2)
    if h != plan.W_sha256 or (registered_W_sha256 is not None and h != registered_W_sha256):
        raise PairingError(f"resample matrix hash {h} != registered {registered_W_sha256 or plan.W_sha256}")


@dataclass(frozen=True)
class UserStats:
    """Per-user sufficient statistics of one metric for one method (and seed) on one manifest."""
    user_keys: np.ndarray
    S: np.ndarray        # per-user credit sum over E2E rows
    N: np.ndarray        # per-user E2E row count (> 0 for every user of the set)
    SR: np.ndarray       # per-user credit sum over RANKABLE rows
    NR: np.ndarray       # per-user RANKABLE row count (may be 0)
    label: str = ""

    @property
    def n_decisions(self) -> int:
        return int(self.N.sum())


def user_stats(credit: np.ndarray, e2e_mask: np.ndarray, rankable_mask: np.ndarray, users: np.ndarray,
               label: str = "") -> UserStats:
    keys, S, N = per_user(credit, e2e_mask, users)
    kr, sr, nr = per_user(credit, np.asarray(rankable_mask, dtype=bool) & np.asarray(e2e_mask, dtype=bool), users)
    SR = np.zeros_like(S)
    NR = np.zeros_like(N)
    pos = np.searchsorted(keys, kr)
    SR[pos], NR[pos] = sr, nr
    return UserStats(keys, S, N, SR, NR, label)


def _check_pairing(plan: ResamplePlan, *stats: UserStats) -> None:
    for st in stats:
        if st.user_keys.shape != plan.user_keys.shape or not np.array_equal(st.user_keys, plan.user_keys):
            raise PairingError(f"statistics {st.label!r} cover a different user set than the resample plan")


def point(st: UserStats, statistic: str):
    if statistic == "macro_e2e":
        return float((st.S / st.N).mean())
    if statistic == "micro_e2e":
        return float(st.S.sum() / st.N.sum())
    has = st.NR > 0
    if not has.any():
        return Undefined("EMPTY_POPULATION")
    if statistic == "macro_rankable":
        return float((st.SR[has] / st.NR[has]).mean())
    if statistic == "micro_rankable":
        return float(st.SR.sum() / st.NR.sum())
    raise ValueError(f"unknown statistic {statistic!r}")


def resampled(plan: ResamplePlan, st: UserStats, statistic: str) -> np.ndarray:
    """The statistic under every resample of `plan` (float64 [R]); NaN never appears (an empty draw raises)."""
    _check_pairing(plan, st)
    out = np.empty(plan.n_resamples, dtype=np.float64)
    m = st.S / st.N
    has = st.NR > 0
    mr = np.divide(st.SR, st.NR, out=np.zeros_like(st.SR), where=has)
    a = 0
    for C in plan.counts():
        b = a + C.shape[0]
        if statistic == "macro_e2e":
            out[a:b] = C @ m / plan.n_users
        elif statistic == "micro_e2e":
            out[a:b] = (C @ st.S) / (C @ st.N)
        elif statistic == "macro_rankable":
            den = C @ has.astype(np.float64)
            if np.any(den == 0):
                raise PairingError("a resample drew no user with a rankable row")
            out[a:b] = (C @ mr) / den
        elif statistic == "micro_rankable":
            den = C @ st.NR
            if np.any(den == 0):
                raise PairingError("a resample drew no rankable row")
            out[a:b] = (C @ st.SR) / den
        else:
            raise ValueError(f"unknown statistic {statistic!r}")
        a = b
    return out


def ci95(x: np.ndarray) -> list[float]:
    lo, hi = np.percentile(np.asarray(x, dtype=np.float64), [2.5, 97.5])
    return [float(lo), float(hi)]


def _meta(plan: ResamplePlan, *stats: UserStats) -> dict:
    return {"n_users": plan.n_users, "n_decisions": [s.n_decisions for s in stats], "n_resamples": plan.n_resamples,
            "seed": plan.seed, "W_sha256": plan.W_sha256, "user_set_sha256": plan.user_set_sha256}


def contrast(plan: ResamplePlan, a: UserStats, b: UserStats, statistic: str, name: str) -> dict:
    """a - b, paired on the same W."""
    _check_pairing(plan, a, b)
    pa, pb = point(a, statistic), point(b, statistic)
    ra, rb = resampled(plan, a, statistic), resampled(plan, b, statistic)
    out = {"name": name, "kind": "delta", "statistic": statistic, "a": a.label, "b": b.label,
           "point": pa - pb, "value_a": pa, "value_b": pb, "ci95": ci95(ra - rb)}
    out.update(_meta(plan, a, b))
    return out


def ratio(plan: ResamplePlan, m: UserStats, c: UserStats, statistic: str, name: str, kind: str = "retained") -> dict:
    """retained = 100 M / C or loss = 100 (C - M) / C, bootstrapped directly per resample (never CI endpoints)."""
    if kind not in ("retained", "loss"):
        raise ValueError("kind must be 'retained' or 'loss'")
    _check_pairing(plan, m, c)
    fn = retained_percent if kind == "retained" else relative_loss_percent
    pm, pc = point(m, statistic), point(c, statistic)
    rm, rc = resampled(plan, m, statistic), resampled(plan, c, statistic)
    out = {"name": name, "kind": kind, "statistic": statistic, "a": m.label, "b": c.label, "point": fn(pm, pc)}
    zero = int(np.count_nonzero(rc == 0))
    if zero:
        out["ci95"] = Undefined(ZERO_DENOMINATOR_IN_RESAMPLES)
        out["n_resamples_zero_denominator"] = zero
    else:
        vals = 100.0 * rm / rc if kind == "retained" else 100.0 * (rc - rm) / rc
        out["ci95"] = ci95(vals)
        out["n_resamples_zero_denominator"] = 0
    out.update(_meta(plan, m, c))
    return out


def seed_contrast(plan: ResamplePlan, a_by_seed: dict[int, UserStats], b_by_seed: dict[int, UserStats],
                  statistic: str, name: str) -> dict:
    """Per-seed values, paired per-seed deltas (seed s vs seed s) and the seed-mean delta, all on the same W."""
    if sorted(a_by_seed) != sorted(b_by_seed):
        raise PairingError("per-seed contrasts need the same seeds for both methods")
    seeds = sorted(a_by_seed)
    per_seed = [contrast(plan, a_by_seed[s], b_by_seed[s], statistic, f"{name}[seed={s}]") for s in seeds]
    diffs = np.mean([resampled(plan, a_by_seed[s], statistic) - resampled(plan, b_by_seed[s], statistic)
                     for s in seeds], axis=0)
    out = {"name": name, "kind": "seed_mean_delta", "statistic": statistic, "seeds": seeds,
           "per_seed": per_seed, "point": float(np.mean([p["point"] for p in per_seed])), "ci95": ci95(diffs)}
    out.update(_meta(plan, *(a_by_seed[s] for s in seeds)))
    return out


def attach(result: dict, contrasts: Iterable[dict]) -> dict:
    """Put registered contrasts into result["bootstrap"] (keyed by name) for schema-checked reports."""
    block = {}
    for c in contrasts:
        if c["name"] in block:
            raise PairingError(f"duplicate contrast name {c['name']!r}")
        block[c["name"]] = c
    result["bootstrap"] = block
    return result


def run_registry(plan: ResamplePlan, stats: dict[str, UserStats],
                 registry: Iterable[tuple[str, str, str, str, str]]) -> list[dict]:
    """Every registered contrast, in registry order, whatever its sign.
    registry rows: (name, kind in {delta, retained, loss}, a, b, statistic)."""
    out = []
    for name, kind, a, b, statistic in registry:
        if kind == "delta":
            out.append(contrast(plan, stats[a], stats[b], statistic, name))
        else:
            out.append(ratio(plan, stats[a], stats[b], statistic, name, kind))
    return out
