"""The FA / FP / PF run loop: participation plan + s(EFE) LR + counters + evaluation + checkpoints.

Before any training, `FLRun` builds the PLANNED ROUND-LR TABLE of the whole main phase from the plan and each
client's valid-decision count: for round r, E_r (planned exposures before), n_r (= passes x valid rows of its
clients), lr_r = lr_peak_local * s((E_r + n_r) / N) under the exposure point "end"; it refuses the run unless the
rounds with n_r > 0 all have lr_r > 0 and the final main round ends at EXACTLY 6N exposures with
lr = peak * s(6.0) (0.01 x peak to 5e-16). The table's sha256 is recorded in the run state and in every checkpoint.

One round r:
  lr_r    = the table's value (constant over the round's local steps; the PF personal LR = pf.lr x the same factor)
  keys    = plan.round_keys(r)                               (seeded sweep, fixed group of 64, logical order)
  report  = run_round(...) or pool.run_round(...)            (n_shards, max_attempts from the frozen config)
  planned n_r == actual sum of n_consumed is ENFORCED; counters.update(report)
  every time EFE crosses a multiple of 0.5 (exact integer test): eval_fn(server) -> metric; strictly-best
    (metric > best + 0.0002) -> best.pt; at 6.0 EFE -> endpoint_6.0.pt and the run stops (no early stopping)
  every `ckpt_every` rounds and at the end: latest.pt (atomic, validated).
Continuation: rounds past 6.0 EFE only after `continue_after_endpoint(record)` with a STILL_IMPROVING record that
matches a recomputation from this run's own evaluations (strictly-best at 5.5 or 6.0); it then uses the continuation
schedule up to 8.0 and stops after 2 consecutive non-improving evaluations or at 8.0.
Resume (`FLRun.resume`) restores theta, the round cursor, the counters, the evaluation history / running best, the
phase, the ledger totals, the PF store with its moments and the global RNG states; the config, plan, init and LR-table
digests must match. Local optimizer state is not checkpointed because FA / FP / PF reset it at every visit; the
per-visit RNG is derived from (seed, round, client key).

Options (all OFF by default; with them off every code path, digest and checkpoint is unchanged):
  * `endpoint_rounds = T` switches the run to a ROUND-COUNT endpoint: exactly T rounds, then the endpoint. Needed
    whenever the consumed exposures are not planned to hit 6N exactly (drop-out; uniform per-round sampling). The
    planned LR table then uses NORMALIZED schedule progress: round r uses peak * s(6.0 * E_end,r / E_total), with
    E_total = the planned consumed exposures of all T rounds (drop-out included, so it is exact), so the s shape and
    the 0.01 x peak endpoint are preserved; the half-EFE evaluations fire on the same normalized units
    floor(12 * E / E_total). The raw EFE (E / N_decisions) is measured and reported every round. No continuation.
  * `dropout_p = p`: each sampled client is dropped independently with p (participation.dropout_split, seeded by
    (run seed, round)); only survivors train and are aggregated (FedAvg: weights = n_consumed); a round with 0
    survivors is a recorded no-op (theta unchanged, no aggregation, no noise). The per-round record (sampled /
    dropped / survived ids) is returned by `step()`; `participation_log` (round, n_sampled, dropped ids, n_survived,
    noop) is checkpointed.
  * `dp = DPConfig(S, z, m)` (method FA only): DP-FedAvg aggregation (dp.py) with the fixed denominator m and noise
    from the (seed, round) generator. Every DP config (z > 0, FA_1024 z = 0, the S calibration) requires a plan with
    sampling="poisson" ("uniform" and "sweep" are refused), group_size == m, and dropout_p = 0. Per-client pre-clip
    norms are kept in `dp_norm_log` (checkpointed) ONLY in the S-calibration pass (clip_norm=None) for
    dp.median_clip_norm; clipped runs never log them (no non-private side output).
  * Poisson plans: the cohort C_r (|C_r| ~ Binomial(N, q), q = m / N) is a function of (seed, r) only.
    - LR schedule (expected-exposure accounting): planned n_r = q x passes x V (V = the valid rows of all N clients,
      i.e. N x E[visit rows]); E_total = T x q x passes x V (= 6.0 x N_decisions for T 384, m 1,024, N 131,072,
      passes 2 when N_decisions = V). The normalized progress 6.0 x E_expected,r / E_total is then exactly
      6.0 x (r + 1) / T, so the LR and the 12 evaluation marks (floor(12 x rounds done / T)) depend on the round index
      only - never on which clients were sampled or on their row counts (a data-dependent LR would sit outside the
      DP accounting).
    - Realised consumption: the table rows still carry the REALISED planned n_r = passes x valid rows of the
      realised cohort (known from the seed), and planned == actual sum of n_consumed is enforced every round, exactly
      as for the other plans; the realised total is reported as E_realised_total, raw EFE every round.
    - Empty cohort: z > 0 -> the noise-only release theta_r + z S xi_r / m (dp.finalize_dp(allow_empty=True),
      then Server.apply); z = 0 (FA_1024, S calibration) and non-DP plans -> a recorded no-op (theta unchanged, no
      post-processing), as the 0-survivor drop-out round.
  * `server_opt = ServerOptConfig("fedadam", lr=eta_s)` attaches the FedAdam server optimiser (server.py) before
    training; every server step (run_round, pool, the noise-only release) then goes through it. With server_opt =
    None the config digest, the checkpoint payload and every bit are unchanged. With it on, the config digest carries
    the full optimiser config and every latest.pt carries the moments m, v and the step count ("server_opt"); resume
    restores them bitwise (and refuses a checkpoint without them, or with them for a FedAvg run). best.pt /
    endpoint_6.0.pt carry theta only (evaluation needs no moments). The same holds for
    `ServerOptConfig("fedavgm", lr=1.0)` (server momentum buffer m).
  * `lr_shape = "fl_const"` (round-count runs only) replaces the central s(.) by an FL-only, ROUND-indexed shape at
    the round's end point x = (r + 1) / T (exposure point "end" kept):
        f(x) = 100 x (x <= 1/100: linear warm-up over the first 1 % of rounds to the peak);  1 (constant peak);
               91/10 - 9 x (x > 9/10: linear cool-down over the last 10 % of rounds to 0.1 x peak)
    computed as an exact rational (fractions.Fraction) and rounded once to float, so the final round is exactly
    0.1 x peak. The evaluation marks, the planned exposures and the central schedule are unchanged; the LR table
    body records "lr_shape" only when it is not "central" (the central table digest is byte-identical).

Frozen item tables (OFF unless called): `freeze_item_tables(adapter, keys, source)` turns the given SHARED parameters
(SASRec: item_embed.weight, output_embed, output_bias = the input item table and the output layer) into FIXED buffers
of the adapter's parameter-role manifest, in place: requires_grad False (not trained, no gradient, excluded from the
optimizer, the clip norm and the FedProx term), absent from extract_shared (not uploaded; VirtualChannel.upload and
every aggregator refuse any other key set), never aggregated (aggregate.py / dp.py keep the server's value for FIXED
buffers, and verify a worker's copy is bit-identical at every visit), and - because the clipping and the noise of
dp.py run over the manifest's shared keys only - a DP variant clips / noises exactly the uploaded parameter count.
`source` (the server's broadcast state) is the one-time bootstrap copy of the frozen tables into a worker. The output
layer's recentering hooks (post_step / server_post_aggregate) become no-ops for a frozen output layer: the common mode
of a frozen table cannot drift, and re-applying the recentering would change its bits. RunConfig and every digest are
untouched: the freeze lives in the manifest (its digest is in every checkpoint, so a frozen run never resumes into an
unfrozen one). Process pools are refused (their workers are rebuilt elsewhere).

Monitoring (never part of RunConfig, the checkpoint payload or any digest):
  * every step() record carries `train_loss` = round_loss_record(the round's visit records): n_clients / n_steps
    (visits with steps > 0), loss_sum (the sum of VisitRecord.loss_sum, the batch-mean CE summed over a visit's
    steps), mean_step_loss = loss_sum / n_steps and client_mean_loss = the mean over those clients of
    loss_sum_u / steps_u (None when no client trained; a non-finite value is recorded as None with finite False).
    It is computed from values the round already produced (no extra forward, no RNG, no state).
  * with `server.observer` set (update_log.UpdateNormObserver) every step() record also carries `update_norms` = the
    observer's record of the round's server step (server_step True; per-group / total norms over the trained subset,
    cos with the previous step) or {"server_step": False} for a round without a step, plus n_clients (omitted for
    z > 0 DP runs: post-noise server-side quantities only).
RunConfig.from_dict builds the solver with client.solver_from_dict (an ItemLRSolver iff the dict carries a non-None
item_lr_mult, else the plain LocalSolver); RunConfig.digest drops a None solver.item_lr_mult, so a plain config digest
does not depend on that option.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace

from .checkpoint import CheckpointError, CheckpointManager, restore_rng_state, rng_state
from .client import ClientData, LocalSolver, PFConfig, solver_from_dict, valid_rows
from .comm import Ledger, VirtualChannel
from .dp import DPConfig
from .participation import ExposureCounters, ParticipationPlan, dropout_split
from .personal import PersonalStore
from .schedule import CEILING, ENDPOINT_EFE, check_exposure_point, round_efe_point, s_of_efe
from .server import Server, ServerOptConfig, run_round

FORMAT_VERSION = 2
PATIENCE = 2


class LRTableError(ValueError):
    pass


LR_SHAPES = ("central", "fl_const")
FL_CONST = {"warmup_frac": "1/100", "cooldown_frac": "1/10", "final_factor": "1/10"}


def fl_const_factor(r: int, T: int) -> float:
    """FL_CONST at the end point x = (r + 1) / T of round r (0-based) of T: exact rational, one rounding."""
    from fractions import Fraction
    r, T = int(r), int(T)
    if not 0 <= r < T:
        raise LRTableError(f"round {r} outside 0..{T - 1}")
    x = Fraction(r + 1, T)
    if x <= Fraction(1, 100):
        f = 100 * x
    elif x <= Fraction(9, 10):
        f = Fraction(1)
    else:
        f = Fraction(91, 10) - 9 * x
    return float(f)


FROZEN_ITEM_KEYS = ("item_embed.weight", "output_embed", "output_bias")   # input item table + output layer
RECENTER_PAIR = ("output_embed", "output_bias")                         # frozen together or not at all


class FreezeError(ValueError):
    pass


def frozen_keys_of(adapter) -> tuple:
    """The frozen item-table keys of an adapter (() when nothing is frozen)."""
    return tuple(getattr(adapter, "fl_frozen_keys", ()) or ())


def _noop_hook() -> None:
    return None


def freeze_item_tables(adapter, keys=FROZEN_ITEM_KEYS, source: Mapping | None = None) -> dict:
    """Freeze `keys` of `adapter` in place (see the module docstring). Idempotent for the same keys;
    refused for another key set, a key that is not a SHARED parameter, a key with a tied alias, or half of the
    recentering pair. `source` = the server's state to copy the frozen tensors from (worker bootstrap). Returns the
    freeze record (keys, numel / bytes frozen and still shared)."""
    from dataclasses import replace as _replace

    import torch

    from .adapter import BufferRule, ParamManifest, Role
    keys = tuple(str(k) for k in keys)
    if not keys or len(set(keys)) != len(keys):
        raise FreezeError(f"freeze keys must be a non-empty set, got {keys}")
    have = frozen_keys_of(adapter)
    man = adapter.manifest
    if have:
        if have != keys:
            raise FreezeError(f"adapter already frozen with {have}, not {keys}")
    else:
        if any(k in keys for k in RECENTER_PAIR) and not all(k in keys for k in RECENTER_PAIR):
            raise FreezeError(f"the output layer {RECENTER_PAIR} is frozen together (recentering pair) or not at all")
        for k in keys:
            e = man.entries.get(k)
            if e is None or e.role != Role.SHARED:
                raise FreezeError(f"{k} is not a SHARED parameter of the manifest (role {getattr(e, 'role', None)})")
            if any(a.alias_of == k for a in man.entries.values() if a.role == Role.ALIAS):
                raise FreezeError(f"{k} has a tied alias (a tied item table cannot be frozen apart from the head)")
    named = dict(adapter.module.named_parameters())
    if source is not None:                                # the one-time bootstrap download of the frozen tables
        with torch.no_grad():
            for k in keys:
                named[k].copy_(source[k])
    if not have:
        for k in keys:
            named[k].requires_grad_(False)
        entries = type(man.entries)((k, _replace(e, role=Role.BUFFER, rule=BufferRule.FIXED) if k in keys else e)
                                    for k, e in man.entries.items())
        adapter.manifest = ParamManifest(entries, man.nonpersistent_buffers, man.private_extra)
        adapter._shared = [named[k] for k in adapter.manifest.shared_keys]
        sd = adapter.module.state_dict(keep_vars=True)
        adapter._buf = {k: sd[k] for k in adapter.manifest.buffer_keys}
        if all(k in keys for k in RECENTER_PAIR):
            adapter.post_step = _noop_hook                # frozen output layer: no recentering (bits kept)
            adapter.server_post_aggregate = _noop_hook
            adapter.has_post_step = False
        adapter.fl_frozen_keys = keys
    m = adapter.manifest
    return {"frozen_keys": list(keys), "frozen_numel": sum(m.entries[k].numel for k in keys),
            "frozen_bytes": sum(m.entries[k].nbytes for k in keys), "shared_numel": m.shared_numel,
            "shared_bytes": m.shared_bytes, "manifest_digest": m.digest()}


def round_loss_record(visits) -> dict:
    """The round's client training-loss summary from its VisitRecords (see the module docstring)."""
    import math
    act = [v for v in visits if int(v.steps) > 0]
    n_steps = sum(int(v.steps) for v in act)
    loss_sum = sum(float(v.loss_sum) for v in act)
    step_mean = loss_sum / n_steps if n_steps else None
    client_mean = sum(float(v.loss_sum) / int(v.steps) for v in act) / len(act) if act else None
    vals = (loss_sum, step_mean, client_mean)
    finite = all(x is None or math.isfinite(x) for x in vals)
    fix = (lambda x: x if x is None or math.isfinite(x) else None)
    return {"n_clients": len(act), "n_steps": n_steps, "loss_sum": fix(loss_sum) if act else None,
            "mean_step_loss": fix(step_mean), "client_mean_loss": fix(client_mean), "finite": finite}


def _round_factor(cfg, r: int, e0: float, e1: float) -> float:
    """The round's LR fraction of the peak: the central s(.) at the exposure point, or FL_CONST by round index."""
    if cfg.lr_shape == "fl_const":
        return fl_const_factor(r, cfg.endpoint_rounds)
    return s_of_efe(round_efe_point(e0, e1, cfg.exposure_point))


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    method: str                          # "FA" | "FP" | "PF"
    seed: int
    lr_peak_local: float
    solver: LocalSolver                  # its `lr` is replaced every round
    mu: float = 0.0
    pf: PFConfig | None = None        # its `lr` is the personal PEAK; scaled by the round's s factor
    n_shards: int = 2
    max_attempts: int = 2
    endpoint_efe: float = ENDPOINT_EFE
    eval_every_half_efe: int = 1         # evaluate at every multiple of 0.5 EFE
    best_delta: float = 0.0002
    ckpt_every: int = 64
    exposure_point: str | None = None  # REQUIRED explicitly: "end" (the study) or, regression only, "start"
    dropout_p: float = 0.0               # seeded client drop-out (0.0 = off)
    dp: DPConfig | None = None        # DP-FedAvg aggregation (None = off)
    endpoint_rounds: int | None = None  # round-count endpoint T (None = the exact 6.0-EFE endpoint)
    server_opt: ServerOptConfig | None = None  # FedAdam / FedAvgM (None = FedAvg, server LR 1)
    lr_shape: str = "central"             # "central" (the s(.) map) or "fl_const" (round-indexed)

    def __post_init__(self):
        if self.method not in ("FA", "FP", "PF"):
            raise ValueError(f"unknown method {self.method!r}")
        if self.mu < 0 or (self.method != "FP" and self.mu != 0) or (self.method == "PF") != (self.pf is not None):
            raise ValueError(f"method {self.method} inconsistent with mu={self.mu} / pf={self.pf}")
        check_exposure_point(self.exposure_point)
        if not 0.0 <= float(self.dropout_p) < 1.0:
            raise ValueError("dropout_p must be in [0, 1)")
        if self.endpoint_rounds is not None and (int(self.endpoint_rounds) != self.endpoint_rounds
                                                 or self.endpoint_rounds < 1):
            raise ValueError("endpoint_rounds must be a positive integer")
        if (self.dropout_p > 0 or self.dp is not None) and self.endpoint_rounds is None:
            raise ValueError("drop-out / DP runs need a round-count endpoint (endpoint_rounds = T)")
        if self.dp is not None and self.method != "FA":
            raise ValueError("DP-FedAvg is defined for method FA only")
        if self.dp is not None and self.dropout_p > 0:
            raise ValueError("drop-out is not used with FA_1024 / DP runs (dropout_p must be 0)")
        if self.server_opt is not None and not isinstance(self.server_opt, ServerOptConfig):
            raise ValueError("server_opt must be a server.ServerOptConfig (or None)")
        if self.lr_shape not in LR_SHAPES:
            raise ValueError(f"lr_shape must be one of {LR_SHAPES}, not {self.lr_shape!r}")
        if self.lr_shape != "central" and self.endpoint_rounds is None:
            raise ValueError("the FL_CONST shape is round-indexed: it needs a round-count endpoint (endpoint_rounds)")

    @property
    def round_endpoint(self) -> bool:
        return self.endpoint_rounds is not None

    @classmethod
    def from_dict(cls, d: Mapping) -> RunConfig:
        """The launcher path: a config without an explicit `exposure_point` key is refused."""
        if "exposure_point" not in d or d["exposure_point"] is None:
            raise ValueError("run config has no explicit exposure_point (no code default is used)")
        d = dict(d)
        d["solver"] = solver_from_dict(d["solver"]) if isinstance(d["solver"], Mapping) else d["solver"]
        if isinstance(d.get("pf"), Mapping):
            d["pf"] = PFConfig(**d["pf"])
        if isinstance(d.get("dp"), Mapping):
            d["dp"] = DPConfig(**d["dp"])
        if isinstance(d.get("server_opt"), Mapping):
            d["server_opt"] = ServerOptConfig(**d["server_opt"])
        return cls(**d)

    def digest(self) -> str:
        """Everything that can change results; `ckpt_every` is operational (a resumed run may change it)."""
        d = asdict(self)
        d.pop("ckpt_every")
        for k, default in (("dropout_p", 0.0), ("dp", None), ("endpoint_rounds", None), ("server_opt", None),
                           ("lr_shape", "central")):
            if d[k] == default:                           # options off: the plain digest, bit for bit
                d.pop(k)
        if isinstance(d.get("solver"), dict) and d["solver"].get("item_lr_mult", 0) is None:
            d["solver"].pop("item_lr_mult")               # digest guard (a plain LocalSolver has no such key)
        return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()


@dataclass
class RunState:
    cursor: int = 0
    evals: list = field(default_factory=list)          # [half_efe_units, round, metric]
    best_metric: float | None = None
    best_half_efe: int | None = None
    done: bool = False
    phase: str = "MAIN"                                  # "MAIN" | "CONTINUATION"
    continuation_record: dict | None = None


def build_lr_table(plan: ParticipationPlan, n_valid_by_key: Mapping[str, int], n_decisions_total: int,
                   cfg: RunConfig, *, continuation: bool = False, start_round: int = 0,
                   start_exposures: int = 0, end_efe: float = ENDPOINT_EFE) -> dict:
    """The planned round-LR table from `start_round` until the exposures reach `end_efe` * N.
    With cfg.endpoint_rounds set: the round-count table (see the module docstring)."""
    if cfg.endpoint_rounds is not None:
        if continuation or start_round or start_exposures:
            raise LRTableError("round-count runs have no continuation table")
        return _build_lr_table_rounds(plan, n_valid_by_key, n_decisions_total, cfg, float(cfg.endpoint_efe))
    N = int(n_decisions_total)
    target = end_efe * N
    rows, E, r = [], int(start_exposures), int(start_round)
    while E < target - 1e-9:
        n = cfg.solver.passes * sum(int(n_valid_by_key[k]) for k in plan.round_keys(r))
        if E + n > target + 1e-9:
            raise LRTableError(f"round {r} would end at {E + n} exposures, past {end_efe} x N = {target} "
                               "(the final unit must end exactly at the endpoint)")
        e0, e1 = E / N, (E + n) / N
        s = s_of_efe(round_efe_point(e0, e1, cfg.exposure_point), continuation=continuation)
        rows.append([r, E, n, E + n, cfg.lr_peak_local * s, s])
        E += n
        r += 1
        if r - start_round > 10 ** 7:
            raise LRTableError("the plan never reaches the endpoint exposures")
    if E != round(target) or abs(E - target) > 1e-9:
        raise LRTableError(f"the final planned round ends at {E} exposures, not at {end_efe} x N = {target} "
                           "(the final unit must end exactly at the endpoint)")
    used = [x[4] for x in rows if x[2] > 0]
    if not used:
        raise LRTableError("no round of the plan has planned exposures")
    final = rows[-1]
    if cfg.exposure_point == "end":                       # the study point; "start" is regression-only
        if min(used) <= 0.0:
            raise LRTableError("a round with planned exposures has LR <= 0 (min LR must be > 0)")
        want = cfg.lr_peak_local * s_of_efe(end_efe, continuation=continuation)
        if final[4] != want:
            raise LRTableError(f"final round LR {final[4]!r} != peak * s({end_efe}) = {want!r}")
    body = {"exposure_point": cfg.exposure_point, "N": N, "continuation": continuation, "end_efe": end_efe,
            "lr_peak_local": cfg.lr_peak_local, "passes": cfg.solver.passes, "rows": rows}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    return {**body, "digest": digest, "n_rounds": len(rows), "min_lr": min(used), "final_lr": final[4],
            "zero_exposure_rounds": sum(1 for x in rows if x[2] == 0)}


def _poisson_table(plan: ParticipationPlan, n_valid_by_key: Mapping[str, int], cfg: RunConfig) -> dict:
    """Poisson plans: E_total = T x q x passes x V (expected exposures; see the module docstring), exact rational."""
    from fractions import Fraction
    T = int(cfg.endpoint_rounds)
    V = sum(int(n_valid_by_key[k]) for k in plan.keys)
    e = Fraction(T * min(plan.group_size, plan.n_clients) * int(cfg.solver.passes) * V, plan.n_clients)
    return {"E_total": int(e) if e.denominator == 1 else float(e), "E_total_exact": [e.numerator, e.denominator],
            "V_valid_rows": V, "inclusion_probability": plan.inclusion_probability,
            "E_total_basis": "expected exposures: T x q x passes x V (q = min(m, N) / N, V = valid rows of all N)"}


def round_survivors(plan: ParticipationPlan, cfg: RunConfig, r: int) -> tuple:
    """(sampled, survivors, dropped) of round r, all in logical order."""
    sampled = plan.round_keys(r)
    surv, drop = dropout_split(sampled, seed=cfg.seed, round_idx=r, p=cfg.dropout_p)
    return sampled, surv, drop


def _build_lr_table_rounds(plan: ParticipationPlan, n_valid_by_key: Mapping[str, int], n_decisions_total: int,
                           cfg: RunConfig, end_efe: float) -> dict:
    N, T = int(n_decisions_total), int(cfg.endpoint_rounds)
    if float(2 * end_efe) != int(2 * end_efe):
        raise LRTableError("the endpoint must be a multiple of 0.5 EFE")
    ns = [cfg.solver.passes * sum(int(n_valid_by_key[k]) for k in round_survivors(plan, cfg, r)[1])
          for r in range(T)]
    poisson = plan.sampling == "poisson"
    E_total = sum(ns)
    if E_total <= 0:
        raise LRTableError("no round of the plan has planned exposures")
    extra = {}
    if poisson:                                           # expected-exposure progress (r + 1) / T
        extra = _poisson_table(plan, n_valid_by_key, cfg)
        extra.update(E_realised_total=E_total, realised_planned_efe=E_total / N,
                     progress="normalized: s(end_efe * E_expected / E_total) = s(end_efe * (r + 1) / T)",
                     eval_units="floor(2 * end_efe * rounds_done / T)",
                     cohort_sizes=[len(round_survivors(plan, cfg, r)[1]) for r in range(T)])
    rows, E = [], 0
    for r, n in enumerate(ns):
        if poisson:
            e0, e1 = end_efe * r / T, end_efe * (r + 1) / T
        else:
            e0, e1 = end_efe * E / E_total, end_efe * (E + n) / E_total
        s = _round_factor(cfg, r, e0, e1)
        rows.append([r, E, n, E + n, cfg.lr_peak_local * s, s])
        E += n
    used = [x[4] for x in rows if x[2] > 0]
    final = rows[-1]
    if cfg.exposure_point == "end":
        if min(used) <= 0.0:
            raise LRTableError("a round with planned exposures has LR <= 0 (min LR must be > 0)")
        want = cfg.lr_peak_local * (s_of_efe(end_efe) if cfg.lr_shape == "central" else fl_const_factor(T - 1, T))
        if final[4] != want:
            raise LRTableError(f"final round LR {final[4]!r} != peak x the shape's endpoint factor = {want!r}")
    body = {"exposure_point": cfg.exposure_point, "N": N, "continuation": False, "end_efe": end_efe,
            "lr_peak_local": cfg.lr_peak_local, "passes": cfg.solver.passes, "rows": rows,
            "endpoint_rounds": T, "E_total": E_total, "progress": "normalized: s(end_efe * E / E_total)",
            "dropout_p": float(cfg.dropout_p), "sampling": plan.sampling}
    body.update(extra)                                    # Poisson: E_total = the expected exposures (+ records)
    if cfg.lr_shape != "central":                         # recorded only when on (central digest unchanged)
        body.update(lr_shape=cfg.lr_shape, lr_shape_def=dict(FL_CONST))
    digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    return {**body, "digest": digest, "n_rounds": len(rows), "min_lr": min(used), "final_lr": final[4],
            "zero_exposure_rounds": sum(1 for x in rows if x[2] == 0), "planned_efe": body["E_total"] / N}


class FLRun:
    def __init__(self, cfg: RunConfig, plan: ParticipationPlan, server: Server, get_client: Callable[[str], ClientData],
                 n_decisions_total: int, *, workers=None, pool=None, store: PersonalStore | None = None,
                 ckpt: CheckpointManager | None = None, eval_fn: Callable[[Server], float] | None = None,
                 n_valid_by_key: Mapping[str, int] | None = None):
        if (workers is None) == (pool is None):
            raise ValueError("give exactly one of workers (in-process) or pool (process pool)")
        if cfg.method == "PF" and store is None:
            raise ValueError("PF needs a PersonalStore")
        if server.init_provenance != "registered":
            raise ValueError("FLRun needs a server started from the registered theta0 (Server(..., init_sha256=...))")
        if plan.sampling != "sweep" and not cfg.round_endpoint:
            raise ValueError("uniform per-round sampling needs a round-count endpoint (endpoint_rounds = T)")
        if cfg.dp is not None:                            # EVERY DP config: DP(z>0), FA_1024 (z=0), S calibration
            if plan.sampling != "poisson":
                raise ValueError("DP-configured runs (DP, FA_1024, S calibration) need Poisson participation (plan "
                                 "sampling='poisson'): the accountant is the "
                                 "Poisson-subsampled Gaussian (add/remove); fixed-size (uniform / sweep) sampling is "
                                 f"refused, got {plan.sampling!r}")
            if plan.group_size != cfg.dp.denominator:
                raise ValueError(f"cohort size {plan.group_size} != the fixed DP denominator m = "
                                 f"{cfg.dp.denominator}")
        frozen = frozen_keys_of(server.adapter)          # server and every worker frozen alike
        if frozen and pool is not None:
            raise ValueError("frozen item tables run in-process only (process-pool workers are rebuilt elsewhere)")
        for w in (workers or ()):
            if frozen_keys_of(w) != frozen:
                raise ValueError(f"worker frozen keys {frozen_keys_of(w)} != the server's {frozen}")
        if cfg.server_opt is not None:                    # fresh moments (resume overwrites them)
            server.attach_server_opt(cfg.server_opt)
        elif server.server_opt is not None:
            raise ValueError("the server has a server optimiser but the run config has none")
        self.cfg, self.plan, self.server, self.get_client = cfg, plan, server, get_client
        for w in (workers or ()):
            if getattr(w, "fl_role", None) != "server":
                w.fl_role = "worker"                      # finetune.py refuses a live worker
        self.workers, self.pool, self.store, self.ckpt, self.eval_fn = workers, pool, store, ckpt, eval_fn
        self.n_decisions_total = int(n_decisions_total)
        if n_valid_by_key is None:                           # tests: count from the clients themselves
            n_valid_by_key = {k: int(valid_rows(get_client(k).examples).numel()) for k in plan.keys}
        self.n_valid_by_key = dict(n_valid_by_key)
        self.lr_table = build_lr_table(plan, self.n_valid_by_key, self.n_decisions_total, cfg)   # BEFORE training
        self.counters = ExposureCounters.for_plan(plan, n_decisions_total)
        self.ledger = Ledger()
        self.channel = VirtualChannel(server.manifest, ledger=self.ledger)
        self.state = RunState()
        self.lr_log: list = []
        self.participation_log: list = []                 # round-count runs only
        self.dp_norm_log: list = []                       # [round, [pre-clip norms of training clients]]

    # ------------------------------------------------------------------------------------------------ one round
    def _planned_row(self, r: int) -> list:
        rows = self.lr_table["rows"]
        i = r - rows[0][0]
        if not 0 <= i < len(rows) or rows[i][0] != r:
            raise LRTableError(f"round {r} is outside the planned LR table")
        return rows[i]

    def step(self) -> dict:
        if self.state.done:
            raise RuntimeError("run already reached its endpoint (continuation needs a STILL_IMPROVING record)")
        if self.cfg.round_endpoint:
            return self._step_rounds()
        cfg, r = self.cfg, self.state.cursor
        clients = [self.get_client(k) for k in self.plan.round_keys(r)]
        row = self._planned_row(r)
        planned = cfg.solver.passes * sum(int(valid_rows(c.examples).numel()) for c in clients)
        if planned != row[2] or self.counters.exposures != row[1]:
            raise LRTableError(f"round {r}: data / counters disagree with the planned LR table")
        e_start = self.counters.exposures / self.n_decisions_total
        e_end = (self.counters.exposures + planned) / self.n_decisions_total
        s = s_of_efe(round_efe_point(e_start, e_end, cfg.exposure_point),
                     continuation=self.state.phase == "CONTINUATION")
        if cfg.lr_peak_local * s != row[4]:
            raise LRTableError(f"round {r}: LR differs from the planned table")
        solver_r = replace(cfg.solver, lr=cfg.lr_peak_local * s)
        pf_r = replace(cfg.pf, lr=cfg.pf.lr * s) if cfg.pf is not None else None
        kw = {"round_idx": r, "seed": cfg.seed, "mu": cfg.mu, "pf": pf_r, "store": self.store, "n_shards": cfg.n_shards,
                  "channel": self.channel, "max_attempts": cfg.max_attempts}
        if self.pool is not None:
            rep = self.pool.run_round(self.server, clients, solver_r, **kw)
        else:
            rep = run_round(self.server, self.workers, clients, solver_r, **kw)
        actual = sum(int(v.n_consumed) for v in rep.visits)
        if actual != planned:
            raise RuntimeError(f"round {r}: actual n_consumed {actual} != planned {planned}")
        before = self.counters.half_efe_units()
        self.counters.update(self.plan, r, rep.visits)
        self.state.cursor += 1
        self.lr_log.append((r, solver_r.lr))
        out = {"round": r, "lr": solver_r.lr, "efe": self.counters.efe, "weight": rep.weight,
               "train_loss": round_loss_record(rep.visits)}                        # monitoring only
        if getattr(self.server, "observer", None) is not None:                    # monitoring only
            out["update_norms"] = self._update_norms(len(clients))
        after = self.counters.half_efe_units()
        if after > before and self.eval_fn is not None and after % self.cfg.eval_every_half_efe == 0:
            self._evaluate(after, r)
        if self.state.phase == "MAIN" and self.counters.reached_efe(cfg.endpoint_efe):
            self.state.done = True
            if self.ckpt is not None:
                self.ckpt.save_endpoint(self.weights_payload())
        elif self.state.phase == "CONTINUATION" and (self.counters.reached_efe(CEILING)
                                                     or self._non_improving_tail() >= PATIENCE):
            self.state.done = True
        if self.ckpt is not None and (self.state.done or self.state.cursor % cfg.ckpt_every == 0):
            self.ckpt.save_latest(self.state_dict())
        return out

    def _half_units(self) -> int:
        """Round-count runs: the normalized evaluation units floor(2 * end_efe * E / E_total) (exact integers);
        Poisson plans: floor(2 * end_efe * rounds_done / T) (data-independent)."""
        if self.plan.sampling == "poisson":
            return (int(2 * self.cfg.endpoint_efe) * self.state.cursor) // int(self.cfg.endpoint_rounds)
        return (int(2 * self.cfg.endpoint_efe) * self.counters.exposures) // int(self.lr_table["E_total"])

    def units_after_row(self, i: int) -> int:
        """Round-count runs: the evaluation units reached once planned row i (0-based) has run."""
        units = int(2 * self.cfg.endpoint_efe)
        if self.plan.sampling == "poisson":
            return (units * (int(i) + 1)) // int(self.cfg.endpoint_rounds)
        return (units * int(self.lr_table["rows"][i][3])) // int(self.lr_table["E_total"])

    def _step_rounds(self) -> dict:
        cfg, r = self.cfg, self.state.cursor
        if r >= cfg.endpoint_rounds:
            raise RuntimeError("round-count run already at its endpoint")
        sampled, survivors, dropped = round_survivors(self.plan, cfg, r)
        clients = [self.get_client(k) for k in survivors]
        row = self._planned_row(r)
        planned = cfg.solver.passes * sum(int(valid_rows(c.examples).numel()) for c in clients)
        if planned != row[2] or self.counters.exposures != row[1]:
            raise LRTableError(f"round {r}: data / counters disagree with the planned LR table")
        if self.plan.sampling == "poisson":               # the progress depends on the round index only
            e0 = cfg.endpoint_efe * r / cfg.endpoint_rounds
            e1 = cfg.endpoint_efe * (r + 1) / cfg.endpoint_rounds
        else:
            E_total = int(self.lr_table["E_total"])
            e0 = cfg.endpoint_efe * self.counters.exposures / E_total
            e1 = cfg.endpoint_efe * (self.counters.exposures + planned) / E_total
        s = _round_factor(cfg, r, e0, e1)
        if cfg.lr_peak_local * s != row[4]:
            raise LRTableError(f"round {r}: LR differs from the planned table")
        solver_r = replace(cfg.solver, lr=cfg.lr_peak_local * s)
        pf_r = replace(cfg.pf, lr=cfg.pf.lr * s) if cfg.pf is not None else None
        rep = None
        if clients:
            kw = {"round_idx": r, "seed": cfg.seed, "mu": cfg.mu, "pf": pf_r, "store": self.store, "n_shards": cfg.n_shards,
                      "channel": self.channel, "max_attempts": cfg.max_attempts}
            if cfg.dp is not None:
                kw.update(dp=cfg.dp, n_sampled=len(sampled))
            if self.pool is not None:
                rep = self.pool.run_round(self.server, clients, solver_r, **kw)
            else:
                rep = run_round(self.server, self.workers, clients, solver_r, **kw)
            actual = sum(int(v.n_consumed) for v in rep.visits)
            if actual != planned:
                raise RuntimeError(f"round {r}: actual n_consumed {actual} != planned {planned}")
        noise_only = None
        if not clients and cfg.dp is not None and cfg.dp.noise_std > 0:
            noise_only = self._noise_only_release(r)          # an empty Poisson round of a z > 0 run
        before = self._half_units()
        self.counters.update(self.plan, r, rep.visits if rep is not None else [], expected_keys=survivors)
        self.state.cursor += 1
        self.lr_log.append((r, solver_r.lr))
        plog = {"round": r, "n_sampled": len(sampled), "dropped": list(dropped), "n_survived": len(survivors),
                "noop": not survivors and noise_only is None}
        if noise_only is not None:
            plog["noise_only"] = True
        self.participation_log.append(plog)
        dp_info = rep.dp if rep is not None else noise_only
        if dp_info is not None and "norms" in dp_info:    # the S-calibration pass only (dp.finalize_dp)
            self.dp_norm_log.append([r, [float(x) for x in dp_info["norms"]]])
        elif (not clients and self.plan.sampling == "poisson" and cfg.dp is not None
              and cfg.dp.clip_norm is None):              # an empty S-calibration round logs no norm
            self.dp_norm_log.append([r, []])
        out = {"round": r, "lr": solver_r.lr, "efe": self.counters.efe, "schedule_efe": e1,
               "weight": rep.weight if rep is not None else 0, "sampled": list(sampled), "dropped": list(dropped),
               "survived": list(survivors), "noop_round": plog["noop"], "dp": dp_info,
               "train_loss": round_loss_record(rep.visits if rep is not None else [])}   # monitoring only
        if getattr(self.server, "observer", None) is not None:                    # monitoring only
            out["update_norms"] = self._update_norms(len(clients))
        after = self._half_units()
        if after > before and self.eval_fn is not None and after % self.cfg.eval_every_half_efe == 0:
            self._evaluate(after, r)
        if self.state.cursor >= cfg.endpoint_rounds:
            self.state.done = True
            if self.ckpt is not None:
                self.ckpt.save_endpoint(self.weights_payload())
        if self.ckpt is not None and (self.state.done or self.state.cursor % cfg.ckpt_every == 0):
            self.ckpt.save_latest(self.state_dict())
        return out

    def _update_norms(self, n_clients: int) -> dict:
        """The observer's record of this round's server step (update_log.py), or the no-step record.
        n_clients (the clients aggregated) is omitted for z > 0 DP runs (post-noise server-side values only)."""
        from .update_log import noop_round_record
        rec = self.server.observer.pop()
        rec = dict(server_step=True, **rec) if rec is not None else noop_round_record()
        if self.cfg.dp is None or float(self.cfg.dp.noise_multiplier) <= 0.0:
            rec["n_clients"] = int(n_clients)
        return rec

    def _noise_only_release(self, r: int) -> dict:
        """theta_{r+1} = theta_r + (0 + z S xi_r) / m for a 0-member round of a z > 0 run (dp.py math, then the
        server's post-aggregate step like every other DP round)."""
        from .dp import DPShardAccumulator, combine_dp_shards, finalize_dp
        theta_r = self.server.broadcast()
        agg = combine_dp_shards([DPShardAccumulator.empty(0, self.server.manifest, theta_r, self.cfg.dp)])
        new, info = finalize_dp(agg, self.server.manifest, theta_r, self.cfg.dp, seed=self.cfg.seed, round_idx=r,
                                n_sampled=0, expected_keys=[], allow_empty=True)
        self.server.apply(new)
        info["noise_only"] = True
        return info

    def _evaluate(self, half_units: int, r: int) -> None:
        m = float(self.eval_fn(self.server))
        improved = self.state.best_metric is None or m > self.state.best_metric + self.cfg.best_delta
        self.state.evals.append([int(half_units), int(r), m, bool(improved), self.state.phase])
        if improved:
            self.state.best_metric, self.state.best_half_efe = m, int(half_units)
            if self.ckpt is not None:
                self.ckpt.save_best(self.weights_payload())

    def _non_improving_tail(self) -> int:
        n = 0
        for e in reversed(self.state.evals):
            if e[4] != "CONTINUATION" or e[3]:
                break
            n += 1
        return n

    def run(self, max_rounds: int | None = None) -> FLRun:
        n = 0
        while not self.state.done and (max_rounds is None or n < max_rounds):
            self.step()
            n += 1
        return self

    # ------------------------------------------------------------------------------------------------ continuation
    def still_improving_record(self) -> dict:
        """Recomputed from this run's own main-phase evaluations (STILL_IMPROVING: strictly-best at 5.5 or 6.0)."""
        main = [e for e in self.state.evals if e[4] == "MAIN"]
        endpoint_units = round(2 * self.cfg.endpoint_efe)
        if not self.state.done or self.state.phase != "MAIN" or not main or main[-1][0] != endpoint_units:
            raise CheckpointError("STILL_IMPROVING is decided only after the 6.0-EFE evaluation of a finished run")
        best = None
        for e in main:
            if e[3]:
                best = e[0]
        verdict = "STILL_IMPROVING" if best is not None and best >= endpoint_units - 1 else "NOT_IMPROVING"
        evals_digest = hashlib.sha256(json.dumps(main, sort_keys=True).encode()).hexdigest()
        return {"verdict": verdict, "run_id": self.cfg.run_id, "config_digest": self.cfg.digest(),
                "best_half_efe": best, "evals_digest": evals_digest}

    def continue_after_endpoint(self, record: Mapping) -> None:
        """Continuation rounds only after a STILL_IMPROVING record that matches this run."""
        if self.cfg.round_endpoint:
            raise CheckpointError("continuation is not defined for round-count (drop-out / DP) runs")
        mine = self.still_improving_record()
        if dict(record) != mine:
            raise CheckpointError("continuation refused: the record does not match this run's own evaluations")
        if mine["verdict"] != "STILL_IMPROVING":
            raise CheckpointError("continuation refused: the run is not STILL_IMPROVING at 6.0 EFE")
        self.lr_table = build_lr_table(self.plan, self.n_valid_by_key, self.n_decisions_total, self.cfg,
                                       continuation=True, start_round=self.state.cursor,
                                       start_exposures=self.counters.exposures, end_efe=CEILING)
        self.state.phase, self.state.done, self.state.continuation_record = "CONTINUATION", False, dict(record)

    # ------------------------------------------------------------------------------------------------ state
    def weights_payload(self) -> dict:
        return {"kind": "weights", "run_id": self.cfg.run_id, "config_digest": self.cfg.digest(),
                "plan_digest": self.plan.digest(), "lr_table_digest": self.lr_table["digest"],
                "round": self.state.cursor, "efe": self.counters.efe, "phase": self.state.phase,
                "theta": dict(self.server.adapter.broadcast_state())}

    def state_dict(self) -> dict:
        d = self._state_dict_base()
        if self.cfg.round_endpoint:                      # round-count state; absent (unchanged payload) otherwise
            d["round_count_state"] = {"participation_log": [dict(x) for x in self.participation_log],
                         "dp_norm_log": [[int(r), list(n)] for r, n in self.dp_norm_log]}
        if self.cfg.server_opt is not None:              # the server optimiser moments (absent for FedAvg)
            d["server_opt"] = self.server.server_opt.state_dict()
        return d

    def _state_dict_base(self) -> dict:
        return {"kind": "resumable", "version": FORMAT_VERSION, "run_id": self.cfg.run_id,
                "config_digest": self.cfg.digest(), "plan_digest": self.plan.digest(),
                "init_sha256": self.server.init_sha256, "lr_table_digest": self.lr_table["digest"],
                "manifest_digest": self.server.manifest.digest(), "cursor": self.state.cursor,
                "server_round": self.server.round, "done": self.state.done, "phase": self.state.phase,
                "continuation_record": self.state.continuation_record,
                "theta": dict(self.server.adapter.broadcast_state()),
                "counters": self.counters.state_dict(), "ledger": self.ledger.state_dict(),
                "evals": [list(e) for e in self.state.evals], "best_metric": self.state.best_metric,
                "best_half_efe": self.state.best_half_efe, "lr_log": [list(x) for x in self.lr_log],
                "store": self.store.state_dict() if self.store is not None else None, "rng": rng_state()}

    def load_state_dict(self, d: dict) -> None:
        if d.get("kind") != "resumable" or d.get("version") != FORMAT_VERSION:
            raise CheckpointError("not a resumable FL checkpoint of this version")
        for key, mine in (("config_digest", self.cfg.digest()), ("plan_digest", self.plan.digest()),
                          ("manifest_digest", self.server.manifest.digest()),
                          ("init_sha256", self.server.init_sha256)):
            if d[key] != mine:
                raise CheckpointError(f"resume refused: {key} differs from this run's")
        self.server.adapter.load_state_(d["theta"])          # already recentered when it was saved
        self.server.round = int(d["server_round"])
        self.counters = ExposureCounters.from_state_dict(d["counters"])
        self.ledger = Ledger.from_state_dict(d["ledger"])
        self.channel = VirtualChannel(self.server.manifest, ledger=self.ledger)
        self.state = RunState(int(d["cursor"]), [list(e) for e in d["evals"]], d["best_metric"], d["best_half_efe"],
                              bool(d["done"]), d["phase"], d["continuation_record"])
        if self.state.phase == "CONTINUATION":
            self.lr_table = build_lr_table(self.plan, self.n_valid_by_key, self.n_decisions_total, self.cfg,
                                           continuation=True, start_round=self._continuation_start(),
                                           start_exposures=self._continuation_start_exposures(), end_efe=CEILING)
        if d["lr_table_digest"] != self.lr_table["digest"]:
            raise CheckpointError("resume refused: the planned LR table differs from this run's")
        self.lr_log = [tuple(x) for x in d["lr_log"]]
        if self.cfg.method == "PF":
            self.store = PersonalStore.from_state_dict(d["store"])
        if self.cfg.round_endpoint:
            if "round_count_state" not in d:
                raise CheckpointError("resume refused: a round-count run checkpoint lacks its participation log")
            self.participation_log = [dict(x) for x in d["round_count_state"]["participation_log"]]
            self.dp_norm_log = [[int(r), [float(v) for v in n]] for r, n in d["round_count_state"]["dp_norm_log"]]
            if len(self.participation_log) != self.state.cursor:
                raise CheckpointError("resume refused: participation log length != round cursor")
        if self.cfg.server_opt is not None:
            if "server_opt" not in d:
                raise CheckpointError("resume refused: a FedAdam run checkpoint lacks the server optimiser state")
            if int(d["server_opt"]["steps"]) > int(d["server_round"]):
                raise CheckpointError("resume refused: server optimiser steps exceed the server rounds")
            self.server.server_opt.load_state_dict(d["server_opt"])
        elif "server_opt" in d:
            raise CheckpointError("resume refused: the checkpoint carries a server optimiser, this run has none")
        restore_rng_state(d["rng"])

    def _continuation_start(self) -> int:
        main_rounds = build_lr_table(self.plan, self.n_valid_by_key, self.n_decisions_total, self.cfg)["n_rounds"]
        return main_rounds

    def _continuation_start_exposures(self) -> int:
        return round(self.cfg.endpoint_efe * self.n_decisions_total)

    @classmethod
    def resume(cls, *args, **kwargs) -> FLRun:
        """Build like the constructor, then continue from `ckpt.load_latest()` (if any)."""
        run = cls(*args, **kwargs)
        if run.ckpt is None:
            raise CheckpointError("resume needs a CheckpointManager")
        payload = run.ckpt.load_latest()
        if payload is not None:
            run.load_state_dict(payload)
        return run

    def summary(self) -> dict:
        out = self._summary_base()
        if self.cfg.round_endpoint:
            pl = self.participation_log
            out["participation"] = {"rounds": len(pl), "sampled": sum(x["n_sampled"] for x in pl),
                                    "dropped": sum(len(x["dropped"]) for x in pl),
                                    "survived": sum(x["n_survived"] for x in pl),
                                    "noop_rounds": sum(1 for x in pl if x["noop"]),
                                    "dropout_p": self.cfg.dropout_p, "endpoint_rounds": self.cfg.endpoint_rounds,
                                    "planned_efe": self.lr_table["planned_efe"]}
            if self.plan.sampling == "poisson":           # Poisson records (absent for sweep / uniform plans)
                sizes = [x["n_sampled"] for x in pl]
                out["participation"].update(
                    sampling="poisson", inclusion_probability=self.plan.inclusion_probability,
                    cohort_size_min=min(sizes) if sizes else None, cohort_size_max=max(sizes) if sizes else None,
                    empty_rounds=sum(1 for n in sizes if n == 0),
                    noise_only_rounds=sum(1 for x in pl if x.get("noise_only")),
                    E_total_expected=self.lr_table["E_total"], E_realised_total=self.lr_table["E_realised_total"])
            if self.cfg.dp is not None:
                out["dp"] = {"clip_norm": self.cfg.dp.clip_norm, "noise_multiplier": self.cfg.dp.noise_multiplier,
                             "denominator": self.cfg.dp.denominator, "rounds_logged": len(self.dp_norm_log)}
        return out

    def _summary_base(self) -> dict:
        return {"run_id": self.cfg.run_id, "method": self.cfg.method, "rounds": self.state.cursor,
                "done": self.state.done, "phase": self.state.phase, "server_digest": self.server.digest(),
                "store_digest": self.store.digest() if self.store is not None else None,
                "counters": self.counters.summary(), "ledger": self.ledger.snapshot(), "evals": self.state.evals,
                "best_metric": self.state.best_metric, "best_half_efe": self.state.best_half_efe,
                "lr_table_digest": self.lr_table["digest"], "plan": self.plan.describe(),
                **({"server_opt": {"cfg": asdict(self.cfg.server_opt), "steps": self.server.server_opt.steps,
                                   "moments_digest": self.server.server_opt.digest()}}
                   if self.cfg.server_opt is not None else {})}
