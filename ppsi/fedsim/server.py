"""In-process synchronous round driver for FA / FP / PF.

One round r:
  1. theta_r = server.broadcast(): an immutable FP32 snapshot of the shared state (+ buffers). For SASRec the server
     has recentered it (at init and after every aggregation), so theta_r — the FedProx anchor — is in the same gauge
     as the per-step recentered local models;
  2. the round's clients are in the given LOGICAL order (duplicates refused); `plan_shards` cuts it into n_shards
     contiguous blocks; shard s runs on worker s % len(workers) — in this driver sequentially, so one worker model is
     reused across many clients (its state is fully reloaded per visit; PF state comes from the store by key);
  3. each visit: virtual download, `client_update` with up to `max_attempts` attempts (every attempt reloads theta_r
     and the committed p_u and re-derives the SAME visit seed, so a retry is bit-identical to a first success; a
     failed attempt is never aggregated; its steps are reported as wasted), virtual upload, streaming accumulation;
     if every attempt fails the round aborts (RoundAborted) and the server state and every p_u are unchanged;
  4. finalize (n_consumed weights, buffer rules, empty/zero-weight refusal, exactly-once check), load into the server
     model, server_post_aggregate (SASRec recentering), then commit the staged PF states.
The process-pool driver (pool.py) reuses steps 2-4 with the same shard reduction, so it gives the same bits.

Server optimisers (Reddi, Charles, Zaheer, Garrett, Rush, Konecny, Kumar & McMahan 2021, ICLR, "Adaptive Federated
Optimization", Algorithm 2 FEDADAM). OFF by default: a Server without a server optimiser applies the
aggregate exactly as before (FedAvg, server LR 1). With `attach_server_opt(ServerOptConfig("fedadam", lr=eta_s))`,
`Server.apply(new_state)` treats new_state as the FedAvg / DP-FedAvg result and uses
    Delta_r = new_state - theta_r                      (per shared key, FP32: the delta FedAvg applies with LR 1; for DP
                                                        arms the noisy fixed-denominator mean -> post-processing)
    m_r = beta1 m_{r-1} + (1 - beta1) Delta_r
    v_r = beta2 v_{r-1} + (1 - beta2) Delta_r^2
    theta_{r+1} = theta_r + eta_s m_r / (sqrt(v_r) + tau)          (no bias correction)
with beta1 0.9, beta2 0.99, tau 1e-3, m_0 = 0 and v_0 = tau^2 (Alg. 2 line 1 "v_{-1} >= tau^2"). Buffers keep the
server's value (aggregate.py rules). Every step is one
elementwise FP32 torch op in a fixed order (no fused kernels), so it is deterministic on a given device; the moments are
committed only after every key's result is finite. Every path that changes theta goes through Server.apply (run_round,
pool.run_round, the noise-only release of an empty Poisson DP round); a recorded no-op round (0 survivors, no aggregation) makes no server
step and leaves m, v unchanged. SASRec recentering (server_post_aggregate) follows the step, as for FedAvg.
FedAvgM (Hsu, Qi & Brown 2019): `ServerOptConfig("fedavgm", lr=1.0)` (beta 0.9):
m_r = beta m_{r-1} + Delta_r, theta_{r+1} = theta_r + eta_s m_r, m_0 = 0; the same state / checkpoint / no-op rules. At the
default eta_s = 1 the step is evaluated as theta_{r+1} = new_state + beta m_{r-1} (algebraically identical; it avoids
re-adding the rounded Delta), so beta = 0 returns new_state itself: FedAvgM(beta 0) == FedAvg bit for bit.
Round observer (OFF by default): `Server.observer` (default None) is called at the end of `Server.apply` as
observer(theta_r, aggregate, theta_{r+1}) — theta_r and theta_{r+1} are fresh broadcast clones, the aggregate is the
state apply was given (for DP the noisy, fixed-denominator mean: post-noise only). It runs after the step and the
post-aggregate hook, draws no random number and modifies no tensor; with observer None apply is the code path above.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field

import torch
from torch import Tensor

from .aggregate import EmptyRoundError, ShardAccumulator, combine_shards, finalize, plan_shards
from .client import (
    ClientData,
    ClientResult,
    FaultPlan,
    LocalSolver,
    PersonalState,
    PFConfig,
    client_update,
    with_attempts,
)
from .comm import VirtualChannel
from .dp import DPConfig, DPShardAccumulator, combine_dp_shards, finalize_dp
from .numerics import state_digest
from .personal import PersonalStore, assert_unique_keys


class RoundAborted(RuntimeError):
    pass


@dataclass
class VisitRecord:
    key: str
    position: int
    shard: int
    n_valid: int
    n_consumed: int
    steps: int
    steps_per_pass: int
    attempts: int
    wasted_steps: int
    loss_sum: float


@dataclass
class RoundReport:
    round_idx: int
    method: str
    n_clients: int
    n_noop: int
    weight: int
    visits: list = field(default_factory=list)
    bytes_down: int = 0
    bytes_up: int = 0
    retry_bytes_down: int = 0
    wall_s: float = 0.0
    state_digest: str = ""
    dp: dict | None = None             # DP-FedAvg round info (dp.finalize_dp); None on the FedAvg path

    def as_json(self) -> dict:
        d = asdict(self)
        d["visits"] = [asdict(v) if not isinstance(v, dict) else v for v in self.visits]
        return d


class InitProvenanceError(RuntimeError):
    pass


class ServerOptError(RuntimeError):
    pass


SERVER_OPTS = ("fedadam", "fedavgm")
FEDADAM_DEFAULTS = {"beta1": 0.9, "beta2": 0.99, "tau": 1e-3}
FEDAVGM_DEFAULTS = {"beta1": 0.9}                         # server momentum beta (Hsu et al. 2019; Hard et al. 2018)
V_INIT = "tau_squared"                                    # v_0 = tau^2 (Reddi et al. 2021 Alg. 2), m_0 = 0


@dataclass(frozen=True)
class ServerOptConfig:
    """Server optimiser. FEDADAM: beta1 / beta2 / tau = Reddi et al.'s defaults, v_init tau^2.
    FEDAVGM: beta1 = the server momentum beta (0.9); beta2 / tau / v_init are None.
    Unset fields take the optimiser's defaults; a field that does not belong to the optimiser is refused."""
    name: str
    lr: float                                             # eta_s (launcher: FedAdam 0.01 / 0.1; FedAvgM 1.0)
    beta1: float | None = None
    beta2: float | None = None
    tau: float | None = None
    v_init: str | None = None

    def __post_init__(self):
        import math
        if self.name not in SERVER_OPTS:
            raise ValueError(f"unknown server optimiser {self.name!r} (known: {SERVER_OPTS})")
        if not (isinstance(self.lr, (int, float)) and math.isfinite(self.lr) and self.lr >= 0):
            raise ValueError(f"server lr must be a finite float >= 0, not {self.lr!r}")
        if self.name == "fedadam":
            for k, v in dict(FEDADAM_DEFAULTS, v_init=V_INIT).items():
                if getattr(self, k) is None:
                    object.__setattr__(self, k, v)
            if not (math.isfinite(self.tau) and self.tau > 0):
                raise ValueError("tau must be a finite positive float")
            if self.v_init != V_INIT:
                raise ValueError(f"v_init must be {V_INIT!r} (v_0 = tau^2, Reddi et al. 2021 Alg. 2)")
            betas = (self.beta1, self.beta2)
        else:
            if self.beta1 is None:
                object.__setattr__(self, "beta1", FEDAVGM_DEFAULTS["beta1"])
            if (self.beta2, self.tau, self.v_init) != (None, None, None):
                raise ValueError("FedAvgM takes the momentum beta1 only (beta2 / tau / v_init must be None)")
            betas = (self.beta1,)
        for b in betas:
            if not 0.0 <= float(b) < 1.0:
                raise ValueError("beta1 / beta2 must be in [0, 1)")


def make_server_opt(cfg: ServerOptConfig, manifest, template: Mapping[str, Tensor]):
    return (FedAdam if cfg.name == "fedadam" else FedAvgM)(cfg, manifest, template)


class FedAdam:
    """Server moments m, v over the manifest's shared keys (FP32, on the server model's device)."""
    MOMENTS = ("m", "v")

    def __init__(self, cfg: ServerOptConfig, manifest, template: Mapping[str, Tensor]):
        self.cfg = cfg
        self.keys = tuple(manifest.shared_keys)
        v0 = float(cfg.tau) * float(cfg.tau)
        self.m = OrderedDict((k, torch.zeros(template[k].shape, dtype=torch.float32, device=template[k].device))
                             for k in self.keys)
        self.v = OrderedDict((k, torch.full(template[k].shape, v0, dtype=torch.float32, device=template[k].device))
                             for k in self.keys)
        self.steps = 0

    def step(self, theta_r: Mapping[str, Tensor], new_state: Mapping[str, Tensor]) -> OrderedDict[str, Tensor]:
        b1, b2, tau, lr = float(self.cfg.beta1), float(self.cfg.beta2), float(self.cfg.tau), float(self.cfg.lr)
        out, m_new, v_new = OrderedDict(), {}, {}
        for k in self.keys:
            th = theta_r[k]
            if th.dtype != torch.float32:
                raise ServerOptError(f"theta {k} is {th.dtype}, expected float32")
            delta = torch.sub(new_state[k].to(th.device), th)
            m = torch.add(torch.mul(self.m[k], b1), torch.mul(delta, 1.0 - b1))
            v = torch.add(torch.mul(self.v[k], b2), torch.mul(torch.mul(delta, delta), 1.0 - b2))
            upd = torch.div(m, torch.add(torch.sqrt(v), tau))
            new = torch.add(th, torch.mul(upd, lr))
            if not (bool(torch.isfinite(m).all()) and bool(torch.isfinite(v).all()) and bool(torch.isfinite(new).all())):
                raise ServerOptError(f"non-finite FedAdam step for {k} (moments unchanged)")
            m_new[k], v_new[k], out[k] = m, v, new
        for k, t in new_state.items():                       # buffers: the aggregate's (= the server's) value
            if k not in out:
                out[k] = t
        self.m.update(m_new)
        self.v.update(v_new)
        self.steps += 1
        return out

    def state_dict(self) -> dict:
        d = {"cfg": asdict(self.cfg), "steps": int(self.steps), "keys": list(self.keys)}
        for name in self.MOMENTS:
            d[name] = OrderedDict((k, t.detach().cpu().clone()) for k, t in getattr(self, name).items())
        return d

    def load_state_dict(self, d: Mapping) -> None:
        if dict(d.get("cfg") or {}) != asdict(self.cfg):
            raise ServerOptError(f"server optimiser config {d.get('cfg')} != this run's {asdict(self.cfg)}")
        if list(d.get("keys") or []) != list(self.keys):
            raise ServerOptError("server optimiser key set differs from the manifest's shared keys")
        for name in self.MOMENTS:
            cur = getattr(self, name)
            for k in self.keys:
                t = d[name][k]
                if t.dtype != torch.float32 or tuple(t.shape) != tuple(cur[k].shape):
                    raise ServerOptError(f"server optimiser {name}[{k}] has dtype {t.dtype} / shape {tuple(t.shape)}")
            setattr(self, name, OrderedDict((k, d[name][k].to(cur[k].device).clone()) for k in self.keys))
        self.steps = int(d["steps"])

    def digest(self) -> str:
        return state_digest(OrderedDict([(f"{n}.{k}", getattr(self, n)[k]) for n in self.MOMENTS for k in self.keys]))


class FedAvgM(FedAdam):
    """FedAvgM (Hsu, Qi & Brown 2019; Hard et al. 2018 Gboard): server momentum buffer m
    (m_0 = 0) over the shared keys: m_r = beta m_{r-1} + Delta_r; theta_{r+1} = theta_r + eta_s m_r (eta_s = 1.0
    by default). Same sign convention as FedAdam (Delta = new - theta = minus Hsu's pseudo-gradient; theta -= v there)."""
    MOMENTS = ("m",)

    def __init__(self, cfg: ServerOptConfig, manifest, template: Mapping[str, Tensor]):
        self.cfg = cfg
        self.keys = tuple(manifest.shared_keys)
        self.m = OrderedDict((k, torch.zeros(template[k].shape, dtype=torch.float32, device=template[k].device))
                             for k in self.keys)
        self.steps = 0

    def step(self, theta_r: Mapping[str, Tensor], new_state: Mapping[str, Tensor]) -> OrderedDict[str, Tensor]:
        beta, lr = float(self.cfg.beta1), float(self.cfg.lr)
        out, m_new = OrderedDict(), {}
        for k in self.keys:
            th = theta_r[k]
            if th.dtype != torch.float32:
                raise ServerOptError(f"theta {k} is {th.dtype}, expected float32")
            agg = new_state[k].to(th.device)
            delta = torch.sub(agg, th)
            m = torch.add(torch.mul(self.m[k], beta), delta)
            if lr == 1.0:                                 # theta + beta m_prev + Delta = agg + beta m_prev (no
                new = agg if beta == 0.0 else torch.add(agg, torch.mul(self.m[k], beta))   # rounding of Delta)
            else:
                new = torch.add(th, torch.mul(m, lr))
            if not (bool(torch.isfinite(m).all()) and bool(torch.isfinite(new).all())):
                raise ServerOptError(f"non-finite FedAvgM step for {k} (momentum unchanged)")
            m_new[k], out[k] = m, new
        for k, t in new_state.items():
            if k not in out:
                out[k] = t
        self.m.update(m_new)
        self.steps += 1
        return out


class Server:
    """Holds the master theta in its own adapter/module (never used as a worker).

    Initial state:
      * REGISTERED path (every run driver requires it): `Server(adapter, init_sha256=<registered theta0 hash>)`.
        The adapter must hold the registered per-seed theta0 (SASRec is ALREADY recentered once when it is built);
        its state_digest must equal `init_sha256`, and it is NOT recentered again (a second recentering changes bits,
        up to 4.7e-10 at full K, and would break the init hash). `init_provenance == "registered"`.
      * RAW path (unit tests only): `Server(adapter)` starts from a raw, non-registered state and applies the one
        initial recentering here (a no-op for families without post-aggregate recentering).
        `init_provenance == "raw_unregistered"`; FLRun refuses such a server.
        `recenter_raw_init=False` exists only for a negative control."""

    def __init__(self, adapter, *, init_sha256: str | None = None, recenter_raw_init: bool = True):
        self.adapter = adapter
        adapter.fl_role = "server"                       # finetune.py refuses a server model
        self.manifest = adapter.manifest
        self.round = 0
        if init_sha256 is not None:
            got = state_digest(adapter.broadcast_state())
            if got != init_sha256:
                raise InitProvenanceError(f"theta0 digest {got[:16]}... != registered init {init_sha256[:16]}...")
            self.init_provenance = "registered"
            self.init_sha256 = init_sha256
        else:
            if recenter_raw_init:
                adapter.server_post_aggregate()
            self.init_provenance = "raw_unregistered"
            self.init_sha256 = state_digest(adapter.broadcast_state())
        self.server_opt = None                                # None = FedAvg (server LR 1)
        self.observer = None                                  # None = no round observer

    def attach_server_opt(self, cfg: ServerOptConfig):
        """Install the server optimiser (fresh moments m_0 = 0, v_0 = tau^2) before the first round."""
        if self.server_opt is not None:
            if self.server_opt.cfg != cfg:
                raise ServerOptError("a different server optimiser is already attached")
            return self.server_opt
        if self.round != 0:
            raise ServerOptError("the server optimiser must be attached before the first round")
        self.server_opt = make_server_opt(cfg, self.manifest, self.adapter.broadcast_state())
        return self.server_opt

    def broadcast(self) -> OrderedDict[str, Tensor]:
        return self.adapter.broadcast_state(clone=True)

    def apply(self, new_state: Mapping[str, Tensor]) -> None:
        obs = self.observer                                   # monitoring only; read-only on every tensor
        if obs is not None:
            theta_r, agg = self.adapter.broadcast_state(clone=True), new_state
        if self.server_opt is not None:                       # new_state = the FedAvg aggregate
            new_state = self.server_opt.step(self.adapter.broadcast_state(clone=True), new_state)   # FedAdam / FedAvgM
        self.adapter.load_state_(new_state)
        self.adapter.server_post_aggregate()
        self.round += 1
        if obs is not None:                                   # (theta_r, the aggregate given, theta_{r+1})
            obs(theta_r, agg, self.adapter.broadcast_state(clone=True))

    def digest(self) -> str:
        return state_digest(self.adapter.broadcast_state())


def method_name(mu: float, pf: PFConfig | None) -> str:
    return "PF" if pf is not None else ("FP" if mu > 0 else "FA")


def visit_with_retry(adapter, theta_r: Mapping[str, Tensor], client: ClientData, solver: LocalSolver, *,
                     round_idx: int, seed: int, mu: float, pf: PFConfig | None,
                     get_personal: Callable[[str], PersonalState | None], fault: FaultPlan | None,
                     max_attempts: int, on_retry: Callable[[int], None] | None = None) -> ClientResult:
    wasted = 0
    errors = []
    for attempt in range(max_attempts):
        if attempt and on_retry is not None:
            on_retry(attempt)
        personal = get_personal(client.key) if pf is not None else None
        try:
            res = client_update(adapter, theta_r, client, solver, round_idx=round_idx, seed=seed, mu=mu, pf=pf,
                                personal=personal, fault=fault, attempt=attempt)
        except Exception as e:  # noqa: BLE001 — any client failure is retried identically, then aborts the round
            errors.append(f"{type(e).__name__}: {e}")
            wasted += int(getattr(e, "steps_done", 0) or 0)
            continue
        return with_attempts(res, attempt + 1, wasted)
    raise RoundAborted(f"client {client.key} failed {max_attempts} attempt(s) in round {round_idx}: {errors}")


def run_round(server: Server, workers: Sequence, clients: Sequence[ClientData], solver: LocalSolver, *,
              round_idx: int, seed: int, mu: float = 0.0, pf: PFConfig | None = None,
              store: PersonalStore | None = None, n_shards: int = 1, channel: VirtualChannel | None = None,
              fault: FaultPlan | None = None, max_attempts: int = 2, dp: DPConfig | None = None,
              n_sampled: int | None = None) -> RoundReport:
    """`dp` switches the aggregation to DP-FedAvg (dp.py: clipped deltas, noise on the sum, fixed
    denominator dp.denominator); `n_sampled` = the round's sampled cohort size before drop-out (default: the number
    of given clients). dp=None is the unchanged n_consumed-weighted FedAvg path."""
    t0 = time.perf_counter()
    if not clients:
        raise EmptyRoundError("empty round: no clients selected")
    keys = [c.key for c in clients]
    assert_unique_keys(keys)
    if pf is not None and store is None:
        raise ValueError("PF needs a PersonalStore")
    if dp is not None and (pf is not None or mu != 0):
        raise ValueError("DP-FedAvg is registered for FA only (no PF state, mu = 0)")
    for w in workers:
        if w is server.adapter:
            raise ValueError("the server model cannot be a worker")
        if w.manifest.digest() != server.manifest.digest():
            raise ValueError("worker and server manifests differ")
    channel = channel or VirtualChannel(server.manifest)
    theta_r = server.broadcast()
    manifest = server.manifest
    report = RoundReport(round_idx, method_name(mu, pf), len(clients), 0, 0)
    staged: dict = {}
    shards = []
    for s, positions in enumerate(plan_shards(len(clients), n_shards)):
        worker = workers[s % len(workers)]
        acc = (ShardAccumulator.empty(s, manifest, theta_r) if dp is None else
               DPShardAccumulator.empty(s, manifest, theta_r, dp))
        for pos in positions:
            c = clients[pos]
            report.bytes_down += channel.download(round_idx, c.key)

            def _retry(_a, _k=c.key):
                report.retry_bytes_down += channel.download(round_idx, _k, retry=True)

            res = visit_with_retry(worker, theta_r, c, solver, round_idx=round_idx, seed=seed, mu=mu, pf=pf,
                                   get_personal=(store.get if store is not None else (lambda _k: None)),
                                   fault=fault, max_attempts=max_attempts, on_retry=_retry)
            report.bytes_up += channel.upload(round_idx, c.key, res.upload)
            if dp is None:
                acc.add(pos, c.key, res.upload, res.n_consumed)  # streaming: consumed before the worker is reused
            else:
                acc.add(pos, c.key, res.upload, res.n_consumed, theta_r)
            if pf is not None and res.n_valid > 0:
                staged[c.key] = res.personal
            report.visits.append(VisitRecord(c.key, pos, s, res.n_valid, res.n_consumed, res.steps,
                                             res.steps_per_pass, res.attempts, res.wasted_steps, res.loss_sum))
        shards.append(acc)
    if dp is None:
        agg = combine_shards(shards)
        new_state = finalize(agg, manifest, theta_r, expected_keys=keys)
    else:
        agg = combine_dp_shards(shards)
        new_state, report.dp = finalize_dp(agg, manifest, theta_r, dp, seed=seed, round_idx=round_idx,
                                           n_sampled=len(clients) if n_sampled is None else int(n_sampled),
                                           expected_keys=keys)
    server.apply(new_state)
    for k, st in staged.items():
        store.commit(k, st)
    report.n_noop, report.weight = agg.n_noop, agg.weight
    report.wall_s = time.perf_counter() - t0
    report.state_digest = server.digest()
    return report
