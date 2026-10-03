"""A process-pool round driver: N concurrent persistent workers, same bits as server.run_round.

Why processes, not threads: the models draw dropout from the process-global RNG, so two concurrent visits in one
process would interleave RNG draws. Each worker is a spawned process holding ONE persistent worker model (built once
from a picklable `WorkerSpec`, cached for the life of the process), like one process per GPU in a multi-GPU pool.

A round: the main process plans the shards (the same `plan_shards` protocol constant), submits one job per shard
(theta_r, the shard's clients with their logical positions, and for PF their committed private states), and collects
the shard accumulators in COMPLETION order; `combine_shards` then adds them in shard-index order, so the result is
bit-identical to the in-process driver with the same n_shards, whatever the timing. Private states travel between the
client-side store and the worker process only (simulator plumbing, never a server message: 0 virtual bytes); they are
committed only after the round finalizes. A failed visit is retried inside the worker with the identical seed; a shard
whose retries are exhausted aborts the round (RoundAborted), leaving the server state and every p_u unchanged.

Determinism caveat (stated tolerance): CPU kernels may reduce in a thread-count-dependent order, so every worker and
the in-process reference use the same fixed intra-op thread count (`WorkerSpec.threads`).
"""
from __future__ import annotations

import importlib
import os
import time
import warnings
from collections.abc import Sequence
from concurrent.futures import FIRST_EXCEPTION, ProcessPoolExecutor, wait
from dataclasses import dataclass
from multiprocessing import get_context

import torch

from .aggregate import EmptyRoundError, ShardAccumulator, combine_shards, finalize, plan_shards
from .client import ClientData, FaultPlan, LocalSolver, PFConfig
from .comm import VirtualChannel
from .dp import DPConfig, DPShardAccumulator, combine_dp_shards, finalize_dp
from .numerics import enforce_strict_fp32
from .personal import PersonalStore, assert_unique_keys
from .server import RoundAborted, RoundReport, Server, VisitRecord, method_name, visit_with_retry


@dataclass(frozen=True)
class WorkerSpec:
    factory: str                     # "package.module:function" returning an adapter
    kwargs: tuple = ()               # tuple of (name, value) pairs
    threads: int = 1
    strict_deterministic: bool = False   # torch.use_deterministic_algorithms(True) WITHOUT warn_only (GPU checks)

    def build(self):
        mod, fn = self.factory.split(":")
        return getattr(importlib.import_module(mod), fn)(**dict(self.kwargs))

    def on_device(self, device: str) -> WorkerSpec:
        kw = tuple((k, v) for k, v in self.kwargs if k != "device") + (("device", device),)
        return WorkerSpec(self.factory, kw, self.threads, self.strict_deterministic)


_WORKER = {}


def require_visible(dev: str) -> None:
    """Refuse a CUDA device this process cannot see, with the likely cause spelled out: a library that runs
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0") on import makes a process (and every child it spawns) see ONLY
    GPU 0 unless CUDA_VISIBLE_DEVICES was exported explicitly."""
    idx = torch.device(dev).index or 0
    n = torch.cuda.device_count()
    if idx >= n:
        raise RuntimeError(f"{dev} is not visible: {n} CUDA device(s), CUDA_VISIBLE_DEVICES="
                           f"{os.environ.get('CUDA_VISIBLE_DEVICES')!r}. Export CUDA_VISIBLE_DEVICES explicitly "
                           "before starting.")


def _worker(spec: WorkerSpec):
    key = repr(spec)
    if key not in _WORKER:
        torch.set_num_threads(spec.threads)
        dev = dict(spec.kwargs).get("device", "cpu")
        if str(dev).startswith("cuda"):
            if spec.strict_deterministic and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
                raise RuntimeError("strict CUDA determinism needs CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA init")
            require_visible(dev)
            torch.cuda.set_device(torch.device(dev))
        enforce_strict_fp32(warn_only=not spec.strict_deterministic)
        _WORKER[key] = spec.build()
    return _WORKER[key]


def _shard_job(spec: WorkerSpec, shard_index: int, positions: list, clients: list, theta_r: dict,
               solver: LocalSolver, round_idx: int, seed: int, mu: float, pf: PFConfig | None,
               personals: dict, fault: FaultPlan | None, max_attempts: int,
               dp: DPConfig | None = None) -> dict:
    adapter = _worker(spec)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dev = adapter.device
        theta_r = {k: v.to(dev) for k, v in theta_r.items()}             # one host->device copy per round
        acc = (ShardAccumulator.empty(shard_index, adapter.manifest, theta_r) if dp is None else
               DPShardAccumulator.empty(shard_index, adapter.manifest, theta_r, dp))
        visits, staged, retries = [], {}, 0
        for pos, c in zip(positions, clients):
            def _retry(_a):
                nonlocal retries
                retries += 1
            res = visit_with_retry(adapter, theta_r, c, solver, round_idx=round_idx, seed=seed, mu=mu, pf=pf,
                                   get_personal=lambda k: (personals[k].clone() if personals.get(k) is not None
                                                           else None),
                                   fault=fault, max_attempts=max_attempts, on_retry=_retry)
            if dp is None:
                acc.add(pos, c.key, res.upload, res.n_consumed)
            else:
                acc.add(pos, c.key, res.upload, res.n_consumed, theta_r)
            if pf is not None and res.n_valid > 0:
                staged[c.key] = res.personal
            visits.append(VisitRecord(c.key, pos, shard_index, res.n_valid, res.n_consumed, res.steps,
                                      res.steps_per_pass, res.attempts, res.wasted_steps, res.loss_sum))
        acc.acc = type(acc.acc)((k, v.cpu()) for k, v in acc.acc.items())  # the partial sum travels to the reducer
    msgs = sorted({f"{w.category.__name__}: {w.message}" for w in caught})
    return {"acc": acc, "visits": visits, "staged": staged, "retries": retries, "pid": os.getpid(),
            "device": str(dev), "t_done": time.perf_counter(), "warnings": msgs}


class ProcessPool:
    """n_workers generic processes (CPU), or one dedicated persistent process per device (`devices=[...]`): then
    shard s runs on device (s + rotation) % n_devices, so a check can swap the shard -> GPU assignment."""

    def __init__(self, spec: WorkerSpec, n_workers: int = 2, devices: Sequence[str] | None = None):
        self.spec = spec
        self.devices = list(devices) if devices else None
        ctx = get_context("spawn")
        if self.devices:
            self.n_workers = len(self.devices)
            self.execs = [ProcessPoolExecutor(max_workers=1, mp_context=ctx) for _ in self.devices]
            self.specs = [spec.on_device(d) for d in self.devices]
        else:
            self.n_workers = n_workers
            self.execs = [ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx)]
            self.specs = [spec]
        self.pids_seen: set = set()
        self.completion_orders: list = []
        self.warnings: set = set()
        self.job_devices: list = []

    def close(self) -> None:
        for ex in self.execs:
            ex.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def run_round(self, server: Server, clients: Sequence[ClientData], solver: LocalSolver, *, round_idx: int,
                  seed: int, mu: float = 0.0, pf: PFConfig | None = None, store: PersonalStore | None = None,
                  n_shards: int = 2, channel: VirtualChannel | None = None, fault: FaultPlan | None = None,
                  max_attempts: int = 2, rotation: int = 0, dp: DPConfig | None = None,
                  n_sampled: int | None = None) -> RoundReport:
        """`dp` / `n_sampled`: as in server.run_round (dp=None = the FedAvg path)."""
        t0 = time.perf_counter()
        if not clients:
            raise EmptyRoundError("empty round: no clients selected")
        keys = [c.key for c in clients]
        assert_unique_keys(keys)
        if pf is not None and store is None:
            raise ValueError("PF needs a PersonalStore")
        if dp is not None and (pf is not None or mu != 0):
            raise ValueError("DP-FedAvg is registered for FA only (no PF state, mu = 0)")
        channel = channel or VirtualChannel(server.manifest)
        theta_r = {k: v.cpu() for k, v in server.broadcast().items()}
        report = RoundReport(round_idx, method_name(mu, pf), len(clients), 0, 0)
        futs = []
        for s, positions in enumerate(plan_shards(len(clients), n_shards)):
            shard_clients = [clients[p] for p in positions]
            for c in shard_clients:
                report.bytes_down += channel.download(round_idx, c.key)
            personals = {c.key: store.get(c.key) for c in shard_clients} if pf is not None else {}
            slot = (s + rotation) % len(self.execs) if self.devices else 0
            futs.append(self.execs[slot].submit(_shard_job, self.specs[slot], s, positions, shard_clients, theta_r,
                                                solver, round_idx, seed, mu, pf, personals, fault, max_attempts,
                                                *((dp,) if dp is not None else ())))
        _done, _ = wait(futs, return_when=FIRST_EXCEPTION)
        for f in futs:
            if f.done() and f.exception() is not None:
                for g in futs:
                    g.cancel()
                e = f.exception()
                if isinstance(e, RoundAborted):
                    raise e
                remote = str(getattr(e, "__cause__", "") or "")           # the worker's traceback, verbatim
                raise RoundAborted(f"shard job failed: {e!r} | remote traceback: {remote[-4000:]}")
        outs = [f.result() for f in futs]
        self.completion_orders.append([o["acc"].shard_index for o in sorted(outs, key=lambda o: o["t_done"])])
        arrival = sorted(outs, key=lambda o: o["t_done"])                  # arrival order is deliberately used
        shards = [o["acc"] for o in arrival]
        agg = (combine_shards(shards) if dp is None else                  # ... and re-ordered by shard index here
               combine_dp_shards(shards))
        visits = sorted((v for o in outs for v in o["visits"]), key=lambda v: v.position)
        for v in visits:
            report.bytes_up += channel.upload_counted(round_idx, v.key)
            for _ in range(v.attempts - 1):
                report.retry_bytes_down += channel.download(round_idx, v.key, retry=True)
        if dp is None:
            new_state = finalize(agg, server.manifest, theta_r, expected_keys=keys)
        else:
            new_state, report.dp = finalize_dp(agg, server.manifest, theta_r, dp, seed=seed, round_idx=round_idx,
                                               n_sampled=len(clients) if n_sampled is None else int(n_sampled),
                                               expected_keys=keys)
        server.apply(new_state)
        if pf is not None:
            for o in outs:
                for k, st in o["staged"].items():
                    store.commit(k, st)
        self.pids_seen.update(o["pid"] for o in outs)
        for o in outs:
            self.warnings.update(o["warnings"])
        self.job_devices.append({o["acc"].shard_index: o["device"] for o in outs})
        report.visits = visits
        report.n_noop, report.weight = agg.n_noop, agg.weight
        report.wall_s = time.perf_counter() - t0
        report.state_digest = server.digest()
        return report
