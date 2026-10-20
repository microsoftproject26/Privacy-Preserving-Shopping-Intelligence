"""fl.bench_visit — per-visit CPU micro-benchmark (FL-01 acceptance: no pathological copies). Synthetic inputs only.

  python -B -m fl.bench_visit --out <json> [--reps 50] [--threads 1] [--families]

For the tiny synthetic model (and, with --families, the snapshot GRU / SASRec at a small K override) it times, as
medians after warm-up: the theta_r load into the worker, a fresh AdamW, one full FA visit (load + optimizer + passes +
upload refs), one FP visit (mu = 0.01) and one PF visit, the streaming accumulation, and the per-step cost, for clients
of 6, 16, 35 and 100 valid decisions at B = 16 and 2 passes, and fits visit_time = fixed + per_step * steps: the fixed
per-visit overhead must stay small next to one step (one theta_r copy per visit, no per-step state copies).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .aggregate import ShardAccumulator
from .client import ClientData, LocalSolver, PFConfig, client_update, make_optimizer
from .numerics import enforce_strict_fp32
from .synthetic import client_examples, make_tiny_adapter


def _t(fn, reps: int, warm: int = 3) -> float:
    for _ in range(warm):
        fn()
    xs = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        xs.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(xs)


def bench_adapter(make, examples_fn, sizes, reps: int) -> dict:
    server, worker = make(1), make(2)
    theta = server.broadcast_state(clone=True)
    solver = LocalSolver(lr=1e-3, passes=2, batch_size=16)
    pf = PFConfig(lr=1e-3)
    out = {"shared_numel": server.manifest.shared_numel, "shared_bytes": server.manifest.shared_bytes,
           "t_load_theta_ms": _t(lambda: worker.load_state_(theta), reps),
           "t_fresh_adamw_ms": _t(lambda: make_optimizer(worker, solver), reps)}
    acc = ShardAccumulator.empty(0, server.manifest, theta)
    up = worker.extract_shared()
    pos = [0]

    def accumulate():
        pos[0] += 1
        acc.add(pos[0], f"k{pos[0]}", up, 7)
    out["t_accumulate_ms"] = _t(accumulate, reps)
    per = {}
    for n in sizes:
        c = ClientData("bench-user", examples_fn(n))
        r = client_update(worker, theta, c, solver, round_idx=0, seed=1)
        steps = r.steps
        t_fa = _t(lambda: client_update(worker, theta, c, solver, round_idx=0, seed=1), reps)
        t_fp = _t(lambda: client_update(worker, theta, c, solver, round_idx=0, seed=1, mu=0.01), reps)
        t_pf = _t(lambda: client_update(worker, theta, c, solver, round_idx=0, seed=1, pf=pf), reps)
        # single-step reference: a 16-row client with one pass = exactly one optimizer step
        one = ClientData("bench-one", examples_fn(16))
        s1 = LocalSolver(lr=1e-3, passes=1, batch_size=16)
        t_one = _t(lambda: client_update(worker, theta, one, s1, round_idx=0, seed=1), reps)
        per[str(n)] = {"steps": steps, "t_visit_fa_ms": t_fa, "t_visit_fp_ms": t_fp, "t_visit_pf_ms": t_pf,
                       "t_one_step_visit_ms": t_one}
    out["by_client_size"] = per
    # linear fit t_visit_FA = fixed + per_step * steps over the client sizes: `fixed` is the per-visit overhead
    # (theta_r load, fresh AdamW, RNG fork, row selection, upload refs) and must stay small next to per_step
    xs = np.array([v["steps"] for v in per.values()], dtype=float)
    ys = np.array([v["t_visit_fa_ms"] for v in per.values()], dtype=float)
    slope, icpt = np.polyfit(xs, ys, 1)
    out["fit_fa"] = {"per_step_ms": float(slope), "fixed_per_visit_ms": float(icpt),
                     "state_bytes_copied_per_visit": server.manifest.shared_bytes,
                     "note": "one theta_r copy into the worker per visit; the upload is by reference and is "
                             "streamed into the FP32 shard accumulator (one read) before the worker is reused"}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--families", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    num = enforce_strict_fp32()
    sizes = (6, 16, 35, 100)
    res = {"what": "FL-01 per-visit CPU micro-benchmark (synthetic)", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "numerics": num, "python": sys.version.split()[0], "reps": a.reps, "passes": 2, "local_batch": 16}
    rng = np.random.default_rng(0)
    res["tiny_tied"] = bench_adapter(lambda s: make_tiny_adapter(K=24, d=8, tied=True, seed=s),
                                     lambda n: client_examples(n, 24, 6, rng), sizes, a.reps)
    res["tiny_untied_recenter"] = bench_adapter(lambda s: make_tiny_adapter(K=24, d=8, tied=False, seed=s),
                                                lambda n: client_examples(n, 24, 6, rng), sizes, a.reps)
    if a.families:
        from .families import build_v2, synthetic_examples_v2
        g = torch.Generator().manual_seed(0)
        for fam in ("GRU", "SASREC"):                   # Clean V2 models (v4models), small K override
            res[f"{fam}_V2_K64"] = bench_adapter(lambda s, f=fam: build_v2(f, seed=s, K=64).adapter,
                                                 lambda n: synthetic_examples_v2(n, 64, g), sizes, max(5, a.reps // 5))
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: {n: round(v["t_visit_fa_ms"], 3) for n, v in res[k]["by_client_size"].items()}
                      for k in res if isinstance(res[k], dict) and "by_client_size" in res[k]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
