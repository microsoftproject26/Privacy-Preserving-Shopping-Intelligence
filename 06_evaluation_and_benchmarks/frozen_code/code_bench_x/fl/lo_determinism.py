"""fl.lo_determinism — the LO determinism receipt over 1,024 synthetic clients (Fable G0 blocker 7; T31).

Runs the procedural LO recipe (theta0 -> 6 passes, persistent AdamW, B = 16, evaluation hook at 2 / 4 / 6 passes,
discard) for every synthetic client on ONE reused worker model, and records per client the final-state sha256 and the
evaluation values (MRR@20 and summed CE on that client's own synthetic query rows). A second run — in a fresh
interpreter, with a different PYTHONHASHSEED and the clients processed in REVERSE order — must reproduce every record
bit-for-bit (CPU, strict FP32, deterministic algorithms, fixed intra-op thread count).

  python -B -m fl.lo_determinism --order reverse --out <json>      (cwd or PYTHONPATH = the ppsi_v4 code directory)

Synthetic inputs only: no dataset row is read. Tolerance: exact (bitwise) on this CPU build; a GPU replay needs its
own receipt (PENDING — cuBLAS/cuDNN determinism settings on the server).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .client import ClientData, LocalSolver
from .local_only import LORecipe, run_local_only
from .numerics import enforce_strict_fp32, state_digest
from .synthetic import census_like_sizes, client_examples, make_tiny_adapter

K, D, LW = 24, 8, 6


def build_population(n_clients: int, seed: int):
    """Census-like sizes; every 16th client has only invalid rows (a no-label user: no update, theta0 predictor)."""
    rng = np.random.default_rng(seed)
    sizes = census_like_sizes(n_clients, rng)
    clients, queries = [], {}
    for i, n in enumerate(sizes):
        pref = rng.dirichlet(np.full(K, 0.3))
        key = f"lo-user-{i:05d}"
        no_label = i % 16 == 15
        ex = client_examples(int(n), K, LW, rng, pref=pref, invalid_frac=1.0 if no_label else 0.1)
        clients.append(ClientData(key, ex))
        queries[key] = client_examples(4, K, LW, rng, pref=pref)
    return clients, queries


def make_eval_hook(q: dict):
    """MRR@20 (strict-greater rank, full K) and summed CE on the client's own synthetic query rows."""
    def hook(adapter, budget: int) -> dict:
        logits = adapter.scores(q)
        tgt = q["target_class"]
        tscore = logits.gather(1, tgt[:, None])
        rank = (logits > tscore).sum(1) + 1
        mrr = torch.where(rank <= 20, 1.0 / rank.to(torch.float64), torch.zeros_like(rank, dtype=torch.float64))
        ce = torch.nn.functional.cross_entropy(logits, tgt, reduction="sum")
        return {"mrr20": float(mrr.mean()), "ce_sum": float(ce)}
    return hook


def run_receipt(n_clients: int = 1024, seed: int = 2026, order: str = "forward", threads: int = 1) -> dict:
    torch.set_num_threads(threads)
    num = enforce_strict_fp32()
    t0 = time.perf_counter()
    adapter = make_tiny_adapter(K=K, d=D, tied=False, dropout=0.1, seed=seed)       # untied + recentering
    theta0 = adapter.broadcast_state(clone=True)
    recipe = LORecipe(theta0_digest=state_digest(theta0), seed=seed, passes=6, eval_at=(2, 4, 6),
                      solver=LocalSolver(lr=5e-3, passes=6, batch_size=16, clip=1.0, weight_decay=1e-5))
    clients, queries = build_population(n_clients, seed + 1)
    seq = clients if order == "forward" else list(reversed(clients))
    per = {}
    steps = 0
    for c in seq:
        r = run_local_only(adapter, theta0, c, recipe, eval_hook=make_eval_hook(queries[c.key]))
        steps += r.steps
        per[c.key] = {"final": r.final_digest, "n_valid": r.n_valid, "n_consumed": r.n_consumed, "steps": r.steps,
                      "no_label": r.no_label, "evals": {str(k): v for k, v in sorted(r.evals.items())}}
    blob = json.dumps({k: per[k] for k in sorted(per)}, sort_keys=True).encode()
    return {"what": "LO determinism receipt (synthetic, CPU)", "n_clients": n_clients, "seed": seed, "order": order,
            "recipe_digest": recipe.digest(), "theta0_digest": recipe.theta0_digest, "numerics": num,
            "python": sys.version.split()[0], "combined_digest": hashlib.sha256(blob).hexdigest(),
            "n_no_label": sum(1 for v in per.values() if v["no_label"]), "steps_total": steps,
            "wall_s": time.perf_counter() - t0, "per_client": per}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--order", choices=("forward", "reverse"), default="forward")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = run_receipt(a.n, a.seed, a.order, a.threads)
    Path(a.out).write_text(json.dumps(res, indent=1, sort_keys=True))
    print(json.dumps({k: res[k] for k in ("n_clients", "order", "combined_digest", "steps_total", "wall_s")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
