"""Secure-aggregation cost and correctness grid (CPU only, deterministic per seed).

    python -m ppsi.fedsim.secagg_grid --out secagg.json [--update-dim 263822 ...] [--seeds 2026 2027 2028]
        [--dropout 0 0.1 0.3] [--k-values 16 32] [--n 64] [--clip 1.0]

Grid: {complete graph, each k-regular graph} x dropout rates x seeds x update lengths. Each cell is one round of
`ppsi.fedsim.secagg` with synthetic L2-clipped updates of the given length (pass the uploaded parameter count of the
model you deploy, e.g. `ppsi.fedsim.comm.visit_bytes(...)["uploaded_numel"]`). Per cell it reports the bytes per client
per round and the overhead ratio against a plain upload, the number of communication rounds, the server's unmasking
CPU time, success or failure under dropout, and whether the decoded sum is bit-identical to the plain fixed-point sum
of the surviving clients. The command exits 1 if any decoded sum differs.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np

from . import secagg as sa

DEFAULT_DIM = 263822          # the encoder of the device-size SASRec (item tables frozen)


def synthetic_encoder(cfg: sa.SecAggConfig, scale: float, seed: int):
    """enc_fn(i): a seeded Gaussian update with L2 norm <= clip (every fourth client sits exactly at the bound)."""
    def enc(i: int) -> np.ndarray:
        g = np.random.default_rng([int(seed), int(i), cfg.d])
        x = g.standard_normal(cfg.d)
        target = cfg.clip if i % 4 == 0 else cfg.clip * float(g.uniform(0.3, 1.0))
        x *= target / max(float(np.linalg.norm(x)), 1e-30)
        return sa.encode_fixed(x, scale, cfg.clip)
    return enc


def run_cell(n: int, d: int, k: int | None, p: float, seed: int, clip: float) -> dict:
    cfg = sa.SecAggConfig(n=n, d=d, k=k, clip=clip, seed=seed)
    scale = sa.fixed_point_scale(clip, n)
    drop = sa.sample_dropped(n, p, seed)
    res = sa.run_round(cfg, synthetic_encoder(cfg, scale, seed), drop)
    out = {"variant": cfg.variant, "k": None if cfg.full else int(k), "d": d, "n": n, "p": p, "seed": seed,
           "t": cfg.t_global, "t_share": cfg.t_share, "n_dropped": len(drop), "n_survivors": len(res.survivors),
           "success": res.success, "failure_reason": res.failure_reason, "bit_identical": res.bit_identical,
           "fixed_point_scale": scale, "unmask_cpu_s": res.unmask_cpu_s, "client_sim_cpu_s": res.client_cpu_s,
           "prg_words_server": res.prg_words_server}
    out.update(sa.cost_summary(cfg, res))
    if res.success:
        ref = sa.decode_fixed(res.plain, scale)
        out["decoded_max_abs"] = float(np.max(np.abs(sa.decode_fixed(res.total, scale) - ref)))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Secure-aggregation cost and correctness grid.")
    ap.add_argument("--out", required=True, help="output JSON path (never overwritten)")
    ap.add_argument("--update-dim", type=int, nargs="+", default=[DEFAULT_DIM])
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    ap.add_argument("--dropout", type=float, nargs="+", default=[0.0, 0.1, 0.3])
    ap.add_argument("--k-values", type=int, nargs="*", default=[16, 32])
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--clip", type=float, default=1.0)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    cells = []
    for d in a.update_dim:
        for k in [None, *a.k_values]:
            for p in a.dropout:
                for seed in a.seeds:
                    c = run_cell(a.n, d, k, p, seed, a.clip)
                    cells.append(c)
                    print(f"d={d} {c['variant']} p={p} seed={seed} ok={c['success']} "
                          f"bit_identical={c['bit_identical']} unmask={c['unmask_cpu_s']:.2f}s "
                          f"x{c['overhead_ratio_up']:.3f}", flush=True)
    bad = [c for c in cells if c["success"] and not c["bit_identical"]]
    doc = {"protocol": sa.PROTOCOL,
           "grid": {"update_dims": a.update_dim, "k_values": a.k_values, "dropout": a.dropout, "seeds": a.seeds,
                    "n": a.n, "clip": a.clip, "threshold_t": sa.threshold(a.n)},
           "sizes": {"key_bytes": sa.KEY_BYTES, "share_bytes": sa.SHARE_BYTES, "ciphertext_bytes": sa.CT_BYTES,
                     "unmask_entry_bytes": sa.UNMASK_ENTRY_BYTES, "header_bytes": sa.HEADER_BYTES},
           "cells": cells, "n_cells": len(cells), "n_failed": sum(not c["success"] for c in cells),
           "n_bit_identity_violations": len(bad),
           "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                           "platform": platform.platform(), "cpu_count": os.cpu_count()}}
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
    print(f"wrote {out}: {len(cells)} cells, {doc['n_failed']} failed to decode, {len(bad)} bit-identity violations")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
