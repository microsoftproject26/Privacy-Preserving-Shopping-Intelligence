"""v4int.e4_secagg_sim — E4 secure-aggregation simulation (PREG2 §7.4; task P2-14b). CPU only, deterministic per seed.

  python -B code/v4int/e4_secagg_sim.py --out <dir>/E4_SECAGG.json [--update-dim 265162 [...] | --dims-json <file>]
         [--seeds 2026 2027 2028] [--clip 1.0] [--n 64] [--k-values 16 32] [--dropout 0 0.1 0.3]

Grid: {full, k-regular (each k in --k-values)} x p in {0, 0.1, 0.3} x seeds x update lengths. Each cell is one round of
fl.secagg (cryptography simulated, sizes counted at real values), with synthetic L2-clipped updates of the real W*
upload length (the dimensions are inputs: the registered text value is 265,162 uploaded parameters, to be re-measured
from fl.comm.visit_bytes(...)["uploaded_numel"] of the W* manifest; pass the measured values with --update-dim or
--dims-json {"name": numel, ...}). Per cell: bytes per client per round and the overhead ratio vs the plain upload,
communication rounds, server unmasking CPU time, success / failure under dropout, and the bit-identity check of the
decoded sum against the plain fixed-point sum of the survivors.
Not run here (declared): the quality TOST on CALIB and the optional MAIN s2026 arm (Q13).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from fl import secagg as sa  # noqa: E402

DEFAULT_DIMS = {"W_STAR_registered_text": 265162}
SPEC_TEXT = ("PREG2 7.4 E4 (Q13): Bonawitz et al. 2017 with the SecAgg+ k-regular option (Bell et al. 2020); cryptography "
             "simulated, sizes counted at real values; fixed-point mod 2^32 with a common scale from the clip bound; "
             "dropout p in {0, 0.1, 0.3}, n = 64, threshold ceil(2n/3); metrics: bytes per client per round (overhead "
             "ratio), communication rounds, unmasking CPU time, success under dropout; criterion: the decoded sum is "
             "bit-identical to the plain fixed-point sum. Quality TOST on CALIB and optional MAIN s2026 (Q13): NOT RUN.")


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def synthetic_encoder(cfg: sa.SecAggConfig, scale: float, seed: int):
    """enc_fn(i): a seeded Gaussian update, L2 norm <= clip (a quarter of the clients sit exactly at the bound)."""
    def enc(i: int) -> np.ndarray:
        g = np.random.default_rng([int(seed), int(i), cfg.d])
        x = g.standard_normal(cfg.d)
        target = cfg.clip if i % 4 == 0 else cfg.clip * float(g.uniform(0.3, 1.0))
        x *= target / max(float(np.linalg.norm(x)), 1e-30)
        return sa.encode_fixed(x, scale, cfg.clip)
    return enc


def run_cell(n: int, d: int, k, p: float, seed: int, clip: float) -> dict:
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
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="E4_SECAGG.json path (refuses to overwrite)")
    ap.add_argument("--update-dim", type=int, nargs="*", default=None)
    ap.add_argument("--dims-json", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    ap.add_argument("--dropout", type=float, nargs="+", default=[0.0, 0.1, 0.3])
    ap.add_argument("--k-values", type=int, nargs="+", default=[16, 32])
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--clip", type=float, default=1.0)
    a = ap.parse_args(argv)
    dims = {}
    if a.dims_json:
        dims.update({str(k): int(v) for k, v in json.loads(Path(a.dims_json).read_text()).items()})
    if a.update_dim:
        dims.update({f"d{v}": int(v) for v in a.update_dim})
    if not dims:
        dims = dict(DEFAULT_DIMS)
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    cells = []
    for name, d in dims.items():
        for k in [None] + list(a.k_values):
            for p in a.dropout:
                for seed in a.seeds:
                    c = run_cell(a.n, d, k, p, seed, a.clip)
                    c["dim_name"] = name
                    cells.append(c)
                    print(f"{name} d={d} {c['variant']} p={p} seed={seed} ok={c['success']} "
                          f"bit_identical={c['bit_identical']} unmask={c['unmask_cpu_s']:.2f}s "
                          f"x{c['overhead_ratio_up']:.3f}", flush=True)
    bad = [c for c in cells if c["success"] and not c["bit_identical"]]
    me = Path(__file__).resolve()
    doc = {"experiment": "E4_SECAGG", "spec_text": SPEC_TEXT, "protocol": sa.PROTOCOL,
           "not_run": ["quality TOST on CALIB (90 % CI in [99, 101] %)", "optional MAIN s2026 arm (Q13)"],
           "grid": {"dims": dims, "k_values": a.k_values, "dropout": a.dropout, "seeds": a.seeds, "n": a.n,
                    "clip": a.clip, "threshold_t": sa.threshold(a.n)},
           "sizes": {"key_bytes": sa.KEY_BYTES, "share_bytes": sa.SHARE_BYTES, "ciphertext_bytes": sa.CT_BYTES,
                     "unmask_entry_bytes": sa.UNMASK_ENTRY_BYTES, "header_bytes": sa.HEADER_BYTES},
           "cells": cells, "n_cells": len(cells), "n_bit_identity_violations": len(bad),
           "environment": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
                           "cpu_count": os.cpu_count(), "device": "cpu"},
           "self_sha256": {"e4_secagg_sim.py": sha256_file(me), "fl/secagg.py": sha256_file(CODE / "fl" / "secagg.py")}}
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
    print(f"wrote {out}; cells {len(cells)}; bit-identity violations {len(bad)}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
