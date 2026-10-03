"""Measures ONE onnxruntime session's peak RSS in an isolated subprocess.

`resource.getrusage(RUSAGE_SELF)` on the whole bench process would conflate the FP32 session, the INT8 session, torch
and everything else the parent holds into a single high-water mark. A fresh subprocess loads exactly one ONNX model,
runs the given queries through it, and reports ITS OWN peak; nothing else has run in that process.

Invoked only as `python -m ppsi.ondevice._rss_worker --onnx PATH --feed FEED.npz --intra-threads N [--warmup N]`
(the caller is `bench.bench_peak_rss_subprocess`). `--feed` is an NPZ of stacked per-key arrays (leading dim =
number of rows) restricted to the ONNX graph's own input names. Prints one JSON line `{"peak_rss_bytes": ...}`.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

import numpy as np


def run_worker(onnx_path: str, feed_npz: str, *, intra_op_num_threads: int, warmup: int) -> dict:
    from .bench import make_session, peak_rss_bytes
    data = np.load(feed_npz)
    names = list(data.files)
    if not names:
        raise ValueError(f"{feed_npz}: no arrays (empty feed)")
    n = int(data[names[0]].shape[0])
    rows = [{name: data[name][i:i + 1] for name in names} for i in range(n)]
    session = make_session(onnx_path, intra_op_num_threads=intra_op_num_threads)
    for i in range(int(warmup)):
        session.run(None, rows[i % len(rows)])
    for row in rows:
        session.run(None, row)
    return {"peak_rss_bytes": peak_rss_bytes()}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="measure one onnxruntime session's peak RSS in an isolated subprocess")
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--feed", required=True, help="NPZ of stacked per-row feed arrays (leading dim = n rows)")
    ap.add_argument("--intra-threads", type=int, default=1, dest="intra_threads")
    ap.add_argument("--warmup", type=int, default=50)
    args = ap.parse_args(argv)
    print(json.dumps(run_worker(args.onnx, args.feed, intra_op_num_threads=args.intra_threads,
                                warmup=args.warmup)))
    return 0


__all__ = ["main", "run_worker"]

if __name__ == "__main__":
    raise SystemExit(main())
