"""bench_store.py — build-time / RAM / batch-latency measurements for CompactStore.

Sonnet A, Wave 1. Run on Lobot, CPU-only (`CUDA_VISIBLE_DEVICES=""`), <=3 threads. Measures,
for both F4 and F0 stores built over splits=(TRAIN, VALIDATION_SELECT):
  - build wall-clock and peak RSS (via psutil, sampled around the build call; also reports
    resource.getrusage ru_maxrss as a second, OS-level source of truth),
  - batch() latency for B=256 random TRAIN positions (warm, repeated) and B=1024 sequential
    VALIDATION_SELECT positions,
  - decision_item_tokens() time over the full TRAIN loss DecisionSet.

Writes JSON to stdout and to --out ($W/tests_full/A/bench_store.json).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("MKL_NUM_THREADS", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(3)

from full_store import CompactStore, configure_threads  # noqa: E402

configure_threads(3)


def rss_mb() -> float:
    import psutil
    return psutil.Process().memory_info().rss / (1024.0 * 1024.0)


def ru_maxrss_mb() -> float:
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def bench_channels(root: Path, channels: str, seed: int) -> dict:
    gc.collect()
    rss_before = rss_mb()
    t0 = time.time()
    store = CompactStore.build(root, splits=("TRAIN", "VALIDATION_SELECT"), channels=channels,
                                buckets=None, context_length=50, n_threads=3, log=print)
    build_seconds = time.time() - t0
    gc.collect()
    rss_after = rss_mb()
    maxrss_after = ru_maxrss_mb()

    train_ds = store.decisions("TRAIN", "loss")
    vs_ds = store.decisions("VALIDATION_SELECT", "eval")

    rng = np.random.default_rng(seed)
    warm_positions = rng.integers(0, len(train_ds), size=256)
    # warm-up call (page in / JIT-ish warmup for numpy fancy indexing), excluded from timing
    _ = store.batch(train_ds, warm_positions)
    n_repeat = 20
    t0 = time.time()
    for _ in range(n_repeat):
        _ = store.batch(train_ds, warm_positions)
    warm_256_s = (time.time() - t0) / n_repeat

    seq_positions = np.arange(min(1024, len(vs_ds)), dtype=np.int64)
    _ = store.batch(vs_ds, seq_positions)
    t0 = time.time()
    for _ in range(n_repeat):
        _ = store.batch(vs_ds, seq_positions)
    seq_1024_s = (time.time() - t0) / n_repeat

    t0 = time.time()
    _ = store.decision_item_tokens(train_ds)
    item_tokens_all_train_s = time.time() - t0

    result = {
        "channels": channels,
        "n_rows": store.n_rows,
        "n_buckets": len(store.buckets),
        "mask_dtype": store._masks_dtype,
        "build_seconds": build_seconds,
        "rss_mb_before_build": rss_before,
        "rss_mb_after_build": rss_after,
        "rss_mb_delta": rss_after - rss_before,
        "ru_maxrss_mb_after_build": maxrss_after,
        "train_loss_decisions": len(train_ds),
        "vs_eval_decisions": len(vs_ds),
        "batch_warm_256_train_s": warm_256_s,
        "batch_seq_1024_vs_s": seq_1024_s,
        "decision_item_tokens_all_train_s": item_tokens_all_train_s,
    }
    del store, train_ds, vs_ds
    gc.collect()
    return result


def _main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="/home/jovyan/shared_data/rees46_nextitem_v1")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--skip-f4", action="store_true")
    ap.add_argument("--skip-f0", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    out = {"dataset_root": str(root), "seed": args.seed}

    if not args.skip_f4:
        print("=== F4 bench ===", flush=True)
        out["F4"] = bench_channels(root, "F4", args.seed)
    if not args.skip_f0:
        print("=== F0 bench ===", flush=True)
        out["F0"] = bench_channels(root, "F0", args.seed)

    print(json.dumps(out, indent=2))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
