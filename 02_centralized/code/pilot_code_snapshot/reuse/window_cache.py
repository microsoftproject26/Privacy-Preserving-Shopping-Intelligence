"""Bucket-major memmapped window cache (Task T-H).

Why this exists (see task brief): the vendored, unmodified `BucketStore`
(``vendor/rees46_eda/loaders.py``) reads EVERY column of a user-bucket's
feature file into RAM -- including several string columns as boxed Python
object arrays -- and `DecisionWindowLoader._buckets` is never evicted. With
~1.17M rows/bucket x 64 buckets that would exhaust RAM if every bucket were
ever touched and kept. This module builds a flat, memmapped, bucket-major
cache of ``DecisionWindowLoader.collate()``'s output at a fixed context
length L=50, processing ONE bucket at a time and evicting it
(``loader._buckets.pop(b, None)``) the moment its rows are written, so peak
RSS stays bounded regardless of how many buckets (and rows) are cached in
total. It also removes per-decision Python collate from the training hot
path: reading is just a fancy-index gather over memmapped ``.npy`` files.

Data-only. No model code. No edits to the vendored loader.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

CODE_DIR = Path(__file__).resolve().parent
VENDOR_DIR = CODE_DIR / "vendor"
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from rees46_eda.loaders import DecisionWindowLoader, detect_layout  # noqa: E402
from rees46_eda.config import (  # noqa: E402
    EVENT_QUALITY_FLAGS, USER_CONTEXT_FEATURES, USER_CONTEXT_MASKS,
)
from rees46_eda.loaders import EVENT_NUMERIC_BLOCK  # noqa: E402
from index_utils import read_index_projected  # noqa: E402

L = 50  # frozen context length for this cache (task brief)

F_NUMERIC = len(EVENT_NUMERIC_BLOCK)          # 19
Q_FLAGS = len(EVENT_QUALITY_FLAGS)            # 15
U_CTX = len(USER_CONTEXT_FEATURES)            # 12
M_CTX = len(USER_CONTEXT_MASKS)               # 6

# Meta columns pulled from the decision index per bucket to feed
# `DecisionWindowLoader.collate()` (the manifest itself carries only
# `decision_id`/`user_bucket`, sometimes plus cohort-only columns that are
# NOT what `collate()` needs).
META_COLUMNS = ["decision_id", "target_class", "loss_mask", "eval_mask", "client_idx"]

# Ordered spec: name -> (per-row shape (without the leading N), dtype).
# Mirrors DecisionWindowLoader.collate()'s batch schema at L=50, restricted
# to exactly the channels the task brief names (model_segment_id is dropped
# -- not requested). Token/index channels are stored int32 (cast back to
# int64 at load time) EXCEPT `decision_id`: measured directly against the
# frozen dataset, decision_id is NOT a small sequential counter -- it ranges
# up to ~1.0e11 (100,032,591,289 observed on TRAIN bucket 0 alone; 432,143 of
# 1,000,000 TRAIN-manifest rows, 43.2%, exceed 2^31-1). The task brief's
# int32 premise ("all indices < 2^31") does not hold for this one field, so
# it is kept int64 end-to-end rather than silently truncated/wrapped -- an
# int32 cast here was caught by the T3 parity test (decision_id mismatch)
# before this fix. Every OTHER field kept in this cache (target_class <=
# 158485, client_idx <= ~4.2M, all vocab-bounded categorical channels,
# lengths/last_valid_index/position_ids <= L=50) was checked against the
# frozen dataset and does fit int32. Numerics stay float32, quality flags
# int8 (as collate() already produces), masks bool.
ARRAY_SPECS: "dict[str, tuple[tuple[int, ...], type]]" = {
    "item_tokens": ((L,), np.int32),
    "category_tokens": ((L,), np.int32),
    "brand_tokens": ((L,), np.int32),
    "main_category_tokens": ((L,), np.int32),
    "daypart_tokens": ((L,), np.int32),
    "weekday_tokens": ((L,), np.int32),
    "position_ids": ((L,), np.int32),
    "attention_mask": ((L,), np.bool_),
    "event_numeric_features": ((L, F_NUMERIC), np.float32),
    "event_quality_flags": ((L, Q_FLAGS), np.int8),
    "user_context": ((U_CTX,), np.float32),
    "user_context_masks": ((M_CTX,), np.float32),
    "lengths": ((), np.int32),
    "last_valid_index": ((), np.int32),
    "target_class": ((), np.int32),
    "loss_mask": ((), np.bool_),
    "eval_mask": ((), np.bool_),
    "decision_id": ((), np.int64),  # NOT int32 -- see note above (exceeds 2^31)
    "client_id": ((), np.int32),
}
# collate() key -> cache array name (identical for every channel we keep;
# only "client_id" in the batch dict maps 1:1 to "client_id" here too).
ARRAY_NAMES = tuple(ARRAY_SPECS.keys())
INT32_NAMES = tuple(n for n, (_, dt) in ARRAY_SPECS.items() if dt == np.int32)


def process_rss_mb() -> float:
    """Current (not peak) resident set size of THIS process, in MiB."""
    import psutil
    return psutil.Process().memory_info().rss / (1024.0 * 1024.0)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_manifest(manifest_path: Path) -> tuple[np.ndarray, np.ndarray]:
    t = pq.read_table(manifest_path, columns=["decision_id", "user_bucket"])
    dec = t["decision_id"].to_numpy(zero_copy_only=False)
    buck = t["user_bucket"].to_numpy(zero_copy_only=False)
    return dec, buck


def fetch_bucket_meta(root: Path, layout: str, split: str, bucket: int) -> dict[int, tuple]:
    """decision_id -> (target_class, loss_mask, eval_mask, client_idx) for one
    bucket of one split, via projected columns only (never the full index).
    """
    t = read_index_projected(root, layout, split, META_COLUMNS, buckets=[bucket])
    dec = t["decision_id"].to_numpy(zero_copy_only=False)
    tgt = t["target_class"].to_numpy(zero_copy_only=False)
    lm = t["loss_mask"].to_numpy(zero_copy_only=False)
    em = t["eval_mask"].to_numpy(zero_copy_only=False)
    cli = t["client_idx"].to_numpy(zero_copy_only=False)
    return {int(d): (int(tg), bool(l), bool(e), int(c))
            for d, tg, l, e, c in zip(dec, tgt, lm, em, cli)}


def collate_rows(loader: DecisionWindowLoader, root: Path, layout: str, split: str,
                  bucket: int, decision_ids: Iterable[int]) -> dict[str, np.ndarray]:
    """Build the decision dicts `DecisionWindowLoader.collate()` expects for
    one bucket's worth of rows and collate them. This is the SAME path used
    by the cache builder and by the parity test's "independent" reference
    collation -- it only assembles collate()'s required per-decision
    metadata (target_class/loss_mask/eval_mask/client_idx) from the decision
    index; it never touches the cache itself.
    """
    meta = fetch_bucket_meta(root, layout, split, bucket)
    decisions = []
    for did in decision_ids:
        tgt, lm, em, cli = meta[int(did)]
        decisions.append({
            "user_bucket": bucket, "decision_id": int(did),
            "target_class": tgt, "loss_mask": lm, "eval_mask": em,
            "client_idx": cli,
        })
    return loader.collate(decisions)


def _open_memmaps(out_dir: Path, n: int, mode: str) -> dict[str, np.memmap]:
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for name, (shape, dtype) in ARRAY_SPECS.items():
        path = out_dir / f"{name}.npy"
        full_shape = (n,) + shape
        if mode == "w+":
            arrays[name] = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=full_shape)
        else:
            arrays[name] = np.load(path, mmap_mode=mode)
    return arrays


def build_cache(manifest_path: Path, root: Path, split: str, out_dir: Path,
                 version_hash: str, log_path: Optional[Path] = None) -> dict:
    """Build the bucket-major memmapped cache for `manifest_path` (a
    `decision_id, user_bucket` parquet) against the frozen dataset at `root`,
    for decision-index split `split` (TRAIN or VALIDATION_SELECT).

    Processes buckets one at a time, in ascending bucket-id order, evicting
    each bucket's `BucketStore` (`loader._buckets.pop(b, None)`) the moment
    its rows are written so at most one bucket's full feature columns are
    resident at once. Manifest row order is preserved: cache row i always
    corresponds to manifest row i, regardless of bucket processing order,
    because writes are placed by absolute manifest row index, not appended.
    """
    layout = detect_layout(root)
    dec_ids, buckets = _read_manifest(manifest_path)
    n = dec_ids.shape[0]

    order_by_bucket: dict[int, list[int]] = {}
    for row_i, b in enumerate(buckets):
        order_by_bucket.setdefault(int(b), []).append(row_i)

    cache = _open_memmaps(out_dir, n, mode="w+")
    loader = DecisionWindowLoader(root, L, layout=layout)

    log_lines = []
    bucket_log = []
    t_start = time.time()
    for b in sorted(order_by_bucket):
        row_idxs = np.asarray(order_by_bucket[b], dtype=np.int64)
        bucket_dec_ids = dec_ids[row_idxs]
        t0 = time.time()
        rss_before = process_rss_mb()

        batch = collate_rows(loader, root, layout, split, b, bucket_dec_ids)

        rss_after_load = process_rss_mb()

        for name in ARRAY_NAMES:
            src = batch[name]
            dtype = ARRAY_SPECS[name][1]
            cache[name][row_idxs] = src.astype(dtype, copy=False)

        loader._buckets.pop(b, None)
        gc.collect()
        rss_after_evict = process_rss_mb()
        dt = time.time() - t0

        line = (f"bucket={b} n_rows={row_idxs.shape[0]} "
                f"rss_before_mb={rss_before:.1f} rss_after_load_mb={rss_after_load:.1f} "
                f"rss_after_evict_mb={rss_after_evict:.1f} elapsed_s={dt:.2f}")
        print(line, flush=True)
        log_lines.append(line)
        bucket_log.append({
            "bucket": b, "n_rows": int(row_idxs.shape[0]),
            "rss_before_mb": rss_before, "rss_after_load_mb": rss_after_load,
            "rss_after_evict_mb": rss_after_evict, "elapsed_s": dt,
        })

    build_seconds = time.time() - t_start

    for name in ARRAY_NAMES:
        cache[name].flush()

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    # blake2b digest of cache contents, in sorted key order, chunked so no
    # single array is ever fully materialised at once (matches T-F's
    # digest convention: fold dtype+shape strings ahead of each array's
    # raw bytes, both per-key and into the overall digest).
    overall = hashlib.blake2b()
    per_key_digest = {}
    for name in sorted(ARRAY_NAMES):
        arr = cache[name]
        h = hashlib.blake2b()
        header = f"{name}|{arr.dtype}|{arr.shape}".encode()
        h.update(header)
        overall.update(header)
        step = max(1, (1 << 24) // max(1, arr[0:1].nbytes or 1))
        for i in range(0, arr.shape[0], step):
            chunk = np.ascontiguousarray(arr[i:i + step]).tobytes()
            h.update(chunk)
            overall.update(chunk)
        per_key_digest[name] = h.hexdigest()

    total_bytes = sum(cache[n].nbytes for n in ARRAY_NAMES)
    disk_bytes = sum((out_dir / f"{n}.npy").stat().st_size for n in ARRAY_NAMES)

    header = {
        "row_count": int(n),
        "context_length": L,
        "split": split,
        "dataset_root": str(root),
        "layout": layout,
        "version_hash": version_hash,
        "source_manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "arrays": {
            name: {"dtype": np.dtype(ARRAY_SPECS[name][1]).name, "shape": list(cache[name].shape)}
            for name in ARRAY_NAMES
        },
        "int32_cast_names": list(INT32_NAMES),
        "digest_blake2b": overall.hexdigest(),
        "per_array_digest_blake2b": per_key_digest,
        "n_buckets": len(order_by_bucket),
        "build_seconds": build_seconds,
        "bucket_log": bucket_log,
        "peak_rss_after_load_mb": max((r["rss_after_load_mb"] for r in bucket_log), default=None),
        "total_array_bytes": int(total_bytes),
        "total_disk_bytes": int(disk_bytes),
    }
    (out_dir / "header.json").write_text(json.dumps(header, indent=2), encoding="utf-8")
    return header


class CacheReader:
    """Reads rows out of a cache built by `build_cache`, as `[B, ...]` torch
    tensors. int32-stored channels are cast to int64 on load (per the task
    brief); every other channel keeps its stored dtype (bool stays bool,
    event_quality_flags stays int8, numerics stay float32).
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.header = json.loads((self.cache_dir / "header.json").read_text(encoding="utf-8"))
        self._arrays: dict[str, np.memmap] = {}

    def _arr(self, name: str) -> np.memmap:
        if name not in self._arrays:
            self._arrays[name] = np.load(self.cache_dir / f"{name}.npy", mmap_mode="r")
        return self._arrays[name]

    def read(self, row_indices) -> dict[str, "object"]:
        import torch
        idx = np.asarray(row_indices, dtype=np.int64)
        out = {}
        for name in self.header["arrays"]:
            arr = np.ascontiguousarray(self._arr(name)[idx])
            if name in self.header.get("int32_cast_names", INT32_NAMES):
                arr = arr.astype(np.int64)
            out[name] = torch.from_numpy(arr)
        return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", required=True, choices=["TRAIN", "VALIDATION_SELECT",
                                                        "VALIDATION_CONFIRM", "TEST"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--version-hash", required=True)
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    header = build_cache(
        Path(args.manifest), Path(args.root), args.split, Path(args.out),
        args.version_hash, Path(args.log) if args.log else None,
    )
    print(json.dumps({k: v for k, v in header.items() if k != "bucket_log"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
