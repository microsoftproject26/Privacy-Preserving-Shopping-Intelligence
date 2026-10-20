"""Projected reads of the decision index.

`DecisionWindowLoader.index()` (vendored, unmodified, in
``vendor/rees46_eda/loaders.py``) reads EVERY column of EVERY requested
partition into memory. Over TRAIN that is ~52.76M rows x 19 columns, which
is the memory-blowup risk T-G exists to avoid. Every function in this module
instead reads only the caller-requested columns via pyarrow column
projection at the parquet level, and NEVER calls ``loader.index()``.

Data-only. No model code, no training loop.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Mirrors DecisionWindowLoader.LAYOUTS's index path, duplicated (not
# imported) so this module has no import-time coupling to the vendored
# loader and can be used standalone for pure index inspection.
INDEX_DIR_OF_LAYOUT = {"live": "indices/decisions", "frozen": "decisions_index"}


def index_root(dataset_root: Path, layout: str) -> Path:
    dataset_root = Path(dataset_root)
    if layout not in INDEX_DIR_OF_LAYOUT:
        raise ValueError(f"unknown layout {layout!r}; expected one of {sorted(INDEX_DIR_OF_LAYOUT)}")
    return dataset_root.joinpath(*INDEX_DIR_OF_LAYOUT[layout].split("/"))


def bucket_dirs(dataset_root: Path, layout: str, split: str,
                 buckets: Optional[Iterable[int]] = None) -> list[Path]:
    """Bucket partition directories for one split, in the same shape
    ``DecisionWindowLoader.index()`` would resolve them (explicit path
    addressing, no Hive partition filter)."""
    root = index_root(dataset_root, layout) / f"split={split}"
    if not root.exists():
        raise FileNotFoundError(f"no decision index for split {split}: {root}")
    if buckets is None:
        return sorted(root.glob("user_bucket=*"))
    return [root / f"user_bucket={int(b)}" for b in buckets]


def read_index_projected(dataset_root: Path, layout: str, split: str,
                          columns: list[str],
                          buckets: Optional[Iterable[int]] = None) -> pa.Table:
    """Read only `columns` (plus a derived `user_bucket`) for one split.

    Column projection happens inside ``pq.read_table(..., columns=...)``, so
    reading e.g. just ``decision_id`` over all of TRAIN costs a small
    fraction of what ``loader.index()``'s full 19-column read would.
    ``user_bucket`` is not a stored column (it is the partition directory
    name); it is appended here exactly as ``DecisionWindowLoader.index()``
    does, so callers get an identical logical shape.
    """
    parts = []
    for d in bucket_dirs(dataset_root, layout, split, buckets):
        files = sorted(d.glob("*.parquet"))
        if not files:
            continue
        b = int(d.name.split("=", 1)[1])
        t = pq.read_table(files, columns=columns)
        if "user_bucket" in t.column_names:
            t = t.drop(["user_bucket"])
        t = t.append_column("user_bucket", pa.array(np.full(t.num_rows, b, np.int64)))
        parts.append(t)
    if not parts:
        raise FileNotFoundError(
            f"no index rows for split={split} buckets={buckets} under {index_root(dataset_root, layout)}")
    return pa.concat_tables(parts)


def all_bucket_ids(dataset_root: Path, layout: str, split: str) -> list[int]:
    """Every user_bucket present for a split, without reading any row data."""
    return sorted(int(d.name.split("=", 1)[1])
                  for d in bucket_dirs(dataset_root, layout, split))
