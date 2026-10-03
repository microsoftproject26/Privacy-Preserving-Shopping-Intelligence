"""Hash definitions used to bind predictions, populations and resample matrices to each other.

  ordered_decision_id_sha256        sha256 of the int64 little-endian decision_id column, manifest order
  eval_/rankable_ordered_..._sha256 the same restricted to eval_mask / rankable rows (the query order hash)
  population_sha256                 sha256 of each selected user's 8-byte big-endian selection_key, ascending order
  user_set_sha256                   sha256 of the 32-byte user digests (client_key_hex decoded), ascending order
  candidate_order_hash              per output class ascending: int64 LE class, uint32 LE byte length, UTF-8 product_id
  W_sha256                          sha256 of the int32 little-endian row-major bootstrap resample matrix
Never Python hash(), never delimiter-joined strings.
"""
from __future__ import annotations

import hashlib
import struct
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ordered_decision_id_sha256(decision_id: np.ndarray, mask: np.ndarray | None = None) -> str:
    d = np.asarray(decision_id)
    if mask is not None:
        d = d[np.asarray(mask, dtype=bool)]
    return sha256_hex(np.ascontiguousarray(d, dtype="<i8").tobytes())


def population_sha256(selection_keys: np.ndarray) -> str:
    k = np.sort(np.asarray(selection_keys, dtype=np.uint64))
    return sha256_hex(k.astype(">u8").tobytes())


def user_set_sha256(client_key_hex: Iterable[str]) -> str:
    keys = sorted(client_key_hex)
    for k in keys:
        if len(k) != 64 or any(c not in "0123456789abcdef" for c in k):
            raise ValueError("client_key_hex must be 64 lowercase hex characters")
    return sha256_hex(b"".join(bytes.fromhex(k) for k in keys))


def candidate_order_hash(product_ids: Sequence[str]) -> str:
    h = hashlib.sha256()
    for cls, pid in enumerate(product_ids):
        b = pid.encode("utf-8")
        h.update(struct.pack("<q", cls))
        h.update(struct.pack("<I", len(b)))
        h.update(b)
    return h.hexdigest()


def matrix_sha256_int32(W: np.ndarray) -> str:
    return sha256_hex(np.ascontiguousarray(W, dtype="<i4").tobytes())


def code_sha256(files: Iterable[Path]) -> str:
    """sha256 over (file name, file bytes) in sorted-name order."""
    h = hashlib.sha256()
    for p in sorted(Path(f) for f in files):
        name = p.name.encode("utf-8")
        h.update(struct.pack("<I", len(name)))
        h.update(name)
        data = p.read_bytes()
        h.update(struct.pack("<Q", len(data)))
        h.update(data)
    return h.hexdigest()
