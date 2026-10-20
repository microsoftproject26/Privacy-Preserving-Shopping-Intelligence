"""sha256 file hashing for prerequisite gate files."""
from __future__ import annotations

from pathlib import Path
import hashlib

_CHUNK = 1 << 20  # 1 MiB


def sha256_file(path: "Path | str") -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()
