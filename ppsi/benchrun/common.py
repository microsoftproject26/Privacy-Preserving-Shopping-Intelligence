"""Constants, refusals and small I/O helpers of the benchmark runner (no torch import here)."""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Mapping
from pathlib import Path

# Wave 1: the sequential-recommendation capacity-probe release; Wave 2: ppsi.benchmarks Amazon Reviews 2023 output.
DATASETS = ("s3_beauty", "s3_sports", "s3_toys", "ml_1m")
WAVE2_DATASETS = ("amazon2023_video_games", "amazon2023_baby_products", "amazon2023_beauty_and_personal_care")
ALL_DATASETS = DATASETS + WAVE2_DATASETS

# The pretraining band: a seeded permutation of the sorted external user ids; the first round(15 %) users are the
# users whose data the server may use for central pretraining (the warm start of the federated arms).
BAND_SEED = 2026
BAND_FRAC = 0.15
BAND_TAG = 0x42414E44                # SeedSequence entropy tag of the band draw (never shared with another stream)

MAX_LEN = 50                         # the context window of past events
PRIMARY_METRIC = "ndcg@10"           # selection metric on the inner validation split
METRICS = ("ndcg@10", "hr@10", "ndcg@20", "hr@20", "mrr@20")
GROUPS = ("ALL", "BAND", "OTHERS")


class BenchRefused(RuntimeError):
    """A refusal of the benchmark runner (configuration, provenance or an unsafe overwrite)."""


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def canon_sha256(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str,
                                     allow_nan=False).encode()).hexdigest()


def _safe(o):
    if isinstance(o, Mapping):
        return {str(k): _safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_safe(v) for v in o]
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "item") and callable(o.item) and getattr(o, "ndim", 1) == 0:
        return o.item()
    return o


def write_json(path, obj) -> None:
    """Atomic strict-JSON write (tmp + replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(_safe(obj), indent=1, sort_keys=True, allow_nan=False, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_exclusive(path, obj) -> None:
    """O_EXCL strict-JSON write: an existing file is never overwritten (write-once records)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(_safe(obj), indent=1, sort_keys=True, allow_nan=False, default=str) + "\n").encode()
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError:
        raise BenchRefused(f"{path} exists: write-once records are never overwritten") from None
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)


def append_jsonl(path, rec: Mapping) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(_safe(rec), sort_keys=True, allow_nan=False) + "\n")


def read_json(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def refuse_overwrite(run_dir) -> None:
    """A finished run (RESULT.json with a DONE_* / INCOMPLETE_* status) is never rerun or overwritten."""
    p = Path(run_dir) / "RESULT.json"
    if p.is_file():
        st = read_json(p).get("status", "")
        if str(st).startswith("DONE") or str(st).startswith("INCOMPLETE"):
            raise BenchRefused(f"{p} exists (status {st}): a finished run is never overwritten")


class WallCap:
    """An optional wall-clock cap in hours; a run that hits it is INCOMPLETE and not eligible for anything."""

    def __init__(self, hours):
        self.t0 = time.perf_counter()
        self.limit = None if not hours else float(hours) * 3600.0

    def hit(self) -> bool:
        return self.limit is not None and (time.perf_counter() - self.t0) > self.limit


def run_dir_for(runs_root, dataset: str, arm: str, seed: int, smoke: bool = False) -> Path:
    base = Path(runs_root) / dataset
    return (base / "SMOKE" if smoke else base) / f"{arm}_s{int(seed)}"


def check_dataset(name: str) -> str:
    if name not in ALL_DATASETS:
        raise BenchRefused(f"dataset {name!r} is not a benchmark dataset {ALL_DATASETS}")
    return name
