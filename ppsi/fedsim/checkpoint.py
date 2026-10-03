"""Atomic, validated checkpoints and the retention rules of a run directory.

Write protocol (`atomic_save`):
  1. disk guard: disk_check(dir, planned peak = the new file's bytes) must not REFUSE (the old file stays until the
     replace, so the temp file is the peak); PAUSE is recorded, not enforced (a live run's checkpoint is not a new
     run);
  2. write `<name>.tmp.<pid>`, flush, fsync;
  3. RELOAD the temp file (torch.load weights_only=True) and VALIDATE: format tag and content digest (sha256 over
     every tensor's raw bytes and every scalar, in canonical key order) must equal the in-memory digest;
  4. os.replace(temp, target) (atomic), then fsync the directory (POSIX);
  a crash anywhere before step 4 leaves the previous target untouched; a stale temp is removed (sha256-logged) by the
  next `CheckpointManager` start.
`load_checkpoint` re-checks format and digest; any mismatch or unreadable file raises CheckpointCorrupt (resume
refuses instead of continuing from a damaged state).

Retention (`CheckpointManager`):
  live run: `latest.pt` (resumable, replaced atomically) + `best.pt` (weights of the strictly-best evaluation) +
  `endpoint_6.0.pt` (weights at the controlled endpoint); `complete()` deletes `latest.pt`; a calibration run
  (retain_weights=False) deletes everything at completion. Only files this manager created are removed, each logged
  with its sha256 and size in `removal_log.jsonl`.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from .numerics import tensor_bytes

FORMAT = "PPSI_FL_CKPT_V1"
FLOOR_GIB = 8.0                 # REFUSE when free - planned peak < 8 GiB
TARGET_GIB = 10.0               # PAUSE when free < 10 GiB
GIB = 2 ** 30


class CheckpointError(RuntimeError):
    pass


class CheckpointCorrupt(CheckpointError):
    pass


class DiskRefused(CheckpointError):
    pass


class SimulatedCrash(RuntimeError):
    """Test-only: raised at a named point of the write protocol."""


CRASH_POINTS = ("after_tmp_write", "after_fsync", "after_validate", "after_replace")


@dataclass(frozen=True)
class DiskDecision:
    state: str  # "OK" | "PAUSE" | "REFUSE"
    free_gib: float
    reason: str


def disk_check(path, planned_peak_gib: float, disk_usage_fn: Callable = shutil.disk_usage,
               floor_gib: float = FLOOR_GIB, target_gib: float = TARGET_GIB) -> DiskDecision:
    """REFUSE if free - planned_peak_gib < floor_gib; PAUSE if free < target_gib; OK otherwise.

    REFUSE is checked first: it accounts for the disk this write plans to consume at its peak (its own atomic temp
    file). PAUSE is the plain free-space rule and applies even to a zero-size plan."""
    free = disk_usage_fn(str(path)).free / GIB
    headroom = free - planned_peak_gib
    if headroom < floor_gib:
        return DiskDecision("REFUSE", free, f"free {free:.2f} GiB - planned peak {planned_peak_gib:.2f} GiB "
                                             f"= headroom {headroom:.2f} GiB < floor {floor_gib} GiB")
    if free < target_gib:
        return DiskDecision("PAUSE", free, f"free {free:.2f} GiB < target {target_gib} GiB")
    return DiskDecision("OK", free, f"free {free:.2f} GiB, headroom {headroom:.2f} GiB >= floor {floor_gib} GiB")


def _walk(obj, prefix: str, h) -> None:
    if isinstance(obj, Tensor):
        h.update(f"{prefix}|T|{obj.dtype}|{tuple(obj.shape)}|".encode())
        h.update(tensor_bytes(obj))
    elif isinstance(obj, dict):
        h.update(f"{prefix}|D|{len(obj)}|".encode())
        for k in sorted(obj, key=str):
            _walk(obj[k], f"{prefix}/{k}", h)
    elif isinstance(obj, (list, tuple)):
        h.update(f"{prefix}|L|{len(obj)}|".encode())
        for i, v in enumerate(obj):
            _walk(v, f"{prefix}[{i}]", h)
    elif obj is None or isinstance(obj, (bool, int, float, str)):
        h.update(f"{prefix}|S|{type(obj).__name__}|{obj!r}|".encode())
    else:
        raise CheckpointError(f"unsupported payload type at {prefix}: {type(obj).__name__}")


def payload_digest(payload: dict) -> str:
    h = hashlib.sha256()
    _walk({k: v for k, v in payload.items() if k not in ("__format__", "__content_digest__")}, "", h)
    return h.hexdigest()


def payload_nbytes(obj) -> int:
    if isinstance(obj, Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(payload_nbytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(payload_nbytes(v) for v in obj)
    return 64


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fsync_dir(d: Path) -> None:
    if os.name != "posix":
        return
    fd = os.open(str(d), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_save(payload: dict, path: Path, *, crash_at: str | None = None,
                disk_usage_fn: Callable = shutil.disk_usage) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if crash_at is not None and crash_at not in CRASH_POINTS:
        raise ValueError(f"unknown crash point {crash_at!r}")
    content = dict(payload)
    content["__format__"] = FORMAT
    content["__content_digest__"] = payload_digest(payload)
    est = payload_nbytes(content) * 1.05 + (1 << 20)
    dec = disk_check(path.parent, est / 2 ** 30, disk_usage_fn=disk_usage_fn)
    if dec.state == "REFUSE":
        raise DiskRefused(dec.reason)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with open(tmp, "wb") as f:
        torch.save(content, f)
        f.flush()
        if crash_at == "after_tmp_write":
            raise SimulatedCrash(crash_at)
        os.fsync(f.fileno())
    if crash_at == "after_fsync":
        raise SimulatedCrash(crash_at)
    back = load_checkpoint(tmp)
    if back["__content_digest__"] != content["__content_digest__"]:
        raise CheckpointCorrupt(f"reload of {tmp} does not validate")
    if crash_at == "after_validate":
        raise SimulatedCrash(crash_at)
    os.replace(tmp, path)
    _fsync_dir(path.parent)
    if crash_at == "after_replace":
        raise SimulatedCrash(crash_at)
    return {"path": str(path), "bytes": path.stat().st_size, "content_digest": content["__content_digest__"],
            "disk": dec.state, "disk_reason": dec.reason}


def load_checkpoint(path: Path) -> dict:
    path = Path(path)
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:  # noqa: BLE001 — truncated / garbled files must never be resumed from
        raise CheckpointCorrupt(f"cannot read {path}: {type(e).__name__}: {e}") from None
    if not isinstance(obj, dict) or obj.get("__format__") != FORMAT:
        raise CheckpointCorrupt(f"{path} is not a {FORMAT} checkpoint")
    if payload_digest(obj) != obj.get("__content_digest__"):
        raise CheckpointCorrupt(f"{path} fails its content digest")
    return obj


class CheckpointManager:
    """One run's checkpoint directory with its retention rules."""

    LATEST, BEST, ENDPOINT = "latest.pt", "best.pt", "endpoint_6.0.pt"

    def __init__(self, root: Path, run_id: str, *, retain_weights: bool = True,
                 disk_usage_fn: Callable = shutil.disk_usage):
        self.dir = Path(root) / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.retain_weights = retain_weights
        self.disk_usage_fn = disk_usage_fn
        self.events = self.dir / "events.jsonl"
        self.removals = self.dir / "removal_log.jsonl"
        for stale in sorted(self.dir.glob("*.tmp.*")):       # our own leftovers from an interrupted write
            self._remove(stale, "stale temp file of an interrupted checkpoint write")

    def _log(self, path: Path, rec: dict) -> None:
        with open(path, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **rec},
                               sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def _remove(self, path: Path, reason: str) -> None:
        if not path.exists():
            return
        rec = {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size, "reason": reason}
        path.unlink()
        self._log(self.removals, rec)

    def _save(self, name: str, payload: dict, crash_at: str | None = None) -> dict:
        info = atomic_save(payload, self.dir / name, crash_at=crash_at, disk_usage_fn=self.disk_usage_fn)
        self._log(self.events, {"saved": name, **{k: info[k] for k in ("bytes", "content_digest", "disk")}})
        return info

    def save_latest(self, payload: dict, crash_at: str | None = None) -> dict:
        return self._save(self.LATEST, payload, crash_at)

    def save_best(self, payload: dict) -> dict:
        return self._save(self.BEST, payload)

    def save_endpoint(self, payload: dict) -> dict:
        return self._save(self.ENDPOINT, payload)

    def load_latest(self) -> dict | None:
        p = self.dir / self.LATEST
        return load_checkpoint(p) if p.exists() else None

    def files(self) -> list:
        return sorted(p.name for p in self.dir.glob("*.pt"))

    def complete(self) -> None:
        self._remove(self.dir / self.LATEST, "run complete: the resumable checkpoint is not retained")
        if not self.retain_weights:
            for name in (self.BEST, self.ENDPOINT):
                self._remove(self.dir / name, "calibration run: no weights retained")


def rng_state() -> dict:
    import random

    import numpy as np
    st = {"torch_cpu": torch.get_rng_state()}
    ns = np.random.get_state()
    st["numpy"] = {"name": ns[0], "keys": torch.from_numpy(ns[1].astype("int64")), "pos": int(ns[2]),
                   "has_gauss": int(ns[3]), "cached_gaussian": float(ns[4])}
    st["python"] = [list(random.getstate()[1]), random.getstate()[0], random.getstate()[2]]
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        st["torch_cuda"] = [s for s in torch.cuda.get_rng_state_all()]
    return st


def restore_rng_state(st: dict) -> None:
    import random

    import numpy as np
    torch.set_rng_state(st["torch_cpu"])
    n = st["numpy"]
    np.random.set_state((n["name"], n["keys"].numpy().astype("uint32"), n["pos"], n["has_gauss"],
                         n["cached_gaussian"]))
    ver, gauss = st["python"][1], st["python"][2]
    random.setstate((ver, tuple(int(x) for x in st["python"][0]), gauss))
    if "torch_cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["torch_cuda"])
