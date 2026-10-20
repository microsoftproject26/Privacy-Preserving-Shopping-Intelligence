"""v4train.checkpoint — atomic checkpoints, strict-JSON records and the retention rules.

Write protocol (package 10 §3; DISK_RETENTION_PLAN rule 5): torch.save to a sibling temp file -> flush -> fsync ->
reload and validate (required keys, config hash, the determinism mode record, the manifest-registry and schema records
of TRAIN-01R2, finite float tensors) -> os.replace -> fsync the directory where the OS allows it. A failure at any step leaves the previous file untouched, so there is always one valid resume point.
Every file this run creates is listed in CHECKPOINTS.json with its sha256; every deletion is appended to
RETENTION_LOG.jsonl with the sha256 of the removed bytes (only files this run created are ever removed).

Kinds:
  latest            resumable: weights, AdamW state, RNG states, data cursor, exposures, schedule position, stop state
  best              weights of the strictly-best evaluation in [0.5, 6.0] (PRACTICAL_BEST)
  endpoint_6.0      weights at 6.0 EFE (CONTROLLED_BUDGET)
  best_extended     weights of the strictly-best continuation evaluation (PRACTICAL_BEST_EXTENDED), if any
Retention (DISK_RETENTION_PLAN rules 1-2; E12.1.8): main runs keep latest + best + endpoint_6.0 while live; at 6.0
`latest` is deleted unless the run is STILL_IMPROVING; after a continuation `latest` is deleted. Calibration runs keep
only `latest` while live and delete it at the end: no weights are retained.
TRAIN-02 (TUNING runs, PREG1_AMENDMENT_02R1 §2): weights-only kinds snapshot_5.0 / snapshot_5.5 / snapshot_6.0 (the
CKPT_AVG inputs) and ckpt_avg (their uniform average); a TUNING run keeps `latest` while live, deletes it at 6.0 (no
continuation), and v4train.ckpt_avg deletes the snapshots and the average after their evaluation, so it retains no
weights. `peak_bytes` is the largest total of weight files seen in the directory (a file being written counted with the
files already there), for the disk-peak report.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Callable, Mapping, Optional

import torch

from v4eval.values import to_jsonable
from v4sup.artifacts import parse_strict_json

from .recipe import DETERMINISM_MODES

KINDS = ("latest", "best", "endpoint_6.0", "best_extended", "snapshot_5.0", "snapshot_5.5", "snapshot_6.0", "ckpt_avg")
REQUIRED = {"latest": ("kind", "meta", "model", "optimizer", "rng", "train_state", "stop_state"),
            "best": ("kind", "meta", "model"), "endpoint_6.0": ("kind", "meta", "model"),
            "best_extended": ("kind", "meta", "model"), "snapshot_5.0": ("kind", "meta", "model"),
            "snapshot_5.5": ("kind", "meta", "model"), "snapshot_6.0": ("kind", "meta", "model"),
            "ckpt_avg": ("kind", "meta", "model")}


class CheckpointError(RuntimeError):
    pass


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def replace_retry(src: Path, dst: Path, tries: int = 40, delay_s: float = 0.05) -> None:
    """os.replace, retried on Windows only: a scanner or indexer briefly holding the just-written target makes
    MoveFileEx fail with ERROR_ACCESS_DENIED (seen on the local test drive). POSIX raises at once."""
    for i in range(tries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if os.name != "nt" or i == tries - 1:
                raise
            time.sleep(delay_s)


def _fsync_dir(d: Path) -> None:
    try:
        fd = os.open(str(d), os.O_RDONLY)
    except OSError:
        return                                   # Windows: directories cannot be opened for fsync
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def write_json_atomic(path: Path, obj) -> str:
    """Strict JSON (no NaN / inf, UNDEFINED explicit) written atomically; returns the file sha256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(to_jsonable(obj), indent=1, sort_keys=True, allow_nan=False)
    parse_strict_json(text)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    parse_strict_json(tmp.read_text(encoding="utf-8"))
    replace_retry(tmp, path)
    _fsync_dir(path.parent)
    return sha256_file(path)


def read_json(path: Path):
    return parse_strict_json(Path(path).read_text(encoding="utf-8"))


def append_jsonl(path: Path, record: Mapping) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(to_jsonable(dict(record)), sort_keys=True, allow_nan=False)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def _validate(obj: Mapping, kind: str, config_sha256: str) -> None:
    missing = [k for k in REQUIRED[kind] if k not in obj]
    if missing:
        raise CheckpointError(f"{kind} checkpoint lacks {missing}")
    if obj["kind"] != kind:
        raise CheckpointError(f"checkpoint kind {obj['kind']!r} != {kind!r}")
    if obj["meta"].get("config_sha256") != config_sha256:
        raise CheckpointError("checkpoint config hash differs from the run's")
    det = obj["meta"].get("determinism")
    if not isinstance(det, Mapping) or det.get("mode") not in DETERMINISM_MODES or \
            not det.get("cublas_workspace_config"):
        raise CheckpointError("checkpoint meta lacks the determinism mode record (PREG1_AMENDMENT_01 item 4)")
    reg = obj["meta"].get("manifest_registry")
    if not isinstance(reg, Mapping) or not all(reg.get(k) for k in ("file_sha256", "self_sha256", "data_version",
                                                                    "role")):
        raise CheckpointError("checkpoint meta lacks the manifest registry record (TRAIN-01R2)")
    sch = obj["meta"].get("schema")
    if not isinstance(sch, Mapping) or not sch.get("schema_id") or not sch.get("schema_self_sha256"):
        raise CheckpointError("checkpoint meta lacks the schema record (TRAIN-01R2)")
    for k, t in obj["model"].items():
        if torch.is_tensor(t) and t.is_floating_point() and not bool(torch.isfinite(t).all()):
            raise CheckpointError(f"non-finite weights in {k}")


class CheckpointStore:
    def __init__(self, run_dir: Path, config_sha256: str, *,
                 fault: Optional[Callable[[str, str], None]] = None):
        self.dir = Path(run_dir) / "checkpoints"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = Path(run_dir)
        self.config_sha256 = config_sha256
        self.index_path = self.run_dir / "CHECKPOINTS.json"
        self.log_path = self.run_dir / "RETENTION_LOG.jsonl"
        self._fault = fault                      # test hook: fault(kind, stage) may raise at "after_tmp"
        self.index = read_json(self.index_path) if self.index_path.exists() else {}
        self.peak_bytes = self._weights_bytes()  # TRAIN-02: the disk peak of weight files seen by this store

    def _weights_bytes(self, extra: int = 0) -> int:
        total = int(extra)
        for q in self.dir.glob("*.pt*"):
            try:
                total += q.stat().st_size
            except OSError:
                pass
        return total

    def path(self, kind: str) -> Path:
        if kind not in KINDS:
            raise CheckpointError(f"unknown checkpoint kind {kind!r}")
        return self.dir / f"{kind}.pt"

    def save(self, kind: str, obj: dict) -> str:
        p = self.path(kind)
        obj = dict(obj, kind=kind)
        tmp = p.with_name(p.name + f".tmp.{os.getpid()}")
        with open(tmp, "wb") as f:
            torch.save(obj, f)
            f.flush()
            os.fsync(f.fileno())
        try:
            if self._fault is not None:
                self._fault(kind, "after_tmp")
            back = torch.load(tmp, map_location="cpu", weights_only=True)
            _validate(back, kind, self.config_sha256)
            digest = sha256_file(tmp)
            self.peak_bytes = max(self.peak_bytes, self._weights_bytes())      # tmp + every file already there
            replace_retry(tmp, p)
        finally:
            if tmp.exists():
                tmp.unlink()
        _fsync_dir(p.parent)
        self.index[kind] = {"file": str(p.relative_to(self.run_dir)).replace("\\", "/"), "sha256": digest,
                            "bytes": p.stat().st_size, "meta": obj["meta"]}
        write_json_atomic(self.index_path, self.index)
        return digest

    def load(self, kind: str) -> dict:
        p = self.path(kind)
        if not p.is_file():
            raise CheckpointError(f"no {kind} checkpoint at {p}")
        rec = self.index.get(kind)
        if rec is None or sha256_file(p) != rec["sha256"]:
            raise CheckpointError(f"{kind} checkpoint does not match CHECKPOINTS.json (sha256)")
        obj = torch.load(p, map_location="cpu", weights_only=True)
        _validate(obj, kind, self.config_sha256)
        return obj

    def exists(self, kind: str) -> bool:
        return self.path(kind).is_file()

    def delete(self, kind: str, reason: str) -> Optional[str]:
        """Remove a checkpoint this run created; the deletion is logged with the removed file's sha256."""
        p = self.path(kind)
        if not p.is_file():
            return None
        rec = self.index.get(kind)
        digest = sha256_file(p)
        if rec is None or rec["sha256"] != digest:
            raise CheckpointError(f"refusing to delete {p}: not a file this run recorded")
        p.unlink()
        append_jsonl(self.log_path, {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "deleted": rec["file"],
                                     "sha256": digest, "bytes": rec["bytes"], "reason": reason})
        self.index.pop(kind)
        write_json_atomic(self.index_path, self.index)
        return digest

    def weight_files(self) -> list:
        return sorted(q.name for q in self.dir.glob("*.pt"))


def finite(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


__all__ = ["KINDS", "CheckpointError", "CheckpointStore", "write_json_atomic", "read_json", "append_jsonl",
           "sha256_file", "finite"]
