"""ckpt_io.py -- atomic checkpoint I/O and disk-floor policy (owner: Sonnet-runtime).

Gate P fix #8 / SASREC_INTERFACES.md sec 7+9:
  - One rotation function. Write a unique `.partial.<pid>` file, flush+fsync the file,
    reload it (torch.load(weights_only=False)) and validate its keys/update BEFORE any
    rotation or replace. Only then: for name=="latest", rotate latest -> previous
    (os.replace), then install the new file as latest (os.replace), then fsync the
    containing directory so the rename is durable.
  - Pre-write free-space check: if free-after-write would drop below the *target*
    (10 GiB), first reclaim allowlisted files; if it would still drop below the
    emergency *floor* (8 GiB), refuse to write at all (raise DiskFloor) -- the caller
    (train_sasrec.py) turns that into exit code 4. Nothing is written or rotated when
    DiskFloor is raised: the previous valid latest.pt is untouched.
  - reclaim() only ever deletes `checkpoints/latest.pt` and `checkpoints/previous.pt`
    of runs strictly under a `runs/` root that both (a) have a SUPERVISOR_DONE marker
    in the run dir and (b) have `summary.json` final_status == "COMPLETE". It never
    touches best_weights.pt, rank files, or anything outside that run tree.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional

import paths  # noqa: F401  (import-first convention; also puts reuse/ on sys.path)

try:
    import torch
except Exception:  # pragma: no cover - torch is always present on the server venv
    torch = None  # type: ignore

GIB = 2 ** 30
TARGET_FREE_BYTES = 10 * GIB
FLOOR_BYTES = 8 * GIB


class DiskFloor(RuntimeError):
    """Raised when a checkpoint write would push free space below the emergency floor.

    Nothing is written to disk when this is raised: the caller must treat this as a
    clean, resumable stop (train_sasrec.py: exit code 4)."""


# ---------------------------------------------------------------------------
# small filesystem helpers
# ---------------------------------------------------------------------------

def sha256_file(path: "str | Path") -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _fsync_file(fh) -> None:
    fh.flush()
    os.fsync(fh.fileno())


def _fsync_dir(dir_path: "str | Path") -> None:
    # Directory fsync is POSIX-only and required to make a rename durable.
    try:
        fd = os.open(str(dir_path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except (OSError, NotImplementedError):
        # Best-effort: some filesystems (or Windows, never used in production here)
        # do not support fsync on directories.
        pass


def _serialize(obj: Any) -> bytes:
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required to serialize checkpoints")
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


def _torch_load(path: "str | Path"):
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required to load checkpoints")
    return torch.load(str(path), weights_only=False, map_location="cpu")


# ---------------------------------------------------------------------------
# disk policy
# ---------------------------------------------------------------------------

def free_bytes(path: "str | Path") -> int:
    p = Path(path)
    while not p.exists():
        if p.parent == p:
            raise FileNotFoundError(f"no existing ancestor found for {path!r}")
        p = p.parent
    return shutil.disk_usage(str(p)).free


def _run_is_reclaimable(run_dir: Path) -> bool:
    if not (run_dir / "SUPERVISOR_DONE").exists():
        return False
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text())
    except Exception:
        return False
    return summary.get("final_status") == "COMPLETE"


def reclaim(runs_root: "str | Path", *, log_path: Optional["str | Path"] = None) -> list[str]:
    """Delete latest.pt/previous.pt for COMPLETE+SUPERVISOR_DONE runs under `runs_root`.

    Returns the list of deleted absolute paths (as strings). Never deletes
    best_weights.pt, rank files, or config/summary/log files, and never touches
    anything outside `runs_root`. Safe to call with an empty/nonexistent root."""
    runs_root = Path(runs_root).resolve()
    deleted: list[str] = []
    if not runs_root.exists():
        return deleted
    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        run_dir = run_dir.resolve()
        # Refuse to touch anything that resolved outside runs_root (defence in depth
        # against a symlinked run dir).
        try:
            run_dir.relative_to(runs_root)
        except ValueError:
            continue
        if not _run_is_reclaimable(run_dir):
            continue
        ckpt_dir = run_dir / "checkpoints"
        for name in ("latest.pt", "previous.pt"):
            target = ckpt_dir / name
            if target.exists():
                try:
                    target.unlink()
                    deleted.append(str(target))
                except OSError:
                    pass
    if deleted and log_path is not None:
        try:
            log_path = Path(log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                for d in deleted:
                    f.write(f"[{ts}] reclaimed {d}\n")
        except OSError:
            pass
    return deleted


def check_disk_and_reclaim(dest_dir: "str | Path", estimate_bytes: int, *,
                            runs_root: Optional["str | Path"] = None,
                            reserve_bytes: int = 0,
                            log_path: Optional["str | Path"] = None) -> dict:
    """Apply the Gate P disk policy before a write of `estimate_bytes` into `dest_dir`.

    Raises DiskFloor (without writing/deleting anything else) if free space would
    still be below FLOOR_BYTES after an attempted reclaim. Returns a small report
    dict on success."""
    dest_dir = Path(dest_dir)
    need = int(estimate_bytes) + int(reserve_bytes)
    before = free_bytes(dest_dir)
    reclaimed: list[str] = []
    if before - need < TARGET_FREE_BYTES:
        root = runs_root if runs_root is not None else (paths.S / "runs")
        reclaimed = reclaim(root, log_path=log_path)
        before = free_bytes(dest_dir)
    if before - need < FLOOR_BYTES:
        raise DiskFloor(
            f"free={before} need={need} floor={FLOOR_BYTES}: refusing to write "
            f"(reclaimed {len(reclaimed)} file(s) first)"
        )
    return {"free_before": before, "estimate_bytes": estimate_bytes,
            "reserve_bytes": reserve_bytes, "reclaimed": reclaimed}


# ---------------------------------------------------------------------------
# atomic writers
# ---------------------------------------------------------------------------

def save_atomic(obj: Any, ckpt_dir: "str | Path", name: str, *, expected_update: Optional[int],
                required_keys: Iterable[str] = (), reserve_bytes: int = 0,
                runs_root: Optional["str | Path"] = None,
                disk_log_path: Optional["str | Path"] = None) -> Path:
    """Write `obj` as `<ckpt_dir>/<name>.pt`, atomically, with the Gate P disk policy.

    For name=="latest" this rotates the existing latest.pt to previous.pt first (a
    rename, not a copy). For any other name (e.g. "best_weights") it is a plain
    validated atomic replace with no rotation. Returns the final path.

    Failure modes are all safe: if `DiskFloor` is raised, or reload/validation fails,
    the pre-existing latest.pt/previous.pt are untouched and no stray `.partial.*`
    files are left behind (they are removed in a `finally`)."""
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    data = _serialize(obj)
    check_disk_and_reclaim(ckpt_dir, len(data), runs_root=runs_root,
                            reserve_bytes=reserve_bytes, log_path=disk_log_path)

    partial = ckpt_dir / f"{name}.partial.{os.getpid()}"
    try:
        with open(partial, "wb") as f:
            f.write(data)
            _fsync_file(f)
        _fsync_dir(ckpt_dir)

        # Reload and validate BEFORE any rotation/replace touches the previous state.
        reloaded = _torch_load(partial)
        missing = [k for k in required_keys if k not in reloaded]
        if missing:
            raise ValueError(f"checkpoint reload missing required keys: {missing}")
        if expected_update is not None:
            got = reloaded.get("update")
            if got != expected_update:
                raise ValueError(f"checkpoint reload update mismatch: expected "
                                  f"{expected_update!r}, got {got!r}")

        final_path = ckpt_dir / f"{name}.pt"
        if name == "latest":
            previous_path = ckpt_dir / "previous.pt"
            if final_path.exists():
                os.replace(final_path, previous_path)
        os.replace(partial, final_path)
        _fsync_dir(ckpt_dir)
        return final_path
    finally:
        if partial.exists():
            try:
                partial.unlink()
            except OSError:
                pass


def save_weights(model_state: Any, path: "str | Path", *, reserve_bytes: int = 0,
                  runs_root: Optional["str | Path"] = None,
                  disk_log_path: Optional["str | Path"] = None) -> Path:
    """Atomically write a plain weights object (e.g. best_weights.pt) to `path`.

    Same partial-write/fsync/reload/replace discipline as save_atomic, but with no
    rotation (best_weights.pt is never reclaimed and has no "previous" sibling)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _serialize(model_state)
    check_disk_and_reclaim(path.parent, len(data), runs_root=runs_root,
                            reserve_bytes=reserve_bytes, log_path=disk_log_path)
    partial = path.parent / f"{path.name}.partial.{os.getpid()}"
    try:
        with open(partial, "wb") as f:
            f.write(data)
            _fsync_file(f)
        _fsync_dir(path.parent)
        _torch_load(partial)  # validate it reloads before installing it
        os.replace(partial, path)
        _fsync_dir(path.parent)
        return path
    finally:
        if partial.exists():
            try:
                partial.unlink()
            except OSError:
                pass


def load_resume(ckpt_dir: "str | Path") -> tuple[Any, str]:
    """Load the most recent valid checkpoint: latest.pt, else previous.pt.

    Raises FileNotFoundError if neither is present or loadable."""
    ckpt_dir = Path(ckpt_dir)
    last_err: Optional[Exception] = None
    for name in ("latest", "previous"):
        p = ckpt_dir / f"{name}.pt"
        if not p.exists():
            continue
        try:
            return _torch_load(p), name
        except Exception as exc:  # corrupt file: fall back to previous
            last_err = exc
            continue
    raise FileNotFoundError(
        f"no loadable checkpoint (latest.pt/previous.pt) under {ckpt_dir}"
    ) from last_err


def _main(argv=None) -> int:  # tiny CLI for manual/server-side poking
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="op", required=True)
    r = sub.add_parser("reclaim")
    r.add_argument("runs_root")
    s = sub.add_parser("sha256")
    s.add_argument("path")
    args = ap.parse_args(argv)
    if args.op == "reclaim":
        deleted = reclaim(args.runs_root)
        print(json.dumps({"deleted": deleted}, indent=2))
        return 0
    if args.op == "sha256":
        print(sha256_file(args.path))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(_main())
