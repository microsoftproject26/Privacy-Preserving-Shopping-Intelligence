"""Atomic status file + append-only event log (package 10 S4: "an atomic status file plus an
append-only event log. The heartbeat is separate from completion.").

Status write pattern: write full payload to a sibling temp file -> flush -> fsync -> os.replace
(atomic on both POSIX and Windows when temp and target share a filesystem/directory). If the
process dies before `os.replace`, the previous file is completely untouched (T46); if it dies
after, the new file is fully there. There is no window in which a reader can observe a
half-written status file.

`merge_status` is the heartbeat path: it only ever updates liveness fields (or a caller-chosen
subset) via read-modify-write, and never itself asserts phase completion -- only `runner.py`'s
artifact-validated DONE path does that.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_status(path: "Path | str") -> "dict | None":
    p = Path(path)
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def write_status_atomic(path: "Path | str", data: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=1, sort_keys=True)
    tmp = p.with_name(p.name + f".tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.replace(tmp, p)  # atomic on POSIX and Windows
    finally:
        # os.replace either succeeded (tmp no longer exists) or raised (tmp is a harmless
        # leftover); either way `p` itself was never partially written.
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def append_event(path: "Path | str", event: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"utc": utc(), **event}, sort_keys=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def merge_status(path: "Path | str", **updates) -> dict:
    """Read-modify-write for fields that must never themselves assert completion (the heartbeat)."""
    cur = read_status(path) or {"created_utc": utc()}
    cur.update(updates)
    cur["updated_utc"] = utc()
    write_status_atomic(path, cur)
    return cur
