"""disk_guard.py — live free-space guard for the shared Lobot volume (Sonnet A, Wave 1).

`FULL_RUN_DISK_BUDGET.md` policy: keep >= 10 GiB free at all times; emergency floor 8 GiB;
any writer whose projected post-write free space would drop below the floor must not write.
This module is the reusable check other agents' writers (checkpoints, logs) can call before a
write. Read-only: never writes, never deletes, never touches the frozen dataset.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

GIB = 2 ** 30
DEFAULT_FLOOR_BYTES = 8 * GIB
TARGET_FREE_BYTES = 10 * GIB


def free_bytes(path: str | Path) -> int:
    """Live free bytes on the filesystem containing `path` (via statvfs/shutil.disk_usage;
    resolves to the nearest existing ancestor if `path` itself does not exist yet)."""
    p = Path(path)
    while not p.exists():
        if p.parent == p:
            raise FileNotFoundError(f"no existing ancestor found for {path!r}")
        p = p.parent
    return shutil.disk_usage(str(p)).free


def check_floor(path: str | Path, planned_bytes: int, floor: int = DEFAULT_FLOOR_BYTES) -> tuple[bool, int]:
    """Returns (ok, free_after) where `free_after` is the LIVE free space minus
    `planned_bytes`, and `ok` is False if that projected value would drop below `floor`.
    Uses live free space (not a cached budget), because the volume is shared with two other
    agents' writers."""
    before = free_bytes(path)
    free_after = before - int(planned_bytes)
    ok = free_after >= int(floor)
    return ok, free_after


def _main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", nargs="?", default=".", help="path (or ancestor) to check")
    ap.add_argument("--planned-bytes", type=int, default=0, help="bytes about to be written")
    ap.add_argument("--floor-gib", type=float, default=8.0, help="emergency floor, GiB")
    ap.add_argument("--target-gib", type=float, default=10.0, help="soft target, GiB")
    args = ap.parse_args()

    floor = int(args.floor_gib * GIB)
    target = int(args.target_gib * GIB)
    before = free_bytes(args.path)
    ok, free_after = check_floor(args.path, args.planned_bytes, floor=floor)
    meets_target = free_after >= target

    result = {
        "path": str(args.path),
        "free_bytes_now": int(before),
        "free_gib_now": before / GIB,
        "planned_bytes": int(args.planned_bytes),
        "free_bytes_after": int(free_after),
        "free_gib_after": free_after / GIB,
        "floor_gib": args.floor_gib,
        "target_gib": args.target_gib,
        "pass_floor": bool(ok),
        "pass_target": bool(meets_target),
    }
    print(json.dumps(result, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_main())
