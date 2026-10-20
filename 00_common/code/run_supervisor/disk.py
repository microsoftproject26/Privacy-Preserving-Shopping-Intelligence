"""Disk guard (company/plans/DISK_RETENTION_PLAN.md S5; package 10 S3).

  REFUSE  -- do not start: free - planned_peak_gib < FLOOR_GIB (8).
  PAUSE   -- do not start new runs: free < TARGET_GIB (10).
  OK      -- otherwise.

REFUSE is checked first and is the more severe condition: it accounts for the disk this specific
phase plans to consume at its peak (including its own atomic temp file), per package 10 S3 ("free
disk >= 10 GiB target; emergency >= 8 GiB AFTER accounting for next atomic temp file"). PAUSE is
the plain free-space rule from DISK_RETENTION_PLAN.md S5 and applies even to a phase with
planned_peak_gib == 0 (e.g. a metrics-only step): the supervisor still will not schedule new work
while the shared volume is below the target.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

FLOOR_GIB = 8.0
TARGET_GIB = 10.0
GIB = 2 ** 30


@dataclass(frozen=True)
class DiskDecision:
    state: str  # "OK" | "PAUSE" | "REFUSE"
    free_gib: float
    reason: str


def free_gib(path: "Path | str", disk_usage_fn: Callable = shutil.disk_usage) -> float:
    return disk_usage_fn(str(path)).free / GIB


def disk_check(path: "Path | str", planned_peak_gib: float, disk_usage_fn: Callable = shutil.disk_usage,
               floor_gib: float = FLOOR_GIB, target_gib: float = TARGET_GIB) -> DiskDecision:
    free = free_gib(path, disk_usage_fn)
    headroom = free - planned_peak_gib
    if headroom < floor_gib:
        return DiskDecision("REFUSE", free, f"free {free:.2f} GiB - planned peak {planned_peak_gib:.2f} GiB "
                                             f"= headroom {headroom:.2f} GiB < floor {floor_gib} GiB")
    if free < target_gib:
        return DiskDecision("PAUSE", free, f"free {free:.2f} GiB < target {target_gib} GiB")
    return DiskDecision("OK", free, f"free {free:.2f} GiB, headroom {headroom:.2f} GiB >= floor {floor_gib} GiB")
