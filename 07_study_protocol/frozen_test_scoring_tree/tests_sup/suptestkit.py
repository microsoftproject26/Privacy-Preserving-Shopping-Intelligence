"""Shared test doubles for tests_sup (mocks nvidia-smi and disk usage, per the SYS-03 task packet:
"Mock nvidia-smi and disk in the tests."). No test in this suite shells out to a real GPU tool or
depends on the real free space of the machine it runs on.
"""
from __future__ import annotations

import subprocess
from collections import namedtuple

DiskUsage = namedtuple("DiskUsage", "total used free")
GIB = 2 ** 30


def fake_disk_usage_fn(free_gib: float, total_gib: float = 200.0):
    """A `shutil.disk_usage`-compatible callable reporting a fixed free space."""
    total = int(total_gib * GIB)
    free = int(free_gib * GIB)

    def _fn(path):
        return DiskUsage(total=total, used=total - free, free=free)

    return _fn


def fake_nvidia_smi_runner(gpu_csv: "str | None" = "0, 00000000:01:00.0\n1, 00000000:02:00.0",
                           apps_csv: "str | None" = "", rc_gpu: int = 0, rc_apps: int = 0,
                           raise_exc: "Exception | None" = None):
    """A fake `runner(argv, **kw) -> subprocess.CompletedProcess` standing in for
    `subprocess.run(["nvidia-smi", ...])`, matching the two queries `gpu.gpu_busy_map` issues."""

    def runner(argv, **kw):
        if raise_exc is not None:
            raise raise_exc
        if any("query-gpu" in a for a in argv):
            return subprocess.CompletedProcess(argv, rc_gpu, stdout=gpu_csv or "", stderr="")
        return subprocess.CompletedProcess(argv, rc_apps, stdout=apps_csv or "", stderr="")

    return runner
