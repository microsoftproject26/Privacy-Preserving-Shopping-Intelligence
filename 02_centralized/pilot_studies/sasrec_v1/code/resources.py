"""resources.py -- live resource sampling: cgroup mem/cpu, disk free, GPU, RSS.

Owner: Sonnet-runtime (SASREC_INTERFACES.md sec 7, 05_TESTS_AND_PRETRAIN_GATE E36).

`sample()` must read the CONTAINER's live cgroup limits, never the host's total
core count from the standard library, and never a hardcoded host size, so the
queue runner and supervisors size concurrency off the actual quota. Everything
here is read-only: no writes except the optional `--loop` JSONL sampler file
this module itself owns.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import paths  # noqa: F401  import-first convention

GIB = 2 ** 30

CGROUP_V2_ROOT = Path("/sys/fs/cgroup")
CGROUP_V1_MEM = Path("/sys/fs/cgroup/memory")
CGROUP_V1_CPU = Path("/sys/fs/cgroup/cpu")


def _read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def cgroup_memory() -> dict:
    """Returns {"limit_bytes": int|None, "current_bytes": int|None, "source": str}."""
    v2_max = _read_text(CGROUP_V2_ROOT / "memory.max")
    if v2_max is not None:
        cur = _read_text(CGROUP_V2_ROOT / "memory.current")
        limit = None if v2_max == "max" else int(v2_max)
        return {"limit_bytes": limit,
                "current_bytes": int(cur) if cur is not None else None,
                "source": "cgroup_v2"}
    limit_txt = _read_text(CGROUP_V1_MEM / "memory.limit_in_bytes")
    cur_txt = _read_text(CGROUP_V1_MEM / "memory.usage_in_bytes")
    if limit_txt is not None:
        limit = int(limit_txt)
        # cgroup v1 "no limit" is a huge sentinel (e.g. 2**63-ish); treat as unset.
        if limit > (1 << 62):
            limit = None
        return {"limit_bytes": limit,
                "current_bytes": int(cur_txt) if cur_txt is not None else None,
                "source": "cgroup_v1"}
    return {"limit_bytes": None, "current_bytes": None, "source": "unavailable"}


def cgroup_cpu_quota_cores() -> dict:
    """Returns {"quota_cores": float|None, "source": str}. This never falls back to
    the host's total core count (the stdlib call this module intentionally avoids)
    -- callers must treat None as "unknown, be conservative"."""
    v2 = _read_text(CGROUP_V2_ROOT / "cpu.max")
    if v2 is not None:
        parts = v2.split()
        if len(parts) == 2 and parts[0] != "max":
            quota, period = int(parts[0]), int(parts[1])
            return {"quota_cores": quota / period, "source": "cgroup_v2"}
        return {"quota_cores": None, "source": "cgroup_v2_max"}
    quota_txt = _read_text(CGROUP_V1_CPU / "cpu.cfs_quota_us")
    period_txt = _read_text(CGROUP_V1_CPU / "cpu.cfs_period_us")
    if quota_txt is not None and period_txt is not None:
        quota, period = int(quota_txt), int(period_txt)
        if quota > 0 and period > 0:
            return {"quota_cores": quota / period, "source": "cgroup_v1"}
        return {"quota_cores": None, "source": "cgroup_v1_unlimited"}
    return {"quota_cores": None, "source": "unavailable"}


def disk_free(path: "str | Path | None" = None) -> dict:
    p = Path(path) if path is not None else paths.S
    while not p.exists():
        if p.parent == p:
            raise FileNotFoundError(f"no existing ancestor for {path!r}")
        p = p.parent
    usage = shutil.disk_usage(str(p))
    return {"path": str(p), "free_bytes": usage.free, "total_bytes": usage.total,
            "used_bytes": usage.used, "free_gib": usage.free / GIB}


def gpu_stats() -> list[dict]:
    """Per-physical-GPU util/mem via nvidia-smi. Empty list if nvidia-smi is absent
    or fails (e.g. no GPU visible in this shell) -- never raises."""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    gpus = []
    for line in out.stdout.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            idx, util, mem_used, mem_total = (int(parts[0]), float(parts[1]),
                                               float(parts[2]), float(parts[3]))
        except ValueError:
            continue
        gpus.append({"index": idx, "utilization_pct": util,
                     "memory_used_mib": mem_used, "memory_total_mib": mem_total})
    return gpus


def rss_of_pids(pids: "list[int]") -> dict:
    """Live RSS of the given PIDs. Uses psutil when available, else /proc fallback.
    Missing/dead PIDs are reported with rss_bytes=None and excluded from the total."""
    per_pid: dict[str, Optional[int]] = {}
    try:
        import psutil
        for pid in pids:
            try:
                per_pid[str(pid)] = psutil.Process(int(pid)).memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
                per_pid[str(pid)] = None
    except ImportError:
        for pid in pids:
            statm = Path(f"/proc/{pid}/statm")
            txt = _read_text(statm)
            if txt is None:
                per_pid[str(pid)] = None
                continue
            try:
                rss_pages = int(txt.split()[1])
                per_pid[str(pid)] = rss_pages * 4096
            except (IndexError, ValueError):
                per_pid[str(pid)] = None
    total = sum(v for v in per_pid.values() if v is not None)
    return {"per_pid": per_pid, "total_bytes": total}


def sample(pids: "list[int] | None" = None, disk_path: "str | Path | None" = None) -> dict:
    return {
        "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cgroup_memory": cgroup_memory(),
        "cgroup_cpu": cgroup_cpu_quota_cores(),
        "disk": disk_free(disk_path),
        "gpus": gpu_stats(),
        "rss": rss_of_pids(pids or []),
    }


def sampler_loop(out_jsonl: "str | Path", interval: float = 60.0,
                  pids: "list[int] | None" = None, iterations: "int | None" = None) -> None:
    """Append one JSON line per interval, forever (or `iterations` times for tests).
    SIGTERM/KeyboardInterrupt exit cleanly; the file is opened append-only so a
    restarted sampler never truncates prior samples."""
    out_path = Path(out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    while iterations is None or n < iterations:
        rec = sample(pids=pids)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
        n += 1
        if iterations is not None and n >= iterations:
            break
        time.sleep(interval)


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--loop", metavar="OUT_JSONL", help="append samples forever to this file")
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--iterations", type=int, default=None, help="test-only: stop after N samples")
    ap.add_argument("--pids", default="", help="comma-separated PIDs to report RSS for")
    ap.add_argument("--once", action="store_true", help="print one sample as JSON and exit")
    args = ap.parse_args(argv)
    pids = [int(x) for x in args.pids.split(",") if x.strip()]
    if args.once or not args.loop:
        print(json.dumps(sample(pids=pids), indent=2))
        return 0
    try:
        sampler_loop(args.loop, interval=args.interval, pids=pids, iterations=args.iterations)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(_main())
