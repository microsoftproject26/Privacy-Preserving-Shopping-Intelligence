"""One heavy baseline process at a time, using the existing runtime monitor.

Run from the project environment. The preparation child has an isolated pinned
pandas environment; training children keep the frozen project environment.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import psutil

from ppsi.federated.runtime_monitor import MonitorLimits, ProcessTreeMonitor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "classical", "session"), required=True)
    parser.add_argument("--batch", default="batch-001")
    parser.add_argument("--raw", default="Dataset/processed_raw_parquet_v1.parquet")
    args = parser.parse_args()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True
    ).strip()
    if branch != "s2-pr-04-05-classical-session-baselines":
        raise RuntimeError("wrong branch; human must switch before execution")
    config = json.loads((ROOT / "config/baselines/s2_pr_04_05.v1.json").read_text())
    policy = config["resources"]
    if psutil.virtual_memory().available < policy["min_available_ram_gib"] * 1024**3:
        raise RuntimeError("BLOCKED_CAPACITY: free at least 6 GiB; do not lower the gate")
    if shutil.disk_usage(ROOT).free < policy["min_free_disk_gib"] * 1024**3:
        raise RuntimeError("BLOCKED_DISK: free at least 5 GiB")
    lock = ROOT / "artifacts/baselines/s2-pr-04-05/ACTIVE_RUN.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    # A leftover lock is reported, never silently removed by a competing process.
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    logs = lock.parent / args.batch / "execution_logs"
    logs.mkdir(parents=True, exist_ok=True)
    command = (
        [
            "uv",
            "run",
            "--no-project",
            "--python",
            "3.11.14",
            "--script",
            str(ROOT / "scripts/baselines/prepare_baseline_views.py"),
            "--repo",
            str(ROOT),
            "--raw",
            args.raw,
        ]
        if args.stage == "prepare"
        else [
            sys.executable,
            str(ROOT / "scripts/baselines/run_baselines.py"),
            "--stage",
            args.stage,
            "--batch",
            args.batch,
        ]
    )
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS"):
        env[key] = str(config["threads"])
    started = time.monotonic()
    reason = None
    child = None
    monitor = None
    try:
        with (logs / f"{stamp}_{args.stage}.log").open("w", encoding="utf-8") as output:
            child = subprocess.Popen(
                command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, env=env, shell=False
            )
            monitor = ProcessTreeMonitor(
                child.pid,
                limits=MonitorLimits(
                    max_tree_rss_fraction_total=policy["max_rss_fraction_total"],
                    min_available_ram_gib=policy["min_available_during_run_gib"],
                ),
            )
            monitor.start()
            while child.poll() is None:
                if monitor.guard_reason:
                    reason = "RESOURCE_GUARD"
                    break
                if time.monotonic() - started > policy["max_stage_wall_seconds"]:
                    reason = "STAGE_TIMEOUT"
                    break
                time.sleep(1)
            if reason:
                monitor.terminate_owned_tree()
            code = child.wait(timeout=30)
            measured = monitor.stop().to_public_dict()
        record = {
            "stage": args.stage,
            "exit_code": code,
            "guard_reason": reason,
            "elapsed_seconds": time.monotonic() - started,
            "resources": measured,
            "log_file": f"{stamp}_{args.stage}.log",
        }
        (logs / f"{stamp}_{args.stage}.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {k: record[k] for k in ("stage", "exit_code", "guard_reason", "elapsed_seconds")},
                indent=2,
            )
        )
        if code != 0 or reason:
            raise SystemExit(1)
    finally:
        if child is not None and child.poll() is None:
            if monitor is not None:
                monitor.terminate_owned_tree()
            else:
                child.terminate()
        if monitor is not None:
            monitor.stop()
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
