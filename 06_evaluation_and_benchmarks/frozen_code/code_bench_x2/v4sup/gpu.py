"""GPU ownership check -- fail-closed (package 10 SS1, 4: "checks GPU ownership fail-closed: any
compute app on a needed GPU means wait; an nvidia-smi error means busy").

Any compute process on a needed GPU -- ours, a teammate's, or unidentified -- means that GPU is
busy. An nvidia-smi error, a missing binary, a non-zero exit, an unparsable line, or a
compute-app row on a PCI bus we did not enumerate all mark EVERY requested GPU busy: this module
never assumes a GPU is free from a partial or failed reading. (Pattern only, independently
reimplemented here; not copied from feature_opt_v1/code/fo_queue.py.)
"""
from __future__ import annotations

import subprocess
from typing import Callable, Iterable

Runner = Callable[..., "subprocess.CompletedProcess"]


def default_runner(argv, **kw):
    return subprocess.run(argv, capture_output=True, text=True, timeout=60, **kw)


def gpu_busy_map(runner: Runner = default_runner) -> "dict[str, list[str]]":
    """{gpu_index_str: [pids]} for every GPU that currently has at least one compute process.

    On any failure to get a clean, fully-mapped reading, returns {"unknown": [reason]} -- the
    caller must treat that as "every requested GPU is busy".
    """
    try:
        q = runner(["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"])
        a = runner(["nvidia-smi", "--query-compute-apps=gpu_bus_id,pid", "--format=csv,noheader"])
    except Exception as e:  # missing binary, permission error, timeout, ...
        return {"unknown": [f"nvidia-smi unavailable: {e}"]}
    if q.returncode != 0 or a.returncode != 0:
        return {"unknown": [f"nvidia-smi returned non-zero (gpu={q.returncode}, apps={a.returncode})"]}
    bus: "dict[str, str]" = {}
    for line in q.stdout.strip().splitlines():
        if not line.strip():
            continue
        parts = line.split(",", 1)
        if len(parts) != 2:
            return {"unknown": [f"unparsable nvidia-smi -L line: {line!r}"]}
        idx, pci = parts
        bus[pci.strip()] = idx.strip()
    if not bus:
        return {"unknown": ["no GPUs enumerated"]}
    busy: "dict[str, list[str]]" = {i: [] for i in bus.values()}
    for line in a.stdout.strip().splitlines():
        if not line.strip():
            continue
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 2 or parts[0] not in bus:
            # a compute app on a PCI bus we did not enumerate: fail closed on everything
            return {i: ["unmapped compute app"] for i in bus.values()}
        busy[bus[parts[0]]].append(parts[1])
    return {i: pids for i, pids in busy.items() if pids}


def gpu_ready(needed: "Iterable[int]", runner: Runner = default_runner) -> "tuple[bool, str]":
    needed = list(needed)
    if not needed:
        return True, "no GPU required"
    busy = gpu_busy_map(runner)
    if "unknown" in busy:
        return False, f"GPU check fail-closed: {busy['unknown']}"
    taken = {g: busy[str(g)] for g in needed if busy.get(str(g))}
    if taken:
        return False, f"GPU(s) busy: {taken}"
    return True, "ok"
