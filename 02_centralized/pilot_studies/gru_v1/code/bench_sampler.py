"""Resource sampler for the Wave-3 system benchmark (owner: Opus).

Samples, every --interval seconds until --stop-file exists (or --max-seconds):
  * per-GPU utilisation / memory used (nvidia-smi),
  * per-PID %CPU and RSS for the PIDs listed in --pid-file (one per line, may grow),
  * cgroup CPU usage (usec) and cgroup IO bytes (read/write) deltas,
  * free disk on the workspace volume.
Writes one JSON object per sample to --out (JSONL) and a summary JSON to --out + ".summary.json".
Read-only with respect to everything except its own output files. No model code.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def gpu_sample() -> list[dict]:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True)
    except Exception as exc:  # pragma: no cover
        return [{"error": str(exc)}]
    out = []
    for line in r.stdout.strip().splitlines():
        idx, util, mem = [x.strip() for x in line.split(",")]
        out.append({"gpu": int(idx), "util": float(util), "mem_mib": float(mem)})
    return out


def cgroup_cpu_usec() -> int | None:
    try:
        for line in Path("/sys/fs/cgroup/cpu.stat").read_text().splitlines():
            k, v = line.split()
            if k == "usage_usec":
                return int(v)
    except Exception:
        return None
    return None


def cgroup_io_bytes() -> tuple[int, int] | None:
    try:
        rb = wb = 0
        for line in Path("/sys/fs/cgroup/io.stat").read_text().splitlines():
            for tok in line.split()[1:]:
                k, _, v = tok.partition("=")
                if k == "rbytes":
                    rb += int(v)
                elif k == "wbytes":
                    wb += int(v)
        return rb, wb
    except Exception:
        return None


def pid_stats(pids: list[int]) -> dict:
    out = {}
    for pid in pids:
        try:
            r = subprocess.run(["ps", "-o", "pcpu=,rss=", "-p", str(pid)],
                               capture_output=True, text=True, timeout=10)
            parts = r.stdout.split()
            if len(parts) == 2:
                out[str(pid)] = {"pcpu": float(parts[0]), "rss_mib": int(parts[1]) / 1024.0}
        except Exception:
            pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stop-file", required=True)
    ap.add_argument("--pid-file", default=None)
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--max-seconds", type=float, default=7200.0)
    ap.add_argument("--disk-path", default="/home/jovyan")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    prev_cpu, prev_io, prev_t = cgroup_cpu_usec(), cgroup_io_bytes(), time.time()
    samples = []
    with out.open("w", encoding="utf-8") as fh:
        while not Path(args.stop_file).exists() and time.time() - t0 < args.max_seconds:
            time.sleep(args.interval)
            now = time.time()
            cpu, io = cgroup_cpu_usec(), cgroup_io_bytes()
            dt = now - prev_t
            rec = {"t": round(now - t0, 1), "gpus": gpu_sample()}
            if cpu is not None and prev_cpu is not None:
                rec["cgroup_cpu_cores"] = (cpu - prev_cpu) / 1e6 / dt
            if io is not None and prev_io is not None:
                rec["cgroup_read_mib_s"] = (io[0] - prev_io[0]) / dt / 2**20
                rec["cgroup_write_mib_s"] = (io[1] - prev_io[1]) / dt / 2**20
            if args.pid_file and Path(args.pid_file).exists():
                pids = [int(x) for x in Path(args.pid_file).read_text().split() if x.strip().isdigit()]
                rec["pids"] = pid_stats(pids)
            st = os.statvfs(args.disk_path)
            rec["disk_free_gib"] = st.f_bavail * st.f_frsize / 2**30
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            samples.append(rec)
            prev_cpu, prev_io, prev_t = cpu, io, now

    def stat(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        vals = sorted(vals)
        return {"mean": sum(vals) / len(vals), "p50": vals[len(vals) // 2], "max": vals[-1], "n": len(vals)}

    summary = {"n_samples": len(samples), "seconds": round(time.time() - t0, 1)}
    gpu_ids = sorted({g["gpu"] for s in samples for g in s["gpus"] if "gpu" in g})
    for g in gpu_ids:
        summary[f"gpu{g}_util"] = stat([x["util"] for s in samples for x in s["gpus"] if x.get("gpu") == g])
        summary[f"gpu{g}_mem_mib"] = stat([x["mem_mib"] for s in samples for x in s["gpus"] if x.get("gpu") == g])
    summary["cgroup_cpu_cores"] = stat([s.get("cgroup_cpu_cores") for s in samples])
    summary["cgroup_read_mib_s"] = stat([s.get("cgroup_read_mib_s") for s in samples])
    pids = sorted({p for s in samples for p in s.get("pids", {})})
    for p in pids:
        summary[f"pid{p}_pcpu"] = stat([s["pids"][p]["pcpu"] for s in samples if p in s.get("pids", {})])
        summary[f"pid{p}_rss_mib"] = stat([s["pids"][p]["rss_mib"] for s in samples if p in s.get("pids", {})])
    summary["disk_free_gib_min"] = min((s["disk_free_gib"] for s in samples), default=None)
    Path(str(out) + ".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
