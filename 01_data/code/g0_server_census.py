"""g0_server_census.py (owner: Opus, lead) — PPSI V4 G0 live resource / software / data-source census. READ-ONLY.

Writes one JSON to the path given as argv[1] (inside .local_ai/ppsi_v4/g0/). Never records process arguments
(they can carry tokens), only command names, owners-by-uid and resource figures.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import ppsi_paths

R = ppsi_paths.WORKSPACE_ROOT
D = ppsi_paths.DATA_ROOT


def sh(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return {"rc": r.returncode, "out": r.stdout.strip()[:20000]}
    except Exception as e:
        return {"rc": None, "error": repr(e)}


def rd(p):
    try:
        return Path(p).read_text().strip()
    except Exception as e:
        return f"UNREADABLE: {e!r}"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


out = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": platform.node(), "platform": platform.platform(),
       "python": sys.version.split()[0], "executable": sys.executable}
# CPU / memory / cgroup
out["cpu"] = {"os_cpu_count": os.cpu_count(), "affinity": sorted(os.sched_getaffinity(0)),
              "model": next((l.split(":", 1)[1].strip() for l in rd("/proc/cpuinfo").splitlines() if l.startswith("model name")), None)}
cg = Path("/sys/fs/cgroup")
out["cgroup"] = {k: rd(cg / k) for k in ("memory.max", "memory.current", "memory.high", "memory.swap.max", "cpu.max", "memory.pressure", "memory.events")}
stat = {}
for l in rd(cg / "memory.stat").splitlines():
    k, _, v = l.partition(" ")
    if k in ("anon", "file", "kernel", "shmem", "file_mapped", "active_file", "inactive_file", "slab"):
        stat[k] = int(v)
out["cgroup"]["memory.stat_selected"] = stat
out["meminfo"] = {l.split(":")[0]: l.split(":")[1].strip() for l in rd("/proc/meminfo").splitlines()[:6]}
# disk
out["disk"] = {str(p): {k: round(v / 2**30, 2) for k, v in shutil.disk_usage(p)._asdict().items()} for p in (Path.home(), D) if p.exists()}
# GPU and compute processes (command name + uid only)
q = sh(["nvidia-smi", "--query-gpu=index,name,pci.bus_id,memory.total,memory.used,utilization.gpu,driver_version,persistence_mode",
        "--format=csv,noheader"])
a = sh(["nvidia-smi", "--query-compute-apps=gpu_bus_id,pid,used_memory", "--format=csv,noheader"])
apps = []
for l in (a.get("out") or "").splitlines():
    if not l.strip():
        continue
    bus, pid, mem = [x.strip() for x in l.split(",")]
    comm = rd(f"/proc/{pid}/comm")
    uid = next((ln.split()[1] for ln in rd(f"/proc/{pid}/status").splitlines() if ln.startswith("Uid:")), None)
    apps.append({"bus": bus, "pid": pid, "used_memory": mem, "comm": comm, "uid": uid})
out["gpu"] = {"query": q, "compute_apps": apps, "cuda_version_line": sh(["bash", "-lc", "nvidia-smi | grep -o 'CUDA Version: [0-9.]*'"]).get("out")}
# processes (names/owners/resources only; top 25 by RSS)
ps = sh(["ps", "-eo", "pid,uid,etimes,pcpu,rss,comm", "--sort=-rss"])
out["processes_top_rss"] = (ps.get("out") or "").splitlines()[:26]
out["tmux_sessions"] = [l.split(":")[0] for l in (sh(["tmux", "ls"]).get("out") or "").splitlines()]
# software versions in the shared env (import only; never install)
vers = {}
for m in ("torch", "numpy", "pyarrow", "polars", "pandas", "duckdb", "scipy", "onnx", "onnxruntime", "flwr", "ray", "yaml", "sklearn"):
    try:
        mod = importlib.import_module(m)
        vers[m] = getattr(mod, "__version__", "unknown")
    except Exception as e:
        vers[m] = f"NOT_IMPORTABLE: {type(e).__name__}"
out["versions"] = vers
try:
    import torch
    out["torch"] = {"cuda_available": torch.cuda.is_available(), "device_count": torch.cuda.device_count(), "cuda": torch.version.cuda,
                    "cudnn": torch.backends.cudnn.version(), "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                    "tf32_matmul_default": torch.backends.cuda.matmul.allow_tf32, "sdpa_available": hasattr(torch.nn.functional, "scaled_dot_product_attention")}
except Exception as e:
    out["torch"] = {"error": repr(e)}
try:
    import onnxruntime as ort
    out["onnxruntime_providers"] = ort.get_available_providers()
except Exception as e:
    out["onnxruntime_providers"] = repr(e)
# data source / lineage (hash only small metadata files; the big tree is verified by the existing frozen verifier record)
v = json.loads((D / "VERSION.json").read_text())
out["dataset"] = {"root": str(D), "exists": D.exists(), "version": v.get("data_version"), "version_hash": v.get("version_hash"),
                  "source_sha256": v.get("components", {}).get("source_sha256"),
                  "metadata_sha256": {f: sha(D / f) for f in ("VERSION.json", "FILES.json", "splits/splits.json", "mappings/catalogue.parquet",
                                                               "mappings/clients.parquet", "mappings/normalizers.json", "source_manifest.json",
                                                               "protocol.resolved.yaml") if (D / f).exists()},
                  "files_json_entries": len(json.loads((D / "FILES.json").read_text()).get("files", [])) if (D / "FILES.json").exists() else None,
                  "last_frozen_verification": str(R / ".local_ai/sasrec_v1/state/verify_frozen_before_retro.json")}
vf = R / ".local_ai/sasrec_v1/state/verify_frozen_before_retro.json"
if vf.exists():
    j = json.loads(vf.read_text())
    out["dataset"]["last_frozen_verification_summary"] = {k: j.get(k) for k in ("all_ok", "utc", "checked", "n_files", "version_hash") if k in j}
src = json.loads((D / "source_manifest.json").read_text()) if (D / "source_manifest.json").exists() else {}
out["raw_source_manifest_excerpt"] = json.dumps(src)[:3000]
raw_candidates = [p for p in (ppsi_paths.SHARED_DATA_ROOT, R / "data", Path.home() / "data") if p.exists()]
out["raw_source_search"] = {str(p): sorted(str(x.relative_to(p)) for x in p.rglob("*.csv*"))[:40] for p in raw_candidates}
# prior workspaces present (read-only inventory)
out["workspaces"] = sorted(x.name for x in (R / ".local_ai").iterdir() if x.is_dir())
out["repo_git_head"] = sh(["git", "-C", str(R), "log", "--oneline", "-1"]).get("out")
out["repo_git_status_short_count"] = len((sh(["git", "-C", str(R), "status", "--short"]).get("out") or "").splitlines())
Path(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True)
tmp = Path(sys.argv[1] + ".tmp")
tmp.write_text(json.dumps(out, indent=1))
tmp.replace(sys.argv[1])
print(json.dumps({"ok": True, "gpu_apps": len(apps), "disk_free_home_gib": out["disk"].get(str(Path.home()), {}).get("free"), "versions": vers}))
