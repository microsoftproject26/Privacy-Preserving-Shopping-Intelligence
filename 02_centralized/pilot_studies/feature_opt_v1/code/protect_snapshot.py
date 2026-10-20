"""Read-only protection snapshot for SASREC_FEATURE_OPT_V1: hashes of active campaign code/config/locks
(no heartbeat/log/state/run files), dataset metadata identity (no full data hashing), process identities."""
import hashlib, json, os, subprocess, sys, time
from pathlib import Path
R = Path("/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence")
SAS, GRU, D = R / ".local_ai/sasrec_v1", R / ".local_ai/gru_v1", Path("/home/jovyan/shared_data/rees46_nextitem_v1")
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
out = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "label": sys.argv[1] if len(sys.argv) > 1 else "snapshot", "files": {}}
for root, subs in ((SAS, ("code", "tools", "config", "review", "locks", "bench")), (GRU, ("code", "locks", "config"))):
    for s in subs:
        base = root / s
        if not base.exists(): continue
        for p in sorted(base.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and p.stat().st_size < 64 << 20:
                out["files"][str(p.relative_to(R))] = sha(p)
out["dataset_metadata"] = {p.name: sha(p) for p in sorted(D.iterdir()) if p.is_file()}
out["dataset_tree"] = {"n_files": sum(1 for _ in D.rglob("*") if _.is_file()),
                       "bytes": sum(p.stat().st_size for p in D.rglob("*") if p.is_file()),
                       "max_mtime": max(p.stat().st_mtime for p in D.rglob("*") if p.is_file())}
ps = subprocess.run(["ps", "-eo", "pid,etimes,rss,args"], capture_output=True, text=True).stdout.splitlines()
out["processes"] = [l.strip()[:300] for l in ps if any(k in l for k in ("sasrec", "gru_v1", "campaign_v3b", "supervise", "train_sasrec", "continue_cooldown")) and "ps -eo" not in l]
tm = subprocess.run(["tmux", "ls"], capture_output=True, text=True).stdout.splitlines()
out["tmux"] = tm
out["disk_free_bytes"] = os.statvfs(str(R)).f_bavail * os.statvfs(str(R)).f_frsize
try: out["cgroup_mem_current"] = int(Path("/sys/fs/cgroup/memory.current").read_text()); out["cgroup_mem_max"] = Path("/sys/fs/cgroup/memory.max").read_text().strip()
except Exception as e: out["cgroup_err"] = repr(e)
dst = R / ".local_ai/feature_opt_v1/state" / f"PROTECT_{out['label']}.json"
dst.write_text(json.dumps(out, indent=1))
print(dst, len(out["files"]), "files;", len(out["processes"]), "procs;", "free GiB", round(out["disk_free_bytes"] / 2**30, 2), "mem GiB", round(out.get("cgroup_mem_current", 0) / 2**30, 1))
