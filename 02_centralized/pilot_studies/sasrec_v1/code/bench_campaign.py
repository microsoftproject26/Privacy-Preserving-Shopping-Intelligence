"""Post-Gate-C real-data systems benchmark for the v2 plan (owner: Opus). 07 section 5.

Runs the two production arms CONCURRENTLY (A0_ID256 on GPU0, A1_RICH256 on GPU1), exactly
as the campaign will run them, in the production training precision (TF32 if the parity
gate passed, else strict FP32), 1000 measured updates each on isolated fresh-init
benchmark states (no run dirs, no checkpoints; discarded). Records throughput, step
times, short/long-context steps, strict-FP32 eval speed, peak VRAM, cgroup memory and the
projected finish against the owner deadline. Writes bench/SYSTEMS_LOCK.json. Systems
numbers only: nothing here reads a validation score.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import paths

PY = os.path.expanduser("~/venvs/ppsi-gpu/bin/python")
TRAINER = str(paths.CODE / "train_sasrec.py")
BENCH = paths.S / "bench"
UPDATES = 1000


def main() -> int:
    parity = json.loads((BENCH / "TF32_PARITY.json").read_text())
    mode = "tf32" if parity.get("pass") is True else "fp32"
    reg = json.loads(paths.REGISTRY_LOCKED.read_text())
    planned = int(reg["schedule"]["planned_epochs"]) if mode == "tf32" else 5
    t = time.time()
    procs = {}
    for rid, gpu in (("A0_ID256", 0), ("A1_RICH256", 1)):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="3", MKL_NUM_THREADS="3",
               OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
        fh = open(BENCH / f"log_pair_{rid}.txt", "w")
        procs[rid] = subprocess.Popen([PY, TRAINER, "--run-id", rid, "--seed", "2026", "--benchmark-updates",
                                       str(UPDATES), "--precision-mode", mode], env=env, stdout=fh, stderr=subprocess.STDOUT)
    peak_mem = 0.0
    while any(p.poll() is None for p in procs.values()):
        try:
            peak_mem = max(peak_mem, int(open("/sys/fs/cgroup/memory.current").read()) / 2**30)
        except Exception:
            pass
        time.sleep(5)
    rcs = {k: p.returncode for k, p in procs.items()}
    if any(rcs.values()):
        raise SystemExit(f"benchmark exit codes {rcs}")
    res = {}
    for rid, gpu in (("A0_ID256", 0), ("A1_RICH256", 1)):
        files = [f for f in glob.glob(str(BENCH / f"bench_{rid}_s2026_*_gpu{gpu}_*.json")) if os.path.getmtime(f) >= t]
        res[rid] = json.loads(Path(max(files, key=os.path.getmtime)).read_text())
    deadline = datetime.fromisoformat(reg["owner_deadline"]["deadline_utc"].replace("Z", "+00:00")).timestamp()
    n_eval = len(range(3, planned + 1))
    proj = {}
    for rid, r in res.items():
        tr = paths.N_TRAIN / r["representative"]["wall_decisions_per_s"]
        ev = paths.N_VS / (r["eval_rows_per_s_strict_fp32"] or 1)
        diag = 112_365 / (r["eval_rows_per_s_strict_fp32"] or 1) * (planned + 6)
        proj[rid] = {"train_seconds_per_epoch": tr, "eval_seconds_full_vs": ev,
                     "total_hours": (planned * tr + n_eval * ev + diag) / 3600}
    slowest = max(p["total_hours"] for p in proj.values())
    finish = time.time() + 600 + slowest * 3600          # ~10 min until launch
    out = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "precision_training": mode,
           "planned_epochs": planned, "eval_epochs": list(range(3, planned + 1)), "concurrent": True,
           "results": res, "projection": proj, "cgroup_memory_peak_gib": peak_mem,
           "projected_training_end_utc": datetime.fromtimestamp(finish, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "projected_report_end_utc": datetime.fromtimestamp(finish + 2 * 3600, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "deadline_utc": reg["owner_deadline"]["deadline_utc"],
           "slack_hours": (deadline - (finish + 2 * 3600)) / 3600,
           "microbatch": 256,
           "note": "concurrent pair measured as it will run; microbatch 256 kept (larger GEMMs are more efficient on A16, synthetic B256/512/1024 trend)"}
    st = os.statvfs(str(paths.S))
    out["disk_free_gib"] = st.f_bavail * st.f_frsize / 2**30
    (BENCH / "SYSTEMS_LOCK.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != "results"}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
