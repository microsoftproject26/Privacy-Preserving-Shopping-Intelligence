"""event_probe.py (owner: Opus): one-shot, read-only campaign status probe for the local event watcher.

Prints one JSON line: a stable `sig` (changes only on campaign events: job status, gates, cooldown
decision, final stage) plus `alarm` reasons (orchestrator dead, disk below the 10 GiB target, a job
STOPPED/FAILED, a stale heartbeat) and a few progress numbers. Never writes anything.
"""
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

S = Path("/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/sasrec_v1")


def jload(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


st = jload(S / "state" / "v3b_state.json") or {}
jobs = {k: {"status": v.get("status"), "gate_go": (v.get("gate") or {}).get("go"), "e5_vs_e4": v.get("e5_vs_e4")}
        for k, v in (st.get("jobs") or {}).items()}
runs = {}
alarm = []
now = time.time()
for d in sorted((S / "runs").glob("*/status.json")):
    s = jload(d) or {}
    name = d.parent.name
    if s.get("state") in ("RUNNING",):
        age = now - d.stat().st_mtime
        if age > 3600:
            alarm.append(f"{name} heartbeat stale {int(age)}s")
    if s.get("state") in ("FAILED", "ERROR") or (d.parent / "SUPERVISOR_STOPPED").exists():
        alarm.append(f"{name} {s.get('state')} stopped={(d.parent / 'SUPERVISOR_STOPPED').exists()}")
    runs[name] = {"state": s.get("state"), "update": s.get("update"),
                  "eta": (s.get("eta") or {}).get("training_end_utc")}
alive = subprocess.run(["pgrep", "-f", "tools/campaign_v3b.py"], capture_output=True).returncode == 0
final_stage_started = bool(st.get("final_stage"))
fs_alive = subprocess.run(["tmux", "has-session", "-t", "sasrec-final"], capture_output=True).returncode == 0
if not alive and not (S / "state" / "FINAL_STAGE_DONE").exists() and not final_stage_started:
    alarm.append("orchestrator not running")          # the orchestrator exits by design after launching the final stage
if final_stage_started and not fs_alive and not (S / "state" / "FINAL_STAGE_DONE").exists():
    alarm.append("final stage session ended without FINAL_STAGE_DONE")
free = shutil.disk_usage(os.path.expanduser("~")).free / 2**30
if free < 10:
    alarm.append(f"disk free {free:.1f} GiB < 10")
final_done = (S / "state" / "FINAL_STAGE_DONE").exists()
retro_done = (S / "locks" / "RETRO_DONE.json").exists()
locked = (S / "locks" / "SASREC_LOCK.json").exists()
FO = S.parent / "feature_opt_v1"
fo = jload(FO / "state" / "fo_status.json") or {}
fo_sig = {"state": fo.get("state"), "waiting_for": fo.get("waiting_for"), "final": fo.get("final"),
          "stages": {k: v.get("state") for k, v in (fo.get("stages") or {}).items()}}
if fo.get("state") in ("FAILED", "BLOCKED"):
    alarm.append(f"feature study {fo.get('state')}: {fo.get('failed_stage') or fo.get('reason')}")
if fo and subprocess.run(["tmux", "has-session", "-t", "fo-queue"], capture_output=True).returncode != 0 \
        and fo.get("state") not in ("DONE", "PAIR_REQUIRED", "FAILED", "BLOCKED"):
    alarm.append("feature study supervisor (tmux fo-queue) not running")
sig = json.dumps({"fo": fo_sig, "jobs": jobs, "go": st.get("cooldowns_go"), "final_stage": bool(st.get("final_stage")),
                  "locked": locked, "retro_done": retro_done, "final_done": final_done,
                  "run_states": {k: ("RUNNING" if v["state"] == "EVALUATING" else v["state"]) for k, v in runs.items()}}, sort_keys=True)
print(json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "sig": sig, "alarm": alarm,
                  "free_gib": round(free, 1), "runs": runs}))
