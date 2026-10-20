"""fo_queue.py (owner: Opus) — detached supervisor for the feature study's approved queue (no LLM polling).

Runs in tmux `fo-queue`. It never touches campaign or teammate processes and never deletes anything.

Start condition (checked every 5 min):
- the campaign's state/FINAL_STAGE_DONE exists;
- no campaign trainer / cooldown / final-stage process is alive;
- the GPU(s) a stage needs have NO compute process at all (ours, the campaign's or a teammate's);
- free disk >= 8 GiB floor + the stage's peak.

Otherwise the status is QUEUED_RESOURCE and the queue waits.

Stages (each at most 2 attempts; a second failure stops the queue with state FAILED):
  T1 protect snapshot -> stream_fit (exact TRAIN pass) -> full profile -> pin_baseline -> probes (+lean parity)
  -> screen CONTROL || C1_D3 (GPU0 || GPU1) -> C2 (only if gate 2(a) named one) -> finalize (gate 2(b)) -> T2 snapshot.
Status: state/fo_status.json. Log: logs/fo_queue.log.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

F = Path("/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/feature_opt_v1")
CAMPAIGN = Path("/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/sasrec_v1")
PY = str(Path.home() / "venvs/ppsi-gpu/bin/python")
STATUS, LOG = F / "state" / "fo_status.json", F / "logs" / "fo_queue.log"
FLOOR_GIB = 8.0
CAMPAIGN_PROCS = ("sasrec_v1/code/train_sasrec.py", "sasrec_v1/code/continue_cooldown.py",
                  "sasrec_v1/code/lock_retro_v3b.py", "sasrec_v1/code/render_report.py", "sasrec_v1/tools/final_stage.sh")
ENV = dict(os.environ, OMP_NUM_THREADS="3", MKL_NUM_THREADS="3", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
           CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONUNBUFFERED="1")


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(msg: str) -> None:
    with open(LOG, "a") as f:
        f.write(f"[{utc()}] {msg}\n")


def status(**kw) -> dict:
    s = json.loads(STATUS.read_text()) if STATUS.exists() else {"created_utc": utc(), "stages": {}}
    s.update(kw); s["updated_utc"] = utc()
    tmp = STATUS.with_suffix(".tmp"); tmp.write_text(json.dumps(s, indent=1)); tmp.replace(STATUS)
    return s


def free_gib() -> float:
    return shutil.disk_usage(str(F)).free / 2**30


def campaign_busy() -> list[str]:
    out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout.splitlines()
    return [l for l in out if any(p in l for p in CAMPAIGN_PROCS)]


def gpu_busy() -> dict:
    """{gpu_index: [pids]} of every compute process, whoever owns it. FAIL-CLOSED (Review C fix):
    an nvidia-smi error, an unparsable line or an app on an unmapped bus marks EVERY GPU busy."""
    all_busy = {"0": ["unknown"], "1": ["unknown"]}
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"], capture_output=True, text=True, timeout=60)
        a = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_bus_id,pid", "--format=csv,noheader"], capture_output=True, text=True, timeout=60)
    except Exception:
        return all_busy
    if q.returncode != 0 or a.returncode != 0:
        return all_busy
    try:
        bus = {l.split(",")[1].strip(): l.split(",")[0].strip() for l in q.stdout.strip().splitlines() if l.strip()}
    except IndexError:
        return all_busy
    if not bus:
        return all_busy
    busy = {i: [] for i in bus.values()}
    for l in a.stdout.strip().splitlines():
        if not l.strip():
            continue
        parts = [x.strip() for x in l.split(",")]
        if len(parts) != 2 or parts[0] not in bus:
            return {i: ["unmapped"] for i in set(bus.values()) | set(all_busy)}
        busy[bus[parts[0]]].append(parts[1])
    return busy


def ready(gpus: tuple, peak_gib: float) -> tuple[bool, str]:
    if not (CAMPAIGN / "state" / "FINAL_STAGE_DONE").exists():
        return False, "campaign FINAL_STAGE_DONE absent"
    cb = campaign_busy()
    if cb:
        return False, f"campaign processes alive: {len(cb)}"
    busy = gpu_busy()
    taken = [g for g in gpus if busy.get(str(g))]
    if taken:
        return False, f"GPU(s) {taken} have compute processes {[busy[str(g)] for g in taken]}"
    fg = free_gib()
    if fg < FLOOR_GIB + peak_gib:
        return False, f"disk free {fg:.2f} GiB < floor {FLOOR_GIB} + peak {peak_gib}"
    return True, "ok"


def run(name: str, argv: list, gpu=None, log_name=None) -> int:
    env = dict(ENV)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu); env["FO_DEVICE"] = "cuda:0"
    lf = open(F / "logs" / (log_name or f"{name}.log"), "a")
    lf.write(f"\n[{utc()}] START {argv} gpu={gpu}\n"); lf.flush()
    t0 = time.time()
    p = subprocess.Popen([PY] + argv, cwd=str(F / "code"), env=env, stdout=lf, stderr=subprocess.STDOUT)
    return p, t0, lf


def stage(name: str, argv: list, gpus: tuple = (), peak_gib: float = 0.2, parallel: list | None = None) -> bool:
    s = status()
    if s["stages"].get(name, {}).get("state") == "DONE":
        return True
    for attempt in (1, 2):
        while True:
            ok, why = ready(gpus, peak_gib)
            if ok:
                break
            status(state="QUEUED_RESOURCE", waiting_for=name, reason=why)
            time.sleep(300)
        status(state="RUNNING", current=name, reason=None)
        log(f"{name} attempt {attempt} start")
        jobs = parallel or [(name, argv, gpus[0] if gpus else None)]
        procs = [(n, *run(n, a, g)) for n, a, g in jobs]
        rcs = {}
        for n, p, t0, lf in procs:
            rcs[n] = p.wait(); lf.write(f"[{utc()}] END rc={rcs[n]} wall={time.time() - t0:.1f}s\n"); lf.close()
        good = all(rc == 0 for rc in rcs.values())
        st = json.loads(STATUS.read_text())
        st["stages"][name] = {"state": "DONE" if good else "FAILED", "attempt": attempt, "rcs": rcs, "end_utc": utc(),
                              "disk_free_gib": round(free_gib(), 2)}
        STATUS.write_text(json.dumps(st, indent=1))
        log(f"{name} attempt {attempt} rcs={rcs}")
        if good:
            return True
    status(state="FAILED", failed_stage=name)
    return False


def main() -> int:
    (F / "logs").mkdir(exist_ok=True); (F / "state").mkdir(exist_ok=True)
    log("fo_queue start")
    if not (F / "review" / "FEATURE_CODE_REVIEW_PASS.json").exists():
        status(state="BLOCKED", reason="no Fable Review C pass file"); log("blocked: no review pass"); return 2
    plan = [
        ("T1_protect", ["protect_snapshot.py", "T1_campaign_done"], (), 0.1, None),
        ("T1_diff", ["protect_diff.py", "T0_study_start", "T1_campaign_done"], (), 0.0, None),
        ("stream_fit", ["stream_fit.py", "default", "all"], (), 0.2, None),
        ("full_profile", ["profile_train_sample.py", "--dataset-root", "/home/jovyan/shared_data/rees46_nextitem_v1",
                          "--buckets", ",".join(str(i) for i in range(64)), "--splits", "TRAIN,VALIDATION_SELECT",
                          "--out", str(F / "audit" / "TRAIN_PROFILE_FULL.json")], (), 0.2, None),
        ("pin_baseline", ["pin_baseline.py"], (), 0.1, None),
        ("probes", ["probe_runner.py"], (0,), 0.5, None),
        ("screen_control_c1", None, (0, 1), 3.0, [("screen_CONTROL", ["screen.py", "--arm", "CONTROL"], 0),
                                               ("screen_C1_D3", ["screen.py", "--arm", "C1_D3"], 1)]),
    ]
    for name, argv, gpus, peak, par in plan:
        if not stage(name, argv, gpus, peak, par):
            return 1
    c2 = json.loads((F / "state" / "C2_DECISION.json").read_text())
    if c2.get("C2"):
        if not stage("screen_C2", ["screen.py", "--arm", "C2"], (0,), 1.5):
            return 1
    else:
        status(c2_skipped="no probe passed gate 2(a)")
    if not stage("finalize", ["finalize_fo.py"], (), 0.1):
        return 1
    stage("T2_protect", ["protect_snapshot.py", "T2_study_done"], (), 0.1)
    stage("T2_diff", ["protect_diff.py", "T0_study_start", "T2_study_done"], (), 0.0)
    fin = json.loads((F / "state" / "FINAL_DECISION.json").read_text())
    status(state="PAIR_REQUIRED" if fin.get("finalist") else "DONE", final=fin.get("decision"))
    (F / "state" / "FO_QUEUE_DONE").write_text(utc())
    log(f"fo_queue done: {fin.get('decision')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
