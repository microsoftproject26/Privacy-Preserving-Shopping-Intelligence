"""X2 scheduler (DL 553 A1 + A2(b); POST_HOC_EXPLORATORY; inner validation only; never a TEST look).

  python -B tools_x2/sched_x2.py --slots 24 --threads 4 [--dry-run] [--only A1|A2B]
Started by company/plans/ops_scripts/bench/run_bench_x2.sh (preflight: sha256 sums, no `final` running, one driver).
Reads runs_bench_x/x2_state/PLAN_X2.json (static job list, write-once). CPU only: every job gets CUDA_VISIBLE_DEVICES=""
and nice -n 19. A job starts when its dependencies (the PRE run) are DONE_MAIN; the plan order (longest chain first)
decides who goes first. Every live `v4bench.run run` process writing under runs_bench_x (also the POST_HOC_X C_SMALL jobs
of DL 555, started elsewhere) holds a slot, so slots x threads bounds the whole benchmark CPU use (owner cap 96 threads).
Relaunching resumes: DONE_MAIN runs are skipped, a run with a checkpoint restarts with --resume, a start without a
checkpoint is set aside as <dir>__STALE_<utc>, a run still alive is not started twice. A job that fails twice is left
alone (FAILED in STATUS.json); its dependants then wait and the scheduler stops when nothing else can run.
GPU lane (opt-in, owner 2026-10-09): --gpu-jobs <json list of run dirs> --gpu N --gpu-max K launches exactly those jobs,
when ready, with `--device cuda --gpu N` (class NVIDIA_A40-FP32-DET, written into the spec by the engine); they hold no
CPU slot (live `--device cuda` processes are not counted), at most K at once. A listed job that already has a CPU
checkpoint is never moved (it stays on CPU).
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
X2 = os.path.dirname(HERE)
V4 = os.path.dirname(X2)
RX = os.path.join(V4, "runs_bench_x")
STATE = os.path.join(RX, "x2_state")
LOGD = os.path.join(STATE, "logs")
PY = "/home/jovyan/venvs/ppsi-gpu/bin/python"
MAX_TRIES = 2
MAX_THREADS = 112              # owner 2026-10-09 ("use the server fully"; DL 565+): was 96


def utc():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def say(msg):
    line = f"[{utc()}] {msg}"
    print(line, flush=True)
    with open(os.path.join(STATE, "scheduler.log"), "a") as f:
        f.write(line + "\n")


def _read(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def done(run_dir) -> bool:
    r = _read(os.path.join(run_dir, "RESULT.json"))
    return bool(r) and r.get("status") == "DONE_MAIN"


def live_cmdlines() -> list:
    out = []
    for p in glob.glob("/proc/[0-9]*/cmdline"):
        try:
            out.append(open(p, "rb").read().replace(b"\0", b" ").decode("utf-8", "replace"))
        except OSError:
            pass
    return out


def has_checkpoint(run_dir) -> bool:
    return bool(glob.glob(os.path.join(run_dir, "**", "latest.pt"), recursive=True))



def in_x2(path) -> bool:
    parts = os.path.abspath(path).split(os.sep)
    return "runs_bench_x" in parts and parts[parts.index("runs_bench_x") + 1].startswith("x2_")


def short(rd) -> str:
    """<runs root>/<dataset>/<run> -> '<x2 root>/<dataset>/<run>' (basenames repeat across datasets)."""
    parts = os.path.abspath(rd).split(os.sep)
    return "/".join(parts[-3:])


def launch(j, threads, gpu=None):
    rd = j["run_dir"]
    if not in_x2(j["runs_root"]) or not in_x2(rd):
        raise SystemExit(f"REFUSED: {rd} is outside runs_bench_x/x2_*")
    extra = [] if gpu is None else ["--device", "cuda", "--gpu", str(int(gpu))]
    if os.path.isdir(rd):
        if has_checkpoint(rd):
            extra = extra + ["--resume"]
        else:                                                # a start that never reached a checkpoint: set aside
            os.rename(rd, f"{rd}__STALE_{utc().replace(':', '')}")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", BM_ALLOW_THREADS="1", OMP_NUM_THREADS=str(threads),
               MKL_NUM_THREADS=str(threads))
    name = f"{j['study']}__{j['dataset']}__{j['label']}_s{j['seed']}"
    log = open(os.path.join(LOGD, name + ".log"), "a")
    cmd = ["nice", "-n", "19", PY, "-B", "-m", "v4bench.run", "run"] + j["argv"] + extra + \
          ["--threads", str(threads), "--runs-root", j["runs_root"]]
    p = subprocess.Popen(cmd, cwd=X2, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    say(f"start pid {p.pid} {name} {' '.join(extra)}")
    return p


def is_cpu_bench(c: str) -> bool:
    return c.startswith(PY) and " -m v4bench.run run " in c and "runs_bench_x" in c and "--device cuda" not in c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=int, default=24)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--only", default=None, choices=("A1", "A2B"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--gpu-jobs", default=None, help="json list of run dirs that run on the GPU lane")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--gpu-max", type=int, default=2)
    a = ap.parse_args()
    gpu_dirs = set(json.load(open(a.gpu_jobs))) if a.gpu_jobs else set()
    if gpu_dirs and a.gpu is None:
        print("REFUSED: --gpu-jobs needs --gpu", file=sys.stderr)
        return 2
    if a.slots * a.threads > MAX_THREADS:
        print(f"REFUSED: at most {MAX_THREADS} CPU threads (slots x threads)", file=sys.stderr)
        return 2
    os.makedirs(LOGD, exist_ok=True)
    plan = json.load(open(os.path.join(STATE, "PLAN_X2.json")))
    jobs = sorted(plan["jobs"], key=lambda j: j["priority"])
    if a.only:
        jobs = [j for j in jobs if j["study"] == a.only]
    running, tries, on_gpu = {}, {}, set()
    say(f"sched_x2 start slots={a.slots} threads={a.threads} jobs={len(jobs)} only={a.only} dry={a.dry_run}")
    while True:
        for rd, p in list(running.items()):
            rc = p.poll()
            if rc is not None:
                del running[rd]
                ok = rc == 0 and done(rd)
                if not ok:
                    tries[rd] = tries.get(rd, 0) + 1
                say(f"end pid {p.pid} rc={rc} {'DONE' if ok else 'FAILED'} {short(rd)}")
        live = live_cmdlines()
        paused = any(" -m v4bench.run final" in c for c in live if c.startswith(PY) or "python" in c.split(" ")[0])
        if paused:
            say("a 'v4bench.run final' process is running: no new launch this round")
        occupied = sum(1 for c in live if is_cpu_bench(c))
        gpu_live = sum(1 for c in live if c.startswith(PY) and " -m v4bench.run run " in c and "--device cuda" in c
                       and "runs_bench_x" in c)
        todo, blocked, orphans, n_done, failed = [], 0, 0, 0, []
        for j in jobs:
            rd = j["run_dir"]
            if done(rd):
                n_done += 1
                continue
            if rd in running:
                continue
            if tries.get(rd, 0) >= MAX_TRIES:
                failed.append(short(rd))
                continue
            key = "-m v4bench.run run " + " ".join(j["argv"])
            if any(key in c for c in live):
                orphans += 1                                 # started by an earlier scheduler, still alive
                continue
            if not all(done(d) for d in j["deps"]):
                blocked += 1
                continue
            todo.append(j)
        if a.dry_run:
            for j in todo:
                say(f"DRY ready p{j['priority']} {j['study']} {j['dataset']}/{j['label']}_s{j['seed']} est {j['est_h']} h")
            say(f"DRY {len(todo)} ready, {blocked} waiting on inputs, {n_done} done, {orphans} alive elsewhere, "
                f"occupied slots {occupied}")
            return 0
        gpu_todo = [j for j in todo if j["run_dir"] in gpu_dirs and not has_checkpoint(j["run_dir"])]
        todo = [j for j in todo if j not in gpu_todo]
        while gpu_todo and not paused and gpu_live < a.gpu_max:
            j = gpu_todo.pop(0)
            running[j["run_dir"]] = launch(j, a.threads, gpu=a.gpu)
            on_gpu.add(j["run_dir"])
            gpu_live += 1
            time.sleep(2)
        n_cpu_running = sum(1 for rd in running if rd not in on_gpu)
        occupied = max(occupied, n_cpu_running + orphans)
        while todo and not paused and occupied < a.slots:
            j = todo.pop(0)
            running[j["run_dir"]] = launch(j, a.threads)
            occupied += 1
            time.sleep(2)
        todo += gpu_todo
        st = {"utc": utc(), "done": n_done, "of": len(jobs), "running": {short(k): v.pid
                                                                         for k, v in running.items()},
              "alive_elsewhere": orphans, "ready_not_started": len(todo), "waiting_on_inputs": blocked,
              "failed": failed, "occupied_slots": occupied, "slots": a.slots,
              "on_gpu": sorted(short(k) for k in running if k in on_gpu), "gpu_jobs_listed": len(gpu_dirs)}
        tmp = os.path.join(STATE, f"STATUS.json.tmp.{os.getpid()}")
        json.dump(st, open(tmp, "w"), indent=1)
        os.replace(tmp, os.path.join(STATE, "STATUS.json"))
        if not running and not orphans and not todo:
            if blocked == 0 and not failed:
                say("sched_x2: nothing left to run - DONE")
                return 0
            say(f"sched_x2: {blocked} jobs wait on inputs that cannot be produced, {len(failed)} failed - stopping")
            return 1
        time.sleep(a.poll)


if __name__ == "__main__":
    raise SystemExit(main())
