"""POST_HOC_X LEAN pipeline scheduler (owner plan 2026-10-06; EXPLORATORY; inner validation only; never a TEST look).

  python -B tools_x/lean_x.py --slots 24 --threads 4 [--datasets "..."] [--dry-run]
Started by company/plans/ops_scripts/bench/run_bench_x.sh (which does the sitting / thread / sha preflight). CPU only:
every job gets CUDA_VISIBLE_DEVICES="" and nice -n 19. Event-driven: a job starts as soon as its inputs exist.

Methodology (REES46, unchanged): FL model SASREC_D64_B2; all users as clients; selection on inner-validation NDCG@10
(PRACTICAL_BEST; CONTROLLED_BUDGET reported); theta_pre = the dataset's best-validation PRE (read-only from runs_bench /
runs_bench_gpu; amazon2023 has no B96 PRE: PRE B96 D0.2 is trained here); seeds 2026 + 2027; R6: seed 2028 when the
two seeds' values differ by more than 0.002. Only the FL training procedure is tuned (seed 2026):
  FA       tau 1e-6, server LR {1e-3, 3e-3}, client LR 0.1 (registered), 1 local pass (registered), B24
  FA_WARM  tau 1e-6, server LR {3e-4, 1e-3}, client LR 0.025 (registered S7), 1 local pass, B24, unfrozen
Each cell runs ONCE at B24 (FL_CONST: the B24 curve carries 12 marks every 2 EFE). Per (dataset, arm), once both cells
are done, the better one is selected; if its best mark is the last (24) it is extended to B96 on the S3 datasets only
(a fresh run - the FL_CONST round schedule depends on T, so a B24 state is not a B96 prefix; amazon2023 / ML-1M: no
B96, it does not fit the 3-day window); the final cell then runs at seed 2027 (and 2028 by R6).
Dropped: FA_WARM_FROZEN (the frozen item tables cannot learn items outside the PRE band: MEASURED validation NDCG@10 =
0.0 for the BAND_UNCOVERED_TARGET users); tau 1e-3 / 1e-4 cells; a passes = 2 cell (no measured evidence that local
steps limit FL here; the measured limiter is FedAdam's tau); C_FULL / C_SMALL (central side is final).
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import xjobs  # noqa: E402

X = xjobs.CODE
RX = xjobs.RX_DEFAULT
PY = "/home/jovyan/venvs/ppsi-gpu/bin/python"
ORDER = ("amazon2023_baby_products", "amazon2023_video_games", "ml_1m", "s3_sports", "s3_beauty", "s3_toys")
S3 = ("s3_beauty", "s3_sports", "s3_toys")
CELLS = {"FA": [("1e-6", "0.001", "0.1"), ("1e-6", "0.003", "0.1")],
         "FA_WARM": [("1e-6", "0.0003", "0.025"), ("1e-6", "0.001", "0.025")]}
# owner follow-up 2026-10-07 (ML-1M FA_WARM 0.1135 < FA 0.1225 at the registered warm client LR 0.025, 4x below the
# cold 0.1): three extra FA_WARM cells on ML-1M and the S3 datasets. They are listed AFTER the incumbent cells, so within
# 0.0002 the incumbent keeps the selection (a new cell must beat the current winner to get seed 2027).
EXTRA_WARM = [("1e-6", "0.001", "0.1"), ("1e-6", "0.003", "0.1"), ("1e-6", "0.001", "0.3")]
EXTRA_WARM_DS = ("ml_1m",) + S3
AMAZON_PRE = xjobs.pre_cell(96, 0.2)
TOL, R6_GAP, MAX_TRIES = xjobs.TOL, xjobs.R6_GAP, 2
LOGD = os.path.join(RX, "logs", "lean")
STATE = os.path.join(RX, "lean")
FINAL_RE = re.compile(r"python[0-9.]*( -[A-Za-z]+)* -m v4bench\.run final")


def utc():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def say(msg):
    line = f"[{utc()}] {msg}"
    print(line, flush=True)
    with open(os.path.join(STATE, "scheduler.log"), "a") as f:
        f.write(line + "\n")


def done(run_dir) -> bool:
    r = xjobs._read(os.path.join(run_dir, "RESULT.json"))
    return bool(r) and r.get("status") == "DONE_MAIN"


def pre_of(ds):
    return AMAZON_PRE if ds.startswith("amazon2023") else xjobs.best_pre(ds, RX)


def is_x(path) -> bool:
    return "runs_bench_x" in os.path.abspath(path).split(os.sep)


def wanted(ds) -> tuple:
    """([(job, [dependency run dirs])], selection record) for the dataset's current state."""
    out, sel = [], {"dataset": ds, "utc": utc(), "arms": {}}
    pre = pre_of(ds)
    sel["theta_pre"] = pre
    if pre is None:
        say(f"{ds}: no PRE found - warm arm skipped")

    def pre_jobs(seed):
        if pre is None:
            return []
        d = xjobs.pre_dir(pre, ds, seed, RX)
        if is_x(d):
            out.append((xjobs.job(pre, ds, seed, RX), []))
            return [d]
        return []

    deps = {s: pre_jobs(s) for s in (2026, 2027)}
    for arm, cells in CELLS.items():
        if arm == "FA_WARM" and pre is None:
            continue
        warm = arm == "FA_WARM"
        dep = (lambda s: deps.get(s, []) if warm else [])
        n0 = len(cells)
        cells = list(cells) + (EXTRA_WARM if (warm and ds in EXTRA_WARM_DS) else [])
        cs = [xjobs.fl(arm, 24, t, s, c, pre=pre if warm else None) for t, s, c in cells]
        for c in cs:
            out.append((xjobs.job(c, ds, 2026, RX), dep(2026)))
        res = [xjobs.result_of(xjobs.job(c, ds, 2026, RX)["run_dir"]) for c in cs]
        rec = {"cells": [{"label": xjobs.job(c, ds, 2026, RX)["label"], "result": r} for c, r in zip(cs, res)]}
        sel["arms"][arm] = rec
        if any(r is None for r in res):
            continue
        best_v = max(r["value"] for r in res)                    # within 0.0002 of the best: an incumbent cell first,
        near = [i for i in range(len(cs)) if res[i]["value"] >= best_v - TOL]     # then the smaller server / client LR
        best_i = min(near, key=lambda i: (i >= n0, float(cs[i]["slr"]), float(cs[i]["clr"])))
        c24, r24 = cs[best_i], res[best_i]
        rec["selected_b24"] = xjobs.job(c24, ds, 2026, RX)["label"]
        final = c24
        if ds in S3 and r24["best_mark"] >= 24.0:                   # still improving at its last mark
            c96 = xjobs.with_(c24, budget=96)
            j96 = xjobs.job(c96, ds, 2026, RX)
            out.append((j96, dep(2026)))
            r96 = xjobs.result_of(j96["run_dir"])
            rec["b96"] = {"label": j96["label"], "result": r96}
            if r96 is None:
                continue
            final = c96 if r96["value"] > r24["value"] + TOL else c24
        jf = {s: xjobs.job(final, ds, s, RX) for s in (2026, 2027, 2028)}
        rec["final"] = jf[2026]["label"]
        out.append((jf[2027], dep(2027)))
        v26, r27 = xjobs.result_of(jf[2026]["run_dir"]), xjobs.result_of(jf[2027]["run_dir"])
        rec["seeds"] = {"2026": v26, "2027": r27}
        if r27 is not None:
            gap = abs(v26["value"] - r27["value"])
            rec["r6_gap"], rec["r6_third_seed"] = gap, gap > R6_GAP
            if gap > R6_GAP:
                d28 = pre_jobs(2028) if warm else []
                out.append((jf[2028], d28))
                rec["seeds"]["2028"] = xjobs.result_of(jf[2028]["run_dir"])
    return out, sel


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


def launch(j, threads):
    rd = j["run_dir"]
    if not is_x(j["runs_root"]) or not is_x(rd):
        raise SystemExit(f"REFUSED: {rd} is outside runs_bench_x")
    extra = []
    if os.path.isdir(rd):
        if has_checkpoint(rd):
            extra = ["--resume"]
        else:                                                # a start that never reached a checkpoint: set aside
            os.rename(rd, f"{rd}__STALE_{utc().replace(':', '')}")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", BM_ALLOW_THREADS="1", OMP_NUM_THREADS=str(threads),
               MKL_NUM_THREADS=str(threads))
    name = f"{j['dataset']}__{j['label']}_s{j['seed']}"
    log = open(os.path.join(LOGD, name + ".log"), "a")
    cmd = ["nice", "-n", "19", PY, "-B", "-m", "v4bench.run", "run"] + j["argv"] + extra + \
          ["--threads", str(threads), "--runs-root", j["runs_root"]]
    p = subprocess.Popen(cmd, cwd=X, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    say(f"start pid {p.pid} {name} {' '.join(extra)}")
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=int, default=24)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--datasets", default=" ".join(ORDER))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--poll", type=int, default=60)
    a = ap.parse_args()
    if a.slots * a.threads > 112:
        print("REFUSED: at most 112 CPU threads", file=sys.stderr)
        return 2
    os.makedirs(LOGD, exist_ok=True)
    os.makedirs(STATE, exist_ok=True)
    dsets = [d for d in ORDER if d in a.datasets.split()]
    running, tries = {}, {}
    say(f"lean_x start slots={a.slots} threads={a.threads} datasets={dsets} dry={a.dry_run}")
    while True:
        for rd, p in list(running.items()):
            rc = p.poll()
            if rc is not None:
                del running[rd]
                ok = rc == 0 and done(rd)
                if not ok:
                    tries[rd] = tries.get(rd, 0) + 1
                say(f"end pid {p.pid} rc={rc} {'DONE' if ok else 'FAILED'} {os.path.basename(rd)}")
        paused = any(FINAL_RE.search(c) for c in live_cmdlines())
        if paused:
            say("a 'v4bench.run final' process is running: no new launch this round")
        jobs, sels = [], []
        for ds in dsets:
            js, sel = wanted(ds)
            jobs += js
            sels.append(sel)
        for sel in sels:
            with open(os.path.join(STATE, f"SELECT_{sel['dataset']}.json"), "w") as f:
                json.dump(sel, f, indent=1, default=str)
        live = live_cmdlines()
        occupied = sum(1 for c in live if c.startswith(PY) and "-m v4bench.run run " in c and "runs_bench_x" in c)
        todo, blocked, orphans = [], 0, 0
        for j, deps in jobs:
            rd = j["run_dir"]
            if done(rd) or rd in running or tries.get(rd, 0) >= MAX_TRIES:
                continue
            key = "-m v4bench.run run " + " ".join(j["argv"])
            if any(key in c for c in live):
                orphans += 1                                 # started by an earlier scheduler, still alive: holds a slot
                continue
            if not all(done(d) for d in deps):
                blocked += 1
                continue
            todo.append(j)
        if a.dry_run:
            for j in todo:
                say(f"DRY ready {j['dataset']}__{j['label']}_s{j['seed']}")
            say(f"DRY {len(todo)} ready, {blocked} waiting on inputs")
            return 0
        occupied = max(occupied, len(running) + orphans)        # every live runs_bench_x job holds a slot (any scheduler)
        while todo and not paused and occupied < a.slots:
            j = todo.pop(0)
            running[j["run_dir"]] = launch(j, a.threads)
            occupied += 1
            time.sleep(2)
        json.dump({"utc": utc(), "running": {os.path.basename(k): v.pid for k, v in running.items()},
                   "ready_not_started": len(todo), "waiting_on_inputs": blocked,
                   "failed": [os.path.basename(k) for k, n in tries.items() if n >= MAX_TRIES]},
                  open(os.path.join(STATE, "STATUS.json"), "w"), indent=1)
        if not running and not orphans and not todo and blocked == 0:
            say("lean_x: nothing left to run - DONE")
            return 0
        if not running and not orphans and not todo and blocked:
            say(f"lean_x: {blocked} jobs wait on inputs that cannot be produced (failed?) - stopping")
            return 1
        time.sleep(a.poll)


if __name__ == "__main__":
    raise SystemExit(main())
