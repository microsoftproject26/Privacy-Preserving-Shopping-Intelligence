#!/bin/bash
# b13_stopper.sh - PREG1 amendment 07 addendum B13 (DL 368): ML-1M Track-B seed-2026 schedule backstop.
# At 2026-10-06T00:00Z every ML-1M Track-B seed-2026 grid cell (tb.cfull / tb.pre / tb.fa / tb.star) with no finished RESULT.json
# is stopped (its own process only) and recorded INCOMPLETE_B13_BACKSTOP; trackb-select then runs on the completed cells only;
# winners' seed-2027 runs are never touched (this script only matches --seed 2026).
# NOT ARMED BY DEFAULT: Ops arms it (see company/handoffs/OPS-B13-P2.md). Dry run: --dry-run [--now <ISO>] lists, changes nothing.
# Clock-only trigger: no value is read (only the "status" key of RESULT.json; metrics.jsonl is only line-counted).
V=${V:-/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4}
PY=${PY:-/home/jovyan/venvs/ppsi-gpu/bin/python}
export V PY
exec "$PY" -B - "$@" <<'PYEOF'
import argparse, datetime as dt, json, os, signal, subprocess, sys, time
from pathlib import Path

V = Path(os.environ["V"])
ap = argparse.ArgumentParser(prog="b13_stopper")
ap.add_argument("--dry-run", action="store_true", help="list only: no sleep, no kill, no write")
ap.add_argument("--now", default=None, help="ISO UTC time used as the clock (dry-run only)")
ap.add_argument("--target", default="2026-10-06T00:00:00Z")
ap.add_argument("--dataset", default="ml_1m")
ap.add_argument("--seed", type=int, default=2026)
ap.add_argument("--runs-root", default=str(V / "runs_bench"))
ap.add_argument("--state-dir", default=None, help="driver state dir (default <runs-root>/_wave1_state)")
ap.add_argument("--code", default=str(V / "code_bench4"))
ap.add_argument("--py", default=os.environ["PY"])
ap.add_argument("--queues", default=str(V / "runs_ds3" / "queues"))
ap.add_argument("--verify-passes", type=int, default=3)
ap.add_argument("--verify-gap", type=float, default=30.0)
a = ap.parse_args()
if a.now and not a.dry_run:
    sys.exit("refused: --now is only allowed with --dry-run (a live stopper uses the real clock)")
RUNS = Path(a.runs_root)
D = a.dataset
SEED = a.seed
STATE = Path(a.state_dir) if a.state_dir else RUNS / "_wave1_state"
Q = Path(a.queues)
Q.mkdir(parents=True, exist_ok=True)
LOG = Q / "b13_stopper.log"
INC = "INCOMPLETE_B13_BACKSTOP"


def utcnow():
    return dt.datetime.now(dt.timezone.utc)


def parse(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(dt.timezone.utc)


def stamp():
    return utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def log(m):
    line = "[%s]%s %s" % (stamp(), " DRY" if a.dry_run else "", m)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def cellname(arm, b, ml, dr):
    return "%s__B%s_L%s_D%s" % (arm, b, ml, dr)


def status_of(run_dir):
    p = run_dir / "RESULT.json"
    if not p.is_file():
        return None
    try:
        return str(json.loads(p.read_text(encoding="utf-8")).get("status"))   # ONLY the status key
    except Exception:
        return "UNREADABLE"


def star():
    return str(json.loads((RUNS / D / "FA_WARM_DECISION.json").read_text(encoding="utf-8"))["FA_WARM_STAR"])


def grid(st_arm):
    """Mirror run_wave1.sh build_dataset: the Track-B seed-2026 cells (Track-A cell 6/50/0.1 is not a grid cell)."""
    mls = [50, 200] if D == "ml_1m" else [50]
    cfb = [6, 24] if D == "ml_1m" else [6, 24, 96]
    cells = []
    for ml in mls:
        for dr in ("0.1", "0.2", "0.5"):
            for b in (6, 24, 96):
                if b == 6 and ml == 50 and dr == "0.1":
                    continue
                if b in cfb:
                    cells.append(dict(job="tb.cfull.%s.%d.%d.%s" % (D, b, ml, dr), sel="C_FULL", arm="C_FULL", b=b, ml=ml, dr=dr))
                cells.append(dict(job="tb.pre.%s.%d.%d.%s" % (D, b, ml, dr), sel="PRE", arm="PRE", b=b, ml=ml, dr=dr))
            if not (ml == 50 and dr == "0.1"):
                cells.append(dict(job="tb.fa.%s.6.%d.%s" % (D, ml, dr), sel="FA", arm="FA", b=None, ml=ml, dr=dr))
                cells.append(dict(job="tb.star.%s.6.%d.%s" % (D, ml, dr), sel="STAR", arm=st_arm, b=None, ml=ml, dr=dr))
    for c in cells:
        c["run"] = "%s_s%d" % (cellname(c["arm"], c["b"] or 6, c["ml"], c["dr"]), SEED)
        c["dir"] = RUNS / D / c["run"]
    return cells


def proc_flags(pid):
    try:
        tok = Path("/proc/%d/cmdline" % pid).read_bytes().decode(errors="replace").split("\0")
    except Exception:
        return None
    tok = [t for t in tok if t != ""]
    for i in range(len(tok) - 2):
        if tok[i] == "-m" and tok[i + 1] == "v4bench.run" and tok[i + 2] == "run":
            fl, j = {}, i + 3
            while j < len(tok):
                if tok[j].startswith("--"):
                    if j + 1 < len(tok) and not tok[j + 1].startswith("--"):
                        fl[tok[j][2:]] = tok[j + 1]
                        j += 2
                    else:
                        fl[tok[j][2:]] = True
                        j += 1
                else:
                    j += 1
            return fl
    return None


def ppid(pid):
    try:
        return int(Path("/proc/%d/stat" % pid).read_text().rsplit(")", 1)[1].split()[1])
    except Exception:
        return None


def cmdline(pid):
    try:
        return Path("/proc/%d/cmdline" % pid).read_bytes().decode(errors="replace").replace("\0", " ").strip()
    except Exception:
        return ""


def find_procs(cells):
    """run -> [pid], matched by EXACT dataset + arm + seed + budget / max_len / dropout (never a broad pattern)."""
    hits = {}
    me = os.getpid()
    for p in os.listdir("/proc"):
        if not p.isdigit() or int(p) == me:
            continue
        fl = proc_flags(int(p))
        if not fl or fl.get("dataset") != D or str(fl.get("seed")) != str(SEED) or "smoke" in fl:
            continue
        for c in cells:
            if fl.get("arm") != c["arm"] or str(fl.get("max-len")) != str(c["ml"]):
                continue
            try:
                if float(fl.get("dropout")) != float(c["dr"]):
                    continue
            except Exception:
                continue
            if c["b"] is not None:
                if str(fl.get("budget-efe")) != str(c["b"]):
                    continue
            elif "budget-efe" in fl:
                continue
            hits.setdefault(c["run"], []).append(int(p))
    return hits


def marker(c, why, pid=None):
    mj = c["dir"] / "metrics.jsonl"
    nm = len(mj.read_text(errors="replace").splitlines()) if mj.is_file() else 0     # line count only
    return {"run_id": "BM_%s_%s" % (D, c["run"]), "arm": c["arm"], "dataset": D, "seed": SEED, "status": INC,
            "reason": "PREG1 amendment 07 addendum B13 (DL 368): no finished RESULT.json at %s; stopped by the Ops "
                      "schedule backstop, clock-only. INCOMPLETE per B10 Q4: never lockable, never a winner." % a.target,
            "state_at_stop": why, "killed_pid": pid, "metrics_lines": nm,
            "has_latest_pt": (c["dir"] / "latest.pt").is_file(), "utc": stamp(), "writer": "b13_stopper.sh"}


def write_marker(c, why, pid=None):
    c["dir"].mkdir(parents=True, exist_ok=True)
    tmp = c["dir"] / ("RESULT.json.tmp.%d" % os.getpid())
    # Compact JSON on purpose: no space after the colon, so run_wave*.sh run_job's grep '"status": "INCOMPLETE' does NOT
    # match and the driver treats the cell as finished (skipped, .ok) instead of re-launching it (the runner would refuse anyway).
    tmp.write_text(json.dumps(marker(c, why, pid), sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    os.replace(tmp, c["dir"] / "RESULT.json")
    (c["dir"] / "STATUS.json").write_text(json.dumps({"run_id": "BM_%s_%s" % (D, c["run"]), "status": INC, "utc": stamp()}, indent=1) + "\n")
    log("MARKED %s %s (%s)" % (c["run"], INC, why))


def kill_run(c, pid):
    p1 = ppid(pid)
    p2 = ppid(p1) if p1 else None
    p3 = ppid(p2) if p2 else None
    log("KILL %s job=%s python_pid=%d job_subshell=%s job_wrapper=%s driver=%s cmd=%s" % (c["run"], c["job"], pid, p1, p2, p3, cmdline(pid)[:200]))
    # the job wrapper first (so the driver never writes <job>.fail), then the subshell, then the run itself
    for x in (p2, p1):
        if x and "run_wave" in cmdline(x):
            try:
                os.kill(x, signal.SIGKILL)
            except ProcessLookupError:
                pass
    for sig, wait in ((signal.SIGTERM, 20), (signal.SIGKILL, 5)):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            break
        for _ in range(wait * 2):
            time.sleep(0.5)
            if not Path("/proc/%d" % pid).exists():
                break
        if not Path("/proc/%d" % pid).exists():
            break
    log("KILLED %s alive_after=%s" % (c["run"], Path("/proc/%d" % pid).exists()))


def passno(npass):
    st_arm = star()
    cells = grid(st_arm)
    stat = {c["run"]: status_of(c["dir"]) for c in cells}
    done = [r for r, s in stat.items() if s and s.startswith("DONE")]
    already = [r for r, s in stat.items() if s and s.startswith("INCOMPLETE")]
    todo = [c for c in cells if not stat[c["run"]]]
    procs = find_procs(cells)
    running = [c for c in todo if c["run"] in procs]
    idle = [c for c in todo if c["run"] not in procs]
    log("PASS %d: grid=%d DONE=%d already_INCOMPLETE=%d RUNNING=%d not_started_or_dead=%d" % (npass, len(cells), len(done), len(already), len(running), len(idle)))
    for c in running:
        log("  would-STOP %s run=%s pids=%s" % (c["job"], c["run"], procs[c["run"]]))
    for c in idle:
        log("  would-MARK-not-started %s run=%s dir_exists=%s" % (c["job"], c["run"], c["dir"].exists()))
    for r in done:
        log("  complete %s" % r)
    fails = sorted(x.name for x in STATE.glob("*%s*.fail" % D)) if STATE.is_dir() else []
    if fails:
        log("  NOTE stale .fail states: %s" % fails)
    arms = {}
    for c in cells:
        arms.setdefault(c["sel"], {"arm": c["arm"], "cells": []})["cells"].append(c)
    trackA = {"C_FULL": "C_FULL", "PRE": "PRE", "FA": "FA", "STAR": st_arm}
    plan = {}
    for sel, info in arms.items():
        if all((stat[c["run"]] or "").startswith("DONE") for c in info["cells"]):
            continue
        winf = RUNS / D / "trackb" / ("WINNER_%s.json" % info["arm"])
        runsel = ["%s_s%d" % (trackA[sel], SEED)] + [c["run"] for c in info["cells"] if (stat[c["run"]] or "").startswith("DONE")]
        a_ok = (status_of(RUNS / D / ("%s_s%d" % (trackA[sel], SEED))) or "").startswith("DONE")
        plan[sel] = (info["arm"], winf, runsel, a_ok)
        log("  trackb-select %s: winner_file_exists=%s track_A_cell_done=%s runs(%d)=%s" % (info["arm"], winf.exists(), a_ok, len(runsel), " ".join(runsel)))
    if a.dry_run:
        return len(running) + len(idle)
    if not (running or idle):
        return 0
    # 1. mark never-started cells FIRST (the driver then skips them instead of launching them into freed slots)
    for c in idle:
        write_marker(c, "not started / no process at the B13 time")
    # 2. stop exactly the matched runs
    for c in running:
        for pid in procs[c["run"]]:
            kill_run(c, pid)
    # 3. trackb-select on the completed cells only, BEFORE any .ok state is written (no driver tbsel can start on an INCOMPLETE cell)
    for sel, (arm, winf, runsel, a_ok) in plan.items():
        if winf.exists():
            log("  SELECT %s: %s already exists (write-once), skipped" % (arm, winf.name))
            continue
        if not a_ok:
            log("  SELECT %s: NOT run, Track-A cell is not DONE - Ops must decide" % arm)
            continue
        cmd = ["nice", "-n", "15", a.py, "-B", "-m", "v4bench.run", "trackb-select", "--dataset", D, "--arm", arm, "--runs"] + runsel
        if a.runs_root != str(V / "runs_bench"):
            cmd += ["--runs-root", a.runs_root]
        outf = Q / ("b13_trackb_select_%s_%s.out" % (D, arm))      # selection values stay in a file, never in the log
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
        with open(outf, "w") as fo:
            rc = subprocess.run(cmd, cwd=a.code, env=env, stdout=fo, stderr=subprocess.STDOUT).returncode
        log("  SELECT %s: trackb-select rc=%d (output in %s; winner file exists=%s)" % (arm, rc, outf.name, winf.exists()))
    # 4. killed runs: marker, then .ok states so the driver's tbsel / win jobs proceed
    for c in running:
        write_marker(c, "running at the B13 time; process stopped", procs[c["run"]][0])
    STATE.mkdir(parents=True, exist_ok=True)
    for c in running + idle:
        try:
            (STATE / ("%s.fail" % c["job"])).unlink()
        except FileNotFoundError:
            pass
        (STATE / ("%s.ok" % c["job"])).touch()
        log("  STATE %s.ok written (B13: INCOMPLETE resolved)" % c["job"])
    return len(running) + len(idle)


target = parse(a.target)
now = parse(a.now) if a.now else utcnow()
log("b13_stopper start: target=%s clock=%s dataset=%s seed=%d runs_root=%s" % (a.target, now.strftime("%Y-%m-%dT%H:%M:%SZ") if a.now else "real", D, SEED, RUNS))
if not a.dry_run:
    pf = Q / "b13_stopper.pid"
    if pf.exists():
        try:
            op = int(pf.read_text())
            os.kill(op, 0)
            sys.exit("refused: another stopper (pid %d) is armed (%s)" % (op, pf))
        except (ValueError, ProcessLookupError, PermissionError):
            pass
    pf.write_text(str(os.getpid()))
    while utcnow() < target:
        time.sleep(max(1.0, min(300.0, (target - utcnow()).total_seconds())))
    log("TARGET reached: executing the backstop")
elif now < target:
    log("dry-run: the clock is before the target; the live stopper would sleep %s more; the listing below is what a pass would do if it ran now (read-only)" % (target - now))
else:
    log("dry-run: the clock is at / after the target: listing what the live stopper would do now")
for k in range(1, (1 if a.dry_run else a.verify_passes) + 1):
    passno(k)
    if not a.dry_run and k < a.verify_passes:
        time.sleep(a.verify_gap)
log("DONE" + (" (dry-run: nothing changed)" if a.dry_run else ""))
PYEOF
