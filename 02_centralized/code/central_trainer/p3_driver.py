"""v4train.p3_driver — the P3 contest queue of one family (TRAIN-03; Fable staggered-P3 ruling, conditions 1, 4, 6).

  python -B <code_root>/v4train/p3_driver.py --family GRU --gpus 0 --out-dir <V4>/runs_p3 [--code-root <V4>/code_p3]
  python -B <code_root>/v4train/p3_driver.py --family SASREC --gpus 0 1 --out-dir <V4>/runs_p3 [...]   # parallel
  [--seeds 2026 2027] [--recipe-id 02B_<FAMILY>] [--tuned-recipes ...] [--history-policy USER_HISTORY_L50] [--dry-run]

For each seed (2026, 2027) it launches one MAIN contest run through <code_root>/v4int/launch_central.py:
    python -B <code_root>/v4int/launch_central.py --family F --seed S --phase MAIN --recipe-id 02B_F --gpu G
           --run-dir <out>/<F>/02B_F_s<S> --history-policy USER_HISTORY_L50 [--tuned-recipes ...] [--resume]
With one GPU the seeds run one after the other on it; with two GPUs they run in parallel, seed i on GPU i (ruling
condition 6: GRU sequential on GPU0 and SASRec parallel are both fine). `--code-root` selects the code tree (e.g. the
separate server tree code_p3/, so the running P2 is untouched). A run dir that already holds a finished public summary
(DONE_CONTINUATION, or DONE_MAIN without STILL_IMPROVING) is not launched again; one with a RUN_START and no finished
summary is resumed. The trainer seals every value (sealed/) and continues automatically on STILL_IMPROVING (C3 rule):
this driver reads ONLY the public STATUS.json / RUN_SUMMARY.json (status, STILL_IMPROVING, marks) and prints and records
no metric value. It writes P3_<FAMILY>_QUEUE_<utc>.json (commands, exit codes, public statuses).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

_ENV0 = dict(os.environ)                     # before any v4train / snapshot import (CUDA_VISIBLE_DEVICES default leak)
_HERE = Path(__file__).resolve().parent
CODE = _HERE.parent
SEEDS = (2026, 2027)
PUBLIC_KEYS = ("status", "still_improving", "marks_done", "continuation", "config_sha256", "sealed")


class P3Error(RuntimeError):
    pass


def child_env() -> dict:
    env = dict(_ENV0, PYTHONDONTWRITEBYTECODE="1")
    env.pop("CUDA_VISIBLE_DEVICES", None)                    # the launcher exports --gpu itself
    return env


def run_dir_of(out_dir: Path, family: str, recipe_id: str, seed: int) -> Path:
    return Path(out_dir) / family / f"{recipe_id}_s{int(seed)}"


def launch_cmd(code_root: Path, family: str, seed: int, gpu: int, run_dir: Path, recipe_id: str, *,
               history_policy: str = "USER_HISTORY_L50", tuned_recipes: Optional[Path] = None, resume: bool = False,
               python: str = sys.executable) -> list:
    cmd = [python, "-B", str(Path(code_root) / "v4int" / "launch_central.py"), "--family", family, "--seed",
           str(int(seed)), "--phase", "MAIN", "--recipe-id", recipe_id, "--gpu", str(int(gpu)), "--run-dir",
           str(run_dir), "--history-policy", history_policy]
    if tuned_recipes:
        cmd += ["--tuned-recipes", str(tuned_recipes)]
    if resume:
        cmd.append("--resume")
    return cmd


def public_state(run_dir: Path) -> dict:
    """The public (unsealed) state of a run: never a metric value."""
    rd = Path(run_dir)
    out = {"run_start": (rd / "RUN_START.json").is_file()}
    p = rd / "RUN_SUMMARY.json"
    if p.is_file():
        s = json.loads(p.read_text(encoding="utf-8"))
        out.update({k: s.get(k) for k in PUBLIC_KEYS})
    st = rd / "STATUS.json"
    if st.is_file():
        out["status_file"] = json.loads(st.read_text(encoding="utf-8")).get("status")
    return out


def finished(state: dict) -> bool:
    return state.get("status") == "DONE_CONTINUATION" or (state.get("status") == "DONE_MAIN"
                                                          and state.get("still_improving") is False)


def plan(args) -> list:
    fam = args.family
    rid = args.recipe_id or f"02B_{fam}"
    gpus = [int(g) for g in args.gpus]
    if len(gpus) not in (1, 2) or len(set(gpus)) != len(gpus):
        raise P3Error("--gpus takes one GPU (seeds sequential) or two distinct GPUs (seeds in parallel)")
    seeds = [int(s) for s in args.seeds]
    if len(gpus) == 2 and len(seeds) != 2:
        raise P3Error("two GPUs run exactly two seeds in parallel")
    jobs = []
    for i, seed in enumerate(seeds):
        rd = run_dir_of(Path(args.out_dir), fam, rid, seed)
        st = public_state(rd)
        gpu = gpus[i] if len(gpus) == 2 else gpus[0]
        jobs.append({"family": fam, "seed": seed, "gpu": gpu, "run_dir": str(rd), "recipe_id": rid,
                     "already_finished": finished(st),
                     "cmd": launch_cmd(Path(args.code_root), fam, seed, gpu, rd, rid, history_policy=args.history_policy,
                                       tuned_recipes=args.tuned_recipes,
                                       resume=st["run_start"] and not finished(st))})
    return jobs


def run_queue(args, popen=subprocess.Popen, say=print) -> int:
    jobs = plan(args)
    if args.dry_run:
        for j in jobs:
            say(" ".join(f'"{c}"' if " " in c else c for c in j["cmd"]))
        return 0
    todo = [j for j in jobs if not j["already_finished"]]
    parallel = len({j["gpu"] for j in todo}) == 2
    t0 = time.time()
    procs = []
    for j in todo:
        say(f"[p3] {j['family']} seed {j['seed']} on GPU {j['gpu']}: launching")
        p = popen(j["cmd"], cwd=str(args.code_root), env=child_env())
        if parallel:
            procs.append((j, p))
        else:
            j["exit_code"] = int(p.wait())
            j["public_state"] = public_state(Path(j["run_dir"]))
            say(f"[p3] {j['family']} seed {j['seed']}: exit {j['exit_code']}, status {j['public_state'].get('status')}")
    for j, p in procs:
        j["exit_code"] = int(p.wait())
        j["public_state"] = public_state(Path(j["run_dir"]))
        say(f"[p3] {j['family']} seed {j['seed']}: exit {j['exit_code']}, status {j['public_state'].get('status')}")
    for j in jobs:
        j.setdefault("public_state", public_state(Path(j["run_dir"])))
    rec = {"task": "TRAIN-03 P3 queue", "family": args.family, "code_root": str(args.code_root),
           "mode": "parallel" if parallel else "sequential", "jobs": jobs, "wall_s": round(time.time() - t0, 1),
           "note": "public states only (TRAIN-03 seal): no metric value is read, printed or recorded here",
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out = Path(args.out_dir) / f"P3_{args.family}_QUEUE_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    ok = all(j.get("exit_code", 0) == 0 and finished(j["public_state"]) for j in jobs)
    say(f"[p3] {out.name} written; all finished: {ok}")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="P3 contest queue of one family (TRAIN-03)")
    ap.add_argument("--family", required=True, type=str.upper, choices=("GRU", "SASREC"))
    ap.add_argument("--gpus", required=True, type=int, nargs="+")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--code-root", type=Path, default=CODE, help="the code tree to launch from (e.g. <V4>/code_p3)")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--recipe-id", default=None, help="default 02B_<FAMILY>")
    ap.add_argument("--tuned-recipes", type=Path, default=None)
    ap.add_argument("--history-policy", default="USER_HISTORY_L50")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if not (Path(args.code_root) / "v4int" / "launch_central.py").is_file():
        print(f"P3 REFUSED: {args.code_root} holds no v4int/launch_central.py", file=sys.stderr)
        return 2
    try:
        return run_queue(args)
    except P3Error as e:
        print(f"P3 REFUSED: {e}", file=sys.stderr)
        return 2


__all__ = ["launch_cmd", "run_dir_of", "public_state", "finished", "plan", "run_queue", "child_env", "P3Error"]

if __name__ == "__main__":
    raise SystemExit(main())
