"""X2 job plan (DL 553 A1 + A2(b); POST_HOC_EXPLORATORY; inner validation only; nothing here trains).

  python -B tools_x2/plan_x2.py [--write]     (prints the plan; --write stores runs_bench_x/x2_state/PLAN_X2.json, O_EXCL)

Fixed FL recipe (DL 551): tau 1e-6, B24, unfrozen, 1 local pass, FL dropout 0.1; the server / client LR of the dataset's
validation winner per arm (runs_bench_x/lean/SELECT_<ds>.json "final", read-only; sha256 recorded); theta_pre = the PRE's
PRACTICAL_BEST ("--pre-ckpt best", as the winners). PRE = the winning PRE cell's budget (B96) and dropout.
  A1 (runs_bench_x/x2_A1): per dataset x split (DJ15, DJ50) x seed (2026, 2027):
      PRE on S (D64, B96, winner dropout)  ->  FA_WARM on F;   FA (cold) on F;   C_SMALL on F (central D64, B24, D0.1)
  A2(b) (runs_bench_x/x2_A2B): per dataset x seed, the registered 15 % band and all users as clients:
      PRE D256 (02B_SASREC recipe, B96, winner dropout)  ->  FA_WARM D256;   FA D256
est_h = INFERENCE hours at 4 threads (scaled from the measured POST_HOC_X / runs_bench timings); used only to start the
longest chains first.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)

from v4bench import common  # noqa: E402

V4 = str(common.V4)
RX = os.path.join(V4, "runs_bench_x")
A1_ROOT = os.path.join(RX, "x2_A1")
A2_ROOT = os.path.join(RX, "x2_A2B")
STATE = os.path.join(RX, "x2_state")
SPLITS = os.path.join(A1_ROOT, "splits")
DATASETS = ("ml_1m", "s3_sports", "s3_beauty", "s3_toys")
SEEDS = (2026, 2027)
SPLIT_IDS = {"DJ15": 0.15, "DJ50": 0.5}
D256 = "SASREC_D256_B3"
# MEASURED (4 threads, POST_HOC_X / runs_bench, s2026): hours of the all-user D64 run (FL B24 FA / FA_WARM winner cell;
# central PRE B96 on the 15 % band; C_SMALL B24); ml_1m PRE B96 on CPU = INFERENCE (53,664 updates x ~0.22 s)
BASE_H = {"s3_beauty": {"FA": 2.7, "FA_WARM": 2.1, "PRE": 0.44, "C_SMALL": 0.40},
          "s3_sports": {"FA": 4.8, "FA_WARM": 5.0, "PRE": 0.57, "C_SMALL": 0.50},
          "s3_toys": {"FA": 2.5, "FA_WARM": 1.6, "PRE": 0.26, "C_SMALL": 0.33},
          "ml_1m": {"FA": 11.0, "FA_WARM": 9.1, "PRE": 3.3, "C_SMALL": 5.7}}
D256_FACTOR = {"FL": 3.3, "PRE": 3.3}          # MEASURED smoke 2026-10-09 (s3_toys: round 2.10 / 0.64 s, step 0.546 / 0.165 s)


def lr_of(label: str) -> tuple:
    m = re.search(r"__s([0-9.e-]+)_c([0-9.e-]+)_p1_t1e-06", label)
    if not m:
        raise SystemExit(f"cannot parse the winner cell {label!r}")
    return m.group(1), m.group(2)


def winners(ds: str) -> dict:
    p = os.path.join(RX, "lean", f"SELECT_{ds}.json")
    s = json.load(open(p))
    fa, fw = s["arms"]["FA"]["final"], s["arms"]["FA_WARM"]["final"]
    for lab in (fa, fw):
        if "__B24_L50_D0.1__" not in lab or "_p1_t1e-06" not in lab:
            raise SystemExit(f"{ds}: winner {lab} is not a fixed-recipe (B24, tau 1e-6, 1 pass) cell")
    pre = s["theta_pre"]
    if "-best" not in fw:
        raise SystemExit(f"{ds}: the FA_WARM winner does not use the PRE's PRACTICAL_BEST")
    return {"select_file": p, "select_sha256": common.sha256_file(p), "FA": fa, "FA_WARM": fw,
            "FA_lr": lr_of(fa), "FA_WARM_lr": lr_of(fw), "pre_budget": int(pre["budget"]),
            "pre_dropout": float(pre["dropout"])}


def run_dir_of(argv: list, runs_root: str) -> tuple:
    from v4bench import recipe, run
    a = run._parse(["run"] + argv)
    arm = recipe.ALIASES.get(a.arm, a.arm)
    label = run._x_label(a, arm, recipe)
    return label, str(common.run_dir_for(runs_root, a.dataset, label, a.seed))


def job(study, ds, seed, arm, argv, runs_root, deps, est_h, **extra) -> dict:
    label, rd = run_dir_of(argv, runs_root)
    return {"study": study, "dataset": ds, "seed": seed, "arm": arm, "label": label, "run_dir": rd,
            "runs_root": runs_root, "argv": argv, "deps": deps, "est_h": round(est_h, 2), **extra}


def fl_argv(ds, arm, seed, lr, extra) -> list:
    return ["--dataset", ds, "--arm", arm, "--seed", str(seed), "--budget-efe", "24", "--dropout", "0.1",
            "--server-tau", "1e-6", "--server-lr", lr[0], "--client-lr", lr[1], "--local-passes", "1"] + extra


def build() -> dict:
    jobs, win = [], {}
    for ds in DATASETS:
        w = win[ds] = winners(ds)
        b = BASE_H[ds]
        for sid, frac in SPLIT_IDS.items():
            sp = os.path.join(SPLITS, f"{ds}__{sid}.json")
            if not os.path.isfile(sp):
                raise SystemExit(f"{sp} is missing: run tools_x2/make_splits.py first")
            sx = ["--user-subset", sp]
            for seed in SEEDS:
                pre_argv = ["--dataset", ds, "--arm", "PRE", "--seed", str(seed), "--budget-efe", str(w["pre_budget"]),
                            "--dropout", f"{w['pre_dropout']:g}"] + sx + ["--subset-role", "S"]
                pre = job("A1", ds, seed, "PRE", pre_argv, A1_ROOT, [], b["PRE"] * frac / 0.15, split=sid)
                fw = job("A1", ds, seed, "FA_WARM",
                         fl_argv(ds, "FA_WARM", seed, w["FA_WARM_lr"], ["--pre-run", pre["run_dir"], "--pre-ckpt", "best"]
                                 + sx + ["--subset-role", "F"]), A1_ROOT, [pre["run_dir"]], b["FA_WARM"] * (1 - frac),
                         split=sid)
                fa = job("A1", ds, seed, "FA", fl_argv(ds, "FA", seed, w["FA_lr"], sx + ["--subset-role", "F"]),
                         A1_ROOT, [], b["FA"] * (1 - frac), split=sid)
                cs = job("A1", ds, seed, "C_SMALL",
                         ["--dataset", ds, "--arm", "C_SMALL", "--seed", str(seed), "--budget-efe", "24", "--dropout", "0.1"]
                         + sx + ["--subset-role", "F"], A1_ROOT, [], b["C_SMALL"] * (1 - frac), split=sid)
                pre["chain_h"] = pre["est_h"] + fw["est_h"]
                jobs += [pre, fw, fa, cs]
        vx = ["--model-variant", D256]
        for seed in SEEDS:
            pre_argv = ["--dataset", ds, "--arm", "PRE", "--seed", str(seed), "--budget-efe", str(w["pre_budget"]),
                        "--dropout", f"{w['pre_dropout']:g}"] + vx
            pre = job("A2B", ds, seed, "PRE", pre_argv, A2_ROOT, [], b["PRE"] * D256_FACTOR["PRE"])
            fw = job("A2B", ds, seed, "FA_WARM",
                     fl_argv(ds, "FA_WARM", seed, w["FA_WARM_lr"], ["--pre-run", pre["run_dir"], "--pre-ckpt", "best"] + vx),
                     A2_ROOT, [pre["run_dir"]], b["FA_WARM"] * D256_FACTOR["FL"])
            fa = job("A2B", ds, seed, "FA", fl_argv(ds, "FA", seed, w["FA_lr"], vx), A2_ROOT, [],
                     b["FA"] * D256_FACTOR["FL"])
            pre["chain_h"] = pre["est_h"] + fw["est_h"]
            jobs += [pre, fw, fa]
    for j in jobs:
        j.setdefault("chain_h", j["est_h"])
    order = sorted(range(len(jobs)), key=lambda i: (-jobs[i]["chain_h"], i))
    jobs = [jobs[i] for i in order]
    for k, j in enumerate(jobs):
        j["priority"] = k
    dirs = [j["run_dir"] for j in jobs]
    if len(set(dirs)) != len(dirs):
        raise SystemExit("two jobs share a run directory")
    return {"format": "X2_PLAN_1", "tag": common.X2_TAG, "authority": "DECISION_LOG 553 (A1, A2(b))",
            "utc": common.utc(), "fixed_recipe": {"server_tau": 1e-6, "budget_efe": 24, "frozen": False,
                                                  "local_passes": 1, "fl_dropout": 0.1, "pre_ckpt": "best"},
            "winners": win, "n_jobs": len(jobs),
            "counts": {s: sum(1 for j in jobs if j["study"] == s) for s in ("A1", "A2B")},
            "est_job_hours_total": round(sum(j["est_h"] for j in jobs), 1), "jobs": jobs}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    plan = build()
    if a.write:
        p = os.path.join(STATE, "PLAN_X2.json")
        common.write_exclusive(p, plan)
        print(f"PLAN written: {p} sha256 {common.sha256_file(p)}")
    for j in plan["jobs"]:
        print(f"{j['priority']:3d} {j['study']:4s} {j['est_h']:6.2f}h chain {j['chain_h']:6.2f}h  "
              f"{j['dataset']}/{os.path.basename(j['run_dir'])}  deps={len(j['deps'])}")
    print(json.dumps({k: plan[k] for k in ("n_jobs", "counts", "est_job_hours_total")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
