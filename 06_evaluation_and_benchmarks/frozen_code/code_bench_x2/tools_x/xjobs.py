"""POST_HOC_X job list v3 (owner correction 2026-10-06: REES46 methodology kept; only the FL TRAINING PROCEDURE is tuned
per dataset). Nothing here trains.

  python -B tools_x/xjobs.py --stage 0..7 [--datasets "a b"] [--rx <runs_bench_x>] [--json]
prints one TSV line per job:
  <run_dir> TAB <log name> TAB <kind central|fl> TAB <device pref any|cpu> TAB <runs root> TAB <v4bench.run args>

Kept from REES46 / the benchmark: the FL model SASREC_D64_B2 only; all users as clients; the same populations; selection
on inner-validation NDCG@10 (PRACTICAL_BEST, CONTROLLED_BUDGET reported - no early stop); the warm start theta_pre from
the server-side PRE; seeds 2026 + 2027 and the R6 third seed 2028 only when the two seeds' inner-validation NDCG@10 differ
by more than 0.002. No central arm is retrained: C_FULL / C_SMALL are not in the grid.
PRE (theta_pre): the best-validation PRE of the dataset (its registered selection value, the PRACTICAL_BEST tracker value
on its own band population), read READ-ONLY from runs_bench (ML-1M: runs_bench_gpu); a PRE that is missing there
(amazon2023 has B6 only; any seed the chosen PRE cell lacks) is trained into runs_bench_x.
Tuned per dataset at seed 2026 (validation only): budget B6 / B24 / B96, FedAdam tau and server LR, client LR, local
passes, for cold FA, FA_WARM (unfrozen) and FA_WARM_FROZEN. Value = RESULT.validation.PRACTICAL_BEST.ALL["ndcg@10"];
within 0.0002 the cheaper cell (budget, passes) wins.
Stages (the driver runs them in order with a barrier):
  0  missing PRE grid at seed 2026 (only datasets without a B24 / B96 PRE: PRE {B24, B96} x {D0.1, 0.2, 0.5}).
  1  coarse, B24, P1: FA tau x server LR (9); FA_WARM (6); FA_WARM_FROZEN (4).
  2  fine, B24: per arm the top-2 cells x passes {2, 4} and the top-1 cell x client LR {x0.3, x3}.
  3  budget: per arm the top-2 cells of stages 1-2 at B6 and at B96.
  4  the seed-2027 PRE of the final warm cells when the read-only root lacks it.
  5  seed 2027 of the final cell per arm.
  6  R6: the seed-2028 PRE of a warm arm whose two seeds differ by > 0.002 (when missing).
  7  R6: seed 2028 of every arm whose two seeds differ by > 0.002.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)
V4 = os.path.dirname(CODE)
RX_DEFAULT = os.path.join(V4, "runs_bench_x")

DATASETS = ("s3_beauty", "s3_sports", "s3_toys", "amazon2023_video_games", "amazon2023_baby_products", "ml_1m")
TOL = 0.0002
R6_GAP = 0.002
ARMS = ("FA", "FA_WARM", "FA_WARM_FROZEN")
# tau x server LR. tau 1e-3 / 1e-4: |delta| << tau (MEASURED smoke) -> step ~ (lr / tau) x delta: LRs give server gains
# lr / tau of 10-100 (1e-3, the registered 0.01 = 10) and 30-300 (1e-4). tau 1e-6: per-coordinate adaptive, step ~ lr:
# LRs bracket the central Adam peak of the same D64 model (7.1e-4).
COLD = [("1e-3", "0.01", "0.1"), ("1e-3", "0.03", "0.1"), ("1e-3", "0.1", "0.1"),
        ("1e-4", "0.003", "0.1"), ("1e-4", "0.01", "0.1"), ("1e-4", "0.03", "0.1"),
        ("1e-6", "0.0003", "0.1"), ("1e-6", "0.001", "0.1"), ("1e-6", "0.003", "0.1")]
# warm (unfrozen): the registered S7 point (server 0.001, client 0.025) plus larger server steps and the cold client LR
WARM = [("1e-3", "0.001", "0.025"), ("1e-3", "0.01", "0.1"), ("1e-6", "0.0001", "0.1"), ("1e-6", "0.0003", "0.1"),
        ("1e-6", "0.001", "0.1"), ("1e-6", "0.0003", "0.025")]
# warm frozen (S8: client factor 1): the registered point (server 0.001, client 0.1) and three larger / adaptive steps
FROZEN = [("1e-3", "0.001", "0.1"), ("1e-3", "0.01", "0.1"), ("1e-6", "0.0003", "0.1"), ("1e-6", "0.001", "0.1")]


def runs_root_for(rx: str, ds: str) -> str:
    return os.path.join(rx, "ml1m") if ds == "ml_1m" else rx


def ro_root(ds: str) -> str:
    """The read-only root of the existing (confirmatory) PRE runs."""
    return os.path.join(V4, "runs_bench_gpu" if ds == "ml_1m" else "runs_bench")


def _read(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


# ------------------------------------------------------------------------------------------------ PRE cells
def pre_cell(budget, dropout):
    return {"arm": "PRE", "budget": int(budget), "dropout": float(dropout)}


def ro_name(pc) -> str:
    return "PRE" if (pc["budget"], pc["dropout"]) == (6, 0.1) else f"PRE__B{pc['budget']}_L50_D{pc['dropout']:g}"


def pre_dir(pc, ds, seed, rx) -> str:
    """The read-only run if it is finished there, else the runs_bench_x run (trained by stage 0 / 4 / 6)."""
    ro = os.path.join(ro_root(ds), ds, f"{ro_name(pc)}_s{seed}")
    r = _read(os.path.join(ro, "RESULT.json"))
    if r and r.get("status") == "DONE_MAIN":
        return ro
    return job(pc, ds, seed, rx)["run_dir"]


def pre_value(d):
    r = _read(os.path.join(d, "RESULT.json"))
    if not r or r.get("status") != "DONE_MAIN" or r.get("arm") != "PRE" or int(r["spec"].get("max_len", 50)) != 50:
        return None
    return float(r["tracker"]["best_value"])            # the PRE's registered selection value (band population)


def stage0_cells(ds):
    have = {(int(pre_cell_of(d)["budget"])) for d in glob.glob(os.path.join(ro_root(ds), ds, "PRE*_s2026"))
            if pre_cell_of(d) is not None and pre_value(d) is not None}
    if 24 in have and 96 in have:
        return []
    return [pre_cell(b, d) for b in (24, 96) for d in (0.1, 0.2, 0.5)]


def pre_cell_of(d):
    n = os.path.basename(d).rsplit("_s", 1)[0]
    if n == "PRE":
        return pre_cell(6, 0.1)
    if n.startswith("PRE__B") and "_L50_D" in n:
        b, dr = n[len("PRE__B"):].split("_L50_D")
        return pre_cell(int(b), float(dr))
    return None


def best_pre(ds, rx):
    cands = []
    for d in glob.glob(os.path.join(ro_root(ds), ds, "PRE*_s2026")):
        pc = pre_cell_of(d)
        v = pre_value(d) if pc else None
        if v is not None:
            cands.append((v, pc))
    for pc in stage0_cells(ds):
        v = pre_value(job(pc, ds, 2026, rx)["run_dir"])
        if v is not None:
            cands.append((v, pc))
    if not cands:
        return None
    cands.sort(key=lambda t: (-t[0], t[1]["budget"]))
    return cands[0][1]


# ------------------------------------------------------------------------------------------------ FL cells and jobs
def fl(arm, budget, tau, slr, clr, passes=1, pre=None):
    c = {"arm": arm, "budget": int(budget), "dropout": 0.1, "tau": tau, "slr": slr, "clr": clr, "passes": int(passes)}
    if pre is not None:
        c["pre"] = dict(pre)
    return c


def argv_of(c, ds, seed, rx) -> list:
    a = ["--dataset", ds, "--arm", c["arm"], "--seed", str(seed), "--budget-efe", str(c["budget"]),
         "--dropout", f"{c['dropout']:g}"]
    if c["arm"] in ARMS:
        a += ["--server-tau", c["tau"], "--server-lr", c["slr"], "--client-lr", c["clr"],
              "--local-passes", str(c["passes"])]
        if "pre" in c:
            a += ["--pre-run", pre_dir(c["pre"], ds, seed, rx), "--pre-ckpt", "best"]
    return a


def job(c, ds, seed, rx) -> dict:
    from v4bench import common, recipe, run
    argv = argv_of(c, ds, seed, rx)
    a = run._parse(["run"] + argv)
    label = run._x_label(a, c["arm"], recipe)
    rr = runs_root_for(rx, ds)
    kind = "fl" if c["arm"] in ARMS else "central"
    return {"dataset": ds, "seed": seed, "cell": c, "arm": c["arm"], "kind": kind, "pref": "cpu" if kind == "fl" else "any",
            "label": label, "run_dir": str(common.run_dir_for(rr, ds, label, seed)), "runs_root": rr, "argv": argv}


def result_of(run_dir):
    r = _read(os.path.join(run_dir, "RESULT.json"))
    if not r or r.get("status") != "DONE_MAIN" or r.get("spec", {}).get("exploratory") != "POST_HOC_X":
        return None
    v = r["validation"]
    return {"value": float(v["PRACTICAL_BEST"]["ALL"]["ndcg@10"]), "best_mark": float(r["checkpoints"]["PRACTICAL_BEST"]["mark"]),
            "controlled_budget": float(v["CONTROLLED_BUDGET"]["ALL"]["ndcg@10"]) if "CONTROLLED_BUDGET" in v else None}


def rank(cells, ds, rx, seed=2026) -> list:
    out, seen = [], set()
    for c in cells:
        rd = job(c, ds, seed, rx)["run_dir"]
        if rd in seen:
            continue
        seen.add(rd)
        r = result_of(rd)
        if r is not None:
            out.append((r["value"], c, r))
    if not out:
        return []
    out.sort(key=lambda t: -t[0])
    best = out[0][0]
    near = sorted([t for t in out if t[0] >= best - TOL], key=lambda t: (t[1]["budget"], t[1]["passes"]))
    return near + [t for t in out if t[0] < best - TOL]


def with_(c, **kw):
    d = dict(c)
    d.update(kw)
    return d


def stage1_cells(ds, rx):
    cs = [fl("FA", 24, t, s, cl) for t, s, cl in COLD]
    p = best_pre(ds, rx)
    if p is not None:
        cs += [fl("FA_WARM", 24, t, s, cl, pre=p) for t, s, cl in WARM]
        cs += [fl("FA_WARM_FROZEN", 24, t, s, cl, pre=p) for t, s, cl in FROZEN]
    return cs


def stage2_cells(ds, rx):
    s1, cs = stage1_cells(ds, rx), []
    for arm in ARMS:
        r = rank([c for c in s1 if c["arm"] == arm], ds, rx)
        for _, c, _x in r[:2]:
            cs += [with_(c, passes=p) for p in (2, 4)]
        for _, c, _x in r[:1]:
            cs += [with_(c, clr=f"{float(c['clr']) * m:.3g}") for m in (0.3, 3.0)]
    return cs


def b24_pool(ds, rx):
    return stage1_cells(ds, rx) + stage2_cells(ds, rx)


def stage3_cells(ds, rx):
    pool, cs = b24_pool(ds, rx), []
    for arm in ARMS:
        for _, c, _x in rank([c for c in pool if c["arm"] == arm], ds, rx)[:2]:
            cs += [with_(c, budget=b) for b in (6, 96)]
    return cs


def final_cells(ds, rx) -> dict:
    pool = b24_pool(ds, rx) + stage3_cells(ds, rx)
    out = {}
    for arm in ARMS:
        r = rank([c for c in pool if c["arm"] == arm], ds, rx)
        out[arm] = r[0] if r else None
    return out


def r6_arms(ds, rx, fin) -> list:
    out = []
    for arm, t in fin.items():
        if t is None:
            continue
        r27 = result_of(job(t[1], ds, 2027, rx)["run_dir"])
        if r27 is not None and abs(t[0] - r27["value"]) > R6_GAP:
            out.append(arm)
    return out


def stage_jobs(stage: int, rx: str, datasets) -> list:
    out, seen = [], set()

    def add(c, ds, seed):
        j = job(c, ds, seed, rx)
        if j["run_dir"] not in seen:
            seen.add(j["run_dir"])
            out.append(j)

    def add_pre(pc, ds, seed):                 # only when the read-only root lacks this PRE seed
        if not pre_dir(pc, ds, seed, rx).startswith(ro_root(ds) + os.sep):
            add(pc, ds, seed)

    for ds in datasets:
        if stage == 0:
            for c in stage0_cells(ds):
                add(c, ds, 2026)
        elif stage in (1, 2, 3):
            for c in {1: stage1_cells, 2: stage2_cells, 3: stage3_cells}[stage](ds, rx):
                add(c, ds, 2026)
        else:
            fin = final_cells(ds, rx)
            r6 = r6_arms(ds, rx, fin) if stage in (6, 7) else []
            for arm, t in fin.items():
                if t is None:
                    continue
                c = t[1]
                if stage == 4 and "pre" in c:
                    add_pre(c["pre"], ds, 2027)
                if stage == 5:
                    add(c, ds, 2027)
                if stage == 6 and arm in r6 and "pre" in c:
                    add_pre(c["pre"], ds, 2028)
                if stage == 7 and arm in r6:
                    add(c, ds, 2028)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True, choices=tuple(range(8)))
    ap.add_argument("--datasets", default=" ".join(DATASETS))
    ap.add_argument("--rx", default=RX_DEFAULT)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    if "runs_bench_x" not in os.path.abspath(a.rx).replace(os.sep, "/").split("/"):
        print("REFUSED: --rx must be under runs_bench_x", file=sys.stderr)
        return 2
    js = stage_jobs(a.stage, a.rx, a.datasets.split())
    if a.json:
        print(json.dumps(js, indent=1, default=str))
        return 0
    for j in js:
        print("\t".join([j["run_dir"], f"{j['dataset']}__{j['label']}_s{j['seed']}", j["kind"], j["pref"],
                         j["runs_root"], " ".join(j["argv"])]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
