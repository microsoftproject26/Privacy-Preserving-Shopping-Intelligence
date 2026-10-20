"""POST_HOC_X selection report on INNER VALIDATION only (never TEST). For every dataset it writes
<runs root>/<dataset>/SELECT_X.json: the PRE used as theta_pre, per arm the ranked seed-2026 cells (value =
PRACTICAL_BEST validation ALL NDCG@10, with the CONTROLLED_BUDGET value), the final cell, its seed-2027 / 2028 values
and the R6 decision. The rule and the candidate lists are xjobs.py's, so the report and the job list agree. The previous
file is kept as SELECT_X.prev.json.

  python -B tools_x/select_x.py [--datasets "a b"] [--rx <runs_bench_x>]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xjobs  # noqa: E402


def report(ds: str, rx: str) -> dict:
    pool = xjobs.b24_pool(ds, rx) + xjobs.stage3_cells(ds, rx)
    ranked = {arm: [{"label": xjobs.job(c, ds, 2026, rx)["label"], "value": v, **r}
                    for v, c, r in xjobs.rank([c for c in pool if c["arm"] == arm], ds, rx)] for arm in xjobs.ARMS}
    fin = xjobs.final_cells(ds, rx)
    final = {}
    for arm, t in fin.items():
        if t is None:
            final[arm] = None
            continue
        seeds = {s: xjobs.result_of(xjobs.job(t[1], ds, s, rx)["run_dir"]) for s in (2026, 2027, 2028)}
        final[arm] = {"label": xjobs.job(t[1], ds, 2026, rx)["label"], "cell": t[1], "seeds": seeds}
    pre = xjobs.best_pre(ds, rx)
    return {"format": "SELECT_X_3", "exploratory": "POST_HOC_X", "dataset": ds,
            "rule": "max inner-validation PRACTICAL_BEST ALL ndcg@10 at seed 2026; within 0.0002 the cheaper cell; "
                    "R6: seed 2028 when |v2026 - v2027| > 0.002",
            "theta_pre": {"cell": pre, "dir_s2026": xjobs.pre_dir(pre, ds, 2026, rx) if pre else None},
            "final": final, "r6_arms": xjobs.r6_arms(ds, rx, fin), "ranked": ranked}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=" ".join(xjobs.DATASETS))
    ap.add_argument("--rx", default=xjobs.RX_DEFAULT)
    a = ap.parse_args()
    if "runs_bench_x" not in os.path.abspath(a.rx).replace(os.sep, "/").split("/"):
        print("REFUSED: select_x writes only under runs_bench_x", file=sys.stderr)
        return 2
    for ds in a.datasets.split():
        out = report(ds, a.rx)
        p = os.path.join(xjobs.runs_root_for(a.rx, ds), ds, "SELECT_X.json")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        if os.path.isfile(p):
            shutil.copyfile(p, p.replace(".json", ".prev.json"))
        with open(p, "w") as f:
            json.dump(out, f, indent=1, default=str)
        print(ds, "pre=", out["theta_pre"]["cell"], {arm: (f["label"], (f["seeds"][2026] or {}).get("value")) if f else None
                                                     for arm, f in out["final"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
