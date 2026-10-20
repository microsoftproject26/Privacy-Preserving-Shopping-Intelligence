"""X2 results collector (DL 553 A1 + A2(b); POST_HOC_EXPLORATORY; INNER VALIDATION only; never TEST).

  python -B tools_x2/report_x2.py       -> runs_bench_x/x2_state/SUMMARY_X2.json (+ a short table on stdout)
Value = RESULT.validation.PRACTICAL_BEST[group]["ndcg@10"] (CONTROLLED_BUDGET too), group = OTHERS (= F's users) for A1,
ALL for A2(b). A1 PRE rows give PRE-alone on F (OTHERS at the PRE's own S-selected mark) = the warm-start reference.
Seed means over the finished seeds; a cell with a missing seed is marked partial.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
V4 = os.path.dirname(os.path.dirname(HERE))
STATE = os.path.join(V4, "runs_bench_x", "x2_state")


def main() -> int:
    plan = json.load(open(os.path.join(STATE, "PLAN_X2.json")))
    cells = {}
    for j in plan["jobs"]:
        p = os.path.join(j["run_dir"], "RESULT.json")
        r = json.load(open(p)) if os.path.isfile(p) else None
        grp = "OTHERS" if j["study"] == "A1" else "ALL"
        k = f"{j['study']}|{j['dataset']}|{j.get('split', 'BAND15')}|{j['arm']}"
        c = cells.setdefault(k, {"group": grp, "seeds": {}})
        if not r or r.get("status") != "DONE_MAIN":
            c["seeds"][str(j["seed"])] = None
            continue
        v = r["validation"]
        c["seeds"][str(j["seed"])] = {"pb": v["PRACTICAL_BEST"][grp]["ndcg@10"],
                                      "pb_mark": r["checkpoints"]["PRACTICAL_BEST"]["mark"],
                                      "cb": v.get("CONTROLLED_BUDGET", {}).get(grp, {}).get("ndcg@10"),
                                      "n_users": v["PRACTICAL_BEST"][grp]["n_users"]}
    for c in cells.values():
        got = [s["pb"] for s in c["seeds"].values() if s]
        c["pb_mean"] = sum(got) / len(got) if got else None
        c["complete"] = len(got) == len(c["seeds"])
    out = {"label": "POST_HOC_EXPLORATORY (inner validation; no TEST)", "plan_sha_note": "PLAN_X2.json", "cells": cells}
    tmp = os.path.join(STATE, f"SUMMARY_X2.json.tmp.{os.getpid()}")
    json.dump(out, open(tmp, "w"), indent=1, sort_keys=True)
    os.replace(tmp, os.path.join(STATE, "SUMMARY_X2.json"))
    for k in sorted(cells):
        c = cells[k]
        m = "  -   " if c["pb_mean"] is None else f"{c['pb_mean']:.4f}"
        print(f"{k:45s} {c['group']:6s} {m} {'' if c['complete'] else '(partial)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
