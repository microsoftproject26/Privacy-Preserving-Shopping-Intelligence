"""X2 (PB100 grid, POST_HOC_EXPLORATORY): the registered P2 CALIB rule (PREG2 §5.6 as v4int.p2_select): metric =
CALIB_VALIDATION E2E client-macro MRR@20 at PRACTICAL_BEST over marks 0.5 .. endpoint (mark 0 never a candidate); tie
band 0.001 INCLUSIVE (exact decimals); inside the band p2_order_key: (1) upload bytes and (2) trained parameters are
equal for every cell here (all dense, items unfrozen), (3) the smaller effective server step (FedAdam eta), (4) the
smaller client LR applied (c; then the item-LR multiplier), (5) X2 additions: fewer exposures (smaller EFE budget, then
fewer local passes), (6) the lexically first label. Cells with tau != 1e-6 (the control) are reported, never selected.
  python x_select2.py <out.json> <top_k> <run_dir> ...     prints the winner and the top_k ranking (run dirs)"""
import glob
import json
import os
import sys
from fractions import Fraction


def curve(rd):
    out = {}
    for f in glob.glob(os.path.join(rd, "evals", "eval_*.result.json")):
        m = float(os.path.basename(f)[len("eval_"):-len(".result.json")])
        out[m] = json.load(open(f))["E2E"]["macro"]["mrr@20"]
    return dict(sorted(out.items()))


def cell(rd):
    a = json.load(open(os.path.join(rd, "ARM_SPEC.json")))
    s = a.get("arm_spec", a)
    c = s["cfg"]
    return {"eta": float(c["server_opt"]["lr"]), "tau": float(c["server_opt"]["tau"]), "c": float(c["solver"]["lr"]),
            "ilr": float(c["solver"].get("item_lr_mult") or 1.0), "passes": int(c["solver"]["passes"]),
            "efe": float(c["endpoint_efe"]),
            "label": s.get("p2_label") or f"{s['arm']}|fedadam|eta={c['server_opt']['lr']:g}|tau={c['server_opt']['tau']:g}"}


def main(argv):
    out, k, rds = argv[0], int(argv[1]), argv[2:]
    rows = []
    for rd in rds:
        if not os.path.isfile(os.path.join(rd, "RUN_SUMMARY.json")):
            rows.append({"run_dir": rd, "excluded": "not finished"})
            continue
        c, x = curve(rd), cell(rd)
        marks = {m: v for m, v in c.items() if m >= 0.5}
        if not marks:
            rows.append({"run_dir": rd, **x, "excluded": "no marks"})
            continue
        bm = max(marks, key=lambda m: (marks[m], -m))
        last = max(marks)
        rows.append({"run_dir": rd, **x, "best_mark": bm, "value": marks[bm], "last_mark": last,
                     "rising_at_end": bm == last, "curve": c})
    ok = [r for r in rows if "value" in r and r["tau"] == 1e-6]
    top = max(Fraction(repr(r["value"])) for r in ok)
    key = lambda r: (Fraction(repr(r["eta"])), Fraction(repr(r["c"])), Fraction(repr(r["ilr"])), r["efe"],  # noqa: E731
                     r["passes"], r["label"])
    band = sorted([r for r in ok if top - Fraction(repr(r["value"])) <= Fraction(1, 1000)], key=key)
    win = band[0]
    ranking = sorted(ok, key=lambda r: (-r["value"], key(r)))
    rec = {"format": "PPSI_V4_X2_PB100_SELECTION_1", "label": "POST_HOC_EXPLORATORY", "rule": __doc__.split("\n  python")[0],
           "rows": rows, "tie_band": [r["run_dir"] for r in band], "winner": win,
           "ranking": [r["run_dir"] for r in ranking]}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1, sort_keys=True)
    print("WINNER", win["run_dir"])
    for r in ranking[:k]:
        print("TOP", r["run_dir"])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
