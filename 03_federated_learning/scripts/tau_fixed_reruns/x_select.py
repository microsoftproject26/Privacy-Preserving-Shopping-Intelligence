"""X-REES46 (POST_HOC_EXPLORATORY): the registered P2 CALIB selection rule (PREG2 §5.6 as implemented in v4int.p2_select)
re-applied to the X grid: metric = CALIB_VALIDATION E2E client-macro MRR@20 at PRACTICAL_BEST over marks 0.5 .. 6.0
(mark 0 never a candidate); tie band 0.001 INCLUSIVE in exact decimal arithmetic; inside the band the smaller effective
server step (FedAdam eta) wins (all cells share bytes / parameters / client LR, so p2_order_key reduces to eta); edge
rule: a winner on the upper eta edge -> ONE extension step outward (EDGE_STEP_UP), none on the lower edge (3e-6 is the
smallest admitted FedAdam eta) -> flag EDGE_LOW_NOT_EXTENDABLE.
  python x_select.py <out.json> <run_dir> [<run_dir> ...]      (exit 3 = edge extension required; the lane runs it)"""
import glob
import json
import os
import sys
from fractions import Fraction

EDGE_STEP_UP = {3e-05: 1e-04}


def curve(rd):
    out = {}
    for f in glob.glob(os.path.join(rd, "evals", "eval_*.result.json")):
        mark = float(os.path.basename(f)[len("eval_"):-len(".result.json")])
        r = json.load(open(f))
        out[mark] = r["E2E"]["macro"]["mrr@20"]
    return dict(sorted(out.items()))


def eta_of(rd):
    a = json.load(open(os.path.join(rd, "ARM_SPEC.json")))
    s = a.get("arm_spec", a)
    return float(s["cfg"]["server_opt"]["lr"]), float(s["cfg"]["server_opt"]["tau"])


def main(argv):
    out, rds = argv[0], argv[1:]
    cand = []
    for rd in rds:
        c = curve(rd)
        marks = {m: v for m, v in c.items() if 0.5 <= m <= 6.0}
        eta, tau = eta_of(rd)
        if not marks:
            cand.append({"run_dir": rd, "eta": eta, "tau": tau, "excluded": "no marks"})
            continue
        bm = max(marks, key=lambda m: (marks[m], -m))
        cand.append({"run_dir": rd, "eta": eta, "tau": tau, "curve": c, "best_mark": bm, "value": marks[bm],
                     "n_marks": len(marks), "endpoint_6_0": marks.get(6.0)})
    ok = [c for c in cand if "value" in c]
    top = max(Fraction(repr(c["value"])) for c in ok)
    band = [c for c in ok if top - Fraction(repr(c["value"])) <= Fraction(1, 1000)]
    win = min(band, key=lambda c: Fraction(repr(c["eta"])))
    etas = sorted({c["eta"] for c in ok})
    flags, need = [], None
    if win["eta"] == etas[-1]:
        if win["eta"] in EDGE_STEP_UP and EDGE_STEP_UP[win["eta"]] not in etas:
            need = EDGE_STEP_UP[win["eta"]]
            flags.append("EDGE_UP_EXTENSION_REQUIRED")
        elif len(etas) > 3:
            flags.append("EDGE_UNRESOLVED")
    if win["eta"] == etas[0] and len(etas) > 1:
        flags.append("EDGE_LOW_NOT_EXTENDABLE")
    rec = {"format": "PPSI_V4_X_REES46_SELECTION_1", "label": "POST_HOC_EXPLORATORY",
           "rule": __doc__.split("\n  python")[0], "candidates": cand, "tie_band": [c["run_dir"] for c in band],
           "winner": {"run_dir": win["run_dir"], "eta": win["eta"], "tau": win["tau"], "best_mark": win["best_mark"],
                      "value": win["value"]}, "flags": flags, "edge_extension_eta": need}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1, sort_keys=True)
    print(json.dumps({"winner": rec["winner"], "flags": flags, "edge_extension_eta": need}))
    return 3 if need is not None else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
