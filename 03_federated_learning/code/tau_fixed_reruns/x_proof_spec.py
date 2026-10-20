"""X-REES46: the VALIDATION (role MAIN) proof spec of an EXP lock's analysis: the lock's validation_proof_entries that
exist, plus in-run MAIN_VALIDATION rank files of FL runs without a REEVAL_A40 copy (PRACTICAL_BEST = the best in-run
mark, CONTROLLED_BUDGET = mark 6.0); contrasts with a missing input are dropped and listed.
  python x_proof_spec.py <lock> <out spec> [NAME|TAG=<FL run dir> ...]"""
import glob
import hashlib
import json
import os
import sys


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def inrun(rd, tag):
    res = {}
    for f in glob.glob(os.path.join(rd, "evals", "eval_*.result.json")):
        m = float(os.path.basename(f)[5:-12])
        if m >= 0.5:
            res[m] = json.load(open(f))["E2E"]["macro"]["mrr@20"]
    m = max(res, key=lambda k: (res[k], -k)) if tag == "PB" else 6.0
    return os.path.join(rd, "evals", f"eval_{m:g}.ranks_int32.npy" if m != int(m) else f"eval_{m:.1f}.ranks_int32.npy"), m


lock = json.load(open(sys.argv[1]))
a = lock["analysis"]
ents = {k: dict(e) for k, e in a["validation_proof_entries"].items()}
for spec in sys.argv[3:]:
    name, rd = spec.split("=", 1)
    parts = name.split("|")
    arm, tag = parts[0], parts[-1]
    seed = int(parts[1]) if len(parts) == 3 else 2026
    p, mark = inrun(rd, tag)
    ents[name] = {"arm": arm, "seed": seed, "rule": {"PB": "PRACTICAL_BEST", "CB": "CONTROLLED_BUDGET"}[tag],
                  "ranks": p, "sha256": None, "source": f"in-run MAIN_VALIDATION eval, mark {mark}"}
keep = {}
for k, e in ents.items():
    if os.path.isfile(e["ranks"]):
        e["sha256"] = sha(e["ranks"])
        keep[k] = e
con = [c for c in a["contrasts"] if all(x in keep for p in c["pairs"] for x in p)]
drop = [c["id"] for c in a["contrasts"] if c not in con]
json.dump({"label": "POST_HOC_EXPLORATORY", "entries": keep, "contrasts": con, "dropped_contrasts": drop,
           "missing_entries": sorted(set(ents) - set(keep))}, open(sys.argv[2], "w"), indent=1)
print(json.dumps({"entries": len(keep), "contrasts": len(con), "dropped": drop, "missing": sorted(set(ents) - set(keep))}))
