"""X-REES46 exploratory analysis (POST_HOC_EXPLORATORY, owner approval DL 539). No new statistics: it imports the frozen
tools/p2_report.py (which imports the frozen tools/retention_report.py, sha-asserted) and uses
  p2_report.P2Engine         retention_report.Engine with ONE shared W per user set = v4eval.bootstrap.make_plan(seed
                             20261004, 1,000 resamples) (the Phase-2 plan)
  retention_report.delta     paired a - b, client-macro E2E, seed mean, percentile 95% CI, harmed / improved share
  retention_report.ratio     R = 100 M / C per seed, bootstrapped directly on the shared W
  p2_report.tost             90% CI inside +-0.0005 (the Q1 margin) for every MRR@20 delta (EQUIVALENT reading)
for MRR@20 (primary), NDCG@10 and HR@10. Rank files are read as given (exploratory ones from runs_x_rees46, confirmatory
ones read-only from runs_ds3/TEST, runs_p2/TEST, sha256-checked against the spec).

  python -B x_rees46/x_analysis.py --spec <spec.json> --role MAIN|MAIN_TEST --out <dir> --code-root <tree>
         [--lock <EXP lock> --lock-sha <sha>]      (MAIN_TEST: the EXP lock + its register entry, checked by reeval_x)
spec: {"label", "entries": {name: {"arm", "seed", "rule", "ranks", "sha256"}},
       "contrasts": [{"id", "kind": "delta"|"ratio", "pairs": [[num_name, den_name], ...], "shared": false, "note"}]}"""
import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
V = HERE.parent
sys.path.insert(0, str(V / "tools"))
LABEL = "POST_HOC_EXPLORATORY"
EVAL = {"MAIN": ("MAIN_VALIDATION_5_100_131072", "MAIN_TRAIN_5_100_131072"),
        "MAIN_TEST": ("MAIN_RETRO_TEST_5_100_131072", "MAIN_TRAIN_5_100_131072")}


def sha_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_rx():
    spec = importlib.util.spec_from_file_location("x_reeval_x", HERE / "reeval_x.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, type=Path)
    ap.add_argument("--role", required=True, choices=tuple(EVAL))
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--code-root", required=True, type=Path)
    ap.add_argument("--lock", type=Path)
    ap.add_argument("--lock-sha")
    args = ap.parse_args(argv)
    if "runs_x_rees46" not in str(args.out.resolve()):
        sys.exit("x_analysis: --out must be under runs_x_rees46")
    if args.out.exists() and any(args.out.iterdir()):
        sys.exit(f"x_analysis: {args.out} exists and is not empty")
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    binding = None
    if args.role == "MAIN_TEST":                                   # the exploratory gate, before any TEST read
        RX = load_rx()
        if args.lock is None or not args.lock.is_file() or RX.sha_file(args.lock) != str(args.lock_sha).lower():
            RX.refuse("MAIN_TEST analysis needs --lock / --lock-sha of the registered EXP lock")
        lock = RX.strict(args.lock)
        binding = RX.register_binding(args.lock_sha.lower(), lock["document"])
    import p2_report as P2
    RR = P2.RR
    RR.bind(args.code_root)
    mid, tid = EVAL[args.role]
    if args.role == "MAIN_TEST":
        aml = V / "company" / "gates" / "ALL_METHODS_LOCK.json"
        reg = V / "company" / "registers" / "TEST_EXPOSURE_REGISTER.json"
        gate = RR.check_test_gate(aml, reg)
        ext_path = V / "contracts" / "manifest_registry" / "REGISTRY_V2_TEST_EXT.json"
        ext = json.loads(ext_path.read_text(encoding="utf-8"))
        loader = RR.test_context_loader(gate, ext_path, ext, aml, reg)
    else:
        loader = RR.production_context_loader()
    ctx = loader(args.role, {"eval_manifest_id": mid, "train_manifest_id": tid})
    eng = P2.P2Engine({args.role: ctx})
    np = RR.LIB.np
    ents, inputs = {}, {}
    for name, d in spec["entries"].items():
        p = Path(d["ranks"])
        got = sha_file(p)
        if d.get("sha256") and got != d["sha256"]:
            sys.exit(f"x_analysis: {name} ranks sha256 {got[:12]} != the spec's")
        ranks = np.load(p)
        if ranks.shape[0] != ctx.manifest.n:
            sys.exit(f"x_analysis: {name} ranks cover {ranks.shape[0]} rows, the {mid} manifest {ctx.manifest.n}")
        RR.check_rank_coverage(name, ctx.manifest, ranks)
        e = RR.Entry(arm=d["arm"], seed=d["seed"], rule=d["rule"], role=args.role, rows_rel=None, ranks_rel=None,
                     row_filter={}, run_dir_rel=None, subset_rel=None, historical=False, meta_declared={})
        e.ranks = ranks
        ents[name] = e
        inputs[name] = {**d, "sha256": got}
    vkey = (args.role, "ALL", None)
    out = {"label": LABEL, "role": args.role, "eval_manifest_id": mid, "bootstrap": "make_plan seed 20261004, 1000",
           "metrics": ["mrr@20 (primary)", "ndcg@10", "hr@10"], "inputs": inputs, "register_binding": binding,
           "tools": {"p2_report.py": sha_file(V / "tools" / "p2_report.py"),
                     "retention_report.py": sha_file(V / "tools" / "retention_report.py"),
                     "x_analysis.py": sha_file(Path(__file__))},
           "points": {}, "contrasts": [], "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for name, e in ents.items():
        out["points"][name] = {f"{m[0]}@{m[1]}": eng.point(e, vkey, m) for m in RR.METRICS}
    for c in spec["contrasts"]:
        pairs = [(ents[a], ents[b]) for a, b in c["pairs"]]
        rec = {"id": c["id"], "kind": c["kind"], "pairs": c["pairs"], "note": c.get("note"), "label": LABEL}
        for m in RR.METRICS:
            key = f"{c['id']}|{m[0]}@{m[1]}"
            if c["kind"] == "delta":
                r = RR.delta(eng, pairs, vkey, m, name=c["id"], shared=c.get("shared", False), boot_key=key)
                if m == RR.PRIMARY:
                    r["tost_q1_margin"] = P2.tost(eng.boot.get(key), P2.Q1_INTERVAL)
            else:
                r = RR.ratio(eng, pairs, vkey, m, strict=True, name=c["id"], shared=c.get("shared", False),
                             boot_key=key)
            rec[f"{m[0]}@{m[1]}"] = r
        out["contrasts"].append(rec)
    args.out.mkdir(parents=True, exist_ok=True)
    txt = RR.dumps(out)
    (args.out / "X_RESULT.json").write_text(txt, encoding="utf-8")
    print(json.dumps({"written": str(args.out / "X_RESULT.json"), "sha256": hashlib.sha256(txt.encode()).hexdigest(),
                      "n_contrasts": len(out["contrasts"])}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
