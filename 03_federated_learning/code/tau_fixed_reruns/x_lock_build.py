"""X-REES46: build an EXPLORATORY TEST lock (format PPSI_V4_X_EXP_TEST_LOCK_1) + the analysis specs (VALIDATION proof and
MAIN_TEST look) for part A (PB100) or part B (X_FA, the tau-fix re-run). Reads checkpoint bytes / run records only (no
metric value). Writes the lock ONLY to a path that does not exist yet.
  python -B x_rees46/x_lock_build.py A|B --out <lock path> [--x-main-dirs <dir> <dir>]"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
V = HERE.parent
sys.path.insert(0, str(HERE))
import reeval_x as RX  # noqa: E402

PB = V / "runs_exp_pb100"
XR = V / "runs_x_rees46"
TD3 = V / "runs_ds3" / "TEST" / "REEVAL"
TP2 = V / "runs_p2" / "TEST" / "REEVAL"
VD3 = V / "runs_ds3" / "REEVAL_A40"
TAG = {"PRACTICAL_BEST": "PB", "CONTROLLED_BUDGET": "CB"}
FILE = {"PRACTICAL_BEST": "best.pt", "CONTROLLED_BUDGET": "endpoint_6.0.pt"}
STEM = {"PRACTICAL_BEST": "best", "CONTROLLED_BUDGET": "endpoint_6.0"}


def fl_entry(key, x_arm, label, run_dir: Path, seed, rule, scorer="reeval", kind="fl"):
    cdir = next((run_dir / "ckpt").iterdir())
    ck = cdir / FILE[rule]
    return {"key": key, "x_arm": x_arm, "arm_label": label, "scorer": scorer, "ckpt": str(ck), "ckpt_kind": kind,
            "checkpoint_sha256": RX.sha_file(ck), "checkpoint_id": f"{cdir.name}:{STEM[rule]}", "run_dir": str(run_dir),
            "variant": "SASREC_D64_B2", "recipe_id": "SMALL_SASREC_D64_B2", "seed": seed, "rule": rule,
            "role": "MAIN_TEST", "out_dir": str(XR / "TEST" / "REEVAL" / key)}


def confirm(name, arm, seed, rule, d: Path, role_file):
    p = d / "evals" / role_file
    return name, {"arm": arm, "seed": seed, "rule": rule, "ranks": str(p), "sha256": RX.sha_file(p) if p.is_file() else None}


def part_a():
    arms = []
    pre = PB / "PRETRAIN_FOR_FL_SMALL_SASREC_D64_B2_PB100K_s2026" / "sealed" / "checkpoints" / "endpoint_6.0.pt"
    arms.append({"key": "PRE100K_s2026_CB", "x_arm": "PRE100K", "arm_label": "PRE", "scorer": "reeval_pb100",
                 "ckpt": str(pre), "ckpt_kind": "central", "checkpoint_sha256": RX.sha_file(pre),
                 "checkpoint_id": "PRETRAIN_FOR_FL_SMALL_SASREC_D64_B2_s2026:endpoint_6.0",
                 "run_dir": str(pre.parents[2]), "variant": "SASREC_D64_B2", "recipe_id": "SMALL_SASREC_D64_B2",
                 "seed": 2026, "rule": "CONTROLLED_BUDGET", "role": "MAIN_TEST",
                 "out_dir": str(XR / "TEST" / "REEVAL" / "PRE100K_s2026_CB"),
                 "note": "theta_pre(100k) = its endpoint_6.0 (no selection, as PRE); the same rank file serves the "
                         "PRACTICAL_BEST pairs (PRE is rule-free in practice; confirmatory PRE PB = CB bytes)"})
    for x, lab, rd in (("FA_WARM_PB100K", "FA_WARM", PB / "FA_WARM_PB100K_s2026"),
                       ("FA_WARM_FROZEN_PB100K", "FA_WARM_FROZEN", PB / "FA_WARM_FROZEN_PB100K_s2026")):
        for rule in ("PRACTICAL_BEST", "CONTROLLED_BUDGET"):
            arms.append(fl_entry(f"{x}_s2026_{TAG[rule]}", x, lab, rd, 2026, rule))
    best = PB / "FA_WARM_PB100K_s2026" / "ckpt" / "FL_FA_WARM_SMALL_s2026" / "best.pt"
    dev = [{"key": "FT_FA_WARM_PB100K_s2026", "x_arm": "FT_FA_WARM_PB100K", "scorer": "device", "role": "MAIN_TEST",
            "seed": 2026, "rule": "PRACTICAL_BEST", "base": str(best), "base_sha256": RX.sha_file(best),
            "out_dir": str(XR / "TEST" / "DEV" / "FT_FA_WARM_PB100K_MAIN_TEST_s2026"),
            "device_args": ["--arm", "FT_FA_WARM", "--seed", "2026", "--role", "MAIN_TEST", "--run-dir", "@OUT@",
                            "--n-shards", "64", "--base", str(best), "--base-kind", "fa_warm"],
            "note": "OPTIONAL: per-user local FT on top of the best PB100 FL model (VALIDATION 0.1513 > FROZEN); "
                    "about 3.5 h GPU (sitting step 7 MEASURED per FT part)"}]
    rc = json.loads((V / "data_pretrain_pb100" / "receipts" / "DATA10_BAND_RECEIPT.json").read_text())
    env = {"PB100_NDEC": rc["n_eligible_examples"], "PB100_SHA": rc["population"]["population_sha256_selection_key_v2"]}
    pinned = ["exp_pb100/reeval_pb100.py", "exp_pb100/pb100_patch.py", "x_rees46/reeval_x.py", "x_rees46/x_analysis.py",
              "tools/p2_report.py", "tools/retention_report.py", "tools/g2_run.py", "tools/test_exposure_register.py"]
    # analysis inputs: exploratory names -> rank files (TEST look) and the VALIDATION proof equivalents
    t, v = {}, {}
    for a in arms:
        rf = a["rule"]
        for rule in (("PRACTICAL_BEST", "CONTROLLED_BUDGET") if a["x_arm"] == "PRE100K" else (rf,)):
            nm = f"{a['x_arm']}|{TAG[rule]}"
            t[nm] = {"arm": a["x_arm"], "seed": 2026, "rule": rule,
                     "ranks": str(Path(a["out_dir"]) / "evals" / "MAIN_TEST.ranks_int32.npy"), "sha256": None}
            v[nm] = {"arm": a["x_arm"], "seed": 2026, "rule": rule,
                     "ranks": str(XR / "PROOF" / f"{a['key']}_MAIN" / "evals" / "MAIN.ranks_int32.npy"), "sha256": None}
    for rule in ("PRACTICAL_BEST", "CONTROLLED_BUDGET"):
        g = TAG[rule]
        for n, arm, d3, p2 in (("PRE", "PRE", f"PRE_s2026_{g}", None), ("C_FULL", "C_FULL", f"C_FULL_s2026_{g}", None),
                               ("FA_WARM_FROZEN", "FA_WARM_FROZEN", f"FA_WARM_FROZEN_s2026_{g}", None),
                               ("FA_WARM", "FA_WARM", f"FA_WARM_s2026_{g}", None),
                               ("P2_WSTAR", "P2_WSTAR", None, f"P2_WSTAR_s2026_{g}")):
            k, e = confirm(f"{n}|{g}", arm, 2026, rule, (TP2 / p2) if p2 else (TD3 / d3), "MAIN_TEST.ranks_int32.npy")
            t[k] = e
            if p2 is None:
                k, e = confirm(f"{n}|{g}", arm, 2026, rule, VD3 / d3, "MAIN.ranks_int32.npy")
                v[k] = e
    con = []
    for g in ("PB", "CB"):
        con += [{"id": f"A1_FA_WARM_PB100K-PRE100K|{g}", "kind": "delta", "pairs": [[f"FA_WARM_PB100K|{g}", f"PRE100K|{g}"]],
                 "note": "the K21 analogue on the 100k base (unfrozen FL - its own server model)"},
                {"id": f"A1f_FA_WARM_FROZEN_PB100K-PRE100K|{g}", "kind": "delta",
                 "pairs": [[f"FA_WARM_FROZEN_PB100K|{g}", f"PRE100K|{g}"]], "note": "K21 analogue, frozen items"},
                {"id": f"A2_FA_WARM_PB100K-PRE|{g}", "kind": "delta", "pairs": [[f"FA_WARM_PB100K|{g}", f"PRE|{g}"]],
                 "note": "vs the 1.2M-user theta_pre (does FL on MAIN from a 100k base reach the big server model?)"},
                {"id": f"A2f_FA_WARM_FROZEN_PB100K-PRE|{g}", "kind": "delta",
                 "pairs": [[f"FA_WARM_FROZEN_PB100K|{g}", f"PRE|{g}"]]},
                {"id": f"A3_PRE100K-PRE|{g}", "kind": "delta", "pairs": [[f"PRE100K|{g}", f"PRE|{g}"]],
                 "note": "the base-size gap FL has to close"},
                {"id": f"A4_FA_WARM_PB100K-P2_WSTAR|{g}", "kind": "delta", "pairs": [[f"FA_WARM_PB100K|{g}", f"P2_WSTAR|{g}"]]}]
        for x in ("PRE100K", "FA_WARM_PB100K", "FA_WARM_FROZEN_PB100K", "PRE", "P2_WSTAR"):
            con.append({"id": f"R_{x}/C_FULL|{g}", "kind": "ratio", "pairs": [[f"{x}|{g}", f"C_FULL|{g}"]]})
    return arms, dev, env, pinned, t, v, con, "EXP_PB100_TEST_LOCK", "code_x1a"


def part_b(main_dirs):
    """X_FA (the tau-fix re-run): the selected cell's MAIN runs, seeds 2026 / 2027, both rules, scored by
    x_rees46/reeval_p2_x.py from code_x1; FT on top is NOT prepared (it needs a launch_device_p2 wrapper: not cheap)."""
    arms, t, v = [], {}, {}
    sel = json.loads((XR / "X_SELECTION_FINAL.json").read_text())
    rids, rshas = set(), set()
    for rd in map(Path, main_dirs):
        L = json.loads((rd / "LAUNCH_0.json").read_text())
        spec = L["arm_spec"]
        seed = int(spec["cfg"]["seed"])
        rids.add(L["run_id"].replace(f"_s{seed}", "_s{seed}"))
        rshas.add(L["p2_recipe"]["sha256"])
        for rule in ("PRACTICAL_BEST", "CONTROLLED_BUDGET"):
            e = fl_entry(f"X_FA_s{seed}_{TAG[rule]}", "X_FA", "X_FA", rd, seed, rule, scorer="reeval_p2_x")
            arms.append(e)
            nm = f"X_FA|{seed}|{TAG[rule]}"
            t[nm] = {"arm": "X_FA", "seed": seed, "rule": rule,
                     "ranks": str(Path(e["out_dir"]) / "evals" / "MAIN_TEST.ranks_int32.npy"), "sha256": None}
            v[nm] = {"arm": "X_FA", "seed": seed, "rule": rule,
                     "ranks": str(XR / "PROOF" / f"{e['key']}_MAIN" / "evals" / "MAIN.ranks_int32.npy"), "sha256": None}
    if len(rids) != 1 or len(rshas) != 1:
        sys.exit(f"the MAIN dirs disagree on run id format / recipe ({rids}, {rshas})")
    env = {"X_TAU": "1e-06", "X_RUN_ID_FMT": rids.pop(), "X_RECIPE_SHA": rshas.pop()}
    con = []
    for rule in ("PRACTICAL_BEST", "CONTROLLED_BUDGET"):
        g = TAG[rule]
        for seed in (2026, 2027):
            for n, arm, d in (("PRE", "PRE", TD3 / f"PRE_s2026_{g}"), ("C_FULL", "C_FULL", TD3 / f"C_FULL_s{seed}_{g}"),
                              ("FA_WARM_FROZEN", "FA_WARM_FROZEN", TD3 / f"FA_WARM_FROZEN_s{seed}_{g}"),
                              ("FA_WARM", "FA_WARM", TD3 / f"FA_WARM_s{seed}_{g}"),
                              ("P2_WSTAR", "P2_WSTAR", TP2 / f"P2_WSTAR_s{seed}_{g}")):
                if n == "PRE" and seed == 2027:
                    continue
                k, e = confirm(f"{n}|{seed}|{g}", arm, 2026 if n == "PRE" else seed, rule, d, "MAIN_TEST.ranks_int32.npy")
                t[k] = e
                if n in ("PRE", "C_FULL"):
                    k, e = confirm(f"{n}|{seed}|{g}", arm, 2026 if n == "PRE" else seed, rule, VD3 / d.name,
                                   "MAIN.ranks_int32.npy")
                    v[k] = e
        X = [f"X_FA|{s}|{g}" for s in (2026, 2027)]
        con += [{"id": f"B1_X_FA-PRE|{g}", "kind": "delta", "pairs": [[x, f"PRE|2026|{g}"] for x in X],
                 "note": "the K21 / Q1 analogue: tau-fixed warm FL (unfrozen) - theta_pre; seed mean, PRE shared"},
                {"id": f"B2_X_FA-P2_WSTAR|{g}", "kind": "delta",
                 "pairs": [[f"X_FA|{s}|{g}", f"P2_WSTAR|{s}|{g}"] for s in (2026, 2027)],
                 "note": "tau fix (+ unfrozen items) vs the confirmatory W* (tau 1e-3, frozen)"},
                {"id": f"B3_X_FA-FA_WARM_FROZEN|{g}", "kind": "delta",
                 "pairs": [[f"X_FA|{s}|{g}", f"FA_WARM_FROZEN|{s}|{g}"] for s in (2026, 2027)]},
                {"id": f"B4_X_FA-FA_WARM|{g}", "kind": "delta",
                 "pairs": [[f"X_FA|{s}|{g}", f"FA_WARM|{s}|{g}"] for s in (2026, 2027)]},
                {"id": f"R_X_FA/C_FULL|{g}", "kind": "ratio",
                 "pairs": [[f"X_FA|{s}|{g}", f"C_FULL|{s}|{g}"] for s in (2026, 2027)]},
                {"id": f"R_P2_WSTAR/C_FULL|{g}", "kind": "ratio",
                 "pairs": [[f"P2_WSTAR|{s}|{g}", f"C_FULL|{s}|{g}"] for s in (2026, 2027)]}]
    pinned = ["x_rees46/reeval_p2_x.py", "x_rees46/reeval_x.py", "x_rees46/launch_fl_x.py", "x_rees46/x_analysis.py",
              "x_rees46/x_select.py", "runs_x_rees46/X_SELECTION_FINAL.json", "tools/p2_report.py",
              "tools/retention_report.py", "tools/g2_run.py", "tools/test_exposure_register.py"]
    return arms, [], env, pinned, t, v, con, "EXP_XREES46_TEST_LOCK", "code_x1"


def part_c(main_dirs):
    """X2 (EXP lock v2): the strongest PB100 FL (laneX_PB.sh winner) MAIN s2026 / s2027, both rules, scored by
    reeval_p2_x.py (init pin = theta_pre(100k)); compared with PRE100K / FA_WARM_PB100K (the A look's TEST rank files)
    and the confirmatory PRE / C_FULL / P2_WSTAR. FT on top is not prepared (no cheap FT path for a P2_W PB100 base)."""
    arms, t, v = [], {}, {}
    rids, rshas, inits = set(), set(), set()
    for rd in map(Path, main_dirs):
        L = json.loads((rd / "LAUNCH_0.json").read_text())
        spec = L["arm_spec"]
        seed = int(spec["cfg"]["seed"])
        rids.add(L["run_id"].replace(f"_s{seed}", "_s{seed}"))
        rshas.add(L["p2_recipe"]["sha256"])
        inits.add((spec["init_from"]["checkpoint_sha256"], spec["init_from"]["state_digest"]))
        for rule in ("PRACTICAL_BEST", "CONTROLLED_BUDGET"):
            e = fl_entry(f"X2_FA_PB100K_s{seed}_{TAG[rule]}", "X2_FA_PB100K", "X_FA", rd, seed, rule, scorer="reeval_p2_x")
            arms.append(e)
            nm = f"X2|{seed}|{TAG[rule]}"
            t[nm] = {"arm": "X2_FA_PB100K", "seed": seed, "rule": rule,
                     "ranks": str(Path(e["out_dir"]) / "evals" / "MAIN_TEST.ranks_int32.npy"), "sha256": None}
            v[nm] = {"arm": "X2_FA_PB100K", "seed": seed, "rule": rule,
                     "ranks": str(XR / "PROOF" / f"{e['key']}_MAIN" / "evals" / "MAIN.ranks_int32.npy"), "sha256": None}
    if len(rids) != 1 or len(rshas) != 1 or len(inits) != 1:
        sys.exit(f"the MAIN dirs disagree ({rids}, {rshas}, {inits})")
    ish, idg = inits.pop()
    pre100 = json.loads((PB / "PRETRAIN_FOR_FL_SMALL_SASREC_D64_B2_PB100K_s2026" / "PRETRAIN_FOR_FL_INIT.json").read_text())
    eps_ = {(float(json.loads((Path(d) / "LAUNCH_0.json").read_text())["arm_spec"]["cfg"]["endpoint_efe"]),
             int(json.loads((Path(d) / "LAUNCH_0.json").read_text())["arm_spec"]["cfg"]["solver"]["passes"]))
            for d in main_dirs}
    if len(eps_) != 1:
        sys.exit(f"the MAIN dirs disagree on budget / passes {eps_}")
    efe_, passes_ = eps_.pop()
    env = {"X_TAU": "1e-06", "X_RUN_ID_FMT": rids.pop(), "X_RECIPE_SHA": rshas.pop(), "X_INIT_SHA": ish,
           "X_INIT_DIGEST": idg, "X_EFE": format(efe_, "g"), "X_PASSES": str(passes_)}
    con = []
    for rule in ("PRACTICAL_BEST", "CONTROLLED_BUDGET"):
        g = TAG[rule]
        for n, key in (("PRE100K", "PRE100K_s2026_CB"), ("FA_WARM_PB100K", f"FA_WARM_PB100K_s2026_{g}")):
            p = XR / "TEST" / "REEVAL" / key / "evals" / "MAIN_TEST.ranks_int32.npy"
            t[f"{n}|2026|{g}"] = {"arm": n, "seed": 2026, "rule": rule, "ranks": str(p),
                                   "sha256": RX.sha_file(p) if p.is_file() else None}
            q = XR / "PROOF" / f"{key}_MAIN" / "evals" / "MAIN.ranks_int32.npy"
            v[f"{n}|2026|{g}"] = {"arm": n, "seed": 2026, "rule": rule, "ranks": str(q),
                                   "sha256": RX.sha_file(q) if q.is_file() else None}
        for seed in (2026, 2027):
            for n, d in (("PRE", TD3 / f"PRE_s2026_{g}"), ("C_FULL", TD3 / f"C_FULL_s{seed}_{g}"),
                         ("P2_WSTAR", TP2 / f"P2_WSTAR_s{seed}_{g}")):
                if n == "PRE" and seed == 2027:
                    continue
                sd = 2026 if n == "PRE" else seed
                k, e = confirm(f"{n}|{seed}|{g}", n, sd, rule, d, "MAIN_TEST.ranks_int32.npy")
                t[k] = e
                if n != "P2_WSTAR":
                    k, e = confirm(f"{n}|{seed}|{g}", n, sd, rule, VD3 / d.name, "MAIN.ranks_int32.npy")
                    v[k] = e
        X = [f"X2|{s}|{g}" for s in (2026, 2027)]
        con += [{"id": f"C1_X2-PRE100K|{g}", "kind": "delta", "shared": True, "pairs": [[x, f"PRE100K|2026|{g}"] for x in X],
                 "note": "the K21 analogue on the 100k base: the strongest FL - its own server model (PRE100K shared)"},
                {"id": f"C2_X2-FA_WARM_PB100K|{g}", "kind": "delta", "shared": True,
                 "pairs": [[x, f"FA_WARM_PB100K|2026|{g}"] for x in X], "note": "tau fix + grid vs the S7 recipe run"},
                {"id": f"C3_X2-PRE|{g}", "kind": "delta", "pairs": [[x, f"PRE|2026|{g}"] for x in X],
                 "note": "does FL from the 100k base reach the 1.2M-user theta_pre?"},
                {"id": f"C4_X2-P2_WSTAR|{g}", "kind": "delta", "pairs": [[f"X2|{s}|{g}", f"P2_WSTAR|{s}|{g}"] for s in (2026, 2027)]},
                {"id": f"R_X2/C_FULL|{g}", "kind": "ratio", "pairs": [[f"X2|{s}|{g}", f"C_FULL|{s}|{g}"] for s in (2026, 2027)]}]
    pinned = ["x_rees46/reeval_p2_x.py", "x_rees46/reeval_x.py", "x_rees46/launch_fl_x2.py", "x_rees46/x_analysis.py",
              "x_rees46/x_select2.py", "x_rees46/x2_admissions.py", "x_rees46/x_execute_fix.py",
              "runs_x_rees46/PB100/X2_SELECTION_FINAL.json", "tools/p2_report.py",
              "tools/retention_report.py", "tools/g2_run.py", "tools/test_exposure_register.py"]
    return arms, [], env, pinned, t, v, con, "EXP_PB100_TEST_LOCK_v2", "code_x1"


def part_d(main_dirs):
    """EXP_XREES46_TEST_LOCK (T2): Part B X_FA warm (keys XW_*) + C1 X_FA cold (keys XC_*), MAIN s2026 + s2027 each, one
    scoring tree code_x1 for all (byte copy of code_p2r4: the P2-style warm arm needs reeval_p2 = code_p2r4; the cold arm
    was trained by code_x1's launch_fl and is scored by code_x1's v4int/reeval.py, byte-identical to code_ds3r7's)."""
    warm = [d for d in main_dirs if "/runs_x_rees46/MAIN/" in d]
    cold = [d for d in main_dirs if "/runs_x_rees46/COLD/MAIN/" in d]
    arms, _dev, env, pinned, t, v, con, _doc, tree = part_b(warm)
    for e in arms:
        e["key"] = "XW" + e["key"][1:]
        e["out_dir"] = str(XR / "TEST" / "REEVAL" / e["key"])
        e["x_arm"] = "X_FA_WARM"
    for d, test in ((t, True), (v, False)):
        for k, e in d.items():
            if e["arm"] == "X_FA":
                e["arm"] = "X_FA_WARM"
                key = [a for a in arms if a["seed"] == e["seed"] and a["rule"] == e["rule"]][0]["key"]
                base = XR / "TEST" / "REEVAL" / key if test else XR / "PROOF" / f"{key}_MAIN"
                e["ranks"] = str(base / "evals" / ("MAIN_TEST.ranks_int32.npy" if test else "MAIN.ranks_int32.npy"))
    con = [c for c in con if not c["id"].startswith(("B3_", "B4_"))]
    for c in con:
        c["id"] = c["id"].replace("X_FA", "X_FA_WARM", 1)
    seeds_c = []
    for rd in map(Path, cold):
        L = json.loads((rd / "LAUNCH_0.json").read_text())
        spec = L["arm_spec"]
        seed = int(spec["cfg"]["seed"])
        if spec.get("init_from") is not None or float(spec["cfg"]["server_opt"]["tau"]) != 1e-6:
            sys.exit(f"{rd}: not a cold tau 1e-6 run")
        seeds_c.append(seed)
        for rule in ("PRACTICAL_BEST", "CONTROLLED_BUDGET"):
            e = fl_entry(f"XC_FA_COLD_s{seed}_{TAG[rule]}", "X_FA_COLD", "FA", rd, seed, rule, scorer="reeval_cold_x")
            arms.append(e)
            nm = f"X_FA_COLD|{seed}|{TAG[rule]}"
            t[nm] = {"arm": "X_FA_COLD", "seed": seed, "rule": rule,
                     "ranks": str(Path(e["out_dir"]) / "evals" / "MAIN_TEST.ranks_int32.npy"), "sha256": None}
            v[nm] = {"arm": "X_FA_COLD", "seed": seed, "rule": rule,
                     "ranks": str(XR / "PROOF" / f"{e['key']}_MAIN" / "evals" / "MAIN.ranks_int32.npy"), "sha256": None}
    for rule in ("PRACTICAL_BEST", "CONTROLLED_BUDGET"):
        g = TAG[rule]
        for seed in (2026, 2027):
            for n, d in (("FA", TD3 / f"FA_s{seed}_{g}"), ("C_SMALL", TD3 / f"C_SMALL_s{seed}_{g}")):
                k, e = confirm(f"{n}|{seed}|{g}", n, seed, rule, d, "MAIN_TEST.ranks_int32.npy")
                t[k] = e
            k, e = confirm(f"C_SMALL|{seed}|{g}", "C_SMALL", seed, rule, VD3 / f"C_SMALL_s{seed}_{g}",
                           "MAIN.ranks_int32.npy")
            v[k] = e
        X = [f"X_FA_COLD|{s}|{g}" for s in (2026, 2027)]
        con += [{"id": f"D1_X_FA_COLD-FA|{g}", "kind": "delta",
                 "pairs": [[f"X_FA_COLD|{s}|{g}", f"FA|{s}|{g}"] for s in (2026, 2027)],
                 "note": "tau fix on the cold arm: X_FA_cold - the confirmatory FA (tau 1e-3), seed-paired"},
                {"id": f"D2_X_FA_COLD-PRE|{g}", "kind": "delta", "pairs": [[x, f"PRE|2026|{g}"] for x in X]},
                {"id": f"R_X_FA_COLD/C_FULL|{g}", "kind": "ratio",
                 "pairs": [[f"X_FA_COLD|{s}|{g}", f"C_FULL|{s}|{g}"] for s in (2026, 2027)], "note": "K17 analogue"},
                {"id": f"R_X_FA_COLD/C_SMALL|{g}", "kind": "ratio",
                 "pairs": [[f"X_FA_COLD|{s}|{g}", f"C_SMALL|{s}|{g}"] for s in (2026, 2027)], "note": "K3 analogue"},
                {"id": f"R_FA/C_FULL|{g}", "kind": "ratio",
                 "pairs": [[f"FA|{s}|{g}", f"C_FULL|{s}|{g}"] for s in (2026, 2027)],
                 "note": "the confirmatory K17 recomputed on the same W, for reference"}]
    pinned = sorted(set(pinned) | {"x_rees46/reeval_cold_x.py", "x_rees46/launch_fl_x3.py", "x_rees46/x_execute_fix.py",
                                   "x_rees46/x_select2.py", "runs_x_rees46/COLD/X_COLD_SELECTION_FINAL.json"})
    pre_dir = V / "runs_pretrain" / "PRETRAIN_FOR_FL_SMALL_SASREC_D64_B2_s2026"
    pre_ck = pre_dir / "sealed" / "checkpoints" / "endpoint_6.0.pt"
    pre_sha = RX.sha_file(pre_ck)
    init = json.loads((pre_dir / "PRETRAIN_FOR_FL_INIT.json").read_text())
    pre = {"theta_pre_checkpoint": str(pre_ck), "checkpoint_sha256": pre_sha,
           "init_record": str(pre_dir / "PRETRAIN_FOR_FL_INIT.json"),
           "init_record_sha256": RX.sha_file(pre_dir / "PRETRAIN_FOR_FL_INIT.json"),
           "run_id": init.get("run_id"), "train_manifest": init.get("train_manifest"),
           "registered_in": "company/gates/ALL_METHODS_LOCK.json (PRE) and P2_METHODS_LOCK phase1_reused",
           "x_fa_warm_init_from_equals_theta_pre": all(
               json.loads((Path(d) / "LAUNCH_0.json").read_text())["arm_spec"]["init_from"]["checkpoint_sha256"]
               == pre_sha for d in warm),
           "x_fa_cold_init": "none (cold, from scratch)"}
    extra = {"pre_provenance": pre,
             "seed_sets": {"X_FA_WARM": sorted({a["seed"] for a in arms if a["x_arm"] == "X_FA_WARM"}),
                           "X_FA_COLD": sorted(set(seeds_c))},
             "scoring_tree_reason": part_d.__doc__}
    return arms, [], env, pinned, t, v, con, "EXP_XREES46_TEST_LOCK", tree, extra


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=("A", "B", "C", "D"))
    ap.add_argument("--x-main-dirs", nargs="*", default=[])
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    if a.out.exists():
        sys.exit(f"{a.out} exists: a lock is written once")
    res = (part_a() if a.part == "A" else part_b(a.x_main_dirs) if a.part == "B" else part_c(a.x_main_dirs) if a.part == "C"
           else part_d(a.x_main_dirs))
    arms, dev, env, pinned, t, v, con, doc, tree = res[:9]
    extra = res[9] if len(res) > 9 else {}
    dig, n = RX.tree_digest(V / tree)
    lock = {"document": doc, "format": RX.FORMAT, "draft": False, "label": RX.LABEL, "study": "PPSI V4 X-REES46",
            "authority": "owner approval DL 539 (exploratory TEST looks, labelled POST_HOC_EXPLORATORY, reported beside "
                         "the confirmatory T0a results, never instead of them); Fable gate + register entry by Ops",
            "binds": "checkpoints by sha256, the scoring tree by digest, the pinned scripts, the analysis spec and "
                     "contrasts; no metric value was opened to build it (VALIDATION values of these runs were seen, DL 539)",
            "bootstrap_seed": 20261004, "n_resamples": 1000,
            "code": {"scoring_tree": {"name": tree, "aggregate_sha256": dig, "n_files": n,
                                      "copied_from": {"code_x1a": "code_ds3r7", "code_x1": "code_p2r4"}[tree] + " (unchanged bytes)",
                                      "rule": "sha256(compact sort_keys JSON of {posix relpath: file sha256})"},
                     "pinned_files": {p: RX.sha_file(V / p) for p in pinned}},
            "scorer_env": env, "sealed_arms": arms, "device_optional": dev,
            "analysis": {"script": "x_rees46/x_analysis.py", "role": "MAIN_TEST", "entries": t, "contrasts": con,
                         "validation_proof_entries": v,
                         "confirmatory_inputs_read_only": "runs_ds3/TEST/REEVAL, runs_p2/TEST/REEVAL (sha256 bound above)"},
            "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra}
    lock["self_sha256"] = RX.self_sha(lock)
    a.out.write_text(json.dumps(lock, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps({"lock": str(a.out), "file_sha256": RX.sha_file(a.out), "self_sha256": lock["self_sha256"],
                      "n_arms": len(arms), "n_device": len(dev), "tree": tree, "tree_digest": dig}))


if __name__ == "__main__":
    main()
