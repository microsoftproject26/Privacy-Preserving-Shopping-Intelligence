"""finalize_fo.py (owner: Opus) — gate 2(b) + deliverables of SASREC_FEATURE_OPT_V1 (VALIDATION_SELECT only).

Reads the probe, lean-parity and screen results and applies Fable's gate 2(b):
- delta(candidate - control) >= max(0.002, 2*sigma_hat);
- paired user-bootstrap 95% CI lower bound > 0;
- micro R@20(candidate) >= control - 0.001.
At most one finalist; a finalist sets PAIR_REQUIRED.

Writes: FEATURE_SELECTION_RESULTS.csv, SELECTED_FEATURES.json, TRANSFORM_STATE.json,
FINAL_FEATURE_OPT_RESULTS.json and state/FINAL_DECISION.json. It never overwrites and never reads VC/TEST.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import time

import numpy as np

import fo_env
import feature_view as fv
import eval_full

GATE_B_MIN, R20_GUARD = 0.002, 0.001
# Review C fix (HIGH): the hygiene deliverable's equivalence claim is gated (thresholds set by Fable)
PARITY_MAX_ABS_LOGIT, PARITY_MIN_RANK_IDENTICAL, ROTATION_MAX_RESIDUAL = 1e-3, 0.999, 1e-6


def jl(*p):
    return json.loads(fo_env.study_path(*p).read_text())


def hygiene_parity(lean: dict, dupf: dict) -> dict:
    """PARITY_OK only if every assumed relation holds on the FULL population and the folded model
    reproduces the incumbent on the cohort; otherwise PARITY_FAILED (and no equivalence claim)."""
    counts = {f"{sp}:{k}": v for sp, d in dupf["checks"].items() for k, v in d.items() if k != "local_hour_rotation_max_residual"}
    resid = max(float(d.get("local_hour_rotation_max_residual", float("inf"))) for d in dupf["checks"].values())
    checks = {"duplicates_full_population_label": dupf.get("label") == "FULL_POPULATION",
              "all_duplicate_and_dead_counts_zero": all(int(v) == 0 for v in counts.values()) and bool(counts),
              "rotation_residual_le_1e-6": resid <= ROTATION_MAX_RESIDUAL,
              "max_abs_logit_le_1e-3": float(lean["max_abs_logit_diff_first_1024"]) <= PARITY_MAX_ABS_LOGIT,
              "rank_identical_ge_0.999": float(lean["ranks_identical_fraction"]) >= PARITY_MIN_RANK_IDENTICAL}
    return {"status": "PARITY_OK" if all(checks.values()) else "PARITY_FAILED", "checks": checks,
            "nonzero_counts": {k: v for k, v in counts.items() if int(v)}, "rotation_residual": resid}


def main() -> int:
    if fo_env.study_path("state", "FINAL_DECISION.json").exists():
        print("already finalized"); return 0
    probes, noise, pin = jl("audit", "GROUP_PROBE_RESULTS.json"), jl("config", "NOISE_REFERENCE.json"), jl("config", "BASELINE_PIN.json")
    if not fo_env.study_path("audit", "LEAN_PARITY.json").exists():
        raise RuntimeError("audit/LEAN_PARITY.json missing: the hygiene deliverable cannot be assessed (rerun probes stage)")
    lean = jl("audit", "LEAN_PARITY.json")
    m1fit = jl("audit", "M1_FIT.json")
    parity = hygiene_parity(lean, jl("audit", "DUPLICATES_FULL.json"))
    c = np.load(fo_env.study_path("runs", "probes", "COHORT_DS.npz"))
    ds = {k: c[k] for k in ("decision_id", "target_class", "eval_mask", "eval_user_key", "client_idx")}
    arms = {}
    for arm in ("CONTROL", "C1_D3", "C2"):
        p = fo_env.study_path("runs", "screen", arm, "result.json")
        if p.exists():
            r = json.loads(p.read_text())
            rk = np.load(fo_env.study_path("runs", "screen", arm, "cohort_ranks.npz"))
            if not np.array_equal(rk["decision_id"], ds["decision_id"]):
                raise RuntimeError(f"{arm}: cohort order mismatch")
            arms[arm] = (r, rk["ranks"].astype(np.int64))
    if "CONTROL" not in arms:
        raise RuntimeError("screen CONTROL missing")
    thr = max(GATE_B_MIN, 2.0 * float(noise["sigma_hat"]))
    ctrl_r, ctrl_rank = arms["CONTROL"]
    screen = {}
    for arm, (r, rank) in arms.items():
        if arm == "CONTROL":
            continue
        ka, va = eval_full.per_user_metric(rank, ds, "mrr", 20)
        kb, vb = eval_full.per_user_metric(ctrl_rank, ds, "mrr", 20)
        boot = eval_full.paired_bootstrap(ka, va, kb, vb, n=1000, seed=20260926)
        d = r["cohort_metrics"]["macro_mrr@20"] - ctrl_r["cohort_metrics"]["macro_mrr@20"]
        dr = r["cohort_metrics"]["micro_recall@20"] - ctrl_r["cohort_metrics"]["micro_recall@20"]
        g = {"delta_ge_threshold": d >= thr, "ci_lower_gt_0": boot["ci95"][0] > 0, "recall_guard": dr >= -R20_GUARD}
        screen[arm] = {"delta_macro_mrr@20_vs_control": d, "delta_micro_recall@20_vs_control": dr, "bootstrap": boot,
                       "threshold": thr, "gate_2b": g, "passes": all(g.values())}
    passing = sorted(((v["delta_macro_mrr@20_vs_control"], a) for a, v in screen.items() if v["passes"]), reverse=True)
    finalist = passing[0][1] if passing else None
    decision = "PAIR_REQUIRED" if finalist else "KEEP_CURRENT"
    base_m = probes["baseline"]["metrics"]

    # ---- FEATURE_SELECTION_RESULTS.csv: diagnostics vs adaptation clearly typed
    rows = [("BASELINE", "reference", base_m["macro_mrr@20"], 0.0, "", "", base_m["micro_recall@20"], "pinned checkpoint, cohort")]
    for n, r in probes["probes"].items():
        b = r["bootstrap_macro_mrr@20"]
        rows.append((n, "inference_probe_reliance(" + r["label"] + ")", r["metrics"]["macro_mrr@20"], r["delta_macro_mrr@20"],
                     b["ci95"][0], b["ci95"][1], r["metrics"]["micro_recall@20"], "not a retraining result"))
    if lean:
        rows.append(("LEAN_FOLD", f"exact_reparametrisation_parity({parity['status']})", lean["lean"]["macro_mrr@20"], lean["delta_macro_mrr@20"], "", "",
                     lean["lean"]["micro_recall@20"], f"max|dlogit|={lean['max_abs_logit_diff_first_1024']:.2e}"))
    for arm, (r, _) in arms.items():
        s = screen.get(arm, {})
        rows.append((f"SCREEN_{arm}", "warm_start_adaptation(exploratory)", r["cohort_metrics"]["macro_mrr@20"],
                     s.get("delta_macro_mrr@20_vs_control", 0.0), (s.get("bootstrap") or {}).get("ci95", ["", ""])[0],
                     (s.get("bootstrap") or {}).get("ci95", ["", ""])[1], r["cohort_metrics"]["micro_recall@20"],
                     "delta vs screen CONTROL" if arm != "CONTROL" else "matched control"))
    with open(fo_env.study_path("FEATURE_SELECTION_RESULTS.csv"), "x", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row", "evidence_type", "cohort_macro_mrr@20", "delta_macro_mrr@20", "ci95_lo", "ci95_hi", "cohort_micro_recall@20", "note"])
        w.writerows(rows)

    # ---- TRANSFORM_STATE.json + SELECTED_FEATURES.json (hygiene deliverable; function-preserving)
    spec = fv.normalizers().spec
    ts = {"view_id": "feature_view_v1_lean", "base_normalizers_sha256": fo_env.sha256_file(fv.normalizers_path()),
          "fit_split": "TRAIN", "fit_population": "TRAIN feature rows with the availability mask = 1 (valid only)",
          "m1_repair": {f: {"mask": fv.M1_FEATURES[f], "clip_lo": spec[f]["clip_lo"], "clip_hi": spec[f]["clip_hi"],
                            "old_mean": spec[f]["mean"], "old_std": spec[f]["std"],
                            "new_mean": m1fit["features"][f]["valid_mean"], "new_std": m1fit["features"][f]["valid_std"],
                            "unavailable_encoding": 0.0, "rule": "z=(clip(x)-new_mean)/new_std if mask==1 else 0.0"}
                        for f in fv.M1_FEATURES},
          "fold_map": {"numeric_affine": fv.DROP_NUM_AFFINE, "numeric_rotation": {"local_hour_(sin,cos)": "R(+3h)·utc_hour_(sin,cos)"},
                       "flag_duplicates": fv.DROP_FLAG_DUP, "dead_flags_dropped": list(fv.DROP_FLAG_DEAD),
                       "mask_duplicates": fv.DROP_MASK_DUP},
          "compatibility": "load only with the lean-folded weights (fold_lean_state_dict) of the pinned checkpoint; an old checkpoint with the lean view without folding is REFUSED",
          "parity": parity,
          "status": "VALID_EQUIVALENT_REPARAMETRISATION" if parity["status"] == "PARITY_OK" else "PARITY_FAILED_NOT_ADOPTED"}
    ts["view_sha256"] = hashlib.sha256(json.dumps(ts, sort_keys=True).encode()).hexdigest()
    sel = {"decision": decision, "model_in_use": "pinned incumbent (unchanged)" if decision == "KEEP_CURRENT" else "pending pair",
           "baseline_pin": {k: pin["chosen"][k] for k in ("seed", "epoch", "sha256", "macro_mrr20")},
           "lean_schema": fv.schema_record(), "transform_state_sha256": ts["view_sha256"],
           "per_column": {"kept_numeric": fv.LEAN_NUM, "kept_flags": fv.LEAN_FLAG, "kept_user_masks": fv.LEAN_UMASK,
                          "removed_as_exact_duplicate": list(fv.DROP_NUM_AFFINE) + list(fv.DROP_NUM_ROTATION) + list(fv.DROP_FLAG_DUP) + list(fv.DROP_MASK_DUP),
                          "removed_as_dead_constant_zero": list(fv.DROP_FLAG_DEAD),
                          "re_encoded_M1": list(fv.M1_FEATURES), "not_used_by_model_stored_only": ["category_code_token"]},
           "lean_view_status": parity["status"],
           "equivalence": ("lean view + folded weights reproduce the incumbent's predictions within the gated tolerances (LEAN_PARITY.json, TRANSFORM_STATE.parity)"
                           if parity["status"] == "PARITY_OK" else None),
           "selected_schema": "lean (equivalent re-parametrisation, hygiene only)" if parity["status"] == "PARITY_OK" else "current full F4 schema (lean parity failed; no equivalence claim)"}
    for name, obj in (("TRANSFORM_STATE.json", ts), ("SELECTED_FEATURES.json", sel)):
        fo_env.write_json_once(fo_env.study_path(name), obj)
    final = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "decision": decision, "finalist": finalist,
             "gate_2b_threshold": thr, "sigma_hat": noise, "screen": screen, "probes_gate_2a": probes["gate_2a"],
             "baseline_cohort": base_m, "lean_parity": dict({k: lean[k] for k in ("max_abs_logit_diff_first_1024", "ranks_identical_fraction",
                                                                                "delta_macro_mrr@20")}, gate=parity),
             "claim_scope": "adaptive development on VALIDATION_SELECT (previously used); no VC/TEST; not a fresh holdout"}
    for p in (fo_env.study_path("FINAL_FEATURE_OPT_RESULTS.json"), fo_env.study_path("state", "FINAL_DECISION.json")):
        fo_env.write_json_once(p, final)
    print(json.dumps({"decision": decision, "finalist": finalist, "threshold": thr}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
