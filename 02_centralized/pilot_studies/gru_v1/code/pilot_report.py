"""Pilot go/no-go EVIDENCE report -- Task T-M.

Reads the two pilot arms' recorded outputs (metrics.jsonl, summary.json,
resource_usage.json, p3_double_eval.json, per_user_mrr_end_to_end_best.json
under experiments/gru/T-M_<arm>/) and applies the FROZEN acceptance criteria
(DECISIONS.md addendum, mirrored in the T-M task brief) MECHANICALLY. Does
NOT decide GO/NO-GO -- only computes the numbers and the mechanical
PASS/FAIL/verdict each frozen threshold implies, with the number that
produced it. Writes artifacts/PILOT_RESULTS.json.

No threshold here is re-derived or tuned; every one is a literal transcription
of the frozen numbers in the task brief / DECISIONS.md addendum. If a number
below looks wrong, that is a run problem to report, not a reason to edit a
threshold.

Pure Python + numpy, no torch, no GPU needed. Run on Lobot after both arms
(run_pilot.py) have completed, from the GPU venv python (any python with
numpy works; convention kept the same as the rest of this task's tooling).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
WORKSPACE = CODE_DIR.parent
ARTIFACTS = WORKSPACE / "artifacts"
EXPERIMENTS_DIR = WORKSPACE / "experiments" / "gru"

RUN_ID_OF = {
    "F0_item_only": "T-M_F0_item_only",
    "F4_rich_side_information": "T-M_F4_rich_side_information",
}
EXPECTED_EXPOSURES = 2_000_000
EMA_WINDOW = 200
EMA_ALPHA = 2.0 / (EMA_WINDOW + 1)

# ---------------------------------------------------------------------------
# FROZEN thresholds -- literal transcription of the task brief / DECISIONS.md
# addendum. Do not edit. Measured baselines transcribed the same way.
# ---------------------------------------------------------------------------
BASELINE_POP = {
    "micro_recall@20": 0.04890, "micro_mrr@20": 0.01145,
    "macro_recall@20": 0.08620, "macro_mrr@20": 0.02189,
}
BASELINE_TRANS_MACRO_MRR20 = 0.13308

GO1_MICRO_RECALL_MIN = 0.07335
GO1_MACRO_RECALL_MIN = 0.12930
GO1_MICRO_MRR_MIN_EXCL = 0.01145
GO1_MACRO_MRR_MIN_EXCL = 0.02189

GO2_APPROVED_MIN = 0.13308
GO2_APPROVED_ONE_CHANGE_MIN = 0.11977  # inclusive lower bound of the middle band

OVERFIT_TOLERANCE = 0.002
P5_MAX_VRAM_GIB = 8.0
P5_MAX_RSS_MB = 24_000.0  # 24 GB, decimal (stricter than 24 GiB=24576 MB)

BOOTSTRAP_N_RESAMPLES = 1000
BOOTSTRAP_SEED = 2026


def log(msg: str) -> None:
    print(f"REPORT: {msg}", flush=True)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_run(group: str) -> dict:
    run_dir = EXPERIMENTS_DIR / RUN_ID_OF[group]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    resource_usage = json.loads((run_dir / "resource_usage.json").read_text(encoding="utf-8"))
    p3 = json.loads((run_dir / "p3_double_eval.json").read_text(encoding="utf-8"))
    per_user = json.loads(
        (run_dir / "per_user_mrr_end_to_end_best.json").read_text(encoding="utf-8"))

    training_records = []
    eval_records = []
    with (run_dir / "metrics.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if "eval_full" in rec:
                eval_records.append(rec)
            elif "loss" in rec:
                training_records.append(rec)

    if len(eval_records) != 2:
        raise RuntimeError(
            f"{group}: expected exactly 2 pass-end eval records, found {len(eval_records)}"
        )
    eval_records.sort(key=lambda r: r["pass_index"])

    return {
        "group": group, "run_dir": str(run_dir), "summary": summary,
        "resource_usage": resource_usage, "p3": p3, "per_user": per_user,
        "training_records": training_records, "eval_records": eval_records,
    }


# ---------------------------------------------------------------------------
# Learning-signal floor: training-loss EMA(200)
# ---------------------------------------------------------------------------

def ema_series(losses: list[float], alpha: float = EMA_ALPHA) -> list[float]:
    if not losses:
        return []
    out = [losses[0]]
    for x in losses[1:]:
        out.append(alpha * x + (1 - alpha) * out[-1])
    return out


def learning_signal_floor(run: dict) -> dict:
    pass1_losses = [r["loss"] for r in run["training_records"] if r["pass_index"] == 0]
    series = ema_series(pass1_losses)
    if len(series) < EMA_WINDOW:
        raise RuntimeError(
            f"{run['group']}: pass 1 has only {len(series)} updates, < EMA window {EMA_WINDOW}"
        )
    ema_first200 = series[EMA_WINDOW - 1]  # running EMA value at update 200
    ema_end_pass1 = series[-1]
    passed = ema_end_pass1 <= ema_first200 - 1.0
    return {
        "n_pass1_updates": len(pass1_losses),
        "ema_first200": ema_first200,
        "ema_end_pass1": ema_end_pass1,
        "delta": ema_first200 - ema_end_pass1,
        "required_delta_min": 1.0,
        "pass": passed,
    }


# ---------------------------------------------------------------------------
# P1 / P3 / P5
# ---------------------------------------------------------------------------

def check_p1(run: dict) -> dict:
    s = run["summary"]
    ru = run["resource_usage"]
    finished_2_passes = (s["pass_index"] == 1 and s["update_number"] == ru["target_update"])
    exact_exposures = s["consumed_decisions"] == EXPECTED_EXPOSURES
    zero_nonfinite = ru["n_nonfinite_losses"] == 0
    return {
        "finished_2_passes": finished_2_passes,
        "pass_index": s["pass_index"], "update_number": s["update_number"],
        "target_update": ru["target_update"],
        "consumed_decisions": s["consumed_decisions"],
        "expected_exposures": EXPECTED_EXPOSURES,
        "exact_exposures": exact_exposures,
        "n_updates_logged": ru["n_updates_logged"],
        "n_nonfinite_losses": ru["n_nonfinite_losses"],
        "zero_nonfinite": zero_nonfinite,
        "pass": finished_2_passes and exact_exposures and zero_nonfinite,
    }


def check_p3(run: dict) -> dict:
    identical = bool(run["p3"]["bit_identical"])
    return {
        "checkpoint": "best",
        "checkpoint_pass_index": run["p3"]["checkpoint_pass_index"],
        "checkpoint_update_number": run["p3"]["checkpoint_update_number"],
        "eval_a_macro_mrr@20": run["p3"]["eval_a"]["end_to_end"]["macro"]["mrr@20"],
        "eval_b_macro_mrr@20": run["p3"]["eval_b"]["end_to_end"]["macro"]["mrr@20"],
        "bit_identical": identical,
        "pass": identical,
    }


def check_p5(run: dict) -> dict:
    ru = run["resource_usage"]
    vram_ok = ru["peak_memory_reserved_gib"] <= P5_MAX_VRAM_GIB
    rss_ok = ru["peak_host_rss_mb_after"] <= P5_MAX_RSS_MB
    return {
        "peak_memory_reserved_gib": ru["peak_memory_reserved_gib"],
        "max_vram_gib": P5_MAX_VRAM_GIB,
        "vram_ok": vram_ok,
        "peak_host_rss_mb": ru["peak_host_rss_mb_after"],
        "max_rss_mb": P5_MAX_RSS_MB,
        "rss_ok": rss_ok,
        "pass": vram_ok and rss_ok,
    }


# ---------------------------------------------------------------------------
# GO-1 / GO-2 (F0 only, BEST checkpoint -- p3.eval_a, identical to eval_b)
# ---------------------------------------------------------------------------

def check_go1(f0_run: dict) -> dict:
    e2e = f0_run["p3"]["eval_a"]["end_to_end"]
    micro_recall = e2e["micro"]["recall@20"]
    macro_recall = e2e["macro"]["recall@20"]
    micro_mrr = e2e["micro"]["mrr@20"]
    macro_mrr = e2e["macro"]["mrr@20"]

    c_micro_recall = micro_recall >= GO1_MICRO_RECALL_MIN
    c_macro_recall = macro_recall >= GO1_MACRO_RECALL_MIN
    c_micro_mrr = micro_mrr > GO1_MICRO_MRR_MIN_EXCL
    c_macro_mrr = macro_mrr > GO1_MACRO_MRR_MIN_EXCL

    return {
        "checkpoint": "best (F0)",
        "micro_recall@20": micro_recall, "threshold_micro_recall_ge": GO1_MICRO_RECALL_MIN,
        "micro_recall_pass": c_micro_recall,
        "macro_recall@20": macro_recall, "threshold_macro_recall_ge": GO1_MACRO_RECALL_MIN,
        "macro_recall_pass": c_macro_recall,
        "micro_mrr@20": micro_mrr, "threshold_micro_mrr_gt": GO1_MICRO_MRR_MIN_EXCL,
        "micro_mrr_pass": c_micro_mrr,
        "macro_mrr@20": macro_mrr, "threshold_macro_mrr_gt": GO1_MACRO_MRR_MIN_EXCL,
        "macro_mrr_pass": c_macro_mrr,
        "pass": c_micro_recall and c_macro_recall and c_micro_mrr and c_macro_mrr,
    }


def check_go2(f0_run: dict) -> dict:
    macro_mrr = f0_run["p3"]["eval_a"]["end_to_end"]["macro"]["mrr@20"]
    if macro_mrr >= GO2_APPROVED_MIN:
        verdict = "APPROVED"
    elif macro_mrr >= GO2_APPROVED_ONE_CHANGE_MIN:
        verdict = "APPROVED_WITH_ONE_CHANGE"
    else:
        verdict = "NO_GO"
    return {
        "checkpoint": "best (F0)",
        "macro_mrr@20": macro_mrr,
        "trans_baseline_macro_mrr@20": BASELINE_TRANS_MACRO_MRR20,
        "approved_min": GO2_APPROVED_MIN,
        "approved_with_one_change_min": GO2_APPROVED_ONE_CHANGE_MIN,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Overfitting check (per arm, informational for F4; gating context is F0)
# ---------------------------------------------------------------------------

def check_overfitting(run: dict) -> dict:
    pass1, pass2 = run["eval_records"][0], run["eval_records"][1]
    m1 = pass1["eval_full"]["end_to_end"]["micro"]["mrr@20"]
    m2 = pass2["eval_full"]["end_to_end"]["micro"]["mrr@20"]
    passed = m2 >= m1 - OVERFIT_TOLERANCE
    return {
        "pass1_micro_mrr@20": m1, "pass2_micro_mrr@20": m2,
        "tolerance": OVERFIT_TOLERANCE, "required_min": m1 - OVERFIT_TOLERANCE,
        "pass": passed,
    }


# ---------------------------------------------------------------------------
# F4 vs F0 paired bootstrap (never gates the GRU decision)
# ---------------------------------------------------------------------------

def bootstrap_f4_vs_f0(f0_run: dict, f4_run: dict) -> dict:
    f0_keys = np.asarray(f0_run["per_user"]["eval_user_key"], dtype=np.int64)
    f4_keys = np.asarray(f4_run["per_user"]["eval_user_key"], dtype=np.int64)
    f0_mrr = np.asarray(f0_run["per_user"]["mrr@20_end_to_end"], dtype=np.float64)
    f4_mrr = np.asarray(f4_run["per_user"]["mrr@20_end_to_end"], dtype=np.float64)

    if f0_keys.shape != f4_keys.shape or not np.array_equal(f0_keys, f4_keys):
        # align by sorting both by key; verify identical key SETS first.
        if set(f0_keys.tolist()) != set(f4_keys.tolist()):
            raise RuntimeError("F0/F4 per-user eval_user_key sets differ -- cannot pair")
        f0_order = np.argsort(f0_keys)
        f4_order = np.argsort(f4_keys)
        f0_keys, f0_mrr = f0_keys[f0_order], f0_mrr[f0_order]
        f4_keys, f4_mrr = f4_keys[f4_order], f4_mrr[f4_order]

    n_users = f0_keys.shape[0]
    diff = f4_mrr - f0_mrr  # per-user paired difference
    point_estimate = float(diff.mean())

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    resample_means = np.empty(BOOTSTRAP_N_RESAMPLES, dtype=np.float64)
    for r in range(BOOTSTRAP_N_RESAMPLES):
        idx = rng.integers(0, n_users, size=n_users)
        resample_means[r] = diff[idx].mean()

    ci_lo, ci_hi = np.percentile(resample_means, [2.5, 97.5])
    return {
        "n_users": int(n_users),
        "n_resamples": BOOTSTRAP_N_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
        "metric": "macro_mrr@20_end_to_end, per-user paired difference (F4 - F0), best checkpoints",
        "point_estimate_mean_diff": point_estimate,
        "ci_95_lower": float(ci_lo),
        "ci_95_upper": float(ci_hi),
        "note": "never gates the GRU go/no-go decision",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    f0 = load_run("F0_item_only")
    f4 = load_run("F4_rich_side_information")

    result: dict = {"task": "T-M", "arms": {}}

    for run in (f0, f4):
        g = run["group"]
        p1 = check_p1(run)
        p3 = check_p3(run)
        p5 = check_p5(run)
        floor = learning_signal_floor(run)
        overfit = check_overfitting(run)
        log(f"{g}: P1={p1['pass']} P3={p3['pass']} P5={p5['pass']} "
            f"learning_floor={floor['pass']} overfit={overfit['pass']}")

        result["arms"][g] = {
            "run_id": RUN_ID_OF[g],
            "summary": run["summary"],
            "resource_usage": run["resource_usage"],
            "passes": {
                "pass_1": run["eval_records"][0]["eval_full"],
                "pass_2": run["eval_records"][1]["eval_full"],
            },
            "engineering_criteria": {
                "P1_finish_2_passes_zero_nonfinite_exact_exposures": p1,
                "P3_bit_identical_repeat_eval": p3,
                "P5_memory_bounds": p5,
                "learning_signal_floor_ema200": floor,
            },
            "overfitting_check": overfit,
        }

    go1 = check_go1(f0)
    go2 = check_go2(f0)
    log(f"GO-1 pass={go1['pass']}  GO-2 verdict={go2['verdict']}")

    bootstrap = bootstrap_f4_vs_f0(f0, f4)
    log(f"bootstrap F4-F0 point={bootstrap['point_estimate_mean_diff']:.6f} "
        f"CI95=[{bootstrap['ci_95_lower']:.6f}, {bootstrap['ci_95_upper']:.6f}]")

    result["baselines"] = {
        "POP": BASELINE_POP,
        "TRANS_macro_mrr@20": BASELINE_TRANS_MACRO_MRR20,
    }
    result["GO1_sanity_floor"] = go1
    result["GO2_vs_trans"] = go2
    result["bootstrap_F4_vs_F0"] = bootstrap

    all_p1 = result["arms"]["F0_item_only"]["engineering_criteria"][
        "P1_finish_2_passes_zero_nonfinite_exact_exposures"]["pass"] and \
        result["arms"]["F4_rich_side_information"]["engineering_criteria"][
        "P1_finish_2_passes_zero_nonfinite_exact_exposures"]["pass"]
    all_p3 = result["arms"]["F0_item_only"]["engineering_criteria"]["P3_bit_identical_repeat_eval"]["pass"] and \
        result["arms"]["F4_rich_side_information"]["engineering_criteria"]["P3_bit_identical_repeat_eval"]["pass"]
    all_p5 = result["arms"]["F0_item_only"]["engineering_criteria"]["P5_memory_bounds"]["pass"] and \
        result["arms"]["F4_rich_side_information"]["engineering_criteria"]["P5_memory_bounds"]["pass"]
    floor_f0 = result["arms"]["F0_item_only"]["engineering_criteria"]["learning_signal_floor_ema200"]["pass"]
    floor_f4 = result["arms"]["F4_rich_side_information"]["engineering_criteria"]["learning_signal_floor_ema200"]["pass"]

    result["mechanical_summary"] = {
        "P1_pass_both_arms": all_p1,
        "P3_pass_both_arms": all_p3,
        "P5_pass_both_arms": all_p5,
        "learning_signal_floor_pass_F0": floor_f0,
        "learning_signal_floor_pass_F4": floor_f4,
        "GO1_pass": go1["pass"],
        "GO2_verdict": go2["verdict"],
        "overfitting_pass_F0": result["arms"]["F0_item_only"]["overfitting_check"]["pass"],
        "overfitting_pass_F4": result["arms"]["F4_rich_side_information"]["overfitting_check"]["pass"],
    }

    out_path = ARTIFACTS / "PILOT_RESULTS.json"
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    log(f"wrote {out_path}")
    log(json.dumps(result["mechanical_summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
