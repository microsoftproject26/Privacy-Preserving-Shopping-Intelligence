"""v3 pre-registered 20k-update go/no-go watcher (owner: Opus). Registry v3 `go_no_go_20k`
(copied from Fable's deep-review VERIFICATION_PLAN before any v3 update).

Runs detached. Waits until BOTH v3 anchors have logged update 20,000, then evaluates on
TRAINING signals only (metrics.jsonl `update_agg`, 100-update means; windows of the 10 records
ending at the named update are used to reduce noise, declared here before any v3 data exists):
  - mean_loss(19k..20k) <= mean_loss(9k..10k) - 0.02
  - mean_grad_norm(19k..20k) <= 1.25 x mean_grad_norm(5k..6k)
  - E_mean_row_norm(20k) <= 0.5
  - [amendment 1, config/run_registry.v3_amendment_1.json] median row norms of E and W_out <= 5.0 absolute
    and W_out ||mean row|| <= 1e-3; the original 3x-init outcome is reported, not gating
  - every logged loss / grad norm finite
Writes review/V3_GATE_20K.json. On NO-GO it stops the v3 campaign (queue tmux, then the two
run sessions; only processes of this workspace) and writes review/V3_NO_GO.json; nothing is deleted.
Later descriptive checkpoints (cohort at 40k / 80k / epoch 1) are appended as they appear.
"""
from __future__ import annotations

import json
import math
import subprocess
import time
from pathlib import Path

S = Path("/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/sasrec_v1")
RUNS = {"A0_ID256": S / "runs" / "A0_ID256_s2026", "A1_RICH256": S / "runs" / "A1_RICH256_s2026"}
OUT = S / "review" / "V3_GATE_20K.json"
DESCRIPTIVE = {"40k_cohort": {"A0_ID256": 0.113, "A1_RICH256": 0.123}, "80k_A0_on_track": 0.140,
               "epoch1_A0_ge_GRU_epoch1": 0.156, "epoch1_A0_unlikely_below": 0.145}


def load(run: Path):
    aggs, diags = [], []
    p = run / "metrics.jsonl"
    if not p.exists():
        return aggs, diags
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("type") == "update_agg":
            aggs.append(r)
        elif r.get("type") == "diag_eval":
            diags.append(r)
    return aggs, diags


def window(aggs, end, key):
    vals = [a[key] for a in aggs if end - 1000 < a["update"] <= end]
    return sum(vals) / len(vals) if len(vals) == 10 else None


def evaluate(aggs):
    by_u = {a["update"]: a for a in aggs}
    l10, l20 = window(aggs, 10_000, "mean_loss"), window(aggs, 20_000, "mean_loss")
    g6, g20 = window(aggs, 6_000, "mean_grad_norm"), window(aggs, 20_000, "mean_grad_norm")
    e100, e20 = by_u[100]["emb"], by_u[20_000]["emb"]
    finite = all(math.isfinite(a["mean_loss"]) and math.isfinite(a["mean_grad_norm"]) for a in aggs if a["update"] <= 20_000)
    checks = {
        "loss_decreasing": l20 <= l10 - 0.02,
        "grad_norm_not_climbing": g20 <= 1.25 * g6,
        "E_common_mode_bounded": e20["E_mean_row_norm"] <= 0.5,
        "E_median_row_norm_le_5_abs": e20["E_median_row_norm"] <= 5.0,        # amendment 1 (Fable)
        "W_median_row_norm_le_5_abs": e20["W_median_row_norm"] <= 5.0,        # amendment 1 (Fable)
        "W_common_mode_le_1e-3": e20["W_mean_row_norm"] <= 1e-3,              # amendment 1 (Fable)
        "all_finite": finite}
    original = {"E_median_row_norm_le_3x_init": e20["E_median_row_norm"] <= 3.0 * e100["E_median_row_norm"],
                "W_median_row_norm_le_3x_init": e20["W_median_row_norm"] <= 3.0 * e100["W_median_row_norm"]}
    return {"go": all(checks.values()), "checks": checks, "original_criterion_4_outcome_reported_not_gating": original,
            "descriptive_E_common_mode_ratio": e20["E_mean_row_norm"] / max(1e-12, e20["E_median_row_norm"]),
            "values": {"loss_9k_10k": l10, "loss_19k_20k": l20, "gn_5k_6k": g6, "gn_19k_20k": g20,
                       "emb_at_100": e100, "emb_at_20k": e20}}


def stop_campaign(reason: str) -> None:
    for sess in ("sasrec-queue", "sasrec-A0_ID256-s2026", "sasrec-A1_RICH256-s2026"):
        subprocess.run(["tmux", "kill-session", "-t", sess], check=False)
    (S / "review" / "V3_NO_GO.json").write_text(json.dumps(
        {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "reason": reason,
         "action": "stopped the v3 queue and both run sessions; nothing deleted"}, indent=1))


def main() -> int:
    result = {"rule": __doc__.split("Runs detached.")[1].split("Writes")[0].strip(), "arms": {}}
    while True:
        data = {k: load(v) for k, v in RUNS.items()}
        if all(any(a["update"] >= 20_000 for a in aggs) for aggs, _ in data.values()):
            break
        time.sleep(60)
    for k, (aggs, _) in data.items():
        result["arms"][k] = evaluate(aggs)
    result["go"] = all(a["go"] for a in result["arms"].values())
    result["decided_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    OUT.write_text(json.dumps(result, indent=1))
    if not result["go"]:
        stop_campaign("pre-registered 20k go/no-go failed: " + json.dumps({k: v["checks"] for k, v in result["arms"].items()}))
        return 1
    # descriptive checkpoints (never stop the campaign)
    seen = set()
    while len(seen) < 3:
        time.sleep(300)
        for k, (aggs, diags) in {k: load(v) for k, v in RUNS.items()}.items():
            for dg in diags:
                u = dg["update"]
                tag = {40_000: "40k", 80_000: "80k", 156_956: "epoch1"}.get(u)
                if tag and (k, tag) not in seen:
                    result.setdefault("descriptive", {}).setdefault(tag, {})[k] = dg["cohort_macro_mrr20"]
                    seen.add((k, tag)) if k == "A0_ID256" else None
        result["descriptive_thresholds"] = DESCRIPTIVE
        OUT.write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
