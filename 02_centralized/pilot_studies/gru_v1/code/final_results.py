"""Assemble FINAL_FULL_GRU_RESULTS.json from the run artifacts (owner: Opus). Read-only on inputs.

python final_results.py [--out $W/FINAL_FULL_GRU_RESULTS.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

W = Path("/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/gru_v1")
RUNS = {"F0_item_only_h256": W / "experiments/full/F0_item_only_h256",
        "F4_rich_side_information_h256": W / "experiments/full/F4_rich_side_information_h256"}
MODELS = ("F0_item_only_h256", "F4_rich_side_information_h256", "POP_full", "TRANS_full")


def jload(p: Path):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_missing": str(p), "_error": repr(exc)}


def sha(p: Path):
    if not Path(p).exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def headline(res: dict) -> dict:
    """Compact metric block for one model on one split."""
    if not isinstance(res, dict) or "end_to_end" not in res:
        return {"_missing": True}
    out = {}
    for den in ("end_to_end", "closed"):
        b = res[den]
        out[den] = {"n": b["n"], "user_count": b["macro"].get("user_count"),
                    "micro": {k: v for k, v in b["micro"].items()},
                    "macro": {k: v for k, v in b["macro"].items() if k != "user_count"}}
    out["cohorts"] = {c: {"n": res["cohorts"][c]["end_to_end"]["n"],
                          "macro_mrr@20": res["cohorts"][c]["end_to_end"]["macro"]["mrr@20"],
                          "micro_recall@20": res["cohorts"][c]["end_to_end"]["micro"]["recall@20"]}
                      for c in ("warm", "unseen")}
    if "descriptive_at_k" in res:
        out["descriptive_at_k"] = {
            k: {den: {"micro": v[den]["micro"], "macro": {kk: vv for kk, vv in v[den]["macro"].items() if kk != "user_count"}}
                for den in ("end_to_end", "closed")}
            for k, v in res["descriptive_at_k"].items()}
    if "slices" in res:
        out["slices"] = res["slices"]
    out["eval_seconds"] = res.get("_elapsed_s")
    return out


def baselines_vs_block() -> dict:
    b = jload(W / "tests_full/C/baselines_full_vs.json")
    out = {}
    for m in ("POP_full", "TRANS_full"):
        res = b.get(m) or b.get(m.split("_")[0]) or {}
        out[m] = headline(res.get("result", res) if isinstance(res, dict) else {})
    return out


def run_block(arm: str, d: Path) -> dict:
    summary = jload(d / "summary.json")
    epochs, evals, diags, aggs = [], [], [], []
    with open(d / "metrics.jsonl", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            t = r.get("type")
            if t == "epoch_end":
                epochs.append(r["selection_record"])
            elif t == "eval":
                e = r["eval_full"]["end_to_end"]
                evals.append({"epoch": r["epoch"] + 1, "update": r["update"],
                              "micro_recall@20": e["micro"]["recall@20"], "micro_mrr@20": e["micro"]["mrr@20"],
                              "micro_ndcg@20": e["micro"]["ndcg@20"], "macro_recall@20": e["macro"]["recall@20"],
                              "macro_mrr@20": e["macro"]["mrr@20"], "macro_ndcg@20": e["macro"]["ndcg@20"],
                              "n": e["n"], "user_count": e["macro"]["user_count"],
                              "eval_seconds": r["eval_full"].get("_elapsed_s")})
            elif t == "diag_eval":
                diags.append({"update": r["update"], "cohort_macro_mrr@20": r["cohort_client_macro_mrr@20"]})
            elif t == "update_agg":
                aggs.append(r)
    losses = {}
    for a in aggs:  # mean train loss per epoch from the 100-update aggregates
        losses.setdefault(a["epoch"], []).append(a["mean_loss"])
    return {
        "summary": summary,
        "config_yaml": (d / "config.yaml").read_text(encoding="utf-8"),
        "environment": jload(d / "environment.json"), "hardware": jload(d / "hardware.json"),
        "git_state": jload(d / "git_state.json"), "start": jload(d / "start.json"),
        "epoch_selection": epochs, "epoch_evals_full_vs": evals,
        "epoch_mean_train_loss": {str(k + 1): sum(v) / len(v) for k, v in sorted(losses.items())},
        "diag_cohort": diags,
        "throughput_dps_median": sorted(a["decisions_per_s"] for a in aggs)[len(aggs) // 2] if aggs else None,
        "peak_gpu_mem_reserved_gib": max((a.get("gpu_mem_reserved_bytes") or 0) for a in aggs) / 2**30 if aggs else None,
        "checkpoints": {n: {"bytes": (d / "checkpoints" / f"{n}.pt").stat().st_size if (d / "checkpoints" / f"{n}.pt").exists() else None,
                            "sha256": sha(d / "checkpoints" / f"{n}.pt")} for n in ("best", "latest", "previous")},
        "supervisor_log_tail": (d / "logs" / "supervisor.log").read_text(encoding="utf-8").splitlines()[-6:],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(W / "FINAL_FULL_GRU_RESULTS.json"))
    args = ap.parse_args()
    vc = jload(W / "final_eval" / "VC_RESULTS.json")
    te = jload(W / "final_eval" / "TEST_RESULTS.json")
    st = os.statvfs("/home/jovyan")
    out = {
        "dataset": {"path": "/home/jovyan/shared_data/rees46_nextitem_v1",
                    "version_hash": "860e5a1c3a81b8e4bbfeff4f245fa37638a43e4f260df179585964d2d4da8dbd",
                    "integrity_before_test": jload(W / "final_eval" / "verify_frozen_before_test.json"),
                    "train_loss_decisions": 40_180_608, "catalogue_size": 158_486,
                    "vs_eval_decisions": 8_826_488, "vs_eval_users": 790_459},
        "protocol": {"path": str(W / "FULL_RUN_PROTOCOL.md"), "sha256": sha(W / "FULL_RUN_PROTOCOL.md")},
        "fable_review": str(W / "FABLE_FULL_RUN_REVIEW.md"),
        "gates": {"data_path_rerun": jload(W / "tests_full/opus/gates_A_rerun.json").get("overall_pass"),
                  "integration": jload(W / "tests_full/opus/integration.json").get("overall_pass"),
                  "benchmark": {k: v for k, v in jload(W / "bench/BENCH_SUMMARY.json").items() if k != "sampler"}},
        "baselines_full_vs": baselines_vs_block(),
        "runs": {arm: run_block(arm, d) for arm, d in RUNS.items()},
        "lock": jload(W / "locks" / "MODEL_LOCK.json"),
        "validation_confirm": {m: headline(vc.get(m, {})) for m in MODELS} | {"oov": vc.get("oov")},
        "test": {m: headline(te.get(m, {})) for m in MODELS} | {"oov": te.get("oov"), "bootstraps": te.get("bootstraps")},
        "locks_files": {n: jload(W / "locks" / n) for n in ("VC_CLAIM.json", "VC_DONE.json", "TEST_CLAIM.json", "TEST_DONE.json")},
        "disk_free_gib_final": st.f_bavail * st.f_frsize / 2**30,
        "recall_at_k_cohort_midtraining": jload(W / "reports_full/recall_at_k_cohort.json"),
    }
    Path(args.out).write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
