"""pin_baseline.py (owner: Opus) — implements config/BASELINE_RULE.md plus Fable decision items 2 and 7.

Reads the campaign's run records read-only and writes, once and never overwritten:
- config/BASELINE_PIN.json: the chosen checkpoint (VS-only rule), its sha256, the registry entry, the
  cohort parquet sha, the snapshot code hashes, and the COMPLETE definition used for cool5 runs.
- config/NOISE_REFERENCE.json: sigma_hat = max over matched epochs of |A1 s2026 - A1 s2027| full-VS
  macro MRR@20 (e3, e4, and e5 when both cool5 runs are COMPLETE). Fallback 0.002 if s2027 is unavailable.

Refuses to run before the campaign's state/FINAL_STAGE_DONE exists (no half-written checkpoint is
ever considered). Never reads retro_eval/ (VC/TEST).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import fo_env

CAMPAIGN = Path(os.environ.get("FO_CAMPAIGN_ROOT",
                               "/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/sasrec_v1"))
RUNS = CAMPAIGN / "runs"
SEEDS = (2026, 2027)
RUN_ID = "A1_RICH256"


def _summary(d: Path):
    p = d / "summary.json"
    if not p.exists():
        return None
    s = json.loads(p.read_text(encoding="utf-8"))
    return s if s.get("final_status") == "COMPLETE" else None


def _vs(summary: dict, epoch: int):
    for r in summary.get("epoch_records", []):
        if int(r.get("epoch", -1)) == epoch and r.get("full_vs"):
            return r["full_vs"]
    return None


def candidates() -> list[dict]:
    out = []
    for seed in SEEDS:
        base = RUNS / f"{RUN_ID}_s{seed}"
        s = _summary(base)
        if s is None:
            continue
        for ep in (3, 4):
            vs = _vs(s, ep)
            if vs is None:
                continue
            f = base / "checkpoints" / f"epoch{ep}_weights.pt"
            if not f.exists() and ep == 3 and s.get("best_epoch") == 3:
                f = Path(s["checkpoints"]["best_weights"]["path"])
            if f.exists():
                out.append({"seed": seed, "epoch": ep, "file": str(f), "macro_mrr20": float(vs["macro_mrr20"]),
                            "micro_recall20": float(vs["micro_recall20"]), "run_dir": str(base),
                            "registry_entry": s["config"]["registry_entry"], "registry_sha256": s["registry_sha256"]})
        cool = RUNS / f"{RUN_ID}_s{seed}_cool5"
        c = _summary(cool)
        f5 = cool / "checkpoints" / "epoch5_weights.pt"
        vs5 = _vs(c, 5) if c else None
        if c is not None and vs5 is not None and f5.exists():   # COMPLETE := summary COMPLETE + e5 record + weights file
            out.append({"seed": seed, "epoch": 5, "file": str(f5), "macro_mrr20": float(vs5["macro_mrr20"]),
                        "micro_recall20": float(vs5["micro_recall20"]), "run_dir": str(cool),
                        "registry_entry": c["config"]["registry_entry"], "registry_sha256": c["registry_sha256"]})
    return out


def choose(cands: list[dict]) -> dict:
    # highest VS macro MRR@20; exact tie -> earlier epoch, then seed 2026
    return sorted(cands, key=lambda r: (-r["macro_mrr20"], r["epoch"], r["seed"]))[0]


def noise_reference(cands: list[dict]) -> dict:
    by = {(r["seed"], r["epoch"]): r["macro_mrr20"] for r in cands}
    diffs = {e: abs(by[(2026, e)] - by[(2027, e)]) for e in (3, 4, 5) if (2026, e) in by and (2027, e) in by}
    if diffs:
        return {"sigma_hat": max(diffs.values()), "basis": "max |A1 s2026 - A1 s2027| full-VS macro MRR@20 over matched epochs",
                "per_epoch_abs_diff": {str(k): v for k, v in diffs.items()}, "fallback_used": False}
    return {"sigma_hat": 0.002, "basis": "fallback (s2027 unavailable): within-arm e3->e4 swings 0.0017/0.0009",
            "per_epoch_abs_diff": {}, "fallback_used": True}


def main() -> int:
    if not (CAMPAIGN / "state" / "FINAL_STAGE_DONE").exists() and not os.environ.get("FO_ALLOW_BEFORE_FINAL"):
        print("campaign final stage not done; refusing to pin (QUEUED_RESOURCE)")
        return 3
    pin_p, noise_p = fo_env.study_path("config", "BASELINE_PIN.json"), fo_env.study_path("config", "NOISE_REFERENCE.json")
    if pin_p.exists():
        print("BASELINE_PIN.json exists (immutable)")
        return 0
    cands = candidates()
    if not cands:
        print("no completed A1 candidate")
        return 4
    best = choose(cands)
    man = fo_env.verify_snapshot()
    pin = {"rule": "config/BASELINE_RULE.md", "pinned_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "chosen": dict(best, sha256=fo_env.sha256_file(best["file"])),
           "candidates": [dict(r) for r in cands],
           "cohort_parquet": str(fo_env.paths.COHORT_PARQUET), "cohort_sha256": fo_env.sha256_file(fo_env.paths.COHORT_PARQUET),
           "snapshot_manifest_sha256": fo_env.sha256_file(fo_env.SNAP / "SNAPSHOT_MANIFEST.json"),
           "evaluator_and_model_sha256": {k: man["files"][k] for k in ("sasrec_model.py", "sasrec_data.py",
                                                                         "reuse/eval_full.py", "reuse/evaluator.py",
                                                                         "reuse/full_store.py")},
           "complete_definition": "summary.json final_status COMPLETE + full_vs record for the epoch + weights file present; sha256 hashed at pin time",
           "vc_test_read": False}
    noise = dict(noise_reference(cands), written_utc=pin["pinned_utc"],
                 rule="Fable scope decision item 2; written before any candidate training")
    for p, obj in ((pin_p, pin), (noise_p, noise)):
        fo_env.write_json_once(p, obj)                # serialise first, then O_EXCL: never overwritten
    print(json.dumps({"chosen": {k: pin["chosen"][k] for k in ("seed", "epoch", "macro_mrr20", "sha256")},
                      "sigma_hat": noise["sigma_hat"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
