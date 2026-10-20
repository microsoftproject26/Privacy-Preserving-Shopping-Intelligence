"""screen.py (owner: Opus) — warm-start adaptation screen (EXPLORATORY; 03_EXPERIMENT_PLAN §3).

One arm per process: CONTROL (unchanged features), C1_D3 (5 causal repeat channels, zero-init), or
C2 (the gate-2(a) removal subset as a permanent coherent-absence adapter).

Every arm:
- starts from the SAME pinned baseline weights (strict load) with a FRESH AdamW;
- trains on the SAME manifest of 5,000,000 unique TRAIN loss decisions, in the same order, in one
  pass: 19,531 updates of 256 plus a final exact update of 64;
- follows the same LR schedule (linear warmup to 1e-4 over 200 updates, cosine to 1e-5 at the final
  update) and the same seed.

The recipe is the campaign's: AdamW (0.9, 0.999), eps 1e-8, WD 1e-5 on matrices, grad clip 1.0,
dropout 0.2, CE, output recentering after each step, TF32 training, strict-FP32 cohort evaluation.

A screen result is warm-start adaptation evidence only, never a from-scratch feature-ablation result.
Writes only under runs/screen/<arm>/.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

import fo_env
import feature_view as fv
import fo_models as fm
import fo_eval as E
import precision
import train_sasrec as ts                     # snapshot copy: build_optimizer, model_inputs, set_lr, constants
from sasrec_objective import ce_sum

N_SCREEN, BATCH, WARMUP, LR_PEAK, LR_END = 5_000_000, 256, 200, 1e-4, 1e-5
MANIFEST_SEED, TRAIN_SEED = 20260927, 2026
ARMS = ("CONTROL", "C1_D3", "C2")


def n_updates(n: int = N_SCREEN, b: int = BATCH) -> int:
    return math.ceil(n / b)


def lr_at(u: int, U: int) -> float:
    """u = 1..U (the update being applied)."""
    if u <= WARMUP:
        return LR_PEAK * u / WARMUP
    t = (u - WARMUP) / max(1, U - WARMUP)
    return LR_END + 0.5 * (LR_PEAK - LR_END) * (1.0 + math.cos(math.pi * t))


def manifest(n_train: int, n: int = N_SCREEN) -> np.ndarray:
    """Deterministic unique sample of TRAIN loss-decision positions; its order IS the data order."""
    return np.random.default_rng(MANIFEST_SEED).choice(n_train, size=n, replace=False).astype(np.int64)


def build_arm(arm: str, reg: dict, sd: dict, device, c2_probe=None, **kw):
    if arm == "CONTROL":
        return fm.build(reg, sd, view="full", device=device, **kw)
    if arm == "C1_D3":
        return fm.build(reg, sd, view="d3", device=device, **kw)
    if arm == "C2":
        if c2_probe is None:
            raise ValueError("C2 requires the gate-2(a) removal probe")
        return fm.build(reg, sd, view="full", probe=c2_probe, device=device, **kw)
    raise ValueError(arm)


def train_arm(model, store, ds_train, positions: np.ndarray, device, log_f=None, microbatch: int = 256) -> dict:
    """One pass over `positions` in order. Returns exposure and optimisation records."""
    U = n_updates(len(positions))
    opt = ts.build_optimizer(model, fused=(device.type == "cuda"))
    model.train()
    seen = np.zeros(len(positions), dtype=bool)
    rec = {"updates": 0, "exposures": 0, "loss_first": None, "loss_last": None, "lr_first": None, "lr_last": None}
    t0 = time.time()
    for u in range(1, U + 1):
        sl = slice((u - 1) * BATCH, min(u * BATCH, len(positions)))
        pos = positions[sl]
        seen[sl] = True
        b = store.batch(ds_train, pos)
        if not bool(torch.as_tensor(b["loss_mask"]).all()):
            raise RuntimeError("a screen row lacks loss_mask")
        n_total = int(((torch.as_tensor(b["loss_mask"])) & (torch.as_tensor(b["target_class"]) >= 0)).sum())
        if n_total != len(pos):
            raise RuntimeError("one target per decision violated")
        opt.zero_grad(set_to_none=True)
        loss_total = 0.0
        for s in range(0, len(pos), microbatch):
            part = {k: (v[s:s + microbatch].to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                    for k, v in b.items()}
            logits = model(**ts.model_inputs(part, model))
            loss_sum, _ = ce_sum(logits, part["target_class"], part["loss_mask"], label_smoothing=0.0)
            (loss_sum / n_total).backward()
            loss_total += float(loss_sum.detach())
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), ts.GRAD_CLIP))
        loss = loss_total / n_total
        if not (math.isfinite(loss) and math.isfinite(gn)):
            raise FloatingPointError(f"non-finite loss/grad at update {u}")
        lr = lr_at(u, U)
        ts.set_lr(opt, lr)
        opt.step()
        model.recenter_output_()
        rec["updates"], rec["exposures"] = u, rec["exposures"] + len(pos)
        if u == 1:
            rec["loss_first"], rec["lr_first"] = loss, lr
        rec["loss_last"], rec["lr_last"] = loss, lr
        if log_f is not None and (u % 500 == 0 or u == U):
            log_f.write(json.dumps({"u": u, "loss": loss, "gn": gn, "lr": lr, "s": round(time.time() - t0, 1)}) + "\n")
            log_f.flush()
    rec["coverage_ok"] = bool(seen.all()) and rec["exposures"] == len(positions)
    rec["train_seconds"] = round(time.time() - t0, 1)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    a = ap.parse_args()
    out_dir = fo_env.study_path("runs", "screen", a.arm)
    if (out_dir / "result.json").exists():
        print("arm already complete (immutable)"); return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    reg_p = fo_env.study_path("config", "candidate_registry.json")
    registry = json.loads(reg_p.read_text())
    if a.arm not in registry["screen_arms"]:
        raise RuntimeError(f"arm {a.arm} not registered in candidate_registry.json")
    pin = json.loads(fo_env.study_path("config", "BASELINE_PIN.json").read_text())
    ch = pin["chosen"]
    if fo_env.sha256_file(ch["file"]) != ch["sha256"]:
        raise RuntimeError("pinned checkpoint changed")
    fo_env.verify_snapshot()
    c2_probe = None
    if a.arm == "C2":
        c2 = json.loads(fo_env.study_path("state", "C2_DECISION.json").read_text())
        if not c2.get("C2"):
            raise RuntimeError("no C2 removal subset passed gate 2(a)")
        cell = json.loads(fo_env.study_path("config", "P8_CALENDAR_CELL.json").read_text())
        c2_probe = fv.ProbeSpec(c2["C2"], calendar_cell=(cell["local_weekday_idx"], cell["daypart_idx"]) if c2["C2"].startswith("P8") else None)
    precision.configure_strict_fp32(TRAIN_SEED)
    device = torch.device(os.environ.get("FO_DEVICE", "cuda:0"))
    import sasrec_data
    store = sasrec_data.build_training_store("F4")
    ds_train = store.decisions("TRAIN", "loss")
    if len(ds_train) != fo_env.N_TRAIN:
        raise RuntimeError(f"TRAIN loss decisions {len(ds_train)} != {fo_env.N_TRAIN}")
    ds_vs = store.decisions("VALIDATION_SELECT", "eval")
    import pyarrow.parquet as pq
    ids = np.ascontiguousarray(pq.read_table(pin["cohort_parquet"], columns=["decision_id"])["decision_id"].to_numpy()).astype(np.int64)
    cohort = ds_vs.take(np.searchsorted(ds_vs.decision_id, ids))
    if not np.array_equal(np.asarray(cohort.decision_id), ids):
        raise RuntimeError("cohort mismatch")
    positions = manifest(len(ds_train))
    man_sha = __import__("hashlib").sha256(positions.tobytes()).hexdigest()
    if man_sha != registry["manifest_sha256"]:
        raise RuntimeError(f"manifest sha {man_sha} != registered {registry['manifest_sha256']}")
    fo_env.assert_isolated()
    if float(ch["registry_entry"]["label_smoothing"]) != 0.0:     # Review C fix: the CE below is unsmoothed
        raise RuntimeError(f"registry label_smoothing {ch['registry_entry']['label_smoothing']} != 0.0 used by the screen")
    sd, _ = fm.load_weights_file(ch["file"])
    model = build_arm(a.arm, ch["registry_entry"], sd, device, c2_probe=c2_probe)
    torch.manual_seed(TRAIN_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(TRAIN_SEED)
    precision.training_precision(True)                  # campaign policy: TF32 training (parity-gated)
    with open(out_dir / "train_log.jsonl", "w") as lf:
        rec = train_arm(model, store, ds_train, positions, device, log_f=lf)
    ranks, m, secs = E.evaluate(model, store, cohort, device)
    ranks_sha = E.save_ranks(out_dir / "cohort_ranks.npz", ranks, cohort)
    torch.save({"state_dict": model.inner.state_dict(), "arm": a.arm, "view": model.view}, out_dir / "final_weights.pt")
    res = {"arm": a.arm, "label": "EXPLORATORY warm-start adaptation (not a from-scratch ablation)",
           "baseline_pin": {k: ch[k] for k in ("seed", "epoch", "sha256")}, "manifest_sha256": man_sha,
           "n_decisions": int(len(positions)), "train": rec, "cohort_metrics": m, "eval_seconds": secs,
           "cohort_ranks_sha256": ranks_sha, "final_weights_sha256": fo_env.sha256_file(out_dir / "final_weights.pt"),
           "precision": {"training": "tf32", "eval": "strict_fp32"}, "schedule": {"warmup": WARMUP, "peak": LR_PEAK, "end": LR_END,
                                                                                 "updates": n_updates()},
           "parameter_mapping": fm.parameter_mapping(sd, model.view) if a.arm != "C2" else {"probe_adapter": c2_probe.name},
           "c2_probe": c2_probe.name if c2_probe else None}
    E.write_once(out_dir / "result.json", res)
    print(json.dumps({"arm": a.arm, "macro": m["macro_mrr@20"], "train_s": rec["train_seconds"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
