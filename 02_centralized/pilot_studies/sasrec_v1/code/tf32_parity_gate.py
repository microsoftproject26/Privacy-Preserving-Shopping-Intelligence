"""TF32 parity gate (owner: Opus). Registry v2 precision.parity_gate, Fable Gate P2 test 1.

Three isolated real-data benchmark states of A0_ID256, 3000 updates each, SAME batches
(benchmark rng 12345), real LR schedule from u=1 (warmup 2000, then peak):
  FP32_A: strict FP32 training, seed 2026      (GPU0)
  TF32_A: TF32 training,        seed 2026      (GPU1, concurrently with FP32_A)
  FP32_B: strict FP32 training, seed 2027      (GPU0, after FP32_A)
PASS iff over updates 2001-3000:
  |mean loss TF32_A - mean loss FP32_A| <= |mean loss FP32_A - mean loss FP32_B| + 0.005 * mean loss FP32_A,
  no NaN/Inf anywhere, and median grad norms of TF32_A and FP32_A within 5%.
plus (pre-registered, review/TF32_PARITY_PREREG.md) the end-of-gate strict-FP32 cohort metric
band and a descriptive parameter-trajectory difference. Writes bench/TF32_PARITY_EVIDENCE.json.
Fable (owner delegate) rules ACCEPTABLE / NOT ACCEPTABLE from it; that ruling is written to
bench/TF32_PARITY.json, the only file train_sasrec.precision_plan reads. Not acceptable ->
registered fallback (strict FP32, 5 epochs). Benchmark states are discarded; they never
initialise a scored run. Requires the Gate C firewall.
"""
from __future__ import annotations

import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import paths

PY = os.path.expanduser("~/venvs/ppsi-gpu/bin/python")
TRAINER = str(paths.CODE / "train_sasrec.py")
OUT = paths.S / "bench"
N_UPDATES = 3000
WINDOW = (2001, 3000)


def launch(tag: str, seed: int, mode: str, gpu: int) -> tuple[subprocess.Popen, Path]:
    log = OUT / f"parity_{tag}.losses.json"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="3", MKL_NUM_THREADS="3",
               OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    fh = open(OUT / f"parity_{tag}.log", "w")
    p = subprocess.Popen([PY, TRAINER, "--run-id", "A0_ID256", "--seed", str(seed), "--benchmark-updates",
                          str(N_UPDATES), "--precision-mode", mode, "--bench-parity", "--bench-loss-log", str(log)],
                         env=env, stdout=fh, stderr=subprocess.STDOUT)
    return p, log


def window_stats(log: Path) -> dict:
    blob = json.loads(log.read_text())
    ups = blob["updates"]
    all_finite = all(math.isfinite(u["loss"]) and math.isfinite(u["grad_norm"]) for u in ups)
    win = [u for u in ups if WINDOW[0] <= u["update"] <= WINDOW[1]]
    if len(win) != WINDOW[1] - WINDOW[0] + 1:
        raise RuntimeError(f"{log}: expected {WINDOW[1] - WINDOW[0] + 1} window updates, got {len(win)}")
    return {"n_updates": len(ups), "all_finite": all_finite,
            "mean_loss_window": statistics.fmean(u["loss"] for u in win),
            "median_grad_norm_window": statistics.median(u["grad_norm"] for u in win),
            "first_loss": ups[0]["loss"], "last_loss": ups[-1]["loss"], "cohort": blob.get("cohort"),
            "param_sample_file": blob.get("param_sample_file")}


def param_delta(a_file: str, t_file: str) -> dict:
    """Descriptive: ||(theta_T - theta0) - (theta_A - theta0)|| / ||theta_A - theta0|| (same init)."""
    import numpy as np
    A, T = np.load(a_file), np.load(t_file)
    out, num_all, den_all = {}, 0.0, 0.0
    for key in A.files:
        if not key.startswith("final::"):
            continue
        name = key[len("final::"):]
        a0, t0 = A[f"init::{name}"], T[f"init::{name}"]
        if not np.array_equal(a0, t0):
            out[name] = {"error": "different init samples"}
            continue
        da, dt = A[key] - a0, T[key] - t0
        num, den = float(np.linalg.norm(dt - da)), float(np.linalg.norm(da))
        out[name] = num / den if den > 0 else None
        num_all += num ** 2; den_all += den ** 2
    return {"per_tensor": out, "overall": (num_all ** 0.5) / (den_all ** 0.5) if den_all > 0 else None}


def main() -> int:
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    pa, la = launch("FP32_A", 2026, "fp32", 0)
    pt, lt = launch("TF32_A", 2026, "tf32", 1)
    rc = {"FP32_A": pa.wait(), "TF32_A": pt.wait()}
    pb, lb = launch("FP32_B", 2027, "fp32", 0)
    rc["FP32_B"] = pb.wait()
    if any(rc.values()):
        res = {"rule_pass": False, "reason": f"benchmark exit codes {rc}", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    else:
        A, T, B = window_stats(la), window_stats(lt), window_stats(lb)
        d_tf32 = abs(T["mean_loss_window"] - A["mean_loss_window"])
        d_seed = abs(A["mean_loss_window"] - B["mean_loss_window"])
        tol = d_seed + 0.005 * A["mean_loss_window"]
        gratio = T["median_grad_norm_window"] / A["median_grad_norm_window"]
        c_tf32 = abs(T["cohort"]["macro_mrr20"] - A["cohort"]["macro_mrr20"])
        c_seed = abs(A["cohort"]["macro_mrr20"] - B["cohort"]["macro_mrr20"])
        checks = {"loss_within_seed_band": d_tf32 <= tol, "all_finite": A["all_finite"] and T["all_finite"] and B["all_finite"],
                  "grad_norm_within_5pct": abs(gratio - 1.0) <= 0.05,
                  "cohort_within_seed_band": c_tf32 <= c_seed + 0.003}
        res = {"rule_pass": all(checks.values()), "checks": checks, "delta_tf32": d_tf32, "delta_seeds": d_seed,
               "cohort_delta_tf32": c_tf32, "cohort_delta_seeds": c_seed,
               "param_trajectory_rel_diff_descriptive": param_delta(A["param_sample_file"], T["param_sample_file"]),
               "tolerance": tol, "grad_median_ratio": gratio, "window": list(WINDOW),
               "FP32_A": A, "TF32_A": T, "FP32_B": B, "rule": __doc__.split("PASS iff")[1].split("Writes")[0].strip(),
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seconds": time.time() - t0}
    res["prereg"] = str(paths.S / "review" / "TF32_PARITY_PREREG.md")
    res["decision"] = "pending Fable review; bench/TF32_PARITY.json is written from Fable's ruling"
    tmp = OUT / "TF32_PARITY_EVIDENCE.json.tmp"
    tmp.write_text(json.dumps(res, indent=2))
    os.replace(tmp, OUT / "TF32_PARITY_EVIDENCE.json")
    print(json.dumps({k: v for k, v in res.items() if k not in ("FP32_A", "TF32_A", "FP32_B")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
