"""Pilot orchestrator -- Task T-M.

Execution and measurement ONLY. Runs the two frozen pilot arms
(F0_item_only, F4_rich_side_information), exactly 2 passes x 1,000,000
decisions each, by calling `train.run_training` UNMODIFIED (imported, not
copied) -- every frozen hyperparameter (optimizer, LR schedule, grad clip,
determinism flags, data order, exact-CE objective, checkpointing, per-pass
VS-cohort evaluation) lives in `train.py` and is not touched here.

This script adds only what T-M's brief asks for on top of the T-K engine:
  1. A pre-flight `nvidia-smi` GPU-0-exclusivity check (abort if any compute
     process is already present on GPU 0).
  2. Peak-VRAM (`max_memory_reserved`) and peak host-RSS sampling per arm.
  3. Wall-clock / throughput measurement per arm.
  4. P3: loads each arm's BEST checkpoint and runs `run_vs_evaluation` on it
     TWICE, diffing the two result dicts for exact (NaN-safe) equality.
  5. A per-user (grouped by `eval_user_key`) end-to-end MRR@20 array for the
     BEST checkpoint of each arm, computed by reusing (not modifying)
     `evaluator._ranks_chunked` / `evaluator._credits_from_ranks` -- this is
     what the F4-vs-F0 paired bootstrap in `pilot_report.py` consumes. Its
     aggregate (mean of per-user means) is cross-checked against the
     official `evaluate()` macro MRR@20 as a consistency check.

Run on Lobot only, GPU 0 pinned (`CUDA_VISIBLE_DEVICES=0`). Never `uv run`.
Launches ONLY the pilot (T-M is explicitly authorised); does not launch full
training.
"""
from __future__ import annotations

import json
import math
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

CODE_DIR = Path(__file__).resolve().parent
WORKSPACE = CODE_DIR.parent
ARTIFACTS = WORKSPACE / "artifacts"
EXPERIMENTS_DIR = WORKSPACE / "experiments" / "gru"

for _p in (str(CODE_DIR), str(CODE_DIR / "vendor")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import train as train_mod  # noqa: E402
from evaluator import _ranks_chunked, _credits_from_ranks  # noqa: E402  (reused, not modified)

ARMS = ["F0_item_only", "F4_rich_side_information"]
PASSES = 2
RUN_ID_OF = {
    "F0_item_only": "T-M_F0_item_only",
    "F4_rich_side_information": "T-M_F4_rich_side_information",
}
EXPECTED_EXPOSURES = PASSES * train_mod.TRAIN_ROWS  # 2,000,000


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] PILOT: {msg}", flush=True)


# ---------------------------------------------------------------------------
# Pre-flight GPU-exclusivity check
# ---------------------------------------------------------------------------

def preflight_nvidia_smi(gpu_index: int = 0) -> None:
    full = subprocess.run(
        ["nvidia-smi", "-i", str(gpu_index)],
        capture_output=True, text=True, timeout=30,
    )
    log("nvidia-smi -i %d:\n%s" % (gpu_index, full.stdout))
    apps = subprocess.run(
        ["nvidia-smi", "-i", str(gpu_index),
         "--query-compute-apps=pid,process_name,used_memory",
         "--format=csv,noheader"],
        capture_output=True, text=True, timeout=30,
    )
    rows = [ln.strip() for ln in apps.stdout.splitlines() if ln.strip()]
    if rows:
        raise RuntimeError(
            f"ABORT: GPU {gpu_index} has {len(rows)} compute process(es) already present "
            f"-- refusing to start the pilot on a non-idle GPU: {rows}"
        )
    log(f"pre-flight OK: GPU {gpu_index} idle, 0 compute processes")


# ---------------------------------------------------------------------------
# Resource sampling
# ---------------------------------------------------------------------------

def peak_rss_mb() -> float:
    """Peak (so far, monotonic non-decreasing) RSS of this process, MB.
    Linux `ru_maxrss` is in KiB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# ---------------------------------------------------------------------------
# Post-arm analysis: P3 double-eval determinism check + per-user MRR array
# ---------------------------------------------------------------------------

def load_best_model(group: str, run_dir: Path, poc: np.ndarray, device: torch.device):
    ckpt = train_mod.load_checkpoint(run_dir, "best")
    model = train_mod.build_model_for_group(group, poc, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def _nan_safe_equal(a, b) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False
        return all(_nan_safe_equal(a[k], b[k]) for k in a)
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return a == b
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_nan_safe_equal(x, y) for x, y in zip(a, b))
    return a == b


def compute_end_to_end_ranks(model, device: torch.device, chunk_size: int = 512,
                              class_chunk: int = 8192):
    """Replays evaluator.evaluate()'s exact rank-computation loop (reusing its
    private `_ranks_chunked` unmodified) but returns the raw per-row `ranks`
    tensor instead of only the aggregate, so per-user credit vectors can be
    built for the F4-vs-F0 bootstrap. Population selection (`eval_mask &
    target_class>=0`) is copy-identical to evaluator.py:186-192.
    """
    from window_cache import CacheReader

    reader = CacheReader(train_mod.CACHE_VS)
    n = reader.header["row_count"]
    eval_user_key = torch.from_numpy(train_mod.eval_user_key_array())
    all_rows = reader.read(np.arange(n))
    target_class = all_rows["target_class"].to(torch.long)
    eval_mask = all_rows["eval_mask"].to(torch.bool)

    ranks = torch.full((n,), -1, dtype=torch.long)
    with torch.no_grad():
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            idx = np.arange(start, end)
            batch = reader.read(idx)
            batch_gpu = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            kw = train_mod.build_named_kwargs(batch_gpu, model)
            logits = model(**kw).detach().to(torch.float32).cpu()

            row_idx = torch.arange(start, end)
            batch_target = target_class[row_idx]
            batch_eval = eval_mask[row_idx]
            need = batch_eval & (batch_target >= 0)
            if bool(need.any()):
                need_rows = row_idx[need]
                sub_scores = logits[need]
                sub_target = batch_target[need]
                ranks[need_rows] = _ranks_chunked(sub_scores, sub_target, class_chunk)
    return ranks, target_class, eval_mask, eval_user_key


def per_user_end_to_end_mrr(ranks, eval_mask, eval_user_key, k: int = 20):
    """Mirrors evaluator._population_block's per-user grouping for the
    end-to-end population (pop_mask = eval_mask), but returns the per-user
    array instead of collapsing it to a single macro mean."""
    credits = _credits_from_ranks(ranks, k)
    mrr = credits[f"mrr@{k}"]
    idx = eval_mask.nonzero(as_tuple=True)[0]
    keys = eval_user_key[idx]
    unique_keys, inverse = torch.unique(keys, return_inverse=True)
    n_users = int(unique_keys.numel())
    sub = mrr[idx]
    sums = torch.zeros(n_users, dtype=torch.float32)
    counts = torch.zeros(n_users, dtype=torch.float32)
    sums.index_add_(0, inverse, sub)
    counts.index_add_(0, inverse, torch.ones_like(sub))
    per_user_mean = sums / counts
    return unique_keys, per_user_mean


def post_arm_analysis(group: str, run_dir: Path, poc: np.ndarray, device: torch.device) -> dict:
    log(f"{group}: post-arm analysis -- P3 double-eval + per-user MRR (best checkpoint)")
    model, ckpt = load_best_model(group, run_dir, poc, device)

    eval_a = train_mod.run_vs_evaluation(model, device)
    eval_b = train_mod.run_vs_evaluation(model, device)
    identical = _nan_safe_equal(eval_a, eval_b)
    log(f"{group}: P3 double-eval identical={identical} "
        f"(macro_mrr@20 A={eval_a['end_to_end']['macro']['mrr@20']!r} "
        f"B={eval_b['end_to_end']['macro']['mrr@20']!r})")

    ranks, target_class, eval_mask, eval_user_key = compute_end_to_end_ranks(model, device)
    unique_keys, per_user_mean = per_user_end_to_end_mrr(ranks, eval_mask, eval_user_key)
    reconstructed_macro = float(per_user_mean.mean().item())
    official_macro = float(eval_a["end_to_end"]["macro"]["mrr@20"])
    consistency_ok = abs(reconstructed_macro - official_macro) < 1e-6
    log(f"{group}: per-user reconstruction macro={reconstructed_macro:.8f} "
        f"official={official_macro:.8f} consistency_ok={consistency_ok} "
        f"n_users={unique_keys.numel()}")

    p3 = {
        "checkpoint_pass_index": ckpt["pass_index"],
        "checkpoint_update_number": ckpt["update_number"],
        "checkpoint_best_metric": ckpt["best_metric"],
        "eval_a": eval_a,
        "eval_b": eval_b,
        "bit_identical": identical,
    }
    (run_dir / "p3_double_eval.json").write_text(
        json.dumps(p3, indent=2, default=str), encoding="utf-8")

    per_user = {
        "eval_user_key": unique_keys.tolist(),
        "mrr@20_end_to_end": per_user_mean.tolist(),
        "reconstructed_macro_mrr@20": reconstructed_macro,
        "official_macro_mrr@20": official_macro,
        "consistency_ok": consistency_ok,
        "checkpoint": "best",
    }
    (run_dir / "per_user_mrr_end_to_end_best.json").write_text(
        json.dumps(per_user), encoding="utf-8")

    del model
    torch.cuda.empty_cache()
    return {"p3": p3, "per_user": per_user}


# ---------------------------------------------------------------------------
# Per-arm run
# ---------------------------------------------------------------------------

def run_arm(group: str, device: torch.device) -> dict:
    run_id = RUN_ID_OF[group]
    run_dir = EXPERIMENTS_DIR / run_id
    n_updates_per_pass = -(-train_mod.TRAIN_ROWS // train_mod.MICROBATCH)
    target_update = PASSES * n_updates_per_pass
    log(f"=== arm {group} run_id={run_id} target_update={target_update} "
        f"(n_updates_per_pass={n_updates_per_pass}, passes={PASSES}) ===")

    torch.cuda.reset_peak_memory_stats(device)
    rss_before_mb = peak_rss_mb()
    t0 = time.perf_counter()

    summary = train_mod.run_training(
        group=group, run_dir=run_dir, target_update=target_update, device=device,
        resume=False, seed=train_mod.SEED, checkpoint_every=500, run_eval=True,
    )

    t1 = time.perf_counter()
    wall_s = t1 - t0
    peak_vram_reserved_bytes = int(torch.cuda.max_memory_reserved(device))

    poc = train_mod.load_product_idx_of_class()
    analysis = post_arm_analysis(group, run_dir, poc, device)

    t2 = time.perf_counter()
    peak_vram_reserved_bytes = max(peak_vram_reserved_bytes,
                                    int(torch.cuda.max_memory_reserved(device)))
    rss_after_mb = peak_rss_mb()

    # zero non-finite losses, over every logged training-update record
    n_nonfinite = 0
    n_updates_logged = 0
    losses = []
    metrics_path = run_dir / "metrics.jsonl"
    with metrics_path.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if "loss" in rec:
                n_updates_logged += 1
                losses.append(rec["loss"])
                if not math.isfinite(rec["loss"]):
                    n_nonfinite += 1

    resource_usage = {
        "group": group,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "target_update": target_update,
        "n_updates_per_pass": n_updates_per_pass,
        "train_wall_clock_s": wall_s,
        "post_arm_analysis_wall_clock_s": t2 - t1,
        "total_wall_clock_s": t2 - t0,
        "throughput_decisions_per_s": summary["consumed_decisions"] / wall_s,
        "peak_memory_reserved_bytes": peak_vram_reserved_bytes,
        "peak_memory_reserved_gib": peak_vram_reserved_bytes / (1024 ** 3),
        "peak_host_rss_mb_before": rss_before_mb,
        "peak_host_rss_mb_after": rss_after_mb,
        "n_updates_logged": n_updates_logged,
        "n_nonfinite_losses": n_nonfinite,
        "consumed_decisions": summary["consumed_decisions"],
        "expected_exposures": EXPECTED_EXPOSURES,
        "exposures_exact": summary["consumed_decisions"] == EXPECTED_EXPOSURES,
        "final_pass_index": summary["pass_index"],
        "final_update_number": summary["update_number"],
        "best_metric": summary["best_metric"],
    }
    (run_dir / "resource_usage.json").write_text(
        json.dumps(resource_usage, indent=2, default=str), encoding="utf-8")

    log(f"=== arm {group} DONE: wall={wall_s:.1f}s throughput="
        f"{resource_usage['throughput_decisions_per_s']:.1f} dec/s "
        f"peak_vram={resource_usage['peak_memory_reserved_gib']:.4f} GiB "
        f"consumed={summary['consumed_decisions']} nonfinite={n_nonfinite} ===")

    return {"summary": summary, "resource_usage": resource_usage, "analysis": analysis}


def main() -> int:
    log(f"host RSS at process start: {peak_rss_mb():.1f} MB")
    preflight_nvidia_smi(0)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available on this worker")
    device = torch.device("cuda:0")
    torch.cuda.init()  # establish the CUDA context; reset_peak_memory_stats()
                        # raises "Invalid device argument" if called first
    train_mod.set_determinism(train_mod.SEED)

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for group in ARMS:
        results[group] = run_arm(group, device)

    out = {
        arm: {
            "summary": results[arm]["summary"],
            "resource_usage": results[arm]["resource_usage"],
        }
        for arm in ARMS
    }
    (ARTIFACTS / "PILOT_RUN_META.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")

    log("PILOT RUN COMPLETE for both arms.")
    log(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
