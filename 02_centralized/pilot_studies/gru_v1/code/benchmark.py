"""Task T-J -- hardware benchmark on one A16 GPU. MEASUREMENT ONLY.

Produces artifacts/BENCHMARK.json with every raw measurement plus the
mechanical E1 (VRAM) / E2 (throughput) verdicts, applying the EXACT
pre-registered rules from the task brief. This script does not tune
anything, does not change any threshold, and does not launch pilot
training -- it only runs timed forward/backward/optimizer steps against
`model.py` / `objective.py` / `window_cache.py` exactly as they exist.

Run on Lobot only, pinned to one GPU (CUDA_VISIBLE_DEVICES=0 is set by the
launcher; this script also defends by setting it itself if unset).
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

WORKSPACE = Path("/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/gru_v1")
CODE_DIR = WORKSPACE / "code"
VENDOR_DIR = CODE_DIR / "vendor"
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(VENDOR_DIR))

DATASET_ROOT = Path("/home/jovyan/shared_data/rees46_nextitem_v1")
CACHE_TRAIN = WORKSPACE / "cache" / "train"
CACHE_VS = WORKSPACE / "cache" / "vs"
ARTIFACTS = WORKSPACE / "artifacts"
LOG_PATH = ARTIFACTS / "benchmark_run.log"

VOCAB_SIZES = {
    "item_tokens": 158489,
    "category_tokens": 663,
    "brand_tokens": 3956,
    "main_category_tokens": 16,
    "daypart_tokens": 8,
    "weekday_tokens": 10,
}
EXPECTED_K = 158486
FULL_EPOCH_DECISIONS = 40_180_608

GROUP_NAMES = ["F0_item_only", "F4_rich_side_information"]
MICROBATCHES = [128, 256, 512]
PRECISIONS = ["FP32", "TF32"]

N_WARMUP = 20
N_TIMED = 50

GROUP_CHANNELS = {
    "F0_item_only": {"categorical": {"item_tokens"}, "numeric": False, "flags": False, "user_ctx": False},
    "F4_rich_side_information": {
        "categorical": {"item_tokens", "category_tokens", "brand_tokens",
                         "main_category_tokens", "daypart_tokens", "weekday_tokens"},
        "numeric": True, "flags": True, "user_ctx": True,
    },
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_product_idx_of_class():
    import pyarrow.parquet as pq
    from rees46_eda.loaders import detect_layout

    layout = detect_layout(DATASET_ROOT)
    if layout != "frozen":
        raise RuntimeError(f"expected frozen layout, got {layout!r}")
    cat = pq.read_table(DATASET_ROOT / "mappings" / "catalogue.parquet")
    poc = cat["product_idx"].to_numpy(zero_copy_only=False)
    if poc.shape[0] != EXPECTED_K:
        raise RuntimeError(f"K mismatch: catalogue.parquet has {poc.shape[0]} rows, expected {EXPECTED_K}")
    return poc, layout


def build_named_kwargs(batch_gpu: dict, group_name: str, idx: slice) -> dict:
    ch = GROUP_CHANNELS[group_name]
    kw = dict(
        item_tokens=batch_gpu["item_tokens"][idx],
        lengths=batch_gpu["lengths"][idx],
        attention_mask=batch_gpu["attention_mask"][idx],
    )
    if "category_tokens" in ch["categorical"]:
        kw["category_tokens"] = batch_gpu["category_tokens"][idx]
        kw["brand_tokens"] = batch_gpu["brand_tokens"][idx]
        kw["main_category_tokens"] = batch_gpu["main_category_tokens"][idx]
        kw["daypart_tokens"] = batch_gpu["daypart_tokens"][idx]
        kw["weekday_tokens"] = batch_gpu["weekday_tokens"][idx]
    if ch["numeric"]:
        kw["event_numeric_features"] = batch_gpu["event_numeric_features"][idx]
    if ch["flags"]:
        kw["event_quality_flags"] = batch_gpu["event_quality_flags"][idx]
    if ch["user_ctx"]:
        kw["user_context"] = batch_gpu["user_context"][idx]
        kw["user_context_masks"] = batch_gpu["user_context_masks"][idx]
    return kw


def make_optimizer(model, torch):
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 else no_decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": 1e-5},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=1e-3,
    )


def run_one_step(torch, model, objective_mod, kw, target_class, loss_mask, optimizer, events):
    optimizer.zero_grad(set_to_none=True)

    events["e0"].record()
    h_last = model.encode(
        item_tokens=kw["item_tokens"], lengths=kw["lengths"], attention_mask=kw["attention_mask"],
        category_tokens=kw.get("category_tokens"), brand_tokens=kw.get("brand_tokens"),
        main_category_tokens=kw.get("main_category_tokens"), daypart_tokens=kw.get("daypart_tokens"),
        weekday_tokens=kw.get("weekday_tokens"), event_numeric_features=kw.get("event_numeric_features"),
        event_quality_flags=kw.get("event_quality_flags"),
    )
    events["e1"].record()

    if model.user_ctx_proj is not None:
        uc_in = torch.cat([kw["user_context"], kw["user_context_masks"]], dim=-1)
        uc = model.user_ctx_proj(uc_in)
        readout_in = torch.cat([h_last, uc], dim=-1)
    else:
        readout_in = h_last
    q = model.readout_proj(readout_in)
    q = model.readout_dropout(q)
    if model.contiguous_head:
        w_out = model.item_embed.weight[model._contig_start: model._contig_start + model.K]
    else:
        w_out = model.item_embed.weight.index_select(0, model.product_idx_of_class)
    logits = q @ w_out.T + model.output_bias
    loss_sum, n = objective_mod.exact_ce_loss(logits, target_class, loss_mask)
    denom = torch.clamp(n.to(loss_sum.dtype), min=1.0)
    loss = loss_sum / denom
    events["e2"].record()

    loss.backward()
    events["e3"].record()

    optimizer.step()
    events["e4"].record()

    return int(n.item())


def run_config(torch, nn, model_mod, objective_mod, poc, device, batch_gpu, group_name, B, precision):
    torch.backends.cuda.matmul.allow_tf32 = (precision == "TF32")
    torch.backends.cudnn.allow_tf32 = (precision == "TF32")

    idx = slice(0, B)
    target_class = batch_gpu["target_class"][idx]
    loss_mask = batch_gpu["loss_mask"][idx]
    lengths = batch_gpu["lengths"][idx]
    tokens_per_step = int(lengths.sum().item())

    model = model_mod.build_model(group_name, VOCAB_SIZES, product_idx_of_class=poc).to(device)
    optimizer = make_optimizer(model, torch)
    kw = build_named_kwargs(batch_gpu, group_name, idx)

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    step_times = []
    phase = {"embed_gru": [], "head_loss": [], "backward": [], "optimizer": []}
    n_valid_list = []

    total_steps = N_WARMUP + N_TIMED
    for i in range(total_steps):
        events = {k: torch.cuda.Event(enable_timing=True) for k in ("e0", "e1", "e2", "e3", "e4")}
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        n_valid = run_one_step(torch, model, objective_mod, kw, target_class, loss_mask, optimizer, events)
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        if i >= N_WARMUP:
            step_times.append(t1 - t0)
            phase["embed_gru"].append(events["e0"].elapsed_time(events["e1"]))
            phase["head_loss"].append(events["e1"].elapsed_time(events["e2"]))
            phase["backward"].append(events["e2"].elapsed_time(events["e3"]))
            phase["optimizer"].append(events["e3"].elapsed_time(events["e4"]))
            n_valid_list.append(n_valid)

    max_alloc = torch.cuda.max_memory_allocated(device)
    max_reserved = torch.cuda.max_memory_reserved(device)

    med_step_s = statistics.median(step_times)
    med_n = statistics.median(n_valid_list)

    result = {
        "microbatch": B,
        "group": group_name,
        "precision": precision,
        "median_step_time_s": med_step_s,
        "median_embed_gru_ms": statistics.median(phase["embed_gru"]),
        "median_head_loss_ms": statistics.median(phase["head_loss"]),
        "median_backward_ms": statistics.median(phase["backward"]),
        "median_optimizer_ms": statistics.median(phase["optimizer"]),
        "max_memory_allocated_bytes": int(max_alloc),
        "max_memory_reserved_bytes": int(max_reserved),
        "max_memory_allocated_gib": max_alloc / (1024 ** 3),
        "max_memory_reserved_gib": max_reserved / (1024 ** 3),
        "n_valid_median": med_n,
        "valid_decisions_per_s": med_n / med_step_s,
        "tokens_per_step": tokens_per_step,
        "tokens_per_s": tokens_per_step / med_step_s,
        "n_valid_samples_min_max": [min(n_valid_list), max(n_valid_list)],
        "implied_full_epoch_gpu_hours": (FULL_EPOCH_DECISIONS / (med_n / med_step_s)) / 3600.0,
    }

    del model, optimizer
    torch.cuda.empty_cache()

    log(f"config B={B} group={group_name} prec={precision}: "
        f"step={med_step_s*1000:.2f}ms valid_dec/s={result['valid_decisions_per_s']:.1f} "
        f"reserved={result['max_memory_reserved_gib']:.3f}GiB alloc={result['max_memory_allocated_gib']:.3f}GiB")

    return result


def gpu_only_benchmarks(torch, nn, model_mod, objective_mod, poc, device):
    from window_cache import CacheReader

    log("loading fixed real batch (B=512) from cache/train onto GPU, held for reuse across configs")
    reader = CacheReader(CACHE_TRAIN)
    idx = list(range(512))
    raw = reader.read(idx)
    batch_gpu = {k: v.to(device) for k, v in raw.items()
                 if k in ("item_tokens", "category_tokens", "brand_tokens", "main_category_tokens",
                          "daypart_tokens", "weekday_tokens", "attention_mask", "lengths",
                          "event_numeric_features", "event_quality_flags", "user_context",
                          "user_context_masks", "target_class", "loss_mask")}

    results = []
    for B in MICROBATCHES:
        for group_name in GROUP_NAMES:
            for precision in PRECISIONS:
                r = run_config(torch, nn, model_mod, objective_mod, poc, device, batch_gpu, group_name, B, precision)
                results.append(r)
    return results


def input_path_benchmark(torch, nn, model_mod, objective_mod, poc, device, chosen_B, chosen_group="F4_rich_side_information", precision="FP32"):
    import psutil
    from window_cache import CacheReader

    log(f"input-path throughput: B={chosen_B} group={chosen_group} precision={precision}, 500 steps, reading from CacheReader")
    torch.backends.cuda.matmul.allow_tf32 = (precision == "TF32")
    torch.backends.cudnn.allow_tf32 = (precision == "TF32")

    reader = CacheReader(CACHE_TRAIN)
    n_rows = reader.header["row_count"]

    model = model_mod.build_model(chosen_group, VOCAB_SIZES, product_idx_of_class=poc).to(device)
    optimizer = make_optimizer(model, torch)

    proc = psutil.Process()
    peak_rss_mb = proc.memory_info().rss / (1024.0 * 1024.0)

    n_steps = 500
    total_valid = 0
    torch.cuda.synchronize(device)
    t_start = time.perf_counter()
    row_cursor = 0
    for i in range(n_steps):
        end = row_cursor + chosen_B
        if end > n_rows:
            row_cursor = 0
            end = chosen_B
        idx = list(range(row_cursor, end))
        row_cursor = end

        raw = reader.read(idx)
        batch_gpu = {k: v.to(device, non_blocking=True) for k, v in raw.items()}
        kw = build_named_kwargs(batch_gpu, chosen_group, slice(0, chosen_B))
        target_class = batch_gpu["target_class"][0:chosen_B]
        loss_mask = batch_gpu["loss_mask"][0:chosen_B]

        events = {k: torch.cuda.Event(enable_timing=True) for k in ("e0", "e1", "e2", "e3", "e4")}
        n_valid = run_one_step(torch, model, objective_mod, kw, target_class, loss_mask, optimizer, events)
        total_valid += n_valid

        if i % 50 == 0:
            torch.cuda.synchronize(device)
            rss = proc.memory_info().rss / (1024.0 * 1024.0)
            peak_rss_mb = max(peak_rss_mb, rss)
            log(f"  input-path step {i}/{n_steps} rss_mb={rss:.1f}")

    torch.cuda.synchronize(device)
    t_end = time.perf_counter()
    rss = proc.memory_info().rss / (1024.0 * 1024.0)
    peak_rss_mb = max(peak_rss_mb, rss)

    elapsed = t_end - t_start
    del model, optimizer
    torch.cuda.empty_cache()

    result = {
        "microbatch": chosen_B,
        "group": chosen_group,
        "precision": precision,
        "n_steps": n_steps,
        "elapsed_s": elapsed,
        "total_valid_decisions": total_valid,
        "end_to_end_valid_decisions_per_s": total_valid / elapsed,
        "peak_host_rss_mb": peak_rss_mb,
        "implied_full_epoch_wallclock_hours": (FULL_EPOCH_DECISIONS / (total_valid / elapsed)) / 3600.0,
    }
    log(f"input-path result: {result['end_to_end_valid_decisions_per_s']:.1f} valid decisions/s, "
        f"peak RSS={peak_rss_mb:.1f} MiB")
    return result


def direct_loader_benchmark():
    import psutil
    from rees46_eda.loaders import DecisionWindowLoader, detect_layout
    from window_cache import fetch_bucket_meta

    log("direct-loader comparison: DecisionWindowLoader, 4096 decisions from 4 buckets, cold + warm")
    layout = detect_layout(DATASET_ROOT)
    proc = psutil.Process()

    buckets = [0, 1, 2, 3]
    per_bucket = 1024
    decisions_by_bucket = {}
    for b in buckets:
        meta = fetch_bucket_meta(DATASET_ROOT, layout, "TRAIN", b)
        dec_ids = sorted(meta.keys())[:per_bucket]
        decisions = []
        for did in dec_ids:
            tgt, lm, em, cli = meta[did]
            decisions.append({
                "user_bucket": b, "decision_id": did,
                "target_class": tgt, "loss_mask": lm, "eval_mask": em, "client_idx": cli,
            })
        decisions_by_bucket[b] = decisions

    all_decisions = []
    for b in buckets:
        all_decisions.extend(decisions_by_bucket[b])
    log(f"  sampled {len(all_decisions)} decisions across buckets {buckets}")

    rss_before_cold = proc.memory_info().rss / (1024.0 * 1024.0)
    loader_cold = DecisionWindowLoader(DATASET_ROOT, 50, layout=layout)
    t0 = time.perf_counter()
    batch_cold = loader_cold.collate(all_decisions)
    t1 = time.perf_counter()
    rss_after_cold = proc.memory_info().rss / (1024.0 * 1024.0)
    cold_elapsed = t1 - t0

    t2 = time.perf_counter()
    batch_warm = loader_cold.collate(all_decisions)
    t3 = time.perf_counter()
    rss_after_warm = proc.memory_info().rss / (1024.0 * 1024.0)
    warm_elapsed = t3 - t2

    n = len(all_decisions)
    result = {
        "n_decisions": n,
        "n_buckets": len(buckets),
        "cold": {
            "elapsed_s": cold_elapsed,
            "decisions_per_s": n / cold_elapsed,
            "rss_before_mb": rss_before_cold,
            "rss_after_mb": rss_after_cold,
        },
        "warm": {
            "elapsed_s": warm_elapsed,
            "decisions_per_s": n / warm_elapsed,
            "rss_before_mb": rss_after_cold,
            "rss_after_mb": rss_after_warm,
        },
    }
    log(f"direct-loader: cold={result['cold']['decisions_per_s']:.1f} dec/s, "
        f"warm={result['warm']['decisions_per_s']:.1f} dec/s")
    return result


def compute_e1_verdict(results):
    def find(B, group, prec):
        return next(r for r in results if r["microbatch"] == B and r["group"] == group and r["precision"] == prec)

    c256 = find(256, "F4_rich_side_information", "FP32")
    reserved_256 = c256["max_memory_reserved_gib"]
    if reserved_256 <= 8.0:
        verdict = (f"PASS: max_memory_reserved={reserved_256:.4f} GiB <= 8 GiB at microbatch=256, "
                   f"F4_rich_side_information, FP32, after {N_TIMED} timed steps -> keep exact CE at microbatch 256.")
        return verdict, 256, 1, c256
    c128 = find(128, "F4_rich_side_information", "FP32")
    reserved_128 = c128["max_memory_reserved_gib"]
    if reserved_128 <= 13.0:
        verdict = (f"MITIGATE: max_memory_reserved={reserved_256:.4f} GiB > 8 GiB at microbatch=256 "
                   f"(F4, FP32); at microbatch=128 it is {reserved_128:.4f} GiB <= 13 GiB -> "
                   f"use microbatch 128 with accumulation 2, still exact CE.")
        return verdict, 128, 2, c128
    verdict = (f"BUG: max_memory_reserved={reserved_128:.4f} GiB > 13 GiB even at microbatch=128 "
               f"(F4, FP32; microbatch=256 was {reserved_256:.4f} GiB) -> implementation bug "
               f"(likely a [B,L,K] head or an unfreed eval buffer). Reported as BUG; objective NOT switched.")
    return verdict, 128, 1, c128


def compute_e2_verdict(results):
    c = next(r for r in results if r["microbatch"] == 256 and r["group"] == "F4_rich_side_information"
             and r["precision"] == "FP32")
    rate = c["valid_decisions_per_s"]
    step_ms = c["median_step_time_s"] * 1000.0
    head_loss_frac = c["median_head_loss_ms"] / step_ms
    if rate >= 2000:
        verdict = (f"PASS: valid_decisions/s={rate:.1f} >= 2000 at microbatch=256, F4, FP32 (GPU-only) "
                   f"-> exact CE primary.")
    elif rate >= 800:
        verdict = (f"KEEP-FLAG: valid_decisions/s={rate:.1f} in [800,2000) at microbatch=256, F4, FP32 "
                   f"(GPU-only) -> keep exact CE for the pilot, flag for the full run.")
    elif head_loss_frac >= 0.60:
        verdict = (f"SAMPLED-CE-CANDIDATE: valid_decisions/s={rate:.1f} < 800 AND head+loss "
                   f"fraction={head_loss_frac:.1%} >= 60% of step time at microbatch=256, F4, FP32 "
                   f"(GPU-only) -> sampled CE becomes a separately-named configuration "
                   f"(NOT switched now; reported only).")
    else:
        verdict = (f"NO-TRIGGER: valid_decisions/s={rate:.1f} < 800 at microbatch=256, F4, FP32 (GPU-only), "
                   f"but head+loss fraction={head_loss_frac:.1%} < 60% of step time -> the rule's AND "
                   f"condition is not met; reported only, exact CE unchanged.")
    return verdict, c


def main() -> int:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    log("=== Task T-J benchmark starting ===")

    import torch
    import torch.nn as nn
    import model as model_mod
    import objective as objective_mod

    log(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} "
        f"device_count={torch.cuda.device_count()} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available on Lobot worker")
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    gpu_total_mem = torch.cuda.get_device_properties(device).total_memory
    log(f"GPU: {gpu_name}, total_memory={gpu_total_mem/(1024**3):.2f} GiB")

    poc, layout = load_product_idx_of_class()
    log(f"product_idx_of_class loaded: K={poc.shape[0]} (expected {EXPECTED_K}), layout={layout}")

    gpu_results = gpu_only_benchmarks(torch, nn, model_mod, objective_mod, poc, device)

    e1_verdict, chosen_B, chosen_accum, e1_config = compute_e1_verdict(gpu_results)
    log(f"E1_verdict: {e1_verdict}")
    e2_verdict, e2_config = compute_e2_verdict(gpu_results)
    log(f"E2_verdict: {e2_verdict}")

    input_path_result = input_path_benchmark(torch, nn, model_mod, objective_mod, poc, device, chosen_B)

    direct_loader_result = direct_loader_benchmark()

    output = {
        "task": "T-J",
        "environment": {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_name": gpu_name,
            "gpu_total_memory_gib": gpu_total_mem / (1024 ** 3),
        },
        "vocab_sizes": VOCAB_SIZES,
        "K": EXPECTED_K,
        "n_warmup_steps": N_WARMUP,
        "n_timed_steps": N_TIMED,
        "full_epoch_decisions": FULL_EPOCH_DECISIONS,
        "gpu_only_configs": gpu_results,
        "input_path_throughput": input_path_result,
        "direct_loader_comparison": direct_loader_result,
        "E1_verdict": e1_verdict,
        "E2_verdict": e2_verdict,
        "chosen_microbatch": chosen_B,
        "chosen_accumulation": chosen_accum,
        "E1_config_used": e1_config,
        "E2_config_used": e2_config,
    }

    out_path = ARTIFACTS / "BENCHMARK.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    log(f"wrote {out_path}")
    log("=== Task T-J benchmark DONE ===")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        log("=== FATAL ERROR ===")
        traceback.print_exc()
        raise
