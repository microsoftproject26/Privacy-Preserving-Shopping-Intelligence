"""run_gru.py -- G0 ONNX export feasibility smoke for ContextGRU (F4_rich_side_information).

Random-initialised weights (torch.manual_seed(2026)), FP32, CPU only, synthetic
inputs only. Writes .../g0/export_smoke/GRU_REPORT.json and the .onnx files.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import common as C

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "g0" / "export_smoke"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> int:
    t_start = time.perf_counter()
    seen = C.isolate_and_verify()
    K = C.get_K()
    model = C.build_gru()
    n_params = sum(p.numel() for p in model.parameters())

    gen = torch.Generator().manual_seed(C.SEED)
    calib = C.make_batch(16, "mixed", gen, K)  # forces at least one length==L row

    report = {
        "family": "GRU", "arch": "ContextGRU (F4_rich_side_information)",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "isolation_modules_checked": len(seen), "K": K, "n_params": n_params,
        "torch_version": torch.__version__,
        "precision": "FP32", "device": "cpu", "seed": C.SEED,
        "note_weights": "random init (torch.manual_seed(2026) before construction); no checkpoint",
    }

    # ---- 1. export: full path ----
    full_wrap = C.GRUFullWrapper(model)
    full_args = C.gru_args(calib)
    dyn_axes_full = {
        "item_tokens": {0: "batch", 1: "seq"}, "lengths": {0: "batch"},
        "attention_mask": {0: "batch", 1: "seq"}, "category_tokens": {0: "batch", 1: "seq"},
        "brand_tokens": {0: "batch", 1: "seq"}, "main_category_tokens": {0: "batch", 1: "seq"},
        "daypart_tokens": {0: "batch", 1: "seq"}, "weekday_tokens": {0: "batch", 1: "seq"},
        "event_numeric_features": {0: "batch", 1: "seq"}, "event_quality_flags": {0: "batch", 1: "seq"},
        "user_context": {0: "batch"}, "user_context_masks": {0: "batch"}, "logits": {0: "batch"},
    }
    full_path = str(OUT_DIR / "gru_full.onnx")
    with torch.no_grad():
        export_full = C.try_export(full_wrap, full_args, full_path, list(C.GRU_FULL_ARGS), ["logits"], dyn_axes_full)
    report["export_full"] = export_full

    # ---- 1b. export: query-encoder only (split option) ----
    query_wrap = C.GRUQueryWrapper(model)
    dyn_axes_q = dict(dyn_axes_full)
    del dyn_axes_q["logits"]
    dyn_axes_q["q"] = {0: "batch"}
    query_path = str(OUT_DIR / "gru_query.onnx")
    with torch.no_grad():
        export_query = C.try_export(query_wrap, full_args, query_path, list(C.GRU_FULL_ARGS), ["q"], dyn_axes_q)
    report["export_query"] = export_query

    # ---- 2. parity ----
    parity_full, parity_query = {}, {}
    sess_full = sess_query = None
    if export_full["status"] == "ok":
        sess_full, load_s_full = C.onnxruntime_session(full_path)
        report["export_full"]["first_load_s"] = load_s_full
        report["export_full"]["declared_input_names"] = [i.name for i in sess_full.get_inputs()]
        report["export_full"]["dropped_input_names"] = [n for n in C.GRU_FULL_ARGS
                                                         if n not in report["export_full"]["declared_input_names"]]
    if export_query["status"] == "ok":
        sess_query, load_s_q = C.onnxruntime_session(query_path)
        report["export_query"]["first_load_s"] = load_s_q
        report["export_query"]["declared_input_names"] = [i.name for i in sess_query.get_inputs()]
        report["export_query"]["dropped_input_names"] = [n for n in C.GRU_FULL_ARGS
                                                          if n not in report["export_query"]["declared_input_names"]]

    with torch.no_grad():
        w_out = (model.item_embed.weight[model._contig_start:model._contig_start + model.K]
                if model.contiguous_head else model.item_embed.weight.index_select(0, model.product_idx_of_class))
        bias = model.output_bias

    gen_p = torch.Generator().manual_seed(C.SEED + 1)
    for name, length_mode, bsz, missing in C.PARITY_SCENARIOS:
        b = C.make_batch(bsz, length_mode, gen_p, K, missing=missing)
        args = C.gru_args(b)
        with torch.no_grad():
            logits_pt = model(**{k: b[k] for k in C.GRU_FULL_ARGS})
            q_pt = model.query(**{k: b[k] for k in C.GRU_FULL_ARGS})

        if sess_full is not None:
            feed, _ = C.filtered_feed(sess_full, C.GRU_FULL_ARGS, args)
            try:
                logits_onnx = sess_full.run(None, feed)[0]
                parity_full[name] = C.compare_logits(logits_pt, logits_onnx, b["target_class"])
            except Exception as e:
                parity_full[name] = {"error": repr(e)}
        if sess_query is not None:
            feed, _ = C.filtered_feed(sess_query, C.GRU_FULL_ARGS, args)
            try:
                q_onnx = sess_query.run(None, feed)[0]
                vec_metrics = C.compare_vectors(q_pt, q_onnx)
                with torch.no_grad():
                    logits_recon = torch.as_tensor(q_onnx) @ w_out.T + bias
                vec_metrics["reconstructed_logits_vs_pytorch_full"] = C.compare_logits(logits_pt, logits_recon, b["target_class"])
                parity_query[name] = vec_metrics
            except Exception as e:
                parity_query[name] = {"error": repr(e)}

    report["parity_full"] = parity_full
    report["parity_query_plus_reconstructed_head"] = parity_query

    # ---- 3. latency (full logits path; onnxruntime CPUExecutionProvider, intra_op=4) ----
    latency = {}
    gen_l = torch.Generator().manual_seed(C.SEED + 2)
    if sess_full is not None:
        for name, length_mode, bsz in C.LATENCY_SCENARIOS:
            b = C.make_batch(bsz, length_mode, gen_l, K)
            feed, _ = C.filtered_feed(sess_full, C.GRU_FULL_ARGS, C.gru_args(b))
            try:
                latency[name] = C.time_session(sess_full, feed, warm=5, reps=50)
            except Exception as e:
                latency[name] = {"error": repr(e)}
    report["latency_full_cpu_proxy"] = latency
    report["latency_label"] = "SERVER-CPU PROXY (AMD EPYC), NOT a phone/browser measurement"

    if export_full["status"] == "ok":
        report["onnx_full_size_mb"] = C.onnx_total_size_mb(full_path)
    if export_query["status"] == "ok":
        report["onnx_query_size_mb"] = C.onnx_total_size_mb(query_path)

    report["process_peak_rss_mb"] = C.rss_mb()
    report["wall_time_s"] = time.perf_counter() - t_start

    out_path = OUT_DIR / "GRU_REPORT.json"
    out_path.write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({"family": "GRU", "export_full": export_full["status"],
                      "export_full_exporter": export_full.get("exporter_used"),
                      "export_query": export_query["status"],
                      "onnx_full_size_mb": report.get("onnx_full_size_mb"),
                      "wall_time_s": report["wall_time_s"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
