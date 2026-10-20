"""run_sasrec.py -- G0 ONNX export feasibility smoke for SASRec-CE-Rich (A1_RICH256).

Random-initialised weights (seed=2026, SASRecCE's own init RNG), FP32, CPU
only, synthetic inputs only. Writes .../g0/export_smoke/SASREC_REPORT.json
and the .onnx files.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

import common as C

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "g0" / "export_smoke"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> int:
    t_start = time.perf_counter()
    seen = C.isolate_and_verify()
    K = C.get_K()
    model = C.build_sasrec()
    n_params = sum(p.numel() for p in model.parameters())

    gen = torch.Generator().manual_seed(C.SEED)
    calib = C.make_batch(16, "mixed", gen, K)  # forces max(lengths)==L==50 -> T bakes to 50, the full window

    report = {
        "family": "SASREC", "arch": "SASRec-CE-Rich A1_RICH256 (d256/2 blocks/4 heads, untied head)",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "isolation_modules_checked": len(seen), "K": K, "n_params": n_params,
        "torch_version": torch.__version__,
        "precision": "FP32", "device": "cpu", "seed": C.SEED,
        "note_weights": "random init (SASRecCE(..., seed=2026)); no checkpoint",
        "note_T_bake": ("SASRecCE._layout computes T = int(lengths.max().item()); this is a Python-level "
                        "data-dependent conversion, so whatever T the export CALIBRATION batch happens to "
                        "have gets baked into the traced graph as a constant. We calibrate with a batch "
                        "guaranteed to contain a length==50 row, so T bakes to the full L=50 window; the "
                        "graph then relies on `lengths`/`attention_mask`/`position_ids` (real runtime tensor "
                        "inputs, not `.item()`'d) to mask out padding per row for any actual lengths in "
                        "[1,50]. This is the 'fixed L=50 with padding' outcome, not a truly dynamic seq axis."),
    }

    # ---- 1. export: full path ----
    full_wrap = C.SASRecFullWrapper(model)
    full_args = C.sasrec_args(calib)
    seqdim = {0: "batch", 1: "seq"}
    dyn_axes_full = {
        "item_tokens": seqdim, "lengths": {0: "batch"}, "attention_mask": seqdim, "position_ids": seqdim,
        "category_tokens": seqdim, "brand_tokens": seqdim, "main_category_tokens": seqdim,
        "daypart_tokens": seqdim, "weekday_tokens": seqdim, "event_numeric_features": seqdim,
        "event_quality_flags": seqdim, "user_context": {0: "batch"}, "user_context_masks": {0: "batch"},
        "logits": {0: "batch"},
    }
    full_path = str(OUT_DIR / "sasrec_full.onnx")
    with torch.no_grad():
        export_full = C.try_export(full_wrap, full_args, full_path, list(C.SASREC_FULL_ARGS), ["logits"], dyn_axes_full)
    report["export_full"] = export_full

    # ---- 1b. export: query-encoder only (split option) ----
    query_wrap = C.SASRecQueryWrapper(model)
    dyn_axes_q = dict(dyn_axes_full)
    del dyn_axes_q["logits"]
    dyn_axes_q["q"] = {0: "batch"}
    query_path = str(OUT_DIR / "sasrec_query.onnx")
    with torch.no_grad():
        export_query = C.try_export(query_wrap, full_args, query_path, list(C.SASREC_FULL_ARGS), ["q"], dyn_axes_q)
    report["export_query"] = export_query

    # ---- 2. parity ----
    parity_full, parity_query = {}, {}
    sess_full = sess_query = None
    if export_full["status"] == "ok":
        sess_full, load_s_full = C.onnxruntime_session(full_path)
        report["export_full"]["first_load_s"] = load_s_full
        report["export_full"]["declared_input_names"] = [i.name for i in sess_full.get_inputs()]
        report["export_full"]["dropped_input_names"] = [n for n in C.SASREC_FULL_ARGS
                                                         if n not in report["export_full"]["declared_input_names"]]
    if export_query["status"] == "ok":
        sess_query, load_s_q = C.onnxruntime_session(query_path)
        report["export_query"]["first_load_s"] = load_s_q
        report["export_query"]["declared_input_names"] = [i.name for i in sess_query.get_inputs()]
        report["export_query"]["dropped_input_names"] = [n for n in C.SASREC_FULL_ARGS
                                                          if n not in report["export_query"]["declared_input_names"]]

    with torch.no_grad():
        w_out = model.output_weight()
        bias = model.output_bias

    gen_p = torch.Generator().manual_seed(C.SEED + 1)
    for name, length_mode, bsz, missing in C.PARITY_SCENARIOS:
        b = C.make_batch(bsz, length_mode, gen_p, K, missing=missing)
        args = C.sasrec_args(b)
        with torch.no_grad():
            logits_pt = model(**{k: b[k] for k in C.SASREC_FULL_ARGS})
            q_pt = model.query(**{k: b[k] for k in C.SASREC_FULL_ARGS})

        if sess_full is not None:
            feed, _ = C.filtered_feed(sess_full, C.SASREC_FULL_ARGS, args)
            try:
                logits_onnx = sess_full.run(None, feed)[0]
                parity_full[name] = C.compare_logits(logits_pt, logits_onnx, b["target_class"])
            except Exception as e:
                parity_full[name] = {"error": repr(e)}
        if sess_query is not None:
            feed, _ = C.filtered_feed(sess_query, C.SASREC_FULL_ARGS, args)
            try:
                q_onnx = sess_query.run(None, feed)[0]
                vec_metrics = C.compare_vectors(q_pt, q_onnx)
                with torch.no_grad():
                    logits_recon = torch.addmm(bias, torch.as_tensor(q_onnx), w_out.t())
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
            feed, _ = C.filtered_feed(sess_full, C.SASREC_FULL_ARGS, C.sasrec_args(b))
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

    out_path = OUT_DIR / "SASREC_REPORT.json"
    out_path.write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({"family": "SASREC", "export_full": export_full["status"],
                      "export_full_exporter": export_full.get("exporter_used"),
                      "export_query": export_query["status"],
                      "onnx_full_size_mb": report.get("onnx_full_size_mb"),
                      "wall_time_s": report["wall_time_s"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
