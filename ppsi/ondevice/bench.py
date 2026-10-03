"""The on-device ONNX bench: export, INT8-quantise, then measure latency, memory and parity under onnxruntime CPU.

`python -m ppsi.ondevice.bench --family GRU|SASREC [--variant NAME] (--k K | --catalogue PATH) [--ckpt PATH]
--out DIR` builds the scoring module, exports it (opset 17, fixed batch 1), INT8-quantises it, and then, under
onnxruntime CPU with `intra_op_num_threads = 1` and batch 1, as a server-CPU proxy for ONNX Runtime Web:
  * median / p95 latency (ms) over `--n-queries` (default 1000) synthetic queries after `--warmup` (default 50)
    untimed calls, for both the FP32 and the INT8 session; each session's peak RSS is measured SEPARATELY in its own
    fresh subprocess (`bench_peak_rss_subprocess` / `_rss_worker`), because the parent process would conflate both
    sessions (and torch) into one number;
  * parity of the FP32 export on the synthetic queries against BOTH the traced wrapper (`scoring.gru_dense_query` /
    SASRec `impl="padded_reference"`: the export matches what it was traced from) AND the production forward (packed
    GRU `ContextGRU.query`; SASRec `impl="unpadded"`: the export matches what training computes): max |delta score|
    and the top-20 SET identity rate (ties can reorder an argsort without changing the set);
  * optionally (`--rows` + `--manifest-n`), the same FP32 / INT8 parity on up to 1,000 REAL decisions, plus the FP32
    and INT8 ranks of those decisions written as full-manifest-length int32 rank arrays (-1 for rows not scored,
    `evals/<tag>.ranks_int32.npy`) that the full-catalogue evaluator reads for INT8 retention. The rows NPZ carries
    one array per model input key (leading dim = number of rows) plus `decision_id`, `target_class` (an OOV row,
    target_class < 0, is not scored: its rank stays -1 = credit 0) and `manifest_row` (each row's position in the
    evaluation manifest). `validate_rows_layout` REFUSES (never clamps) a row whose `lengths` is < 1 or whose padding
    is not a contiguous valid prefix.
The result JSON also records the onnxruntime version and the session configuration (`onnxruntime_info`).

`peak_rss_bytes()` prefers the OS high-water mark (`resource.getrusage`, POSIX); `psutil` current RSS is the
documented, approximate fallback on platforms without `resource` (Windows).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from ppsi.seqrec.synthetic import L_MAX, take

from .export import OPSET, add_model_args, export_model, model_from_args
from .io import sha256_file, write_json_atomic
from .quantize import quantize_int8
from .scoring import ScoringModule, example_batch, example_inputs

DEFAULT_N_QUERIES = 1000
DEFAULT_WARMUP = 50
DEFAULT_INTRA_THREADS = 1
DEFAULT_TOP_K = 20
FP32_TAG = "onnx_fp32"
REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------------------------- pure helpers
def peak_rss_bytes() -> int | None:
    """The process's peak RSS so far, in bytes: the OS high-water mark on POSIX (`resource.getrusage`), else a
    `psutil` current-RSS approximation, else None."""
    try:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF)
        return int(ru.ru_maxrss * (1 if sys.platform == "darwin" else 1024))   # macOS reports bytes, Linux KB
    except Exception:  # noqa: BLE001, S110  (no resource module, e.g. on Windows)
        pass
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except Exception:  # noqa: BLE001
        return None


def latency_stats(seconds: np.ndarray) -> dict:
    """median / p95 (ms) of a 1-D array of per-query wall times in seconds."""
    ms = np.asarray(seconds, dtype=np.float64) * 1000.0
    if ms.size == 0:
        raise ValueError("no timed queries")
    return {"median_ms": float(np.median(ms)), "p95_ms": float(np.percentile(ms, 95, method="linear")),
            "n_queries": int(ms.size)}


def parity_stats(scores_a: np.ndarray, scores_b: np.ndarray, *, k: int = DEFAULT_TOP_K) -> dict:
    """max |delta score| and the top-`k` SET identity rate between two `[N, K]` score arrays (a set, not an ordered
    list: a tie can legally reorder an argsort without changing the retrieved set)."""
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError(f"score arrays must be equal-shape [N, K]; got {a.shape} vs {b.shape}")
    if a.shape[1] < k:
        raise ValueError(f"k={k} exceeds the catalogue size {a.shape[1]}")
    delta = np.abs(a - b)
    top_a = np.sort(np.argsort(-a, axis=1, kind="stable")[:, :k], axis=1)
    top_b = np.sort(np.argsort(-b, axis=1, kind="stable")[:, :k], axis=1)
    identical = np.all(top_a == top_b, axis=1)
    return {"n_rows": int(a.shape[0]), "top_k": int(k), "max_abs_delta": float(delta.max()) if delta.size else 0.0,
            "top_k_identity_rate": float(identical.mean()) if identical.size else 1.0}


def rows_to_batches(built, scoring: ScoringModule, n: int, *, seed: int) -> list:
    """`n` synthetic single-decision batches (each tensor's leading dim is 1) in `scoring.input_keys` order."""
    batch = example_batch(built, n=n, seed=seed)
    return [{k: take(batch, [i])[k] for k in scoring.input_keys} for i in range(n)]


def load_rows_npz(path) -> dict:
    """The `--rows` NPZ (module docstring: one array per input key, plus decision_id / target_class /
    manifest_row)."""
    data = np.load(path)
    required = ("decision_id", "target_class", "manifest_row")
    missing = [k for k in required if k not in data.files]
    if missing:
        raise ValueError(f"{path}: rows NPZ is missing {missing}")
    return {k: data[k] for k in data.files}


def validate_rows_layout(rows_data: Mapping[str, np.ndarray], input_keys: Sequence[str]) -> None:
    """Refuse (never silently clamp) a decision whose `lengths` is < 1, or whose padding is not a contiguous valid
    prefix of that length (real events at columns 0..len-1, PAD after). Runs BEFORE any row reaches the model:
    `scoring.gru_dense_query`'s clamp is defensive only and must never decide that bad data is acceptable."""
    if "lengths" not in input_keys:
        raise ValueError("input_keys has no 'lengths' entry to validate against")
    if "lengths" not in rows_data:
        raise ValueError("--rows NPZ has no 'lengths' array")
    lengths = np.asarray(rows_data["lengths"])
    if lengths.ndim != 1:
        raise ValueError("--rows 'lengths' must be 1-D (one value per row)")
    bad_len = np.flatnonzero(lengths < 1)
    if bad_len.size:
        raise ValueError(f"--rows rows {bad_len[:10].tolist()} have lengths < 1 (every decision needs >= 1 real "
                         f"context event); refused, not clamped")
    am = rows_data.get("attention_mask")
    if am is not None:
        am = np.asarray(am)
        if am.ndim != 2 or am.shape[0] != lengths.shape[0]:
            raise ValueError(f"--rows 'attention_mask' must be [n_rows, L] matching 'lengths' ({lengths.shape[0]})")
        ar = np.arange(am.shape[1])[None, :]
        expect = ar < lengths[:, None]
        bad_pad = np.flatnonzero(np.any(am.astype(bool) != expect, axis=1))
        if bad_pad.size:
            raise ValueError(f"--rows rows {bad_pad[:10].tolist()} are not right-padded (attention_mask is not a "
                             f"contiguous prefix of length `lengths`); refused, not repacked")


def real_rows_from_npz(rows_data: Mapping[str, np.ndarray], input_keys: Sequence[str], *, limit: int) -> list:
    """Up to `limit` real per-decision batches (torch tensors, leading dim 1), in `input_keys` order, from a
    validated `--rows` NPZ."""
    validate_rows_layout(rows_data, input_keys)
    missing = [k for k in input_keys if k not in rows_data]
    if missing:
        raise ValueError(f"--rows NPZ is missing required input keys: {missing}")
    n = min(int(limit), int(rows_data[input_keys[0]].shape[0]))
    return [{k: torch.as_tensor(rows_data[k][i:i + 1]) for k in input_keys} for i in range(n)]


def ranks_from_session(session, rows: Mapping[str, np.ndarray], input_keys: Sequence[str],
                       manifest_n: int) -> tuple:
    """(ranks, n_oov_skipped): the full manifest-length int64 rank array (-1 where not scored) from an onnxruntime
    session's per-row full-catalogue scores (the evaluator's exact rank rule), and the number of rows skipped because
    `target_class < 0` (OOV: left at -1 = credit 0, as the evaluator treats them). `manifest_row` is range-checked
    for every row, skipped or not."""
    from ppsi.evaluation.fullrank.ranking import rank_from_scores
    n = int(rows["decision_id"].shape[0])
    out = np.full(int(manifest_n), -1, dtype=np.int64)
    n_oov = 0
    for i in range(n):
        row = int(rows["manifest_row"][i])
        if not (0 <= row < manifest_n):
            raise ValueError(f"rows[{i}].manifest_row {row} out of range [0, {manifest_n})")
        target = int(rows["target_class"][i])
        if target < 0:
            n_oov += 1
            continue
        feed = {name: np.asarray(rows[name][i:i + 1]) for name in input_keys}
        scores = session.run(None, feed)[0]
        s = torch.as_tensor(np.asarray(scores, dtype=np.float64))
        t = torch.as_tensor(rows["target_class"][i:i + 1], dtype=torch.int64)
        out[row] = int(rank_from_scores(s, t).ranks.item())
    return out, n_oov


# --------------------------------------------------------------------------------------------- onnxruntime
DEFAULT_GRAPH_OPT_LEVEL = "ORT_ENABLE_ALL"


def make_session(onnx_path, *, intra_op_num_threads: int = DEFAULT_INTRA_THREADS):
    """A CPU-only onnxruntime session, single-threaded, with the graph optimisation level set explicitly so it is a
    fixed, recorded part of the protocol rather than whatever an onnxruntime version defaults to."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(intra_op_num_threads)
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = getattr(ort.GraphOptimizationLevel, DEFAULT_GRAPH_OPT_LEVEL)
    return ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])


def onnxruntime_info(intra_op_num_threads: int = DEFAULT_INTRA_THREADS) -> dict:
    """The onnxruntime version and the fixed session configuration `make_session` uses."""
    import onnxruntime as ort
    return {"onnxruntime_version": ort.__version__, "graph_optimization_level": DEFAULT_GRAPH_OPT_LEVEL,
            "intra_op_num_threads": int(intra_op_num_threads), "inter_op_num_threads": 1,
            "execution_mode": "ORT_SEQUENTIAL", "providers": ["CPUExecutionProvider"]}


def bench_latency(session, numpy_rows: Sequence[Mapping[str, np.ndarray]], *, warmup: int) -> dict:
    """Median / p95 latency (ms) over `numpy_rows` (batch 1 each), after `warmup` untimed calls. No RSS number here:
    peak RSS is measured per session in an isolated subprocess (`bench_peak_rss_subprocess`)."""
    if not numpy_rows:
        raise ValueError("no queries to bench")
    for i in range(int(warmup)):
        session.run(None, numpy_rows[i % len(numpy_rows)])
    times = np.empty(len(numpy_rows), dtype=np.float64)
    for i, feed in enumerate(numpy_rows):
        t0 = time.perf_counter()
        session.run(None, feed)
        times[i] = time.perf_counter() - t0
    stats = latency_stats(times)
    stats["warmup"] = int(warmup)
    return stats


def bench_peak_rss_subprocess(onnx_path, numpy_rows: Sequence[Mapping[str, np.ndarray]], *,
                              intra_op_num_threads: int, warmup: int) -> int | None:
    """Peak RSS of loading `onnx_path` and running `numpy_rows` through it, measured in a FRESH subprocess
    (`_rss_worker`), so it is that session's own footprint. Raises `subprocess.CalledProcessError` (worker stderr
    attached) if the worker fails; silently returning None would hide a real failure."""
    if not numpy_rows:
        raise ValueError("no queries to bench")
    stacked = {k: np.concatenate([row[k] for row in numpy_rows], axis=0) for k in numpy_rows[0]}
    fd, feed_path = tempfile.mkstemp(suffix=".npz")
    os.close(fd)
    try:
        np.savez(feed_path, **stacked)
        proc = subprocess.run([sys.executable, "-B", "-m", "ppsi.ondevice._rss_worker", "--onnx", str(onnx_path),
                               "--feed", feed_path, "--intra-threads", str(int(intra_op_num_threads)),
                               "--warmup", str(int(warmup))],
                              cwd=str(REPO_ROOT), capture_output=True, text=True, check=True)
        line = proc.stdout.strip().splitlines()[-1]
        return json.loads(line)["peak_rss_bytes"]
    finally:
        Path(feed_path).unlink(missing_ok=True)


def onnx_scores_batch(session, numpy_rows: Sequence[Mapping[str, np.ndarray]]) -> np.ndarray:
    return np.concatenate([np.asarray(session.run(None, feed)[0], dtype=np.float64) for feed in numpy_rows], axis=0)


def pytorch_scores_batch(scoring: ScoringModule, rows: Sequence[Mapping[str, torch.Tensor]]) -> np.ndarray:
    """Scores from the ONNX-traceable wrapper (`gru_dense_query` / SASRec `impl='padded_reference'`): what was
    actually exported, so this checks that the ONNX file matches its own Python source."""
    keys = scoring.input_keys
    with torch.no_grad():
        scoring.eval()
        outs = [scoring(*[row[k] for k in keys]).double().numpy() for row in rows]
    return np.concatenate(outs, axis=0)


def production_scores_batch(built, rows: Sequence[Mapping[str, torch.Tensor]]) -> np.ndarray:
    """Scores from the production forward path (packed-sequence `ContextGRU.query`; SASRec `impl='unpadded'`, the
    trained configuration), not the ONNX-traceable stand-in. Temporarily switches SASRec's `impl` back to the
    trained default and restores it afterwards (`impl` never changes the weights or their shapes)."""
    module = built.module
    prev_impl = getattr(module, "impl", None)
    if prev_impl is not None:
        module.impl = "unpadded"
    try:
        with torch.no_grad():
            module.eval()
            outs = [built.adapter.logits(built.adapter.query(row)).double().numpy() for row in rows]
    finally:
        if prev_impl is not None:
            module.impl = prev_impl
    return np.concatenate(outs, axis=0)


# --------------------------------------------------------------------------------------------- orchestration
def run_pipeline(args: argparse.Namespace) -> dict:
    t_start = time.perf_counter()
    made = model_from_args(args)
    scoring, built = made["scoring"], made["built"]
    input_keys = list(scoring.input_keys)

    rows_data = None
    if args.rows is not None:
        if args.manifest_n is None:
            raise SystemExit("--manifest-n is required together with --rows")
        if args.tag == FP32_TAG:
            raise SystemExit(f"--tag must differ from the reserved FP32 tag {FP32_TAG!r}")
        rows_data = load_rows_npz(args.rows)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    example = example_inputs(built, scoring, n=1, seed=args.seed, min_len=L_MAX)   # full-length trace
    export_rec = export_model(scoring, example, out_dir / "model_fp32.onnx")
    quantize_rec = quantize_int8(out_dir / "model_fp32.onnx", out_dir / "model_int8.onnx")

    rows = rows_to_batches(built, scoring, args.n_queries, seed=args.seed)
    sess_fp32 = make_session(out_dir / "model_fp32.onnx", intra_op_num_threads=args.intra_threads)
    sess_int8 = make_session(out_dir / "model_int8.onnx", intra_op_num_threads=args.intra_threads)
    # feed only the ONNX graph's own declared inputs (a subset of input_keys: the export drops inputs that are never
    # read as tensor data, e.g. attention_mask / position_ids)
    onnx_input_names = [i.name for i in sess_fp32.get_inputs()]
    numpy_rows = [{k: row[k].numpy() for k in onnx_input_names} for row in rows]

    latency_fp32 = bench_latency(sess_fp32, numpy_rows, warmup=args.warmup)
    latency_int8 = bench_latency(sess_int8, numpy_rows, warmup=args.warmup)
    latency_fp32["peak_rss_bytes"] = bench_peak_rss_subprocess(out_dir / "model_fp32.onnx", numpy_rows,
                                                               intra_op_num_threads=args.intra_threads,
                                                               warmup=args.warmup)
    latency_int8["peak_rss_bytes"] = bench_peak_rss_subprocess(out_dir / "model_int8.onnx", numpy_rows,
                                                               intra_op_num_threads=args.intra_threads,
                                                               warmup=args.warmup)

    onnx_fp32_scores = onnx_scores_batch(sess_fp32, numpy_rows)
    wrapper_scores = pytorch_scores_batch(scoring, rows)
    production_scores = production_scores_batch(built, rows)

    result = {"family": built.family, "variant": built.variant, "K": int(built.K), "seed": int(args.seed),
              "opset": OPSET, "intra_op_num_threads": int(args.intra_threads), "input_keys": input_keys,
              "onnx_input_names": onnx_input_names, "onnxruntime": onnxruntime_info(args.intra_threads),
              "checkpoint": args.ckpt, "checkpoint_sha256": sha256_file(args.ckpt) if args.ckpt else None,
              "checkpoint_meta": made["checkpoint_meta"],
              "model_sha256": {"fp32": export_rec["sha256"], "int8": quantize_rec["int8"]["sha256"]},
              "export": export_rec, "quantize": quantize_rec,
              "latency_fp32_ms": latency_fp32, "latency_int8_ms": latency_int8,
              "parity_synthetic_fp32_vs_export_wrapper": parity_stats(onnx_fp32_scores, wrapper_scores,
                                                                      k=DEFAULT_TOP_K),
              "parity_synthetic_fp32_vs_production": parity_stats(onnx_fp32_scores, production_scores,
                                                                  k=DEFAULT_TOP_K),
              "parity_real_fp32_vs_production": None, "parity_real_int8_vs_production": None,
              "parity_note": ("parity_synthetic_* are on --n-queries synthetic decisions (always computed); "
                              "parity_real_* are on the --rows real decisions (only when --rows is given); "
                              "the two are never averaged together"),
              "retention": None}

    if rows_data is not None:
        real_rows = real_rows_from_npz(rows_data, input_keys, limit=DEFAULT_N_QUERIES)     # validated; refuses bad rows
        real_numpy_rows = [{k: row[k].numpy() for k in onnx_input_names} for row in real_rows]
        real_production = production_scores_batch(built, real_rows)
        result["parity_real_fp32_vs_production"] = parity_stats(onnx_scores_batch(sess_fp32, real_numpy_rows),
                                                                real_production, k=DEFAULT_TOP_K)
        result["parity_real_int8_vs_production"] = parity_stats(onnx_scores_batch(sess_int8, real_numpy_rows),
                                                                real_production, k=DEFAULT_TOP_K)
        ort_info = onnxruntime_info(args.intra_threads)
        retention = {}
        for tag, sess, model_key in ((FP32_TAG, sess_fp32, "fp32"), (args.tag, sess_int8, "int8")):
            ranks, n_oov = ranks_from_session(sess, rows_data, onnx_input_names, args.manifest_n)
            ranks_path = out_dir / "evals" / f"{tag}.ranks_int32.npy"
            ranks_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(ranks_path, ranks.astype(np.int32))
            n_rows = int(rows_data["decision_id"].shape[0])
            retention[tag] = {
                "tag": tag, "model": model_key, "model_sha256": result["model_sha256"][model_key],
                "ranks_path": str(ranks_path), "ranks_sha256": sha256_file(ranks_path),
                "manifest_n": int(args.manifest_n), "n_rows_in_npz": n_rows, "n_oov_skipped": int(n_oov),
                "n_rows_scored": n_rows - int(n_oov), "onnxruntime_version": ort_info["onnxruntime_version"],
                "intra_op_num_threads": ort_info["intra_op_num_threads"],
                "graph_optimization_level": ort_info["graph_optimization_level"],
                "rows_npz_sha256": sha256_file(args.rows)}
        result["retention"] = retention

    result["wall_seconds"] = float(time.perf_counter() - t_start)
    result["result_sha256"] = write_json_atomic(out_dir / "BENCH_RESULT.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="export + INT8-quantise + bench a scoring model under onnxruntime CPU")
    add_model_args(ap)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-queries", type=int, default=DEFAULT_N_QUERIES, dest="n_queries")
    ap.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    ap.add_argument("--intra-threads", type=int, default=DEFAULT_INTRA_THREADS, dest="intra_threads")
    ap.add_argument("--rows", default=None, help="NPZ of real decisions (module docstring)")
    ap.add_argument("--manifest-n", type=int, default=None, dest="manifest_n")
    ap.add_argument("--tag", default="onnx_int8", help="INT8 ranks file: evals/<tag>.ranks_int32.npy (the FP32 ranks "
                                                       "of the same rows go to evals/onnx_fp32.ranks_int32.npy)")
    ap.add_argument("--quiet", action="store_true", help="print only the timing summary, not the result")
    args = ap.parse_args(argv)
    result = run_pipeline(args)
    if args.quiet:
        print(json.dumps({"wall_seconds": result["wall_seconds"], "latency_fp32_ms": result["latency_fp32_ms"],
                          "latency_int8_ms": result["latency_int8_ms"], "result_sha256": result["result_sha256"]},
                         indent=1))
    else:
        print(json.dumps(result, indent=1))
    return 0


__all__ = ["FP32_TAG", "bench_latency", "bench_peak_rss_subprocess", "latency_stats", "load_rows_npz", "main",
           "make_session", "onnx_scores_batch", "onnxruntime_info", "parity_stats", "peak_rss_bytes",
           "production_scores_batch", "pytorch_scores_batch", "ranks_from_session", "real_rows_from_npz",
           "rows_to_batches", "run_pipeline", "validate_rows_layout"]

if __name__ == "__main__":
    raise SystemExit(main())
