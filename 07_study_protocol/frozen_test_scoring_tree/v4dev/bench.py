"""v4dev.bench — the ONNX protocol bench + orchestration CLI (DEV-01 items 3-4).

`python -B -m v4dev.bench --ckpt <path> --family GRU|SASREC [--variant NAME] --v1-root <path> --out <dir>` is the one
Ops command: build the checkpoint's scoring module, export it (v4dev.export, opset 17, fixed batch 1), INT8-quantise
it (v4dev.quantize), then under onnxruntime CPU with `intra_op_num_threads = 1`, batch 1:
  * median / p95 latency (ms) over `--n-queries` (default 1000) synthetic queries, after `--warmup` (default 50)
    untimed calls, for both the FP32 and the INT8 session; each session's peak RSS is measured SEPARATELY, in its
    own fresh subprocess (`bench_peak_rss_subprocess` / `v4dev._rss_worker`) — `resource.getrusage` on the parent
    bench process would conflate both sessions (and torch, and everything else it holds) into one number, not
    either session's own footprint (Opus review C.2);
  * parity of the FP32 ONNX export on synthetic queries, against BOTH `v4dev.scoring`'s ONNX-traceable reimplementation
    (`gru_dense_query` / SASRec `impl="padded_reference"` — confirms the export matches what it was traced from) AND
    the real PRODUCTION forward (packed GRU `ContextGRU.query`; SASRec `impl="unpadded"` — confirms the export
    matches what the server/training code actually computes; Opus review C.1): max |Δscore| and top-20 SET identity
    rate (ties can reorder an argsort without changing the set, so identity is checked as a set, not an ordered list);
  * (optional, `--rows` + `--manifest-n`): the SAME FP32/INT8-vs-production parity, but on up to 1,000 REAL rows
    (Opus review C.3; the synthetic parity above is always computed and always labelled `_synthetic_`, so the two
    are never confused), PLUS INT8 scores of these real, already-materialized decisions written as a per-row int32
    rank array in the existing convention (v4train.trainer / v4train.ckpt_avg: full manifest length, -1 for rows
    not scored, `evals/<tag>.ranks_int32.npy`) the frozen T3 evaluator / v4train.bootstrap_ci can read for the
    retention numbers. Rows are supplied as an NPZ this package does not itself produce (materializing
    MAIN_VALIDATION decisions is a v4int/data-pipeline job outside DEV-01's scope): one array per input key
    `scoring.input_keys` (leading dim = number of rows, batch-of-1 semantics per row), plus `decision_id`,
    `target_class` (rankable rows: [0, K); an OOV row, target_class < 0, is NOT scored: its rank stays -1 = credit 0
    exactly as `v4eval` treats it, and it is counted in `n_oov_skipped`, REV-ANA-01 D-F1) and `manifest_row` (each row's position in the full
    evaluation manifest, so the sparse ranks array can be scattered into `--manifest-n` length). `validate_rows_layout`
    REFUSES (never silently clamps) a row whose `lengths` is < 1 or whose padding is not a contiguous valid prefix
    of that length (Opus review C.4).

Feature view (DEV-04): the module is built with the checkpoint's OWN registered view (v4dev.checkpointio.
resolve_feature_view: checkpoint meta + RUN_START -> the registered schema; C_SMALL is a derived SCHEMA-01R1 view), and
`--rows` must carry that view (`v4dev.dump_rows.check_rows_view`, run right after the build and before any export or
scoring: every key the view consumes, full Clean V2 float widths, the recorded column names / view / base self-hash
-- required for a non-identity projection -- and the run's eval manifest / history policy). Any mismatch refuses.
BENCH_RESULT.json records `feature_view` and `rows_view_check`.

The output JSON also records the onnxruntime version and the graph optimisation level the session was built with
(`onnxruntime_info`; Opus review C.4).

Every onnxruntime / `onnx`-package call is lazy-imported inside the function that needs it (NOT installed in the
local dev venv; only torch/numpy/psutil are). `peak_rss_bytes()` prefers the OS high-water mark (`resource.getrusage`,
POSIX — the true peak, not a sampled proxy); `psutil` current RSS is the (documented, approximate) fallback on
platforms without `resource` (i.e. local Windows dev only — the server is POSIX). It reports the CALLING process's
own peak, which is why it is only ever called from inside `v4dev._rss_worker`'s isolated subprocess, never from the
parent bench process."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import torch

from .checkpointio import build_scoring_module, sha256_file
from .dump_rows import check_rows_view
from .export import OPSET, export_model
from .quantize import quantize_int8
from .scoring import ScoringModule, example_batch

DEFAULT_N_QUERIES = 1000
DEFAULT_WARMUP = 50
DEFAULT_INTRA_THREADS = 1
DEFAULT_TOP_K = 20
FP32_TAG = "onnx_fp32"      # K16: the FP32 ONNX ranks of the SAME rows (evals/onnx_fp32.ranks_int32.npy)


# --------------------------------------------------------------------------------------------- pure helpers (local)
def peak_rss_bytes() -> Optional[int]:
    """The process's peak RSS so far, in bytes: the OS high-water mark on POSIX (`resource.getrusage`), else a
    `psutil` current-RSS approximation, else None."""
    try:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF)
        return int(ru.ru_maxrss * (1 if sys.platform == "darwin" else 1024))   # macOS reports bytes, Linux KB
    except Exception:
        pass
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


def latency_stats(seconds: np.ndarray) -> dict:
    """median / p95 (ms) of a 1-D array of per-query wall times in seconds."""
    ms = np.asarray(seconds, dtype=np.float64) * 1000.0
    if ms.size == 0:
        raise ValueError("no timed queries")
    return {"median_ms": float(np.median(ms)), "p95_ms": float(np.percentile(ms, 95, method="linear")),
            "n_queries": int(ms.size)}


def parity_stats(scores_a: np.ndarray, scores_b: np.ndarray, *, k: int = DEFAULT_TOP_K) -> dict:
    """max |Δscore| and the top-`k` SET identity rate between two `[N, K]` score arrays (a set, not an ordered list:
    a tie can legally reorder an argsort without changing the retrieved set)."""
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
    from v4models.synthetic import take
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
    """Refuse (never silently clamp) a `--rows` decision whose `lengths` is < 1, or whose padding is not a
    contiguous valid prefix of that length (DATA_CONTRACT_CLEAN_V2 rezero convention: real events at columns
    0..len-1, PAD after). Opus review C.4: `v4dev.scoring.gru_dense_query`'s `.clamp(min=0)` is defensive only (it
    stops a length-0 row IndexErroring the gather) and must never be the thing that decides bad real data is
    acceptable -- this runs BEFORE any row reaches the model."""
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
    validated `--rows` NPZ (module docstring's NPZ contract: one array per `scoring.input_keys` name)."""
    validate_rows_layout(rows_data, input_keys)
    missing = [k for k in input_keys if k not in rows_data]
    if missing:
        raise ValueError(f"--rows NPZ is missing required input keys: {missing}")
    n = min(int(limit), int(rows_data[input_keys[0]].shape[0]))
    return [{k: torch.as_tensor(rows_data[k][i:i + 1]) for k in input_keys} for i in range(n)]


def ranks_from_session(session, rows: Mapping[str, np.ndarray], input_keys: Sequence[str],
                       manifest_n: int) -> tuple:
    """(ranks, n_oov_skipped): the full manifest-length int64 rank array (-1 where not scored) from an onnxruntime
    session's per-row full-K scores (`v4eval.ranking.rank_from_scores`: the exact full-catalogue rank rule, one block
    since K is the whole output), and the number of rows skipped because `target_class < 0` (OOV: `rank_from_scores`
    raises on them; they are left at -1 = credit 0, as `v4eval` does -- REV-ANA-01 D-F1). `manifest_row` is range-checked
    for every row, skipped or not."""
    from v4eval.ranking import rank_from_scores
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
        r = rank_from_scores(s, t)
        out[row] = int(r.ranks.item())
    return out, n_oov


def ranks_from_int8_session(session, rows: Mapping[str, np.ndarray], input_keys: Sequence[str],
                            manifest_n: int) -> np.ndarray:
    """`ranks_from_session`'s rank array only (kept for callers that do not need the OOV count)."""
    return ranks_from_session(session, rows, input_keys, manifest_n)[0]


# ---------------------------------------------------------------------------------------- onnxruntime (server-only)
DEFAULT_GRAPH_OPT_LEVEL = "ORT_ENABLE_ALL"


def make_session(onnx_path, *, intra_op_num_threads: int = DEFAULT_INTRA_THREADS):
    """A CPU-only onnxruntime session, single-threaded (T3-style discipline: fixed, recorded runtime flags). The
    graph optimisation level is set explicitly (not left to whatever onnxruntime defaults to) so it is a fixed,
    reproducible part of the protocol, not just whatever a given onnxruntime version happens to default to."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(intra_op_num_threads)
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = getattr(ort.GraphOptimizationLevel, DEFAULT_GRAPH_OPT_LEVEL)
    return ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])


def onnxruntime_info(intra_op_num_threads: int = DEFAULT_INTRA_THREADS) -> dict:
    """The onnxruntime version and the fixed runtime configuration `make_session` builds every session with (Opus
    review C.4: record it, do not just assume the reader knows)."""
    import onnxruntime as ort
    return {"onnxruntime_version": ort.__version__, "graph_optimization_level": DEFAULT_GRAPH_OPT_LEVEL,
            "intra_op_num_threads": int(intra_op_num_threads), "inter_op_num_threads": 1,
            "execution_mode": "ORT_SEQUENTIAL", "providers": ["CPUExecutionProvider"]}


def bench_latency(session, numpy_rows: Sequence[Mapping[str, np.ndarray]], *, warmup: int) -> dict:
    """Median / p95 latency (ms) over `numpy_rows` (batch 1 each), after `warmup` untimed calls. Carries no RSS
    number (Opus review C.2): peak RSS is measured separately, per session, in an isolated subprocess
    (`bench_peak_rss_subprocess`) -- this in-process loop shares its process with whatever OTHER session the caller
    also built, so its own `peak_rss_bytes()` would conflate the two."""
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
                              intra_op_num_threads: int, warmup: int) -> Optional[int]:
    """Peak RSS of loading `onnx_path` and running `numpy_rows` through it, measured in a FRESH subprocess
    (`v4dev._rss_worker`) so it is that session's own footprint, isolated from the parent bench process and from any
    other session it also holds (Opus review C.2). Raises `subprocess.CalledProcessError` (worker stderr attached)
    if the worker fails for any reason, including a missing onnxruntime -- silently returning None there would hide
    a real failure the caller (already onnxruntime-only past this point) needs to see."""
    if not numpy_rows:
        raise ValueError("no queries to bench")
    stacked = {k: np.concatenate([row[k] for row in numpy_rows], axis=0) for k in numpy_rows[0]}
    code_dir = Path(__file__).resolve().parents[1]
    fd, feed_path = tempfile.mkstemp(suffix=".npz")
    os.close(fd)
    try:
        np.savez(feed_path, **stacked)
        proc = subprocess.run([sys.executable, "-B", "-m", "v4dev._rss_worker", "--onnx", str(onnx_path),
                              "--feed", feed_path, "--intra-threads", str(int(intra_op_num_threads)),
                              "--warmup", str(int(warmup))],
                             cwd=str(code_dir), capture_output=True, text=True, check=True)
        line = proc.stdout.strip().splitlines()[-1]
        return json.loads(line)["peak_rss_bytes"]
    finally:
        Path(feed_path).unlink(missing_ok=True)


def onnx_scores_batch(session, numpy_rows: Sequence[Mapping[str, np.ndarray]]) -> np.ndarray:
    return np.concatenate([np.asarray(session.run(None, feed)[0], dtype=np.float64) for feed in numpy_rows], axis=0)


def pytorch_scores_batch(scoring: ScoringModule, rows: Sequence[Mapping[str, torch.Tensor]]) -> np.ndarray:
    """Scores from `v4dev.scoring`'s ONNX-traceable wrapper (`gru_dense_query` / SASRec `impl='padded_reference'`)
    -- what was actually EXPORTED and traced, so this is a check that the ONNX file matches its own Python source."""
    keys = scoring.input_keys
    with torch.no_grad():
        scoring.eval()
        outs = [scoring(*[row[k] for k in keys]).double().numpy() for row in rows]
    return np.concatenate(outs, axis=0)


def production_scores_batch(built, rows: Sequence[Mapping[str, torch.Tensor]]) -> np.ndarray:
    """Scores from the REAL production forward path (packed-sequence `ContextGRU.query`; SASRec `impl='unpadded'`,
    the trained/deployed configuration) -- NOT `v4dev.scoring`'s ONNX-traceable stand-in. Opus review C.1: parity
    must also be checked against what the server/training code actually computes, not only against v4dev's own
    reimplementation of it. Temporarily flips SASRec's `impl` back to the trained default and restores it afterward
    (a plain runtime attribute; v4models' own tests establish `impl` never changes the weights or their shapes)."""
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


# --------------------------------------------------------------------------------------------------- orchestration
def load_rows_meta(rows_path) -> Optional[dict]:
    """`<rows' directory>/rows.meta.json` (written by v4dev.dump_rows), or None when absent."""
    p = Path(rows_path).parent / "rows.meta.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


TEST_SPLIT = "RETRO_TEST"


def check_bench_rows(made: Mapping, rows_data: Mapping, rows_meta: Optional[Mapping], *, ckpt_path=None) -> dict:
    """DEV-04: refuse `--rows` that do not carry the checkpoint's registered view (`check_rows_view`). The recorded
    column-order provenance is REQUIRED whenever the model's projection is not the identity (a derived view selects
    columns by name index); the run binding is the checkpoint's RUN_START eval manifest / history policy.
    DEV-05: a `feature_view_binding.checkpoint_sha256` must be THIS checkpoint's. RETRO_TEST rows
    (v4int.dump_rows_retro, `meta.split` RETRO_TEST) come from the TEST counterpart manifest, not the run's VALIDATION
    eval manifest, so for them the eval-manifest equality is replaced by a REQUIRED binding: a feature_view_binding of
    split RETRO_TEST made with `--ckpt` of this very checkpoint (sha256), plus the gate's register entry id."""
    from .dump_rows import DumpRowsError
    built = made["built"]
    proj = getattr(built, "projection", None)
    rec = made.get("feature_view") or {}
    schema = made.get("schema")
    if schema is None:
        raise ValueError("the scoring module was built without a schema object; cannot check the rows' view")
    meta = dict(rows_meta or {})
    fb = meta.get("feature_view_binding") or {}
    ck_sha = sha256_file(ckpt_path) if ckpt_path is not None else None
    if fb.get("checkpoint_sha256") and ck_sha is not None and fb["checkpoint_sha256"] != ck_sha:
        raise DumpRowsError(f"the rows were bound to checkpoint sha256 {fb['checkpoint_sha256']}, not this --ckpt "
                            f"({ck_sha})")
    run_binding = {"eval_manifest_id": rec.get("eval_manifest_id"), "history_policy": rec.get("history_policy")}
    if meta.get("split") == TEST_SPLIT or fb.get("split") == TEST_SPLIT:
        if fb.get("split") != TEST_SPLIT or meta.get("split") != TEST_SPLIT:
            raise DumpRowsError("RETRO_TEST rows need meta.split and feature_view_binding.split both RETRO_TEST "
                                "(v4int.dump_rows_retro --ckpt)")
        if ck_sha is None or fb.get("checkpoint_sha256") != ck_sha:
            raise DumpRowsError("RETRO_TEST rows must be bound to this checkpoint (dump_rows_retro --ckpt); refused")
        if meta.get("test_exposure_register_entry_id") is None or \
                fb.get("test_exposure_register_entry_id") != meta.get("test_exposure_register_entry_id"):
            raise DumpRowsError("RETRO_TEST rows lack a consistent TEST gate register entry id")
        if fb.get("run_eval_manifest_id") is not None and rec.get("eval_manifest_id") is not None and \
                fb["run_eval_manifest_id"] != rec["eval_manifest_id"]:
            raise DumpRowsError("the RETRO_TEST rows were bound to a run with another VALIDATION eval manifest")
        run_binding = {"history_policy": rec.get("history_policy")}
    out = check_rows_view(rows_data, meta, schema, require_batch_view=bool(proj is not None and not proj.is_identity),
                          run_binding=run_binding)
    out["split"] = meta.get("split")
    out["checkpoint_bound"] = bool(fb.get("checkpoint_sha256")) and fb.get("checkpoint_sha256") == ck_sha
    return out


def run_pipeline(args: argparse.Namespace) -> dict:
    t_start = time.perf_counter()
    made = build_scoring_module(args.family, variant=args.variant, ckpt_path=args.ckpt, seed=args.seed, k=args.k,
                                catalogue=args.catalogue, v1_root=args.v1_root,
                                schema_id=getattr(args, "schema_id", None),
                                schema_registry=getattr(args, "schema_registry", None))
    scoring, built = made["scoring"], made["built"]
    input_keys = list(scoring.input_keys)

    rows_data = rows_meta = rows_view = None
    if args.rows is not None:                                  # refuse a wrong view before any export / scoring
        if args.manifest_n is None:
            raise SystemExit("--manifest-n is required together with --rows")
        if args.tag == FP32_TAG:
            raise SystemExit(f"--tag must differ from the reserved FP32 tag {FP32_TAG!r}")
        rows_data = load_rows_npz(args.rows)
        rows_meta = load_rows_meta(args.rows)
        rows_view = check_bench_rows(made, rows_data, rows_meta, ckpt_path=args.ckpt)
        if rows_meta is not None and rows_meta.get("manifest_n") is not None \
                and int(rows_meta["manifest_n"]) != int(args.manifest_n):
            raise SystemExit(f"--manifest-n {args.manifest_n} != rows.meta.json manifest_n {rows_meta['manifest_n']}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from v4models.synthetic import L_MAX

    from .scoring import example_inputs
    example = example_inputs(built, scoring, n=1, seed=args.seed, min_len=L_MAX)   # full-length trace (v4dev.scoring)
    export_rec = export_model(scoring, example, out_dir / "model_fp32.onnx")
    quantize_rec = quantize_int8(out_dir / "model_fp32.onnx", out_dir / "model_int8.onnx")

    rows = rows_to_batches(built, scoring, args.n_queries, seed=args.seed)

    sess_fp32 = make_session(out_dir / "model_fp32.onnx", intra_op_num_threads=args.intra_threads)
    sess_int8 = make_session(out_dir / "model_int8.onnx", intra_op_num_threads=args.intra_threads)
    # feed only the ONNX graph's own declared inputs (a subset of input_keys: v4dev.export drops any input never
    # read as real tensor data, e.g. attention_mask/position_ids -- redundant with lengths in both families here)
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
    parity_synthetic_vs_export_wrapper = parity_stats(onnx_fp32_scores, wrapper_scores, k=DEFAULT_TOP_K)
    parity_synthetic_vs_production = parity_stats(onnx_fp32_scores, production_scores, k=DEFAULT_TOP_K)

    result = {"family": built.family, "variant": built.variant, "K": int(built.K), "seed": int(args.seed),
              "opset": OPSET, "intra_op_num_threads": int(args.intra_threads), "input_keys": input_keys,
              "onnx_input_names": onnx_input_names, "onnxruntime": onnxruntime_info(args.intra_threads),
              "checkpoint": str(args.ckpt), "checkpoint_sha256": sha256_file(args.ckpt),
              "checkpoint_meta": made["checkpoint_meta"], "feature_view": made.get("feature_view"),
              "rows_view_check": rows_view,
              "model_sha256": {"fp32": export_rec["sha256"], "int8": quantize_rec["int8"]["sha256"]},
              "export": export_rec, "quantize": quantize_rec,
              "latency_fp32_ms": latency_fp32, "latency_int8_ms": latency_int8,
              "parity_synthetic_fp32_vs_export_wrapper": parity_synthetic_vs_export_wrapper,
              "parity_synthetic_fp32_vs_production": parity_synthetic_vs_production,
              "parity_real_fp32_vs_production": None, "parity_real_int8_vs_production": None,
              "parity_note": ("parity_synthetic_* are on --n-queries synthetic decisions (always computed); "
                              "parity_real_* are on the --rows real decisions (only when --rows is given) -- "
                              "the two are never averaged together"),
              "retention": None, "retention_note": None}

    if args.rows is not None:
        real_rows = real_rows_from_npz(rows_data, input_keys, limit=DEFAULT_N_QUERIES)     # validated; refuses bad rows
        real_numpy_rows = [{k: row[k].numpy() for k in onnx_input_names} for row in real_rows]
        real_onnx_fp32 = onnx_scores_batch(sess_fp32, real_numpy_rows)
        real_onnx_int8 = onnx_scores_batch(sess_int8, real_numpy_rows)
        real_production = production_scores_batch(built, real_rows)
        result["parity_real_fp32_vs_production"] = parity_stats(real_onnx_fp32, real_production, k=DEFAULT_TOP_K)
        result["parity_real_int8_vs_production"] = parity_stats(real_onnx_int8, real_production, k=DEFAULT_TOP_K)

        ort_info = onnxruntime_info(args.intra_threads)
        rows_npz_sha = sha256_file(args.rows)
        meta_path = Path(args.rows).parent / "rows.meta.json"
        rows_meta_sha = sha256_file(meta_path) if meta_path.is_file() else None
        retention = {}
        for tag, sess, model_key in ((FP32_TAG, sess_fp32, "fp32"), (args.tag, sess_int8, "int8")):
            ranks, n_oov = ranks_from_session(sess, rows_data, onnx_input_names, args.manifest_n)
            ranks_path = out_dir / "evals" / f"{tag}.ranks_int32.npy"
            ranks_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(ranks_path, ranks.astype(np.int32))
            retention[tag] = {
                "tag": tag, "model": model_key, "model_sha256": result["model_sha256"][model_key],
                "ranks_path": str(ranks_path), "ranks_sha256": sha256_file(ranks_path),
                "manifest_n": int(args.manifest_n), "n_rows_in_npz": int(rows_data["decision_id"].shape[0]),
                "n_oov_skipped": int(n_oov), "n_rows_scored": int(rows_data["decision_id"].shape[0]) - int(n_oov),
                "onnxruntime_version": ort_info["onnxruntime_version"],
                "intra_op_num_threads": ort_info["intra_op_num_threads"],
                "graph_optimization_level": ort_info["graph_optimization_level"],
                "rows_npz_sha256": rows_npz_sha, "rows_meta_sha256": rows_meta_sha}
        result["retention"] = retention
        result["rows_first_users"] = (rows_meta or {}).get("first_users")
    else:
        result["retention_note"] = ("no --rows/--manifest-n given: INT8 retention ranks were not written this run "
                                    "(OPEN: Ops must materialize MAIN_VALIDATION rows and re-run with --rows)")

    result["wall_seconds"] = float(time.perf_counter() - t_start)
    from v4train.checkpoint import write_json_atomic
    result["result_sha256"] = write_json_atomic(out_dir / "BENCH_RESULT.json", result)
    return result


# --------------------------------------------------------------------------------------------- INT8 pass 2: latency only
LATENCY_PASS2_NAME = "LATENCY_PASS2.json"


def read_pass1_record(out_dir) -> dict:
    """The few fields of pass 1's own BENCH_RESULT.json that pass 2 must assert against (model / rows / checkpoint
    shas, the fixed timing protocol). ONLY these identity / protocol keys are read; pass 1's accuracy and latency
    values are never read or copied (DL 368: pass-1 latency is 'loaded host', never the device number)."""
    p = Path(out_dir) / "BENCH_RESULT.json"
    if not p.is_file():
        raise SystemExit(f"--latency-only: {p} (pass 1's record) is missing")
    rec = json.loads(p.read_text(encoding="utf-8"))
    try:
        ret = rec["retention"]
        if not ret:
            raise KeyError("retention")
        rows_shas = {b["rows_npz_sha256"] for b in ret.values()}
        meta_shas = {b.get("rows_meta_sha256") for b in ret.values()}
        out = {"pass1_record_path": str(p), "pass1_record_file_sha256": sha256_file(p),
               "model_sha256": dict(rec["model_sha256"]), "checkpoint_sha256": rec["checkpoint_sha256"],
               "family": rec["family"], "variant": rec["variant"], "K": int(rec["K"]), "seed": int(rec["seed"]),
               "intra_op_num_threads": int(rec["intra_op_num_threads"]),
               "n_queries": int(rec["latency_fp32_ms"]["n_queries"]), "warmup": int(rec["latency_fp32_ms"]["warmup"]),
               "rows_npz_sha256": None, "rows_meta_sha256": None, "rows_first_users": rec.get("rows_first_users")}
        if int(rec["latency_int8_ms"]["n_queries"]) != out["n_queries"] or int(rec["latency_int8_ms"]["warmup"]) != out["warmup"]:
            raise SystemExit("--latency-only: pass 1's FP32 and INT8 timing protocols differ in its own record")
    except KeyError as e:
        raise SystemExit(f"--latency-only: pass 1's record lacks {e} (pass 1 must have run with --rows)") from None
    if len(rows_shas) != 1 or len(meta_shas) != 1:
        raise SystemExit("--latency-only: pass 1's record names more than one rows file")
    out["rows_npz_sha256"], out["rows_meta_sha256"] = next(iter(rows_shas)), next(iter(meta_shas))
    for k in ("fp32", "int8"):
        if not out["model_sha256"].get(k):
            raise SystemExit(f"--latency-only: pass 1's record has no {k} model sha256")
    return out


def assert_pass1_identity(out_dir, rows_path, rec: Mapping) -> dict:
    """Assert, from the files on disk, that the INT8 / FP32 models and the rows file are byte-identical to what pass 1
    recorded (sha256). Any difference refuses; nothing is exported or re-evaluated."""
    out_dir = Path(out_dir)
    found = {"fp32": sha256_file(out_dir / "model_fp32.onnx") if (out_dir / "model_fp32.onnx").is_file() else None,
             "int8": sha256_file(out_dir / "model_int8.onnx") if (out_dir / "model_int8.onnx").is_file() else None,
             "rows_npz": sha256_file(rows_path) if Path(rows_path).is_file() else None}
    for k in ("fp32", "int8"):
        if found[k] != rec["model_sha256"][k]:
            raise SystemExit(f"--latency-only: model_{k}.onnx sha256 {found[k]} != pass 1's record {rec['model_sha256'][k]}")
    if found["rows_npz"] != rec["rows_npz_sha256"]:
        raise SystemExit(f"--latency-only: rows file {rows_path} sha256 {found['rows_npz']} != pass 1's record {rec['rows_npz_sha256']}")
    meta = Path(rows_path).parent / "rows.meta.json"
    found["rows_meta"] = sha256_file(meta) if meta.is_file() else None
    if rec["rows_meta_sha256"] != found["rows_meta"]:
        raise SystemExit(f"--latency-only: rows.meta.json sha256 {found['rows_meta']} != pass 1's record {rec['rows_meta_sha256']}")
    return found


def run_latency_only(args: argparse.Namespace) -> dict:
    """INT8 pass 2 (DL 368): measure single-thread FP32 / INT8 ONNX latency ONLY, on the pinned pass-1 artifact dir (`--out`),
    exactly as pass 1 did (same synthetic queries, n-queries, warmup, intra threads, session config). No export, no
    quantisation, no accuracy / parity / RSS. Writes `<out>/LATENCY_PASS2.json` (write-once); every pass-1 file is
    left untouched (sha re-checked after the timing)."""
    t_start = time.perf_counter()
    out_dir = Path(args.out)
    if not out_dir.is_dir():
        raise SystemExit(f"--latency-only: {out_dir} is not a directory")
    target = out_dir / LATENCY_PASS2_NAME
    if target.exists():
        raise SystemExit(f"--latency-only: {target} already exists (write-once; never overwritten)")
    rows_path = out_dir / "rows.npz" if args.rows is None else Path(args.rows)
    rec = read_pass1_record(out_dir)
    for name, got, want in (("--seed", int(args.seed), rec["seed"]), ("--n-queries", int(args.n_queries), rec["n_queries"]),
                            ("--warmup", int(args.warmup), rec["warmup"]),
                            ("--intra-threads", int(args.intra_threads), rec["intra_op_num_threads"])):
        if got != want:
            raise SystemExit(f"--latency-only: {name} {got} != pass 1's {want} (pass 2 must time the same way)")
    ck_sha = sha256_file(args.ckpt)
    if ck_sha != rec["checkpoint_sha256"]:
        raise SystemExit(f"--latency-only: --ckpt sha256 {ck_sha} != pass 1's record {rec['checkpoint_sha256']}")
    ident_before = assert_pass1_identity(out_dir, rows_path, rec)
    made = build_scoring_module(args.family, variant=args.variant, ckpt_path=args.ckpt, seed=args.seed, k=args.k,
                                catalogue=args.catalogue, v1_root=args.v1_root,
                                schema_id=getattr(args, "schema_id", None),
                                schema_registry=getattr(args, "schema_registry", None))
    scoring, built = made["scoring"], made["built"]
    if (built.family, built.variant, int(built.K)) != (rec["family"], rec["variant"], rec["K"]):
        raise SystemExit(f"--latency-only: built ({built.family}, {built.variant}, K={built.K}) != pass 1 "
                         f"({rec['family']}, {rec['variant']}, K={rec['K']})")
    rows = rows_to_batches(built, scoring, args.n_queries, seed=args.seed)          # the same synthetic queries as pass 1
    load_start = list(os.getloadavg()) if hasattr(os, "getloadavg") else None
    sess_fp32 = make_session(out_dir / "model_fp32.onnx", intra_op_num_threads=args.intra_threads)
    sess_int8 = make_session(out_dir / "model_int8.onnx", intra_op_num_threads=args.intra_threads)
    onnx_input_names = [i.name for i in sess_fp32.get_inputs()]
    numpy_rows = [{k: row[k].numpy() for k in onnx_input_names} for row in rows]
    latency_fp32 = bench_latency(sess_fp32, numpy_rows, warmup=args.warmup)
    latency_int8 = bench_latency(sess_int8, numpy_rows, warmup=args.warmup)
    load_end = list(os.getloadavg()) if hasattr(os, "getloadavg") else None
    ident_after = assert_pass1_identity(out_dir, rows_path, rec)                     # nothing under the pinned dir changed
    if ident_after != ident_before:
        raise SystemExit("--latency-only: a pinned pass-1 file changed during the timing")
    result = {"mode": "latency_only_pass2", "family": built.family, "variant": built.variant, "K": int(built.K),
              "seed": int(args.seed), "opset": OPSET, "intra_op_num_threads": int(args.intra_threads),
              "onnx_input_names": onnx_input_names, "onnxruntime": onnxruntime_info(args.intra_threads),
              "pinned_dir": str(out_dir),
              "pass1_record": {k: rec[k] for k in (
                  "pass1_record_path", "pass1_record_file_sha256", "model_sha256", "checkpoint_sha256", "rows_npz_sha256",
                  "rows_meta_sha256", "rows_first_users", "n_queries", "warmup", "intra_op_num_threads", "seed")},
              "asserted_sha256": ident_before, "checkpoint_sha256": ck_sha,
              "latency_fp32_ms": latency_fp32, "latency_int8_ms": latency_int8,
              "host": {"loadavg_start": load_start, "loadavg_end": load_end, "cpu_count": os.cpu_count()},
              "no_export_no_accuracy_no_rss": True,
              "note": ("DL 368 pass 2: this is the reported device latency (single-thread, quiet host); pass-1 latency "
                       "in BENCH_RESULT.json is 'loaded host' and is never used as the device number"),
              "wall_seconds": float(time.perf_counter() - t_start)}
    from v4train.checkpoint import write_json_atomic
    if target.exists():
        raise SystemExit(f"--latency-only: {target} appeared during the run (write-once)")
    result["result_sha256"] = write_json_atomic(target, result)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="DEV-01: export + quantize + bench the ONNX device protocol")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--family", required=True, choices=["GRU", "SASREC"])
    ap.add_argument("--variant", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--v1-root", default=None)
    ap.add_argument("--catalogue", default=None)
    ap.add_argument("--k", type=int, default=None, help="synthetic catalogue override (tests only, never on server)")
    ap.add_argument("--n-queries", type=int, default=DEFAULT_N_QUERIES, dest="n_queries")
    ap.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    ap.add_argument("--intra-threads", type=int, default=DEFAULT_INTRA_THREADS, dest="intra_threads")
    ap.add_argument("--rows", default=None, help="NPZ of materialized real decisions (module docstring)")
    ap.add_argument("--manifest-n", type=int, default=None, dest="manifest_n")
    ap.add_argument("--schema-id", default=None, dest="schema_id",
                    help="the registered schema id (optional; must equal the checkpoint's own record)")
    ap.add_argument("--schema-registry", default=None, dest="schema_registry",
                    help="derived-schema registry override (default: the one the run's RUN_START records)")
    ap.add_argument("--latency-only", action="store_true", dest="latency_only",
                    help="INT8 pass 2 (DL 368): time FP32 / INT8 single-thread latency ONLY on the pinned pass-1 artifact dir "
                         "given as --out (sha asserted against its BENCH_RESULT.json; no export, no accuracy); writes "
                         "<out>/LATENCY_PASS2.json, never overwrites a pass-1 file")
    ap.add_argument("--quiet", action="store_true", help="print only the timing / provenance summary, not the result")
    ap.add_argument("--tag", default="onnx_int8", help="INT8 ranks: evals/<tag>.ranks_int32.npy (the FP32 ONNX ranks of "
                                                       "the same rows are always written as evals/onnx_fp32.ranks_int32.npy)")
    args = ap.parse_args(argv)
    if args.latency_only:
        result = run_latency_only(args)
        print(json.dumps({"mode": result["mode"], "latency_fp32_ms": result["latency_fp32_ms"],
                          "latency_int8_ms": result["latency_int8_ms"], "host": result["host"],
                          "result_sha256": result["result_sha256"]}, indent=1))
        return 0
    result = run_pipeline(args)
    if args.quiet:
        fv = result.get("feature_view") or {}
        print(json.dumps({"wall_seconds": result["wall_seconds"], "schema_id": fv.get("schema_id"),
                          "input_keys": result["input_keys"], "onnx_input_names": result["onnx_input_names"],
                          "result_sha256": result["result_sha256"]}, indent=1))
    else:
        print(json.dumps(result, indent=1))
    return 0


__all__ = ["TEST_SPLIT", "load_rows_meta", "check_bench_rows", "peak_rss_bytes", "latency_stats", "parity_stats", "rows_to_batches", "load_rows_npz",
           "validate_rows_layout", "real_rows_from_npz", "ranks_from_session", "ranks_from_int8_session", "FP32_TAG", "make_session",
           "onnxruntime_info", "bench_latency", "bench_peak_rss_subprocess", "onnx_scores_batch",
           "pytorch_scores_batch", "production_scores_batch", "run_pipeline", "LATENCY_PASS2_NAME", "read_pass1_record",
           "assert_pass1_identity", "run_latency_only", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
