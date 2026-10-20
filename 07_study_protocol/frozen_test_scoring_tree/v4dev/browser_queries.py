"""v4dev.browser_queries — build `queries.json` for the DEV-03 browser latency page (`v4dev/browser/`) from a
`--rows` NPZ (DEV-02's `dump_rows.py`; the same NPZ layout `v4dev.bench.load_rows_npz` / `validate_rows_layout`
accept: one array per model input key, plus the three meta fields `decision_id` / `target_class` / `manifest_row`).

CLI (Ops / owner, local; no onnx / onnxruntime import anywhere in this module -- only torch/numpy, same as the rest
of `v4dev` locally, per `v4dev.bench`'s module docstring):
    python -B -m v4dev.browser_queries --rows <rows.npz> --out <dir>/queries.json [--n-queries 1000]
        [--ckpt <path> --family GRU|SASREC [--variant NAME] [--v1-root <path> | --catalogue <path>] [--seed N] [--k N]]

Always: the first `--n-queries` (default 1,000, DEV-03 packet) rows of the NPZ's model-input arrays, plus each
array's ONNX tensor dtype and per-row shape (`shapes[key]`; the page adds its own leading batch-of-1 dimension), so
`v4dev/browser/bench.js` can build an `ort.Tensor` for any key without guessing a dtype from its name.

Model-input selection (`select_input_keys`): with `--ckpt` + `--family` given, exactly `scoring.input_keys` for THAT
built model (the correct, minimal set -- mirrors `v4dev.export`/`v4dev.bench`'s own `build_scoring_module` call).
Without a checkpoint (DEV-02's rows dump and a trained checkpoint are separate, parallel tasks; a queries.json is
still useful for a latency-only run before one exists), every NPZ array except the three meta fields -- a safe
superset for either family, since `v4dev/browser/bench.js` narrows the feed to the ONNX session's OWN declared
`session.inputNames` before every `run()` call, exactly as `v4dev.bench.run_pipeline`'s `onnx_input_names` narrowing
does (harmless extra keys, never a missing one).

Provenance (REV-ANA-01 D-F3): queries.json carries a `provenance` block -- `rows_npz_sha256` (the --rows file used),
`rows_meta_sha256` (rows.meta.json beside it, when present), `ckpt_sha256` (the --ckpt file, or BENCH_RESULT's) and, with
`--bench-result <BENCH_RESULT.json>`, `model_fp32_sha256` / `model_int8_sha256` / `bench_result_sha256`. `bench.js`
refuses a model file whose sha256 differs from the `model_*_sha256` here. Given both --ckpt and --bench-result, the
checkpoint sha256 must equal BENCH_RESULT's `checkpoint_sha256` (else refused). `rows_npz_sha256` binds the file used
(np.savez zip timestamps make it non-reproducible across regenerations of identical content).

Reference top-20 ("the PyTorch/ONNX CPU path when available", DEV-03 packet): with `--ckpt` given, the first `n`
rows are scored by `v4dev.bench.pytorch_scores_batch` (the ONNX-traceable wrapper -- what was actually exported;
runs on CPU, needs no onnxruntime/onnx import) and the top-`--top-k` (default 20) argsort per row is recorded as
`reference.top20`; `reference.available` is `false` and `top20` is `null` otherwise -- the browser page's parity
metric is then skipped (there is no ground truth to compare against).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from .bench import DEFAULT_TOP_K, load_rows_npz, pytorch_scores_batch
from .checkpointio import build_scoring_module, sha256_file

#: the three fields `v4dev.bench.load_rows_npz` requires that are never a model input (decision label / bookkeeping).
META_KEYS = ("decision_id", "target_class", "manifest_row")

#: numpy dtype name -> the ONNX Runtime Web tensor type string `bench.js` builds a typed array from. Kept explicit
#: (not "whatever str(dtype) happens to spell") so an unsupported dtype is refused here, not silently mis-fed in JS.
_ONNX_DTYPE = {"int64": "int64", "int32": "int32", "int16": "int16", "int8": "int8", "uint8": "uint8",
              "uint16": "uint16", "uint32": "uint32", "uint64": "uint64", "float32": "float32",
              "float64": "float64", "bool": "bool"}


class BrowserQueriesError(ValueError):
    pass


def _onnx_dtype(arr: np.ndarray) -> str:
    name = str(np.asarray(arr).dtype)
    if name not in _ONNX_DTYPE:
        raise BrowserQueriesError(f"unsupported numpy dtype {name!r} for an ONNX Runtime Web feed "
                                  f"(add it to browser_queries._ONNX_DTYPE only once bench.js also handles it)")
    return _ONNX_DTYPE[name]


def select_input_keys(rows_data: Mapping[str, np.ndarray], input_keys: Optional[Sequence[str]]) -> list:
    """The model-input array names to carry into queries.json (module docstring)."""
    if input_keys is not None:
        missing = [k for k in input_keys if k not in rows_data]
        if missing:
            raise BrowserQueriesError(f"--rows NPZ is missing required input keys for this model: {missing}")
        return list(input_keys)
    keys = [k for k in rows_data.keys() if k not in META_KEYS]
    if not keys:
        raise BrowserQueriesError("--rows NPZ has no model-input arrays (only the meta fields are present)")
    return keys


def resolve_n_rows(rows_data: Mapping[str, np.ndarray], keys: Sequence[str], n_queries: int) -> int:
    """`min(n_queries, rows available)`, after checking every selected key has the same row count (a mismatched NPZ
    is refused here, not silently truncated to the shortest array)."""
    n_total = int(np.asarray(rows_data[keys[0]]).shape[0])
    for k in keys:
        n_k = int(np.asarray(rows_data[k]).shape[0])
        if n_k != n_total:
            raise BrowserQueriesError(f"--rows arrays have mismatched row counts: {keys[0]}={n_total}, {k}={n_k}")
    n = min(int(n_queries), n_total)
    if n <= 0:
        raise BrowserQueriesError("no rows to write (n_queries <= 0, or the --rows NPZ is empty)")
    return n


def reference_top20(rows_data: Mapping[str, np.ndarray], scoring, input_keys: Sequence[str], n: int, *,
                    top_k: int = DEFAULT_TOP_K) -> dict:
    """Reference top-`top_k` item indices for the first `n` rows via `v4dev.bench.pytorch_scores_batch` (module
    docstring: the exported ONNX graph's own PyTorch source, CPU, no onnxruntime needed)."""
    import torch
    rows = [{k: torch.as_tensor(np.asarray(rows_data[k])[i:i + 1]) for k in input_keys} for i in range(n)]
    scores = pytorch_scores_batch(scoring, rows)
    if scores.shape[1] < top_k:
        raise BrowserQueriesError(f"top_k={top_k} exceeds the catalogue size {scores.shape[1]}")
    top = np.argsort(-scores, axis=1, kind="stable")[:, :top_k]
    return {"available": True, "source": "pytorch_cpu_wrapper", "top_k": int(top_k), "top20": top.tolist()}


def build_provenance(rows_path, *, ckpt_path=None, bench_result_path=None) -> dict:
    """The `provenance` block (module docstring). Missing pieces are simply absent (bench.js then records that the
    model sha was not verified); a checkpoint / BENCH_RESULT disagreement is refused."""
    prov: dict = {"rows_npz_sha256": sha256_file(rows_path)}
    meta = Path(rows_path).parent / "rows.meta.json"
    if meta.is_file():
        prov["rows_meta_sha256"] = sha256_file(meta)
    if ckpt_path is not None:
        prov["ckpt_sha256"] = sha256_file(ckpt_path)
    if bench_result_path is not None:
        br = json.loads(Path(bench_result_path).read_text(encoding="utf-8"))
        models = br.get("model_sha256") or {}
        for key in ("fp32", "int8"):
            if isinstance(models.get(key), str) and models[key]:
                prov[f"model_{key}_sha256"] = models[key]
        br_ckpt = br.get("checkpoint_sha256")
        if br_ckpt:
            if "ckpt_sha256" in prov and prov["ckpt_sha256"] != br_ckpt:
                raise BrowserQueriesError(f"--ckpt sha256 {prov['ckpt_sha256']} != BENCH_RESULT checkpoint_sha256 "
                                          f"{br_ckpt}: the bench was run on a different checkpoint")
            prov["ckpt_sha256"] = br_ckpt
        if not models:
            raise BrowserQueriesError("BENCH_RESULT.json has no `model_sha256` block (bench run before DEV-04?)")
        prov["bench_result_sha256"] = sha256_file(bench_result_path)
    return prov


def build_queries(rows_data: Mapping[str, np.ndarray], keys: Sequence[str], n: int, *,
                  reference: Optional[dict] = None, provenance: Optional[dict] = None) -> dict:
    """The full `queries.json` document (module docstring): `n` rows of every array in `keys`, plus dtype/shape
    metadata and (optionally) the reference top-20."""
    inputs, dtypes, shapes = {}, {}, {}
    for k in keys:
        arr = np.asarray(rows_data[k])[:n]
        dtypes[k] = _onnx_dtype(arr)
        shapes[k] = list(arr.shape[1:])          # per-row shape; the page adds its own leading batch-of-1 dim
        inputs[k] = arr.tolist()
    doc = {"schema": "v4dev.browser_queries.v1", "n_rows": int(n), "input_keys": list(keys),
          "dtypes": dtypes, "shapes": shapes, "inputs": inputs, "provenance": dict(provenance or {}),
          "reference": reference if reference is not None else
                       {"available": False, "source": None, "top_k": DEFAULT_TOP_K, "top20": None}}
    return doc


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="DEV-03: build queries.json for v4dev/browser from a --rows NPZ")
    ap.add_argument("--rows", required=True, help="the DEV-02 rows NPZ (v4dev.bench's --rows layout)")
    ap.add_argument("--out", required=True, help="output queries.json path")
    ap.add_argument("--n-queries", type=int, default=1000, dest="n_queries")
    ap.add_argument("--ckpt", default=None, help="checkpoint for the reference top-20 (optional; PyTorch CPU path)")
    ap.add_argument("--family", default=None, choices=["GRU", "SASREC"])
    ap.add_argument("--variant", default=None)
    ap.add_argument("--v1-root", default=None)
    ap.add_argument("--catalogue", default=None)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--k", type=int, default=None, help="synthetic catalogue override (tests only, never on server)")
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, dest="top_k")
    ap.add_argument("--schema-registry", default=None, dest="schema_registry",
                    help="derived-schema registry override (default: the one the checkpoint run's RUN_START records)")
    ap.add_argument("--bench-result", default=None, dest="bench_result",
                    help="the v4dev.bench BENCH_RESULT.json whose model shas bench.js must match (recommended)")
    args = ap.parse_args(argv)

    if (args.ckpt is None) != (args.family is None):
        raise SystemExit("--ckpt and --family must be given together (or neither)")

    rows_data = load_rows_npz(args.rows)

    scoring = None
    input_keys = None
    if args.ckpt is not None:
        made = build_scoring_module(args.family, variant=args.variant, ckpt_path=args.ckpt, seed=args.seed,
                                    k=args.k, catalogue=args.catalogue, v1_root=args.v1_root,
                                    schema_registry=args.schema_registry)   # DEV-04: the checkpoint's own view
        scoring = made["scoring"]
        input_keys = list(scoring.input_keys)

    keys = select_input_keys(rows_data, input_keys)
    n = resolve_n_rows(rows_data, keys, args.n_queries)

    reference = None
    if scoring is not None:
        reference = reference_top20(rows_data, scoring, keys, n, top_k=args.top_k)

    provenance = build_provenance(args.rows, ckpt_path=args.ckpt, bench_result_path=args.bench_result)
    doc = build_queries(rows_data, keys, n, reference=reference, provenance=provenance)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    from v4train.checkpoint import write_json_atomic
    sha = write_json_atomic(out_path, doc)
    print(json.dumps({"path": str(out_path), "n_rows": doc["n_rows"], "input_keys": doc["input_keys"],
                      "reference_available": doc["reference"]["available"], "provenance": doc["provenance"],
                      "sha256": sha}, indent=1))
    return 0


__all__ = ["BrowserQueriesError", "META_KEYS", "select_input_keys", "resolve_n_rows", "reference_top20",
           "build_provenance", "build_queries", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
