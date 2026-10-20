"""Build `queries.json` for the browser latency page (`ppsi/ondevice/browser/`).

From a rows NPZ (the layout `bench.load_rows_npz` / `validate_rows_layout` accept: one array per model input key,
plus `decision_id` / `target_class` / `manifest_row`) or, without one, from synthetic decisions of the given model:
    python -m ppsi.ondevice.browser_queries --out DIR/queries.json [--rows rows.npz] [--n-queries 1000]
        [--family GRU|SASREC [--variant NAME] (--k K | --catalogue PATH) [--ckpt PATH]] [--bench-result PATH]

The file carries the first `--n-queries` rows of every model-input array plus each array's ONNX tensor dtype and
per-row shape (`shapes[key]`; the page adds its own leading batch-of-1 dimension), so `browser/bench.js` can build
an `ort.Tensor` for any key without guessing a dtype from its name.

Model-input selection (`select_input_keys`): with a model given (`--family`), exactly that model's
`scoring.input_keys`; without one, every NPZ array except the three meta fields (a safe superset: the page narrows
the feed to the ONNX session's own `inputNames` before every run).

Provenance: `rows_npz_sha256` (when rows are given), `ckpt_sha256` (with `--ckpt`) and, with `--bench-result`, the
`model_fp32_sha256` / `model_int8_sha256` the bench recorded; the page refuses a model file whose sha256 differs.

Reference top-20: with a model given, the first `n` rows are scored by `bench.pytorch_scores_batch` (the exported
graph's own PyTorch source, CPU) and the top-`--top-k` per row is recorded as `reference.top20`; otherwise
`reference.available` is false and the page skips its parity metric.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from .bench import DEFAULT_TOP_K, load_rows_npz, pytorch_scores_batch
from .export import add_model_args, model_from_args
from .io import sha256_file, write_json_atomic
from .scoring import example_batch

#: the three fields `bench.load_rows_npz` requires that are never a model input
META_KEYS = ("decision_id", "target_class", "manifest_row")

#: numpy dtype name -> the ONNX Runtime Web tensor type string `bench.js` builds a typed array from. Explicit, so an
#: unsupported dtype is refused here instead of being mis-fed in JS.
_ONNX_DTYPE = {"int64": "int64", "int32": "int32", "int16": "int16", "int8": "int8", "uint8": "uint8",
               "uint16": "uint16", "uint32": "uint32", "uint64": "uint64", "float32": "float32",
               "float64": "float64", "bool": "bool"}


class BrowserQueriesError(ValueError):
    pass


def _onnx_dtype(arr: np.ndarray) -> str:
    name = str(np.asarray(arr).dtype)
    if name not in _ONNX_DTYPE:
        raise BrowserQueriesError(f"unsupported numpy dtype {name!r} for an ONNX Runtime Web feed "
                                  f"(add it to _ONNX_DTYPE only once bench.js also handles it)")
    return _ONNX_DTYPE[name]


def select_input_keys(rows_data: Mapping[str, np.ndarray], input_keys: Sequence[str] | None) -> list:
    """The model-input array names to carry into queries.json (module docstring)."""
    if input_keys is not None:
        missing = [k for k in input_keys if k not in rows_data]
        if missing:
            raise BrowserQueriesError(f"the rows are missing required input keys for this model: {missing}")
        return list(input_keys)
    keys = [k for k in rows_data if k not in META_KEYS]
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
    """Reference top-`top_k` item indices for the first `n` rows via `bench.pytorch_scores_batch` (the exported
    graph's own PyTorch source, CPU, no onnxruntime needed)."""
    import torch
    rows = [{k: torch.as_tensor(np.asarray(rows_data[k])[i:i + 1]) for k in input_keys} for i in range(n)]
    scores = pytorch_scores_batch(scoring, rows)
    if scores.shape[1] < top_k:
        raise BrowserQueriesError(f"top_k={top_k} exceeds the catalogue size {scores.shape[1]}")
    top = np.argsort(-scores, axis=1, kind="stable")[:, :top_k]
    return {"available": True, "source": "pytorch_cpu_wrapper", "top_k": int(top_k), "top20": top.tolist()}


def build_provenance(*, rows_path=None, ckpt_path=None, bench_result_path=None) -> dict:
    """The `provenance` block (module docstring). Missing pieces are simply absent (the page then records that the
    model sha256 was not verified); a checkpoint / bench-result disagreement is refused."""
    prov: dict = {}
    if rows_path is not None:
        prov["rows_npz_sha256"] = sha256_file(rows_path)
    if ckpt_path is not None:
        prov["ckpt_sha256"] = sha256_file(ckpt_path)
    if bench_result_path is not None:
        br = json.loads(Path(bench_result_path).read_text(encoding="utf-8"))
        models = br.get("model_sha256") or {}
        if not models:
            raise BrowserQueriesError("the bench result has no `model_sha256` block")
        for key in ("fp32", "int8"):
            if isinstance(models.get(key), str) and models[key]:
                prov[f"model_{key}_sha256"] = models[key]
        br_ckpt = br.get("checkpoint_sha256")
        if br_ckpt:
            if "ckpt_sha256" in prov and prov["ckpt_sha256"] != br_ckpt:
                raise BrowserQueriesError(f"--ckpt sha256 {prov['ckpt_sha256']} != the bench result's "
                                          f"checkpoint_sha256 {br_ckpt}: the bench ran on a different checkpoint")
            prov["ckpt_sha256"] = br_ckpt
        prov["bench_result_sha256"] = sha256_file(bench_result_path)
    return prov


def build_queries(rows_data: Mapping[str, np.ndarray], keys: Sequence[str], n: int, *,
                  reference: dict | None = None, provenance: dict | None = None) -> dict:
    """The full `queries.json` document (module docstring): `n` rows of every array in `keys`, plus dtype/shape
    metadata and (optionally) the reference top-20."""
    inputs, dtypes, shapes = {}, {}, {}
    for k in keys:
        arr = np.asarray(rows_data[k])[:n]
        dtypes[k] = _onnx_dtype(arr)
        shapes[k] = list(arr.shape[1:])          # per-row shape; the page adds its own leading batch-of-1 dim
        inputs[k] = arr.tolist()
    doc = {"schema": "ppsi.ondevice.browser_queries.v1", "n_rows": int(n), "input_keys": list(keys),
          "dtypes": dtypes, "shapes": shapes, "inputs": inputs, "provenance": dict(provenance or {}),
          "reference": reference if reference is not None else
                       {"available": False, "source": None, "top_k": DEFAULT_TOP_K, "top20": None}}
    return doc


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="build queries.json for the browser latency page")
    ap.add_argument("--out", required=True, help="output queries.json path")
    ap.add_argument("--rows", default=None, help="a rows NPZ (bench --rows layout); default: synthetic decisions")
    ap.add_argument("--n-queries", type=int, default=1000, dest="n_queries")
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, dest="top_k")
    ap.add_argument("--bench-result", default=None, dest="bench_result",
                    help="the bench's BENCH_RESULT.json whose model sha256s the page must match (recommended)")
    add_model_args(ap, family_required=False)
    args = ap.parse_args(argv)
    if args.ckpt is not None and args.family is None:
        raise SystemExit("--ckpt needs --family (and the model options that built it)")

    scoring = built = None
    input_keys = None
    if args.family is not None:
        made = model_from_args(args)
        scoring, built = made["scoring"], made["built"]
        input_keys = list(scoring.input_keys)
    if args.rows is not None:
        rows_data = load_rows_npz(args.rows)
    elif built is not None:
        rows_data = {k: v.numpy() for k, v in example_batch(built, args.n_queries, seed=args.seed).items()
                     if hasattr(v, "numpy")}
    else:
        raise SystemExit("give --rows, or a model (--family ...) to draw synthetic decisions from")

    keys = select_input_keys(rows_data, input_keys)
    n = resolve_n_rows(rows_data, keys, args.n_queries)
    reference = reference_top20(rows_data, scoring, keys, n, top_k=args.top_k) if scoring is not None else None
    provenance = build_provenance(rows_path=args.rows, ckpt_path=args.ckpt, bench_result_path=args.bench_result)
    doc = build_queries(rows_data, keys, n, reference=reference, provenance=provenance)
    out_path = Path(args.out)
    sha = write_json_atomic(out_path, doc)
    print(json.dumps({"path": str(out_path), "n_rows": doc["n_rows"], "input_keys": doc["input_keys"],
                      "reference_available": doc["reference"]["available"], "provenance": doc["provenance"],
                      "sha256": sha}, indent=1))
    return 0


__all__ = ["META_KEYS", "BrowserQueriesError", "build_provenance", "build_queries", "main", "reference_top20",
           "resolve_n_rows", "select_input_keys"]

if __name__ == "__main__":
    raise SystemExit(main())
