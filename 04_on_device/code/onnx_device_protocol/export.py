"""v4dev.export — deterministic ONNX opset-17 export of a scoring checkpoint (DEV-01 item 1).

`export_model` exports the traced graph of a `v4dev.scoring.ScoringModule` with a fixed batch of 1 (no
`dynamic_axes`: every axis, including the window length, is the shape of the one traced example — "a fixed batch 1
inference signature", DEV-01 item 1) at opset 17, using the classic (non-dynamo) `torch.onnx` exporter for a stable,
reviewable graph. Determinism is verified, not assumed: every export re-exports to a sibling temp file and requires
byte-identical output before returning. Traces with every row full-length (`v4models.synthetic.L_MAX`, not the
varying real lengths `v4dev.bench`'s queries use): see `v4dev.scoring.example_inputs` for why (SASRec's
`_layout` bakes a data-dependent `T = lengths.max()` at trace time; tracing at the maximal T = L_MAX keeps the
graph valid for any shorter real length at inference). `input_names` given to the exporter is `scoring.input_keys`
in full, but the returned record's `input_names` is the graph's OWN, possibly smaller, retained set (see
`_graph_input_names`); GRU and SASRec each drop `attention_mask` (and SASRec `position_ids`) this way — both are
redundant with `lengths` in these models' own math, never read for anything but validation, so tracing correctly
never turns them into graph inputs.

Needs the `onnx` package (present on the SERVER; NOT in the local dev venv — `_onnx_export` raises ModuleNotFoundError
at the `import onnx` line locally, so local tests `pytest.importorskip("onnx")` before calling it).

CLI (Ops, on the server): `python -B -m v4dev.export --ckpt <path> --family GRU|SASREC [--variant NAME]
--v1-root <path-to-v1-mappings> --out <dir>` — writes `<out>/model_fp32.onnx` and `<out>/EXPORT_MANIFEST.json`
(strict JSON: family, variant, K, seed, init_sha256, checkpoint path/meta, the ONNX artifact record incl. sha256s).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import torch

from v4models.synthetic import L_MAX

from .checkpointio import build_scoring_module, sha256_file
from .scoring import ScoringModule, example_inputs

OPSET = 17


def _onnx_export(module: ScoringModule, example: Sequence[torch.Tensor], out_path: Path, *,
                 input_names: Sequence[str], output_names: Sequence[str]) -> None:
    """One `torch.onnx.export` call, opset 17, static shapes, the classic (non-dynamo) TorchScript-traced exporter
    when this torch build offers the `dynamo` switch (older builds have only the classic exporter)."""
    import onnx  # noqa: F401  (torch.onnx.export needs it to serialize the graph; surfaces a clear ImportError)
    kwargs = dict(input_names=list(input_names), output_names=list(output_names), opset_version=OPSET,
                  do_constant_folding=True, dynamic_axes=None, training=torch.onnx.TrainingMode.EVAL)
    try:
        torch.onnx.export(module, tuple(example), str(out_path), dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(module, tuple(example), str(out_path), **kwargs)


def _graph_input_names(onnx_path: Path) -> list:
    """The ONNX graph's OWN declared input names, read back after export. Not necessarily all of `scoring.input_keys`:
    a name whose tensor is used only for a static-shape or validation check (never a real op input feeding the
    output — e.g. `attention_mask` / `position_ids`, redundant with `lengths` in both families here) never becomes a
    graph node and is dropped by tracing, not by any optimization pass DEV-01 chose. This is the authoritative feed
    contract; `v4dev.bench` builds onnxruntime feeds from the session's own `get_inputs()`, not from this list, but
    it is recorded for the record."""
    import onnx
    m = onnx.load(str(onnx_path))
    return [i.name for i in m.graph.input]


def export_model(scoring: ScoringModule, example: Sequence[torch.Tensor], out_path: Path) -> dict:
    """Export `scoring` (eval mode) to `out_path`; re-export to a temp file and require byte-identical output
    (DEV-01 item 1: "Deterministic export. File sha256."). Returns the artifact record."""
    scoring.eval()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    declared = list(scoring.input_keys)
    names_out = ["scores"]
    _onnx_export(scoring, example, out_path, input_names=declared, output_names=names_out)
    digest = sha256_file(out_path)
    check_path = out_path.with_name(out_path.name + ".detcheck.tmp")
    try:
        _onnx_export(scoring, example, check_path, input_names=declared, output_names=names_out)
        digest2 = sha256_file(check_path)
    finally:
        if check_path.exists():
            check_path.unlink()
    if digest2 != digest:
        raise RuntimeError(f"non-deterministic ONNX export: {out_path} sha256 {digest} != re-export sha256 {digest2}")
    return {"path": str(out_path), "sha256": digest, "bytes": out_path.stat().st_size, "opset": OPSET,
            "declared_input_keys": declared, "input_names": _graph_input_names(out_path), "output_names": names_out,
            "K": int(scoring.K), "deterministic_reexport_sha256": digest2}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="DEV-01: export a scoring checkpoint to ONNX (opset 17, fixed batch 1)")
    ap.add_argument("--ckpt", required=True, help="checkpoint file (v4train record or a bare state dict)")
    ap.add_argument("--family", required=True, choices=["GRU", "SASREC"])
    ap.add_argument("--variant", default=None, help="v4models.variants name; default = the family's current build")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--seed", type=int, default=2026, help="module construction seed (overwritten by --ckpt weights)")
    ap.add_argument("--v1-root", default=None, help="the real, hash-checked v1 mapping files (server catalogue)")
    ap.add_argument("--catalogue", default=None, help="an explicit local catalogue.parquet (alternative to --v1-root)")
    ap.add_argument("--k", type=int, default=None, help="synthetic catalogue override (tests only, never on server)")
    ap.add_argument("--schema-id", default=None, dest="schema_id",
                    help="the registered schema id (optional; must equal the checkpoint's own record, DEV-04)")
    ap.add_argument("--schema-registry", default=None, dest="schema_registry",
                    help="derived-schema registry override (default: the one the run's RUN_START records)")
    args = ap.parse_args(argv)

    made = build_scoring_module(args.family, variant=args.variant, ckpt_path=args.ckpt, seed=args.seed, k=args.k,
                                catalogue=args.catalogue, v1_root=args.v1_root, schema_id=args.schema_id,
                                schema_registry=args.schema_registry)
    scoring, built = made["scoring"], made["built"]
    example = example_inputs(built, scoring, n=1, seed=args.seed, min_len=L_MAX)   # full-length trace (see scoring.py)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = export_model(scoring, example, out_dir / "model_fp32.onnx")
    manifest = {"family": built.family, "variant": built.variant, "K": built.K, "seed": int(args.seed),
                "init_sha256": built.init_sha256, "checkpoint": str(args.ckpt),
                "checkpoint_meta": made["checkpoint_meta"], "feature_view": made["feature_view"], "onnx": rec}
    from v4train.checkpoint import write_json_atomic
    write_json_atomic(out_dir / "EXPORT_MANIFEST.json", manifest)
    print(json.dumps(manifest, indent=1))
    return 0


__all__ = ["OPSET", "export_model", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
