"""Deterministic ONNX opset-17 export of a scoring model.

`export_model` exports the traced graph of a `ScoringModule` with a fixed batch of 1 (no `dynamic_axes`: every axis,
including the window length, is the shape of the one traced example) at opset 17, using the classic (non-dynamo)
`torch.onnx` exporter for a stable, reviewable graph. Determinism is verified, not assumed: every export re-exports
to a sibling temp file and requires byte-identical output before returning. The trace uses full-length rows (see
`scoring.example_inputs`). The record's `input_names` is the graph's OWN retained input set: tracing drops inputs
that are only used for validation (GRU and SASRec drop `attention_mask`, SASRec also `position_ids`; both are
redundant with `lengths` in these models' math).

CLI: `python -m ppsi.ondevice.export --family GRU|SASREC [--variant NAME] (--k K | --catalogue PATH)
[--ckpt PATH] [--widths 15,15,12,4] --out DIR` writes `DIR/model_fp32.onnx` and `DIR/EXPORT_MANIFEST.json`.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import torch

from ppsi.seqrec.synthetic import L_MAX

from .io import build_scoring_module, parse_widths, sha256_file, write_json_atomic
from .scoring import ScoringModule, example_inputs

OPSET = 17


def _onnx_export(module: ScoringModule, example: Sequence[torch.Tensor], out_path: Path, *,
                 input_names: Sequence[str], output_names: Sequence[str]) -> None:
    """One `torch.onnx.export` call, opset 17, static shapes, the classic TorchScript-traced exporter when this torch
    build offers the `dynamo` switch (older builds have only the classic exporter)."""
    import onnx  # noqa: F401  (torch.onnx.export needs it to serialise the graph; surfaces a clear ImportError)
    kwargs = {"input_names": list(input_names), "output_names": list(output_names), "opset_version": OPSET,
              "do_constant_folding": True, "dynamic_axes": None, "training": torch.onnx.TrainingMode.EVAL}
    try:
        torch.onnx.export(module, tuple(example), str(out_path), dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(module, tuple(example), str(out_path), **kwargs)


def _graph_input_names(onnx_path: Path) -> list:
    """The ONNX graph's OWN declared input names, read back after export (the authoritative feed contract)."""
    import onnx
    m = onnx.load(str(onnx_path))
    return [i.name for i in m.graph.input]


def export_model(scoring: ScoringModule, example: Sequence[torch.Tensor], out_path: Path) -> dict:
    """Export `scoring` (eval mode) to `out_path`; re-export to a temp file and require byte-identical output.
    Returns the artifact record."""
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
            "declared_input_keys": declared, "input_names": _graph_input_names(out_path),
            "output_names": names_out, "K": int(scoring.K), "deterministic_reexport_sha256": digest2}


def add_model_args(ap: argparse.ArgumentParser, *, family_required: bool = True) -> None:
    ap.add_argument("--family", required=family_required, default=None, choices=["GRU", "SASREC"])
    ap.add_argument("--variant", default=None, help="ppsi.seqrec.variants name; default = the family's default size")
    ap.add_argument("--ckpt", default=None, help="state-dict file (bare, or {'model': ..., 'meta': ...}); "
                                                 "default: a fresh seeded init")
    ap.add_argument("--seed", type=int, default=2026, help="module construction seed")
    ap.add_argument("--widths", default="15,15,12,4", help="numeric,flags,user_context,user_mask")
    ap.add_argument("--k", type=int, default=None, help="catalogue size (classes arange(3, 3+K))")
    ap.add_argument("--catalogue", default=None, help="a local catalogue.parquet with a product_idx column")


def model_from_args(args: argparse.Namespace) -> dict:
    if (args.k is None) == (args.catalogue is None):
        raise SystemExit("give exactly one of --k or --catalogue")
    return build_scoring_module(args.family, widths=parse_widths(args.widths), variant=args.variant,
                                ckpt_path=args.ckpt, seed=args.seed, k=args.k, catalogue=args.catalogue)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="export a scoring model to ONNX (opset 17, fixed batch 1)")
    add_model_args(ap)
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args(argv)
    made = model_from_args(args)
    scoring, built = made["scoring"], made["built"]
    example = example_inputs(built, scoring, n=1, seed=args.seed, min_len=L_MAX)   # full-length trace
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = export_model(scoring, example, out_dir / "model_fp32.onnx")
    manifest = {"family": built.family, "variant": built.variant, "K": built.K, "seed": int(args.seed),
                "init_sha256": built.init_sha256, "checkpoint": args.ckpt,
                "checkpoint_meta": made["checkpoint_meta"], "widths": built.widths.as_json(), "onnx": rec}
    write_json_atomic(out_dir / "EXPORT_MANIFEST.json", manifest)
    print(json.dumps(manifest, indent=1))
    return 0


__all__ = ["OPSET", "add_model_args", "export_model", "main", "model_from_args"]

if __name__ == "__main__":
    raise SystemExit(main())
