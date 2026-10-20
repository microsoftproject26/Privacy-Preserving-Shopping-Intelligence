"""INT8 dynamic quantisation of an exported ONNX model.

Thin wrapper around `onnxruntime.quantization.quantize_dynamic` (weight-only dynamic INT8: no calibration data is
needed). Records both artifacts' sizes and sha256s (FP32 and INT8) for the size and parity reporting of the bench.

CLI: `python -m ppsi.ondevice.quantize --fp32 model_fp32.onnx --out DIR` writes `DIR/model_int8.onnx` and
`DIR/QUANTIZE_MANIFEST.json`.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .io import sha256_file, write_json_atomic


def _size_record(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def quantize_int8(fp32_path, int8_path) -> dict:
    """Dynamic INT8 quantisation of `fp32_path` (must exist) into `int8_path` (parent created if needed). Returns
    `{"fp32": {...}, "int8": {...}}` size / sha256 records."""
    from onnxruntime.quantization import QuantType, quantize_dynamic
    fp32_path, int8_path = Path(fp32_path), Path(int8_path)
    if not fp32_path.is_file():
        raise FileNotFoundError(f"no FP32 ONNX model at {fp32_path}")
    int8_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)
    return {"fp32": _size_record(fp32_path), "int8": _size_record(int8_path)}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="INT8 dynamic quantisation of an exported ONNX scoring model")
    ap.add_argument("--fp32", required=True, help="the FP32 ONNX model (ppsi.ondevice.export output)")
    ap.add_argument("--out", required=True, help="output directory (model_int8.onnx is written here)")
    args = ap.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = quantize_int8(args.fp32, out_dir / "model_int8.onnx")
    write_json_atomic(out_dir / "QUANTIZE_MANIFEST.json", rec)
    print(json.dumps(rec, indent=1))
    return 0


__all__ = ["main", "quantize_int8"]

if __name__ == "__main__":
    raise SystemExit(main())
