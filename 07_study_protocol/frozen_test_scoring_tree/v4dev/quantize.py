"""v4dev.quantize — INT8 dynamic quantisation of the exported ONNX model (DEV-01 item 2).

Thin, server-only wrapper around `onnxruntime.quantization.quantize_dynamic` (weight-only dynamic INT8 — no
calibration data needed, matching the "INT8 dynamic quantisation" the task names). Records both artifacts' sizes and
sha256s (FP32 and INT8), for the size and parity reporting bench.py folds into its JSON.

Needs `onnxruntime` (present on the SERVER; NOT in the local dev venv — the `import` inside `quantize_int8` raises
ModuleNotFoundError locally, so local tests `pytest.importorskip("onnxruntime")` before calling it).

CLI: `python -B -m v4dev.quantize --fp32 <model_fp32.onnx> --out <dir>` — writes `<out>/model_int8.onnx` and prints
the size/sha256 record (also written to `<out>/QUANTIZE_MANIFEST.json`).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from .checkpointio import sha256_file


def _size_record(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def quantize_int8(fp32_path, int8_path) -> dict:
    """Dynamic INT8 quantisation of `fp32_path` (must exist) into `int8_path` (parent created if needed). Returns
    `{"fp32": {...}, "int8": {...}}` size/sha256 records."""
    from onnxruntime.quantization import QuantType, quantize_dynamic
    fp32_path, int8_path = Path(fp32_path), Path(int8_path)
    if not fp32_path.is_file():
        raise FileNotFoundError(f"no FP32 ONNX model at {fp32_path}")
    int8_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)
    return {"fp32": _size_record(fp32_path), "int8": _size_record(int8_path)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="DEV-01: INT8 dynamic quantisation of an exported ONNX scoring model")
    ap.add_argument("--fp32", required=True, help="the FP32 ONNX model (v4dev.export output)")
    ap.add_argument("--out", required=True, help="output directory (model_int8.onnx written here)")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = quantize_int8(args.fp32, out_dir / "model_int8.onnx")
    from v4train.checkpoint import write_json_atomic
    write_json_atomic(out_dir / "QUANTIZE_MANIFEST.json", rec)
    print(json.dumps(rec, indent=1))
    return 0


__all__ = ["quantize_int8", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
