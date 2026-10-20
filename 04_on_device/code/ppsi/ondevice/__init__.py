"""On-device inference: export a sequence model to ONNX, INT8-quantise it, and benchmark it.

  scoring          the fixed-signature, single-decision scoring wrapper (BuiltModel -> ONNX-traceable nn.Module)
  export           deterministic ONNX opset-17 export (byte-identical re-export check)
  quantize         ONNX Runtime INT8 dynamic quantisation
  bench            onnxruntime CPU latency / peak RSS / parity bench, and INT8 retention rank arrays
  browser_queries  queries.json for the ONNX Runtime Web pages in `browser/`
  io               checkpoint loading, sha256 and atomic JSON helpers

onnx / onnxruntime are imported lazily inside the functions that need them.
"""
from __future__ import annotations

__all__: list = []
