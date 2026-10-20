"""v4dev — the on-device ONNX protocol (DEV-01; contracts/PREG1_AMENDMENT_04.md §3, company/reviews/
DOWNSTREAM_FINAL_PLAN_FABLE.md (2)): export a scoring model to ONNX, INT8-quantise it, and bench it under
onnxruntime CPU as a disclosed server-CPU proxy for ONNX Runtime Web.

Modules:
  scoring    the fixed-signature, single-decision K-score wrapper (v4models BuiltModel -> ONNX-traceable nn.Module).
  checkpointio  read-only checkpoint loading (v4train checkpoint records and plain state-dict files).
  export     deterministic ONNX opset-17 export (torch.onnx; needs the `onnx` package).
  quantize   ONNX Runtime INT8 dynamic quantisation (onnxruntime.quantization; server-only).
  bench      onnxruntime CPU latency/RSS/parity bench + the INT8 retention rank-array write (server-only).

onnxruntime and the `onnx` package are installed on the SERVER, not in the local dev venv: every module that touches
them imports lazily inside the function that needs them, so `import v4dev.*` and every non-runtime helper (input-key
derivation, sha256, percentile/rank-array logic, checkpoint parsing) work locally with plain torch + numpy.
"""
from __future__ import annotations

__all__: list = []
