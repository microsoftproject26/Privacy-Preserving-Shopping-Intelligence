"""Quantize the exported model to INT8 and measure whether it is worth deploying.

Reproduce::

    uv run --locked python scripts/deployment/benchmark_int8.py \
        --report docs/evidence/s2-se-02/int8_benchmark.v1.json

Produces the three numbers CP-DEPLOY's INT8 rule asks for: size ratio, CPU latency, and
how far the outputs move. It does not decide; the report states what was measured and
which part of the rule cannot be checked without the frozen evaluation set.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import onnxruntime
import torch

from ppsi.deployment.onnx_export import deterministic_example_batch, export_session_gru
from ppsi.deployment.quantization import compare_fp32_and_int8, quantize_int8
from ppsi.models.batch_spec import phase1_batch_spec_v1
from ppsi.models.session_gru import build_model

# One shape for latency so the two models are timed identically, several for agreement so
# a single well-behaved shape cannot hide a badly-behaved one.
AGREEMENT_SHAPES = ((4, 6, 5), (1, 6, 5), (8, 12, 20), (4, 1, 5))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark INT8 against FP32")
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--keep", type=Path, default=None, help="Directory to keep both models in")
    args = parser.parse_args(argv)

    spec = phase1_batch_spec_v1()
    model = build_model(args.seed, batch_spec=spec).eval()
    batches = [
        deterministic_example_batch(spec, rows=r, history=h, candidates=c, seed=args.seed)
        for r, h, c in AGREEMENT_SHAPES
    ]

    workspace = Path(args.keep) if args.keep else Path(tempfile.mkdtemp())
    workspace.mkdir(parents=True, exist_ok=True)
    fp32 = workspace / "session_gru.fp32.onnx"
    int8 = workspace / "session_gru.int8.onnx"

    export_session_gru(model, fp32, example_batch=batches[0], batch_spec=spec)
    quantize_int8(fp32, int8)

    comparison = compare_fp32_and_int8(fp32, int8, batches, repetitions=args.repetitions)

    report = comparison.as_dict()
    report["seed"] = args.seed
    report["shapes_compared"] = [
        {"rows": r, "history": h, "candidates": c} for r, h, c in AGREEMENT_SHAPES
    ]
    report["environment"] = {
        "onnxruntime": onnxruntime.__version__,
        "torch": torch.__version__,
        "provider": "CPUExecutionProvider",
        "machine": platform.machine(),
        "processor_note": "latency is machine-specific; the ratio is the portable figure",
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(
        f"fp32 {comparison.fp32_bytes} B at {comparison.fp32_latency['median_ms']:.2f} ms; "
        f"int8 {comparison.int8_bytes} B at {comparison.int8_latency['median_ms']:.2f} ms; "
        f"size ratio {comparison.size_ratio:.2f}x; "
        f"size rule {'met' if comparison.is_smaller_enough else 'not met'}; "
        f"speed rule {'met' if comparison.is_not_slower else 'not met'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
