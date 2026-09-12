"""Export the S2-DS-08 deployment candidate and record the evidence Phase 1 closes on.

Reproduce::

    uv run --locked python scripts/deployment/export_final_model.py \
        --report docs/evidence/s2-se-08/final_export.v1.json

Add `--weights <path>` to export the trained checkpoint. Its SHA-256 is checked against
the frozen contract before it is loaded, so the wrong file stops the export rather than
producing an artifact that looks correct.
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

from ppsi.deployment.final_export import (
    build_deployment_model,
    candidate_identity,
    load_deployment_candidate,
)
from ppsi.deployment.onnx_export import (
    OPSET_VERSION,
    check_graph,
    compare_against_onnx,
    deterministic_example_batch,
    export_session_gru,
    history_channels_of,
    serialized_size_bytes,
)
from ppsi.deployment.quantization import compare_fp32_and_int8, quantize_int8
from ppsi.models.batch_spec import phase1_batch_spec_v1

TOLERANCE = 1e-4

# Every axis the contract lets vary, on its own and together. The final export is the one
# that has to survive real traffic, so a single-shape check would be the weakest possible
# evidence for the strongest possible claim.
PARITY_SHAPES = (
    (4, 20, 100),
    (1, 20, 100),
    (2, 20, 100),
    (16, 20, 100),
    (4, 1, 100),
    (4, 20, 1),
    (4, 20, 20),
    (3, 7, 55),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export the Phase 1 deployment candidate")
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--keep", type=Path, default=None)
    parser.add_argument("--repetitions", type=int, default=30)
    args = parser.parse_args(argv)

    candidate, config = load_deployment_candidate()
    spec = phase1_batch_spec_v1()
    model = build_deployment_model(candidate, config, weights=args.weights, batch_spec=spec)

    batches = [
        deterministic_example_batch(spec, rows=r, history=h, candidates=c, seed=candidate.seed)
        for r, h, c in PARITY_SHAPES
    ]

    workspace = Path(args.keep) if args.keep else Path(tempfile.mkdtemp())
    workspace.mkdir(parents=True, exist_ok=True)
    fp32 = workspace / "phase1_final.fp32.onnx"
    int8 = workspace / "phase1_final.int8.onnx"

    export_session_gru(model, fp32, example_batch=batches[0], batch_spec=spec)
    parity = compare_against_onnx(model, fp32, batches, tolerance=TOLERANCE)
    quantize_int8(fp32, int8)
    quantization = compare_fp32_and_int8(
        fp32, int8, batches, repetitions=args.repetitions, channels=history_channels_of(model)
    )

    fp32_decision = (
        "INT8" if quantization.is_smaller_enough and quantization.is_not_slower else "FP32"
    )

    report = {
        "schema": "FinalDeploymentExport",
        "version": 1,
        "task_id": "S2-SE-08",
        "candidate": candidate_identity(candidate, weights_loaded=args.weights is not None),
        "parity": parity.as_dict(),
        "parity_shapes": [
            {"rows": r, "history": h, "candidates": c} for r, h, c in PARITY_SHAPES
        ],
        "heads_verified": list(parity.max_absolute_difference),
        "fp32": {"serialized_bytes": serialized_size_bytes(fp32)},
        "quantization": quantization.as_dict(),
        "recommended_artifact": fp32_decision,
        "recommendation_basis": (
            "Size and speed only. The per-task quality delta CP-DEPLOY's rule asks for needs "
            "the frozen evaluation set, which carries user_id and is not in this repository. "
            "Keep FP32 if that delta later exceeds 1% on any headline metric."
        ),
        "environment": {
            "onnxruntime": onnxruntime.__version__,
            "torch": torch.__version__,
            "opset_version": OPSET_VERSION,
            "provider": "CPUExecutionProvider",
            "machine": platform.machine(),
        },
        "graph": dict(check_graph(fp32)),
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(
        f"{candidate.checkpoint_id}: "
        f"{'trained weights' if args.weights else 'architecture only'}; "
        f"fp32 {report['fp32']['serialized_bytes']} B, int8 {quantization.int8_bytes} B "
        f"({quantization.size_ratio:.2f}x); worst parity {parity.worst_absolute:.3e} "
        f"across {len(PARITY_SHAPES)} shapes; recommend {fp32_decision}"
    )
    return 0 if parity.within_tolerance else 1


if __name__ == "__main__":
    raise SystemExit(main())
