"""Export a Phase 1 model to ONNX and write the parity evidence.

Reproduce::

    uv run --locked python scripts/deployment/export_onnx.py \
        --output artifacts/onnx/session_gru.onnx \
        --report docs/evidence/s2-se-01/onnx_parity.v1.json

The ONNX binary itself is not committed. It is reproducible from a seed and a checkpoint,
and a 14 MB binary in git would be a copy of something the command regenerates exactly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import onnxruntime
import torch

from ppsi.deployment.onnx_export import (
    OPSET_VERSION,
    check_graph,
    compare_against_onnx,
    deterministic_example_batch,
    export_session_gru,
    serialized_size_bytes,
)
from ppsi.models.batch_spec import phase1_batch_spec_v1
from ppsi.models.checkpoint import load_encoder
from ppsi.models.session_gru import build_model

# Shapes the parity sweep covers. Batch size, history length and candidate width are the
# three axes declared dynamic, so each is varied on its own and then together. An export
# that only works at the shape it was traced on passes a single-shape check and fails in
# deployment.
PARITY_SHAPES = (
    (4, 6, 5),
    (1, 6, 5),
    (2, 6, 5),
    (7, 6, 5),
    (16, 6, 5),
    (4, 1, 5),
    (4, 12, 5),
    (4, 6, 1),
    (4, 6, 20),
    (3, 12, 20),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a Phase 1 model to ONNX")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Trained encoder to export. Without it the model is built from --seed alone, "
        "which proves the export path but is not a deployment candidate.",
    )
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args(argv)

    spec = phase1_batch_spec_v1()
    model = build_model(args.seed, batch_spec=spec)
    if args.checkpoint is not None:
        load_encoder(model, args.checkpoint)
    model.eval()

    example = deterministic_example_batch(spec, rows=4, history=6, candidates=5, seed=args.seed)
    export_session_gru(model, args.output, example_batch=example, batch_spec=spec)

    batches = [
        deterministic_example_batch(
            spec, rows=rows, history=history, candidates=candidates, seed=args.seed
        )
        for rows, history, candidates in PARITY_SHAPES
    ]
    parity = compare_against_onnx(model, args.output, batches, tolerance=args.tolerance)

    report = parity.as_dict()
    report["graph"] = dict(check_graph(args.output))
    report["serialized_bytes"] = serialized_size_bytes(args.output)
    report["shapes_checked"] = [
        {"rows": rows, "history": history, "candidates": candidates}
        for rows, history, candidates in PARITY_SHAPES
    ]
    report["runtime"] = {
        "onnxruntime": onnxruntime.__version__,
        "torch": torch.__version__,
        "provider": "CPUExecutionProvider",
        "opset_version": OPSET_VERSION,
    }
    report["checkpoint"] = str(args.checkpoint) if args.checkpoint else None
    report["seed"] = args.seed

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(
        f"exported {report['serialized_bytes']} bytes; worst absolute difference "
        f"{parity.worst_absolute:.3e} across {len(PARITY_SHAPES)} shapes "
        f"(tolerance {args.tolerance})"
    )
    return 0 if parity.within_tolerance else 1


if __name__ == "__main__":
    raise SystemExit(main())
