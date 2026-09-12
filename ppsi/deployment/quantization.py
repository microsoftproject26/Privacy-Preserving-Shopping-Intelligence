"""Dynamic INT8 quantization and the size, latency and quality evidence to judge it.

CP-DEPLOY has to decide whether INT8 is accepted, and the rule it states is concrete:
at least 2x smaller, no more than 1% relative loss on each approved headline metric, and
not slower under the same benchmark. This produces those three numbers on one model so
the decision is made against measurements rather than expectations.

Nothing here decides. INT8 is frequently a loss on a small CPU model, and reporting that
honestly is the useful outcome, not a failure of the attempt.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ppsi.deployment.onnx_export import (
    OUTPUT_NAMES,
    batch_to_onnx_inputs,
    serialized_size_bytes,
)
from ppsi.models.batch_spec import HISTORY_CHANNELS
from ppsi.training.batch import Phase1Batch

DEFAULT_WARMUPS = 5
DEFAULT_REPETITIONS = 30

SIZE_RATIO_TARGET = 2.0
"""CP-DEPLOY's stated acceptance rule: INT8 must be at least this much smaller."""

MAX_RELATIVE_QUALITY_LOSS = 0.01
"""CP-DEPLOY's stated acceptance rule: no more than 1% relative loss per headline metric."""


def quantize_int8(source: Path | str, destination: Path | str) -> Path:
    """Write a dynamically quantized INT8 copy of an ONNX model.

    Dynamic rather than static: static quantization needs a calibration set drawn from
    real task examples, and those carry `user_id` and are deliberately not in this
    repository. Dynamic quantization needs no calibration data, which is what makes this
    attempt reproducible here at all.
    """

    from onnxruntime.quantization import QuantType, quantize_dynamic

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=str(source),
        model_output=str(destination),
        weight_type=QuantType.QInt8,
    )
    return destination


def _session(path: Path | str) -> Any:
    import onnxruntime

    options = onnxruntime.SessionOptions()
    # One thread. A latency number measured while the machine decides how many cores to
    # use is not comparable with another one measured the same way.
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return onnxruntime.InferenceSession(
        str(path), options, providers=["CPUExecutionProvider"]
    )


def measure_latency_ms(
    onnx_path: Path | str,
    batch: Phase1Batch,
    *,
    warmups: int = DEFAULT_WARMUPS,
    repetitions: int = DEFAULT_REPETITIONS,
    channels: Sequence[str] = HISTORY_CHANNELS,
) -> dict[str, float]:
    """Median and p95 wall-clock milliseconds for one forward pass.

    Warmups are discarded rather than averaged in: the first calls pay for graph
    initialisation and arena allocation, which is a real cost but not the per-request one.
    """

    if repetitions < 1:
        raise ValueError("at least one repetition is required")

    session = _session(onnx_path)
    inputs = batch_to_onnx_inputs(batch, channels)
    outputs = list(OUTPUT_NAMES)

    for _ in range(warmups):
        session.run(outputs, inputs)

    samples: list[float] = []
    for _ in range(repetitions):
        started = time.perf_counter()
        session.run(outputs, inputs)
        samples.append((time.perf_counter() - started) * 1000.0)

    samples.sort()
    index = min(len(samples) - 1, round(0.95 * (len(samples) - 1)))
    return {
        "median_ms": statistics.median(samples),
        "p95_ms": samples[index],
        "repetitions": repetitions,
        "warmups": warmups,
        "threads": 1,
    }


def output_agreement(
    reference_path: Path | str,
    candidate_path: Path | str,
    batches: Sequence[Phase1Batch],
    channels: Sequence[str] = HISTORY_CHANNELS,
) -> dict[str, float]:
    """Worst per-head disagreement between two ONNX models on identical inputs.

    This is the quality signal available without a labelled evaluation set. It is not a
    metric delta, and the report says so rather than letting it be read as one.
    """

    reference = _session(reference_path)
    candidate = _session(candidate_path)
    outputs = list(OUTPUT_NAMES)
    worst = dict.fromkeys(OUTPUT_NAMES, 0.0)

    for batch in batches:
        inputs = batch_to_onnx_inputs(batch, channels)
        for name, want, got in zip(
            OUTPUT_NAMES, reference.run(outputs, inputs), candidate.run(outputs, inputs)
        ):
            worst[name] = max(worst[name], float(abs(want - got).max()))
    return worst


@dataclass(frozen=True, slots=True)
class QuantizationComparison:
    """Everything CP-DEPLOY's INT8 rule needs, and the verdict that follows from it."""

    fp32_bytes: int
    int8_bytes: int
    fp32_latency: dict[str, float]
    int8_latency: dict[str, float]
    agreement: dict[str, float]

    @property
    def size_ratio(self) -> float:
        return self.fp32_bytes / self.int8_bytes

    @property
    def latency_ratio(self) -> float:
        return self.fp32_latency["median_ms"] / self.int8_latency["median_ms"]

    @property
    def is_smaller_enough(self) -> bool:
        return self.size_ratio >= SIZE_RATIO_TARGET

    @property
    def is_not_slower(self) -> bool:
        return self.int8_latency["median_ms"] <= self.fp32_latency["median_ms"]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "Int8QuantizationComparison",
            "version": 1,
            "task_id": "S2-SE-02",
            "method": "dynamic, QInt8 weights, no calibration set",
            "acceptance_rule": {
                "size_ratio_at_least": SIZE_RATIO_TARGET,
                "max_relative_quality_loss": MAX_RELATIVE_QUALITY_LOSS,
                "must_not_be_slower": True,
                "source": "CP-DEPLOY Decision 1",
            },
            "fp32": {"serialized_bytes": self.fp32_bytes, "latency": self.fp32_latency},
            "int8": {"serialized_bytes": self.int8_bytes, "latency": self.int8_latency},
            "size_ratio": round(self.size_ratio, 4),
            "latency_ratio_fp32_over_int8": round(self.latency_ratio, 4),
            "max_output_disagreement": {k: v for k, v in sorted(self.agreement.items())},
            "quality_note": (
                "Output disagreement is not a metric delta. A per-task quality delta needs "
                "the frozen evaluation set, which carries user_id and is not in this "
                "repository, so CP-DEPLOY should treat this as a bound and not as the "
                "1% headline-metric check its rule names."
            ),
            "meets_size_rule": self.is_smaller_enough,
            "meets_speed_rule": self.is_not_slower,
        }


def compare_fp32_and_int8(
    fp32_path: Path | str,
    int8_path: Path | str,
    batches: Sequence[Phase1Batch],
    *,
    warmups: int = DEFAULT_WARMUPS,
    repetitions: int = DEFAULT_REPETITIONS,
    channels: Sequence[str] = HISTORY_CHANNELS,
) -> QuantizationComparison:
    """Measure both models on the same machine, threads, shapes and repetitions."""

    if not batches:
        raise ValueError("at least one batch is required")
    benchmark_batch = batches[0]

    return QuantizationComparison(
        fp32_bytes=serialized_size_bytes(fp32_path),
        int8_bytes=serialized_size_bytes(int8_path),
        fp32_latency=measure_latency_ms(
            fp32_path, benchmark_batch, warmups=warmups, repetitions=repetitions, channels=channels
        ),
        int8_latency=measure_latency_ms(
            int8_path, benchmark_batch, warmups=warmups, repetitions=repetitions, channels=channels
        ),
        agreement=output_agreement(fp32_path, int8_path, batches, channels),
    )
