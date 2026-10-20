"""Small runtime measurements for experiment and export evidence.

The helpers return plain Python values. Persisted measurements remain separate
artifacts that ``ExperimentResult v1`` can reference; this module does not
create a second experiment-result schema.
"""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


def serialized_file_size(path: str | Path) -> int:
    """Return the actual serialized file size in bytes."""

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    if not file_path.is_file():
        raise IsADirectoryError(file_path)
    return file_path.stat().st_size


def tensor_payload_bytes(
    tensors: Tensor | Iterable[Tensor] | Mapping[str, Tensor],
) -> int:
    """Return logical tensor bytes as ``numel * element_size``.

    This intentionally excludes serialization and protocol overhead, so the
    result must not be reported as network or wire traffic.
    """

    if isinstance(tensors, Tensor):
        values: Iterable[Tensor] = (tensors,)
    elif isinstance(tensors, Mapping):
        values = tensors.values()
    elif isinstance(tensors, (str, bytes)):
        raise TypeError("tensors must contain torch.Tensor values")
    else:
        try:
            values = iter(tensors)
        except TypeError as error:
            raise TypeError("tensors must be a tensor, mapping, or iterable") from error

    total = 0
    for tensor in values:
        if not isinstance(tensor, Tensor):
            raise TypeError("tensors must contain only torch.Tensor values")
        total += tensor.numel() * tensor.element_size()
    return total


def measure_cpu_latency(
    operation: Callable[[], Any],
    *,
    warmup_runs: int = 5,
    repetitions: int = 20,
    provider: str = "pytorch",
    input_shape: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Benchmark a no-argument CPU inference callable with a monotonic clock.

    Example::

        result = measure_cpu_latency(
            lambda: model(example), provider="pytorch", input_shape=example.shape
        )
    """

    if not callable(operation):
        raise TypeError("operation must be callable")
    _validate_count("warmup_runs", warmup_runs, allow_zero=True)
    _validate_count("repetitions", repetitions, allow_zero=False)
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError("provider must be a non-empty string")
    normalized_shape = _normalize_shape(input_shape)

    samples: list[float] = []
    with torch.inference_mode():
        for _ in range(warmup_runs):
            operation()
        for _ in range(repetitions):
            started_ns = time.perf_counter_ns()
            operation()
            elapsed_ns = time.perf_counter_ns() - started_ns
            samples.append(max(elapsed_ns, 0) / 1_000_000)

    sorted_samples = sorted(samples)
    p95_index = math.ceil(0.95 * len(sorted_samples)) - 1
    return {
        "measurement_kind": "cpu_inference_latency",
        "device": "cpu",
        "provider": provider.strip(),
        "clock": "perf_counter_ns",
        "warmup_runs": warmup_runs,
        "repetitions": repetitions,
        "input_shape": list(normalized_shape) if normalized_shape is not None else None,
        "samples_ms": samples,
        "p50_ms": float(statistics.median(sorted_samples)),
        "p95_ms": float(sorted_samples[p95_index]),
    }


def _validate_count(name: str, value: int, *, allow_zero: bool) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")


def _normalize_shape(input_shape: Sequence[int] | None) -> tuple[int, ...] | None:
    if input_shape is None:
        return None
    if isinstance(input_shape, (str, bytes)):
        raise TypeError("input_shape must be a sequence of integers")
    shape = tuple(input_shape)
    if any(isinstance(size, bool) or not isinstance(size, int) for size in shape):
        raise TypeError("input_shape must contain only integers")
    if any(size < 0 for size in shape):
        raise ValueError("input_shape dimensions must be non-negative")
    return shape
