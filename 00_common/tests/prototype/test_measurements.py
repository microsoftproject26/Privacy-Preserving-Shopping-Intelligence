from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ppsi import measurements
from ppsi.measurements import (
    measure_cpu_latency,
    serialized_file_size,
    tensor_payload_bytes,
)


def test_serialized_file_size_uses_actual_bytes(tmp_path: Path) -> None:
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"model\x00bytes")

    assert serialized_file_size(artifact) == 11


def test_serialized_file_size_rejects_missing_file_and_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        serialized_file_size(tmp_path / "missing.bin")
    with pytest.raises(IsADirectoryError):
        serialized_file_size(tmp_path)


def test_tensor_payload_bytes_matches_hand_calculation() -> None:
    tensors = {
        "weights": torch.zeros((2, 3), dtype=torch.float32),
        "ids": torch.zeros(4, dtype=torch.int64),
    }

    assert tensor_payload_bytes(tensors) == (2 * 3 * 4) + (4 * 8)
    assert tensor_payload_bytes(tensors["weights"]) == 24
    assert tensor_payload_bytes([]) == 0


def test_tensor_payload_bytes_rejects_non_tensors() -> None:
    with pytest.raises(TypeError, match="torch.Tensor"):
        tensor_payload_bytes([torch.zeros(1), 2])
    with pytest.raises(TypeError, match="tensor, mapping, or iterable"):
        tensor_payload_bytes(3)


def test_cpu_latency_records_samples_percentiles_and_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter(
        [
            0,
            1_000_000,
            1_000_000,
            3_000_000,
            3_000_000,
            7_000_000,
            7_000_000,
            11_000_000,
        ]
    )
    monkeypatch.setattr(measurements.time, "perf_counter_ns", lambda: next(ticks))

    result = measure_cpu_latency(
        lambda: None,
        warmup_runs=0,
        repetitions=4,
        provider="test-provider",
        input_shape=(2, 4),
    )

    assert result == {
        "measurement_kind": "cpu_inference_latency",
        "device": "cpu",
        "provider": "test-provider",
        "clock": "perf_counter_ns",
        "warmup_runs": 0,
        "repetitions": 4,
        "input_shape": [2, 4],
        "samples_ms": [1.0, 2.0, 4.0, 4.0],
        "p50_ms": 3.0,
        "p95_ms": 4.0,
    }


def test_cpu_latency_runs_warmups_and_a_tiny_cpu_model() -> None:
    model = torch.nn.Linear(4, 2).eval()
    example = torch.ones((1, 4))
    calls = 0

    def infer() -> torch.Tensor:
        nonlocal calls
        calls += 1
        return model(example)

    result = measure_cpu_latency(
        infer,
        warmup_runs=2,
        repetitions=5,
        input_shape=example.shape,
    )

    assert calls == 7
    assert len(result["samples_ms"]) == 5
    assert result["p95_ms"] >= result["p50_ms"] >= 0
    assert result["input_shape"] == [1, 4]


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"warmup_runs": -1}, ValueError),
        ({"warmup_runs": True}, TypeError),
        ({"repetitions": 0}, ValueError),
        ({"repetitions": 1.5}, TypeError),
        ({"provider": ""}, ValueError),
        ({"input_shape": (1, -1)}, ValueError),
        ({"input_shape": (1, 2.5)}, TypeError),
    ],
)
def test_cpu_latency_rejects_invalid_configuration(kwargs: dict, error: type[Exception]) -> None:
    with pytest.raises(error):
        measure_cpu_latency(lambda: None, **kwargs)


def test_cpu_latency_requires_callable() -> None:
    with pytest.raises(TypeError, match="callable"):
        measure_cpu_latency(None)
