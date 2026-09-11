from __future__ import annotations

import json
from pathlib import Path

import pytest

from ppsi.deployment.onnx_export import (
    OUTPUT_NAMES,
    deterministic_example_batch,
    export_session_gru,
    serialized_size_bytes,
)
from ppsi.deployment.quantization import (
    MAX_RELATIVE_QUALITY_LOSS,
    SIZE_RATIO_TARGET,
    QuantizationComparison,
    compare_fp32_and_int8,
    measure_latency_ms,
    output_agreement,
    quantize_int8,
)
from ppsi.models.batch_spec import phase1_batch_spec_v1
from ppsi.models.session_gru import build_model

EVIDENCE = (
    Path(__file__).resolve().parents[2] / "docs" / "evidence" / "s2-se-02" / "int8_benchmark.v1.json"
)


@pytest.fixture(scope="module")
def spec():
    return phase1_batch_spec_v1()


@pytest.fixture(scope="module")
def batch(spec):
    return deterministic_example_batch(spec, rows=2, history=4, candidates=3)


@pytest.fixture(scope="module")
def models(tmp_path_factory, spec, batch):
    directory = tmp_path_factory.mktemp("int8")
    fp32 = export_session_gru(
        build_model(13, batch_spec=spec).eval(),
        directory / "m.fp32.onnx",
        example_batch=batch,
        batch_spec=spec,
    )
    return fp32, quantize_int8(fp32, directory / "m.int8.onnx")


def test_quantizing_produces_a_substantially_smaller_model(models) -> None:
    fp32, int8 = models

    assert serialized_size_bytes(int8) < serialized_size_bytes(fp32)
    assert serialized_size_bytes(fp32) / serialized_size_bytes(int8) >= SIZE_RATIO_TARGET


def test_the_quantized_model_still_runs_and_produces_every_head(models, batch) -> None:
    fp32, int8 = models

    agreement = output_agreement(fp32, int8, [batch])

    assert set(agreement) == set(OUTPUT_NAMES)
    assert all(value >= 0.0 for value in agreement.values())


def test_latency_samples_are_well_formed(models, batch) -> None:
    fp32, _ = models

    latency = measure_latency_ms(fp32, batch, warmups=2, repetitions=5)

    assert latency["median_ms"] > 0.0
    assert latency["p95_ms"] >= latency["median_ms"]
    assert latency["repetitions"] == 5
    assert latency["threads"] == 1


def test_a_single_repetition_is_allowed_and_zero_is_refused(models, batch) -> None:
    fp32, _ = models

    assert measure_latency_ms(fp32, batch, warmups=0, repetitions=1)["repetitions"] == 1
    with pytest.raises(ValueError, match="repetition"):
        measure_latency_ms(fp32, batch, repetitions=0)


def test_the_acceptance_rules_follow_from_the_measurements() -> None:
    fast_and_small = QuantizationComparison(
        fp32_bytes=1000,
        int8_bytes=250,
        fp32_latency={"median_ms": 2.0},
        int8_latency={"median_ms": 1.0},
        agreement={name: 0.0 for name in OUTPUT_NAMES},
    )
    assert fast_and_small.is_smaller_enough
    assert fast_and_small.is_not_slower
    assert fast_and_small.size_ratio == 4.0

    barely_smaller_and_slower = QuantizationComparison(
        fp32_bytes=1000,
        int8_bytes=800,
        fp32_latency={"median_ms": 1.0},
        int8_latency={"median_ms": 2.0},
        agreement={name: 0.0 for name in OUTPUT_NAMES},
    )
    assert not barely_smaller_and_slower.is_smaller_enough
    assert not barely_smaller_and_slower.is_not_slower


def test_comparison_refuses_an_empty_batch_list(models) -> None:
    fp32, int8 = models

    with pytest.raises(ValueError, match="at least one batch"):
        compare_fp32_and_int8(fp32, int8, [])


def test_the_committed_evidence_states_the_rule_it_was_judged_against() -> None:
    report = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    rule = report["acceptance_rule"]
    assert rule["size_ratio_at_least"] == SIZE_RATIO_TARGET
    assert rule["max_relative_quality_loss"] == MAX_RELATIVE_QUALITY_LOSS
    assert rule["must_not_be_slower"] is True
    assert report["method"].startswith("dynamic")


def test_the_evidence_does_not_present_output_drift_as_a_quality_metric() -> None:
    # The rule asks for a headline-metric delta. This run cannot produce one, and the
    # report has to say so rather than letting a max-absolute-difference stand in for it.
    report = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    assert "not a metric delta" in report["quality_note"]
    assert "max_output_disagreement" in report
    assert "quality_delta" not in report


def test_the_evidence_records_both_sides_of_every_rule() -> None:
    report = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    assert report["fp32"]["serialized_bytes"] > report["int8"]["serialized_bytes"]
    assert report["size_ratio"] >= SIZE_RATIO_TARGET
    assert report["meets_size_rule"] is True
    assert isinstance(report["meets_speed_rule"], bool)
    assert report["fp32"]["latency"]["threads"] == report["int8"]["latency"]["threads"] == 1
