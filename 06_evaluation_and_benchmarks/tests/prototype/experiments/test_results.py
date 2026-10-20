from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.experiments.contracts import build_run_id
from scripts.experiments.results import (
    ResultsError,
    build_quality_report,
    load_results,
    main,
    validate_result_for_reporting,
)


def _fixture() -> dict:
    return json.loads(
        Path("fixtures/experiments/contracts/experiment_result.json").read_text(encoding="utf-8")
    )


def _result(
    *,
    regime: str,
    value: float,
    metric_id: str = "accuracy",
    direction: str = "MAXIMIZE",
    seed: int = 13,
    config_sha256: str = "a" * 64,
) -> dict:
    result = deepcopy(_fixture())
    result["regime"] = regime
    result["seed"] = seed
    result["config_ref"]["sha256"] = config_sha256
    result["compatibility"]["seed"] = seed
    result["run_id"] = build_run_id(
        regime=regime,
        tasks=result["tasks"],
        cohorts=result["evaluation_cohorts"],
        seed=seed,
        config_sha256=config_sha256,
        attempt=1,
    )
    result["metrics"] = [
        {
            "schema": "metric_record_v1",
            "version": "1",
            "metric_id": metric_id,
            "task": "T1",
            "cohort": "C1",
            "value": value,
            "direction": direction,
            "unit": "FRACTION",
            "support": 100,
        }
    ]
    return result


def _write(directory: Path, name: str, result: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.result.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    return path


def test_load_and_compatible_qr_report(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    r1 = _result(regime="R1", value=0.8)
    r2a = _result(regime="R2A", value=0.72)
    _write(results_dir, "r1", r1)
    _write(results_dir, "r2a", r2a)

    loaded = load_results(results_dir)
    report = build_quality_report(loaded)

    assert report["source_run_ids"] == sorted([r1["run_id"], r2a["run_id"]])
    assert report["comparisons"] == [
        {
            "comparison_run_id": r2a["run_id"],
            "comparison_regime": "R2A",
            "seed": 13,
            "task": "T1",
            "cohort": "C1",
            "metric_id": "accuracy",
            "direction": "MAXIMIZE",
            "unit": "FRACTION",
            "comparison_value": 0.72,
            "baseline_run_id": r1["run_id"],
            "r1_value": 0.8,
            "status": "QR",
            "quality_retention": pytest.approx(0.9),
        }
    ]


def test_invalid_record_and_duplicate_run_id_name_both_files(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    invalid = _result(regime="R1", value=0.8)
    invalid["schema"] = "wrong"
    _write(results_dir, "invalid", invalid)

    with pytest.raises(ResultsError, match="invalid.result.json"):
        load_results(results_dir)

    invalid["schema"] = "experiment_result_v1"
    _write(results_dir, "first", invalid)
    (results_dir / "invalid.result.json").unlink()
    _write(results_dir, "second", invalid)
    with pytest.raises(ResultsError, match="first.result.json and second.result.json"):
        load_results(results_dir)


def test_reporting_rejects_cross_field_drift_duplicate_metric_and_nonfinite_value() -> None:
    result = _result(regime="R1", value=0.8)
    result["compatibility"]["git_sha"] = "e" * 40
    with pytest.raises(ResultsError, match="git_sha"):
        validate_result_for_reporting(result)

    result = _result(regime="R1", value=0.8)
    result["metrics"].append(deepcopy(result["metrics"][0]))
    with pytest.raises(ResultsError, match="duplicate metric"):
        validate_result_for_reporting(result)

    result = _result(regime="R1", value=float("nan"))
    with pytest.raises(ResultsError, match="finite"):
        validate_result_for_reporting(result)

    result = _result(regime="R1", value=0.8)
    result["tasks"] = ["T2"]
    with pytest.raises(ResultsError, match="run_id tasks"):
        validate_result_for_reporting(result)


def test_missing_and_incompatible_r1_are_explicit() -> None:
    comparison = _result(regime="R2A", value=0.7)
    pending = build_quality_report([comparison])["comparisons"][0]
    assert pending["status"] == "PENDING_R1"
    assert "quality_retention" not in pending

    baseline = _result(regime="R1", value=0.8)
    comparison["compatibility"]["model_config_sha256"] = "b" * 64
    incompatible = build_quality_report([comparison, baseline])["comparisons"][0]
    assert incompatible["status"] == "INCOMPATIBLE"
    assert incompatible["compatibility_mismatches"] == ["model_config_sha256"]
    assert "quality_retention" not in incompatible


@pytest.mark.parametrize(
    ("metric_id", "direction"),
    [("log_loss", "MINIMIZE"), ("brier_score", "MAXIMIZE")],
)
def test_log_loss_and_brier_are_absolute_only(metric_id: str, direction: str) -> None:
    comparison = _result(regime="R2A", value=0.2, metric_id=metric_id, direction=direction)

    entry = build_quality_report([comparison])["comparisons"][0]

    assert entry["status"] == "ABSOLUTE_ONLY"
    assert entry["comparison_value"] == 0.2
    assert "quality_retention" not in entry


def test_zero_r1_and_ambiguous_r1_fail_safely() -> None:
    zero_r1 = _result(regime="R1", value=0.0)
    comparison = _result(regime="R2A", value=0.5)
    undefined = build_quality_report([zero_r1, comparison])["comparisons"][0]
    assert undefined["status"] == "QR_UNDEFINED"
    assert "quality_retention" not in undefined

    second_r1 = _result(regime="R1", value=0.0, config_sha256="b" * 64)
    with pytest.raises(ResultsError, match="ambiguous R1"):
        build_quality_report([zero_r1, second_r1, comparison])


def test_report_is_deterministic_and_cli_writes_it(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    output = tmp_path / "quality-retention.json"
    r1 = _result(regime="R1", value=1.0)
    r2a = _result(regime="R2A", value=0.9)
    _write(results_dir, "z-r2a", r2a)
    _write(results_dir, "a-r1", r1)

    forward = build_quality_report([r1, r2a])
    reverse = build_quality_report([r2a, r1])

    assert forward == reverse
    assert main([str(results_dir), "--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == forward
