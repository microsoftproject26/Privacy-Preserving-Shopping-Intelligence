"""Validate ExperimentResult v1 files and build a deterministic QR report."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.experiments.compatibility import compare_compatibility
from scripts.experiments.contracts import ContractValidationError, parse_run_id
from scripts.experiments.schemas import validate_experiment_result


class ResultsError(ValueError):
    """A result directory cannot be validated or compared safely."""


def load_results(directory: str | Path) -> list[dict[str, Any]]:
    """Load and validate deterministic ``*.result.json`` records from a directory."""

    results_path = Path(directory)
    if not results_path.exists():
        raise FileNotFoundError(results_path)
    if not results_path.is_dir():
        raise NotADirectoryError(results_path)

    paths = sorted(results_path.glob("*.result.json"))
    if not paths:
        raise ResultsError(f"no *.result.json files found in {results_path}")

    records: list[dict[str, Any]] = []
    sources: dict[str, Path] = {}
    for path in paths:
        try:
            raw = json.loads(path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as error:
            raise ResultsError(f"{path.name}: invalid JSON: {error.msg}") from error
        try:
            record = validate_result_for_reporting(raw, source=path.name)
        except ContractValidationError as error:
            raise ResultsError(f"{path.name}: invalid ExperimentResult v1: {error}") from error

        run_id = record["run_id"]
        if run_id in sources:
            raise ResultsError(
                f"duplicate run_id {run_id!r} in {sources[run_id].name} and {path.name}"
            )
        sources[run_id] = path
        records.append(record)
    return records


def validate_result_for_reporting(
    record: dict[str, Any], *, source: str = "result"
) -> dict[str, Any]:
    """Apply the frozen validator plus cross-field checks needed for safe QR."""

    validated = validate_experiment_result(record)
    parsed = parse_run_id(validated["run_id"])
    if parsed["tasks"] != validated["tasks"]:
        raise ResultsError(f"{source}: run_id tasks do not match result tasks")
    if parsed["cohorts"] != validated["evaluation_cohorts"]:
        raise ResultsError(f"{source}: run_id cohorts do not match evaluation_cohorts")

    compatibility = validated["compatibility"]
    for field in ("tasks", "training_cohort", "evaluation_cohorts", "seed", "git_sha"):
        if validated[field] != compatibility[field]:
            raise ResultsError(f"{source}: {field} does not match compatibility.{field}")
    if validated["environment_lock_ref"]["sha256"] != compatibility["environment_lock_sha256"]:
        raise ResultsError(f"{source}: environment lock does not match compatibility identity")
    initialization = validated["initialization"]
    if (
        initialization["kind"] == "COMMON_INITIALIZATION"
        and initialization["common_initialization_ref"]["sha256"]
        != compatibility["common_initialization_sha256"]
    ):
        raise ResultsError(f"{source}: common initialization does not match compatibility identity")

    _reject_duplicate_metrics(validated, source=source)
    for metric in validated["metrics"]:
        if metric["task"] not in validated["tasks"]:
            raise ResultsError(f"{source}: metric task {metric['task']!r} is not in result tasks")
        if metric["cohort"] not in validated["evaluation_cohorts"]:
            raise ResultsError(
                f"{source}: metric cohort {metric['cohort']!r} is not in evaluation_cohorts"
            )
        if not math.isfinite(metric["value"]):
            raise ResultsError(f"{source}: metric {_metric_label(metric)} value must be finite")
    return validated


def build_quality_report(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Build QR records for succeeded non-R1 runs without changing frozen contracts."""

    validated = _validate_records(results)
    baselines = [
        result
        for result in validated
        if result["state"] == "SUCCEEDED" and result["regime"] == "R1"
    ]
    comparisons: list[dict[str, Any]] = []

    for result in validated:
        if result["state"] != "SUCCEEDED" or result["regime"] == "R1":
            continue
        for metric in result["metrics"]:
            entry = _comparison_identity(result, metric)
            if _is_absolute_only(metric):
                entry.update(
                    status="ABSOLUTE_ONLY",
                    reason="QR is defined only for approved higher-is-better metrics",
                )
                comparisons.append(entry)
                continue

            compatible, mismatches = _matching_baselines(result, metric, baselines)
            if len(compatible) > 1:
                run_ids = sorted(candidate[0]["run_id"] for candidate in compatible)
                raise ResultsError(
                    f"ambiguous R1 baseline for {result['run_id']} / {_metric_label(metric)}: "
                    f"{run_ids}"
                )
            if not compatible:
                if mismatches:
                    entry.update(
                        status="INCOMPATIBLE",
                        reason="R1 exists but frozen comparison identities differ",
                        compatibility_mismatches=sorted(mismatches),
                    )
                else:
                    entry.update(
                        status="PENDING_R1",
                        reason="no succeeded R1 result matches seed, task, cohort, and metric",
                    )
                comparisons.append(entry)
                continue

            baseline, baseline_metric = compatible[0]
            entry.update(
                baseline_run_id=baseline["run_id"],
                r1_value=baseline_metric["value"],
            )
            if baseline_metric["value"] <= 0:
                entry.update(status="QR_UNDEFINED", reason="R1 metric value is not positive")
            else:
                entry.update(
                    status="QR",
                    quality_retention=metric["value"] / baseline_metric["value"],
                )
            comparisons.append(entry)

    comparisons.sort(
        key=lambda entry: (
            entry["comparison_run_id"],
            entry["task"],
            entry["cohort"],
            entry["metric_id"],
        )
    )
    return {
        "report": "quality_retention",
        "source_run_ids": sorted(result["run_id"] for result in validated),
        "comparisons": comparisons,
    }


def _validate_records(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    run_ids: set[str] = set()
    for index, result in enumerate(results):
        try:
            record = validate_result_for_reporting(result, source=f"results[{index}]")
        except ContractValidationError as error:
            raise ResultsError(f"results[{index}]: invalid ExperimentResult v1: {error}") from error
        run_id = record["run_id"]
        if run_id in run_ids:
            raise ResultsError(f"duplicate run_id {run_id!r}")
        run_ids.add(run_id)
        validated.append(record)
    return sorted(validated, key=lambda record: record["run_id"])


def _reject_duplicate_metrics(result: dict[str, Any], *, source: str) -> None:
    seen: set[tuple[str, str, str]] = set()
    for metric in result["metrics"]:
        key = (metric["task"], metric["cohort"], metric["metric_id"])
        if key in seen:
            raise ResultsError(f"{source}: duplicate metric {_metric_label(metric)}")
        seen.add(key)


def _matching_baselines(
    comparison: dict[str, Any],
    metric: dict[str, Any],
    baselines: Sequence[dict[str, Any]],
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], set[str]]:
    compatible: list[tuple[dict[str, Any], dict[str, Any]]] = []
    mismatches: set[str] = set()
    for baseline in baselines:
        if baseline["seed"] != comparison["seed"]:
            continue
        baseline_metrics = [
            candidate
            for candidate in baseline["metrics"]
            if _metric_key(candidate) == _metric_key(metric)
        ]
        for baseline_metric in baseline_metrics:
            outcome = compare_compatibility(baseline["compatibility"], comparison["compatibility"])
            if outcome["is_compatible"]:
                compatible.append((baseline, baseline_metric))
            else:
                mismatches.update(outcome["mismatches"])
    return compatible, mismatches


def _metric_key(metric: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        metric["task"],
        metric["cohort"],
        metric["metric_id"],
        metric["direction"],
        metric["unit"],
    )


def _metric_label(metric: dict[str, Any]) -> str:
    return f"{metric['task']}/{metric['cohort']}/{metric['metric_id']}"


def _is_absolute_only(metric: dict[str, Any]) -> bool:
    normalized_id = "".join(
        character for character in metric["metric_id"].lower() if character.isalnum()
    )
    return (
        metric["direction"] != "MAXIMIZE" or "logloss" in normalized_id or "brier" in normalized_id
    )


def _comparison_identity(result: dict[str, Any], metric: dict[str, Any]) -> dict[str, Any]:
    return {
        "comparison_run_id": result["run_id"],
        "comparison_regime": result["regime"],
        "seed": result["seed"],
        "task": metric["task"],
        "cohort": metric["cohort"],
        "metric_id": metric["metric_id"],
        "direction": metric["direction"],
        "unit": metric["unit"],
        "comparison_value": metric["value"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate ExperimentResult v1 files and build a Quality Retention report."
    )
    parser.add_argument("results_directory", type=Path)
    parser.add_argument("--output", type=Path, help="Write JSON here instead of stdout.")
    args = parser.parse_args(argv)

    try:
        report = build_quality_report(load_results(args.results_directory))
        rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output is None:
            print(rendered, end="")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
            print(f"Validated {len(report['source_run_ids'])} result(s); wrote {args.output}")
    except (OSError, ResultsError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
