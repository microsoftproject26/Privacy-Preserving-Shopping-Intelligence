"""Run-level ExperimentResult v1 construction from existing S1-PR-03 contracts."""

from __future__ import annotations

from typing import Any

from scripts.experiments.compatibility import build_compatibility_tuple
from scripts.experiments.contracts import build_run_id
from scripts.experiments.schemas import (
    validate_experiment_config,
    validate_experiment_result,
    validate_metric_record,
)


def make_metric_record(
    *,
    metric_id: str,
    task: str,
    cohort: str,
    value: float,
    direction: str,
    unit: str,
    support: int,
) -> dict[str, Any]:
    record = {
        "schema": "metric_record_v1",
        "version": "1",
        "metric_id": metric_id,
        "task": task,
        "cohort": cohort,
        "value": value,
        "direction": direction,
        "unit": unit,
        "support": support,
    }
    return validate_metric_record(record)


def build_experiment_result(
    *,
    experiment_config: dict[str, Any],
    config_ref: dict[str, Any],
    git_sha: str,
    state: str,
    attempt: int,
    started_at_utc: str,
    ended_at_utc: str,
    metrics: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    system_measurements: dict[str, Any] | None = None,
    checkpoint_ref: dict[str, Any] | None = None,
    federated_metadata: dict[str, Any] | None = None,
    failure: dict[str, Any] | None = None,
    resume_count: int = 0,
) -> dict[str, Any]:
    config = validate_experiment_config(experiment_config)
    run_id = build_run_id(
        regime=config["regime"],
        tasks=config["tasks"],
        cohorts=config["evaluation_cohorts"],
        seed=config["seed"],
        config_sha256=config_ref["sha256"],
        attempt=attempt,
    )
    result: dict[str, Any] = {
        "schema": "experiment_result_v1",
        "version": "1",
        "run_id": run_id,
        "attempt": attempt,
        "state": state,
        "regime": config["regime"],
        "tasks": config["tasks"],
        "training_cohort": config["training_cohort"],
        "evaluation_cohorts": config["evaluation_cohorts"],
        "seed": config["seed"],
        "config_ref": config_ref,
        "compatibility": build_compatibility_tuple(config, git_sha),
        "git_sha": git_sha,
        "environment_lock_ref": config["environment_lock_ref"],
        "initialization": config["initialization"],
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "resume_count": resume_count,
        "metrics": metrics,
        "artifacts": artifacts,
        "system_measurements": system_measurements
        or {
            "schema": "system_measurement_reference_set_v1",
            "version": "1",
            "status": "NOT_APPLICABLE",
            "null_reason": "S1-PR-05 contract smoke does not persist system measurements",
        },
    }
    if checkpoint_ref is not None:
        result["checkpoint_ref"] = checkpoint_ref
    if federated_metadata is not None:
        result["federated_metadata"] = federated_metadata
    if state == "FAILED":
        if failure is None:
            raise ValueError("FAILED result requires failure details")
        result["failure"] = failure
    elif failure is not None:
        raise ValueError(f"{state} result must not contain failure details")
    return validate_experiment_result(result)
