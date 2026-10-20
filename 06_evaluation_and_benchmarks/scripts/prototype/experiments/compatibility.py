"""Compatibility logic for comparing experiments (Phase 5)."""

from __future__ import annotations

from typing import Any

from scripts.experiments.contracts import (
    ContractValidationError,
    reject_unknown_fields,
    strict_nonempty_string,
    validate_cohort,
    validate_cohort_set,
    validate_schema_header,
    validate_seed,
    validate_sha256,
    validate_tasks,
)
from scripts.experiments.schemas import validate_experiment_config

_COMPATIBILITY_FIELDS = frozenset(
    {
        "schema",
        "version",
        "tasks",
        "training_cohort",
        "evaluation_cohorts",
        "seed",
        "source_dataset_sha256",
        "canonical_data_contract_sha256",
        "cohort_manifest_sha256",
        "split_manifest_sha256",
        "task_examples_manifest_sha256",
        "evaluation_manifest_sha256",
        "representation_sha256",
        "model_config_sha256",
        "objective_config_sha256",
        "shared_trainer_core_sha256",
        "common_initialization_sha256",
        "evaluation_protocol_sha256",
        "evaluator_sha256",
        "environment_lock_sha256",
        "git_sha",
    }
)


def validate_compatibility_tuple(record: Any) -> dict[str, Any]:
    """Validate a ComparisonCompatibility v1 tuple."""
    if not isinstance(record, dict):
        raise ContractValidationError("compatibility tuple must be a dict")

    reject_unknown_fields(record, _COMPATIBILITY_FIELDS, "comparison_compatibility_v1")
    validate_schema_header(record, "comparison_compatibility_v1")

    validate_tasks(record.get("tasks"), "tasks")
    validate_cohort(record.get("training_cohort"), "training_cohort")
    validate_cohort_set(record.get("evaluation_cohorts"), "evaluation_cohorts")
    validate_seed(record.get("seed"))

    for field in _COMPATIBILITY_FIELDS:
        if field.endswith("_sha256"):
            validate_sha256(record.get(field), field)

    strict_nonempty_string(record.get("git_sha"), "git_sha")

    return record


def build_compatibility_tuple(config: dict[str, Any], git_sha: str) -> dict[str, Any]:
    """Build a ComparisonCompatibility v1 tuple from an ExperimentConfig and git SHA.

    If the config is missing a common_initialization_ref (e.g., R5), this builds
    an incompatible tuple with a dummy zeroed SHA-256 for the missing init,
    which will deterministically fail comparison with R1/R2A.
    """
    config = validate_experiment_config(config)

    init = config["initialization"]
    if init["kind"] == "COMMON_INITIALIZATION":
        init_sha = init["common_initialization_ref"]["sha256"]
    else:
        # Pretrained configs don't have common_initialization.
        init_sha = "0" * 64

    tuple_record = {
        "schema": "comparison_compatibility_v1",
        "version": "1",
        "tasks": config["tasks"],
        "training_cohort": config["training_cohort"],
        "evaluation_cohorts": config["evaluation_cohorts"],
        "seed": config["seed"],
        "source_dataset_sha256": config["source_dataset_ref"]["sha256"],
        "canonical_data_contract_sha256": config["canonical_data_contract_ref"]["sha256"],
        "cohort_manifest_sha256": config["cohort_manifest_ref"]["sha256"],
        "split_manifest_sha256": config["split_manifest_ref"]["sha256"],
        "task_examples_manifest_sha256": config["task_examples_manifest_ref"]["sha256"],
        "evaluation_manifest_sha256": config["evaluation_manifest_ref"]["sha256"],
        "representation_sha256": config["representation_ref"]["sha256"],
        "model_config_sha256": config["model_config_ref"]["sha256"],
        "objective_config_sha256": config["objective_config_ref"]["sha256"],
        "shared_trainer_core_sha256": config["shared_trainer_core_ref"]["sha256"],
        "common_initialization_sha256": init_sha,
        "evaluation_protocol_sha256": config["evaluation_protocol_ref"]["sha256"],
        "evaluator_sha256": config["evaluator_ref"]["sha256"],
        "environment_lock_sha256": config["environment_lock_ref"]["sha256"],
        "git_sha": git_sha,
    }

    return validate_compatibility_tuple(tuple_record)


def compare_compatibility(tuple1: dict[str, Any], tuple2: dict[str, Any]) -> dict[str, Any]:
    """Compare two compatibility tuples.

    Returns:
    {
        "is_compatible": bool,
        "mismatches": list[str]  # Field names that mismatched
    }
    """
    validate_compatibility_tuple(tuple1)
    validate_compatibility_tuple(tuple2)

    mismatches = []
    for field in _COMPATIBILITY_FIELDS:
        if field in {"schema", "version"}:
            continue
        if tuple1.get(field) != tuple2.get(field):
            mismatches.append(field)

    mismatches.sort()
    return {
        "is_compatible": len(mismatches) == 0,
        "mismatches": mismatches,
    }
