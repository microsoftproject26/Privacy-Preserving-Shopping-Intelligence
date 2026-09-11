"""Tests for experiment schemas (ModelConfig, etc.)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.experiments.contracts import ContractValidationError
from scripts.experiments.schemas import validate_model_config

# ── Helpers ──────────────────────────────────────────────────────────────


def _make_model_config(**overrides: Any) -> dict[str, Any]:
    base = {
        "schema": "model_config_v1",
        "version": "1",
        "model_config_id": "test_mc",
        "model_family": "test_family",
        "architecture_id": "fixture_linear_v1",
        "architecture_version": "1",
        "parameter_dtype": "float32",
        "architecture_parameters": {
            "input_dim": 3,
            "output_dim": 2,
            "bias": True,
        },
        "task_heads": [
            {
                "task": "T1",
                "head_id": "h1",
                "head_version": "1",
                "output_dim": 2,
            }
        ],
    }
    base.update(overrides)
    return base


# ── ModelConfig ──────────────────────────────────────────────────────────


class TestModelConfig:
    def test_valid(self) -> None:
        cfg = _make_model_config()
        assert validate_model_config(cfg) == cfg

    def test_invalid_schema(self) -> None:
        cfg = _make_model_config(schema="wrong_v1")
        with pytest.raises(ContractValidationError, match="schema"):
            validate_model_config(cfg)

    def test_invalid_version(self) -> None:
        cfg = _make_model_config(version="2")
        with pytest.raises(ContractValidationError, match="version"):
            validate_model_config(cfg)

    def test_unsupported_dtype(self) -> None:
        cfg = _make_model_config(parameter_dtype="float16")
        with pytest.raises(ContractValidationError, match="float32"):
            validate_model_config(cfg)

    def test_unsupported_architecture(self) -> None:
        cfg = _make_model_config(architecture_id="unknown_arch")
        with pytest.raises(ContractValidationError, match="unknown architecture"):
            validate_model_config(cfg)

    def test_fixture_linear_params(self) -> None:
        cfg = _make_model_config()
        cfg["architecture_parameters"]["input_dim"] = -1
        with pytest.raises(ContractValidationError, match=">="):
            validate_model_config(cfg)

    def test_missing_task_heads(self) -> None:
        cfg = _make_model_config(task_heads=[])
        with pytest.raises(ContractValidationError, match="non-empty list"):
            validate_model_config(cfg)

    def test_bad_task_order(self) -> None:
        cfg = _make_model_config(
            task_heads=[
                {"task": "T2", "head_id": "h2", "head_version": "1", "output_dim": 2},
                {"task": "T1", "head_id": "h1", "head_version": "1", "output_dim": 2},
            ]
        )
        with pytest.raises(ContractValidationError, match="canonical order"):
            validate_model_config(cfg)

    def test_unknown_fields(self) -> None:
        cfg = _make_model_config(extra="bad")
        with pytest.raises(ContractValidationError, match="unknown fields"):
            validate_model_config(cfg)

    def test_frozen_shared_session_encoder(self) -> None:
        cfg = _make_model_config(
            architecture_id="shared_session_encoder_v1",
            architecture_parameters={
                "core": "gru",
                "hidden": 128,
                "layers": 1,
                "dropout": 0.3,
                "history_channels": ["category_id", "event_type_id"],
                "use_gap": True,
                "history_length": 20,
                "batch_spec": "phase1_batch_spec_v1",
                "parameter_count": 2379263,
                "readout": "gather at the final valid position",
                "architecture_selected_by": "S2-DS-05 validation comparison",
            },
        )
        assert validate_model_config(cfg) == cfg

    def test_shared_session_encoder_rejects_channel_drift(self) -> None:
        cfg = _make_model_config(
            architecture_id="shared_session_encoder_v1",
            architecture_parameters={
                "core": "gru",
                "hidden": 128,
                "layers": 1,
                "dropout": 0.3,
                "history_channels": ["event_type_id", "category_id"],
                "use_gap": True,
                "history_length": 20,
                "batch_spec": "phase1_batch_spec_v1",
                "parameter_count": 2379263,
                "readout": "gather at the final valid position",
                "architecture_selected_by": "S2-DS-05 validation comparison",
            },
        )
        with pytest.raises(ContractValidationError, match="frozen channel order"):
            validate_model_config(cfg)

    def test_published_s2_ds_08_config_and_initializations_share_one_hash(self) -> None:
        root = Path(__file__).resolve().parents[2]
        contract_dir = root / "config" / "experiments" / "s2-ds-08"
        config_path = contract_dir / "model_config.v1.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        assert validate_model_config(config) == config

        digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
        for seed in (13, 42, 2026):
            initialization = json.loads(
                (contract_dir / f"common_initialization.seed{seed}.v1.json").read_text(
                    encoding="utf-8"
                )
            )
            assert initialization["model_config_ref"]["sha256"] == digest


# ── ExperimentConfig ─────────────────────────────────────────────────────

from scripts.experiments.schemas import validate_experiment_config
from tests.experiments.test_primitives import _make_artifact_ref


def _make_experiment_config(**overrides: Any) -> dict[str, Any]:
    base = {
        "schema": "experiment_config_v1",
        "version": "1",
        "config_id": "test_cfg",
        "regime": "R1",
        "tasks": ["T1"],
        "training_cohort": "C1",
        "evaluation_cohorts": ["C1"],
        "seed": 13,
        "source_dataset_ref": _make_artifact_ref(),
        "canonical_data_contract_ref": _make_artifact_ref(),
        "cohort_manifest_ref": _make_artifact_ref(),
        "split_manifest_ref": _make_artifact_ref(),
        "task_examples_manifest_ref": _make_artifact_ref(),
        "evaluation_manifest_ref": _make_artifact_ref(),
        "representation_ref": _make_artifact_ref(),
        "model_config_ref": _make_artifact_ref(),
        "objective_config_ref": _make_artifact_ref(),
        "shared_trainer_core_ref": _make_artifact_ref(),
        "evaluation_protocol_ref": _make_artifact_ref(),
        "evaluator_ref": _make_artifact_ref(),
        "environment_lock_ref": _make_artifact_ref(),
        "initialization": {
            "kind": "COMMON_INITIALIZATION",
            "common_initialization_ref": _make_artifact_ref(),
        },
        "regime_config": {},
    }
    base.update(overrides)
    return base


class TestExperimentConfig:
    def test_valid_r1(self) -> None:
        cfg = _make_experiment_config()
        assert validate_experiment_config(cfg) == cfg

    def test_valid_r5(self) -> None:
        cfg = _make_experiment_config(
            regime="R5",
            initialization={
                "kind": "PRETRAINED_R1_CHECKPOINT",
                "pretrained_checkpoint_ref": _make_artifact_ref(),
                "parent_run_id": "run-v1__r1__t1__c1__s13__cfg0123456789ab__a1",
            },
        )
        assert validate_experiment_config(cfg) == cfg

    def test_r1_with_r5_init_rejected(self) -> None:
        cfg = _make_experiment_config(
            regime="R1",
            initialization={
                "kind": "PRETRAINED_R1_CHECKPOINT",
                "pretrained_checkpoint_ref": _make_artifact_ref(),
                "parent_run_id": "some_run_id",
            },
        )
        with pytest.raises(ContractValidationError, match="COMMON_INITIALIZATION for regime R1"):
            validate_experiment_config(cfg)

    def test_r5_with_r1_init_rejected(self) -> None:
        cfg = _make_experiment_config(
            regime="R5",
            initialization={
                "kind": "COMMON_INITIALIZATION",
                "common_initialization_ref": _make_artifact_ref(),
            },
        )
        with pytest.raises(ContractValidationError, match="PRETRAINED_R1_CHECKPOINT for regime R5"):
            validate_experiment_config(cfg)

    def test_missing_required_ref(self) -> None:
        cfg = _make_experiment_config()
        del cfg["model_config_ref"]
        with pytest.raises(ContractValidationError, match="must be a dict"):
            validate_experiment_config(cfg)

    def test_invalid_seed(self) -> None:
        cfg = _make_experiment_config(seed=99)
        with pytest.raises(ContractValidationError, match="seed"):
            validate_experiment_config(cfg)

    def test_optional_measurement_ref(self) -> None:
        cfg = _make_experiment_config(measurement_protocol_ref=_make_artifact_ref())
        assert validate_experiment_config(cfg) == cfg


# ── MetricRecord ─────────────────────────────────────────────────────────

from scripts.experiments.schemas import validate_metric_record


def _make_metric_record(**overrides: Any) -> dict[str, Any]:
    base = {
        "schema": "metric_record_v1",
        "version": "1",
        "metric_id": "test_metric",
        "task": "T1",
        "cohort": "C1",
        "value": 0.95,
        "direction": "MAXIMIZE",
        "unit": "FRACTION",
        "support": 1000,
    }
    base.update(overrides)
    return base


class TestMetricRecord:
    def test_valid_with_support(self) -> None:
        rec = _make_metric_record()
        assert validate_metric_record(rec) == rec

    def test_valid_with_null_reason(self) -> None:
        rec = _make_metric_record()
        del rec["support"]
        rec["support_null_reason"] = "no data"
        assert validate_metric_record(rec) == rec

    def test_both_support_and_reason_rejected(self) -> None:
        rec = _make_metric_record(support_null_reason="no data")
        with pytest.raises(ContractValidationError, match="exactly one"):
            validate_metric_record(rec)

    def test_neither_support_nor_reason_rejected(self) -> None:
        rec = _make_metric_record()
        del rec["support"]
        with pytest.raises(ContractValidationError, match="exactly one"):
            validate_metric_record(rec)


# ── SystemMeasurementReferenceSet ────────────────────────────────────────

from scripts.experiments.schemas import validate_system_measurement_reference_set


def _make_sys_measurement(**overrides: Any) -> dict[str, Any]:
    base = {
        "schema": "system_measurement_reference_set_v1",
        "version": "1",
        "status": "AVAILABLE",
        "refs": [_make_artifact_ref()],
    }
    base.update(overrides)
    return base


class TestSystemMeasurementReferenceSet:
    def test_available(self) -> None:
        rec = _make_sys_measurement()
        assert validate_system_measurement_reference_set(rec) == rec

    def test_pending(self) -> None:
        rec = _make_sys_measurement(status="PENDING", null_reason="still running")
        del rec["refs"]
        assert validate_system_measurement_reference_set(rec) == rec

    def test_available_with_reason_rejected(self) -> None:
        rec = _make_sys_measurement(null_reason="bad")
        with pytest.raises(ContractValidationError, match="cannot have null_reason"):
            validate_system_measurement_reference_set(rec)

    def test_pending_with_refs_rejected(self) -> None:
        rec = _make_sys_measurement(status="PENDING", null_reason="wait")
        with pytest.raises(ContractValidationError, match="cannot have refs"):
            validate_system_measurement_reference_set(rec)


# ── ExperimentResult ─────────────────────────────────────────────────────

from scripts.experiments.contracts import build_run_id
from scripts.experiments.schemas import validate_experiment_result
from tests.experiments.test_primitives import GOOD_SHA, GOOD_UTC


def _make_experiment_result(**overrides: Any) -> dict[str, Any]:
    rid = build_run_id(
        regime="R1", tasks=["T1"], cohorts=["C1"], seed=13, config_sha256=GOOD_SHA, attempt=1
    )
    base = {
        "schema": "experiment_result_v1",
        "version": "1",
        "run_id": rid,
        "attempt": 1,
        "state": "SUCCEEDED",
        "regime": "R1",
        "tasks": ["T1"],
        "training_cohort": "C1",
        "evaluation_cohorts": ["C1"],
        "seed": 13,
        "config_ref": _make_artifact_ref(sha256=GOOD_SHA),
        "compatibility": {
            "schema": "comparison_compatibility_v1",
            "version": "1",
            "tasks": ["T1"],
            "training_cohort": "C1",
            "evaluation_cohorts": ["C1"],
            "seed": 13,
            "source_dataset_sha256": GOOD_SHA,
            "canonical_data_contract_sha256": GOOD_SHA,
            "cohort_manifest_sha256": GOOD_SHA,
            "split_manifest_sha256": GOOD_SHA,
            "task_examples_manifest_sha256": GOOD_SHA,
            "evaluation_manifest_sha256": GOOD_SHA,
            "representation_sha256": GOOD_SHA,
            "model_config_sha256": GOOD_SHA,
            "objective_config_sha256": GOOD_SHA,
            "shared_trainer_core_sha256": GOOD_SHA,
            "common_initialization_sha256": GOOD_SHA,
            "evaluation_protocol_sha256": GOOD_SHA,
            "evaluator_sha256": GOOD_SHA,
            "environment_lock_sha256": GOOD_SHA,
            "git_sha": "f" * 40,
        },
        "git_sha": "f" * 40,
        "environment_lock_ref": _make_artifact_ref(),
        "initialization": {
            "kind": "COMMON_INITIALIZATION",
            "common_initialization_ref": _make_artifact_ref(),
        },
        "started_at_utc": GOOD_UTC,
        "ended_at_utc": GOOD_UTC,
        "metrics": [_make_metric_record()],
        "artifacts": [_make_artifact_ref()],
        "system_measurements": _make_sys_measurement(),
    }
    base.update(overrides)
    return base


class TestExperimentResult:
    def test_valid_succeeded(self) -> None:
        res = _make_experiment_result()
        assert validate_experiment_result(res) == res

    def test_planned_no_timestamps(self) -> None:
        res = _make_experiment_result(state="PLANNED")
        del res["started_at_utc"]
        del res["ended_at_utc"]
        assert validate_experiment_result(res) == res

    def test_running_no_end_time(self) -> None:
        res = _make_experiment_result(state="RUNNING")
        del res["ended_at_utc"]
        assert validate_experiment_result(res) == res

    def test_failed_requires_failure(self) -> None:
        res = _make_experiment_result(state="FAILED")
        with pytest.raises(ContractValidationError, match="requires failure details"):
            validate_experiment_result(res)
        res["failure"] = {"reason": "test"}
        assert validate_experiment_result(res) == res

    def test_run_id_consistency(self) -> None:
        res = _make_experiment_result(seed=42)
        with pytest.raises(ContractValidationError, match="do not match"):
            validate_experiment_result(res)


# ── CommonInitialization ─────────────────────────────────────────────────

from scripts.experiments.schemas import validate_common_initialization


def _make_common_initialization(**overrides: Any) -> dict[str, Any]:
    base = {
        "schema": "common_initialization_v1",
        "version": "1",
        "regime": "R1",
        "tasks": ["T1"],
        "model_config_ref": _make_artifact_ref(),
        "seed": 13,
        "initializer_git_sha": "f" * 40,
    }
    base.update(overrides)
    return base


class TestCommonInitialization:
    def test_valid(self) -> None:
        init = _make_common_initialization()
        assert validate_common_initialization(init) == init

    def test_invalid_seed(self) -> None:
        init = _make_common_initialization(seed=99)
        with pytest.raises(ContractValidationError, match="seed"):
            validate_common_initialization(init)

    def test_invalid_git_sha(self) -> None:
        init = _make_common_initialization(initializer_git_sha="")
        with pytest.raises(ContractValidationError, match="non-empty string"):
            validate_common_initialization(init)
