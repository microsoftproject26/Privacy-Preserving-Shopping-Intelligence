"""Downstream producer-interface contract tests.

These tests validate the frozen S1-PR-03 interfaces that downstream
Issues (#18, #19, #27, #36, #51) will consume.  They do NOT implement
downstream features — they verify that the producer contracts are
stable, loadable, and usable via the public module API.
"""

from __future__ import annotations

from typing import Any

import pytest
from test_primitives import GOOD_SHA
from test_schemas import (
    _make_common_initialization,
    _make_experiment_config,
    _make_experiment_result,
    _make_model_config,
    _make_sys_measurement,
)

from scripts.experiments.compatibility import (
    build_compatibility_tuple,
    compare_compatibility,
    validate_compatibility_tuple,
)
from scripts.experiments.contracts import (
    ContractValidationError,
    build_run_id,
    canonical_sha256,
    parse_run_id,
    validate_sha256,
)
from scripts.experiments.schemas import (
    validate_common_initialization,
    validate_experiment_config,
    validate_experiment_result,
    validate_model_config,
    validate_system_measurement_reference_set,
)

# ── #18: Data Registry — ExperimentConfig / ModelConfig / CommonInit refs ──


class TestIssue18DataRegistryContracts:
    """#18 will consume ExperimentConfig refs, ModelConfig, and
    CommonInitialization via the public module.  Verify the producer
    interface is loadable, validatable, and does not require internal
    implementation knowledge."""

    def test_experiment_config_loadable_via_public_api(self) -> None:
        cfg = _make_experiment_config()
        result = validate_experiment_config(cfg)
        assert result["schema"] == "experiment_config_v1"
        assert result["version"] == "1"

    def test_model_config_consumed_without_internals(self) -> None:
        mc = _make_model_config()
        result = validate_model_config(mc)
        assert result["architecture_id"] == "fixture_linear_v1"
        assert result["parameter_dtype"] == "float32"

    def test_common_init_consumed_via_refs(self) -> None:
        ci = _make_common_initialization()
        result = validate_common_initialization(ci)
        assert result["schema"] == "common_initialization_v1"
        assert "model_config_ref" in result

    def test_all_experiment_config_refs_are_artifact_ref_v1(self) -> None:
        """All refs in ExperimentConfig are ArtifactRef v1 with sha256."""
        cfg = _make_experiment_config()
        validated = validate_experiment_config(cfg)
        ref_fields = [
            k for k in validated if k.endswith("_ref") and k != "measurement_protocol_ref"
        ]
        for field in ref_fields:
            ref = validated[field]
            if isinstance(ref, dict) and "schema" in ref:
                assert ref["schema"] == "artifact_ref_v1"
                validate_sha256(ref["sha256"])


# ── #19: Model Registry — run-v1 identity, seed stability ────────────────


class TestIssue19ModelRegistryContracts:
    """#19 will consume run-v1 identity and seed contracts.  Verify
    build/parse, seed stability, and malformed rejection."""

    def test_build_and_parse_run_id(self) -> None:
        rid = build_run_id(
            regime="R1",
            tasks=["T1", "T2", "T3"],
            cohorts=["C1", "C2", "C3"],
            seed=42,
            config_sha256=GOOD_SHA,
            attempt=1,
        )
        parsed = parse_run_id(rid)
        assert parsed["regime"] == "R1"
        assert parsed["seed"] == 42
        assert parsed["attempt"] == 1

    def test_seed_identity_stable(self) -> None:
        """Same inputs always produce the same run-ID."""
        kwargs: dict[str, Any] = {
            "regime": "R2A",
            "tasks": ["T1"],
            "cohorts": ["C1"],
            "seed": 13,
            "config_sha256": GOOD_SHA,
            "attempt": 1,
        }
        assert build_run_id(**kwargs) == build_run_id(**kwargs)

    def test_malformed_run_id_rejected(self) -> None:
        with pytest.raises(ContractValidationError, match="malformed"):
            parse_run_id("invalid-id-format")

    def test_attempt_zero_rejected_by_registry_consumer(self) -> None:
        with pytest.raises(ContractValidationError):
            build_run_id(
                regime="R1",
                tasks=["T1"],
                cohorts=["C1"],
                seed=13,
                config_sha256=GOOD_SHA,
                attempt=0,
            )


# ── #27: FL Orchestration — ExperimentResult + SystemMeasurement ─────────


class TestIssue27OrchestrationContracts:
    """#27 will consume ExperimentResult with SystemMeasurement references.
    Verify the producer interface rejects raw measurement values."""

    def test_experiment_result_accepts_sys_measurement_refs(self) -> None:
        result = _make_experiment_result()
        validated = validate_experiment_result(result)
        sm = validated["system_measurements"]
        assert sm["schema"] == "system_measurement_reference_set_v1"
        assert sm["status"] == "AVAILABLE"

    def test_sys_measurement_available_requires_refs(self) -> None:
        sm = _make_sys_measurement(status="AVAILABLE")
        validated = validate_system_measurement_reference_set(sm)
        assert isinstance(validated["refs"], list)
        assert len(validated["refs"]) > 0

    def test_raw_measurement_fields_rejected(self) -> None:
        """ExperimentResult must not contain raw size/RAM/latency/upload/
        download/wire values — only references."""
        result = _make_experiment_result()
        for forbidden_field in [
            "model_size_bytes",
            "peak_ram_bytes",
            "inference_latency_ms",
            "upload_bytes",
            "download_bytes",
            "wire_size_bytes",
        ]:
            result_with_raw = {**result, forbidden_field: 12345}
            with pytest.raises(ContractValidationError, match="unknown fields"):
                validate_experiment_result(result_with_raw)


# ── #36: Evaluation — compatibility API ──────────────────────────────────


class TestIssue36EvaluationContracts:
    """#36 will consume comparison-compatibility API.  Verify that the
    public API returns exact tuple/hash and field-level mismatch evidence."""

    def test_compatibility_tuple_deterministic(self) -> None:
        cfg = _make_experiment_config()
        git_sha = "f" * 40
        t1 = build_compatibility_tuple(cfg, git_sha)
        t2 = build_compatibility_tuple(cfg, git_sha)
        assert canonical_sha256(t1) == canonical_sha256(t2)

    def test_compatible_r1_r2a_fixtures(self) -> None:
        cfg = _make_experiment_config()
        git_sha = "f" * 40
        tup = build_compatibility_tuple(cfg, git_sha)
        result = compare_compatibility(tup, tup)
        assert result["is_compatible"] is True
        assert result["mismatches"] == []

    def test_incompatible_field_produces_mismatch_evidence(self) -> None:
        cfg1 = _make_experiment_config()
        cfg2 = _make_experiment_config(seed=42)
        git_sha = "f" * 40
        t1 = build_compatibility_tuple(cfg1, git_sha)
        t2 = build_compatibility_tuple(cfg2, git_sha)
        result = compare_compatibility(t1, t2)
        assert result["is_compatible"] is False
        assert "seed" in result["mismatches"]

    def test_every_sha_field_mismatch_detected(self) -> None:
        """Each SHA-256 field in the tuple, when changed, produces a mismatch."""
        cfg = _make_experiment_config()
        git_sha = "f" * 40
        base_tuple = build_compatibility_tuple(cfg, git_sha)
        sha_fields = [f for f in base_tuple if f.endswith("_sha256")]
        for field in sha_fields:
            modified = dict(base_tuple)
            modified[field] = "b" * 64
            result = compare_compatibility(base_tuple, modified)
            assert result["is_compatible"] is False, f"{field} mismatch not detected"
            assert field in result["mismatches"], f"{field} not in mismatches"


# ── #51: Reporting — ExperimentResult consumption ────────────────────────


class TestIssue51ReportingContracts:
    """#51 will consume ExperimentResult v1 for reporting.  Verify that
    results can be loaded/validated from the public module and that
    compatibility identity is directly consumable."""

    def test_experiment_result_loadable_from_public_module(self) -> None:
        result = _make_experiment_result()
        validated = validate_experiment_result(result)
        assert validated["schema"] == "experiment_result_v1"
        assert validated["version"] == "1"

    def test_compatibility_consumed_from_result_without_parallel_schema(self) -> None:
        """The compatibility tuple embedded in ExperimentResult can be
        independently validated without creating a separate schema."""
        result = _make_experiment_result()
        validated = validate_experiment_result(result)
        compat = validated["compatibility"]
        revalidated = validate_compatibility_tuple(compat)
        assert revalidated["schema"] == "comparison_compatibility_v1"

    def test_result_metrics_consumable(self) -> None:
        result = _make_experiment_result()
        validated = validate_experiment_result(result)
        metrics = validated["metrics"]
        assert isinstance(metrics, list)
        assert len(metrics) > 0
        assert metrics[0]["schema"] == "metric_record_v1"


# ── #26: S1-SE-06 Price Representation + Time-Gap Encoding ────────────────


class TestIssue26ProducerContracts:
    """#26 will consume ExperimentConfig for leakage-safe fitting identity.
    Verify that the public API exposes stable identities for representation,
    splits, and external config hash."""

    def test_config_exposes_fitting_identities(self) -> None:
        cfg = _make_experiment_config()
        validated = validate_experiment_config(cfg)

        # Verify split/training-manifest identity is accessible
        assert "split_manifest_ref" in validated
        assert "sha256" in validated["split_manifest_ref"]
        validate_sha256(validated["split_manifest_ref"]["sha256"])

        # Verify representation/config identity is accessible
        assert "representation_ref" in validated
        assert "sha256" in validated["representation_ref"]
        validate_sha256(validated["representation_ref"]["sha256"])

    def test_external_canonical_sha256_accessible(self) -> None:
        cfg = _make_experiment_config()
        validated = validate_experiment_config(cfg)

        # Config itself doesn't contain the hash, we compute it externally
        cfg_hash = canonical_sha256(validated)
        validate_sha256(cfg_hash)

    def test_changing_fitting_identity_changes_config_identity(self) -> None:
        cfg1 = _make_experiment_config()
        cfg2 = _make_experiment_config()

        # Change scientific fitting identity (e.g. representation)
        cfg2["representation_ref"]["sha256"] = "b" * 64

        hash1 = canonical_sha256(validate_experiment_config(cfg1))
        hash2 = canonical_sha256(validate_experiment_config(cfg2))

        assert hash1 != hash2
