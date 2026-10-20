"""Tests for regime catalog and artifact map validation via internal validators."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ppsi_paths import REPO_ROOT, prototype_path, prototype_view
from scripts.experiments.contracts import ContractValidationError
from scripts.experiments.schemas import (
    validate_experiment_artifact_map,
    validate_experiment_contract_artifact_manifest,
    validate_experiment_contract_validation_summary,
    validate_regime_catalog,
)
from scripts.experiments.validate_experiment_contracts import (
    _verify_artifact_manifest,
    _verify_validation_summary,
)

_ROOT = REPO_ROOT
_EVIDENCE_DIR = prototype_path("docs/evidence/s1-pr-03")


def _prototype_layout_view(dest: Path) -> Path:
    """The artifact manifest names every artifact by its prototype-layout path (docs/, scripts/).

    Verify it against a byte copy of the current files laid out that way.
    """
    artifact_map = json.loads(
        (_ROOT / "config" / "experiments" / "experiment_artifact_map.v1.json").read_text(
            encoding="utf-8"
        )
    )
    paths = [entry["path"].rstrip("/") for entry in artifact_map["artifacts"].values()]
    return prototype_view(dest, paths)


class TestRegimeCatalog:
    def test_regime_catalog_validates(self) -> None:
        data_path = _ROOT / "config" / "experiments" / "regime_catalog.v1.json"
        assert data_path.exists(), f"Missing {data_path}"
        data = json.loads(data_path.read_text(encoding="utf-8"))
        result = validate_regime_catalog(data)
        expected_regimes = {"R1", "R2A", "R2B", "R3", "R4", "R5"}
        assert set(result["regimes"].keys()) == expected_regimes
        assert result["regimes"]["R1"] == {
            "description": "Centralized reference baseline",
            "orchestration_type": "CENTRALIZED",
        }
        assert result["regimes"]["R2A"] == {
            "description": "Standard federated averaging (FedAvg) baseline",
            "orchestration_type": "FEDAVG",
        }
        assert {
            regime_id: regime["orchestration_type"]
            for regime_id, regime in result["regimes"].items()
        } == {
            "R1": "CENTRALIZED",
            "R2A": "FEDAVG",
            "R2B": "FEDADAM",
            "R3": "PERSONALIZED_FL",
            "R4": "STRICT_LOCAL",
            "R5": "LOCAL_ADAPTATION",
        }

    def test_regime_catalog_requires_exact_frozen_set(self) -> None:
        data_path = _ROOT / "config" / "experiments" / "regime_catalog.v1.json"
        data = json.loads(data_path.read_text(encoding="utf-8"))
        data["regimes"].pop("R5")

        with pytest.raises(ContractValidationError, match="frozen v1 set"):
            validate_regime_catalog(data)

    def test_regime_catalog_rejects_wrong_orchestration(self) -> None:
        data_path = _ROOT / "config" / "experiments" / "regime_catalog.v1.json"
        data = json.loads(data_path.read_text(encoding="utf-8"))
        data["regimes"]["R2B"]["orchestration_type"] = "FEDAVG"

        with pytest.raises(ContractValidationError, match="R2B.*FEDADAM"):
            validate_regime_catalog(data)

    def test_regime_catalog_json_schema_freezes_exact_contract(self) -> None:
        schema_path = _ROOT / "config" / "experiments" / "schemas" / "regime_catalog.v1.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        regimes = schema["properties"]["regimes"]
        expected = {
            "R1": "CENTRALIZED",
            "R2A": "FEDAVG",
            "R2B": "FEDADAM",
            "R3": "PERSONALIZED_FL",
            "R4": "STRICT_LOCAL",
            "R5": "LOCAL_ADAPTATION",
        }

        assert regimes["additionalProperties"] is False
        assert regimes["required"] == list(expected)
        assert set(regimes["properties"]) == set(expected)
        for regime_id, orchestration in expected.items():
            regime = regimes["properties"][regime_id]
            assert regime["properties"]["orchestration_type"]["const"] == orchestration

    def test_regime_catalog_rejects_unknown_fields(self) -> None:
        data = {
            "schema": "regime_catalog_v1",
            "version": "1",
            "regimes": {},
            "extra": "bad",
        }
        with pytest.raises(ContractValidationError, match="unknown fields"):
            validate_regime_catalog(data)

    def test_regime_catalog_rejects_wrong_schema(self) -> None:
        data = {
            "schema": "wrong_v1",
            "version": "1",
            "regimes": {},
        }
        with pytest.raises(ContractValidationError, match="schema"):
            validate_regime_catalog(data)


class TestArtifactMap:
    def test_artifact_map_validates(self) -> None:
        data_path = _ROOT / "config" / "experiments" / "experiment_artifact_map.v1.json"
        assert data_path.exists(), f"Missing {data_path}"
        data = json.loads(data_path.read_text(encoding="utf-8"))
        result = validate_experiment_artifact_map(data)
        assert isinstance(result["artifacts"], dict)

    def test_artifact_map_rejects_unknown_fields(self) -> None:
        data = {
            "schema": "experiment_artifact_map_v1",
            "version": "1",
            "artifacts": {},
            "extra": "bad",
        }
        with pytest.raises(ContractValidationError, match="unknown fields"):
            validate_experiment_artifact_map(data)


class TestContractEvidence:
    def test_real_evidence_validates_and_tampered_hash_fails(self, tmp_path: Path) -> None:
        evidence_dir = _EVIDENCE_DIR
        summary = json.loads(
            (evidence_dir / "experiment_contract_validation_summary.v1.json").read_text(
                encoding="utf-8"
            )
        )
        manifest = json.loads(
            (evidence_dir / "experiment_contract_artifact_manifest.v1.json").read_text(
                encoding="utf-8"
            )
        )

        validate_experiment_contract_validation_summary(summary)
        validate_experiment_contract_artifact_manifest(manifest)
        expected_fixture_paths = list(summary["validated_fixtures"])
        _verify_validation_summary(
            summary,
            validated_fixture_paths=expected_fixture_paths,
        )
        view = _prototype_layout_view(tmp_path)
        _verify_artifact_manifest(view, manifest)

        manifest["artifacts"]["RegimeCatalog"]["sha256"] = "0" * 64

        with pytest.raises(ContractValidationError, match="SHA-256 mismatch for RegimeCatalog"):
            _verify_artifact_manifest(view, manifest)

    def test_validation_summary_rejects_empty_claims(self) -> None:
        summary_path = _EVIDENCE_DIR / "experiment_contract_validation_summary.v1.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        validated_fixture_paths = list(summary["validated_fixtures"])

        for field in ("validated_schemas", "validated_fixtures"):
            tampered = {**summary, field: []}
            with pytest.raises(ContractValidationError, match="validation summary"):
                _verify_validation_summary(
                    tampered,
                    validated_fixture_paths=validated_fixture_paths,
                )
