"""Downstream mapping and catalog tests for schemas.

Verifies that all frozen schema contracts are valid JSON and provide
expected metadata ($schema, $id, title) for artifact catalog mapping.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ppsi_paths import REPO_ROOT


def get_schema_files() -> list[Path]:
    """Find all JSON schemas in the config/experiments/schemas directory."""
    root = REPO_ROOT
    schemas_dir = root / "config" / "experiments" / "schemas"
    return list(schemas_dir.glob("*.schema.json"))


def test_schemas_directory_exists() -> None:
    root = REPO_ROOT
    schemas_dir = root / "config" / "experiments" / "schemas"
    assert schemas_dir.exists(), f"Schema directory not found: {schemas_dir}"
    assert schemas_dir.is_dir()


@pytest.mark.parametrize("schema_path", get_schema_files(), ids=lambda p: p.name)
def test_schema_valid_json_and_metadata(schema_path: Path) -> None:
    """Verify that each schema is valid JSON and has required metadata."""
    with open(schema_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "$schema" in data, f"{schema_path.name} missing $schema"
    assert "$id" in data, f"{schema_path.name} missing $id"
    assert "title" in data, f"{schema_path.name} missing title"
    assert "type" in data, f"{schema_path.name} missing type"

    # All our schemas represent objects
    assert data["type"] == "object"


def test_missing_schema_fails_canonical_validation(tmp_path: Path) -> None:
    """Prove that removing or failing to register a frozen schema fails validation."""
    from scripts.experiments.validate_experiment_contracts import _validate_schemas_loadable

    # Mock a fake root directory structure
    schema_dir = tmp_path / "config" / "experiments" / "schemas"
    schema_dir.mkdir(parents=True)

    # Write only one schema (so the others are missing)
    fake_schema = schema_dir / "experiment_config.v1.schema.json"
    fake_schema.write_text(
        json.dumps(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "properties": {"schema": {"const": "experiment_config_v1"}},
            }
        )
    )

    ok, fail = _validate_schemas_loadable(tmp_path)

    # Should report 1 ok (experiment_config_v1) but >0 fails because 12 are missing
    assert ok == 1
    assert fail > 0
