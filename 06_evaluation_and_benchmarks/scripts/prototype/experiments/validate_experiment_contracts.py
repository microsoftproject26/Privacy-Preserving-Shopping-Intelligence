"""Validate all frozen S1-PR-03 experiment contracts.

Canonical entry point:
    uv run --locked python scripts/experiments/validate_experiment_contracts.py

Validates:
- JSON schemas load correctly
- Fixture records pass strict Python validators
- Catalogs and maps pass validation
- Contract freeze record passes validation
- Evidence records match their frozen schemas and declared artifact hashes
- Schema set completeness

Exits nonzero on any mismatch.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path when running as a script.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.experiments.contracts import ContractValidationError, canonical_json_bytes
from scripts.experiments.schemas import (
    validate_common_initialization,
    validate_experiment_artifact_map,
    validate_experiment_config,
    validate_experiment_contract_artifact_manifest,
    validate_experiment_contract_freeze,
    validate_experiment_contract_validation_summary,
    validate_experiment_result,
    validate_metric_record,
    validate_model_config,
    validate_regime_catalog,
    validate_system_measurement_reference_set,
)

# Complete frozen schema ID → validator mapping.
_VALIDATORS: dict[str, Callable[[Any], dict[str, Any]]] = {
    "model_config_v1": validate_model_config,
    "experiment_config_v1": validate_experiment_config,
    "metric_record_v1": validate_metric_record,
    "system_measurement_reference_set_v1": validate_system_measurement_reference_set,
    "experiment_result_v1": validate_experiment_result,
    "common_initialization_v1": validate_common_initialization,
    "regime_catalog_v1": validate_regime_catalog,
    "experiment_artifact_map_v1": validate_experiment_artifact_map,
    "experiment_contract_validation_summary_v1": validate_experiment_contract_validation_summary,
    "experiment_contract_artifact_manifest_v1": validate_experiment_contract_artifact_manifest,
    "experiment_contract_freeze_v1": validate_experiment_contract_freeze,
}

# The complete frozen S1-PR-03 schema set.
FROZEN_SCHEMA_IDS: frozenset[str] = frozenset(
    {
        "artifact_ref_v1",
        "model_config_v1",
        "experiment_config_v1",
        "experiment_result_v1",
        "metric_record_v1",
        "system_measurement_reference_set_v1",
        "common_initialization_v1",
        "comparison_compatibility_v1",
        "experiment_contract_freeze_v1",
        "experiment_artifact_map_v1",
        "regime_catalog_v1",
        "experiment_contract_validation_summary_v1",
        "experiment_contract_artifact_manifest_v1",
    }
)

_EVIDENCE_DIR = Path("docs/evidence/s1-pr-03")
_ARTIFACT_MANIFEST = _EVIDENCE_DIR / "experiment_contract_artifact_manifest.v1.json"
_VALIDATION_SUMMARY = _EVIDENCE_DIR / "experiment_contract_validation_summary.v1.json"
_TEXT_SUFFIXES = frozenset({".md", ".py", ".toml", ".txt", ".yaml", ".yml"})


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _file_sha256(path: Path) -> str:
    """Hash JSON canonically, normalize known text, and preserve binary bytes."""
    if path.suffix.lower() == ".json":
        data = canonical_json_bytes(_read_json(path))
    else:
        data = path.read_bytes()
        if path.suffix.lower() in _TEXT_SUFFIXES:
            text = data.decode("utf-8-sig")
            data = text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _artifact_sha256(
    path: Path,
    *,
    root: Path,
    excluded_path: Path | None = None,
) -> str:
    """Hash one artifact or a deterministic tree of its canonical file hashes."""

    if path.is_file():
        return _file_sha256(path)
    if not path.is_dir():
        raise FileNotFoundError(path)

    files = sorted(
        (item for item in path.rglob("*") if item.is_file() and item != excluded_path),
        key=lambda item: item.relative_to(root).as_posix(),
    )

    digest = hashlib.sha256()
    for file_path in files:
        relative = file_path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(_file_sha256(file_path).encode("ascii"))
    return digest.hexdigest()


def _verify_artifact_manifest(root: Path, manifest: Any | None = None) -> None:
    """Validate the real S1-PR-03 manifest and every referenced content hash."""

    manifest_path = root / _ARTIFACT_MANIFEST
    if manifest is None:
        manifest = _read_json(manifest_path)
    manifest = validate_experiment_contract_artifact_manifest(manifest)

    artifact_map = validate_experiment_artifact_map(
        _read_json(root / "config" / "experiments" / "experiment_artifact_map.v1.json")
    )
    expected_paths = {
        logical_id: entry["path"] for logical_id, entry in artifact_map["artifacts"].items()
    }
    records = manifest["artifacts"]
    if set(records) != set(expected_paths):
        raise ContractValidationError("artifact manifest logical IDs do not match artifact map")

    for logical_id, expected_path in sorted(expected_paths.items()):
        record = records[logical_id]
        if record["path"] != expected_path:
            raise ContractValidationError(
                f"artifact manifest path mismatch for {logical_id}: "
                f"{record['path']} != {expected_path}"
            )

        artifact_path = root / expected_path
        # Decision 13 freezes the manifest last and forbids self-hashing.
        excluded_path = manifest_path if artifact_path == manifest_path.parent else None
        actual_hash = _artifact_sha256(
            artifact_path,
            root=root,
            excluded_path=excluded_path,
        )
        if actual_hash != record["sha256"]:
            raise ContractValidationError(
                f"artifact SHA-256 mismatch for {logical_id}: {actual_hash} != {record['sha256']}"
            )


def _verify_validation_summary(
    summary: Any,
    *,
    validated_fixture_paths: list[str],
) -> None:
    """Verify that the stored PASS summary names exactly what was validated."""

    summary = validate_experiment_contract_validation_summary(summary)
    if summary["status"] != "PASS":
        raise ContractValidationError("validation summary status must be PASS")

    expected_schema_ids = sorted(FROZEN_SCHEMA_IDS)
    if summary["validated_schemas"] != expected_schema_ids:
        raise ContractValidationError(
            "validation summary schema IDs do not match the frozen schema set"
        )

    expected_fixture_paths = sorted(validated_fixture_paths)
    if summary["validated_fixtures"] != expected_fixture_paths:
        raise ContractValidationError(
            "validation summary fixture paths do not match the validated records"
        )


def _validate_schemas_loadable(root: Path) -> tuple[int, int]:
    """Validate all JSON schema files load as valid JSON with required fields."""
    schema_dir = root / "config" / "experiments" / "schemas"
    if not schema_dir.exists():
        print("ERROR: schemas directory does not exist")
        return 0, 1

    ok = 0
    fail = 0

    found_schemas = set()
    for path in sorted(schema_dir.glob("*.json")):
        try:
            data = _read_json(path)
            assert isinstance(data, dict), "Schema must be a dict"
            assert "$schema" in data or "schema" in data, "Missing schema identifier"
            schema_id = data.get("properties", {}).get("schema", {}).get("const")
            if schema_id:
                found_schemas.add(schema_id)
            print(f"  SCHEMA OK: {path.name}")
            ok += 1
        except (json.JSONDecodeError, AssertionError, KeyError, ValueError) as e:
            print(f"  SCHEMA FAIL: {path.name} — {e}")
            fail += 1

    missing = FROZEN_SCHEMA_IDS - found_schemas
    extra = found_schemas - FROZEN_SCHEMA_IDS

    if missing:
        print(f"  ERROR: Missing schema files for {missing}")
        fail += len(missing)
    if extra:
        print(f"  ERROR: Extra/unknown schema files for {extra}")
        fail += len(extra)

    return ok, fail


def _validate_fixtures(
    root: Path,
    *,
    validated_fixture_paths: list[str] | None = None,
) -> tuple[int, int]:
    """Validate all fixture, config, and evidence JSON records."""
    directories = [
        root / "fixtures" / "experiments" / "contracts",
        root / "config" / "experiments",
        root / _EVIDENCE_DIR,
    ]

    ok = 0
    fail = 0
    validated_paths: list[str] = []
    for directory in directories:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.json")):
            try:
                data = _read_json(path)
                schema_name = data.get("schema")
                if not schema_name:
                    print(f"  SKIP: {path.name} (no schema field)")
                    continue
                if schema_name not in _VALIDATORS:
                    print(f"  FAIL: {path.name} — no validator for {schema_name}")
                    fail += 1
                    continue
                validated = _VALIDATORS[schema_name](data)
                if (
                    schema_name == "experiment_contract_validation_summary_v1"
                    and validated["status"] != "PASS"
                ):
                    raise ContractValidationError("validation summary status must be PASS")
                print(f"  FIXTURE OK: {path.name} [{schema_name}]")
                ok += 1
                validated_paths.append(path.relative_to(root).as_posix())
            except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
                print(f"  FIXTURE FAIL: {path.name} — {e}")
                fail += 1
    if validated_fixture_paths is not None:
        validated_fixture_paths.extend(sorted(validated_paths))
    return ok, fail


def main() -> int:
    root = Path(__file__).parent.parent.parent
    errors = 0

    print("=" * 60)
    print("S1-PR-03 Experiment Contract Validation")
    print("=" * 60)

    # 1. Schema files
    print("\n[1] JSON Schema Files")
    s_ok, s_fail = _validate_schemas_loadable(root)
    errors += s_fail
    print(f"    {s_ok} OK, {s_fail} FAIL")

    # 2. Fixture/config records
    print("\n[2] Fixture & Config Records")
    validated_fixture_paths: list[str] = []
    f_ok, f_fail = _validate_fixtures(
        root,
        validated_fixture_paths=validated_fixture_paths,
    )
    errors += f_fail
    print(f"    {f_ok} OK, {f_fail} FAIL")

    # 3. Evidence records and referenced artifact hashes
    print("\n[3] Evidence Integrity")
    try:
        _verify_validation_summary(
            _read_json(root / _VALIDATION_SUMMARY),
            validated_fixture_paths=validated_fixture_paths,
        )
        _verify_artifact_manifest(root)
        print("  EVIDENCE OK: validation summary, artifact manifest, and referenced hashes")
    except (json.JSONDecodeError, OSError, ValueError, KeyError, TypeError) as e:
        print(f"  EVIDENCE FAIL: {e}")
        errors += 1

    # 4. Schema set completeness
    print("\n[4] Frozen Schema Set Completeness")
    print(f"    Expected {len(FROZEN_SCHEMA_IDS)} schema IDs")
    for sid in sorted(FROZEN_SCHEMA_IDS):
        print(f"      ✓ {sid}")

    # 5. Summary
    print("\n" + "=" * 60)
    if errors > 0:
        print(f"FAILED — {errors} error(s)")
        return 1
    print(f"PASSED — {s_ok} schemas, {f_ok} fixtures validated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
