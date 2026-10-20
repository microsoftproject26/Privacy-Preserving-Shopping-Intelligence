"""Tests for Compatibility v1."""

from __future__ import annotations

import pytest
from test_schemas import _make_experiment_config

from scripts.experiments.compatibility import (
    build_compatibility_tuple,
    compare_compatibility,
    validate_compatibility_tuple,
)
from scripts.experiments.contracts import ContractValidationError


def test_build_and_validate_compatibility_tuple() -> None:
    cfg = _make_experiment_config()
    git_sha = "f" * 40

    tup = build_compatibility_tuple(cfg, git_sha)
    assert tup["schema"] == "comparison_compatibility_v1"
    assert tup["git_sha"] == git_sha
    assert tup["tasks"] == ["T1"]
    assert tup["seed"] == 13
    assert (
        tup["common_initialization_sha256"]
        == cfg["initialization"]["common_initialization_ref"]["sha256"]
    )

    assert validate_compatibility_tuple(tup) == tup


def test_build_pretrained_compatibility_tuple() -> None:
    cfg = _make_experiment_config(
        regime="R5",
        initialization={
            "kind": "PRETRAINED_R1_CHECKPOINT",
            "pretrained_checkpoint_ref": {
                "schema": "artifact_ref_v1",
                "version": "1",
                "logical_id": "test_artifact",
                "artifact_schema": "model_config_v1",
                "artifact_version": "1",
                "uri": "foo",
                "sha256": "b" * 64,
            },
            "parent_run_id": "run-v1__r1__t1__c1__s13__cfg0123456789ab__a1",
        },
    )
    tup = build_compatibility_tuple(cfg, "f" * 40)
    assert tup["common_initialization_sha256"] == "0" * 64
    assert validate_compatibility_tuple(tup) == tup


def test_compare_compatibility_identical() -> None:
    cfg = _make_experiment_config()
    git_sha = "f" * 40
    tup1 = build_compatibility_tuple(cfg, git_sha)
    tup2 = build_compatibility_tuple(cfg, git_sha)

    result = compare_compatibility(tup1, tup2)
    assert result["is_compatible"] is True
    assert result["mismatches"] == []


def test_compare_compatibility_mismatch() -> None:
    cfg1 = _make_experiment_config()
    cfg2 = _make_experiment_config(seed=42)
    git_sha = "f" * 40

    tup1 = build_compatibility_tuple(cfg1, git_sha)
    tup2 = build_compatibility_tuple(cfg2, "e" * 40)

    result = compare_compatibility(tup1, tup2)
    assert result["is_compatible"] is False
    assert result["mismatches"] == ["git_sha", "seed"]


def test_invalid_compatibility_tuple_rejected() -> None:
    cfg = _make_experiment_config()
    tup = build_compatibility_tuple(cfg, "f" * 40)
    tup["extra"] = "bad"

    with pytest.raises(ContractValidationError, match="unknown"):
        validate_compatibility_tuple(tup)
