from __future__ import annotations

import hashlib
import json
import shutil
from copy import deepcopy
from pathlib import Path

import pytest

from ppsi.training.identity import (
    DEFAULT_TRAINER_CORE_FILES,
    build_trainer_core_manifest,
    canonical_manifest_bytes,
    verify_trainer_core_manifest,
)
from ppsi.training.result import build_experiment_result, make_metric_record
from ppsi_paths import REPO_ROOT, prototype_path


def _load_config(repo_root: Path) -> dict:
    return json.loads(
        (repo_root / "fixtures/experiments/contracts/experiment_config_r1.json").read_text(
            encoding="utf-8"
        )
    )


def _config_ref(config: dict) -> dict:
    return {
        **config["source_dataset_ref"],
        "logical_id": "ExperimentConfigFixture",
        "artifact_schema": "experiment_config_v1",
    }


def test_success_and_failure_records_validate_against_existing_contract():
    repo_root = REPO_ROOT
    config = _load_config(repo_root)
    metric = make_metric_record(
        metric_id="contract_smoke_t1_loss",
        task="T1",
        cohort="C1",
        value=0.5,
        direction="MINIMIZE",
        unit="UNITLESS",
        support=3,
    )
    success = build_experiment_result(
        experiment_config=config,
        config_ref=_config_ref(config),
        git_sha="f" * 40,
        state="SUCCEEDED",
        attempt=1,
        started_at_utc="2026-08-27T00:00:00Z",
        ended_at_utc="2026-08-27T00:00:01Z",
        metrics=[metric],
        artifacts=[],
    )
    assert success["schema"] == "experiment_result_v1"
    assert success["state"] == "SUCCEEDED"

    failure = build_experiment_result(
        experiment_config=config,
        config_ref=_config_ref(config),
        git_sha="f" * 40,
        state="FAILED",
        attempt=1,
        started_at_utc="2026-08-27T00:00:00Z",
        ended_at_utc="2026-08-27T00:00:01Z",
        metrics=[],
        artifacts=[],
        failure={"type": "TrainingCoreError", "message": "injected failure"},
    )
    assert failure["state"] == "FAILED"
    assert failure["failure"]["type"] == "TrainingCoreError"


def test_trainer_core_manifest_changes_only_for_behavior_bundle(tmp_path: Path):
    for relative in DEFAULT_TRAINER_CORE_FILES:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(prototype_path(relative), destination)

    before = build_trainer_core_manifest(tmp_path)
    verify_trainer_core_manifest(tmp_path, before)

    unrelated = tmp_path / "docs/unrelated.md"
    unrelated.parent.mkdir(parents=True, exist_ok=True)
    unrelated.write_text("not behavior defining", encoding="utf-8")
    after_unrelated = build_trainer_core_manifest(tmp_path)
    assert after_unrelated["sha256"] == before["sha256"]

    behavior_file = tmp_path / DEFAULT_TRAINER_CORE_FILES[0]
    behavior_file.write_text(
        behavior_file.read_text(encoding="utf-8") + "\n# semantic change under review\n",
        encoding="utf-8",
    )
    after_behavior = build_trainer_core_manifest(tmp_path)
    assert after_behavior["sha256"] != before["sha256"]


def test_trainer_core_manifest_requires_exact_file_set(tmp_path: Path):
    for relative in DEFAULT_TRAINER_CORE_FILES:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(prototype_path(relative), destination)

    manifest = build_trainer_core_manifest(tmp_path)
    missing = deepcopy(manifest)
    missing["files"] = missing["files"][:-1]
    missing_without_sha = {key: value for key, value in missing.items() if key != "sha256"}
    missing["sha256"] = hashlib.sha256(canonical_manifest_bytes(missing_without_sha)).hexdigest()
    with pytest.raises(ValueError, match="file set does not match"):
        verify_trainer_core_manifest(tmp_path, missing)
