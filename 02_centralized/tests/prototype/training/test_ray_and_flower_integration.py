from __future__ import annotations

import importlib
from copy import deepcopy

import pytest
import ray
import torch
from flwr.app import ArrayRecord

from ppsi.training.centralized import CentralizedAdapter
from ppsi.training.core import LocalTrainerCore
from ppsi.training.fixtures import (
    default_batch_spec,
    make_phase1_batch,
    make_stub_model,
)
from ppsi.training.flower import (
    ContributingRowsSmokeWeightPolicy,
    FlowerLocalAdapter,
)
from ppsi.training.objective import ContractSmokeObjective
from ppsi.training.sampler import TrainingCursor
from ppsi.training.state import load_shared_state, pack_shared_state
from scripts import generate_training_artifact_manifest, training_smoke
from scripts.federated.fl_synthetic_smoke import SmokeValidationError
from scripts.training_smoke import execute, validate_config


def _core(model):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    return LocalTrainerCore(
        model=model,
        batch_spec=default_batch_spec(),
        objective=ContractSmokeObjective(),
        optimizer=optimizer,
        scheduler=scheduler,
        device="cpu",
    )


def test_reusable_training_module_imports_in_real_ray_worker():
    ray.init(num_cpus=1, include_dashboard=False, ignore_reinit_error=True)

    @ray.remote
    def worker_import():
        module = importlib.import_module("ppsi.training")
        return module.Phase1Batch.__name__

    try:
        assert ray.get(worker_import.remote()) == "Phase1Batch"
    finally:
        ray.shutdown()


def test_arrayrecord_round_trip_preserves_centralized_flower_step_parity():
    batch = make_phase1_batch()
    source = make_stub_model(seed=13)
    initial = pack_shared_state(source, source.shared_state_spec())
    array_record = ArrayRecord.from_torch_state_dict(initial)
    transported = array_record.to_torch_state_dict()

    centralized_model = make_stub_model(seed=42)
    load_shared_state(
        centralized_model,
        transported,
        centralized_model.shared_state_spec(),
    )
    centralized_core = _core(centralized_model)
    centralized_summary = CentralizedAdapter(centralized_core).train_batches(
        [batch],
        cursor=TrainingCursor(None, 0, 0, 0),
    )
    centralized_state = pack_shared_state(
        centralized_model,
        centralized_model.shared_state_spec(),
    )

    flower_model = make_stub_model(seed=2026)
    flower_core = _core(flower_model)
    flower_result = FlowerLocalAdapter(
        flower_core,
        flower_model.shared_state_spec(),
        ContributingRowsSmokeWeightPolicy(),
    ).fit(transported, [batch], outer_round=1)

    assert centralized_summary.contributing_examples == flower_result.aggregation_weight
    for task, stat in centralized_summary.task_stats.items():
        other = flower_result.summary.task_stats[task]
        assert stat == other
    for key, value in centralized_state.items():
        torch.testing.assert_close(
            value,
            flower_result.shared_state[key],
            rtol=0,
            atol=0,
        )


def test_unified_trainer_runs_through_real_flower_three_round_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
):
    summary = execute(
        {
            "schema": "unified_trainer_smoke_v1",
            "version": "1",
            "seed": 13,
            "num_clients": 2,
            "num_rounds": 3,
            "repeat_runs": 2,
            "learning_rate": 0.01,
            "tolerance": 1e-6,
        }
    )
    assert summary["status"] == "PASS"
    assert summary["identities"]["source_git_sha"]
    assert summary["aggregation_parity"]["atol"] == 1e-6
    assert summary["aggregation_parity"]["rtol"] == 0.0
    round_diffs = [
        record["max_abs_diff"]
        for repetition in summary["repetitions"]
        for record in repetition["tracing_log"]
    ]
    assert summary["aggregation_parity"]["max_abs_diff"] == max(round_diffs)
    assert len(summary["repetitions"]) == 2
    for run_index, repetition in enumerate(summary["repetitions"]):
        assert repetition["run"] == run_index
        assert repetition["selected_client_ids"] == [[0, 1], [0, 1], [0, 1]]
        assert [record["round"] for record in repetition["tracing_log"]] == [1, 2, 3]
        for record in repetition["tracing_log"]:
            assert record["selected_client_count"] == 2
            assert record["selected_logical_ids"] == [0, 1]
            assert [client["logical_client_id"] for client in record["clients"]] == [0, 1]
    assert (
        summary["repetitions"][0]["final_model_digest"]
        == summary["repetitions"][1]["final_model_digest"]
    )

    monkeypatch.setattr(
        generate_training_artifact_manifest,
        "_require_source_git_sha",
        lambda value: value,
    )
    generate_training_artifact_manifest.validate_smoke_summary(summary)

    missing_round = deepcopy(summary)
    missing_round["repetitions"][0]["tracing_log"].pop()
    with pytest.raises(ValueError, match="round count"):
        generate_training_artifact_manifest.validate_smoke_summary(missing_round)

    wrong_clients = deepcopy(summary)
    wrong_clients["repetitions"][0]["tracing_log"][0]["selected_client_count"] = 1
    with pytest.raises(ValueError, match="exactly two clients"):
        generate_training_artifact_manifest.validate_smoke_summary(wrong_clients)

    wrong_client_order = deepcopy(summary)
    wrong_client_order["repetitions"][0]["tracing_log"][0]["clients"].reverse()
    with pytest.raises(ValueError, match="clients 0 and 1 in order"):
        generate_training_artifact_manifest.validate_smoke_summary(wrong_client_order)

    wrong_max = deepcopy(summary)
    wrong_max["aggregation_parity"]["max_abs_diff"] = 0.0
    with pytest.raises(ValueError, match="measured round maximum"):
        generate_training_artifact_manifest.validate_smoke_summary(wrong_max)

    broken_chain = deepcopy(summary)
    broken_chain["repetitions"][0]["tracing_log"][1]["server_input_digest"] = "a" * 64
    with pytest.raises(ValueError, match="redistribution chain"):
        generate_training_artifact_manifest.validate_smoke_summary(broken_chain)

    wrong_client_digest = deepcopy(summary)
    wrong_client_digest["repetitions"][0]["tracing_log"][0]["clients"][0]["received_digest"] = (
        "b" * 64
    )
    with pytest.raises(ValueError, match="match the round server input"):
        generate_training_artifact_manifest.validate_smoke_summary(wrong_client_digest)

    wrong_final_digest = deepcopy(summary)
    wrong_final_digest["repetitions"][0]["final_model_digest"] = "f" * 64
    with pytest.raises(ValueError, match="final aggregated digest"):
        generate_training_artifact_manifest.validate_smoke_summary(wrong_final_digest)

    wrong_loss_rounds = deepcopy(summary)
    first_loss = wrong_loss_rounds["repetitions"][0]["global_loss_history"][0]
    wrong_loss_rounds["repetitions"][0]["global_loss_history"][0] = (99, first_loss[1])
    with pytest.raises(ValueError, match="zero through num_rounds"):
        generate_training_artifact_manifest.validate_smoke_summary(wrong_loss_rounds)


def test_unified_trainer_smoke_v1_requires_exactly_two_clients():
    config = {
        "schema": "unified_trainer_smoke_v1",
        "version": "1",
        "seed": 13,
        "num_clients": 3,
        "num_rounds": 3,
        "repeat_runs": 2,
        "learning_rate": 0.01,
        "tolerance": 1e-6,
    }
    with pytest.raises(SmokeValidationError, match="num_clients == 2"):
        validate_config(config)


def test_unified_trainer_smoke_v1_keeps_frozen_parity_tolerance():
    config = {
        "schema": "unified_trainer_smoke_v1",
        "version": "1",
        "seed": 13,
        "num_clients": 2,
        "num_rounds": 3,
        "repeat_runs": 2,
        "learning_rate": 0.01,
        "tolerance": 1e-5,
    }
    with pytest.raises(SmokeValidationError, match="tolerance == 1e-6"):
        validate_config(config)


def test_canonical_evidence_rejects_untracked_files(monkeypatch: pytest.MonkeyPatch):
    source_git_sha = "a" * 40

    class GitResult:
        returncode = 0

    monkeypatch.setattr(training_smoke, "_git_sha", lambda: source_git_sha)
    monkeypatch.setattr(training_smoke.subprocess, "run", lambda *args, **kwargs: GitResult())

    def fake_check_output(command, **kwargs):
        assert command == ["git", "status", "--porcelain", "--untracked-files=all"]
        return "?? local-only.py\n"

    monkeypatch.setattr(training_smoke.subprocess, "check_output", fake_check_output)

    with pytest.raises(SmokeValidationError, match="clean working tree"):
        training_smoke.validate_source_git_sha(source_git_sha, require_clean_head=True)
