from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from ppsi.training.checkpoint import (
    CheckpointError,
    CheckpointIdentity,
    build_checkpoint_payload,
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from ppsi.training.core import LocalTrainerCore
from ppsi.training.fixtures import default_batch_spec, make_phase1_batch, make_stub_model
from ppsi.training.objective import ContractSmokeObjective
from ppsi.training.sampler import (
    DeterministicBatchSampler,
    LoaderContract,
    TrainingCursor,
    deterministic_epoch_batches,
)


def _identity(fill="a"):
    digest = fill * 64
    return CheckpointIdentity(
        experiment_config_sha256=digest,
        common_initialization_sha256="b" * 64,
        input_data_sha256="c" * 64,
        shared_trainer_core_sha256="d" * 64,
        objective_config_sha256="e" * 64,
        environment_lock_sha256="f" * 64,
        git_sha="1" * 40,
    )


def _loader_contract():
    return LoaderContract(
        schema="loader_contract_v1",
        version="1",
        dataset_identity_sha256="9" * 64,
        dataset_length=4,
        batch_size=1,
        drop_last=False,
        sampler_version="deterministic_epoch_sampler_v1",
        run_seed=13,
        num_workers=0,
    )


def _make_core():
    model = make_stub_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    core = LocalTrainerCore(
        model=model,
        batch_spec=default_batch_spec(),
        objective=ContractSmokeObjective(),
        optimizer=optimizer,
        scheduler=scheduler,
        device="cpu",
    )
    return core


def _assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for lval, rval in zip(left, right, strict=True):
            _assert_nested_equal(lval, rval)
    else:
        assert left == right


def test_deterministic_sampler_resumes_from_next_batch():
    contract = _loader_contract()
    all_batches = deterministic_epoch_batches(contract, epoch=0)
    resumed = list(DeterministicBatchSampler(contract, epoch=0, next_batch_index=1))
    assert resumed == all_batches[1:]


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("run_id", " "),
        ("attempt", True),
        ("best_criterion", float("nan")),
        ("best_criterion", True),
    ],
)
def test_checkpoint_builder_rejects_invalid_metadata(field: str, bad_value: object):
    core = _make_core()
    values = {
        "run_id": "run-v1__r1__t1__c1__s13__cfgaaaaaaaaaaaa__a1",
        "attempt": 1,
        "best_criterion": 1.0,
    }
    values[field] = bad_value

    with pytest.raises(ValueError, match=field):
        build_checkpoint_payload(
            run_id=values["run_id"],
            attempt=values["attempt"],
            model=core.model,
            optimizer=core.optimizer,
            scheduler=core.scheduler,
            grad_scaler=None,
            cursor=TrainingCursor(None, 0, 0, 0),
            best_criterion=values["best_criterion"],
            identity=_identity(),
            loader_contract=_loader_contract(),
            scheduler_step_unit="OPTIMIZER_STEP",
        )


def test_checkpoint_exact_next_batch_and_next_step_replay(tmp_path: Path):
    torch.manual_seed(13)
    contract = _loader_contract()
    order = deterministic_epoch_batches(contract, epoch=0)
    first_index = order[0][0]
    second_index = order[1][0]
    batches = {index: make_phase1_batch(seed=13 + index) for index in range(4)}

    original = _make_core()
    original.train_step(batches[first_index])
    cursor = TrainingCursor(None, 0, 1, original.optimizer_step_count)
    payload = build_checkpoint_payload(
        run_id="run-v1__r1__t1__c1__s13__cfgaaaaaaaaaaaa__a1",
        attempt=1,
        model=original.model,
        optimizer=original.optimizer,
        scheduler=original.scheduler,
        grad_scaler=None,
        cursor=cursor,
        best_criterion=1.0,
        identity=_identity(),
        loader_contract=contract,
        scheduler_step_unit="OPTIMIZER_STEP",
    )
    artifact = save_checkpoint(tmp_path / "checkpoint.pt", payload)

    expected_summary = original.train_step(batches[second_index])
    expected_model = deepcopy(original.model.state_dict())
    expected_optimizer = deepcopy(original.optimizer.state_dict())
    expected_scheduler = deepcopy(original.scheduler.state_dict())

    restored = _make_core()
    loaded = load_checkpoint(
        artifact.path,
        expected_sha256=artifact.sha256,
        expected_identity=_identity(),
        expected_loader_contract=contract,
        core=restored,
        grad_scaler=None,
        map_location="cpu",
    )
    assert restored.optimizer_step_count == loaded.cursor.optimizer_step == 1
    resumed_index = next(
        iter(
            DeterministicBatchSampler(
                contract,
                epoch=loaded.cursor.local_epoch,
                next_batch_index=loaded.cursor.next_batch_index,
            )
        )
    )[0]
    assert resumed_index == second_index
    replay_summary = restored.train_step(batches[resumed_index])

    assert replay_summary.total_loss == expected_summary.total_loss
    for key, tensor in expected_model.items():
        torch.testing.assert_close(tensor, restored.model.state_dict()[key], rtol=0, atol=0)
    _assert_nested_equal(expected_optimizer, restored.optimizer.state_dict())
    _assert_nested_equal(expected_scheduler, restored.scheduler.state_dict())


def test_hash_and_identity_mismatch_fail_before_resume(tmp_path: Path):
    core = _make_core()
    contract = _loader_contract()
    payload = build_checkpoint_payload(
        run_id="run-v1__r1__t1__c1__s13__cfgaaaaaaaaaaaa__a1",
        attempt=1,
        model=core.model,
        optimizer=core.optimizer,
        scheduler=core.scheduler,
        grad_scaler=None,
        cursor=TrainingCursor(None, 0, 0, 0),
        best_criterion=None,
        identity=_identity(),
        loader_contract=contract,
        scheduler_step_unit="OPTIMIZER_STEP",
    )
    artifact = save_checkpoint(tmp_path / "checkpoint.pt", payload)

    with pytest.raises(CheckpointError, match="SHA-256 mismatch"):
        load_checkpoint(
            artifact.path,
            expected_sha256="0" * 64,
            expected_identity=_identity(),
            expected_loader_contract=contract,
            core=core,
            grad_scaler=None,
        )

    with pytest.raises(CheckpointError, match="identity mismatch"):
        load_checkpoint(
            artifact.path,
            expected_sha256=artifact.sha256,
            expected_identity=_identity("0"),
            expected_loader_contract=contract,
            core=core,
            grad_scaler=None,
        )


def test_rng_capture_restore():
    import random

    import numpy as np

    random.seed(13)
    np.random.seed(13)
    torch.manual_seed(13)
    state = capture_rng_state()
    expected = (random.random(), float(np.random.rand()), float(torch.rand(())))
    restore_rng_state(state)
    actual = (random.random(), float(np.random.rand()), float(torch.rand(())))
    assert actual == expected


def test_failed_atomic_replace_preserves_previous_checkpoint(tmp_path: Path, monkeypatch):
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"previous-good-checkpoint")
    core = _make_core()
    payload = build_checkpoint_payload(
        run_id="run-v1__r1__t1__c1__s13__cfgaaaaaaaaaaaa__a1",
        attempt=1,
        model=core.model,
        optimizer=core.optimizer,
        scheduler=core.scheduler,
        grad_scaler=None,
        cursor=TrainingCursor(None, 0, 0, 0),
        best_criterion=None,
        identity=_identity(),
        loader_contract=_loader_contract(),
        scheduler_step_unit="OPTIMIZER_STEP",
    )

    def always_fail(src, dst):
        raise PermissionError("injected replace failure")

    monkeypatch.setattr("ppsi.training.checkpoint.os.replace", always_fail)
    with pytest.raises(PermissionError, match="injected"):
        save_checkpoint(path, payload)
    assert path.read_bytes() == b"previous-good-checkpoint"


def test_checkpoint_payload_is_a_snapshot_not_live_model_view():
    core = _make_core()
    before = {key: value.clone() for key, value in core.model.state_dict().items()}
    payload = build_checkpoint_payload(
        run_id="run-v1__r1__t1__c1__s13__cfgaaaaaaaaaaaa__a1",
        attempt=1,
        model=core.model,
        optimizer=core.optimizer,
        scheduler=core.scheduler,
        grad_scaler=None,
        cursor=TrainingCursor(None, 0, 0, 0),
        best_criterion=None,
        identity=_identity(),
        loader_contract=_loader_contract(),
        scheduler_step_unit="OPTIMIZER_STEP",
    )
    with torch.no_grad():
        for parameter in core.model.parameters():
            parameter.add_(10.0)
    for key, tensor in before.items():
        torch.testing.assert_close(payload["model_state_dict"][key], tensor, rtol=0, atol=0)


def test_cuda_rng_incompatibility_fails_before_model_mutation(tmp_path: Path, monkeypatch):
    source = _make_core()
    payload = build_checkpoint_payload(
        run_id="run-v1__r1__t1__c1__s13__cfgaaaaaaaaaaaa__a1",
        attempt=1,
        model=source.model,
        optimizer=source.optimizer,
        scheduler=source.scheduler,
        grad_scaler=None,
        cursor=TrainingCursor(None, 0, 0, 0),
        best_criterion=None,
        identity=_identity(),
        loader_contract=_loader_contract(),
        scheduler_step_unit="OPTIMIZER_STEP",
    )
    payload["rng_state"]["torch_cuda"] = [torch.tensor([1], dtype=torch.uint8)]
    artifact = save_checkpoint(tmp_path / "cuda-checkpoint.pt", payload)

    target = _make_core()
    before = {key: value.clone() for key, value in target.model.state_dict().items()}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(CheckpointError, match="CUDA RNG state"):
        load_checkpoint(
            artifact.path,
            expected_sha256=artifact.sha256,
            expected_identity=_identity(),
            expected_loader_contract=_loader_contract(),
            core=target,
            grad_scaler=None,
            strict_cuda_rng=True,
        )
    for key, tensor in before.items():
        torch.testing.assert_close(tensor, target.model.state_dict()[key], rtol=0, atol=0)


def test_late_checkpoint_failure_rolls_back_all_runtime_state(tmp_path: Path):
    torch.manual_seed(13)
    source = _make_core()
    source.train_step(make_phase1_batch(seed=42))
    payload = build_checkpoint_payload(
        run_id="run-v1__r1__t1__c1__s13__cfgaaaaaaaaaaaa__a1",
        attempt=1,
        model=source.model,
        optimizer=source.optimizer,
        scheduler=source.scheduler,
        grad_scaler=None,
        cursor=TrainingCursor(None, 0, 1, source.optimizer_step_count),
        best_criterion=None,
        identity=_identity(),
        loader_contract=_loader_contract(),
        scheduler_step_unit="OPTIMIZER_STEP",
    )
    payload["optimizer_state_dict"]["param_groups"] = []
    artifact = save_checkpoint(tmp_path / "late-failure.pt", payload)

    target = _make_core()
    target.optimizer_step_count = 7
    before_model = deepcopy(target.model.state_dict())
    before_optimizer = deepcopy(target.optimizer.state_dict())
    before_scheduler = deepcopy(target.scheduler.state_dict())
    before_rng = capture_rng_state()

    with pytest.raises(CheckpointError, match="runtime state was restored"):
        load_checkpoint(
            artifact.path,
            expected_sha256=artifact.sha256,
            expected_identity=_identity(),
            expected_loader_contract=_loader_contract(),
            core=target,
            grad_scaler=None,
        )

    _assert_nested_equal(before_model, target.model.state_dict())
    _assert_nested_equal(before_optimizer, target.optimizer.state_dict())
    _assert_nested_equal(before_scheduler, target.scheduler.state_dict())
    assert target.optimizer_step_count == 7
    after_rng = capture_rng_state()
    assert before_rng["python"] == after_rng["python"]
    assert before_rng["numpy"][0] == after_rng["numpy"][0]
    np.testing.assert_array_equal(before_rng["numpy"][1], after_rng["numpy"][1])
    assert before_rng["numpy"][2:] == after_rng["numpy"][2:]
    torch.testing.assert_close(before_rng["torch_cpu"], after_rng["torch_cpu"], rtol=0, atol=0)
    # A list of CUDA RNG-state tensors, one per device. `==` on tensors is elementwise, so
    # asserting on it raises "Boolean value of Tensor with more than one value is
    # ambiguous" - on a CUDA machine. On a CPU-only machine the list is empty and `==`
    # happens to work, which is why this passed everywhere it was run before.
    assert len(before_rng["torch_cuda"]) == len(after_rng["torch_cuda"])
    assert all(torch.equal(before, after) for before, after
               in zip(before_rng["torch_cuda"], after_rng["torch_cuda"], strict=True))


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("run_id", "", "run_id"),
        ("attempt", "not-an-int", "attempt"),
        ("best_criterion", float("nan"), "best_criterion"),
    ],
)
def test_invalid_checkpoint_metadata_fails_before_runtime_mutation(
    tmp_path: Path,
    field: str,
    bad_value: object,
    message: str,
):
    source = _make_core()
    source.train_step(make_phase1_batch(seed=42))
    payload = build_checkpoint_payload(
        run_id="run-v1__r1__t1__c1__s13__cfgaaaaaaaaaaaa__a1",
        attempt=1,
        model=source.model,
        optimizer=source.optimizer,
        scheduler=source.scheduler,
        grad_scaler=None,
        cursor=TrainingCursor(None, 0, 1, source.optimizer_step_count),
        best_criterion=1.0,
        identity=_identity(),
        loader_contract=_loader_contract(),
        scheduler_step_unit="OPTIMIZER_STEP",
    )
    payload[field] = bad_value
    artifact = save_checkpoint(tmp_path / f"bad-{field}.pt", payload)

    target = _make_core()
    target.optimizer_step_count = 7
    before_model = deepcopy(target.model.state_dict())
    before_optimizer = deepcopy(target.optimizer.state_dict())
    before_scheduler = deepcopy(target.scheduler.state_dict())
    before_rng = capture_rng_state()

    with pytest.raises(CheckpointError, match=message):
        load_checkpoint(
            artifact.path,
            expected_sha256=artifact.sha256,
            expected_identity=_identity(),
            expected_loader_contract=_loader_contract(),
            core=target,
            grad_scaler=None,
        )

    _assert_nested_equal(before_model, target.model.state_dict())
    _assert_nested_equal(before_optimizer, target.optimizer.state_dict())
    _assert_nested_equal(before_scheduler, target.scheduler.state_dict())
    assert target.optimizer_step_count == 7
    after_rng = capture_rng_state()
    assert before_rng["python"] == after_rng["python"]
    assert before_rng["numpy"][0] == after_rng["numpy"][0]
    np.testing.assert_array_equal(before_rng["numpy"][1], after_rng["numpy"][1])
    assert before_rng["numpy"][2:] == after_rng["numpy"][2:]
    torch.testing.assert_close(before_rng["torch_cpu"], after_rng["torch_cpu"], rtol=0, atol=0)
    # A list of CUDA RNG-state tensors, one per device. `==` on tensors is elementwise, so
    # asserting on it raises "Boolean value of Tensor with more than one value is
    # ambiguous" - on a CUDA machine. On a CPU-only machine the list is empty and `==`
    # happens to work, which is why this passed everywhere it was run before.
    assert len(before_rng["torch_cuda"]) == len(after_rng["torch_cuda"])
    assert all(torch.equal(before, after) for before, after
               in zip(before_rng["torch_cuda"], after_rng["torch_cuda"], strict=True))
