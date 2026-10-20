from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from ppsi.training.centralized import CentralizedAdapter
from ppsi.training.core import LocalTrainerCore
from ppsi.training.fixtures import (
    default_batch_spec,
    make_no_contribution_batch,
    make_phase1_batch,
    make_stub_model,
)
from ppsi.training.flower import (
    ContributingRowsSmokeWeightPolicy,
    FlowerLocalAdapter,
)
from ppsi.training.objective import ContractSmokeObjective
from ppsi.training.outputs import StepStatus
from ppsi.training.sampler import TrainingCursor
from ppsi.training.state import load_shared_state, pack_shared_state


def _make_core(model, lr=0.01, scheduler=False):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    lr_scheduler = (
        torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9) if scheduler else None
    )
    return LocalTrainerCore(
        model=model,
        batch_spec=default_batch_spec(),
        objective=ContractSmokeObjective(),
        optimizer=optimizer,
        scheduler=lr_scheduler,
        device="cpu",
    )


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


def test_train_updates_and_eval_does_not():
    batch = make_phase1_batch()
    model = make_stub_model()
    core = _make_core(model)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    summary = core.train_step(batch)
    assert summary.optimizer_step_performed
    assert any(not torch.equal(before[key], value) for key, value in model.state_dict().items())

    before_eval = {key: value.clone() for key, value in model.state_dict().items()}
    eval_summary = core.eval_step(batch)
    assert not eval_summary.optimizer_step_performed
    for key, value in model.state_dict().items():
        torch.testing.assert_close(before_eval[key], value, rtol=0, atol=0)


def test_no_contribution_does_not_touch_model_optimizer_or_scheduler():
    model = make_stub_model()
    core = _make_core(model, scheduler=True)
    before_model = deepcopy(model.state_dict())
    before_optimizer = deepcopy(core.optimizer.state_dict())
    before_scheduler = deepcopy(core.scheduler.state_dict())
    summary = core.train_step(make_no_contribution_batch())
    assert summary.status == StepStatus.NO_CONTRIBUTING_TASK
    assert not summary.optimizer_step_performed
    assert core.optimizer_step_count == 0
    assert core.optimizer.state_dict() == before_optimizer
    assert core.scheduler.state_dict() == before_scheduler
    for key, value in model.state_dict().items():
        torch.testing.assert_close(before_model[key], value, rtol=0, atol=0)


def test_nonfinite_valid_input_fails_closed():
    batch = make_phase1_batch()
    continuous = batch.history_continuous_features.clone()
    continuous[0, 0, 0] = float("nan")
    bad = replace(batch, history_continuous_features=continuous)
    core = _make_core(make_stub_model())
    with pytest.raises(Exception, match="NaN/Inf"):
        core.train_step(bad)


def test_zero_history_representation_is_exact_zero_and_query_still_matters():
    batch = make_phase1_batch(batch_size=1)
    empty = replace(
        batch,
        lengths=torch.zeros_like(batch.lengths),
        history_mask=torch.zeros_like(batch.history_mask),
        history_categorical_ids={
            name: torch.zeros_like(values) for name, values in batch.history_categorical_ids.items()
        },
        history_continuous_features=torch.zeros_like(batch.history_continuous_features),
    )
    model = make_stub_model()
    history = model.encode_history(empty)
    assert torch.equal(history, torch.zeros_like(history))

    changed_query = replace(
        empty,
        query_continuous_features=empty.query_continuous_features + 1.0,
    )
    output_a = model(empty)
    output_b = model(changed_query)
    assert not torch.equal(output_a.t1_logits, output_b.t1_logits)


def test_centralized_and_flower_actual_local_step_are_identical():
    torch.manual_seed(13)
    batch = make_phase1_batch()
    initial_model = make_stub_model()
    initial_spec = initial_model.shared_state_spec()
    initial_state = pack_shared_state(initial_model, initial_spec)

    centralized_model = make_stub_model(seed=42)
    load_shared_state(centralized_model, initial_state, centralized_model.shared_state_spec())
    centralized_core = _make_core(centralized_model, scheduler=True)

    flower_model = make_stub_model(seed=2026)
    load_shared_state(flower_model, initial_state, flower_model.shared_state_spec())
    flower_core = _make_core(flower_model, scheduler=True)

    with torch.no_grad():
        centralized_before = centralized_model(batch)
        flower_before = flower_model(batch)
    torch.testing.assert_close(
        centralized_before.t1_logits, flower_before.t1_logits, rtol=0, atol=0
    )
    torch.testing.assert_close(centralized_before.t2_logit, flower_before.t2_logit, rtol=0, atol=0)
    torch.testing.assert_close(
        centralized_before.t3_scores, flower_before.t3_scores, rtol=0, atol=0
    )

    centralized = CentralizedAdapter(centralized_core)
    centralized_summary = centralized.train_batches(
        [batch],
        cursor=TrainingCursor(None, 0, 0, 0),
    )
    centralized_state = pack_shared_state(centralized_model, centralized_model.shared_state_spec())

    flower = FlowerLocalAdapter(
        flower_core,
        flower_model.shared_state_spec(),
        ContributingRowsSmokeWeightPolicy(),
    )
    flower_result = flower.fit(initial_state, [batch], outer_round=1)

    assert centralized_summary.contributing_examples == flower_result.aggregation_weight
    assert centralized_summary.task_stats.keys() == flower_result.summary.task_stats.keys()
    for task in centralized_summary.task_stats:
        left = centralized_summary.task_stats[task]
        right = flower_result.summary.task_stats[task]
        assert left.denominator == right.denominator
        assert left.support_unit == right.support_unit
        assert left.numerator == right.numerator
    assert centralized_state.keys() == flower_result.shared_state.keys()
    for key in centralized_state:
        torch.testing.assert_close(
            centralized_state[key], flower_result.shared_state[key], rtol=0, atol=0
        )
    _assert_nested_equal(
        centralized_core.optimizer.state_dict(), flower_core.optimizer.state_dict()
    )
    _assert_nested_equal(
        centralized_core.scheduler.state_dict(), flower_core.scheduler.state_dict()
    )
