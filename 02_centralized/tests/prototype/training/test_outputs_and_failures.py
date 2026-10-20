from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import nn

from ppsi.training.core import LocalTrainerCore, TrainerPolicy, TrainingCoreError
from ppsi.training.fixtures import default_batch_spec, make_phase1_batch, make_stub_model
from ppsi.training.objective import ContractSmokeObjective
from ppsi.training.outputs import (
    ObjectiveResult,
    RawModelOutput,
    StepStatus,
    SupportUnit,
    TaskLossComponent,
)
from ppsi.training.stub_model import Phase1StubModel


def _make_core(model, objective):
    return LocalTrainerCore(
        model=model,
        batch_spec=default_batch_spec(),
        objective=objective,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        device="cpu",
    )


def test_raw_output_shapes_are_finite_and_activation_free():
    batch = make_phase1_batch()
    model = make_stub_model()
    output = model(batch)
    assert tuple(output.t1_logits.shape) == (batch.batch_size, model.category_count)
    assert tuple(output.t2_logit.shape) == (batch.batch_size, 1)
    assert tuple(output.t3_scores.shape) == (batch.batch_size, batch.candidate_width)
    assert torch.isfinite(output.t1_logits).all()
    assert torch.isfinite(output.t2_logit).all()
    assert torch.isfinite(output.t3_scores).all()
    assert not any(
        isinstance(module, (nn.Sigmoid, nn.Softmax, nn.LogSoftmax)) for module in model.modules()
    )


def test_raw_output_values_are_not_probability_activated():
    batch = make_phase1_batch(batch_size=1)
    model = make_stub_model()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.context_projection.bias.fill_(2.0)
        model.candidate_projection.bias.fill_(2.0)
        model.t1_head.bias.copy_(torch.linspace(-3.0, 3.0, model.category_count))
        model.t2_head.bias.fill_(2.5)

    output = model(batch)
    torch.testing.assert_close(output.t1_logits[0], model.t1_head.bias, rtol=0, atol=0)
    torch.testing.assert_close(output.t2_logit, torch.full_like(output.t2_logit, 2.5), rtol=0, atol=0)
    assert torch.any(output.t1_logits < 0)
    assert torch.any(output.t1_logits > 1)
    assert torch.all(output.t3_scores > 1)


def test_candidate_axis_permutation_preserves_alignment():
    batch = make_phase1_batch()
    model = make_stub_model()
    model.eval()
    output = model(batch)
    permutation = torch.arange(batch.candidate_width - 1, -1, -1)
    permuted = replace(
        batch,
        candidate_ids=batch.candidate_ids[:, permutation],
        candidate_categorical_ids={
            name: values[:, permutation] for name, values in batch.candidate_categorical_ids.items()
        },
        candidate_continuous_features=batch.candidate_continuous_features[:, permutation],
        candidate_mask=batch.candidate_mask[:, permutation],
        t3_gains=batch.t3_gains[:, permutation],
    )
    permuted_output = model(permuted)
    torch.testing.assert_close(output.t1_logits, permuted_output.t1_logits, rtol=0, atol=0)
    torch.testing.assert_close(output.t2_logit, permuted_output.t2_logit, rtol=0, atol=0)
    torch.testing.assert_close(
        output.t3_scores[:, permutation],
        permuted_output.t3_scores,
        rtol=1e-6,
        atol=1e-7,
    )


class NonFiniteOutputModel(Phase1StubModel):
    def forward(self, batch):
        output = super().forward(batch)
        t1_logits = output.t1_logits.clone()
        t1_logits[0, 0] = float("nan")
        return RawModelOutput(t1_logits, output.t2_logit, output.t3_scores)


def test_nonfinite_raw_output_fails_closed():
    reference = make_stub_model()
    model = NonFiniteOutputModel(batch_spec=reference.batch_spec, config=reference.config)
    core = _make_core(model, ContractSmokeObjective())
    with pytest.raises(Exception, match="NaN/Inf"):
        core.train_step(make_phase1_batch())


class NonFiniteObjective:
    objective_id = "nonfinite_test_objective"

    def __call__(self, batch, output):
        numerator = output.t1_logits.sum() * float("nan")
        component = TaskLossComponent(numerator, 1, SupportUnit.EXAMPLES)
        return ObjectiveResult(
            status=StepStatus.NORMAL,
            total_loss=component.mean,
            task_losses={"T1": component},
            examples_processed=batch.batch_size,
            contributing_examples=1,
            contributing_tasks=("T1",),
        )


def test_nonfinite_contributing_loss_fails_closed():
    model = make_stub_model()
    core = _make_core(model, NonFiniteObjective())
    with pytest.raises(TrainingCoreError, match="loss contains NaN/Inf"):
        core.train_step(make_phase1_batch())


def test_nonfinite_gradient_fails_closed():
    model = make_stub_model()
    first_parameter = next(model.parameters())
    first_parameter.register_hook(lambda grad: torch.full_like(grad, float("nan")))
    core = _make_core(model, ContractSmokeObjective())
    with pytest.raises(TrainingCoreError, match="Gradient .* contains NaN/Inf"):
        core.train_step(make_phase1_batch())


def test_objective_is_defensive_for_all_masked_t3_tensor():
    batch = make_phase1_batch(batch_size=1)
    all_masked = replace(
        batch,
        t1_present=torch.zeros_like(batch.t1_present),
        t2_present=torch.zeros_like(batch.t2_present),
        t3_present=torch.ones_like(batch.t3_present),
        candidate_mask=torch.zeros_like(batch.candidate_mask),
    )
    model = make_stub_model()
    result = ContractSmokeObjective()(all_masked, model(all_masked))
    assert result.status == StepStatus.NO_CONTRIBUTING_TASK
    assert result.total_loss is None


def test_gradient_clipping_is_called_only_when_configured(monkeypatch):
    calls: list[float] = []
    original = nn.utils.clip_grad_norm_

    def spy(parameters, max_norm, *args, **kwargs):
        calls.append(float(max_norm))
        return original(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(nn.utils, "clip_grad_norm_", spy)

    model_without = make_stub_model()
    core_without = _make_core(model_without, ContractSmokeObjective())
    core_without.train_step(make_phase1_batch())
    assert calls == []

    model_with = make_stub_model()
    core_with = LocalTrainerCore(
        model=model_with,
        batch_spec=default_batch_spec(),
        objective=ContractSmokeObjective(),
        optimizer=torch.optim.SGD(model_with.parameters(), lr=0.01),
        policy=TrainerPolicy(gradient_clip_norm=0.25),
        device="cpu",
    )
    core_with.train_step(make_phase1_batch())
    assert calls == [0.25]
