from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import torch

from ppsi.training.fixtures import make_phase1_batch, make_stub_model
from ppsi.training.objective import ContractSmokeObjective


def _loss_and_gradients(model, batch):
    model.zero_grad(set_to_none=True)
    output = model(batch)
    result = ContractSmokeObjective()(batch, output)
    assert result.total_loss is not None
    result.total_loss.backward()
    grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return output, result.total_loss.detach().clone(), grads


def _assert_gradients_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        torch.testing.assert_close(left[key], right[key], rtol=0.0, atol=0.0)


def test_history_padding_poison_does_not_change_output_loss_or_gradients():
    batch = make_phase1_batch()
    poisoned = replace(
        batch,
        history_categorical_ids={
            name: values.clone() for name, values in batch.history_categorical_ids.items()
        },
        history_continuous_features=batch.history_continuous_features.clone(),
    )
    for values in poisoned.history_categorical_ids.values():
        values[~batch.history_mask] = 1
    poisoned.history_continuous_features[~batch.history_mask] = 999.0

    model_a = make_stub_model()
    model_b = deepcopy(model_a)
    output_a, loss_a, grads_a = _loss_and_gradients(model_a, batch)
    output_b, loss_b, grads_b = _loss_and_gradients(model_b, poisoned)
    torch.testing.assert_close(output_a.t1_logits, output_b.t1_logits, rtol=0, atol=0)
    torch.testing.assert_close(output_a.t2_logit, output_b.t2_logit, rtol=0, atol=0)
    torch.testing.assert_close(output_a.t3_scores, output_b.t3_scores, rtol=0, atol=0)
    torch.testing.assert_close(loss_a, loss_b, rtol=0, atol=0)
    _assert_gradients_equal(grads_a, grads_b)


def test_candidate_mask_poison_does_not_change_loss_or_gradients():
    batch = make_phase1_batch()
    candidate_ids = batch.candidate_ids.clone()
    candidate_ids[~batch.candidate_mask] = 7
    candidate_channels = {
        name: values.clone() for name, values in batch.candidate_categorical_ids.items()
    }
    for values in candidate_channels.values():
        values[~batch.candidate_mask] = 5
    candidate_continuous = batch.candidate_continuous_features.clone()
    candidate_continuous[~batch.candidate_mask] = 777.0
    gains = batch.t3_gains.clone()
    gains[~batch.candidate_mask] = 123.0
    poisoned = replace(
        batch,
        candidate_ids=candidate_ids,
        candidate_categorical_ids=candidate_channels,
        candidate_continuous_features=candidate_continuous,
        t3_gains=gains,
    )

    model_a = make_stub_model()
    model_b = deepcopy(model_a)
    _, loss_a, grads_a = _loss_and_gradients(model_a, batch)
    _, loss_b, grads_b = _loss_and_gradients(model_b, poisoned)
    torch.testing.assert_close(loss_a, loss_b, rtol=0, atol=0)
    _assert_gradients_equal(grads_a, grads_b)


def test_absent_task_target_poison_never_changes_loss_or_gradients():
    batch = make_phase1_batch()
    poisoned = replace(
        batch,
        t1_target=batch.t1_target.clone(),
        t2_target=batch.t2_target.clone(),
        t3_gains=batch.t3_gains.clone(),
    )
    poisoned.t1_target[~batch.t1_present] = 5
    poisoned.t2_target[~batch.t2_present] = -999.0
    ignored_t3 = ~batch.t3_present.unsqueeze(1) | ~batch.candidate_mask
    poisoned.t3_gains[ignored_t3] = 999.0

    model_a = make_stub_model()
    model_b = deepcopy(model_a)
    _, loss_a, grads_a = _loss_and_gradients(model_a, batch)
    _, loss_b, grads_b = _loss_and_gradients(model_b, poisoned)
    torch.testing.assert_close(loss_a, loss_b, rtol=0, atol=0)
    _assert_gradients_equal(grads_a, grads_b)
