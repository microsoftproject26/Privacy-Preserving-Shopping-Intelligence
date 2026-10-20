"""Copy to tests/mvp on the REAL repo. This file is not part of pack-only test results."""

from types import SimpleNamespace

import pytest
import torch

from ppsi.training.outputs import RawModelOutput, StepStatus
from ppsi.training.t1_mvp_objective import T1ContributingWeightPolicy, T1MVPObjective


def batch(present, *, t2=False, t3=False):
    n = len(present)
    return SimpleNamespace(
        batch_size=n,
        t1_present=torch.tensor(present),
        t1_target=torch.zeros(n, dtype=torch.int64),
        t2_present=torch.full((n,), t2),
        t3_present=torch.full((n,), t3),
    )


def outputs(n):
    return RawModelOutput(
        torch.zeros(n, 588, requires_grad=True), torch.zeros(n, 1), torch.zeros(n, 1)
    )


def test_real_objective_result_type_and_support():
    result = T1MVPObjective()(batch([True, False, True]), outputs(3))
    assert result.status == StepStatus.NORMAL
    assert result.contributing_tasks == ("T1",)
    assert result.contributing_examples == 2
    assert result.task_losses["T1"].denominator == 2
    assert torch.isfinite(result.total_loss)


def test_real_objective_absent_status():
    result = T1MVPObjective()(batch([False, False]), outputs(2))
    assert result.status == StepStatus.NO_CONTRIBUTING_TASK
    assert result.total_loss is None


@pytest.mark.parametrize("other", ["t2", "t3"])
def test_real_objective_refuses_other_tasks(other):
    with pytest.raises(ValueError):
        T1MVPObjective()(batch([True], **{other: True}), outputs(1))


def test_weight_matches_true_contributors():
    summary = SimpleNamespace(
        task_stats={"T1": SimpleNamespace(denominator=5)}, contributing_examples=5
    )
    assert T1ContributingWeightPolicy()(summary) == 5
    summary.contributing_examples = 8
    with pytest.raises(ValueError):
        T1ContributingWeightPolicy()(summary)
