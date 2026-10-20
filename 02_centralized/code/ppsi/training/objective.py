"""Objective interface implementation used only for contract smoke tests.

ContractSmokeObjective is deliberately non-scientific. It proves masking,
head alignment and backward behavior without freezing the final T3 ranking
loss or final multi-task objective weights.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ppsi.training.batch import Phase1Batch
from ppsi.training.outputs import (
    ObjectiveResult,
    RawModelOutput,
    StepStatus,
    SupportUnit,
    TaskLossComponent,
)


class ContractSmokeObjective:
    """Equal-weight sum of available task means for interface validation only."""

    objective_id = "contract_smoke_objective_v1_NON_SCIENTIFIC"

    def __call__(self, batch: Phase1Batch, output: RawModelOutput) -> ObjectiveResult:
        task_losses: dict[str, TaskLossComponent] = {}
        task_means: list[torch.Tensor] = []

        if batch.t1_present.any():
            logits = output.t1_logits[batch.t1_present]
            targets = batch.t1_target[batch.t1_present]
            if torch.any(targets < 0) or torch.any(targets >= logits.shape[1]):
                raise ValueError("Present T1 targets must be inside the category-logit range")
            numerator = F.cross_entropy(logits, targets, reduction="sum")
            denominator = int(targets.numel())
            component = TaskLossComponent(numerator, denominator, SupportUnit.EXAMPLES)
            task_losses["T1"] = component
            task_means.append(component.mean)

        if batch.t2_present.any():
            logits = output.t2_logit[batch.t2_present]
            targets = batch.t2_target[batch.t2_present]
            numerator = F.binary_cross_entropy_with_logits(logits, targets, reduction="sum")
            denominator = int(targets.shape[0])
            component = TaskLossComponent(numerator, denominator, SupportUnit.EXAMPLES)
            task_losses["T2"] = component
            task_means.append(component.mean)

        valid_t3 = batch.t3_present.unsqueeze(1) & batch.candidate_mask
        if valid_t3.any():
            # Non-scientific MSE used only to exercise candidate masking and gradients.
            diff = output.t3_scores[valid_t3] - batch.t3_gains[valid_t3]
            numerator = torch.square(diff).sum()
            denominator = int(valid_t3.sum().item())
            component = TaskLossComponent(numerator, denominator, SupportUnit.CANDIDATES)
            task_losses["T3"] = component
            task_means.append(component.mean)

        examples_processed = batch.batch_size
        contributing_examples = int(batch.contributing_row_mask().sum().item())

        if not task_means:
            return ObjectiveResult(
                status=StepStatus.NO_CONTRIBUTING_TASK,
                total_loss=None,
                task_losses={},
                examples_processed=examples_processed,
                contributing_examples=0,
                contributing_tasks=(),
                diagnostics={"objective_id": self.objective_id},
            )

        total_loss = torch.stack(task_means).sum()
        return ObjectiveResult(
            status=StepStatus.NORMAL,
            total_loss=total_loss,
            task_losses=task_losses,
            examples_processed=examples_processed,
            contributing_examples=contributing_examples,
            contributing_tasks=tuple(sorted(task_losses)),
            diagnostics={"objective_id": self.objective_id},
        )
