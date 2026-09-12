"""Masked T2 binary cross-entropy with logits for the fast matched comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from ppsi.training.outputs import (
    ObjectiveResult,
    StepStatus,
    SupportUnit,
    TaskLossComponent,
)


def masked_t2_sum(logits: torch.Tensor, targets: torch.Tensor, present: torch.Tensor):
    """Finite masked BCE numerator and integer support; absent targets never contribute."""
    if logits.ndim == 2 and logits.shape[1] == 1:
        logits = logits.squeeze(-1)
    if targets.ndim == 2 and targets.shape[1] == 1:
        targets = targets.squeeze(-1)
    if present.ndim == 2 and present.shape[1] == 1:
        present = present.squeeze(-1)

    if logits.shape != targets.shape or present.shape != targets.shape:
        raise ValueError(f"T2 shapes mismatch: {logits.shape}, {targets.shape}, {present.shape}")
    if present.dtype != torch.bool:
        raise ValueError("T2 requires boolean presence mask")
    support = int(present.sum().item())
    if support == 0:
        return None, 0
    kept_logits = logits[present]
    kept_targets = targets[present].to(torch.float32)
    if not bool(torch.isfinite(kept_logits).all()):
        raise ValueError("nonfinite contributing T2 logits")
    if bool(((kept_targets < 0.0) | (kept_targets > 1.0)).any()):
        raise ValueError("present T2 target outside [0, 1]")
    numerator = F.binary_cross_entropy_with_logits(kept_logits, kept_targets, reduction="sum")
    if not bool(torch.isfinite(numerator)):
        raise ValueError("nonfinite contributing T2 loss")
    return numerator, support


class T2MVPObjective:
    objective_id = "t2_mvp_masked_bce_v1"

    def __call__(self, batch: Any, output: Any) -> ObjectiveResult:
        if bool(batch.t1_present.any()) or bool(batch.t3_present.any()):
            raise ValueError("T2 MVP objective cannot consume present T1/T3 tasks")
        numerator, support = masked_t2_sum(output.t2_logit, batch.t2_target, batch.t2_present)
        if support == 0:
            return ObjectiveResult(
                status=StepStatus.NO_CONTRIBUTING_TASK,
                total_loss=None,
                task_losses={},
                examples_processed=batch.batch_size,
                contributing_examples=0,
                contributing_tasks=(),
                diagnostics={"objective_id": self.objective_id},
            )
        component = TaskLossComponent(numerator, support, SupportUnit.EXAMPLES)
        return ObjectiveResult(
            status=StepStatus.NORMAL,
            total_loss=component.mean,
            task_losses={"T2": component},
            examples_processed=batch.batch_size,
            contributing_examples=support,
            contributing_tasks=("T2",),
            diagnostics={"objective_id": self.objective_id},
        )


@dataclass(frozen=True)
class T2ContributingWeightPolicy:
    policy_id: str = "mvp_t2_contributing_examples_v1"

    def __call__(self, summary: Any) -> int:
        stat = summary.task_stats.get("T2")
        support = 0 if stat is None else stat.denominator
        if isinstance(support, bool) or not isinstance(support, int) or support < 0:
            raise ValueError("T2 contribution weight must be a non-negative integer")
        if support != summary.contributing_examples:
            raise ValueError("T2-only support disagrees with contributing row count")
        return support
