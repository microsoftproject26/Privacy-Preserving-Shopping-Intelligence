"""Masked T1 cross-entropy for the explicitly scoped real-data MVP.

This does not stand in for the final multi-task objective. Presence of T2 or T3
is rejected, so a caller cannot silently omit a task and claim all-head training.
"""

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


def masked_t1_sum(logits: torch.Tensor, targets: torch.Tensor, present: torch.Tensor):
    """Finite masked CE numerator and integer support; absent targets are never read."""
    if logits.ndim != 2 or targets.shape != logits.shape[:1] or present.shape != targets.shape:
        raise ValueError("T1 shapes must be [B,C], [B], [B]")
    if present.dtype != torch.bool or targets.dtype != torch.int64:
        raise ValueError("T1 requires boolean presence and int64 targets")
    support = int(present.sum().item())
    if support == 0:
        return None, 0
    kept_logits, kept_targets = logits[present], targets[present]
    if not bool(torch.isfinite(kept_logits).all()):
        raise ValueError("nonfinite contributing T1 logits")
    if bool(((kept_targets < 0) | (kept_targets >= logits.shape[1])).any()):
        raise ValueError("present T1 target outside category vocabulary")
    numerator = F.cross_entropy(kept_logits, kept_targets, reduction="sum")
    if not bool(torch.isfinite(numerator)):
        raise ValueError("nonfinite contributing T1 loss")
    return numerator, support


class T1MVPObjective:
    objective_id = "t1_mvp_masked_cross_entropy_v1"

    def __call__(self, batch: Any, output: Any) -> ObjectiveResult:
        if bool(batch.t2_present.any()) or bool(batch.t3_present.any()):
            raise ValueError("T1 MVP objective cannot consume present T2/T3 tasks")
        numerator, support = masked_t1_sum(output.t1_logits, batch.t1_target, batch.t1_present)
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
            task_losses={"T1": component},
            examples_processed=batch.batch_size,
            contributing_examples=support,
            contributing_tasks=("T1",),
            diagnostics={"objective_id": self.objective_id},
        )


@dataclass(frozen=True)
class T1ContributingWeightPolicy:
    policy_id: str = "mvp_t1_contributing_examples_v1"

    def __call__(self, summary: Any) -> int:
        stat = summary.task_stats.get("T1")
        support = 0 if stat is None else stat.denominator
        if isinstance(support, bool) or not isinstance(support, int) or support < 0:
            raise ValueError("T1 contribution weight must be a non-negative integer")
        if support != summary.contributing_examples:
            raise ValueError("T1-only support disagrees with contributing row count")
        return support
