"""Raw model outputs and mergeable local training summaries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import torch
from torch import Tensor

from ppsi.training.batch import Phase1Batch


class OutputValidationError(ValueError):
    """Raised when a model violates the raw-output contract."""


class StepStatus(StrEnum):
    NORMAL = "NORMAL"
    NO_CONTRIBUTING_TASK = "NO_CONTRIBUTING_TASK"


class SupportUnit(StrEnum):
    EXAMPLES = "EXAMPLES"
    CANDIDATES = "CANDIDATES"


@dataclass(frozen=True, slots=True)
class RawModelOutput:
    """Activation-free shared output contract."""

    t1_logits: Tensor
    t2_logit: Tensor
    t3_scores: Tensor


@dataclass(frozen=True, slots=True)
class TaskLossComponent:
    """One additive task-loss numerator and its support."""

    numerator: Tensor
    denominator: int
    support_unit: SupportUnit

    @property
    def mean(self) -> Tensor:
        if self.denominator <= 0:
            raise ZeroDivisionError("Task loss component has zero support")
        return self.numerator / self.denominator


@dataclass(frozen=True, slots=True)
class ObjectiveResult:
    """Loss result returned by an injected Objective implementation."""

    status: StepStatus
    total_loss: Tensor | None
    task_losses: Mapping[str, TaskLossComponent]
    examples_processed: int
    contributing_examples: int
    contributing_tasks: tuple[str, ...]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TaskStepStat:
    numerator: float
    denominator: int
    support_unit: SupportUnit

    @property
    def mean(self) -> float | None:
        return None if self.denominator == 0 else self.numerator / self.denominator


@dataclass(frozen=True, slots=True)
class LocalStepSummary:
    """Adapter-mergeable result from one shared local step."""

    status: StepStatus
    total_loss: float | None
    task_stats: Mapping[str, TaskStepStat]
    examples_processed: int
    contributing_examples: int
    contributing_tasks: tuple[str, ...]
    optimizer_step_performed: bool
    optimizer_step_index: int
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def to_scalar_metrics(self, *, prefix: str) -> dict[str, int | float | str]:
        metrics: dict[str, int | float | str] = {
            f"{prefix}_status": self.status.value,
            f"{prefix}_examples_processed": self.examples_processed,
            f"{prefix}_contributing_examples": self.contributing_examples,
            f"{prefix}_optimizer_step_performed": int(self.optimizer_step_performed),
            f"{prefix}_optimizer_step_index": self.optimizer_step_index,
        }
        if self.total_loss is not None:
            metrics[f"{prefix}_total_loss"] = self.total_loss
        for task, stat in sorted(self.task_stats.items()):
            task_key = task.lower()
            metrics[f"{prefix}_{task_key}_loss_numerator"] = stat.numerator
            metrics[f"{prefix}_{task_key}_loss_denominator"] = stat.denominator
        return metrics


def validate_raw_model_output(
    output: RawModelOutput,
    batch: Phase1Batch,
    *,
    category_count: int,
) -> None:
    """Fail closed when a model violates the frozen raw-output shape contract."""

    if not isinstance(output, RawModelOutput):
        raise OutputValidationError("model output must be RawModelOutput")
    batch_size = batch.batch_size
    candidate_width = batch.candidate_width
    expected = {
        "t1_logits": (output.t1_logits, (batch_size, category_count)),
        "t2_logit": (output.t2_logit, (batch_size, 1)),
        "t3_scores": (output.t3_scores, (batch_size, candidate_width)),
    }
    for field_name, (tensor, shape) in expected.items():
        if not isinstance(tensor, Tensor):
            raise OutputValidationError(f"{field_name}: expected torch.Tensor")
        if not tensor.is_floating_point():
            raise OutputValidationError(f"{field_name}: expected a floating dtype")
        if tuple(tensor.shape) != shape:
            raise OutputValidationError(
                f"{field_name}: expected shape {shape}, got {tuple(tensor.shape)}"
            )
        if not torch.isfinite(tensor).all():
            raise OutputValidationError(f"{field_name}: NaN/Inf values are not allowed")
