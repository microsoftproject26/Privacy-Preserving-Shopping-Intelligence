"""Flower-independent shared local train/eval core."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from ppsi.training.batch import Phase1Batch, Phase1BatchSpec, validate_phase1_batch
from ppsi.training.outputs import (
    LocalStepSummary,
    ObjectiveResult,
    StepStatus,
    TaskStepStat,
    validate_raw_model_output,
)
from ppsi.training.protocol import Objective, Phase1Model, require_nn_module
from ppsi.training.state import SharedStateSpec, pack_shared_state, shared_state_digest


class TrainingCoreError(RuntimeError):
    """Raised when local training cannot preserve a required invariant."""


@dataclass(frozen=True, slots=True)
class TrainerPolicy:
    """Behavior that changes local step semantics and must be versioned."""

    scheduler_step_unit: str = "OPTIMIZER_STEP"
    gradient_clip_norm: float | None = None

    def __post_init__(self) -> None:
        if self.scheduler_step_unit != "OPTIMIZER_STEP":
            raise ValueError("#18 supports only OPTIMIZER_STEP scheduler semantics")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive or None")


class LocalTrainerCore:
    """One local step implementation shared by centralized and Flower adapters."""

    def __init__(
        self,
        *,
        model: Phase1Model,
        batch_spec: Phase1BatchSpec,
        objective: Objective,
        optimizer: Optimizer,
        scheduler: Any | None = None,
        policy: TrainerPolicy | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.model = require_nn_module(model)
        self.batch_spec = batch_spec
        self.objective = objective
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.policy = policy or TrainerPolicy()
        self.device = torch.device(device)
        self.model.to(self.device)
        self.optimizer_step_count = 0
        self._initial_optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        self._initial_scheduler_state = (
            None if self.scheduler is None else copy.deepcopy(self.scheduler.state_dict())
        )

    @property
    def shared_state_spec(self) -> SharedStateSpec:
        return self.model.shared_state_spec()  # type: ignore[attr-defined]

    def reset_optimizer_for_new_server_round(self) -> None:
        """Reset client-local optimization state for a completed-round boundary."""

        self.optimizer.load_state_dict(copy.deepcopy(self._initial_optimizer_state))
        if self.scheduler is not None:
            self.scheduler.load_state_dict(copy.deepcopy(self._initial_scheduler_state))
        self.optimizer_step_count = 0

    def _validate_model_and_gradients(self) -> None:
        for name, parameter in self.model.named_parameters():
            if not torch.isfinite(parameter).all():
                raise TrainingCoreError(f"Parameter {name} contains NaN/Inf")
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise TrainingCoreError(f"Gradient {name} contains NaN/Inf")

    def _summarize(
        self,
        result: ObjectiveResult,
        *,
        optimizer_step_performed: bool,
    ) -> LocalStepSummary:
        stats = {
            task: TaskStepStat(
                numerator=float(component.numerator.detach().cpu().item()),
                denominator=component.denominator,
                support_unit=component.support_unit,
            )
            for task, component in sorted(result.task_losses.items())
        }
        total = (
            None if result.total_loss is None else float(result.total_loss.detach().cpu().item())
        )
        diagnostics = dict(result.diagnostics)
        diagnostics["shared_state_digest"] = shared_state_digest(
            pack_shared_state(self.model, self.shared_state_spec)
        )
        return LocalStepSummary(
            status=result.status,
            total_loss=total,
            task_stats=stats,
            examples_processed=result.examples_processed,
            contributing_examples=result.contributing_examples,
            contributing_tasks=result.contributing_tasks,
            optimizer_step_performed=optimizer_step_performed,
            optimizer_step_index=self.optimizer_step_count,
            diagnostics=diagnostics,
        )

    def train_step(self, batch: Phase1Batch) -> LocalStepSummary:
        validate_phase1_batch(batch, self.batch_spec)
        batch_on_device = batch.to(self.device)
        self.model.train()
        output = self.model(batch_on_device)
        validate_raw_model_output(
            output,
            batch_on_device,
            category_count=self.model.category_count,  # type: ignore[attr-defined]
        )
        result = self.objective(batch_on_device, output)

        if result.status == StepStatus.NO_CONTRIBUTING_TASK:
            return self._summarize(result, optimizer_step_performed=False)
        if result.total_loss is None or result.total_loss.ndim != 0:
            raise TrainingCoreError("Contributing objective must return a scalar total_loss")
        if not torch.isfinite(result.total_loss):
            raise TrainingCoreError("Contributing loss contains NaN/Inf")

        self.optimizer.zero_grad(set_to_none=True)
        result.total_loss.backward()
        self._validate_model_and_gradients()

        if self.policy.gradient_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.policy.gradient_clip_norm)
            self._validate_model_and_gradients()

        self.optimizer.step()
        self.optimizer_step_count += 1
        if self.scheduler is not None:
            self.scheduler.step()
        self._validate_model_and_gradients()
        return self._summarize(result, optimizer_step_performed=True)

    @torch.no_grad()
    def eval_step(self, batch: Phase1Batch) -> LocalStepSummary:
        validate_phase1_batch(batch, self.batch_spec)
        batch_on_device = batch.to(self.device)
        self.model.eval()
        output = self.model(batch_on_device)
        validate_raw_model_output(
            output,
            batch_on_device,
            category_count=self.model.category_count,  # type: ignore[attr-defined]
        )
        result = self.objective(batch_on_device, output)
        return self._summarize(result, optimizer_step_performed=False)
