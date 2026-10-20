"""Minimal protocols consumed by the shared trainer."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import nn

from ppsi.training.batch import Phase1Batch
from ppsi.training.outputs import ObjectiveResult, RawModelOutput

if False:  # pragma: no cover - type-checking import without a runtime cycle
    from ppsi.training.state import SharedStateSpec


@runtime_checkable
class Phase1Model(Protocol):
    """Model surface required by the trainer without freezing final architecture."""

    category_count: int

    def __call__(self, batch: Phase1Batch) -> RawModelOutput: ...

    def shared_state_spec(self) -> SharedStateSpec: ...


@runtime_checkable
class Objective(Protocol):
    """Scientific objective is injected and identified separately from the trainer."""

    objective_id: str

    def __call__(self, batch: Phase1Batch, output: RawModelOutput) -> ObjectiveResult: ...


def require_nn_module(model: Phase1Model) -> nn.Module:
    if not isinstance(model, nn.Module):
        raise TypeError("Phase1Model implementations must also inherit torch.nn.Module")
    return model
