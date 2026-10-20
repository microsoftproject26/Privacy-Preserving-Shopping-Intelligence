"""Unified Phase-1 trainer contracts and adapters."""

from ppsi.training.batch import (
    BatchValidationError,
    CategoricalChannelSpec,
    Phase1Batch,
    Phase1BatchSpec,
    validate_canonical_phase1_batch,
    validate_phase1_batch,
)
from ppsi.training.core import LocalTrainerCore, TrainerPolicy, TrainingCoreError
from ppsi.training.outputs import RawModelOutput, StepStatus

__all__ = [
    "BatchValidationError",
    "CategoricalChannelSpec",
    "LocalTrainerCore",
    "Phase1Batch",
    "Phase1BatchSpec",
    "RawModelOutput",
    "StepStatus",
    "TrainerPolicy",
    "TrainingCoreError",
    "validate_canonical_phase1_batch",
    "validate_phase1_batch",
]
