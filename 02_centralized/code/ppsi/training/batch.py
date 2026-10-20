"""Phase1Batch v1 tensor contract and validators.

The runtime validator checks semantic structure. The canonical producer
validator adds storage-normalization rules. Keeping those layers separate lets
contract tests poison masked values to prove that masks, not filler values,
control training semantics.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace

import torch
from torch import Tensor

_CHANNEL_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


class BatchValidationError(ValueError):
    """Raised when a Phase1Batch invariant is violated."""


@dataclass(frozen=True, slots=True)
class CategoricalChannelSpec:
    """Versioned metadata for one categorical tensor channel."""

    name: str
    pad_id: int
    vocab_size: int | None = None

    def __post_init__(self) -> None:
        if not _CHANNEL_NAME_RE.fullmatch(self.name):
            raise ValueError(f"Invalid channel name: {self.name!r}")
        if isinstance(self.pad_id, bool) or not isinstance(self.pad_id, int):
            raise TypeError("pad_id must be an int, not bool")
        if self.pad_id < 0:
            raise ValueError("pad_id must be non-negative")
        if self.vocab_size is not None:
            if isinstance(self.vocab_size, bool) or not isinstance(self.vocab_size, int):
                raise TypeError("vocab_size must be an int or None")
            if self.vocab_size <= 0:
                raise ValueError("vocab_size must be positive")
            if not 0 <= self.pad_id < self.vocab_size:
                raise ValueError("pad_id must be inside the declared vocabulary")


@dataclass(frozen=True, slots=True)
class Phase1BatchSpec:
    """Representation-bound shape metadata consumed by Phase1Batch v1."""

    schema: str = "phase1_batch_spec_v1"
    version: str = "1"
    history_categorical: tuple[CategoricalChannelSpec, ...] = ()
    query_categorical: tuple[CategoricalChannelSpec, ...] = ()
    candidate_categorical: tuple[CategoricalChannelSpec, ...] = ()
    history_continuous_dim: int = 0
    query_continuous_dim: int = 0
    candidate_continuous_dim: int = 0
    candidate_id_pad_id: int = 0
    candidate_id_vocab_size: int | None = None
    t1_absent_fill: int = 0
    t2_absent_fill: float = 0.0
    t3_absent_fill: float = 0.0

    def __post_init__(self) -> None:
        if self.schema != "phase1_batch_spec_v1":
            raise ValueError("Unsupported batch spec schema")
        if self.version != "1":
            raise ValueError("Unsupported batch spec version")
        for field_name in (
            "history_continuous_dim",
            "query_continuous_dim",
            "candidate_continuous_dim",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative int")
        for group_name in (
            "history_categorical",
            "query_categorical",
            "candidate_categorical",
        ):
            specs = getattr(self, group_name)
            names = [spec.name for spec in specs]
            if len(names) != len(set(names)):
                raise ValueError(f"Duplicate names in {group_name}")
        if isinstance(self.candidate_id_pad_id, bool) or not isinstance(
            self.candidate_id_pad_id, int
        ):
            raise TypeError("candidate_id_pad_id must be an int")
        if self.candidate_id_pad_id < 0:
            raise ValueError("candidate_id_pad_id must be non-negative")
        if self.candidate_id_vocab_size is not None:
            if self.candidate_id_vocab_size <= 0:
                raise ValueError("candidate_id_vocab_size must be positive")
            if not 0 <= self.candidate_id_pad_id < self.candidate_id_vocab_size:
                raise ValueError("candidate_id_pad_id must be inside candidate vocabulary")

    @property
    def history_channel_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.history_categorical)

    @property
    def query_channel_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.query_categorical)

    @property
    def candidate_channel_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.candidate_categorical)


@dataclass(frozen=True, slots=True)
class Phase1Batch:
    """Shared tensor-only batch contract for centralized and Flower adapters.

    The dataclass freezes the structure, not the mutability of Tensor storage.
    Callers should treat instances as immutable values.
    """

    history_categorical_ids: Mapping[str, Tensor]
    history_continuous_features: Tensor
    lengths: Tensor
    history_mask: Tensor
    query_categorical_ids: Mapping[str, Tensor]
    query_continuous_features: Tensor
    candidate_ids: Tensor
    candidate_categorical_ids: Mapping[str, Tensor]
    candidate_continuous_features: Tensor
    candidate_mask: Tensor
    t1_target: Tensor
    t2_target: Tensor
    t3_gains: Tensor
    t1_present: Tensor
    t2_present: Tensor
    t3_present: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.lengths.shape[0])

    @property
    def history_width(self) -> int:
        return int(self.history_mask.shape[1])

    @property
    def candidate_width(self) -> int:
        return int(self.candidate_mask.shape[1])

    def contributing_row_mask(self) -> Tensor:
        t3_has_candidate = self.t3_present & self.candidate_mask.any(dim=1)
        return self.t1_present | self.t2_present | t3_has_candidate

    def to(self, device: torch.device | str) -> Phase1Batch:
        """Return a new batch with every Tensor moved to the requested device."""

        return replace(
            self,
            history_categorical_ids={
                name: tensor.to(device) for name, tensor in self.history_categorical_ids.items()
            },
            history_continuous_features=self.history_continuous_features.to(device),
            lengths=self.lengths.to(device),
            history_mask=self.history_mask.to(device),
            query_categorical_ids={
                name: tensor.to(device) for name, tensor in self.query_categorical_ids.items()
            },
            query_continuous_features=self.query_continuous_features.to(device),
            candidate_ids=self.candidate_ids.to(device),
            candidate_categorical_ids={
                name: tensor.to(device) for name, tensor in self.candidate_categorical_ids.items()
            },
            candidate_continuous_features=self.candidate_continuous_features.to(device),
            candidate_mask=self.candidate_mask.to(device),
            t1_target=self.t1_target.to(device),
            t2_target=self.t2_target.to(device),
            t3_gains=self.t3_gains.to(device),
            t1_present=self.t1_present.to(device),
            t2_present=self.t2_present.to(device),
            t3_present=self.t3_present.to(device),
        )


def _require_tensor(value: object, field: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise BatchValidationError(f"{field}: expected torch.Tensor")
    return value


def _require_dtype(tensor: Tensor, dtype: torch.dtype, field: str) -> None:
    if tensor.dtype != dtype:
        raise BatchValidationError(f"{field}: expected {dtype}, got {tensor.dtype}")


def _require_shape(tensor: Tensor, expected: tuple[int, ...], field: str) -> None:
    if tuple(tensor.shape) != expected:
        raise BatchValidationError(f"{field}: expected shape {expected}, got {tuple(tensor.shape)}")


def _require_finite(tensor: Tensor, field: str) -> None:
    if tensor.is_floating_point() and not torch.isfinite(tensor).all():
        raise BatchValidationError(f"{field}: NaN/Inf values are not allowed")


def _validate_channel_map(
    values: Mapping[str, Tensor],
    specs: tuple[CategoricalChannelSpec, ...],
    expected_shape: tuple[int, ...],
    field: str,
) -> None:
    if not isinstance(values, Mapping):
        raise BatchValidationError(f"{field}: expected a mapping")
    expected_names = tuple(spec.name for spec in specs)
    if set(values.keys()) != set(expected_names):
        raise BatchValidationError(
            f"{field}: expected channels {sorted(expected_names)}, got {sorted(values.keys())}"
        )
    spec_by_name = {spec.name: spec for spec in specs}
    for name in expected_names:
        tensor = _require_tensor(values[name], f"{field}.{name}")
        _require_dtype(tensor, torch.int64, f"{field}.{name}")
        _require_shape(tensor, expected_shape, f"{field}.{name}")
        spec = spec_by_name[name]
        if tensor.numel() > 0 and int(tensor.min()) < 0:
            raise BatchValidationError(f"{field}.{name}: IDs must be non-negative")
        if (
            spec.vocab_size is not None
            and tensor.numel() > 0
            and int(tensor.max()) >= spec.vocab_size
        ):
            raise BatchValidationError(f"{field}.{name}: IDs must be in [0, {spec.vocab_size})")


def validate_phase1_batch(batch: Phase1Batch, spec: Phase1BatchSpec) -> None:
    """Validate runtime semantics without relying on canonical filler values."""

    if not isinstance(batch, Phase1Batch):
        raise BatchValidationError("batch: expected Phase1Batch")

    lengths = _require_tensor(batch.lengths, "lengths")
    _require_dtype(lengths, torch.int64, "lengths")
    if lengths.ndim != 1:
        raise BatchValidationError("lengths: expected rank 1")
    batch_size = int(lengths.shape[0])
    if batch_size < 1:
        raise BatchValidationError("B must be >= 1")

    history_mask = _require_tensor(batch.history_mask, "history_mask")
    _require_dtype(history_mask, torch.bool, "history_mask")
    if history_mask.ndim != 2 or history_mask.shape[0] != batch_size:
        raise BatchValidationError("history_mask: expected shape [B,L]")
    history_width = int(history_mask.shape[1])
    if history_width < 1:
        raise BatchValidationError("physical L must be >= 1")

    if torch.any(lengths < 0) or torch.any(lengths > history_width):
        raise BatchValidationError("lengths must satisfy 0 <= length <= L")
    expected_history_mask = torch.arange(history_width, device=lengths.device).unsqueeze(
        0
    ) < lengths.unsqueeze(1)
    if not torch.equal(history_mask, expected_history_mask):
        raise BatchValidationError(
            "history_mask must be exact right-padding: mask[b,j] == (j < lengths[b])"
        )

    _validate_channel_map(
        batch.history_categorical_ids,
        spec.history_categorical,
        (batch_size, history_width),
        "history_categorical_ids",
    )

    history_continuous = _require_tensor(
        batch.history_continuous_features, "history_continuous_features"
    )
    _require_dtype(history_continuous, torch.float32, "history_continuous_features")
    _require_shape(
        history_continuous,
        (batch_size, history_width, spec.history_continuous_dim),
        "history_continuous_features",
    )
    _require_finite(history_continuous, "history_continuous_features")

    _validate_channel_map(
        batch.query_categorical_ids,
        spec.query_categorical,
        (batch_size,),
        "query_categorical_ids",
    )
    query_continuous = _require_tensor(batch.query_continuous_features, "query_continuous_features")
    _require_dtype(query_continuous, torch.float32, "query_continuous_features")
    _require_shape(
        query_continuous,
        (batch_size, spec.query_continuous_dim),
        "query_continuous_features",
    )
    _require_finite(query_continuous, "query_continuous_features")

    candidate_ids = _require_tensor(batch.candidate_ids, "candidate_ids")
    _require_dtype(candidate_ids, torch.int64, "candidate_ids")
    if candidate_ids.ndim != 2 or candidate_ids.shape[0] != batch_size:
        raise BatchValidationError("candidate_ids: expected shape [B,K]")
    candidate_width = int(candidate_ids.shape[1])
    if candidate_width < 1:
        raise BatchValidationError("physical K must be >= 1")
    if candidate_ids.numel() > 0 and int(candidate_ids.min()) < 0:
        raise BatchValidationError("candidate_ids: IDs must be non-negative")
    if (
        spec.candidate_id_vocab_size is not None
        and candidate_ids.numel() > 0
        and int(candidate_ids.max()) >= spec.candidate_id_vocab_size
    ):
        raise BatchValidationError(
            f"candidate_ids: IDs must be in [0, {spec.candidate_id_vocab_size})"
        )

    candidate_mask = _require_tensor(batch.candidate_mask, "candidate_mask")
    _require_dtype(candidate_mask, torch.bool, "candidate_mask")
    _require_shape(candidate_mask, (batch_size, candidate_width), "candidate_mask")

    _validate_channel_map(
        batch.candidate_categorical_ids,
        spec.candidate_categorical,
        (batch_size, candidate_width),
        "candidate_categorical_ids",
    )
    candidate_continuous = _require_tensor(
        batch.candidate_continuous_features, "candidate_continuous_features"
    )
    _require_dtype(candidate_continuous, torch.float32, "candidate_continuous_features")
    _require_shape(
        candidate_continuous,
        (batch_size, candidate_width, spec.candidate_continuous_dim),
        "candidate_continuous_features",
    )
    _require_finite(candidate_continuous, "candidate_continuous_features")

    fields = {
        "t1_target": (batch.t1_target, torch.int64, (batch_size,)),
        "t2_target": (batch.t2_target, torch.float32, (batch_size, 1)),
        "t3_gains": (batch.t3_gains, torch.float32, (batch_size, candidate_width)),
        "t1_present": (batch.t1_present, torch.bool, (batch_size,)),
        "t2_present": (batch.t2_present, torch.bool, (batch_size,)),
        "t3_present": (batch.t3_present, torch.bool, (batch_size,)),
    }
    for field, (tensor, dtype, shape) in fields.items():
        tensor = _require_tensor(tensor, field)
        _require_dtype(tensor, dtype, field)
        _require_shape(tensor, shape, field)
        _require_finite(tensor, field)

    if batch.t2_present.any():
        present_t2 = batch.t2_target[batch.t2_present]
        if torch.any(present_t2 < 0) or torch.any(present_t2 > 1):
            raise BatchValidationError("Present T2 targets must be in [0,1]")

    invalid_t3_rows = batch.t3_present & ~candidate_mask.any(dim=1)
    if invalid_t3_rows.any():
        rows = torch.nonzero(invalid_t3_rows, as_tuple=False).flatten().tolist()
        raise BatchValidationError(
            f"t3_present requires at least one valid candidate; invalid rows={rows}"
        )


def validate_canonical_phase1_batch(batch: Phase1Batch, spec: Phase1BatchSpec) -> None:
    """Validate canonical storage conventions in addition to runtime semantics."""

    validate_phase1_batch(batch, spec)

    history_padding = ~batch.history_mask
    for channel in spec.history_categorical:
        values = batch.history_categorical_ids[channel.name]
        if batch.history_mask.any() and torch.any(values[batch.history_mask] == channel.pad_id):
            raise BatchValidationError(
                f"history_categorical_ids.{channel.name}: pad ID is not valid data"
            )
        if history_padding.any() and not torch.all(values[history_padding] == channel.pad_id):
            raise BatchValidationError(
                f"history_categorical_ids.{channel.name}: non-canonical padding value"
            )

    for channel in spec.query_categorical:
        values = batch.query_categorical_ids[channel.name]
        if torch.any(values == channel.pad_id):
            raise BatchValidationError(
                f"query_categorical_ids.{channel.name}: pad ID is not valid data"
            )
    if batch.history_continuous_features.numel() > 0:
        expanded = history_padding.unsqueeze(-1).expand_as(batch.history_continuous_features)
        if expanded.any() and not torch.all(batch.history_continuous_features[expanded] == 0):
            raise BatchValidationError(
                "history_continuous_features: padded positions must be canonical zero"
            )

    candidate_padding = ~batch.candidate_mask
    if batch.candidate_mask.any() and torch.any(
        batch.candidate_ids[batch.candidate_mask] == spec.candidate_id_pad_id
    ):
        raise BatchValidationError("candidate_ids: pad ID is not valid data")
    if candidate_padding.any() and not torch.all(
        batch.candidate_ids[candidate_padding] == spec.candidate_id_pad_id
    ):
        raise BatchValidationError("candidate_ids: non-canonical padding value")
    for channel in spec.candidate_categorical:
        values = batch.candidate_categorical_ids[channel.name]
        if batch.candidate_mask.any() and torch.any(
            values[batch.candidate_mask] == channel.pad_id
        ):
            raise BatchValidationError(
                f"candidate_categorical_ids.{channel.name}: pad ID is not valid data"
            )
        if candidate_padding.any() and not torch.all(values[candidate_padding] == channel.pad_id):
            raise BatchValidationError(
                f"candidate_categorical_ids.{channel.name}: non-canonical padding value"
            )
    if batch.candidate_continuous_features.numel() > 0:
        expanded = candidate_padding.unsqueeze(-1).expand_as(batch.candidate_continuous_features)
        if expanded.any() and not torch.all(batch.candidate_continuous_features[expanded] == 0):
            raise BatchValidationError(
                "candidate_continuous_features: masked positions must be canonical zero"
            )

    if (~batch.t1_present).any() and not torch.all(
        batch.t1_target[~batch.t1_present] == spec.t1_absent_fill
    ):
        raise BatchValidationError("t1_target: absent rows must contain canonical filler")
    if (~batch.t2_present).any() and not torch.all(
        batch.t2_target[~batch.t2_present] == spec.t2_absent_fill
    ):
        raise BatchValidationError("t2_target: absent rows must contain canonical filler")

    ignored_t3 = ~batch.candidate_mask | ~batch.t3_present.unsqueeze(1)
    if ignored_t3.any() and not torch.all(batch.t3_gains[ignored_t3] == spec.t3_absent_fill):
        raise BatchValidationError("t3_gains: ignored positions must contain canonical filler")
