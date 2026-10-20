"""Deterministic shared-state selection and validation for Flower adapters."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

import torch
from torch import Tensor, nn


class SharedStateError(ValueError):
    """Raised when model/shared-state identity is malformed or incompatible."""


class StateKind(StrEnum):
    PARAMETER = "PARAMETER"
    BUFFER = "BUFFER"


class StateOwnership(StrEnum):
    SHARED = "SHARED"
    LOCAL = "LOCAL"


@dataclass(frozen=True, slots=True)
class SharedStateEntry:
    key: str
    kind: StateKind
    ownership: StateOwnership
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True, slots=True)
class SharedStateSpec:
    """Explicit ordered state ownership; #18 uses all safe state as shared."""

    schema: str
    version: str
    entries: tuple[SharedStateEntry, ...]

    @classmethod
    def all_shared_floating(cls, model: nn.Module) -> SharedStateSpec:
        parameter_names = set(dict(model.named_parameters()).keys())
        buffer_names = set(dict(model.named_buffers()).keys())
        entries: list[SharedStateEntry] = []
        for key, tensor in sorted(model.state_dict().items()):
            if key in parameter_names:
                kind = StateKind.PARAMETER
            elif key in buffer_names:
                kind = StateKind.BUFFER
            else:
                raise SharedStateError(f"State key {key!r} is neither parameter nor buffer")
            if not tensor.is_floating_point():
                raise SharedStateError(
                    f"Shared state {key!r} has non-floating dtype {tensor.dtype}; "
                    "declare a future explicit policy instead of averaging it by accident"
                )
            entries.append(
                SharedStateEntry(
                    key=key,
                    kind=kind,
                    ownership=StateOwnership.SHARED,
                    shape=tuple(tensor.shape),
                    dtype=str(tensor.dtype),
                )
            )
        return cls(schema="shared_state_spec_v1", version="1", entries=tuple(entries))

    @property
    def shared_entries(self) -> tuple[SharedStateEntry, ...]:
        return tuple(entry for entry in self.entries if entry.ownership == StateOwnership.SHARED)

    @property
    def local_entries(self) -> tuple[SharedStateEntry, ...]:
        return tuple(entry for entry in self.entries if entry.ownership == StateOwnership.LOCAL)

    @property
    def shared_keys(self) -> tuple[str, ...]:
        return tuple(entry.key for entry in self.shared_entries)

    def to_manifest(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "version": self.version,
            "entries": [
                {
                    "key": entry.key,
                    "kind": entry.kind.value,
                    "ownership": entry.ownership.value,
                    "shape": list(entry.shape),
                    "dtype": entry.dtype,
                }
                for entry in self.entries
            ],
        }

    def validate_model(self, model: nn.Module) -> None:
        state = model.state_dict()
        state_keys = set(state.keys())
        spec_keys = {entry.key for entry in self.entries}
        if state_keys != spec_keys:
            raise SharedStateError(
                f"State spec keys differ from model keys: missing={sorted(state_keys - spec_keys)}, "
                f"extra={sorted(spec_keys - state_keys)}"
            )
        for entry in self.entries:
            tensor = state[entry.key]
            if tuple(tensor.shape) != entry.shape:
                raise SharedStateError(f"State shape mismatch for {entry.key}")
            if str(tensor.dtype) != entry.dtype:
                raise SharedStateError(f"State dtype mismatch for {entry.key}")
            if entry.ownership == StateOwnership.SHARED:
                if not tensor.is_floating_point():
                    raise SharedStateError(f"Shared state {entry.key} must be floating")
                if not torch.isfinite(tensor).all():
                    raise SharedStateError(f"Shared state {entry.key} contains NaN/Inf")


def pack_shared_state(model: nn.Module, spec: SharedStateSpec) -> dict[str, Tensor]:
    """Pack shared model state in deterministic key order on CPU."""

    spec.validate_model(model)
    state = model.state_dict()
    return {
        entry.key: state[entry.key].detach().cpu().contiguous().clone()
        for entry in spec.shared_entries
    }


def validate_packed_shared_state(packed: Mapping[str, Tensor], spec: SharedStateSpec) -> None:
    if not isinstance(packed, Mapping):
        raise SharedStateError("Packed shared state must be a mapping")
    expected_keys = spec.shared_keys
    if tuple(sorted(packed.keys())) != tuple(sorted(expected_keys)):
        raise SharedStateError(
            f"Packed keys mismatch: expected={sorted(expected_keys)}, got={sorted(packed.keys())}"
        )
    expected = {entry.key: entry for entry in spec.shared_entries}
    for key in expected_keys:
        tensor = packed[key]
        entry = expected[key]
        if not isinstance(tensor, Tensor):
            raise SharedStateError(f"Packed state {key} is not a Tensor")
        if tuple(tensor.shape) != entry.shape:
            raise SharedStateError(
                f"Packed state {key} shape mismatch: {tuple(tensor.shape)} != {entry.shape}"
            )
        if str(tensor.dtype) != entry.dtype:
            raise SharedStateError(
                f"Packed state {key} dtype mismatch: {tensor.dtype} != {entry.dtype}"
            )
        if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
            raise SharedStateError(f"Packed state {key} must be finite floating data")


def load_shared_state(
    model: nn.Module,
    packed: Mapping[str, Tensor],
    spec: SharedStateSpec,
) -> None:
    """Load only declared shared keys while preserving future local state."""

    spec.validate_model(model)
    validate_packed_shared_state(packed, spec)
    current = model.state_dict()
    for entry in spec.shared_entries:
        current[entry.key] = packed[entry.key].to(
            device=current[entry.key].device,
            dtype=current[entry.key].dtype,
        )
    model.load_state_dict(current, strict=True)


def shared_state_digest(packed: Mapping[str, Tensor]) -> str:
    """Digest key, dtype, shape and exact contiguous bytes in sorted order."""

    digest = hashlib.sha256()
    for key in sorted(packed.keys()):
        tensor = packed[key].detach().cpu().contiguous()
        metadata = json.dumps(
            {"key": key, "dtype": str(tensor.dtype), "shape": list(tensor.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
