"""Checkpoint v1 with atomic persistence and exact CPU replay state."""

from __future__ import annotations

import copy
import hashlib
import math
import os
import random
import re
import tempfile
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

from ppsi.training.core import LocalTrainerCore
from ppsi.training.sampler import LoaderContract, TrainingCursor

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class CheckpointError(RuntimeError):
    """Raised when checkpoint integrity or semantic compatibility fails."""


@dataclass(frozen=True, slots=True)
class CheckpointIdentity:
    experiment_config_sha256: str
    common_initialization_sha256: str
    input_data_sha256: str
    shared_trainer_core_sha256: str
    objective_config_sha256: str
    environment_lock_sha256: str
    git_sha: str

    def __post_init__(self) -> None:
        for name in (
            "experiment_config_sha256",
            "common_initialization_sha256",
            "input_data_sha256",
            "shared_trainer_core_sha256",
            "objective_config_sha256",
            "environment_lock_sha256",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
                raise ValueError(f"{name} must contain 64 lowercase hex characters")
        if not self.git_sha:
            raise ValueError("git_sha must be non-empty")

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, str]) -> CheckpointIdentity:
        return cls(**value)


@dataclass(frozen=True, slots=True)
class CheckpointArtifact:
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    cursor: TrainingCursor
    best_criterion: float | None
    run_id: str
    attempt: int


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any], *, strict_cuda: bool = True) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None:
        if not torch.cuda.is_available():
            if strict_cuda:
                raise CheckpointError(
                    "Checkpoint contains CUDA RNG state but CUDA is unavailable for exact replay"
                )
        else:
            torch.cuda.set_rng_state_all(cuda_state)


def build_checkpoint_payload(
    *,
    run_id: str,
    attempt: int,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Any | None,
    grad_scaler: Any | None,
    cursor: TrainingCursor,
    best_criterion: float | None,
    identity: CheckpointIdentity,
    loader_contract: LoaderContract,
    scheduler_step_unit: str,
) -> dict[str, Any]:
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be a non-empty string")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise ValueError("attempt must be an integer >= 1")
    if best_criterion is not None:
        if (
            isinstance(best_criterion, bool)
            or not isinstance(best_criterion, (int, float))
            or not math.isfinite(float(best_criterion))
        ):
            raise ValueError("best_criterion must be finite or None")
        best_criterion = float(best_criterion)
    if scheduler_step_unit != "OPTIMIZER_STEP":
        raise ValueError("Unsupported scheduler step unit")
    return {
        "schema": "checkpoint_v1",
        "version": "1",
        "run_id": run_id,
        "attempt": attempt,
        "model_state_dict": {
            key: value.detach().clone() for key, value in model.state_dict().items()
        },
        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        "scheduler_state_dict": (
            None if scheduler is None else copy.deepcopy(scheduler.state_dict())
        ),
        "grad_scaler_state": (
            None if grad_scaler is None else copy.deepcopy(grad_scaler.state_dict())
        ),
        "cursor": cursor.to_dict(),
        "best_criterion": best_criterion,
        "identity": identity.to_dict(),
        "loader_contract": loader_contract.to_dict(),
        "scheduler_step_unit": scheduler_step_unit,
        "rng_state": capture_rng_state(),
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "python_version": ".".join(map(str, __import__("sys").version_info[:3])),
        "torch_version": torch.__version__,
        "amp_enabled": grad_scaler is not None,
    }


def _validate_payload_header(
    payload: dict[str, Any],
    *,
    expected_identity: CheckpointIdentity,
    expected_loader_contract: LoaderContract,
) -> None:
    if payload.get("schema") != "checkpoint_v1" or payload.get("version") != "1":
        raise CheckpointError("Unsupported checkpoint schema/version")
    actual_identity = CheckpointIdentity.from_dict(payload["identity"])
    if actual_identity != expected_identity:
        raise CheckpointError(
            f"Checkpoint semantic identity mismatch: {actual_identity} != {expected_identity}"
        )
    actual_loader = LoaderContract.from_dict(payload["loader_contract"])
    if actual_loader != expected_loader_contract:
        raise CheckpointError(
            f"Checkpoint loader contract mismatch: {actual_loader} != {expected_loader_contract}"
        )
    if payload.get("scheduler_step_unit") != "OPTIMIZER_STEP":
        raise CheckpointError("Unsupported checkpoint scheduler semantics")
    TrainingCursor.from_dict(payload["cursor"])


def save_checkpoint(path: Path, payload: dict[str, Any]) -> CheckpointArtifact:
    """Write one checkpoint atomically; the returned SHA is the integrity identity."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        with temp_path.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        digest = file_sha256(temp_path)

        for attempt in range(5):
            try:
                os.replace(temp_path, path)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.1 * (2**attempt))
        return CheckpointArtifact(path=path, sha256=digest)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _validate_rng_compatibility(state: dict[str, Any], *, strict_cuda: bool) -> None:
    if not isinstance(state, dict):
        raise CheckpointError("Checkpoint RNG state must be a dict")
    for key in ("python", "numpy", "torch_cpu", "torch_cuda"):
        if key not in state:
            raise CheckpointError(f"Checkpoint RNG state is missing {key}")
    if state["torch_cuda"] is not None and strict_cuda and not torch.cuda.is_available():
        raise CheckpointError(
            "Checkpoint contains CUDA RNG state but CUDA is unavailable for exact replay"
        )


def _validate_model_state(model: nn.Module, state: object) -> None:
    if not isinstance(state, Mapping):
        raise CheckpointError("Checkpoint model_state_dict must be a mapping")
    current = model.state_dict()
    if set(state) != set(current):
        missing = sorted(set(current) - set(state))
        unexpected = sorted(set(state) - set(current))
        raise CheckpointError(
            f"Checkpoint model keys mismatch: missing={missing}, unexpected={unexpected}"
        )
    for name, expected in current.items():
        incoming = state[name]
        if not isinstance(incoming, torch.Tensor):
            raise CheckpointError(f"Checkpoint model tensor {name} is not a Tensor")
        if incoming.shape != expected.shape:
            raise CheckpointError(
                f"Checkpoint model tensor {name} shape mismatch: "
                f"{tuple(incoming.shape)} != {tuple(expected.shape)}"
            )
        if incoming.dtype != expected.dtype:
            raise CheckpointError(
                f"Checkpoint model tensor {name} dtype mismatch: "
                f"{incoming.dtype} != {expected.dtype}"
            )
        if (incoming.is_floating_point() or incoming.is_complex()) and not torch.isfinite(
            incoming
        ).all():
            raise CheckpointError(f"Checkpoint model tensor {name} contains NaN/Inf")


def load_checkpoint(
    path: Path,
    *,
    expected_sha256: str,
    expected_identity: CheckpointIdentity,
    expected_loader_contract: LoaderContract,
    core: LocalTrainerCore,
    grad_scaler: Any | None,
    map_location: torch.device | str = "cpu",
    strict_cuda_rng: bool = True,
) -> LoadedCheckpoint:
    """Verify bytes and semantics before mutating runtime state."""

    actual_sha = file_sha256(path)
    if actual_sha != expected_sha256:
        raise CheckpointError(f"Checkpoint SHA-256 mismatch: {actual_sha} != {expected_sha256}")

    # Trusted internal artifact only. weights_only=False is required for optimizer/RNG state.
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise CheckpointError("Checkpoint payload must be a dict")
    _validate_payload_header(
        payload,
        expected_identity=expected_identity,
        expected_loader_contract=expected_loader_contract,
    )

    scheduler_state = payload["scheduler_state_dict"]
    if (core.scheduler is None) != (scheduler_state is None):
        raise CheckpointError("Scheduler presence does not match checkpoint")
    scaler_state = payload["grad_scaler_state"]
    if (grad_scaler is None) != (scaler_state is None):
        raise CheckpointError("GradScaler presence does not match checkpoint")
    _validate_rng_compatibility(payload["rng_state"], strict_cuda=strict_cuda_rng)

    _validate_model_state(core.model, payload["model_state_dict"])
    cursor = TrainingCursor.from_dict(payload["cursor"])
    run_id = payload.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise CheckpointError("Checkpoint run_id must be a non-empty string")
    attempt = payload.get("attempt")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise CheckpointError("Checkpoint attempt must be an integer >= 1")
    if "best_criterion" not in payload:
        raise CheckpointError("Checkpoint best_criterion is missing")
    best_criterion = payload["best_criterion"]
    if best_criterion is not None:
        if (
            isinstance(best_criterion, bool)
            or not isinstance(best_criterion, (int, float))
            or not math.isfinite(float(best_criterion))
        ):
            raise CheckpointError("Checkpoint best_criterion must be finite or None")
        best_criterion = float(best_criterion)
    loaded = LoadedCheckpoint(
        cursor=cursor,
        best_criterion=best_criterion,
        run_id=run_id,
        attempt=attempt,
    )
    snapshot = {
        "model": {key: value.detach().clone() for key, value in core.model.state_dict().items()},
        "optimizer": copy.deepcopy(core.optimizer.state_dict()),
        "scheduler": (
            None if core.scheduler is None else copy.deepcopy(core.scheduler.state_dict())
        ),
        "grad_scaler": (None if grad_scaler is None else copy.deepcopy(grad_scaler.state_dict())),
        "optimizer_step_count": core.optimizer_step_count,
        "rng": capture_rng_state(),
    }

    try:
        core.model.load_state_dict(payload["model_state_dict"], strict=True)
        core.optimizer.load_state_dict(payload["optimizer_state_dict"])
        if core.scheduler is not None:
            core.scheduler.load_state_dict(scheduler_state)
        if grad_scaler is not None:
            grad_scaler.load_state_dict(scaler_state)
        restore_rng_state(payload["rng_state"], strict_cuda=strict_cuda_rng)
        core.optimizer_step_count = cursor.optimizer_step
    except Exception as exc:
        try:
            core.model.load_state_dict(snapshot["model"], strict=True)
            core.optimizer.load_state_dict(snapshot["optimizer"])
            if core.scheduler is not None:
                core.scheduler.load_state_dict(snapshot["scheduler"])
            if grad_scaler is not None:
                grad_scaler.load_state_dict(snapshot["grad_scaler"])
            core.optimizer_step_count = snapshot["optimizer_step_count"]
            restore_rng_state(snapshot["rng"], strict_cuda=strict_cuda_rng)
        except (KeyError, RuntimeError, TypeError, ValueError) as rollback_exc:
            raise CheckpointError(
                f"Checkpoint load failed and runtime rollback also failed: {rollback_exc}"
            ) from exc
        raise CheckpointError(
            "Checkpoint state application failed; runtime state was restored"
        ) from exc

    return loaded
