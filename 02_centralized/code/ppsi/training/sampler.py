"""Deterministic within-client batch ordering and resume cursor.

This module does not own federated client sampling. S1-PR-06 owns which
logical clients participate in each server round.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import asdict, dataclass

import torch
from torch.utils.data import Sampler


@dataclass(frozen=True, slots=True)
class TrainingCursor:
    outer_round: int | None
    local_epoch: int
    next_batch_index: int
    optimizer_step: int

    def __post_init__(self) -> None:
        if self.outer_round is not None and self.outer_round < 0:
            raise ValueError("outer_round must be non-negative or None")
        for name in ("local_epoch", "next_batch_index", "optimizer_step"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

    def to_dict(self) -> dict[str, int | None]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, int | None]) -> TrainingCursor:
        return cls(
            outer_round=value["outer_round"],
            local_epoch=int(value["local_epoch"]),
            next_batch_index=int(value["next_batch_index"]),
            optimizer_step=int(value["optimizer_step"]),
        )


@dataclass(frozen=True, slots=True)
class LoaderContract:
    schema: str
    version: str
    dataset_identity_sha256: str
    dataset_length: int
    batch_size: int
    drop_last: bool
    sampler_version: str
    run_seed: int
    num_workers: int = 0

    def __post_init__(self) -> None:
        if self.schema != "loader_contract_v1" or self.version != "1":
            raise ValueError("Unsupported loader contract")
        if len(self.dataset_identity_sha256) != 64:
            raise ValueError("dataset_identity_sha256 must contain 64 hex characters")
        int(self.dataset_identity_sha256, 16)
        if self.dataset_length < 0:
            raise ValueError("dataset_length must be non-negative")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_workers != 0:
            raise ValueError("Exact-replay P0 path requires num_workers=0")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> LoaderContract:
        return cls(
            schema=str(value["schema"]),
            version=str(value["version"]),
            dataset_identity_sha256=str(value["dataset_identity_sha256"]),
            dataset_length=int(value["dataset_length"]),
            batch_size=int(value["batch_size"]),
            drop_last=bool(value["drop_last"]),
            sampler_version=str(value["sampler_version"]),
            run_seed=int(value["run_seed"]),
            num_workers=int(value["num_workers"]),
        )


def derive_epoch_seed(contract: LoaderContract, epoch: int) -> int:
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    material = (
        f"{contract.sampler_version}|{contract.run_seed}|"
        f"{contract.dataset_identity_sha256}|{contract.dataset_length}|{epoch}"
    ).encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**63 - 1)


def deterministic_epoch_indices(contract: LoaderContract, epoch: int) -> list[int]:
    generator = torch.Generator().manual_seed(derive_epoch_seed(contract, epoch))
    return torch.randperm(contract.dataset_length, generator=generator).tolist()


def deterministic_epoch_batches(contract: LoaderContract, epoch: int) -> list[list[int]]:
    indices = deterministic_epoch_indices(contract, epoch)
    batches = [
        indices[start : start + contract.batch_size]
        for start in range(0, len(indices), contract.batch_size)
    ]
    if contract.drop_last and batches and len(batches[-1]) < contract.batch_size:
        batches.pop()
    return batches


class DeterministicBatchSampler(Sampler[list[int]]):
    """Regenerate a stable epoch permutation and resume from next_batch_index."""

    def __init__(
        self,
        contract: LoaderContract,
        *,
        epoch: int,
        next_batch_index: int = 0,
    ) -> None:
        self.contract = contract
        self.epoch = epoch
        self.next_batch_index = next_batch_index
        self._batches = deterministic_epoch_batches(contract, epoch)
        if not 0 <= next_batch_index <= len(self._batches):
            raise ValueError("next_batch_index is outside the epoch batch range")

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._batches[self.next_batch_index :]

    def __len__(self) -> int:
        return len(self._batches) - self.next_batch_index

    @property
    def total_batch_count(self) -> int:
        return len(self._batches)
