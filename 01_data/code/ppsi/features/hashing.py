"""Small deterministic product-hashing feature used before model input."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

HASH_ALGORITHM = "blake2b-64-v1"
PAD_BUCKET = 0
_PERSON = b"ppsi-item-v1"
_UINT64_MAX = (1 << 64) - 1


@dataclass(frozen=True, slots=True)
class ProductHashConfig:
    """Model-facing values that remain explicit until architecture review."""

    bucket_count: int
    seed: int
    residual_dim: int

    def __post_init__(self) -> None:
        for name, value in (
            ("bucket_count", self.bucket_count),
            ("seed", self.seed),
            ("residual_dim", self.residual_dim),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.bucket_count < 2:
            raise ValueError("bucket_count must leave bucket 0 for padding")
        if not 0 <= self.seed <= _UINT64_MAX:
            raise ValueError("seed must fit in uint64")
        if self.residual_dim < 1:
            raise ValueError("residual_dim must be positive")


def hash_product_id(item_id: str, config: ProductHashConfig) -> int:
    """Hash a complete namespaced CanonicalEvent item ID into a non-padding bucket."""

    if not isinstance(config, ProductHashConfig):
        raise TypeError("config must be ProductHashConfig")
    if not isinstance(item_id, str) or not item_id:
        raise ValueError("item_id must be a non-empty string")

    digest = hashlib.blake2b(
        item_id.encode("utf-8"),
        digest_size=8,
        key=config.seed.to_bytes(8, "big"),
        person=_PERSON,
    ).digest()
    return 1 + int.from_bytes(digest, "big") % (config.bucket_count - 1)


def hash_product_ids(
    item_ids: Iterable[str | None],
    *,
    shape: Sequence[int],
    config: ProductHashConfig,
) -> Tensor:
    """Hash flat row-major IDs and reshape them for Phase1Batch."""

    if isinstance(item_ids, (str, bytes)):
        raise TypeError("item_ids must be an iterable, not a string")
    output_shape = tuple(shape)
    if not output_shape or any(
        isinstance(size, bool) or not isinstance(size, int) or size < 0 for size in output_shape
    ):
        raise ValueError("shape must contain non-negative integer dimensions")

    values = list(item_ids)
    expected = math.prod(output_shape)
    if len(values) != expected:
        raise ValueError(f"shape requires {expected} values, got {len(values)}")

    buckets = [
        PAD_BUCKET if item_id is None else hash_product_id(item_id, config) for item_id in values
    ]
    return torch.tensor(buckets, dtype=torch.int64).reshape(output_shape)


class ProductHashEmbedding(nn.Module):
    """Learned product-only residual; category and brand remain separate inputs."""

    def __init__(self, config: ProductHashConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(
            config.bucket_count,
            config.residual_dim,
            padding_idx=PAD_BUCKET,
        )

    def forward(self, bucket_ids: Tensor) -> Tensor:
        if bucket_ids.dtype != torch.int64:
            raise TypeError("bucket_ids must use torch.int64")
        if bucket_ids.numel() and (
            int(bucket_ids.min()) < 0 or int(bucket_ids.max()) >= self.config.bucket_count
        ):
            raise ValueError("bucket_ids are outside the configured range")
        output = self.embedding(bucket_ids)
        return output.masked_fill(bucket_ids.eq(PAD_BUCKET).unsqueeze(-1), 0.0)
