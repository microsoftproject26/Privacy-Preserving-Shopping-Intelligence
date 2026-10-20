from __future__ import annotations

import pytest
import torch

from ppsi.features.hashing import (
    PAD_BUCKET,
    ProductHashConfig,
    ProductHashEmbedding,
    hash_product_id,
    hash_product_ids,
)


def config(*, seed: int = 13) -> ProductHashConfig:
    return ProductHashConfig(bucket_count=257, seed=seed, residual_dim=8)


def test_hash_is_stable_namespaced_and_never_padding() -> None:
    value = hash_product_id("rees46:item:44600062", config())

    assert value == 33
    assert value != PAD_BUCKET
    assert value != hash_product_id("other:item:44600062", config())
    assert value != hash_product_id("rees46:item:44600062", config(seed=14))


def test_config_and_item_id_validation() -> None:
    with pytest.raises(ValueError):
        ProductHashConfig(bucket_count=1, seed=13, residual_dim=8)
    with pytest.raises(ValueError):
        ProductHashConfig(bucket_count=257, seed=-1, residual_dim=8)
    with pytest.raises(ValueError):
        hash_product_id("", config())
    with pytest.raises(ValueError):
        hash_product_id(None, config())


def test_batch_hashing_preserves_shape_dtype_and_padding() -> None:
    buckets = hash_product_ids(
        ["rees46:item:1", None, "rees46:item:2", "rees46:item:3"],
        shape=(2, 2),
        config=config(),
    )

    assert buckets.dtype == torch.int64
    assert tuple(buckets.shape) == (2, 2)
    assert buckets[0, 1].item() == PAD_BUCKET


def test_embedding_preserves_shape_and_zeros_padding() -> None:
    cfg = config()
    buckets = hash_product_ids(
        ["rees46:item:1", None, "rees46:item:2", "rees46:item:3"],
        shape=(2, 2),
        config=cfg,
    )
    output = ProductHashEmbedding(cfg)(buckets)

    assert tuple(output.shape) == (2, 2, cfg.residual_dim)
    assert torch.equal(output[0, 1], torch.zeros(cfg.residual_dim))
