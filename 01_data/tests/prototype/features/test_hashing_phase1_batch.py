from __future__ import annotations

from dataclasses import replace

import torch

from ppsi.features.hashing import PAD_BUCKET, ProductHashConfig, hash_product_ids
from ppsi.training.batch import CategoricalChannelSpec, validate_canonical_phase1_batch
from ppsi.training.fixtures import default_batch_spec, make_phase1_batch


def test_hashed_products_fit_existing_phase1_batch_channels() -> None:
    cfg = ProductHashConfig(bucket_count=257, seed=13, residual_dim=8)
    batch = make_phase1_batch()
    spec = default_batch_spec()
    batch_size, history_width = batch.history_mask.shape
    _, candidate_width = batch.candidate_mask.shape

    history = hash_product_ids(
        [
            f"rees46:item:h-{row}-{column}" if batch.history_mask[row, column] else None
            for row in range(batch_size)
            for column in range(history_width)
        ],
        shape=(batch_size, history_width),
        config=cfg,
    )
    candidates = hash_product_ids(
        [
            f"rees46:item:c-{row}-{column}" if batch.candidate_mask[row, column] else None
            for row in range(batch_size)
            for column in range(candidate_width)
        ],
        shape=(batch_size, candidate_width),
        config=cfg,
    )

    original_category = batch.history_categorical_ids["category_id"].clone()
    history_channels = dict(batch.history_categorical_ids)
    history_channels["item_id"] = history
    history_specs = tuple(
        CategoricalChannelSpec(channel.name, PAD_BUCKET, cfg.bucket_count)
        if channel.name == "item_id"
        else channel
        for channel in spec.history_categorical
    )

    hashed_spec = replace(
        spec,
        history_categorical=history_specs,
        candidate_id_pad_id=PAD_BUCKET,
        candidate_id_vocab_size=cfg.bucket_count,
    )
    hashed_batch = replace(
        batch,
        history_categorical_ids=history_channels,
        candidate_ids=candidates,
    )

    validate_canonical_phase1_batch(hashed_batch, hashed_spec)
    assert torch.equal(hashed_batch.history_categorical_ids["category_id"], original_category)
