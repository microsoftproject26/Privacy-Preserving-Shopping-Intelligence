"""Deterministic synthetic fixtures for S1-PR-05 contract and smoke tests.

No real REES46 data is used here. Real REES46 TaskExamples enter through
S1-PR-07 (#20) after the interface is frozen.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from ppsi.training.batch import (
    CategoricalChannelSpec,
    Phase1Batch,
    Phase1BatchSpec,
    validate_canonical_phase1_batch,
)
from ppsi.training.stub_model import Phase1StubModel, StubModelConfig


def default_batch_spec() -> Phase1BatchSpec:
    return Phase1BatchSpec(
        history_categorical=(
            CategoricalChannelSpec("item_id", pad_id=0, vocab_size=64),
            CategoricalChannelSpec("category_id", pad_id=0, vocab_size=16),
            CategoricalChannelSpec("event_type_id", pad_id=0, vocab_size=8),
        ),
        query_categorical=(CategoricalChannelSpec("query_context_id", pad_id=0, vocab_size=8),),
        candidate_categorical=(
            CategoricalChannelSpec("candidate_category_id", pad_id=0, vocab_size=16),
        ),
        history_continuous_dim=2,
        query_continuous_dim=2,
        candidate_continuous_dim=1,
        candidate_id_pad_id=0,
        candidate_id_vocab_size=64,
    )


def default_stub_model_config() -> StubModelConfig:
    return StubModelConfig(category_count=6, embedding_dim=4, hidden_dim=8)


def make_stub_model(seed: int = 13) -> Phase1StubModel:
    if seed not in {13, 42, 2026}:
        raise ValueError("seed must be one of 13, 42, 2026")
    # fork_rng prevents fixture construction from mutating the caller's RNG state.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return Phase1StubModel(
            batch_spec=default_batch_spec(),
            config=default_stub_model_config(),
        )


def make_phase1_batch(
    *,
    batch_size: int = 3,
    seed: int = 13,
    client_id: int = 0,
) -> Phase1Batch:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    spec = default_batch_spec()
    history_width = 4
    candidate_width = 5
    generator = torch.Generator().manual_seed(seed + 1009 * client_id + batch_size)

    base_lengths = [4, 2, 0]
    lengths = torch.tensor(
        [base_lengths[i % len(base_lengths)] for i in range(batch_size)],
        dtype=torch.int64,
    )
    history_mask = torch.arange(history_width).unsqueeze(0) < lengths.unsqueeze(1)

    history_channels: dict[str, torch.Tensor] = {}
    for channel in spec.history_categorical:
        values = torch.randint(
            1,
            channel.vocab_size or 2,
            (batch_size, history_width),
            generator=generator,
            dtype=torch.int64,
        )
        values[~history_mask] = channel.pad_id
        history_channels[channel.name] = values

    history_continuous = torch.randn(
        batch_size,
        history_width,
        spec.history_continuous_dim,
        generator=generator,
        dtype=torch.float32,
    )
    history_continuous[~history_mask] = 0.0

    query_channels = {
        channel.name: torch.randint(
            1,
            channel.vocab_size or 2,
            (batch_size,),
            generator=generator,
            dtype=torch.int64,
        )
        for channel in spec.query_categorical
    }
    query_continuous = torch.randn(
        batch_size,
        spec.query_continuous_dim,
        generator=generator,
        dtype=torch.float32,
    )

    valid_candidate_counts = [5, 3, 0]
    candidate_mask = torch.zeros(batch_size, candidate_width, dtype=torch.bool)
    for row in range(batch_size):
        candidate_mask[row, : valid_candidate_counts[row % 3]] = True

    candidate_ids = torch.randint(
        1,
        spec.candidate_id_vocab_size or 2,
        (batch_size, candidate_width),
        generator=generator,
        dtype=torch.int64,
    )
    candidate_ids[~candidate_mask] = spec.candidate_id_pad_id

    candidate_channels: dict[str, torch.Tensor] = {}
    for channel in spec.candidate_categorical:
        values = torch.randint(
            1,
            channel.vocab_size or 2,
            (batch_size, candidate_width),
            generator=generator,
            dtype=torch.int64,
        )
        values[~candidate_mask] = channel.pad_id
        candidate_channels[channel.name] = values

    candidate_continuous = torch.randn(
        batch_size,
        candidate_width,
        spec.candidate_continuous_dim,
        generator=generator,
        dtype=torch.float32,
    )
    candidate_continuous[~candidate_mask] = 0.0

    t1_present = torch.tensor(
        [(i % 3) != 2 for i in range(batch_size)],
        dtype=torch.bool,
    )
    t2_present = torch.tensor(
        [(i % 3) == 0 for i in range(batch_size)],
        dtype=torch.bool,
    )
    t3_present = torch.tensor(
        [(i % 3) == 0 for i in range(batch_size)],
        dtype=torch.bool,
    )

    t1_target = (
        torch.arange(batch_size, dtype=torch.int64) % default_stub_model_config().category_count
    )
    t1_target[~t1_present] = spec.t1_absent_fill
    t2_target = torch.zeros(batch_size, 1, dtype=torch.float32)
    t2_target[t2_present] = 1.0
    t3_gains = torch.zeros(batch_size, candidate_width, dtype=torch.float32)
    for row in range(batch_size):
        if t3_present[row]:
            t3_gains[row, 0] = 3.0
            if candidate_width > 1 and candidate_mask[row, 1]:
                t3_gains[row, 1] = 1.0

    batch = Phase1Batch(
        history_categorical_ids=history_channels,
        history_continuous_features=history_continuous,
        lengths=lengths,
        history_mask=history_mask,
        query_categorical_ids=query_channels,
        query_continuous_features=query_continuous,
        candidate_ids=candidate_ids,
        candidate_categorical_ids=candidate_channels,
        candidate_continuous_features=candidate_continuous,
        candidate_mask=candidate_mask,
        t1_target=t1_target,
        t2_target=t2_target,
        t3_gains=t3_gains,
        t1_present=t1_present,
        t2_present=t2_present,
        t3_present=t3_present,
    )
    validate_canonical_phase1_batch(batch, spec)
    return batch


def make_no_contribution_batch() -> Phase1Batch:
    batch = make_phase1_batch(batch_size=1)
    return replace(
        batch,
        t1_target=torch.zeros_like(batch.t1_target),
        t2_target=torch.zeros_like(batch.t2_target),
        t3_gains=torch.zeros_like(batch.t3_gains),
        t1_present=torch.zeros_like(batch.t1_present),
        t2_present=torch.zeros_like(batch.t2_present),
        t3_present=torch.zeros_like(batch.t3_present),
        candidate_ids=torch.zeros_like(batch.candidate_ids),
        candidate_categorical_ids={
            name: torch.zeros_like(values)
            for name, values in batch.candidate_categorical_ids.items()
        },
        candidate_continuous_features=torch.zeros_like(batch.candidate_continuous_features),
        candidate_mask=torch.zeros_like(batch.candidate_mask),
    )


def make_client_batches(logical_client_id: int) -> list[Phase1Batch]:
    if logical_client_id < 0:
        raise ValueError("logical_client_id must be non-negative")
    # Unequal client contribution sizes exercise weighted Flower aggregation.
    batch_size = 2 if logical_client_id == 0 else 4
    return [make_phase1_batch(batch_size=batch_size, client_id=logical_client_id)]
