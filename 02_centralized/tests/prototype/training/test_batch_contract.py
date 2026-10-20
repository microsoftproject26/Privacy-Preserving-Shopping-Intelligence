from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from ppsi.training.batch import (
    BatchValidationError,
    validate_canonical_phase1_batch,
    validate_phase1_batch,
)
from ppsi.training.fixtures import make_phase1_batch


def test_valid_batch_passes_both_validators(phase1_batch, batch_spec):
    validate_phase1_batch(phase1_batch, batch_spec)
    validate_canonical_phase1_batch(phase1_batch, batch_spec)


def test_b_equals_one_and_zero_history_are_valid(batch_spec):
    batch = make_phase1_batch(batch_size=1)
    assert batch.lengths.tolist() == [4]
    validate_canonical_phase1_batch(batch, batch_spec)

    empty = replace(
        batch,
        lengths=torch.zeros_like(batch.lengths),
        history_mask=torch.zeros_like(batch.history_mask),
        history_categorical_ids={
            name: torch.zeros_like(values) for name, values in batch.history_categorical_ids.items()
        },
        history_continuous_features=torch.zeros_like(batch.history_continuous_features),
    )
    validate_canonical_phase1_batch(empty, batch_spec)


def test_wrong_history_dtype_fails(phase1_batch, batch_spec):
    channels = dict(phase1_batch.history_categorical_ids)
    channels["item_id"] = channels["item_id"].to(torch.int32)
    with pytest.raises(BatchValidationError, match="expected torch.int64"):
        validate_phase1_batch(replace(phase1_batch, history_categorical_ids=channels), batch_spec)


def test_length_mask_mismatch_fails(phase1_batch, batch_spec):
    bad_mask = phase1_batch.history_mask.clone()
    bad_mask[0, -1] = False
    with pytest.raises(BatchValidationError, match="right-padding"):
        validate_phase1_batch(replace(phase1_batch, history_mask=bad_mask), batch_spec)


def test_left_padding_fails(phase1_batch, batch_spec):
    lengths = torch.tensor([2, 2, 0], dtype=torch.int64)
    mask = torch.tensor(
        [[False, False, True, True], [True, True, False, False], [False, False, False, False]],
        dtype=torch.bool,
    )
    with pytest.raises(BatchValidationError, match="right-padding"):
        validate_phase1_batch(replace(phase1_batch, lengths=lengths, history_mask=mask), batch_spec)


def test_non_contiguous_mask_fails(phase1_batch, batch_spec):
    mask = phase1_batch.history_mask.clone()
    mask[0] = torch.tensor([True, False, True, True])
    with pytest.raises(BatchValidationError, match="right-padding"):
        validate_phase1_batch(replace(phase1_batch, history_mask=mask), batch_spec)


def test_negative_or_too_large_length_fails(phase1_batch, batch_spec):
    for bad_length in (-1, phase1_batch.history_width + 1):
        lengths = phase1_batch.lengths.clone()
        lengths[0] = bad_length
        with pytest.raises(BatchValidationError, match="0 <= length <= L"):
            validate_phase1_batch(replace(phase1_batch, lengths=lengths), batch_spec)


def test_candidate_alignment_mismatch_fails(phase1_batch, batch_spec):
    with pytest.raises(BatchValidationError, match="candidate_mask"):
        validate_phase1_batch(
            replace(phase1_batch, candidate_mask=phase1_batch.candidate_mask[:, :-1]),
            batch_spec,
        )


def test_t3_present_requires_valid_candidate(phase1_batch, batch_spec):
    candidate_mask = phase1_batch.candidate_mask.clone()
    candidate_mask[0] = False
    with pytest.raises(BatchValidationError, match="requires at least one valid candidate"):
        validate_phase1_batch(replace(phase1_batch, candidate_mask=candidate_mask), batch_spec)


def test_all_candidates_masked_is_legal_when_t3_absent(phase1_batch, batch_spec):
    batch = replace(
        phase1_batch,
        candidate_ids=torch.zeros_like(phase1_batch.candidate_ids),
        candidate_categorical_ids={
            name: torch.zeros_like(values)
            for name, values in phase1_batch.candidate_categorical_ids.items()
        },
        candidate_continuous_features=torch.zeros_like(phase1_batch.candidate_continuous_features),
        candidate_mask=torch.zeros_like(phase1_batch.candidate_mask),
        t3_gains=torch.zeros_like(phase1_batch.t3_gains),
        t3_present=torch.zeros_like(phase1_batch.t3_present),
    )
    validate_canonical_phase1_batch(batch, batch_spec)


def test_runtime_validator_allows_finite_poisoned_padding_but_canonical_rejects(
    phase1_batch, batch_spec
):
    history = phase1_batch.history_continuous_features.clone()
    history[~phase1_batch.history_mask] = 1234.5
    poisoned = replace(phase1_batch, history_continuous_features=history)
    validate_phase1_batch(poisoned, batch_spec)
    with pytest.raises(BatchValidationError, match="canonical zero"):
        validate_canonical_phase1_batch(poisoned, batch_spec)


def test_canonical_validator_rejects_pad_id_in_valid_history_position(
    phase1_batch, batch_spec
):
    channels = dict(phase1_batch.history_categorical_ids)
    channels["item_id"] = channels["item_id"].clone()
    channels["item_id"][phase1_batch.history_mask] = batch_spec.history_categorical[0].pad_id
    with pytest.raises(BatchValidationError, match="pad ID is not valid data"):
        validate_canonical_phase1_batch(
            replace(phase1_batch, history_categorical_ids=channels),
            batch_spec,
        )


def test_canonical_validator_rejects_pad_id_in_valid_candidate_position(
    phase1_batch, batch_spec
):
    candidate_ids = phase1_batch.candidate_ids.clone()
    candidate_ids[phase1_batch.candidate_mask] = batch_spec.candidate_id_pad_id
    with pytest.raises(BatchValidationError, match="pad ID is not valid data"):
        validate_canonical_phase1_batch(
            replace(phase1_batch, candidate_ids=candidate_ids),
            batch_spec,
        )


def test_canonical_validator_rejects_pad_id_in_query_position(phase1_batch, batch_spec):
    channels = dict(phase1_batch.query_categorical_ids)
    channels["query_context_id"] = torch.full_like(
        channels["query_context_id"],
        batch_spec.query_categorical[0].pad_id,
    )
    with pytest.raises(BatchValidationError, match="pad ID is not valid data"):
        validate_canonical_phase1_batch(
            replace(phase1_batch, query_categorical_ids=channels),
            batch_spec,
        )


def test_negative_categorical_or_candidate_ids_fail(phase1_batch, batch_spec):
    channels = dict(phase1_batch.history_categorical_ids)
    channels["item_id"] = channels["item_id"].clone()
    channels["item_id"][0, 0] = -1
    with pytest.raises(BatchValidationError, match="non-negative"):
        validate_phase1_batch(
            replace(phase1_batch, history_categorical_ids=channels),
            batch_spec,
        )

    candidates = phase1_batch.candidate_ids.clone()
    candidates[0, 0] = -1
    with pytest.raises(BatchValidationError, match="non-negative"):
        validate_phase1_batch(replace(phase1_batch, candidate_ids=candidates), batch_spec)


def test_empty_optional_feature_collections_and_zero_width_tensors_are_legal():
    from ppsi.training.batch import Phase1Batch, Phase1BatchSpec

    spec = Phase1BatchSpec(
        history_categorical=(),
        query_categorical=(),
        candidate_categorical=(),
        history_continuous_dim=0,
        query_continuous_dim=0,
        candidate_continuous_dim=0,
        candidate_id_pad_id=0,
        candidate_id_vocab_size=2,
    )
    batch = Phase1Batch(
        history_categorical_ids={},
        history_continuous_features=torch.empty(1, 1, 0, dtype=torch.float32),
        lengths=torch.tensor([0], dtype=torch.int64),
        history_mask=torch.tensor([[False]], dtype=torch.bool),
        query_categorical_ids={},
        query_continuous_features=torch.empty(1, 0, dtype=torch.float32),
        candidate_ids=torch.tensor([[0]], dtype=torch.int64),
        candidate_categorical_ids={},
        candidate_continuous_features=torch.empty(1, 1, 0, dtype=torch.float32),
        candidate_mask=torch.tensor([[False]], dtype=torch.bool),
        t1_target=torch.tensor([0], dtype=torch.int64),
        t2_target=torch.tensor([[0.0]], dtype=torch.float32),
        t3_gains=torch.tensor([[0.0]], dtype=torch.float32),
        t1_present=torch.tensor([False], dtype=torch.bool),
        t2_present=torch.tensor([False], dtype=torch.bool),
        t3_present=torch.tensor([False], dtype=torch.bool),
    )
    validate_canonical_phase1_batch(batch, spec)


def test_present_t2_target_must_be_binary_range(phase1_batch, batch_spec):
    target = phase1_batch.t2_target.clone()
    target[phase1_batch.t2_present] = 1.5
    with pytest.raises(BatchValidationError, match=r"\[0,1\]"):
        validate_phase1_batch(replace(phase1_batch, t2_target=target), batch_spec)
