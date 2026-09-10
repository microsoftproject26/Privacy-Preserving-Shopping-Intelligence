"""Contract and semantics tests for the shared session encoder.

Six rows before three million. The left-padding defect that drove MRR@20 to 0.4509 -
below a baseline that learns nothing - passed every shape, dtype and range check that
existed at the time, so the checks here are about *content*: what the encoder is actually
handed, and whether two ways of reading a padded sequence agree.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn.utils.rnn import pack_padded_sequence

from ppsi.data.batching import windows_to_batch
from ppsi.data.sequences import build_windows
from ppsi.models.batch_spec import (
    CATEGORY_OOV,
    CATEGORY_PAD,
    EVENT_PAD,
    PRICE_BAND_PAD,
    phase1_batch_spec_v1,
)
from ppsi.models.session_gru import (
    SessionGRUConfig,
    build_model,
    common_initialization,
    parameter_count,
)
from ppsi.training.batch import validate_canonical_phase1_batch
from ppsi.training.core import LocalTrainerCore
from ppsi.training.objective import ContractSmokeObjective
from ppsi.training.outputs import validate_raw_model_output
from ppsi.training.protocol import Phase1Model
from ppsi.training.sampler import TrainingCursor
from ppsi.training.state import SharedStateSpec

HISTORY = 5


@pytest.fixture
def frame() -> pd.DataFrame:
    """Two sessions belonging to two clients, canonically ordered.

    Session 100 runs four events; session 200 runs two. One event carries a category
    unseen in TRAIN, encoded -1 exactly as the upstream builders encode it.
    """
    return pd.DataFrame(
        {
            "session": [100, 100, 100, 100, 200, 200],
            "order": [0, 1, 2, 3, 4, 5],
            "user": [7, 7, 7, 7, 9, 9],
            "category": np.array([3, 3, 11, -1, 42, 5], dtype="int32"),
            "product_bucket": np.array([31, 31, 77, 88, 12, 19], dtype="int32"),
            "event_code": np.array([1, 1, 2, 1, 1, 3], dtype="int8"),
            "brand_bucket": np.array([4, 4, 6, 6, 2, 2], dtype="int32"),
            "price_band": np.array([1, 1, 3, 2, 4, 0], dtype="int8"),
            "event_time": pd.to_datetime(
                [
                    "2019-10-01 00:00:00",
                    "2019-10-01 00:00:10",
                    "2019-10-01 00:01:00",
                    "2019-10-01 00:01:05",
                    "2019-10-02 08:00:00",
                    "2019-10-02 08:00:30",
                ],
                utc=True,
            ),
        }
    )


@pytest.fixture
def windows(frame):
    decisions = np.array([1, 2, 3, 5])
    targets = np.array([11, 5, 42, 3])
    return build_windows(frame, decisions, targets, history_length=HISTORY)


# --- the window itself ------------------------------------------------------------


def test_history_is_right_padded_with_the_decision_last(windows, frame):
    """Real events occupy 0..lengths-1 and the decision event sits at lengths-1.

    This is defect #1. Left-padded, `pack_padded_sequence` and a gather at lengths-1
    both read pure padding for every window shorter than L - which is most of them.
    """
    decisions = np.array([1, 2, 3, 5])
    expected = frame["category"].to_numpy()[decisions]
    expected = np.where(expected < 0, CATEGORY_OOV, expected)
    last = windows.category[np.arange(len(windows)), windows.lengths - 1]
    assert np.array_equal(last, expected), (
        f"the decision's own category should sit at lengths-1; got {last} want {expected}"
    )


def test_padding_sits_after_the_real_events(windows):
    columns = np.arange(windows.history_length)[None, :]
    padded = columns >= windows.lengths[:, None]
    assert np.all(windows.category[padded] == CATEGORY_PAD)
    assert np.all(windows.product[padded] == 0)
    assert np.all(windows.event[padded] == EVENT_PAD)
    assert np.all(windows.price_band[padded] == PRICE_BAND_PAD)
    assert np.all(windows.gap[padded] == 0.0)


def test_the_window_never_reaches_into_a_previous_session(windows):
    """Decision at order 5 is the second event of session 200, so its history is 2."""
    assert windows.lengths.tolist() == [2, 3, 4, 2]


def test_an_unseen_category_becomes_oov_as_an_input(windows):
    """-1 disqualifies a decision as a *target*, upstream. As an *input* it is OOV."""
    assert CATEGORY_OOV in windows.category
    assert -1 not in windows.category


def test_the_gap_is_zero_at_a_session_start_and_positive_after(windows):
    assert np.all(windows.gap >= 0.0)
    # Decision at order 1 has history [order 0, order 1]; the first is the session start.
    assert windows.gap[0, 0] == 0.0
    assert windows.gap[0, 1] == pytest.approx(float(np.log1p(10)), rel=1e-6)


def test_the_gap_channel_actually_carries_values(windows):
    """A regression test for a channel that died silently.

    S2-SMOKE converted timestamps with `astype("int64") // 10**9`, which is correct only
    for datetime64[ns]. Under pandas 3 the column is datetime64[us], the divisor is a
    thousand times too large, and every gap floors to zero. No exception, no failing
    shape check - the feature is simply absent, and an ablation would have reported that
    inter-event timing carries no signal.

    Asserting the channel is non-constant is what would have caught it.
    """
    real = windows.gap[np.arange(windows.history_length)[None, :] < windows.lengths[:, None]]
    assert real.max() > 0.0, "every time gap is zero; the timestamp unit is being assumed"
    assert len(np.unique(real)) > 1, "the gap channel is constant and therefore carries nothing"


# --- the batch contract -----------------------------------------------------------


def test_the_adapter_produces_a_canonically_valid_batch(windows):
    spec = phase1_batch_spec_v1()
    batch = windows_to_batch(windows, np.arange(len(windows)), spec)
    validate_canonical_phase1_batch(batch, spec)
    assert batch.batch_size == 4
    assert batch.history_width == HISTORY


def test_the_history_mask_matches_the_lengths_exactly(windows):
    spec = phase1_batch_spec_v1()
    batch = windows_to_batch(windows, np.arange(len(windows)), spec)
    expected = torch.arange(HISTORY).unsqueeze(0) < batch.lengths.unsqueeze(1)
    assert torch.equal(batch.history_mask, expected)


# --- the model contract -----------------------------------------------------------


def test_the_model_satisfies_the_phase1_model_protocol():
    model = build_model(13)
    assert isinstance(model, Phase1Model)
    assert isinstance(model.category_count, int)


def test_the_output_passes_the_raw_output_validator(windows):
    spec = phase1_batch_spec_v1()
    batch = windows_to_batch(windows, np.arange(len(windows)), spec)
    model = build_model(13)
    output = model(batch)
    validate_raw_model_output(output, batch, category_count=model.category_count)
    assert output.t1_logits.shape == (4, model.category_count)
    assert output.t2_logit.shape == (4, 1)
    assert output.t3_scores.shape == (4, batch.candidate_width)


def test_shared_state_is_all_floating_so_federated_averaging_can_build():
    """BatchNorm would fail this: its num_batches_tracked buffer is int64."""
    model = build_model(13)
    spec = SharedStateSpec.all_shared_floating(model)
    assert spec.shared_keys
    spec.validate_model(model)


def test_one_trainer_core_step_runs_end_to_end(windows):
    """We do not train through the core, but the federated lane does, so it must work."""
    spec = phase1_batch_spec_v1()
    batch = windows_to_batch(windows, np.arange(len(windows)), spec)
    model = build_model(13)
    core = LocalTrainerCore(
        model=model,
        batch_spec=spec,
        objective=ContractSmokeObjective(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        device="cpu",
    )
    summary = core.train_step(batch)
    assert summary.optimizer_step_performed
    assert summary.contributing_examples == 4
    _ = TrainingCursor(None, 0, 0, 0)


# --- the readout decision ---------------------------------------------------------


def test_gathering_at_lengths_minus_one_equals_packing(windows):
    """The export-friendly readout must not change the answer.

    `pack_padded_sequence` obstructs the ONNX export S2-SE-01 owes, so the encoder runs
    the whole padded sequence and gathers. With right-padding the two are identical - and
    if they ever disagree, the padding is on the wrong side, which is defect #1
    announcing itself a second way.
    """
    spec = phase1_batch_spec_v1()
    batch = windows_to_batch(windows, np.arange(len(windows)), spec)
    model = build_model(13).eval()

    with torch.no_grad():
        gathered = model.encode_history(batch)

        parts = [
            model.history_embeddings[name](batch.history_categorical_ids[name])
            for name in model.config.channels
        ]
        parts.append(batch.history_continuous_features)
        projected = torch.tanh(model.input_projection(torch.cat(parts, dim=-1)))
        packed = pack_padded_sequence(
            projected, batch.lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = model.encoder(packed)
        packed_result = hidden[-1]

    difference = (gathered - packed_result).abs().max().item()
    assert difference < 1e-5, f"gather and pack disagree by {difference:.2e}"


# --- reproducibility for the federated lane ---------------------------------------


def test_common_initialization_is_reproducible_and_seed_specific():
    """S2-PR-06 must start R2a from exactly where the matching R1 started."""
    first_state, first_digest = common_initialization(13)
    again_state, again_digest = common_initialization(13)
    other_state, other_digest = common_initialization(42)

    assert first_digest == again_digest
    assert first_digest != other_digest
    assert set(first_state) == set(other_state)
    for key in first_state:
        assert torch.equal(first_state[key], again_state[key])


def test_an_unapproved_seed_is_refused():
    with pytest.raises(ValueError, match="seed must be one of"):
        common_initialization(7)


def test_building_a_model_does_not_disturb_global_rng():
    """Otherwise two runs differing only in architecture also differ in batch order."""
    torch.manual_seed(99)
    before = torch.randn(3)
    torch.manual_seed(99)
    build_model(13)
    after = torch.randn(3)
    assert torch.equal(before, after)


# --- the ablation ladder ----------------------------------------------------------


def test_the_model_can_consume_a_subset_of_channels(windows):
    """The batch always carries all five; the model decides what to read.

    This is what lets the spec stay frozen while the ladder adds one channel at a time.
    """
    spec = phase1_batch_spec_v1()
    batch = windows_to_batch(windows, np.arange(len(windows)), spec)
    smoke_replica = SessionGRUConfig(channels=("category_id", "event_type_id"), use_gap=True)
    model = build_model(13, config=smoke_replica)
    output = model(batch)
    validate_raw_model_output(output, batch, category_count=model.category_count)
    assert parameter_count(model) < parameter_count(build_model(13))


def test_a_zero_length_history_encodes_to_exactly_zero():
    """The contract's ZERO_HIDDEN case: no history must mean no contribution."""
    spec = phase1_batch_spec_v1()
    model = build_model(13).eval()

    frame = pd.DataFrame(
        {
            "session": [1],
            "order": [0],
            "user": [1],
            "category": np.array([2], dtype="int32"),
            "product_bucket": np.array([5], dtype="int32"),
            "event_code": np.array([1], dtype="int8"),
            "brand_bucket": np.array([3], dtype="int32"),
            "price_band": np.array([1], dtype="int8"),
            "event_time": pd.to_datetime(["2019-10-01"], utc=True),
        }
    )
    built = build_windows(frame, np.array([0]), np.array([2]), history_length=3)
    batch = windows_to_batch(built, np.array([0]), spec)
    # Phase1Batch is frozen with slots, so it has no __dict__; clone-and-replace is the
    # idiom the package's own fixtures use.
    zeroed = replace(
        batch,
        lengths=torch.zeros_like(batch.lengths),
        history_mask=torch.zeros_like(batch.history_mask),
    )
    with torch.no_grad():
        assert torch.all(model.encode_history(zeroed) == 0)
