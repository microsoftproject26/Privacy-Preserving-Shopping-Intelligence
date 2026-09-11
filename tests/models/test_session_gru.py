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
    config = SessionGRUConfig()
    first_state, first_digest = common_initialization(13, config=config)
    again_state, again_digest = common_initialization(13, config=config)
    other_state, other_digest = common_initialization(42, config=config)

    assert first_digest == again_digest
    assert first_digest != other_digest
    assert set(first_state) == set(other_state)
    for key in first_state:
        assert torch.equal(first_state[key], again_state[key])


def test_an_unapproved_seed_is_refused():
    with pytest.raises(ValueError, match="seed must be one of"):
        common_initialization(7, config=SessionGRUConfig())


def test_common_initialization_refuses_to_guess_the_architecture():
    """Calling it bare used to build the default five-channel encoder.

    R1 uses a two-channel config; the default has five. So `common_initialization(13)` read
    as "the shared starting point" while producing a different architecture with a different
    digest, and a federated lane following the handoff literally would have started
    somewhere the centralized lane never was. The signature now refuses to guess.
    """
    with pytest.raises(TypeError):
        common_initialization(13)


def test_the_selected_and_default_architectures_are_not_interchangeable():
    """The reason the argument is required, stated as a measurement."""
    selected = SessionGRUConfig(channels=("category_id", "event_type_id"))
    _, selected_digest = common_initialization(13, config=selected)
    _, default_digest = common_initialization(13, config=SessionGRUConfig())
    assert selected_digest != default_digest


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


def test_t3_uses_the_query_item(windows) -> None:
    """T3 must depend on the query product, and the check must survive zero-init.

    The first T3 reranker scored `session . candidate` and the query never entered it. The
    frozen retrieval order it was competing against is co-occurrence between the *query item*
    and each candidate, so the model could not represent what the baseline does - let alone
    improve on it - and the resulting "learning loses" conclusion was about a dot product
    rather than about T3.

    Permuting the query tensors across a batch must move the T3 scores. It must also leave
    T1 alone, which is what proves the permutation is reaching the query path specifically
    and not perturbing the batch at large.

    The cross head's final layer is zero-initialised, so an untrained model emits zero for
    any input and cannot tell "unused" from "currently zero". The weights are therefore
    given real values first, which is the state every trained model is in.
    """
    import dataclasses

    spec = phase1_batch_spec_v1()
    batch = windows_to_batch(windows, np.arange(len(windows)), spec)
    model = build_model(13).eval()
    torch.nn.init.normal_(model.t3_cross[-1].weight, std=0.1)
    torch.nn.init.normal_(model.t3_cross[-1].bias, std=0.1)

    with torch.no_grad():
        before = model(batch)
        rolled = {k: torch.roll(v, 1, dims=0)
                  for k, v in batch.query_categorical_ids.items()}
        after = model(dataclasses.replace(batch, query_categorical_ids=rolled))

    moved = (after.t3_scores - before.t3_scores).abs().max().item()
    assert moved > 1e-6, (
        f"permuting the query moved T3 scores by {moved}; T3 is blind to the query item "
        "and cannot represent the query-candidate relationship the baseline is built on")
    unmoved = (after.t1_logits - before.t1_logits).abs().max().item()
    assert unmoved == 0.0, (
        f"permuting the query moved T1 by {unmoved}; the permutation is not isolated to "
        "the query path, so the T3 result above proves nothing")


def test_t3_starts_exactly_at_the_retrieval_order(windows) -> None:
    """An untrained model must score precisely the negated retrieval rank.

    This is what makes every later T3 number a measured departure from the frozen ordering
    rather than a difference between two independent fits. If the cross head emits anything
    at initialisation, epoch 0 is no longer the baseline and the whole comparison loses its
    anchor.
    """
    spec = phase1_batch_spec_v1()
    batch = windows_to_batch(windows, np.arange(len(windows)), spec)
    model = build_model(13).eval()
    with torch.no_grad():
        scores = model(batch).t3_scores
        expected = model.t3_rank_weight * batch.candidate_continuous_features[..., 0]
    assert torch.equal(scores, expected), (
        "the untrained T3 head does not reproduce the rank prior exactly; the zero "
        "initialisation of the cross head's final layer has been lost")


@pytest.mark.parametrize("core", ["gru", "lstm", "tcn", "transformer"])
def test_every_sequence_core_is_causal(core: str, windows) -> None:
    """A core that can see past the decision is reading the future, and its metric is a lie.

    The whole padded sequence goes through every core - none of them packs, because packing
    obstructs ONNX export and `S2-SE-01` has to export whichever wins. That is only
    equivalent to packing if the core cannot look forward, so this changes everything
    *after* each row's decision position and requires the encoded vector not to move.

    A TCN with the padding on the wrong side and a transformer without its causal mask both
    pass every shape and dtype check in this file while quietly failing here.
    """
    spec = phase1_batch_spec_v1()
    batch = windows_to_batch(windows, np.arange(len(windows)), spec)
    model = build_model(13, config=SessionGRUConfig(core=core)).eval()

    with torch.no_grad():
        before = model.encode_history(batch)

        # Overwrite every position strictly after the decision with a different category.
        tampered = {k: v.clone() for k, v in batch.history_categorical_ids.items()}
        length = batch.history_categorical_ids["category_id"].shape[1]
        after_decision = (torch.arange(length).unsqueeze(0)
                          >= batch.lengths.unsqueeze(1))
        for name, tensor in tampered.items():
            vocabulary = next(c.vocab_size for c in spec.history_categorical
                              if c.name == name)
            tensor[after_decision] = (tensor[after_decision] + 1) % vocabulary

        import dataclasses
        after = model.encode_history(
            dataclasses.replace(batch, history_categorical_ids=tampered))

    moved = (after - before).abs().max().item()
    assert moved < 1e-6, (
        f"the {core} core moved by {moved:.2e} when only post-decision padding changed. "
        "It is reading the future, so gathering at lengths-1 is not equivalent to packing "
        "and every number it produces is contaminated.")


def test_the_recurrent_cores_keep_their_published_parameter_names() -> None:
    """Renaming an encoder parameter orphans every checkpoint, and nothing else complains.

    `S2-DS-05` needed a pluggable sequence core, and the obvious way to write one - wrap
    each module so they all return a tensor - renames `encoder.weight_ih_l0` to
    `encoder.module.weight_ih_l0`. Every test in this file still passed. Every checkpoint
    this project has published stopped loading: `s2_ds_01_gru_t1_seed13.pt`, both T2 heads,
    all three joint models.

    The loader's guard caught it, which is the only reason it was a wasted afternoon rather
    than a wrong result. This makes the name itself the contract.
    """
    names = set(build_model(13, config=SessionGRUConfig(core="gru")).state_dict())
    assert "encoder.weight_ih_l0" in names, (
        "the GRU core's parameters are no longer named encoder.weight_ih_l0. Every "
        "published checkpoint is now unloadable.")
    assert not any(key.startswith("encoder.module.") for key in names), (
        "the recurrent core is wrapped again; that renames every encoder parameter")

    lstm = set(build_model(13, config=SessionGRUConfig(core="lstm")).state_dict())
    assert "encoder.weight_ih_l0" in lstm
