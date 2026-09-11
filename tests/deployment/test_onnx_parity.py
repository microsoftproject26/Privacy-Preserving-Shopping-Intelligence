from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from ppsi.deployment.onnx_export import (
    INPUT_NAMES,
    OPSET_VERSION,
    OUTPUT_NAMES,
    check_graph,
    compare_against_onnx,
    deterministic_example_batch,
    export_session_gru,
)
from ppsi.models.batch_spec import phase1_batch_spec_v1
from ppsi.models.session_gru import build_model

TOLERANCE = 1e-4


@pytest.fixture(scope="module")
def spec():
    return phase1_batch_spec_v1()


@pytest.fixture(scope="module")
def model(spec):
    return build_model(13, batch_spec=spec).eval()


@pytest.fixture(scope="module")
def exported(tmp_path_factory, model, spec):
    destination = tmp_path_factory.mktemp("onnx") / "session_gru.onnx"
    example = deterministic_example_batch(spec, rows=4, history=6, candidates=5)
    return export_session_gru(model, destination, example_batch=example, batch_spec=spec)


def test_the_graph_is_valid_and_pins_its_opset(exported) -> None:
    info = check_graph(exported)

    assert info["opset_imports"]["ai.onnx"] == OPSET_VERSION
    assert tuple(info["graph_inputs"]) == INPUT_NAMES
    assert tuple(info["graph_outputs"]) == OUTPUT_NAMES


def test_outputs_agree_with_pytorch_on_the_traced_shape(model, exported, spec) -> None:
    batch = deterministic_example_batch(spec, rows=4, history=6, candidates=5)

    report = compare_against_onnx(model, exported, batch, tolerance=TOLERANCE)

    assert report.within_tolerance
    assert report.worst_absolute < TOLERANCE
    assert set(report.max_absolute_difference) == set(OUTPUT_NAMES)


@pytest.mark.parametrize("rows", [1, 2, 7, 16])
def test_a_batch_size_other_than_the_traced_one_still_agrees(model, exported, spec, rows) -> None:
    # The indexed gather in `encode_history` used to bake the traced batch size into the
    # graph as a constant, so every other batch size failed inside ONNX Runtime rather
    # than returning a wrong number. This is the regression guard for that.
    batch = deterministic_example_batch(spec, rows=rows, history=6, candidates=5, seed=99)

    report = compare_against_onnx(model, exported, batch, tolerance=TOLERANCE)

    assert report.within_tolerance


@pytest.mark.parametrize("history,candidates", [(1, 5), (12, 5), (6, 1), (6, 20), (12, 20)])
def test_history_length_and_candidate_width_are_genuinely_dynamic(
    model, exported, spec, history, candidates
) -> None:
    batch = deterministic_example_batch(
        spec, rows=3, history=history, candidates=candidates, seed=7
    )

    report = compare_against_onnx(model, exported, batch, tolerance=TOLERANCE)

    assert report.within_tolerance


def test_an_empty_history_row_agrees_and_contributes_zero(model, exported, spec) -> None:
    batch = deterministic_example_batch(spec, rows=3, history=5, candidates=4)
    assert int(batch.lengths[0]) == 0

    report = compare_against_onnx(model, exported, batch, tolerance=TOLERANCE)

    assert report.within_tolerance


def test_padding_positions_do_not_change_the_outputs(model, spec) -> None:
    # Padded positions must be inert. Rewriting what sits in them has to leave every head
    # unchanged, otherwise the export is reading past the decision.
    batch = deterministic_example_batch(spec, rows=3, history=6, candidates=4, seed=5)
    mask = batch.history_mask
    mutated_ids = {
        name: torch.where(mask, tensor, torch.full_like(tensor, channel.pad_id))
        for (name, tensor), channel in zip(
            batch.history_categorical_ids.items(), spec.history_categorical
        )
    }
    for name, tensor in mutated_ids.items():
        assert torch.equal(tensor, batch.history_categorical_ids[name])

    with torch.no_grad():
        before = model(batch)
    noisy_gap = batch.history_continuous_features.clone()
    noisy_gap[~mask] = 99.0
    with torch.no_grad():
        after = model(replace(batch, history_continuous_features=noisy_gap))

    assert torch.allclose(before.t1_logits, after.t1_logits, atol=TOLERANCE)
    assert torch.allclose(before.t2_logit, after.t2_logit, atol=TOLERANCE)


def test_repeated_runs_of_the_exported_model_are_identical(model, exported, spec) -> None:
    batch = deterministic_example_batch(spec, rows=4, history=6, candidates=5)

    first = compare_against_onnx(model, exported, batch, tolerance=TOLERANCE)
    second = compare_against_onnx(model, exported, batch, tolerance=TOLERANCE)

    assert first.max_absolute_difference == second.max_absolute_difference


def test_a_drifted_signature_is_refused(model, exported, spec, monkeypatch) -> None:
    import ppsi.deployment.onnx_export as module

    monkeypatch.setattr(module, "OUTPUT_NAMES", ("t1_logits", "t2_logit", "renamed"))
    batch = deterministic_example_batch(spec, rows=2, history=4, candidates=3)

    with pytest.raises(ValueError, match="drifted"):
        compare_against_onnx(model, exported, batch)


def test_the_wrapper_refuses_the_wrong_number_of_inputs(model, spec) -> None:
    from ppsi.deployment.onnx_export import SessionGRUExportWrapper

    wrapper = SessionGRUExportWrapper(model, batch_spec=spec)

    with pytest.raises(ValueError, match="expected"):
        wrapper(torch.zeros(2, 3, dtype=torch.long))
