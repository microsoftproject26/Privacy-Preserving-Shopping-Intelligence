from __future__ import annotations

import json
from pathlib import Path

import pytest

from ppsi.deployment.final_export import (
    CANDIDATE_PATH,
    MODEL_CONFIG_PATH,
    DeploymentContractError,
    build_deployment_model,
    candidate_identity,
    load_deployment_candidate,
)
from ppsi.deployment.onnx_export import (
    OUTPUT_NAMES,
    check_graph,
    compare_against_onnx,
    deterministic_example_batch,
    export_session_gru,
    history_channels_of,
    input_names_for,
)
from ppsi.models.batch_spec import phase1_batch_spec_v1
from ppsi.models.session_gru import parameter_count

EVIDENCE = (
    Path(__file__).resolve().parents[2] / "docs" / "evidence" / "s2-se-08" / "final_export.v1.json"
)
TOLERANCE = 1e-4


@pytest.fixture(scope="module")
def contract():
    return load_deployment_candidate()


@pytest.fixture(scope="module")
def spec():
    return phase1_batch_spec_v1()


@pytest.fixture(scope="module")
def model(contract, spec):
    candidate, config = contract
    return build_deployment_model(candidate, config, batch_spec=spec)


@pytest.fixture(scope="module")
def exported(tmp_path_factory, model, spec):
    batch = deterministic_example_batch(spec, rows=4, history=20, candidates=100)
    return export_session_gru(
        model, tmp_path_factory.mktemp("final") / "final.onnx", example_batch=batch, batch_spec=spec
    )


def test_the_architecture_is_read_from_the_frozen_files_not_assumed(contract) -> None:
    candidate, config = contract

    assert candidate.checkpoint_id == "s2_ds_08_joint_lambda1.0_seed13"
    assert candidate.model_config_id == "s2_ds_08_shared_gru_t1_t2_v1"
    # The final model consumes two history channels, not the batch spec's five. Assuming
    # five would export a graph with three inputs nothing reads.
    assert config.channels == ("category_id", "event_type_id")
    assert config.core == "gru"
    assert config.hidden == 128


def test_the_built_model_matches_the_declared_parameter_count(model, contract) -> None:
    candidate, _ = contract

    assert parameter_count(model) == candidate.declared_parameter_count == 2_379_263


def test_a_config_that_drifts_from_the_contract_stops_the_export(contract, spec) -> None:
    from dataclasses import replace

    candidate, config = contract

    with pytest.raises(DeploymentContractError, match="diverged"):
        build_deployment_model(candidate, replace(config, hidden=64), batch_spec=spec)


def test_the_wrong_checkpoint_is_refused_before_it_is_loaded(contract, tmp_path, spec) -> None:
    candidate, config = contract
    impostor = tmp_path / "not-the-candidate.pt"
    impostor.write_bytes(b"not a checkpoint")

    with pytest.raises(DeploymentContractError, match="not the approved candidate"):
        build_deployment_model(candidate, config, weights=impostor, batch_spec=spec)


def test_a_candidate_naming_a_different_model_config_is_refused(tmp_path) -> None:
    candidate = json.loads(CANDIDATE_PATH.read_text(encoding="utf-8"))
    candidate["model_config_ref"] = "some_other_model"
    forged = tmp_path / "deployment_candidate.v1.json"
    forged.write_text(json.dumps(candidate), encoding="utf-8")

    with pytest.raises(DeploymentContractError, match="model config"):
        load_deployment_candidate(MODEL_CONFIG_PATH, forged)


def test_the_exported_signature_follows_the_model_not_the_batch(exported, model) -> None:
    info = check_graph(exported)

    assert tuple(info["graph_inputs"]) == input_names_for(history_channels_of(model))
    assert tuple(info["graph_outputs"]) == OUTPUT_NAMES
    assert len(info["graph_inputs"]) == 12


def test_all_three_heads_agree_with_pytorch(model, exported, spec) -> None:
    batch = deterministic_example_batch(spec, rows=4, history=20, candidates=100)

    report = compare_against_onnx(model, exported, batch, tolerance=TOLERANCE)

    assert set(report.max_absolute_difference) == {"t1_logits", "t2_logit", "t3_scores"}
    assert report.within_tolerance


@pytest.mark.parametrize(
    "rows,history,candidates", [(1, 20, 100), (16, 20, 100), (4, 1, 100), (4, 20, 1), (3, 7, 55)]
)
def test_parity_holds_at_every_shape_the_contract_allows(
    model, exported, spec, rows, history, candidates
) -> None:
    batch = deterministic_example_batch(
        spec, rows=rows, history=history, candidates=candidates, seed=13
    )

    assert compare_against_onnx(model, exported, batch, tolerance=TOLERANCE).within_tolerance


def test_the_evidence_says_whether_trained_weights_were_used(contract) -> None:
    candidate, _ = contract

    without = candidate_identity(candidate, weights_loaded=False)
    with_weights = candidate_identity(candidate, weights_loaded=True)

    assert "architecture only" in without["status"]
    assert with_weights["status"] == "deployment artifact"
    assert without["weights_sha256_required"] == candidate.weights_sha256


def test_the_committed_evidence_names_the_candidate_and_clears_tolerance() -> None:
    report = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    assert report["candidate"]["deployment_candidate_checkpoint_id"] == (
        "s2_ds_08_joint_lambda1.0_seed13"
    )
    assert report["parity"]["within_tolerance"] is True
    assert set(report["heads_verified"]) == {"t1_logits", "t2_logit", "t3_scores"}
    assert len(report["parity_shapes"]) >= 8


def test_the_recommendation_follows_from_the_measurements() -> None:
    report = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    quantization = report["quantization"]

    expected = (
        "INT8"
        if quantization["meets_size_rule"] and quantization["meets_speed_rule"]
        else "FP32"
    )
    assert report["recommended_artifact"] == expected
    assert "quality delta" in report["recommendation_basis"]
