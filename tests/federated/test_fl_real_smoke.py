"""Tests for scripts.federated.fl_real_smoke (S1-PR-07).

These tests cover all logic that can be tested without private data:
- Config validation
- Artifact ref construction
- Public trace sanitization
- Oracle cross-check logic
- Model construction
- Server-side evaluation with toy data

Never skip/xfail to hide defects.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import torch
from flwr.app import ConfigRecord, Message, RecordDict

from ppsi.federated.task_examples import default_batch_spec, make_t1_smoke_batches
from ppsi.training.identity import file_sha256
from scripts.experiments.results import validate_result_for_reporting
from scripts.experiments.schemas import (
    validate_experiment_config,
    validate_experiment_result,
)
from scripts.federated.fl_real_smoke import (
    SmokeValidationError,
    assert_no_client_leakage,
    build_smoke_model,
    get_digest,
    load_smoke_config,
    make_artifact_ref,
    sanitize_trace_for_public,
    server_evaluate,
    server_round_to_sampler_round,
    sort_client_replies_by_logical_id,
    validate_smoke_config,
    weighted_average_state_dicts,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_minimal_config() -> dict[str, Any]:
    return {
        "schema": "fl_real_smoke_v1",
        "version": "1",
        "seed": 13,
        "num_rounds": 3,
        "clients_per_round": 4,
        "repeat_runs": 2,
        "max_train_examples_per_client": 32,
        "validation_example_limit": 256,
        "learning_rate": 0.02,
        "momentum": 0.0,
        "local_epochs": 1,
        "sampler_version": "client_sampler_v1",
        "aggregation_atol": 1e-6,
        "paths": {
            "cohort_manifest": "x",
            "base_client_manifest": "x",
            "t1_train": "x",
            "t1_validation": "x",
            "vocabulary": "x",
            "derived_client_manifest": "x",
            "sampling_trace": "x",
            "train_smoke_slice": "x",
            "validation_smoke_slice": "x",
            "input_evidence": "x",
            "initialization_evidence": "x",
            "experiment_config_evidence": "x",
            "summary": "x",
            "results_directory": "x",
        },
    }


def _make_minimal_config_file(tmp_path: Path) -> Path:
    cfg = _make_minimal_config()
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_validate_smoke_config_valid() -> None:
    validate_smoke_config(_make_minimal_config())  # no exception


def test_validate_smoke_config_wrong_schema() -> None:
    cfg = _make_minimal_config()
    cfg["schema"] = "wrong"
    with pytest.raises(SmokeValidationError, match="Invalid config schema"):
        validate_smoke_config(cfg)


def test_validate_smoke_config_wrong_seed() -> None:
    cfg = _make_minimal_config()
    cfg["seed"] = 42
    with pytest.raises(SmokeValidationError, match="seed must be 13"):
        validate_smoke_config(cfg)


def test_validate_smoke_config_wrong_num_rounds() -> None:
    cfg = _make_minimal_config()
    cfg["num_rounds"] = 2
    with pytest.raises(SmokeValidationError, match="num_rounds must be 3"):
        validate_smoke_config(cfg)


def test_validate_smoke_config_wrong_clients_per_round() -> None:
    cfg = _make_minimal_config()
    cfg["clients_per_round"] = 2
    with pytest.raises(SmokeValidationError, match="clients_per_round must be 4"):
        validate_smoke_config(cfg)


def test_validate_smoke_config_too_few_repeat_runs() -> None:
    cfg = _make_minimal_config()
    cfg["repeat_runs"] = 1
    with pytest.raises(SmokeValidationError, match="repeat_runs must be >= 2"):
        validate_smoke_config(cfg)


def test_load_smoke_config_roundtrip(tmp_path: Path) -> None:
    cfg_path = _make_minimal_config_file(tmp_path)
    loaded = load_smoke_config(cfg_path)
    assert loaded["seed"] == 13
    assert loaded["num_rounds"] == 3


# ---------------------------------------------------------------------------
# Artifact ref construction
# ---------------------------------------------------------------------------


def test_make_artifact_ref_fields() -> None:
    ref = make_artifact_ref("test-logical-id", "test_schema_v1", "some/uri/path.json", "a" * 64)
    assert ref["schema"] == "artifact_ref_v1"
    assert ref["version"] == "1"
    assert ref["logical_id"] == "test-logical-id"
    assert ref["artifact_schema"] == "test_schema_v1"
    assert ref["artifact_version"] == "1"
    assert ref["uri"] == "some/uri/path.json"
    assert ref["sha256"] == "a" * 64


def test_make_artifact_ref_no_absolute_path() -> None:
    # Absolute path in uri must not be present (will fail schema validation later)
    ref = make_artifact_ref("x", "y", "relative/path.json", "b" * 64)
    assert not ref["uri"].startswith("/")
    assert not ref["uri"].startswith("C:")


# ---------------------------------------------------------------------------
# get_digest
# ---------------------------------------------------------------------------


def test_get_digest_deterministic() -> None:
    sd = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0])}
    d1 = get_digest(sd)
    d2 = get_digest(sd)
    assert d1 == d2
    assert len(d1) == 64


def test_get_digest_detects_change() -> None:
    sd1 = {"a": torch.tensor([1.0])}
    sd2 = {"a": torch.tensor([2.0])}
    assert get_digest(sd1) != get_digest(sd2)


# ---------------------------------------------------------------------------
# weighted_average_state_dicts (oracle)
# ---------------------------------------------------------------------------


def test_oracle_averaging_two_clients() -> None:
    sd1 = {"w": torch.tensor([1.0, 0.0])}
    sd2 = {"w": torch.tensor([3.0, 4.0])}
    result = weighted_average_state_dicts([(sd1, 1), (sd2, 3)])
    # Expected: (1*1 + 3*3) / 4 = 10/4 = 2.5, (1*0 + 3*4) / 4 = 3.0
    expected = torch.tensor([2.5, 3.0])
    assert torch.allclose(result["w"], expected, atol=1e-6)


def test_oracle_averaging_equal_weights() -> None:
    sd1 = {"a": torch.tensor([0.0])}
    sd2 = {"a": torch.tensor([2.0])}
    result = weighted_average_state_dicts([(sd1, 1), (sd2, 1)])
    assert torch.allclose(result["a"], torch.tensor([1.0]), atol=1e-6)


def test_oracle_fails_empty() -> None:
    with pytest.raises(SmokeValidationError, match="No client updates"):
        weighted_average_state_dicts([])


def test_oracle_fails_non_positive_examples() -> None:
    sd = {"a": torch.tensor([1.0])}
    with pytest.raises(SmokeValidationError, match="num_examples must be positive"):
        weighted_average_state_dicts([(sd, 0)])


def test_oracle_fails_key_mismatch() -> None:
    sd1 = {"a": torch.tensor([1.0])}
    sd2 = {"b": torch.tensor([1.0])}
    with pytest.raises(SmokeValidationError, match="Keys do not match"):
        weighted_average_state_dicts([(sd1, 1), (sd2, 1)])


# ---------------------------------------------------------------------------
# Build smoke model
# ---------------------------------------------------------------------------


def test_build_smoke_model_correct_shape() -> None:
    spec = default_batch_spec()
    model = build_smoke_model(10, spec)
    # Check that model has T1 head with 10 output categories
    from ppsi.training.stub_model import Phase1StubModel

    assert isinstance(model, Phase1StubModel)
    assert model.config.category_count == 10


def test_build_smoke_model_logits_shape_588() -> None:
    spec = default_batch_spec()
    model = build_smoke_model(588, spec)
    assert model.config.category_count == 588

    # Forward pass on a batch of size B=3
    df = pl.DataFrame(
        {
            "client_id": ["client-v1-abc"] * 3,
            "session": ["s1"] * 3,
            "decision_order": [0, 1, 2],
            "label_code": [0, 60, 587],
        }
    )
    batches = make_t1_smoke_batches(df, spec=spec, batch_size=3)
    assert len(batches) == 1
    batch = batches[0]
    out = model(batch)
    assert out.t1_logits.shape == (3, 588)


def test_build_smoke_model_different_categories() -> None:
    spec = default_batch_spec()
    m1 = build_smoke_model(5, spec)
    m2 = build_smoke_model(20, spec)
    assert m1.config.category_count == 5
    assert m2.config.category_count == 20


# ---------------------------------------------------------------------------
# Server-side evaluation
# ---------------------------------------------------------------------------


def test_server_evaluate_basic() -> None:
    """server_evaluate runs without error on a tiny toy batch."""
    spec = default_batch_spec()
    model = build_smoke_model(6, spec)

    val_df = pl.DataFrame(
        {
            "client_id": ["c1"] * 4,
            "session": ["s"] * 4,
            "decision_order": [0, 1, 2, 3],
            "label_code": [0, 1, 2, 3],
        }
    )
    metrics = server_evaluate(model, val_df, spec)
    assert "cross_entropy" in metrics
    assert "accuracy_at_1" in metrics
    assert "support" in metrics
    assert metrics["support"] == 4
    assert 0.0 <= metrics["accuracy_at_1"] <= 1.0
    assert metrics["cross_entropy"] > 0.0
    assert torch.isfinite(torch.tensor(metrics["cross_entropy"]))
    assert torch.isfinite(torch.tensor(metrics["accuracy_at_1"]))


def test_server_evaluate_empty_slice_fails() -> None:
    spec = default_batch_spec()
    model = build_smoke_model(6, spec)
    empty_df = pl.DataFrame(
        schema={
            "client_id": pl.String,
            "session": pl.String,
            "decision_order": pl.Int64,
            "label_code": pl.Int64,
        }
    )
    with pytest.raises(SmokeValidationError, match="no T1 rows"):
        server_evaluate(model, empty_df, spec)


def test_server_evaluate_accuracy_at_1_range() -> None:
    spec = default_batch_spec()
    model = build_smoke_model(6, spec)
    val_df = pl.DataFrame(
        {
            "client_id": ["c"] * 10,
            "session": ["s"] * 10,
            "decision_order": list(range(10)),
            "label_code": [i % 6 for i in range(10)],
        }
    )
    m = server_evaluate(model, val_df, spec)
    assert 0.0 <= m["accuracy_at_1"] <= 1.0
    assert m["support"] == 10


# ---------------------------------------------------------------------------
# Sanitize trace for public
# ---------------------------------------------------------------------------


def _make_fake_trace_entry(server_round: int) -> dict:
    return {
        "server_round": server_round,
        "sampler_round_index": server_round - 1,
        "selection_digest": "abc" * 21 + "a",
        "server_input_digest": "def" * 21 + "d",
        "selected_client_ids": ["client-v1-abc", "client-v1-xyz"],  # private
        "selected_client_count": 2,
        "clients": [
            {
                "logical_client_id": "client-v1-abc",
                "num_examples": 10,
                "local_train_loss": 1.0,
                "received_digest": "",
                "updated_digest": "",
            },
            {
                "logical_client_id": "client-v1-xyz",
                "num_examples": 8,
                "local_train_loss": 1.2,
                "received_digest": "",
                "updated_digest": "",
            },
        ],
        "aggregation_oracle_pass": True,
        "max_abs_diff": 1e-8,
        "aggregated_digest": "ghi" * 21 + "g",
        "contributing_examples": 18,
    }


def test_sanitize_trace_removes_client_ids() -> None:
    trace = [_make_fake_trace_entry(1), _make_fake_trace_entry(2)]
    public = sanitize_trace_for_public(trace)
    for entry in public:
        assert "selected_client_ids" not in entry


def test_sanitize_trace_preserves_metadata() -> None:
    trace = [_make_fake_trace_entry(1)]
    public = sanitize_trace_for_public(trace)
    assert len(public) == 1
    entry = public[0]
    assert entry["server_round"] == 1
    assert entry["selected_client_count"] == 2
    assert entry["successful_client_count"] == 2
    assert entry["contributing_examples"] == 18
    assert entry["aggregation_oracle_pass"] is True
    assert entry["max_abs_diff"] == 1e-8


def test_sanitize_trace_computes_mean_loss() -> None:
    trace = [_make_fake_trace_entry(1)]
    public = sanitize_trace_for_public(trace)
    expected_mean = (1.0 + 1.2) / 2
    assert abs(public[0]["mean_local_train_loss"] - expected_mean) < 1e-9


def test_sanitize_trace_empty() -> None:
    assert sanitize_trace_for_public([]) == []


# ---------------------------------------------------------------------------
# RealDataTracingFedAvg critical behavior
# ---------------------------------------------------------------------------


def test_server_round_to_sampler_round_mapping() -> None:
    assert server_round_to_sampler_round(1) == 0
    assert server_round_to_sampler_round(2) == 1
    assert server_round_to_sampler_round(3) == 2
    with pytest.raises(ValueError, match="server_round must be >= 1"):
        server_round_to_sampler_round(0)


def test_sort_client_replies_by_logical_id_deterministic() -> None:
    cids = ["client-v1-zzz", "client-v1-aaa", "client-v1-mmm"]
    replies = []
    for cid in cids:
        rd = RecordDict()
        rd.configs_records["config"] = ConfigRecord({"logical_client_id": cid})
        msg = Message(content=rd, dst_node_id=0, message_type="train")
        replies.append(msg)

    # Permutation 1
    sorted1 = sort_client_replies_by_logical_id(replies)
    # Permutation 2: reversed
    sorted2 = sort_client_replies_by_logical_id(list(reversed(replies)))

    ids1 = [m.content.configs_records["config"]["logical_client_id"] for m in sorted1]
    ids2 = [m.content.configs_records["config"]["logical_client_id"] for m in sorted2]
    assert ids1 == ["client-v1-aaa", "client-v1-mmm", "client-v1-zzz"]
    assert ids1 == ids2


# ---------------------------------------------------------------------------
# Public privacy check
# ---------------------------------------------------------------------------


def test_assert_no_client_leakage_detects_and_passes(tmp_path: Path) -> None:
    # Clean file passes
    clean_p = tmp_path / "clean.json"
    clean_p.write_text(json.dumps({"key": "clean_value", "count": 10}), encoding="utf-8")
    assert_no_client_leakage([clean_p])

    # Leaking file with fake client-v1 ID is rejected
    leak_p = tmp_path / "leak.json"
    fake_cid = "client-v1-" + "a" * 64
    leak_p.write_text(json.dumps({"user": fake_cid}), encoding="utf-8")
    with pytest.raises(SmokeValidationError, match="Privacy leak detected"):
        assert_no_client_leakage([leak_p])

    # Leaking file with selected_client_ids key is rejected
    key_leak_p = tmp_path / "key_leak.json"
    key_leak_p.write_text(json.dumps({"selected_client_ids": ["x"]}), encoding="utf-8")
    with pytest.raises(SmokeValidationError, match="selected_client_ids"):
        assert_no_client_leakage([key_leak_p])


# ---------------------------------------------------------------------------
# ExperimentConfig & ExperimentResult referential integrity
# ---------------------------------------------------------------------------


def test_experiment_config_and_result_validation_and_integrity() -> None:
    cfg_evidence_path = Path("docs/evidence/s1-pr-07/fl_real_smoke_experiment_config.v1.json")
    if cfg_evidence_path.is_file():
        cfg = json.loads(cfg_evidence_path.read_text(encoding="utf-8"))
        validate_experiment_config(cfg)

    res_dir = Path("artifacts/experiment-results")
    result_files = list(res_dir.glob("run-v1__r2a__t1__c1__s13__cfg*.result.json"))
    for rf in result_files:
        res = json.loads(rf.read_text(encoding="utf-8"))
        validate_experiment_result(res)
        validate_result_for_reporting(res, source=rf.name)
        cfg_uri = Path(res["config_ref"]["uri"])
        if cfg_uri.is_file():
            assert file_sha256(cfg_uri) == res["config_ref"]["sha256"]
            expected_prefix = f"run-v1__r2a__t1__c1__s13__cfg{res['config_ref']['sha256'][:12]}"
            assert res["run_id"].startswith(expected_prefix)
