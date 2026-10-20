"""Hand-worked, dependency-free checks of the S2-PR-02 numerical primitives."""

import json

import pytest

from ppsi.federated.convergence import analyze_curve, sum_measured_communication


def curve(values, rounds=None):
    rounds = range(1, len(values) + 1) if rounds is None else rounds
    return [{"server_round": r, "value": v} for r, v in zip(rounds, values, strict=True)]


def test_inclusive_boundary_and_confirmation_not_first_crossing():
    out = analyze_curve(curve([0.95, 0.95, 0.95, 1.0]), r1_value=1.0)
    assert out["self_convergence"]["first_qualifying_round"] == 1
    assert out["self_convergence"]["confirmation_round"] == 3
    assert out["r1_target"]["crossing_round"] == 1


def test_spike_is_not_sustained_convergence():
    out = analyze_curve(curve([0.2, 0.97, 0.5, 1.0]), r1_value=1.0)
    assert out["self_convergence"]["status"] == "NOT_REACHED"
    assert out["r1_target"]["crossing_round"] == 2


def test_three_qualifying_values():
    out = analyze_curve(curve([0.2, 0.94, 0.96, 0.95, 1.0]), r1_value=1.0)
    assert out["self_convergence"]["first_qualifying_round"] == 3
    assert out["self_convergence"]["confirmation_round"] == 5
    assert out["r1_target"]["crossing_round"] == 2


@pytest.mark.parametrize("gap", [None, float("nan"), float("inf"), float("-inf")])
def test_missing_or_nonfinite_observation_breaks_streak(gap):
    out = analyze_curve(curve([0.95, gap, 0.97, 0.96, 1.0]))
    assert out["self_convergence"]["first_qualifying_round"] == 3
    assert out["self_convergence"]["confirmation_round"] == 5
    assert out["missing_observation_rounds"] == [2]
    json.dumps(out, allow_nan=False)


def test_missing_scheduled_round_breaks_streak():
    out = analyze_curve(curve([0.95, 0.96, 0.97, 1.0], [1, 3, 4, 5]))
    assert out["self_convergence"]["first_qualifying_round"] == 3
    assert out["self_convergence"]["confirmation_round"] == 5
    assert out["cadence_gap_pairs"] == [[1, 3]]


def test_nonunit_declared_cadence():
    out = analyze_curve(curve([0.95, 0.95, 1.0], [0, 5, 10]), cadence_rounds=5)
    assert out["self_convergence"]["confirmation_round"] == 10


@pytest.mark.parametrize(
    "values,status",
    [
        ([], "NO_VALID_OBSERVATIONS"),
        ([None], "NO_VALID_OBSERVATIONS"),
        ([0, 0, 0], "NO_POSITIVE_REFERENCE"),
        ([0.95, 1.0], "INSUFFICIENT_VALID_OBSERVATIONS"),
    ],
)
def test_degenerate_curves(values, status):
    assert analyze_curve(curve(values))["self_convergence"]["status"] == status


def test_target_pending_zero_and_never_crossed():
    points = curve([0.1, 0.2, 0.3])
    assert analyze_curve(points)["r1_target"]["status"] == "PENDING_R1"
    assert analyze_curve(points, r1_value=0)["r1_target"]["status"] == "UNDEFINED_ZERO_R1"
    assert analyze_curve(points, r1_value=1)["r1_target"]["status"] == "NOT_REACHED"


@pytest.mark.parametrize(
    "points",
    [
        curve([0.2, 0.3], [2, 1]),
        curve([0.2, 0.3], [1, 1]),
        curve([0.2], [-1]),
        curve([0.2], [True]),
        curve([True]),
        curve([-0.1]),
        curve([1.1]),
        curve(["0.5"]),
    ],
)
def test_invalid_curve_input(points):
    with pytest.raises((ValueError, TypeError)):
        analyze_curve(points)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cadence_rounds": 0},
        {"consecutive": False},
        {"self_fraction": 0},
        {"target_fraction": 1.2},
        {"r1_value": -1},
        {"r1_value": float("nan")},
    ],
)
def test_invalid_policy_or_reference(kwargs):
    with pytest.raises((ValueError, TypeError)):
        analyze_curve(curve([0.5]), **kwargs)


RUN = "fixture-run"
SHA = "a" * 64


def communication(round_number, upload=10, download=100):
    return {
        "server_round": round_number,
        "upload_bytes": upload,
        "download_bytes": download,
        "run_id": RUN,
        "config_sha256": SHA,
        "measurement_basis": "MEASURED_APPLICATION_MODEL_PAYLOAD",
    }


def total(records, through_round=3):
    return sum_measured_communication(
        records, through_round=through_round, expected_run_id=RUN, expected_config_sha256=SHA
    )


def test_measured_bytes_include_confirmation_round():
    out = total(
        [communication(1, 10), communication(2, 20), communication(3, 30), communication(4, 99)]
    )
    assert out["status"] == "AVAILABLE"
    assert out["upload_bytes"] == 60
    assert out["download_bytes"] == 300
    assert out["total_bytes"] == 360


def test_missing_bytes_are_not_zero():
    assert total(None)["status"] == "PENDING_COMMUNICATION"
    out = total([communication(1), communication(3)])
    assert out["status"] == "INCOMPLETE_COMMUNICATION"
    assert out["missing_rounds"] == [2]
    assert out["total_bytes"] is None
    assert total(None, through_round=None)["status"] == "PENDING_CROSSING"


@pytest.mark.parametrize(
    "field,value",
    [
        ("upload_bytes", -1),
        ("download_bytes", True),
        ("run_id", "wrong"),
        ("config_sha256", "b" * 64),
        ("measurement_basis", "ESTIMATED_PARAMETER_BYTES"),
    ],
)
def test_communication_rejects_invalid_or_mismatched_records(field, value):
    rec = communication(1)
    rec[field] = value
    with pytest.raises((TypeError, ValueError)):
        total([rec], through_round=1)


def test_duplicate_aggregate_round_fails():
    with pytest.raises(ValueError):
        total([communication(1), communication(1)], through_round=1)


def test_deterministic_strict_json():
    first = analyze_curve(curve([0.2, 0.94, 0.96, 0.95, 1.0]), r1_value=1.0)
    second = analyze_curve(curve([0.2, 0.94, 0.96, 0.95, 1.0]), r1_value=1.0)
    assert json.dumps(first, sort_keys=True, allow_nan=False) == json.dumps(
        second, sort_keys=True, allow_nan=False
    )


# ---------------------------------------------------------------------------
# Project-facing identity guard integration (S2-PR-02)
#
# These exercise the real project validators, not the pure arithmetic above.
# ---------------------------------------------------------------------------

from ppsi.federated.convergence import ConvergenceIdentityError, analyze_run_curve
from ppsi.training.result import build_experiment_result, make_metric_record
from scripts.experiments.compatibility import build_compatibility_tuple

REF_SHA = "a" * 64
GIT_SHA = "f" * 40
METRIC_ID = "t1.next_distinct.mrr_at_20.macro"


def _ref(sha: str = REF_SHA) -> dict:
    return {
        "schema": "artifact_ref_v1",
        "version": "1",
        "logical_id": "toy",
        "artifact_schema": "toy_v1",
        "artifact_version": "1",
        "uri": "docs/evidence/toy.json",
        "sha256": sha,
    }


def _policy(**overrides) -> dict:
    base = {
        "schema": "convergence_policy_v1",
        "version": "1",
        "metric_id": METRIC_ID,
        "task": "T1",
        "cohort": "C1",
        "split": "VALIDATION",
        "direction": "MAXIMIZE",
        "unit": "FRACTION",
        "self_fraction": 0.95,
        "self_consecutive_evaluations": 3,
        "target_fraction": 0.9,
        "evaluation_cadence_rounds": 1,
        "cadence_must_match_input": True,
    }
    base.update(overrides)
    return base


def _r1_config(**overrides) -> dict:
    base = {
        "schema": "experiment_config_v1",
        "version": "1",
        "config_id": "toy_r1_cfg",
        "regime": "R1",
        "tasks": ["T1"],
        "training_cohort": "C1",
        "evaluation_cohorts": ["C1"],
        "seed": 13,
        "source_dataset_ref": _ref(),
        "canonical_data_contract_ref": _ref(),
        "cohort_manifest_ref": _ref(),
        "split_manifest_ref": _ref(),
        "task_examples_manifest_ref": _ref(),
        "evaluation_manifest_ref": _ref(),
        "representation_ref": _ref(),
        "model_config_ref": _ref(),
        "objective_config_ref": _ref(),
        "shared_trainer_core_ref": _ref(),
        "evaluation_protocol_ref": _ref(),
        "evaluator_ref": _ref(),
        "environment_lock_ref": _ref(),
        "initialization": {
            "kind": "COMMON_INITIALIZATION",
            "common_initialization_ref": _ref(),
        },
        "regime_config": {"evaluation_split": "VALIDATION"},
    }
    base.update(overrides)
    return base


def _r1_result(config: dict, *, value: float = 0.5, state: str = "SUCCEEDED", **overrides) -> dict:
    metric = make_metric_record(
        metric_id=METRIC_ID,
        task="T1",
        cohort="C1",
        value=value,
        direction="MAXIMIZE",
        unit="FRACTION",
        support=100,
    )
    kwargs = {
        "experiment_config": config,
        "config_ref": _ref(),
        "git_sha": GIT_SHA,
        "state": state,
        "attempt": 1,
        "started_at_utc": "2026-01-01T00:00:00Z",
        "ended_at_utc": "2026-01-01T01:00:00Z",
        "metrics": [metric],
        "artifacts": [],
    }
    kwargs.update(overrides)
    return build_experiment_result(**kwargs)


def _identity(config: dict, **overrides) -> dict:
    base = {
        "metric_id": METRIC_ID,
        "task": "T1",
        "cohort": "C1",
        "evaluation_split": "VALIDATION",
        "direction": "MAXIMIZE",
        "unit": "FRACTION",
        "cadence_rounds": 1,
        "run_id": "toy-run",
        "config_sha256": REF_SHA,
        "compatibility": build_compatibility_tuple(config, GIT_SHA),
    }
    base.update(overrides)
    return base


RISING = [0.20, 0.40, 0.96, 0.97, 0.98]


def _write_r1_bundle(tmp_path, *, config_overrides=None, value=0.5, state="SUCCEEDED", **result_kw):
    """Write a genuine config/result pair on disk and return verifiable references."""
    from ppsi.training.identity import file_sha256

    config = _r1_config(**(config_overrides or {}))
    config_path = tmp_path / "r1_config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    config_ref = {
        "schema": "artifact_ref_v1",
        "version": "1",
        "logical_id": "r1_resolved_config",
        "artifact_schema": "experiment_config_v1",
        "artifact_version": "1",
        "uri": config_path.name,
        "sha256": file_sha256(config_path),
    }
    result = _r1_result(config, value=value, state=state, config_ref=config_ref, **result_kw)
    result_path = tmp_path / "r1_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    result_ref = {
        "schema": "artifact_ref_v1",
        "version": "1",
        "logical_id": "r1_experiment_result",
        "artifact_schema": "experiment_result_v1",
        "artifact_version": "1",
        "uri": result_path.name,
        "sha256": file_sha256(result_path),
    }
    return {
        "config": config,
        "config_path": config_path,
        "result": result,
        "result_path": result_path,
        "result_ref": result_ref,
    }


def _write_communication(tmp_path, records, name="communication.json"):
    from ppsi.training.identity import file_sha256

    path = tmp_path / name
    path.write_text(json.dumps(records), encoding="utf-8")
    return {"uri": path.name, "sha256": file_sha256(path)}


def _byte_rows(count=5):
    return [
        {
            "server_round": r,
            "upload_bytes": 10 * r,
            "download_bytes": 100,
            "run_id": "toy-run",
            "config_sha256": REF_SHA,
            "measurement_basis": "MEASURED_APPLICATION_MODEL_PAYLOAD",
        }
        for r in range(1, count + 1)
    ]


def test_wrapper_without_r1_still_analyses_self_convergence():
    config = _r1_config()
    out = analyze_run_curve(
        curve_points=curve(RISING), policy=_policy(), curve_identity=_identity(config)
    )
    assert out["real_r1_status"] == "PENDING_R1"
    assert out["actual_communication_status"] == "PENDING_50_INSTRUMENTATION"
    assert out["curve_analysis"]["self_convergence"]["status"] == "REACHED"
    assert out["curve_analysis"]["r1_target"]["status"] == "PENDING_R1"
    assert out["communication"]["self_convergence"]["status"] == "PENDING_COMMUNICATION"
    assert out["communication"]["r1_target"]["status"] == "PENDING_CROSSING"
    assert out["identity_guard"]["r1_identity_verified"] is False
    assert out["source_refs"]["r1_result_ref"] is None
    assert out["policy_identity"]["metric_id"] == METRIC_ID


def test_wrapper_accepts_an_r1_result_bound_to_its_verified_files(tmp_path):
    bundle = _write_r1_bundle(tmp_path, value=0.5)
    out = analyze_run_curve(
        curve_points=curve(RISING),
        policy=_policy(),
        curve_identity=_identity(bundle["config"]),
        r1_result=bundle["result"],
        r1_result_ref=bundle["result_ref"],
        repo_root=tmp_path,
    )
    assert out["real_r1_status"] == "AVAILABLE"
    assert out["identity_guard"]["r1_identity_verified"] is True
    assert out["source_refs"]["r1_result_ref"]["uri"] == "r1_result.json"
    # 0.90 * 0.5 = 0.45; rounds 1-2 are below it, so round 3 is the first crossing.
    assert out["curve_analysis"]["r1_target"]["crossing_round"] == 3


def test_an_unreferenced_r1_result_is_never_source_verified(tmp_path):
    bundle = _write_r1_bundle(tmp_path)
    with pytest.raises(ConvergenceIdentityError, match="artifact reference"):
        analyze_run_curve(
            curve_points=curve(RISING),
            policy=_policy(),
            curve_identity=_identity(bundle["config"]),
            r1_result=bundle["result"],
            repo_root=tmp_path,
        )


def test_tampered_in_memory_r1_value_fails_against_the_matching_file(tmp_path):
    """The file hash still passes; the in-memory payload no longer matches it."""
    bundle = _write_r1_bundle(tmp_path, value=0.5)
    tampered = json.loads(json.dumps(bundle["result"]))
    tampered["metrics"][0]["value"] = 0.99
    with pytest.raises(ConvergenceIdentityError, match="differs from the verified file"):
        analyze_run_curve(
            curve_points=curve(RISING),
            policy=_policy(),
            curve_identity=_identity(bundle["config"]),
            r1_result=tampered,
            r1_result_ref=bundle["result_ref"],
            repo_root=tmp_path,
        )


def test_changed_in_memory_config_fails_against_the_referenced_config(tmp_path):
    bundle = _write_r1_bundle(tmp_path)
    swapped = json.loads(json.dumps(bundle["config"]))
    swapped["regime_config"] = {"evaluation_split": "TEST"}
    with pytest.raises(ConvergenceIdentityError, match="differs from the resolved config"):
        analyze_run_curve(
            curve_points=curve(RISING),
            policy=_policy(),
            curve_identity=_identity(bundle["config"]),
            r1_result=bundle["result"],
            r1_result_ref=bundle["result_ref"],
            r1_config=swapped,
            repo_root=tmp_path,
        )


def test_a_config_file_declaring_the_wrong_split_is_rejected(tmp_path):
    bundle = _write_r1_bundle(
        tmp_path, config_overrides={"regime_config": {"evaluation_split": "TEST"}}
    )
    with pytest.raises(ConvergenceIdentityError, match="evaluation split"):
        analyze_run_curve(
            curve_points=curve(RISING),
            policy=_policy(),
            curve_identity=_identity(bundle["config"]),
            r1_result=bundle["result"],
            r1_result_ref=bundle["result_ref"],
            repo_root=tmp_path,
        )


def test_a_config_file_with_no_split_declaration_is_rejected(tmp_path):
    bundle = _write_r1_bundle(tmp_path, config_overrides={"regime_config": {}})
    with pytest.raises(ConvergenceIdentityError, match="evaluation split"):
        analyze_run_curve(
            curve_points=curve(RISING),
            policy=_policy(),
            curve_identity=_identity(bundle["config"]),
            r1_result=bundle["result"],
            r1_result_ref=bundle["result_ref"],
            repo_root=tmp_path,
        )


@pytest.mark.parametrize(
    "config_overrides", [{"seed": 42}, {"tasks": ["T2"]}, {"evaluation_cohorts": ["C2"]}]
)
def test_changed_identity_is_rejected(tmp_path, config_overrides):
    curve_config = _r1_config()
    bundle = _write_r1_bundle(tmp_path, config_overrides=config_overrides)
    with pytest.raises(ConvergenceIdentityError):
        analyze_run_curve(
            curve_points=curve(RISING),
            policy=_policy(),
            curve_identity=_identity(curve_config),
            r1_result=bundle["result"],
            r1_result_ref=bundle["result_ref"],
            repo_root=tmp_path,
        )


def test_failed_or_wrong_regime_r1_is_rejected(tmp_path):
    failed = _write_r1_bundle(
        tmp_path,
        state="FAILED",
        failure={"reason_code": "CRASH", "message": "toy failure", "retryable": False},
    )
    with pytest.raises(ConvergenceIdentityError, match="R1 state"):
        analyze_run_curve(
            curve_points=curve(RISING),
            policy=_policy(),
            curve_identity=_identity(failed["config"]),
            r1_result=failed["result"],
            r1_result_ref=failed["result_ref"],
            repo_root=tmp_path,
        )

    other = tmp_path / "r2a"
    other.mkdir()
    r2a = _write_r1_bundle(other, config_overrides={"regime": "R2A", "config_id": "toy_r2a_cfg"})
    with pytest.raises(ConvergenceIdentityError, match="R1 regime"):
        analyze_run_curve(
            curve_points=curve(RISING),
            policy=_policy(),
            curve_identity=_identity(r2a["config"]),
            r1_result=r2a["result"],
            r1_result_ref=r2a["result_ref"],
            repo_root=other,
        )


def test_wrong_metric_identity_is_rejected(tmp_path):
    other_metric = make_metric_record(
        metric_id="t1.overall.mrr_at_20.micro",
        task="T1",
        cohort="C1",
        value=0.5,
        direction="MAXIMIZE",
        unit="FRACTION",
        support=100,
    )
    bundle = _write_r1_bundle(tmp_path, metrics=[other_metric])
    with pytest.raises(ConvergenceIdentityError, match="no metric"):
        analyze_run_curve(
            curve_points=curve(RISING),
            policy=_policy(),
            curve_identity=_identity(bundle["config"]),
            r1_result=bundle["result"],
            r1_result_ref=bundle["result_ref"],
            repo_root=tmp_path,
        )


def test_ambiguous_duplicate_metric_is_rejected(tmp_path):
    metric = make_metric_record(
        metric_id=METRIC_ID,
        task="T1",
        cohort="C1",
        value=0.5,
        direction="MAXIMIZE",
        unit="FRACTION",
        support=100,
    )
    bundle = _write_r1_bundle(tmp_path, metrics=[metric, dict(metric)])
    with pytest.raises(ConvergenceIdentityError, match="ambiguous"):
        analyze_run_curve(
            curve_points=curve(RISING),
            policy=_policy(),
            curve_identity=_identity(bundle["config"]),
            r1_result=bundle["result"],
            r1_result_ref=bundle["result_ref"],
            repo_root=tmp_path,
        )


def test_lower_is_better_policy_is_refused():
    config = _r1_config()
    policy = _policy(direction="MINIMIZE", metric_id="t1.validation.logloss", unit="UNITLESS")
    identity = _identity(
        config, direction="MINIMIZE", metric_id="t1.validation.logloss", unit="UNITLESS"
    )
    with pytest.raises(ConvergenceIdentityError, match="unsupported"):
        analyze_run_curve(curve_points=curve(RISING), policy=policy, curve_identity=identity)


def test_cadence_must_match_declared_policy():
    config = _r1_config()
    with pytest.raises(ConvergenceIdentityError, match="cadence"):
        analyze_run_curve(
            curve_points=curve(RISING),
            policy=_policy(),
            curve_identity=_identity(config, cadence_rounds=2),
        )


def test_measured_bytes_need_a_verifiable_source(tmp_path):
    config = _r1_config()
    records = _byte_rows(3)
    with pytest.raises(ConvergenceIdentityError, match="communication_source_ref"):
        analyze_run_curve(
            curve_points=curve(RISING),
            policy=_policy(),
            curve_identity=_identity(config),
            communication_records=records,
        )
    source = _write_communication(tmp_path, records)
    with pytest.raises(ConvergenceIdentityError, match="SHA-256 mismatch"):
        analyze_run_curve(
            curve_points=curve(RISING),
            policy=_policy(),
            curve_identity=_identity(config),
            communication_records=records,
            communication_source_ref={"uri": source["uri"], "sha256": "c" * 64},
            repo_root=tmp_path,
        )


def test_altered_byte_rows_fail_against_an_unchanged_verified_file(tmp_path):
    """The file is untouched and its hash passes; the caller's rows were edited."""
    config = _r1_config()
    records = _byte_rows(5)
    source = _write_communication(tmp_path, records)
    altered = json.loads(json.dumps(records))
    altered[0]["upload_bytes"] = 999_999
    with pytest.raises(ConvergenceIdentityError, match="differ from the verified communication"):
        analyze_run_curve(
            curve_points=curve(RISING),
            policy=_policy(),
            curve_identity=_identity(config),
            communication_records=altered,
            communication_source_ref=source,
            repo_root=tmp_path,
        )


def test_verified_measured_bytes_are_summed_to_the_confirmation_round(tmp_path):
    config = _r1_config()
    records = _byte_rows(5)
    source = _write_communication(tmp_path, records)
    out = analyze_run_curve(
        curve_points=curve(RISING),
        policy=_policy(),
        curve_identity=_identity(config),
        communication_records=records,
        communication_source_ref=source,
        repo_root=tmp_path,
    )
    assert out["identity_guard"]["communication_source_verified"] is True
    assert out["actual_communication_status"] == "AVAILABLE"
    # Self convergence confirms at round 5; upload 10+20+30+40+50 = 150.
    assert out["curve_analysis"]["self_convergence"]["confirmation_round"] == 5
    assert out["communication"]["self_convergence"]["upload_bytes"] == 150
    assert out["communication"]["self_convergence"]["download_bytes"] == 500


def test_a_verified_source_never_turns_a_short_interval_into_available(tmp_path):
    """A passing file hash must not upgrade an interval the file cannot cover."""
    config = _r1_config()
    records = _byte_rows(2)  # the curve confirms at round 5
    source = _write_communication(tmp_path, records)
    out = analyze_run_curve(
        curve_points=curve(RISING),
        policy=_policy(),
        curve_identity=_identity(config),
        communication_records=records,
        communication_source_ref=source,
        repo_root=tmp_path,
    )
    assert out["identity_guard"]["communication_source_verified"] is True
    self_comm = out["communication"]["self_convergence"]
    assert self_comm["status"] == "INCOMPLETE_COMMUNICATION"
    assert self_comm["missing_rounds"] == [3, 4, 5]
    assert self_comm["total_bytes"] is None


def test_fixture_json_cases_match_recorded_expectations():
    from pathlib import Path

    fixture = json.loads(
        Path("fixtures/federated/convergence_cases.v1.json").read_text(encoding="utf-8")
    )
    assert fixture["artifact_kind"] == "FIXTURE_PROOF"
    for case in fixture["cases"]:
        out = analyze_curve(
            [
                {"server_round": r, "value": v}
                for r, v in zip(case["rounds"], case["values"], strict=True)
            ],
            r1_value=case.get("r1_value"),
        )
        assert out["self_convergence"]["status"] == case["expected_self_status"], case["case_id"]
        assert out["self_convergence"]["first_qualifying_round"] == case["expected_first_round"]
        assert (
            out["self_convergence"]["confirmation_round"] == case["expected_confirmation_round"]
        ), case["case_id"]
        assert out["r1_target"]["status"] == case["expected_target_status"], case["case_id"]
        assert out["r1_target"]["crossing_round"] == case["expected_target_round"], case["case_id"]
