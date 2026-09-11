"""Pure curve/byte arithmetic for S2-PR-02.

These functions do not authenticate a scientific R1 reference. The project-facing
wrapper must validate ExperimentResult identity, metric and VALIDATION lineage
before passing a real R1 value or measured byte records to these primitives.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any


def _integer(name: str, value: Any, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _fraction(name: str, value: Any, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result > 1 or result < 0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    if result == 0 and not allow_zero:
        raise ValueError(f"{name} must be > 0")
    return result


def normalize_curve(points: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Reject unsorted/duplicate rounds; retain missing/nonfinite observations as gaps."""
    normalized: list[dict[str, Any]] = []
    previous = -1
    for point in points:
        round_number = _integer("server_round", point["server_round"])
        if round_number <= previous:
            raise ValueError("curve rounds must be unique and strictly increasing")
        previous = round_number
        value = point["value"]
        if value is None:
            normalized.append({"server_round": round_number, "value": None})
            continue
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("curve values must be real numbers or None")
        value = float(value)
        if not math.isfinite(value):
            value = None  # Missing observation, not an interpolated score.
        elif not 0 <= value <= 1:
            raise ValueError("this v1 curve policy requires scores in [0, 1]")
        normalized.append({"server_round": round_number, "value": value})
    return normalized


def analyze_curve(
    points: Sequence[Mapping[str, Any]],
    *,
    cadence_rounds: int = 1,
    self_fraction: float = 0.95,
    consecutive: int = 3,
    target_fraction: float = 0.90,
    r1_value: float | None = None,
) -> dict[str, Any]:
    """Analyze a completed VALIDATION curve without fitting or interpolation.

    Self convergence is retrospective: 95% of this completed curve's best score,
    sustained for three scheduled observations. Return both the beginning of the
    qualifying streak and its confirmation round. Target crossing is the first
    observed score >= 90% of a separately verified matching R1 value.
    """
    cadence_rounds = _integer("cadence_rounds", cadence_rounds, 1)
    consecutive = _integer("consecutive", consecutive, 1)
    self_fraction = _fraction("self_fraction", self_fraction)
    target_fraction = _fraction("target_fraction", target_fraction)
    normalized = normalize_curve(points)
    values = [p["value"] for p in normalized if p["value"] is not None]
    best = max(values) if values else None
    missing_rounds = [p["server_round"] for p in normalized if p["value"] is None]
    gap_pairs = [
        [a["server_round"], b["server_round"]]
        for a, b in itertools.pairwise(normalized)
        if b["server_round"] - a["server_round"] != cadence_rounds
    ]
    self_result: dict[str, Any] = {
        "status": "NOT_REACHED",
        "threshold": None if best is None else self_fraction * best,
        "best_validation_value": best,
        "first_qualifying_round": None,
        "confirmation_round": None,
        "required_consecutive_evaluations": consecutive,
        "retrospective": True,
    }
    if best is None:
        self_result["status"] = "NO_VALID_OBSERVATIONS"
    elif best == 0:
        self_result["status"] = "NO_POSITIVE_REFERENCE"
    elif len(values) < consecutive:
        self_result["status"] = "INSUFFICIENT_VALID_OBSERVATIONS"
    else:
        streak = 0
        first = None
        previous_round = None
        for p in normalized:
            round_number, value = p["server_round"], p["value"]
            if previous_round is not None and round_number - previous_round != cadence_rounds:
                streak, first = 0, None
            previous_round = round_number
            if value is None or value < self_result["threshold"]:
                streak, first = 0, None
                continue
            if streak == 0:
                first = round_number
            streak += 1
            if streak == consecutive:
                self_result.update(
                    status="REACHED",
                    first_qualifying_round=first,
                    confirmation_round=round_number,
                )
                break

    target_result: dict[str, Any] = {
        "status": "PENDING_R1",
        "threshold": None,
        "crossing_round": None,
        "required_consecutive_evaluations": 1,
    }
    if r1_value is not None:
        r1_value = _fraction("r1_value", r1_value, allow_zero=True)
        if r1_value == 0:
            target_result["status"] = "UNDEFINED_ZERO_R1"
        else:
            threshold = target_fraction * r1_value
            target_result.update(status="NOT_REACHED", threshold=threshold)
            for p in normalized:
                if p["value"] is not None and p["value"] >= threshold:
                    target_result.update(status="REACHED", crossing_round=p["server_round"])
                    break
    return {
        "schema": "convergence_curve_analysis_v1",
        "version": "1",
        "cadence_rounds": cadence_rounds,
        "observation_count": len(normalized),
        "valid_observation_count": len(values),
        "missing_observation_rounds": missing_rounds,
        "cadence_gap_pairs": gap_pairs,
        "self_convergence": self_result,
        "r1_target": target_result,
    }


def sum_measured_communication(
    records: Sequence[Mapping[str, Any]] | None,
    *,
    through_round: int | None,
    expected_run_id: str,
    expected_config_sha256: str,
) -> dict[str, Any]:
    """Sum complete, already-aggregated #50 training-round model-payload records.

    The interval is rounds 1..through_round inclusive. Round-1 download includes
    the initial model sent for training. Server-only initial validation is outside
    this boundary. Each input round must already include any measured retries.
    Missing measurements are never zero. Parent code verifies source artifact SHA.
    """
    result: dict[str, Any] = {
        "status": "PENDING_CROSSING" if through_round is None else "PENDING_COMMUNICATION",
        "boundary": "TRAIN_ROUNDS_1_THROUGH_SELECTED_BOUNDARY_INCLUSIVE",
        "through_round": through_round,
        "upload_bytes": None,
        "download_bytes": None,
        "total_bytes": None,
    }
    if through_round is None:
        return result
    through_round = _integer("through_round", through_round)
    if not isinstance(expected_run_id, str) or not expected_run_id:
        raise ValueError("expected_run_id must be a non-empty string")
    if (
        not isinstance(expected_config_sha256, str)
        or len(expected_config_sha256) != 64
        or any(c not in "0123456789abcdef" for c in expected_config_sha256)
    ):
        raise ValueError("expected_config_sha256 must be lowercase SHA-256 hex")
    if records is None:
        return result
    by_round: dict[int, tuple[int, int]] = {}
    for rec in records:
        if (
            rec.get("run_id") != expected_run_id
            or rec.get("config_sha256") != expected_config_sha256
        ):
            raise ValueError("communication record belongs to a different run/config")
        if rec.get("measurement_basis") != "MEASURED_APPLICATION_MODEL_PAYLOAD":
            raise ValueError("estimated tensor sizes are not measured communication")
        round_number = _integer("server_round", rec["server_round"], 1)
        if round_number in by_round:
            raise ValueError("duplicate aggregate communication round")
        by_round[round_number] = (
            _integer("upload_bytes", rec["upload_bytes"]),
            _integer("download_bytes", rec["download_bytes"]),
        )
    missing = sorted(set(range(1, through_round + 1)) - set(by_round))
    if missing:
        result.update(status="INCOMPLETE_COMMUNICATION", missing_rounds=missing)
        return result
    upload = sum(by_round[r][0] for r in range(1, through_round + 1))
    download = sum(by_round[r][1] for r in range(1, through_round + 1))
    result.update(
        status="AVAILABLE",
        upload_bytes=upload,
        download_bytes=download,
        total_bytes=upload + download,
    )
    return result


# ---------------------------------------------------------------------------
# Project-facing identity guard (S2-PR-02)
#
# The primitives above are deliberately unauthenticated arithmetic. The wrapper
# below is the only supported way to feed a real R1 value or measured byte
# records into them: it refuses anything whose identity it cannot verify against
# the frozen project contracts.
# ---------------------------------------------------------------------------

SUPPORTED_POLICY_SCHEMA = "convergence_policy_v1"
SUPPORTED_DIRECTIONS = frozenset({"MAXIMIZE"})


class ConvergenceIdentityError(ValueError):
    """Raised when a curve, policy, R1 result or byte source fails identity checks."""


def _require_mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConvergenceIdentityError(f"{name} must be a mapping")
    return value


def _require_exact(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ConvergenceIdentityError(f"{name} mismatch: expected {expected!r}, got {actual!r}")


def _verify_referenced_file(ref: Mapping[str, Any], *, label: str, repo_root: Any) -> Any:
    """Verify a referenced artifact and return the content that was actually verified.

    The verified file is the authority. Callers must use the returned content rather
    than a separately supplied in-memory payload, so a passing hash can never vouch
    for something the file does not contain.
    """
    from pathlib import Path

    from ppsi.training.identity import file_sha256

    uri = ref.get("uri")
    sha = ref.get("sha256")
    if not isinstance(uri, str) or not uri:
        raise ConvergenceIdentityError(f"{label} ref requires a non-empty uri")
    if not isinstance(sha, str) or len(sha) != 64:
        raise ConvergenceIdentityError(f"{label} ref requires a SHA-256 hex digest")
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    path = root / uri
    if not path.is_file():
        raise ConvergenceIdentityError(f"{label} ref points at a missing file: {uri}")
    actual = file_sha256(path)
    if actual != sha:
        raise ConvergenceIdentityError(
            f"{label} ref SHA-256 mismatch for {uri}: recorded {sha}, actual {actual}"
        )
    import json as _json

    try:
        return _json.loads(path.read_text(encoding="utf-8"))
    except _json.JSONDecodeError as exc:
        raise ConvergenceIdentityError(f"{label} ref is not readable JSON: {uri}") from exc


def _select_single_metric(
    result: Mapping[str, Any],
    *,
    metric_id: str,
    task: str,
    cohort: str,
    direction: str,
    unit: str,
) -> Mapping[str, Any]:
    """Return the one metric matching the frozen identity, or fail."""
    matches = [
        m
        for m in result.get("metrics", [])
        if m.get("metric_id") == metric_id and m.get("task") == task and m.get("cohort") == cohort
    ]
    if not matches:
        raise ConvergenceIdentityError(
            f"R1 result carries no metric {metric_id!r} for task {task!r} cohort {cohort!r}"
        )
    if len(matches) > 1:
        raise ConvergenceIdentityError(
            f"R1 result carries {len(matches)} metrics named {metric_id!r}; identity is ambiguous"
        )
    metric = matches[0]
    _require_exact("R1 metric direction", metric.get("direction"), direction)
    _require_exact("R1 metric unit", metric.get("unit"), unit)
    value = metric.get("value")
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ConvergenceIdentityError("R1 metric value must be a finite real number")
    return metric


def _normalize_communication(records: Any) -> list[tuple[Any, ...]]:
    """Canonical comparison form for aggregate round records.

    Only the fields the arithmetic actually consumes are compared, in round order, so
    two payloads bind if and only if they mean the same thing to the summation.
    """
    if not isinstance(records, list):
        raise ConvergenceIdentityError("communication records must be a list")
    rows = []
    for rec in records:
        rec = _require_mapping("communication record", rec)
        rows.append(
            (
                rec.get("server_round"),
                rec.get("upload_bytes"),
                rec.get("download_bytes"),
                rec.get("run_id"),
                rec.get("config_sha256"),
                rec.get("measurement_basis"),
            )
        )
    return sorted(rows, key=lambda row: (row[0] is None, row[0]))


def analyze_run_curve(
    *,
    curve_points: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
    curve_identity: Mapping[str, Any],
    r1_result: Mapping[str, Any] | None = None,
    r1_config: Mapping[str, Any] | None = None,
    r1_result_ref: Mapping[str, Any] | None = None,
    communication_records: Sequence[Mapping[str, Any]] | None = None,
    communication_source_ref: Mapping[str, Any] | None = None,
    repo_root: Any = None,
) -> dict[str, Any]:
    """Analyze a completed run curve after verifying every identity it depends on.

    ``curve_identity`` must carry the run's own ``task``, ``cohort``,
    ``evaluation_split``, ``metric_id``, ``direction``, ``unit``,
    ``cadence_rounds``, ``run_id``, ``config_sha256`` and its
    ``comparison_compatibility_v1`` tuple. An R1 value is only accepted when the
    supplied ExperimentResult validates, is a SUCCEEDED R1 run, is bound to a
    resolved ExperimentConfig that explicitly declares the VALIDATION split,
    exposes exactly one matching metric, and whose complete compatibility tuple
    compares equal to the curve's own.
    """
    from scripts.experiments.compatibility import (
        build_compatibility_tuple,
        compare_compatibility,
        validate_compatibility_tuple,
    )
    from scripts.experiments.contracts import build_run_id
    from scripts.experiments.schemas import validate_experiment_config, validate_experiment_result

    policy = _require_mapping("policy", policy)
    identity = _require_mapping("curve_identity", curve_identity)
    _require_exact("policy schema", policy.get("schema"), SUPPORTED_POLICY_SCHEMA)

    metric_id = policy["metric_id"]
    task = policy["task"]
    cohort = policy["cohort"]
    split = policy["split"]
    direction = policy["direction"]
    unit = policy["unit"]

    if direction not in SUPPORTED_DIRECTIONS:
        raise ConvergenceIdentityError(
            f"policy direction {direction!r} is unsupported; this v1 policy only ranks "
            "higher-is-better fractions and must not be applied to LogLoss or Brier"
        )

    # The curve must be the same measurement the policy describes.
    _require_exact("curve metric_id", identity.get("metric_id"), metric_id)
    _require_exact("curve task", identity.get("task"), task)
    _require_exact("curve cohort", identity.get("cohort"), cohort)
    _require_exact("curve evaluation_split", identity.get("evaluation_split"), split)
    _require_exact("curve direction", identity.get("direction"), direction)
    _require_exact("curve unit", identity.get("unit"), unit)

    cadence = identity.get("cadence_rounds", policy.get("evaluation_cadence_rounds", 1))
    if policy.get("cadence_must_match_input", True):
        _require_exact("curve cadence_rounds", cadence, policy["evaluation_cadence_rounds"])

    curve_compat = validate_compatibility_tuple(
        dict(_require_mapping("curve compatibility", identity.get("compatibility")))
    )

    run_id = identity.get("run_id")
    config_sha256 = identity.get("config_sha256")
    if not isinstance(run_id, str) or not run_id:
        raise ConvergenceIdentityError("curve_identity.run_id must be a non-empty string")
    if not isinstance(config_sha256, str) or len(config_sha256) != 64:
        raise ConvergenceIdentityError("curve_identity.config_sha256 must be a SHA-256 hex digest")

    guard: dict[str, Any] = {
        "policy_metric_id": metric_id,
        "evaluation_split": split,
        "cadence_rounds": cadence,
        "r1_identity_verified": False,
        "r1_rejection_reason": None,
        "communication_source_verified": False,
    }

    r1_value: float | None = None
    if r1_result is None:
        guard["r1_rejection_reason"] = "NO_R1_RESULT_SUPPLIED"
    else:
        # The referenced file is the authority. A separately supplied payload is only
        # accepted when it is identical to the content that was hash-verified on disk,
        # so a passing hash can never vouch for something the file does not contain.
        if r1_result_ref is None:
            raise ConvergenceIdentityError(
                "an R1 result requires its artifact reference; an unreferenced in-memory "
                "result is never treated as source-verified"
            )
        file_result = _verify_referenced_file(
            _require_mapping("r1_result_ref", r1_result_ref),
            label="R1 result",
            repo_root=repo_root,
        )
        if dict(_require_mapping("r1_result", r1_result)) != file_result:
            raise ConvergenceIdentityError(
                "the supplied R1 result differs from the verified file it references"
            )
        validated = validate_experiment_result(dict(file_result))
        _require_exact("R1 state", validated.get("state"), "SUCCEEDED")
        _require_exact("R1 regime", validated.get("regime"), "R1")
        _require_exact("R1 seed", validated.get("seed"), curve_compat["seed"])
        if task not in validated.get("tasks", []):
            raise ConvergenceIdentityError(f"R1 result does not cover task {task!r}")
        if cohort not in validated.get("evaluation_cohorts", []):
            raise ConvergenceIdentityError(f"R1 result does not evaluate cohort {cohort!r}")

        # Follow the result's own config_ref rather than trusting a caller-supplied config.
        result_config_ref = _require_mapping("R1 config_ref", validated.get("config_ref"))
        file_config = _verify_referenced_file(
            result_config_ref, label="R1 resolved config", repo_root=repo_root
        )
        if r1_config is not None and dict(r1_config) != file_config:
            raise ConvergenceIdentityError(
                "the supplied r1_config differs from the resolved config the result references"
            )
        validated_config = validate_experiment_config(dict(file_config))

        # Run-id integrity: the recorded id must be derivable from this exact config.
        expected_run_id = build_run_id(
            regime=validated_config["regime"],
            tasks=validated_config["tasks"],
            cohorts=validated_config["evaluation_cohorts"],
            seed=validated_config["seed"],
            config_sha256=result_config_ref["sha256"],
            attempt=validated["attempt"],
        )
        if validated["run_id"] != expected_run_id:
            raise ConvergenceIdentityError(
                "R1 run_id does not match the resolved config it references"
            )

        # The VALIDATION declaration is read from the verified config, never inferred.
        regime_config = validated_config.get("regime_config")
        declared_split = (
            regime_config.get("evaluation_split") if isinstance(regime_config, Mapping) else None
        )
        if declared_split != split:
            raise ConvergenceIdentityError(
                f"R1 config does not declare an explicit {split} evaluation split; the split "
                "is never inferred from a filename, a metric name or a caller-supplied value"
            )
        config_tuple = build_compatibility_tuple(validated_config, validated["git_sha"])
        if config_tuple != validated["compatibility"]:
            raise ConvergenceIdentityError(
                "R1 resolved config does not reproduce the result's compatibility tuple"
            )
        comparison = compare_compatibility(
            curve_compat, validate_compatibility_tuple(dict(validated["compatibility"]))
        )
        if not comparison["is_compatible"]:
            raise ConvergenceIdentityError(
                f"R1 compatibility tuple differs from the curve on {comparison['mismatches']}"
            )
        metric = _select_single_metric(
            validated,
            metric_id=metric_id,
            task=task,
            cohort=cohort,
            direction=direction,
            unit=unit,
        )
        r1_value = float(metric["value"])
        guard["r1_identity_verified"] = True
        guard["r1_source_ref"] = {
            "uri": result_config_ref["uri"],
            "result_uri": r1_result_ref["uri"],
        }

    analysis = analyze_curve(
        curve_points,
        cadence_rounds=cadence,
        self_fraction=policy["self_fraction"],
        consecutive=policy["self_consecutive_evaluations"],
        target_fraction=policy["target_fraction"],
        r1_value=r1_value,
    )

    bound_records = communication_records
    if communication_records is not None:
        if communication_source_ref is None:
            raise ConvergenceIdentityError(
                "measured communication records require a verifiable communication_source_ref"
            )
        source_content = _verify_referenced_file(
            _require_mapping("communication_source_ref", communication_source_ref),
            label="communication source",
            repo_root=repo_root,
        )
        file_records = (
            source_content.get("records") if isinstance(source_content, Mapping) else source_content
        )
        if not isinstance(file_records, list):
            raise ConvergenceIdentityError(
                "the verified communication source does not hold a list of round records"
            )
        if _normalize_communication(file_records) != _normalize_communication(
            communication_records
        ):
            raise ConvergenceIdentityError(
                "the supplied byte records differ from the verified communication source"
            )
        # The arithmetic consumes the file's own records, not the caller's copy.
        bound_records = file_records
        guard["communication_source_verified"] = True
        guard["communication_source_ref"] = dict(communication_source_ref)

    self_boundary = analysis["self_convergence"]["confirmation_round"]
    target_boundary = analysis["r1_target"]["crossing_round"]
    communication = {
        "self_convergence": sum_measured_communication(
            bound_records,
            through_round=self_boundary,
            expected_run_id=run_id,
            expected_config_sha256=config_sha256,
        ),
        "r1_target": sum_measured_communication(
            bound_records,
            through_round=target_boundary,
            expected_run_id=run_id,
            expected_config_sha256=config_sha256,
        ),
    }

    return {
        "schema": "convergence_run_analysis_v1",
        "version": "1",
        "run_id": run_id,
        "identity_guard": guard,
        "policy_identity": {
            "schema": policy.get("schema"),
            "metric_id": metric_id,
            "task": task,
            "cohort": cohort,
            "evaluation_split": split,
            "direction": direction,
            "unit": unit,
            "self_fraction": policy["self_fraction"],
            "self_consecutive_evaluations": policy["self_consecutive_evaluations"],
            "target_fraction": policy["target_fraction"],
            "cadence_rounds": cadence,
        },
        "source_refs": {
            "r1_result_ref": dict(r1_result_ref) if r1_result_ref is not None else None,
            "communication_source_ref": (
                dict(communication_source_ref) if communication_source_ref is not None else None
            ),
        },
        "curve_analysis": analysis,
        "communication": communication,
        "real_r1_status": "AVAILABLE" if r1_value is not None else "PENDING_R1",
        "actual_communication_status": (
            "AVAILABLE" if guard["communication_source_verified"] else "PENDING_50_INSTRUMENTATION"
        ),
    }
