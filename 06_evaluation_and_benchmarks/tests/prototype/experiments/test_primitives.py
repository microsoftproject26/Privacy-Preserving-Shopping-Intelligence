"""Phase 1 tests: primitives, canonical JSON, hashing, refs, run-ID, states."""

from __future__ import annotations

import json

import pytest

from scripts.experiments.contracts import (
    ALLOWED_SEEDS,
    METRIC_DIRECTIONS,
    METRIC_UNITS,
    REGIMES,
    RUN_STATES,
    TERMINAL_STATES,
    ContractValidationError,
    build_run_id,
    canonical_json_bytes,
    canonical_sha256,
    parse_run_id,
    reject_unknown_fields,
    strict_bool,
    strict_finite_float,
    strict_int,
    strict_nonneg_int,
    strict_positive_int,
    validate_artifact_ref,
    validate_cohort_set,
    validate_metric_direction,
    validate_metric_unit,
    validate_regime,
    validate_run_id_config_prefix,
    validate_run_state,
    validate_schema_header,
    validate_seed,
    validate_sha256,
    validate_state_transition,
    validate_tasks,
    validate_utc_timestamp,
    validate_version_string,
)

# ── Helpers ──────────────────────────────────────────────────────────────

GOOD_SHA = "a" * 64
GOOD_UTC = "2026-01-15T10:30:00Z"


def _make_artifact_ref(**overrides: object) -> dict:
    base = {
        "schema": "artifact_ref_v1",
        "version": "1",
        "logical_id": "test_artifact",
        "artifact_schema": "model_config_v1",
        "artifact_version": "1",
        "uri": "config/experiments/schemas/model_config.v1.schema.json",
        "sha256": GOOD_SHA,
    }
    base.update(overrides)
    return base


# ── strict_int ───────────────────────────────────────────────────────────


class TestStrictInt:
    def test_valid_int(self) -> None:
        assert strict_int(42, "x") == 42

    def test_bool_rejected(self) -> None:
        with pytest.raises(ContractValidationError, match="not bool"):
            strict_int(True, "x")

    def test_float_rejected(self) -> None:
        with pytest.raises(ContractValidationError):
            strict_int(3.14, "x")

    def test_string_rejected(self) -> None:
        with pytest.raises(ContractValidationError):
            strict_int("42", "x")

    def test_minimum_enforced(self) -> None:
        assert strict_int(5, "x", minimum=5) == 5
        with pytest.raises(ContractValidationError, match=">="):
            strict_int(4, "x", minimum=5)


class TestStrictPositiveInt:
    def test_one(self) -> None:
        assert strict_positive_int(1, "x") == 1

    def test_zero_rejected(self) -> None:
        with pytest.raises(ContractValidationError):
            strict_positive_int(0, "x")


class TestStrictNonnegInt:
    def test_zero(self) -> None:
        assert strict_nonneg_int(0, "x") == 0

    def test_negative_rejected(self) -> None:
        with pytest.raises(ContractValidationError):
            strict_nonneg_int(-1, "x")


# ── strict_bool ──────────────────────────────────────────────────────────


class TestStrictBool:
    def test_valid(self) -> None:
        assert strict_bool(True, "x") is True
        assert strict_bool(False, "x") is False

    def test_int_rejected(self) -> None:
        with pytest.raises(ContractValidationError):
            strict_bool(1, "x")


# ── strict_finite_float ──────────────────────────────────────────────────


class TestStrictFiniteFloat:
    def test_valid(self) -> None:
        assert strict_finite_float(3.14, "x") == 3.14
        assert strict_finite_float(0, "x") == 0.0

    def test_nan_rejected(self) -> None:
        with pytest.raises(ContractValidationError, match="NaN"):
            strict_finite_float(float("nan"), "x")

    def test_inf_rejected(self) -> None:
        with pytest.raises(ContractValidationError, match="NaN"):
            strict_finite_float(float("inf"), "x")

    def test_bool_rejected(self) -> None:
        with pytest.raises(ContractValidationError):
            strict_finite_float(True, "x")


# ── validate_version_string ──────────────────────────────────────────────


class TestVersionString:
    def test_valid_string(self) -> None:
        assert validate_version_string("1") == "1"

    def test_integer_rejected(self) -> None:
        with pytest.raises(ContractValidationError, match="string"):
            validate_version_string(1)

    def test_empty_rejected(self) -> None:
        with pytest.raises(ContractValidationError, match="non-empty"):
            validate_version_string("")


# ── validate_seed ────────────────────────────────────────────────────────


class TestValidateSeed:
    @pytest.mark.parametrize("seed", sorted(ALLOWED_SEEDS))
    def test_allowed(self, seed: int) -> None:
        assert validate_seed(seed) == seed

    def test_disallowed(self) -> None:
        with pytest.raises(ContractValidationError):
            validate_seed(7)

    def test_zero_rejected(self) -> None:
        with pytest.raises(ContractValidationError):
            validate_seed(0)

    def test_bool_rejected(self) -> None:
        with pytest.raises(ContractValidationError):
            validate_seed(True)


# ── validate_sha256 ──────────────────────────────────────────────────────


class TestValidateSHA256:
    def test_valid(self) -> None:
        assert validate_sha256(GOOD_SHA) == GOOD_SHA

    def test_uppercase_rejected(self) -> None:
        with pytest.raises(ContractValidationError, match="lowercase"):
            validate_sha256("A" * 64)

    def test_short_rejected(self) -> None:
        with pytest.raises(ContractValidationError):
            validate_sha256("a" * 63)

    def test_non_hex_rejected(self) -> None:
        with pytest.raises(ContractValidationError):
            validate_sha256("g" * 64)


# ── validate_utc_timestamp ───────────────────────────────────────────────


class TestValidateUTC:
    def test_valid(self) -> None:
        assert validate_utc_timestamp(GOOD_UTC) == GOOD_UTC

    def test_non_utc_rejected(self) -> None:
        with pytest.raises(ContractValidationError, match="UTC"):
            validate_utc_timestamp("2026-01-15T10:30:00+05:00")

    def test_invalid_date_rejected(self) -> None:
        with pytest.raises(ContractValidationError, match="timestamp"):
            validate_utc_timestamp("not-a-dateZ")


# ── validate_regime ──────────────────────────────────────────────────────


class TestValidateRegime:
    @pytest.mark.parametrize("regime", sorted(REGIMES))
    def test_valid(self, regime: str) -> None:
        assert validate_regime(regime) == regime

    def test_unknown_rejected(self) -> None:
        with pytest.raises(ContractValidationError):
            validate_regime("R99")


# ── validate_tasks ───────────────────────────────────────────────────────


class TestValidateTasks:
    def test_canonical_order(self) -> None:
        assert validate_tasks(["T1", "T2", "T3"]) == ["T1", "T2", "T3"]

    def test_subset(self) -> None:
        assert validate_tasks(["T1", "T3"]) == ["T1", "T3"]

    def test_non_canonical_rejected(self) -> None:
        with pytest.raises(ContractValidationError, match="canonical order"):
            validate_tasks(["T2", "T1"])

    def test_duplicate_rejected(self) -> None:
        with pytest.raises(ContractValidationError, match="duplicate"):
            validate_tasks(["T1", "T1"])

    def test_invalid_task_rejected(self) -> None:
        with pytest.raises(ContractValidationError, match="invalid task"):
            validate_tasks(["T1", "T99"])

    def test_empty_rejected(self) -> None:
        with pytest.raises(ContractValidationError, match="non-empty"):
            validate_tasks([])


# ── validate_cohort_set ──────────────────────────────────────────────────


class TestValidateCohortSet:
    def test_full_set(self) -> None:
        assert validate_cohort_set(["C1", "C2", "C3"]) == ["C1", "C2", "C3"]

    def test_subset(self) -> None:
        assert validate_cohort_set(["C1"]) == ["C1"]

    def test_non_canonical_rejected(self) -> None:
        with pytest.raises(ContractValidationError, match="canonical order"):
            validate_cohort_set(["C2", "C1"])


# ── canonical JSON ───────────────────────────────────────────────────────


class TestCanonicalJSON:
    def test_key_order_stable(self) -> None:
        assert canonical_sha256({"b": 2, "a": 1}) == canonical_sha256({"a": 1, "b": 2})

    def test_utf8_stability(self) -> None:
        b = canonical_json_bytes({"name": "café"})
        assert "café".encode() in b

    def test_nan_rejected(self) -> None:
        with pytest.raises(ContractValidationError, match="NaN"):
            canonical_json_bytes({"x": float("nan")})

    def test_inf_rejected(self) -> None:
        with pytest.raises(ContractValidationError, match="NaN"):
            canonical_json_bytes({"x": float("inf")})

    def test_nested_nan_rejected(self) -> None:
        with pytest.raises(ContractValidationError, match="NaN"):
            canonical_json_bytes({"outer": {"inner": float("nan")}})

    def test_pretty_vs_compact_same_hash(self) -> None:
        data = {"version": "1", "seed": 42}
        pretty = json.dumps(data, indent=2)
        reparsed = json.loads(pretty)
        assert canonical_sha256(reparsed) == canonical_sha256(data)

    def test_non_string_key_rejected(self) -> None:
        # json.dumps would fail on non-string keys, but our _reject_nonfinite
        # catches it explicitly.
        with pytest.raises(ContractValidationError, match="non-string key"):
            canonical_json_bytes({1: "bad"})


# ── reject_unknown_fields ────────────────────────────────────────────────


class TestRejectUnknownFields:
    def test_valid(self) -> None:
        reject_unknown_fields({"a": 1}, {"a"}, "test")

    def test_unknown_rejected(self) -> None:
        with pytest.raises(ContractValidationError, match="unknown fields"):
            reject_unknown_fields({"a": 1, "b": 2}, {"a"}, "test")


# ── validate_schema_header ───────────────────────────────────────────────


class TestSchemaHeader:
    def test_valid(self) -> None:
        validate_schema_header({"schema": "test_v1", "version": "1"}, "test_v1")

    def test_wrong_schema(self) -> None:
        with pytest.raises(ContractValidationError, match="schema"):
            validate_schema_header({"schema": "wrong", "version": "1"}, "test_v1")

    def test_wrong_version(self) -> None:
        with pytest.raises(ContractValidationError, match="version"):
            validate_schema_header({"schema": "test_v1", "version": "2"}, "test_v1")

    def test_int_version_rejected(self) -> None:
        with pytest.raises(ContractValidationError, match="version"):
            validate_schema_header({"schema": "test_v1", "version": 1}, "test_v1")


# ── validate_artifact_ref ────────────────────────────────────────────────


class TestArtifactRef:
    def test_valid(self) -> None:
        ref = _make_artifact_ref()
        result = validate_artifact_ref(ref)
        assert result["sha256"] == GOOD_SHA

    def test_unknown_field_rejected(self) -> None:
        ref = _make_artifact_ref(extra="bad")
        with pytest.raises(ContractValidationError, match="unknown"):
            validate_artifact_ref(ref)

    def test_missing_logical_id_rejected(self) -> None:
        ref = _make_artifact_ref()
        del ref["logical_id"]
        with pytest.raises(ContractValidationError, match="logical_id"):
            validate_artifact_ref(ref)

    def test_absolute_uri_rejected(self) -> None:
        ref = _make_artifact_ref(uri="/absolute/path")
        with pytest.raises(ContractValidationError, match="repo-relative"):
            validate_artifact_ref(ref)

    def test_windows_uri_rejected(self) -> None:
        ref = _make_artifact_ref(uri="C:\\Users\\test")
        with pytest.raises(ContractValidationError, match="repo-relative"):
            validate_artifact_ref(ref)

    def test_bad_sha_rejected(self) -> None:
        ref = _make_artifact_ref(sha256="ABC")
        with pytest.raises(ContractValidationError, match="lowercase"):
            validate_artifact_ref(ref)


# ── Run-ID v1 ────────────────────────────────────────────────────────────


class TestRunID:
    def test_build_parse_roundtrip(self) -> None:
        rid = build_run_id(
            regime="R2A",
            tasks=["T1", "T2", "T3"],
            cohorts=["C1", "C2", "C3"],
            seed=13,
            config_sha256=GOOD_SHA,
            attempt=1,
        )
        assert rid == f"run-v1__r2a__t1-t2-t3__c1-c2-c3__s13__cfg{'a' * 12}__a1"
        parsed = parse_run_id(rid)
        assert parsed["regime"] == "R2A"
        assert parsed["tasks"] == ["T1", "T2", "T3"]
        assert parsed["cohorts"] == ["C1", "C2", "C3"]
        assert parsed["seed"] == 13
        assert parsed["attempt"] == 1

    def test_attempt_change_changes_id(self) -> None:
        common = {
            "regime": "R1",
            "tasks": ["T1"],
            "cohorts": ["C1"],
            "seed": 42,
            "config_sha256": GOOD_SHA,
        }
        r1 = build_run_id(**common, attempt=1)
        r2 = build_run_id(**common, attempt=2)
        assert r1 != r2

    def test_seed_change_changes_id(self) -> None:
        common = {
            "regime": "R1",
            "tasks": ["T1"],
            "cohorts": ["C1"],
            "config_sha256": GOOD_SHA,
            "attempt": 1,
        }
        r1 = build_run_id(**common, seed=13)
        r2 = build_run_id(**common, seed=42)
        assert r1 != r2

    def test_config_change_changes_id(self) -> None:
        common = {"regime": "R1", "tasks": ["T1"], "cohorts": ["C1"], "seed": 13, "attempt": 1}
        r1 = build_run_id(**common, config_sha256="a" * 64)
        r2 = build_run_id(**common, config_sha256="b" * 64)
        assert r1 != r2

    def test_malformed_rejected(self) -> None:
        with pytest.raises(ContractValidationError, match="malformed"):
            parse_run_id("not-a-run-id")

    def test_attempt_zero_rejected(self) -> None:
        with pytest.raises(ContractValidationError):
            build_run_id(
                regime="R1",
                tasks=["T1"],
                cohorts=["C1"],
                seed=13,
                config_sha256=GOOD_SHA,
                attempt=0,
            )

    def test_config_prefix_validation(self) -> None:
        rid = build_run_id(
            regime="R1",
            tasks=["T1"],
            cohorts=["C1"],
            seed=13,
            config_sha256=GOOD_SHA,
            attempt=1,
        )
        validate_run_id_config_prefix(rid, GOOD_SHA)
        with pytest.raises(ContractValidationError, match="prefix"):
            validate_run_id_config_prefix(rid, "b" * 64)


# ── State transitions ───────────────────────────────────────────────────


class TestStateTransitions:
    def test_valid_transitions(self) -> None:
        validate_state_transition("PLANNED", "RUNNING")
        validate_state_transition("RUNNING", "SUCCEEDED")
        validate_state_transition("RUNNING", "FAILED")
        validate_state_transition("RUNNING", "INTERRUPTED")
        validate_state_transition("INTERRUPTED", "RUNNING")

    def test_terminal_no_exit(self) -> None:
        for state in TERMINAL_STATES:
            with pytest.raises(ContractValidationError, match="not allowed"):
                validate_state_transition(state, "RUNNING")

    def test_invalid_backward(self) -> None:
        with pytest.raises(ContractValidationError, match="not allowed"):
            validate_state_transition("SUCCEEDED", "PLANNED")


# ── Enum validators ─────────────────────────────────────────────────────


class TestEnumValidators:
    @pytest.mark.parametrize("direction", sorted(METRIC_DIRECTIONS))
    def test_valid_direction(self, direction: str) -> None:
        assert validate_metric_direction(direction) == direction

    def test_unknown_direction(self) -> None:
        with pytest.raises(ContractValidationError):
            validate_metric_direction("UP")

    @pytest.mark.parametrize("unit", sorted(METRIC_UNITS))
    def test_valid_unit(self, unit: str) -> None:
        assert validate_metric_unit(unit) == unit

    def test_unknown_unit(self) -> None:
        with pytest.raises(ContractValidationError):
            validate_metric_unit("SCORE")

    @pytest.mark.parametrize("state", sorted(RUN_STATES))
    def test_valid_state(self, state: str) -> None:
        assert validate_run_state(state) == state

    def test_unknown_state(self) -> None:
        with pytest.raises(ContractValidationError):
            validate_run_state("DONE")


class TestAttemptTransitions:
    def test_resume_running_to_interrupted(self) -> None:
        from scripts.experiments.contracts import validate_attempt_transition

        rid = build_run_id(
            regime="R1", tasks=["T1"], cohorts=["C1"], seed=13, config_sha256=GOOD_SHA, attempt=1
        )
        validate_attempt_transition(rid, "RUNNING", rid, "INTERRUPTED")

    def test_resume_terminal_rejected(self) -> None:
        from scripts.experiments.contracts import validate_attempt_transition

        rid = build_run_id(
            regime="R1", tasks=["T1"], cohorts=["C1"], seed=13, config_sha256=GOOD_SHA, attempt=1
        )
        with pytest.raises(ContractValidationError, match="terminal state"):
            validate_attempt_transition(rid, "FAILED", rid, "FAILED")

    def test_new_attempt_valid(self) -> None:
        from scripts.experiments.contracts import validate_attempt_transition

        rid1 = build_run_id(
            regime="R1", tasks=["T1"], cohorts=["C1"], seed=13, config_sha256=GOOD_SHA, attempt=1
        )
        rid2 = build_run_id(
            regime="R1", tasks=["T1"], cohorts=["C1"], seed=13, config_sha256=GOOD_SHA, attempt=2
        )
        validate_attempt_transition(rid1, "FAILED", rid2, "PLANNED")

    def test_new_attempt_must_match_config(self) -> None:
        from scripts.experiments.contracts import validate_attempt_transition

        rid1 = build_run_id(
            regime="R1", tasks=["T1"], cohorts=["C1"], seed=13, config_sha256=GOOD_SHA, attempt=1
        )
        rid2 = build_run_id(
            regime="R2A", tasks=["T1"], cohorts=["C1"], seed=13, config_sha256=GOOD_SHA, attempt=2
        )
        with pytest.raises(ContractValidationError, match="must match regime"):
            validate_attempt_transition(rid1, "FAILED", rid2, "PLANNED")

    def test_new_attempt_number_must_increase(self) -> None:
        from scripts.experiments.contracts import validate_attempt_transition

        rid1 = build_run_id(
            regime="R1", tasks=["T1"], cohorts=["C1"], seed=13, config_sha256=GOOD_SHA, attempt=2
        )
        rid2 = build_run_id(
            regime="R1", tasks=["T1"], cohorts=["C1"], seed=13, config_sha256=GOOD_SHA, attempt=1
        )
        with pytest.raises(ContractValidationError, match="must be > old attempt"):
            validate_attempt_transition(rid1, "FAILED", rid2, "PLANNED")
