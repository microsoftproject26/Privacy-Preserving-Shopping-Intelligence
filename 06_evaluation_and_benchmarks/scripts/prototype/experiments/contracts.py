"""Experiment contract primitives, validators, and identity utilities.

Implements the S1-PR-03 frozen contract (14_FINAL_CONTRACT_FREEZE.md).
Approval basis: technical freeze by Ahmed/project-owner delegation.
Another-member PR review required before merge.

All versioned records use ``version`` as a string (e.g. ``"1"``).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALLOWED_SEEDS: frozenset[int] = frozenset({13, 42, 2026})

REGIMES: frozenset[str] = frozenset({"R1", "R2A", "R2B", "R3", "R4", "R5"})

TASK_ORDER: tuple[str, ...] = ("T1", "T2", "T3")

COHORT_ORDER: tuple[str, ...] = ("C1", "C2", "C3")

RUN_STATES: frozenset[str] = frozenset(
    {"PLANNED", "RUNNING", "INTERRUPTED", "SUCCEEDED", "FAILED", "CANCELLED"}
)

TERMINAL_STATES: frozenset[str] = frozenset({"SUCCEEDED", "FAILED", "CANCELLED"})

METRIC_DIRECTIONS: frozenset[str] = frozenset({"MAXIMIZE", "MINIMIZE", "NEUTRAL"})

METRIC_UNITS: frozenset[str] = frozenset({"FRACTION", "RATIO", "COUNT", "UNITLESS"})

MEASUREMENT_STATUSES: frozenset[str] = frozenset({"AVAILABLE", "PENDING", "NOT_APPLICABLE"})

ARTIFACT_KINDS: frozenset[str] = frozenset({"FIXTURE_PROOF", "AUTHORITATIVE"})

INITIALIZATION_KINDS: frozenset[str] = frozenset(
    {"COMMON_INITIALIZATION", "PRETRAINED_R1_CHECKPOINT"}
)

# Regimes that use COMMON_INITIALIZATION.
COMMON_INIT_REGIMES: frozenset[str] = frozenset({"R1", "R2A", "R2B", "R3", "R4"})

# Regimes that use PRETRAINED_R1_CHECKPOINT.
PRETRAINED_INIT_REGIMES: frozenset[str] = frozenset({"R5"})

SHA256_RE: re.Pattern[str] = re.compile(r"^[0-9a-f]{64}$")

# Run-ID grammar v1 regex.
_RUN_ID_RE: re.Pattern[str] = re.compile(
    r"^run-v1__"
    r"(?P<regime>r1|r2a|r2b|r3|r4|r5)__"
    r"(?P<task_set>(?:t[123])(?:-t[123])*)__"
    r"(?P<cohort_set>(?:c[123])(?:-c[123])*)__"
    r"s(?P<seed>\d+)__"
    r"cfg(?P<cfg_prefix>[0-9a-f]{12})__"
    r"a(?P<attempt>[1-9]\d*)$"
)

# State transitions: from_state -> set of valid to_states.
STATE_TRANSITIONS: dict[str, frozenset[str]] = {
    "PLANNED": frozenset({"RUNNING", "CANCELLED"}),
    "RUNNING": frozenset({"INTERRUPTED", "SUCCEEDED", "FAILED", "CANCELLED"}),
    "INTERRUPTED": frozenset({"RUNNING", "CANCELLED"}),
    "SUCCEEDED": frozenset(),
    "FAILED": frozenset(),
    "CANCELLED": frozenset(),
}


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------


class ContractValidationError(ValueError):
    """Raised when a contract invariant is violated."""


# ---------------------------------------------------------------------------
# Primitive validators
# ---------------------------------------------------------------------------


def strict_bool(value: Any, field: str) -> bool:
    """Validate a strict boolean. Integers are NOT accepted."""
    if not isinstance(value, bool):
        raise ContractValidationError(f"{field}: must be bool, got {type(value).__name__}")
    return value


def strict_int(value: Any, field: str, *, minimum: int | None = None) -> int:
    """Validate a strict integer. Booleans are NOT accepted."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractValidationError(
            f"{field}: must be int (not bool), got {type(value).__name__}"
        )
    if minimum is not None and value < minimum:
        raise ContractValidationError(f"{field}: must be >= {minimum}, got {value}")
    return value


def strict_positive_int(value: Any, field: str) -> int:
    """Validate a strict positive integer (>= 1)."""
    return strict_int(value, field, minimum=1)


def strict_nonneg_int(value: Any, field: str) -> int:
    """Validate a strict non-negative integer (>= 0)."""
    return strict_int(value, field, minimum=0)


def strict_finite_float(value: Any, field: str) -> float:
    """Validate a finite float. NaN/Inf rejected."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ContractValidationError(f"{field}: must be numeric, got {type(value).__name__}")
    fval = float(value)
    if not math.isfinite(fval):
        raise ContractValidationError(f"{field}: NaN/Inf not allowed")
    return fval


def strict_nonempty_string(value: Any, field: str) -> str:
    """Validate a non-empty string."""
    if not isinstance(value, str) or not value:
        raise ContractValidationError(f"{field}: must be non-empty string")
    return value


def validate_version_string(value: Any, field: str = "version") -> str:
    """Validate that version is a string (not int)."""
    if not isinstance(value, str):
        raise ContractValidationError(
            f"{field}: version must be a string, got {type(value).__name__}"
        )
    if not value:
        raise ContractValidationError(f"{field}: version must be non-empty")
    return value


def validate_seed(seed: Any) -> int:
    """Validate seed is exactly 13, 42, or 2026."""
    seed = strict_int(seed, "seed")
    if seed not in ALLOWED_SEEDS:
        raise ContractValidationError(f"seed: must be one of {sorted(ALLOWED_SEEDS)}, got {seed}")
    return seed


def validate_sha256(value: Any, field: str = "sha256") -> str:
    """Validate a SHA-256 hex string (64 lowercase hex characters)."""
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ContractValidationError(f"{field}: must be exactly 64 lowercase hex characters")
    return value


def validate_utc_timestamp(value: Any, field: str = "timestamp") -> str:
    """Validate an ISO 8601 UTC timestamp ending in 'Z'."""
    if not isinstance(value, str):
        raise ContractValidationError(f"{field}: must be a string")
    if not value.endswith("Z"):
        raise ContractValidationError(f"{field}: must end with 'Z' (UTC)")
    try:
        datetime.fromisoformat(value)
    except ValueError as exc:
        raise ContractValidationError(f"{field}: invalid ISO 8601 timestamp") from exc
    return value


def validate_regime(value: Any, field: str = "regime") -> str:
    """Validate regime is one of the known regimes."""
    if not isinstance(value, str) or value not in REGIMES:
        raise ContractValidationError(f"{field}: must be one of {sorted(REGIMES)}, got {value!r}")
    return value


def validate_cohort(value: Any, field: str = "cohort") -> str:
    """Validate a single cohort identifier."""
    if not isinstance(value, str) or value not in COHORT_ORDER:
        raise ContractValidationError(
            f"{field}: must be one of {list(COHORT_ORDER)}, got {value!r}"
        )
    return value


def validate_tasks(tasks: Any, field: str = "tasks") -> list[str]:
    """Validate and return tasks in canonical T1, T2, T3 order."""
    if not isinstance(tasks, list) or not tasks:
        raise ContractValidationError(f"{field}: must be a non-empty list")
    seen: set[str] = set()
    for t in tasks:
        if not isinstance(t, str) or t not in TASK_ORDER:
            raise ContractValidationError(f"{field}: invalid task {t!r}")
        if t in seen:
            raise ContractValidationError(f"{field}: duplicate task {t!r}")
        seen.add(t)
    canonical = [t for t in TASK_ORDER if t in seen]
    if tasks != canonical:
        raise ContractValidationError(
            f"{field}: must be in canonical order {canonical}, got {tasks}"
        )
    return canonical


def validate_cohort_set(cohorts: Any, field: str = "cohorts") -> list[str]:
    """Validate and return cohorts in canonical C1, C2, C3 order."""
    if not isinstance(cohorts, list) or not cohorts:
        raise ContractValidationError(f"{field}: must be a non-empty list")
    seen: set[str] = set()
    for c in cohorts:
        if not isinstance(c, str) or c not in COHORT_ORDER:
            raise ContractValidationError(f"{field}: invalid cohort {c!r}")
        if c in seen:
            raise ContractValidationError(f"{field}: duplicate cohort {c!r}")
        seen.add(c)
    canonical = [c for c in COHORT_ORDER if c in seen]
    if cohorts != canonical:
        raise ContractValidationError(
            f"{field}: must be in canonical order {canonical}, got {cohorts}"
        )
    return canonical


def validate_run_state(value: Any, field: str = "state") -> str:
    """Validate a run state enum."""
    if not isinstance(value, str) or value not in RUN_STATES:
        raise ContractValidationError(
            f"{field}: must be one of {sorted(RUN_STATES)}, got {value!r}"
        )
    return value


def validate_state_transition(from_state: str, to_state: str) -> None:
    """Validate that a state transition is allowed."""
    validate_run_state(from_state, "from_state")
    validate_run_state(to_state, "to_state")
    allowed = STATE_TRANSITIONS[from_state]
    if to_state not in allowed:
        raise ContractValidationError(
            f"state transition {from_state} -> {to_state} is not allowed; "
            f"valid transitions from {from_state}: {sorted(allowed) or 'none (terminal)'}"
        )


def validate_attempt_transition(
    old_run_id: str, old_state: str, new_run_id: str, new_state: str
) -> None:
    """Validate resume and rerun immutability rules."""
    old_parsed = parse_run_id(old_run_id)
    new_parsed = parse_run_id(new_run_id)

    validate_run_state(old_state, "old_state")
    validate_run_state(new_state, "new_state")

    is_terminal = old_state in TERMINAL_STATES
    same_run = old_run_id == new_run_id

    if same_run:
        if is_terminal:
            raise ContractValidationError(
                f"Cannot resume terminal state {old_state}. A new attempt is required."
            )
        validate_state_transition(old_state, new_state)
    else:
        # A new run ID is provided. It must be for a new attempt.
        # Ensure it's the exact same config/seed/etc but attempt is strictly greater.
        for key in ["regime", "tasks", "cohorts", "seed", "cfg_prefix"]:
            if old_parsed[key] != new_parsed[key]:
                raise ContractValidationError(f"New attempt must match {key}")
        if new_parsed["attempt"] <= old_parsed["attempt"]:
            raise ContractValidationError(
                f"New attempt {new_parsed['attempt']} must be > old attempt {old_parsed['attempt']}"
            )


def validate_metric_direction(value: Any, field: str = "direction") -> str:
    """Validate a metric direction enum."""
    if not isinstance(value, str) or value not in METRIC_DIRECTIONS:
        raise ContractValidationError(
            f"{field}: must be one of {sorted(METRIC_DIRECTIONS)}, got {value!r}"
        )
    return value


def validate_metric_unit(value: Any, field: str = "unit") -> str:
    """Validate a metric unit enum."""
    if not isinstance(value, str) or value not in METRIC_UNITS:
        raise ContractValidationError(
            f"{field}: must be one of {sorted(METRIC_UNITS)}, got {value!r}"
        )
    return value


# ---------------------------------------------------------------------------
# Canonical JSON / hashing
# ---------------------------------------------------------------------------


def _reject_nonfinite(value: Any, path: str = "") -> None:
    """Recursively reject NaN/Inf floats in a JSON-like structure."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ContractValidationError(f"NaN/Inf not allowed at {path or 'root'}")
    if isinstance(value, bool):
        return  # bool is a subclass of int; skip
    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str):
                raise ContractValidationError(f"non-string key {k!r} at {path or 'root'}")
            _reject_nonfinite(v, f"{path}.{k}" if path else k)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            _reject_nonfinite(v, f"{path}[{i}]")


def canonical_json_bytes(value: Any) -> bytes:
    """Produce canonical UTF-8 JSON bytes for identity hashing.

    Rules (Decision 2 of 14_FINAL_CONTRACT_FREEZE.md):
    - sort_keys=True, separators=(",",":"), ensure_ascii=False, allow_nan=False
    - reject NaN/Inf before serialization
    - reject non-string object keys
    - no BOM
    """
    _reject_nonfinite(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """SHA-256 of canonical JSON bytes."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


# ---------------------------------------------------------------------------
# Unknown field rejection
# ---------------------------------------------------------------------------


def reject_unknown_fields(
    record: dict[str, Any],
    known_fields: frozenset[str] | set[str],
    schema_name: str,
) -> None:
    """Raise if record contains fields not in known_fields."""
    if not isinstance(record, dict):
        raise ContractValidationError(f"{schema_name}: expected dict")
    unknown = set(record.keys()) - set(known_fields)
    if unknown:
        raise ContractValidationError(f"{schema_name}: unknown fields {sorted(unknown)}")


# ---------------------------------------------------------------------------
# Schema/version validation
# ---------------------------------------------------------------------------


def validate_schema_header(
    record: dict[str, Any],
    expected_schema: str,
    expected_version: str = "1",
) -> None:
    """Validate the schema/version header of a record."""
    if not isinstance(record, dict):
        raise ContractValidationError("record must be a dict")
    schema = record.get("schema")
    if schema != expected_schema:
        raise ContractValidationError(f"schema: expected {expected_schema!r}, got {schema!r}")
    version = record.get("version")
    if version != expected_version:
        raise ContractValidationError(f"version: expected {expected_version!r}, got {version!r}")


# ---------------------------------------------------------------------------
# ArtifactRef v1
# ---------------------------------------------------------------------------

_ARTIFACT_REF_FIELDS: frozenset[str] = frozenset(
    {"schema", "version", "logical_id", "artifact_schema", "artifact_version", "uri", "sha256"}
)


def validate_artifact_ref(ref: Any, field: str = "artifact_ref") -> dict[str, Any]:
    """Validate an ArtifactRef v1 record."""
    if not isinstance(ref, dict):
        raise ContractValidationError(f"{field}: must be a dict")
    reject_unknown_fields(ref, _ARTIFACT_REF_FIELDS, f"{field} (artifact_ref_v1)")
    validate_schema_header(ref, "artifact_ref_v1")
    strict_nonempty_string(ref.get("logical_id"), f"{field}.logical_id")
    strict_nonempty_string(ref.get("artifact_schema"), f"{field}.artifact_schema")
    validate_version_string(ref.get("artifact_version"), f"{field}.artifact_version")
    uri = strict_nonempty_string(ref.get("uri"), f"{field}.uri")
    # Reject machine-specific absolute paths (Windows drive letters, UNC, Unix root).
    if re.match(r"^[A-Za-z]:|^\\\\|^/", uri):
        raise ContractValidationError(
            f"{field}.uri: must be repo-relative (no absolute paths), got {uri!r}"
        )
    validate_sha256(ref.get("sha256"), f"{field}.sha256")
    return ref


# ---------------------------------------------------------------------------
# Run-ID v1 build / parse
# ---------------------------------------------------------------------------


def build_run_id(
    *,
    regime: str,
    tasks: list[str],
    cohorts: list[str],
    seed: int,
    config_sha256: str,
    attempt: int,
) -> str:
    """Build a deterministic run-ID v1 string.

    Grammar: run-v1__<regime>__<task-set>__<cohort-set>__s<seed>__cfg<12hex>__a<attempt>
    """
    validate_regime(regime)
    tasks = validate_tasks(tasks)
    cohorts = validate_cohort_set(cohorts)
    validate_seed(seed)
    validate_sha256(config_sha256, "config_sha256")
    strict_positive_int(attempt, "attempt")
    task_str = "-".join(t.lower() for t in tasks)
    cohort_str = "-".join(c.lower() for c in cohorts)
    return (
        f"run-v1__{regime.lower()}__{task_str}__{cohort_str}__"
        f"s{seed}__cfg{config_sha256[:12]}__a{attempt}"
    )


def parse_run_id(run_id: str) -> dict[str, Any]:
    """Parse a run-ID v1 string into its components.

    Returns a dict with keys: regime, tasks, cohorts, seed, cfg_prefix, attempt.
    Raises ContractValidationError on malformed input.
    """
    if not isinstance(run_id, str):
        raise ContractValidationError("run_id: must be a string")
    m = _RUN_ID_RE.fullmatch(run_id)
    if m is None:
        raise ContractValidationError(f"run_id: malformed run-ID {run_id!r}")
    regime = m.group("regime").upper()
    tasks = [t.upper() for t in m.group("task_set").split("-")]
    cohorts = [c.upper() for c in m.group("cohort_set").split("-")]
    seed = int(m.group("seed"))
    cfg_prefix = m.group("cfg_prefix")
    attempt = int(m.group("attempt"))

    # Validate parsed values against contract rules.
    validate_regime(regime)
    validate_tasks(tasks)
    validate_cohort_set(cohorts)
    validate_seed(seed)
    strict_positive_int(attempt, "attempt")

    return {
        "regime": regime,
        "tasks": tasks,
        "cohorts": cohorts,
        "seed": seed,
        "cfg_prefix": cfg_prefix,
        "attempt": attempt,
    }


def validate_run_id_config_prefix(run_id: str, full_config_sha256: str) -> None:
    """Validate that a run-ID's config prefix matches the full config SHA-256."""
    parsed = parse_run_id(run_id)
    validate_sha256(full_config_sha256, "full_config_sha256")
    expected_prefix = full_config_sha256[:12]
    if parsed["cfg_prefix"] != expected_prefix:
        raise ContractValidationError(
            f"run_id config prefix {parsed['cfg_prefix']!r} does not match "
            f"full config SHA-256 prefix {expected_prefix!r}"
        )
