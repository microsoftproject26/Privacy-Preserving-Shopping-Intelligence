"""Experiment contract schemas and validators.

Implements structural validation for:
- model_config_v1

The shared GRU architecture was ratified for the Phase-1 MVP by CP-ARCH.
"""

from __future__ import annotations

from typing import Any

from scripts.experiments.contracts import (
    COMMON_INIT_REGIMES,
    INITIALIZATION_KINDS,
    PRETRAINED_INIT_REGIMES,
    ContractValidationError,
    reject_unknown_fields,
    strict_bool,
    strict_nonempty_string,
    strict_positive_int,
    validate_artifact_ref,
    validate_cohort,
    validate_cohort_set,
    validate_regime,
    validate_schema_header,
    validate_seed,
    validate_tasks,
    validate_version_string,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODEL_CONFIG_FIELDS: frozenset[str] = frozenset(
    {
        "schema",
        "version",
        "model_config_id",
        "model_family",
        "architecture_id",
        "architecture_version",
        "parameter_dtype",
        "architecture_parameters",
        "task_heads",
    }
)

_TASK_HEAD_FIELDS: frozenset[str] = frozenset({"task", "head_id", "head_version", "output_dim"})

_REGIME_ORCHESTRATION_TYPES: dict[str, str] = {
    "R1": "CENTRALIZED",
    "R2A": "FEDAVG",
    "R2B": "FEDADAM",
    "R3": "PERSONALIZED_FL",
    "R4": "STRICT_LOCAL",
    "R5": "LOCAL_ADAPTATION",
}


# ---------------------------------------------------------------------------
# Architecture Validators
# ---------------------------------------------------------------------------


def validate_fixture_linear_v1(params: Any) -> dict[str, Any]:
    """Validate fixture_linear_v1 architecture_parameters."""
    if not isinstance(params, dict):
        raise ContractValidationError("fixture_linear_v1 parameters must be dict")
    reject_unknown_fields(params, {"input_dim", "output_dim", "bias"}, "fixture_linear_v1_params")
    strict_positive_int(params.get("input_dim"), "fixture_linear_v1.input_dim")
    strict_positive_int(params.get("output_dim"), "fixture_linear_v1.output_dim")
    strict_bool(params.get("bias"), "fixture_linear_v1.bias")
    return params


def validate_shared_session_encoder_v1(params: Any) -> dict[str, Any]:
    """Validate the ModelConfig frozen by the Phase-1 architecture checkpoint."""
    if not isinstance(params, dict):
        raise ContractValidationError("shared_session_encoder_v1 parameters must be dict")

    fields = {
        "core",
        "hidden",
        "layers",
        "dropout",
        "history_channels",
        "use_gap",
        "history_length",
        "batch_spec",
        "parameter_count",
        "readout",
        "architecture_selected_by",
    }
    reject_unknown_fields(params, fields, "shared_session_encoder_v1_params")

    if params.get("core") != "gru":
        raise ContractValidationError("shared_session_encoder_v1.core: must be 'gru'")
    strict_positive_int(params.get("hidden"), "shared_session_encoder_v1.hidden")
    strict_positive_int(params.get("layers"), "shared_session_encoder_v1.layers")
    dropout = params.get("dropout")
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
        raise ContractValidationError("shared_session_encoder_v1.dropout: must be numeric")
    if not 0 <= float(dropout) < 1:
        raise ContractValidationError("shared_session_encoder_v1.dropout: must be in [0, 1)")

    channels = params.get("history_channels")
    if channels != ["category_id", "event_type_id"]:
        raise ContractValidationError(
            "shared_session_encoder_v1.history_channels: must equal the frozen channel order"
        )
    strict_bool(params.get("use_gap"), "shared_session_encoder_v1.use_gap")
    strict_positive_int(params.get("history_length"), "shared_session_encoder_v1.history_length")
    if params.get("batch_spec") != "phase1_batch_spec_v1":
        raise ContractValidationError(
            "shared_session_encoder_v1.batch_spec: must be 'phase1_batch_spec_v1'"
        )
    strict_positive_int(params.get("parameter_count"), "shared_session_encoder_v1.parameter_count")
    strict_nonempty_string(params.get("readout"), "shared_session_encoder_v1.readout")
    strict_nonempty_string(
        params.get("architecture_selected_by"),
        "shared_session_encoder_v1.architecture_selected_by",
    )
    return params


_ARCHITECTURE_VALIDATORS = {
    "fixture_linear_v1": validate_fixture_linear_v1,
    "shared_session_encoder_v1": validate_shared_session_encoder_v1,
}

# ---------------------------------------------------------------------------
# ModelConfig v1
# ---------------------------------------------------------------------------


def validate_model_config(record: Any) -> dict[str, Any]:
    """Validate a ModelConfig v1 record."""
    if not isinstance(record, dict):
        raise ContractValidationError("model_config must be a dict")

    reject_unknown_fields(record, _MODEL_CONFIG_FIELDS, "model_config_v1")
    validate_schema_header(record, "model_config_v1")

    strict_nonempty_string(record.get("model_config_id"), "model_config_id")
    strict_nonempty_string(record.get("model_family"), "model_family")
    arch_id = strict_nonempty_string(record.get("architecture_id"), "architecture_id")
    validate_version_string(record.get("architecture_version"), "architecture_version")

    dtype = record.get("parameter_dtype")
    if dtype != "float32":
        raise ContractValidationError(f"parameter_dtype: only float32 supported, got {dtype!r}")

    params = record.get("architecture_parameters")
    if arch_id not in _ARCHITECTURE_VALIDATORS:
        raise ContractValidationError(f"architecture_id: unknown architecture {arch_id!r}")
    _ARCHITECTURE_VALIDATORS[arch_id](params)

    heads = record.get("task_heads")
    if not isinstance(heads, list) or not heads:
        raise ContractValidationError("task_heads: must be a non-empty list")

    tasks_in_heads = []
    for i, head in enumerate(heads):
        if not isinstance(head, dict):
            raise ContractValidationError(f"task_heads[{i}]: must be a dict")
        reject_unknown_fields(head, _TASK_HEAD_FIELDS, f"task_heads[{i}]")
        task = strict_nonempty_string(head.get("task"), f"task_heads[{i}].task")
        strict_nonempty_string(head.get("head_id"), f"task_heads[{i}].head_id")
        validate_version_string(head.get("head_version"), f"task_heads[{i}].head_version")
        strict_positive_int(head.get("output_dim"), f"task_heads[{i}].output_dim")
        tasks_in_heads.append(task)

    # The tasks must be unique and in canonical order.
    # While ModelConfig doesn't explicitly mandate tasks must be the *full* set,
    # the tasks it defines heads for should at least follow canonical rules.
    validate_tasks(tasks_in_heads, "task_heads.tasks")

    return record


# ---------------------------------------------------------------------------
# ExperimentConfig v1
# ---------------------------------------------------------------------------

_EXPERIMENT_CONFIG_FIELDS: frozenset[str] = frozenset(
    {
        "schema",
        "version",
        "config_id",
        "regime",
        "tasks",
        "training_cohort",
        "evaluation_cohorts",
        "seed",
        "source_dataset_ref",
        "canonical_data_contract_ref",
        "cohort_manifest_ref",
        "split_manifest_ref",
        "task_examples_manifest_ref",
        "evaluation_manifest_ref",
        "representation_ref",
        "model_config_ref",
        "objective_config_ref",
        "shared_trainer_core_ref",
        "evaluation_protocol_ref",
        "evaluator_ref",
        "environment_lock_ref",
        "initialization",
        "regime_config",
        "measurement_protocol_ref",
    }
)


def validate_initialization_union(
    init: Any, regime: str, field: str = "initialization"
) -> dict[str, Any]:
    """Validate initialization union rules based on regime."""
    if not isinstance(init, dict):
        raise ContractValidationError(f"{field}: must be a dict")

    kind = init.get("kind")
    if kind not in INITIALIZATION_KINDS:
        raise ContractValidationError(
            f"{field}.kind: must be one of {sorted(INITIALIZATION_KINDS)}"
        )

    if regime in COMMON_INIT_REGIMES:
        if kind != "COMMON_INITIALIZATION":
            raise ContractValidationError(
                f"{field}.kind: must be COMMON_INITIALIZATION for regime {regime}"
            )
        reject_unknown_fields(init, {"kind", "common_initialization_ref"}, f"{field}")
        validate_artifact_ref(
            init.get("common_initialization_ref"), f"{field}.common_initialization_ref"
        )

    elif regime in PRETRAINED_INIT_REGIMES:
        if kind != "PRETRAINED_R1_CHECKPOINT":
            raise ContractValidationError(
                f"{field}.kind: must be PRETRAINED_R1_CHECKPOINT for regime {regime}"
            )
        reject_unknown_fields(
            init, {"kind", "pretrained_checkpoint_ref", "parent_run_id"}, f"{field}"
        )
        validate_artifact_ref(
            init.get("pretrained_checkpoint_ref"), f"{field}.pretrained_checkpoint_ref"
        )
        strict_nonempty_string(init.get("parent_run_id"), f"{field}.parent_run_id")

    return init


def validate_experiment_config(record: Any) -> dict[str, Any]:
    """Validate an ExperimentConfig v1 record."""
    if not isinstance(record, dict):
        raise ContractValidationError("experiment_config must be a dict")

    reject_unknown_fields(record, _EXPERIMENT_CONFIG_FIELDS, "experiment_config_v1")
    validate_schema_header(record, "experiment_config_v1")

    strict_nonempty_string(record.get("config_id"), "config_id")
    regime = validate_regime(record.get("regime"), "regime")
    validate_tasks(record.get("tasks"), "tasks")
    validate_cohort(record.get("training_cohort"), "training_cohort")
    validate_cohort_set(record.get("evaluation_cohorts"), "evaluation_cohorts")
    validate_seed(record.get("seed"))

    # Required refs
    required_refs = [
        "source_dataset_ref",
        "canonical_data_contract_ref",
        "cohort_manifest_ref",
        "split_manifest_ref",
        "task_examples_manifest_ref",
        "evaluation_manifest_ref",
        "representation_ref",
        "model_config_ref",
        "objective_config_ref",
        "shared_trainer_core_ref",
        "evaluation_protocol_ref",
        "evaluator_ref",
        "environment_lock_ref",
    ]
    for ref_field in required_refs:
        validate_artifact_ref(record.get(ref_field), ref_field)

    # Optional measurement_protocol_ref
    if "measurement_protocol_ref" in record:
        validate_artifact_ref(record["measurement_protocol_ref"], "measurement_protocol_ref")

    validate_initialization_union(record.get("initialization"), regime)

    regime_config = record.get("regime_config")
    if not isinstance(regime_config, dict):
        raise ContractValidationError("regime_config: must be a dict")

    return record


# ---------------------------------------------------------------------------
# MetricRecord v1
# ---------------------------------------------------------------------------

_METRIC_RECORD_FIELDS = frozenset(
    {
        "schema",
        "version",
        "metric_id",
        "task",
        "cohort",
        "value",
        "direction",
        "unit",
        "support",
        "support_null_reason",
    }
)


def validate_metric_record(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ContractValidationError("metric_record must be dict")
    reject_unknown_fields(record, _METRIC_RECORD_FIELDS, "metric_record_v1")
    validate_schema_header(record, "metric_record_v1")

    strict_nonempty_string(record.get("metric_id"), "metric_id")
    strict_nonempty_string(record.get("task"), "task")
    strict_nonempty_string(record.get("cohort"), "cohort")

    val = record.get("value")
    if not isinstance(val, (int, float)) or isinstance(val, bool):
        raise ContractValidationError("value must be numeric")

    from scripts.experiments.contracts import validate_metric_direction, validate_metric_unit

    validate_metric_direction(record.get("direction"))
    validate_metric_unit(record.get("unit"))

    has_support = "support" in record
    has_reason = "support_null_reason" in record
    if has_support == has_reason:
        raise ContractValidationError("Must have exactly one of support or support_null_reason")
    if has_support:
        from scripts.experiments.contracts import strict_nonneg_int

        strict_nonneg_int(record["support"], "support")
    else:
        strict_nonempty_string(record["support_null_reason"], "support_null_reason")

    return record


# ---------------------------------------------------------------------------
# SystemMeasurementReferenceSet v1
# ---------------------------------------------------------------------------

_SYS_MEASUREMENT_FIELDS = frozenset({"schema", "version", "status", "refs", "null_reason"})


def validate_system_measurement_reference_set(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ContractValidationError("system_measurement_reference_set must be dict")
    reject_unknown_fields(record, _SYS_MEASUREMENT_FIELDS, "system_measurement_reference_set_v1")
    validate_schema_header(record, "system_measurement_reference_set_v1")

    from scripts.experiments.contracts import MEASUREMENT_STATUSES

    status = record.get("status")
    if status not in MEASUREMENT_STATUSES:
        raise ContractValidationError(f"status must be in {sorted(MEASUREMENT_STATUSES)}")

    if status == "AVAILABLE":
        if "null_reason" in record:
            raise ContractValidationError("AVAILABLE status cannot have null_reason")
        refs = record.get("refs")
        if not isinstance(refs, list):
            raise ContractValidationError("AVAILABLE status must have refs list")
        for i, ref in enumerate(refs):
            validate_artifact_ref(ref, f"refs[{i}]")
    else:
        if "refs" in record:
            raise ContractValidationError(f"{status} status cannot have refs")
        strict_nonempty_string(record.get("null_reason"), "null_reason")

    return record


# ---------------------------------------------------------------------------
# ExperimentResult v1
# ---------------------------------------------------------------------------

_EXPERIMENT_RESULT_FIELDS = frozenset(
    {
        "schema",
        "version",
        "run_id",
        "attempt",
        "state",
        "regime",
        "tasks",
        "training_cohort",
        "evaluation_cohorts",
        "seed",
        "config_ref",
        "compatibility",
        "git_sha",
        "environment_lock_ref",
        "initialization",
        "started_at_utc",
        "ended_at_utc",
        "resume_count",
        "checkpoint_ref",
        "metrics",
        "federated_metadata",
        "artifacts",
        "system_measurements",
        "failure",
    }
)


def validate_experiment_result(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ContractValidationError("experiment_result must be dict")
    reject_unknown_fields(record, _EXPERIMENT_RESULT_FIELDS, "experiment_result_v1")
    validate_schema_header(record, "experiment_result_v1")

    from scripts.experiments.compatibility import validate_compatibility_tuple
    from scripts.experiments.contracts import (
        parse_run_id,
        validate_run_id_config_prefix,
        validate_run_state,
        validate_utc_timestamp,
    )

    strict_nonempty_string(record.get("run_id"), "run_id")
    attempt = strict_positive_int(record.get("attempt"), "attempt")
    state = validate_run_state(record.get("state"))
    regime = validate_regime(record.get("regime"))
    validate_tasks(record.get("tasks"))
    validate_cohort(record.get("training_cohort"))
    validate_cohort_set(record.get("evaluation_cohorts"))
    validate_seed(record.get("seed"))

    # parse run_id consistency
    parsed = parse_run_id(record["run_id"])
    if (
        parsed["regime"] != regime
        or parsed["attempt"] != attempt
        or parsed["seed"] != record["seed"]
    ):
        raise ContractValidationError("run_id components do not match result fields")

    validate_artifact_ref(record.get("config_ref"), "config_ref")
    validate_run_id_config_prefix(record["run_id"], record["config_ref"]["sha256"])

    validate_compatibility_tuple(record.get("compatibility"))
    strict_nonempty_string(record.get("git_sha"), "git_sha")
    validate_artifact_ref(record.get("environment_lock_ref"), "environment_lock_ref")

    validate_initialization_union(record.get("initialization"), regime)

    # Timestamps & state rules
    start = record.get("started_at_utc")
    end = record.get("ended_at_utc")
    if state == "PLANNED":
        if start or end:
            raise ContractValidationError("PLANNED cannot have started_at_utc or ended_at_utc")
    elif state == "RUNNING":
        validate_utc_timestamp(start, "started_at_utc")
        if end:
            raise ContractValidationError("RUNNING cannot have ended_at_utc")
    else:  # INTERRUPTED or TERMINAL
        validate_utc_timestamp(start, "started_at_utc")
        validate_utc_timestamp(end, "ended_at_utc")

    if state == "FAILED":
        fail = record.get("failure")
        if not isinstance(fail, dict):
            raise ContractValidationError("FAILED requires failure details object")
    else:
        if "failure" in record:
            raise ContractValidationError(f"State {state} forbids failure details")

    if "resume_count" in record:
        from scripts.experiments.contracts import strict_nonneg_int

        strict_nonneg_int(record["resume_count"], "resume_count")
    if "checkpoint_ref" in record:
        validate_artifact_ref(record["checkpoint_ref"], "checkpoint_ref")

    metrics = record.get("metrics")
    if not isinstance(metrics, list):
        raise ContractValidationError("metrics must be a list")
    for i, m in enumerate(metrics):
        validate_metric_record(m)

    artifacts = record.get("artifacts")
    if not isinstance(artifacts, list):
        raise ContractValidationError("artifacts must be a list")
    for i, a in enumerate(artifacts):
        validate_artifact_ref(a, f"artifacts[{i}]")

    validate_system_measurement_reference_set(record.get("system_measurements"))

    if "federated_metadata" in record and not isinstance(record["federated_metadata"], dict):
        raise ContractValidationError("federated_metadata must be dict")

    return record


# ---------------------------------------------------------------------------
# CommonInitialization v1
# ---------------------------------------------------------------------------

_COMMON_INITIALIZATION_FIELDS = frozenset(
    {"schema", "version", "regime", "tasks", "model_config_ref", "seed", "initializer_git_sha"}
)


def validate_common_initialization(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ContractValidationError("common_initialization must be dict")
    reject_unknown_fields(record, _COMMON_INITIALIZATION_FIELDS, "common_initialization_v1")
    validate_schema_header(record, "common_initialization_v1")

    validate_regime(record.get("regime"))
    validate_tasks(record.get("tasks"))
    validate_artifact_ref(record.get("model_config_ref"), "model_config_ref")
    validate_seed(record.get("seed"))
    strict_nonempty_string(record.get("initializer_git_sha"), "initializer_git_sha")

    return record


# ---------------------------------------------------------------------------
# RegimeCatalog v1
# ---------------------------------------------------------------------------


def validate_regime_catalog(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ContractValidationError("regime_catalog must be dict")
    reject_unknown_fields(record, {"schema", "version", "regimes"}, "regime_catalog_v1")
    validate_schema_header(record, "regime_catalog_v1")

    regimes = record.get("regimes")
    if not isinstance(regimes, dict):
        raise ContractValidationError("regimes must be dict")

    expected_ids = set(_REGIME_ORCHESTRATION_TYPES)
    actual_ids = set(regimes)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        unknown = sorted(actual_ids - expected_ids)
        raise ContractValidationError(
            f"regimes must contain the frozen v1 set: missing={missing}, unknown={unknown}"
        )

    for regime_id, expected_orchestration in _REGIME_ORCHESTRATION_TYPES.items():
        regime = regimes[regime_id]
        if not isinstance(regime, dict):
            raise ContractValidationError(f"regimes[{regime_id}] must be dict")
        reject_unknown_fields(
            regime,
            {"description", "orchestration_type"},
            f"regimes[{regime_id}]",
        )
        strict_nonempty_string(regime.get("description"), f"regimes[{regime_id}].description")
        orchestration = strict_nonempty_string(
            regime.get("orchestration_type"),
            f"regimes[{regime_id}].orchestration_type",
        )
        if orchestration != expected_orchestration:
            raise ContractValidationError(
                f"regimes[{regime_id}].orchestration_type must be "
                f"{expected_orchestration}, got {orchestration}"
            )
    return record


# ---------------------------------------------------------------------------
# ExperimentArtifactMap v1
# ---------------------------------------------------------------------------


def validate_experiment_artifact_map(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ContractValidationError("experiment_artifact_map must be dict")
    reject_unknown_fields(record, {"schema", "version", "artifacts"}, "experiment_artifact_map_v1")
    validate_schema_header(record, "experiment_artifact_map_v1")

    artifacts = record.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ContractValidationError("artifacts must be dict")
    return record


# ---------------------------------------------------------------------------
# ExperimentContractValidationSummary v1
# ---------------------------------------------------------------------------


def validate_experiment_contract_validation_summary(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ContractValidationError("experiment_contract_validation_summary must be dict")
    reject_unknown_fields(
        record,
        {"schema", "version", "status", "timestamp_utc", "validated_schemas", "validated_fixtures"},
        "experiment_contract_validation_summary_v1",
    )
    validate_schema_header(record, "experiment_contract_validation_summary_v1")

    status = record.get("status")
    if status not in {"PASS", "FAIL"}:
        raise ContractValidationError(f"status must be PASS or FAIL, got {status}")

    from scripts.experiments.contracts import validate_utc_timestamp

    validate_utc_timestamp(record.get("timestamp_utc"), "timestamp_utc")

    if not isinstance(record.get("validated_schemas"), list):
        raise ContractValidationError("validated_schemas must be list")
    if not isinstance(record.get("validated_fixtures"), list):
        raise ContractValidationError("validated_fixtures must be list")

    return record


# ---------------------------------------------------------------------------
# ExperimentContractArtifactManifest v1
# ---------------------------------------------------------------------------


def validate_experiment_contract_artifact_manifest(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ContractValidationError("experiment_contract_artifact_manifest must be dict")
    reject_unknown_fields(
        record, {"schema", "version", "artifacts"}, "experiment_contract_artifact_manifest_v1"
    )
    validate_schema_header(record, "experiment_contract_artifact_manifest_v1")

    artifacts = record.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ContractValidationError("artifacts must be dict")

    from scripts.experiments.contracts import validate_sha256

    for k, v in artifacts.items():
        if not isinstance(v, dict):
            raise ContractValidationError(f"artifacts[{k}] must be dict")
        reject_unknown_fields(v, {"path", "sha256"}, f"artifacts[{k}]")
        strict_nonempty_string(v.get("path"), f"artifacts[{k}].path")
        validate_sha256(v.get("sha256"), f"artifacts[{k}].sha256")

    return record


# ---------------------------------------------------------------------------
# ExperimentContractFreeze v1
# ---------------------------------------------------------------------------


def validate_experiment_contract_freeze(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ContractValidationError("experiment_contract_freeze must be dict")
    validate_schema_header(record, "experiment_contract_freeze_v1")
    return record
