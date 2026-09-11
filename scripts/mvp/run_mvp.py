"""The scoped T1 MVP stage CLI, and the Flower ClientApp/ServerApp it runs.

One command, one stage, one owner for each artifact:

``preflight``    freeze the executing source and create the single common initialization
``baselines``    score the three frozen #32 count baselines over all VALIDATION decisions
``canary``       mechanical checks in a throwaway namespace, never a scientific result
``centralized``  replay the private schedule through one persistent trainer core
``federated``    the same schedule through real Flower FedAvg with measured payload bytes
``compare``      the matched-pair comparison and the isolated quality-retention attempt
``verify``       adversarial re-checks of everything the other stages claimed
``report``       assemble the report/notebook inputs from verified files only
``all``          the mandatory sequence above, and nothing it did not actually execute

What this file is not: a training framework, a second evaluator, or a place where policy
lives. Every size, seed and lifecycle comes from the execution policy passed in with
``--config``, which is itself frozen into the run's source snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import time
from dataclasses import replace as dataclass_replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ppsi.baselines.t1_simple import T1CountBaselines
from ppsi.evaluation.t1 import (
    evaluate_t1_ranks,
    metric_records_from_t1_summary,
    rank_targets_from_scores,
)
from ppsi.federated.communication import CommunicationLedger
from ppsi.federated.mvp_data import load_prepared
from ppsi.federated.mvp_oracle import (
    ORACLE_POLICY_ID,
    check_aggregation,
)
from ppsi.federated.mvp_runner import (
    build_trainer,
    client_workload,
    evaluate_full_validation,
    headline_metric,
    model_identity,
    reset_client_stream,
    set_deterministic_execution,
    workload_batches,
)
from ppsi.federated.mvp_support import (
    exposure_digest,
    metric_delta,
    raw_file_sha256,
    require_complete_replies,
    require_finite_metric,
)
from ppsi.models.session_gru import common_initialization, parameter_count
from ppsi.training.identity import file_sha256
from ppsi.training.result import build_experiment_result
from ppsi.training.state import pack_shared_state, shared_state_digest
from scripts.experiments.results import (
    build_quality_report,
    load_results,
    validate_result_for_reporting,
)
from scripts.experiments.schemas import validate_experiment_config

BRANCH = "s2-pr-mvp-integration"

# Every file whose bytes can change what the pilot computes. Frozen before execution.
SOURCE_FILES = (
    "ppsi/baselines/t1_simple.py",
    "ppsi/data/batching.py",
    "ppsi/data/rees46.py",
    "ppsi/data/sequences.py",
    "ppsi/evaluation/t1.py",
    "ppsi/federated/clients.py",
    "ppsi/federated/communication.py",
    "ppsi/federated/mvp_data.py",
    "ppsi/federated/mvp_oracle.py",
    "ppsi/federated/mvp_runner.py",
    "ppsi/federated/mvp_support.py",
    "ppsi/federated/sampling.py",
    "ppsi/models/batch_spec.py",
    "ppsi/models/session_gru.py",
    "ppsi/training/batch.py",
    "ppsi/training/checkpoint.py",
    "ppsi/training/core.py",
    "ppsi/training/flower.py",
    "ppsi/training/identity.py",
    "ppsi/training/result.py",
    "ppsi/training/state.py",
    "ppsi/training/t1_mvp_objective.py",
    "scripts/experiments/compatibility.py",
    "scripts/experiments/results.py",
    "scripts/experiments/schemas.py",
    "scripts/federated/fl_real_smoke.py",
    "scripts/mvp/prepare_mvp.py",
    "scripts/mvp/run_mvp.py",
    "uv.lock",
)

BASELINE_VARIANTS = ("popularity", "markov", "last_category")


class StageError(RuntimeError):
    """A stage refused to continue; the reason is the message, never a silent pass."""


# ---------------------------------------------------------------------------
# Small shared helpers, in the pattern the baseline lane already uses
# ---------------------------------------------------------------------------


def now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def read_json(rel: str | Path) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def write_json(rel: str | Path, data: dict, *, replace: bool = False) -> Path:
    """Write canonical JSON atomically, refusing to silently change a published file."""
    path = ROOT / rel
    payload = json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        if path.read_text(encoding="utf-8") != payload:
            raise StageError(f"refusing to overwrite an existing artifact in place: {rel}")
        return path
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    temporary.replace(path)
    return path


def ref(rel: str | Path, schema: str, logical: str | None = None) -> dict:
    path = ROOT / rel
    if not path.is_file():
        raise FileNotFoundError(str(rel))
    return {
        "schema": "artifact_ref_v1",
        "version": "1",
        "logical_id": logical or schema,
        "artifact_schema": schema,
        "artifact_version": "1",
        "uri": str(rel).replace("\\", "/"),
        "sha256": file_sha256(path),
    }


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def require_branch() -> str:
    branch = git("branch", "--show-current")
    if branch != BRANCH:
        raise StageError(
            f"wrong branch: on {branch!r}, this pilot executes on {BRANCH!r}. "
            "Switching branches is the human's action, not this script's."
        )
    return branch


def source_snapshot(config_rel: str) -> dict:
    """The executing source, by content. Uncommitted implementation is declared as such.

    The active execution policy is frozen alongside the code, because it is behaviour
    defining: a different policy file is a different experiment.
    """
    status = git("status", "--porcelain")
    frozen_files = (*SOURCE_FILES, str(config_rel).replace("\\", "/"))
    return {
        "schema": "mvp_source_snapshot_v1",
        "version": "1",
        "git_head": git("rev-parse", "HEAD"),
        "git_branch": git("branch", "--show-current"),
        "execution_provenance": "WORKING_TREE_RUN_WITH_EXPLICIT_SOURCE_HASHES",
        "working_tree_clean": status == "",
        "uncommitted_paths": sorted(line[3:] for line in status.splitlines()) if status else [],
        "files_sha256": {rel: file_sha256(ROOT / rel) for rel in sorted(frozen_files)},
        "libraries": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "torch", "flwr", "ray", "pyarrow", "polars", "psutil")
        },
        "python": sys.version.split()[0],
    }


def logical_digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


SOURCE_FREEZE_NAME = "source_snapshot.v1.json"


def freeze_source(ctx: Context) -> dict:
    """Establish, or re-verify, the frozen behavior-defining source for this run id.

    The first ``preflight`` of a run id writes this file. Every later invocation compares
    the working tree against it, so a stage can never execute source that differs from the
    source the run claims to have executed. The remedy for a real change is a new run id,
    never an edited snapshot.
    """
    path = ROOT / ctx.public_path(SOURCE_FREEZE_NAME)
    current = source_snapshot(ctx.config_rel)
    if not path.is_file():
        current["frozen_for_run"] = ctx.run
        current["freeze_status"] = "FROZEN_BEFORE_SCIENTIFIC_EXECUTION"
        write_json(ctx.public_path(SOURCE_FREEZE_NAME), current)
        log(f"source frozen: {len(current['files_sha256'])} behaviour-defining files")
        return current
    verify_frozen_source(ctx, stage="freeze")
    return read_json(ctx.public_path(SOURCE_FREEZE_NAME))


def verify_frozen_source(ctx: Context, *, stage: str) -> dict:
    """Fail closed if any behaviour-defining file differs from the frozen snapshot."""
    path = ROOT / ctx.public_path(SOURCE_FREEZE_NAME)
    if not path.is_file():
        raise StageError(
            f"{stage} cannot run before the source freeze exists; run the preflight stage, "
            "which establishes it once for this run id"
        )
    frozen = read_json(ctx.public_path(SOURCE_FREEZE_NAME))["files_sha256"]
    expected = (*SOURCE_FILES, str(ctx.config_rel).replace("\\", "/"))
    if sorted(frozen) != sorted(expected):
        raise StageError(
            "the frozen snapshot covers a different file set from this code; that is a "
            "source change and needs a new run id, not an edited snapshot"
        )
    drifted = sorted(rel for rel in expected if file_sha256(ROOT / rel) != frozen[rel])
    if drifted:
        raise StageError(
            f"SOURCE_DRIFT before {stage}: {drifted}. Behaviour-defining source changed after "
            f"the freeze for {ctx.run}. Start a new run id; never patch the recorded hashes."
        )
    return {"stage": stage, "files_checked": len(SOURCE_FILES), "drift": []}


class Context:
    """Paths, policy and prepared inputs resolved once per invocation."""

    def __init__(self, config_rel: str, run: str) -> None:
        self.policy = read_json(config_rel)
        if self.policy.get("schema") != "mvp_execution_policy_v1":
            raise StageError("unexpected execution policy schema")
        if run != self.policy["run_family"]:
            raise StageError("run id does not match the declared run family")
        self.config_rel = str(config_rel).replace("\\", "/")
        self.run = run
        self.public = Path(self.policy["outputs"]["public"])
        self.private = Path(self.policy["outputs"]["private"])
        self.baseline_out = Path(self.policy["outputs"]["baseline_results"])
        self.pair_out = Path(self.policy["outputs"]["pilot_pair_registry"])
        self.prepared_dir = ROOT / self.private / "prepared"
        self.seed = int(self.policy["pilot"]["seed"])
        self._prepared = None

    @property
    def prepared(self):
        if self._prepared is None:
            if not self.prepared_dir.is_dir():
                raise StageError(
                    "prepared inputs are missing; run scripts/mvp/prepare_mvp.py first"
                )
            self._prepared = load_prepared(self.prepared_dir)
        return self._prepared

    def public_path(self, name: str) -> str:
        return str(self.public / name).replace("\\", "/")

    def private_path(self, name: str) -> Path:
        path = ROOT / self.private / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def check_resources(self, *, stage: str) -> dict:
        resources = self.policy["resources"]
        available = psutil.virtual_memory().available
        floor = resources["min_available_during_run_gib"] * 1024**3
        if available < floor:
            raise StageError(
                f"CAPACITY_BLOCKED before {stage}: {available / 1024**3:.2f} GiB available"
            )
        return {
            "available_ram_bytes": int(available),
            "total_ram_bytes": int(psutil.virtual_memory().total),
            "logical_cpus": psutil.cpu_count(logical=True),
        }


# ---------------------------------------------------------------------------
# Stage: preflight
# ---------------------------------------------------------------------------


def stage_preflight(ctx: Context) -> dict:
    """Freeze the executing source and create the one common initialization."""
    branch = require_branch()
    frozen = freeze_source(ctx)
    prepared = ctx.prepared
    identity = model_identity(ctx.policy)
    set_deterministic_execution(ctx.policy, seed=ctx.seed)

    state, model_lane_digest = common_initialization(
        ctx.seed, config=identity.config, batch_spec=identity.spec
    )
    init_path = ctx.private_path("common_initialization/common_init_seed13.pt")
    if init_path.exists():
        existing = torch.load(init_path, map_location="cpu", weights_only=True)
        if sorted(existing) != sorted(state) or any(
            not torch.equal(existing[k], state[k]) for k in state
        ):
            raise StageError(
                "an existing common initialization differs from the one this policy builds; "
                "a changed architecture needs a new run id, not an overwritten artifact"
            )
    else:
        temporary = init_path.with_suffix(".pt.tmp")
        torch.save(state, temporary)
        temporary.replace(init_path)

    model, _ = build_trainer(identity, ctx.policy, state)
    packed = pack_shared_state(model, model.shared_state_spec())
    record = {
        "schema": "mvp_preflight_v1",
        "version": "1",
        "run": ctx.run,
        "scope": ctx.policy["scope"],
        "branch": branch,
        "generated_at_utc": now(),
        "model": identity.to_dict(),
        "parameter_count": parameter_count(model),
        "common_initialization": {
            "role": "FRESH_UNTRAINED_ARTIFACT_LOADED_BY_BOTH_REGIMES",
            "private_uri": str(init_path.relative_to(ROOT)).replace("\\", "/"),
            "raw_file_sha256": raw_file_sha256(init_path),
            "model_lane_state_digest": model_lane_digest,
            "training_state_codec_digest": shared_state_digest(packed),
            "digest_conventions_are_distinct": True,
            "not_a_trained_checkpoint": True,
        },
        "prepared": {
            "data_manifest_sha256": prepared.data_manifest_sha256,
            "train_rows": prepared.train.rows,
            "validation_rows": prepared.validation.rows,
            "population": len(prepared.population),
            "rounds": len(prepared.schedule),
        },
        "resources": ctx.check_resources(stage="preflight"),
        "source_freeze": {
            "uri": ctx.public_path(SOURCE_FREEZE_NAME),
            "files_frozen": len(frozen["files_sha256"]),
            "git_head": frozen["git_head"],
            "status": frozen["freeze_status"],
            "drift_policy": "A later behaviour-defining change requires a new run id.",
        },
    }
    write_json(ctx.public_path("preflight.v1.json"), record, replace=True)
    log(f"preflight complete: {record['parameter_count']:,} parameters")
    return record


# ---------------------------------------------------------------------------
# Stage: baselines
# ---------------------------------------------------------------------------


def _load_count_tables(ctx: Context, scope: str) -> T1CountBaselines:
    payload = np.load(ctx.prepared_dir / f"count_tables_{scope}.npz", allow_pickle=False)
    return T1CountBaselines(
        target_counts=payload["target_counts"],
        transition_counts=payload["transition_counts"],
        train_decisions=int(payload["train_decisions"]),
    )


def _score_baseline(
    tables: T1CountBaselines, current: np.ndarray, targets: np.ndarray, *, chunk: int
) -> dict[str, np.ndarray]:
    """Raw ranks for all three variants over one frozen split, in decision order."""
    total = len(current)
    ranks = {variant: np.empty(total, dtype=np.int64) for variant in BASELINE_VARIANTS}
    for start in range(0, total, chunk):
        stop = min(start + chunk, total)
        for variant in BASELINE_VARIANTS:
            scores = tables.scores(variant, current[start:stop])
            chunk_ranks = rank_targets_from_scores(
                torch.from_numpy(scores), torch.from_numpy(targets[start:stop]), category_count=588
            )
            ranks[variant][start:stop] = chunk_ranks.to(torch.int64).numpy()
    return ranks


def _baseline_config(
    ctx: Context, *, variant: str, scope: str, directory: str, init_ref: dict
) -> dict:
    protocol_ref = ref(
        "docs/evidence/s1-ds-05-06/data_protocol_v1.proposed.json", "data_protocol_v1"
    )
    prepare_ref = ref(ctx.public_path("prepare_summary.v1.json"), "mvp_prepare_summary_v1")
    cfg = {
        "schema": "experiment_config_v1",
        "version": "1",
        "config_id": f"{ctx.run}_{scope}_{variant}",
        "regime": "R1",
        "tasks": ["T1"],
        "training_cohort": "C1",
        "evaluation_cohorts": ["C1"],
        "seed": ctx.seed,
        "source_dataset_ref": prepare_ref,
        "canonical_data_contract_ref": protocol_ref,
        "cohort_manifest_ref": ref(
            "data/protocol/INTERNAL_DO_NOT_UPLOAD_cohort_manifest_v1.proposed.parquet",
            "cohort_manifest_v1",
        ),
        "split_manifest_ref": protocol_ref,
        "task_examples_manifest_ref": prepare_ref,
        "evaluation_manifest_ref": ref(
            "data/examples/INTERNAL_DO_NOT_UPLOAD_task_examples_t1_v1.proposed.parquet",
            "task_example_v1",
        ),
        "representation_ref": ref(ctx.config_rel, "mvp_execution_policy_v1"),
        "model_config_ref": ref(ctx.config_rel, "mvp_execution_policy_v1"),
        "objective_config_ref": ref(ctx.config_rel, "mvp_execution_policy_v1"),
        "shared_trainer_core_ref": ref("ppsi/baselines/t1_simple.py", "python_source_v1"),
        "evaluation_protocol_ref": ref(
            "docs/decisions/mvp-execution.md", "mvp_execution_decision_v1"
        ),
        "evaluator_ref": ref("ppsi/evaluation/t1.py", "python_source_v1"),
        "environment_lock_ref": ref("uv.lock", "uv_lock_v1"),
        "initialization": {
            "kind": "COMMON_INITIALIZATION",
            "common_initialization_ref": init_ref,
        },
        "regime_config": {
            "orchestration_type": "CENTRALIZED",
            "result_role": "MVP_T1_COUNT_BASELINE_NOT_FINAL_R1",
            "evaluation_split": "VALIDATION",
            "model": variant,
            "fit_scope": scope,
            "fitted_on": "TRAIN_ONLY",
            "search": "NOT_APPLICABLE_FIXED_PREDECLARED_RULES",
            "score_convention": "RAW_NO_SUPPRESSION",
            "final_r1_denominator": False,
            "quality_retention_claim": False,
            "neural_trainer_used": False,
        },
    }
    validate_experiment_config(cfg)
    return cfg


def stage_baselines(ctx: Context) -> dict:
    """The three frozen count baselines, scored over every VALIDATION decision."""
    require_branch()
    started = now()
    prepared = ctx.prepared
    meta = prepared.validation_metadata()
    current = np.load(
        ctx.prepared_dir / "validation_current_category.npy", allow_pickle=False
    ).astype(np.int64)
    targets = np.asarray(prepared.validation.arrays["target"], dtype=np.int64)
    if len(current) != prepared.validation.rows:
        raise StageError("validation current-category column does not cover the prepared rows")

    verify_frozen_source(ctx, stage="baselines")
    snapshot_ref = ref(ctx.public_path(SOURCE_FREEZE_NAME), "mvp_source_snapshot_v1")
    init_ref = ref(
        ctx.public_path("preflight.v1.json"),
        "mvp_preflight_v1",
        "common_initialization_declaration",
    )
    chunk = int(ctx.policy["pilot"]["evaluation_batch_size"])

    published = []
    for scope in ("full", "pilot"):
        tables = _load_count_tables(ctx, scope)
        log(f"scoring {scope}-scope baselines over {len(current):,} decisions")
        ranks = _score_baseline(tables, current, targets, chunk=chunk)
        for variant in BASELINE_VARIANTS:
            summary = evaluate_t1_ranks(
                ranks[variant],
                meta["category_changed"],
                meta["client_ids"],
                train_history_counts=meta["train_history_counts"],
                mrr_cutoff=20,
            )
            summary_dict = summary.to_dict()
            directory = str(ctx.baseline_out / f"{scope}-{variant}").replace("\\", "/")
            metrics_path = f"{directory}/t1_metrics.v1.json"
            write_json(
                metrics_path,
                {
                    "schema": "mvp_t1_baseline_metrics_v1",
                    "version": "1",
                    "run": ctx.run,
                    "variant": variant,
                    "fit_scope": scope,
                    "train_decisions": tables.train_decisions,
                    "count_tables_content_sha256": tables.content_sha256(),
                    "score_convention": "RAW_NO_SUPPRESSION",
                    "evaluation": summary_dict,
                    "headline": headline_metric(summary_dict),
                    "diagnostic_note": (
                        "overall slice and micro averages are diagnostics; the headline is "
                        "t1.next_distinct.mrr_at_20.macro"
                    ),
                },
                replace=True,
            )
            cfg = _baseline_config(
                ctx, variant=variant, scope=scope, directory=directory, init_ref=init_ref
            )
            cfg_path = f"{directory}/experiment_config.v1.json"
            write_json(cfg_path, cfg, replace=True)
            result = build_experiment_result(
                experiment_config=cfg,
                config_ref=ref(cfg_path, "experiment_config_v1"),
                git_sha=git("rev-parse", "HEAD"),
                state="SUCCEEDED",
                attempt=1,
                started_at_utc=started,
                ended_at_utc=now(),
                metrics=metric_records_from_t1_summary(summary),
                artifacts=[ref(metrics_path, "mvp_t1_baseline_metrics_v1"), snapshot_ref],
                system_measurements={
                    "schema": "system_measurement_reference_set_v1",
                    "version": "1",
                    "status": "NOT_APPLICABLE",
                    "null_reason": "count baseline scoring performs no training or communication",
                },
            )
            result_path = f"{directory}/experiment_result.v1.json"
            validate_result_for_reporting(result, source=Path(result_path).name)
            write_json(result_path, result, replace=True)
            published.append(
                {
                    "variant": variant,
                    "fit_scope": scope,
                    "run_id": result["run_id"],
                    "headline": headline_metric(summary_dict),
                    "result_ref": ref(result_path, "experiment_result_v1"),
                    "metrics_ref": ref(metrics_path, "mvp_t1_baseline_metrics_v1"),
                }
            )
            log(
                f"  {scope}/{variant}: macro MRR@20 "
                f"{published[-1]['headline']['value']:.4f} over "
                f"{published[-1]['headline']['support_clients']:,} clients"
            )
        del ranks

    report = {
        "schema": "mvp_baseline_summary_v1",
        "version": "1",
        "run": ctx.run,
        "generated_at_utc": now(),
        "validation_decisions": prepared.validation.rows,
        "history_count_basis": meta["history_count_basis"],
        "variants": published,
        "scope_note": (
            "full = fitted on all 3,113,814 frozen T1 TRAIN decisions; pilot = fitted on the "
            "1,000-client pilot TRAIN decisions only. The two are never mixed in one row."
        ),
        "source_snapshot_ref": snapshot_ref,
    }
    write_json(ctx.public_path("baseline_summary.v1.json"), report, replace=True)
    return report


# ---------------------------------------------------------------------------
# Shared pieces of both training regimes
# ---------------------------------------------------------------------------


def _round_plan(ctx: Context) -> list[dict]:
    """The precomputed schedule, checked against the policy it claims to follow."""
    schedule = ctx.prepared.schedule
    expected_rounds = int(ctx.policy["pilot"]["rounds"])
    per_round = int(ctx.policy["pilot"]["clients_per_round"])
    if len(schedule) != expected_rounds:
        raise StageError("the prepared schedule does not have the declared number of rounds")
    for entry in schedule:
        if len(entry["selected_client_ids"]) != per_round:
            raise StageError("a scheduled round does not have the declared participant count")
    return schedule


def _exposure_records(ctx: Context) -> tuple[list[tuple[int, str, list[str]]], dict]:
    """The (round, client, ordered decision keys) triples both regimes must share."""
    batch_size = int(ctx.policy["pilot"]["batch_size"])
    records: list[tuple[int, str, list[str]]] = []
    rows_per_round: dict[int, int] = {}
    for entry in _round_plan(ctx):
        server_round = int(entry["server_round"])
        total = 0
        for client_id in entry["selected_client_ids"]:
            workload = client_workload(
                ctx.prepared,
                client_id,
                server_round=server_round,
                seed=ctx.seed,
                batch_size=batch_size,
            )
            records.append((server_round, client_id, workload.decision_keys))
            total += workload.rows
        rows_per_round[server_round] = total
    stats = {
        "rounds": len(rows_per_round),
        "participations": len(records),
        "unique_clients": len({client for _, client, _ in records}),
        "total_rows": sum(rows_per_round.values()),
        "rows_per_round": rows_per_round,
    }
    return records, stats


def _evaluation_rounds(ctx: Context) -> list[int]:
    return [int(r) for r in ctx.policy["pilot"]["full_validation_at_rounds"]]


def _evaluate_and_record(ctx: Context, model, server_round: int, regime: str) -> dict:
    """One full frozen VALIDATION pass, stored privately with its raw ranks."""
    started = time.perf_counter()
    summary, ranks = evaluate_full_validation(
        model,
        ctx.prepared,
        model_identity(ctx.policy).spec,
        batch_size=int(ctx.policy["pilot"]["evaluation_batch_size"]),
        category_count=int(ctx.policy["model"]["category_count"]),
    )
    path = ctx.private_path(f"{regime}/ranks_round_{server_round:02d}.npy")
    np.save(path, ranks, allow_pickle=False)
    return {
        "server_round": server_round,
        "seconds": round(time.perf_counter() - started, 2),
        "evaluation": summary,
        "headline": headline_metric(summary),
        "ranks_private_uri": str(path.relative_to(ROOT)).replace("\\", "/"),
        "ranks_sha256": raw_file_sha256(path),
    }


# ---------------------------------------------------------------------------
# Stage: canary
# ---------------------------------------------------------------------------


def dropout_free_identity(identity):
    """The same architecture with dropout switched off, for one toy agreement fixture.

    ``dataclasses.replace``, not ``copy.replace``: this runs on Python 3.11, where
    ``copy.replace`` does not exist. Only this throwaway fixture is dropout-free; the
    measured pilot keeps the dropout the policy declares.
    """
    return type(identity)(
        config=dataclass_replace(identity.config, dropout=0.0),
        spec=identity.spec,
        category_count=identity.category_count,
    )


def stage_canary(ctx: Context) -> dict:
    """Mechanical checks only. Nothing measured here is a scientific result."""
    require_branch()
    verify_frozen_source(ctx, stage="canary")
    identity = model_identity(ctx.policy)
    spec = identity.spec
    prepared = ctx.prepared
    set_deterministic_execution(ctx.policy, seed=ctx.seed)
    state = torch.load(
        ctx.private_path("common_initialization/common_init_seed13.pt"),
        map_location="cpu",
        weights_only=True,
    )

    # Two eligible TRAIN clients, chosen by schedule position only, never by their score.
    clients = _round_plan(ctx)[0]["selected_client_ids"][:2]
    checks: dict[str, Any] = {}

    from ppsi.data.batching import windows_to_batch
    from ppsi.training.batch import validate_phase1_batch
    from ppsi.training.flower import FlowerLocalAdapter
    from ppsi.training.t1_mvp_objective import T1ContributingWeightPolicy

    batches_by_client = {}
    for client_id in clients:
        workload = client_workload(
            prepared,
            client_id,
            server_round=1,
            seed=ctx.seed,
            batch_size=int(ctx.policy["pilot"]["batch_size"]),
        )
        built = workload_batches(prepared, workload, spec)
        for batch in built:
            validate_phase1_batch(batch, spec)
        batches_by_client[client_id] = built
    checks["real_batches_validate"] = True
    checks["canary_clients"] = len(clients)
    checks["canary_batches"] = sum(len(v) for v in batches_by_client.values())
    first = batches_by_client[clients[0]][0]

    model, core = build_trainer(identity, ctx.policy, state)
    reset_client_stream(seed=ctx.seed, server_round=1, client_id=clients[0])
    summary = core.train_step(first)
    checks["finite_first_step_loss"] = require_finite_metric(summary.total_loss)
    checks["contributing_examples"] = summary.contributing_examples
    checks["gradients_finite"] = all(
        bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None
    )
    if not checks["gradients_finite"]:
        raise StageError("a gradient was not finite on real data")

    # A TRAIN-only toy overfit: the loop can reduce a loss. No real-data threshold is set.
    _, toy_core = build_trainer(identity, ctx.policy, state)
    toy_losses = [require_finite_metric(toy_core.train_step(first).total_loss) for _ in range(12)]
    checks["toy_overfit_loss_start"] = toy_losses[0]
    checks["toy_overfit_loss_end"] = toy_losses[-1]
    checks["toy_overfit_reduces_loss"] = toy_losses[-1] < toy_losses[0]
    if not checks["toy_overfit_reduces_loss"]:
        raise StageError("the shared training loop did not reduce a repeated-batch loss")

    # Fresh init reload is tensor-identical on both construction paths.
    left, _ = build_trainer(identity, ctx.policy, state)
    right, _ = build_trainer(identity, ctx.policy, state)
    checks["fresh_init_reload_identical"] = all(
        torch.equal(left.state_dict()[key], right.state_dict()[key]) for key in left.state_dict()
    )
    if not checks["fresh_init_reload_identical"]:
        raise StageError("two loads of the same common initialization differ")

    # One client, one batch, dropout disabled ONLY inside this separate toy fixture.
    fixture_identity = dropout_free_identity(identity)
    dropout_free = fixture_identity.config
    fixture_state, _ = common_initialization(ctx.seed, config=dropout_free, batch_spec=spec)
    central_model, central_core = build_trainer(fixture_identity, ctx.policy, fixture_state)
    central_core.train_step(first)
    central_after = pack_shared_state(central_model, central_model.shared_state_spec())

    flower_model, flower_core = build_trainer(fixture_identity, ctx.policy, fixture_state)
    adapter = FlowerLocalAdapter(
        core=flower_core,
        shared_state_spec=flower_model.shared_state_spec(),
        aggregation_weight_policy=T1ContributingWeightPolicy(),
    )
    fit = adapter.fit(
        pack_shared_state(flower_model, flower_model.shared_state_spec()),
        [first],
        outer_round=1,
    )
    diffs = [
        float(torch.abs(central_after[key] - fit.shared_state[key]).max().item())
        for key in central_after
    ]
    checks["central_vs_flower_local_max_abs_diff"] = max(diffs)
    checks["central_vs_flower_local_within_atol"] = max(diffs) <= 1e-6
    if not checks["central_vs_flower_local_within_atol"]:
        raise StageError("the two execution paths disagree on one identical local update")
    checks["dropout_disabled_only_in_this_fixture"] = True
    checks["aggregation_weight_is_contributing_examples"] = fit.aggregation_weight

    # A small validation smoke batch: shapes and finiteness only, never a score claim.
    smoke = windows_to_batch(prepared.validation.windows(), np.arange(64), spec, validate=True)
    model.eval()
    with torch.no_grad():
        output = model(smoke)
    checks["validation_logits_shape"] = list(output.t1_logits.shape)
    checks["validation_logits_finite"] = bool(torch.isfinite(output.t1_logits).all())

    record = {
        "schema": "mvp_canary_v1",
        "version": "1",
        "run": ctx.run,
        "namespace": "CANARY_NOT_A_SCIENTIFIC_RESULT",
        "generated_at_utc": now(),
        "client_selection_basis": "the first two entries of the frozen round-1 schedule",
        "checks": checks,
        "counted_in_final_rounds": False,
        "common_initialization_reloaded_after_canary": True,
        "resources": ctx.check_resources(stage="canary"),
    }
    write_json(ctx.public_path("canary.v1.json"), record, replace=True)
    log(
        f"canary passed; local update agreement {checks['central_vs_flower_local_max_abs_diff']:.2e}"
    )
    return record


# ---------------------------------------------------------------------------
# Stage: centralized
# ---------------------------------------------------------------------------


def stage_centralized(ctx: Context) -> dict:
    """Replay the identical schedule through ONE persistent trainer core."""
    require_branch()
    verify_frozen_source(ctx, stage="centralized")
    started_at = now()
    identity = model_identity(ctx.policy)
    prepared = ctx.prepared
    set_deterministic_execution(ctx.policy, seed=ctx.seed)
    state = torch.load(
        ctx.private_path("common_initialization/common_init_seed13.pt"),
        map_location="cpu",
        weights_only=True,
    )
    model, core = build_trainer(identity, ctx.policy, state)
    batch_size = int(ctx.policy["pilot"]["batch_size"])

    evaluations = []
    if 0 in _evaluation_rounds(ctx):
        evaluations.append(_evaluate_and_record(ctx, model, 0, "centralized"))
        log(f"round 0 headline {evaluations[-1]['headline']['value']:.5f}")

    exposure: list[tuple[int, str, list[str]]] = []
    rounds = []
    for entry in _round_plan(ctx):
        server_round = int(entry["server_round"])
        round_started = time.perf_counter()
        rows = batches = steps = 0
        numerator = 0.0
        denominator = 0
        for client_id in entry["selected_client_ids"]:
            workload = client_workload(
                prepared,
                client_id,
                server_round=server_round,
                seed=ctx.seed,
                batch_size=batch_size,
            )
            exposure.append((server_round, client_id, workload.decision_keys))
            reset_client_stream(seed=ctx.seed, server_round=server_round, client_id=client_id)
            for batch in workload_batches(prepared, workload, identity.spec):
                summary = core.train_step(batch)
                stat = summary.task_stats.get("T1")
                if stat is None or summary.contributing_examples <= 0:
                    raise StageError("a real T1 batch contributed nothing")
                numerator += require_finite_metric(stat.numerator)
                denominator += int(stat.denominator)
                steps += int(summary.optimizer_step_performed)
                batches += 1
            rows += workload.rows
        rounds.append(
            {
                "server_round": server_round,
                "clients": len(entry["selected_client_ids"]),
                "rows": rows,
                "batches": batches,
                "optimizer_steps": steps,
                "loss_numerator": numerator,
                "loss_denominator": denominator,
                "mean_train_loss": numerator / denominator,
                "seconds": round(time.perf_counter() - round_started, 2),
                "selection_digest": entry["selected_digest"],
            }
        )
        _save_round_checkpoint(ctx, "centralized", server_round, model, core)
        if server_round in _evaluation_rounds(ctx):
            evaluations.append(_evaluate_and_record(ctx, model, server_round, "centralized"))
            log(
                f"round {server_round} headline "
                f"{evaluations[-1]['headline']['value']:.5f} "
                f"(train loss {rounds[-1]['mean_train_loss']:.4f})"
            )
        else:
            log(f"round {server_round} train loss {rounds[-1]['mean_train_loss']:.4f}")

    record = {
        "schema": "mvp_centralized_run_v1",
        "version": "1",
        "run": ctx.run,
        "regime": "R1",
        "result_role": "MVP_MATCHED_T1_CENTRALIZED_NOT_FINAL_R1",
        "optimizer_lifecycle": ctx.policy["pilot"]["centralized_optimizer_lifecycle"],
        "started_at_utc": started_at,
        "ended_at_utc": now(),
        "rounds": rounds,
        "evaluations": evaluations,
        "exposure_sha256": exposure_digest(exposure),
        "totals": {
            "rows": sum(r["rows"] for r in rounds),
            "batches": sum(r["batches"] for r in rounds),
            "optimizer_steps": sum(r["optimizer_steps"] for r in rounds),
            "participations": sum(r["clients"] for r in rounds),
            "unique_clients": len({client for _, client, _ in exposure}),
        },
        "resources": ctx.check_resources(stage="centralized"),
    }
    write_json(ctx.public_path("centralized_run.v1.json"), record, replace=True)
    return record


def _save_round_checkpoint(ctx: Context, regime: str, server_round: int, model, core) -> Path:
    """A completed-round checkpoint with cursor, optimizer and RNG state."""
    from ppsi.training.checkpoint import (
        CheckpointIdentity,
        build_checkpoint_payload,
        save_checkpoint,
    )
    from ppsi.training.sampler import LoaderContract, TrainingCursor

    prepared = ctx.prepared
    identity = CheckpointIdentity(
        experiment_config_sha256=file_sha256(ROOT / ctx.config_rel),
        common_initialization_sha256=raw_file_sha256(
            ctx.private_path("common_initialization/common_init_seed13.pt")
        ),
        input_data_sha256=prepared.data_manifest_sha256,
        shared_trainer_core_sha256=file_sha256(ROOT / "ppsi/training/core.py"),
        objective_config_sha256=file_sha256(ROOT / "ppsi/training/t1_mvp_objective.py"),
        environment_lock_sha256=file_sha256(ROOT / "uv.lock"),
        git_sha=git("rev-parse", "HEAD"),
    )
    payload = build_checkpoint_payload(
        run_id=f"{ctx.run}-{regime}",
        attempt=1,
        model=model,
        optimizer=core.optimizer,
        scheduler=None,
        grad_scaler=None,
        cursor=TrainingCursor(
            outer_round=server_round,
            local_epoch=0,
            next_batch_index=0,
            optimizer_step=core.optimizer_step_count,
        ),
        best_criterion=None,
        identity=identity,
        loader_contract=LoaderContract(
            schema="loader_contract_v1",
            version="1",
            dataset_identity_sha256=prepared.data_manifest_sha256,
            dataset_length=prepared.train.rows,
            batch_size=int(ctx.policy["pilot"]["batch_size"]),
            drop_last=False,
            sampler_version="mvp_client_epoch_order_v1",
            run_seed=ctx.seed,
        ),
        scheduler_step_unit="OPTIMIZER_STEP",
    )
    path = ctx.private_path(f"{regime}/checkpoints/round_{server_round:02d}.pt")
    save_checkpoint(path, payload)
    return path


# ---------------------------------------------------------------------------
# Stage: federated — real Flower FedAvg over the same schedule
# ---------------------------------------------------------------------------

# One read-only prepared bundle per Ray worker process, keyed by the manifest digest of
# the inputs it describes. A worker never caches a client's model or optimizer: those
# belong to one client for one round and reusing them would leak state between clients.
_WORKER_CACHE: dict[str, Any] = {}


def _worker_bundle(prepared_dir: str, data_manifest_sha256: str) -> Any:
    cached = _WORKER_CACHE.get(data_manifest_sha256)
    if cached is None:
        cached = load_prepared(Path(prepared_dir))
        if cached.data_manifest_sha256 != data_manifest_sha256:
            raise StageError("prepared inputs on this worker are not the ones the server declared")
        _WORKER_CACHE[data_manifest_sha256] = cached
    return cached


def build_mvp_client_app():
    """The real-data ClientApp. Every input arrives in the message, not from a global."""
    from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict
    from flwr.clientapp import ClientApp

    from ppsi.training.flower import FlowerLocalAdapter
    from ppsi.training.t1_mvp_objective import T1ContributingWeightPolicy
    from scripts.federated.fl_synthetic_smoke import get_digest

    app = ClientApp()

    @app.train()
    def train(message: Message, context) -> Message:
        config_rec = message.content.configs_records.get("config", ConfigRecord())
        logical_client_id = str(config_rec["logical_client_id"])
        server_round = int(config_rec["server_round"])
        prepared_dir = str(config_rec["prepared_dir"])
        data_manifest = str(config_rec["data_manifest_sha256"])
        policy_path = str(config_rec["policy_path"])
        seed = int(config_rec["experiment_seed"])

        policy = json.loads(Path(policy_path).read_text(encoding="utf-8"))
        torch.set_num_threads(int(policy["resources"]["client_torch_threads"]))
        prepared = _worker_bundle(prepared_dir, data_manifest)

        scheduled = {
            entry["server_round"]: entry["selected_client_ids"] for entry in prepared.schedule
        }
        if logical_client_id not in scheduled.get(server_round, []):
            raise StageError("this client is not scheduled for this round")

        identity = model_identity(policy)
        workload = client_workload(
            prepared,
            logical_client_id,
            server_round=server_round,
            seed=seed,
            batch_size=int(policy["pilot"]["batch_size"]),
        )
        owned = set(prepared.rows_for(logical_client_id).tolist())
        if not set(workload.row_order.tolist()).issubset(owned):
            raise StageError("a selected row does not belong to this client's TRAIN rows")

        incoming = message.content.parameters_records["arrays"].to_torch_state_dict()
        received_digest = get_digest(incoming)

        model, core = build_trainer(identity, policy, None)
        adapter = FlowerLocalAdapter(
            core=core,
            shared_state_spec=model.shared_state_spec(),
            aggregation_weight_policy=T1ContributingWeightPolicy(),
        )
        reset_client_stream(seed=seed, server_round=server_round, client_id=logical_client_id)
        batches = workload_batches(prepared, workload, identity.spec)
        result = adapter.fit(incoming, batches, outer_round=server_round)
        if result.aggregation_weight <= 0:
            raise StageError("a client returned a non-positive aggregation weight")

        stat = result.summary.task_stats.get("T1")
        if stat is None or stat.mean is None:
            raise StageError("a client produced no contributing T1 loss")
        train_loss = require_finite_metric(stat.mean)
        updated = dict(result.shared_state)
        updated_array_record = ArrayRecord.from_torch_state_dict(updated)

        payload = RecordDict()
        payload.parameters_records["arrays"] = updated_array_record
        payload.metrics_records["metrics"] = MetricRecord(
            {
                "num-examples": result.aggregation_weight,
                "train_loss": train_loss,
                "rows": float(workload.rows),
                "batches": float(len(batches)),
            }
        )
        payload.configs_records["config"] = ConfigRecord(
            {
                "logical_client_id": logical_client_id,
                # The inherited strategy compares these against its own get_digest values,
                # which encode differently from the training-state codec. Mixing the two
                # conventions would look like a redistribution failure.
                "received_digest": received_digest,
                "updated_digest": get_digest(updated),
                "local_train_loss": train_loss,
                "server_round": server_round,
            }
        )
        return message.create_reply(payload)

    return app


COMPLETED_ROUNDS_DIR = "federated/completed_rounds"
LAST_COMPLETED_NAME = "federated/last_completed_round.json"


def federated_resume_identity(ctx: Context) -> dict:
    """Everything a resumed run must match before it may continue this one.

    A resume that differs in any of these is a different experiment wearing the same run
    id, so the mismatch is fatal rather than reconciled.
    """
    init_path = ctx.private_path("common_initialization/common_init_seed13.pt")
    pilot = ctx.policy["pilot"]
    return {
        "run": ctx.run,
        "policy_sha256": file_sha256(ROOT / ctx.config_rel),
        "source_freeze_sha256": file_sha256(ROOT / ctx.public_path(SOURCE_FREEZE_NAME)),
        "data_manifest_sha256": ctx.prepared.data_manifest_sha256,
        "common_initialization_sha256": raw_file_sha256(init_path),
        "schedule_digest": logical_digest(
            [entry["selected_digest"] for entry in ctx.prepared.schedule]
        ),
        "seed": ctx.seed,
        "rounds": int(pilot["rounds"]),
        "clients_per_round": int(pilot["clients_per_round"]),
        "batch_size": int(pilot["batch_size"]),
    }


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def persist_completed_round(
    directory: Path,
    *,
    run: str,
    server_round: int,
    state: dict,
    identity: dict,
    trace: dict,
    round_record: dict,
    ledger_records: list[dict],
    evaluations: list[dict],
) -> dict:
    """Write one *completed* round's recovery point, pointer last.

    Order matters. The state file, then the checkpoint description, then the pointer that
    advertises the round as resumable. A crash anywhere leaves the previous pointer intact,
    so a partially written round can never be mistaken for a completed one.

    Paths inside the checkpoint are relative to ``directory``, so the recovery point stays
    self-describing wherever the private run directory lives.
    """
    rounds_dir = directory / COMPLETED_ROUNDS_DIR
    rounds_dir.mkdir(parents=True, exist_ok=True)
    state_path = rounds_dir / f"round_{server_round:02d}.state.pt"
    temporary = state_path.with_suffix(".pt.tmp")
    torch.save(state, temporary)
    temporary.replace(state_path)

    payload = {
        "schema": "mvp_federated_completed_round_v1",
        "version": "1",
        "run": run,
        "completed_server_round": server_round,
        "completed_at_utc": now(),
        "identity": identity,
        "selection": {
            "selection_digest": trace["selection_digest"],
            "selected_client_count": trace["selected_client_count"],
            "sampler_round_index": server_round - 1,
        },
        "aggregate": {
            "digest": trace["aggregated_digest"],
            "state_uri": state_path.relative_to(directory).as_posix(),
            "state_sha256": raw_file_sha256(state_path),
        },
        "round_record": round_record,
        "communication_records_through_round": ledger_records,
        "communication_totals_through_round": _totals_of(ledger_records),
        "evaluations_through_round": evaluations,
        "partial_rounds_are_never_recorded_here": True,
    }
    checkpoint_path = rounds_dir / f"round_{server_round:02d}.checkpoint.json"
    _atomic_write_text(checkpoint_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    pointer_path = directory / LAST_COMPLETED_NAME
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(
        pointer_path,
        json.dumps(
            {
                "schema": "mvp_federated_resume_pointer_v1",
                "version": "1",
                "run": run,
                "completed_server_round": server_round,
                "checkpoint_uri": checkpoint_path.relative_to(directory).as_posix(),
                "checkpoint_sha256": raw_file_sha256(checkpoint_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    return payload


def _totals_of(records: list[dict]) -> dict:
    totals = {
        "download_bytes": 0,
        "upload_bytes": 0,
        "download_transmissions": 0,
        "upload_transmissions": 0,
    }
    for record in records:
        totals[f"{record['direction']}_bytes"] += int(record["payload_bytes"])
        totals[f"{record['direction']}_transmissions"] += 1
    return totals


def load_last_completed_round(directory: Path, identity: dict) -> dict | None:
    """The last fully completed, identity-compatible round, or nothing."""
    pointer_path = directory / LAST_COMPLETED_NAME
    if not pointer_path.is_file():
        return None
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    checkpoint_path = directory / pointer["checkpoint_uri"]
    if not checkpoint_path.is_file():
        raise StageError("the resume pointer names a checkpoint that is not on disk")
    if raw_file_sha256(checkpoint_path) != pointer["checkpoint_sha256"]:
        raise StageError("the resume checkpoint does not match the hash its pointer recorded")
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if payload["completed_server_round"] != pointer["completed_server_round"]:
        raise StageError("the resume pointer and its checkpoint disagree about the round")
    differing = sorted(
        key
        for key in set(identity) | set(payload["identity"])
        if identity.get(key) != payload["identity"].get(key)
    )
    if differing:
        raise StageError(
            f"INCOMPATIBLE_RESUME: {differing} differ from the persisted federated run. "
            "A changed experiment continues under a new run id, never by resuming this one."
        )
    state_path = directory / payload["aggregate"]["state_uri"]
    if (
        not state_path.is_file()
        or raw_file_sha256(state_path) != payload["aggregate"]["state_sha256"]
    ):
        raise StageError("the persisted aggregate state is missing or altered")
    return payload


def build_mvp_strategy_class():
    """Subclass the proven strategy: measure bytes, keep the reply set complete, and
    validate the aggregate against a derived float32 bound instead of a fixed tolerance.

    ``round_offset`` exists only for a resumed run. Flower always counts its own rounds
    from one, but a client's data order and RNG stream are derived from the *logical*
    round of the 20-round schedule, so the offset is added back before anything that
    touches identity or exposure.

    The inherited strategy is untouched. Its own absolute-tolerance comparison is given
    this round's derived global allowance so it cannot fire before the authoritative
    per-tensor check below, and every structural check it performs still applies.
    """
    from scripts.federated.fl_real_smoke import (
        RealDataTracingFedAvg,
        sort_client_replies_by_logical_id,
    )
    from scripts.federated.fl_synthetic_smoke import weighted_average_state_dicts

    class MVPTracingFedAvg(RealDataTracingFedAvg):
        """Byte ledger at the real message boundary, plus the scale-aware oracle."""

        def __init__(
            self, *, ledger: CommunicationLedger, round_offset: int = 0, **kwargs: Any
        ) -> None:
            super().__init__(aggregation_atol=float("inf"), **kwargs)
            self.ledger = ledger
            self.round_offset = int(round_offset)
            self.reply_ids_by_round: dict[int, list[str]] = {}
            self.oracle_by_round: dict[int, dict] = {}

        def logical_round(self, server_round: int) -> int:
            return int(server_round) + self.round_offset

        def configure_train(self, server_round, arrays, config, grid):
            messages = super().configure_train(server_round, arrays, config, grid)
            logical = self.logical_round(server_round)
            for message in messages:
                conf = message.content.configs_records["config"]
                # The client derives its epoch order and RNG stream from this number.
                conf["server_round"] = logical
                self.ledger.measure(
                    message.content.parameters_records["arrays"],
                    server_round=logical,
                    client_id=str(conf["logical_client_id"]),
                    direction="download",
                )
            return messages

        def aggregate_train(self, server_round, replies):
            materialized = list(replies)
            logical = self.logical_round(server_round)
            received: list[str] = []
            for message in materialized:
                if message.has_error():
                    # A failed reply is a missing measurement, never an upload of zero.
                    raise StageError(f"client error in round {logical}: {message.error}")
                conf = message.content.configs_records["config"]
                client_id = str(conf["logical_client_id"])
                received.append(client_id)
                self.ledger.measure(
                    message.content.parameters_records["arrays"],
                    server_round=logical,
                    client_id=client_id,
                    direction="upload",
                )
            # The parent checks that replies exist; the policy requires the complete set.
            require_complete_replies(list(self.selected_ids_by_round[server_round - 1]), received)
            self.reply_ids_by_round[logical] = sorted(received)

            # Rebuild the same updates the parent will use, in the same sorted order, so
            # the independent float64 oracle below describes exactly this aggregation.
            updates = [
                (
                    message.content.parameters_records["arrays"].to_torch_state_dict(),
                    int(message.content.metrics_records["metrics"]["num-examples"]),
                )
                for message in sort_client_replies_by_logical_id(materialized)
            ]
            arrays, metrics = super().aggregate_train(server_round, materialized)
            if arrays is None:
                raise StageError("Flower aggregation returned nothing")
            diagnostics = check_aggregation(
                updates=updates,
                oracle_state=weighted_average_state_dicts(updates),
                flower_state=arrays.to_torch_state_dict(),
                server_round=logical,
            )
            self.oracle_by_round[logical] = diagnostics
            entry = self.tracing_log[-1]
            entry["aggregation_oracle_pass"] = diagnostics["oracle_pass"]
            entry["max_abs_diff"] = diagnostics["max_abs_diff"]
            return arrays, metrics

    return MVPTracingFedAvg


def _ledger_rows(ledger: CommunicationLedger) -> list[dict]:
    return [
        {
            "server_round": record.server_round,
            "client_id": record.client_id,
            "direction": record.direction,
            "payload_bytes": record.payload_bytes,
        }
        for record in ledger.records
    ]


def _restore_ledger(rows: list[dict]) -> CommunicationLedger:
    from ppsi.federated.communication import CommunicationRecord

    ledger = CommunicationLedger()
    for row in rows:
        ledger.add(
            CommunicationRecord(
                server_round=int(row["server_round"]),
                client_id=str(row["client_id"]),
                direction=str(row["direction"]),
                payload_bytes=int(row["payload_bytes"]),
            )
        )
    return ledger


def stage_federated(ctx: Context) -> dict:
    """One real Flower FedAvg run over the identical schedule, with measured bytes."""
    require_branch()
    verify_frozen_source(ctx, stage="federated")
    from flwr.app import ArrayRecord, ConfigRecord, MetricRecord
    from flwr.serverapp import ServerApp
    from flwr.simulation import run_simulation

    started_at = now()
    identity = model_identity(ctx.policy)
    prepared = ctx.prepared
    set_deterministic_execution(ctx.policy, seed=ctx.seed)
    schedule = _round_plan(ctx)
    total_rounds = int(ctx.policy["pilot"]["rounds"])
    per_round = int(ctx.policy["pilot"]["clients_per_round"])
    evaluation_rounds = _evaluation_rounds(ctx)

    resume_identity = federated_resume_identity(ctx)
    private_root = ROOT / ctx.private
    resumed = load_last_completed_round(private_root, resume_identity)
    completed = int(resumed["completed_server_round"]) if resumed else 0
    if completed >= total_rounds:
        raise StageError(
            f"the federated run already completed round {completed} of {total_rounds}; its "
            "evidence is on disk and is not re-executed in place"
        )
    if resumed:
        log(f"resuming the federated run from completed round {completed}")
        initial_state = torch.load(
            private_root / resumed["aggregate"]["state_uri"],
            map_location="cpu",
            weights_only=True,
        )
        ledger = _restore_ledger(resumed["communication_records_through_round"])
        prior_rounds = [
            json.loads(
                (
                    private_root / COMPLETED_ROUNDS_DIR / f"round_{index:02d}.checkpoint.json"
                ).read_text(encoding="utf-8")
            )["round_record"]
            for index in range(1, completed + 1)
        ]
        prior_evaluations = list(resumed["evaluations_through_round"])
    else:
        initial_state = torch.load(
            ctx.private_path("common_initialization/common_init_seed13.pt"),
            map_location="cpu",
            weights_only=True,
        )
        ledger = CommunicationLedger()
        prior_rounds = []
        prior_evaluations = []

    # Re-keyed so Flower's own round 1 maps to the next unfinished logical round.
    selected_ids = {
        index: list(schedule[index + completed]["selected_client_ids"])
        for index in range(total_rounds - completed)
    }
    selection_digests = {
        index: schedule[index + completed]["selected_digest"]
        for index in range(total_rounds - completed)
    }

    strategy_class = build_mvp_strategy_class()
    holder: dict[str, Any] = {
        "evaluations": list(prior_evaluations),
        "rounds": list(prior_rounds),
        "final_state": None,
    }

    app = ServerApp()

    @app.main()
    def server_main(grid, context) -> None:
        set_deterministic_execution(ctx.policy, seed=ctx.seed)
        strategy = strategy_class(
            ledger=ledger,
            round_offset=completed,
            expected_clients=per_round,
            selected_ids_by_round=selected_ids,
            selection_digests_by_round=selection_digests,
            fraction_train=1.0,
            fraction_evaluate=0.0,
            min_train_nodes=per_round,
            min_evaluate_nodes=0,
            min_available_nodes=per_round,
            weighted_by_key="num-examples",
        )

        def evaluate_fn(server_round: int, arrays: ArrayRecord) -> MetricRecord:
            # Flower calls this once with its initial arrays and once after each round it
            # has fully aggregated, which is exactly the completed-round boundary.
            logical = completed if server_round == 0 else strategy.logical_round(server_round)
            server_state = {
                key: value.detach().cpu().clone()
                for key, value in arrays.to_torch_state_dict().items()
            }
            already = next(
                (e for e in holder["evaluations"] if int(e["server_round"]) == logical), None
            )
            evaluated = None
            if logical in evaluation_rounds and already is None:
                model, _ = build_trainer(identity, ctx.policy, server_state)
                evaluated = _evaluate_and_record(ctx, model, logical, "federated")
                holder["evaluations"].append(evaluated)
                log(f"[federated] round {logical} headline {evaluated['headline']['value']:.5f}")
            elif already is not None:
                evaluated = already

            if server_round >= 1:
                holder["final_state"] = server_state
                trace = strategy.tracing_log[-1]
                entry = schedule[logical - 1]
                round_record = _federated_round_record(
                    ctx,
                    entry=entry,
                    trace=trace,
                    ledger=ledger,
                    logical_round=logical,
                    oracle=strategy.oracle_by_round[logical],
                )
                holder["rounds"].append(round_record)
                persist_completed_round(
                    private_root,
                    run=ctx.run,
                    server_round=logical,
                    state=server_state,
                    identity=resume_identity,
                    trace=trace,
                    round_record=round_record,
                    ledger_records=_ledger_rows(ledger),
                    evaluations=holder["evaluations"],
                )

            if evaluated is None:
                # Truthful minimal status: no full evaluation ran at this round.
                return MetricRecord({"full_validation_executed": 0.0})
            return MetricRecord(
                {
                    "full_validation_executed": 1.0,
                    "t1_next_distinct_mrr_at_20_macro": evaluated["headline"]["value"],
                    "support_clients": float(evaluated["headline"]["support_clients"]),
                    "support_decisions": float(evaluated["headline"]["support_decisions"]),
                }
            )

        holder["result"] = strategy.start(
            grid=grid,
            initial_arrays=ArrayRecord.from_torch_state_dict(initial_state),
            num_rounds=total_rounds - completed,
            train_config=ConfigRecord(
                {
                    "prepared_dir": str(ctx.prepared_dir),
                    "data_manifest_sha256": prepared.data_manifest_sha256,
                    "policy_path": str(ROOT / ctx.config_rel),
                    "experiment_seed": ctx.seed,
                }
            ),
            evaluate_fn=evaluate_fn,
        )
        holder["tracing_log"] = strategy.tracing_log
        holder["reply_ids_by_round"] = strategy.reply_ids_by_round

    concurrency = int(ctx.policy["resources"]["ray_total_cpus"])
    run_started = time.perf_counter()
    run_simulation(
        server_app=app,
        client_app=build_mvp_client_app(),
        num_supernodes=per_round,
        backend_name="ray",
        backend_config={
            "init_args": {
                "num_cpus": concurrency,
                "num_gpus": 0,
                "include_dashboard": False,
            },
            "client_resources": {
                "num_cpus": int(ctx.policy["resources"]["ray_client_cpus"]),
                "num_gpus": 0,
            },
        },
    )
    elapsed = time.perf_counter() - run_started
    if "tracing_log" not in holder:
        raise StageError("the federated run did not complete its server main")

    rounds = sorted(holder["rounds"], key=lambda r: r["server_round"])
    if [r["server_round"] for r in rounds] != list(range(1, total_rounds + 1)):
        raise StageError("the federated run did not execute every scheduled round exactly once")
    final_state = holder["final_state"]
    if final_state is None:
        raise StageError("no aggregated state was captured for the final round")
    final_path = ctx.private_path(f"federated/aggregate_round_{total_rounds:02d}.pt")
    temporary = final_path.with_suffix(".pt.tmp")
    torch.save(final_state, temporary)
    temporary.replace(final_path)

    exposure: list[tuple[int, str, list[str]]] = []
    for entry in schedule:
        for client_id in entry["selected_client_ids"]:
            workload = client_workload(
                prepared,
                client_id,
                server_round=int(entry["server_round"]),
                seed=ctx.seed,
                batch_size=int(ctx.policy["pilot"]["batch_size"]),
            )
            exposure.append((int(entry["server_round"]), client_id, workload.decision_keys))

    ledger_path = ctx.private_path("federated/communication_ledger.json")
    _atomic_write_text(ledger_path, json.dumps(_ledger_rows(ledger), separators=(",", ":")))

    record = {
        "schema": "mvp_federated_run_v1",
        "version": "1",
        "run": ctx.run,
        "regime": "R2A",
        "result_role": "MVP_MATCHED_T1_FEDAVG_NOT_FINAL_R2A",
        "optimizer_lifecycle": ctx.policy["pilot"]["federated_optimizer_lifecycle"],
        "started_at_utc": started_at,
        "ended_at_utc": now(),
        "wall_seconds": round(elapsed, 2),
        "rounds": rounds,
        "evaluations": sorted(holder["evaluations"], key=lambda e: e["server_round"]),
        "exposure_sha256": exposure_digest(exposure),
        "recovery": {
            "resumed_from_completed_round": completed,
            "rounds_executed_in_this_attempt": total_rounds - completed,
            "completed_round_checkpoints": (ctx.private / COMPLETED_ROUNDS_DIR).as_posix(),
            "policy": "COMPLETE_ROUND_BOUNDARY_ONLY_NEW_ATTEMPT_PRESERVE_ALL_PRIOR_BYTES",
            "partial_round_completion": "NEVER_RECORDED",
        },
        "population": {
            "declared_pilot_population": len(prepared.population),
            "supernode_execution_slots": per_round,
            "scheduled_participations": sum(r["clients"] for r in rounds),
            "measured_unique_clients_trained": len({c for _, c, _ in exposure}),
            "note": (
                "execution slots are not study clients; the pilot population is 1,000 and "
                "each round trains 50 of them"
            ),
        },
        "aggregation_oracle": {
            "policy_id": ORACLE_POLICY_ID,
            "reference": "FLOAT64_WEIGHTED_AVERAGE_INDEPENDENT_ORACLE",
            "enforced_in": ("the MVP strategy subclass, per tensor, on the real Flower aggregate"),
            "inherited_fixed_tolerance": (
                "SUPERSEDED. The inherited comparison is given an infinite absolute "
                "tolerance so it cannot fire ahead of the derived per-tensor bound; every "
                "structural check it performs still applies, and non-finite values are "
                "rejected explicitly rather than by comparing NaN against a tolerance."
            ),
            "structural_checks_retained": ctx.policy["pilot"][
                "aggregation_structural_checks_retained"
            ],
            "per_round": [r["aggregation_oracle"] for r in rounds],
        },
        "communication": {
            "measurement_boundary": (
                "serialized model payload of each ArrayRecord at the real Flower message "
                "boundary; framing, config and metric records are excluded"
            ),
            "counted_once_per_transmission": True,
            "run_totals": ledger.run_totals(),
            "private_ledger_uri": str(ledger_path.relative_to(ROOT)).replace("\\", "/"),
        },
        "final_aggregate": {
            "server_round": total_rounds,
            "private_uri": str(final_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": raw_file_sha256(final_path),
            "digest": rounds[-1]["aggregated_digest"],
        },
        "totals": {
            "rows": sum(r["rows"] for r in rounds),
            "contributing_examples": sum(r["contributing_examples"] for r in rounds),
            "max_oracle_abs_diff": max(float(r["max_abs_diff"]) for r in rounds),
            "max_oracle_ratio": max(float(r["aggregation_oracle"]["worst_ratio"]) for r in rounds),
            "max_oracle_tight_ratio": max(
                float(r["aggregation_oracle"]["worst_tight_ratio"]) for r in rounds
            ),
            "every_round_within_tight_diagnostic_bound": all(
                bool(r["aggregation_oracle"]["passes_tight_diagnostic_bound"]) for r in rounds
            ),
            "aggregation_oracle_policy": ORACLE_POLICY_ID,
        },
        "flower": {
            "version": importlib.metadata.version("flwr"),
            "ray_total_cpus": concurrency,
            "ray_client_cpus": int(ctx.policy["resources"]["ray_client_cpus"]),
            "gpus_requested": 0,
        },
        "resources": ctx.check_resources(stage="federated"),
    }
    write_json(ctx.public_path("federated_run.v1.json"), record, replace=True)
    return record


def _federated_round_record(
    ctx: Context,
    *,
    entry: dict,
    trace: dict,
    ledger: CommunicationLedger,
    logical_round: int,
    oracle: dict,
) -> dict:
    """The public row for one completed round, built at the round boundary itself."""
    rows = 0
    for client_id in entry["selected_client_ids"]:
        workload = client_workload(
            ctx.prepared,
            client_id,
            server_round=logical_round,
            seed=ctx.seed,
            batch_size=int(ctx.policy["pilot"]["batch_size"]),
        )
        rows += workload.rows
    bytes_round = ledger.round_totals()[logical_round]
    if trace["selection_digest"] != entry["selected_digest"]:
        raise StageError("the executed round does not carry its scheduled selection digest")
    return {
        "server_round": logical_round,
        "clients": trace["selected_client_count"],
        "rows": rows,
        "contributing_examples": trace["contributing_examples"],
        "selection_digest": trace["selection_digest"],
        "aggregation_oracle_pass": oracle["oracle_pass"],
        "max_abs_diff": oracle["max_abs_diff"],
        "aggregation_oracle": oracle,
        "aggregated_digest": trace["aggregated_digest"],
        "mean_train_loss": sum(c["local_train_loss"] for c in trace["clients"])
        / len(trace["clients"]),
        "download_bytes": bytes_round["download_bytes"],
        "upload_bytes": bytes_round["upload_bytes"],
        "download_transmissions": bytes_round["download_transmissions"],
        "upload_transmissions": bytes_round["upload_transmissions"],
    }


# ---------------------------------------------------------------------------
# Stage: compare — the matched pair, and an isolated quality-retention attempt
# ---------------------------------------------------------------------------


def _pair_config(ctx: Context, *, regime: str, run: dict, init_ref: dict) -> dict:
    """One ExperimentConfig per regime. Only the regime block may differ."""
    protocol_ref = ref(
        "docs/evidence/s1-ds-05-06/data_protocol_v1.proposed.json", "data_protocol_v1"
    )
    prepare_ref = ref(ctx.public_path("prepare_summary.v1.json"), "mvp_prepare_summary_v1")
    centralized = regime == "R1"
    cfg = {
        "schema": "experiment_config_v1",
        "version": "1",
        "config_id": f"{ctx.run}_{'centralized' if centralized else 'fedavg'}",
        "regime": regime,
        "tasks": ["T1"],
        "training_cohort": "C1",
        "evaluation_cohorts": ["C1"],
        "seed": ctx.seed,
        "source_dataset_ref": prepare_ref,
        "canonical_data_contract_ref": protocol_ref,
        "cohort_manifest_ref": ref(
            "data/protocol/INTERNAL_DO_NOT_UPLOAD_cohort_manifest_v1.proposed.parquet",
            "cohort_manifest_v1",
        ),
        "split_manifest_ref": protocol_ref,
        "task_examples_manifest_ref": prepare_ref,
        "evaluation_manifest_ref": ref(
            "data/examples/INTERNAL_DO_NOT_UPLOAD_task_examples_t1_v1.proposed.parquet",
            "task_example_v1",
        ),
        "representation_ref": ref("ppsi/models/batch_spec.py", "python_source_v1"),
        "model_config_ref": ref(ctx.public_path("preflight.v1.json"), "mvp_preflight_v1"),
        "objective_config_ref": ref("ppsi/training/t1_mvp_objective.py", "python_source_v1"),
        "shared_trainer_core_ref": ref("ppsi/training/core.py", "python_source_v1"),
        "evaluation_protocol_ref": ref(
            "docs/decisions/mvp-execution.md", "mvp_execution_decision_v1"
        ),
        "evaluator_ref": ref("ppsi/evaluation/t1.py", "python_source_v1"),
        "environment_lock_ref": ref("uv.lock", "uv_lock_v1"),
        "initialization": {
            "kind": "COMMON_INITIALIZATION",
            "common_initialization_ref": init_ref,
        },
        "regime_config": {
            "orchestration_type": "CENTRALIZED" if centralized else "FEDAVG",
            "result_role": run["result_role"],
            "evaluation_split": "VALIDATION",
            "optimizer_lifecycle": run["optimizer_lifecycle"],
            "rounds": int(ctx.policy["pilot"]["rounds"]),
            "clients_per_round": int(ctx.policy["pilot"]["clients_per_round"]),
            "pilot_population": len(ctx.prepared.population),
            "exposure_sha256": run["exposure_sha256"],
            "score_convention": "RAW_NO_SUPPRESSION",
            "checkpoint_selection": ctx.policy["pilot"]["checkpoint_selection"],
            "comparison_label": ctx.policy["pilot"]["comparison_label"],
            "final_r1_denominator": False,
            "final_r2a_reference": False,
            "quality_retention_claim": False,
            "neural_trainer_used": True,
        },
    }
    validate_experiment_config(cfg)
    return cfg


def _publish_pair_result(ctx: Context, *, regime: str, run: dict, snapshot_ref: dict) -> dict:
    """Publish one half of the pair into the isolated pilot-pair registry directory."""
    init_ref = ref(
        ctx.public_path("preflight.v1.json"),
        "mvp_preflight_v1",
        "common_initialization_declaration",
    )
    cfg = _pair_config(ctx, regime=regime, run=run, init_ref=init_ref)
    name = "centralized" if regime == "R1" else "fedavg"
    directory = str(ctx.pair_out).replace("\\", "/")
    cfg_path = f"{directory}/{name}.experiment_config.v1.json"
    write_json(cfg_path, cfg, replace=True)
    final_round = int(ctx.policy["pilot"]["rounds"])
    evaluation = next(e for e in run["evaluations"] if int(e["server_round"]) == final_round)
    run_path = ctx.public_path(
        "centralized_run.v1.json" if regime == "R1" else "federated_run.v1.json"
    )

    from ppsi.evaluation.t1 import T1EvaluationSummary

    summary = T1EvaluationSummary(
        status=evaluation["evaluation"]["status"],
        decision_count=evaluation["evaluation"]["decision_count"],
        client_count=evaluation["evaluation"]["client_count"],
        mrr_cutoff=evaluation["evaluation"]["mrr_cutoff"],
        slices=evaluation["evaluation"]["slices"],
        history_stratification_status=evaluation["evaluation"]["history_stratification_status"],
        history_buckets=evaluation["evaluation"].get("history_buckets"),
    )
    system = {
        "schema": "system_measurement_reference_set_v1",
        "version": "1",
        "status": "NOT_APPLICABLE",
        "null_reason": "centralized replay performs no client-server transmission",
    }
    federated_metadata = None
    if regime == "R2A":
        # The measured payload bytes live in the run record, which this set references
        # rather than restating. The contract allows refs, not inline measurements.
        system = {
            "schema": "system_measurement_reference_set_v1",
            "version": "1",
            "status": "AVAILABLE",
            "refs": [ref(run_path, "mvp_run_record_v1", "federated_payload_byte_measurements")],
        }
        federated_metadata = {
            "rounds_completed": final_round,
            "clients_per_round": int(ctx.policy["pilot"]["clients_per_round"]),
            "pilot_population": len(ctx.prepared.population),
            "unique_clients_trained": run["population"]["measured_unique_clients_trained"],
            "aggregation_oracle_policy": run["totals"]["aggregation_oracle_policy"],
            "max_oracle_ratio": run["totals"]["max_oracle_ratio"],
            "max_oracle_abs_diff": run["totals"]["max_oracle_abs_diff"],
            "upload_bytes": run["communication"]["run_totals"]["upload_bytes"],
            "download_bytes": run["communication"]["run_totals"]["download_bytes"],
        }
    result = build_experiment_result(
        experiment_config=cfg,
        config_ref=ref(cfg_path, "experiment_config_v1"),
        git_sha=git("rev-parse", "HEAD"),
        state="SUCCEEDED",
        attempt=1,
        started_at_utc=run["started_at_utc"],
        ended_at_utc=run["ended_at_utc"],
        metrics=metric_records_from_t1_summary(summary),
        artifacts=[ref(run_path, "mvp_run_record_v1"), snapshot_ref],
        system_measurements=system,
        federated_metadata=federated_metadata,
    )
    result_path = f"{directory}/{name}.result.json"
    validate_result_for_reporting(result, source=Path(result_path).name)
    write_json(result_path, result, replace=True)
    return {
        "regime": regime,
        "run_id": result["run_id"],
        "config_ref": ref(cfg_path, "experiment_config_v1"),
        "result_ref": ref(result_path, "experiment_result_v1"),
        "headline": evaluation["headline"],
        "evaluation": evaluation["evaluation"],
    }


def stage_compare(ctx: Context) -> dict:
    """The matched-exposure comparison: identical membership, two declared lifecycles."""
    require_branch()
    verify_frozen_source(ctx, stage="compare")
    central = read_json(ctx.public_path("centralized_run.v1.json"))
    federated = read_json(ctx.public_path("federated_run.v1.json"))
    prepared = ctx.prepared

    if central["exposure_sha256"] != federated["exposure_sha256"]:
        raise StageError(
            "the two regimes did not see the same rows in the same order; this pair is not "
            "matched-exposure and must not be compared"
        )
    snapshot_ref = ref(ctx.public_path(SOURCE_FREEZE_NAME), "mvp_source_snapshot_v1")
    published = [
        _publish_pair_result(ctx, regime="R1", run=central, snapshot_ref=snapshot_ref),
        _publish_pair_result(ctx, regime="R2A", run=federated, snapshot_ref=snapshot_ref),
    ]

    # Quality retention through the existing reporter, on the isolated pair only.
    quality = build_quality_report(load_results(ROOT / ctx.pair_out))
    write_json(
        ctx.public_path("pilot_quality_retention.v1.json"),
        {
            "schema": "mvp_pilot_quality_retention_v1",
            "version": "1",
            "run": ctx.run,
            "generated_at_utc": now(),
            "scope": "PILOT_PAIR_ONLY_NOT_THE_PROJECT_R1_DENOMINATOR",
            "registry_directory": str(ctx.pair_out).replace("\\", "/"),
            "report": quality,
            "note": (
                "The R1 side of this ratio is the matched centralized pilot, not a final "
                "full-data R1 reference. It answers how much of this pilot's centralized "
                "quality FedAvg retained under identical exposure."
            ),
        },
        replace=True,
    )

    identity_fields = {
        "model_sha256": file_sha256(ROOT / "ppsi/models/session_gru.py"),
        "batch_spec_sha256": file_sha256(ROOT / "ppsi/models/batch_spec.py"),
        "common_init_sha256": read_json(ctx.public_path("preflight.v1.json"))[
            "common_initialization"
        ]["raw_file_sha256"],
        "data_manifest_sha256": prepared.data_manifest_sha256,
        "evaluation_membership_sha256": logical_digest(
            {
                "rows": prepared.validation.rows,
                "arrays": prepared.manifest["splits"]["VALIDATION"]["array_sha256"],
            }
        ),
        "evaluator_sha256": file_sha256(ROOT / "ppsi/evaluation/t1.py"),
        "exposure_sha256": central["exposure_sha256"],
        "seed": ctx.seed,
        "metric_id": "t1.next_distinct.mrr_at_20.macro",
        "score_convention": "RAW_NO_SUPPRESSION",
        "validation_decisions": prepared.validation.rows,
    }
    left = dict(identity_fields, value=published[0]["headline"]["value"])
    right = dict(identity_fields, value=published[1]["headline"]["value"])
    delta = metric_delta(left, right)

    baselines = read_json(ctx.public_path("baseline_summary.v1.json"))
    comparison = {
        "schema": "mvp_comparison_v1",
        "version": "1",
        "run": ctx.run,
        "generated_at_utc": now(),
        "comparison_label": ctx.policy["pilot"]["comparison_label"],
        "headline_metric": "t1.next_distinct.mrr_at_20.macro",
        "score_convention": "RAW_NO_SUPPRESSION",
        "validation_decisions": prepared.validation.rows,
        "support_clients": published[0]["headline"]["support_clients"],
        "identity": identity_fields,
        "regimes": {
            "centralized": {
                "run_id": published[0]["run_id"],
                "result_ref": published[0]["result_ref"],
                "config_ref": published[0]["config_ref"],
                "optimizer_lifecycle": central["optimizer_lifecycle"],
                "headline": published[0]["headline"],
                "rounds": central["totals"],
                "wall_seconds": sum(r["seconds"] for r in central["rounds"]),
            },
            "fedavg": {
                "run_id": published[1]["run_id"],
                "result_ref": published[1]["result_ref"],
                "config_ref": published[1]["config_ref"],
                "optimizer_lifecycle": federated["optimizer_lifecycle"],
                "headline": published[1]["headline"],
                "rounds": federated["totals"],
                "wall_seconds": federated["wall_seconds"],
                "communication": federated["communication"],
                "population": federated["population"],
            },
        },
        "difference": {
            "fedavg_minus_centralized": delta,
            "computed_between": "identical metric id, convention, membership and support",
        },
        "round_zero_agreement": {
            "centralized": next(
                e["headline"]["value"] for e in central["evaluations"] if e["server_round"] == 0
            ),
            "fedavg_round_zero_evaluated": any(
                e["server_round"] == 0 for e in federated["evaluations"]
            ),
        },
        "baseline_context": [
            {
                "variant": entry["variant"],
                "fit_scope": entry["fit_scope"],
                "value": entry["headline"]["value"],
            }
            for entry in baselines["variants"]
        ],
        "limitations": [
            (
                "This is a scoped T1 pilot on a 1,000-client population, not the final "
                "R1/R2A reference and not the original multi-task MVP commitment."
            ),
            (
                "Exposure is matched; optimization trajectories are not. Centralized keeps "
                "one Adam state for the run and FedAvg resets it every server round."
            ),
            (
                "The centralized side is not an all-data upper bound: it replays exactly "
                "the federated schedule."
            ),
            (
                "Measured bytes are model payload at the Flower message boundary, not "
                "total network traffic."
            ),
            "No differential privacy, secure aggregation or device isolation is claimed.",
            "TEST was never read, fitted on, or used for selection.",
        ],
    }
    write_json(ctx.public_path("comparison.v1.json"), comparison, replace=True)
    log(
        f"comparison: centralized {published[0]['headline']['value']:.5f} vs fedavg "
        f"{published[1]['headline']['value']:.5f} (delta {delta:+.5f})"
    )
    return comparison


# ---------------------------------------------------------------------------
# Stage: verify
# ---------------------------------------------------------------------------


def stage_verify(ctx: Context) -> dict:
    """Recheck every claim from the files themselves, and fail closed on any gap."""
    require_branch()
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: Any = None) -> None:
        checks.append({"check": name, "passed": bool(ok), "detail": detail})

    public = ROOT / ctx.public
    required = [
        "prepare_summary.v1.json",
        "preflight.v1.json",
        "baseline_summary.v1.json",
        "canary.v1.json",
        "centralized_run.v1.json",
        "federated_run.v1.json",
        "comparison.v1.json",
        "pilot_quality_retention.v1.json",
    ]
    missing = [name for name in required if not (public / name).is_file()]
    check("all_public_stage_outputs_present", not missing, {"missing": missing})

    # Every artifact_ref in every public file still hashes to what it recorded.
    broken: list[str] = []
    checked = 0

    def visit(node: Any) -> None:
        nonlocal checked
        if isinstance(node, dict):
            if node.get("schema") == "artifact_ref_v1" and "uri" in node:
                path = ROOT / node["uri"]
                checked += 1
                if not path.is_file() or file_sha256(path) != node["sha256"]:
                    broken.append(node["uri"])
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    for name in required:
        if (public / name).is_file():
            visit(read_json(ctx.public_path(name)))
    for path in sorted((ROOT / ctx.pair_out).glob("*.json")):
        visit(json.loads(path.read_text(encoding="utf-8")))
    check("artifact_reference_hashes_match", not broken, {"checked": checked, "broken": broken})

    prepared = ctx.prepared
    manifest = prepared.manifest
    drifted = [
        f"{split}/{name}"
        for split, body in manifest["splits"].items()
        for name, digest in body["array_sha256"].items()
        if raw_file_sha256(prepared.root / f"{split.lower()}_{name}.npy") != digest
    ]
    check("prepared_arrays_unchanged_since_preparation", not drifted, {"drifted": drifted})

    central = read_json(ctx.public_path("centralized_run.v1.json"))
    federated = read_json(ctx.public_path("federated_run.v1.json"))
    comparison = read_json(ctx.public_path("comparison.v1.json"))

    check(
        "exposure_digest_matches_between_regimes",
        central["exposure_sha256"] == federated["exposure_sha256"],
    )
    recomputed, stats = _exposure_records(ctx)
    check(
        "exposure_digest_reproduces_from_prepared_inputs",
        exposure_digest(recomputed) == central["exposure_sha256"],
        stats,
    )
    check(
        "scheduled_participations_and_unique_clients_reported_separately",
        stats["participations"]
        == int(ctx.policy["pilot"]["rounds"]) * int(ctx.policy["pilot"]["clients_per_round"])
        and stats["unique_clients"] <= len(prepared.population),
        {
            "population": len(prepared.population),
            "participations": stats["participations"],
            "unique_clients": stats["unique_clients"],
        },
    )
    check(
        "every_federated_round_passed_the_fedavg_oracle",
        all(r["aggregation_oracle_pass"] for r in federated["rounds"]),
        {"max_abs_diff": federated["totals"]["max_oracle_abs_diff"]},
    )
    check(
        "every_round_stayed_inside_its_derived_float32_bound",
        all(float(r["aggregation_oracle"]["worst_ratio"]) <= 1.0 for r in federated["rounds"]),
        {
            "policy": federated["totals"]["aggregation_oracle_policy"],
            "max_ratio_of_allowance_used": federated["totals"]["max_oracle_ratio"],
        },
    )
    check(
        "the_derived_bound_did_not_become_a_blanket_loose_tolerance",
        federated["totals"]["max_oracle_ratio"] <= 1.0
        and federated["totals"]["max_oracle_tight_ratio"] <= 1.0,
        {
            "max_ratio_against_authorised_bound": federated["totals"]["max_oracle_ratio"],
            "max_ratio_against_tighter_textbook_bound": federated["totals"][
                "max_oracle_tight_ratio"
            ],
        },
    )
    check(
        "oracle_diagnostics_carry_no_client_identity",
        all("client_id" not in json.dumps(r["aggregation_oracle"]) for r in federated["rounds"]),
    )
    check(
        "every_round_received_the_complete_reply_set",
        all(
            r["clients"] == int(ctx.policy["pilot"]["clients_per_round"])
            for r in federated["rounds"]
        ),
    )
    check(
        "measured_bytes_are_positive_in_both_directions",
        federated["communication"]["run_totals"]["upload_bytes"] > 0
        and federated["communication"]["run_totals"]["download_bytes"] > 0,
        federated["communication"]["run_totals"],
    )
    check(
        "byte_transmissions_match_participations",
        federated["communication"]["run_totals"]["upload_transmissions"] == stats["participations"]
        and federated["communication"]["run_totals"]["download_transmissions"]
        == stats["participations"],
    )

    # Round 0 is the same untrained state on both paths, so its metric must be identical.
    zero_central = next(e for e in central["evaluations"] if e["server_round"] == 0)
    federated_zero = [e for e in federated["evaluations"] if e["server_round"] == 0]
    check(
        "round_zero_predictions_identical_across_regimes",
        bool(federated_zero) and zero_central["ranks_sha256"] == federated_zero[0]["ranks_sha256"],
        {
            "centralized_ranks_sha256": zero_central["ranks_sha256"],
            "federated_ranks_sha256": federated_zero[0]["ranks_sha256"] if federated_zero else None,
        },
    )

    # The published headline must be recomputable from the stored raw ranks.
    meta = prepared.validation_metadata()
    recomputed_headlines = {}
    for label, run in (("centralized", central), ("fedavg", federated)):
        final = max(run["evaluations"], key=lambda e: e["server_round"])
        ranks = np.load(ROOT / final["ranks_private_uri"], allow_pickle=False)
        summary = evaluate_t1_ranks(
            ranks,
            meta["category_changed"],
            meta["client_ids"],
            train_history_counts=meta["train_history_counts"],
            mrr_cutoff=20,
        ).to_dict()
        recomputed_headlines[label] = headline_metric(summary)
        check(
            f"{label}_headline_recomputes_from_stored_ranks",
            abs(recomputed_headlines[label]["value"] - final["headline"]["value"]) < 1e-12,
            {
                "published": final["headline"]["value"],
                "recomputed": recomputed_headlines[label]["value"],
            },
        )
        check(
            f"{label}_ranks_are_raw_one_based",
            int(ranks.min()) >= 1 and int(ranks.max()) <= 588,
            {"min": int(ranks.min()), "max": int(ranks.max())},
        )
    check(
        "full_validation_support_is_the_frozen_count",
        comparison["validation_decisions"] == int(ctx.policy["pilot"]["full_validation_rows"]),
        {"decisions": comparison["validation_decisions"]},
    )

    # Privacy: no opaque client identity and no raw user id may reach a public file.
    import re

    leak_pattern = re.compile(r"client-v1-[0-9a-f]{64}")
    leaks = []
    for path in sorted(public.rglob("*.json")) + sorted((ROOT / ctx.baseline_out).rglob("*.json")):
        text = path.read_text(encoding="utf-8")
        if leak_pattern.search(text):
            leaks.append(str(path.relative_to(ROOT)).replace("\\", "/"))
    check("no_client_identity_in_public_outputs", not leaks, {"files": leaks})

    check(
        "no_pilot_result_entered_the_global_results_directory",
        not any(
            path.name.startswith(("centralized.", "fedavg."))
            for path in (ROOT / "artifacts/experiment-results").glob("*.json")
        ),
    )
    check(
        "test_examples_were_not_read",
        int(read_json(ctx.public_path("prepare_summary.v1.json"))["sealed_test"]["files_opened"])
        == 0,
        read_json(ctx.public_path("prepare_summary.v1.json"))["sealed_test"],
    )

    passed = all(entry["passed"] for entry in checks)
    record = {
        "schema": "mvp_verification_v1",
        "version": "1",
        "run": ctx.run,
        "generated_at_utc": now(),
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "failed_checks": [entry["check"] for entry in checks if not entry["passed"]],
    }
    write_json(ctx.public_path("verification.v1.json"), record, replace=True)
    if not passed:
        raise StageError(f"verification failed: {record['failed_checks']}")
    log(f"verification passed {len(checks)} checks")
    return record


# ---------------------------------------------------------------------------
# Stage: report
# ---------------------------------------------------------------------------


def stage_report(ctx: Context) -> dict:
    """Collect verified numbers for the report, notebook and slides. No new claims."""
    require_branch()
    verification = read_json(ctx.public_path("verification.v1.json"))
    if verification["status"] != "PASS":
        raise StageError("the report stage refuses to run on unverified results")
    comparison = read_json(ctx.public_path("comparison.v1.json"))
    baselines = read_json(ctx.public_path("baseline_summary.v1.json"))
    central = read_json(ctx.public_path("centralized_run.v1.json"))
    federated = read_json(ctx.public_path("federated_run.v1.json"))
    prepare = read_json(ctx.public_path("prepare_summary.v1.json"))

    table = [
        {
            "system": f"{entry['variant']} ({entry['fit_scope']} TRAIN)",
            "family": "count baseline",
            "t1_next_distinct_mrr_at_20_macro": entry["headline"]["value"],
            "support_clients": entry["headline"]["support_clients"],
        }
        for entry in baselines["variants"]
    ]
    for label, key in (("GRU centralized pilot", "centralized"), ("GRU FedAvg pilot", "fedavg")):
        block = comparison["regimes"][key]
        table.append(
            {
                "system": label,
                "family": "neural pilot",
                "t1_next_distinct_mrr_at_20_macro": block["headline"]["value"],
                "support_clients": block["headline"]["support_clients"],
            }
        )

    record = {
        "schema": "mvp_report_inputs_v1",
        "version": "1",
        "run": ctx.run,
        "generated_at_utc": now(),
        "scope_statement": (
            "A scoped T1 next-distinct-category pilot: a real matched centralized/FedAvg pair "
            "on a 1,000-client population, evaluated on all 438,185 frozen VALIDATION "
            "decisions. It is a preliminary demonstrator, not the original multi-task MVP "
            "commitment and not a Phase-1 freeze."
        ),
        "result_table": table,
        "headline_comparison": comparison["difference"],
        "communication": federated["communication"]["run_totals"],
        "communication_boundary": federated["communication"]["measurement_boundary"],
        "population": federated["population"],
        "runtime_seconds": {
            "centralized": sum(r["seconds"] for r in central["rounds"]),
            "federated": federated["wall_seconds"],
        },
        "training_curves": {
            "centralized": [
                {"server_round": r["server_round"], "mean_train_loss": r["mean_train_loss"]}
                for r in central["rounds"]
            ],
            "fedavg": [
                {"server_round": r["server_round"], "mean_train_loss": r["mean_train_loss"]}
                for r in federated["rounds"]
            ],
        },
        "validation_curve": {
            "centralized": [
                {"server_round": e["server_round"], "value": e["headline"]["value"]}
                for e in central["evaluations"]
            ],
            "fedavg": [
                {"server_round": e["server_round"], "value": e["headline"]["value"]}
                for e in federated["evaluations"]
            ],
        },
        "data": {
            "c1_users": ctx.policy["data"]["base_c1_users"],
            "example_slice_users": ctx.policy["data"]["example_slice_users"],
            "train_decisions": ctx.policy["data"]["t1_train_rows"],
            "validation_decisions": ctx.policy["data"]["t1_validation_rows"],
            "pilot_train_rows": prepare["splits"]["TRAIN"]["rows"],
            "pilot_population": prepare["population"]["selected"],
        },
        "limitations": comparison["limitations"],
        "verification_status": verification["status"],
        "verification_checks": len(verification["checks"]),
    }
    write_json(ctx.public_path("report_inputs.v1.json"), record, replace=True)
    return record


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

STAGES = {
    "preflight": stage_preflight,
    "baselines": stage_baselines,
    "canary": stage_canary,
    "centralized": stage_centralized,
    "federated": stage_federated,
    "compare": stage_compare,
    "verify": stage_verify,
    "report": stage_report,
}

MANDATORY_SEQUENCE = (
    "preflight",
    "baselines",
    "canary",
    "centralized",
    "federated",
    "compare",
    "verify",
    "report",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one stage of the scoped T1 MVP.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--stage", required=True, choices=(*STAGES, "all"))
    args = parser.parse_args(argv)

    ctx = Context(args.config, args.run)
    stages = MANDATORY_SEQUENCE if args.stage == "all" else (args.stage,)
    executed = []
    for name in stages:
        log(f"=== stage {name} ===")
        started = time.perf_counter()
        STAGES[name](ctx)
        executed.append({"stage": name, "seconds": round(time.perf_counter() - started, 1)})
        log(f"=== stage {name} finished in {executed[-1]['seconds']}s ===")
    print(json.dumps({"run": ctx.run, "executed_stages": executed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
