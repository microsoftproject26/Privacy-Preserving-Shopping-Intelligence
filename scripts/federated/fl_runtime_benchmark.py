"""Shared S2-PR-01/#35 scale and S2-PR-03/#37 profiling driver.

This is a *runtime* benchmark. It measures how a fixed, preregistered federated
workload behaves as the declared client population and worker concurrency change.
It is explicitly not science: no result here may be read as an R2A finding, a model
quality claim, or a capacity guarantee for another model or machine.

Three quantities are kept independent and are never conflated:

* ``N`` -- declared population, the number of virtual Flower SuperNodes.
* ``M`` -- participants per round, the clients that actually receive train messages.
* ``C`` -- backend concurrency, the CPU cap given to the Ray backend.

Each trial runs in its own supervised subprocess so a resource guard can stop one
trial without taking down the session. The parent samples the child's process tree
and refuses to escalate after any resource failure.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import polars as pl
import psutil
import torch
from flwr.app import ArrayRecord, ConfigRecord, Message, MessageType, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common import Context
from flwr.serverapp import Grid, ServerApp
from flwr.simulation import run_simulation
from torch import optim

from ppsi.federated.runtime_monitor import (
    GIB,
    MonitorLimits,
    ProcessTreeMonitor,
    summarize_samples,
)
from ppsi.federated.sampling import DEFAULT_SAMPLER_VERSION, build_trace, sample_clients
from ppsi.federated.task_examples import (
    load_t1_category_spec,
    load_t1_client_counts,
    make_t1_smoke_batches,
    prepare_t1_smoke_slices,
    verify_t1_task_example_file,
    verify_vocabulary,
)
from ppsi.training.core import LocalTrainerCore
from ppsi.training.fixtures import default_batch_spec
from ppsi.training.flower import ContributingRowsSmokeWeightPolicy, FlowerLocalAdapter
from ppsi.training.identity import file_sha256
from ppsi.training.objective import ContractSmokeObjective
from ppsi.training.state import shared_state_digest
from scripts.experiments.schemas import validate_experiment_config, validate_experiment_result
from scripts.federated.fl_real_smoke import (
    RealDataTracingFedAvg,
    build_smoke_model,
    make_artifact_ref,
    server_evaluate,
    server_round_to_sampler_round,
)
from scripts.federated.fl_synthetic_smoke import SmokeValidationError, get_digest

# ---------------------------------------------------------------------------
# Canonical paths
# ---------------------------------------------------------------------------

_WORKLOAD_CONFIG = Path("config/federated/fl_runtime_workload.v1.json")
_SCALE_CONFIG = Path("config/federated/fl_scale_matrix.v1.json")
_PROFILE_CONFIG = Path("config/federated/fl_profile_matrix.v1.json")
_CONVERGENCE_CONFIG = Path("config/federated/convergence_policy.v1.json")

_SCALE_EVIDENCE_DIR = Path("docs/evidence/s2-pr-01")
_PROFILE_EVIDENCE_DIR = Path("docs/evidence/s2-pr-03")
_RUNTIME_INPUTS = _SCALE_EVIDENCE_DIR / "runtime_inputs.v1.json"
_RUNTIME_INITIALIZATION = _SCALE_EVIDENCE_DIR / "runtime_initialization.v1.json"
_SCALE_SUMMARY = _SCALE_EVIDENCE_DIR / "scale_summary.v1.json"
_PROFILE_SUMMARY = _PROFILE_EVIDENCE_DIR / "profile_summary.v1.json"
_RUNTIME_RECOMMENDATIONS = _PROFILE_EVIDENCE_DIR / "runtime_recommendations.v1.json"
_PRIVATE_ROOT = Path("artifacts/federated-runtime")
_RESULT_ROOT = Path("artifacts/experiment-results")

_INPUT_EVIDENCE = Path("docs/evidence/s1-pr-07/task_example_inputs.v1.json")
_EXPECTED_MANIFEST_LOGICAL_SHA = "a96fb92fc6c43e458f9d8692491c7927e49d17741cb37e72d303420734820764"
_EXPECTED_MANIFEST_COUNT = 388_789

_PURPOSE = "NON_SCIENTIFIC_FL_RUNTIME_BENCHMARK"


class RuntimeBenchmarkError(RuntimeError):
    """Raised when a runtime benchmark precondition or invariant fails."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads((_REPO_ROOT / path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    target = _REPO_ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return file_sha256(target)


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        shell=False,
    )
    return result.stdout.strip()


def trial_id(population: int, participants: int, concurrency: int, repetition: int) -> str:
    return f"n{population}-m{participants}-c{concurrency}-rep{repetition}"


def capture_hardware() -> dict[str, Any]:
    """Record the machine actually used, never a remembered specification."""
    import platform

    import flwr
    import numpy
    import ray

    vm = psutil.virtual_memory()
    import shutil
    import tempfile

    return {
        "schema": "runtime_hardware_identity_v1",
        "version": "1",
        "logical_cpus": psutil.cpu_count(logical=True),
        "physical_cpus": psutil.cpu_count(logical=False),
        "total_ram_bytes": vm.total,
        "available_ram_bytes": vm.available,
        "repo_volume_free_bytes": shutil.disk_usage(_REPO_ROOT).free,
        "temp_volume_free_bytes": shutil.disk_usage(tempfile.gettempdir()).free,
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "flwr": flwr.__version__,
        "ray": ray.__version__,
        "polars": pl.__version__,
        "numpy": numpy.__version__,
        "psutil": psutil.__version__,
        "gpu_memory": "NOT_APPLICABLE",
        "device": "cpu",
    }


def choose_scale_concurrency(
    hardware: dict[str, Any], scale_config: dict[str, Any]
) -> dict[str, Any]:
    """Freeze one concurrency for the whole scale matrix, before any run."""
    policy = scale_config["concurrency_policy"]
    safety = scale_config["safety"]
    logical = hardware["logical_cpus"]
    total_gib = hardware["total_ram_bytes"] / GIB
    available_gib = hardware["available_ram_bytes"] / GIB

    if (
        logical >= policy["default_min_logical_cpus"]
        and total_gib >= policy["default_min_total_ram_gib"]
        and available_gib >= policy["default_min_available_ram_gib"]
    ):
        return {"concurrency": policy["default"], "admissible": True, "reason": "DEFAULT_ADMITTED"}
    if (
        total_gib >= safety["min_total_ram_gib"]
        and available_gib >= safety["min_available_ram_gib"]
    ):
        return {
            "concurrency": policy["fallback"],
            "admissible": True,
            "reason": "FALLBACK_ADMITTED_INSUFFICIENT_RAM_FOR_DEFAULT",
        }
    return {
        "concurrency": None,
        "admissible": False,
        "reason": (
            f"CAPACITY_BLOCKER: total {total_gib:.2f} GiB / available {available_gib:.2f} GiB "
            f"below the minimum {safety['min_total_ram_gib']} / {safety['min_available_ram_gib']} GiB"
        ),
    }


def check_disk(hardware: dict[str, Any], scale_config: dict[str, Any]) -> tuple[bool, str]:
    need = scale_config["safety"]["min_free_disk_gib"] * GIB
    if hardware["repo_volume_free_bytes"] < need:
        return False, "insufficient free space on the repository volume"
    if hardware["temp_volume_free_bytes"] < need:
        return False, "insufficient free space on the temporary volume"
    return True, "OK"


# Every file whose content can change what a trial measures. Untracked new code is
# included by content hash, because a base commit SHA cannot identify it.
_EXECUTED_SOURCE_FILES = (
    "scripts/federated/fl_runtime_benchmark.py",
    "ppsi/federated/runtime_monitor.py",
    "ppsi/federated/convergence.py",
    "scripts/federated/fl_real_smoke.py",
    "scripts/federated/fl_synthetic_smoke.py",
    "ppsi/federated/sampling.py",
    "ppsi/federated/task_examples.py",
    "ppsi/training/core.py",
    "ppsi/training/flower.py",
    "ppsi/training/objective.py",
    "ppsi/training/fixtures.py",
    "ppsi/training/state.py",
    "config/federated/fl_runtime_workload.v1.json",
    "config/federated/fl_scale_matrix.v1.json",
    "config/federated/fl_profile_matrix.v1.json",
)


def build_source_snapshot() -> dict[str, Any]:
    """Hash every source file that can change a measurement, before running it."""
    files: dict[str, str] = {}
    for rel in _EXECUTED_SOURCE_FILES:
        path = _REPO_ROOT / rel
        if not path.is_file():
            raise RuntimeBenchmarkError(f"executed source file is missing: {rel}")
        files[rel] = file_sha256(path)
    return {
        "schema": "fl_runtime_source_snapshot_v1",
        "version": "1",
        "execution_provenance": "WORKING_TREE_BUNDLE_RUN",
        "git_head": _git_head(),
        "files_sha256": files,
        "note": (
            "The driver and monitor are untracked working-tree code at run time, so the "
            "git head alone does not identify them; each file is pinned by content."
        ),
    }


# ---------------------------------------------------------------------------
# Stage: prepare
# ---------------------------------------------------------------------------


def _verify_declared_inputs(workload: dict[str, Any]) -> dict[str, Any]:
    """Require full SHA and row-count equality against the merged S1-PR-07 evidence."""
    evidence = _load_json(_INPUT_EVIDENCE)
    declared: dict[str, Any] = {}

    def _check(label: str, key: str, *, expected_rows: int | None = None) -> None:
        entry = evidence.get(key)
        if not isinstance(entry, dict) or "path" not in entry or "sha256" not in entry:
            raise RuntimeBenchmarkError(
                f"merged input evidence does not declare a usable {label} reference"
            )
        path = _REPO_ROOT / entry["path"]
        if not path.is_file():
            raise RuntimeBenchmarkError(f"declared {label} input is missing: {entry['path']}")
        actual = file_sha256(path)
        if actual != entry["sha256"]:
            raise RuntimeBenchmarkError(
                f"{label} SHA-256 mismatch: evidence {entry['sha256']}, actual {actual}. "
                "This is a blocker; expected hashes are never edited to match a file."
            )
        if expected_rows is not None:
            rows = pl.scan_parquet(path).select(pl.len()).collect().item()
            if rows != expected_rows:
                raise RuntimeBenchmarkError(
                    f"{label} row count mismatch: expected {expected_rows}, actual {rows}"
                )
        declared[label] = {"uri": entry["path"], "sha256": actual}

    _check("t1_train", "t1_train", expected_rows=evidence["t1_train_row_count"])
    _check("t1_validation", "t1_validation", expected_rows=evidence["t1_validation_row_count"])
    _check("cohort_manifest", "cohort_manifest")

    # The frozen vocabulary keeps the upstream raw-file identity recorded by S1.
    vocab_entry = evidence["vocabulary"]
    actual_vocab = verify_vocabulary(_REPO_ROOT / vocab_entry["path"])
    if actual_vocab != vocab_entry["sha256"]:
        raise RuntimeBenchmarkError("vocabulary SHA-256 mismatch against merged evidence")
    declared["vocabulary"] = {"uri": vocab_entry["path"], "sha256": actual_vocab}

    if evidence.get("sealed_test_accessed") is not False:
        raise RuntimeBenchmarkError(
            "merged input evidence does not assert sealed_test_accessed=false"
        )
    return declared


def _load_base_manifest(workload: dict[str, Any]) -> tuple[pl.DataFrame, str]:
    """Read the existing private #19 ClientManifest and verify its logical hash."""
    from ppsi.federated.clients import manifest_content_sha256

    path = _REPO_ROOT / workload["protected_base_manifest"]
    if not path.is_file():
        raise RuntimeBenchmarkError(
            f"the existing #19 client manifest is missing at {workload['protected_base_manifest']}; "
            "regenerate it cohort-only with build_client_manifest before benchmarking"
        )
    df = pl.read_parquet(path)
    logical = manifest_content_sha256(df)
    if logical != _EXPECTED_MANIFEST_LOGICAL_SHA:
        raise RuntimeBenchmarkError(
            f"client manifest logical hash mismatch: expected {_EXPECTED_MANIFEST_LOGICAL_SHA}, "
            f"got {logical}"
        )
    if df.height != _EXPECTED_MANIFEST_COUNT:
        raise RuntimeBenchmarkError(
            f"client manifest row count mismatch: expected {_EXPECTED_MANIFEST_COUNT}, got {df.height}"
        )
    return df, logical


def _describe_counts(counts: list[int]) -> dict[str, Any]:
    """TaskExample counts, deliberately not raw-event history counts."""
    ordered = sorted(counts)
    n = len(ordered)

    def _pct(p: float) -> int:
        return ordered[min(n - 1, max(0, math.ceil(p * n) - 1))]

    return {
        "measure": "T1_TRAIN_TASKEXAMPLE_COUNTS_NOT_RAW_EVENT_HISTORY",
        "client_count": n,
        "min": ordered[0],
        "max": ordered[-1],
        "mean": sum(ordered) / n,
        "p50": _pct(0.50),
        "p90": _pct(0.90),
        "p95": _pct(0.95),
        "p99": _pct(0.99),
    }


def stage_prepare(workload: dict[str, Any], scale_config: dict[str, Any]) -> dict[str, Any]:
    """Verify inputs, build the benchmark pool and materialize the shared slices once."""
    print("[prepare] verifying declared inputs against merged S1-PR-07 evidence")
    inputs = _verify_declared_inputs(workload)

    train_uri = inputs["t1_train"]["uri"]
    val_uri = inputs["t1_validation"]["uri"]
    category_count, valid_codes = load_t1_category_spec(_REPO_ROOT / inputs["vocabulary"]["uri"])
    if category_count != workload["category_count"]:
        raise RuntimeBenchmarkError("vocabulary category count does not match the workload config")

    print("[prepare] verifying T1 TaskExample files with the existing validator")
    evidence = _load_json(_INPUT_EVIDENCE)
    verify_t1_task_example_file(
        _REPO_ROOT / train_uri,
        expected_sha_prefix=inputs["t1_train"]["sha256"][:16],
        expected_row_count=int(evidence["t1_train_row_count"]),
        expected_split="TRAIN",
        file_label="t1_train",
        expected_category_count=category_count,
    )
    verify_t1_task_example_file(
        _REPO_ROOT / val_uri,
        expected_sha_prefix=inputs["t1_validation"]["sha256"][:16],
        expected_row_count=int(evidence["t1_validation_row_count"]),
        expected_split="VALIDATION",
        file_label="t1_validation",
        expected_category_count=category_count,
    )

    manifest_df, manifest_logical_sha = _load_base_manifest(workload)
    base_ids = set(manifest_df["client_id"].to_list())

    print("[prepare] loading T1 client example counts")
    counts_df = load_t1_client_counts(_REPO_ROOT / train_uri, _REPO_ROOT / val_uri, base_ids)
    train_counts = counts_df.filter(pl.col("t1_train_example_count") > 0)
    stats_before_filter = _describe_counts(train_counts["t1_train_example_count"].to_list())

    minimum = int(workload["minimum_train_examples_for_benchmark"])
    eligible = (
        train_counts.filter(pl.col("t1_train_example_count") >= minimum)
        .select("client_id")
        .to_series()
        .to_list()
    )
    eligible.sort()
    excluded = train_counts.height - len(eligible)
    print(
        f"[prepare] benchmark pool: {len(eligible)} clients with >= {minimum} T1 TRAIN examples "
        f"({excluded} excluded from the benchmark only)"
    )

    points = list(scale_config["population_points"])
    largest = max(points)
    pool_sufficient = len(eligible) >= largest

    populations: dict[int, list[str]] = {}
    if pool_sufficient:
        for n in points:
            populations[n] = eligible[:n]

    # Six deterministic round samples per population using the existing #19 sampler.
    clients_per_round = int(workload["clients_per_round"])
    num_rounds = int(workload["num_rounds"])
    seed = int(workload["seed"])
    samples: dict[str, dict[str, Any]] = {}
    union: set[str] = set()
    traces: list[dict[str, Any]] = []
    for n, pool in populations.items():
        per_round: dict[str, Any] = {}
        for server_round in range(1, num_rounds + 1):
            round_index = server_round_to_sampler_round(server_round)
            result = sample_clients(pool, seed, round_index, clients_per_round)
            per_round[str(round_index)] = {
                "selected_client_ids": result.selected_client_ids,
                "selected_digest": result.selected_digest,
            }
            union.update(result.selected_client_ids)
            traces.append(
                {
                    "population": n,
                    **build_trace(
                        result,
                        sampler_version=DEFAULT_SAMPLER_VERSION,
                        experiment_seed=seed,
                        round_index=round_index,
                        eligible_pool_count=len(pool),
                        clients_per_round=clients_per_round,
                    ).to_dict(),
                }
            )
        samples[str(n)] = per_round

    prepared: dict[str, Any] = {}
    if union:
        union_ids = sorted(union)
        print(f"[prepare] materializing one shared slice for {len(union_ids)} union clients")
        train_out = Path(workload["train_slice_output"])
        val_out = Path(workload["validation_slice_output"])
        train_sha, val_sha = prepare_t1_smoke_slices(
            _REPO_ROOT / train_uri,
            _REPO_ROOT / val_uri,
            union_ids,
            category_count=category_count,
            valid_codes=valid_codes,
            max_train_examples_per_client=int(workload["examples_per_client"]),
            max_validation_examples=int(workload["validation_example_limit"]),
            train_output_path=_REPO_ROOT / train_out,
            validation_output_path=_REPO_ROOT / val_out,
        )
        # Every selected client must execute exactly the same amount of work.
        slice_df = pl.read_parquet(_REPO_ROOT / train_out)
        per_client = slice_df.group_by("client_id").len()
        expected = int(workload["examples_per_client"])
        bad = per_client.filter(pl.col("len") != expected)
        if bad.height:
            raise RuntimeBenchmarkError(
                f"{bad.height} prepared clients do not hold exactly {expected} TRAIN rows"
            )
        if set(per_client["client_id"].to_list()) != set(union_ids):
            raise RuntimeBenchmarkError("prepared TRAIN slice does not cover the selected union")
        val_df = pl.read_parquet(_REPO_ROOT / val_out)
        if val_df.height != int(workload["validation_example_limit"]):
            raise RuntimeBenchmarkError(
                f"prepared VALIDATION slice holds {val_df.height} rows, expected "
                f"{workload['validation_example_limit']}"
            )
        prepared = {
            "train_slice": {"uri": train_out.as_posix(), "sha256": train_sha},
            "validation_slice": {"uri": val_out.as_posix(), "sha256": val_sha},
            "union_client_count": len(union_ids),
            "train_rows_per_client": expected,
            "validation_rows": val_df.height,
        }
        # Private population manifest and sampling trace; never overwrite #19/#20 outputs.
        population_path = _REPO_ROOT / workload["population_output"]
        population_path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(
            {
                "population": [n for n, pool in populations.items() for _ in pool],
                "client_id": [cid for pool in populations.values() for cid in pool],
            }
        ).write_parquet(population_path)
        trace_path = _REPO_ROOT / workload["sampling_trace_output"]
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(
            "\n".join(json.dumps(t, sort_keys=True) for t in traces) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        del slice_df, per_client, val_df
    del counts_df, train_counts, manifest_df, base_ids

    inputs_evidence = {
        "schema": "fl_runtime_inputs_v1",
        "version": "1",
        "purpose": _PURPOSE,
        "scientific": False,
        "source_inputs": inputs,
        "base_client_manifest": {
            "uri": workload["protected_base_manifest"],
            "logical_content_sha256": manifest_logical_sha,
            "client_count": _EXPECTED_MANIFEST_COUNT,
            "note": (
                "Frozen T1 examples are a smaller measurement slice of C1; these "
                f"{_EXPECTED_MANIFEST_COUNT} users are not the T1 example population."
            ),
        },
        "taskexample_counts_before_benchmark_filter": stats_before_filter,
        "benchmark_pool": {
            "minimum_train_examples": minimum,
            "eligible_client_count": len(eligible),
            "excluded_from_benchmark_only": excluded,
            "population_points": points,
            "pool_sufficient_for_largest_point": pool_sufficient,
            "order": "ascending_opaque_client_id_nested_prefixes",
            "note": (
                "An explicit benchmark subset for comparable runtime work. Not a new "
                "cohort, not a FedAvg eligibility policy, and never TEST-informed."
            ),
        },
        "population_selection_digests": {
            n: {r: s["selected_digest"] for r, s in rounds.items()} for n, rounds in samples.items()
        },
        "prepared_slices": prepared,
        "workload_ref": {
            "uri": _WORKLOAD_CONFIG.as_posix(),
            "sha256": file_sha256(_REPO_ROOT / _WORKLOAD_CONFIG),
        },
    }
    inputs_sha = _write_json(_RUNTIME_INPUTS, inputs_evidence)

    # Deterministic seed-13 smoke initialization proof; never a scientific claim.
    spec = default_batch_spec()
    torch.manual_seed(seed)
    model = build_smoke_model(category_count, spec)
    initial_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    init_evidence = {
        "schema": "fl_runtime_initialization_v1",
        "version": "1",
        "purpose": _PURPOSE,
        "scientific": False,
        "seed": seed,
        "category_count": category_count,
        "model_id": workload["model_id"],
        "initial_state_digest": get_digest(initial_state),
        "parameter_keys": sorted(initial_state),
        "note": (
            "Smoke fixture initialization for runtime measurement only. This is not a "
            "final scientific CommonInitialization artifact."
        ),
    }
    init_sha = _write_json(_RUNTIME_INITIALIZATION, init_evidence)

    private_samples = _REPO_ROOT / _PRIVATE_ROOT / "population_samples.json"
    private_samples.parent.mkdir(parents=True, exist_ok=True)
    private_samples.write_text(
        json.dumps(
            {"populations": {k: v for k, v in samples.items()}, "pool_size": len(eligible)},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
        newline="\n",
    )

    return {
        "inputs_sha256": inputs_sha,
        "initialization_sha256": init_sha,
        "eligible_pool_count": len(eligible),
        "pool_sufficient": pool_sufficient,
        "populations": {int(k): len(v) for k, v in populations.items()},
        "prepared": prepared,
        "category_count": category_count,
        "initial_state_digest": init_evidence["initial_state_digest"],
    }


# ---------------------------------------------------------------------------
# Runtime strategy: population N, participants M, concurrency C stay separate
# ---------------------------------------------------------------------------


class RuntimeFedAvg(RealDataTracingFedAvg):
    """Narrow strategy subclass that routes by logical identity, not by random draw.

    The completed #20 strategy samples nodes through ``FedAvg.configure_train`` and
    then relabels whichever messages come back. That is fine when the population and
    the participant count are the same number, but it cannot express N != M. This
    subclass builds exactly M messages itself and addresses them to the nodes that
    the frozen sampler actually chose, while leaving the parent's aggregation and
    oracle behaviour untouched.
    """

    def __init__(
        self,
        *,
        population_ids: list[str],
        node_registration_timeout: float,
        train_slice_path: str,
        learning_rate: float,
        momentum: float,
        local_epochs: int,
        batch_size: int,
        examples_per_client: int,
        seed: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.population_ids = list(population_ids)
        self.node_registration_timeout = float(node_registration_timeout)
        self.train_slice_path = train_slice_path
        self.learning_rate = float(learning_rate)
        self.momentum = float(momentum)
        self.local_epochs = int(local_epochs)
        self.batch_size = int(batch_size)
        self.examples_per_client = int(examples_per_client)
        self.seed = int(seed)
        self._node_by_logical_id: dict[str, int] = {}
        self.stage_timings: list[dict[str, Any]] = []
        self._configure_end_ns: int | None = None

    # -- node map ----------------------------------------------------------

    def bind_population_to_nodes(self, grid: Grid) -> dict[str, Any]:
        """Wait for exactly N nodes, then freeze one logical-to-node map for the run.

        The node ids are an ephemeral transport detail of this simulation. They are
        never a scientific identity and are never derived from ``hash()``.
        """
        expected = len(self.population_ids)
        deadline = time.monotonic() + self.node_registration_timeout
        observed: list[int] = []
        while time.monotonic() < deadline:
            observed = sorted(set(grid.get_node_ids()))
            if len(observed) >= expected:
                break
            time.sleep(0.25)
        if len(observed) != expected:
            raise SmokeValidationError(
                f"expected exactly {expected} registered SuperNodes, observed {len(observed)}; "
                "a partially registered population is never treated as the declared population"
            )
        self._node_by_logical_id = dict(zip(self.population_ids, observed, strict=True))
        return {
            "declared_population": expected,
            "registered_nodes": len(observed),
            "map_frozen": True,
        }

    # -- configure ---------------------------------------------------------

    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ):
        configure_start_ns = time.perf_counter_ns()
        if not self._node_by_logical_id:
            self.bind_population_to_nodes(grid)

        state_dict = arrays.to_torch_state_dict()
        current_digest = get_digest(state_dict)
        if (
            server_round > 1
            and self.previous_aggregate_digest is not None
            and current_digest != self.previous_aggregate_digest
        ):
            raise SmokeValidationError(
                f"Redistribution failure in round {server_round}: "
                f"expected {self.previous_aggregate_digest}, got {current_digest}"
            )

        round_index = server_round_to_sampler_round(server_round)
        selected = list(self.selected_ids_by_round[round_index])
        if len(selected) != self.expected_clients:
            raise SmokeValidationError(
                f"round {server_round}: sampler produced {len(selected)} ids, "
                f"expected {self.expected_clients}"
            )
        if len(set(selected)) != len(selected):
            raise SmokeValidationError(f"round {server_round}: duplicate selected client ids")
        unknown = [cid for cid in selected if cid not in self._node_by_logical_id]
        if unknown:
            raise SmokeValidationError(
                f"round {server_round}: {len(unknown)} selected ids are outside the population"
            )

        messages: list[Message] = []
        for logical_id in selected:
            # Every message carries its own config object; sharing one would let the
            # last write overwrite every recipient's client id.
            client_config = ConfigRecord(
                {
                    "logical_client_id": logical_id,
                    "server_round": server_round,
                    "server_digest": current_digest,
                    "train_slice_path": self.train_slice_path,
                    "learning_rate": self.learning_rate,
                    "momentum": self.momentum,
                    "local_epochs": self.local_epochs,
                    "batch_size": self.batch_size,
                    "examples_per_client": self.examples_per_client,
                    "seed": self.seed,
                }
            )
            content = RecordDict(
                {
                    self.arrayrecord_key: ArrayRecord(copy.deepcopy(arrays.to_torch_state_dict())),
                    self.configrecord_key: client_config,
                }
            )
            messages.append(
                grid.create_message(
                    content=content,
                    message_type=MessageType.TRAIN,
                    dst_node_id=self._node_by_logical_id[logical_id],
                    group_id=str(server_round),
                )
            )

        # Parent-compatible trace entry so aggregate_train needs no second oracle.
        self.tracing_log.append(
            {
                "server_round": server_round,
                "sampler_round_index": round_index,
                "selection_digest": self.selection_digests_by_round[round_index],
                "server_input_digest": current_digest,
                "selected_client_ids": list(selected),
                "selected_client_count": len(selected),
                "clients": [],
                "aggregation_oracle_pass": False,
                "aggregated_digest": None,
                "contributing_examples": 0,
                "max_abs_diff": None,
                "node_ids": [self._node_by_logical_id[cid] for cid in selected],
            }
        )
        self._configure_end_ns = time.perf_counter_ns()
        self.stage_timings.append(
            {
                "server_round": server_round,
                "configure_start_ns": configure_start_ns,
                "configure_end_ns": self._configure_end_ns,
            }
        )
        return messages

    # -- aggregate ---------------------------------------------------------

    def aggregate_train(self, server_round: int, replies):
        aggregate_entry_ns = time.perf_counter_ns()
        replies_list = list(replies)
        entry = self.tracing_log[-1]
        expected_ids = set(entry["selected_client_ids"])
        node_for_id = {
            cid: nid
            for cid, nid in zip(entry["selected_client_ids"], entry["node_ids"], strict=True)
        }

        if len(replies_list) != self.expected_clients:
            raise SmokeValidationError(
                f"round {server_round}: expected exactly {self.expected_clients} replies, "
                f"received {len(replies_list)}; partial aggregation is never accepted"
            )
        seen: set[str] = set()
        client_timings: list[dict[str, Any]] = []
        for msg in replies_list:
            if msg.has_error():
                raise SmokeValidationError(f"Client error in round {server_round}: {msg.error}")
            conf = msg.content.configs_records["config"]
            met = msg.content.metrics_records["metrics"]
            logical_id = str(conf.get("logical_client_id", ""))
            if logical_id not in expected_ids:
                raise SmokeValidationError(
                    f"round {server_round}: reply from an unexpected logical client"
                )
            if logical_id in seen:
                raise SmokeValidationError(f"round {server_round}: duplicate reply for one client")
            seen.add(logical_id)
            if msg.metadata.src_node_id != node_for_id[logical_id]:
                raise SmokeValidationError(
                    f"round {server_round}: reply arrived from a node this client was not mapped to"
                )
            weight = met["num-examples"]
            if isinstance(weight, bool) or int(weight) != weight or int(weight) <= 0:
                raise SmokeValidationError(
                    f"round {server_round}: aggregation weight must be a positive integer"
                )
            if int(weight) != self.examples_per_client:
                raise SmokeValidationError(
                    f"round {server_round}: expected weight {self.examples_per_client}, "
                    f"got {int(weight)}"
                )
            loss = float(conf.get("local_train_loss", float("nan")))
            if not math.isfinite(loss):
                raise SmokeValidationError(f"round {server_round}: non-finite local loss")
            state = msg.content.parameters_records["arrays"].to_torch_state_dict()
            for key, tensor in state.items():
                if not torch.isfinite(tensor).all():
                    raise SmokeValidationError(
                        f"round {server_round}: non-finite values in updated tensor {key}"
                    )
            # Recompute with the same packing digest the adapter used, so this verifies
            # the client's claim rather than comparing two different hash functions.
            recomputed = shared_state_digest(state)
            claimed = str(conf.get("updated_digest", ""))
            if recomputed != claimed:
                raise SmokeValidationError(
                    f"round {server_round}: client update digest does not match its own tensors"
                )
            client_timings.append(
                {
                    "data_load_seconds": float(conf.get("data_load_seconds", 0.0)),
                    "local_fit_seconds": float(conf.get("local_fit_seconds", 0.0)),
                    "client_start_ns": int(conf.get("client_start_ns", 0)),
                    "client_end_ns": int(conf.get("client_end_ns", 0)),
                }
            )
        if seen != expected_ids:
            raise SmokeValidationError(
                f"round {server_round}: the reply id set differs from the selected id set"
            )

        arrays, metrics = super().aggregate_train(server_round, replies_list)
        aggregate_end_ns = time.perf_counter_ns()

        if entry["contributing_examples"] != self.expected_clients * self.examples_per_client:
            raise SmokeValidationError(
                f"round {server_round}: contributing examples "
                f"{entry['contributing_examples']} != "
                f"{self.expected_clients * self.examples_per_client}"
            )
        if not entry["aggregation_oracle_pass"]:
            raise SmokeValidationError(f"round {server_round}: aggregation oracle did not pass")

        stage = self.stage_timings[-1]
        stage.update(
            aggregate_entry_ns=aggregate_entry_ns,
            aggregate_end_ns=aggregate_end_ns,
            client_timings=client_timings,
        )
        return arrays, metrics


# ---------------------------------------------------------------------------
# Client callback factory: captures immutable values only
# ---------------------------------------------------------------------------


def build_runtime_client_app(category_count: int) -> ClientApp:
    """Build a ClientApp whose closure holds no frames and no population tensors."""
    app = ClientApp()

    @app.train()
    def runtime_train(message: Message, context: Context) -> Message:
        client_start_ns = time.perf_counter_ns()
        config_rec = message.content.configs_records["config"]
        logical_client_id = str(config_rec["logical_client_id"])
        server_round = int(config_rec["server_round"])
        seed = int(config_rec["seed"])
        expected_rows = int(config_rec["examples_per_client"])

        torch.manual_seed(seed)
        torch.set_num_threads(1)

        state_dict = message.content.parameters_records["arrays"].to_torch_state_dict()
        received_digest = get_digest(state_dict)
        if received_digest != str(config_rec["server_digest"]):
            raise SmokeValidationError("client received a state that is not the announced one")

        load_start_ns = time.perf_counter_ns()
        # Read only this client's rows out of the small prepared slice.
        client_df = (
            pl.scan_parquet(str(config_rec["train_slice_path"]))
            .filter(pl.col("client_id") == logical_client_id)
            .collect()
        )
        if client_df.height != expected_rows:
            raise SmokeValidationError(
                f"prepared slice holds {client_df.height} rows for this client, "
                f"expected {expected_rows}"
            )
        spec = default_batch_spec()
        batches = make_t1_smoke_batches(
            client_df, spec=spec, batch_size=int(config_rec["batch_size"])
        )
        if not batches:
            raise SmokeValidationError("client produced zero batches from its prepared rows")
        load_end_ns = time.perf_counter_ns()

        model = build_smoke_model(category_count, spec)
        model.load_state_dict(state_dict)
        optimizer = optim.SGD(
            model.parameters(),
            lr=float(config_rec["learning_rate"]),
            momentum=float(config_rec["momentum"]),
        )
        core = LocalTrainerCore(
            model=model,
            batch_spec=spec,
            objective=ContractSmokeObjective(),
            optimizer=optimizer,
            device="cpu",
        )
        adapter = FlowerLocalAdapter(
            core=core,
            shared_state_spec=model.shared_state_spec(),
            aggregation_weight_policy=ContributingRowsSmokeWeightPolicy(),
        )
        all_batches = batches * int(config_rec["local_epochs"])

        fit_start_ns = time.perf_counter_ns()
        result = adapter.fit(state_dict, all_batches, outer_round=server_round)
        fit_end_ns = time.perf_counter_ns()

        weight = result.aggregation_weight
        if isinstance(weight, bool) or int(weight) != weight or int(weight) <= 0:
            raise SmokeValidationError("client produced a non-positive or non-integral weight")
        scalar_metrics = result.scalar_metrics()
        if "train_loss" not in scalar_metrics:
            raise SmokeValidationError(
                "client metrics are missing train_loss; 0.0 is never assumed"
            )
        train_loss = float(scalar_metrics["train_loss"])
        if not math.isfinite(train_loss):
            raise SmokeValidationError("client produced a non-finite local loss")

        client_end_ns = time.perf_counter_ns()
        record = RecordDict()
        record.parameters_records["arrays"] = ArrayRecord.from_torch_state_dict(
            dict(result.shared_state)
        )
        record.metrics_records["metrics"] = MetricRecord(
            {"num-examples": int(weight), "train_loss": train_loss}
        )
        record.configs_records["config"] = ConfigRecord(
            {
                "logical_client_id": logical_client_id,
                "received_digest": received_digest,
                "updated_digest": result.updated_state_digest,
                "local_train_loss": train_loss,
                "data_load_seconds": (load_end_ns - load_start_ns) / 1e9,
                "local_fit_seconds": (fit_end_ns - fit_start_ns) / 1e9,
                "client_start_ns": client_start_ns,
                "client_end_ns": client_end_ns,
            }
        )
        return message.create_reply(record)

    return app


# ---------------------------------------------------------------------------
# One trial, executed inside its own supervised subprocess
# ---------------------------------------------------------------------------


def run_trial_child(spec: dict[str, Any]) -> dict[str, Any]:
    """Run exactly one Flower simulation. This executes in the child process."""
    workload = spec["workload"]
    population_ids: list[str] = spec["population_ids"]
    samples: dict[str, Any] = spec["samples"]
    concurrency = int(spec["concurrency"])
    category_count = int(spec["category_count"])
    seed = int(workload["seed"])
    clients_per_round = int(workload["clients_per_round"])
    num_rounds = int(workload["num_rounds"])

    torch.manual_seed(seed)
    torch.set_num_threads(int(workload["torch_num_threads"]))

    selected_ids_by_round = {int(k): list(v["selected_client_ids"]) for k, v in samples.items()}
    selection_digests_by_round = {int(k): v["selected_digest"] for k, v in samples.items()}

    spec_batch = default_batch_spec()
    val_df = pl.read_parquet(spec["validation_slice_path"])

    # Identical deterministic starting point for every independent trial.
    torch.manual_seed(seed)
    base_model = build_smoke_model(category_count, spec_batch)
    initial_state = {k: v.detach().clone() for k, v in base_model.state_dict().items()}
    initial_digest = get_digest(initial_state)
    if initial_digest != spec["expected_initial_digest"]:
        raise SmokeValidationError(
            "trial initial state digest does not match the recorded initialization proof"
        )

    holder: dict[str, Any] = {
        "server_evaluations": [],
        "round_wall_seconds": {},
        "server_evaluation_seconds": {},
        "server_entry_to_node_binding_seconds": None,
        "node_binding": None,
        "error": None,
    }

    server_app = ServerApp()

    @server_app.main()
    def _server_main(grid: Grid, context: Context) -> None:
        simulation_entry_ns = time.perf_counter_ns()
        torch.manual_seed(seed)
        torch.set_num_threads(int(workload["torch_num_threads"]))

        strategy = RuntimeFedAvg(
            population_ids=population_ids,
            node_registration_timeout=float(spec["node_registration_timeout_seconds"]),
            train_slice_path=spec["train_slice_path"],
            learning_rate=float(workload["learning_rate"]),
            momentum=float(workload["momentum"]),
            local_epochs=int(workload["local_epochs"]),
            batch_size=int(workload["batch_size"]),
            examples_per_client=int(workload["examples_per_client"]),
            seed=seed,
            expected_clients=clients_per_round,
            aggregation_atol=float(workload["aggregation_atol"]),
            selected_ids_by_round=selected_ids_by_round,
            selection_digests_by_round=selection_digests_by_round,
            fraction_train=0.0,
            fraction_evaluate=0.0,
            min_train_nodes=clients_per_round,
            min_evaluate_nodes=0,
            min_available_nodes=clients_per_round,
            weighted_by_key="num-examples",
        )
        holder["node_binding"] = strategy.bind_population_to_nodes(grid)
        # Measured from ServerApp callback entry through node binding. This is not the
        # whole simulation startup; Ray/worker bring-up happens before this callback runs.
        holder["server_entry_to_node_binding_seconds"] = (
            time.perf_counter_ns() - simulation_entry_ns
        ) / 1e9

        def evaluate_fn(server_round: int, arrays: ArrayRecord) -> MetricRecord:
            eval_start_ns = time.perf_counter_ns()
            state = arrays.to_torch_state_dict()
            model = build_smoke_model(category_count, spec_batch)
            model.load_state_dict(state)
            metrics = server_evaluate(model, val_df, spec_batch)
            eval_end_ns = time.perf_counter_ns()
            holder["server_evaluation_seconds"][server_round] = (eval_end_ns - eval_start_ns) / 1e9
            holder["final_state"] = {k: v.detach().clone() for k, v in state.items()}
            holder["server_evaluations"].append(
                {
                    "server_round": server_round,
                    "cross_entropy": metrics["cross_entropy"],
                    "accuracy_at_1": metrics["accuracy_at_1"],
                    "support": int(metrics["support"]),
                    "state_digest": get_digest(state),
                }
            )
            if server_round >= 1 and strategy.stage_timings:
                stage = strategy.stage_timings[-1]
                if stage["server_round"] == server_round:
                    holder["round_wall_seconds"][server_round] = (
                        eval_end_ns - stage["configure_start_ns"]
                    ) / 1e9
            return MetricRecord(
                {
                    "cross_entropy": metrics["cross_entropy"],
                    "accuracy_at_1": metrics["accuracy_at_1"],
                    "support": float(metrics["support"]),
                }
            )

        # The configured per-round wait is the production timeout, not a default.
        strategy.start(
            grid=grid,
            initial_arrays=ArrayRecord(initial_state),
            num_rounds=num_rounds,
            timeout=float(spec["round_timeout_seconds"]),
            train_config=ConfigRecord({"purpose": _PURPOSE}),
            evaluate_fn=evaluate_fn,
        )
        holder["tracing_log"] = strategy.tracing_log
        holder["stage_timings"] = strategy.stage_timings

    client_app = build_runtime_client_app(category_count)

    trial_start_ns = time.perf_counter_ns()
    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=len(population_ids),  # N, never M
        backend_name="ray",
        backend_config={
            "init_args": {
                "num_cpus": concurrency,
                "num_gpus": 0,
                "include_dashboard": False,
            },
            "client_resources": {"num_cpus": 1, "num_gpus": 0},
        },
    )
    trial_end_ns = time.perf_counter_ns()

    tracing_log = holder.get("tracing_log", [])
    stage_timings = holder.get("stage_timings", [])
    evaluations = sorted(holder["server_evaluations"], key=lambda e: e["server_round"])

    rounds = [entry["server_round"] for entry in tracing_log]
    if rounds != list(range(1, num_rounds + 1)):
        raise SmokeValidationError(f"expected rounds 1..{num_rounds}, observed {rounds}")
    eval_rounds = [e["server_round"] for e in evaluations]
    if eval_rounds != list(range(num_rounds + 1)):
        raise SmokeValidationError(f"expected evaluations 0..{num_rounds}, observed {eval_rounds}")
    if evaluations[0]["state_digest"] == evaluations[-1]["state_digest"]:
        raise SmokeValidationError("the global model did not change between round 0 and the end")

    per_round: list[dict[str, Any]] = []
    for entry, stage in zip(tracing_log, stage_timings, strict=True):
        server_round = entry["server_round"]
        client_timings = stage.get("client_timings", [])
        starts = [c["client_start_ns"] for c in client_timings]
        ends = [c["client_end_ns"] for c in client_timings]
        per_round.append(
            {
                "server_round": server_round,
                "selected_client_count": entry["selected_client_count"],
                "selection_digest": entry["selection_digest"],
                "contributing_examples": entry["contributing_examples"],
                "aggregation_oracle_pass": entry["aggregation_oracle_pass"],
                "max_abs_diff": entry["max_abs_diff"],
                "dispatch_to_replies_seconds": (
                    stage["aggregate_entry_ns"] - stage["configure_end_ns"]
                )
                / 1e9,
                "aggregation_seconds": (stage["aggregate_end_ns"] - stage["aggregate_entry_ns"])
                / 1e9,
                "server_evaluation_seconds": holder["server_evaluation_seconds"].get(server_round),
                "round_wall_seconds": holder["round_wall_seconds"].get(server_round),
                "client_data_load_seconds": summarize_samples(
                    [c["data_load_seconds"] for c in client_timings]
                ),
                "client_local_fit_seconds": summarize_samples(
                    [c["local_fit_seconds"] for c in client_timings]
                ),
                "observed_client_span_seconds": (
                    (max(ends) - min(starts)) / 1e9 if client_timings else None
                ),
            }
        )

    final_state = holder.get("final_state")
    if final_state is not None:
        state_path = Path(spec["output_path"]).parent / "final_state.pt"
        torch.save(final_state, state_path)

    return {
        "schema": "fl_runtime_trial_v1",
        "version": "1",
        "trial_id": spec["trial_id"],
        "status": "SUCCEEDED",
        "declared_population": len(population_ids),
        "participants_per_round": clients_per_round,
        "configured_concurrency": concurrency,
        "node_binding": holder["node_binding"],
        "server_entry_to_node_binding_seconds": holder["server_entry_to_node_binding_seconds"],
        "trial_wall_seconds": (trial_end_ns - trial_start_ns) / 1e9,
        "rounds": per_round,
        "server_evaluations": evaluations,
        "initial_state_digest": initial_digest,
        "final_state_digest": evaluations[-1]["state_digest"],
        "unique_participating_clients": len(
            {cid for entry in tracing_log for cid in entry["selected_client_ids"]}
        ),
        "total_fit_calls": sum(entry["selected_client_count"] for entry in tracing_log),
        "private_selected_ids": {
            str(entry["server_round"]): entry["selected_client_ids"] for entry in tracing_log
        },
    }


# ---------------------------------------------------------------------------
# Parent-side supervision
# ---------------------------------------------------------------------------


def launch_trial(
    spec: dict[str, Any],
    *,
    safety: dict[str, Any],
) -> dict[str, Any]:
    """Run one trial in a child process while sampling its process tree."""
    tid = spec["trial_id"]
    private_dir = _REPO_ROOT / _PRIVATE_ROOT / tid
    private_dir.mkdir(parents=True, exist_ok=True)
    spec_path = private_dir / "trial_spec.json"
    out_path = private_dir / "trial_output.json"
    log_path = private_dir / "child.log"
    spec = dict(spec)
    spec["output_path"] = str(out_path)
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True), encoding="utf-8", newline="\n")
    if out_path.exists():
        out_path.unlink()

    child_env = dict(os.environ)
    child_env.update(
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        POLARS_MAX_THREADS="1",
    )

    limits = MonitorLimits(
        max_tree_rss_fraction_total=float(safety["max_sampled_tree_rss_fraction_total"]),
        min_available_ram_gib=float(safety["min_available_ram_during_trial_gib"]),
        min_available_ram_fraction_total=float(
            safety["min_available_ram_during_trial_fraction_total"]
        ),
    )
    max_wall = float(safety["max_trial_wall_seconds"])

    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--trial-spec", str(spec_path)],
            cwd=str(_REPO_ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=child_env,
            shell=False,
        )
        monitor = ProcessTreeMonitor(
            process.pid, poll_seconds=float(safety["monitor_poll_seconds"]), limits=limits
        )
        monitor.start()
        guard_reason: str | None = None
        timed_out = False
        while True:
            if process.poll() is not None:
                break
            if monitor.guard_reason is not None:
                guard_reason = monitor.guard_reason
                break
            if time.monotonic() - started > max_wall:
                timed_out = True
                break
            time.sleep(0.2)

        cleanup: dict[str, Any] = {"cleanup_required": False}
        if guard_reason is not None or timed_out:
            cleanup = monitor.terminate_owned_tree()
            cleanup["cleanup_required"] = True
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            cleanup = monitor.terminate_owned_tree()
            cleanup["cleanup_required"] = True
        report = monitor.stop()

    elapsed = time.monotonic() - started
    exit_code = process.returncode
    measurement = report.to_public_dict()

    if timed_out:
        status, reason = "FAILED_TIMEOUT", f"trial exceeded {max_wall} seconds of wall time"
    elif guard_reason is not None:
        status, reason = "FAILED_RESOURCE_GUARD", guard_reason
    elif exit_code != 0:
        status, reason = "FAILED_WORKER_ERROR", f"child exited with code {exit_code}"
    elif not out_path.is_file():
        status, reason = "FAILED_NO_OUTPUT", "child produced no trial output"
    else:
        status, reason = "SUCCEEDED", None

    payload: dict[str, Any] = {}
    if out_path.is_file():
        try:
            payload = json.loads(out_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            status, reason = "FAILED_NO_OUTPUT", "child output was not valid JSON"

    if status == "SUCCEEDED" and not report.measurement_complete:
        status = "FAILED_INCOMPLETE_MEASUREMENT"
        reason = "the sampled memory profile was incomplete and cannot be called verified"

    return {
        "trial_id": tid,
        "status": status,
        "reason": reason,
        "source_snapshot": spec.get("source_snapshot"),
        "exit_code": exit_code,
        "supervised_wall_seconds": elapsed,
        "measurement": measurement,
        "cleanup": cleanup,
        "child_log_uri": (_PRIVATE_ROOT / tid / "child.log").as_posix(),
        "payload": payload,
    }


# ---------------------------------------------------------------------------
# ExperimentConfig / ExperimentResult for every attempted trial
# ---------------------------------------------------------------------------


def _ref(logical_id: str, artifact_schema: str, uri: str) -> dict[str, Any]:
    """Reference an artifact using the identity convention its owner established.

    The frozen vocabulary keeps the upstream raw-file SHA recorded by S1; every other
    artifact keeps the repository canonical convention. Nothing is normalized here to
    make the recorded values look uniform.
    """
    path = _REPO_ROOT / uri
    if not path.is_file():
        raise RuntimeBenchmarkError(f"cannot reference a missing artifact: {uri}")
    sha = (
        verify_vocabulary(path)
        if uri.endswith("vocabulary_v1.proposed.json")
        else file_sha256(path)
    )
    return make_artifact_ref(logical_id, artifact_schema, uri, sha)


def build_resolved_config(
    *,
    workload: dict[str, Any],
    population: int,
    concurrency: int,
    repetition: int,
    hardware: dict[str, Any],
) -> dict[str, Any]:
    """Resolve one ExperimentConfig per trial so each repetition has its own identity."""
    return {
        "schema": "experiment_config_v1",
        "version": "1",
        "config_id": (
            f"fl_runtime_benchmark_r2a_t1_n{population}"
            f"_m{workload['clients_per_round']}_c{concurrency}_rep{repetition}_non_scientific"
        ),
        "regime": "R2A",
        "tasks": ["T1"],
        "training_cohort": "C1",
        "evaluation_cohorts": ["C1"],
        "seed": int(workload["seed"]),
        "source_dataset_ref": _ref(
            "fl_runtime_input_manifest_v1", "fl_runtime_inputs_v1", _RUNTIME_INPUTS.as_posix()
        ),
        "canonical_data_contract_ref": _ref(
            "data_protocol_v1",
            "data_protocol_v1",
            "docs/evidence/s1-ds-05-06/data_protocol_v1.proposed.json",
        ),
        "cohort_manifest_ref": _ref(
            "cohort_manifest_v1",
            "cohort_manifest_v1",
            "data/protocol/INTERNAL_DO_NOT_UPLOAD_cohort_manifest_v1.proposed.parquet",
        ),
        "split_manifest_ref": _ref(
            "data_protocol_v1_split",
            "data_protocol_v1",
            "docs/evidence/s1-ds-05-06/data_protocol_v1.proposed.json",
        ),
        "task_examples_manifest_ref": _ref(
            "fl_runtime_input_manifest_v1", "fl_runtime_inputs_v1", _RUNTIME_INPUTS.as_posix()
        ),
        "evaluation_manifest_ref": _ref(
            "t1_validation_parquet",
            "task_examples_t1_v1",
            "data/examples/INTERNAL_DO_NOT_UPLOAD_task_examples_t1_v1.proposed.parquet",
        ),
        "representation_ref": _ref(
            "vocabulary_v1",
            "vocabulary_v1",
            "docs/evidence/s1-d1-ds-07/vocabulary_v1.proposed.json",
        ),
        # All N and C points share this identity: the workload itself never changes.
        "model_config_ref": _ref(
            "fl_runtime_workload_v1", "fl_runtime_workload_v1", _WORKLOAD_CONFIG.as_posix()
        ),
        "objective_config_ref": _ref(
            "contract_smoke_objective_v1", "python_source_v1", "ppsi/training/objective.py"
        ),
        "shared_trainer_core_ref": _ref(
            "shared_trainer_core_manifest_v1",
            "shared_trainer_core_manifest_v1",
            "docs/evidence/s1-pr-05/shared_trainer_core_manifest.v1.json",
        ),
        "evaluation_protocol_ref": _ref(
            "adr_001_evaluation_protocol",
            "adr_001_evaluation_protocol",
            "docs/decisions/ADR-001-evaluation-protocol.md",
        ),
        # The #20 diagnostic evaluator, never the #31 scientific headline evaluator.
        "evaluator_ref": _ref(
            "fl_real_smoke_script_v1", "python_source_v1", "scripts/federated/fl_real_smoke.py"
        ),
        "environment_lock_ref": _ref("uv_lock_v1", "uv_lock_v1", "uv.lock"),
        "initialization": {
            "kind": "COMMON_INITIALIZATION",
            "common_initialization_ref": _ref(
                "fl_runtime_initialization_v1",
                "fl_runtime_initialization_v1",
                _RUNTIME_INITIALIZATION.as_posix(),
            ),
        },
        "regime_config": {
            "orchestration_type": "FEDAVG",
            "scientific": False,
            "purpose": _PURPOSE,
            "declared_population": population,
            "clients_per_round": int(workload["clients_per_round"]),
            "configured_concurrency": concurrency,
            "repetition_index": repetition,
            "num_rounds": int(workload["num_rounds"]),
            "local_epochs": int(workload["local_epochs"]),
            "learning_rate": float(workload["learning_rate"]),
            "momentum": float(workload["momentum"]),
            "optimizer": workload["optimizer"],
            "examples_per_client": int(workload["examples_per_client"]),
            "aggregation_weight_policy": "contributing_rows_smoke_weight_v1_NON_SCIENTIFIC",
            "runner_ref_uri": "scripts/federated/fl_runtime_benchmark.py",
            "hardware_logical_cpus": hardware["logical_cpus"],
            "gpu_memory": "NOT_APPLICABLE",
        },
    }


def _retire_superseded_results(config_path: Path, *, keep: Path) -> list[str]:
    """Remove this trial's own orphaned results after its config was rewritten.

    Re-running a trial rewrites its resolved config in place, which changes the config
    hash and therefore the run id. The previous result file then points at a config it
    no longer describes. Only results naming *this* trial's config path with a stale
    hash are retired; results belonging to any other run are never touched.
    """
    root = _REPO_ROOT / _RESULT_ROOT
    if not root.is_dir():
        return []
    tracked = set(
        subprocess.run(
            ["git", "ls-files", _RESULT_ROOT.as_posix()],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            shell=False,
        ).stdout.split()
    )
    current_sha = file_sha256(_REPO_ROOT / config_path)
    retired: list[str] = []
    for candidate in sorted(root.glob("*.result.json")):
        if candidate == _REPO_ROOT / keep:
            continue
        try:
            record = json.loads(candidate.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        ref = record.get("config_ref") or {}
        if ref.get("uri") != config_path.as_posix() or ref.get("sha256") == current_sha:
            continue
        if candidate.relative_to(_REPO_ROOT).as_posix() in tracked:
            # A tracked result belongs to someone else's merged work.
            continue
        # Preserve the superseded attempt with its metadata instead of erasing it.
        archive = _REPO_ROOT / _PRIVATE_ROOT / "superseded-results"
        archive.mkdir(parents=True, exist_ok=True)
        candidate.replace(archive / candidate.name)
        retired.append(candidate.name)
    return retired


def write_trial_evidence(
    *,
    outcome: dict[str, Any],
    resolved_config: dict[str, Any],
    git_sha: str,
    started_at: str,
    ended_at: str,
) -> dict[str, Any]:
    """Persist the resolved config, the measurement and one ExperimentResult."""
    from ppsi.training.result import build_experiment_result, make_metric_record

    tid = outcome["trial_id"]
    public_dir = _SCALE_EVIDENCE_DIR / "trials" / tid
    config_path = public_dir / "experiment_config.v1.json"
    config_sha = _write_json(config_path, resolved_config)
    config_ref = make_artifact_ref(
        "fl_runtime_resolved_config_v1",
        "experiment_config_v1",
        config_path.as_posix(),
        config_sha,
    )

    measurement = dict(outcome["measurement"])
    measurement["trial_id"] = tid
    measurement_path = public_dir / "runtime_measurement.v1.json"
    measurement_sha = _write_json(measurement_path, measurement)
    measurement_ref = make_artifact_ref(
        "fl_runtime_measurement_v1",
        "runtime_process_tree_measurement_v1",
        measurement_path.as_posix(),
        measurement_sha,
    )

    payload = outcome.get("payload") or {}
    succeeded = outcome["status"] == "SUCCEEDED"
    metrics: list[dict[str, Any]] = []
    if succeeded and payload.get("server_evaluations"):
        final = payload["server_evaluations"][-1]
        metrics = [
            make_metric_record(
                metric_id="smoke_t1_validation_cross_entropy_non_scientific",
                task="T1",
                cohort="C1",
                value=float(final["cross_entropy"]),
                direction="MINIMIZE",
                unit="UNITLESS",
                support=int(final["support"]),
            ),
            make_metric_record(
                metric_id="smoke_t1_validation_accuracy_at_1_non_scientific",
                task="T1",
                cohort="C1",
                value=float(final["accuracy_at_1"]),
                direction="NEUTRAL",
                unit="FRACTION",
                support=int(final["support"]),
            ),
        ]

    if succeeded:
        system_measurements = {
            "schema": "system_measurement_reference_set_v1",
            "version": "1",
            "status": "AVAILABLE",
            "refs": [measurement_ref],
        }
    else:
        system_measurements = {
            "schema": "system_measurement_reference_set_v1",
            "version": "1",
            "status": "PENDING",
            "null_reason": f"trial did not complete: {outcome['status']}",
        }

    failure = None
    if not succeeded:
        failure = {
            "reason_code": outcome["status"],
            # Never copy a raw child traceback: it can carry user paths and ids.
            "message": str(outcome.get("reason") or outcome["status"])[:400],
            "retryable": False,
        }

    result = build_experiment_result(
        experiment_config=resolved_config,
        config_ref=config_ref,
        git_sha=git_sha,
        state="SUCCEEDED" if succeeded else "FAILED",
        attempt=1,
        started_at_utc=started_at,
        ended_at_utc=ended_at,
        metrics=metrics,
        artifacts=[measurement_ref],
        system_measurements=system_measurements,
        failure=failure,
        federated_metadata={
            "orchestration_type": "FEDAVG",
            "num_rounds": int(resolved_config["regime_config"]["num_rounds"]),
            "clients_per_round": int(resolved_config["regime_config"]["clients_per_round"]),
            "sampler_version": DEFAULT_SAMPLER_VERSION,
            "aggregation_weight_policy": "contributing_rows_smoke_weight_v1_NON_SCIENTIFIC",
        },
    )
    result_path = _RESULT_ROOT / f"{result['run_id']}.result.json"
    result_sha = _write_json(result_path, result)
    _retire_superseded_results(config_path, keep=result_path)

    # Public trial summary: counts, digests and timings only.
    public_rounds = []
    for row in payload.get("rounds", []):
        public_rounds.append({k: v for k, v in row.items() if k != "private_selected_ids"})
    warm_excluded = {
        int(r) for r in _load_json(_WORKLOAD_CONFIG)["warmup_rounds_excluded_from_timing_summary"]
    }
    warm_rounds = [r for r in public_rounds if r["server_round"] not in warm_excluded]
    warm_stats = summarize_samples(
        [r["round_wall_seconds"] for r in warm_rounds if r["round_wall_seconds"] is not None]
    )
    summary = {
        "schema": "fl_runtime_trial_summary_v1",
        "version": "1",
        "trial_id": tid,
        "status": outcome["status"],
        "reason": outcome["reason"],
        "declared_population": payload.get("declared_population"),
        "participants_per_round": payload.get("participants_per_round"),
        "configured_concurrency": payload.get("configured_concurrency"),
        "node_binding": payload.get("node_binding"),
        "server_entry_to_node_binding_seconds": payload.get("server_entry_to_node_binding_seconds"),
        "trial_wall_seconds": payload.get("trial_wall_seconds"),
        "supervised_wall_seconds": outcome["supervised_wall_seconds"],
        "rounds": public_rounds,
        "server_evaluations": payload.get("server_evaluations", []),
        "warm_round_wall_seconds": warm_stats,
        "warm_rounds_excluded": sorted(warm_excluded),
        "unique_participating_clients": payload.get("unique_participating_clients"),
        "total_fit_calls": payload.get("total_fit_calls"),
        "initial_state_digest": payload.get("initial_state_digest"),
        "final_state_digest": payload.get("final_state_digest"),
        "measurement_ref": measurement_ref,
        "config_ref": config_ref,
        "source_snapshot": outcome.get("source_snapshot"),
        "result_ref": make_artifact_ref(
            "fl_runtime_experiment_result_v1",
            "experiment_result_v1",
            result_path.as_posix(),
            result_sha,
        ),
        "cleanup": outcome["cleanup"],
        "limitations": [
            "Runtime measurement only; never an R2A scientific result.",
            "Only M clients train per round; the declared population is not all trained.",
            "Timings and RSS are observations and are not expected to reproduce exactly.",
        ],
    }
    summary_path = public_dir / "trial_summary.v1.json"
    summary_sha = _write_json(summary_path, summary)

    # Private detail stays outside the public evidence tree.
    private_dir = _REPO_ROOT / _PRIVATE_ROOT / tid
    private_dir.mkdir(parents=True, exist_ok=True)
    (private_dir / "selected_ids.json").write_text(
        json.dumps(payload.get("private_selected_ids", {}), indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )
    return {
        "trial_id": tid,
        "status": outcome["status"],
        "reason": outcome["reason"],
        "summary_ref": make_artifact_ref(
            "fl_runtime_trial_summary_v1",
            "fl_runtime_trial_summary_v1",
            summary_path.as_posix(),
            summary_sha,
        ),
        "config_ref": config_ref,
        "measurement_ref": measurement_ref,
        "result_run_id": result["run_id"],
        "warm_round_wall_seconds": warm_stats,
        "peak_tree_rss_bytes": measurement.get("peak_tree_rss_bytes"),
        "declared_population": payload.get("declared_population"),
        "configured_concurrency": payload.get("configured_concurrency"),
        "selection_digests": [r["selection_digest"] for r in public_rounds],
        "contributing_examples": [r["contributing_examples"] for r in public_rounds],
        "final_state_digest": payload.get("final_state_digest"),
    }


# ---------------------------------------------------------------------------
# Stage: scale (#35)
# ---------------------------------------------------------------------------


def _load_prepared_context(workload: dict[str, Any]) -> dict[str, Any]:
    inputs = _load_json(_RUNTIME_INPUTS)
    init = _load_json(_RUNTIME_INITIALIZATION)
    samples_path = _REPO_ROOT / _PRIVATE_ROOT / "population_samples.json"
    if not samples_path.is_file():
        raise RuntimeBenchmarkError("run --stage prepare before scale: population samples missing")
    samples = json.loads(samples_path.read_text(encoding="utf-8"))
    prepared = inputs.get("prepared_slices") or {}
    if not prepared:
        raise RuntimeBenchmarkError("prepared slices are missing from the runtime input evidence")
    return {
        "inputs": inputs,
        "initialization": init,
        "samples": samples,
        "train_slice_path": str(_REPO_ROOT / prepared["train_slice"]["uri"]),
        "validation_slice_path": str(_REPO_ROOT / prepared["validation_slice"]["uri"]),
        "category_count": init["category_count"],
        "expected_initial_digest": init["initial_state_digest"],
    }


def _population_ids(population: int, workload: dict[str, Any]) -> list[str]:
    path = _REPO_ROOT / workload["population_output"]
    df = pl.read_parquet(path).filter(pl.col("population") == population)
    ids = sorted(df["client_id"].to_list())
    if len(ids) != population:
        raise RuntimeBenchmarkError(
            f"population manifest holds {len(ids)} ids for the {population} point"
        )
    return ids


def _trial_spec(
    *,
    population: int,
    concurrency: int,
    repetition: int,
    workload: dict[str, Any],
    context: dict[str, Any],
    safety: dict[str, Any],
) -> dict[str, Any]:
    return {
        "trial_id": trial_id(
            population, int(workload["clients_per_round"]), concurrency, repetition
        ),
        "workload": workload,
        "population_ids": _population_ids(population, workload),
        "samples": context["samples"]["populations"][str(population)],
        "concurrency": concurrency,
        "category_count": context["category_count"],
        "train_slice_path": context["train_slice_path"],
        "validation_slice_path": context["validation_slice_path"],
        "expected_initial_digest": context["expected_initial_digest"],
        "node_registration_timeout_seconds": safety["node_registration_timeout_seconds"],
        "round_timeout_seconds": safety["round_timeout_seconds"],
        "source_snapshot": build_source_snapshot(),
    }


def _run_one_trial(
    *,
    population: int,
    concurrency: int,
    repetition: int,
    workload: dict[str, Any],
    context: dict[str, Any],
    safety: dict[str, Any],
    hardware: dict[str, Any],
    git_sha: str,
) -> dict[str, Any]:
    spec = _trial_spec(
        population=population,
        concurrency=concurrency,
        repetition=repetition,
        workload=workload,
        context=context,
        safety=safety,
    )
    print(f"[trial] {spec['trial_id']}: launching supervised subprocess")
    started_at = _utc_now()
    outcome = launch_trial(spec, safety=safety)
    ended_at = _utc_now()
    resolved = build_resolved_config(
        workload=workload,
        population=population,
        concurrency=concurrency,
        repetition=repetition,
        hardware=hardware,
    )
    record = write_trial_evidence(
        outcome=outcome,
        resolved_config=resolved,
        git_sha=git_sha,
        started_at=started_at,
        ended_at=ended_at,
    )
    print(
        f"[trial] {spec['trial_id']}: {record['status']}"
        + (f" ({record['reason']})" if record["reason"] else "")
    )
    return record


def _point_is_stable(
    record: dict[str, Any], scale_config: dict[str, Any], workload: dict[str, Any]
) -> tuple[bool, str]:
    rules = scale_config["stable_point"]
    if record["status"] != "SUCCEEDED":
        return False, record["reason"] or record["status"]
    summary = _load_json(Path(record["summary_ref"]["uri"]))
    rounds = summary["rounds"]
    if len(rounds) != rules["required_completed_rounds"]:
        return False, f"completed {len(rounds)} rounds"
    if len(summary["server_evaluations"]) != rules["required_completed_rounds"] + 1:
        return False, "server evaluation count is not rounds + 1"
    for row in rounds:
        if row["selected_client_count"] != rules["required_clients_every_round"]:
            return (
                False,
                f"round {row['server_round']} selected {row['selected_client_count']} clients",
            )
        if row["contributing_examples"] != rules["expected_examples_every_round"]:
            return (
                False,
                f"round {row['server_round']} contributed {row['contributing_examples']} examples",
            )
        if not row["aggregation_oracle_pass"]:
            return False, f"round {row['server_round']} oracle did not pass"
    if summary["initial_state_digest"] == summary["final_state_digest"]:
        return False, "the global model did not change"
    warm = summary["warm_round_wall_seconds"]
    if warm["n"] != len(rules["timing_sample_rounds"]):
        return False, f"expected {len(rules['timing_sample_rounds'])} warm samples, got {warm['n']}"
    if warm["p95"] is not None and warm["p95"] > rules["max_warm_round_p95_seconds"]:
        return False, f"warm p95 {warm['p95']:.1f}s exceeded {rules['max_warm_round_p95_seconds']}s"
    return True, "OK"


def _compare_repeats(
    first: dict[str, Any], second: dict[str, Any], scale_config: dict[str, Any]
) -> dict[str, Any]:
    """Repeat checks compare the experiment, never the wall clock."""
    atol = float(scale_config["stable_point"]["repeat_parameter_atol"])
    same_selection = first["selection_digests"] == second["selection_digests"]
    same_contrib = first["contributing_examples"] == second["contributing_examples"]
    byte_exact = first["final_state_digest"] == second["final_state_digest"]

    max_abs_diff: float | None = None
    within_tolerance: bool | None = None
    paths = [
        _REPO_ROOT / _PRIVATE_ROOT / record["trial_id"] / "final_state.pt"
        for record in (first, second)
    ]
    if all(p.is_file() for p in paths):
        a = torch.load(paths[0], map_location="cpu", weights_only=True)
        b = torch.load(paths[1], map_location="cpu", weights_only=True)
        if set(a) == set(b):
            max_abs_diff = max(
                float(torch.abs(a[k].to(torch.float64) - b[k].to(torch.float64)).max().item())
                for k in a
            )
            within_tolerance = max_abs_diff <= atol
    return {
        "selection_digests_identical": same_selection,
        "contributing_examples_identical": same_contrib,
        "final_state_byte_exact": byte_exact,
        "final_state_max_abs_diff": max_abs_diff,
        "final_state_within_atol": within_tolerance,
        "atol": atol,
        "rtol": float(scale_config["stable_point"]["repeat_parameter_rtol"]),
        "repeatable": bool(
            same_selection
            and same_contrib
            and (within_tolerance if within_tolerance is not None else byte_exact)
        ),
        "note": "Timing and RSS are observations and are never compared for equality.",
    }


def stage_scale(
    workload: dict[str, Any],
    scale_config: dict[str, Any],
    *,
    hardware: dict[str, Any],
    git_sha: str,
    budget_deadline: float,
) -> dict[str, Any]:
    safety = scale_config["safety"]
    decision = choose_scale_concurrency(hardware, scale_config)
    disk_ok, disk_reason = check_disk(hardware, scale_config)

    points: list[dict[str, Any]] = []
    limitations = [
        "Runtime benchmark only; not an R2A scientific result and not a capacity guarantee.",
        "Only 20 clients train per round; the declared population is never fully trained.",
        "Peak memory is a sampled process-tree RSS upper bound, not unique physical RAM.",
        "Safety thresholds are conservative plan decisions, not measured hardware capability.",
    ]

    if not decision["admissible"] or not disk_ok:
        reason = decision["reason"] if not decision["admissible"] else disk_reason
        for n in scale_config["population_points"]:
            points.append(
                {"declared_population": n, "status": "NOT_ATTEMPTED_SAFETY", "reason": reason}
            )
        summary = {
            "schema": "fl_scale_summary_v1",
            "version": "1",
            "status": "BLOCKED_CAPACITY",
            "local_gate": "BLOCKED",
            "purpose": _PURPOSE,
            "scientific": False,
            "hardware": hardware,
            "concurrency_decision": decision,
            "points": points,
            "chosen_population": None,
            "repeatability": None,
            "limitations": limitations + [reason],
        }
        _write_json(_SCALE_SUMMARY, summary)
        return summary

    concurrency = int(decision["concurrency"])
    context = _load_prepared_context(workload)
    inputs = context["inputs"]
    if not inputs["benchmark_pool"]["pool_sufficient_for_largest_point"]:
        for n in scale_config["population_points"]:
            points.append(
                {
                    "declared_population": n,
                    "status": "BLOCKED_POOL",
                    "reason": (
                        f"only {inputs['benchmark_pool']['eligible_client_count']} clients hold at "
                        f"least {inputs['benchmark_pool']['minimum_train_examples']} T1 TRAIN examples"
                    ),
                }
            )
        summary = {
            "schema": "fl_scale_summary_v1",
            "version": "1",
            "status": "BLOCKED_POOL",
            "local_gate": "BLOCKED",
            "purpose": _PURPOSE,
            "scientific": False,
            "hardware": hardware,
            "concurrency_decision": decision,
            "points": points,
            "chosen_population": None,
            "repeatability": None,
            "limitations": limitations,
        }
        _write_json(_SCALE_SUMMARY, summary)
        return summary

    stop_escalation = False
    passing: list[tuple[int, dict[str, Any]]] = []
    for n in sorted(scale_config["population_points"]):
        if stop_escalation:
            points.append(
                {
                    "declared_population": n,
                    "status": "NOT_ATTEMPTED_SAFETY",
                    "reason": "a smaller population already failed a resource guard",
                }
            )
            continue
        if time.monotonic() > budget_deadline:
            points.append(
                {
                    "declared_population": n,
                    "status": "NOT_ATTEMPTED_BUDGET",
                    "reason": "the real execution budget was exhausted",
                }
            )
            continue
        record = _run_one_trial(
            population=n,
            concurrency=concurrency,
            repetition=1,
            workload=workload,
            context=context,
            safety=safety,
            hardware=hardware,
            git_sha=git_sha,
        )
        stable, why = _point_is_stable(record, scale_config, workload)
        points.append(
            {
                "declared_population": n,
                "participants_per_round": int(workload["clients_per_round"]),
                "configured_concurrency": concurrency,
                "status": "STABLE" if stable else record["status"],
                "reason": None if stable else why,
                "trial_id": record["trial_id"],
                "trial_summary_ref": record["summary_ref"],
                "result_run_id": record["result_run_id"],
                "warm_round_wall_seconds": record["warm_round_wall_seconds"],
                "peak_tree_rss_bytes": record["peak_tree_rss_bytes"],
            }
        )
        if stable:
            passing.append((n, record))
        elif record["status"] in {"FAILED_RESOURCE_GUARD", "FAILED_TIMEOUT"}:
            stop_escalation = True

    chosen = None
    repeatability = None
    if passing:
        for n, first in reversed(passing):
            if time.monotonic() > budget_deadline:
                repeatability = {
                    "status": "NOT_ATTEMPTED_BUDGET",
                    "reason": "the real execution budget was exhausted before the repeat trial",
                }
                break
            second = _run_one_trial(
                population=n,
                concurrency=concurrency,
                repetition=2,
                workload=workload,
                context=context,
                safety=safety,
                hardware=hardware,
                git_sha=git_sha,
            )
            stable, why = _point_is_stable(second, scale_config, workload)
            comparison = _compare_repeats(first, second, scale_config)
            points.append(
                {
                    "declared_population": n,
                    "participants_per_round": int(workload["clients_per_round"]),
                    "configured_concurrency": concurrency,
                    "status": "STABLE_CONFIRMATION" if stable else second["status"],
                    "reason": None if stable else why,
                    "trial_id": second["trial_id"],
                    "trial_summary_ref": second["summary_ref"],
                    "result_run_id": second["result_run_id"],
                    "warm_round_wall_seconds": second["warm_round_wall_seconds"],
                    "peak_tree_rss_bytes": second["peak_tree_rss_bytes"],
                }
            )
            repeatability = {
                "population": n,
                "confirmation_stable": stable,
                "confirmation_reason": None if stable else why,
                **comparison,
            }
            if stable and comparison["repeatable"]:
                chosen = n
                break

    local_gate = "PASS" if chosen is not None else "FAIL"
    summary = {
        "schema": "fl_scale_summary_v1",
        "version": "1",
        "status": "COMPLETE" if chosen is not None else "INCOMPLETE",
        "local_gate": local_gate,
        "purpose": _PURPOSE,
        "scientific": False,
        "hardware": hardware,
        "concurrency_decision": decision,
        "benchmark_pool": inputs["benchmark_pool"],
        "taskexample_counts_before_benchmark_filter": inputs[
            "taskexample_counts_before_benchmark_filter"
        ],
        "points": points,
        "chosen_population": chosen,
        "repeatability": repeatability,
        "local_gate_note": (
            "Local technical acceptance for this bundle only. It is not a GitHub closure "
            "and no issue is Done before the human merges the delivering PR."
        ),
        "limitations": limitations,
    }
    _write_json(_SCALE_SUMMARY, summary)
    return summary


# ---------------------------------------------------------------------------
# Stage: profile (#37)
# ---------------------------------------------------------------------------


def stage_profile(
    workload: dict[str, Any],
    scale_config: dict[str, Any],
    profile_config: dict[str, Any],
    *,
    hardware: dict[str, Any],
    git_sha: str,
    budget_deadline: float,
) -> dict[str, Any]:
    scale_summary = _load_json(_SCALE_SUMMARY)
    if scale_summary.get("local_gate") != "PASS":
        summary = {
            "schema": "fl_profile_summary_v1",
            "version": "1",
            "status": "NOT_RUN_DEPENDENCY",
            "reason": (
                "#37 requires the documented #35 local acceptance gate to pass first; "
                f"the gate is {scale_summary.get('local_gate')}"
            ),
            "points": [],
            "scientific_protocol": False,
        }
        _write_json(_PROFILE_SUMMARY, summary)
        _write_json(
            _RUNTIME_RECOMMENDATIONS,
            {
                "schema": "fl_runtime_recommendations_v1",
                "version": "1",
                "status": "NOT_RUN_DEPENDENCY",
                "scientific_protocol": False,
                "reason": summary["reason"],
            },
        )
        return summary

    population = int(scale_summary["chosen_population"])
    scale_concurrency = int(scale_summary["concurrency_decision"]["concurrency"])
    safety = scale_config["safety"]
    context = _load_prepared_context(workload)
    preflight = profile_config["concurrency_4_preflight"]

    points: list[dict[str, Any]] = []
    stop_escalation = False
    for concurrency in profile_config["concurrency_points"]:
        if stop_escalation:
            points.append(
                {
                    "configured_concurrency": concurrency,
                    "status": "NOT_ATTEMPTED_SAFETY",
                    "reason": "a lower concurrency already failed; escalation is not attempted",
                }
            )
            continue
        if concurrency >= 4 and not (
            hardware["logical_cpus"] >= preflight["min_logical_cpus"]
            and hardware["total_ram_bytes"] / GIB >= preflight["min_total_ram_gib"]
            and hardware["available_ram_bytes"] / GIB >= preflight["min_available_ram_gib"]
        ):
            points.append(
                {
                    "configured_concurrency": concurrency,
                    "status": "NOT_ATTEMPTED_SAFETY",
                    "reason": "hardware does not satisfy the declared concurrency-4 preflight",
                }
            )
            continue
        if (
            concurrency > 1
            and hardware["available_ram_bytes"] / GIB
            < scale_config["concurrency_policy"]["default_min_available_ram_gib"]
        ):
            points.append(
                {
                    "configured_concurrency": concurrency,
                    "status": "NOT_ATTEMPTED_SAFETY",
                    "reason": "available RAM is below the declared multi-worker admission bound",
                }
            )
            continue

        trials: list[dict[str, Any]] = []
        reuse_ok = (
            concurrency == scale_concurrency
            and profile_config["reuse_identical_successful_scale_trials"]
        )
        for repetition in (1, 2):
            tid = trial_id(population, int(workload["clients_per_round"]), concurrency, repetition)
            existing = _REPO_ROOT / _SCALE_EVIDENCE_DIR / "trials" / tid / "trial_summary.v1.json"
            if reuse_ok and existing.is_file():
                summary_json = json.loads(existing.read_text(encoding="utf-8"))
                if summary_json["status"] == "SUCCEEDED":
                    measurement_uri = summary_json["measurement_ref"]["uri"]
                    measured_peak = json.loads(
                        (_REPO_ROOT / measurement_uri).read_text(encoding="utf-8")
                    )["peak_tree_rss_bytes"]
                    trials.append(
                        {
                            "trial_id": tid,
                            "source": "REUSED_FROM_SCALE",
                            "warm_round_wall_seconds": summary_json["warm_round_wall_seconds"],
                            "peak_tree_rss_bytes": measured_peak,
                            "summary_uri": (
                                _SCALE_EVIDENCE_DIR / "trials" / tid / "trial_summary.v1.json"
                            ).as_posix(),
                        }
                    )
                    continue
            if time.monotonic() > budget_deadline:
                trials.append({"trial_id": tid, "source": "NOT_ATTEMPTED_BUDGET"})
                continue
            record = _run_one_trial(
                population=population,
                concurrency=concurrency,
                repetition=repetition,
                workload=workload,
                context=context,
                safety=safety,
                hardware=hardware,
                git_sha=git_sha,
            )
            stable, why = _point_is_stable(record, scale_config, workload)
            trials.append(
                {
                    "trial_id": record["trial_id"],
                    "source": "MEASURED",
                    "status": record["status"],
                    "stable": stable,
                    "reason": None if stable else why,
                    "warm_round_wall_seconds": record["warm_round_wall_seconds"],
                    "peak_tree_rss_bytes": record["peak_tree_rss_bytes"],
                    "summary_uri": record["summary_ref"]["uri"],
                }
            )
            if not stable and record["status"] in {"FAILED_RESOURCE_GUARD", "FAILED_TIMEOUT"}:
                stop_escalation = True
                break

        measured = [t for t in trials if t.get("source") in {"MEASURED", "REUSED_FROM_SCALE"}]
        twice_stable = len(measured) == 2 and all(
            t.get("stable", True) for t in measured if t["source"] == "MEASURED"
        )
        pooled = [value for t in measured for value in _pooled_warm_samples(t)]
        points.append(
            {
                "configured_concurrency": concurrency,
                "declared_population": population,
                "participants_per_round": int(workload["clients_per_round"]),
                "status": "TWICE_STABLE" if twice_stable else "PARTIAL",
                "trials": trials,
                "pooled_warm_round_wall_seconds": summarize_samples(pooled),
                "max_trial_tree_rss_bytes": max(
                    [
                        t["peak_tree_rss_bytes"]
                        for t in measured
                        if isinstance(t.get("peak_tree_rss_bytes"), int)
                    ]
                    or [0]
                )
                or None,
            }
        )

    return _finalize_profile(points, population, hardware, workload, scale_summary)


def _pooled_warm_samples(trial: dict[str, Any]) -> list[float]:
    uri = trial.get("summary_uri")
    if not uri:
        return []
    path = _REPO_ROOT / uri
    if not path.is_file():
        return []
    summary = json.loads(path.read_text(encoding="utf-8"))
    excluded = set(summary.get("warm_rounds_excluded", []))
    return [
        row["round_wall_seconds"]
        for row in summary.get("rounds", [])
        if row["server_round"] not in excluded and row.get("round_wall_seconds") is not None
    ]


def _finalize_profile(
    points: list[dict[str, Any]],
    population: int,
    hardware: dict[str, Any],
    workload: dict[str, Any],
    scale_summary: dict[str, Any],
) -> dict[str, Any]:
    twice_stable = [p for p in points if p["status"] == "TWICE_STABLE"]
    comparative = len(twice_stable) >= 2

    normal = None
    low_memory = None
    if twice_stable:
        normal = min(
            twice_stable,
            key=lambda p: (
                p["pooled_warm_round_wall_seconds"]["p50"]
                if p["pooled_warm_round_wall_seconds"]["p50"] is not None
                else float("inf"),
                p["configured_concurrency"],
            ),
        )["configured_concurrency"]
        low_memory = min(
            twice_stable,
            key=lambda p: (
                p["max_trial_tree_rss_bytes"] if p["max_trial_tree_rss_bytes"] else float("inf"),
                p["configured_concurrency"],
            ),
        )["configured_concurrency"]

    summary = {
        "schema": "fl_profile_summary_v1",
        "version": "1",
        "status": "COMPLETE" if comparative else "PARTIAL",
        "purpose": _PURPOSE,
        "scientific_protocol": False,
        "declared_population": population,
        "participants_per_round": int(workload["clients_per_round"]),
        "points": points,
        "comparative_recommendation_supported": comparative,
        "timing_definitions": {
            "startup_seconds": "simulation entry to the first train configure call",
            "data_load_seconds": "client prepared-slice read, filter and batch construction",
            "local_fit_seconds": "around adapter.fit on each client",
            "dispatch_to_replies_seconds": (
                "train configure completion to aggregate entry; transport, scheduling and "
                "client processing, not the sum of client fits"
            ),
            "aggregation_seconds": "aggregation plus the required oracle checks",
            "server_evaluation_seconds": "around the server diagnostic evaluation",
            "round_wall_seconds": "configure start through that round's evaluation completion",
            "trial_wall_seconds": "supervised subprocess launch to clean exit",
        },
        "statistics": {
            "p50": "statistics.median",
            "p95": "nearest rank, sorted[ceil(0.95 * n) - 1]",
            "warm_rounds": workload["warmup_rounds_excluded_from_timing_summary"],
            "samples_per_trial": 5,
            "pooled_samples_per_point": 10,
            "caveat": "A p95 over ten observations is descriptive, not a production tail bound.",
        },
        "measurement_limitations": [
            "Sampled process-tree RSS can double-count shared pages.",
            "GPU memory is NOT_APPLICABLE on this CPU-only run, not a measured zero.",
            "Configured concurrency is a cap; observed overlap is reported separately.",
            "Per-process RSS, tree RSS and system available memory are never mixed.",
        ],
        "scale_summary_ref": {
            "uri": _SCALE_SUMMARY.as_posix(),
            "sha256": file_sha256(_REPO_ROOT / _SCALE_SUMMARY),
        },
    }
    _write_json(_PROFILE_SUMMARY, summary)

    recommendations = {
        "schema": "fl_runtime_recommendations_v1",
        "version": "1",
        "status": "COMPLETE" if comparative else "PARTIAL",
        "scientific_protocol": False,
        "hardware": hardware,
        "workload_ref": {
            "uri": _WORKLOAD_CONFIG.as_posix(),
            "sha256": file_sha256(_REPO_ROOT / _WORKLOAD_CONFIG),
        },
        "chosen_population": population,
        "participants_per_round": int(workload["clients_per_round"]),
        "normal_concurrency": normal,
        "normal_rationale": (
            "lowest pooled warm-round p50 among twice-stable points; ties resolve to lower "
            "concurrency"
        ),
        "low_memory_concurrency": low_memory,
        "low_memory_rationale": (
            "lowest maximum trial process-tree RSS among twice-stable points; ties resolve to "
            "lower concurrency"
        ),
        "admissible_configurations": [
            p["configured_concurrency"] for p in points if p["status"] == "TWICE_STABLE"
        ],
        "failed_or_skipped_configurations": [
            {"configured_concurrency": p["configured_concurrency"], "status": p["status"]}
            for p in points
            if p["status"] != "TWICE_STABLE"
        ],
        "comparative_recommendation_supported": comparative,
        "limitations": [
            "Measured on one machine with one stub workload; not a guarantee elsewhere.",
            "An untested lower-memory machine is never claimed to be safe.",
        ],
    }
    _write_json(_RUNTIME_RECOMMENDATIONS, recommendations)
    return summary


# ---------------------------------------------------------------------------
# Stage: verify
# ---------------------------------------------------------------------------


def stage_verify() -> dict[str, Any]:
    """Re-verify every saved URI and hash without rerunning any measurement."""
    checked: list[dict[str, Any]] = []
    failures: list[str] = []

    def _check(uri: str, sha: str, source: str) -> None:
        path = _REPO_ROOT / uri
        if not path.is_file():
            failures.append(f"{source}: missing {uri}")
            checked.append({"uri": uri, "source": source, "ok": False, "reason": "MISSING"})
            return
        actual = (
            verify_vocabulary(path)
            if uri.endswith("vocabulary_v1.proposed.json")
            else file_sha256(path)
        )
        ok = actual == sha
        if not ok:
            failures.append(f"{source}: SHA mismatch for {uri}")
        checked.append({"uri": uri, "source": source, "ok": ok})

    def _walk(node: Any, source: str) -> None:
        if isinstance(node, dict):
            if isinstance(node.get("uri"), str) and isinstance(node.get("sha256"), str):
                _check(node["uri"], node["sha256"], source)
            for value in node.values():
                _walk(value, source)
        elif isinstance(node, list):
            for value in node:
                _walk(value, source)

    # Required deliverables. A missing one is a verification failure, never a skip:
    # an empty or partial evidence directory must not be able to report PASS.
    required = [
        _RUNTIME_INPUTS,
        _RUNTIME_INITIALIZATION,
        _SCALE_SUMMARY,
        _PROFILE_SUMMARY,
        _RUNTIME_RECOMMENDATIONS,
        Path("docs/evidence/s2-pr-02/convergence_validation.v1.json"),
    ]
    documents: list[Path] = []
    for doc in required:
        if not (_REPO_ROOT / doc).is_file():
            failures.append(f"required deliverable missing: {doc.as_posix()}")
            continue
        documents.append(doc)

    # Follow the summaries themselves to the trials they declare.
    declared_trials: set[str] = set()
    for summary_path in (_SCALE_SUMMARY, _PROFILE_SUMMARY):
        full = _REPO_ROOT / summary_path
        if not full.is_file():
            continue
        declared_trials |= set(
            re.findall(r"trials/(n\d+-m\d+-c\d+-rep\d+)", full.read_text(encoding="utf-8"))
        )
    for name in sorted(declared_trials):
        trial_dir = _SCALE_EVIDENCE_DIR / "trials" / name
        for leaf in (
            "experiment_config.v1.json",
            "runtime_measurement.v1.json",
            "trial_summary.v1.json",
        ):
            doc = trial_dir / leaf
            if not (_REPO_ROOT / doc).is_file():
                failures.append(f"declared trial {name} is missing {leaf}")
                continue
            documents.append(doc)

    validated_results = 0
    for doc in documents:
        payload = json.loads((_REPO_ROOT / doc).read_text(encoding="utf-8"))
        _walk(payload, doc.as_posix())
        if payload.get("schema") == "experiment_config_v1":
            validate_experiment_config(payload)
        run_id = payload.get("result_ref", {}).get("uri") if isinstance(payload, dict) else None
        if run_id:
            result_path = _REPO_ROOT / run_id
            if not result_path.is_file():
                failures.append(f"{doc.as_posix()}: declared result is missing {run_id}")
            else:
                validate_experiment_result(json.loads(result_path.read_text(encoding="utf-8")))
                validated_results += 1

    if not declared_trials:
        failures.append("no trial is declared by the scale or profile summary")

    return {
        "status": "PASS" if not failures else "FAIL",
        "checked_reference_count": len(checked),
        "required_deliverables": len(required),
        "declared_trials": sorted(declared_trials),
        "validated_result_documents": validated_results,
        "failures": failures,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="S2-PR-01/03 federated runtime benchmark.")
    parser.add_argument(
        "--stage", choices=["prepare", "scale", "profile", "verify", "all"], default="all"
    )
    parser.add_argument("--workload-config", default=_WORKLOAD_CONFIG.as_posix())
    parser.add_argument("--scale-config", default=_SCALE_CONFIG.as_posix())
    parser.add_argument("--profile-config", default=_PROFILE_CONFIG.as_posix())
    parser.add_argument("--trial-spec", default=None, help="Internal child entry point.")
    args = parser.parse_args()

    if args.trial_spec:
        spec = json.loads(Path(args.trial_spec).read_text(encoding="utf-8"))
        payload = run_trial_child(spec)
        Path(spec["output_path"]).write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8", newline="\n"
        )
        return 0

    workload = _load_json(Path(args.workload_config))
    scale_config = _load_json(Path(args.scale_config))
    profile_config = _load_json(Path(args.profile_config))
    hardware = capture_hardware()
    git_sha = _git_head()
    budget_deadline = time.monotonic() + float(
        scale_config["safety"]["maximum_real_execution_budget_seconds"]
    )

    exit_code = 0
    if args.stage in {"prepare", "all"}:
        prepared = stage_prepare(workload, scale_config)
        print(f"[prepare] eligible benchmark pool: {prepared['eligible_pool_count']} clients")

    if args.stage in {"scale", "all"}:
        summary = stage_scale(
            workload,
            scale_config,
            hardware=hardware,
            git_sha=git_sha,
            budget_deadline=budget_deadline,
        )
        print(f"[scale] status={summary['status']} local_gate={summary['local_gate']}")
        if summary["local_gate"] != "PASS":
            exit_code = 1

    if args.stage == "all":
        print("[convergence] validating the S2-PR-02 fixture oracles")
        result = subprocess.run(
            [
                sys.executable,
                str(_REPO_ROOT / "scripts/federated/validate_convergence.py"),
                "--config",
                _CONVERGENCE_CONFIG.as_posix(),
            ],
            cwd=str(_REPO_ROOT),
            shell=False,
            check=False,
        )
        if result.returncode != 0:
            exit_code = 1

    if args.stage in {"profile", "all"}:
        summary = stage_profile(
            workload,
            scale_config,
            profile_config,
            hardware=hardware,
            git_sha=git_sha,
            budget_deadline=budget_deadline,
        )
        print(f"[profile] status={summary['status']}")

    if args.stage in {"verify", "all"}:
        verification = stage_verify()
        print(
            f"[verify] {verification['status']}: "
            f"{verification['checked_reference_count']} references checked"
        )
        for failure in verification["failures"]:
            print(f"  FAIL {failure}")
        if verification["status"] != "PASS":
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
