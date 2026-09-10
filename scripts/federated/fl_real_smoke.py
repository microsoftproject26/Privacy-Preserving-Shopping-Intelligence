"""S1-PR-07 — Real REES46 Federated Smoke Test.

NON_SCIENTIFIC_REAL_DATA_PIPELINE_SMOKE

This script proves the end-to-end path:
    real frozen T1 TaskExamples
    → real C1 users / #19 opaque client IDs
    → deterministic task-eligible client sampling
    → Phase1Batch (zero-history)
    → LocalTrainerCore + FlowerLocalAdapter
    → Flower/Ray FedAvg
    → structured evidence/result

It does NOT prove recommendation quality, final R2a, final GRU, R1 comparison,
QR, communication efficiency, or TEST performance.

Usage (canonical):
    uv run --locked python scripts/federated/fl_real_smoke.py --config config/fl_real_smoke.v1.json
"""

from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import logging
import math
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import flwr
import polars as pl
import ray
import torch
import torch.nn.functional as F
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from flwr.simulation import run_simulation
from torch import Tensor, optim

from ppsi.federated.clients import (
    build_client_manifest,
    manifest_content_sha256,
)
from ppsi.federated.sampling import (
    DEFAULT_SAMPLER_VERSION,
    build_trace,
    sample_clients,
)
from ppsi.federated.task_examples import (
    default_batch_spec,
    load_t1_category_spec,
    load_t1_client_counts,
    make_t1_smoke_batches,
    prepare_t1_smoke_slices,
    verify_t1_task_example_file,
    verify_vocabulary,
)
from ppsi.training.batch import Phase1BatchSpec
from ppsi.training.core import LocalTrainerCore
from ppsi.training.flower import ContributingRowsSmokeWeightPolicy, FlowerLocalAdapter
from ppsi.training.identity import file_sha256
from ppsi.training.objective import ContractSmokeObjective
from ppsi.training.result import build_experiment_result, make_metric_record
from ppsi.training.state import pack_shared_state
from ppsi.training.stub_model import Phase1StubModel, StubModelConfig
from scripts.experiments.results import validate_result_for_reporting
from scripts.experiments.schemas import (
    validate_experiment_config,
    validate_experiment_result,
)
from scripts.federated.fl_synthetic_smoke import (
    SmokeValidationError,
    get_digest,
    weighted_average_state_dicts,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def server_round_to_sampler_round(server_round: int) -> int:
    """Map Flower 1-indexed server_round to 0-indexed sampler round_index."""
    if server_round < 1:
        raise ValueError(f"server_round must be >= 1, got {server_round}")
    return server_round - 1


def sort_client_replies_by_logical_id(replies: list[Message]) -> list[Message]:
    """Sort Flower client reply messages deterministically by opaque logical_client_id."""

    def _sort_key(msg: Message) -> str:
        return str(
            msg.content.configs_records.get("config", ConfigRecord()).get("logical_client_id", "")
        )

    return sorted(replies, key=_sort_key)


def assert_no_client_leakage(paths_to_check: list[Path]) -> None:
    """Ensure no opaque client IDs (client-v1-[0-9a-f]{64}) leak into public files."""
    pattern = re.compile(r"client-v1-[0-9a-f]{64}")
    for p in paths_to_check:
        if not p.is_file():
            continue
        content = p.read_text(encoding="utf-8", errors="replace")
        matches = pattern.findall(content)
        if matches:
            raise SmokeValidationError(
                f"Privacy leak detected in public artifact {p}: found {len(matches)} client IDs: "
                f"{matches[:3]}..."
            )
        if p.suffix == ".json":
            try:
                data = json.loads(content)
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
                data = None
            if isinstance(data, dict) and "selected_client_ids" in data:
                raise SmokeValidationError(f"Public artifact {p} contains selected_client_ids")


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def validate_smoke_config(config: dict[str, Any]) -> None:
    if config.get("schema") != "fl_real_smoke_v1":
        raise SmokeValidationError(f"Invalid config schema: {config.get('schema')!r}")
    if config.get("seed") != 13:
        raise SmokeValidationError("seed must be 13")
    if config.get("num_rounds", 0) != 3:
        raise SmokeValidationError("num_rounds must be 3")
    if config.get("clients_per_round", 0) != 4:
        raise SmokeValidationError("clients_per_round must be 4")
    if config.get("repeat_runs", 0) < 2:
        raise SmokeValidationError("repeat_runs must be >= 2")


def load_smoke_config(config_path: Path) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    validate_smoke_config(config)
    return config


# ---------------------------------------------------------------------------
# Artifact helpers
# ---------------------------------------------------------------------------


def make_artifact_ref(
    logical_id: str,
    artifact_schema: str,
    uri: str,
    sha256: str,
) -> dict[str, Any]:
    """Build an ArtifactRef v1 record."""
    return {
        "schema": "artifact_ref_v1",
        "version": "1",
        "logical_id": logical_id,
        "artifact_schema": artifact_schema,
        "artifact_version": "1",
        "uri": uri,
        "sha256": sha256,
    }


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

_EXPECTED_CATEGORY_COUNT = 588
_MODEL_ID = "phase1_stub_t1_real_data_smoke_NON_SCIENTIFIC"


def build_smoke_model(category_count: int, spec: Phase1BatchSpec) -> Phase1StubModel:
    return Phase1StubModel(
        batch_spec=spec,
        config=StubModelConfig(
            category_count=category_count,
            embedding_dim=4,
            hidden_dim=8,
        ),
    )


# ---------------------------------------------------------------------------
# Server-side evaluation
# ---------------------------------------------------------------------------


def server_evaluate(
    model: Phase1StubModel,
    val_df: pl.DataFrame,
    spec: Phase1BatchSpec,
) -> dict[str, float]:
    """Evaluate model on validation smoke slice. Returns cross_entropy, accuracy@1, support."""
    model.eval()
    loss_sum = 0.0
    correct = 0
    support = 0
    batches = make_t1_smoke_batches(val_df, spec=spec, batch_size=8)
    with torch.no_grad():
        for batch in batches:
            output = model(batch)
            logits = output.t1_logits[batch.t1_present]
            targets = batch.t1_target[batch.t1_present]
            loss_sum += float(F.cross_entropy(logits, targets, reduction="sum").item())
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            support += len(targets)
    if support == 0:
        raise SmokeValidationError("Validation slice has no T1 rows")
    cross_entropy = loss_sum / support
    accuracy_at_1 = correct / support
    if not torch.isfinite(torch.tensor(cross_entropy)):
        raise SmokeValidationError(f"Non-finite cross_entropy: {cross_entropy}")
    if not (0.0 <= accuracy_at_1 <= 1.0):
        raise SmokeValidationError(f"accuracy@1={accuracy_at_1} out of [0,1]")
    return {
        "cross_entropy": cross_entropy,
        "accuracy_at_1": accuracy_at_1,
        "support": support,
    }


# ---------------------------------------------------------------------------
# RealDataTracingFedAvg
# ---------------------------------------------------------------------------


class RealDataTracingFedAvg(FedAvg):
    """FedAvg subclass for S1-PR-07 real-data smoke.

    Adds:
    - Deterministic client ID injection from precomputed sampling.
    - Sort replies by opaque logical_client_id before oracle and Flower aggregation.
    - Oracle FedAvg cross-check at each round.
    - Redistribution digest verification.
    """

    def __init__(
        self,
        *,
        expected_clients: int,
        aggregation_atol: float,
        selected_ids_by_round: dict[int, list[str]],
        selection_digests_by_round: dict[int, str],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.expected_clients = expected_clients
        self.aggregation_atol = aggregation_atol
        self.selected_ids_by_round = selected_ids_by_round
        self.selection_digests_by_round = selection_digests_by_round
        self.previous_aggregate_digest: str | None = None
        self.tracing_log: list[dict[str, Any]] = []

    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ):
        state_dict = arrays.to_torch_state_dict()
        current_digest = get_digest(state_dict)

        # Verify redistribution
        if (
            server_round > 1
            and self.previous_aggregate_digest is not None
            and current_digest != self.previous_aggregate_digest
        ):
            raise SmokeValidationError(
                f"Redistribution failure in round {server_round}: "
                f"expected {self.previous_aggregate_digest}, got {current_digest}"
            )

        config["server_digest"] = current_digest

        messages = list(super().configure_train(server_round, arrays, config, grid))
        if len(messages) != self.expected_clients:
            raise SmokeValidationError(
                f"Expected {self.expected_clients} clients, got {len(messages)}"
            )

        # Inject precomputed opaque client IDs (round_index = server_round - 1)
        round_index = server_round_to_sampler_round(server_round)
        sampled_ids = self.selected_ids_by_round[round_index]
        if len(sampled_ids) != len(messages):
            raise SmokeValidationError(
                f"Round {server_round}: sampled {len(sampled_ids)} IDs but {len(messages)} messages"
            )

        for i, msg in enumerate(messages):
            msg.content = copy.deepcopy(msg.content)
            new_conf = ConfigRecord(msg.content.configs_records.get("config", ConfigRecord()))
            new_conf["logical_client_id"] = sampled_ids[i]
            new_conf["server_round"] = server_round
            msg.content.configs_records["config"] = new_conf

        self.tracing_log.append(
            {
                "server_round": server_round,
                "sampler_round_index": round_index,
                "selection_digest": self.selection_digests_by_round[round_index],
                "server_input_digest": current_digest,
                "selected_client_ids": list(sampled_ids),
                "selected_client_count": len(sampled_ids),
                "clients": [],
                "aggregation_oracle_pass": False,
                "aggregated_digest": None,
                "contributing_examples": 0,
                "max_abs_diff": None,
            }
        )
        return messages

    def aggregate_train(self, server_round: int, replies):
        replies_list = list(replies)
        if not replies_list:
            raise SmokeValidationError("No client replies received")

        tracing_entry = self.tracing_log[-1]

        # Fail on any client error
        for msg in replies_list:
            if msg.has_error():
                raise SmokeValidationError(f"Client error in round {server_round}: {msg.error}")

        # Sort replies by opaque logical_client_id to eliminate Ray nondeterminism
        sorted_replies = sort_client_replies_by_logical_id(replies_list)

        updates: list[tuple[dict[str, Tensor], int]] = []
        total_examples = 0
        for msg in sorted_replies:
            arr_rec = msg.content.parameters_records["arrays"]
            met_rec = msg.content.metrics_records["metrics"]
            conf_rec = msg.content.configs_records["config"]

            received = str(conf_rec.get("received_digest", ""))
            expected = tracing_entry["server_input_digest"]
            if received != expected:
                raise SmokeValidationError(
                    f"Client digest mismatch: expected {expected}, got {received}"
                )

            sd = arr_rec.to_torch_state_dict()
            num_examples = int(met_rec["num-examples"])
            if num_examples <= 0:
                raise SmokeValidationError("Client returned non-positive num-examples")
            updates.append((sd, num_examples))
            total_examples += num_examples

            tracing_entry["clients"].append(
                {
                    "logical_client_id": str(conf_rec.get("logical_client_id", "")),
                    "received_digest": received,
                    "updated_digest": str(conf_rec.get("updated_digest", "")),
                    "num_examples": num_examples,
                    "local_train_loss": float(conf_rec.get("local_train_loss", 0.0)),
                }
            )

        # Pure oracle FedAvg
        oracle_result = weighted_average_state_dicts(updates)

        # Flower FedAvg
        flower_array_record, metrics = super().aggregate_train(server_round, sorted_replies)
        if flower_array_record is None:
            raise SmokeValidationError("Flower aggregation returned None")

        flower_state = flower_array_record.to_torch_state_dict()

        # Cross-check oracle vs Flower
        max_abs_diff = 0.0
        for key in oracle_result:
            if key not in flower_state:
                raise SmokeValidationError(f"Key {key} missing from Flower state")
            diff = float(torch.abs(oracle_result[key] - flower_state[key]).max().item())
            max_abs_diff = max(max_abs_diff, diff)
            if diff > self.aggregation_atol:
                raise SmokeValidationError(
                    f"Oracle mismatch for {key}: max_abs_diff={diff} > atol={self.aggregation_atol}"
                )

        tracing_entry["aggregation_oracle_pass"] = True
        tracing_entry["max_abs_diff"] = max_abs_diff
        tracing_entry["contributing_examples"] = total_examples

        new_digest = get_digest(flower_state)
        self.previous_aggregate_digest = new_digest
        tracing_entry["aggregated_digest"] = new_digest

        return flower_array_record, metrics


# ---------------------------------------------------------------------------
# Global state between client and server workers (same pattern as reference)
# ---------------------------------------------------------------------------

_TRAIN_SLICE_PATH: str = ""
_SPEC: Phase1BatchSpec | None = None
_CATEGORY_COUNT: int = 0
_SMOKE_CONFIG: dict[str, Any] = {}

# Results accumulator
_SMOKE_RESULT = None
_SMOKE_TRACING_LOG: list[dict[str, Any]] = []


# ---------------------------------------------------------------------------
# ClientApp
# ---------------------------------------------------------------------------

real_client_app = ClientApp()


@real_client_app.train()
def real_train(message: Message, context: Context) -> Message:
    torch.manual_seed(13)
    torch.set_num_threads(1)

    config_rec = message.content.configs_records.get("config", ConfigRecord())
    logical_client_id: str = str(config_rec["logical_client_id"])
    server_round: int = int(config_rec["server_round"])

    # Load incoming state
    array_record = message.content.parameters_records["arrays"]
    state_dict = array_record.to_torch_state_dict()
    received_digest = get_digest(state_dict)

    spec = _SPEC if _SPEC is not None else default_batch_spec()
    model = build_smoke_model(_CATEGORY_COUNT or _EXPECTED_CATEGORY_COUNT, spec)
    model.load_state_dict(state_dict)

    # Load client batches from tiny train slice
    train_df = pl.read_parquet(_TRAIN_SLICE_PATH)
    client_df = train_df.filter(pl.col("client_id") == logical_client_id)
    batches = make_t1_smoke_batches(client_df, spec=spec, batch_size=8)

    if not batches:
        raise SmokeValidationError(
            f"Client {logical_client_id[:16]}... has zero batches in train slice."
        )

    lr = float(_SMOKE_CONFIG.get("learning_rate", 0.02))
    momentum = float(_SMOKE_CONFIG.get("momentum", 0.0))
    local_epochs = int(_SMOKE_CONFIG.get("local_epochs", 1))

    opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    core = LocalTrainerCore(
        model=model,
        batch_spec=spec,
        objective=ContractSmokeObjective(),
        optimizer=opt,
        device="cpu",
    )
    adapter = FlowerLocalAdapter(
        core=core,
        shared_state_spec=model.shared_state_spec(),
        aggregation_weight_policy=ContributingRowsSmokeWeightPolicy(),
    )

    all_batches = batches * local_epochs
    result = adapter.fit(state_dict, all_batches, outer_round=server_round)

    if result.aggregation_weight <= 0:
        raise SmokeValidationError(
            f"Client {logical_client_id[:16]}... returned zero aggregation weight."
        )

    updated_state = result.shared_state
    updated_digest = result.updated_state_digest
    scalar_metrics = result.scalar_metrics()
    train_loss = float(scalar_metrics.get("train_loss", 0.0))

    updated_array_record = ArrayRecord.from_torch_state_dict(dict(updated_state))
    metrics_rec = MetricRecord(
        {
            "num-examples": result.aggregation_weight,
            "train_loss": train_loss,
        }
    )
    configs_rec = ConfigRecord(
        {
            "logical_client_id": logical_client_id,
            "received_digest": received_digest,
            "updated_digest": updated_digest,
            "local_train_loss": train_loss,
        }
    )
    record_dict = RecordDict()
    record_dict.parameters_records["arrays"] = updated_array_record
    record_dict.metrics_records["metrics"] = metrics_rec
    record_dict.configs_records["config"] = configs_rec
    return message.create_reply(record_dict)


# ---------------------------------------------------------------------------
# ServerApp factory
# ---------------------------------------------------------------------------


def get_server_app(
    config: dict[str, Any],
    *,
    selected_ids_by_round: dict[int, list[str]],
    selection_digests_by_round: dict[int, str],
    val_df: pl.DataFrame,
    spec: Phase1BatchSpec,
    initial_state: dict[str, Tensor],
    category_count: int,
) -> ServerApp:
    app = ServerApp()

    @app.main()
    def main(grid: Grid, context: Context) -> None:
        global _SMOKE_RESULT, _SMOKE_TRACING_LOG

        torch.manual_seed(config["seed"])
        torch.set_num_threads(1)

        num_rounds = config["num_rounds"]
        clients_per_round = config["clients_per_round"]
        aggregation_atol = config.get("aggregation_atol", 1e-6)

        initial_arrays = ArrayRecord.from_torch_state_dict(initial_state)
        train_config = ConfigRecord(
            {
                "learning_rate": config.get("learning_rate", 0.02),
                "local_epochs": config.get("local_epochs", 1),
            }
        )

        strategy = RealDataTracingFedAvg(
            expected_clients=clients_per_round,
            aggregation_atol=aggregation_atol,
            selected_ids_by_round=selected_ids_by_round,
            selection_digests_by_round=selection_digests_by_round,
            fraction_train=1.0,
            fraction_evaluate=0.0,
            min_train_nodes=clients_per_round,
            min_evaluate_nodes=0,
            min_available_nodes=clients_per_round,
            weighted_by_key="num-examples",
        )

        def evaluate_fn(server_round: int, arrays: ArrayRecord) -> MetricRecord:
            state = arrays.to_torch_state_dict()
            m = build_smoke_model(category_count, spec)
            m.load_state_dict(state)
            metrics = server_evaluate(m, val_df, spec)
            return MetricRecord(
                {
                    "cross_entropy": metrics["cross_entropy"],
                    "accuracy_at_1": metrics["accuracy_at_1"],
                    "support": float(metrics["support"]),
                }
            )

        result = strategy.start(
            grid=grid,
            initial_arrays=initial_arrays,
            num_rounds=num_rounds,
            train_config=train_config,
            evaluate_fn=evaluate_fn,
        )
        _SMOKE_RESULT = result
        _SMOKE_TRACING_LOG = strategy.tracing_log

    return app


def run_smoke_simulation(
    config: dict[str, Any],
    *,
    selected_ids_by_round: dict[int, list[str]],
    selection_digests_by_round: dict[int, str],
    val_df: pl.DataFrame,
    spec: Phase1BatchSpec,
    initial_state: dict[str, Tensor],
    category_count: int,
):
    global _SMOKE_RESULT, _SMOKE_TRACING_LOG
    _SMOKE_RESULT = None
    _SMOKE_TRACING_LOG = []

    server_app = get_server_app(
        config,
        selected_ids_by_round=selected_ids_by_round,
        selection_digests_by_round=selection_digests_by_round,
        val_df=val_df,
        spec=spec,
        initial_state=initial_state,
        category_count=category_count,
    )
    run_simulation(
        server_app=server_app,
        client_app=real_client_app,
        num_supernodes=config["clients_per_round"],
        backend_name="ray",
        backend_config={"client_resources": {"num_cpus": 1}},
    )
    return _SMOKE_RESULT, list(_SMOKE_TRACING_LOG)


# ---------------------------------------------------------------------------
# Public trace sanitizer (privacy)
# ---------------------------------------------------------------------------


def sanitize_trace_for_public(
    tracing_log: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove raw opaque client ID lists from trace records for public evidence."""
    public = []
    for entry in tracing_log:
        sanitized: dict[str, Any] = {
            "server_round": entry.get("server_round"),
            "sampler_round_index": entry.get("sampler_round_index"),
            "selection_digest": entry.get("selection_digest", ""),
            "selected_client_count": entry.get("selected_client_count"),
            "successful_client_count": len(entry.get("clients", [])),
            "contributing_examples": entry.get("contributing_examples", 0),
            "mean_local_train_loss": (
                sum(c.get("local_train_loss", 0.0) for c in entry.get("clients", []))
                / max(len(entry.get("clients", [])), 1)
            ),
            "aggregation_oracle_pass": entry.get("aggregation_oracle_pass", False),
            "max_abs_diff": entry.get("max_abs_diff"),
            "server_input_digest": entry.get("server_input_digest"),
            "aggregated_digest": entry.get("aggregated_digest"),
        }
        # Explicitly assert no raw IDs in public record
        assert "selected_client_ids" not in sanitized
        public.append(sanitized)
    return public


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="S1-PR-07 Real Federated Smoke Test")
    parser.add_argument("--config", type=Path, required=True, help="Path to fl_real_smoke.v1.json")
    args = parser.parse_args()

    global _TRAIN_SLICE_PATH, _SPEC, _CATEGORY_COUNT, _SMOKE_CONFIG

    config = load_smoke_config(args.config)
    _SMOKE_CONFIG = config
    logger.info("Config loaded and validated.")

    repo_root = Path(__file__).resolve().parents[2]
    paths = config["paths"]

    def rp(key: str) -> Path:
        return repo_root / paths[key]

    # === INPUT VERIFICATION ===
    logger.info("Verifying input files...")

    cohort_sha = file_sha256(rp("cohort_manifest"))
    cohort_sha_prefix = cohort_sha[:16]
    expected_cohort_prefix = "32d4b8ce4bb84f78"
    if cohort_sha_prefix != expected_cohort_prefix:
        raise SmokeValidationError(
            f"Cohort manifest SHA-256 prefix mismatch: "
            f"expected {expected_cohort_prefix!r}, got {cohort_sha_prefix!r}"
        )
    logger.info("Cohort manifest: prefix OK.")

    train_sha = verify_t1_task_example_file(
        rp("t1_train"),
        expected_sha_prefix="42b1617c1d2f1b5a",
        expected_row_count=3_113_814,
        expected_split="TRAIN",
        file_label="T1 TRAIN",
    )
    logger.info("T1 TRAIN: verified.")

    val_sha = verify_t1_task_example_file(
        rp("t1_validation"),
        expected_sha_prefix="e5e8522510371741",
        expected_row_count=438_185,
        expected_split="VALIDATION",
        file_label="T1 VALIDATION",
    )
    logger.info("T1 VALIDATION: verified.")

    vocab_sha = verify_vocabulary(
        rp("vocabulary"), expected_category_count=_EXPECTED_CATEGORY_COUNT
    )
    category_count, valid_codes = load_t1_category_spec(rp("vocabulary"))
    if category_count != _EXPECTED_CATEGORY_COUNT:
        raise SmokeValidationError(
            f"Expected category_count == {_EXPECTED_CATEGORY_COUNT}, got {category_count}"
        )
    _CATEGORY_COUNT = category_count

    spec = default_batch_spec()
    _SPEC = spec

    # === BASE CLIENT MANIFEST ===
    expected_logical_hash = "a96fb92fc6c43e458f9d8692491c7927e49d17741cb37e72d303420734820764"
    expected_c1_count = 388789
    base_manifest_path = rp("base_client_manifest")

    if base_manifest_path.is_file():
        logger.info("Loading existing base client manifest...")
        base_manifest = pl.read_parquet(base_manifest_path)
    else:
        logger.info("Regenerating base client manifest from cohort manifest...")
        base_manifest = build_client_manifest(str(rp("cohort_manifest")))
        base_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        base_manifest.write_parquet(base_manifest_path)

    actual_hash = manifest_content_sha256(base_manifest)
    actual_count = len(base_manifest)
    if actual_hash != expected_logical_hash:
        raise SmokeValidationError(
            f"Base manifest logical hash mismatch: expected {expected_logical_hash!r}, got {actual_hash!r}"
        )
    if actual_count != expected_c1_count:
        raise SmokeValidationError(
            f"Base manifest count mismatch: expected {expected_c1_count}, got {actual_count}"
        )
    logger.info(f"Base manifest: {actual_count} C1 clients, hash prefix {actual_hash[:16]}.")

    base_client_ids = set(base_manifest["client_id"].to_list())

    # === T1 CLIENT COUNTS ===
    logger.info("Computing T1 client counts...")
    counts_df = load_t1_client_counts(rp("t1_train"), rp("t1_validation"), base_client_ids)
    t1_unique_train_clients = int(
        counts_df.filter(pl.col("t1_train_example_count") > 0)["client_id"].n_unique()
    )
    t1_unique_val_clients = int(
        counts_df.filter(pl.col("t1_validation_example_count") > 0)["client_id"].n_unique()
    )
    t1_eligible_count = int(
        counts_df.filter(pl.col("eligible_for_t1_smoke"))["client_id"].n_unique()
    )
    logger.info(
        f"T1 clients: train={t1_unique_train_clients}, val={t1_unique_val_clients}, "
        f"eligible={t1_eligible_count}"
    )

    # Write derived manifest
    derived_manifest_df = (
        counts_df.with_columns(
            pl.lit("fl_real_smoke_client_manifest_v1").alias("schema"),
            pl.lit("1").alias("version"),
            pl.lit(train_sha).alias("t1_train_source_sha256"),
            pl.lit(val_sha).alias("t1_validation_source_sha256"),
            pl.lit(actual_hash).alias("base_client_manifest_content_sha256"),
        )
        .select(
            [
                "schema",
                "version",
                "client_id",
                "t1_train_example_count",
                "t1_validation_example_count",
                "eligible_for_t1_smoke",
                "t1_train_source_sha256",
                "t1_validation_source_sha256",
                "base_client_manifest_content_sha256",
            ]
        )
        .sort("client_id")
    )
    derived_manifest_path = rp("derived_client_manifest")
    derived_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    derived_manifest_df.write_parquet(derived_manifest_path)
    derived_manifest_sha = file_sha256(derived_manifest_path)
    logger.info("Derived smoke client manifest written.")

    # === DETERMINISTIC CLIENT SELECTION ===
    seed = config["seed"]
    num_rounds = config["num_rounds"]
    clients_per_round = config["clients_per_round"]
    sampler_version = config.get("sampler_version", DEFAULT_SAMPLER_VERSION)

    eligible_ids = sorted(counts_df.filter(pl.col("eligible_for_t1_smoke"))["client_id"].to_list())
    if len(eligible_ids) < clients_per_round:
        raise SmokeValidationError(
            f"Not enough eligible clients: {len(eligible_ids)} < {clients_per_round}"
        )

    logger.info(f"Precomputing sampling for {num_rounds} rounds ({len(eligible_ids)} eligible)...")
    selected_ids_by_round: dict[int, list[str]] = {}
    selection_digests_by_round: dict[int, str] = {}
    sampling_traces = []

    for round_index in range(num_rounds):
        result = sample_clients(
            eligible_client_ids=eligible_ids,
            experiment_seed=seed,
            round_index=round_index,
            clients_per_round=clients_per_round,
            sampler_version=sampler_version,
        )
        selected_ids_by_round[round_index] = result.selected_client_ids
        selection_digests_by_round[round_index] = result.selected_digest
        trace = build_trace(
            result,
            sampler_version=sampler_version,
            experiment_seed=seed,
            round_index=round_index,
            eligible_pool_count=len(eligible_ids),
            clients_per_round=clients_per_round,
        )
        sampling_traces.append(trace)

    sampling_trace_path = rp("sampling_trace")
    sampling_trace_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sampling_trace_path, "w", encoding="utf-8") as f:
        for trace in sampling_traces:
            f.write(trace.to_json() + "\n")
    logger.info(f"Sampling trace written: {num_rounds} rounds.")

    # Union of all selected clients
    union_selected = sorted({cid for ids in selected_ids_by_round.values() for cid in ids})
    logger.info(f"Union of selected clients across all rounds: {len(union_selected)}")

    # === MATERIALIZE SMOKE SLICES ===
    max_train = config.get("max_train_examples_per_client", 32)
    max_val = config.get("validation_example_limit", 256)

    logger.info("Materializing smoke slices...")
    _train_slice_sha, _val_slice_sha = prepare_t1_smoke_slices(
        rp("t1_train"),
        rp("t1_validation"),
        union_selected,
        category_count=category_count,
        valid_codes=valid_codes,
        max_train_examples_per_client=max_train,
        max_validation_examples=max_val,
        train_output_path=rp("train_smoke_slice"),
        validation_output_path=rp("validation_smoke_slice"),
    )

    train_slice_df = pl.read_parquet(rp("train_smoke_slice"))
    val_slice_df = pl.read_parquet(rp("validation_smoke_slice"))
    _TRAIN_SLICE_PATH = str(rp("train_smoke_slice"))

    logger.info(f"Smoke slices: train={len(train_slice_df)} rows, val={len(val_slice_df)} rows")

    # === MODEL INITIALIZATION ===
    logger.info("Initializing model...")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = build_smoke_model(category_count, spec)

    initial_state = pack_shared_state(model, model.shared_state_spec())
    initial_digest = get_digest(initial_state)
    logger.info(f"Initial state digest: {initial_digest[:16]}...")

    # Initial server-side evaluation
    val_init_metrics = server_evaluate(model, val_slice_df, spec)
    logger.info(
        f"Initial eval: CE={val_init_metrics['cross_entropy']:.4f}, "
        f"acc@1={val_init_metrics['accuracy_at_1']:.4f}, "
        f"support={val_init_metrics['support']}"
    )

    # Write initialization evidence
    init_evidence = {
        "schema": "fl_real_smoke_initialization_v1",
        "version": "1",
        "artifact_kind": "FIXTURE_PROOF",
        "scientific": False,
        "seed": seed,
        "model_id": _MODEL_ID,
        "category_count": category_count,
        "embedding_dim": 4,
        "hidden_dim": 8,
        "history_mode": "ZERO_HISTORY_TASKEXAMPLE_SMOKE",
        "initial_state_digest": initial_digest,
        "initial_validation_metrics": val_init_metrics,
    }
    init_evidence_path = rp("initialization_evidence")
    init_evidence_path.parent.mkdir(parents=True, exist_ok=True)
    init_evidence_path.write_text(json.dumps(init_evidence, indent=2), encoding="utf-8")
    init_evidence_sha = file_sha256(init_evidence_path)

    # Write input evidence
    input_evidence = {
        "schema": "fl_real_smoke_input_manifest_v1",
        "version": "1",
        "cohort_manifest": {
            "path": str(rp("cohort_manifest").relative_to(repo_root)).replace("\\", "/"),
            "sha256": cohort_sha,
            "sha256_prefix": cohort_sha_prefix,
        },
        "cohort_manifest_sha256": cohort_sha,
        "cohort_manifest_sha256_prefix": cohort_sha_prefix,
        "t1_train": {
            "path": str(rp("t1_train").relative_to(repo_root)).replace("\\", "/"),
            "sha256": train_sha,
            "rows": 3_113_814,
        },
        "t1_train_sha256": train_sha,
        "t1_train_sha256_prefix": train_sha[:16],
        "t1_train_row_count": 3_113_814,
        "t1_validation": {
            "path": str(rp("t1_validation").relative_to(repo_root)).replace("\\", "/"),
            "sha256": val_sha,
            "rows": 438_185,
        },
        "t1_validation_sha256": val_sha,
        "t1_validation_sha256_prefix": val_sha[:16],
        "t1_validation_row_count": 438_185,
        "vocabulary": {
            "path": str(rp("vocabulary").relative_to(repo_root)).replace("\\", "/"),
            "sha256": vocab_sha,
            "category_count": _EXPECTED_CATEGORY_COUNT,
        },
        "vocabulary_sha256": vocab_sha,
        "vocabulary_category_count": _EXPECTED_CATEGORY_COUNT,
        "base_client_manifest": {
            "path": str(rp("base_client_manifest").relative_to(repo_root)).replace("\\", "/"),
            "logical_hash": actual_hash,
            "client_count": actual_count,
        },
        "base_client_manifest_logical_hash": actual_hash,
        "base_client_manifest_count": actual_count,
        "t1_unique_train_clients": t1_unique_train_clients,
        "t1_unique_validation_clients": t1_unique_val_clients,
        "t1_eligible_count": t1_eligible_count,
        "t1_smoke_eligible_client_count": t1_eligible_count,
        "derived_client_manifest_sha256": derived_manifest_sha,
        "raw_events_used": False,
        "sealed_test_accessed": False,
    }
    input_evidence_path = rp("input_evidence")
    input_evidence_path.parent.mkdir(parents=True, exist_ok=True)
    input_evidence_path.write_text(json.dumps(input_evidence, indent=2), encoding="utf-8")
    input_evidence_sha = file_sha256(input_evidence_path)
    logger.info("Input evidence written.")

    # === EXPERIMENT CONFIG ===
    uv_lock_sha = file_sha256(repo_root / "uv.lock")
    config_sha = file_sha256(args.config)
    shared_trainer_path = (
        repo_root / "docs" / "evidence" / "s1-pr-05" / "shared_trainer_core_manifest.v1.json"
    )
    shared_trainer_sha = (
        file_sha256(shared_trainer_path) if shared_trainer_path.is_file() else "0" * 64
    )
    data_protocol_path = (
        repo_root / "docs" / "evidence" / "s1-ds-05-06" / "data_protocol_v1.proposed.json"
    )
    data_protocol_sha = (
        file_sha256(data_protocol_path) if data_protocol_path.is_file() else "0" * 64
    )
    objective_sha = file_sha256(repo_root / "ppsi" / "training" / "objective.py")
    evaluator_sha = file_sha256(Path(__file__))
    adr_path = repo_root / "docs" / "decisions" / "ADR-001-evaluation-protocol.md"
    adr_sha = file_sha256(adr_path) if adr_path.is_file() else "0" * 64

    def _relpath(p: Path) -> str:
        return str(p.relative_to(repo_root)).replace("\\", "/")

    exp_config: dict[str, Any] = {
        "schema": "experiment_config_v1",
        "version": "1",
        "config_id": "fl_real_smoke_r2a_t1_v1_non_scientific",
        "regime": "R2A",
        "tasks": ["T1"],
        "training_cohort": "C1",
        "evaluation_cohorts": ["C1"],
        "seed": seed,
        "source_dataset_ref": make_artifact_ref(
            "fl_real_smoke_input_manifest_v1",
            "fl_real_smoke_input_manifest_v1",
            paths["input_evidence"],
            input_evidence_sha,
        ),
        "canonical_data_contract_ref": make_artifact_ref(
            "data_protocol_v1",
            "data_protocol_v1",
            _relpath(data_protocol_path),
            data_protocol_sha,
        ),
        "cohort_manifest_ref": make_artifact_ref(
            "cohort_manifest_v1",
            "cohort_manifest_v1",
            paths["cohort_manifest"],
            cohort_sha,
        ),
        "split_manifest_ref": make_artifact_ref(
            "data_protocol_v1_split",
            "data_protocol_v1",
            _relpath(data_protocol_path),
            data_protocol_sha,
        ),
        "task_examples_manifest_ref": make_artifact_ref(
            "fl_real_smoke_input_manifest_v1",
            "fl_real_smoke_input_manifest_v1",
            paths["input_evidence"],
            input_evidence_sha,
        ),
        "evaluation_manifest_ref": make_artifact_ref(
            "t1_validation_parquet",
            "task_examples_v1",
            paths["t1_validation"],
            val_sha,
        ),
        "representation_ref": make_artifact_ref(
            "vocabulary_v1",
            "vocabulary_v1",
            paths["vocabulary"],
            vocab_sha,
        ),
        "model_config_ref": make_artifact_ref(
            "fl_real_smoke_config_v1",
            "fl_real_smoke_v1",
            "config/fl_real_smoke.v1.json",
            config_sha,
        ),
        "objective_config_ref": make_artifact_ref(
            "contract_smoke_objective_v1",
            "python_module_v1",
            "ppsi/training/objective.py",
            objective_sha,
        ),
        "shared_trainer_core_ref": make_artifact_ref(
            "shared_trainer_core_manifest_v1",
            "shared_trainer_core_manifest_v1",
            _relpath(shared_trainer_path),
            shared_trainer_sha,
        ),
        "evaluation_protocol_ref": make_artifact_ref(
            "adr_001_evaluation_protocol",
            "adr_v1",
            _relpath(adr_path)
            if adr_path.is_file()
            else "docs/decisions/ADR-001-evaluation-protocol.md",
            adr_sha,
        ),
        "evaluator_ref": make_artifact_ref(
            "fl_real_smoke_script_v1",
            "python_module_v1",
            "scripts/federated/fl_real_smoke.py",
            evaluator_sha,
        ),
        "environment_lock_ref": make_artifact_ref(
            "uv_lock_v1",
            "uv_lock_v1",
            "uv.lock",
            uv_lock_sha,
        ),
        "initialization": {
            "kind": "COMMON_INITIALIZATION",
            "common_initialization_ref": make_artifact_ref(
                "fl_real_smoke_initialization_v1",
                "fl_real_smoke_initialization_v1",
                paths["initialization_evidence"],
                init_evidence_sha,
            ),
        },
        "regime_config": {
            "orchestration_type": "FEDAVG",
            "scientific": False,
            "purpose": "NON_SCIENTIFIC_REAL_DATA_PIPELINE_SMOKE",
            "num_rounds": num_rounds,
            "clients_per_round": clients_per_round,
            "local_epochs": config.get("local_epochs", 1),
            "learning_rate": config.get("learning_rate", 0.02),
            "momentum": config.get("momentum", 0.0),
            "optimizer": config.get("optimizer", "SGD"),
            "aggregation_weight_policy": "contributing_rows_smoke_weight_v1_NON_SCIENTIFIC",
        },
    }

    validate_experiment_config(exp_config)
    exp_config_path = rp("experiment_config_evidence")
    exp_config_path.parent.mkdir(parents=True, exist_ok=True)
    exp_config_path.write_text(json.dumps(exp_config, indent=2), encoding="utf-8")
    exp_config_sha = file_sha256(exp_config_path)
    logger.info("ExperimentConfig written and validated.")

    # === REPEAT SIMULATIONS ===
    started_at_utc = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    repeat_runs = config.get("repeat_runs", 2)
    all_repetitions = []

    for rep in range(repeat_runs):
        logger.info(f"Starting repetition {rep + 1}/{repeat_runs}...")
        # Re-init model to same state for reproducibility
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            _model = build_smoke_model(category_count, spec)
        rep_initial_state = dict(pack_shared_state(_model, _model.shared_state_spec()))

        smoke_result, tracing_log = run_smoke_simulation(
            config,
            selected_ids_by_round=selected_ids_by_round,
            selection_digests_by_round=selection_digests_by_round,
            val_df=val_slice_df,
            spec=spec,
            initial_state=rep_initial_state,
            category_count=category_count,
        )

        # Extract server-side eval metrics
        validation_history: list[dict[str, float]] = []
        if (
            smoke_result is not None
            and hasattr(smoke_result, "evaluate_metrics_serverapp")
            and smoke_result.evaluate_metrics_serverapp
        ):
            for round_num in sorted(smoke_result.evaluate_metrics_serverapp.keys()):
                m = smoke_result.evaluate_metrics_serverapp[round_num]
                validation_history.append(
                    {
                        "server_round": round_num,
                        "cross_entropy": float(m.get("cross_entropy", float("nan"))),
                        "accuracy_at_1": float(m.get("accuracy_at_1", float("nan"))),
                        "support": int(m.get("support", 0)),
                    }
                )

        final_digest = tracing_log[-1]["aggregated_digest"] if tracing_log else ""
        all_repetitions.append(
            {
                "rep": rep,
                "tracing_log": tracing_log,
                "validation_history": validation_history,
                "final_digest": final_digest,
            }
        )

    ended_at_utc = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    # === ACCEPTANCE CHECKS ===
    logger.info("Running acceptance checks...")
    rep0 = all_repetitions[0]
    tracing_log_0 = rep0["tracing_log"]

    if len(tracing_log_0) != num_rounds:
        raise SmokeValidationError(f"Expected {num_rounds} rounds, got {len(tracing_log_0)}")

    for entry in tracing_log_0:
        sr = entry["server_round"]
        if entry["selected_client_count"] != clients_per_round:
            raise SmokeValidationError(
                f"Round {sr}: {entry['selected_client_count']} selected != {clients_per_round}"
            )
        if len(entry["clients"]) != clients_per_round:
            raise SmokeValidationError(
                f"Round {sr}: {len(entry['clients'])} successful != {clients_per_round}"
            )
        if entry["contributing_examples"] <= 0:
            raise SmokeValidationError(f"Round {sr}: contributing_examples must be > 0")
        if not entry["aggregation_oracle_pass"]:
            raise SmokeValidationError(f"Round {sr}: aggregation oracle FAILED")
        mad = entry.get("max_abs_diff", 0.0)
        if mad is not None and mad > config.get("aggregation_atol", 1e-6):
            raise SmokeValidationError(f"Round {sr}: max_abs_diff={mad}")
        for c in entry["clients"]:
            if not torch.isfinite(torch.tensor(c["local_train_loss"])):
                raise SmokeValidationError(f"Round {sr}: non-finite local_train_loss")

    final_digest_0 = rep0["final_digest"]
    if final_digest_0 == initial_digest:
        raise SmokeValidationError("Model did not update (initial == final digest)")

    # Validate validation history: for num_rounds=3, require exactly 4 records (0, 1, 2, 3)
    val_history_0 = rep0["validation_history"]
    expected_val_records = num_rounds + 1
    if len(val_history_0) != expected_val_records:
        raise SmokeValidationError(
            f"Expected exactly {expected_val_records} validation records, got {len(val_history_0)}"
        )
    expected_support = config.get("validation_example_limit", 256)
    for expected_round, val_record in enumerate(val_history_0):
        actual_round = val_record.get("server_round")
        if actual_round != expected_round:
            raise SmokeValidationError(
                f"Validation record {expected_round} has server_round {actual_round} != {expected_round}"
            )
        support = val_record.get("support", 0)
        if support != expected_support:
            raise SmokeValidationError(
                f"Validation record round {expected_round} has support {support} != {expected_support}"
            )
        ce = val_record.get("cross_entropy", float("nan"))
        if not math.isfinite(ce):
            raise SmokeValidationError(f"Non-finite cross_entropy in round {expected_round}: {ce}")
        acc = val_record.get("accuracy_at_1", float("nan"))
        if not (math.isfinite(acc) and 0.0 <= acc <= 1.0):
            raise SmokeValidationError(
                f"accuracy_at_1 out of [0, 1] in round {expected_round}: {acc}"
            )

    logger.info("Acceptance: PASS")

    # === REPRODUCIBILITY ===
    reproducibility_pass = True
    rep_details: dict[str, Any] = {"repeat_runs": repeat_runs}
    if len(all_repetitions) > 1:
        rep1 = all_repetitions[1]
        tl0 = tracing_log_0
        tl1 = rep1["tracing_log"]
        # Selection digests identical
        sel_match = all(
            tl0[i]["selection_digest"] == tl1[i]["selection_digest"] for i in range(num_rounds)
        )
        # Final digest identical
        fd_match = rep0["final_digest"] == rep1["final_digest"]
        # Validation metrics within tolerance
        tol = config.get("aggregation_atol", 1e-6)
        val_match = True
        hist0 = rep0["validation_history"]
        hist1 = rep1["validation_history"]
        if len(hist0) != len(hist1):
            raise SmokeValidationError(
                f"Repeat validation history length mismatch: {len(hist0)} != {len(hist1)}"
            )
        for j, (h0, h1) in enumerate(zip(hist0, hist1, strict=True)):
            for k in ("cross_entropy", "accuracy_at_1"):
                diff = abs(h0.get(k, 0.0) - h1.get(k, 0.0))
                if diff > tol:
                    val_match = False
                    logger.warning(f"Validation {k} differs at step {j}: {diff}")
        reproducibility_pass = sel_match and fd_match and val_match
        rep_details = {
            "repeat_runs": repeat_runs,
            "pass": reproducibility_pass,
            "reproducibility_pass": reproducibility_pass,
            "selection_digests_match": sel_match,
            "final_digest_match": fd_match,
            "validation_metrics_match": val_match,
        }
    else:
        rep_details = {"repeat_runs": 1, "pass": True, "reproducibility_pass": True}

    if not reproducibility_pass:
        raise SmokeValidationError("Reproducibility check FAILED.")
    logger.info("Reproducibility: PASS")

    # === PROVENANCE ===
    git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True
    ).strip()
    try:
        diff_bytes = subprocess.check_output(["git", "diff", "--binary"], cwd=str(repo_root))
        diff_sha = hashlib.sha256(diff_bytes).hexdigest()
    except (subprocess.SubprocessError, OSError):
        diff_sha = "unavailable"

    # Explicit source-file SHA-256 mapping for implementation files used by this run
    execution_source_files = [
        "config/fl_real_smoke.v1.json",
        "ppsi/federated/task_examples.py",
        "scripts/federated/fl_real_smoke.py",
        "ppsi/federated/clients.py",
        "ppsi/federated/sampling.py",
        "ppsi/training/batch.py",
        "ppsi/training/core.py",
        "ppsi/training/flower.py",
        "ppsi/training/objective.py",
        "ppsi/training/stub_model.py",
    ]
    execution_source_files_sha256 = {}
    for rel_path in execution_source_files:
        src_path = repo_root / rel_path
        if src_path.is_file():
            execution_source_files_sha256[rel_path] = file_sha256(src_path)

    # === EXPERIMENT RESULT ===
    last_val = rep0["validation_history"][-1] if rep0["validation_history"] else {}

    config_ref = make_artifact_ref(
        "FLRealSmokeResolvedExperimentConfig",
        "experiment_config_v1",
        str(rp("experiment_config_evidence").relative_to(repo_root)).replace("\\", "/"),
        exp_config_sha,
    )

    metric_ce = make_metric_record(
        metric_id="smoke_t1_validation_cross_entropy_non_scientific",
        task="T1",
        cohort="C1",
        value=float(last_val.get("cross_entropy", 0.0)),
        direction="MINIMIZE",
        unit="UNITLESS",
        support=int(last_val.get("support", 0)),
    )
    metric_acc = make_metric_record(
        metric_id="smoke_t1_validation_accuracy_at_1_non_scientific",
        task="T1",
        cohort="C1",
        value=float(last_val.get("accuracy_at_1", 0.0)),
        direction="NEUTRAL",
        unit="FRACTION",
        support=int(last_val.get("support", 0)),
    )

    public_tracing = sanitize_trace_for_public(tracing_log_0)
    federated_metadata = {
        "purpose": "NON_SCIENTIFIC_REAL_DATA_PIPELINE_SMOKE",
        "scientific": False,
        "scientific_claims": [],
        "num_rounds": num_rounds,
        "clients_per_round": clients_per_round,
        "local_epochs": config.get("local_epochs", 1),
        "max_examples_per_client": max_train,
        "weight_policy_id": "contributing_rows_smoke_weight_v1_NON_SCIENTIFIC",
        "round_selection_digests": [selection_digests_by_round[i] for i in range(num_rounds)],
        "successful_client_counts": [len(e["clients"]) for e in tracing_log_0],
        "contributing_examples_per_round": [e["contributing_examples"] for e in tracing_log_0],
        "oracle_pass_all_rounds": all(e["aggregation_oracle_pass"] for e in tracing_log_0),
        "max_abs_diff_all_rounds": [e.get("max_abs_diff") for e in tracing_log_0],
        "initial_digest": initial_digest,
        "final_digest": final_digest_0,
        "global_model_changed": initial_digest != final_digest_0,
        "repeat_runs": repeat_runs,
        "reproducibility": rep_details,
        "raw_events_used": False,
        "sealed_test_accessed": False,
    }

    exp_result = build_experiment_result(
        experiment_config=exp_config,
        config_ref=config_ref,
        git_sha=git_sha,
        state="SUCCEEDED",
        attempt=1,
        started_at_utc=started_at_utc,
        ended_at_utc=ended_at_utc,
        metrics=[metric_ce, metric_acc],
        artifacts=[
            make_artifact_ref(
                "derived_client_manifest",
                "fl_real_smoke_client_manifest_v1",
                paths["derived_client_manifest"],
                derived_manifest_sha,
            ),
            make_artifact_ref(
                "input_evidence",
                "fl_real_smoke_input_manifest_v1",
                paths["input_evidence"],
                input_evidence_sha,
            ),
            make_artifact_ref(
                "initialization_evidence",
                "fl_real_smoke_initialization_v1",
                paths["initialization_evidence"],
                init_evidence_sha,
            ),
        ],
        system_measurements={
            "schema": "system_measurement_reference_set_v1",
            "version": "1",
            "status": "NOT_APPLICABLE",
            "null_reason": "S1-PR-07 makes no system or communication claim",
        },
        federated_metadata=federated_metadata,
    )

    results_dir = rp("results_directory")
    results_dir.mkdir(parents=True, exist_ok=True)
    run_id = exp_result["run_id"]
    result_path = results_dir / f"{run_id}.result.json"
    result_path.write_text(json.dumps(exp_result, indent=2), encoding="utf-8")
    result_sha = file_sha256(result_path)

    record = json.loads(result_path.read_text(encoding="utf-8"))
    validate_experiment_result(record)
    validate_result_for_reporting(record, source=result_path.name)
    logger.info(f"ExperimentResult written and validated: {result_path.name}")

    # === PUBLIC SUMMARY ===
    summary = {
        "schema": "fl_real_smoke_summary_v1",
        "version": "1",
        "status": "PASS",
        "purpose": "NON_SCIENTIFIC_REAL_DATA_PIPELINE_SMOKE",
        "scientific": False,
        "scientific_claims": [],
        "execution_provenance": "WORKING_TREE_MVP_SMOKE",
        "input_identities": {
            "t1_train_sha256": train_sha,
            "t1_validation_sha256": val_sha,
            "vocabulary_sha256": vocab_sha,
            "base_client_manifest_logical_hash": actual_hash,
        },
        "derived_manifest_sha256": derived_manifest_sha,
        "initialization_digest": initial_digest,
        "config_sha256": config_sha,
        "result_sha256": result_sha,
        "runtime_versions": {
            "python": sys.version.split()[0],
            "flower": flwr.__version__,
            "ray": ray.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "validation_history": rep0["validation_history"],
        "rounds": public_tracing,
        "reproducibility": rep_details,
        "global_model_changed": initial_digest != final_digest_0,
        "aggregation_oracle_pass": all(e["aggregation_oracle_pass"] for e in tracing_log_0),
        "integration_findings": {
            "global_model_changed": initial_digest != final_digest_0,
            "oracle_pass_all_rounds": all(e["aggregation_oracle_pass"] for e in tracing_log_0),
        },
        "limitations": [
            "Zero-history representation — no sequential event history",
            "Non-scientific stub model — not the final project GRU",
            "No R1 baseline — QR not calculated",
            "No T2/T3 tasks, no TEST set, no communication measurement",
        ],
        "source_git_sha": git_sha,
        "git_sha": git_sha,
        "tracked_diff_sha256": diff_sha,
        "execution_source_files_sha256": execution_source_files_sha256,
        "raw_events_used": False,
        "sealed_test_accessed": False,
    }

    summary_path = rp("summary")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"Summary written: {summary_path}")

    # === PRIVACY SCAN ===
    public_evidence_files = list(rp("summary").parent.glob("*.json"))
    public_paths_to_check = public_evidence_files + [result_path]
    notebook_path = repo_root / "notebooks" / "S1_PR_07_Real_Flower_Smoke.ipynb"
    if notebook_path.is_file():
        public_paths_to_check.append(notebook_path)
    assert_no_client_leakage(public_paths_to_check)
    logger.info("Public privacy check: PASS (no client-v1 IDs found in public files)")

    logger.info("=" * 60)
    logger.info("S1-PR-07 SMOKE TEST: PASS")
    logger.info(f"  Rounds: {num_rounds}, clients/round: {clients_per_round}")
    logger.info(f"  Final CE: {last_val.get('cross_entropy', 'N/A')}")
    logger.info(f"  Final acc@1: {last_val.get('accuracy_at_1', 'N/A')}")
    logger.info(f"  Model changed: {initial_digest != final_digest_0}")
    logger.info(f"  Reproducibility: {reproducibility_pass}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
