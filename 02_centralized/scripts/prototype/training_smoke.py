"""S1-PR-05 unified trainer smoke using the proven Flower 1.33 lifecycle.

This script uses deterministic synthetic Phase1Batch fixtures. It does not
train on REES46 and does not define final scientific hyperparameters or loss.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import re
import subprocess
import sys
from pathlib import Path

import ppsi_paths

# Kept from the flat layout; the installed package map makes ppsi and scripts importable.
_REPO_ROOT = ppsi_paths.REPO_ROOT
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import flwr
import ray
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.serverapp import Grid, ServerApp
from flwr.simulation import run_simulation

from ppsi.training.core import LocalTrainerCore
from ppsi.training.fixtures import (
    default_batch_spec,
    make_client_batches,
    make_stub_model,
)
from ppsi.training.flower import (
    ContributingRowsSmokeWeightPolicy,
    FlowerLocalAdapter,
)
from ppsi.training.identity import (
    DEFAULT_TRAINER_CORE_FILES,
    build_trainer_core_manifest,
    file_sha256,
)
from ppsi.training.initialization import (
    generate_stub_initialization_fixture,
    load_verified_stub_initialization,
)
from ppsi.training.objective import ContractSmokeObjective
from ppsi.training.state import pack_shared_state
from scripts.federated.fl_synthetic_smoke import (
    SmokeValidationError,
    TracingFedAvg,
    get_digest,
    weighted_average_state_dicts,
)

logger = logging.getLogger(__name__)
REPO_ROOT = ppsi_paths.REPO_ROOT

client_app = ClientApp()
SMOKE_RESULT = None
SMOKE_TRACING_LOG: list[dict] = []
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class UnifiedTrainerTracingFedAvg(TracingFedAvg):
    """Record numeric aggregation parity for the S1-PR-05 evidence."""

    def aggregate_train(self, server_round: int, replies):
        replies_list = list(replies)
        flower_arrays, metrics = super().aggregate_train(server_round, replies_list)
        if flower_arrays is None:
            return flower_arrays, metrics

        updates = [
            (
                message.content.parameters_records["arrays"].to_torch_state_dict(),
                int(message.content.metrics_records["metrics"]["num-examples"]),
            )
            for message in replies_list
        ]
        oracle_state = weighted_average_state_dicts(updates)
        flower_state = flower_arrays.to_torch_state_dict()
        self.tracing_log[-1]["max_abs_diff"] = max(
            float(torch.max(torch.abs(oracle_state[key] - flower_state[key])).item())
            for key in oracle_state
        )
        return flower_arrays, metrics


def _make_core(*, learning_rate: float) -> LocalTrainerCore:
    model = make_stub_model(seed=13)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    return LocalTrainerCore(
        model=model,
        batch_spec=default_batch_spec(),
        objective=ContractSmokeObjective(),
        optimizer=optimizer,
        device="cpu",
    )


@client_app.train()
def train(message: Message, context: Context) -> Message:
    torch.manual_seed(13)
    torch.set_num_threads(1)
    config = message.content.configs_records.get("config", ConfigRecord())
    incoming = message.content.parameters_records["arrays"].to_torch_state_dict()
    logical_client_id = int(config["logical_client_id"])
    learning_rate = float(config["learning_rate"])

    core = _make_core(learning_rate=learning_rate)
    adapter = FlowerLocalAdapter(core, core.shared_state_spec, ContributingRowsSmokeWeightPolicy())
    fit_result = adapter.fit(
        incoming,
        make_client_batches(logical_client_id),
        outer_round=int(config.get("server_round", 0)),
    )
    if fit_result.aggregation_weight <= 0:
        raise SmokeValidationError("Synthetic client produced zero aggregation weight")

    record = RecordDict()
    record.parameters_records["arrays"] = ArrayRecord.from_torch_state_dict(
        dict(fit_result.shared_state)
    )
    scalar_metrics = fit_result.scalar_metrics()
    record.metrics_records["metrics"] = MetricRecord(scalar_metrics)
    # TracingFedAvg owns the canonical lifecycle digest format from S1-PR-02.
    # The adapter's richer state digest remains an internal diagnostic only.
    record.configs_records["config"] = ConfigRecord(
        {
            "received_digest": get_digest(incoming),
            "updated_digest": get_digest(dict(fit_result.shared_state)),
            "logical_client_id": logical_client_id,
            "local_train_loss": float(scalar_metrics["train_loss"]),
        }
    )
    return message.create_reply(record)


@client_app.evaluate()
def evaluate(message: Message, context: Context) -> Message:
    torch.manual_seed(13)
    torch.set_num_threads(1)
    config = message.content.configs_records.get("config", ConfigRecord())
    incoming = message.content.parameters_records["arrays"].to_torch_state_dict()
    logical_client_id = int(config["logical_client_id"])
    core = _make_core(learning_rate=float(config.get("learning_rate", 0.01)))
    adapter = FlowerLocalAdapter(core, core.shared_state_spec, ContributingRowsSmokeWeightPolicy())
    summary = adapter.evaluate(
        incoming,
        make_client_batches(logical_client_id),
        outer_round=int(config.get("server_round", 0)),
    )
    loss = sum(stat.mean for stat in summary.task_stats.values() if stat.mean is not None)

    record = RecordDict()
    record.metrics_records["metrics"] = MetricRecord(
        {"eval_loss": loss, "num-examples": summary.contributing_examples}
    )
    return message.create_reply(record)


def evaluate_fn(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    state = arrays.to_torch_state_dict()
    core = _make_core(learning_rate=0.01)
    adapter = FlowerLocalAdapter(core, core.shared_state_spec, ContributingRowsSmokeWeightPolicy())
    summaries = [
        adapter.evaluate(state, make_client_batches(client_id), outer_round=server_round)
        for client_id in (0, 1)
    ]
    numerators: dict[str, float] = {}
    denominators: dict[str, int] = {}
    for summary in summaries:
        for task, stat in summary.task_stats.items():
            numerators[task] = numerators.get(task, 0.0) + stat.numerator
            denominators[task] = denominators.get(task, 0) + stat.denominator
    loss = sum(
        numerators[task] / denominators[task] for task in numerators if denominators[task] > 0
    )
    return MetricRecord({"global_loss": loss})


def get_server_app(config: dict) -> ServerApp:
    app = ServerApp()

    @app.main()
    def main(grid: Grid, context: Context) -> None:
        global SMOKE_RESULT, SMOKE_TRACING_LOG
        torch.manual_seed(13)
        torch.set_num_threads(1)
        model = make_stub_model(seed=42)
        initialization = generate_stub_initialization_fixture(
            batch_spec=default_batch_spec(),
            model_config=model.config,
            seed=config["seed"],
        )
        load_verified_stub_initialization(
            model,
            initialization,
            expected_seed=config["seed"],
            expected_state_sha256=initialization.state_sha256,
            expected_model_config_sha256=initialization.model_config_sha256,
        )
        initial_state = pack_shared_state(model, model.shared_state_spec())
        initial_arrays = ArrayRecord.from_torch_state_dict(initial_state)

        train_config = ConfigRecord(
            {
                "learning_rate": config["learning_rate"],
            }
        )
        strategy = UnifiedTrainerTracingFedAvg(
            expected_clients=config["num_clients"],
            tolerance=config["tolerance"],
            fraction_train=1.0,
            fraction_evaluate=1.0,
            min_train_nodes=config["num_clients"],
            min_evaluate_nodes=config["num_clients"],
            min_available_nodes=config["num_clients"],
            weighted_by_key="num-examples",
        )
        SMOKE_RESULT = strategy.start(
            grid=grid,
            initial_arrays=initial_arrays,
            num_rounds=config["num_rounds"],
            train_config=train_config,
            evaluate_fn=evaluate_fn,
        )
        SMOKE_TRACING_LOG = list(strategy.tracing_log)

    return app


def run_unified_trainer_smoke(config: dict):
    global SMOKE_RESULT, SMOKE_TRACING_LOG
    SMOKE_RESULT = None
    SMOKE_TRACING_LOG = []
    run_simulation(
        server_app=get_server_app(config),
        client_app=client_app,
        num_supernodes=config["num_clients"],
        backend_name="ray",
        backend_config={"client_resources": {"num_cpus": 1}},
    )
    return SMOKE_RESULT, list(SMOKE_TRACING_LOG)


def validate_config(config: dict) -> None:
    if config.get("schema") != "unified_trainer_smoke_v1":
        raise SmokeValidationError("Invalid unified trainer smoke schema")
    if config.get("version") != "1":
        raise SmokeValidationError("Invalid unified trainer smoke version")
    if config.get("seed") != 13:
        raise SmokeValidationError("Smoke seed must be 13")
    if config.get("num_clients") != 2:
        raise SmokeValidationError("unified_trainer_smoke_v1 requires num_clients == 2")
    if config.get("num_rounds", 0) < 3:
        raise SmokeValidationError("num_rounds must be >= 3")
    if config.get("repeat_runs", 0) < 2:
        raise SmokeValidationError("repeat_runs must be >= 2")
    if float(config.get("learning_rate", 0)) <= 0:
        raise SmokeValidationError("learning_rate must be positive")
    if float(config.get("tolerance", 0)) != 1e-6:
        raise SmokeValidationError("unified_trainer_smoke_v1 requires tolerance == 1e-6")


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def _loss_history(result) -> list[tuple[int, float]]:
    if result is None:
        raise SmokeValidationError("Flower returned no result")
    history = []
    for round_id in sorted(result.evaluate_metrics_serverapp):
        metrics = result.evaluate_metrics_serverapp[round_id]
        if "global_loss" in metrics:
            history.append((round_id, float(metrics["global_loss"])))
    if len(history) < 4:  # initial evaluation + three rounds
        raise SmokeValidationError("Missing expected global-loss history")
    if not all(torch.isfinite(torch.tensor(value)) for _, value in history):
        raise SmokeValidationError("Non-finite global loss")
    return history


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        encoding="utf-8",
    ).strip()


def validate_source_git_sha(source_git_sha: str, *, require_clean_head: bool) -> str:
    if not _GIT_SHA_RE.fullmatch(source_git_sha):
        raise SmokeValidationError("source_git_sha must be 40 lowercase hex characters")
    exists = subprocess.run(
        ["git", "cat-file", "-e", f"{source_git_sha}^{{commit}}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if exists.returncode != 0:
        raise SmokeValidationError("source_git_sha does not identify an available commit")
    if require_clean_head:
        if source_git_sha != _git_sha():
            raise SmokeValidationError("source_git_sha must equal the current HEAD")
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=REPO_ROOT,
            text=True,
            encoding="utf-8",
        ).strip()
        if status:
            raise SmokeValidationError("Canonical evidence requires a clean working tree")
    return source_git_sha


def execute(config: dict, *, source_git_sha: str | None = None) -> dict:
    validate_config(config)
    source_git_sha = validate_source_git_sha(
        source_git_sha or _git_sha(),
        require_clean_head=False,
    )
    repetitions = []
    for run_index in range(config["repeat_runs"]):
        result, trace = run_unified_trainer_smoke(config)
        for record in trace:
            record["clients"].sort(key=lambda client: client["logical_client_id"])
        if len(trace) != config["num_rounds"]:
            raise SmokeValidationError("Not all Flower rounds completed")
        if not all(entry["aggregation_oracle_pass"] for entry in trace):
            raise SmokeValidationError("FedAvg oracle did not pass every round")
        initial_digest = trace[0]["server_input_digest"]
        final_digest = trace[-1]["aggregated_digest"]
        if final_digest == initial_digest:
            raise SmokeValidationError("Global model did not change")
        repetitions.append(
            {
                "run": run_index,
                "global_loss_history": _loss_history(result),
                "final_model_digest": final_digest,
                "selected_client_ids": [entry["selected_logical_ids"] for entry in trace],
                "tracing_log": trace,
            }
        )

    reference = repetitions[0]
    for repetition in repetitions[1:]:
        if repetition["final_model_digest"] != reference["final_model_digest"]:
            raise SmokeValidationError("Final model digest differs across repetitions")
        if repetition["selected_client_ids"] != reference["selected_client_ids"]:
            raise SmokeValidationError("Client order differs across repetitions")

    initialization = generate_stub_initialization_fixture(
        batch_spec=default_batch_spec(),
        model_config=make_stub_model(seed=42).config,
        seed=config["seed"],
    )
    # identity.py joins the prototype-layout names onto a root and hashes itself, so it runs unchanged
    # on a view laid out the old way; the manifest equals the one over the real files.
    with ppsi_paths.prototype_root(DEFAULT_TRAINER_CORE_FILES) as trainer_root:
        trainer_manifest = build_trainer_core_manifest(trainer_root)
    identities = {
        "config_sha256": _canonical_sha256(config),
        "input_fixture_sha256": _canonical_sha256(
            {
                "schema": "deterministic_phase1_fixture_v1",
                "version": "1",
                "seed": config["seed"],
                "client_batch_sizes": {"0": 2, "1": 4},
            }
        ),
        "objective_config_sha256": _canonical_sha256(
            {
                "schema": "contract_smoke_objective_config_v1",
                "version": "1",
                "objective_id": ContractSmokeObjective.objective_id,
                "scientific": False,
            }
        ),
        "shared_trainer_core_sha256": trainer_manifest["sha256"],
        "source_git_sha": source_git_sha,
    }
    max_abs_diff = max(
        float(record["max_abs_diff"])
        for repetition in repetitions
        for record in repetition["tracing_log"]
    )
    return {
        "schema": "unified_trainer_smoke_summary_v1",
        "version": "1",
        "status": "PASS",
        "seed": config["seed"],
        "identities": identities,
        "aggregation_parity": {
            "atol": 1e-6,
            "rtol": 0.0,
            "max_abs_diff": max_abs_diff,
        },
        "initialization": {
            "artifact_kind": initialization.artifact_kind,
            "state_sha256": initialization.state_sha256,
            "model_config_sha256": initialization.model_config_sha256,
            "state_size_bytes": initialization.state_size_bytes,
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "flower": flwr.__version__,
            "ray": ray.__version__,
            "platform": platform.platform(),
            "uv_lock_sha256": file_sha256(REPO_ROOT / "uv.lock"),
        },
        "repetitions": repetitions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-git-sha", required=True)
    args = parser.parse_args()
    try:
        source_git_sha = validate_source_git_sha(
            args.source_git_sha,
            require_clean_head=True,
        )
        summary = execute(load_config(args.config), source_git_sha=source_git_sha)
    except Exception as exc:
        logger.exception("Unified trainer smoke failed")
        summary = {
            "schema": "unified_trainer_smoke_summary_v1",
            "version": "1",
            "status": "FAIL",
            "error": str(exc),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                summary,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        raise SystemExit(2) from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    logger.info("Summary written to %s", args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
