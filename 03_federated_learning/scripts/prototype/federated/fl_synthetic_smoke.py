import argparse
import copy
import hashlib
import json
import logging
import platform
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import flwr
import ray
import torch
import torch.nn.functional as F
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from flwr.simulation import run_simulation
from torch import Tensor, nn, optim

from ppsi.training.identity import file_sha256

logger = logging.getLogger(__name__)


class SmokeValidationError(Exception):
    """Raised when the smoke test validation fails or invariants are violated."""


# --- Pure Oracle / Validation Logic ---


def get_digest(state_dict: dict[str, Tensor]) -> str:
    """Computes a SHA-256 digest of the state dict for redistribution tracing."""
    m = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        m.update(key.encode("utf-8"))
        # Ensure contiguous array for consistent hashing
        m.update(state_dict[key].cpu().contiguous().numpy().tobytes())
    return m.hexdigest()


def validate_state_dict(state_dict: dict[str, Tensor]) -> None:
    """Validates that a state dict has finite values."""
    for key, tensor in state_dict.items():
        if not torch.isfinite(tensor).all():
            raise SmokeValidationError(f"Tensor {key} contains non-finite values (NaN/Inf)")


def weighted_average_state_dicts(updates: list[tuple[dict[str, Tensor], int]]) -> dict[str, Tensor]:
    """Pure oracle for FedAvg aggregation independent of Flower internals."""
    if not updates:
        raise SmokeValidationError("No client updates provided for aggregation")

    total_examples = 0
    accumulated: dict[str, Tensor] = {}
    reference_state = updates[0][0]
    reference_keys = sorted(reference_state.keys())

    for state_dict, num_examples in updates:
        if num_examples <= 0:
            raise SmokeValidationError(f"num_examples must be positive, got {num_examples}")

        validate_state_dict(state_dict)

        current_keys = sorted(state_dict.keys())
        if current_keys != reference_keys:
            raise SmokeValidationError(
                f"Keys do not match: expected {reference_keys}, got {current_keys}"
            )

        for key in reference_keys:
            tensor = state_dict[key]
            if key not in accumulated:
                accumulated[key] = tensor.to(torch.float64) * num_examples
            else:
                if tensor.shape != reference_state[key].shape:
                    raise SmokeValidationError(f"Shape mismatch for {key}")
                accumulated[key] += tensor.to(torch.float64) * num_examples

        total_examples += num_examples

    result: dict[str, Tensor] = {}
    for key in reference_keys:
        averaged = accumulated[key] / total_examples
        result[key] = averaged.to(reference_state[key].dtype)

    return result


# --- Model ---


class TinyLinearModel(nn.Module):
    """A deterministic one-parameter model for smoke testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


# --- Data & Local Training ---


def get_deterministic_data(logical_client_id: int) -> tuple[Tensor, Tensor]:
    """Returns exact deterministic data based on client ID."""
    if logical_client_id < 0:
        raise ValueError(f"Invalid logical_client_id: {logical_client_id}. Must be >= 0.")
    torch.manual_seed(13 + logical_client_id)
    num_examples = 1 if logical_client_id == 0 else 3
    x = torch.randn(num_examples, 1)
    y = 2.0 * x
    return x, y


def local_train(model: nn.Module, x: Tensor, y: Tensor, lr: float, epochs: int) -> float:
    """Performs local SGD."""
    if x.numel() == 0:
        raise SmokeValidationError("Client dataset is empty")
    optimizer = optim.SGD(model.parameters(), lr=lr)
    model.train()
    loss = None
    for _ in range(epochs):
        optimizer.zero_grad()
        loss_t = F.mse_loss(model(x), y)
        loss_t.backward()
        optimizer.step()
        loss = loss_t.item()
    return loss if loss is not None else 0.0


def local_evaluate(model: nn.Module, x: Tensor, y: Tensor) -> float:
    model.eval()
    with torch.no_grad():
        loss = F.mse_loss(model(x), y)
    return loss.item()


# --- Flower ClientApp ---

client_app = ClientApp()


@client_app.train()
def train(message: Message, context: Context) -> Message:
    torch.manual_seed(13)
    torch.set_num_threads(1)
    config = message.content.configs_records.get("config", ConfigRecord())

    if "inject_f02_exception" in config:
        raise SmokeValidationError("injected client failure")

    array_record = message.content.parameters_records["arrays"]
    state_dict = array_record.to_torch_state_dict()
    received_digest = get_digest(state_dict)

    model = TinyLinearModel()
    model.load_state_dict(state_dict)

    logical_client_id = int(config["logical_client_id"])
    x, y = get_deterministic_data(logical_client_id)

    if "inject_f01_empty_data" in config:
        x = torch.empty(0, 1)
        y = torch.empty(0, 1)

    lr = float(config.get("learning_rate", 0.1))
    epochs = int(config.get("local_epochs", 1))
    loss = local_train(model, x, y, lr, epochs)

    updated_array_record = ArrayRecord.from_torch_state_dict(model.state_dict())

    metrics = MetricRecord(
        {
            "train_loss": loss,
            "num-examples": len(x),
        }
    )

    configs = ConfigRecord(
        {
            "received_digest": received_digest,
            "updated_digest": get_digest(model.state_dict()),
            "logical_client_id": logical_client_id,
            "local_train_loss": loss,
        }
    )

    record_dict = RecordDict()
    record_dict.parameters_records["arrays"] = updated_array_record
    record_dict.metrics_records["metrics"] = metrics
    record_dict.configs_records["config"] = configs

    return message.create_reply(record_dict)


@client_app.evaluate()
def evaluate(message: Message, context: Context) -> Message:
    torch.manual_seed(13)
    torch.set_num_threads(1)
    config = message.content.configs_records.get("config", ConfigRecord())

    array_record = message.content.parameters_records["arrays"]
    state_dict = array_record.to_torch_state_dict()
    model = TinyLinearModel()
    model.load_state_dict(state_dict)

    logical_client_id = int(config["logical_client_id"])
    x, y = get_deterministic_data(logical_client_id)
    loss = local_evaluate(model, x, y)

    metrics = MetricRecord(
        {
            "eval_loss": loss,
            "num-examples": len(x),
        }
    )

    record_dict = RecordDict()
    record_dict.metrics_records["metrics"] = metrics
    return message.create_reply(record_dict)


# --- TracingFedAvg and ServerApp ---


class TracingFedAvg(FedAvg):
    def __init__(self, expected_clients: int, tolerance: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expected_clients = expected_clients
        self.tolerance = tolerance
        self.previous_aggregate_digest = None
        self.tracing_log = []

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ):
        state_dict = arrays.to_torch_state_dict()
        current_digest = get_digest(state_dict)

        if (
            server_round > 1
            and self.previous_aggregate_digest is not None
            and current_digest != self.previous_aggregate_digest
        ):
            raise SmokeValidationError(
                f"Redistribution failure in round {server_round}: expected {self.previous_aggregate_digest}, got {current_digest}"
            )

        config["server_digest"] = current_digest

        messages = list(super().configure_train(server_round, arrays, config, grid))
        if len(messages) != self.expected_clients:
            raise SmokeValidationError(
                f"Expected {self.expected_clients} clients, got {len(messages)}"
            )

        logical_ids = sorted(range(len(messages)))
        for i, msg in enumerate(messages):
            msg.content = copy.deepcopy(msg.content)
            new_conf = ConfigRecord(msg.content.configs_records.get("config", ConfigRecord()))
            new_conf["logical_client_id"] = logical_ids[i]
            msg.content.configs_records["config"] = new_conf

        self.tracing_log.append(
            {
                "round": server_round,
                "server_input_digest": current_digest,
                "clients": [],
                "aggregation_oracle_pass": False,
                "redistribution_pass": (server_round > 1),
                "aggregated_digest": None,
                "selected_client_count": len(messages),
                "selected_logical_ids": logical_ids,
            }
        )

        return messages

    def configure_evaluate(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ):
        messages = list(super().configure_evaluate(server_round, arrays, config, grid))
        logical_ids = sorted(range(len(messages)))
        for i, msg in enumerate(messages):
            msg.content = copy.deepcopy(msg.content)
            new_conf = ConfigRecord(msg.content.configs_records.get("config", ConfigRecord()))
            new_conf["logical_client_id"] = logical_ids[i]
            msg.content.configs_records["config"] = new_conf
        return messages

    def aggregate_train(self, server_round: int, replies):
        replies_list = list(replies)
        if not replies_list:
            raise SmokeValidationError("No participant replied")

        tracing_entry = self.tracing_log[-1]
        updates = []
        for msg in replies_list:
            if msg.has_error():
                raise SmokeValidationError(f"Client error received: {msg.error}")

            arr_rec = msg.content.parameters_records["arrays"]
            met_rec = msg.content.metrics_records["metrics"]
            conf_rec = msg.content.configs_records["config"]

            if "received_digest" not in conf_rec:
                raise SmokeValidationError("Client did not report received_digest")

            if conf_rec["received_digest"] != tracing_entry["server_input_digest"]:
                raise SmokeValidationError(
                    f"Client digest mismatch. Expected {tracing_entry['server_input_digest']}, got {conf_rec['received_digest']}"
                )

            state_dict = arr_rec.to_torch_state_dict()
            num_examples = int(met_rec["num-examples"])
            updates.append((state_dict, num_examples))

            client_trace = {
                "logical_client_id": int(conf_rec["logical_client_id"]),
                "received_digest": str(conf_rec["received_digest"]),
                "updated_digest": str(conf_rec["updated_digest"]),
                "num_examples": num_examples,
                "train_loss": float(conf_rec["local_train_loss"]),
            }
            tracing_entry["clients"].append(client_trace)

        oracle_result = weighted_average_state_dicts(updates)
        flower_array_record, metrics = super().aggregate_train(server_round, replies_list)
        if flower_array_record is None:
            raise SmokeValidationError("Flower aggregation returned None")

        flower_state = flower_array_record.to_torch_state_dict()
        for key in oracle_result:
            if key not in flower_state:
                raise SmokeValidationError(f"Key {key} missing from Flower aggregated state")
            if not torch.allclose(
                oracle_result[key], flower_state[key], atol=self.tolerance, rtol=0.0
            ):
                raise SmokeValidationError(f"Oracle mismatch for {key} > {self.tolerance}")

        tracing_entry["aggregation_oracle_pass"] = True
        self.previous_aggregate_digest = get_digest(flower_state)
        tracing_entry["aggregated_digest"] = self.previous_aggregate_digest

        return flower_array_record, metrics


def evaluate_fn(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    torch.manual_seed(13)
    torch.set_num_threads(1)
    state_dict = arrays.to_torch_state_dict()
    model = TinyLinearModel()
    model.load_state_dict(state_dict)

    x0, y0 = get_deterministic_data(0)
    x1, y1 = get_deterministic_data(1)
    x = torch.cat([x0, x1], dim=0)
    y = torch.cat([y0, y1], dim=0)

    loss = local_evaluate(model, x, y)
    return MetricRecord({"global_loss": loss})


SMOKE_RESULT = None
SMOKE_TRACING_LOG = []


def get_server_app(config: dict) -> ServerApp:
    app = ServerApp()

    @app.main()
    def main(grid: Grid, context: Context) -> None:
        global SMOKE_RESULT
        torch.manual_seed(13)
        torch.set_num_threads(1)

        num_clients = config["num_clients"]
        num_rounds = config["num_rounds"]
        tolerance = config.get("tolerance", 1e-7)

        train_config = ConfigRecord()
        train_config["learning_rate"] = config.get("learning_rate", 0.1)
        train_config["local_epochs"] = config.get("local_epochs", 1)

        if config.get("inject_f02_exception"):
            train_config["inject_f02_exception"] = True
        if config.get("inject_f01_empty_data"):
            train_config["inject_f01_empty_data"] = True

        initial_model = TinyLinearModel()
        torch.manual_seed(13)
        initial_model.linear.weight.data.normal_()
        initial_arrays = ArrayRecord.from_torch_state_dict(initial_model.state_dict())

        strategy = TracingFedAvg(
            expected_clients=num_clients,
            tolerance=tolerance,
            fraction_train=1.0,
            fraction_evaluate=1.0,
            min_train_nodes=num_clients,
            min_evaluate_nodes=num_clients,
            min_available_nodes=num_clients,
            weighted_by_key="num-examples",
        )

        SMOKE_RESULT = strategy.start(
            grid=grid,
            initial_arrays=initial_arrays,
            num_rounds=num_rounds,
            train_config=train_config,
            evaluate_fn=evaluate_fn,
        )
        global SMOKE_TRACING_LOG
        SMOKE_TRACING_LOG = strategy.tracing_log

    return app


def run_smoke_test(config: dict):
    global SMOKE_RESULT, SMOKE_TRACING_LOG
    SMOKE_RESULT = None
    SMOKE_TRACING_LOG = []
    app = get_server_app(config)

    # Check if a client exception will happen; if so, we can let Ray throw it and catch it in main
    # Wait, the prompt says F02 exception should be deterministic.
    run_simulation(
        server_app=app,
        client_app=client_app,
        num_supernodes=config["num_clients"],
        backend_name="ray",
        backend_config={"client_resources": {"num_cpus": 1}},
    )
    return SMOKE_RESULT, SMOKE_TRACING_LOG


def validate_config(config: dict) -> None:
    if config.get("schema") != "fl_synthetic_smoke_v1":
        raise SmokeValidationError("Invalid config schema")
    if config.get("version") != 1:
        raise SmokeValidationError("Invalid config version")
    if config.get("seed") != 13:
        raise SmokeValidationError("Invalid seed, must be 13")
    if config.get("num_clients", 0) < 2:
        raise SmokeValidationError("num_clients must be >= 2")
    if config.get("num_rounds", 0) < 3:
        raise SmokeValidationError("num_rounds must be >= 3")
    if config.get("repeat_runs", 0) < 2:
        raise SmokeValidationError("repeat_runs must be >= 2")
    if "learning_rate" not in config:
        config["learning_rate"] = 0.1
    if "local_epochs" not in config:
        config["local_epochs"] = 1
    if "tolerance" not in config:
        config["tolerance"] = 1e-7


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    validate_config(config)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Flower Synthetic FedAvg Smoke Test")
    parser.add_argument("--config", type=Path, required=True, help="Path to config JSON")
    parser.add_argument("--output", type=Path, required=True, help="Path to write summary JSON")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        logger.info("Configuration validated successfully.")

        config_hash = file_sha256(args.config)

        runtime_metadata = {
            "python_version": sys.version.split()[0],
            "flower_version": flwr.__version__,
            "ray_version": ray.__version__,
            "torch_version": torch.__version__,
            "platform": platform.platform(),
            "command": " ".join(sys.argv),
        }

        repetitions_evidence = []
        repeat_runs = config.get("repeat_runs", 2)

        for i in range(repeat_runs):
            logger.info(f"Starting repetition {i + 1}/{repeat_runs}")
            result, tracing = run_smoke_test(config)

            global_loss_history = []
            if (
                result
                and 0 in result.evaluate_metrics_serverapp
                and "global_loss" in result.evaluate_metrics_serverapp[0]
            ):
                for r in sorted(result.evaluate_metrics_serverapp.keys()):
                    global_loss_history.append(
                        (r, float(result.evaluate_metrics_serverapp[r]["global_loss"]))
                    )

            if not global_loss_history:
                raise SmokeValidationError("Missing global loss history")

            initial_loss = global_loss_history[0][1]
            final_loss = global_loss_history[-1][1]

            if final_loss >= initial_loss:
                raise SmokeValidationError(
                    f"Final global loss {final_loss} not less than initial loss {initial_loss}"
                )

            if len(global_loss_history) - 1 < config["num_rounds"]:
                raise SmokeValidationError("Not all rounds completed")

            final_digest = tracing[-1]["aggregated_digest"]

            repetitions_evidence.append(
                {
                    "run": i,
                    "global_loss_history": global_loss_history,
                    "tracing_log": list(tracing),
                    "final_model_digest": final_digest,
                }
            )

        first = repetitions_evidence[0]
        for other in repetitions_evidence[1:]:
            if other["final_model_digest"] != first["final_model_digest"]:
                raise SmokeValidationError("Final model parameters differ across repetitions")
            if (
                abs(other["global_loss_history"][-1][1] - first["global_loss_history"][-1][1])
                > config["tolerance"]
            ):
                raise SmokeValidationError("Final global loss differs across repetitions")

        uv_lock_hash = ""
        if Path("uv.lock").exists():
            uv_lock_hash = file_sha256(Path("uv.lock"))

        summary = {
            "schema": "fl_synthetic_smoke_summary_v1",
            "version": 1,
            "status": "PASS",
            "seed": config["seed"],
            "config_sha256": config_hash,
            "uv_lock_sha256": uv_lock_hash,
            "runtime_metadata": runtime_metadata,
            "repetitions": repetitions_evidence,
            "execution_details": {
                "num_clients": config["num_clients"],
                "num_rounds": config["num_rounds"],
            },
            "metrics": {"global_loss_history": first["global_loss_history"]},
            "tracing_log": first["tracing_log"],
        }

    except Exception as e:
        logger.exception("Smoke test failed")
        summary = {
            "schema": "fl_synthetic_smoke_summary_v1",
            "version": 1,
            "status": "FAIL",
            "error": str(e),
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        sys.exit(2)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary written to {args.output}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
