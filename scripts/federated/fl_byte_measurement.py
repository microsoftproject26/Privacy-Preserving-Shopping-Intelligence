"""Measure Flower model-payload bytes at the real send and receive boundary.

Runs a small deterministic FedAvg simulation and records, for every round and
every client, how many serialized bytes went server-to-client and how many came
back client-to-server. The figures come from the arrays Flower actually
transmits, so they include serialization overhead a parameter count cannot see.

Reproduce::

    uv run --locked python scripts/federated/fl_byte_measurement.py \
        --output docs/evidence/s2-se-05/communication_bytes.v1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from flwr.simulation import run_simulation

from ppsi.federated.clients import client_id_from_user
from ppsi.federated.communication import (
    DOWNLOAD,
    UPLOAD,
    CommunicationLedger,
    array_record_bytes,
)

SEED = 13
NUM_CLIENTS = 2
NUM_ROUNDS = 2
ARRAYS_KEY = "arrays"
CONFIG_KEY = "config"

LEDGER = CommunicationLedger()


def opaque_client_id(partition: int) -> str:
    """An opaque ID for a simulated partition, using the existing identity contract."""

    return client_id_from_user(f"simulated-partition-{partition}")


class TinyModel(torch.nn.Module):
    """Small enough to run anywhere, large enough that bytes are not trivial."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(16, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def deterministic_data(partition: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(SEED + partition)
    x = torch.randn(8, 16, generator=generator)
    y = torch.randn(8, 8, generator=generator)
    return x, y


class ByteMeasuringFedAvg(FedAvg):
    """FedAvg that measures each outgoing and returning model payload.

    Measuring here rather than inside the client is deliberate: this is the one
    place that sees every transmission of a round, including a client whose
    reply never arrives.
    """

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ):
        messages = list(super().configure_train(server_round, arrays, config, grid))
        for partition, message in enumerate(messages):
            outgoing = message.content.array_records[ARRAYS_KEY]
            LEDGER.measure(
                outgoing,
                server_round=server_round,
                client_id=opaque_client_id(partition),
                direction=DOWNLOAD,
            )
            assigned = ConfigRecord(message.content.config_records.get(CONFIG_KEY, ConfigRecord()))
            assigned["partition"] = partition
            message.content.config_records[CONFIG_KEY] = assigned
        return messages

    def aggregate_train(self, server_round: int, replies):
        replies = list(replies)
        for reply in replies:
            if reply.has_error():
                # A reply that never arrived transmitted nothing; it is not a zero-byte
                # upload, so it is left out rather than recorded as one.
                continue
            returned = reply.content.array_records[ARRAYS_KEY]
            partition = int(reply.content.config_records[CONFIG_KEY]["partition"])
            LEDGER.measure(
                returned,
                server_round=server_round,
                client_id=opaque_client_id(partition),
                direction=UPLOAD,
            )
        return super().aggregate_train(server_round, replies)


client_app = ClientApp()


@client_app.train()
def train(message: Message, context: Context) -> Message:
    torch.manual_seed(SEED)
    torch.set_num_threads(1)

    config = message.content.config_records[CONFIG_KEY]
    partition = int(config["partition"])

    model = TinyModel()
    model.load_state_dict(message.content.array_records[ARRAYS_KEY].to_torch_state_dict())

    x, y = deterministic_data(partition)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    optimizer.zero_grad()
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()
    optimizer.step()

    reply = RecordDict()
    reply.array_records[ARRAYS_KEY] = ArrayRecord.from_torch_state_dict(model.state_dict())
    reply.metric_records["metrics"] = MetricRecord({"num-examples": len(x)})
    reply.config_records[CONFIG_KEY] = ConfigRecord({"partition": partition})
    return Message(reply, reply_to=message)


def build_server_app() -> ServerApp:
    app = ServerApp()

    @app.main()
    def main(grid: Grid, context: Context) -> None:
        torch.manual_seed(SEED)
        torch.set_num_threads(1)

        initial = TinyModel()
        strategy = ByteMeasuringFedAvg(
            fraction_train=1.0,
            fraction_evaluate=0.0,
            min_train_nodes=NUM_CLIENTS,
            min_available_nodes=NUM_CLIENTS,
            weighted_by_key="num-examples",
        )
        strategy.start(
            grid=grid,
            initial_arrays=ArrayRecord.from_torch_state_dict(initial.state_dict()),
            num_rounds=NUM_ROUNDS,
            train_config=ConfigRecord(),
        )

    return app


def run_measurement() -> CommunicationLedger:
    """Run the simulation and return the populated ledger."""

    global LEDGER
    LEDGER = CommunicationLedger()
    run_simulation(
        server_app=build_server_app(),
        client_app=client_app,
        num_supernodes=NUM_CLIENTS,
        backend_name="ray",
        backend_config={"client_resources": {"num_cpus": 1}},
    )
    return LEDGER


def expected_payload_bytes() -> int:
    """The serialized size of one model payload, computed without a simulation."""

    return array_record_bytes(ArrayRecord.from_torch_state_dict(TinyModel().state_dict()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure Flower model-payload bytes")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    ledger = run_measurement()
    report = ledger.report(
        context={
            "flower_backend": "ray simulation",
            "clients": NUM_CLIENTS,
            "rounds": NUM_ROUNDS,
            "seed": SEED,
            "model": "Linear(16, 8)",
            "serialized_payload_bytes_per_transmission": expected_payload_bytes(),
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    totals = report["run_totals"]
    print(
        f"download {totals['download_bytes']} B in {totals['download_transmissions']} "
        f"transmissions; upload {totals['upload_bytes']} B in "
        f"{totals['upload_transmissions']} transmissions"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
