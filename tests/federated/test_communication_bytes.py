from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from flwr.app import ArrayRecord

from ppsi.federated.communication import (
    DOWNLOAD,
    UPLOAD,
    CommunicationLedger,
    CommunicationRecord,
    array_record_bytes,
)

EVIDENCE_PATH = (
    Path(__file__).resolve().parents[2] / "docs" / "evidence" / "s2-se-05" / "communication_bytes.v1.json"
)


def tiny_record() -> ArrayRecord:
    return ArrayRecord.from_torch_state_dict(torch.nn.Linear(16, 8).state_dict())


def test_known_size_payload_is_measured_after_serialization() -> None:
    record = tiny_record()
    raw_element_bytes = sum(tensor.numel() * 4 for tensor in record.to_torch_state_dict().values())

    measured = array_record_bytes(record)

    # 8x16 + 8 float32 parameters are 544 bytes of numbers. What Flower sends is
    # larger, because each array carries its own serialization header. A
    # parameter-count estimate would report the smaller number.
    assert raw_element_bytes == 544
    assert measured == 800
    assert measured > raw_element_bytes
    assert measured == sum(len(array.data) for array in record.values())


def test_payload_bytes_scale_with_the_model_not_the_client_count() -> None:
    small = array_record_bytes(ArrayRecord.from_torch_state_dict(torch.nn.Linear(4, 4).state_dict()))
    large = array_record_bytes(
        ArrayRecord.from_torch_state_dict(torch.nn.Linear(256, 256).state_dict())
    )

    assert large > small


def test_directions_are_totalled_separately() -> None:
    ledger = CommunicationLedger()
    record = tiny_record()
    per_payload = array_record_bytes(record)

    for server_round in (1, 2):
        for client in ("client-a", "client-b"):
            ledger.measure(record, server_round=server_round, client_id=client, direction=DOWNLOAD)
        ledger.measure(record, server_round=server_round, client_id="client-a", direction=UPLOAD)

    totals = ledger.run_totals()
    assert totals["download_bytes"] == 4 * per_payload
    assert totals["upload_bytes"] == 2 * per_payload
    assert totals["download_bytes"] != totals["upload_bytes"]
    assert totals["total_bytes"] == totals["download_bytes"] + totals["upload_bytes"]

    per_round = ledger.round_totals()
    assert sorted(per_round) == [1, 2]
    assert per_round[1]["download_transmissions"] == 2
    assert per_round[1]["upload_transmissions"] == 1


def test_a_raw_numeric_user_identifier_is_refused() -> None:
    with pytest.raises(ValueError, match="opaque client ID"):
        CommunicationRecord(server_round=1, client_id="554748815", direction=UPLOAD, payload_bytes=1)


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"server_round": 0}, ValueError),
        ({"client_id": ""}, ValueError),
        ({"direction": "sideways"}, ValueError),
        ({"payload_bytes": -1}, ValueError),
        ({"payload_bytes": 1.5}, TypeError),
    ],
)
def test_malformed_records_are_rejected(kwargs, error) -> None:
    base = {"server_round": 1, "client_id": "client-a", "direction": UPLOAD, "payload_bytes": 10}

    with pytest.raises(error):
        CommunicationRecord(**{**base, **kwargs})


def test_a_record_cannot_carry_payload_contents() -> None:
    record = CommunicationRecord(
        server_round=1, client_id="client-a", direction=UPLOAD, payload_bytes=800
    )

    assert set(record.as_dict()) == {"server_round", "client_id", "direction", "payload_bytes"}
    with pytest.raises(TypeError):
        CommunicationRecord(
            server_round=1,
            client_id="client-a",
            direction=UPLOAD,
            payload_bytes=800,
            payload=b"weights",
        )


def test_committed_evidence_separates_directions_for_at_least_two_clients() -> None:
    report = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))

    assert report["schema"] == "CommunicationRecord"
    assert report["client_count"] >= 2
    assert len(report["per_round"]) >= 2

    totals = report["run_totals"]
    assert totals["download_bytes"] > 0
    assert totals["upload_bytes"] > 0
    assert totals["total_bytes"] == totals["download_bytes"] + totals["upload_bytes"]

    per_payload = report["context"]["serialized_payload_bytes_per_transmission"]
    assert totals["download_bytes"] == per_payload * totals["download_transmissions"]
    assert totals["upload_bytes"] == per_payload * totals["upload_transmissions"]

    for row in report["records"]:
        assert set(row) == {"server_round", "client_id", "direction", "payload_bytes"}
        assert not row["client_id"].isdigit()
        assert row["client_id"].startswith("client-v1-")
