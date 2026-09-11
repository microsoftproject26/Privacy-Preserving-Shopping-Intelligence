"""Directional byte accounting for Flower model payloads.

The number this records is the length of the serialized arrays that actually
cross the send and receive boundary, not a parameter count multiplied by a
dtype width. Serialization carries real overhead, so the two differ.

The boundary is the model payload only:

- ``download`` is server to client;
- ``upload`` is client to server.

That is narrower than total network traffic. gRPC framing, headers, config
records and metrics are not counted, and this module says so rather than
letting a payload figure be read as a wire figure.

Nothing here stores payload contents, and :class:`CommunicationRecord` has no
field that could hold them. Client identities must already be opaque; see
``ppsi.federated.clients.client_id_from_user``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

DOWNLOAD = "download"
"""Server to client."""

UPLOAD = "upload"
"""Client to server."""

DIRECTIONS = (DOWNLOAD, UPLOAD)

MEASUREMENT_BOUNDARY = "model payload after serialization; excludes transport framing"


class SerializedArray(Protocol):
    """Any array exposing the serialized bytes Flower transmits."""

    data: bytes


def array_record_bytes(record: Mapping[str, SerializedArray] | Any) -> int:
    """Return the serialized payload bytes of one record of arrays.

    Accepts a Flower ``ArrayRecord`` or any mapping whose values expose the
    serialized ``data`` buffer. Duck-typed so the ledger stays usable from
    tests without constructing a Flower message.
    """

    try:
        arrays = record.values()
    except AttributeError as error:
        raise TypeError("record must be a mapping of name to serialized array") from error

    total = 0
    for array in arrays:
        payload = getattr(array, "data", None)
        if payload is None:
            raise TypeError("every array must expose serialized 'data' bytes")
        total += len(payload)
    return total


@dataclass(frozen=True, slots=True)
class CommunicationRecord:
    """One measured model-payload transmission."""

    server_round: int
    client_id: str
    direction: str
    payload_bytes: int

    def __post_init__(self) -> None:
        if isinstance(self.server_round, bool) or not isinstance(self.server_round, int):
            raise TypeError("server_round must be an integer")
        if self.server_round < 1:
            raise ValueError("server rounds start at 1")
        if not isinstance(self.client_id, str) or not self.client_id:
            raise ValueError("client_id must be a non-empty string")
        if self.client_id.isdigit():
            raise ValueError("client_id looks like a raw user identifier; pass an opaque client ID")
        if self.direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}")
        if isinstance(self.payload_bytes, bool) or not isinstance(self.payload_bytes, int):
            raise TypeError("payload_bytes must be an integer")
        if self.payload_bytes < 0:
            raise ValueError("payload_bytes must not be negative")

    def as_dict(self) -> dict[str, int | str]:
        return {
            "server_round": self.server_round,
            "client_id": self.client_id,
            "direction": self.direction,
            "payload_bytes": self.payload_bytes,
        }


class CommunicationLedger:
    """Collects measured transmissions and totals them per round and per run."""

    def __init__(self) -> None:
        self._records: list[CommunicationRecord] = []

    @property
    def records(self) -> tuple[CommunicationRecord, ...]:
        return tuple(self._records)

    def add(self, record: CommunicationRecord) -> CommunicationRecord:
        self._records.append(record)
        return record

    def measure(
        self,
        arrays: Mapping[str, SerializedArray] | Any,
        *,
        server_round: int,
        client_id: str,
        direction: str,
    ) -> CommunicationRecord:
        """Measure one record of arrays and add it to the ledger."""

        return self.add(
            CommunicationRecord(
                server_round=server_round,
                client_id=client_id,
                direction=direction,
                payload_bytes=array_record_bytes(arrays),
            )
        )

    def round_totals(self) -> dict[int, dict[str, int]]:
        """Directional bytes and transmission counts for every measured round."""

        totals: dict[int, dict[str, int]] = {}
        for record in self._records:
            bucket = totals.setdefault(
                record.server_round,
                {
                    "download_bytes": 0,
                    "upload_bytes": 0,
                    "download_transmissions": 0,
                    "upload_transmissions": 0,
                },
            )
            bucket[f"{record.direction}_bytes"] += record.payload_bytes
            bucket[f"{record.direction}_transmissions"] += 1
        return dict(sorted(totals.items()))

    def run_totals(self) -> dict[str, int]:
        """Directional bytes and transmission counts for the whole run."""

        totals = {
            "download_bytes": 0,
            "upload_bytes": 0,
            "download_transmissions": 0,
            "upload_transmissions": 0,
        }
        for record in self._records:
            totals[f"{record.direction}_bytes"] += record.payload_bytes
            totals[f"{record.direction}_transmissions"] += 1
        totals["total_bytes"] = totals["download_bytes"] + totals["upload_bytes"]
        return totals

    def client_ids(self) -> tuple[str, ...]:
        return tuple(sorted({record.client_id for record in self._records}))

    def report(self, *, context: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Build the JSON-ready artifact. Bytes only; never payload contents."""

        return {
            "schema": "CommunicationRecord",
            "version": 1,
            "task_id": "S2-SE-05",
            "measurement_boundary": MEASUREMENT_BOUNDARY,
            "context": dict(context or {}),
            "client_count": len(self.client_ids()),
            "per_round": {str(k): v for k, v in self.round_totals().items()},
            "run_totals": self.run_totals(),
            "records": [record.as_dict() for record in self._records],
        }


def extend(ledger: CommunicationLedger, records: Iterable[CommunicationRecord]) -> None:
    for record in records:
        ledger.add(record)
