"""Small deterministic guards for the scoped T1 MVP, not a new training framework."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


def raw_file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_text_sha256(path: Path | str) -> str:
    text = Path(path).read_bytes().decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def require_unique_clients(client_ids: list[str]) -> list[str]:
    if not client_ids or not all(isinstance(v, str) and v for v in client_ids):
        raise ValueError("eligible clients must be nonempty opaque strings")
    if any(v.isdigit() for v in client_ids):
        raise ValueError("raw numeric user IDs may not be used as federated client IDs")
    if len(set(client_ids)) != len(client_ids):
        raise ValueError("duplicate eligible client IDs")
    return sorted(client_ids)


def client_epoch_order(rows, *, seed: int, server_round: int, client_id: str) -> np.ndarray:
    """One permutation of every local row, never a cap or sampling with replacement."""
    values = np.asarray(rows)
    if values.ndim != 1 or values.dtype.kind not in "iu" or len(values) == 0:
        raise ValueError("local rows must be a nonempty integer vector")
    if np.any(values < 0) or len(np.unique(values)) != len(values):
        raise ValueError("local rows must be unique and non-negative")
    # A bool IS an int, so it is a forbidden value rather than a wrong type; anything
    # else that is not an int is a type error. Both stay rejected.
    if seed is True or seed is False:
        raise ValueError("seed must be an integer, not a boolean")
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if isinstance(server_round, bool) or not isinstance(server_round, int) or server_round < 1:
        raise ValueError("server round must be a positive integer")
    require_unique_clients([client_id])
    material = json.dumps(
        ["mvp-client-order-v1", seed, server_round, client_id], separators=(",", ":")
    ).encode("utf-8")
    local_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    order = np.sort(values.astype(np.int64, copy=True))
    np.random.default_rng(local_seed).shuffle(order)
    return order


def client_rng_seed(*, seed: int, server_round: int, client_id: str) -> int:
    """A stable per-client-round dropout stream independent of Ray worker assignment."""
    if seed is True or seed is False:
        raise ValueError("seed must be an integer, not a boolean")
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if isinstance(server_round, bool) or not isinstance(server_round, int) or server_round < 1:
        raise ValueError("server round must be a positive integer")
    require_unique_clients([client_id])
    material = json.dumps(
        ["mvp-local-rng-v1", seed, server_round, client_id], separators=(",", ":")
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & ((1 << 63) - 1)


def exposure_digest(records: list[tuple[int, str, list[str]]]) -> str:
    """Digest ordered round/client/decision keys. This structure is PRIVATE."""
    digest = hashlib.sha256(b"mvp-exposure-v1\n")
    seen = set()
    for server_round, client_id, keys in records:
        require_unique_clients([client_id])
        if (
            isinstance(server_round, bool)
            or not isinstance(server_round, int)
            or server_round < 1
            or (server_round, client_id) in seen
        ):
            raise ValueError("invalid or duplicate round/client exposure")
        if (
            not keys
            or len(set(keys)) != len(keys)
            or not all(isinstance(k, str) and k for k in keys)
        ):
            raise ValueError("decision keys must be unique nonempty string lists per local epoch")
        seen.add((server_round, client_id))
        digest.update(json.dumps([server_round, client_id, keys], separators=(",", ":")).encode())
        digest.update(b"\n")
    if not seen:
        raise ValueError("no exposure records")
    return digest.hexdigest()


def require_complete_replies(selected: list[str], received: list[str]) -> None:
    expected = require_unique_clients(selected)
    actual = require_unique_clients(received)
    if actual != expected:
        raise ValueError("incomplete or unexpected client reply set; partial aggregation forbidden")


def require_finite_metric(value: Any) -> float:
    if isinstance(value, (bool, str)) or value is None:
        raise ValueError("metric must be an actual finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("nonfinite metric")
    return number


def require_comparison_identity(left: dict, right: dict) -> None:
    """Precheck BEFORE official registry compatibility; does not replace that validator."""
    required = (
        "model_sha256",
        "batch_spec_sha256",
        "common_init_sha256",
        "data_manifest_sha256",
        "evaluation_membership_sha256",
        "evaluator_sha256",
        "exposure_sha256",
        "seed",
        "metric_id",
        "score_convention",
        "validation_decisions",
    )
    missing = [key for key in required if key not in left or key not in right]
    if missing:
        raise ValueError(f"missing comparison identity: {missing}")
    for record in (left, right):
        for key in required:
            if record[key] is None or record[key] == "":
                raise ValueError(f"unmeasured comparison identity: {key}")
            if key.endswith("_sha256") and (
                not isinstance(record[key], str) or not re.fullmatch(r"[0-9a-f]{64}", record[key])
            ):
                raise ValueError(f"invalid comparison digest: {key}")
        for key in ("seed", "validation_decisions"):
            value = record[key]
            if value is True or value is False:
                raise ValueError(f"comparison {key} must be an integer, not a boolean")
            if not isinstance(value, int):
                raise TypeError(f"comparison {key} must be an integer")
        if record["validation_decisions"] <= 0:
            raise ValueError("comparison validation support must be positive")
    mismatched = [key for key in required if left[key] != right[key]]
    if mismatched:
        raise ValueError(f"incomparable run identities: {mismatched}")
    if left["score_convention"] != "RAW_NO_SUPPRESSION":
        raise ValueError("this MVP uses the frozen raw T1 convention only")


def metric_delta(left: dict, right: dict) -> float:
    """Right minus left, only after explicit semantic identity equality. Not QR."""
    require_comparison_identity(left, right)
    return require_finite_metric(right["value"]) - require_finite_metric(left["value"])
