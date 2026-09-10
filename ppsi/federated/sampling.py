"""Deterministic per-round client sampling for simulated federated learning.

This module owns which logical clients participate in each server round.
It does NOT own within-client batch ordering (see ppsi.training.sampler).
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SAMPLER_VERSION = "client_sampler_v1"


# ---------------------------------------------------------------------------
# Seed derivation
# ---------------------------------------------------------------------------


def derive_round_seed(
    sampler_version: str,
    experiment_seed: int,
    round_index: int,
) -> int:
    """Derive a deterministic 64-bit seed for a specific round.

    Uses SHA-256 of the concatenation of sampler version, experiment seed,
    and round index. Does NOT use Python ``hash()``.
    """
    material = f"{sampler_version}|{experiment_seed}|{round_index}".encode()
    digest = hashlib.sha256(material).digest()
    return int.from_bytes(digest[:8], "big")


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SamplingResult:
    """Result of a single round's client sampling."""

    selected_client_ids: list[str]
    selected_digest: str


def selection_digest(selected_ids: list[str]) -> str:
    """Compute a deterministic SHA-256 digest of sorted selected client IDs."""
    canonical = "\n".join(sorted(selected_ids))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def sample_clients(
    eligible_client_ids: list[str],
    experiment_seed: int,
    round_index: int,
    clients_per_round: int,
    *,
    sampler_version: str = DEFAULT_SAMPLER_VERSION,
) -> SamplingResult:
    """Select clients for a federated round, without replacement.

    Parameters
    ----------
    eligible_client_ids
        Full pool of eligible client IDs.
    experiment_seed
        The experiment-level seed (e.g. 13, 42, 2026).
    round_index
        Zero-based round index.
    clients_per_round
        Number of clients to select.
    sampler_version
        Versioned sampler identifier for seed derivation.

    Returns
    -------
    SamplingResult
        Selected client IDs and their deterministic digest.

    Raises
    ------
    ValueError
        If clients_per_round > len(eligible_client_ids) or inputs are invalid.
    """
    if not isinstance(eligible_client_ids, list):
        raise TypeError("eligible_client_ids must be a list")
    if clients_per_round <= 0:
        raise ValueError("clients_per_round must be positive")
    if round_index < 0:
        raise ValueError("round_index must be non-negative")

    pool = sorted(eligible_client_ids)

    if clients_per_round > len(pool):
        raise ValueError(
            f"clients_per_round ({clients_per_round}) exceeds eligible pool "
            f"size ({len(pool)}). Silent shrinking is not allowed."
        )

    seed = derive_round_seed(sampler_version, experiment_seed, round_index)
    rng = random.Random(seed)
    selected = rng.sample(pool, clients_per_round)

    digest = selection_digest(selected)
    return SamplingResult(selected_client_ids=selected, selected_digest=digest)


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ClientSamplingTrace:
    """Per-round trace record for auditing and reproducibility."""

    trace_version: str
    sampler_version: str
    experiment_seed: int
    round_index: int
    eligible_pool_count: int
    clients_per_round: int
    selected_client_ids: list[str]
    selected_digest: str
    failed_client_ids: list[str] = field(default_factory=list)
    retried_client_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)


def build_trace(
    result: SamplingResult,
    *,
    sampler_version: str,
    experiment_seed: int,
    round_index: int,
    eligible_pool_count: int,
    clients_per_round: int,
) -> ClientSamplingTrace:
    """Build a ClientSamplingTrace from a SamplingResult."""
    return ClientSamplingTrace(
        trace_version="client_sampling_trace_v1",
        sampler_version=sampler_version,
        experiment_seed=experiment_seed,
        round_index=round_index,
        eligible_pool_count=eligible_pool_count,
        clients_per_round=clients_per_round,
        selected_client_ids=result.selected_client_ids,
        selected_digest=result.selected_digest,
    )
