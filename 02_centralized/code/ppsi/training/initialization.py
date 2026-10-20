"""S1-PR-05 glue for verified, non-authoritative stub initialization.

The byte format, seed restrictions and canonical hashing are reused from the
already-merged S1-PR-03 implementation. These fixtures are contract proofs,
not authoritative final GRU initialization artifacts.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

from ppsi.training.batch import Phase1BatchSpec
from ppsi.training.state import SharedStateSpec, load_shared_state
from ppsi.training.stub_model import Phase1StubModel, StubModelConfig
from scripts.experiments.contracts import canonical_sha256, validate_seed
from scripts.experiments.initialization import (
    deserialize_state_dict,
    deterministic_rng,
    serialize_state_dict,
)


class InitializationVerificationError(ValueError):
    """Raised before training when fixture initialization identity is invalid."""


@dataclass(frozen=True, slots=True)
class StubInitializationFixture:
    """Clearly non-authoritative initialization proof for the #18 stub model."""

    artifact_kind: str
    seed: int
    model_config_sha256: str
    state_sha256: str
    state_size_bytes: int
    payload: bytes


def stub_model_config_sha256(
    batch_spec: Phase1BatchSpec,
    model_config: StubModelConfig,
) -> str:
    """Canonical identity of the complete stub shape/config contract."""

    value = {
        "schema": "s1_pr_05_stub_model_config_v1",
        "version": "1",
        "artifact_kind": "FIXTURE_PROOF",
        "batch_spec": asdict(batch_spec),
        "model_config": asdict(model_config),
    }
    return canonical_sha256(value)


def generate_stub_initialization_fixture(
    *,
    batch_spec: Phase1BatchSpec,
    model_config: StubModelConfig,
    seed: int,
) -> StubInitializationFixture:
    """Generate deterministic bytes through the frozen #16 serializer."""

    with deterministic_rng(seed):
        model = Phase1StubModel(batch_spec=batch_spec, config=model_config)
    payload = serialize_state_dict(model.state_dict())
    return StubInitializationFixture(
        artifact_kind="FIXTURE_PROOF",
        seed=seed,
        model_config_sha256=stub_model_config_sha256(batch_spec, model_config),
        state_sha256=hashlib.sha256(payload).hexdigest(),
        state_size_bytes=len(payload),
        payload=payload,
    )


def load_verified_stub_initialization(
    model: Phase1StubModel,
    fixture: StubInitializationFixture,
    *,
    expected_seed: int,
    expected_state_sha256: str,
    expected_model_config_sha256: str,
    shared_state_spec: SharedStateSpec | None = None,
) -> None:
    """Verify every fixture identity before mutating the target model."""

    if fixture.artifact_kind != "FIXTURE_PROOF":
        raise InitializationVerificationError("Stub initialization must be FIXTURE_PROOF")
    validate_seed(expected_seed)
    if fixture.seed != expected_seed:
        raise InitializationVerificationError("Stub initialization seed mismatch")
    actual_state_sha = hashlib.sha256(fixture.payload).hexdigest()
    if actual_state_sha != fixture.state_sha256 or actual_state_sha != expected_state_sha256:
        raise InitializationVerificationError("Stub initialization state SHA-256 mismatch")

    actual_model_config_sha = stub_model_config_sha256(model.batch_spec, model.config)
    if (
        fixture.model_config_sha256 != expected_model_config_sha256
        or actual_model_config_sha != expected_model_config_sha256
    ):
        raise InitializationVerificationError("Stub ModelConfig identity mismatch")

    state = deserialize_state_dict(fixture.payload)
    spec = shared_state_spec or model.shared_state_spec()
    # #18 has all state shared. load_shared_state validates key/shape/dtype/finiteness.
    load_shared_state(model, state, spec)
