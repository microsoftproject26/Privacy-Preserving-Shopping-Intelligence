from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from ppsi.training.fixtures import default_batch_spec, default_stub_model_config
from ppsi.training.initialization import (
    InitializationVerificationError,
    generate_stub_initialization_fixture,
    load_verified_stub_initialization,
)
from ppsi.training.stub_model import Phase1StubModel, StubModelConfig
from scripts.experiments.contracts import ContractValidationError


@pytest.mark.parametrize("seed", [13, 42, 2026])
def test_stub_common_initialization_is_deterministic_and_loadable(seed: int):
    spec = default_batch_spec()
    config = default_stub_model_config()
    first = generate_stub_initialization_fixture(
        batch_spec=spec,
        model_config=config,
        seed=seed,
    )
    second = generate_stub_initialization_fixture(
        batch_spec=spec,
        model_config=config,
        seed=seed,
    )
    assert first == second

    target = Phase1StubModel(batch_spec=spec, config=config)
    load_verified_stub_initialization(
        target,
        first,
        expected_seed=seed,
        expected_state_sha256=first.state_sha256,
        expected_model_config_sha256=first.model_config_sha256,
    )

    source = Phase1StubModel(batch_spec=spec, config=config)
    load_verified_stub_initialization(
        source,
        second,
        expected_seed=seed,
        expected_state_sha256=second.state_sha256,
        expected_model_config_sha256=second.model_config_sha256,
    )
    for key, tensor in source.state_dict().items():
        torch.testing.assert_close(tensor, target.state_dict()[key], rtol=0, atol=0)


def test_disallowed_seed_fails_before_generation():
    with pytest.raises(ContractValidationError):
        generate_stub_initialization_fixture(
            batch_spec=default_batch_spec(),
            model_config=default_stub_model_config(),
            seed=999,
        )


def test_tampered_payload_or_wrong_hash_fails_before_model_mutation():
    spec = default_batch_spec()
    config = default_stub_model_config()
    fixture = generate_stub_initialization_fixture(
        batch_spec=spec,
        model_config=config,
        seed=13,
    )
    model = Phase1StubModel(batch_spec=spec, config=config)
    before = {key: value.clone() for key, value in model.state_dict().items()}

    with pytest.raises(InitializationVerificationError, match="state SHA-256"):
        load_verified_stub_initialization(
            model,
            replace(fixture, payload=fixture.payload + b"tampered"),
            expected_seed=13,
            expected_state_sha256=fixture.state_sha256,
            expected_model_config_sha256=fixture.model_config_sha256,
        )
    for key, tensor in before.items():
        torch.testing.assert_close(tensor, model.state_dict()[key], rtol=0, atol=0)

    with pytest.raises(InitializationVerificationError, match="state SHA-256"):
        load_verified_stub_initialization(
            model,
            fixture,
            expected_seed=13,
            expected_state_sha256="0" * 64,
            expected_model_config_sha256=fixture.model_config_sha256,
        )

    with pytest.raises(InitializationVerificationError, match="seed mismatch"):
        load_verified_stub_initialization(
            model,
            fixture,
            expected_seed=42,
            expected_state_sha256=fixture.state_sha256,
            expected_model_config_sha256=fixture.model_config_sha256,
        )


def test_wrong_model_config_identity_fails_before_load():
    spec = default_batch_spec()
    config = default_stub_model_config()
    fixture = generate_stub_initialization_fixture(
        batch_spec=spec,
        model_config=config,
        seed=13,
    )
    wrong_model = Phase1StubModel(
        batch_spec=spec,
        config=StubModelConfig(
            category_count=config.category_count,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim + 1,
        ),
    )
    with pytest.raises(InitializationVerificationError, match="ModelConfig identity"):
        load_verified_stub_initialization(
            wrong_model,
            fixture,
            expected_seed=13,
            expected_state_sha256=fixture.state_sha256,
            expected_model_config_sha256=fixture.model_config_sha256,
        )
