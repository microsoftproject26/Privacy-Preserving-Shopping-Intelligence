"""Tests for CommonInitialization generation and serialization (S1-PR-03)."""

from __future__ import annotations

import hashlib
import json

import pytest
import torch
from test_schemas import _make_model_config

from ppsi_paths import REPO_ROOT
from scripts.experiments.contracts import ContractValidationError
from scripts.experiments.initialization import (
    MAGIC_BYTES,
    deserialize_state_dict,
    generate_fixture_linear_v1_state,
    serialize_state_dict,
)

# S1-PR-03 required seeds
_SEEDS = [13, 42, 2026]


def test_independent_regeneration_identical():
    """Independent regeneration produces identical bytes/hash/size against ON-DISK proofs."""
    root = REPO_ROOT
    init_dir = root / "fixtures" / "experiments" / "common_initialization"
    cfg_path = init_dir / "fixture_model_config.v1.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    for seed in _SEEDS:
        state = generate_fixture_linear_v1_state(cfg, seed)
        bytes_data = serialize_state_dict(state)

        # Check against on-disk
        bin_path = init_dir / f"seed-{seed}.state.v1.bin"
        disk_bytes = bin_path.read_bytes()

        assert bytes_data == disk_bytes
        assert len(bytes_data) == len(disk_bytes)
        assert hashlib.sha256(bytes_data).hexdigest() == hashlib.sha256(disk_bytes).hexdigest()

        # Verify metadata
        meta_path = init_dir / f"seed-{seed}.common_init.v1.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        assert meta["artifact_kind"] == "FIXTURE_PROOF"
        assert meta["seed"] == seed
        assert meta["state_size_bytes"] == len(disk_bytes)
        assert meta["state_sha256"] == hashlib.sha256(disk_bytes).hexdigest()


def test_serialize_deserialize_restores_tensors():
    """serialize -> deserialize restores tensor values exactly."""
    root = REPO_ROOT
    init_dir = root / "fixtures" / "experiments" / "common_initialization"
    cfg_path = init_dir / "fixture_model_config.v1.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    state = generate_fixture_linear_v1_state(cfg, 42)
    bin_path = init_dir / "seed-42.state.v1.bin"
    disk_bytes = bin_path.read_bytes()

    restored = deserialize_state_dict(disk_bytes)
    assert state.keys() == restored.keys()

    for k in state:
        torch.testing.assert_close(state[k], restored[k], rtol=0, atol=0)


def test_model_config_change_invalidates_prior_init():
    """model-config change invalidates prior init."""
    cfg1 = _make_model_config()
    state1 = generate_fixture_linear_v1_state(cfg1, 42)
    bytes1 = serialize_state_dict(state1)

    # Change model config (e.g., add a task)
    cfg2 = _make_model_config()
    cfg2["task_heads"].append(
        {
            "task": "T2",
            "head_id": "h2",
            "head_version": "1",
            "output_dim": 2,
        }
    )

    state2 = generate_fixture_linear_v1_state(cfg2, 42)
    bytes2 = serialize_state_dict(state2)

    assert bytes1 != bytes2
    assert hashlib.sha256(bytes1).hexdigest() != hashlib.sha256(bytes2).hexdigest()


def test_state_tamper_fails():
    """state tamper fails."""
    cfg = _make_model_config()
    state = generate_fixture_linear_v1_state(cfg, 42)
    bytes_data = bytearray(serialize_state_dict(state))

    # Flip a byte in the payload
    bytes_data[-1] ^= 0xFF

    # It might fail metadata check if we hit metadata, or tensor values will be wrong
    with pytest.raises((Exception,)):
        # We don't have a hash-check on deserialization yet since the SHA is external,
        # but parsing invalid metadata will fail JSON decode.
        # However, tampering payload changes tensor values, breaking cryptographic equality.
        restored = deserialize_state_dict(bytes(bytes_data))
        for k in state:
            torch.testing.assert_close(state[k], restored[k], rtol=0, atol=0)


def test_missing_state_fails():
    """missing state fails."""
    with pytest.raises(ContractValidationError, match="empty"):
        serialize_state_dict({})


def test_wrong_architecture_fails():
    """wrong architecture/config hash fails."""
    cfg = _make_model_config(architecture_id="unknown")
    with pytest.raises(ContractValidationError, match="Only fixture_linear_v1"):
        generate_fixture_linear_v1_state(cfg, 42)


def test_unsupported_nonfinite_state_fails():
    """unsupported/nonfinite state fails."""
    cfg = _make_model_config()
    state = generate_fixture_linear_v1_state(cfg, 42)

    # Inject nonfinite
    state["backbone.weight"][0, 0] = float("inf")
    with pytest.raises(ContractValidationError, match="finite"):
        serialize_state_dict(state)


def test_generator_restores_caller_rng():
    """generator restores caller Torch RNG state."""
    torch.manual_seed(999)
    val1 = torch.rand(1).item()

    torch.manual_seed(999)
    cfg = _make_model_config()
    _ = generate_fixture_linear_v1_state(cfg, 42)

    val2 = torch.rand(1).item()
    assert val1 == val2


def test_not_torch_save_bytes():
    """torch.save/pickle bytes are NOT the scientific identity."""
    cfg = _make_model_config()
    state = generate_fixture_linear_v1_state(cfg, 42)
    bytes_data = serialize_state_dict(state)

    import io

    buf = io.BytesIO()
    torch.save(state, buf)
    torch_bytes = buf.getvalue()

    assert bytes_data != torch_bytes
    assert bytes_data.startswith(MAGIC_BYTES)
