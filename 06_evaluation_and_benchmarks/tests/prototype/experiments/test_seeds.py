"""Audit test for frozen seeds.

This test file acts as the executable proof that the only
authorized random seeds for experiments are 13, 42, and 2026.
"""

from __future__ import annotations

import pytest

from scripts.experiments.contracts import ContractValidationError, validate_seed


def test_frozen_authorized_seeds() -> None:
    """Proof that exactly 13, 42, and 2026 are authorized."""
    assert validate_seed(13) == 13
    assert validate_seed(42) == 42
    assert validate_seed(2026) == 2026


@pytest.mark.parametrize("unauthorized_seed", [0, 1, 100, 1234, -1, 43, 2025])
def test_unauthorized_seeds_rejected(unauthorized_seed: int) -> None:
    """Proof that any other integer seed is strictly rejected."""
    with pytest.raises(ContractValidationError, match="must be one of"):
        validate_seed(unauthorized_seed)


def test_seed_type_strictness() -> None:
    """Proof that seeds must be strict integers."""
    with pytest.raises(ContractValidationError, match="must be int"):
        validate_seed(13.0)  # type: ignore

    with pytest.raises(ContractValidationError, match="must be int"):
        validate_seed(True)  # type: ignore

    with pytest.raises(ContractValidationError, match="must be int"):
        validate_seed("13")  # type: ignore
