from __future__ import annotations

import pytest

from ppsi.training.fixtures import default_batch_spec, make_phase1_batch


@pytest.fixture
def batch_spec():
    return default_batch_spec()


@pytest.fixture
def phase1_batch():
    return make_phase1_batch()
