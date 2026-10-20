"""DATA-07 test configuration: ppsi_env FIRST (snapshot isolation), strict FP32, 1 CPU thread, no byte-code.

Negative controls are `xfail(strict=True, raises=AssertionError)` tests named test_nc_*: each runs a key-property check
against a deliberately wrong variant and MUST fail at its assertion (an XPASS fails the suite).
Real-data fixtures read the frozen v1 dataset read-only (bucket 0; TRAIN and the two VALIDATION splits, never TEST).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
CODE = HERE.parent
for _p in (str(CODE), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import ppsi_env  # noqa: E402,F401  (mandatory isolation, before any snapshot import)

import pytest  # noqa: E402

import v4train  # noqa: E402,F401
import torch  # noqa: E402

from fl.numerics import enforce_strict_fp32  # noqa: E402


@pytest.fixture(autouse=True)
def _strict_numerics():
    torch.set_num_threads(1)
    enforce_strict_fp32()
    yield
    enforce_strict_fp32()


@pytest.fixture(scope="session")
def bucket0():
    """The full bucket-0 event table (all users; every non-TEST row)."""
    from trunckit import REAL_DS, have_real
    if not have_real():
        pytest.skip("the frozen v1 dataset is not present on this host")
    from v4trunc.events import load_bucket
    return load_bucket(REAL_DS, 0)


@pytest.fixture(scope="session")
def sample_table(bucket0):
    """300 bucket-0 users with >= 1 VALIDATION evaluation decision (deterministic)."""
    from trunckit import sample_users
    return sample_users(bucket0, 300, seed=0)


@pytest.fixture(scope="session")
def builder(sample_table):
    from trunckit import make_builder
    return make_builder(sample_table)


@pytest.fixture(scope="session")
def derived_panel(sample_table):
    from v4trunc.panel import derive_panel
    return derive_panel(sample_table, label="WARM_TRUNCATION")
