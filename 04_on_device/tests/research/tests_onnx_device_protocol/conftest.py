"""DEV-01 test configuration: ppsi_env FIRST, strict FP32 (TF32 off) and deterministic algorithms for every test,
1 CPU thread. Pattern of code/tests_models/conftest.py (and code/tests_fl/conftest.py).

Negative controls are `xfail(strict=True)` tests named test_nc_*: each asserts a deliberately WRONG property and
MUST fail with AssertionError against the real (correct) implementation (an XPASS fails the suite).
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
import torch  # noqa: E402

from fl.numerics import assert_strict_fp32, enforce_strict_fp32  # noqa: E402

THREADS = 1


@pytest.fixture(autouse=True)
def _strict_numerics():
    torch.set_num_threads(THREADS)
    enforce_strict_fp32()
    assert_strict_fp32()
    yield
    enforce_strict_fp32()
