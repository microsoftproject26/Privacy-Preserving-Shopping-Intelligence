"""FL-01 test configuration: ppsi_env FIRST, strict FP32 and deterministic algorithms for every test, 1 CPU thread.

Negative controls are `xfail(strict=True)` tests: each runs a key-property check against a deliberately wrong variant
and MUST fail (an XPASS fails the suite). They are named test_nc_*.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
import ppsi_env  # noqa: E402,F401  (mandatory isolation, before any snapshot import)

import pytest  # noqa: E402
import torch  # noqa: E402

from fl.numerics import enforce_strict_fp32  # noqa: E402

THREADS = 1


@pytest.fixture(autouse=True)
def _strict_numerics():
    torch.set_num_threads(THREADS)
    enforce_strict_fp32()
    yield
    enforce_strict_fp32()

