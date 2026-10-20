"""EVAL-02 test configuration: ppsi_env FIRST (snapshot isolation), no byte-code, strict FP32, 1 CPU thread.

The frozen T3 runtime flags (contracts/PREG1_LOCK_VALUES.json) are applied before every test.
Negative controls are `xfail(strict=True, raises=AssertionError)` tests named test_nc_*: each runs a key-property check
against a deliberately wrong variant and MUST fail; an XPASS fails the suite (pattern of code/tests_fl/fl_testkit.py).
All data are synthetic; no manifest, ledger or TEST row is read.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for p in (str(CODE), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
import ppsi_env  # noqa: E402,F401  (mandatory isolation, before any snapshot import)

import pytest  # noqa: E402
import torch  # noqa: E402

from fl.numerics import enforce_strict_fp32  # noqa: E402
from v4eval.frozen_config import apply_runtime_flags, load  # noqa: E402

_FROZEN = load()


def _frozen_flags():
    torch.set_num_threads(1)
    enforce_strict_fp32()
    apply_runtime_flags(_FROZEN)          # T3: deterministic algorithms with warn_only=False


_frozen_flags()                           # also before module-scoped fixtures


@pytest.fixture(autouse=True)
def _strict_numerics():
    _frozen_flags()
    yield
    _frozen_flags()                       # a test that changed a flag cannot leak it
