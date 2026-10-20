"""TRAIN-01 test configuration: ppsi_env FIRST, strict FP32 (TF32 off) and deterministic algorithms for every test,
1 CPU thread. Pattern of code/tests_fl/conftest.py.

Negative controls are `xfail(strict=True, raises=AssertionError)` tests named test_nc_*: each runs a key-property
check against a deliberately wrong variant and MUST fail (an XPASS fails the suite).
Run directories live under code/tests_train/_work/<test> and are removed after each test.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True
# The test process plays the launcher: CUBLAS_WORKSPACE_CONFIG is exported before torch / CUDA start, as on the server
# (the trainer's STRICT determinism mode refuses to start without it; TRAIN-01R1). run_all.py exports the T3 value.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
HERE = Path(__file__).resolve().parent
CODE = HERE.parent
for _p in (str(CODE), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import ppsi_env  # noqa: E402,F401  (mandatory isolation, before any snapshot import)

import pytest  # noqa: E402
import torch  # noqa: E402

import v4eval.frozen_config  # noqa: E402,F401  (EVAL-03's loader has landed: the real one is used, no stub)
from fl.numerics import enforce_strict_fp32  # noqa: E402

THREADS = 1
WORK = HERE / "_work"


def pytest_configure(config):
    """FIX-DELTA-01 F5: key p2kit's V4-internal P2 work dir by this session's --basetemp (process id without one)."""
    bt = config.getoption("basetemp", None)
    os.environ["PPSI_TESTS_TRAIN_BASETEMP"] = str(Path(bt).resolve()) if bt else f"pid{os.getpid()}"


@pytest.fixture(autouse=True)
def _strict_numerics():
    torch.set_num_threads(THREADS)
    enforce_strict_fp32()
    yield
    enforce_strict_fp32()


@pytest.fixture
def work(request):
    d = WORK / request.node.name.replace("[", "_").replace("]", "").replace("/", "_")
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)
    try:
        WORK.rmdir()
    except OSError:
        pass
