"""SCHEMA-01 test configuration (tests_fs): ppsi_env FIRST, strict FP32 (TF32 off), deterministic algorithms, 1 CPU
thread. The test process plays the launcher: CUBLAS_WORKSPACE_CONFIG is exported before torch starts (the trainer's
STRICT mode refuses to start without it), as run_fs.export_environment does.

Negative controls are `xfail(strict=True, raises=AssertionError)` tests named test_nc_*.
Run directories live under code/tests_fs/_work/<test> and are removed after each test.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
HERE = Path(__file__).resolve().parent
CODE = HERE.parent
for _p in (str(CODE), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import ppsi_env  # noqa: E402,F401  (mandatory isolation, before any snapshot import)

import pytest  # noqa: E402
import torch  # noqa: E402

from fl.numerics import enforce_strict_fp32  # noqa: E402

THREADS = 1
WORK = HERE / "_work"


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
