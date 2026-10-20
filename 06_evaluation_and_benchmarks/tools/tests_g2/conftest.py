"""G2-01 test configuration: the V4 code tree (code/) FIRST, then code/tests_train (the TRAIN-05 / TRAIN-03 fixtures),
then tools/ (g2_run). Synthetic data only; the P2 stage evidence (which the tuned-recipes registry stores V4-relative)
lives under tools/tests_g2/_work and is removed at the end of the session; everything else is in pytest tmp dirs."""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent
V4 = TOOLS.parent
CODE = V4 / "code"
for _p in (str(TOOLS), str(CODE / "tests_train"), str(CODE)):
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
import ppsi_env  # noqa: E402,F401  (mandatory isolation, before any other V4 import)

import pytest  # noqa: E402

WORK = HERE / "_work"


@pytest.fixture(scope="session", autouse=True)
def _clean_work():
    shutil.rmtree(WORK, ignore_errors=True)
    yield
    shutil.rmtree(WORK, ignore_errors=True)
