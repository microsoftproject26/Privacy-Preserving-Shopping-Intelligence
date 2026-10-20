"""ANA-01 test configuration: the V4 code tree (code/) FIRST, then tools/ (retention_report). Synthetic blocks only,
all under pytest tmp dirs (run with --basetemp on F:, company/tasks/_COMMON_RULES_06B.md)."""
from __future__ import annotations

import os
import sys
from pathlib import Path
import ppsi_paths

sys.dont_write_bytecode = True
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
HERE = Path(__file__).resolve().parent
CODE = ppsi_paths.code_dir("v4train").parent          # the code folder that holds v4train
TOOLS = ppsi_paths.locate("tools/retention_report.py").parent
for _p in (str(HERE), str(TOOLS), str(CODE)):
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
import ppsi_env  # noqa: E402,F401  (mandatory isolation, before any other V4 import)

import pytest  # noqa: E402

import retention_report as R  # noqa: E402

R.bind(CODE)


@pytest.fixture(scope="session")
def lib():
    return R.LIB
