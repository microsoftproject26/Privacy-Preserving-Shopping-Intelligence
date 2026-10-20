"""SYS-03 test configuration: puts code/ on sys.path so `import v4sup...` works with no install
step. Test doubles for nvidia-smi and disk usage live in suptestkit.py (imported directly by the
tests that need them), mirroring code/tests_fl/fl_testkit.py's pattern.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import pytest  # noqa: E402


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws
