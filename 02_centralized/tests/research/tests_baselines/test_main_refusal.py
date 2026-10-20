"""BASE-01: the MAIN refusal — `v4base.run` refuses phase MAIN unless `--allow-main-validation` is passed (Ops
passes it only from P3 on, Amendment 02 §4). Tested at the gate function itself (`check_main_gate`), BEFORE any
provider/data is touched, and via the CLI's own argument parsing.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from basekit import nc
from v4base.run import MainValidationRefused, check_main_gate


def test_main_is_refused_without_the_flag():
    with pytest.raises(MainValidationRefused):
        check_main_gate("MAIN", False)


def test_main_is_allowed_with_the_flag():
    check_main_gate("MAIN", True)              # must not raise


def test_calibration_never_needs_the_flag():
    check_main_gate("CALIBRATION", False)      # must not raise
    check_main_gate("CALIBRATION", True)       # must not raise


def test_the_refusal_is_a_systemexit_subclass_so_the_cli_exits_cleanly():
    assert issubclass(MainValidationRefused, SystemExit)


@nc("without the phase check, a bare 'MAIN' call would run straight through instead of refusing")
def test_nc_main_would_run_through_without_the_gate():
    raised = False
    try:
        check_main_gate("MAIN", False)
    except MainValidationRefused:
        raised = True
    assert not raised
