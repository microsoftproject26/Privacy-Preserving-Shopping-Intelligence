"""DATA-06 test configuration: ppsi_env FIRST (snapshot isolation, T43), then the shared fixtures.

Negative controls are strict-xfail tests named test_nc_* (data_testkit.nc): each runs a key-property check against a
deliberately wrong variant and MUST fail with AssertionError; an XPASS fails the suite.
Reuses the FS-01 test helpers (code/tests_fs_data/fs_testkit.py: read_tables, load_json, m01_assertions,
m02_assertions - an independent re-implementation of the MANIFEST_SPEC M01/M02 assertions, read-only) and the
DATA-04 negative-control marker (code/tests_data/data_testkit.py: nc). Real-data tests read only ppsi_v4/data_v3 and
the registry; they never open the v1 dataset, so no TEST row can be read. They are skipped where data_v3 is absent
(local host) or when D6_TESTS_SYNTHETIC_ONLY=1 (unused here; kept for symmetry with FS-01's pre-build gate).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import ppsi_paths

sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
for p in (str(CODE), str(TESTS), str(CODE / "tests_data"), str(CODE / "tests_fs_data")):
    if p not in sys.path:
        sys.path.insert(0, p)
import ppsi_env  # noqa: E402,F401  (mandatory isolation, before any snapshot import)

import pytest  # noqa: E402

V4 = ppsi_paths.PROJECT_ROOT.resolve()   # the ppsi_v4 workspace (contracts, data)
DATA_V3 = V4 / "data_v3"
REG = V4 / "contracts" / "manifest_registry" / "REGISTRY_V2.json"
REAL = (DATA_V3 / "receipts" / "DATA06_BUILD_RECEIPT.json").is_file() and os.environ.get("D6_TESTS_SYNTHETIC_ONLY") != "1"
real_only = pytest.mark.skipif(not REAL, reason="real DATA-06 build not present on this host (or synthetic-only gate)")
