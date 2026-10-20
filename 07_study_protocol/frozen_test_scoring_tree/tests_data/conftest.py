"""DATA-04 test configuration: ppsi_env FIRST (snapshot isolation, T43), then shared fixtures.

Negative controls are strict-xfail tests named test_nc_* (data_testkit.nc): each runs a key-property check against a
deliberately wrong variant and MUST fail; an XPASS fails the suite.
Real-data fixtures use the frozen v1 copy (local byte-identical copy, or the server copy) and never read TEST rows.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
for p in (str(CODE), str(TESTS)):
    if p not in sys.path:
        sys.path.insert(0, p)
import ppsi_env  # noqa: E402,F401  (mandatory isolation, before any snapshot import)

import pytest  # noqa: E402

from v4data.common import load_schema  # noqa: E402
from v4data.transform import build_spec  # noqa: E402


@pytest.fixture(scope="session")
def schema():
    return load_schema()


@pytest.fixture(scope="session")
def spec(schema):
    return build_spec(schema)


@pytest.fixture(scope="session")
def fake_build(tmp_path_factory):
    """The builder run end-to-end on the synthetic frozen-layout dataset (one bucket, expectations off)."""
    from data_testkit import standard_users, write_dataset
    from v4data.manifests import ManifestBuilder
    users, cat = standard_users()
    root = tmp_path_factory.mktemp("fake_ds")
    write_dataset(root, users, cat)
    out = tmp_path_factory.mktemp("fake_out")
    b = ManifestBuilder(ds_root=root, out_root=out, buckets=[0], expect=False, log=lambda *_: None)
    b.pass_support()
    b.pass_aggregate()
    b.select_populations()
    b.pass_extract()
    man = b.assemble()
    hashes = b.write(man)
    return {"builder": b, "root": root, "out": out, "man": man, "hashes": hashes, "users": users, "catalogue": cat}


@pytest.fixture(scope="session")
def real_store():
    """Clean V2 store over the real frozen v1 bucket 0, TRAIN + VALIDATION_SELECT (never TEST)."""
    from data_testkit import REAL_DS
    from v4data.store import CleanV2Store
    return CleanV2Store.build(splits=("TRAIN", "VALIDATION_SELECT"), buckets=[0], ds_root=REAL_DS,
                              log=lambda *_: None)
