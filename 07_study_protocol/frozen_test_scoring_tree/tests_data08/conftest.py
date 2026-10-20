"""DATA-08 test configuration: ppsi_env FIRST (snapshot isolation, T43), then the shared fixtures.

Negative controls are strict-xfail tests named test_nc_* (data_testkit.nc): each runs a key-property check against a
deliberately wrong variant and MUST fail with AssertionError; an XPASS fails the suite.
Synthetic fixtures use the DATA-04 frozen-layout testkit (code/tests_data/data_testkit.py, read-only): 14 pool users
(5 <= n_eligible <= 100) with TRAIN, VALIDATION and TEST rows, plus the testkit's standard users (strata A0/A1/B, outside
the pool). Synthetic bands mirror the locked layout: TIEBREAK [0, 8), U_NEW_VAL [8, 10), U_NEW_TEST [10, 12),
FS [12, 14), MAIN [0, 4), CALIB [0, 2).
Real-data tests read only ppsi_v4/data_cfull, data_v3 / data_fs population files and the registries; they never open
the v1 dataset. They are skipped where data_cfull is absent (local host) or when D8_TESTS_SYNTHETIC_ONLY=1.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
for p in (str(CODE), str(TESTS), str(CODE / "tests_data"), str(CODE / "tests_fs_data")):
    if p not in sys.path:
        sys.path.insert(0, p)
import ppsi_env  # noqa: E402,F401  (mandatory isolation, before any snapshot import)

import pytest  # noqa: E402

V4 = CODE.parent
DATA_CFULL = V4 / "data_cfull"
REAL = (DATA_CFULL / "receipts" / "DATA08_BUILD_RECEIPT.json").is_file() \
    and os.environ.get("D8_TESTS_SYNTHETIC_ONLY") != "1"
real_only = pytest.mark.skipif(not REAL, reason="real DATA-08 build not present on this host (or synthetic-only gate)")

N_POOL = 14
SYN_BANDS = {"MAIN": (0, 4), "CALIB": (0, 2), "TIEBREAK": (0, 8), "U_NEW_VAL": (8, 10), "U_NEW_TEST": (10, 12),
             "FS": (12, 14)}
CATALOGUE = ["pA", "pB", "pC", "pD", "pE", "pFm", "pGo"]


def quiet(*_a, **_k):
    return None


def pool_users():
    """N_POOL users with 6..(6+i%4) eligible TRAIN examples (one TRAIN segment), a VALIDATION segment and TEST rows."""
    from data_testkit import make_user
    out = []
    for i in range(N_POOL):
        n_tr = 7 + i % 4                                       # 6..9 eligible (the last event has no target)
        day = 2 + i
        ev = [("TRAIN", f"2019-10-{day:02d}T10:{j:02d}:00", CATALOGUE[(i + j) % 5], 0) for j in range(n_tr)]
        ev += [("VALIDATION_SELECT", f"2019-11-17T{8 + i % 10:02d}:0{j}:00", CATALOGUE[(i + j + 1) % 5], 1)
               for j in range(3)]
        ev += [("TEST", f"2019-11-25T{8 + i % 10:02d}:0{j}:00", CATALOGUE[(i + j + 2) % 5], 2) for j in range(2)]
        out.append(make_user(f"pool-{i:02d}", ev))
    return out


@pytest.fixture(scope="session")
def synth_ds(tmp_path_factory):
    from data_testkit import standard_users, write_dataset
    std, cat = standard_users()
    assert cat == CATALOGUE
    root = tmp_path_factory.mktemp("d8_ds")
    write_dataset(root, pool_users() + std, cat)
    return root


@pytest.fixture(scope="session")
def synth_build(synth_ds, tmp_path_factory):
    """The DATA-08 build on the synthetic dataset (partial build: buckets [0], population constants not checked)."""
    from v4data08.d8_build import D8Builder, run_build
    out = tmp_path_factory.mktemp("d8_v4") / "data_cfull"
    b = D8Builder(ds_root=synth_ds, out_root=out, buckets=[0], expect=False, log=quiet)
    rec = run_build(b, out, bands=SYN_BANDS, write=True)
    return {"rec": rec, "out": out, "builder": b, "root": synth_ds}


@pytest.fixture(scope="session")
def synth_ref(synth_ds, tmp_path_factory):
    """The LOCKED DATA-04 builder end-to-end on the same synthetic dataset (reference M01 layout / schema)."""
    from v4data.manifests import ManifestBuilder
    out = tmp_path_factory.mktemp("d8_ref")
    ref = ManifestBuilder(ds_root=synth_ds, out_root=out, buckets=[0], expect=False, log=quiet)
    ref.pass_support()
    ref.pass_aggregate()
    ref.select_populations()
    ref.pass_extract()
    man = ref.assemble()
    ref.write(man)
    return {"out": out, "ref": ref}
