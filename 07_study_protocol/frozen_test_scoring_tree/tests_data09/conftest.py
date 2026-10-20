"""DATA-09 test configuration: ppsi_env FIRST (snapshot isolation, T43), then the shared fixtures.

HARD RULE (DATA-09 packet): every test is SYNTHETIC. No test opens the frozen v1 dataset (local or server) or any real
RETRO_TEST row / label / feature / metric; the real files touched are only contracts/ (read), the real
TEST_EXPOSURE_REGISTER (read, to prove the real gate is CLOSED) and code. Real-data tests of the server build are in
test_d9_real.py and are skipped unless data_test/receipts/DATA09_BUILD_RECEIPT.json exists AND the real gate is open.

Negative controls are strict-xfail tests named test_nc_* (data_testkit.nc): each runs a key-property check against a
deliberately wrong variant and MUST fail with AssertionError; an XPASS fails the suite.

Synthetic dataset (one bucket, the DATA-04 frozen layout via code/tests_data/data_testkit.py, read-only): 14 pool users
(stratum C) with TRAIN / VALIDATION / TEST rows whose TEST tails cover every RETRO_TEST status and censor class
(NO_OBSERVED_NEXT, OBSERVED_MATURED_IN_GRACE incl. an OOV grace target, AMBIGUOUS_BOUNDARY -> LABEL_GRACE,
BOUNDARY_UNRESOLVED -> LABEL_GRACE, an in-partition OOV target), stratum-B / stratum-D panel users (one of each without
a TEST row), cold users (A0 / A1 with TEST eval rows, a TEST-only user, a TEST-only user without an eval row, a cold user
with censored TEST rows only) and the testkit's standard users. Synthetic bands: MAIN [0, 4), CALIB [0, 2),
U_NEW_VAL [8, 10), U_NEW_TEST [10, 12); panels of 2 (monkeypatched v4data06.d6_common constants, restored after use).
Synthetic TEST gate: a tmp ALL_METHODS_LOCK.json + a tmp hash-chained register (STRUCTURAL_COUNT, MANIFEST_BUILD).
"""
from __future__ import annotations

import contextlib
import hashlib
import json
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
DATA_TEST = V4 / "data_test"
CATALOGUE = ["pA", "pB", "pC", "pD", "pE", "pFm", "pGo"]
SYN = {"N_MAIN": 4, "N_CALIB": 2, "N_PANEL": 2,
       "BANDS": {"MAIN": (0, 4), "CALIB": (0, 2), "U_NEW_VAL": (8, 10), "U_NEW_TEST": (10, 12)}}
GRACE_TS = "2019-11-30T05:00:00"


def quiet(*_a, **_k):
    return None


def real_gate_open() -> bool:
    try:
        from v4data09.d9_gate import check_test_gate
        check_test_gate()
        return True
    except Exception:
        return False


REAL = (DATA_TEST / "receipts" / "DATA09_BUILD_RECEIPT.json").is_file() \
    and os.environ.get("D9_TESTS_SYNTHETIC_ONLY") != "1" and real_gate_open()
real_only = pytest.mark.skipif(not REAL, reason="real DATA-09 build absent or the real TEST gate closed (local host)")
# tests that prove the REAL gate refuses: they run wherever the real gate is closed (every host before the lock) and are
# skipped on the server after ALL_METHODS_LOCK + the MANIFEST_BUILD entry exist (the look itself)
REAL_GATE_OPEN = real_gate_open()
closed_gate_only = pytest.mark.skipif(REAL_GATE_OPEN, reason="the real TEST gate is open on this host (post-lock)")


# ------------------------------------------------------------------------------------------------ synthetic users
def _pool_users():
    from data_testkit import _ts, make_user
    out = []
    for i in range(14):
        n_tr = 7 + i % 4
        day = 2 + i
        ev = [("TRAIN", f"2019-10-{day:02d}T10:{j:02d}:00", CATALOGUE[(i + j) % 5], 0) for j in range(n_tr)]
        ev += [("VALIDATION_SELECT", f"2019-11-17T{8 + i % 10:02d}:0{j}:00", CATALOGUE[(i + j + 1) % 5], 1)
               for j in range(3)]
        tails, over = {}, {}
        if i != 13:                                          # pool-13: no RETRO_TEST row at all
            prods = [CATALOGUE[(i + j + 2) % 5] for j in range(3)]
            if i == 6:
                prods[1] = "rare-t"                          # an in-partition OOV target (support 0)
            ev += [("TEST", f"2019-11-25T{8 + i % 10:02d}:0{j}:00", prods[j], 2) for j in range(3)]
            tail = len(ev) - 1
            k = i % 5
            if k in (1, 4):
                over[tail] = {"target_product_id": "pB" if k == 1 else "rare-g", "target_ts": _ts(GRACE_TS),
                              "target_status": "OBSERVED_MATURED_IN_GRACE", "next_split": "LABEL_GRACE",
                              "next_block_type": "SINGLE_ITEM"}
            elif k == 2:
                tails[2] = ("AMBIGUOUS_BOUNDARY", "LABEL_GRACE", "AMBIGUOUS_TIME_BLOCK_BARRIER")
            elif k == 3:
                tails[2] = ("BOUNDARY_UNRESOLVED", "LABEL_GRACE", "SINGLE_ITEM")
        out.append(make_user(f"pool-{i:02d}", ev, tails=tails, overrides=over))
    return out


def _panel_and_cold_users():
    from data_testkit import _ts, make_user
    out = []
    for i in range(4):                                       # stratum B (1 eligible); sparse-3 has no TEST row
        ev = [("TRAIN", f"2019-10-2{i}T09:00:00", "pA", 0), ("TRAIN", f"2019-10-2{i}T09:05:00", "pB", 0),
              ("VALIDATION_CONFIRM", f"2019-11-21T1{i}:00:00", "pC", 1),
              ("VALIDATION_CONFIRM", f"2019-11-21T1{i}:05:00", "pD", 1)]
        if i != 3:
            ev += [("TEST", f"2019-11-26T1{i}:00:00", "pE", 2), ("TEST", f"2019-11-26T1{i}:05:00", "pA", 2)]
        out.append(make_user(f"sparse-{i}", ev))
    for i in range(3):                                       # stratum D (> 100 eligible); heavy-2 has no TEST row
        ev = [("TRAIN", f"2019-10-{3 + i:02d}T{8 + j // 60:02d}:{j % 60:02d}:00", CATALOGUE[j % 5], 0) for j in range(103)]
        ev += [("VALIDATION_SELECT", f"2019-11-18T1{i}:00:00", "pA", 1),
               ("VALIDATION_SELECT", f"2019-11-18T1{i}:05:00", "pB", 1)]
        if i != 2:
            ev += [("TEST", f"2019-11-27T1{i}:00:00", "pC", 2), ("TEST", f"2019-11-27T1{i}:05:00", "pD", 2),
                   ("TEST", f"2019-11-27T1{i}:10:00", "pE", 2)]
        out.append(make_user(f"heavy-{i}", ev))
    out.append(make_user("cold-a0-t", [
        ("VALIDATION_SELECT", "2019-11-18T12:00:00", "pA", 0), ("VALIDATION_SELECT", "2019-11-18T12:03:00", "pB", 0),
        ("TEST", "2019-11-24T12:00:00", "pC", 1), ("TEST", "2019-11-24T12:03:00", "pD", 1),
        ("TEST", "2019-11-24T12:06:00", "pE", 1)],
        overrides={4: {"target_product_id": "pA", "target_ts": _ts(GRACE_TS), "target_status": "OBSERVED_MATURED_IN_GRACE",
                       "next_split": "LABEL_GRACE", "next_block_type": "SINGLE_ITEM"}}))
    out.append(make_user("cold-a1-t", [
        ("TRAIN", "2019-10-11T08:00:00", "pA", 0),
        ("TEST", "2019-11-28T08:00:00", "pB", 1), ("TEST", "2019-11-28T08:05:00", "pC", 1),
        ("TEST", "2019-11-29T20:00:00", "pD", 2)],                  # BU -> LABEL_GRACE: CEN_RIGHT_EDGE
        tails={2: ("BOUNDARY_UNRESOLVED", "LABEL_GRACE", "SINGLE_ITEM")}))
    out.append(make_user("testonly-0", [("TEST", "2019-11-29T08:00:00", "pA", 0), ("TEST", "2019-11-29T08:05:00", "pB", 0)]))
    out.append(make_user("testonly-1", [("TEST", "2019-11-29T09:00:00", "pC", 0)]))
    out.append(make_user("cold-censored", [
        ("TEST", "2019-11-24T15:00:00", "pA", 0), ("TEST", "2019-11-25T15:00:00", "pB", 1)],
        tails={0: ("AMBIGUOUS_BOUNDARY", "TEST", "AMBIGUOUS_TIME_BLOCK_BARRIER")}))
    return out


def synthetic_users():
    from data_testkit import standard_users
    std, cat = standard_users()
    assert cat == CATALOGUE
    return _pool_users() + _panel_and_cold_users() + std


def write_synthetic_dataset(root: Path) -> Path:
    from data_testkit import write_dataset
    return write_dataset(Path(root), synthetic_users(), CATALOGUE)


@contextlib.contextmanager
def synthetic_bands():
    """The synthetic band sizes for v4data06.d6_build.compute_bands (unedited), restored on exit."""
    from v4data06 import d6_common as DC
    saved = {k: getattr(DC, k) for k in ("N_MAIN", "N_CALIB", "N_PANEL", "BANDS")}
    try:
        DC.N_MAIN, DC.N_CALIB, DC.N_PANEL = SYN["N_MAIN"], SYN["N_CALIB"], SYN["N_PANEL"]
        DC.BANDS = dict(saved["BANDS"], **SYN["BANDS"])
        yield
    finally:
        for k, v in saved.items():
            setattr(DC, k, v)


# ------------------------------------------------------------------------------------------------ synthetic gate
def _canon(o) -> bytes:
    return json.dumps(o, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode()


FIELDS = ("actor_role", "source", "split", "access_purpose", "observations", "observations_description", "decision",
          "all_methods_lock_sha256", "manifest_header_sha256", "model_checkpoint_sha256", "code_sha256", "receipt_path")


def make_register(path: Path, specs: list) -> Path:
    """A hash-chained register (the tools/test_exposure_register.py entry and chain rule) from partial entry specs."""
    entries, prev = [], None
    for i, s in enumerate(specs, 1):
        e = {"entry_id": i, "utc": "2026-10-01T00:00:00Z"}
        base = {"actor_role": "OPERATIONS_DIRECTOR (synthetic)", "source": "synthetic", "split": "RETRO_TEST",
                "access_purpose": "STRUCTURAL_COUNT", "observations": "NONE", "observations_description": "synthetic",
                "decision": "NO_FEEDBACK", "all_methods_lock_sha256": None, "manifest_header_sha256": None,
                "model_checkpoint_sha256": [], "code_sha256": [], "receipt_path": "synthetic"}
        base.update(s)
        e.update({k: base[k] for k in FIELDS})
        e["prev_entry_sha256"] = prev
        e["entry_sha256"] = hashlib.sha256(_canon(e)).hexdigest()
        prev = e["entry_sha256"]
        entries.append(e)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"register_id": "TEST_EXPOSURE_REGISTER", "spec_version": "MANIFEST_SPEC_V4 1.2.0",
                                      "entries": entries}, indent=1), encoding="utf-8")
    return Path(path)


def make_gate_files(d: Path, *, purpose: str = "MANIFEST_BUILD", lock_text: str = '{"document": "ALL_METHODS_LOCK"}',
                    extra_specs: list = ()) -> tuple:
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    lock = d / "ALL_METHODS_LOCK.json"
    lock.write_text(lock_text, encoding="utf-8")
    sha = hashlib.sha256(lock.read_bytes()).hexdigest()
    reg = make_register(d / "TEST_EXPOSURE_REGISTER.json",
                        [{"access_purpose": "STRUCTURAL_COUNT", "observations": "COUNTS_ONLY"},
                         {"access_purpose": purpose, "all_methods_lock_sha256": sha}] + list(extra_specs))
    return lock, reg


@pytest.fixture(scope="session")
def gate_files(tmp_path_factory):
    return make_gate_files(tmp_path_factory.mktemp("d9_gate"))


@pytest.fixture(scope="session")
def synth_gate(gate_files):
    from v4data09.d9_gate import check_test_gate
    return check_test_gate(*gate_files)


@pytest.fixture(scope="session")
def synth_ds(tmp_path_factory):
    return write_synthetic_dataset(tmp_path_factory.mktemp("d9_ds"))


def run_synthetic_build(ds: Path, out: Path, gate, write: bool = True) -> dict:
    from v4data09.d9_build import D9Builder, run_build
    with synthetic_bands():
        b = D9Builder(ds_root=ds, out_root=out, buckets=[0], expect=False, log=quiet, gate=gate)
        rec = run_build(b, out, write=write)
    return {"rec": rec, "out": out, "builder": b, "root": ds}


@pytest.fixture(scope="session")
def synth_build(synth_ds, synth_gate, tmp_path_factory):
    out = tmp_path_factory.mktemp("d9_v4") / "data_test"
    return run_synthetic_build(synth_ds, out, synth_gate)
