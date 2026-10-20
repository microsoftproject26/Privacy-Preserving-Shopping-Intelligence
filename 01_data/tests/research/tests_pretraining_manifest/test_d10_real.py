"""DATA-10 real-data tests (server, after job_build.sh / the registry step): the build receipt and counts, the PRETRAIN
band against the DATA-06 / FS-01 / DATA-08 population files (disjoint from MAIN, CALIB, U_NEW_VAL, U_NEW_HOLDOUT, FS and
every DATA-06 panel file; PRETRAIN u MAIN = TIEBREAK; exact cover of the pool), TRAIN-only rows (column statistics, no
full materialisation), the registry extension rule on the real files and the provider verification path through the
PRETRAIN wrapper. Skipped where data_pretrain is absent."""
from __future__ import annotations

import copy
import json

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from conftest import DATA_PRETRAIN, V4, real_only
from data_testkit import nc
from fs_testkit import load_json
from v4data.common import TRAIN_END_US, TRAIN_START_US, sha256_file
from v4data10 import d10_common as D10
from v4data10 import d10_finalize as F
from v4data10 import d10_registry as R
from v4fs_data import fs_common as FC

MDIR = DATA_PRETRAIN / "manifests"
EXT = V4 / F.EXT_REL
BASE = V4 / F.BASE_REL
POP3 = V4 / "data_v3" / "population"
DISJOINT_FILES = ("MAIN", "CALIB", "U_NEW_VAL", "U_NEW_HOLDOUT", "SPARSE", "HEAVY", "TRUE_COLD", "COLD_H",
                  "WARM_TRUNCATION")


def _keys(path):
    return np.frombuffer(path.read_bytes(), dtype=">u8").astype(np.uint64)


def _pretrain():
    return _keys(DATA_PRETRAIN / "population" / "PRETRAIN_NONMAIN_SELECTION_KEY_V2.u64be")


def _pop(name):
    if name == "FS":
        return V4 / "data_fs" / "population" / "FS_SELECTION_KEY_V2.u64be"
    if name == "POOL":
        return POP3 / "POOL_5_100_SELECTION_KEY_V2.u64be"
    return POP3 / f"{name}_SELECTION_KEY_V2.u64be"


@real_only
def test_real_build_receipt_and_counts():
    rec = load_json(DATA_PRETRAIN / "receipts" / "DATA10_BUILD_RECEIPT.json")
    assert FC.verify_self_hash(rec, "receipt_sha256") and rec["status"] == "PASS" and rec["full_build"] is True
    assert rec["n_failed"] == 0 and rec["test_split_read"] is False
    e = rec["manifests"][D10.TRAIN_ID]
    assert e["n_users"] == D10.N_USERS == 1_217_610 and e["n_decisions"] == D10.N_DECISIONS == 24_089_824
    h = load_json(MDIR / f"{D10.TRAIN_ID}.header.json")
    assert h["status"] == D10.ROLE and h["header_content_sha256"] == FC.header_content_sha256(h)
    assert h["population_sha256_selection_key_v2"] == D10.PRETRAIN_SHA_V2 and all(h["assertions"].values())


@real_only
def test_real_population_file_is_the_measured_band():
    p = DATA_PRETRAIN / "population" / "PRETRAIN_NONMAIN_SELECTION_KEY_V2.u64be"
    assert p.stat().st_size == 8 * D10.N_USERS and sha256_file(p) == D10.PRETRAIN_SHA_V2


@real_only
def test_real_clients_equal_the_band_file():
    cli = pq.read_table(MDIR / f"{D10.TRAIN_ID}.clients.parquet", columns=["client_key_hex"])
    ck = np.sort(FC.keys_v2_of_hex(cli["client_key_hex"].to_pylist()))
    del cli
    assert np.array_equal(ck, _pretrain())


@real_only
@pytest.mark.parametrize("name", DISJOINT_FILES + ("FS",))
def test_real_band_disjoint_from_every_registered_population(name):
    assert np.intersect1d(_pretrain(), _keys(_pop(name))).size == 0, name


@real_only
def test_real_union_with_main_is_tiebreak_and_the_pool_is_covered_once():
    pre, main = _pretrain(), _keys(_pop("MAIN"))
    tb = np.sort(np.concatenate([pre, main]))
    assert tb.size == FC.BANDS["TIEBREAK"][1] and FC.population_sha256(tb) == D10.TB_SHA_V2
    union = np.sort(np.concatenate([tb] + [_keys(_pop(n)) for n in ("U_NEW_VAL", "U_NEW_HOLDOUT", "FS")]))
    assert np.array_equal(union, np.sort(_keys(_pop("POOL"))))


@nc("MAIN is outside the band: an overlap assertion between them must fail")
@real_only
def test_nc_real_main_overlaps_band():
    assert np.intersect1d(_pretrain(), _keys(_pop("MAIN"))).size > 0


@real_only
def test_real_decisions_are_train_period_only():
    t = pq.read_table(MDIR / f"{D10.TRAIN_ID}.decisions.parquet",
                      columns=["old_split", "target_ts_us", "decision_ts_us"])
    assert t.num_rows == D10.N_DECISIONS
    assert pc.unique(t["old_split"]).to_pylist() == ["TRAIN"]
    assert pc.max(t["target_ts_us"]).as_py() < TRAIN_END_US and pc.min(t["target_ts_us"]).as_py() >= 0
    assert pc.min(t["decision_ts_us"]).as_py() >= TRAIN_START_US and pc.max(t["decision_ts_us"]).as_py() < TRAIN_END_US


@real_only
def test_real_extension_registry_rule():
    from v4data08.d8_finalize import base_pin_checks
    ext, base = json.loads(EXT.read_text(encoding="utf-8")), json.loads(BASE.read_text(encoding="utf-8"))
    chk = R.verify_ext_registry(ext, base, sha256_file(BASE))
    assert chk["all_ok"], {k: v for k, v in chk.items() if not v}
    pins = base_pin_checks(V4)
    assert all(v is True for k, v in pins.items() if not k.startswith("_"))
    assert ext["manifests"][D10.TRAIN_ID]["header_sha256"] == sha256_file(MDIR / f"{D10.TRAIN_ID}.header.json")


@nc("a real extension with one REGISTRY_V2 entry changed must fail the rule")
@real_only
def test_nc_real_tampered_extension():
    ext, base = json.loads(EXT.read_text(encoding="utf-8")), json.loads(BASE.read_text(encoding="utf-8"))
    bad = copy.deepcopy(ext)
    bad["manifests"][D10.EVAL_ID]["n_decisions"] += 1
    assert R.verify_ext_registry(R.with_self(bad), base, sha256_file(BASE))["all_ok"]


@real_only
def test_real_provider_verification_through_the_pretrain_wrapper(monkeypatch):
    """v4int.verify.verify_manifest_by_registry (the providers' path) on both PRETRAIN manifests via the extension,
    under DATA_PROTOCOL_LOCK_V2, with the shim's wrapper 3 (REGISTRY_V2 verified by the original function)."""
    import v4int.verify as VF
    from v4int.pretrain_small import install_verify_wrapper
    monkeypatch.setattr(VF, "verify_registry", VF.verify_registry)
    install_verify_wrapper(EXT, BASE)
    for mid, kind, n in ((D10.TRAIN_ID, "TRAIN", D10.N_DECISIONS), (D10.EVAL_ID, "EVAL", None)):
        vm = VF.verify_manifest_by_registry(mid, EXT, VF.DataPaths.default(), root=V4, expect_kind=kind)
        assert all(v is True for v in vm.checks.values())
        assert vm.registry["self_sha256"] == json.loads(EXT.read_text(encoding="utf-8"))["self_sha256"]
        if n is not None:
            assert int(vm.canonical_decision_id.size) == n
        vm.columns = None
