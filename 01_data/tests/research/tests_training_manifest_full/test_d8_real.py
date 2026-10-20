"""DATA-08 real-data tests (server, after job_build.sh / the registry step): the build receipt and counts, the TB band
against the DATA-06 / FS-01 population files (exclusion of U_NEW_VAL / U_NEW_HOLDOUT / FS, exact cover of the pool),
TRAIN-only rows (column statistics, no full materialisation), the registry extension rule on the real files and the
provider verification path through the C_FULL wrapper. Skipped where data_cfull is absent."""
from __future__ import annotations

import copy
import json

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from conftest import DATA_CFULL, V4, real_only
from data_testkit import nc
from fs_testkit import load_json
from v4data.common import TRAIN_END_US, TRAIN_START_US, sha256_file
from v4data08 import d8_common as D8
from v4data08 import d8_finalize as F
from v4data08 import d8_registry as R
from v4fs_data import fs_common as FC

MDIR = DATA_CFULL / "manifests"
EXT = V4 / F.EXT_REL
BASE = V4 / F.BASE_REL


def _keys(path):
    return np.frombuffer(path.read_bytes(), dtype=">u8").astype(np.uint64)


def _pop(name):
    return {"TB": DATA_CFULL / "population" / "TIEBREAK_SELECTION_KEY_V2.u64be",
            "POOL": V4 / "data_v3" / "population" / "POOL_5_100_SELECTION_KEY_V2.u64be",
            "MAIN": V4 / "data_v3" / "population" / "MAIN_SELECTION_KEY_V2.u64be",
            "U_NEW_VAL": V4 / "data_v3" / "population" / "U_NEW_VAL_SELECTION_KEY_V2.u64be",
            "U_NEW_HOLDOUT": V4 / "data_v3" / "population" / "U_NEW_HOLDOUT_SELECTION_KEY_V2.u64be",
            "FS": V4 / "data_fs" / "population" / "FS_SELECTION_KEY_V2.u64be"}[name]


@real_only
def test_real_build_receipt_and_counts():
    rec = load_json(DATA_CFULL / "receipts" / "DATA08_BUILD_RECEIPT.json")
    assert FC.verify_self_hash(rec, "receipt_sha256") and rec["status"] == "PASS" and rec["full_build"] is True
    assert rec["n_failed"] == 0 and rec["test_split_read"] is False
    e = rec["manifests"][D8.TRAIN_ID]
    assert e["n_users"] == D8.TB_N_USERS == 1_348_682 and e["n_decisions"] == D8.TB_N_DECISIONS == 26_677_600
    h = load_json(MDIR / f"{D8.TRAIN_ID}.header.json")
    assert h["status"] == D8.ROLE and h["header_content_sha256"] == FC.header_content_sha256(h)
    assert h["population_sha256_selection_key_v2"] == D8.TB_SHA_V2 and all(h["assertions"].values())


@real_only
def test_real_population_file_is_the_locked_tb_band():
    p = _pop("TB")
    assert p.stat().st_size == 8 * D8.TB_N_USERS and sha256_file(p) == D8.TB_SHA_V2


@real_only
def test_real_clients_equal_tb_and_tb_excludes_unew_and_fs():
    tb = _keys(_pop("TB"))
    cli = pq.read_table(MDIR / f"{D8.TRAIN_ID}.clients.parquet", columns=["client_key_hex"])
    ck = np.sort(FC.keys_v2_of_hex(cli["client_key_hex"].to_pylist()))
    del cli
    assert np.array_equal(ck, tb)
    for name in ("U_NEW_VAL", "U_NEW_HOLDOUT", "FS"):
        assert np.intersect1d(tb, _keys(_pop(name))).size == 0, name
    assert np.isin(_keys(_pop("MAIN")), tb).all()
    union = np.sort(np.concatenate([tb] + [_keys(_pop(n)) for n in ("U_NEW_VAL", "U_NEW_HOLDOUT", "FS")]))
    assert np.array_equal(union, np.sort(_keys(_pop("POOL"))))


@nc("MAIN is inside TB: a disjointness assertion between them must fail")
@real_only
def test_nc_real_main_not_disjoint_from_tb():
    assert np.intersect1d(_keys(_pop("TB")), _keys(_pop("MAIN"))).size == 0


@real_only
def test_real_decisions_are_train_period_only():
    t = pq.read_table(MDIR / f"{D8.TRAIN_ID}.decisions.parquet",
                      columns=["old_split", "target_ts_us", "decision_ts_us"])
    assert t.num_rows == D8.TB_N_DECISIONS
    assert pc.unique(t["old_split"]).to_pylist() == ["TRAIN"]
    assert pc.max(t["target_ts_us"]).as_py() < TRAIN_END_US and pc.min(t["target_ts_us"]).as_py() >= 0
    assert pc.min(t["decision_ts_us"]).as_py() >= TRAIN_START_US and pc.max(t["decision_ts_us"]).as_py() < TRAIN_END_US


@real_only
def test_real_extension_registry_rule():
    ext, base = json.loads(EXT.read_text(encoding="utf-8")), json.loads(BASE.read_text(encoding="utf-8"))
    chk = R.verify_ext_registry(ext, base, sha256_file(BASE))
    assert chk["all_ok"], {k: v for k, v in chk.items() if not v}
    pins = F.base_pin_checks(V4)
    assert all(v is True for k, v in pins.items() if not k.startswith("_"))
    assert ext["manifests"][D8.TRAIN_ID]["header_sha256"] == sha256_file(MDIR / f"{D8.TRAIN_ID}.header.json")


@nc("a real extension with one REGISTRY_V2 entry changed must fail the rule")
@real_only
def test_nc_real_tampered_extension():
    ext, base = json.loads(EXT.read_text(encoding="utf-8")), json.loads(BASE.read_text(encoding="utf-8"))
    bad = copy.deepcopy(ext)
    bad["manifests"][D8.EVAL_ID]["n_decisions"] += 1
    assert R.verify_ext_registry(R.with_self(bad), base, sha256_file(BASE))["all_ok"]


@real_only
def test_real_provider_verification_through_the_c_full_wrapper(monkeypatch):
    """v4int.verify.verify_manifest_by_registry (the providers' path) on both C_FULL manifests via the extension,
    under DATA_PROTOCOL_LOCK_V2, with the shim's wrapper 3 (REGISTRY_V2 verified by the original function)."""
    import v4int.verify as VF
    from v4int.c_full import install_verify_wrapper
    monkeypatch.setattr(VF, "verify_registry", VF.verify_registry)
    install_verify_wrapper(EXT, BASE)
    for mid, kind, n in ((D8.TRAIN_ID, "TRAIN", D8.TB_N_DECISIONS), (D8.EVAL_ID, "EVAL", None)):
        vm = VF.verify_manifest_by_registry(mid, EXT, VF.DataPaths.default(), root=V4, expect_kind=kind)
        assert all(v is True for v in vm.checks.values())
        assert vm.registry["self_sha256"] == json.loads(EXT.read_text(encoding="utf-8"))["self_sha256"]
        if n is not None:
            assert int(vm.canonical_decision_id.size) == n
        vm.columns = None
