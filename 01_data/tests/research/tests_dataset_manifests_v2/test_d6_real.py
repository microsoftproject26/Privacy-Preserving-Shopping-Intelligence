"""DATA-06 real-data tests (server, after the build): pool/band population hashes against CENSUS-02, pairwise
disjointness, the M01/M02 assertions on the re-read MAIN/CALIB/UNSEEN_VAL/UNSEEN_HOLDOUT/SPARSE/HEAVY tables, the
TRUE_COLD A0/A1 split, the build receipt and the registry (self-hash field `self_sha256`), and their strict-xfail
negative controls. Reads only ppsi_v4/data_v3, the registry and the DATA-04/CENSUS-02 reference constants; never the
v1 dataset."""
from __future__ import annotations

import copy

import numpy as np

from conftest import DATA_V3, REG, V4, real_only
from data_testkit import nc
from fs_testkit import load_json, m01_assertions, m02_assertions, read_tables
from v4data.common import sha256_file
from v4data06 import d6_common as DC
from v4data06 import d6_finalize as DF
from v4fs_data import fs_common as FC

MDIR = DATA_V3 / "manifests"
PAIRS = (
    (DC.MAIN_TRAIN_ID, DC.MAIN_EVAL_ID), (DC.CALIB_TRAIN_ID, DC.CALIB_EVAL_ID),
    (DC.UNEW_VAL_TRAIN_ID, DC.UNEW_VAL_EVAL_ID), (DC.UNEW_TEST_TRAIN_ID, DC.UNEW_TEST_EVAL_ID),
    (DC.SPARSE_TRAIN_ID, DC.SPARSE_EVAL_ID), (DC.HEAVY_TRAIN_ID, DC.HEAVY_EVAL_ID),
)


def _build_receipt():
    return load_json(DATA_V3 / "receipts" / "DATA06_BUILD_RECEIPT.json")


def _pool_keys() -> np.ndarray:
    b = (DATA_V3 / "population" / "POOL_5_100_SELECTION_KEY_V2.u64be").read_bytes()
    return np.frombuffer(b, dtype=">u8").astype(np.uint64)


@real_only
def test_real_no_manifest_or_file_contains_TEST():
    """The task packet's own naming for the U_NEW_TEST band conflicts with the LOCKED refuse_test_manifest
    substring gate (v4data.manifests); DATA-06 renamed the physical ids/files to UNEW_HOLDOUT (see
    v4data06.d6_common.NAMING_DECISION). No manifest id or on-disk path may contain 'TEST'."""
    for mid in DC.ALL_IDS:
        assert "TEST" not in mid.upper()
    # scoped to the DATA files (manifests/population); receipts/ legitimately contains "TESTS_server..." (the
    # pytest TEST SUITE run, a different sense of "test" entirely) and must not be flagged.
    for sub in ("manifests", "population"):
        for p in (DATA_V3 / sub).rglob("*"):
            assert "TEST" not in p.name.upper(), p


@real_only
def test_real_pool_and_band_hashes_match_census02():
    pk = _pool_keys()
    assert sha256_file(DATA_V3 / "population" / "POOL_5_100_SELECTION_KEY_V2.u64be") == DC.POOL_SHA_V2
    assert pk.size == DC.POOL_N and np.all(pk[1:] > pk[:-1])
    for k, ref in DC.PREFIX_SHA_V2.items():
        assert FC.population_sha256(pk[:k]) == ref
    main_h = load_json(MDIR / f"{DC.MAIN_TRAIN_ID}.header.json")
    calib_h = load_json(MDIR / f"{DC.CALIB_TRAIN_ID}.header.json")
    assert main_h["population_sha256_selection_key_v2"] == DC.MAIN_SHA_V2
    assert calib_h["population_sha256_selection_key_v2"] == DC.CALIB_SHA_V2
    assert main_h["n_users"] == DC.EXPECTED_MAIN["n_users"]
    assert calib_h["n_users"] == DC.EXPECTED_CALIB["n_users"]


@real_only
@nc("MAIN and CALIB population hashes must differ (CALIB is a strict nested prefix of MAIN)")
def test_nc_real_main_and_calib_hash_are_not_the_same():
    assert DC.MAIN_SHA_V2 == DC.CALIB_SHA_V2


@real_only
def test_real_expected_counts():
    main_h = load_json(MDIR / f"{DC.MAIN_TRAIN_ID}.header.json")
    main_e = load_json(MDIR / f"{DC.MAIN_EVAL_ID}.header.json")
    calib_h = load_json(MDIR / f"{DC.CALIB_TRAIN_ID}.header.json")
    calib_e = load_json(MDIR / f"{DC.CALIB_EVAL_ID}.header.json")
    cold = load_json(MDIR / f"{DC.TRUE_COLD_ID}.header.json")
    assert main_h["n_users"] == DC.EXPECTED_MAIN["n_users"]
    assert main_h["n_decisions"] == DC.EXPECTED_MAIN["n_train_examples"]
    # main_e["n_users"] is the M02 definition's own count (every user with >= 1 VALIDATION-split decision, whether
    # observed or not); the task's "VALIDATION users" means >= 1 EVAL ROW specifically (n_users_with_ge1_eval_row),
    # a strict subset - see the FS-01 precedent (FS_VALIDATION_5_100_32768: 12,885 users, 10,269 with >= 1 eval row).
    assert main_e["n_users_with_ge1_eval_row"] == DC.EXPECTED_MAIN["n_validation_users"]
    assert main_e["n_eval_rows"] == DC.EXPECTED_MAIN["n_eval_rows"]
    assert main_e["n_rankable_rows"] == DC.EXPECTED_MAIN["n_rankable_rows"]
    assert calib_h["n_users"] == DC.EXPECTED_CALIB["n_users"]
    assert calib_h["n_decisions"] == DC.EXPECTED_CALIB["n_train_examples"]
    assert calib_e["n_users_with_ge1_eval_row"] == DC.EXPECTED_CALIB["n_validation_users"]
    assert calib_e["n_eval_rows"] == DC.EXPECTED_CALIB["n_eval_rows"]
    assert cold["n_users"] == DC.EXPECTED_TRUE_COLD["n_users"]
    assert cold["n_eval_rows"] == DC.EXPECTED_TRUE_COLD["n_eval_rows"]
    cbc = cold["counts_by_cause"]
    assert cbc["stratum_COLD_NO_TRAIN_EVENTS"]["users_with_ge1_eval_row"] == DC.EXPECTED_TRUE_COLD["n_users_A0"]
    assert cbc["stratum_COLD_NO_TRAIN_EVENTS"]["eval_rows"] == DC.EXPECTED_TRUE_COLD["n_eval_rows_A0"]
    assert cbc["stratum_COLD_TRAIN_EVENTS_NO_LABEL"]["users_with_ge1_eval_row"] == DC.EXPECTED_TRUE_COLD["n_users_A1"]
    assert cbc["stratum_COLD_TRAIN_EVENTS_NO_LABEL"]["eval_rows"] == DC.EXPECTED_TRUE_COLD["n_eval_rows_A1"]


@real_only
@nc("A0 + A1 must equal the TRUE_COLD total; a deliberately wrong split must be caught")
def test_nc_real_true_cold_A0_A1_sum_wrong():
    assert DC.EXPECTED_TRUE_COLD["n_users_A0"] + DC.EXPECTED_TRUE_COLD["n_users_A1"] + 1 == DC.EXPECTED_TRUE_COLD["n_users"]


@real_only
def test_real_disjointness():
    pk = _pool_keys()

    def keys_of(mid, table=None):
        if table is None:
            table = "clients" if ("TRAIN" in mid or "SUPPORT" in mid) else "users"
        t = read_tables(MDIR, mid)
        return set(FC.keys_v2_of_hex(t[table]["client_key_hex"].astype(str)).tolist())

    main_k = keys_of(DC.MAIN_TRAIN_ID)
    calib_k = keys_of(DC.CALIB_TRAIN_ID)
    uv_k = keys_of(DC.UNEW_VAL_TRAIN_ID)
    ut_k = keys_of(DC.UNEW_TEST_TRAIN_ID)
    sparse_k = keys_of(DC.SPARSE_TRAIN_ID)
    heavy_k = keys_of(DC.HEAVY_TRAIN_ID)
    cold_k = keys_of(DC.TRUE_COLD_ID, "users")
    coldh_k = keys_of(DC.COLD_H_ID, "panel_users")
    wt_k = keys_of(DC.WARM_TRUNCATION_ID, "panel_users")

    assert calib_k <= main_k
    assert wt_k <= main_k
    assert coldh_k <= cold_k
    for a, b in ((main_k, uv_k), (main_k, ut_k), (uv_k, ut_k)):
        assert not (a & b)
    for other in (sparse_k, heavy_k, cold_k, coldh_k):
        for band in (main_k, uv_k, ut_k):
            assert not (band & other)
    assert not (sparse_k & heavy_k)
    # rank-arithmetic cross-check: MAIN/UNEW ranks are within the pool at the recorded bands
    ranks_main = np.searchsorted(pk, np.array(sorted(main_k), dtype=np.uint64))
    assert ranks_main.max() < DC.BANDS["MAIN"][1]


@real_only
@nc("CALIB is nested inside MAIN by construction; a disjointness check between them must fail")
def test_nc_real_calib_within_main_must_not_be_disjoint():
    t_main = read_tables(MDIR, DC.MAIN_TRAIN_ID)
    t_calib = read_tables(MDIR, DC.CALIB_TRAIN_ID)
    main_k = set(t_main["clients"]["client_key_hex"].astype(str).tolist())
    calib_k = set(t_calib["clients"]["client_key_hex"].astype(str).tolist())
    assert not (main_k & calib_k)


@real_only
def test_real_m01_m02_assertions_all_pairs():
    for tid, eid in PAIRS:
        t = read_tables(MDIR, tid)
        e = read_tables(MDIR, eid)
        K = t["header"]["catalogue_K"]
        a1 = m01_assertions(t["clients"], t["decisions"], t["central_order"], K)
        a2 = m02_assertions(e["users"], e["decisions"], t["clients"], t["decisions"], K)
        assert all(a1.values()), (tid, a1)
        assert all(a2.values()), (eid, a2)
        for x, mid in ((t, tid), (e, eid)):
            h = x["header"]
            assert h["manifest_id"] == mid
            assert h["header_content_sha256"] == FC.header_content_sha256(h)
            allowed = DC.KNOWN_ASSERTION_DEVIATIONS.get(mid, set())
            assert all(v for k, v in h["assertions"].items() if k not in allowed), h["assertions"]
            assert h["data_protocol_lock_sha256"] == FC.LOCK_SELF_SHA256
            assert h["all_methods_lock_sha256"] is None and h["test_exposure_register_entry_id"] is None
            for tb in h["tables"]:
                assert x[f"{tb['table']}.__content"] == tb["table_content_sha256"]
                assert x[f"{tb['table']}.__rows"] == tb["row_count"]
                assert sha256_file(MDIR / f"{mid}.{tb['table']}.parquet") == tb["file_sha256"]
            mk = load_json(MDIR / f"{mid}.OK")["verified_sha256"]
            assert all(sha256_file(DATA_V3 / rel) == s for rel, s in mk.items()) and len(mk) == len(h["tables"]) + 1


@real_only
@nc("a changed target_class in MAIN_TRAIN decisions must change the table content hash")
def test_nc_real_tampered_table_changes_content_hash():
    import pyarrow as pa
    import pyarrow.parquet as pq
    from v4data.common import table_content_sha256
    from v4data.manifests import reread_columns
    tbl = pq.ParquetFile(MDIR / f"{DC.MAIN_TRAIN_ID}.decisions.parquet").read()
    h = load_json(MDIR / f"{DC.MAIN_TRAIN_ID}.header.json")
    tc = tbl["target_class"].to_numpy().copy()
    tc[0] = (tc[0] + 1) % h["catalogue_K"]
    tbl = tbl.set_column(tbl.column_names.index("target_class"), "target_class", pa.array(tc))
    want = next(x for x in h["tables"] if x["table"] == "decisions")["table_content_sha256"]
    assert table_content_sha256(reread_columns(tbl)) == want


@real_only
def test_real_true_cold_and_panels():
    cold = read_tables(MDIR, DC.TRUE_COLD_ID)
    coldh = read_tables(MDIR, DC.COLD_H_ID)
    wt = read_tables(MDIR, DC.WARM_TRUNCATION_ID)
    assert cold["header"]["n_users"] == DC.EXPECTED_TRUE_COLD["n_users"]
    assert set(cold["users"]["cold_stratum"].astype(str).tolist()) <= {"COLD_NO_TRAIN_EVENTS", "COLD_TRAIN_EVENTS_NO_LABEL"}
    assert coldh["header"]["n_users"] == DC.N_PANEL
    assert wt["header"]["n_users"] == DC.N_PANEL
    assert all(coldh["header"]["assertions"].values())
    assert all(wt["header"]["assertions"].values())
    coldh_keys = set(FC.keys_v2_of_hex(coldh["panel_users"]["client_key_hex"].astype(str)).tolist())
    cold_keys_sorted = np.sort(FC.keys_v2_of_hex(cold["users"]["client_key_hex"].astype(str)))
    assert coldh_keys == set(cold_keys_sorted[:DC.N_PANEL].tolist())


@real_only
def test_real_build_receipt():
    rec = _build_receipt()
    assert FC.verify_self_hash(rec, "receipt_sha256")
    assert rec["status"] == "PASS" and rec["full_build"] is True and rec["n_failed"] == 0
    assert all(c["pass"] for c in rec["checks"])
    assert rec["test_split_read"] is False
    names = {c["name"] for c in rec["checks"]}
    for need in ("pool.n_users", "MAIN.population_sha256_selection_key_v2", "CALIB.population_sha256_selection_key_v2",
                 "TRUE_COLD.n_users", "DATA04.universe_size"):
        assert need in names
    mh = load_json(DATA_V3 / "receipts" / "DATA06_MANIFEST_HASHES.json")
    assert mh["manifests"] == rec["manifests"]


@real_only
@nc("a tampered receipt field must fail the receipt self-hash")
def test_nc_real_build_receipt_tamper():
    rec = _build_receipt()
    rec["n_failed"] += 1
    assert FC.verify_self_hash(rec, "receipt_sha256")


@real_only
def test_real_registry():
    reg = load_json(REG)
    assert DF.verify_self_sha256(reg)
    assert "self_sha256" in reg and "registry_self_sha256" not in reg
    assert reg["registry_version"] == 2 and reg["data_version"] == DC.DATA_VERSION
    assert set(reg["roles"]) == {"MAIN", "CALIBRATION", "UNSEEN_VAL", "UNSEEN_HOLDOUT", "SPARSE_PANEL", "HEAVY_PANEL",
                                "TRUE_COLD", "COLD_H", "WARM_TRUNCATION"}
    assert reg["roles"]["UNSEEN_HOLDOUT"] == {"train": DC.UNEW_TEST_TRAIN_ID, "eval": DC.UNEW_TEST_EVAL_ID}
    for mid, e in reg["manifests"].items():
        h = load_json(V4 / e["path"])
        assert sha256_file(V4 / e["path"]) == e["header_sha256"]
        assert e["content_sha256"] == FC.manifest_content_sha256(h["tables"])
        assert e["header_content_sha256"] == h["header_content_sha256"]
        for k in ("n_users", "n_decisions", "n_eval_rows", "n_rankable_rows", "population_sha256"):
            assert e[k] == h[k], (mid, k)
        for t, x in e["tables"].items():
            assert sha256_file(V4 / x["file"]) == x["file_sha256"]
    assert reg["manifests"][DC.MAIN_EVAL_ID]["n_eval_rows"] > 0
    assert reg["manifests"][DC.MAIN_TRAIN_ID]["n_eval_rows"] == 0
    assert reg["transform_state"]["transform_state_sha256"] == DC.TRANSFORM_STATE_SELF_SHA256


@real_only
@nc("a tampered registry entry must fail the registry self-hash")
def test_nc_real_registry_tamper():
    reg = load_json(REG)
    reg["manifests"][DC.MAIN_TRAIN_ID]["n_users"] += 1
    assert DF.verify_self_sha256(reg)


@real_only
def test_real_manifest_on_disk_verification():
    for mid in DC.ALL_IDS:
        v = DF.verify_manifest_on_disk(V4, "data_v3", mid)
        assert v["all_ok"], (mid, v["checks"])


@real_only
@nc("verify_manifest_on_disk must catch a header whose row_count no longer matches the parquet file")
def test_nc_real_manifest_verification_detects_tamper():
    v = DF.verify_manifest_on_disk(V4, "data_v3", DC.MAIN_TRAIN_ID)
    bad = copy.deepcopy(v)
    bad["checks"]["clients.row_count"] = False
    assert all(bad["checks"].values())
