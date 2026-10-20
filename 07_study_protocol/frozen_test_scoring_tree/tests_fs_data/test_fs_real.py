"""FS-01 real-data tests (server, after the build): equivalence against MANIFEST_HASHES.json, pool membership and
top-key selection of FS, disjointness from every held-out band, the M01 / M02 assertions on the re-read FS tables,
the registry, and their strict-xfail negative controls. They read only data_fs/, data_v2 receipts / TRANSFORM_STATE
and the registry - never the v1 dataset."""
from __future__ import annotations

import copy
import json

import numpy as np

from conftest import DATA_FS, V4, real_only
from data_testkit import nc
from fs_testkit import load_json, m01_assertions, m02_assertions, read_tables
from census_v2 import core as C2
from v4data.common import VALIDATION_END_US, sha256_file
from v4fs_data import fs_common as FC

MH = V4 / "data_v2" / "receipts" / "MANIFEST_HASHES.json"
MDIR = DATA_FS / "manifests"
REG = V4 / "contracts" / "manifest_registry" / "REGISTRY_FS.json"


def _equiv():
    return load_json(DATA_FS / "receipts" / "EQUIVALENCE_M05_M06.json")


def _recheck_equivalence(rec: dict, mh: dict) -> list:
    """Independent of the receipt's own booleans: every in-verdict 'got' value must equal MANIFEST_HASHES."""
    bad = []
    for mid, r in rec["a_M05_M06_definitions_vs_MANIFEST_HASHES"].items():
        ref = mh["manifests"][mid]
        for k, v in r["fields"].items():
            if not v["in_verdict"]:
                continue
            if k.startswith("tables."):
                _, t, f = k.split(".")
                want = ref["tables"][t][f]
            else:
                want = ref[k]
            if v["got"] != want:
                bad.append((mid, k))
        need = {f"tables.{t}.table_content_sha256" for t in ref["tables"]} | {"header_content_sha256",
                                                                               "ordered_decision_id_sha256",
                                                                               "population_sha256", "n_decisions"}
        bad += [(mid, k, "missing") for k in need if k not in r["fields"]]
    for k, v in rec["b_FS_configuration_vs_M05_M06"].items():
        if not v["equal"]:
            bad.append(("b", k))
    for k, v in rec["c_server_files_rehashed"].items():
        mid, t = k.split(".")
        if v["table_content_sha256"] != mh["manifests"][mid]["tables"][t]["table_content_sha256"]:
            bad.append(("c", k))
    return bad


@real_only
def test_real_equivalence_M05_M06_exact():
    rec, mh = _equiv(), load_json(MH)
    assert FC.verify_self_hash(rec, "receipt_sha256")
    assert rec["full_build"] is True and rec["all_equal"] is True and rec["status"] == "PASS"
    assert rec["manifest_hashes_file_sha256"] == sha256_file(MH)
    assert rec["old_M05_population_sha256"] == FC.OLD_M05_POP_SHA and rec["old_M05_n_users"] == 65536
    assert not _recheck_equivalence(rec, mh)
    assert len(rec["c_server_files_rehashed"]) == 5


@real_only
@nc("a single changed MANIFEST_HASHES table hash must be detected by the re-check")
def test_nc_real_equivalence_detects_tampered_reference():
    rec, mh = _equiv(), copy.deepcopy(load_json(MH))
    mh["manifests"]["M06_CALIB_VALIDATION_WARM"]["tables"]["decisions"]["table_content_sha256"] = "0" * 64
    assert not _recheck_equivalence(rec, mh)


def _pool_keys() -> np.ndarray:
    b = (DATA_FS / "population" / "POOL_5_100_SELECTION_KEY_V2.u64be").read_bytes()
    return np.frombuffer(b, dtype=">u8").astype(np.uint64)


@real_only
def test_real_fs_users_are_the_top_keys_of_the_census_pool():
    p = DATA_FS / "population" / "POOL_5_100_SELECTION_KEY_V2.u64be"
    assert sha256_file(p) == FC.POOL_SHA_V2                      # byte-identical to the CENSUS-02 pool hash
    pk = _pool_keys()
    assert pk.size == FC.POOL_N and np.all(pk[1:] > pk[:-1])
    for k, ref in FC.PREFIX_SHA_V2.items():
        assert FC.population_sha256(pk[:k]) == ref
    t = read_tables(MDIR, FC.FS_TRAIN_ID)
    kv2 = np.sort(FC.keys_v2_of_hex(t["clients"]["client_key_hex"].astype(str)))
    assert kv2.size == FC.FS_N
    assert np.array_equal(kv2, pk[-FC.FS_N:])                     # all in the pool, and exactly its top 32,768
    assert np.all(np.isin(kv2, pk))
    n = t["clients"]["n_eligible_train_examples"]
    assert np.all((n >= C2.C_LO) & (n <= C2.C_HI))
    assert FC.population_sha256(kv2) == t["header"]["population_sha256_selection_key_v2"]
    assert sha256_file(DATA_FS / "population" / "FS_SELECTION_KEY_V2.u64be") == FC.population_sha256(kv2)


@real_only
def test_real_fs_disjoint_from_every_band():
    pk = _pool_keys()
    kv2 = np.sort(FC.keys_v2_of_hex(read_tables(MDIR, FC.FS_TRAIN_ID)["clients"]["client_key_hex"].astype(str)))
    fs_keys = set(kv2.tolist())
    ranks = np.searchsorted(pk, kv2)
    assert np.array_equal(pk[ranks], kv2)
    assert ranks[0] == FC.BANDS["FS"][0] and ranks[-1] == FC.BANDS["FS"][1] - 1 and ranks.size == FC.FS_N
    for name in FC.DISJOINT_FROM:
        x, y = FC.BANDS[name]
        assert not (fs_keys & set(pk[x:y].tolist())), name
        assert ranks[0] >= y or ranks[-1] < x, name
    assert FC.BANDS["U_NEW_TEST"][1] == FC.BANDS["FS"][0] and FC.BANDS["FS"][1] == FC.POOL_N


@real_only
@nc("the CALIB prefix is inside MAIN, so a disjointness check between them must fail")
def test_nc_real_disjointness_check_detects_overlap():
    pk = _pool_keys()
    x, y = FC.BANDS["MAIN"]
    assert not (set(pk[:32768].tolist()) & set(pk[x:y].tolist()))


@real_only
def test_real_m01_m02_assertions_and_hashes():
    t = read_tables(MDIR, FC.FS_TRAIN_ID)
    e = read_tables(MDIR, FC.FS_EVAL_ID)
    K = t["header"]["catalogue_K"]
    a1 = m01_assertions(t["clients"], t["decisions"], t["central_order"], K)
    a2 = m02_assertions(e["users"], e["decisions"], t["clients"], t["decisions"], K)
    assert all(a1.values()), a1
    assert all(a2.values()), a2
    for x, mid in ((t, FC.FS_TRAIN_ID), (e, FC.FS_EVAL_ID)):
        h = x["header"]
        assert h["manifest_id"] == mid and h["status"] == FC.FS_STATUS and h["run_label"] == FC.FS_STATUS
        assert h["header_content_sha256"] == FC.header_content_sha256(h)
        assert all(h["assertions"].values()) and all(c["match"] for c in h["expected_counts_check"])
        assert h["data_protocol_lock_sha256"] == FC.LOCK_SELF_SHA256
        assert h["data_protocol_lock_file_sha256"] == FC.LOCK_FILE_SHA256
        assert h["builder_code_sha256"] == FC.DATA04_BUILDER_CODE_SHA256
        assert h["code"]["fs_code_sha256"]
        assert h["all_methods_lock_sha256"] is None and h["test_exposure_register_entry_id"] is None
        for tb in h["tables"]:
            assert x[f"{tb['table']}.__content"] == tb["table_content_sha256"]
            assert x[f"{tb['table']}.__rows"] == tb["row_count"]
            assert sha256_file(MDIR / f"{mid}.{tb['table']}.parquet") == tb["file_sha256"]
        mk = load_json(MDIR / f"{mid}.OK")["verified_sha256"]
        assert all(sha256_file(DATA_FS / rel) == s for rel, s in mk.items()) and len(mk) == len(h["tables"]) + 1
    assert t["header"]["n_users"] == FC.FS_N and int(t["clients"]["in_calib_prefix"].sum()) == 0
    assert e["header"]["parent_manifests"] == [FC.FS_TRAIN_ID]
    assert np.all(e["decisions"]["decision_ts_us"] < VALIDATION_END_US)


@real_only
@nc("a changed target_class in the FS decisions must change the table content hash")
def test_nc_real_tampered_table_changes_content_hash():
    import pyarrow.parquet as pq
    from v4data.common import table_content_sha256
    from v4data.manifests import reread_columns
    tbl = pq.ParquetFile(MDIR / f"{FC.FS_TRAIN_ID}.decisions.parquet").read()
    h = load_json(MDIR / f"{FC.FS_TRAIN_ID}.header.json")
    tc = tbl["target_class"].to_numpy().copy()
    tc[0] = (tc[0] + 1) % h["catalogue_K"]
    import pyarrow as pa
    tbl = tbl.set_column(tbl.column_names.index("target_class"), "target_class", pa.array(tc))
    want = next(x for x in h["tables"] if x["table"] == "decisions")["table_content_sha256"]
    assert table_content_sha256(reread_columns(tbl)) == want


@real_only
def test_real_registry():
    reg = load_json(REG)
    assert FC.verify_self_hash(reg, "self_sha256")                 # REGISTRY_FS v2 key (DL 45)
    assert reg["data_version"] == FC.DATA_VERSION
    assert reg["roles"] == {"FEATURE_SEARCH_SCREEN": {"train": FC.FS_TRAIN_ID, "eval": FC.FS_EVAL_ID}}
    for mid, e in reg["manifests"].items():
        h = load_json(V4 / e["path"])
        assert sha256_file(V4 / e["path"]) == e["header_sha256"]
        assert e["content_sha256"] == FC.manifest_content_sha256(h["tables"])
        assert e["header_content_sha256"] == h["header_content_sha256"]
        for k in ("n_users", "n_decisions", "n_eval_rows", "n_rankable_rows", "population_sha256"):
            assert e[k] == h[k], k
        for t, x in e["tables"].items():
            assert sha256_file(V4 / x["file"]) == x["file_sha256"]
    assert reg["manifests"][FC.FS_EVAL_ID]["n_eval_rows"] > 0 and reg["manifests"][FC.FS_TRAIN_ID]["n_eval_rows"] == 0
    from v4data.transform_state import load_state
    st = load_state(V4 / reg["transform_state"]["path"])
    assert st["transform_state_sha256"] == reg["transform_state"]["transform_state_sha256"] == FC.TRANSFORM_STATE_SELF_SHA256
    assert sha256_file(V4 / reg["transform_state"]["path"]) == reg["transform_state"]["file_sha256"]
    assert reg["equivalence"]["all_equal"] is True
    assert sha256_file(V4 / reg["equivalence"]["receipt"]) == reg["equivalence"]["receipt_file_sha256"]


@real_only
@nc("a tampered registry entry must fail the registry self-hash")
def test_nc_real_registry_tamper():
    reg = load_json(REG)
    reg["manifests"][FC.FS_TRAIN_ID]["n_users"] += 1
    assert FC.verify_self_hash(reg, "self_sha256")                 # REGISTRY_FS v2 key (DL 45)


@real_only
def test_real_build_receipt():
    rec = load_json(DATA_FS / "receipts" / "FS_BUILD_RECEIPT.json")
    assert FC.verify_self_hash(rec, "receipt_sha256")
    assert rec["status"] == "PASS" and rec["full_build"] is True and rec["n_failed"] == 0
    assert all(c["pass"] for c in rec["checks"])
    names = {c["name"] for c in rec["checks"]}
    for need in ("pool.n_users", "pool.population_sha256_selection_key_v2", "fs.ranks",
                 "DATA04.universe_size", "DATA04.VALIDATION.censor.CEN_NO_NEXT"):
        assert need in names
    assert rec["test_split_read"] is False
    mh = load_json(DATA_FS / "receipts" / "MANIFEST_HASHES_FS.json")
    assert FC.verify_self_hash(mh, "receipt_sha256") and mh["manifests"] == rec["manifests"]
    json.dumps(rec, allow_nan=False)
