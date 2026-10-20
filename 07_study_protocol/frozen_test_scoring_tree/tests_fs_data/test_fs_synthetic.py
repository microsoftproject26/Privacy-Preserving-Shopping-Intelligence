"""FS-01 synthetic tests (the DATA-04 frozen-layout testkit): the equivalence proof, the FS selection, the M01 / M02
assertions on re-read tables, the TEST gate, and their strict-xfail negative controls."""
from __future__ import annotations

import numpy as np
import pyarrow.parquet as pq
import pytest

from conftest import SYN, quiet
from data_testkit import assert_raises, nc
from fs_testkit import m01_assertions, m02_assertions, read_tables
from census_v2 import core as C2
from v4data.common import TestGateError, VALIDATION_END_US, digest_rows, refuse_test_split, sha256_bytes
from v4data.manifests import refuse_test_manifest
from v4fs_data import fs_common as FC


# ------------------------------------------------------------------------------------------ equivalence proof
def test_equivalence_reproduces_locked_M05_M06(synth_equiv):
    rec = synth_equiv["rec"]
    assert rec["a_all_equal"], {m: {k: v for k, v in r["fields"].items() if not v["equal"] and v["in_verdict"]}
                                for m, r in rec["a_M05_M06_definitions_vs_MANIFEST_HASHES"].items()}
    assert rec["b_all_equal"], {k: v for k, v in rec["b_FS_configuration_vs_M05_M06"].items() if not v["equal"]}
    assert rec["c_all_equal"]
    assert rec["all_equal"] and rec["status"] == "PASS"
    # the header content hash and every table content hash are part of the verdict
    for mid in ("M05_CALIB_TRAIN", "M06_CALIB_VALIDATION_WARM"):
        f = rec["a_M05_M06_definitions_vs_MANIFEST_HASHES"][mid]["fields"]
        assert f["header_content_sha256"]["equal"] and f["header_content_sha256"]["in_verdict"]
        assert all(v["equal"] for k, v in f.items() if k.endswith("table_content_sha256"))


def test_equivalence_receipt_self_hash(synth_equiv):
    import json
    rec = json.loads((synth_equiv["out"] / "receipts" / "EQUIVALENCE_M05_M06.json").read_text(encoding="utf-8"))
    assert FC.verify_self_hash(rec, "receipt_sha256")


def _entry_for(synth, users, calib_prefix=True):
    from v4fs_data.fs_build import PopulationBuilder
    b = PopulationBuilder(ds_root=synth["root"], out_root=synth["out_ref"], buckets=[0], expect=False, log=quiet)
    b.prepare()
    b.select_populations()
    P = users(b)
    b.set_population(P, calib_prefix=calib_prefix)
    b.extract()
    tm, _ = b.assemble_pair("M05_M06")
    enc = FC.encode_tables(b, tm["manifest_id"], tm)
    meta = FC.tables_meta_of(enc)
    return FC.hashes_entry(b._header(tm, meta), meta, None)


def test_compare_detects_nothing_on_identical_population(synth):
    got = _entry_for(synth, lambda b: np.asarray(b.calib))
    assert FC.compare_entries(got, synth["hashes"]["M05_CALIB_TRAIN"])["all_equal"]


@nc("dropping one user of the old M05 population must break the equivalence")
def test_nc_equivalence_detects_population_change(synth):
    got = _entry_for(synth, lambda b: np.sort(np.asarray(b.calib))[:-1])
    assert FC.compare_entries(got, synth["hashes"]["M05_CALIB_TRAIN"])["all_equal"]


@nc("without the calibration flag the M05 clients table differs (in_calib_prefix)")
def test_nc_equivalence_detects_calib_flag(synth):
    got = _entry_for(synth, lambda b: np.asarray(b.calib), calib_prefix=False)
    assert FC.compare_entries(got, synth["hashes"]["M05_CALIB_TRAIN"])["all_equal"]


# ------------------------------------------------------------------------------------------ FS selection
def _pool_order(b, lo, hi):
    pool = np.flatnonzero((b.u_nelig >= lo) & (b.u_nelig <= hi))
    keys = C2.selection_keys_v2(digest_rows(b.u_dig[pool]))
    o = np.argsort(keys)
    return pool[o], keys[o]


def test_fs_is_top_keys_of_pool_and_disjoint(synth_fs):
    b, rec = synth_fs["builder"], synth_fs["rec"]
    ps, ks = _pool_order(b, SYN["lo"], SYN["hi"])
    fs = np.sort(b.cohort)
    assert np.array_equal(fs, np.sort(ps[-SYN["n_fs"]:]))
    assert np.all((b.u_nelig[fs] >= SYN["lo"]) & (b.u_nelig[fs] <= SYN["hi"]))
    x, y = SYN["bands"]["MAIN"]
    assert np.intersect1d(fs, ps[x:y]).size == 0
    assert rec["population"]["fs_population_sha256_selection_key_v2"] == FC.population_sha256(ks[-SYN["n_fs"]:])
    assert all(v["disjoint"] for v in rec["population"]["disjointness"].values())


@nc("the SMALLEST keys are not the FS sample")
def test_nc_smallest_keys_are_not_fs(synth_fs):
    b = synth_fs["builder"]
    ps, _ = _pool_order(b, SYN["lo"], SYN["hi"])
    assert np.array_equal(np.sort(b.cohort), np.sort(ps[:SYN["n_fs"]]))


@nc("a band that overlaps FS must be reported as not disjoint")
def test_nc_overlapping_band_is_detected(synth_fs):
    b = synth_fs["builder"]
    sel = FC.select_fs(b.u_nelig, b.u_dig, lo=SYN["lo"], hi=SYN["hi"], n_fs=SYN["n_fs"],
                       bands={"FS": (1, 3), "OVERLAP": (0, 2)}, disjoint_from=("OVERLAP",))
    assert sel["all_disjoint"]


def test_select_fs_pool_bounds_and_oversize_refusal():
    dig = np.array([bytes([i + 1]) * 32 for i in range(4)], dtype="S32")
    with pytest.raises(AssertionError):                      # BuildBlocked: FS larger than the pool
        FC.select_fs(np.array([5, 5, 5, 5]), dig, n_fs=5, bands={}, disjoint_from=())
    sel = FC.select_fs(np.array([5, 4, 100, 101]), dig, n_fs=2, bands={}, disjoint_from=())
    assert sel["pool_size"] == 2 and sorted(sel["fs"].tolist()) == [0, 2]      # n = 4 and n = 101 are outside 5-100
    k = C2.selection_keys_v2([bytes([1]) * 32, bytes([3]) * 32])
    sel1 = FC.select_fs(np.array([5, 4, 100, 101]), dig, n_fs=1, bands={}, disjoint_from=())
    assert sel1["fs"].tolist() == [0 if k[0] > k[1] else 2]                    # the LARGEST key


# ------------------------------------------------------------------------------------------ M01 / M02 assertions
def test_fs_manifests_verify_on_disk(synth_fs):
    from v4fs_data.fs_finalize import verify_manifest_on_disk
    for mid in (SYN["train_id"], SYN["eval_id"]):
        v = verify_manifest_on_disk(synth_fs["v4"], "data_fs", mid)
        bad = {k: x for k, x in v["checks"].items() if not x and k != "header.status"}
        assert not bad, bad
        assert v["header"]["status"] == "PARTIAL_TEST_BUILD"       # synthetic = not a full build
        assert v["header"]["run_label"] == FC.FS_STATUS


def test_m01_m02_assertions_hold_on_reread_tables(synth_fs):
    mdir = synth_fs["out"] / "manifests"
    t = read_tables(mdir, SYN["train_id"])
    e = read_tables(mdir, SYN["eval_id"])
    K = t["header"]["catalogue_K"]
    a1 = m01_assertions(t["clients"], t["decisions"], t["central_order"], K)
    a2 = m02_assertions(e["users"], e["decisions"], t["clients"], t["decisions"], K)
    assert all(a1.values()), a1
    assert all(a2.values()), a2
    assert all(t["header"]["assertions"].values()) and all(e["header"]["assertions"].values())
    assert int(t["clients"]["in_calib_prefix"].sum()) == 0
    for x in (t, e):
        for tb in x["header"]["tables"]:
            assert x[f"{tb['table']}.__content"] == tb["table_content_sha256"]


@nc("a tampered target_ts at TRAIN_END must violate the independent M01 assertions")
def test_nc_m01_assertions_detect_tamper(synth_fs):
    t = read_tables(synth_fs["out"] / "manifests", SYN["train_id"])
    dec = dict(t["decisions"])
    dec["target_ts_us"] = dec["target_ts_us"].copy()
    from v4data.common import TRAIN_END_US
    dec["target_ts_us"][0] = TRAIN_END_US
    assert all(m01_assertions(t["clients"], dec, t["central_order"], t["header"]["catalogue_K"]).values())


@nc("an eval row flipped to censored must violate the independent M02 assertions")
def test_nc_m02_assertions_detect_tamper(synth_fs):
    mdir = synth_fs["out"] / "manifests"
    t, e = read_tables(mdir, SYN["train_id"]), read_tables(mdir, SYN["eval_id"])
    dec = dict(e["decisions"])
    dec["eval_mask"] = dec["eval_mask"].copy()
    i = int(np.flatnonzero(dec["eval_mask"])[0])
    dec["eval_mask"][i] = False
    assert all(m02_assertions(e["users"], dec, t["clients"], t["decisions"], t["header"]["catalogue_K"]).values())


# ------------------------------------------------------------------------------------------ TEST gate
def test_test_gate_refuses_every_test_path(synth, tmp_path):
    from v4fs_data.fs_build import PopulationBuilder, run_fs
    b = PopulationBuilder(ds_root=synth["root"], out_root=tmp_path, buckets=[0], expect=False, log=quiet)
    assert_raises(TestGateError, run_fs, b, tmp_path, lo=1, hi=100, n_fs=2, bands=SYN["bands"],
                  disjoint_from=("MAIN",), train_id="FS_TEST_5_100_32768", eval_id=SYN["eval_id"], write=False)
    assert not hasattr(b, "u_dig")                               # refused before any dataset read
    for mid in ("M07_TEST_WARM", "M08_TEST_COLD_PANEL", "FS_RETRO_TEST_X"):
        assert_raises(TestGateError, refuse_test_manifest, mid)
        assert_raises(TestGateError, FC.encode_tables, b, mid, {"tables": {}})
        assert_raises(TestGateError, FC.write_manifest, tmp_path, mid, [], {})
    for s in ("TEST", "RETRO_TEST"):
        assert_raises(TestGateError, refuse_test_split, s)


def test_no_test_row_reaches_fs_manifests(synth, synth_fs):
    led = pq.read_table(synth["root"] / "decisions" / "user_bucket=0" / "ledger.parquet",
                        columns=["decision_id", "split"])          # SYNTHETIC ledger only (planted TEST rows)
    test_ids = set(np.asarray(led["decision_id"])[np.asarray(led["split"]) == "TEST"].tolist())
    assert test_ids, "the fixture must plant TEST rows"
    mdir = synth_fs["out"] / "manifests"
    for mid in (SYN["train_id"], SYN["eval_id"]):
        d = read_tables(mdir, mid)["decisions"]
        assert not (set(d["decision_id"].tolist()) & test_ids)
        assert np.all(d["decision_ts_us"] < VALIDATION_END_US)


@nc("an unfiltered scan of the synthetic ledger does see TEST rows (read_non_test is what removes them)")
def test_nc_unfiltered_scan_sees_test(synth):
    led = pq.read_table(synth["root"] / "decisions" / "user_bucket=0" / "ledger.parquet", columns=["split"])
    assert "TEST" not in set(led["split"].to_pylist())


# ------------------------------------------------------------------------------------------ hashing helpers
def test_self_hash_and_header_rule():
    d = FC.with_self_hash({"a": 1, "b": [1, 2]}, "registry_self_sha256")
    assert FC.verify_self_hash(d, "registry_self_sha256")
    h = {"x": 1, "build_utc": "t", "tables": [{"table": "t", "row_count": 1, "table_content_sha256": "c",
                                               "file_sha256": "f"}]}
    h2 = dict(h, build_utc="other", tables=[dict(h["tables"][0], file_sha256="g")])
    assert FC.header_content_sha256(h) == FC.header_content_sha256(h2)
    assert FC.manifest_content_sha256(h["tables"]) == FC.manifest_content_sha256(h2["tables"])
    assert sha256_bytes(b"") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


@nc("a tampered registry must fail its self-hash")
def test_nc_tampered_self_hash():
    d = FC.with_self_hash({"a": 1}, "registry_self_sha256")
    d["a"] = 2
    assert FC.verify_self_hash(d, "registry_self_sha256")
