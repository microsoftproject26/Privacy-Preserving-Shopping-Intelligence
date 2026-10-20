"""DATA-08 synthetic tests: band exclusion (TB only; U_NEW_VAL / U_NEW_TEST / FS excluded), TRAIN-only rows, the locked
M01 layout / schema, header and receipt integrity, the TEST gate - and their strict-xfail negative controls."""
from __future__ import annotations

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from conftest import SYN_BANDS
from data_testkit import assert_raises, nc
from fs_testkit import load_json, m01_assertions, read_tables
from v4data.common import TRAIN_END_US, TestGateError, sha256_file
from v4data.manifests import refuse_test_manifest
from v4data08 import d8_common as D8
from v4fs_data import fs_common as FC


def _tables(sb):
    return read_tables(sb["out"] / "manifests", D8.TRAIN_ID)


def _band_keys(sb, name):
    rec = sb["rec"]
    x, y = SYN_BANDS[name]
    return set(rec["_keys_v2"][rec["_pool_sorted"][x:y]].tolist())


def _client_keys(t):
    return set(FC.keys_v2_of_hex(t["clients"]["client_key_hex"].astype(str)).tolist())


def _ledger(root):
    return pq.read_table(root / "decisions" / "user_bucket=0" / "ledger.parquet")


# ------------------------------------------------------------------------------------------ band
def test_clients_are_exactly_the_tiebreak_band(synth_build):
    t = _tables(synth_build)
    assert _client_keys(t) == _band_keys(synth_build, "TIEBREAK")
    assert t["clients"]["client_ord"].size == SYN_BANDS["TIEBREAK"][1] - SYN_BANDS["TIEBREAK"][0]


def test_unew_and_fs_bands_excluded_main_calib_included(synth_build):
    ck = _client_keys(_tables(synth_build))
    for name in D8.EXCLUDED_BANDS:
        assert not (ck & _band_keys(synth_build, name)), name
    assert _band_keys(synth_build, "MAIN") <= ck and _band_keys(synth_build, "CALIB") <= ck


def test_tb_band_function_checks_cover_and_disjoint():
    pool = np.arange(100, 114)
    sel = D8.tb_band(pool, SYN_BANDS)
    assert np.array_equal(sel["tb"], pool[:8])
    assert all(obs == exp for obs, exp in sel["checks"].values()), sel["checks"]


def test_reused_population_definition_is_the_locked_band():
    assert D8.BANDS is FC.BANDS and D8.BANDS["TIEBREAK"] == (0, 1_348_682)
    assert D8.TB_N_USERS == 1_348_682 and D8.TB_N_DECISIONS == 26_677_600
    assert D8.AMENDMENT_06_STATED_USERS == FC.POOL_N - 2 * 65_536          # the superseded DL40 layout
    assert FC.BANDS["U_NEW_VAL"][0] == FC.BANDS["TIEBREAK"][1] and FC.BANDS["FS"][1] == FC.POOL_N


@nc("a band that reaches into U_NEW_VAL must fail the exclusion / cover checks")
def test_nc_band_overlapping_unew_val_fails():
    bad = dict(SYN_BANDS, TIEBREAK=(0, 9))
    sel = D8.tb_band(np.arange(14), bad)
    assert all(obs == exp for obs, exp in sel["checks"].values())


@nc("the U_NEW_VAL band's users are not the manifest's clients")
def test_nc_unew_val_is_not_the_client_set(synth_build):
    assert _client_keys(_tables(synth_build)) & _band_keys(synth_build, "U_NEW_VAL")


# ------------------------------------------------------------------------------------------ TRAIN-only rows
def test_m01_assertions_on_reread_tables(synth_build):
    t = _tables(synth_build)
    a = m01_assertions(t["clients"], t["decisions"], t["central_order"], synth_build["builder"].K)
    assert all(a.values()), {k: v for k, v in a.items() if not v}


def test_rows_are_train_only_and_no_test_or_validation_id(synth_build):
    t = _tables(synth_build)
    led = _ledger(synth_build["root"])
    did = set(t["decisions"]["decision_id"].tolist())
    for split in ("TEST", "VALIDATION_SELECT", "VALIDATION_CONFIRM"):
        ids = set(led.filter(pc.equal(led["split"], split))["decision_id"].to_pylist())
        assert ids and not (did & ids), split
    assert set(t["decisions"]["old_split"].astype(str)) == {"TRAIN"}
    assert int(t["decisions"]["target_ts_us"].max()) < TRAIN_END_US


def test_decisions_equal_an_independent_count(synth_build):
    """Independent of the builder: TB users' TRAIN rows with loss_mask, a catalogue target and target_ts < TRAIN_END."""
    from census_v2 import core as C2
    from v4data.common import digest_rows, digests_of
    t = _tables(synth_build)
    led = _ledger(synth_build["root"])
    users = led["user_id"].to_pylist()
    tts = led["target_ts"].cast("int64").to_pylist()
    tbk = _band_keys(synth_build, "TIEBREAK")
    k_of = {u: int(C2.selection_keys_v2(digest_rows(digests_of([u])))[0]) for u in set(users)}
    want = sorted(d for u, s, lm, tc, ts, d in zip(users, led["split"].to_pylist(), led["loss_mask"].to_pylist(),
                                                  led["target_class"].to_pylist(), tts, led["decision_id"].to_pylist())
                  if k_of[u] in tbk and s == "TRAIN" and lm and tc >= 0 and ts is not None and ts < TRAIN_END_US)
    assert want and sorted(t["decisions"]["decision_id"].tolist()) == want
    b = synth_build["builder"]
    assert synth_build["rec"]["manifests"][D8.TRAIN_ID]["n_decisions"] == len(want) == int(
        b.u_nelig[synth_build["rec"]["_tb"]].sum())


@nc("a VALIDATION row planted into the decisions table must violate the TRAIN-only assertions")
def test_nc_validation_row_breaks_train_only(synth_build):
    t = _tables(synth_build)
    dec = {k: v.copy() for k, v in t["decisions"].items()}
    dec["old_split"] = dec["old_split"].astype(object)
    dec["old_split"][0] = "VALIDATION_SELECT"
    dec["target_ts_us"][0] = TRAIN_END_US + 1
    a = m01_assertions(t["clients"], dec, t["central_order"], synth_build["builder"].K)
    assert all(a.values())


# ------------------------------------------------------------------------------------------ layout / schema
def test_schema_equals_the_locked_m01_layout(synth_build, synth_ref):
    mine = load_json(synth_build["out"] / "manifests" / f"{D8.TRAIN_ID}.header.json")
    ref = load_json(synth_ref["out"] / "manifests" / "M01_COHORT_TRAIN.header.json")
    assert [x["table"] for x in mine["tables"]] == [x["table"] for x in ref["tables"]] == \
        ["clients", "decisions", "central_order"]
    for tname in ("clients", "decisions", "central_order"):
        a = pq.read_schema(synth_build["out"] / "manifests" / f"{D8.TRAIN_ID}.{tname}.parquet")
        b = pq.read_schema(synth_ref["out"] / "manifests" / f"M01_COHORT_TRAIN.{tname}.parquet")
        assert a.remove_metadata().equals(b.remove_metadata()), tname


@nc("a TRAIN table without one of its locked columns is not the M01 layout")
def test_nc_schema_detects_missing_column(synth_build, synth_ref):
    a = pq.read_schema(synth_build["out"] / "manifests" / f"{D8.TRAIN_ID}.decisions.parquet")
    b = pq.read_schema(synth_ref["out"] / "manifests" / "M01_COHORT_TRAIN.decisions.parquet")
    a = a.remove(a.get_field_index("old_split"))
    assert a.remove_metadata().equals(b.remove_metadata())


# ------------------------------------------------------------------------------------------ header / receipt / gate
def test_header_identity_and_content_hash(synth_build):
    h = load_json(synth_build["out"] / "manifests" / f"{D8.TRAIN_ID}.header.json")
    assert h["manifest_id"] == D8.TRAIN_ID and h["task"] == D8.TASK and h["data_version"] == D8.DATA_VERSION
    assert h["definition"]["manifest_spec_definition"] == D8.DEFINITION and h["parent_manifests"] == []
    assert h["status"] == "PARTIAL_TEST_BUILD"                 # a partial build is never labelled C_FULL
    assert h["header_content_sha256"] == FC.header_content_sha256(h)
    assert h["population_v2"]["band"] == "TIEBREAK" and h["population_v2"]["amendment_06_stated_users"] == 1_315_914
    assert all(h["assertions"].values())
    mk = load_json(synth_build["out"] / "manifests" / f"{D8.TRAIN_ID}.OK")["verified_sha256"]
    assert all(sha256_file(synth_build["out"] / rel) == s for rel, s in mk.items())


def test_build_receipt_self_hash_and_no_failure(synth_build):
    rec = load_json(synth_build["out"] / "receipts" / "DATA08_BUILD_RECEIPT.json")
    assert FC.verify_self_hash(rec, "receipt_sha256") and rec["n_failed"] == 0 and rec["status"] == "PASS"
    assert rec["test_split_read"] is False and rec["role"] == "C_FULL" and rec["eval_manifest"] == D8.EVAL_ID
    assert rec["n_validation_rows_collected_and_released"] > 0


@nc("a tampered build-receipt field must fail its self-hash")
def test_nc_tampered_receipt(synth_build):
    rec = load_json(synth_build["out"] / "receipts" / "DATA08_BUILD_RECEIPT.json")
    rec["manifests"][D8.TRAIN_ID]["n_decisions"] += 1
    assert FC.verify_self_hash(rec, "receipt_sha256")


def test_test_gate_refuses_test_ids_and_accepts_ours():
    refuse_test_manifest(D8.TRAIN_ID)
    assert "TEST" not in D8.ROLE.upper() and "TEST" not in D8.TRAIN_ID.upper()
    assert_raises(TestGateError, refuse_test_manifest, "FULL_TEST_5_100_TIEBREAK")
    assert_raises(TestGateError, FC.encode_tables, None, "C_FULL_TEST", {"tables": {}})


def test_synthetic_ledger_does_contain_test_rows_the_build_never_read(synth_build):
    led = _ledger(synth_build["root"])
    assert pc.sum(pc.cast(pc.equal(led["split"], "TEST"), "int64")).as_py() > 0
    rec = synth_build["rec"]
    assert rec["test_split_read"] is False
