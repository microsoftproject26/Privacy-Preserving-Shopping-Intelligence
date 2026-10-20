"""DATA-10 synthetic tests: band arithmetic (TIEBREAK minus MAIN), disjointness from MAIN / CALIB / U_NEW_VAL /
U_NEW_TEST / FS and the DATA-06 panels, TRAIN-only rows, the locked M01 layout / schema, header and receipt integrity,
the band-only receipt, the TEST gate - and their strict-xfail negative controls."""
from __future__ import annotations

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from conftest import SYN_BANDS, SYN_PRETRAIN, quiet
from data_testkit import assert_raises, nc
from fs_testkit import load_json, m01_assertions, read_tables
from v4data.common import TRAIN_END_US, TestGateError, sha256_file
from v4data.manifests import BuildBlocked, refuse_test_manifest
from v4data10 import d10_common as D10
from v4fs_data import fs_common as FC


def _tables(sb):
    return read_tables(sb["out"] / "manifests", D10.TRAIN_ID)


def _keys_of_ranks(sb, x, y):
    rec = sb["rec"]
    return set(rec["_keys_v2"][rec["_pool_sorted"][x:y]].tolist())


def _band_keys(sb, name):
    return _keys_of_ranks(sb, *SYN_BANDS[name])


def _client_keys(t):
    return set(FC.keys_v2_of_hex(t["clients"]["client_key_hex"].astype(str)).tolist())


def _ledger(root):
    return pq.read_table(root / "decisions" / "user_bucket=0" / "ledger.parquet")


# ------------------------------------------------------------------------------------------ band arithmetic
def test_registered_band_arithmetic():
    assert D10.BANDS is FC.BANDS
    assert D10.RANKS == (131_072, 1_348_682) == (FC.BANDS["MAIN"][1], FC.BANDS["TIEBREAK"][1])
    assert D10.N_USERS == 1_217_610 == FC.BANDS["TIEBREAK"][1] - FC.BANDS["MAIN"][1]
    assert D10.N_DECISIONS == 24_089_824 == 26_677_600 - 2_587_776
    assert FC.BANDS["CALIB"][1] <= FC.BANDS["MAIN"][1] and FC.BANDS["U_NEW_VAL"][0] == FC.BANDS["TIEBREAK"][1]
    assert len(D10.PRETRAIN_SHA_V2) == 64 and D10.PRETRAIN_SHA_V2 not in (D10.TB_SHA_V2, D10.MAIN_SHA_V2)
    assert D10.EVAL_ID == "UNEW_VAL_VALIDATION_32768" and D10.ROLE == "PRETRAIN"


def test_band_ranks_refuse_a_main_that_is_not_a_prefix():
    assert_raises(BuildBlocked, D10.band_ranks, dict(SYN_BANDS, MAIN=(1, 4)))
    assert_raises(BuildBlocked, D10.band_ranks, dict(SYN_BANDS, MAIN=(0, 8)))


def test_pretrain_band_function_checks_cover_and_disjoint():
    pool = np.arange(100, 114)
    sel = D10.pretrain_band(pool, SYN_BANDS)
    assert sel["ranks"] == list(SYN_PRETRAIN) and np.array_equal(sel["users"], pool[4:8])
    assert all(obs == exp for obs, exp in sel["checks"].values()), sel["checks"]
    for name in ("MAIN", "CALIB", "U_NEW_VAL", "U_NEW_TEST", "FS"):
        assert f"disjoint.PRETRAIN_NONMAIN_vs_{name}" in sel["checks"]


def test_scaled_bands_for_partial_smoke_keep_the_locked_layout():
    for P in (14, 90_454, 1_000_003):
        sb = D10.scaled_bands(P)
        assert sb["FS"][1] == P and sb["MAIN"][0] == sb["CALIB"][0] == sb["TIEBREAK"][0] == 0
        assert sb["CALIB"][1] <= sb["MAIN"][1] < sb["TIEBREAK"][1] == sb["U_NEW_VAL"][0]
        assert sb["U_NEW_VAL"][1] == sb["U_NEW_TEST"][0] and sb["U_NEW_TEST"][1] == sb["FS"][0]
        sel = D10.pretrain_band(np.arange(P), sb)
        assert all(o == e for o, e in sel["checks"].values())
    assert D10.scaled_bands(D10.POOL_N) == dict(FC.BANDS)          # identity on the full pool


@nc("a TIEBREAK band that reaches into U_NEW_VAL must fail the disjointness / cover checks")
def test_nc_band_overlapping_unew_val_fails():
    sel = D10.pretrain_band(np.arange(14), dict(SYN_BANDS, TIEBREAK=(0, 9)))
    assert all(obs == exp for obs, exp in sel["checks"].values())


def test_panel_checks_detect_overlap_and_stratum():
    users = np.array([5, 6, 7])
    ok = D10.panel_checks(users, {"SPARSE": np.array([1, 2]), "WARM_TRUNCATION": np.array([0])},
                          np.array([True] * 8))
    assert all(o == e for o, e in ok.values()) and "PRETRAIN_NONMAIN.all_in_pool_stratum_C" in ok


@nc("a panel that contains a PRETRAIN user must fail the panel disjointness check")
def test_nc_panel_overlap_fails():
    chk = D10.panel_checks(np.array([5, 6, 7]), {"HEAVY": np.array([7, 9])})
    assert all(o == e for o, e in chk.values())


# ------------------------------------------------------------------------------------------ the synthetic build
def test_clients_are_exactly_the_pretrain_band(synth_build):
    t = _tables(synth_build)
    assert _client_keys(t) == _keys_of_ranks(synth_build, *SYN_PRETRAIN)
    assert t["clients"]["client_ord"].size == SYN_PRETRAIN[1] - SYN_PRETRAIN[0]


def test_main_calib_unew_fs_excluded_and_union_with_main_is_tiebreak(synth_build):
    ck = _client_keys(_tables(synth_build))
    for name in D10.EXCLUDED_BANDS:
        assert not (ck & _band_keys(synth_build, name)), name
    assert ck | _band_keys(synth_build, "MAIN") == _band_keys(synth_build, "TIEBREAK")


def test_build_checks_include_every_disjointness_and_panel(synth_build):
    names = {c["name"]: c for c in synth_build["rec"]["checks"]}
    for n in [f"disjoint.PRETRAIN_NONMAIN_vs_{x}" for x in D10.EXCLUDED_BANDS + D10.PANELS] + [
            "pool_covered_exactly_once_by_PRETRAIN_MAIN_UNEW_FS", "PRETRAIN_NONMAIN_union_MAIN_eq_TIEBREAK",
            "PRETRAIN_NONMAIN.all_in_pool_stratum_C", "M01_rows.none_from_MAIN", "M01_rows.all_in_PRETRAIN"]:
        assert n in names and names[n]["pass"], n


@nc("the MAIN band's users are not the manifest's clients")
def test_nc_main_is_not_the_client_set(synth_build):
    assert _client_keys(_tables(synth_build)) & _band_keys(synth_build, "MAIN")


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
    """Independent of the builder: PRETRAIN users' TRAIN rows with loss_mask, a catalogue target, target_ts < TRAIN_END."""
    from census_v2 import core as C2
    from v4data.common import digest_rows, digests_of
    t = _tables(synth_build)
    led = _ledger(synth_build["root"])
    users = led["user_id"].to_pylist()
    tts = led["target_ts"].cast("int64").to_pylist()
    band = _keys_of_ranks(synth_build, *SYN_PRETRAIN)
    k_of = {u: int(C2.selection_keys_v2(digest_rows(digests_of([u])))[0]) for u in set(users)}
    want = sorted(d for u, s, lm, tc, ts, d in zip(users, led["split"].to_pylist(), led["loss_mask"].to_pylist(),
                                                  led["target_class"].to_pylist(), tts, led["decision_id"].to_pylist())
                  if k_of[u] in band and s == "TRAIN" and lm and tc >= 0 and ts is not None and ts < TRAIN_END_US)
    assert want and sorted(t["decisions"]["decision_id"].tolist()) == want
    b = synth_build["builder"]
    assert synth_build["rec"]["manifests"][D10.TRAIN_ID]["n_decisions"] == len(want) == int(
        b.u_nelig[synth_build["rec"]["_users"]].sum())


@nc("a VALIDATION row planted into the decisions table must violate the TRAIN-only assertions")
def test_nc_validation_row_breaks_train_only(synth_build):
    t = _tables(synth_build)
    dec = {k: v.copy() for k, v in t["decisions"].items()}
    dec["old_split"] = dec["old_split"].astype(object)
    dec["old_split"][0] = "VALIDATION_SELECT"
    dec["target_ts_us"][0] = TRAIN_END_US + 1
    a = m01_assertions(t["clients"], dec, t["central_order"], synth_build["builder"].K)
    assert all(a.values())


def test_schema_equals_the_locked_m01_layout(synth_build, synth_ref):
    mine = load_json(synth_build["out"] / "manifests" / f"{D10.TRAIN_ID}.header.json")
    ref = load_json(synth_ref["out"] / "manifests" / "M01_COHORT_TRAIN.header.json")
    assert [x["table"] for x in mine["tables"]] == [x["table"] for x in ref["tables"]] == \
        ["clients", "decisions", "central_order"]
    for tname in ("clients", "decisions", "central_order"):
        a = pq.read_schema(synth_build["out"] / "manifests" / f"{D10.TRAIN_ID}.{tname}.parquet")
        b = pq.read_schema(synth_ref["out"] / "manifests" / f"M01_COHORT_TRAIN.{tname}.parquet")
        assert a.remove_metadata().equals(b.remove_metadata()), tname


@nc("a TRAIN table without one of its locked columns is not the M01 layout")
def test_nc_schema_detects_missing_column(synth_build, synth_ref):
    a = pq.read_schema(synth_build["out"] / "manifests" / f"{D10.TRAIN_ID}.decisions.parquet")
    b = pq.read_schema(synth_ref["out"] / "manifests" / "M01_COHORT_TRAIN.decisions.parquet")
    a = a.remove(a.get_field_index("old_split"))
    assert a.remove_metadata().equals(b.remove_metadata())


def test_header_identity_and_content_hash(synth_build):
    h = load_json(synth_build["out"] / "manifests" / f"{D10.TRAIN_ID}.header.json")
    assert h["manifest_id"] == D10.TRAIN_ID and h["task"] == D10.TASK and h["data_version"] == D10.DATA_VERSION
    assert h["definition"]["manifest_spec_definition"] == D10.DEFINITION and h["parent_manifests"] == []
    assert h["status"] == "PARTIAL_TEST_BUILD" and h["run_label"] == D10.ROLE     # partial: never labelled PRETRAIN
    assert h["header_content_sha256"] == FC.header_content_sha256(h)
    assert h["population_v2"]["band"] == D10.BAND and h["population_v2"]["rank_range"] == list(SYN_PRETRAIN)
    assert h["selection_refusal"] == D10.SELECTION_REFUSAL and all(h["assertions"].values())
    mk = load_json(synth_build["out"] / "manifests" / f"{D10.TRAIN_ID}.OK")["verified_sha256"]
    assert all(sha256_file(synth_build["out"] / rel) == s for rel, s in mk.items())
    pf = synth_build["out"] / "population" / "PRETRAIN_NONMAIN_SELECTION_KEY_V2.u64be"
    assert sha256_file(pf) == h["population_sha256_selection_key_v2"]


def test_build_receipt_self_hash_and_no_failure(synth_build):
    rec = load_json(synth_build["out"] / "receipts" / "DATA10_BUILD_RECEIPT.json")
    assert FC.verify_self_hash(rec, "receipt_sha256") and rec["n_failed"] == 0 and rec["status"] == "PASS"
    assert rec["test_split_read"] is False and rec["role"] == "PRETRAIN" and rec["eval_manifest"] == D10.EVAL_ID
    assert rec["n_validation_rows_collected_and_released"] > 0 and rec["full_build"] is False


@nc("a tampered build-receipt field must fail its self-hash")
def test_nc_tampered_receipt(synth_build):
    rec = load_json(synth_build["out"] / "receipts" / "DATA10_BUILD_RECEIPT.json")
    rec["manifests"][D10.TRAIN_ID]["n_decisions"] += 1
    assert FC.verify_self_hash(rec, "receipt_sha256")


def test_band_only_receipt_matches_the_build(synth_ds, synth_build, tmp_path):
    from v4data10.d10_build import D10Builder, run_band_only
    b = D10Builder(ds_root=synth_ds, out_root=tmp_path, buckets=[0], expect=False, log=quiet)
    rec = run_band_only(b, tmp_path, bands=SYN_BANDS, write=True)
    back = load_json(tmp_path / "receipts" / "DATA10_BAND_RECEIPT.json")
    assert FC.verify_self_hash(back, "receipt_sha256") and back["n_failed"] == 0
    e = synth_build["rec"]["manifests"][D10.TRAIN_ID]
    assert rec["population"]["n_users"] == e["n_users"] and rec["n_eligible_examples"] == e["n_decisions"]
    assert rec["population"]["population_sha256_selection_key_v2"] == \
        synth_build["rec"]["population"]["population_sha256_selection_key_v2"]


def test_test_gate_refuses_test_ids_and_accepts_ours():
    refuse_test_manifest(D10.TRAIN_ID)
    assert "TEST" not in D10.ROLE.upper() and "TEST" not in D10.TRAIN_ID.upper() and "TEST" not in D10.EVAL_ID
    assert_raises(TestGateError, refuse_test_manifest, "PRETRAIN_TEST_5_100_NONMAIN")
    assert_raises(TestGateError, FC.encode_tables, None, "PRETRAIN_TEST", {"tables": {}})


def test_synthetic_ledger_does_contain_test_rows_the_build_never_read(synth_build):
    led = _ledger(synth_build["root"])
    assert pc.sum(pc.cast(pc.equal(led["split"], "TEST"), "int64")).as_py() > 0
    assert synth_build["rec"]["test_split_read"] is False
