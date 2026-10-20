"""DATA-09 synthetic build tests: the RETRO_TEST rows of every population against an INDEPENDENT ledger reference (counts,
grace maturation, censor classes, eval_mask / rankable, eval_user_key densification = the D6 rule, class positives, the
M08 population rule, panel users without a RETRO_TEST row), the M02 / M03 schema + grace_matured, headers / receipts /
gate fields, the full-population counters and the G0 comparison - with strict-xfail negative controls."""
from __future__ import annotations

import copy

import numpy as np
import pyarrow.parquet as pq
import pytest

from conftest import SYN, quiet, synthetic_bands
from d9_testkit import Ledger, load_json, m07_assertions, read_tables_d9
from data_testkit import assert_raises, nc
from v4data.common import GRACE_END_US, TEST_END_US, sha256_file
from v4data.manifests import BuildBlocked
from v4data09 import d9_common as D9
from v4data09 import d9_finalize as F
from v4data09.d9_gate import TestGateRefused
from v4fs_data import fs_common as FC


@pytest.fixture(scope="module")
def led(synth_build):
    return Ledger(synth_build["root"])


def _t(sb, mid):
    return read_tables_d9(sb["out"] / "manifests", mid)


def _row_users(t, led):
    return {led.by_did[int(d)]["user_id"] for d in t["decisions"]["decision_id"]}


def _population(mid, led):
    b = SYN["BANDS"]
    if mid == D9.MAIN_TEST_ID:
        return led.band(*b["MAIN"])
    if mid == D9.UNEW_TEST_ID:
        return led.band(*b["U_NEW_TEST"])
    if mid == D9.SPARSE_TEST_ID:
        return led.stratum_with_eval(1, 4, SYN["N_PANEL"])
    if mid == D9.HEAVY_TEST_ID:
        return led.stratum_with_eval(101, None, SYN["N_PANEL"])
    return led.m08_users()


# ------------------------------------------------------------------------------------------ populations and rows
@pytest.mark.parametrize("mid", D9.TEST_IDS)
def test_rows_are_every_retro_test_decision_of_the_population(synth_build, led, mid):
    t = _t(synth_build, mid)
    pop = _population(mid, led)
    want = sorted(r["decision_id"] for r in led.test_rows(pop))
    assert want and t["decisions"]["decision_id"].tolist() == want
    assert _row_users(t, led) <= set(pop)
    assert set(t["users"]["client_key_hex"].astype(str)) == {led.hex[u] for u in pop if led.test_rows([u])}


def test_panel_users_without_a_retro_test_row_are_absent_and_counted(synth_build, led):
    rec = synth_build["rec"]
    for mid in (D9.SPARSE_TEST_ID, D9.HEAVY_TEST_ID, D9.MAIN_TEST_ID):
        pop = _population(mid, led)
        missing = [u for u in pop if not led.test_rows([u])]
        assert rec["population_extra"][mid]["n_population_users_without_retro_test_row"] == len(missing)
        assert not ({led.hex[u] for u in missing} & set(_t(synth_build, mid)["users"]["client_key_hex"].astype(str)))
    assert rec["population_extra"][D9.SPARSE_TEST_ID]["n_population_users_without_retro_test_row"] >= 1


def test_true_cold_is_the_literal_m08_rule(synth_build, led):
    t = _t(synth_build, D9.TRUE_COLD_TEST_ID)
    users = set(_row_users(t, led))
    assert users == set(led.m08_users())
    assert "testonly-0" in users and "cold-a1-t" in users              # TEST-only / A1 users included
    assert "testonly-1" not in users and "cold-censored" not in users  # no RETRO_TEST eval row -> excluded
    assert all(led.nelig[u] == 0 for u in users)
    strata = dict(zip(t["users"]["client_key_hex"].astype(str), t["users"]["cold_stratum"].astype(str)))
    assert strata[led.hex["cold-a1-t"]] == "COLD_TRAIN_EVENTS_NO_LABEL"
    assert strata[led.hex["testonly-0"]] == "COLD_NO_TRAIN_EVENTS"
    assert synth_build["rec"]["population_extra"][D9.TRUE_COLD_TEST_ID]["n_test_only_users"] == 1


@nc("restricting M08 to the TRUE_COLD_VALIDATION users (VALIDATION-cold) is NOT the registered rule")
def test_nc_true_cold_restricted_to_validation_cold_users(synth_build, led):
    t = _t(synth_build, D9.TRUE_COLD_TEST_ID)
    val_cold = {u for u in led.m08_users() if led.nval_eval[u] >= 1}
    assert set(_row_users(t, led)) == val_cold


# ------------------------------------------------------------------------------------------ assertions / grace / censor
@pytest.mark.parametrize("mid", D9.TEST_IDS)
def test_independent_m07_assertions_hold(synth_build, mid):
    t = _t(synth_build, mid)
    a = m07_assertions(t["users"], t["decisions"])
    assert all(a.values()), {k: v for k, v in a.items() if not v}
    assert all(v is True for v in t["header"]["assertions"].values())


def _all_decisions(sb):
    parts = [_t(sb, m)["decisions"] for m in D9.TEST_IDS]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def test_grace_matured_rows(synth_build, led):
    d = _all_decisions(synth_build)
    gr = d["grace_matured"].astype(bool)
    assert gr.sum() >= 3
    for i in np.flatnonzero(gr):
        r = led.by_did[int(d["decision_id"][i])]
        assert r["target_status"] == "OBSERVED_MATURED_IN_GRACE" and r["eval_mask"]
        assert TEST_END_US <= int(d["target_ts_us"][i]) < GRACE_END_US
        assert d["censor_class"][i] == "NONE" and bool(d["eval_mask"][i])
        assert bool(d["rankable"][i]) == (r["target_class"] >= 0) and int(d["target_class"][i]) == r["target_class"]
        assert d["target_category_state"][i] == "NOT_APPLICABLE"
    oov_grace = gr & d["target_oov"].astype(bool)
    assert oov_grace.any() and set(d["oov_cause"][oov_grace].astype(str)) == {"ABSENT_FROM_TRAIN"}


def test_censor_classes_on_retro_test_rows(synth_build, led):
    d = _all_decisions(synth_build)
    seen = set()
    for did, cen in zip(d["decision_id"].tolist(), d["censor_class"].astype(str).tolist()):
        r = led.by_did[did]
        want = {"NO_OBSERVED_NEXT": "CEN_NO_NEXT", "AMBIGUOUS_BOUNDARY": "CEN_AMBIGUOUS_NEXT_BARRIER",
                "BOUNDARY_UNRESOLVED": "CEN_RIGHT_EDGE"}.get(r["target_status"], "NONE")
        assert cen == want, (did, r["target_status"], cen)
        seen.add(cen)
    assert seen == {"NONE", "CEN_NO_NEXT", "CEN_AMBIGUOUS_NEXT_BARRIER", "CEN_RIGHT_EDGE"}


def test_in_partition_oov_target_and_metadata_join(synth_build, led):
    d = _all_decisions(synth_build)
    ip = d["eval_mask"].astype(bool) & ~d["grace_matured"].astype(bool)
    assert set(d["target_category_state"][ip].astype(str)) <= {"KNOWN", "MISSING", "OOV"}
    oov = ip & d["target_oov"].astype(bool)
    assert oov.any() and not d["rankable"][oov].any()


def test_eval_user_key_is_the_d6_densification_rule(synth_build, led):
    for mid in D9.TEST_IDS:
        t = _t(synth_build, mid)
        hexs = t["users"]["client_key_hex"].astype(str)
        assert list(hexs) == sorted(hexs) and np.array_equal(t["users"]["eval_user_key"], np.arange(hexs.size))
        for did, k in zip(t["decisions"]["decision_id"].tolist(), t["decisions"]["eval_user_key"].tolist()):
            assert hexs[k] == led.hex[led.by_did[did]["user_id"]]


@pytest.mark.parametrize("mid", D9.TEST_IDS)
def test_class_positives_counted_over_the_support_population(synth_build, led, mid):
    t = _t(synth_build, mid)
    K = synth_build["builder"].K
    ref = led.band(*SYN["BANDS"]["MAIN"]) if mid == D9.TRUE_COLD_TEST_ID else _population(mid, led)
    pos = led.class_positives(ref, K)
    rk = t["decisions"]["rankable"].astype(bool)
    tc = t["decisions"]["target_class"].astype(np.int64)
    tp = t["decisions"]["target_class_cohort_train_positives"].astype(np.int64)
    assert np.array_equal(tp[rk], pos[tc[rk]]) and np.all(tp[~rk] == -1)


# ------------------------------------------------------------------------------------------ schema
def test_schema_is_m02_m03_plus_grace_matured(synth_build, synth_ds, tmp_path):
    from v4data.manifests import ManifestBuilder
    ref = ManifestBuilder(ds_root=synth_ds, out_root=tmp_path, buckets=[0], expect=False, log=quiet)
    ref.pass_support()
    ref.pass_aggregate()
    ref.select_populations()
    ref.pass_extract()
    ref.write(ref.assemble())
    md = tmp_path / "manifests"
    for mid, locked in ((D9.MAIN_TEST_ID, "M02_VALIDATION_WARM"), (D9.TRUE_COLD_TEST_ID, "M03_COLD_PANEL_VAL")):
        mine_d = pq.read_schema(synth_build["out"] / "manifests" / f"{mid}.decisions.parquet").remove_metadata()
        ref_d = pq.read_schema(md / f"{locked}.decisions.parquet").remove_metadata()
        assert mine_d.names == ref_d.names + ["grace_matured"]
        assert str(mine_d.field("grace_matured").type) == "bool"
        assert mine_d.remove(mine_d.get_field_index("grace_matured")).equals(ref_d)
        mine_u = pq.read_schema(synth_build["out"] / "manifests" / f"{mid}.users.parquet").remove_metadata()
        assert mine_u.equals(pq.read_schema(md / f"{locked}.users.parquet").remove_metadata())


# ------------------------------------------------------------------------------------------ header / receipt / gate
@pytest.mark.parametrize("mid", D9.TEST_IDS)
def test_header_gate_fields_and_integrity(synth_build, synth_gate, mid):
    h = load_json(synth_build["out"] / "manifests" / f"{mid}.header.json")
    assert h["manifest_id"] == mid and h["task"] == D9.TASK and h["v4_split"] == "RETRO_TEST"
    assert h["all_methods_lock_sha256"] == synth_gate.lock_sha256
    assert h["test_exposure_register_entry_id"] == synth_gate.entry_id
    assert h["definition"]["manifest_spec_definition"] == D9.DEFINITION[mid]
    assert h["status"] == "PARTIAL_TEST_BUILD" and h["run_label"] == D9.ROLE_OF_ID[mid]
    assert h["parent_manifests"] == ([D9.PARENT[mid]] if D9.PARENT[mid] else [])
    assert h["header_content_sha256"] == FC.header_content_sha256(h)
    mk = load_json(synth_build["out"] / "manifests" / f"{mid}.OK")["verified_sha256"]
    assert all(sha256_file(synth_build["out"] / rel) == s for rel, s in mk.items())


def test_build_receipt(synth_build, synth_gate):
    rec = load_json(synth_build["out"] / "receipts" / "DATA09_BUILD_RECEIPT.json")
    assert FC.verify_self_hash(rec, "receipt_sha256") and rec["status"] == "PASS" and rec["n_failed"] == 0
    assert rec["test_gate"]["all_methods_lock_sha256"] == synth_gate.lock_sha256
    assert rec["test_split_read"] is True and set(rec["manifests"]) == set(D9.TEST_IDS)
    assert rec["target_metadata_join"]["target_pid_mismatch"] == rec["target_metadata_join"]["target_ts_mismatch"] == 0


def test_full_population_counters_match_an_independent_count(synth_build, led):
    c = synth_build["rec"]["full_population_retro_test_counters"]
    tr = [r for r in led.rows if r["split"] == "TEST"]
    p = c["per_old_split_TEST"]
    assert p["decisions"] == len(tr) and p["eval_mask_true"] == sum(r["eval_mask"] for r in tr)
    assert p["distinct_users"] == len({r["user_id"] for r in tr})
    assert p["target_status_counts"]["OBSERVED_MATURED_IN_GRACE"] == sum(
        r["target_status"] == "OBSERVED_MATURED_IN_GRACE" for r in tr)
    assert c["v4_retro_test"] == p
    ev_users = {r["user_id"] for r in tr if r["eval_mask"]}
    assert c["test_users"] == {"total_with_ge1_eval_row": len(ev_users),
                               "warm_ge1_eligible_train_example": sum(led.nelig[u] > 0 for u in ev_users),
                               "cold_zero_eligible_train_examples": sum(led.nelig[u] == 0 for u in ev_users)}


def _synthetic_census(led):
    tr = [r for r in led.rows if r["split"] == "TEST"]
    blk = {"decisions": len(tr), "has_observed_target": sum(r["has_observed_target"] for r in tr),
           "loss_mask_true": sum(r["loss_mask"] for r in tr), "eval_mask_true": sum(r["eval_mask"] for r in tr),
           "target_oov_true": sum(r["target_oov"] for r in tr),
           "target_class_ge0": sum(r["target_class"] >= 0 for r in tr),
           "distinct_users": len({r["user_id"] for r in tr}),
           "target_status_counts": {s: sum(r["target_status"] == s for r in tr) for s in sorted(
               {r["target_status"] for r in tr})}}
    ev_users = {r["user_id"] for r in tr if r["eval_mask"]}
    tu = {"total_with_ge1_eval_row": len(ev_users),
          "warm_ge1_eligible_train_example": sum(led.nelig[u] > 0 for u in ev_users),
          "cold_zero_eligible_train_examples": sum(led.nelig[u] == 0 for u in ev_users)}
    return {"per_old_split": {"TEST": blk}, "v4_retro_test": copy.deepcopy(blk)}, {"s5": {"test_users": tu}}


def test_g0_integrity_matches_a_synthetic_census(synth_build, led):
    sc, cc = _synthetic_census(led)
    res = F.g0_integrity(synth_build["rec"]["full_population_retro_test_counters"], sc, cc)
    assert res["all_match"] and res["n_compared"] >= 20, res


@nc("a census that differs by one TEST decision must fail the G0 integrity comparison")
def test_nc_g0_integrity_detects_a_mismatch(synth_build, led):
    sc, cc = _synthetic_census(led)
    sc["per_old_split"]["TEST"]["decisions"] += 1
    assert F.g0_integrity(synth_build["rec"]["full_population_retro_test_counters"], sc, cc)["all_match"]


# ------------------------------------------------------------------------------------------ refusals
def test_censor_rule_stops_on_an_unmatched_retro_test_row(synth_build):
    b = synth_build["builder"]
    E = b.E
    r = {"status": np.array([E.c("target_status", "BOUNDARY_UNRESOLVED")]),
         "nsplit": np.array([E.c("next_split", "NONE")]), "vs": np.array([False]), "eval": np.array([False]),
         "has_obs": np.array([False])}
    assert_raises(BuildBlocked, b._censor_test, r, np.array([True]))
    r2 = dict(r, status=np.array([E.c("target_status", "OBSERVED_MATURED_IN_GRACE")]))
    assert_raises(BuildBlocked, b._censor_test, r2, np.array([False]))       # grace outside TEST


def test_gated_emitter_refuses_foreign_ids_and_foreign_gates(synth_build, synth_gate, tmp_path):
    import dataclasses
    b = synth_build["builder"]
    m = synth_build["rec"]["_man"][D9.MAIN_TEST_ID][0]
    assert_raises(BuildBlocked, D9.encode_tables_d9, b, "MAIN_VALIDATION_5_100_131072", m, synth_gate)
    assert_raises(TestGateRefused, D9.encode_tables_d9, b, D9.MAIN_TEST_ID, m, None)
    enc = D9.encode_tables_d9(b, D9.MAIN_TEST_ID, m, synth_gate)
    h = dict(load_json(synth_build["out"] / "manifests" / f"{D9.MAIN_TEST_ID}.header.json"),
             test_exposure_register_entry_id=99)
    assert_raises(TestGateRefused, D9.write_manifest_d9, tmp_path, D9.MAIN_TEST_ID, enc, h, synth_gate)
    forged = dataclasses.replace(synth_gate, entry_id=99)
    assert_raises(TestGateRefused, D9.write_manifest_d9, tmp_path, D9.MAIN_TEST_ID, enc, h, forged)
    assert not (tmp_path / "manifests").exists()


def test_the_locked_field_dictionary_is_untouched():
    from v4data.manifests import FIELD_TYPES
    assert "grace_matured" not in FIELD_TYPES and D9.FIELD_TYPES_D9["grace_matured"] == ("bool", None)
    assert {k: v for k, v in D9.FIELD_TYPES_D9.items() if k != "grace_matured"} == FIELD_TYPES


def test_synthetic_bands_are_restored_after_use():
    from v4data06 import d6_common as DC
    with synthetic_bands():
        assert DC.N_MAIN == SYN["N_MAIN"]
    assert DC.N_MAIN == 131_072 and DC.BANDS is FC.BANDS and DC.N_PANEL == 16_384


# ------------------------------------------------------------------------------------------ negative controls
def _tampered(sb, mid, fn):
    t = _t(sb, mid)
    dec = {k: v.copy() for k, v in t["decisions"].items()}
    usr = {k: v.copy() for k, v in t["users"].items()}
    fn(dec, usr)
    return m07_assertions(usr, dec)


@nc("a VALIDATION row planted into a RETRO_TEST manifest must violate the M07 assertions")
def test_nc_validation_row_breaks_m07(synth_build):
    def f(dec, usr):
        dec["old_split"] = dec["old_split"].astype(object)
        dec["old_split"][0] = "VALIDATION_CONFIRM"
        dec["decision_ts_us"][0] = dec["decision_ts_us"][0] - 5 * 86_400_000_000
    assert all(_tampered(synth_build, D9.MAIN_TEST_ID, f).values())


@nc("a grace target at or after the end of LABEL_GRACE must violate the M07 assertions")
def test_nc_grace_target_after_label_grace(synth_build):
    def f(dec, usr):
        i = int(np.flatnonzero(dec["grace_matured"].astype(bool))[0])
        dec["target_ts_us"][i] = GRACE_END_US
    mid = next(m for m in D9.TEST_IDS if _t(synth_build, m)["decisions"]["grace_matured"].any())
    assert all(_tampered(synth_build, mid, f).values())


@nc("a censored row marked evaluable must violate the M07 assertions")
def test_nc_censored_row_marked_eval(synth_build):
    def f(dec, usr):
        i = int(np.flatnonzero(~dec["eval_mask"].astype(bool))[0])
        dec["eval_mask"][i] = True
    assert all(_tampered(synth_build, D9.MAIN_TEST_ID, f).values())


@nc("eval_user_key not densified by client_key_hex order must violate the D6 rule")
def test_nc_eval_user_key_not_dense(synth_build):
    def f(dec, usr):
        usr["client_key_hex"] = usr["client_key_hex"][::-1].copy()
    assert all(_tampered(synth_build, D9.MAIN_TEST_ID, f).values())


@nc("a population that includes a non-band user's TEST rows is not the MAIN band's RETRO_TEST row set")
def test_nc_main_rows_with_a_foreign_user(synth_build, led):
    t = _t(synth_build, D9.MAIN_TEST_ID)
    pop = _population(D9.MAIN_TEST_ID, led) + ["testonly-0"]
    assert t["decisions"]["decision_id"].tolist() == sorted(r["decision_id"] for r in led.test_rows(pop))
