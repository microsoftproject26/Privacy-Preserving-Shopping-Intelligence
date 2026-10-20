"""Hashing, identity, no-raw-id, transform-state, snapshot-isolation and census-constant tests (DATA-04).

Proves: the table_content_sha256 byte layout of MANIFEST_SPEC; enum and string encodings hash identically; the
salted client key and selection key follow CLIENT_CONTRACT C1.2/C2.2 exactly (including digests with trailing zero
bytes); the builder refuses raw-id and free-string columns; TRANSFORM_STATE self-hash and tamper detection; the
snapshot is byte-unchanged and isolated; the quoted census constants equal the census files (local only).
Does not prove: anonymity (the key is a pseudonym; CLIENT_CONTRACT C1.5).
"""
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from data_testkit import REAL_DS, assert_raises, nc
from v4data import reference_counts as REF
from v4data.common import (SALT, EnumValues, V4, digests_of, hex_of, population_sha256, selection_keys_of,
                           table_content_sha256)
from v4data.manifests import BuildBlocked, ManifestBuilder


def test_table_content_sha256_layout():
    cols = [("a", "int64", np.array([1, -2], dtype=np.int64)), ("s", "string", np.array(["x", "yz"], dtype=object)),
            ("b", "bool", np.array([True, False]))]
    h = hashlib.sha256()
    for name, code, payload in (("a", 1, struct.pack("<qq", 1, -2)),
                                ("s", 6, struct.pack("<I", 1) + b"x" + struct.pack("<I", 2) + b"yz"),
                                ("b", 5, bytes([1, 0]))):
        h.update(struct.pack("<I", len(name)) + name.encode() + struct.pack("<B", code) + payload)
    assert table_content_sha256(cols) == h.hexdigest()


def test_enum_and_string_payloads_hash_identically():
    vocab = ["NONE", "CEN_NO_NEXT", "CEN_LEGACY_BARRIER"]
    codes = np.array([0, 2, 1, 0, 0])
    e = EnumValues(codes, vocab)
    assert table_content_sha256([("c", "string", e)]) == table_content_sha256([("c", "string", e.strings())])


def test_client_key_and_selection_key():
    uid = "513101234"
    d = hashlib.sha256(SALT + b"|" + uid.encode()).digest()
    got = digests_of([uid])
    assert hex_of(got)[0] == d.hex()
    assert int(selection_keys_of(got)[0]) == int.from_bytes(d[:8], "big")
    assert population_sha256(np.array([5, 9], dtype=np.uint64)) == hashlib.sha256(
        (5).to_bytes(8, "big") + (9).to_bytes(8, "big")).hexdigest()


def test_hex_of_keeps_trailing_zero_bytes():
    d = np.zeros(1, dtype="S32")
    raw = bytes(range(1, 31)) + b"\x00\x00"
    d[0] = raw
    assert hex_of(d)[0] == raw.hex() and len(hex_of(d)[0]) == 64


@nc("numpy 'S' element access strips trailing zero bytes, so a naive hex is not the key")
def test_nc_naive_hex():
    d = np.zeros(1, dtype="S32")
    raw = bytes(range(1, 31)) + b"\x00\x00"
    d[0] = raw
    assert d[0].hex() == raw.hex()


def test_no_raw_id_columns_refused(fake_build):
    b: ManifestBuilder = fake_build["builder"]
    assert_raises(BuildBlocked, b._check_no_raw_ids, "X", "t", {"user_id": np.array(["u"], dtype=object)})
    assert_raises(BuildBlocked, b._check_no_raw_ids, "X", "t", {"decision_id": np.array([1]),
                                                               "client_key_hex": np.array(["not-a-key"], dtype=object)})
    assert_raises(BuildBlocked, b._check_no_raw_ids, "X", "t", {"old_split": np.array(["TRAIN"], dtype=object)})


@nc("a name-only raw-id check accepts a free-text string column")
def test_nc_name_only_raw_id_check(fake_build):
    b = fake_build["builder"]
    forb = set(b.spec["global_rules"]["no_raw_ids"]["forbidden_columns"])
    tbl = {"old_split": np.array(["user-513101234"], dtype=object)}
    naive_ok = not (set(tbl) & forb)
    assert not naive_ok


def test_written_fake_manifests_verify(fake_build):
    out = Path(fake_build["out"]) / "manifests"
    for mid, h in fake_build["hashes"].items():
        assert (out / f"{mid}.OK").is_file()
        header = json.loads((out / f"{mid}.header.json").read_text())
        for t in header["tables"]:
            assert t["table_content_sha256"] == h["tables"][t["table"]]["table_content_sha256"]
        assert header["all_methods_lock_sha256"] is None and header["previously_seen_evaluation_period"] is True


def test_transform_state_self_hash_and_tamper(tmp_path, schema):
    from v4data.transform_state import build_state, load_state, write_state
    p, st = write_state(tmp_path, REAL_DS)
    back = load_state(p)
    assert back["transform_state_sha256"] == st["transform_state_sha256"]
    assert back["widths"] == {"event_numeric_features": 15, "event_quality_flags": 15, "user_context": 12,
                              "user_context_masks": 4, "channels_total": 53}
    d = json.loads(p.read_text())
    d["f1"]["new_std"] = 0.9933321718650419
    p.write_text(json.dumps(d))
    assert_raises(ValueError, load_state, p)


def test_snapshot_unchanged_and_isolated(real_store):
    import ppsi_env
    ppsi_env.verify_snapshot()
    seen = ppsi_env.assert_isolated()
    assert "full_store" in seen and "rees46_eda.loaders" in seen


def _census(name):
    p = V4 / "g0" / "data_census" / name
    if not p.is_file():
        pytest.skip("census files are local-only (not on the server)")
    return json.loads(p.read_text())


def test_reference_counts_equal_census_files():
    sc, lt, cs = _census("SPLIT_COUNTS.json"), _census("LABEL_TIME_AUDIT.json"), _census("CATALOGUE_SUPPORT.json")
    cc, cp = _census("CLIENT_CENSUS.json"), _census("COHORT_PROJECTIONS.json")
    qa = V4 / "g0" / "receipts_qa01"
    cr = json.loads((qa / "COHORT_RECEIPTS.json").read_text())
    cst = json.loads((qa / "COLD_STRATA.json").read_text())
    ps = sc["per_old_split"]
    vu = sc["v4_validation_union"]
    F = {k: v[0] for k, v in REF.FULL_POPULATION.items()}
    P = {k: v[0] for k, v in REF.POPULATION.items()}
    for s in ("TRAIN", "VALIDATION_SELECT", "VALIDATION_CONFIRM"):
        assert F[f"{s}.decisions"] == ps[s]["decisions"] and F[f"{s}.distinct_users"] == ps[s]["distinct_users"]
    assert F["TRAIN.loss_mask_true"] == ps["TRAIN"]["loss_mask_true"]
    assert F["TRAIN.target_oov_true"] == ps["TRAIN"]["target_oov_true"]
    for st, n in ps["TRAIN"]["target_status_counts"].items():
        assert F[f"TRAIN.status.{st}"] == n
    for k in ("decisions", "eval_mask_true", "target_oov_true", "target_class_ge0", "distinct_users"):
        assert F[f"VALIDATION.{k}"] == vu[k]
    for st, n in vu["target_status_counts"].items():
        assert F[f"VALIDATION.status.{st}"] == n
    assert F["label_time.train_loss_target_ts_ge_train_end"] == lt["train_loss_and_target_ts_ge_train_end"]["count"]
    assert F["label_time.val_eval_target_ts_ge_val_end"] == lt["val_union_eval_and_target_ts_ge_val_end"]["count"]
    assert F["catalogue.K1_products_with_train_support"] == cs["verification"]["distinct_products_with_any_train_support_K1"]
    assert F["VALIDATION.oov_absent_from_train_s1_excluded"] == \
        cs["thresholds"]["1"]["validation_union_label_coverage"]["excluded_count"]
    s5 = cc["s5"]
    assert P["universe_size"] == cp["universe_size"] == s5["users_with_ge1_example"]
    assert P["users_in_train_zero_eligible"] == s5["users_in_train_with_zero_eligible_examples"]
    assert P["val_users_cold"] == s5["validation_union_users"]["cold_zero_eligible_train_examples"]
    for n in ("16384", "65536", "262144", "ALL"):
        assert P[f"cohort{n}.population_sha256"] == cp["cohorts"][n]["selected_keys_sha256_hex"]
    for n in ("65536", "262144"):
        c = cp["cohorts"][n]
        assert P[f"cohort{n}.eligible_examples"] == c["eligible_train_examples_total"]
        assert P[f"cohort{n}.val_eval_rows"] == c["validation_union_eval_rows_total"]
        assert P[f"cohort{n}.users_ge1_eval"] == c["users_with_ge1_validation_eval_row"]
        r = cr["cohorts"][n]
        assert P[f"cohort{n}.rankable"] == r["validation_union_label_coverage_support_ge5"]["rankable_target_count"]
        assert P[f"cohort{n}.oov"] == r["validation_union_label_coverage_support_ge5"]["oov_target_count"]
        assert P[f"cohort{n}.classes_ge1_positive"] == r["catalogue_classes_with_cohort_train_positives"][
            "classes_with_ge1_cohort_train_positive"]
        assert P[f"cohort{n}.val_classes_zero_positive"] == r["classes_as_validation_target_with_zero_cohort_train_positives"]
    for s in ("COLD_NO_TRAIN_EVENTS", "COLD_TRAIN_EVENTS_NO_LABEL"):
        assert P[f"cold.{s}.users"] == cst["VALIDATION_union"][s]["users"]
        assert P[f"cold.{s}.eval_rows"] == cst["VALIDATION_union"][s]["eval_rows"]
