"""T03-T07 as they apply to a sidecar + manifest builder over frozen v1 (PKG/12_ACCEPTANCE_TESTS.md L8-L12).

Proves: half-open V4 bounds; VALIDATION = VS u VC by time; legacy barrier rows stay censored (not re-targeted);
TRAIN labels at/after TRAIN_END never enter M01; no GRACE/TEST decision enters a pre-lock manifest; OOV targets keep
their place (no A->rareB->C reconnection) and revisits are kept; the frozen v1 session rule (gap > 3600 s breaks,
3600 does not) holds on real data.
Does not prove: anything about a raw rebuild (none is done) or about TEST rows (never read).
"""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from data_testkit import REAL_DS, _ts, make_user, nc, standard_users, write_dataset
from v4data.common import (GRACE_END_US, TEST_END_US, TRAIN_END_US, VALIDATION_END_US, subperiod_of, utc_us,
                           v4_split_of)
from v4data.manifests import BuildBlocked, ManifestBuilder, read_non_test


def _dec(man, mid, table="decisions"):
    return man[mid]["tables"][table]


def _enum(col):
    return col.strings().tolist()


# ------------------------------------------------------------------------------------------------ T03
def test_t03_half_open_bounds():
    ts = np.array([utc_us(x) for x in ("2019-11-15T23:59:59.999999Z", "2019-11-16T00:00:00Z", "2019-11-22T23:59:59Z",
                                       "2019-11-23T00:00:00Z", "2019-11-29T23:59:59Z", "2019-11-30T00:00:00Z",
                                       "2019-09-30T23:59:59Z")])
    assert v4_split_of(ts).tolist() == [0, 1, 1, 2, 2, -1, -1]
    sub = subperiod_of(np.array([utc_us("2019-11-19T23:59:59Z"), utc_us("2019-11-20T00:00:00Z"),
                                 utc_us("2019-11-23T00:00:00Z")]))
    assert sub.tolist() == [0, 1, 2]


@nc("a closed right endpoint puts the TRAIN_END instant into TRAIN")
def test_nc_t03_closed_interval():
    ts = np.array([TRAIN_END_US])
    closed = np.where(ts <= TRAIN_END_US, 0, 1)
    assert closed.tolist() == v4_split_of(ts).tolist()


def test_t03_real_bucket0_membership_by_time():
    t = read_non_test(REAL_DS / "decisions" / "user_bucket=0" / "ledger.parquet", ["split", "decision_ts"])
    split = t["split"].to_numpy(zero_copy_only=False)
    ts = pc.cast(t["decision_ts"], pa.int64()).to_numpy()
    v4 = v4_split_of(ts)
    assert np.array_equal(v4 == 0, split == "TRAIN")
    assert np.array_equal(v4 == 1, np.isin(split, ["VALIDATION_SELECT", "VALIDATION_CONFIRM"]))
    assert np.array_equal(subperiod_of(ts) == 0, split == "VALIDATION_SELECT")
    assert not np.any(v4 == 2), "a RETRO_TEST-time decision was materialised"


def test_t03_legacy_barrier_kept_censored(fake_build):
    d = _dec(fake_build["man"], "M02_VALIDATION_WARM")
    cen = np.array(_enum(d["censor_class"]))
    st = np.array(_enum(d["target_status"]))
    assert (cen == "CEN_LEGACY_BARRIER").sum() == 1 and (cen == "CEN_RIGHT_EDGE").sum() == 1
    assert (cen == "CEN_AMBIGUOUS_NEXT_BARRIER").sum() == 1
    bu = st == "BOUNDARY_UNRESOLVED"
    assert np.all(~d["eval_mask"][bu]) and np.all(d["target_class"][bu] == -1), "a legacy-barrier row was re-targeted"
    assert np.array_equal(cen == "NONE", d["eval_mask"])


# ------------------------------------------------------------------------------------------------ T04
def test_t04_train_label_at_train_end_excluded(fake_build):
    b = fake_build["builder"]
    d = _dec(fake_build["man"], "M01_COHORT_TRAIN")
    assert np.all((d["target_ts_us"] >= 0) & (d["target_ts_us"] < TRAIN_END_US))
    assert b.recon["label_time.train_loss_target_ts_ge_train_end"] == 1, "the planted leak was not detected"
    leak_ts = _ts("2019-11-15T23:59:00")
    assert leak_ts not in set(d["decision_ts_us"].tolist())


@nc("a filter without the target_ts < TRAIN_END condition admits the planted leak")
def test_nc_t04_filter_without_label_time(fake_build):
    t = pq.ParquetFile(fake_build["root"] / "decisions" / "user_bucket=0" / "ledger.parquet").read()
    naive = pc.and_(pc.equal(t["split"], "TRAIN"), t["loss_mask"])
    ts = pc.cast(pc.filter(t["target_ts"], naive), pa.int64()).to_numpy()
    assert np.all(ts < TRAIN_END_US)


def _mini_builder(tmp_path, users, cat):
    root = write_dataset(tmp_path / "ds", users, cat)
    b = ManifestBuilder(ds_root=root, out_root=tmp_path / "out", buckets=[0], expect=False, log=lambda *_: None)
    b.pass_support(); b.pass_aggregate(); b.select_populations(); b.pass_extract()
    return b


def test_t04_validation_target_after_end_blocks(tmp_path):
    """A planted VALIDATION label at VALIDATION_END stops the build (the target-row join catches it first)."""
    users, cat = standard_users()
    bad = make_user("warm-late", [("TRAIN", "2019-10-03T09:00:00", "pA", 0), ("TRAIN", "2019-10-03T09:05:00", "pB", 0),
                                  ("VALIDATION_CONFIRM", "2019-11-22T23:58:00", "pA", 1),
                                  ("VALIDATION_CONFIRM", "2019-11-22T23:59:00", "pB", 1)],
                    overrides={2: {"target_ts": _ts("2019-11-23T00:00:00")}})
    with pytest.raises(BuildBlocked):
        _mini_builder(tmp_path, users + [bad], cat)
    assert not list((tmp_path / "out").rglob("*.OK")), "a success marker survived a failed build"


def test_t04_manifest_assertion_flags_post_validation_label(fake_build):
    b = fake_build["builder"]
    v = {k: a.copy() for k, a in b.rows_val.items()}
    coh = (v["role"] & 1) != 0
    v = {k: a[coh] for k, a in v.items()}
    i = int(np.flatnonzero(v["eval"])[0])
    v["tts"][i] = VALIDATION_END_US
    m = b._eval_manifest("M02_VALIDATION_WARM", v, b.class_pos["M01"], "warm", "M01_COHORT_TRAIN")
    assert m["assertions"]["no_eval_target_at_or_after_validation_end"] is False
    with pytest.raises(BuildBlocked):
        b._assert_manifest(m)


# ------------------------------------------------------------------------------------------------ T05
def test_t05_no_grace_or_test_decision_in_any_manifest(fake_build):
    man = fake_build["man"]
    test_ts = {_ts("2019-11-24T10:00:00"), _ts("2019-11-24T10:05:00")}
    for mid, m in man.items():
        for tname, tbl in m["tables"].items():
            for col in ("decision_ts_us", "query_decision_ts_us"):
                if col in tbl:
                    v = tbl[col]
                    assert np.all(v < VALIDATION_END_US), f"{mid}.{tname} holds a post-VALIDATION decision"
                    assert not (set(v.tolist()) & test_ts)
    assert fake_build["builder"].recon["label_time.decisions_ts_ge_2019_11_30_non_test"] == 0


def test_t05_grace_split_rows_never_materialised(tmp_path):
    users, cat = standard_users()
    g = make_user("grace-user", [("TRAIN", "2019-10-03T09:00:00", "pA", 0), ("TRAIN", "2019-10-03T09:05:00", "pB", 0),
                                 ("LABEL_GRACE", "2019-11-30T10:00:00", "pC", 1)])
    root = write_dataset(tmp_path / "ds", users + [g], cat)
    t = read_non_test(root / "decisions" / "user_bucket=0" / "ledger.parquet", ["split", "decision_ts"])
    ts = pc.cast(t["decision_ts"], pa.int64()).to_numpy()
    assert np.all(ts < GRACE_END_US) and not np.any(ts >= TEST_END_US)
    assert set(t["split"].to_pylist()) <= {"TRAIN", "VALIDATION_SELECT", "VALIDATION_CONFIRM"}


# ------------------------------------------------------------------------------------------------ T06
def test_t06_target_before_filter_no_reconnection(fake_build):
    man, cat = fake_build["man"], fake_build["catalogue"]
    d = _dec(man, "M02_VALIDATION_WARM")
    cls = {p: i for i, p in enumerate(cat)}
    a_row = np.flatnonzero(d["decision_ts_us"] == _ts("2019-11-16T00:00:00"))
    assert a_row.size == 1, "the decision at exactly TRAIN_END must be a VALIDATION decision"
    i = int(a_row[0])
    assert d["eval_mask"][i] and d["target_oov"][i] and d["target_class"][i] == -1 and not d["rankable"][i]
    assert d["oov_cause"].strings()[i] == "ABSENT_FROM_TRAIN"
    assert d["target_class"][i] != cls["pC"], "A -> rare B -> C was reconnected into A -> C"
    x = np.flatnonzero(d["decision_ts_us"] == _ts("2019-11-16T00:30:00"))
    assert d["target_class"][int(x[0])] == cls["pC"]
    y = np.flatnonzero(d["decision_ts_us"] == _ts("2019-11-21T09:00:00"))
    assert d["oov_cause"].strings()[int(y[0])] == "BELOW_SUPPORT" and d["target_train_support"][int(y[0])] == 1


def test_t06_revisit_kept(fake_build):
    d = _dec(fake_build["man"], "M01_COHORT_TRAIN")
    cls = {p: i for i, p in enumerate(fake_build["catalogue"])}
    ts = d["decision_ts_us"].tolist()
    i_ab = ts.index(_ts("2019-10-02T10:00:00"))
    i_ba = ts.index(_ts("2019-10-02T10:10:00"))
    assert d["target_class"][i_ab] == cls["pB"] and d["target_class"][i_ba] == cls["pA"], "A -> B -> A lost a step"


# ------------------------------------------------------------------------------------------------ T07
def _sessions_bucket0():
    t = read_non_test(REAL_DS / "decisions" / "user_bucket=0" / "ledger.parquet",
                      ["user_id", "decision_ts", "decision_id", "session_index"])
    u = pc.index_in(t["user_id"], value_set=pc.unique(t["user_id"])).to_numpy()
    ts = pc.cast(t["decision_ts"], pa.int64()).to_numpy()
    did = t["decision_id"].to_numpy()
    s = t["session_index"].to_numpy()
    o = np.lexsort((did, ts, u))
    u, ts, s = u[o], ts[o], s[o]
    same_u = u[1:] == u[:-1]
    gap = (ts[1:] - ts[:-1]) / 1e6
    change = s[1:] != s[:-1]
    return same_u, gap, change


def test_t07_session_gap_rule_real_bucket0():
    same_u, gap, change = _sessions_bucket0()
    assert np.all(gap[same_u] > 0), "two decisions of one user share a timestamp (ambiguity must be a barrier)"
    # gap <= 3600 s (including exactly 3600) never breaks a session; equivalently every break has a gap > 3600 s.
    # The converse is not required: a barrier block inside a long decision gap can keep the raw session alive.
    assert not np.any(same_u & (gap <= 3600) & change), "a gap <= 3600 s broke a session"
    assert np.any(same_u & change), "fixture has no session breaks"
    assert np.any(same_u & ~change & (gap > 3000) & (gap <= 3600)), "fixture has no near-threshold same-session gap"


@nc("a 7200 s rule does not describe the frozen sessions (breaks exist in (3600, 7200])")
def test_nc_t07_wrong_threshold():
    same_u, gap, change = _sessions_bucket0()
    assert not np.any(same_u & change & (gap <= 7200))
