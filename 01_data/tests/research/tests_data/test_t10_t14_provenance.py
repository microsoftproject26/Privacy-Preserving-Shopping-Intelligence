"""T10 and T14 (PKG/12_ACCEPTANCE_TESTS.md L15, L19) as they apply to the sidecar over frozen v1.

Proves: every frozen statistic is TRAIN-fitted and quoted, tokens stay inside the TRAIN vocabularies, a
future-only value is an OOV token (never a known index); every kept channel carries a locality label; the
LOCAL_HISTORY counts of a user come only from that user's own earlier events, with the v1 replay-stream semantics
(TRAIN, then the decision's own legacy partition; the old VS events are NOT in a VC decision's history, disclosure
B9); item observed age is anchored to one frozen TRAIN first-seen time per product.
Does not prove: privacy (pooled bootstrap information is present by design) or anything about TEST.
"""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from data_testkit import REAL_DS, nc
from v4data.common import TRAIN_END_US
from v4data.manifests import read_non_test
from v4data.transform_state import build_state


# ------------------------------------------------------------------------------------------------ T10
def test_t10_frozen_statistics_are_train_fitted_and_quoted(schema):
    st = build_state(REAL_DS, schema)
    assert st["fit"]["split"] == "TRAIN" and st["fit"]["population"] == "FULL_TRAIN"
    assert "fits nothing" in st["fit"]["source"]
    assert st["f1"]["new_std"] == 0.9945344391661521 and st["f1"]["n_valid"] == 48434230


def _vocab_rows(name):
    return pq.ParquetFile(REAL_DS / "mappings" / f"{name}.parquet").metadata.num_rows


def test_t10_tokens_inside_train_vocabularies(real_store):
    ds = real_store.decision_set("VALIDATION_SELECT", "eval")
    b = real_store.batch(ds, np.arange(min(4096, len(ds))))
    lim = {"item_tokens": 158486 + 3, "category_tokens": _vocab_rows("vocab_category_id") + 3,
           "brand_tokens": _vocab_rows("vocab_brand") + 3, "main_category_tokens": _vocab_rows("vocab_main_category") + 3}
    for k, n in lim.items():
        t = b[k].numpy()
        assert t.max() < n and t.min() >= 0, k


def test_t10_future_only_values_are_oov(fake_build):
    d = fake_build["man"]["M02_VALIDATION_WARM"]["tables"]["decisions"]
    ts = d["decision_ts_us"]
    from data_testkit import _ts
    i = int(np.flatnonzero(ts == _ts("2019-11-16T00:00:00"))[0])       # target rare-x, never seen in TRAIN
    assert d["target_category_state"].strings()[i] == "OOV"
    j = int(np.flatnonzero(ts == _ts("2019-11-20T00:00:00"))[0])       # target pFm: brand MISSING
    assert d["target_brand_state"].strings()[j] == "MISSING"


def _valid_mean(split_set):
    t = read_non_test(REAL_DS / "model_ready" / "user_bucket=0" / "features.parquet",
                      ["split", "log1p_time_since_prev_user_event_s", "prev_user_event_available"])
    m = np.isin(t["split"].to_numpy(zero_copy_only=False), split_set) & (t["prev_user_event_available"].to_numpy() > 0)
    x = np.clip(t["log1p_time_since_prev_user_event_s"].to_numpy().astype(np.float64)[m], 0.0, 14.877890586853027)
    return float(x.mean())


@nc("a fit that pools TRAIN with VALIDATION rows is not the TRAIN-only fit")
def test_nc_t10_fit_with_validation(spec):
    assert abs(_valid_mean(["TRAIN"]) - _valid_mean(["TRAIN", "VALIDATION_SELECT"])) < 1e-9


# ------------------------------------------------------------------------------------------------ T14
def test_t14_every_kept_channel_has_a_locality_label(schema):
    kept = [c for c in schema["channels"] if c["decision"] != "DROP_REDUNDANT"]
    assert len(kept) == 53
    labels = {c["locality"] for c in kept}
    assert labels <= {"LOCAL_EVENT", "LOCAL_HISTORY", "SHARED_CATALOGUE", "SHARED_TRAIN_STATISTIC"}
    assert "UNSUPPORTED_FUTURE" not in labels


def _replay_frame():
    t = read_non_test(REAL_DS / "model_ready" / "user_bucket=0" / "features.parquet",
                      ["user_id", "split", "block_ts", "decision_id", "log1p_prev_view_count", "has_history"])
    u = pc.index_in(t["user_id"], value_set=pc.unique(t["user_id"])).to_numpy()
    sp = t["split"].to_numpy(zero_copy_only=False)
    sc = np.select([sp == "TRAIN", sp == "VALIDATION_SELECT", sp == "VALIDATION_CONFIRM"], [0, 1, 2], -1)
    ts = pc.cast(t["block_ts"], pa.int64()).to_numpy()
    did = t["decision_id"].to_numpy()
    cnt = np.rint(np.expm1(t["log1p_prev_view_count"].to_numpy().astype(np.float64))).astype(np.int64)
    hh = t["has_history"].to_numpy()
    return u, sc, ts, did, cnt, hh


def _rank_within(keys_sorted_groups):
    g = keys_sorted_groups
    start = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]]))
    idx = np.arange(g.size)
    return idx - np.repeat(start, np.diff(np.append(start, g.size)))


def _expected_counts(u, sc, ts, did, include_vs_in_vc=False):
    """The v1 replay (pipeline6.py:42-47): TRAIN stream; VS/VC streams = the user's TRAIN events then that
    partition's earlier events. `include_vs_in_vc` is the WRONG variant used by the negative control."""
    o = np.lexsort((did, ts, sc, u))
    uo, so = u[o], sc[o]
    rank = _rank_within(uo * 4 + so)
    ntrain = np.bincount(u[sc == 0], minlength=u.max() + 1)
    nvs = np.bincount(u[sc == 1], minlength=u.max() + 1)
    exp = np.where(so == 0, rank, ntrain[uo] + rank)
    if include_vs_in_vc:
        exp = np.where(so == 2, exp + nvs[uo], exp)
    out = np.empty_like(exp)
    out[o] = exp
    return out


def test_t14_history_counts_are_own_user_replay():
    u, sc, ts, did, cnt, hh = _replay_frame()
    exp = _expected_counts(u, sc, ts, did)
    assert np.array_equal(cnt, exp), "a user's history count includes events that are not its own replay stream"
    assert np.array_equal(hh > 0, exp > 0)


@nc("counting across users (a pooled replay) contaminates LOCAL_HISTORY")
def test_nc_t14_cross_user_counting():
    u, sc, ts, did, cnt, hh = _replay_frame()
    o = np.lexsort((did, ts, sc))
    pooled = np.empty_like(cnt)
    pooled[o] = _rank_within(sc[o])
    assert np.array_equal(cnt, pooled)


@nc("VC history built with the old VS events is not the frozen minimal-diff replay (disclosure B9)")
def test_nc_t14_vs_events_in_vc_history():
    u, sc, ts, did, cnt, hh = _replay_frame()
    assert np.array_equal(cnt, _expected_counts(u, sc, ts, did, include_vs_in_vc=True))


def _age_frame():
    t = read_non_test(REAL_DS / "model_ready" / "user_bucket=0" / "features.parquet",
                      ["split", "block_ts", "product_idx", "log1p_item_observed_age_s", "item_age_unavailable"])
    sp = t["split"].to_numpy(zero_copy_only=False)
    ts = pc.cast(t["block_ts"], pa.int64()).to_numpy() / 1e6
    pid = t["product_idx"].to_numpy()
    age = t["log1p_item_observed_age_s"].to_numpy().astype(np.float64)
    un = t["item_age_unavailable"].to_numpy()
    return sp, ts, pid, age, un


def test_t14_item_age_single_frozen_train_anchor():
    sp, ts, pid, age, un = _age_frame()
    ok = (~un) & (pid >= 3)
    anchor = ts[ok] - np.expm1(age[ok])
    p = pid[ok]
    o = np.argsort(p, kind="stable")
    p, anchor = p[o], anchor[o]
    start = np.flatnonzero(np.concatenate([[True], p[1:] != p[:-1]]))
    mx = np.maximum.reduceat(anchor, start)
    mn = np.minimum.reduceat(anchor, start)
    assert np.all(mx - mn < 30.0), "an item has more than one first-seen anchor"
    assert np.all(mx < TRAIN_END_US / 1e6), "an anchor lies after TRAIN_END (first seen outside TRAIN)"
    assert np.all(age[ok] > 0) and np.all(age[un] == 0)


@nc("a first-seen over all months would make items first seen in VALIDATION available later in VALIDATION")
def test_nc_t14_all_months_first_seen():
    """Uses the raw product identity (transiently, in memory) because items never seen in TRAIN share the OOV token."""
    t = read_non_test(REAL_DS / "model_ready" / "user_bucket=0" / "features.parquet",
                      ["split", "block_ts", "product_id", "item_age_unavailable"])
    sp = t["split"].to_numpy(zero_copy_only=False)
    ts = pc.cast(t["block_ts"], pa.int64()).to_numpy()
    pid = t["product_id"].to_pylist()
    un = t["item_age_unavailable"].to_numpy()
    first_any = {}
    for i in np.argsort(ts, kind="stable").tolist():
        first_any.setdefault(pid[i], ts[i])
    idx = np.flatnonzero((sp != "TRAIN") & un)
    assert idx.size > 0, "fixture has no never-seen-in-TRAIN items"
    leak_available = np.array([ts[i] > first_any[pid[i]] for i in idx.tolist()])
    assert not np.any(leak_available)
