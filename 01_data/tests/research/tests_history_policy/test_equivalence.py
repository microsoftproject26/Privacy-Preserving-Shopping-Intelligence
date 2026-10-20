"""Equivalence and monotonicity proofs (real bucket-0 rows unless stated).

* Monotonicity (exhaustive): USER length >= SESSION length, with equality iff the user has no earlier segment
  (rank + 1 == position_in_segment) or the session already holds L events.
* A decision whose user has no earlier segment: USER_HISTORY_L50 batch == SESSION_L50 batch, bitwise.
* SESSION_L50 == the snapshot loader, bitwise: (a) the INT-01 / v1 streaming CleanV2Store over the decision splits
  only; (b) the vendored rees46_eda DecisionWindowLoader.collate (loaders.py L263-L270) on the v1-rich batch.
* A USER window does not depend on which later partitions the store holds (TRAIN decisions from a TRAIN store and
  from a TRAIN + VALIDATION store are bitwise equal).
* Only the input window changes: user_context, masks, targets, masks and ids are bitwise equal across policies, and
  the last position is the current event under both.
"""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
import torch

from histkit import POLICIES, REAL_DS, batches_equal, synthetic_index, take_rows
from v4data.common import VALIDATION_SPLITS
from v4data.manifests import read_non_test
from v4hist.policy import CONTEXT_LENGTH as L, SESSION_L50, USER_HISTORY_L50
from v4hist.stream import HistoryStore
from v4hist.windows import USER_ONLY_KEYS, session_lengths, user_lengths

B = 256


def _mono(ix, rows):
    u = user_lengths(ix, rows).astype(np.int64)
    s = session_lengths(ix, rows).astype(np.int64)
    assert np.all(u >= s), "a USER window is shorter than its SESSION window"
    no_earlier = (ix.user_rank[rows].astype(np.int64) + 1) == ix.position[rows]
    full = ix.position[rows] >= L
    assert np.array_equal(u == s, no_earlier | full), "equality is not exactly (no earlier segment) or (session >= L)"
    return int(no_earlier.sum()), int(full.sum()), int((u > s).sum())


def test_monotonicity_exhaustive_real_and_synthetic(hist_val):
    ix = hist_val.index
    n_no, n_full, n_longer = _mono(ix, np.arange(ix.n_rows))
    assert n_no > 0 and n_full > 0 and n_longer > 0.3 * ix.n_rows
    syn = synthetic_index(seed=3, n_users=300)
    _mono(syn, np.arange(syn.n_rows))


def _sets(hist_train, hist_val):
    return (("TRAIN", hist_train, "loss"), ("VALIDATION", hist_val, "eval"))


def test_no_earlier_segment_user_equals_session_bitwise(hist_train, hist_val):
    rng = np.random.default_rng(1)
    for name, hs, kind in _sets(hist_train, hist_val):
        du, dsess = hs.decision_set(kind), hs.raw_decision_set(kind)
        ix = hs.index
        cand = np.flatnonzero((ix.user_rank[du.end_row].astype(np.int64) + 1) == ix.position[du.end_row])
        assert cand.size > 1_000, name
        pos = rng.choice(cand, min(cand.size, 80 * B), replace=False)
        for s in range(0, pos.size, B):
            p = pos[s:s + B]
            bu, bs = dict(hs.batch(du, p)), dict(hs.batch(dsess, p))
            for k in USER_ONLY_KEYS:                       # USER-only keys (DECISION_LOG 47/48), equal here too
                a, b = bu.pop(k), bs.pop(k)
                assert torch.equal(a, b), (name, k)
            tok = hs.batch(du, p)["session_offset_tokens"]  # one session: every valid token is 1
            assert np.array_equal(tok.numpy(), bu["attention_mask"].numpy().astype(np.int64)), name
            ok, why = batches_equal(bu, bs)
            assert ok, (name, why)


def test_session_equals_the_snapshot_streaming_store_bitwise(hist_val, hist_train_on_full, real_train_store,
                                                             real_val_store):
    """SESSION_L50 through v4hist on the TRAIN + VALIDATION store == the INT-01 / v1 CleanV2Store over the decision
    splits only (20,480 random rows + whole batches + the tail, per split)."""
    from v4int.cache import merged_decision_set
    rng = np.random.default_rng(2)
    pairs = ((HistoryStore(hist_val.store, hist_val.index, SESSION_L50, VALIDATION_SPLITS), "eval", real_val_store,
              VALIDATION_SPLITS),
             (HistoryStore(hist_train_on_full.store, hist_train_on_full.index, SESSION_L50, ("TRAIN",)), "loss",
              real_train_store, ("TRAIN",)))
    for hs, kind, ref, splits in pairs:
        ds = hs.decision_set(kind)
        rds = merged_decision_set(ref, splits, kind)
        assert np.array_equal(ds.decision_id, rds.decision_id)
        assert np.array_equal(ds.length, rds.length)
        n = len(ds)
        batches = [rng.choice(n, B, replace=False) for _ in range(80)]
        batches += [np.arange(0, B), np.arange(B, 2 * B), np.arange((n // B) * B, n)]
        for p in batches:
            ok, why = batches_equal(hs.batch(ds, p), ref.batch(rds, p))
            assert ok, (kind, why)


def test_session_equals_the_vendored_loader_collate(hist_val, tmp_path):
    """The v1 batch under SESSION_L50 == rees46_eda.loaders.DecisionWindowLoader.collate (vendored, unmodified), on
    every decision of 300 bucket-0 users. The loader's BucketStore reads a copy of those users' non-TEST feature rows
    (read_non_test), so no TEST row is materialised; windows never leave a user, so the subset is exact."""
    from rees46_eda.loaders import BucketStore, DecisionWindowLoader
    f = REAL_DS / "model_ready" / "user_bucket=0" / "features.parquet"
    names = pq.read_schema(f).names
    t = read_non_test(f, names)
    users = pc.unique(t["user_id"])
    chosen = users.take(pa.array(np.sort(np.random.default_rng(9).choice(len(users), 300, replace=False))))
    sub = t.filter(pc.is_in(t["user_id"], value_set=chosen))
    del t
    p = tmp_path / "features_subset.parquet"
    pq.write_table(sub, p)
    loader = DecisionWindowLoader.__new__(DecisionWindowLoader)    # only the collate state; no clients mapping read
    loader.L = L
    from rees46_eda.loaders import Normalizers
    loader.normalizers = Normalizers.load(REAL_DS / "mappings" / "normalizers.json")
    loader.client_idx_of_user = {}
    loader._buckets = {0: BucketStore(p)}
    ids = set(sub["decision_id"].to_numpy().tolist())
    ds = HistoryStore(hist_val.store, hist_val.index, SESSION_L50, VALIDATION_SPLITS).decision_set("eval")
    dtr = HistoryStore(hist_val.store, hist_val.index, SESSION_L50, ("TRAIN",)).decision_set("loss")
    n_cmp = 0
    for d in (ds, dtr):
        pos = np.flatnonzero(np.isin(d.decision_id, np.fromiter(ids, dtype=np.int64)))
        assert pos.size > 500
        for s in range(0, pos.size, B):
            pp = pos[s:s + B]
            mine = hist_val.store.v1_batch(d, pp)
            recs = [{"user_bucket": 0, "decision_id": int(d.decision_id[i]), "target_class": int(d.target_class[i]),
                     "loss_mask": bool(d.loss_mask[i]), "eval_mask": bool(d.eval_mask[i]),
                     "client_idx": int(d.client_idx[i])} for i in pp]
            ref = loader.collate(recs)
            for k, v in mine.items():
                if not torch.is_tensor(v):
                    continue
                rv = ref[k]
                assert v.numpy().dtype == rv.dtype and v.numpy().shape == rv.shape, k
                assert np.array_equal(v.numpy(), rv), k
            n_cmp += pp.size
    assert n_cmp > 2_000


def test_user_windows_do_not_depend_on_later_partitions(hist_train, hist_train_on_full):
    a, b = hist_train.decision_set("loss"), hist_train_on_full.decision_set("loss")
    assert np.array_equal(a.decision_id, b.decision_id) and np.array_equal(a.length, b.length)
    assert not np.array_equal(a.end_row, b.end_row)                   # different stores, different row numbers
    rng = np.random.default_rng(4)
    for _ in range(80):
        p = rng.choice(len(a), B, replace=False)
        ok, why = batches_equal(hist_train.batch(a, p), hist_train_on_full.batch(b, p))
        assert ok, why


def test_only_the_input_window_changes(hist_train, hist_val):
    rng = np.random.default_rng(6)
    same = ("user_context", "user_context_masks", "target_class", "loss_mask", "eval_mask", "decision_id", "client_id")
    for name, hs, kind in _sets(hist_train, hist_val):
        du, dsess = hs.decision_set(kind), hs.raw_decision_set(kind)
        for f in ("decision_id", "end_row", "target_class", "loss_mask", "eval_mask", "client_idx", "prefix_position"):
            assert np.array_equal(getattr(du, f), getattr(dsess, f)), (name, f)
        longer = 0
        for _ in range(40):
            p = rng.choice(len(du), B, replace=False)
            bu, bs = hs.batch(du, p), hs.batch(dsess, p)
            for k in same:
                assert torch.equal(bu[k], bs[k]), (name, k)
            r = torch.arange(B)
            lu, ls = bu["last_valid_index"], bs["last_valid_index"]
            for k in ("item_tokens", "category_tokens", "brand_tokens", "daypart_tokens", "weekday_tokens"):
                assert torch.equal(bu[k][r, lu], bs[k][r, ls]), (name, k)       # the current event is last in both
            for k in ("event_numeric_features", "event_quality_flags"):
                assert torch.equal(bu[k][r, lu], bs[k][r, ls]), (name, k)
            longer += int((bu["lengths"] > bs["lengths"]).sum())
        assert longer > 0, name


def test_take_rows_helper():
    b = {"x": torch.arange(6).view(3, 2), "n": "tag"}
    t = take_rows(b, [2, 0])
    assert t["x"].tolist() == [[4, 5], [0, 1]] and t["n"] == "tag"


@pytest.mark.parametrize("policy", POLICIES)
def test_lengths_within_bounds(hist_val, policy):
    ix = hist_val.index
    ln = user_lengths(ix, np.arange(ix.n_rows)) if policy == USER_HISTORY_L50 else session_lengths(ix, np.arange(ix.n_rows))
    assert int(ln.min()) >= 1 and int(ln.max()) <= L and ln.dtype == np.int16
