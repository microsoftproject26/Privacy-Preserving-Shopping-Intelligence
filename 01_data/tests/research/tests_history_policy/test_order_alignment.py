"""The ordering assumption, proven on real rows: in the store's (user_id, segment_index, position_in_segment) order a
user's rows are one contiguous run, chronological (event time and partition never decrease), a model segment never
spans a partition fence, and the index built from the feature file is row-for-row the snapshot CompactStore."""
from __future__ import annotations

import copy

import numpy as np
import pyarrow.parquet as pq

from histkit import REAL_DS, assert_raises
from v4data.manifests import _ts_us
from v4hist.windows import (SPLIT_CODES, HistoryAlignmentError, HistoryOrderError, RowIndex, align_to_store,
                            build_row_index)

NON_TEST = ("TRAIN", "VALIDATION_SELECT", "VALIDATION_CONFIRM")


def test_real_rows_are_contiguous_and_chronological_per_user(hist_val):
    ix = hist_val.index
    assert ix.n_rows > 1_000_000 and ix.n_users > 70_000
    oc = ix.order_checks
    assert oc["ts_decreasing"] == 0 and oc["partition_decreasing"] == 0 and oc["segment_spans_partitions"] == 0
    assert oc["user_row_pairs_checked"] == ix.n_rows - ix.n_users
    uo = ix.user_ord.astype(np.int64)
    d = np.diff(uo)
    assert np.all((d == 0) | (d == 1)), "a user's rows are not one contiguous run"
    same = d == 0
    assert np.all(np.diff(ix.user_rank.astype(np.int64))[same] == 1) and np.all(ix.user_rank[1:][~same] == 0)
    assert np.all(np.diff(ix.ts_us)[same] >= 0), "event time decreases inside a user"
    assert np.all(np.diff(ix.split_code.astype(np.int64))[same] >= 0), "partition decreases inside a user"
    inseg = same & (ix.position[1:] != 1)
    assert np.all(ix.split_code[1:][inseg] == ix.split_code[:-1][inseg]), "a segment spans two partitions"
    assert np.all(ix.position[1:][inseg] == ix.position[:-1][inseg] + 1)
    # users with several segments and several partitions exist (the property is not vacuous)
    starts = np.flatnonzero(np.concatenate([[True], ~same]))
    n_seg = np.add.reduceat((ix.position == 1).astype(np.int64), starts)
    changes = np.concatenate([[False], (np.diff(ix.split_code.astype(np.int64)) != 0) & same]).astype(np.int64)
    n_part = np.add.reduceat(changes, starts) + 1
    assert int((n_seg > 1).sum()) > 10_000 and int((n_part > 1).sum()) > 5_000


def test_index_timestamps_and_rows_equal_the_decision_index(hist_val):
    """block_ts of every retained row == its decision_ts in decisions_index, and every row is a decision."""
    ix = hist_val.index
    seen = 0
    for s in NON_TEST:
        t = pq.read_table(list((REAL_DS / "decisions_index" / f"split={s}" / "user_bucket=0").glob("*.parquet")),
                          columns=["decision_id", "decision_ts", "position_in_segment"])
        rows = ix.rows_of(t["decision_id"].to_numpy())
        assert np.array_equal(ix.ts_us[rows], _ts_us(t["decision_ts"]))
        assert np.array_equal(ix.position[rows], t["position_in_segment"].to_numpy().astype(np.int32))
        assert np.all(ix.split_code[rows] == SPLIT_CODES[s])
        seen += t.num_rows
    assert seen == ix.n_rows


def test_index_is_row_for_row_the_compact_store(hist_train, hist_val, real_train_store):
    a = align_to_store(hist_val.index, hist_val.store.cs)
    assert sum(a["decisions_checked"].values()) == sum(int(v["decision_id"].size)
                                                       for v in hist_val.store.cs._split_decisions.values())
    assert set(a["decisions_checked"]) == set(NON_TEST)
    b = align_to_store(hist_train.index, real_train_store.cs)                 # an independently built TRAIN store
    assert b["decisions_checked"]["TRAIN"] > 600_000


def test_alignment_refuses_a_shifted_or_foreign_index(hist_train, hist_val):
    bad = copy.copy(hist_val.index)
    bad.decision_id = np.roll(hist_val.index.decision_id, 1)
    assert_raises(HistoryAlignmentError, align_to_store, bad, hist_val.store.cs)
    assert_raises(HistoryAlignmentError, align_to_store, hist_train.index, hist_val.store.cs)   # other partitions
    other = build_row_index(REAL_DS, [1], ("TRAIN",))
    assert_raises(HistoryAlignmentError, align_to_store, other, hist_train.store.cs)          # other bucket


def test_order_checks_refuse_non_chronological_rows():
    kw = dict(user_ord=[0, 0, 0, 1, 1], position=[1, 2, 3, 1, 2], split_code=[0, 0, 0, 0, 0])
    RowIndex.from_arrays(ts_us=[1, 2, 2, 5, 6], **kw)                          # ties are fine
    assert_raises(HistoryOrderError, RowIndex.from_arrays, ts_us=[1, 3, 2, 5, 6], **kw)
    assert_raises(HistoryOrderError, RowIndex.from_arrays, user_ord=[0, 0, 0], position=[1, 2, 3],
                  ts_us=[1, 2, 3], split_code=[0, 1, 1])                        # a segment across a fence
    assert_raises(HistoryOrderError, RowIndex.from_arrays, user_ord=[0, 0], position=[1, 1], ts_us=[1, 2],
                  split_code=[1, 0])                                            # partition goes back
    assert_raises(HistoryOrderError, RowIndex.from_arrays, user_ord=[0, 0], position=[1, 3], ts_us=[1, 2],
                  split_code=[0, 0])                                            # a position gap
