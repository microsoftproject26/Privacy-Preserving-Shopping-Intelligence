"""T41 golden replay (EVALUATION_CONTRACT_V4 E11.4-E11.7; PKG/12 L52) on real rows of the frozen v1 bucket 0.

With h = the full available history the E11.4 builder must reproduce the PRODUCTION batch (the USER_HISTORY_L50
StoreEvalProvider / StoreTrainProvider path) bitwise on every key, apart from the declared PARTIAL_OOV_IDENTITY cells
of rows the builder itself flags. Negative controls: a builder that leaks one extra (future) event, one without the v1
carry-in and one that breaks the v1 replay-stream rule must each FAIL the golden replay.
"""
from __future__ import annotations

import numpy as np
import pytest

from trunckit import (REAL_DS, FutureEventBuilder, MergedStreamBuilder, NoCarryBuilder, make_builder, nc,
                      val_query_rows)
from v4data.common import VALIDATION_SPLITS
from v4trunc.golden import assert_golden, compare_batches, golden_replay


def _val_ids(table, n=None):
    ids = np.sort(table.decision_id[val_query_rows(table)])
    return ids if n is None else ids[:n]


def _golden(builder, ids, splits=VALIDATION_SPLITS, kind="eval"):
    return golden_replay(builder, ids, splits=splits, kind=kind, ds_root=REAL_DS, buckets=[0])


def test_t41_golden_replay_validation_queries(builder, sample_table):
    ids = _val_ids(sample_table)
    rec = _golden(builder, ids)
    assert rec["n_compared"] == ids.size > 2000
    assert_golden(rec)
    assert rec["admitted_partial_rows"] <= sum(rec["flagged_rows"].values())


def test_t41_golden_replay_train_decisions(builder, sample_table):
    """TRAIN loss decisions (Study B support inputs use the TRAIN replay stream) reproduce the TRAIN provider path."""
    t = sample_table
    rows = np.flatnonzero((t.split_code == 0) & t.meta["loss_mask"])
    ids = np.sort(t.decision_id[rows])[::7][:1500]
    rec = _golden(builder, ids, splits=("TRAIN",), kind="loss")
    assert rec["n_compared"] == ids.size > 500
    assert_golden(rec)


def test_golden_compare_admits_only_flagged_partial_cells():
    import torch
    a = torch.zeros(3, 12)
    b = a.clone()
    b[0, 7] = 0.5                     # brand share, flagged row -> admitted
    b[1, 7] = 0.5                     # brand share, unflagged row -> a failure
    b[2, 3] = 1.0                     # another column of a flagged row -> a failure
    flags = [{"brand_oov_collision": True}, {}, {"brand_oov_collision": True}]
    c = compare_batches({"user_context": a}, {"user_context": b}, flags)
    assert c["admitted_rows"] == 1 and c["diff_rows"] == {"user_context": 2}


@nc("a builder leaking one extra (future) event must fail the golden replay")
def test_nc_golden_future_event(sample_table):
    assert_golden(_golden(make_builder(sample_table, FutureEventBuilder), _val_ids(sample_table, 600)))


@nc("window sums without the v1 bucket-stream carry-in must fail the golden replay")
def test_nc_golden_no_carry_in(sample_table):
    assert_golden(_golden(make_builder(sample_table, NoCarryBuilder), _val_ids(sample_table)))


def _vc_rows_with_vs_history(t):
    vc_users = np.unique(t.row_user[t.split_code == 2])
    vs_users = np.unique(t.row_user[t.split_code == 1])
    both = np.intersect1d(vc_users, vs_users)
    return np.flatnonzero((t.split_code == 2) & t.meta["eval_mask"] & np.isin(t.row_user, both))


def test_merged_stream_nc_has_rows(sample_table):
    """REV-ANA-01 B-F5: the merged-stream NC below is never vacuous: the sample holds VC queries with VS history."""
    assert _vc_rows_with_vs_history(sample_table).size > 0


@nc("a VALIDATION_CONFIRM query replaying VALIDATION_SELECT events must fail the golden replay")
def test_nc_golden_merged_stream(sample_table):
    t = sample_table
    rows = _vc_rows_with_vs_history(t)
    assert rows.size > 0                                   # B-F5: never skipped (precondition test above)
    assert_golden(_golden(make_builder(t, MergedStreamBuilder), np.sort(t.decision_id[rows])))
