"""Adapter tests: counts = statistics.csv, chronological sequences, the inner validation rule, the band, and the
holdout read only by load_holdout_view."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ppsi.benchrun import data as D


def _load(rel):
    return D.load(rel["name"], rel["root"])


def test_counts_equal_statistics(release):
    d = _load(release)
    chk = D.statistics_check(release["name"], release["root"], d)
    assert chk["ok"], chk
    assert d.U == release["stats"]["n_users"]
    assert int(d.full_items.size) == release["stats"]["leave_one_out_n_train_interactions"]
    assert int(d.seq_items.size) == int(d.full_items.size) - d.U
    assert d.N == d.full_items.size - 2 * d.U
    assert d.K == len({r[1] for r in release["train_rows"]})
    assert int(np.diff(d.user_dec_offsets).sum()) == d.N


def test_sequences_are_chronological_per_user(release):
    d = _load(release)
    fr = pd.DataFrame(release["train_rows"], columns=["user_id", "item_id", "datetime", "weight"])
    fr["_o"] = np.arange(len(fr))
    for ui, uid in enumerate(d.user_ids):
        want = fr[fr.user_id == uid].sort_values(["datetime", "_o"], kind="stable")["item_id"].to_numpy()
        got = d.catalog[d.full_items[d.full_offsets[ui]:d.full_offsets[ui + 1]]]
        assert np.array_equal(want, got)


def reference_val_mask(interactions):
    """The reference trainer's leave-one-out validation rule: rank rows by datetime DESCENDING with a stable sort per
    user and take rank 0. The Series '&' re-aligns the rank-0 flags to the original row order (file order)."""
    users = interactions["user_id"].unique()
    rank = (interactions.sort_values("datetime", ascending=False, kind="stable")
            .groupby("user_id", sort=False).cumcount())
    return ((interactions["user_id"].isin(users)) & (rank == 0)).values


def test_inner_validation_equals_the_reference_rule_including_timestamp_ties(release):
    d = _load(release)
    fr = pd.read_csv(release["root"] / release["name"] / "leave_one_out" / "train.csv", parse_dates=["datetime"])
    want = fr[reference_val_mask(fr)].set_index("user_id")["item_id"]
    got = pd.Series(d.catalog[d.val_item], index=d.user_ids)
    assert (want.sort_index().to_numpy() == got.sort_index().to_numpy()).all() and len(want) == d.U
    assert d.stats["n_val_row_not_last"] > 0                 # the fixture has ties, so the rule is really exercised
    for u in range(d.U):     # the validation row leaves the inner sequence and keeps its place in the FULL sequence
        full = d.full_items[d.full_offsets[u]:d.full_offsets[u + 1]]
        inner = d.seq_items[d.seq_offsets[u]:d.seq_offsets[u + 1]]
        assert np.array_equal(np.delete(full, d.val_pos[u]), inner) and full[d.val_pos[u]] == d.val_item[u]


def test_nc_last_row_rule_would_differ_on_ties(release):
    """Negative control: 'the last train row' is NOT the reference validation row when timestamps tie."""
    d = _load(release)
    last = d.full_items[d.full_offsets[1:] - 1]
    assert (last != d.val_item).any()


def test_validation_view_context_seen_and_target(release):
    d = _load(release)
    v = d.validation_view()
    assert len(v.users) == d.U
    for j, u in enumerate(v.users):
        seq = d.seq_items[d.seq_offsets[u]:d.seq_offsets[u + 1]]
        assert v.targets[j] == d.val_item[u] and v.ends[j] == len(seq) == d.lengths[u] - 1
    sel = np.arange(len(v.users))
    b = v.batch(sel)
    for j, u in enumerate(v.users):
        seq = d.seq_items[d.seq_offsets[u]:d.seq_offsets[u + 1]]
        ctx = seq[-d.max_len:]
        n = int(b["lengths"][j])
        assert n == len(ctx)
        assert np.array_equal(b["item_tokens"][j, :n].numpy() - D.ITEM_OFFSET, ctx)
        assert not b["attention_mask"][j, n:].any() and (b["item_tokens"][j, n:] == 0).all()
        assert np.array_equal(b["position_ids"][j, :n].numpy(), np.arange(1, n + 1))
    rows, items = v.seen_pairs(sel)
    for j, u in enumerate(v.users):                      # seen = the inner history (the validation row is the target)
        seq = d.seq_items[d.seq_offsets[u]:d.seq_offsets[u + 1]]
        assert sorted(items[rows == j]) == sorted(seq)


def test_train_decisions_never_contain_the_validation_target(release):
    d = _load(release)
    for u in range(d.U):
        a, b = d.dec_range(u)
        L = int(d.lengths[u])
        assert b - a == max(L - 2, 0)
        assert (d.dec_pos[a:b] <= L - 2).all() and (d.dec_pos[a:b] >= 1).all()


def test_band_size_seed_and_disjointness(release):
    d = _load(release)
    assert int(d.band.sum()) == int(np.floor(0.15 * d.U + 0.5))
    again = D.band_users(d.user_ids)
    assert np.array_equal(again, d.user_ids[d.band])
    assert not np.array_equal(D.band_users(d.user_ids, seed=2027), again)
    rows = d.band_rows()
    assert (d.band[d.dec_user[rows]]).all()
    rest = np.setdiff1d(np.arange(d.N), rows)
    assert not d.band[d.dec_user[rest]].any()
    assert np.intersect1d(rows, rest).size == 0 and rows.size + rest.size == d.N


def test_load_does_not_read_the_holdout(release):
    (release["root"] / release["name"] / "leave_one_out" / "holdout.csv").unlink()
    d = _load(release)                                  # the train side loads without any holdout file
    assert d.U == release["stats"]["n_users"]


def test_test_view_population_candidates_and_context(release):
    d = _load(release)
    tv = D.load_holdout_view(d, release["root"])
    n_hold = len(release["holdout_rows"])
    assert len(tv.users) == n_hold < d.U                 # only users that HAVE a holdout row are scored
    assert tv.source == "full" and (tv.ends == d.lengths[tv.users]).all()
    ho = pd.DataFrame(release["holdout_rows"], columns=["user_id", "item_id", "datetime", "weight"]).set_index("user_id")
    for j, u in enumerate(tv.users):
        assert d.catalog[tv.targets[j]] == ho.loc[d.user_ids[u], "item_id"]        # one target per user
    sel = np.arange(len(tv.users))
    rows, items = tv.seen_pairs(sel)
    for j, u in enumerate(tv.users):                      # seen = the WHOLE train history, validation row included
        full = d.full_items[d.full_offsets[u]:d.full_offsets[u + 1]]
        assert sorted(items[rows == j]) == sorted(full) and d.val_item[u] in items[rows == j]
    b = tv.batch(sel)
    for j, u in enumerate(tv.users):                      # context = last <= 50 of the FULL sequence (no retrain)
        full = d.full_items[d.full_offsets[u]:d.full_offsets[u + 1]][-d.max_len:]
        n = int(b["lengths"][j])
        assert np.array_equal(b["item_tokens"][j, :n].numpy() - D.ITEM_OFFSET, full)
    assert np.all(np.diff(d.catalog) > 0)                 # class order = ascending external item id (tie rule)


def test_cold_holdout_items_are_kept_as_misses(tmp_path):
    from benchrun_testkit import make_release
    rel = make_release(tmp_path / "p", cold_holdout=3)
    d = _load(rel)
    tv = D.load_holdout_view(d, rel["root"])
    assert tv.n_cold_items == 3 and int((tv.targets < 0).sum()) == 3 and tv.n_unknown_users == 0


def test_nc_wrong_sort_would_be_detected(release):
    """Negative control: sequences in FILE order differ from chronological (the fixture shuffles the file)."""
    fr = pd.read_csv(release["root"] / release["name"] / "leave_one_out" / "train.csv")
    d = _load(release)
    file_order = fr.groupby("user_id", sort=True)["item_id"].apply(list)
    diff = sum(not np.array_equal(np.asarray(file_order.iloc[i]),
                                  d.catalog[d.seq_items[d.seq_offsets[i]:d.seq_offsets[i + 1]]])
               for i in range(d.U))
    assert diff > 0
