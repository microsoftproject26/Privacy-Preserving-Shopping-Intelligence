"""Checks of a converted dataset directory (layout, dtypes, one holdout per user, no leakage; counts only)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import layout


def read_split(path):
    """Read a split file the way the benchmark runner does (int64 ids, parsed datetimes)."""
    return pd.read_csv(path, dtype={"user_id": "int64", "item_id": "int64"}, parse_dates=["datetime"])


def verify_layout(out_dir) -> dict:
    out = Path(out_dir)
    tr = read_split(out / "leave_one_out" / "train.csv")
    ho = read_split(out / "leave_one_out" / "holdout.csv")
    st = pd.read_csv(out / "statistics.csv")
    res = {}
    res["columns_ok"] = list(tr.columns) == layout.COLUMNS and list(ho.columns) == layout.COLUMNS
    res["stat_columns_ok"] = list(st.columns) == layout.STAT_COLUMNS
    res["dtypes_ok"] = (str(tr.user_id.dtype), str(tr.item_id.dtype), str(tr.rating.dtype), str(tr.weight.dtype)) == \
        ("int64", "int64", "float64", "float64") and str(tr.datetime.dtype).startswith("datetime64")
    res["one_holdout_per_user"] = bool(ho.user_id.is_unique)
    res["holdout_users_in_train"] = bool(set(ho.user_id) <= set(tr.user_id))
    res["holdout_in_user_train_pairs"] = int(pd.merge(ho[["user_id", "item_id"]], tr[["user_id", "item_id"]],
                                                     on=["user_id", "item_id"]).shape[0])
    s = st.iloc[0]
    res["stats_match"] = bool(int(s.leave_one_out_n_train_interactions) == len(tr)
                              and int(s.leave_one_out_n_holdout_interactions) == len(ho)
                              and int(s.n_interactions) == len(tr) + len(ho)
                              and int(s.n_items) == pd.concat([tr.item_id, ho.item_id]).nunique())
    res["train_sorted_by_user"] = bool((tr.user_id.diff().dropna() >= 0).all())
    res["n_train"], res["n_holdout"], res["n_users_train"] = len(tr), len(ho), int(tr.user_id.nunique())
    return res
