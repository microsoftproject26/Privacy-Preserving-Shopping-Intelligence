"""Amazon Reviews 2023 official 5-core benchmark -> the leave-one-out layout (leave_one_out/train.csv + holdout.csv).

last_out_w_his (CONVERTED): per user the official split has train = all but the last two interactions, valid = the
second-last, test = the last (ordered by timestamp). train.csv = official train rows + valid rows (file order preserved,
stable by user); holdout = the official test row (exactly one per user).

timestamp_w_his (NOT converted): the official split is GLOBAL-time: train = timestamp < T1,
valid = T1 <= timestamp < T2, test = timestamp >= T2. A user has 0..184 test rows and most users have none, so it is not
one holdout per user. `timestamp_diagnostics` measures that.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import layout

CATEGORIES = {"Video_Games": "amazon2023_video_games", "Baby_Products": "amazon2023_baby_products",
              "Beauty_and_Personal_Care": "amazon2023_beauty_and_personal_care"}
USECOLS = ["user_id", "parent_asin", "rating", "timestamp"]


def _read(raw_dir, split_dir, cat, part):
    p = Path(raw_dir) / split_dir / f"{cat}.{part}.csv.gz"
    return pd.read_csv(p, usecols=USECOLS, dtype={"user_id": str, "parent_asin": str, "timestamp": np.int64})


def convert_last_out(raw_dir, cat: str, out_dir) -> dict:
    tr, va, te = (_read(raw_dir, "last_out_w_his", cat, p) for p in ("train", "valid", "test"))
    if te["user_id"].duplicated().any():
        raise ValueError("official last_out test has >1 row for a user")
    users = np.unique(np.concatenate([tr.user_id.to_numpy(), va.user_id.to_numpy(), te.user_id.to_numpy()]))
    if set(te.user_id) != set(users) or set(va.user_id) != set(users):
        raise ValueError("train/valid/test user sets differ")
    items = np.unique(np.concatenate([tr.parent_asin.to_numpy(), va.parent_asin.to_numpy(), te.parent_asin.to_numpy()]))
    train_raw = pd.concat([tr, va], ignore_index=True)           # official order: train rows, then valid rows

    def enc(df):
        return (np.searchsorted(users, df.user_id.to_numpy()) + 1, np.searchsorted(items, df.parent_asin.to_numpy()) + 1)

    u, i = enc(train_raw)
    order = np.argsort(u, kind="stable")                         # group by user, keep official (chronological) order
    train_raw, u, i = train_raw.iloc[order].reset_index(drop=True), u[order], i[order]
    train = layout.make_frame(u, i, train_raw.rating, layout.dt_from_ms(train_raw.timestamp))
    uh, ih = enc(te)
    o2 = np.argsort(uh, kind="stable")
    te = te.iloc[o2].reset_index(drop=True)
    holdout = layout.make_frame(uh[o2], ih[o2], te.rating, layout.dt_from_ms(te.timestamp))
    st = layout.write_dataset(out_dir, train, holdout, n_users=len(users))
    out = Path(out_dir)
    pd.DataFrame({"user_id": np.arange(1, len(users) + 1), "orig_user_id": users}).to_csv(out / "user_map.csv", index=False)
    pd.DataFrame({"item_id": np.arange(1, len(items) + 1), "orig_parent_asin": items}).to_csv(out / "item_map.csv", index=False)
    cold = int((~holdout["item_id"].isin(set(train["item_id"]))).sum())
    in_train = int(pd.merge(holdout[["user_id", "item_id"]], train[["user_id", "item_id"]],
                            on=["user_id", "item_id"]).shape[0])
    stats = {"dataset": CATEGORIES[cat], "source_category": cat, "variant": "last_out_w_his",
             "source_rows": {"train": len(tr), "valid": len(va), "test": len(te)},
             "n_users": len(users), "n_items": int(st.n_items[0]), "n_interactions": int(st.n_interactions[0]),
             "train_rows": len(train), "holdout_rows": len(holdout),
             "holdout_items_not_in_train_catalog": cold, "holdout_item_in_user_train": in_train,
             "min_date": st.min_date[0], "max_date": st.max_date[0]}
    layout.write_stats(out, stats)
    layout.write_sums(out)
    return stats


def timestamp_diagnostics(raw_dir, cat: str) -> dict:
    """Why timestamp_w_his is not a one-holdout-per-user split (MEASURED numbers; informational)."""
    tr, va, te = (_read(raw_dir, "timestamp_w_his", cat, p) for p in ("train", "valid", "test"))
    all_users = set(tr.user_id) | set(va.user_id) | set(te.user_id)
    tc = te.user_id.value_counts()
    return {"rows": [len(tr), len(va), len(te)], "users_total": len(all_users),
            "users_with_test_rows": int(tc.size), "users_with_gt1_test_rows": int((tc > 1).sum()),
            "max_test_rows_per_user": int(tc.max()),
            "train_ts_max": int(tr.timestamp.max()), "valid_ts_min": int(va.timestamp.min()),
            "valid_ts_max": int(va.timestamp.max()), "test_ts_min": int(te.timestamp.min())}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Convert Amazon Reviews 2023 5-core categories to the leave-one-out layout")
    ap.add_argument("--raw-root", required=True, help="directory holding Amazon2023-<category>/ folders")
    ap.add_argument("--out-root", required=True, help="output directory (one sub-folder per dataset)")
    ap.add_argument("--cats", nargs="*", default=list(CATEGORIES))
    a = ap.parse_args(argv)
    for cat in a.cats:
        raw = Path(a.raw_root) / f"Amazon2023-{cat}"
        out = Path(a.out_root) / CATEGORIES[cat]
        s = convert_last_out(raw, cat, out)
        d = timestamp_diagnostics(raw, cat)
        (out / "TIMESTAMP_VARIANT_DIAGNOSTICS.json").write_text(json.dumps(d, indent=1))
        layout.write_sums(out)
        print(cat, {k: s[k] for k in ("n_users", "n_items", "n_interactions", "train_rows", "holdout_rows",
                                       "holdout_items_not_in_train_catalog", "holdout_item_in_user_train")})
        print(cat, "timestamp_w_his NOT converted:", d)


if __name__ == "__main__":
    main()
