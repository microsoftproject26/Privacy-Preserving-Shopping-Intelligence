"""MBHT (KDD'22) RecBole .inter files -> the leave-one-out layout.

Source columns: session_id, item_id_list, item_type_list, item_id (tab separated, RecBole header with ':token' suffixes).
MEASURED structure of the TRAIN file: rows sharing a session_id are augmented prefixes of one sequence (each row = the
history before one target purchase + that target; the next row's list contains the previous target typed 4); lists are
capped at 199 items. So a session's full sequence = (its longest list) + (that row's target); the shorter rows add nothing.
The TEST file has one row per session (session_id 1..N, an id space independent of the train file): history + target.

Output users: every train session (all events in train.csv) followed by every test session (history in train.csv, the
test target = the ONLY holdout row). A session is a pseudo-user: the real users cannot be reconstructed from the
released files.
Items keep their MBHT ids (RecBole 1-based ids; the MBHT fixed-100-popular lists use these ids).
Datetime is SYNTHETIC: 1970-01-01 00:00:00 + (position in the sequence) seconds. rating = 1.0, weight = 1.0.
Behaviour types go to side file item_type.csv (user_id, item_id, datetime, item_type, part); the type of every
target is 4 (the marker the source itself writes for earlier targets inside later lists; semantic unverified).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import layout

DATASETS = {"tmall_beh": "mbht_taobao", "ijcai_beh": "mbht_tmall"}     # folder -> output dataset
PAPER = {"mbht_taobao": {"items": 99038}, "mbht_tmall": {"items": 808354}}   # MBHT paper Table 1 (items)
TARGET_TYPE = 4
CAP = 199


def _read_inter(path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, header=0, names=["sid", "il", "tl", "t"])
    df["sid"] = df["sid"].astype(np.int64)
    df["n"] = df["il"].str.count(" ") + 1
    return df


def _longest(df: pd.DataFrame) -> pd.DataFrame:
    """Per session the longest-list row (ties: the later row in the file)."""
    df = df.assign(_o=np.arange(len(df)))
    df = df.sort_values(["sid", "n", "_o"], kind="stable")
    return df.drop_duplicates("sid", keep="last").sort_values("sid", kind="stable").reset_index(drop=True)


def _explode(rows: pd.DataFrame, user_ids: np.ndarray, with_target: bool):
    """-> (user, item, type, pos) arrays; position is 1-based inside the user's full sequence."""
    us, its, tys, ps = [], [], [], []
    for uid, il, tl, t in zip(user_ids, rows["il"], rows["tl"], rows["t"]):
        a = np.fromstring(il, dtype=np.int64, sep=" ")
        b = np.fromstring(tl, dtype=np.int64, sep=" ")
        if a.size != b.size:
            raise ValueError("item_id_list / item_type_list length mismatch")
        if with_target:
            a = np.append(a, np.int64(t))
            b = np.append(b, TARGET_TYPE)
        us.append(np.full(a.size, uid, dtype=np.int64)); its.append(a); tys.append(b); ps.append(np.arange(1, a.size + 1))
    return np.concatenate(us), np.concatenate(its), np.concatenate(tys), np.concatenate(ps)


def convert(folder, src_name: str, out_dir) -> dict:
    folder = Path(folder)
    name = DATASETS[src_name]
    tr_raw = _read_inter(folder / f"{src_name}.train.inter")
    te_raw = _read_inter(folder / f"{src_name}.test.inter")
    tr = _longest(tr_raw)
    te = te_raw.sort_values("sid", kind="stable").reset_index(drop=True)
    if te["sid"].duplicated().any():
        raise ValueError("test file has repeated session ids")
    n_tr = len(tr)
    uid_tr = np.arange(1, n_tr + 1, dtype=np.int64)
    uid_te = n_tr + np.arange(1, len(te) + 1, dtype=np.int64)
    u1, i1, t1, p1 = _explode(tr, uid_tr, with_target=True)
    u2, i2, t2, p2 = _explode(te, uid_te, with_target=False)
    ht = te["t"].astype(np.int64).to_numpy()
    hpos = te["n"].to_numpy() + 1
    # the test history goes to train.csv; the test target is the holdout
    u_all, i_all, ty_all, p_all = (np.concatenate([x, y]) for x, y in ((u1, u2), (i1, i2), (t1, t2), (p1, p2)))
    train = layout.make_frame(u_all, i_all, 1.0, layout.dt_from_seconds(p_all))
    holdout = layout.make_frame(uid_te, ht, 1.0, layout.dt_from_seconds(hpos))
    st = layout.write_dataset(out_dir, train, holdout, n_users=n_tr + len(te))
    out = Path(out_dir)
    side = pd.DataFrame({"user_id": np.concatenate([u_all, uid_te]), "item_id": np.concatenate([i_all, ht]),
                         "datetime": pd.concat([train["datetime"], holdout["datetime"]], ignore_index=True),
                         "item_type": np.concatenate([ty_all, np.full(len(te), TARGET_TYPE)]),
                         "part": ["train"] * len(train) + ["holdout"] * len(holdout)})
    side.to_csv(out / "item_type.csv", index=False)
    pd.DataFrame({"user_id": np.concatenate([uid_tr, uid_te]), "source": ["train"] * n_tr + ["test"] * len(te),
                  "orig_session_id": np.concatenate([tr["sid"].to_numpy(), te["sid"].to_numpy()])}
                 ).to_csv(out / "user_map.csv", index=False)
    seen_pairs = pd.merge(holdout[["user_id", "item_id"]].drop_duplicates(), train[["user_id", "item_id"]].drop_duplicates(),
                          on=["user_id", "item_id"]).shape[0]
    tcat = set(train["item_id"])
    stats = {"dataset": name, "source_folder": src_name, "variant": "mbht_session_longest_plus_target",
             "source_rows": {"train": len(tr_raw), "test": len(te_raw)},
             "train_sessions": n_tr, "test_sessions": len(te), "n_users": n_tr + len(te),
             "n_items": int(st.n_items[0]), "paper_items": PAPER[name]["items"],
             "n_interactions": int(st.n_interactions[0]), "train_rows": len(train), "holdout_rows": len(holdout),
             "train_sessions_at_cap_199": int((tr["n"] >= CAP).sum()),
             "holdout_items_not_in_train_catalog": int(sum(1 for x in ht if x not in tcat)),
             "holdout_item_in_user_train": int(seen_pairs),
             "item_type_counts": {int(k): int(v) for k, v in pd.Series(ty_all).value_counts().sort_index().items()},
             "note": "sessions are pseudo-users; paper users/interactions cannot be reconstructed"}
    layout.write_stats(out, stats)
    layout.write_sums(out)
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description="Convert the MBHT Taobao / Tmall sessions to the leave-one-out layout")
    ap.add_argument("--mbht-root", required=True, help=".../MBHT_dataset")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--folders", nargs="*", default=list(DATASETS))
    a = ap.parse_args(argv)
    for f in a.folders:
        s = convert(Path(a.mbht_root) / f, f, Path(a.out_root) / DATASETS[f])
        print(f, "->", s["dataset"], json.dumps({k: v for k, v in s.items() if k != "note"}, sort_keys=True))


if __name__ == "__main__":
    main()
