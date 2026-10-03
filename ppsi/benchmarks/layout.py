"""Shared writers: the leave-one-out (ml-1m style) file layout, statistics.csv, STATS.json and SHA256SUMS."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["user_id", "item_id", "rating", "datetime", "weight"]
STAT_COLUMNS = ["n_users", "n_items", "n_interactions", "avg_length", "sparsity", "min_date", "max_date",
                "leave_one_out_n_train_interactions", "leave_one_out_n_holdout_interactions",
                "leave_one_out_holdout_ratio", "loo_correction_leave_one_out"]
FMT_SEC = "%Y-%m-%d %H:%M:%S"


def dt_from_ms(ts_ms) -> pd.Series:
    """ms epoch -> 'YYYY-MM-DD HH:MM:SS.mmm' (UTC). Millisecond precision keeps distinct reviews distinct."""
    d = pd.to_datetime(np.asarray(ts_ms, dtype=np.int64), unit="ms")
    return pd.Series(d.strftime(FMT_SEC + ".%f").str[:-3])


def dt_from_seconds(sec) -> pd.Series:
    d = pd.to_datetime(np.asarray(sec, dtype=np.int64), unit="s")
    return pd.Series(d.strftime(FMT_SEC))


def make_frame(user, item, rating, datetime, weight=1.0) -> pd.DataFrame:
    f = pd.DataFrame({"user_id": np.asarray(user, dtype=np.int64), "item_id": np.asarray(item, dtype=np.int64),
                      "rating": np.asarray(rating, dtype=np.float64),
                      "datetime": pd.Series(datetime).to_numpy(), "weight": float(weight)})
    return f[COLUMNS]


def statistics_row(train: pd.DataFrame, holdout: pd.DataFrame, n_users: int) -> pd.DataFrame:
    """Same definitions as the public benchmark release (checked on ml_1m / s3_beauty):
    avg_length = n_inter / n_users; sparsity = 1 - n_inter / (U * I); holdout_ratio = h / (t + h);
    loo_correction = h / n_users."""
    n_items = int(pd.concat([train["item_id"], holdout["item_id"]]).nunique())
    n_inter = len(train) + len(holdout)
    dts = pd.concat([train["datetime"], holdout["datetime"]])
    row = {"n_users": n_users, "n_items": n_items, "n_interactions": n_inter,
           "avg_length": n_inter / n_users, "sparsity": 1.0 - n_inter / (n_users * n_items),
           "min_date": dts.min(), "max_date": dts.max(),
           "leave_one_out_n_train_interactions": len(train), "leave_one_out_n_holdout_interactions": len(holdout),
           "leave_one_out_holdout_ratio": len(holdout) / (len(train) + len(holdout)), "loo_correction_leave_one_out": len(holdout) / n_users}
    return pd.DataFrame([row])[STAT_COLUMNS]


def write_dataset(out_dir, train: pd.DataFrame, holdout: pd.DataFrame, n_users: int) -> pd.DataFrame:
    out = Path(out_dir)
    (out / "leave_one_out").mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "leave_one_out" / "train.csv", index=False)
    holdout.to_csv(out / "leave_one_out" / "holdout.csv", index=False)
    st = statistics_row(train, holdout, n_users)
    st.to_csv(out / "statistics.csv", index=False)
    return st


def sha256_file(path, chunk=1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def write_sums(out_dir) -> dict:
    out = Path(out_dir)
    sums = {}
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS":
            sums[str(p.relative_to(out)).replace("\\", "/")] = sha256_file(p)
    (out / "SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n, h in sums.items()), encoding="utf-8")
    return sums


def write_stats(out_dir, stats: dict) -> None:
    (Path(out_dir) / "STATS.json").write_text(json.dumps(stats, indent=1, sort_keys=True, default=str), encoding="utf-8")
