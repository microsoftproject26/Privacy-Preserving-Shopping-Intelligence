"""Benchmark-runner test helper: a synthetic leave-one-out release in the ppsi.benchmarks layout (no real data)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def make_release(root: Path, name: str = "s3_beauty", n_users: int = 40, n_items: int = 30, seed: int = 7,
                 drop_holdout: int = 2, rating: bool = False, cold_holdout: int = 0) -> dict:
    """Write <root>/<name>/{statistics.csv, leave_one_out/{train,holdout}.csv} in the release format."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    tr, ho = [], []
    base = pd.Timestamp("1970-01-01")
    for u in range(1, n_users + 1):
        L = int(rng.integers(5, 13))
        items = rng.choice(np.arange(1, n_items + 1), size=L + 1, replace=False)
        for t in range(L + 1):
            tt = t + 1
            if u % 5 == 0 and t == L - 1:                  # the last two train rows share a timestamp
                tt = t
            row = (u, int(items[t]), base + pd.Timedelta(nanoseconds=tt), 1.0)
            (tr if t < L else ho).append(row)
    ho = ho[:len(ho) - drop_holdout]                       # a few users have no holdout row
    for k in range(cold_holdout):                          # holdout items that are not in the train catalogue
        u, _it, ts, w = ho[k]
        ho[k] = (u, 9000 + k, ts, w)
    d = Path(root) / name / "leave_one_out"
    d.mkdir(parents=True)
    cols = ["user_id", "item_id", "datetime", "weight"]
    rows = list(tr)
    rng.shuffle(rows)                                      # file order != chronological: the adapter must sort
    tdf, hdf = pd.DataFrame(rows, columns=cols), pd.DataFrame(ho, columns=cols)
    if rating:
        for df in (tdf, hdf):
            df.insert(2, "rating", 4.0)
    tdf.to_csv(d / "train.csv", index=False)
    hdf.to_csv(d / "holdout.csv", index=False)
    stats = {"n_users": n_users, "n_items": n_items, "n_interactions": len(tr) + len(ho),
             "leave_one_out_n_train_interactions": len(tr), "leave_one_out_n_holdout_interactions": len(ho)}
    pd.DataFrame([stats]).to_csv(Path(root) / name / "statistics.csv", index=False)
    return {"root": Path(root), "name": name, "stats": stats, "train_rows": rows, "holdout_rows": ho}
