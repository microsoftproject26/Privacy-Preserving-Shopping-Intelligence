"""Benchmark-runner fixtures: a synthetic leave-one-out release (no real data), and the global torch settings (which
the runner sets for strict numerics) restored after each test module."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch


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


@pytest.fixture()
def release(tmp_path):
    return make_release(tmp_path / "processed")


@pytest.fixture(autouse=True, scope="module")
def _restore_torch_settings():
    saved = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(),
             torch.is_deterministic_algorithms_warn_only_enabled(), torch.get_float32_matmul_precision(),
             torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
             torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    yield
    threads, det, warn_only, precision, cudnn_det, cudnn_bench, tf32_mm, tf32_cudnn = saved
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(det, warn_only=warn_only)
    torch.set_float32_matmul_precision(precision)
    torch.backends.cudnn.deterministic = cudnn_det
    torch.backends.cudnn.benchmark = cudnn_bench
    torch.backends.cuda.matmul.allow_tf32 = tf32_mm
    torch.backends.cudnn.allow_tf32 = tf32_cudnn
