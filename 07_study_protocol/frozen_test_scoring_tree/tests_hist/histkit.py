"""DATA-05 test kit: the synthetic history fixture (users with several model segments across TRAIN / VALIDATION_SELECT
/ VALIDATION_CONFIRM / TEST, generated in memory; no dataset row), a naive reference resolver, and helpers.

The synthetic fixture is the only place a TEST-partition row exists in these tests: it is fabricated, never read.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
for _p in (str(CODE / "tests_int"), str(CODE / "tests_data")):
    if _p not in sys.path:
        sys.path.append(_p)

import v4train  # noqa: E402,F401  (ppsi_env first)

import numpy as np  # noqa: E402

from intkit import REAL_DS, arrow_of, assert_raises, batches_equal, nc  # noqa: E402,F401  (read-only import)
from v4data.common import TEST_END_US, TRAIN_END_US, TRAIN_START_US, VALIDATION_END_US, VS_END_US  # noqa: E402
from v4hist.policy import CONTEXT_LENGTH, SESSION_L50, USER_HISTORY_L50  # noqa: E402
from v4hist.windows import RowIndex  # noqa: E402

PARTS = ((0, TRAIN_START_US, TRAIN_END_US), (1, TRAIN_END_US, VS_END_US), (2, VS_END_US, VALIDATION_END_US),
         (3, VALIDATION_END_US, TEST_END_US))
Q = lambda *_: None  # noqa: E731
POLICIES = (USER_HISTORY_L50, SESSION_L50)
S = 1_000_000                                                    # one second in microseconds


def synthetic_index(seed: int = 0, n_users: int = 300) -> RowIndex:
    """Users in (user, chronological) order. Each user has 0-3 model segments per partition; a segment has 1-80
    events 0-600 s apart (ties allowed), segments are > 3600 s apart, and every partition fence starts a segment."""
    rng = np.random.default_rng(seed)
    uo, pos, ts, sc = [], [], [], []
    for u in range(n_users):
        for code, lo, hi in PARTS:
            k = int(rng.choice([0, 1, 2, 3], p=[0.25, 0.35, 0.25, 0.15]))
            t = lo + int(rng.integers(0, 3600 * S))
            for _ in range(k):
                n = int(rng.choice([int(rng.integers(1, 6)), int(rng.integers(6, 81))]))
                gaps = np.where(rng.random(n) < 0.1, 0, rng.integers(1, 600 * S, size=n))
                gaps[0] = 0
                times = t + np.cumsum(gaps)
                if int(times[-1]) >= hi:
                    break
                uo.append(np.full(n, u)); pos.append(np.arange(1, n + 1)); ts.append(times)  # noqa: E702
                sc.append(np.full(n, code))
                t = int(times[-1]) + 3601 * S + int(rng.integers(0, 86_400 * S))
    uo, pos, ts, sc = (np.concatenate(x) for x in (uo, pos, ts, sc))
    return RowIndex.from_arrays(user_ord=uo, position=pos, ts_us=ts, split_code=sc, session_id=v1_sessions(uo, ts),
                                decision_id=np.arange(uo.size, dtype=np.int64) * 7 + 11)


def v1_sessions(user_ord, ts_us) -> np.ndarray:
    """The v1 behavioural session rule on synthetic rows (EDA pipeline4 sess2): a new session iff a new user or a gap
    > 3600 s to the previous event; never reset at a partition fence."""
    uo, ts = np.asarray(user_ord), np.asarray(ts_us, dtype=np.int64)
    new = np.ones(uo.size, dtype=bool)
    if uo.size > 1:
        new[1:] = (uo[1:] != uo[:-1]) | ((ts[1:] - ts[:-1]) > 3600 * S)
    return np.cumsum(new).astype(np.int32)


def naive_window(index: RowIndex, r: int, policy: str, L: int = CONTEXT_LENGTH) -> list:
    """The reference definition, one decision at a time: the user's rows up to and including r (USER), or the
    segment's rows up to and including r (SESSION), last L of them."""
    u = int(index.user_ord[r])
    rows = []
    j = r
    while j >= 0 and int(index.user_ord[j]) == u and len(rows) < L:
        if policy == SESSION_L50 and j < r and int(index.position[j + 1]) == 1:
            break
        rows.append(j)
        j -= 1
    return rows[::-1]


def raw_rows(bucket: int, splits, columns) -> dict:
    """Raw (stored, pre-normalization) v1 feature columns of one bucket's retained rows, in the snapshot store's row
    order (read_non_test, the store's partitions, np.lexsort((position, segment, user))), plus decision_id for the
    alignment check against a RowIndex. Test reference only (the direct-formula checks)."""
    import pyarrow as pa
    import pyarrow.compute as pc
    from v4data.manifests import read_non_test
    f = REAL_DS / "model_ready" / f"user_bucket={int(bucket)}" / "features.parquet"
    keys = ["user_id", "segment_index", "position_in_segment", "decision_id", "split"]
    t = read_non_test(f, keys + [c for c in columns if c not in keys])
    t = t.filter(pc.is_in(t["split"], value_set=pa.array(list(splits), type=pa.string())))
    o = np.lexsort((t["position_in_segment"].to_numpy(), t["segment_index"].to_numpy(),
                    t["user_id"].to_numpy(zero_copy_only=False)))
    return {c: np.asarray(t[c].to_numpy(zero_copy_only=False))[o] for c in list(columns) + ["decision_id"]}


def take_rows(batch: dict, rows) -> dict:
    """The rows `rows` of a batch (tensors sliced on dim 0; other fields kept)."""
    import torch
    r = torch.as_tensor(np.asarray(rows, dtype=np.int64))
    return {k: (v.index_select(0, r) if torch.is_tensor(v) and v.dim() >= 1 else v) for k, v in batch.items()}


__all__ = ["REAL_DS", "nc", "assert_raises", "batches_equal", "synthetic_index", "naive_window", "take_rows", "Q",
           "POLICIES", "arrow_of"]
