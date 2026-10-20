"""v4hist.census (owner: Data/integration engineer, DATA-05) — the 5-100 pool's decisions and window-length statistics.

Pool (CENSUS-02, binding definitions reused read-only): users with 5 <= n <= 100, n = n_eligible_TRAIN_examples under
catalogue A. Each bucket's decision ledger is scanned through v4data.manifests.read_non_test (TEST never
materialised), the row flags come from the DATA-04 functions through census_v2.census02.RowDefs (elig, eval, splits),
and the stratum from census_v2.core.stratum_codes. Users stay in memory as per-bucket ordinals; no identifier is
returned. The pool's supervised TRAIN decisions are its `elig` rows (== the store's TRAIN loss decisions), and its
VALIDATION eval rows are `val & eval` rows (== the store's VALIDATION eval decisions).
Counts only: nothing here reads a model output, and no TEST row, label or feature is read.
"""
from __future__ import annotations

from pathlib import Path

import ppsi_env  # noqa: F401  (isolation first)

import numpy as np
import pyarrow.compute as pc

from census_v2 import core
from census_v2.census02 import COLS as CENSUS_COLS, RowDefs
from v4data.common import load_manifest_spec
from v4data.manifests import read_non_test

from .policy import CONTEXT_LENGTH
from .windows import RowIndex

# the ledger columns ManifestBuilder._row_arrays reads (census02.COLS without the two product-id strings)
LEDGER_COLS = [c for c in CENSUS_COLS if c not in ("product_id", "target_product_id")]


def row_defs() -> RowDefs:
    return RowDefs(load_manifest_spec())


def pool_bucket(defs: RowDefs, ds_root, b: int) -> dict:
    """The pool decisions of one bucket (a user never crosses a bucket, so per-user counts are complete here)."""
    p = Path(ds_root) / "decisions" / f"user_bucket={int(b)}" / "ledger.parquet"
    t = read_non_test(p, LEDGER_COLS)
    if t["user_id"].null_count:
        raise AssertionError("NULL user_id in the ledger")
    r = defs._row_arrays(t)
    uniq = pc.unique(t["user_id"])
    ui = np.asarray(pc.index_in(t["user_id"], value_set=uniq).to_numpy(zero_copy_only=False), dtype=np.int64)
    nu = len(uniq)
    did = np.asarray(t["decision_id"].to_numpy(zero_copy_only=False), dtype=np.int64)
    del t, uniq
    elig, tr, val, ev = r["elig"], r["tr"], r["val"], r["eval"]
    n = np.bincount(ui[elig], minlength=nu).astype(np.int64)
    ntrain = np.bincount(ui[tr], minlength=nu).astype(np.int64)
    strat = core.stratum_codes(n, ntrain)
    in_pool = (strat == core.S_C)[ui]
    tid = np.sort(did[elig & in_pool])
    vid = np.sort(did[val & ev & in_pool])
    return {"bucket": int(b), "users": int(nu), "pool_users": int(np.count_nonzero(strat == core.S_C)),
            "pool_train_examples": int(np.count_nonzero(elig & in_pool)), "pool_val_eval_rows": int(vid.size),
            "strata_users": np.bincount(strat, minlength=5).astype(np.int64).tolist(),
            "all_elig": int(elig.sum()), "all_val_eval": int((val & ev).sum()),
            "train_ids": tid, "val_ids": vid}


def length_stats(lengths, L: int = CONTEXT_LENGTH) -> dict:
    """Share at length 1, quantiles (census_v2.core.quantiles: numpy linear), share at the cap, the full histogram."""
    a = np.asarray(lengths, dtype=np.int64)
    n = int(a.size)
    if n == 0:
        return {"n": 0}
    if int(a.min()) < 1 or int(a.max()) > int(L):
        raise AssertionError("a window length outside [1, L]")
    h = np.bincount(a, minlength=int(L) + 1)
    return {"n": n, "share_length_1": float(h[1] / n), "share_at_cap_L": float(h[int(L)] / n),
            "quantiles": core.quantiles(a), "histogram_lengths_1_to_L": h[1:].astype(np.int64).tolist()}


def window_composition(index: RowIndex, rows, lengths) -> dict:
    """Per decision (window [r - len + 1, r]): the number of model segments it spans, its TRAIN-period rows, and
    whether its first row is a TRAIN-period event."""
    r = np.asarray(rows, dtype=np.int64)
    lo = r - np.asarray(lengths, dtype=np.int64) + 1
    seg_start = np.concatenate([[0], np.cumsum(index.position == 1, dtype=np.int64)])
    train = np.concatenate([[0], np.cumsum(index.split_code == 0, dtype=np.int64)])
    n_seg = seg_start[r + 1] - seg_start[lo + 1] + 1              # segment starts in (lo, r], plus lo's segment
    n_train = train[r + 1] - train[lo]
    return {"n_segments": n_seg, "n_train_rows": n_train, "starts_in_train": index.split_code[lo] == 0}


def composition_stats(comp: dict, lengths) -> dict:
    ln = np.asarray(lengths, dtype=np.int64)
    n = int(ln.size)
    if n == 0:
        return {"n": 0}
    seg = comp["n_segments"]
    return {"n": n, "segments_per_window": core.quantiles(seg), "share_spanning_more_than_one_segment":
            float(np.count_nonzero(seg > 1) / n),
            "share_of_window_positions_from_TRAIN": float(comp["n_train_rows"].sum() / ln.sum()),
            "share_of_decisions_whose_window_includes_TRAIN_events": float(np.count_nonzero(comp["n_train_rows"] > 0)
                                                                           / n)}


__all__ = ["row_defs", "pool_bucket", "length_stats", "window_composition", "composition_stats", "LEDGER_COLS"]
