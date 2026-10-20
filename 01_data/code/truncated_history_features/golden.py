"""v4trunc.golden — T41 (PKG/12 L52; EVALUATION_CONTRACT_V4 E11.4-E11.7): golden replay and the leakage controls.

golden_replay(builder, decision_ids, split_kind)
  For every decision, the builder at h = FULL (the full available prior history) must reproduce the PRODUCTION batch
  bitwise: every key of v4int.cache.build_subset_store(..., history_policy=USER_HISTORY_L50) -> CleanV2Store.batch ->
  v4int.providers.add_session_offsets (the StoreEvalProvider / StoreTrainProvider path), float tensors compared by
  their bit patterns, dtypes and shapes included. The only admitted differences are the declared PARTIAL_OOV_IDENTITY
  cells (v4trunc.rules): a row the builder itself flags (brand / main-category OOV collision, audit of the block) may
  differ in exactly the V2 user_context columns of that channel; any other difference fails T41. The receipt counts
  compared rows, flagged rows and differing cells per key.
Controls (E11.7)
  * leak control: a builder that uses the full history for user_context (or one extra event) must produce features
    that DIFFER from the E11.4 output on the WARM_TRUNCATION panel, and the check must detect it (tests_trunc NCs);
  * deletion invariance: the E11.4 output must not change when every event of the user before the permitted window is
    deleted from storage (the event table with those rows physically removed, a fresh builder).
"""
from __future__ import annotations

from typing import Callable, Iterable, Optional, Sequence

import numpy as np

from .builder import FULL, TruncatedHistoryBuilder

V2_UCTX_PARTIAL = {"brand_oov_collision": (7,),          # recent_brand_match_share
                   "main_oov_collision": (5, 6)}         # recent_distinct_known_main_categories, recent_category_match_share


def _np(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:                                  # pragma: no cover
        pass
    return x


def diff_rows(a, b) -> Optional[np.ndarray]:
    """[B] bool: rows whose values differ bitwise (None when the values are not row-aligned arrays)."""
    a, b = _np(a), _np(b)
    if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
        return None
    if a.dtype != b.dtype or a.shape != b.shape:
        n = a.shape[0] if a.ndim else 1
        return np.ones(n, dtype=bool)
    if a.dtype.kind == "f":
        av = a.view(np.dtype(f"i{a.dtype.itemsize}"))
        bv = b.view(np.dtype(f"i{b.dtype.itemsize}"))
        d = av != bv
    else:
        d = a != b
    return d.reshape(d.shape[0], -1).any(axis=1) if d.ndim else np.array([bool(d)])


def compare_batches(prod: dict, mine: dict, flags: Sequence[dict]) -> dict:
    """Per key: differing rows; separately the rows that differ ONLY inside their admitted PARTIAL cells."""
    out = {"keys_missing": sorted(set(prod) - set(mine)), "keys_extra": sorted(set(mine) - set(prod)),
           "diff_rows": {}, "admitted_rows": 0, "non_array_diff": []}
    B = len(flags)
    admitted = np.zeros(B, dtype=bool)
    for k in sorted(set(prod) & set(mine)):
        a, b = prod[k], mine[k]
        d = diff_rows(a, b)
        if d is None:
            if str(a) != str(b):
                out["non_array_diff"].append(k)
            continue
        if k == "user_context" and d.any():
            A, C = _np(a), _np(b)
            for i in np.flatnonzero(d).tolist():
                cols = set()
                for flag, cs in V2_UCTX_PARTIAL.items():
                    if flags[i].get(flag):
                        cols |= set(cs)
                diff_cols = set(np.flatnonzero(A[i].view(np.int32) != C[i].view(np.int32)).tolist())
                if diff_cols and diff_cols <= cols:
                    d[i] = False
                    admitted[i] = True
        if d.any():
            out["diff_rows"][k] = int(d.sum())
    out["admitted_rows"] = int(admitted.sum())
    return out


def production_eval_batches(decision_ids: np.ndarray, *, splits: Sequence[str], kind: str, state_sha256: str,
                            ds_root, buckets: Optional[Sequence[int]] = None, batch_size: int = 256,
                            log: Callable = lambda *_: None):
    """(positions, batch) of the production USER_HISTORY_L50 path for strictly ascending decision ids."""
    from v4int.cache import build_subset_store
    from v4int.providers import add_session_offsets
    st, ds, _ = build_subset_store(np.asarray(decision_ids, dtype=np.int64), splits=splits, kind=kind,
                                   state_sha256=state_sha256, history_policy="USER_HISTORY_L50", ds_root=ds_root,
                                   buckets=buckets, log=log)
    for s in range(0, len(ds), int(batch_size)):
        pos = np.arange(s, min(len(ds), s + int(batch_size)), dtype=np.int64)
        yield pos, add_session_offsets(st.batch(ds, pos), st, ds, pos)


def golden_replay(builder: TruncatedHistoryBuilder, decision_ids: np.ndarray, *, splits: Sequence[str], kind: str,
                  ds_root, buckets: Optional[Sequence[int]] = None, batch_size: int = 256,
                  log: Callable = lambda *_: None) -> dict:
    """T41 golden replay over `decision_ids` (strictly ascending, every one a retained decision of `kind`)."""
    ids = np.asarray(decision_ids, dtype=np.int64)
    t = builder.table
    rows = np.array([t.row_of(int(d)) for d in ids.tolist()], dtype=np.int64)
    rec = {"n_compared": 0, "diff_rows": {}, "keys_missing": [], "keys_extra": [], "non_array_diff": [],
           "admitted_partial_rows": 0, "flagged_rows": {k: 0 for k in V2_UCTX_PARTIAL}}
    for pos, prod in production_eval_batches(ids, splits=splits, kind=kind, state_sha256=builder.state_sha256,
                                             ds_root=ds_root, buckets=buckets, batch_size=batch_size, log=log):
        blocks = [builder.block(int(r), FULL) for r in rows[pos].tolist()]
        mine = builder.batch_of_blocks(blocks)
        flags = [b.audit for b in blocks]
        for k in rec["flagged_rows"]:
            rec["flagged_rows"][k] += int(sum(bool(f.get(k)) for f in flags))
        c = compare_batches(prod, mine, flags)
        rec["n_compared"] += int(pos.size)
        for k, v in c["diff_rows"].items():
            rec["diff_rows"][k] = rec["diff_rows"].get(k, 0) + v
        rec["admitted_partial_rows"] += c["admitted_rows"]
        for f in ("keys_missing", "keys_extra", "non_array_diff"):
            rec[f] = sorted(set(rec[f]) | set(c[f]))
    rec["pass"] = bool(rec["n_compared"] == ids.size and not rec["diff_rows"] and not rec["keys_missing"]
                       and not rec["keys_extra"] and not rec["non_array_diff"])
    return rec


def assert_golden(rec: dict) -> None:
    assert rec["pass"], (f"T41 golden replay FAILED: diff rows {rec['diff_rows']}, missing {rec['keys_missing']}, "
                         f"extra {rec['keys_extra']}, non-array {rec['non_array_diff']}")


def batches_equal(a: dict, b: dict) -> bool:
    if set(a) != set(b):
        return False
    for k in a:
        d = diff_rows(a[k], b[k])
        if d is None:
            if str(a[k]) != str(b[k]):
                return False
        elif d.any():
            return False
    return True


def deletion_invariance(builder: TruncatedHistoryBuilder, items: Iterable[tuple], make_builder: Callable) -> dict:
    """For each (q_row, h): the batch from `builder` == the batch from a fresh builder over the event table with every
    row of q's user BEFORE the permitted window physically deleted. `make_builder(table)` builds the fresh builder."""
    t = builder.table
    n, bad = 0, []
    for q_row, h in items:
        u = int(t.row_user[q_row])
        start = int(t.user_start[u])
        first = int(q_row) - int(h)
        drop = np.arange(start, first, dtype=np.int64)
        t2 = t.drop_rows(drop)
        b2 = make_builder(t2)
        q2 = t2.row_of(int(t.decision_id[q_row]))
        ok = batches_equal(builder.batch([(int(q_row), int(h))]), b2.batch([(q2, int(h))]))
        n += 1
        if not ok:
            bad.append(int(t.decision_id[q_row]))
    return {"n_checked": n, "n_changed": len(bad), "changed_decisions": bad[:20]}


__all__ = ["golden_replay", "assert_golden", "compare_batches", "diff_rows", "batches_equal", "deletion_invariance",
           "production_eval_batches", "V2_UCTX_PARTIAL"]
