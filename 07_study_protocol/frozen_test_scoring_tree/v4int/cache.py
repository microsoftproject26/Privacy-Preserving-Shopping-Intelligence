"""v4int.cache — the manifest-scoped Clean V2 store (a materialized in-RAM cache over the unmodified data path).

Why: `v4data.store.CleanV2Store.build(splits=("TRAIN",))` holds EVERY TRAIN event row of all 64 buckets (the streaming
store; its RSS and build time are measured by INT-01 on the server), while a cohort manifest touches only its users'
windows. Two concurrent central runs (one per GPU), each with an evaluation provider, must fit the 64 GiB cgroup.

How (no locked code is changed):
  1. the snapshot CompactStore is built through `CleanV2Store.build(splits, buckets=group)` for one bucket group at a
     time (a user lives in exactly one bucket, and a window never leaves its segment, so a group store holds the
     complete windows of its users);
  2. the group's decision set is taken exactly as the streaming path takes it (`decision_set(split, kind)`; for
     VALIDATION the two splits merged in ascending decision_id, as tests_eval/gpu_t3_receipt.merged_eval_set);
  3. only the manifest's decisions are kept; for each, the event rows [end_row - length + 1, end_row] are marked and the
     union of marked rows is copied, IN ORDER, into contiguous arrays (every event array CompactStore.batch reads), and
     end_row is remapped to the row's rank in that union — each window stays one contiguous run of the same values;
  4. the result is a CompactStore instance (the snapshot class; only its data attributes are set) wrapped in the
     UNMODIFIED `CleanV2Store`, so every batch is produced by the same code as the streaming path:
     CleanV2Store.batch -> CompactStore.batch (window gather) -> clean_v2_view (RC1-RC8, EXACT_INVERSE re-encode).
The only representational difference is the user-context mask dtype (uint8 when every group's masks are binary, else
float32); CompactStore.batch casts it to float32, so the values are identical. INT-01 proves cache == stream bitwise on
the server (>= 20,000 random rows plus whole trainer batches) instead of assuming it.
Nothing is written to disk.

History policy (DATA-05; DECISION_LOG 35-36): `history_policy` is a REQUIRED keyword (no default).
  SESSION_L50       the path above, byte for byte (the group store holds the decision splits; lengths are the store's
                    own min(50, position_in_segment)).
  USER_HISTORY_L50  each group store holds v4hist.policy.store_splits(policy, splits) (TRAIN for TRAIN decisions;
                    TRAIN + VALIDATION_SELECT + VALIDATION_CONFIRM for VALIDATION decisions, never TEST), the group's
                    v4hist RowIndex is built and proven aligned to the group store (v4hist.windows.align_to_store), and
                    the decision set's lengths become min(50, rank in the user's stream + 1)
                    (v4hist.windows.policy_decision_set) BEFORE the window rows are marked, so the kept rows and the
                    remapped end_row cover exactly the USER_HISTORY windows. A user lives in one bucket, so a group
                    store holds each of its users' complete prior stream. The index is dropped after each group,
                    except the behavioural session id of every kept row (DECISION_LOG 47), which is kept alongside the
                    cache as `session_id` (int32, cache row order) for the `session_offset_tokens` batch key.
"""
from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import v4train  # noqa: F401  (ppsi_env first)

import numpy as np

from v4data.common import N_BUCKETS, VALIDATION_SPLITS, dataset_root, refuse_test_split
from v4data.store import CleanV2Store
from v4hist.policy import needs_row_index, require_policy, store_splits
from v4hist.windows import align_to_store, build_row_index, policy_decision_set

EVENT_ARRAYS = ("_item_tokens", "_category_tokens", "_brand_tokens", "_main_category_tokens", "_daypart_tokens",
                "_weekday_tokens", "_event_numeric_features", "_event_quality_flags", "_user_context",
                "_user_context_masks")
DS_FIELDS = ("decision_id", "end_row", "length", "target_class", "loss_mask", "eval_mask", "client_idx", "user_bucket",
             "prior_view_count", "prefix_position")


class CacheBuildError(RuntimeError):
    pass


def rss_bytes() -> Optional[int]:
    """Current resident set size (Linux /proc; None elsewhere)."""
    try:
        with open("/proc/self/status", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def peak_rss_bytes() -> Optional[int]:
    try:
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024          # Linux: KiB
    except (ImportError, AttributeError):
        return None


def merged_decision_set(store: CleanV2Store, splits: Sequence[str], kind: str):
    """The streaming path's decision set: one split as-is; several splits (VALIDATION) merged in ascending decision_id
    exactly as tests_eval/gpu_t3_receipt.merged_eval_set does."""
    from full_store import DecisionSet                       # snapshot copy (ppsi_env)
    parts = [store.decision_set(s, kind) for s in splits]
    if len(parts) == 1:
        return parts[0]
    cat = lambda f: np.concatenate([getattr(p, f) for p in parts])  # noqa: E731
    did = cat("decision_id")
    o = np.argsort(did, kind="stable")
    ds = DecisionSet(split="VALIDATION" if tuple(splits) == tuple(VALIDATION_SPLITS) else "+".join(splits), kind=kind,
                     decision_id=did[o], end_row=cat("end_row")[o], length=cat("length")[o],
                     target_class=cat("target_class")[o], loss_mask=cat("loss_mask")[o], eval_mask=cat("eval_mask")[o],
                     client_idx=cat("client_idx")[o], user_bucket=cat("user_bucket")[o], eval_user_key=None,
                     prior_view_count=cat("prior_view_count")[o], prefix_position=cat("prefix_position")[o])
    if ds.decision_id.size > 1 and not np.all(np.diff(ds.decision_id) > 0):
        raise CacheBuildError("merged decision ids are not strictly ascending")
    return ds


def _members(sorted_ids: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Boolean mask: which of `ids` are in the strictly ascending `sorted_ids`."""
    if sorted_ids.size == 0:
        return np.zeros(ids.shape, dtype=bool)
    p = np.searchsorted(sorted_ids, ids)
    pc = np.minimum(p, sorted_ids.size - 1)
    return (p < sorted_ids.size) & (sorted_ids[pc] == ids)


@dataclass
class SubsetStoreInfo:
    splits: tuple
    kind: str
    n_decisions: int
    n_event_rows: int
    buckets: list
    bucket_groups: int
    masks_dtype: str
    build_s: float
    group_log: list = field(default_factory=list)
    cache_nbytes: int = 0
    peak_rss_bytes_after: Optional[int] = None
    history_policy: Optional[str] = None
    store_splits: tuple = ()
    mean_window_length: Optional[float] = None

    def as_json(self) -> dict:
        return dict(self.__dict__)


def cache_nbytes(store: CleanV2Store, ds) -> int:
    n = sum(int(getattr(store.cs, a).nbytes) for a in EVENT_ARRAYS)
    return n + sum(int(np.asarray(getattr(ds, f)).nbytes) for f in DS_FIELDS)


def build_subset_store(decision_ids: np.ndarray, *, splits: Sequence[str], kind: str, state_sha256: Optional[str],
                       history_policy: str, ds_root: Optional[Path] = None, buckets: Optional[Sequence[int]] = None,
                       group_size: int = 8, n_threads: int = 3, log: Callable = print, test_gate=None):
    """(CleanV2Store over only the manifest's windows, its DecisionSet in ascending decision_id, SubsetStoreInfo).

    `decision_ids` must be strictly ascending; every one must be a retained decision of `kind` in `splits`, else the
    build is refused (a manifest row the store does not hold is never silently dropped). `splits` are the DECISION
    splits; `history_policy` (required) decides the store partitions and the window lengths.

    `test_gate` (INT-07): None = the path above, unchanged (a TEST split is refused). A v4data09 GateRecord = the GATED
    RETRO_TEST branch (v4int.retro_store): decision splits VALIDATION_SELECT + VALIDATION_CONFIRM + TEST, store
    partitions TRAIN + VALIDATION_SELECT + VALIDATION_CONFIRM + TEST (USER_HISTORY_L50), the gated store / row index,
    the gate re-validated at every group, and every kept decision must be a TEST-partition decision."""
    from full_store import CompactStore, DecisionSet         # snapshot copies (ppsi_env)
    policy = require_policy(history_policy)
    if test_gate is None:
        for s in splits:
            refuse_test_split(s)
        ss = store_splits(policy, splits)
        build_store, build_index, wrap_store = CleanV2Store.build, build_row_index, CleanV2Store
    else:
        from v4int import retro_store as RS                  # gated; imported only on this branch
        ss, build_store, build_index, wrap_store = RS.gated_path_builders(test_gate, policy, splits)
    want = np.ascontiguousarray(decision_ids, dtype=np.int64)
    if want.size > 1 and not np.all(want[1:] > want[:-1]):
        raise CacheBuildError("decision_ids must be strictly ascending")
    root = Path(ds_root) if ds_root is not None else dataset_root()
    all_b = list(range(N_BUCKETS)) if buckets is None else sorted(int(b) for b in buckets)
    groups = [all_b[i:i + int(group_size)] for i in range(0, len(all_b), int(group_size))]
    t0 = time.time()
    ev = {a: [] for a in EVENT_ARRAYS}
    dsp = {f: [] for f in DS_FIELDS}
    mask_dtypes, glog, base = [], [], 0
    sids = []
    last = None
    for g in groups:
        tg = time.time()
        st = build_store(splits=ss, buckets=g, ds_root=root, state_sha256=state_sha256,
                         n_threads=n_threads, log=lambda *_: None)
        gds = merged_decision_set(st, splits, kind)
        aligned, gsid = None, None
        if needs_row_index(policy):
            ri = build_index(root, st.cs.buckets, ss)
            aligned = align_to_store(ri, st.cs)
            gds = policy_decision_set(gds, ri, policy)
            gsid = ri.session_id
            del ri
        sel = np.flatnonzero(_members(want, gds.decision_id))
        if test_gate is not None and not bool(np.all(RS.retro_decision_mask(test_gate, st, gds.decision_id[sel]))):
            raise CacheBuildError(f"bucket group {g}: a RETRO_TEST manifest decision is not a TEST-partition decision")
        n_rows = int(st.cs.n_rows)
        kept_rows = 0
        if sel.size:
            end = gds.end_row[sel].astype(np.int64)
            ln = gds.length[sel].astype(np.int64)
            if np.any(ln < 1) or np.any(end - ln + 1 < 0) or np.any(end >= n_rows):
                raise CacheBuildError(f"bucket group {g}: a window lies outside the group store")
            diff = np.zeros(n_rows + 1, dtype=np.int64)
            np.add.at(diff, end - ln + 1, 1)
            np.add.at(diff, end + 1, -1)
            rows = np.flatnonzero(np.cumsum(diff[:-1]) > 0)
            new_end = np.searchsorted(rows, end)
            if not np.array_equal(np.searchsorted(rows, end - ln + 1), new_end - (ln - 1)) \
                    or not np.array_equal(rows[new_end], end):
                raise CacheBuildError(f"bucket group {g}: a window is not contiguous in the kept rows")
            for a in EVENT_ARRAYS:
                ev[a].append(np.ascontiguousarray(getattr(st.cs, a)[rows]))
            if gsid is not None:
                sids.append(np.ascontiguousarray(gsid[rows], dtype=np.int32))
            for f in DS_FIELDS:
                v = new_end + base if f == "end_row" else getattr(gds, f)[sel]
                dsp[f].append(np.asarray(v))
            kept_rows = int(rows.size)
            base += kept_rows
        mask_dtypes.append(str(st.cs._masks_dtype))
        glog.append({"buckets": [int(g[0]), int(g[-1])], "group_store_rows": n_rows, "kept_decisions": int(sel.size),
                     "kept_rows": kept_rows, "masks_dtype": str(st.cs._masks_dtype), "s": round(time.time() - tg, 2),
                     "rss_bytes": rss_bytes(),
                     "index_alignment_decisions_checked": (None if aligned is None
                                                           else int(sum(aligned["decisions_checked"].values())))})
        last = {"layout": st.cs.layout, "context_length": int(st.cs.context_length), "version_hash": st.cs.version_hash,
                "spec": st.spec, "runtime": st.runtime_input_schema_sha256}
        del st, gds
        gc.collect()
        _trim()
        log(f"[v4int.cache] group {g[0]}-{g[-1]}: kept {glog[-1]['kept_decisions']} decisions / "
            f"{glog[-1]['kept_rows']} rows ({glog[-1]['s']} s)")
    if last is None:
        raise CacheBuildError("no bucket group was built")
    mdt = np.uint8 if all(m == "uint8" for m in mask_dtypes) else np.float32
    cs = CompactStore.__new__(CompactStore)
    cs.dataset_root, cs.layout, cs.splits, cs.channels = root, last["layout"], tuple(ss), "F4"
    cs.context_length, cs.buckets, cs.version_hash = last["context_length"], all_b, last["version_hash"]
    for a in EVENT_ARRAYS:
        parts = ev[a]
        if a == "_user_context_masks":
            parts = [p.astype(mdt) if p.dtype != mdt else p for p in parts]
        setattr(cs, a, np.concatenate(parts) if parts else np.zeros(0))
        ev[a] = None
    cs.n_rows = int(cs._item_tokens.shape[0])
    cs._masks_dtype = np.dtype(mdt).name
    cs._split_decisions, cs._decisions_cache = {}, {}
    cs.build_seconds, cs.bucket_log = time.time() - t0, glog
    cat = {f: (np.concatenate(v) if v else np.zeros(0, dtype=np.int64)) for f, v in dsp.items()}
    o = np.argsort(cat["decision_id"], kind="stable")
    ds = DecisionSet(split=("VALIDATION" if tuple(splits) == tuple(VALIDATION_SPLITS) else "+".join(splits))
                     if test_gate is None else "RETRO_TEST", kind=kind,
                     decision_id=cat["decision_id"][o].astype(np.int64), end_row=cat["end_row"][o].astype(np.int64),
                     length=cat["length"][o], target_class=cat["target_class"][o].astype(np.int64),
                     loss_mask=cat["loss_mask"][o].astype(bool), eval_mask=cat["eval_mask"][o].astype(bool),
                     client_idx=cat["client_idx"][o].astype(np.int64), user_bucket=cat["user_bucket"][o],
                     eval_user_key=None, prior_view_count=cat["prior_view_count"][o],
                     prefix_position=cat["prefix_position"][o])
    del cat, dsp
    if not np.array_equal(ds.decision_id, want):
        missing = int(want.size - _members(ds.decision_id, want).sum())
        raise CacheBuildError(f"the store does not hold every manifest decision: {missing} of {want.size} missing "
                              f"(kind {kind!r}, splits {tuple(splits)})")
    store = wrap_store(cs, last["spec"], state_sha256)
    # behavioural session id per kept row (USER_HISTORY_L50 only): an attribute beside the snapshot store, read only by
    # v4hist.windows.session_offset_tokens; CompactStore.batch ignores it
    store.session_id = (np.concatenate(sids) if sids else np.zeros(0, dtype=np.int32)) if needs_row_index(policy) \
        else None
    if store.session_id is not None and store.session_id.shape[0] != cs.n_rows:
        raise CacheBuildError("session ids are not aligned with the kept rows")
    if store.runtime_input_schema_sha256 != last["runtime"]:
        raise CacheBuildError("runtime input schema changed between the group stores and the cache")
    gc.collect()
    _trim()
    info = SubsetStoreInfo(splits=tuple(splits), kind=kind, n_decisions=int(ds.decision_id.size),
                           n_event_rows=int(cs.n_rows), buckets=[int(all_b[0]), int(all_b[-1])],
                           bucket_groups=len(groups), masks_dtype=cs._masks_dtype, build_s=round(time.time() - t0, 2),
                           group_log=glog, cache_nbytes=cache_nbytes(store, ds), peak_rss_bytes_after=peak_rss_bytes(),
                           history_policy=policy, store_splits=tuple(ss),
                           mean_window_length=(float(np.mean(ds.length)) if ds.decision_id.size else None))
    return store, ds, info


def _trim() -> None:
    """Return freed heap pages to the OS: pyarrow's memory pool (the group stores' Parquet decode) and glibc."""
    try:
        import pyarrow as pa
        pa.default_memory_pool().release_unused()
    except Exception:
        pass
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


__all__ = ["build_subset_store", "merged_decision_set", "SubsetStoreInfo", "CacheBuildError", "cache_nbytes",
           "rss_bytes", "peak_rss_bytes", "EVENT_ARRAYS"]
