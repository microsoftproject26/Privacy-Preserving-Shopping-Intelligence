"""full_store.py — compact per-event store in RAM for the full-data GRU run (Wave 1, Sonnet A).

Owner: Sonnet A. Contract: `FULL_INTERFACES.md` section 1. Do not import anything from
`train_full.py` (Sonnet B) or `eval_full.py` (Sonnet C).

Why this exists: the pilot's bucket-major memmapped window cache (`window_cache.py`) stores
one pre-collated window per DECISION (~5.9 KB/decision on disk). Scaled to the full 40.18M
TRAIN + 8.83M VALIDATION_SELECT decision population that would be ~237 GB on disk -- forbidden
(`FULL_RUN_DISK_BUDGET.md`). Instead, this module stores each EVENT ROW exactly ONCE, in
compact dtypes, entirely in process RAM (no disk write, no memmap): a per-event compact store
plus a decision -> (end_row, length) index. A window is then a vectorised gather of a
contiguous row range, computed on demand in `CompactStore.batch()`.

Every numeric/categorical value is computed EXACTLY as `rees46_eda.loaders.DecisionWindowLoader
.collate()` computes it (same `Normalizers.apply`, same `MISSING_FLAG_OF`/`USER_MISSING_FLAG_OF`
zeroing, same row order via the same `np.lexsort` key), just done ONCE per row instead of once
per (row, window) pair. `code/test_full_store.py` proves this reproduction is byte-exact against
both the pilot cache and a fresh `DecisionWindowLoader.collate()`.

Data-only. No model code. No edits to any vendored or frozen module.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

CODE_DIR = Path(__file__).resolve().parent
VENDOR_DIR = CODE_DIR / "vendor"
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from rees46_eda.loaders import (  # noqa: E402
    Normalizers, MISSING_FLAG_OF, USER_MISSING_FLAG_OF, EVENT_NUMERIC_BLOCK, detect_layout,
)
from rees46_eda.config import (  # noqa: E402
    EVENT_QUALITY_FLAGS, USER_CONTEXT_FEATURES, USER_CONTEXT_MASKS,
    NORMALIZED_EVENT_NUMERIC, NORMALIZED_SESSION_PREFIX, NORMALIZED_USER_CONTEXT,
)
from index_utils import read_index_projected  # noqa: E402

EXPECTED_VERSION_HASH = "860e5a1c3a81b8e4bbfeff4f245fa37638a43e4f260df179585964d2d4da8dbd"
L = 50

F_NUMERIC = len(EVENT_NUMERIC_BLOCK)      # 19
Q_FLAGS = len(EVENT_QUALITY_FLAGS)        # 15
U_CTX = len(USER_CONTEXT_FEATURES)        # 12
M_CTX = len(USER_CONTEXT_MASKS)           # 6

# Columns projected from decisions_index for the per-decision metadata join.
META_COLUMNS = ["decision_id", "client_idx", "target_class", "loss_mask", "eval_mask",
                "position_in_segment"]


def configure_threads(n: int = 3) -> None:
    """Best-effort thread capping for pyarrow (numpy/torch capped by the caller's env vars
    per FULL_INTERFACES.md's common rules -- those must be set before numpy/torch import,
    which this library cannot guarantee since it may be imported after them)."""
    try:
        pa.set_cpu_count(n)
        pa.set_io_thread_count(n)
    except Exception:
        pass


configure_threads(3)


def _release_freed_memory() -> None:
    """Return freed heap pages to the OS after a build (memory only; never touches data).

    The per-bucket decode churns through many short-lived buffers and boxed user_id strings;
    glibc keeps those arenas mapped unless asked, which is what inflated settled RSS."""
    import gc
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# DecisionSet
# ---------------------------------------------------------------------------

@dataclass
class DecisionSet:
    split: str
    kind: str
    decision_id: np.ndarray
    end_row: np.ndarray
    length: np.ndarray
    target_class: np.ndarray
    loss_mask: np.ndarray
    eval_mask: np.ndarray
    client_idx: np.ndarray
    user_bucket: np.ndarray
    eval_user_key: Optional[np.ndarray]
    prior_view_count: np.ndarray
    prefix_position: np.ndarray

    def __len__(self) -> int:
        return int(self.decision_id.shape[0])

    def take(self, positions) -> "DecisionSet":
        positions = np.asarray(positions)
        return DecisionSet(
            split=self.split, kind=self.kind,
            decision_id=self.decision_id[positions],
            end_row=self.end_row[positions],
            length=self.length[positions],
            target_class=self.target_class[positions],
            loss_mask=self.loss_mask[positions],
            eval_mask=self.eval_mask[positions],
            client_idx=self.client_idx[positions],
            user_bucket=self.user_bucket[positions],
            eval_user_key=(self.eval_user_key[positions] if self.eval_user_key is not None else None),
            prior_view_count=self.prior_view_count[positions],
            prefix_position=self.prefix_position[positions],
        )

    def sha256(self) -> str:
        return hashlib.sha256(np.ascontiguousarray(self.decision_id, dtype=np.int64).tobytes()).hexdigest()


def _empty_decision_arrays() -> dict:
    return {
        "decision_id": np.zeros(0, dtype=np.int64),
        "global_row": np.zeros(0, dtype=np.int64),
        "length": np.zeros(0, dtype=np.int16),
        "target_class": np.zeros(0, dtype=np.int64),
        "loss_mask": np.zeros(0, dtype=bool),
        "eval_mask": np.zeros(0, dtype=bool),
        "client_idx": np.zeros(0, dtype=np.int64),
        "user_bucket": np.zeros(0, dtype=np.int16),
        "prior_view_count": np.zeros(0, dtype=np.int32),
        "prefix_position": np.zeros(0, dtype=np.int32),
    }


# ---------------------------------------------------------------------------
# Per-bucket build worker
# ---------------------------------------------------------------------------

def _needed_feature_columns(channels: str) -> list[str]:
    base = ["user_id", "segment_index", "position_in_segment", "decision_id", "split",
            "product_idx", "log1p_prev_view_count"]
    if channels == "F0":
        return base
    if channels != "F4":
        raise ValueError(f"channels must be 'F0' or 'F4', got {channels!r}")
    extra = (["category_idx", "brand_idx", "main_category_idx", "daypart_idx", "local_weekday_idx"]
             + list(EVENT_NUMERIC_BLOCK) + list(EVENT_QUALITY_FLAGS)
             + list(USER_CONTEXT_FEATURES) + list(USER_CONTEXT_MASKS))
    seen = set(base)
    cols = list(base)
    for c in extra:
        if c not in seen:
            seen.add(c)
            cols.append(c)
    return cols


def _all_feature_buckets(dataset_root: Path) -> list[int]:
    root = Path(dataset_root) / "model_ready"
    return sorted(int(p.name.split("=", 1)[1]) for p in root.glob("user_bucket=*"))


def _bucket_precount(dataset_root: Path, b: int, splits: tuple, channels: str) -> tuple[int, bool]:
    """Cheap prepass: retained row count for bucket `b`, plus (F4 only) whether every
    USER_CONTEXT_MASKS value is exactly 0.0/1.0 (decides uint8 vs float32 mask storage)."""
    path = Path(dataset_root) / "model_ready" / f"user_bucket={b}" / "features.parquet"
    cols = ["split"] + (list(USER_CONTEXT_MASKS) if channels == "F4" else [])
    tbl = pq.read_table(path, columns=cols)
    keep = pc.is_in(tbl["split"], value_set=pa.array(list(splits), type=pa.string()))
    tbl = tbl.filter(keep)
    n = tbl.num_rows
    masks_ok = True
    if channels == "F4" and n > 0:
        for name in USER_CONTEXT_MASKS:
            vals = tbl[name].to_numpy(zero_copy_only=False)
            if not np.all((vals == 0.0) | (vals == 1.0)):
                masks_ok = False
    return n, masks_ok


def _process_bucket(dataset_root: Path, layout: str, b: int, splits: tuple, channels: str,
                     ctx_len: int, normalizers: Normalizers, bucket_offset: int) -> dict:
    path = Path(dataset_root) / "model_ready" / f"user_bucket={b}" / "features.parquet"
    cols = _needed_feature_columns(channels)
    tbl = pq.read_table(path, columns=cols)
    keep = pc.is_in(tbl["split"], value_set=pa.array(list(splits), type=pa.string()))
    tbl = tbl.filter(keep)
    n = tbl.num_rows

    result: dict = {"n": n, "decisions": {s: _empty_decision_arrays() for s in splits}}
    if n == 0:
        result["item_tokens"] = np.zeros(0, dtype=np.int32)
        if channels == "F4":
            result["category_tokens"] = np.zeros(0, dtype=np.int16)
            result["brand_tokens"] = np.zeros(0, dtype=np.int16)
            result["main_category_tokens"] = np.zeros(0, dtype=np.uint8)
            result["daypart_tokens"] = np.zeros(0, dtype=np.uint8)
            result["weekday_tokens"] = np.zeros(0, dtype=np.uint8)
            result["event_numeric_features"] = np.zeros((0, F_NUMERIC), dtype=np.float32)
            result["event_quality_flags"] = np.zeros((0, Q_FLAGS), dtype=np.int8)
            result["user_context"] = np.zeros((0, U_CTX), dtype=np.float32)
            result["user_context_masks_f32"] = np.zeros((0, M_CTX), dtype=np.float32)
        return result

    raw = {name: tbl[name].to_numpy(zero_copy_only=False) for name in cols if name != "split"}

    user_id = raw["user_id"]
    segment_index = raw["segment_index"]
    position_in_segment = raw["position_in_segment"]
    decision_id_local = raw["decision_id"]
    if np.any(np.isnan(segment_index)):
        raise AssertionError(f"bucket {b}: NaN segment_index encountered")

    # Reproduces BucketStore's row order EXACTLY: np.lexsort((position_in_segment,
    # segment_index, user_id)), applied here to the retained (in-`splits`) subset only.
    # Equivalent to lexsorting the FULL bucket then dropping out-of-split rows, because the
    # sort key of a retained row never depends on a dropped row (lexsort is stable).
    order = np.lexsort((position_in_segment, segment_index, user_id))

    def reorder(a):
        return a[order]

    user_id = reorder(user_id)
    segment_index = reorder(segment_index)
    position_in_segment = reorder(position_in_segment)
    decision_id_local = reorder(decision_id_local)
    product_idx = reorder(raw["product_idx"])
    prev_view_raw = reorder(raw["log1p_prev_view_count"])

    newseg = np.empty(n, dtype=bool)
    newseg[0] = True
    if n > 1:
        newseg[1:] = (user_id[1:] != user_id[:-1]) | (segment_index[1:] != segment_index[:-1])
    seg_start = np.flatnonzero(newseg)
    seg_of_row = np.cumsum(newseg) - 1
    local_position = np.arange(n) - seg_start[seg_of_row] + 1
    if not np.array_equal(local_position, position_in_segment):
        bad = np.flatnonzero(local_position != position_in_segment)[:5]
        raise AssertionError(
            f"bucket {b}: retained-row position mismatch at local rows {bad.tolist()} -- a "
            "retained decision's window would touch a row dropped by the `splits` filter")

    result["item_tokens"] = product_idx.astype(np.int32, copy=False)
    length_local = np.minimum(ctx_len, position_in_segment).astype(np.int16)

    if channels == "F4":
        result["category_tokens"] = reorder(raw["category_idx"]).astype(np.int16, copy=False)
        result["brand_tokens"] = reorder(raw["brand_idx"]).astype(np.int16, copy=False)
        result["main_category_tokens"] = reorder(raw["main_category_idx"]).astype(np.uint8, copy=False)
        result["daypart_tokens"] = reorder(raw["daypart_idx"]).astype(np.uint8, copy=False)
        result["weekday_tokens"] = reorder(raw["local_weekday_idx"]).astype(np.uint8, copy=False)

        numeric = np.zeros((n, F_NUMERIC), dtype=np.float32)
        price_missing = reorder(raw["price_missing"])
        price_conflict = reorder(raw["price_conflict"])
        pmiss = price_missing | price_conflict
        for fi, name in enumerate(EVENT_NUMERIC_BLOCK):
            rv = reorder(raw[name]).astype(np.float64)
            if name in NORMALIZED_EVENT_NUMERIC or name in NORMALIZED_SESSION_PREFIX:
                v = normalizers.apply(name, rv)
            else:
                v = rv.astype(np.float32)
            flag = MISSING_FLAG_OF.get(name)
            if flag == "price_missing_or_conflict":
                v = np.where(pmiss, 0.0, v).astype(np.float32)
            elif flag is not None:
                fcol = reorder(raw[flag])
                v = np.where(fcol, 0.0, v).astype(np.float32)
            numeric[:, fi] = v
        result["event_numeric_features"] = numeric

        flags = np.zeros((n, Q_FLAGS), dtype=np.int8)
        for qi, name in enumerate(EVENT_QUALITY_FLAGS):
            flags[:, qi] = reorder(raw[name]).astype(np.int8)
        result["event_quality_flags"] = flags

        uctx = np.zeros((n, U_CTX), dtype=np.float32)
        for ui, name in enumerate(USER_CONTEXT_FEATURES):
            rv = reorder(raw[name]).astype(np.float64)
            if name in NORMALIZED_USER_CONTEXT:
                val = normalizers.apply(name, rv)
            else:
                val = rv.astype(np.float32)
            sup = USER_MISSING_FLAG_OF.get(name)
            if sup is not None:
                supcol = reorder(raw[sup])
                val = np.where(supcol == 0.0, 0.0, val).astype(np.float32)
            uctx[:, ui] = val
        result["user_context"] = uctx

        umask_f = np.zeros((n, M_CTX), dtype=np.float32)
        for mi, name in enumerate(USER_CONTEXT_MASKS):
            umask_f[:, mi] = reorder(raw[name]).astype(np.float32)
        result["user_context_masks_f32"] = umask_f

    # -- decisions_index join, per split, vectorised (sort + searchsorted, no Python dict) --
    order_by_did = np.argsort(decision_id_local, kind="stable")
    sorted_did = decision_id_local[order_by_did]

    for s in splits:
        t = read_index_projected(dataset_root, layout, s, META_COLUMNS, buckets=[b])
        dec = t["decision_id"].to_numpy(zero_copy_only=False)
        cli = t["client_idx"].to_numpy(zero_copy_only=False)
        tgt = t["target_class"].to_numpy(zero_copy_only=False)
        lm = t["loss_mask"].to_numpy(zero_copy_only=False)
        em = t["eval_mask"].to_numpy(zero_copy_only=False)
        pos_idx = t["position_in_segment"].to_numpy(zero_copy_only=False)
        if dec.shape[0] == 0:
            continue

        pos_in_sorted = np.searchsorted(sorted_did, dec)
        pos_in_sorted = np.clip(pos_in_sorted, 0, n - 1)
        local_rows = order_by_did[pos_in_sorted]
        matched = decision_id_local[local_rows] == dec
        if not np.all(matched):
            bad = dec[~matched][:5]
            raise AssertionError(f"bucket {b} split {s}: decisions_index decision_id(s) not found "
                                  f"among retained feature rows: {bad.tolist()}")
        if not np.array_equal(position_in_segment[local_rows], pos_idx):
            raise AssertionError(f"bucket {b} split {s}: position_in_segment disagreement between "
                                  "decisions_index and features for a matched decision_id")
        if np.any(cli == 0):
            raise AssertionError(f"bucket {b} split {s}: client_idx == 0 (PAD) in decisions_index")

        keep2 = lm | em
        if not np.any(keep2):
            continue
        sel_local = local_rows[keep2]
        global_row = (bucket_offset + sel_local).astype(np.int64)
        tgt_sel = tgt[keep2]
        lm_sel = lm[keep2]
        em_sel = em[keep2]
        if np.any(lm_sel & (tgt_sel < 0)):
            raise AssertionError(f"bucket {b} split {s}: loss_mask true with target_class < 0")
        prefix_pos = position_in_segment[sel_local]
        if np.any(prefix_pos >= 2 ** 31):
            raise AssertionError(f"bucket {b} split {s}: position_in_segment overflows int32 storage")
        prev_view_sel = prev_view_raw[sel_local].astype(np.float64)

        result["decisions"][s] = {
            "decision_id": dec[keep2].astype(np.int64),
            "global_row": global_row,
            "length": length_local[sel_local].astype(np.int16),
            "target_class": tgt_sel.astype(np.int64),
            "loss_mask": lm_sel.astype(bool),
            "eval_mask": em_sel.astype(bool),
            "client_idx": cli[keep2].astype(np.int64),
            "user_bucket": np.full(int(keep2.sum()), b, dtype=np.int16),
            "prior_view_count": np.rint(np.expm1(prev_view_sel)).astype(np.int32),
            "prefix_position": prefix_pos.astype(np.int32),
        }

    return result


def _global_sorted_distinct_eval_users(dataset_root: Path, layout: str, split: str) -> np.ndarray:
    """Exactly `build_manifests.build_vs_cohort`'s distinct-user construction: pyarrow
    hash-based unique + sort_indices over the FULL split's eval_mask population, all
    buckets, regardless of which buckets this store was built with."""
    t = read_index_projected(dataset_root, layout, split, ["user_id", "eval_mask"], buckets=None)
    t_ev = t.filter(t["eval_mask"])
    uid_col = t_ev["user_id"]
    distinct_arrow = pc.unique(uid_col)
    sort_idx = pc.sort_indices(distinct_arrow)
    distinct_sorted_arrow = distinct_arrow.take(sort_idx)
    return distinct_sorted_arrow.to_numpy(zero_copy_only=False)


def _lookup_user_ids_for_decisions(dataset_root: Path, layout: str, split: str, buckets,
                                    decision_id_sorted: np.ndarray) -> np.ndarray:
    t = read_index_projected(dataset_root, layout, split, ["decision_id", "user_id", "eval_mask"],
                              buckets=buckets)
    t = t.filter(t["eval_mask"])
    dec = t["decision_id"].to_numpy(zero_copy_only=False)
    uid = t["user_id"].to_numpy(zero_copy_only=False)
    order = np.argsort(dec, kind="stable")
    dec_sorted = dec[order]
    uid_sorted = uid[order]
    pos = np.searchsorted(dec_sorted, decision_id_sorted)
    pos = np.clip(pos, 0, dec_sorted.shape[0] - 1) if dec_sorted.shape[0] else pos
    matched_uid = uid_sorted[pos] if dec_sorted.shape[0] else np.zeros(decision_id_sorted.shape[0], dtype=object)
    if dec_sorted.shape[0]:
        assert np.array_equal(dec_sorted[pos], decision_id_sorted), (
            "eval decision_id not found while looking up user_id for eval_user_key")
    return matched_uid


# ---------------------------------------------------------------------------
# CompactStore
# ---------------------------------------------------------------------------

class CompactStore:
    context_length: int
    channels: str
    splits: tuple

    @classmethod
    def build(cls, dataset_root: Path, *, splits: tuple, channels: str,
              buckets: Optional[list] = None, context_length: int = 50,
              n_threads: int = 3, log=print) -> "CompactStore":
        t_start = time.time()
        dataset_root = Path(dataset_root)
        layout = detect_layout(dataset_root)
        if channels not in ("F0", "F4"):
            raise ValueError(f"channels must be 'F0' or 'F4', got {channels!r}")
        splits = tuple(splits)
        if not splits:
            raise ValueError("splits must be non-empty")

        version = json.loads((dataset_root / "VERSION.json").read_text(encoding="utf-8"))["version_hash"]
        if version != EXPECTED_VERSION_HASH:
            raise AssertionError(f"dataset version_hash {version} != expected {EXPECTED_VERSION_HASH}")
        normalizers = Normalizers.load(dataset_root / "mappings" / "normalizers.json")

        all_buckets = sorted(int(b) for b in (buckets if buckets is not None
                                                else _all_feature_buckets(dataset_root)))
        n_threads = max(1, min(3, int(n_threads)))
        log(f"[full_store] build channels={channels} splits={splits} n_buckets={len(all_buckets)} "
            f"n_threads={n_threads} context_length={context_length}")

        # -- prepass: retained-row counts (for preallocation) + mask-binary proof --
        counts: dict = {}
        masks_binary = True
        for b in all_buckets:
            n_b, mb_ok = _bucket_precount(dataset_root, b, splits, channels)
            counts[b] = n_b
            masks_binary = masks_binary and mb_ok
        n_total = sum(counts.values())
        offsets: dict = {}
        off = 0
        for b in all_buckets:
            offsets[b] = off
            off += counts[b]
        assert off == n_total
        mask_dtype = np.uint8 if (channels == "F4" and masks_binary) else np.float32
        log(f"[full_store] prepass done n_total_rows={n_total} mask_dtype={np.dtype(mask_dtype).name}")

        arrays: dict = {"item_tokens": np.empty(n_total, dtype=np.int32)}
        if channels == "F4":
            arrays["category_tokens"] = np.empty(n_total, dtype=np.int16)
            arrays["brand_tokens"] = np.empty(n_total, dtype=np.int16)
            arrays["main_category_tokens"] = np.empty(n_total, dtype=np.uint8)
            arrays["daypart_tokens"] = np.empty(n_total, dtype=np.uint8)
            arrays["weekday_tokens"] = np.empty(n_total, dtype=np.uint8)
            arrays["event_numeric_features"] = np.empty((n_total, F_NUMERIC), dtype=np.float32)
            arrays["event_quality_flags"] = np.empty((n_total, Q_FLAGS), dtype=np.int8)
            arrays["user_context"] = np.empty((n_total, U_CTX), dtype=np.float32)
            arrays["user_context_masks"] = np.empty((n_total, M_CTX), dtype=mask_dtype)

        def work(b):
            # Copy this bucket's rows into the preallocated arrays INSIDE the worker and return
            # only the small per-decision metadata. Holding every bucket's full result until
            # all 64 finished (the original design) kept a second copy of the whole store alive
            # at the end of the build (measured peak 27.6 GiB). Buckets own disjoint
            # [offset, offset+n) slices, so concurrent writes never overlap. Values are unchanged.
            res = _process_bucket(dataset_root, layout, b, splits, channels, context_length,
                                  normalizers, offsets[b])
            n_b = res["n"]
            if n_b > 0:
                lo, hi = offsets[b], offsets[b] + n_b
                arrays["item_tokens"][lo:hi] = res["item_tokens"]
                if channels == "F4":
                    arrays["category_tokens"][lo:hi] = res["category_tokens"]
                    arrays["brand_tokens"][lo:hi] = res["brand_tokens"]
                    arrays["main_category_tokens"][lo:hi] = res["main_category_tokens"]
                    arrays["daypart_tokens"][lo:hi] = res["daypart_tokens"]
                    arrays["weekday_tokens"][lo:hi] = res["weekday_tokens"]
                    arrays["event_numeric_features"][lo:hi] = res["event_numeric_features"]
                    arrays["event_quality_flags"][lo:hi] = res["event_quality_flags"]
                    arrays["user_context"][lo:hi] = res["user_context"]
                    m = res["user_context_masks_f32"]
                    arrays["user_context_masks"][lo:hi] = m.astype(mask_dtype) if mask_dtype == np.uint8 else m
            return {"n": n_b, "decisions": res["decisions"]}

        results: dict = {}
        if n_threads > 1 and len(all_buckets) > 1:
            with ThreadPoolExecutor(max_workers=n_threads) as ex:
                futs = {b: ex.submit(work, b) for b in all_buckets}
                for b, fut in futs.items():
                    results[b] = fut.result()
        else:
            for b in all_buckets:
                results[b] = work(b)

        split_accum: dict = {s: [] for s in splits}
        bucket_log = []
        for b in all_buckets:
            res = results[b]
            bucket_log.append({"bucket": b, "n_rows": int(res["n"])})
            for s in splits:
                split_accum[s].append(res["decisions"][s])
        del results

        store = cls.__new__(cls)
        store.dataset_root = dataset_root
        store.layout = layout
        store.splits = splits
        store.channels = channels
        store.context_length = int(context_length)
        store.buckets = all_buckets
        store.n_rows = int(n_total)
        store.version_hash = version
        store._masks_dtype = np.dtype(mask_dtype).name if channels == "F4" else None
        store._item_tokens = arrays["item_tokens"]
        if channels == "F4":
            store._category_tokens = arrays["category_tokens"]
            store._brand_tokens = arrays["brand_tokens"]
            store._main_category_tokens = arrays["main_category_tokens"]
            store._daypart_tokens = arrays["daypart_tokens"]
            store._weekday_tokens = arrays["weekday_tokens"]
            store._event_numeric_features = arrays["event_numeric_features"]
            store._event_quality_flags = arrays["event_quality_flags"]
            store._user_context = arrays["user_context"]
            store._user_context_masks = arrays["user_context_masks"]

        store._split_decisions = {}
        for s in splits:
            pieces = [p for p in split_accum[s]]
            if pieces:
                merged = {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
            else:
                merged = _empty_decision_arrays()
            store._split_decisions[s] = merged
        store._decisions_cache = {}
        del split_accum
        _release_freed_memory()
        store.build_seconds = time.time() - t_start
        store.bucket_log = bucket_log
        log(f"[full_store] build done n_rows={n_total} seconds={store.build_seconds:.1f} "
            f"mask_dtype={store._masks_dtype}")
        return store

    # -- decisions ---------------------------------------------------------
    def decisions(self, split: str, kind: str) -> DecisionSet:
        if split not in self.splits:
            raise ValueError(f"split {split!r} not loaded by this store (loaded: {self.splits})")
        if kind not in ("loss", "eval"):
            raise ValueError(f"kind must be 'loss' or 'eval', got {kind!r}")
        key = (split, kind)
        if key in self._decisions_cache:
            return self._decisions_cache[key]

        arr = self._split_decisions[split]
        mask = arr["loss_mask"] if kind == "loss" else arr["eval_mask"]
        idx = np.flatnonzero(mask)
        decision_id = arr["decision_id"][idx]
        order = np.argsort(decision_id, kind="stable")
        decision_id = decision_id[order]
        if decision_id.shape[0] > 1:
            assert np.all(np.diff(decision_id) > 0), (
                f"decision_id not strictly ascending for split={split} kind={kind} "
                "(duplicate decision_id?)")
        sel = idx[order]

        end_row = arr["global_row"][sel].astype(np.int64)
        length = arr["length"][sel]
        target_class = arr["target_class"][sel].astype(np.int64)
        loss_mask = arr["loss_mask"][sel].astype(bool)
        eval_mask = arr["eval_mask"][sel].astype(bool)
        client_idx = arr["client_idx"][sel].astype(np.int64)
        user_bucket = arr["user_bucket"][sel].astype(np.int16)
        prior_view_count = arr["prior_view_count"][sel]
        prefix_position = arr["prefix_position"][sel]

        if kind == "loss":
            if np.any(loss_mask):
                assert np.all(target_class[loss_mask] >= 0), "loss_mask true with target_class < 0"
            eval_user_key = None
        else:
            eval_user_key = self._compute_eval_user_key(split, decision_id)

        ds = DecisionSet(split=split, kind=kind, decision_id=decision_id, end_row=end_row,
                          length=length, target_class=target_class, loss_mask=loss_mask,
                          eval_mask=eval_mask, client_idx=client_idx, user_bucket=user_bucket,
                          eval_user_key=eval_user_key, prior_view_count=prior_view_count,
                          prefix_position=prefix_position)
        self._decisions_cache[key] = ds
        return ds

    def _compute_eval_user_key(self, split: str, decision_id_sorted: np.ndarray) -> np.ndarray:
        distinct_users = _global_sorted_distinct_eval_users(self.dataset_root, self.layout, split)
        user_ids = _lookup_user_ids_for_decisions(self.dataset_root, self.layout, split,
                                                   self.buckets, decision_id_sorted)
        key = np.searchsorted(distinct_users, user_ids).astype(np.int64)
        if key.shape[0]:
            assert np.array_equal(distinct_users[key], user_ids), "eval_user_key lookup mismatch"
        del distinct_users, user_ids
        _release_freed_memory()
        return key

    # -- batch assembly ------------------------------------------------------
    def batch(self, ds: DecisionSet, positions) -> dict:
        import torch
        positions = np.asarray(positions, dtype=np.int64)
        Lc = self.context_length
        end_row = ds.end_row[positions]
        length = ds.length[positions].astype(np.int64)
        rel = np.arange(Lc, dtype=np.int64)
        idx = end_row[:, None] - (length[:, None] - 1) + rel[None, :]
        valid = rel[None, :] < length[:, None]
        idx_clipped = np.where(valid, idx, 0)

        def gather_tok(arr):
            g = arr[idx_clipped]
            g = np.where(valid, g, 0).astype(np.int64)
            return g

        out: dict = {}
        out["item_tokens"] = torch.from_numpy(gather_tok(self._item_tokens))
        out["attention_mask"] = torch.from_numpy(np.ascontiguousarray(valid))
        pos_ids = np.where(valid, rel[None, :] + 1, 0).astype(np.int64)
        out["position_ids"] = torch.from_numpy(pos_ids)
        out["lengths"] = torch.from_numpy(length)
        out["last_valid_index"] = torch.from_numpy((length - 1).astype(np.int64))
        out["target_class"] = torch.from_numpy(ds.target_class[positions].astype(np.int64))
        out["loss_mask"] = torch.from_numpy(ds.loss_mask[positions].astype(bool))
        out["eval_mask"] = torch.from_numpy(ds.eval_mask[positions].astype(bool))
        out["decision_id"] = torch.from_numpy(ds.decision_id[positions].astype(np.int64))
        out["client_id"] = torch.from_numpy(ds.client_idx[positions].astype(np.int64))

        if self.channels == "F4":
            out["category_tokens"] = torch.from_numpy(gather_tok(self._category_tokens))
            out["brand_tokens"] = torch.from_numpy(gather_tok(self._brand_tokens))
            out["main_category_tokens"] = torch.from_numpy(gather_tok(self._main_category_tokens))
            out["daypart_tokens"] = torch.from_numpy(gather_tok(self._daypart_tokens))
            out["weekday_tokens"] = torch.from_numpy(gather_tok(self._weekday_tokens))

            num = self._event_numeric_features[idx_clipped]
            num = np.where(valid[..., None], num, 0.0).astype(np.float32)
            out["event_numeric_features"] = torch.from_numpy(num)

            flags = self._event_quality_flags[idx_clipped]
            flags = np.where(valid[..., None], flags, 0).astype(np.int8)
            out["event_quality_flags"] = torch.from_numpy(flags)

            uctx = self._user_context[end_row].astype(np.float32, copy=False)
            out["user_context"] = torch.from_numpy(np.ascontiguousarray(uctx))
            umask = self._user_context_masks[end_row].astype(np.float32, copy=False)
            out["user_context_masks"] = torch.from_numpy(np.ascontiguousarray(umask))

        return out

    def decision_item_tokens(self, ds: DecisionSet) -> np.ndarray:
        return self._item_tokens[ds.end_row].astype(np.int64)
