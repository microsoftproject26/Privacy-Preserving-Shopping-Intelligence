"""v4hist.windows (owner: Data/integration engineer, DATA-05) — the window resolver of both history policies.

Authority: DECISION_LOG 35-36 (owner). USER_HISTORY_L50 (locked) = the last 50 representative events of the user's
whole prior stream, across model segments and partition fences, strictly causal, identical for TRAIN / VALIDATION /
TEST. SESSION_L50 = the v1 session-scoped window (snap_v1/reuse/vendor/rees46_eda/loaders.py L263-L270:
take = min(L, position_in_segment), asserted inside the model segment), kept as a named diagnostic only.
Targets, loss_mask, eval_mask, censoring and the same-partition target rule are UNCHANGED: only the input window moves.

How (no locked code is changed; the snapshot CompactStore and CleanV2Store are used as they are):
  * `build_row_index` reads, per user bucket, the key columns of the SAME feature file the CompactStore reads
    (user_id, segment_index, position_in_segment, decision_id, block_ts, split) through v4data.manifests.read_non_test
    (TEST rows are never materialised), keeps the store's partitions, and sorts with the store's own key
    np.lexsort((position_in_segment, segment_index, user_id)) (full_store.py L237, loaders.py L96). Buckets are
    concatenated in ascending order, exactly as CompactStore.build places them, so index row i IS store row i.
    Build-time proofs (fail-closed, HistoryOrderError): positions are 1..k inside every segment (the keys are unique,
    so both sorts give one order), and inside every user the event timestamp and the partition never decrease, and
    a model segment never spans two partitions. A user's rows are therefore one contiguous, chronological run.
  * `align_to_store` proves index row i == store row i: the same buckets and per-bucket row counts, and for every
    retained decision of the store, the same decision_id, position_in_segment and partition at its global row.
  * For a decision at store row r of user u: USER_HISTORY_L50 length = min(50, rank_u(r) + 1), i.e. the slice
    [max(user_start[u], r - 49), r]; SESSION_L50 length = min(50, position_in_segment(r)), the store's own value
    (asserted equal). `policy_decision_set` returns the store's DecisionSet with `length` replaced (USER) or unchanged
    (SESSION). The snapshot CompactStore.batch then gathers rows [end_row - length + 1, end_row], so the current event
    is the last position, exactly as in the session window, and it fills every per-event channel (the 6 categorical
    token channels, the numeric and flag blocks), attention_mask, position_ids (1..length), lengths and
    last_valid_index from those rows; user_context / user_context_masks are the decision row's values, unchanged.
    CleanV2Store.batch -> clean_v2_view then produces the Clean V2 layout (15 numeric, 15 flags, 12 + 4).

Per-event channels keep their STORED per-event values (documented choice, DATA-05 item 1). Several are
session-relative in v1: log1p_segment_elapsed_s and the prefix_* / run-length counts restart at every model-segment
start, and log1p_prefix_elapsed_s (a dropped v1 column) likewise. In a USER_HISTORY window that spans several
segments these channels therefore form one restarting run per segment; nothing is recomputed across the boundary.
Every such value was computed from events at or before its own event, so it is causal for any later decision.

Session offsets (DECISION_LOG 47, owner): USER_HISTORY_L50 batches carry `session_offset_tokens` int64 [B, 50]:
for window position j of decision d, offset = session(d) - session(j), the number of v1 BEHAVIOURAL sessions between
event j's session and the decision's current session (current = 0, previous = 1, ...), token = min(offset, 7) + 1 on
valid positions and 0 on PAD (vocabulary 9). The session is the stored v1 `session_index` (EDA pipeline4.py `sess2`:
a new session starts iff the gap between consecutive raw blocks of the user, over ALL blocks, pre-compression, is
> 3600 s; it is never reset at a partition fence), so a session straddling a fence is one session, unlike the model
segments (which also break at fences and at barrier blocks). Every session involved is at or before the decision, so
the offset is strictly causal, and the same function serves TRAIN and evaluation. SESSION_L50 batches keep the exact
v1 key set (no session_offset_tokens: inside one model segment every offset would be 0, i.e. every token 1).

Session-mode candidate features (DECISION_LOG 48, owner; CANDIDATE = droppable), USER_HISTORY_L50 only, computed from
the window's own `main_category_tokens` (the stored per-event v1 main category, vocabulary 16: PAD 0, MISSING 1,
OOV 2, known 3..15; the category -> main_category map is not a function on real rows, so the per-event stored token is
the exact value) and the same behavioural sessions, all INSIDE the 50-event window:
  session_mode_main_category_tokens int64 [B, 50]: an event of a PAST session (offset >= 1) gets the mode of the known
      main categories over that session's events in the window; an event of the CURRENT session gets the causal prefix
      mode up to and including itself; ties go to the category whose latest occurrence is most recent; no known
      category -> the UNKNOWN token (MISSING = 1); PAD 0;
  current_matches_session_mode float32 [B] and current_matches_session_mode_available float32 [B]: 1 iff the decision
      event's known main category equals the current-session prefix mode at the decision position (which includes the
      decision event); the mask is 0 when either is unknown (then the value is 0);
  session_mode_share float32 [B]: count(mode) / known-category events of the current session so far (in the window,
      including the decision event); 0 whenever the mask is 0.

`check_windows` is the leakage / structure proof used by the tests and the server measurement: every window row is
the decision's user, at or before the decision in the user's stream, has a timestamp <= the decision's and < the
end of the decision's partition, is never from a later partition (so never TEST for a TRAIN / VALIDATION decision),
the window is one right-padded contiguous run ending at the decision, and its length follows the policy rule.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence

import ppsi_env  # noqa: F401  (isolation first: the snapshot modules come from code/snap_v1)

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from v4data.common import TEST_END_US, TRAIN_END_US, VALIDATION_END_US, VS_END_US, refuse_test_split
from v4data.manifests import _ts_us, read_non_test

from .policy import (CONTEXT_LENGTH, NON_TEST_ORDER, RESOLVER_ID, SESSION_L50, USER_HISTORY_L50, HistoryPolicyError,
                     history_splits, require_policy, store_splits)

INDEX_COLUMNS = ("user_id", "segment_index", "position_in_segment", "decision_id", "block_ts", "split",
                 "session_index")
SESSION_OFFSET_KEY = "session_offset_tokens"
SESSION_OFFSET_CAP = 7                       # offsets >= 7 share one token
SESSION_OFFSET_VOCAB = SESSION_OFFSET_CAP + 2   # 0 = PAD, 1..8 = offset 0..>=7
SESSION_MODE_KEY = "session_mode_main_category_tokens"
MATCH_KEY = "current_matches_session_mode"
MATCH_MASK_KEY = "current_matches_session_mode_available"
SHARE_KEY = "session_mode_share"
USER_ONLY_KEYS = (SESSION_OFFSET_KEY, SESSION_MODE_KEY, MATCH_KEY, MATCH_MASK_KEY, SHARE_KEY)
MAIN_CATEGORY_VOCAB = 16                      # snap_v1/reuse/train.py VOCAB_SIZES["main_category_tokens"]
MAIN_KNOWN_START = 3                          # PAD 0, MISSING 1, OOV 2; known >= 3
MAIN_UNKNOWN = 1                              # the UNKNOWN token of the session-mode channel (= MISSING)
# partition codes in chronological order; TEST (3) exists only for synthetic fixtures and the checks (the real index
# builder refuses TEST)
SPLIT_CODES = {"TRAIN": 0, "VALIDATION_SELECT": 1, "VALIDATION_CONFIRM": 2, "TEST": 3}
TEST_CODE = SPLIT_CODES["TEST"]
PARTITION_END_US = np.array([TRAIN_END_US, VS_END_US, VALIDATION_END_US, TEST_END_US], dtype=np.int64)


class HistoryOrderError(AssertionError):
    """The (user_id, segment_index, position_in_segment) order is not chronological, or segment positions are broken."""


class HistoryAlignmentError(AssertionError):
    """The row index and the CompactStore rows disagree (index row i must be store row i)."""


class HistoryLeakError(AssertionError):
    """A window row violates causality, user identity, the partition rule or the policy's length rule."""


def _need(cond, msg: str, exc=HistoryLeakError) -> None:
    if not bool(cond):
        raise exc(msg)


# ---------------------------------------------------------------------------------------------------------- index
@dataclass
class RowIndex:
    """Per-row metadata of a CompactStore's retained rows, in the store's own row order (row i == store row i).
    Users are ordinals of this index only (never a raw identifier)."""
    splits: tuple
    buckets: tuple
    bucket_rows: tuple                   # ((bucket, n_rows), ...) in store order
    decision_id: np.ndarray              # int64 [n]
    user_ord: np.ndarray                 # int32 [n]
    user_rank: np.ndarray                # int32 [n]  0-based rank of the row inside its user's retained stream
    position: np.ndarray                 # int32 [n]  position_in_segment (1-based)
    ts_us: np.ndarray                    # int64 [n]  the event's block_ts, UTC microseconds
    split_code: np.ndarray               # int8  [n]  SPLIT_CODES
    n_users: int
    session_id: Optional[np.ndarray] = None   # int32 [n]  the v1 behavioural session_index (per user, never fence-reset)
    order_checks: dict = field(default_factory=dict)
    _did_order: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def n_rows(self) -> int:
        return int(self.decision_id.shape[0])

    def rows_of(self, decision_ids) -> np.ndarray:
        """Index rows of `decision_ids` (each must be present exactly once)."""
        if self._did_order is None:
            self._did_order = np.argsort(self.decision_id, kind="stable")
        d = np.asarray(decision_ids, dtype=np.int64)
        srt = self.decision_id[self._did_order]
        p = np.searchsorted(srt, d)
        pc_ = np.minimum(p, max(srt.size - 1, 0))
        if srt.size == 0 or not np.array_equal(srt[pc_], d):
            raise HistoryAlignmentError("a requested decision_id is not a row of this index")
        return self._did_order[pc_].astype(np.int64)

    def nbytes(self) -> int:
        arrs = [self.decision_id, self.user_ord, self.user_rank, self.position, self.ts_us, self.split_code]
        if self.session_id is not None:
            arrs.append(self.session_id)
        return int(sum(a.nbytes for a in arrs))

    @classmethod
    def from_arrays(cls, *, user_ord, position, ts_us, split_code, decision_id=None, splits=None,
                    session_id=None, check: bool = True) -> "RowIndex":
        """An index over rows already in (user, chronological) order — synthetic fixtures. user_rank is derived; the
        same order checks as the real builder run when `check`. `session_id` None = one session per model segment."""
        uo = np.asarray(user_ord, dtype=np.int64)
        n = uo.size
        newu = np.ones(n, dtype=bool)
        if n > 1:
            newu[1:] = uo[1:] != uo[:-1]
        starts = np.flatnonzero(newu)
        rank = (np.arange(n) - starts[np.cumsum(newu) - 1]).astype(np.int32)
        pos = np.asarray(position, dtype=np.int32)
        ts = np.asarray(ts_us, dtype=np.int64)
        sc = np.asarray(split_code, dtype=np.int8)
        newseg = _segment_starts(newu, pos)
        sid = (np.cumsum(newseg) if session_id is None else np.asarray(session_id)).astype(np.int32)
        oc = _order_checks(newu, newseg, pos, ts, sc, sid) if check else {}
        did = np.arange(n, dtype=np.int64) if decision_id is None else np.asarray(decision_id, dtype=np.int64)
        return cls(splits=tuple(splits or ()), buckets=(), bucket_rows=(), decision_id=did,
                   user_ord=uo.astype(np.int32), user_rank=rank, position=pos, ts_us=ts, split_code=sc,
                   n_users=int(newu.sum()), order_checks=oc, session_id=sid)


def _segment_starts(newu: np.ndarray, pos: np.ndarray) -> np.ndarray:
    return newu | (pos == 1)


def _order_checks(newu, newseg, pos, ts, sc, sid=None) -> dict:
    """The chronology proofs of one bucket (or a synthetic fixture), fail-closed."""
    n = pos.size
    seg_start = np.flatnonzero(newseg)
    local = np.arange(n) - seg_start[np.cumsum(newseg) - 1] + 1
    _need(np.array_equal(local, pos), "position_in_segment is not 1..k inside a model segment (keys not unique or "
                                      "rows missing): the store order cannot be reproduced", HistoryOrderError)
    same_user = ~newu[1:] if n > 1 else np.zeros(0, dtype=bool)
    same_seg = ~newseg[1:] if n > 1 else np.zeros(0, dtype=bool)
    ts_bad = int(np.count_nonzero(ts[1:][same_user] < ts[:-1][same_user])) if n > 1 else 0
    sp_bad = int(np.count_nonzero(sc[1:][same_user] < sc[:-1][same_user])) if n > 1 else 0
    fence_bad = int(np.count_nonzero(sc[1:][same_seg] != sc[:-1][same_seg])) if n > 1 else 0
    _need(ts_bad == 0, f"{ts_bad} rows: the event timestamp decreases inside a user's stream (order not "
                       "chronological)", HistoryOrderError)
    _need(sp_bad == 0, f"{sp_bad} rows: the partition decreases inside a user's stream", HistoryOrderError)
    _need(fence_bad == 0, f"{fence_bad} rows: a model segment spans two partitions (fence reset violated)",
          HistoryOrderError)
    _need(bool(np.all(ts >= 0)), "a NULL / negative event timestamp", HistoryOrderError)
    out = {"rows": int(n), "users": int(newu.sum()), "segments": int(newseg.sum()),
           "user_row_pairs_checked": int(same_user.sum()), "ts_decreasing": ts_bad, "partition_decreasing": sp_bad,
           "segment_spans_partitions": fence_bad}
    if sid is not None and n > 1:
        d = np.diff(sid.astype(np.int64))[same_user]
        ses_bad = int(np.count_nonzero(d < 0))
        _need(ses_bad == 0, f"{ses_bad} rows: the behavioural session decreases inside a user", HistoryOrderError)
        new_ses = np.ones(n, dtype=bool)
        new_ses[1:] = ~same_user | (np.diff(sid.astype(np.int64)) != 0)
        cross = int(np.count_nonzero(new_ses & ~newseg))
        _need(cross == 0, f"{cross} rows: a behavioural session starts inside a model segment", HistoryOrderError)
        fence = np.zeros(n, dtype=bool)
        fence[1:] = same_user & (sc[1:] != sc[:-1])
        out.update(sessions=int(new_ses.sum()), session_starts_inside_segment=cross, session_decreasing=ses_bad,
                   fence_rows=int(fence.sum()), fence_rows_same_session=int(np.count_nonzero(fence & ~new_ses)),
                   session_gaps_gt1=int(np.count_nonzero(d > 1)))
    return out


def _split_codes(col) -> np.ndarray:
    idx = pc.index_in(col, value_set=pa.array(list(NON_TEST_ORDER), type=pa.string()))
    if idx.null_count:
        raise HistoryOrderError("a row outside TRAIN / VALIDATION_SELECT / VALIDATION_CONFIRM reached the index")
    return np.asarray(idx.to_numpy(zero_copy_only=False), dtype=np.int8)


def _bucket_index(ds_root: Path, b: int, splits: tuple) -> dict:
    path = Path(ds_root) / "model_ready" / f"user_bucket={int(b)}" / "features.parquet"
    t = read_non_test(path, list(INDEX_COLUMNS))                     # split IN non-TEST as a scan filter
    t = t.filter(pc.is_in(t["split"], value_set=pa.array(list(splits), type=pa.string())))
    n = t.num_rows
    if n == 0:
        z = lambda dt: np.zeros(0, dtype=dt)  # noqa: E731
        return {"n": 0, "users": 0, "decision_id": z(np.int64), "user_ord": z(np.int64), "user_rank": z(np.int32),
                "position": z(np.int32), "ts_us": z(np.int64), "split_code": z(np.int8), "session_id": z(np.int32),
                "checks": {"rows": 0}}
    if t["user_id"].null_count:
        raise HistoryOrderError(f"bucket {b}: NULL user_id")
    uid = t["user_id"].to_numpy(zero_copy_only=False)
    seg = t["segment_index"].to_numpy(zero_copy_only=False)
    pos = t["position_in_segment"].to_numpy(zero_copy_only=False)
    if np.any(np.isnan(seg)):
        raise HistoryOrderError(f"bucket {b}: NaN segment_index")
    o = np.lexsort((pos, seg, uid))                  # the store's key (full_store.py L237; loaders.py L96)
    uid, seg, pos = uid[o], seg[o], pos[o]
    newu = np.ones(n, dtype=bool)
    newu[1:] = uid[1:] != uid[:-1]
    newseg = newu.copy()
    newseg[1:] |= seg[1:] != seg[:-1]
    del uid
    ts = _ts_us(t["block_ts"])[o]
    sc = _split_codes(t["split"])[o]
    did = np.asarray(t["decision_id"].to_numpy(zero_copy_only=False), dtype=np.int64)[o]
    ses = np.asarray(t["session_index"].to_numpy(zero_copy_only=False), dtype=np.float64)[o]
    del t
    if np.any(~np.isfinite(ses)) or np.any(ses != np.round(ses)) or np.any(ses < 0) or np.any(ses >= 2 ** 31):
        raise HistoryOrderError(f"bucket {b}: session_index is not a non-negative integer")
    sid = ses.astype(np.int32)
    pos32 = np.asarray(pos, dtype=np.int32)
    checks = _order_checks(newu, newseg, pos32, ts, sc, sid)
    starts = np.flatnonzero(newu)
    uof = np.cumsum(newu) - 1
    rank = (np.arange(n) - starts[uof]).astype(np.int32)
    return {"n": int(n), "users": int(starts.size), "decision_id": did, "user_ord": uof.astype(np.int64),
            "user_rank": rank, "position": pos32, "ts_us": ts, "split_code": sc, "session_id": sid, "checks": checks}


def build_row_index(ds_root, buckets: Sequence[int], splits: Sequence[str]) -> RowIndex:
    """The RowIndex of the rows a CompactStore built with (`splits`, `buckets`) retains, in its row order."""
    sp = tuple(splits)
    for s in sp:
        refuse_test_split(s)
        if s not in NON_TEST_ORDER:
            raise HistoryPolicyError(f"unknown split {s!r}")
    bl = sorted(int(b) for b in buckets)                # CompactStore.build sorts its buckets
    parts, rows, off_u, checks = [], [], 0, {}
    for b in bl:
        r = _bucket_index(Path(ds_root), b, sp)
        r["user_ord"] = r["user_ord"] + off_u
        off_u += r["users"]
        parts.append(r)
        rows.append((b, r["n"]))
        checks[str(b)] = r["checks"]
    cat = lambda k, dt: (np.concatenate([p[k] for p in parts]).astype(dt, copy=False) if parts  # noqa: E731
                         else np.zeros(0, dtype=dt))
    if off_u >= 2 ** 31:
        raise HistoryOrderError("more users than an int32 ordinal holds")
    agg = {k: int(sum(c.get(k, 0) for c in checks.values()))
           for k in ("sessions", "session_starts_inside_segment", "session_decreasing", "fence_rows",
                     "fence_rows_same_session", "session_gaps_gt1")}
    agg.update({"rows": int(sum(c.get("rows", 0) for c in checks.values())),
           "users": int(off_u),
           "segments": int(sum(c.get("segments", 0) for c in checks.values())),
           "user_row_pairs_checked": int(sum(c.get("user_row_pairs_checked", 0) for c in checks.values())),
           "ts_decreasing": int(sum(c.get("ts_decreasing", 0) for c in checks.values())),
           "partition_decreasing": int(sum(c.get("partition_decreasing", 0) for c in checks.values())),
           "segment_spans_partitions": int(sum(c.get("segment_spans_partitions", 0) for c in checks.values()))})
    return RowIndex(splits=sp, buckets=tuple(bl), bucket_rows=tuple(rows), decision_id=cat("decision_id", np.int64),
                    user_ord=cat("user_ord", np.int32), user_rank=cat("user_rank", np.int32),
                    position=cat("position", np.int32), ts_us=cat("ts_us", np.int64),
                    split_code=cat("split_code", np.int8), n_users=int(off_u), order_checks=agg,
                    session_id=cat("session_id", np.int32))


def align_to_store(index: RowIndex, cs) -> dict:
    """Prove index row i == CompactStore row i (fail-closed). `cs` is a snapshot CompactStore built by
    CompactStore.build (it carries bucket_log and _split_decisions)."""
    _need(set(index.splits) == set(cs.splits), f"index splits {index.splits} != store splits {tuple(cs.splits)}",
          HistoryAlignmentError)
    _need(tuple(index.buckets) == tuple(int(b) for b in cs.buckets), "index and store buckets differ",
          HistoryAlignmentError)
    log = tuple((int(x["bucket"]), int(x["n_rows"])) for x in cs.bucket_log)
    _need(log == tuple(index.bucket_rows), "per-bucket row counts differ between the index and the store",
          HistoryAlignmentError)
    _need(int(cs.n_rows) == index.n_rows, "row counts differ between the index and the store", HistoryAlignmentError)
    checked = {}
    for s, arr in cs._split_decisions.items():
        g = np.asarray(arr["global_row"], dtype=np.int64)
        if g.size == 0:
            checked[s] = 0
            continue
        _need(np.array_equal(index.decision_id[g], np.asarray(arr["decision_id"], dtype=np.int64)),
              f"{s}: decision_id at the store's global rows differs from the index", HistoryAlignmentError)
        _need(np.array_equal(index.position[g], np.asarray(arr["prefix_position"], dtype=np.int32)),
              f"{s}: position_in_segment at the store's global rows differs from the index", HistoryAlignmentError)
        _need(bool(np.all(index.split_code[g] == SPLIT_CODES[s])), f"{s}: partition code mismatch",
              HistoryAlignmentError)
        checked[s] = int(g.size)
    return {"n_rows": index.n_rows, "buckets": [int(b) for b in index.buckets], "decisions_checked": checked}


# -------------------------------------------------------------------------------------------------------- lengths
def session_lengths(index: RowIndex, rows, L: int = CONTEXT_LENGTH) -> np.ndarray:
    return np.minimum(int(L), index.position[np.asarray(rows, dtype=np.int64)].astype(np.int64)).astype(np.int16)


def user_lengths(index: RowIndex, rows, L: int = CONTEXT_LENGTH) -> np.ndarray:
    return np.minimum(int(L), index.user_rank[np.asarray(rows, dtype=np.int64)].astype(np.int64) + 1).astype(np.int16)


def policy_lengths(index: RowIndex, rows, policy, L: int = CONTEXT_LENGTH) -> np.ndarray:
    p = require_policy(policy)
    return user_lengths(index, rows, L) if p == USER_HISTORY_L50 else session_lengths(index, rows, L)


def policy_decision_set(ds, index: Optional[RowIndex], policy, L: int = CONTEXT_LENGTH):
    """The snapshot DecisionSet `ds` (end_row = store rows) under `policy`: SESSION_L50 returns `ds` itself (the v1
    lengths, asserted equal to the index rule when an index is given); USER_HISTORY_L50 returns a copy whose `length`
    is min(L, rank + 1). Every other field (targets, masks, ids) is untouched."""
    p = require_policy(policy)
    end = np.asarray(ds.end_row, dtype=np.int64)
    if index is None:
        if p == USER_HISTORY_L50:
            raise HistoryPolicyError("USER_HISTORY_L50 needs the row index of the store")
        return ds
    _need(end.size == 0 or (int(end.min()) >= 0 and int(end.max()) < index.n_rows),
          "a decision's end_row lies outside the index", HistoryAlignmentError)
    _need(np.array_equal(np.asarray(ds.length, dtype=np.int64), session_lengths(index, end, L).astype(np.int64)),
          "the store's own window length != min(L, position_in_segment) of the index (misaligned index)",
          HistoryAlignmentError)
    _need(np.array_equal(index.decision_id[end], np.asarray(ds.decision_id, dtype=np.int64)),
          "decision_id at end_row differs from the index (misaligned index)", HistoryAlignmentError)
    if p == SESSION_L50:
        return ds
    ln = user_lengths(index, end, L).astype(np.asarray(ds.length).dtype, copy=False)
    return dataclasses.replace(ds, length=ln)


def window_rows(end_row, length, L: int = CONTEXT_LENGTH) -> tuple:
    """([B, L] int64 row indices, [B, L] bool valid): exactly the gather of the snapshot CompactStore.batch
    (idx = end_row - (length - 1) + rel, valid = rel < length)."""
    end = np.asarray(end_row, dtype=np.int64)
    ln = np.asarray(length, dtype=np.int64)
    rel = np.arange(int(L), dtype=np.int64)
    idx = end[:, None] - (ln[:, None] - 1) + rel[None, :]
    valid = rel[None, :] < ln[:, None]
    return idx, valid


# --------------------------------------------------------------------------------------------------------- checks
def check_windows(index: RowIndex, decision_rows, idx, valid, *, policy, L: int = CONTEXT_LENGTH) -> dict:
    """The leakage and structure proof of B windows. `decision_rows` are the decisions' own index rows (taken from
    their decision identity, never from the window); `idx` / `valid` are the resolver's windows. Raises
    HistoryLeakError (an AssertionError) on the first violated property; returns counts."""
    pol = require_policy(policy)
    r = np.asarray(decision_rows, dtype=np.int64)
    idx = np.asarray(idx, dtype=np.int64)
    valid = np.asarray(valid, dtype=bool)
    B = int(r.shape[0])
    _need(idx.shape == (B, int(L)) and valid.shape == (B, int(L)), f"window shape {idx.shape} != ({B}, {L})")
    lens = valid.sum(axis=1).astype(np.int64)
    _need(bool(np.all(lens >= 1)), "a window is empty (lengths >= 1 is the contract)")
    iv = idx[valid]
    _need(bool(np.all((iv >= 0) & (iv < index.n_rows))), "a window row lies outside the store")
    rv = np.repeat(r, lens)
    scv, scr = index.split_code[iv], index.split_code[rv]
    _need(not bool(np.any((scv == TEST_CODE) & (scr < TEST_CODE))),
          "a TEST-partition row appears in the window of a non-TEST decision")
    _need(bool(np.all(scv <= scr)), "a window row is from a later partition than its decision")
    tsv, tsr = index.ts_us[iv], index.ts_us[rv]
    _need(bool(np.all(tsv < PARTITION_END_US[scr.astype(np.int64)])),
          "a window row's event timestamp is not before the end of the decision's partition")
    _need(bool(np.all(index.user_ord[iv] == index.user_ord[rv])), "a window row belongs to another user")
    _need(bool(np.all(index.user_rank[iv] <= index.user_rank[rv])),
          "a window row comes after the decision in the user's stream (the window includes a future event)")
    _need(bool(np.all(tsv <= tsr)), "a window row's event timestamp is after the decision's timestamp")
    _need(bool(np.array_equal(valid, np.arange(int(L))[None, :] < lens[:, None])),
          "a window is not a right-padded prefix")
    last = idx[np.arange(B), lens - 1]
    _need(bool(np.array_equal(last, r)), "the window's last position is not the decision event itself")
    if int(L) > 1:
        both = valid[:, 1:] & valid[:, :-1]
        _need(bool(np.all((idx[:, 1:] - idx[:, :-1])[both] == 1)), "a window is not one contiguous run of rows")
    want = policy_lengths(index, r, pol, L).astype(np.int64)
    _need(bool(np.array_equal(lens, want)), f"window lengths differ from the {pol} rule")
    back = index.user_rank[rv].astype(np.int64) - index.user_rank[iv].astype(np.int64)
    earlier_seg = back >= index.position[rv]
    if pol == SESSION_L50:
        _need(not bool(np.any(earlier_seg)), "a SESSION_L50 window leaves its model segment")
    earlier_part = scv < scr
    dec_with_part = np.zeros(B, dtype=bool)
    if earlier_part.any():
        dec_with_part[np.repeat(np.arange(B), lens)[earlier_part]] = True
    return {"policy": pol, "n_decisions": B, "n_window_rows": int(iv.size),
            "rows_from_earlier_segments": int(earlier_seg.sum()), "rows_from_earlier_partitions": int(earlier_part.sum()),
            "decisions_with_earlier_partition_rows": int(dec_with_part.sum())}


def session_offset_tokens(session_id: np.ndarray, end_row, length, L: int = CONTEXT_LENGTH,
                          cap: int = SESSION_OFFSET_CAP) -> np.ndarray:
    """int64 [B, L]: min(session(d) - session(j), cap) + 1 on the window's valid positions, 0 on PAD. `session_id` is
    aligned to the rows that `end_row` indexes (a RowIndex's rows or a cache's kept rows). Fail-closed on a negative
    offset (a window row in a later session than its decision would be a future-event leak)."""
    end = np.asarray(end_row, dtype=np.int64)
    idx, valid = window_rows(end, length, L)
    ic = np.where(valid, idx, 0)
    sid = np.asarray(session_id, dtype=np.int64)
    off = sid[end][:, None] - sid[ic]
    _need(not bool(np.any(off[valid] < 0)), "a window row lies in a later session than its decision (future leak)")
    return np.where(valid, np.minimum(off, int(cap)) + 1, 0).astype(np.int64)


def check_session_offsets(index: RowIndex, decision_rows, idx, valid, tokens, L: int = CONTEXT_LENGTH) -> dict:
    """The session-offset proof of B windows against the index's behavioural sessions (the decisions' own rows come
    from their identity): PAD <-> 0; tokens in 1..8; token = min(session(d) - session(j), 7) + 1 with every offset >= 0
    (causal); offset 0 exactly on the current-session positions; non-decreasing going back in the window; the
    decision's own position is 1."""
    _need(index.session_id is not None, "the index carries no behavioural session ids")
    r = np.asarray(decision_rows, dtype=np.int64)
    idx = np.asarray(idx, dtype=np.int64)
    v = np.asarray(valid, dtype=bool)
    tok = np.asarray(tokens)
    _need(tok.dtype == np.int64 and tok.shape == v.shape, f"session_offset_tokens must be int64 {v.shape}")
    _need(bool(np.all(tok[~v] == 0)), "a PAD position carries a non-zero session token")
    _need(bool(np.all((tok[v] >= 1) & (tok[v] <= SESSION_OFFSET_CAP + 1))), "a valid session token outside 1..8")
    sid = index.session_id.astype(np.int64)
    off = sid[r][:, None] - sid[np.where(v, idx, 0)]
    _need(not bool(np.any(off[v] < 0)), "a window position lies in a later session than the decision (future leak)")
    want = np.where(v, np.minimum(off, SESSION_OFFSET_CAP) + 1, 0)
    _need(bool(np.array_equal(tok, want)), "session tokens differ from the behavioural-session offsets")
    cur = v & (sid[np.where(v, idx, 0)] == sid[r][:, None])
    _need(bool(np.array_equal(tok == 1, cur)), "offset 0 is not exactly the current-session positions")
    if tok.shape[1] > 1:
        both = v[:, 1:] & v[:, :-1]
        _need(bool(np.all((tok[:, :-1] >= tok[:, 1:])[both])), "session offsets decrease going back in the window")
    lens = v.sum(axis=1)
    _need(bool(np.all(tok[np.arange(r.size), lens - 1] == 1)), "the decision's own position is not offset 0")
    return {"n_decisions": int(r.size), "n_positions": int(v.sum()),
            "token_counts_1_to_8": [int(np.count_nonzero(tok[v] == k)) for k in range(1, SESSION_OFFSET_CAP + 2)],
            "decisions_with_an_earlier_session": int(np.count_nonzero((tok > 1).any(axis=1)))}


def session_mode_features(main_tokens, valid, session_at_pos, vocab: int = MAIN_CATEGORY_VOCAB) -> dict:
    """The DECISION_LOG 48 features of B windows, vectorised. `main_tokens` int [B, L] (0 on PAD), `valid` bool
    [B, L] (a right-padded prefix), `session_at_pos` int [B, L] the behavioural session id of each position (sessions
    are contiguous runs of a chronological window; the last valid position is the decision's current session)."""
    m = np.asarray(main_tokens, dtype=np.int64)
    v = np.asarray(valid, dtype=bool)
    sp = np.asarray(session_at_pos, dtype=np.int64)
    B, L = m.shape
    _need(bool(np.all((m[v] >= 0) & (m[v] < vocab))), f"a main_category token outside the vocabulary of {vocab}")
    known = v & (m >= MAIN_KNOWN_START)
    cats = np.arange(vocab)
    oh = (m[..., None] == cats[None, None, :]) & known[..., None]                     # [B, L, V]
    cum = np.cumsum(oh, axis=1, dtype=np.int32)
    j = np.arange(L)
    start = v.copy()
    start[:, 1:] = v[:, 1:] & (sp[:, 1:] != sp[:, :-1])
    rs = np.maximum.accumulate(np.where(start, j[None, :], 0), axis=1)              # the run start of each position
    rows = np.arange(B)[:, None]
    base = cum[rows, rs] - oh[rows, rs].astype(np.int32)                             # counts before the run start
    cnt = cum - base                                                                 # within-run counts up to j
    last = np.maximum.accumulate(np.where(oh, j[None, :, None], -1), axis=1)         # latest occurrence <= j
    score = np.where(cnt > 0, cnt.astype(np.int64) * (L + 1) + last, -1)
    best = score.argmax(axis=2)
    has = score.max(axis=2) >= 0
    prefix = np.where(has, best, MAIN_UNKNOWN)
    end = v.copy()
    end[:, :-1] = v[:, :-1] & (start[:, 1:] | ~v[:, 1:])                             # the run's last position
    re = np.minimum.accumulate(np.where(end, j[None, :], L)[:, ::-1], axis=1)[:, ::-1]
    re = np.minimum(re, L - 1)
    full = prefix[rows, re]                                                          # the run's full mode
    lens = v.sum(axis=1)
    _need(bool(np.all(lens >= 1)), "an empty window")
    lp = lens - 1
    cur_s = sp[np.arange(B), lp]
    current = v & (sp == cur_s[:, None])
    mode_tok = np.where(v, np.where(current, prefix, full), 0).astype(np.int64)
    cur_cat = m[np.arange(B), lp]
    pm = prefix[np.arange(B), lp]
    avail = (cur_cat >= MAIN_KNOWN_START) & (pm >= MAIN_KNOWN_START)
    match = np.where(avail, (cur_cat == pm).astype(np.float32), np.float32(0.0)).astype(np.float32)
    ccnt = cnt[np.arange(B), lp]                                                     # [B, V] current-session counts
    nk = ccnt.sum(axis=1)
    cm = ccnt[np.arange(B), np.where(pm >= MAIN_KNOWN_START, pm, 0)]
    share = np.where(avail, cm / np.maximum(nk, 1), 0.0).astype(np.float32)
    return {SESSION_MODE_KEY: mode_tok, MATCH_KEY: match, MATCH_MASK_KEY: avail.astype(np.float32), SHARE_KEY: share}


def add_session_features(batch: dict, session_id: np.ndarray, end_row, length) -> dict:
    """Adds the DECISION_LOG 47 + 48 USER_HISTORY_L50 keys to a served batch (in place; returns it). `session_id` is
    aligned to the rows `end_row` indexes (a RowIndex or a cache's kept rows)."""
    import torch
    end = np.asarray(end_row, dtype=np.int64)
    ln = np.asarray(length, dtype=np.int64)
    tok = session_offset_tokens(session_id, end, ln)
    idx, valid = window_rows(end, ln)
    sp = np.asarray(session_id, dtype=np.int64)[np.where(valid, idx, 0)]
    mt = batch["main_category_tokens"]
    mt = mt.numpy() if hasattr(mt, "numpy") else np.asarray(mt)
    feats = session_mode_features(mt, valid, sp)
    batch[SESSION_OFFSET_KEY] = torch.from_numpy(tok)
    for k, a in feats.items():
        batch[k] = torch.from_numpy(np.ascontiguousarray(a))
    return batch


TOKEN_ARRAYS = (("item_tokens", "_item_tokens"), ("category_tokens", "_category_tokens"),
                ("brand_tokens", "_brand_tokens"), ("main_category_tokens", "_main_category_tokens"),
                ("daypart_tokens", "_daypart_tokens"), ("weekday_tokens", "_weekday_tokens"))


def _np(x) -> np.ndarray:
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:                                                   # pragma: no cover
        pass
    return np.asarray(x)


def check_served_batch(batch: Mapping, cs, spec, idx, valid) -> dict:
    """The served Clean V2 batch equals the store's event rows at the window rows (every per-event channel, the
    attention mask, positions and lengths): ties the bytes a model receives to the rows `check_windows` proved."""
    v = np.asarray(valid, dtype=bool)
    ic = np.where(v, np.asarray(idx, dtype=np.int64), 0)
    for key, attr in TOKEN_ARRAYS:
        want = np.where(v, getattr(cs, attr)[ic], 0).astype(np.int64)
        _need(np.array_equal(_np(batch[key]), want), f"served {key} differs from the store rows of the window")
    num = np.where(v[..., None], cs._event_numeric_features[ic], 0.0).astype(np.float32)[..., list(spec.num_src)]
    _need(np.array_equal(_np(batch["event_numeric_features"]), num),
          "served event_numeric_features differ from the store rows of the window")
    flg = np.where(v[..., None], cs._event_quality_flags[ic], 0).astype(np.int8)[..., list(spec.flag_src)]
    _need(np.array_equal(_np(batch["event_quality_flags"]), flg),
          "served event_quality_flags differ from the store rows of the window")
    lens = v.sum(axis=1).astype(np.int64)
    _need(np.array_equal(_np(batch["attention_mask"]), v), "served attention_mask differs from the window")
    _need(np.array_equal(_np(batch["lengths"]), lens), "served lengths differ from the window")
    _need(np.array_equal(_np(batch["last_valid_index"]), lens - 1), "served last_valid_index differs")
    rel = np.arange(v.shape[1], dtype=np.int64)
    _need(np.array_equal(_np(batch["position_ids"]), np.where(v, rel[None, :] + 1, 0)),
          "served position_ids are not 1..length on valid positions")
    return {"rows": int(v.shape[0]), "window_rows": int(v.sum())}


def history_record(policy, decision_splits) -> dict:
    """The policy block recorded in every provider fingerprint and receipt."""
    p = require_policy(policy)
    return {"history_policy": p, "context_length": CONTEXT_LENGTH, "resolver": RESOLVER_ID,
            "decision_splits": list(decision_splits), "store_splits": list(store_splits(p, decision_splits)),
            "window_rule": ("rows [max(user_start, r - (L - 1)), r] of the user's retained stream in (user_id, "
                            "segment_index, position_in_segment) order" if p == USER_HISTORY_L50 else
                            "rows [r - min(L, position_in_segment) + 1, r] (v1 loaders.py L263-L270)"),
            "per_event_channels": "stored per-event values (session-relative channels restart at each segment start)",
            "session_offset_tokens": ({"key": SESSION_OFFSET_KEY, "dtype": "int64", "cap": SESSION_OFFSET_CAP,
                                       "vocab": SESSION_OFFSET_VOCAB,
                                       "rule": "min(session_index(d) - session_index(j), 7) + 1; PAD 0; v1 behavioural "
                                               "session (gap > 3600 s over all raw blocks; never fence-reset)"}
                                      if p == USER_HISTORY_L50 else None),
            "session_mode_features": ({"keys": {SESSION_MODE_KEY: "int64 [B, 50]", MATCH_KEY: "float32 [B]",
                                                MATCH_MASK_KEY: "float32 [B]", SHARE_KEY: "float32 [B]"},
                                       "status": "CANDIDATE (DECISION_LOG 48; droppable)",
                                       "main_category_vocab": MAIN_CATEGORY_VOCAB, "unknown_token": MAIN_UNKNOWN,
                                       "scope": "inside the 50-event window; ties -> most recent occurrence"}
                                      if p == USER_HISTORY_L50 else None)}


def input_contract_sha256(feature_schema_self_sha256: str, policy, transform_state_sha256: Optional[str] = None) -> str:
    """The comparison-key `feature_contract_hash` of a run: the frozen feature schema bound to the history policy
    and the context length, so rows of different policies are never same-conditions rows under the locked
    v4eval.rows COMPARISON_KEY (which carries no separate history field). A NON-default transform state (v4int.renorm,
    DECISION_LOG 42) is bound too; the default (locked Clean V2) state is not, so the default key is unchanged."""
    from v4data.common import canonical_json_bytes, sha256_bytes
    obj = {"feature_schema_self_sha256": str(feature_schema_self_sha256), "history_policy": require_policy(policy),
           "context_length": CONTEXT_LENGTH}
    if transform_state_sha256 is not None:
        obj["transform_state_sha256"] = str(transform_state_sha256)
    return sha256_bytes(canonical_json_bytes(obj))


__all__ = ["session_offset_tokens", "check_session_offsets", "SESSION_OFFSET_KEY", "SESSION_OFFSET_CAP",
           "SESSION_OFFSET_VOCAB", "session_mode_features", "add_session_features", "SESSION_MODE_KEY", "MATCH_KEY",
           "MATCH_MASK_KEY", "SHARE_KEY", "USER_ONLY_KEYS", "MAIN_CATEGORY_VOCAB", "MAIN_UNKNOWN",
           "RowIndex", "build_row_index", "align_to_store", "session_lengths", "user_lengths", "policy_lengths",
           "policy_decision_set", "window_rows", "check_windows", "check_served_batch", "history_record",
           "input_contract_sha256", "history_splits", "store_splits", "SPLIT_CODES", "TEST_CODE", "PARTITION_END_US",
           "HistoryOrderError", "HistoryAlignmentError", "HistoryLeakError", "HistoryPolicyError", "INDEX_COLUMNS"]
