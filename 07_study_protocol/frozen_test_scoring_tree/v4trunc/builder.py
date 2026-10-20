"""v4trunc.builder — E11.4: the features of query q under budget h, as the clean builder produces them for a user whose
observed history is H_h(q) followed by e_q (EVALUATION_CONTRACT_V4 §11.2, §11.4-§11.7).

Two stages, the second of which is the UNMODIFIED production path:
  1. `block(q_row, h)` — the virtual history. listed rows = the h store-order predecessors of e_q inside its user
     (H_h(q); h = None means the full available history) followed by e_q. For every listed row the stored v1 columns
     are taken per v4trunc.rules: LOCAL_EVENT / SHARED unchanged; the per-event LOCAL_HISTORY columns recomputed over
     the listed rows only (recompute.segment_channels, the PREV_GAP rule on the earliest listed row); the query's
     user-context columns recomputed by the v1 replay over the listed rows of its stream (recompute.user_context_query,
     with the user's v1 carry-in); the user-context columns of every NON-query row are zeroed (the production gather
     reads only the decision row's, so nothing of the full history can reach a tensor).
  2. `batch(items)` — the listed rows are encoded exactly as the snapshot CompactStore encodes stored rows (v1
     normalizers, MISSING_FLAG_OF zeroing; a line-for-line copy of full_store._process_bucket's F4 block, proven
     byte-identical by T41), placed in a CompactStore instance (the snapshot class, data attributes only, as
     v4int.cache does), and served by the unmodified CleanV2Store.batch -> CompactStore.batch -> clean_v2_view, then
     the DATA-05 USER_HISTORY_L50 keys (v4hist.windows.add_session_features over the truncated window) and the optional
     transform_state renormalization (v4int.renorm) — the same calls, in the same order, as StoreEvalProvider._make.
The window length is min(50, len(listed rows)) (USER_HISTORY_L50: the current event is the last position).

Normalization is the frozen one (v1 normalizers + the Clean V2 re-encode of TRANSFORM_STATE); nothing is refit per h
or per client (E11.5). Shared token channels, item age and the output catalogue are the frozen end-of-TRAIN values.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import ppsi_env  # noqa: F401  (snapshot isolation first)

import numpy as np

from rees46_eda.config import (EVENT_QUALITY_FLAGS, NORMALIZED_EVENT_NUMERIC, NORMALIZED_SESSION_PREFIX,  # snapshot
                               NORMALIZED_USER_CONTEXT, USER_CONTEXT_FEATURES, USER_CONTEXT_MASKS)
from rees46_eda.loaders import EVENT_NUMERIC_BLOCK, MISSING_FLAG_OF, USER_MISSING_FLAG_OF, Normalizers  # snapshot

from .events import RAW_COLUMNS, SPLITS, EventTable
from .recompute import first_permitted_prev_gap, segment_channels, stream_rows, user_context_query
from .rules import (COLUMN_RULES, KEEP_EVENT, PARTIAL_OOV_IDENTITY, PARTIAL_ORIGIN, PREV_GAP_RULE, RECOMPUTE_SEGMENT,
                    RECOMPUTE_USER, SHARED_FROZEN, STREAMS)

CONTEXT_LENGTH = 50
SEGMENT_COLUMNS = ("prefix_representative_count", "prefix_item_count", "prefix_known_category_count",
                   "prefix_category_switch_count", "current_known_category_run_length", "log1p_segment_elapsed_s",
                   "log1p_prefix_elapsed_s")
UCTX_COLUMNS = tuple(USER_CONTEXT_FEATURES) + tuple(USER_CONTEXT_MASKS)
PREV_GAP_COLUMNS = ("prev_gap_unavailable", "log1p_prev_raw_view_gap_s")
FULL = None                                         # h = FULL: the full available prior history (T41, C_ref)


class RuleCoverageError(AssertionError):
    """The builder's hard-coded column handling disagrees with v4trunc.rules.COLUMN_RULES: refused at import."""


def check_rule_coverage(column_rules=None, raw_columns=None) -> dict:
    """REV-ANA-01 B-F4: the builder serves a column by its hard-coded sets, so those sets must EQUAL the rule classes
    of v4trunc.rules (checked at import, fail-closed): RECOMPUTE_SEGMENT + PARTIAL_ORIGIN == SEGMENT_COLUMNS,
    PREV_GAP_RULE == PREV_GAP_COLUMNS, RECOMPUTE_USER + PARTIAL_OOV_IDENTITY == UCTX_COLUMNS, KEEP_EVENT +
    SHARED_FROZEN == every other served column (copied unchanged), the rule table covers exactly RAW_COLUMNS, and any
    other rule (e.g. UNSUPPORTED_FUTURE) raises. Returns {rule class: sorted columns}."""
    rules = COLUMN_RULES if column_rules is None else column_rules
    raw = set(RAW_COLUMNS if raw_columns is None else raw_columns)
    if set(rules) != raw:
        raise RuleCoverageError(f"COLUMN_RULES != the served stored columns: missing {sorted(raw - set(rules))}, "
                                f"extra {sorted(set(rules) - raw)}")
    groups = {"segment": {RECOMPUTE_SEGMENT, PARTIAL_ORIGIN}, "prev_gap": {PREV_GAP_RULE},
              "user_context": {RECOMPUTE_USER, PARTIAL_OOV_IDENTITY}, "unchanged": {KEEP_EVENT, SHARED_FROZEN}}
    have = {g: set() for g in groups}
    for c, (_loc, r) in rules.items():
        g = [k for k, v in groups.items() if r in v]
        if not g:
            raise RuleCoverageError(f"stored column {c!r} has rule {r!r}, which the E11.4 builder cannot serve "
                                    "(UNSUPPORTED_FUTURE / unknown rules refuse)")
        have[g[0]].add(c)
    expect = {"segment": set(SEGMENT_COLUMNS), "prev_gap": set(PREV_GAP_COLUMNS), "user_context": set(UCTX_COLUMNS)}
    expect["unchanged"] = raw - expect["segment"] - expect["prev_gap"] - expect["user_context"]
    bad = {g: {"rules": sorted(have[g]), "builder": sorted(expect[g])} for g in groups if have[g] != expect[g]}
    if bad:
        raise RuleCoverageError(f"builder column sets != COLUMN_RULES rule classes: {bad}")
    return {g: sorted(v) for g, v in have.items()}


RULE_COVERAGE = check_rule_coverage()               # import-time: a rule/builder disagreement never serves a batch


class BudgetError(ValueError):
    """h exceeds the query's available prior events (the panel never asks for that)."""


@dataclass
class VirtualBlock:
    query_row: int
    user: int
    h: Optional[int]
    rows: np.ndarray                 # event-table rows, store order, query last
    raw: dict                        # RAW_COLUMNS -> [len(rows)] stored dtypes, after the E11.4 rules
    session: np.ndarray              # stored behavioural session per listed row
    position: np.ndarray             # recomputed position_in_segment per listed row
    audit: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.rows.size)


def load_normalizers(ds_root: Path) -> Normalizers:
    return Normalizers.load(Path(ds_root) / "mappings" / "normalizers.json")


def encode_v1(raw: dict, normalizers: Normalizers) -> dict:
    """Stored v1 columns -> the CompactStore per-row arrays. A copy of full_store._process_bucket's F4 block (same
    dtypes, same Normalizers.apply, same MISSING_FLAG_OF / USER_MISSING_FLAG_OF zeroing); T41 proves it byte-exact."""
    n = int(np.asarray(raw["product_idx"]).shape[0])
    out = {"_item_tokens": np.asarray(raw["product_idx"]).astype(np.int32, copy=False),
           "_category_tokens": np.asarray(raw["category_idx"]).astype(np.int16, copy=False),
           "_brand_tokens": np.asarray(raw["brand_idx"]).astype(np.int16, copy=False),
           "_main_category_tokens": np.asarray(raw["main_category_idx"]).astype(np.uint8, copy=False),
           "_daypart_tokens": np.asarray(raw["daypart_idx"]).astype(np.uint8, copy=False),
           "_weekday_tokens": np.asarray(raw["local_weekday_idx"]).astype(np.uint8, copy=False)}
    numeric = np.zeros((n, len(EVENT_NUMERIC_BLOCK)), dtype=np.float32)
    pmiss = np.asarray(raw["price_missing"]) | np.asarray(raw["price_conflict"])
    for fi, name in enumerate(EVENT_NUMERIC_BLOCK):
        rv = np.asarray(raw[name]).astype(np.float64)
        if name in NORMALIZED_EVENT_NUMERIC or name in NORMALIZED_SESSION_PREFIX:
            v = normalizers.apply(name, rv)
        else:
            v = rv.astype(np.float32)
        flag = MISSING_FLAG_OF.get(name)
        if flag == "price_missing_or_conflict":
            v = np.where(pmiss, 0.0, v).astype(np.float32)
        elif flag is not None:
            v = np.where(np.asarray(raw[flag]), 0.0, v).astype(np.float32)
        numeric[:, fi] = v
    out["_event_numeric_features"] = numeric
    flags = np.zeros((n, len(EVENT_QUALITY_FLAGS)), dtype=np.int8)
    for qi, name in enumerate(EVENT_QUALITY_FLAGS):
        flags[:, qi] = np.asarray(raw[name]).astype(np.int8)
    out["_event_quality_flags"] = flags
    uctx = np.zeros((n, len(USER_CONTEXT_FEATURES)), dtype=np.float32)
    for ui, name in enumerate(USER_CONTEXT_FEATURES):
        rv = np.asarray(raw[name]).astype(np.float64)
        val = normalizers.apply(name, rv) if name in NORMALIZED_USER_CONTEXT else rv.astype(np.float32)
        sup = USER_MISSING_FLAG_OF.get(name)
        if sup is not None:
            val = np.where(np.asarray(raw[sup]) == 0.0, 0.0, val).astype(np.float32)
        uctx[:, ui] = val
    out["_user_context"] = uctx
    masks = np.zeros((n, len(USER_CONTEXT_MASKS)), dtype=np.float32)
    for mi, name in enumerate(USER_CONTEXT_MASKS):
        masks[:, mi] = np.asarray(raw[name]).astype(np.float32)
    out["_user_context_masks"] = masks
    return out


class TruncatedHistoryBuilder:
    """E11.4 over one EventTable. `state_sha256` = the Clean V2 TRANSFORM_STATE self-hash the production provider
    passes to CleanV2Store (recorded in every batch); `transform_state` = an optional renormalization file (v4int.renorm,
    DECISION_LOG 42; None = the locked Clean V2 state, as the production default)."""

    def __init__(self, table: EventTable, *, normalizers: Normalizers, state_sha256: Optional[str],
                 transform_state: Optional[Path] = None, spec=None):
        from v4data.transform import build_spec
        from v4int.renorm import resolve_transform_state
        self.table = table
        self.normalizers = normalizers
        self.state_sha256 = state_sha256
        self.spec = spec if spec is not None else build_spec()
        self.renorm, self.transform_state = resolve_transform_state(transform_state, self.spec, state_sha256)

    # ---------------------------------------------------------------- stage 1: the virtual history
    def listed_rows(self, q_row: int, h: Optional[int]) -> np.ndarray:
        t = self.table
        u = int(t.row_user[q_row])
        start = int(t.user_start[u])
        avail = int(q_row) - start                       # prior events available (store order = chronological)
        if h is FULL:
            first = start
        else:
            h = int(h)
            if h < 0 or h > avail:
                raise BudgetError(f"h={h} but the query has {avail} prior events")
            first = int(q_row) - h
        return np.arange(first, int(q_row) + 1, dtype=np.int64)

    def user_context(self, rows: np.ndarray) -> dict:
        """The query's (last listed row's) v1 user context over the listed rows of its replay stream."""
        t = self.table
        q_row = int(rows[-1])
        sm = stream_rows(t.split_code[rows], int(t.split_code[q_row]), STREAMS, SPLITS)
        sr = rows[sm]
        qi = int(np.flatnonzero(sr == q_row)[0])
        stream = SPLITS[int(t.split_code[q_row])]
        u = int(t.row_user[q_row])
        return user_context_query(ts_s=t.ts_us[sr] // 1_000_000, decision_id=t.decision_id[sr], item=t.item[sr],
                                  main_code=t.main_code[sr], brand_code=t.brand_code[sr], log_price=t.log_price[sr],
                                  session=t.session[sr], q=qi, carry_in=float(t.carry[stream][u]))

    def block(self, q_row: int, h: Optional[int]) -> VirtualBlock:
        t = self.table
        rows = self.listed_rows(int(q_row), h)
        raw = {c: np.array(t.raw[c][rows], copy=True) for c in RAW_COLUMNS}
        seg = segment_channels(t.segment[rows], t.ts_us[rows], t.main_code[rows])
        for c in SEGMENT_COLUMNS:
            raw[c] = seg[c].astype(raw[c].dtype, copy=False)
        raw["prev_gap_unavailable"], raw["log1p_prev_raw_view_gap_s"] = first_permitted_prev_gap(
            raw["prev_gap_unavailable"], raw["log1p_prev_raw_view_gap_s"])
        uc = self.user_context(rows)
        for c in UCTX_COLUMNS:
            col = np.zeros(rows.size, dtype=raw[c].dtype)
            col[-1] = uc["columns"][c]
            raw[c] = col
        seg_first = rows[0] == int(t.user_start[int(t.row_user[q_row])])
        first_in_seg = np.ones(rows.size, dtype=bool)
        first_in_seg[1:] = t.segment[rows][1:] != t.segment[rows][:-1]
        stored_pos = t.position[rows]
        origin_truncated = bool(np.any(first_in_seg & (stored_pos > 1)))
        audit = dict(uc["audit"], n_listed=int(rows.size), full_history_from_user_start=bool(seg_first),
                     segment_origin_truncated=origin_truncated,
                     segment_main_oov_rows=int(np.count_nonzero(t.main_code[rows] == 2)))
        return VirtualBlock(query_row=int(q_row), user=int(t.row_user[q_row]), h=h, rows=rows, raw=raw,
                            session=t.session[rows].astype(np.int64), position=seg["position_in_segment"], audit=audit)

    # ---------------------------------------------------------------- stage 2: the production path
    def store(self, blocks: Sequence[VirtualBlock]):
        """(CleanV2Store over the blocks' encoded rows, DecisionSet with one decision per block in block order,
        per-row session ids)."""
        from full_store import CompactStore, DecisionSet          # snapshot copies
        from v4data.store import CleanV2Store
        t = self.table
        enc = [encode_v1(b.raw, self.normalizers) for b in blocks]
        cs = CompactStore.__new__(CompactStore)
        cs.dataset_root, cs.layout, cs.splits, cs.channels = None, "frozen", SPLITS, "F4"
        cs.context_length, cs.buckets, cs.version_hash = CONTEXT_LENGTH, [], None
        for a in enc[0] if enc else ():
            setattr(cs, a, np.concatenate([e[a] for e in enc]))
        n_rows = int(sum(b.n for b in blocks))
        cs.n_rows = n_rows
        cs._masks_dtype = "float32"
        cs._split_decisions, cs._decisions_cache = {}, {}
        cs.build_seconds, cs.bucket_log = 0.0, []
        sizes = np.array([b.n for b in blocks], dtype=np.int64)
        end = np.cumsum(sizes) - 1
        q = np.array([b.query_row for b in blocks], dtype=np.int64)
        pvc = np.array([float(b.raw["log1p_prev_view_count"][-1]) for b in blocks], dtype=np.float64)
        ds = DecisionSet(split="TRUNCATED", kind="eval", decision_id=t.decision_id[q].astype(np.int64),
                         end_row=end.astype(np.int64), length=np.minimum(CONTEXT_LENGTH, sizes).astype(np.int16),
                         target_class=t.meta["target_class"][q].astype(np.int64),
                         loss_mask=t.meta["loss_mask"][q].astype(bool), eval_mask=t.meta["eval_mask"][q].astype(bool),
                         client_idx=t.meta["client_idx"][q].astype(np.int64),
                         user_bucket=t.user_bucket[t.row_user[q]].astype(np.int16), eval_user_key=None,
                         prior_view_count=np.rint(np.expm1(pvc)).astype(np.int32),
                         prefix_position=np.array([int(b.position[-1]) for b in blocks], dtype=np.int32))
        st = CleanV2Store(cs, self.spec, self.state_sha256)
        sid = np.concatenate([b.session for b in blocks]).astype(np.int32) if blocks else np.zeros(0, np.int32)
        return st, ds, sid

    def batch_of_blocks(self, blocks: Sequence[VirtualBlock]) -> dict:
        """The production batch of the blocks' queries (StoreEvalProvider._make, call for call)."""
        from v4hist.windows import add_session_features
        st, ds, sid = self.store(blocks)
        pos = np.arange(len(blocks), dtype=np.int64)
        b = add_session_features(st.batch(ds, pos), sid, ds.end_row[pos], ds.length[pos])
        if self.renorm is not None:
            b = self.renorm.apply(b)
        return b

    def batch(self, items: Iterable[tuple]) -> dict:
        """items: (query event-table row, h) pairs (h = None: full history)."""
        return self.batch_of_blocks([self.block(int(r), h) for r, h in items])


__all__ = ["TruncatedHistoryBuilder", "VirtualBlock", "encode_v1", "load_normalizers", "BudgetError", "FULL",
           "CONTEXT_LENGTH", "SEGMENT_COLUMNS", "UCTX_COLUMNS", "PREV_GAP_COLUMNS", "RuleCoverageError",
           "check_rule_coverage", "RULE_COVERAGE"]
