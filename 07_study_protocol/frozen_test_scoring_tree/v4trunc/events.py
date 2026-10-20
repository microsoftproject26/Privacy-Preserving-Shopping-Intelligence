"""v4trunc.events — per-user event tables for the truncated-history builder (read-only, never TEST).

An observed event of user u (EVALUATION_CONTRACT_V4 §11.2) is one stored v1 feature row of u in TRAIN,
VALIDATION_SELECT or VALIDATION_CONFIRM (= one ledger decision row: a representative after consecutive-view
compression). Rows are kept in the snapshot CompactStore order, np.lexsort((position_in_segment, segment_index,
user_id)) (full_store.py L237), which inside a user is chronological (v4hist.windows proves it on every non-TEST row;
re-checked here per table). The files are read through v4data.manifests.read_non_test: TEST rows are never
materialised.

Besides the stored v1 columns the production encoder reads (`RAW_COLUMNS`), a table carries what the recompute needs:
block_ts (UTC microseconds; the v1 replay orders by epoch seconds, so second granularity is asserted), session_index,
an exact per-bucket product code (dense rank of the product_id string, as the v1 replay's dense_rank), the main
category / brand identity codes (the TRAIN-fitted tokens: known >= 2, MISSING 1 -> unknown), the float64 log price
ln(1 + price_obs) (math.log = the C library log, as DuckDB's ln), the decisions_index metadata (client_idx, targets,
masks, target_ts, has_observed_target) and the v1 replay CARRY-IN of every user and stream.

Carry-in (why and what): the v1 replay (EDA userctx.window_sum) sums a user's recent log prices as a difference of ONE
float64 cumulative sum over the whole bucket stream (all users, ordered by user_id, ts, decision_id). The prefix that
precedes a user therefore enters every one of its window sums as rounding noise (a per-user direct sum differs in the
last float32 bit on ~1% of current_minus_recent_mean_log_price rows; MEASURED bucket 0). The builder reproduces the v1
arithmetic exactly by starting the user's cumulative sum at C0 = the v1 bucket-stream prefix sum over the rows of all
users that precede u in that stream. C0 is a function of OTHER users' rows only: deleting any of u's own events never
changes it (E11.7), and it carries no information about u's history.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import ppsi_env  # noqa: F401  (snapshot isolation first)

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from rees46_eda.config import EVENT_QUALITY_FLAGS, USER_CONTEXT_FEATURES, USER_CONTEXT_MASKS  # snapshot copy
from rees46_eda.loaders import EVENT_NUMERIC_BLOCK                                             # snapshot copy
from v4data.common import digests_of, hex_of, refuse_test_split
from v4data.manifests import _ts_us, read_non_test

from .rules import COLUMN_RULES, STREAMS

TOKEN_COLUMNS = ("product_idx", "category_idx", "brand_idx", "main_category_idx", "daypart_idx", "local_weekday_idx")
RAW_COLUMNS = TOKEN_COLUMNS + tuple(EVENT_NUMERIC_BLOCK) + tuple(EVENT_QUALITY_FLAGS) + tuple(USER_CONTEXT_FEATURES) \
    + tuple(USER_CONTEXT_MASKS)
KEY_COLUMNS = ("user_id", "decision_id", "block_ts", "split", "segment_index", "position_in_segment", "session_index",
               "product_id", "price_obs")
FEATURE_COLUMNS = tuple(dict.fromkeys(KEY_COLUMNS + RAW_COLUMNS))
META_COLUMNS = ("decision_id", "client_idx", "target_class", "loss_mask", "eval_mask", "target_ts",
                "has_observed_target", "decision_ts")
SPLITS = ("TRAIN", "VALIDATION_SELECT", "VALIDATION_CONFIRM")
SPLIT_CODE = {s: i for i, s in enumerate(SPLITS)}
_missing_rules = [c for c in RAW_COLUMNS if c not in COLUMN_RULES]
assert not _missing_rules, f"stored columns without an E11.4 rule: {_missing_rules}"


class EventTableError(AssertionError):
    """The event table violates a builder precondition (order, granularity, coverage): refused."""


@dataclass
class EventTable:
    """Rows of a set of users, users contiguous, each user's rows in store (chronological) order."""
    user_hex: np.ndarray            # object [U] client_key_hex (salted digest; never the raw id)
    user_start: np.ndarray          # int64 [U]
    user_count: np.ndarray          # int64 [U]
    user_bucket: np.ndarray         # int64 [U]
    carry: dict                     # stream -> float64 [U] (v1 bucket-stream prefix sum before the user)
    row_user: np.ndarray            # int64 [n]
    decision_id: np.ndarray         # int64 [n]
    ts_us: np.ndarray               # int64 [n]
    split_code: np.ndarray          # int8  [n] 0 TRAIN, 1 VS, 2 VC
    segment: np.ndarray             # int64 [n] stored segment_index
    position: np.ndarray            # int64 [n] stored position_in_segment
    session: np.ndarray             # int64 [n] stored session_index (behavioural session)
    item: np.ndarray                # int64 [n] exact product identity code
    main_code: np.ndarray           # int64 [n] main category identity (-1 unknown)
    brand_code: np.ndarray          # int64 [n] brand identity (-1 unknown)
    log_price: np.ndarray           # float64 [n] ln(1 + price_obs), NaN = unavailable
    raw: dict                       # stored v1 column -> [n] (RAW_COLUMNS), stored dtypes
    meta: dict                      # client_idx, target_class, loss_mask, eval_mask, target_ts_us, has_obs, in_index
    source: dict = field(default_factory=dict)
    _row_of_did: Optional[dict] = field(default=None, repr=False)
    _user_of_hex: Optional[dict] = field(default=None, repr=False)

    @property
    def n_rows(self) -> int:
        return int(self.decision_id.shape[0])

    @property
    def n_users(self) -> int:
        return int(self.user_hex.shape[0])

    def row_of(self, decision_id: int) -> int:
        if self._row_of_did is None:
            self._row_of_did = {int(d): i for i, d in enumerate(self.decision_id.tolist())}
        try:
            return self._row_of_did[int(decision_id)]
        except KeyError:
            raise EventTableError(f"decision {decision_id} is not a row of this event table") from None

    def user_of_hex(self, key: str) -> int:
        if self._user_of_hex is None:
            self._user_of_hex = {str(h): i for i, h in enumerate(self.user_hex.tolist())}
        try:
            return self._user_of_hex[str(key)]
        except KeyError:
            raise EventTableError("a requested client key is not a user of this event table") from None

    def rows_of_user(self, u: int) -> np.ndarray:
        s = int(self.user_start[u])
        return np.arange(s, s + int(self.user_count[u]), dtype=np.int64)

    def drop_rows(self, rows: np.ndarray) -> "EventTable":
        """A copy with `rows` physically deleted (the E11.7 deletion control). Carry-ins are unchanged: they depend on
        other users' rows only."""
        keep = np.ones(self.n_rows, dtype=bool)
        keep[np.asarray(rows, dtype=np.int64)] = False
        return self.select_rows(keep)

    def select_rows(self, keep: np.ndarray) -> "EventTable":
        keep = np.asarray(keep, dtype=bool)
        ru = self.row_user[keep]
        cnt = np.bincount(ru, minlength=self.n_users).astype(np.int64)
        start = np.concatenate([[0], np.cumsum(cnt)[:-1]]).astype(np.int64)
        return EventTable(user_hex=self.user_hex.copy(), user_start=start, user_count=cnt,
                          user_bucket=self.user_bucket.copy(), carry={k: v.copy() for k, v in self.carry.items()},
                          row_user=ru, decision_id=self.decision_id[keep], ts_us=self.ts_us[keep],
                          split_code=self.split_code[keep], segment=self.segment[keep], position=self.position[keep],
                          session=self.session[keep], item=self.item[keep], main_code=self.main_code[keep],
                          brand_code=self.brand_code[keep], log_price=self.log_price[keep],
                          raw={k: v[keep] for k, v in self.raw.items()}, meta={k: v[keep] for k, v in self.meta.items()},
                          source=dict(self.source, rows_deleted=int((~keep).sum())))

    def check(self) -> dict:
        """Builder preconditions (fail-closed): users contiguous; inside a user, time and partition never decrease,
        positions are 1..k inside every stored segment and a segment never spans two partitions; second-granular
        timestamps; every row is a decisions_index row."""
        n = self.n_rows
        if n == 0:
            return {"rows": 0}
        ru = self.row_user
        if not np.array_equal(np.repeat(np.arange(self.n_users), self.user_count), ru):
            raise EventTableError("users are not contiguous / counts disagree")
        same = ru[1:] == ru[:-1]
        if np.any(self.ts_us[1:][same] < self.ts_us[:-1][same]):
            raise EventTableError("time decreases inside a user's stream")
        if np.any(self.split_code[1:][same] < self.split_code[:-1][same]):
            raise EventTableError("the partition decreases inside a user's stream")
        newseg = np.ones(n, dtype=bool)
        newseg[1:] = ~same | (self.segment[1:] != self.segment[:-1])
        st = np.flatnonzero(newseg)
        local = np.arange(n) - st[np.cumsum(newseg) - 1] + 1
        if not np.array_equal(local, self.position):
            raise EventTableError("stored positions are not 1..k inside every segment")
        inseg = ~newseg[1:]
        if np.any(self.split_code[1:][inseg] != self.split_code[:-1][inseg]):
            raise EventTableError("a segment spans two partitions")
        if np.any(self.ts_us % 1_000_000 != 0):
            raise EventTableError("a block_ts has a sub-second part (the v1 replay orders by epoch seconds)")
        if not bool(np.all(self.meta["in_index"])):
            raise EventTableError("a feature row has no decisions_index row")
        return {"rows": n, "users": self.n_users, "segments": int(newseg.sum())}


# ------------------------------------------------------------------------------------------------ helpers
def log_price_of(price_obs: np.ndarray) -> np.ndarray:
    """float64 ln(1 + price_obs) where price_obs is finite and > 0, else NaN (EDA pipeline6: log1p_price is NULL
    otherwise, and the replay reads coalesce(log1p_price, NaN)). math.log = the C library log (DuckDB's ln)."""
    p = np.asarray(price_obs, dtype=np.float64)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p) & (p > 0)
    out[ok] = np.fromiter((math.log(1.0 + x) for x in p[ok].tolist()), dtype=np.float64, count=int(ok.sum()))
    return out


def identity_code(token: np.ndarray) -> np.ndarray:
    """TRAIN-fitted token -> replay identity code: MISSING (1) / PAD (0) -> -1 (unknown); OOV (2) and known (>= 3)
    keep their token (OOV collapses distinct unseen values: rules.PARTIAL_OOV_IDENTITY)."""
    t = np.asarray(token, dtype=np.int64)
    return np.where(t >= 2, t, -1).astype(np.int64)


def stream_carry_in(user_id: np.ndarray, ts_s: np.ndarray, decision_id: np.ndarray, split_code: np.ndarray,
                    log_price: np.ndarray) -> dict:
    """{stream: {user_id: C0}}: the v1 bucket-stream prefix sum before each user (EDA pipeline6._fetch_stream order
    user_id, ts, decision_id; userctx.window_sum's global cumsum of the valid log prices)."""
    out = {}
    pv_all = np.where(np.isfinite(log_price), log_price, 0.0)
    for stream, splits in STREAMS.items():
        codes = [SPLIT_CODE[s] for s in splits]
        idx = np.flatnonzero(np.isin(split_code, codes))
        if idx.size == 0:
            out[stream] = {}
            continue
        o = np.lexsort((decision_id[idx], ts_s[idx], user_id[idx]))
        r = idx[o]
        c = np.concatenate(([0.0], np.cumsum(pv_all[r], dtype=np.float64)))
        u = user_id[r]
        first = np.flatnonzero(np.concatenate(([True], u[1:] != u[:-1])))
        out[stream] = dict(zip(u[first].tolist(), c[first].tolist()))
    return out


def _col(t: pa.Table, name: str) -> np.ndarray:
    return t[name].to_numpy(zero_copy_only=False)


# ------------------------------------------------------------------------------------------------ real data
def feature_path(ds_root: Path, b: int) -> Path:
    return Path(ds_root) / "model_ready" / f"user_bucket={int(b)}" / "features.parquet"


def load_bucket(ds_root: Path, b: int, *, want_hex: Optional[Iterable[str]] = None) -> EventTable:
    """The event table of bucket `b` (all its users, or those whose client_key_hex is in `want_hex`)."""
    from index_utils import read_index_projected       # snapshot copy
    ds_root = Path(ds_root)
    t = read_non_test(feature_path(ds_root, b), list(FEATURE_COLUMNS))
    n_all = t.num_rows
    uid = _col(t, "user_id").astype(object)
    ts_us = _ts_us(t["block_ts"])
    if np.any(ts_us % 1_000_000 != 0):
        raise EventTableError(f"bucket {b}: sub-second block_ts (the v1 replay orders by epoch seconds)")
    sc_arr = pc.index_in(t["split"], value_set=pa.array(list(SPLITS), type=pa.string()))
    if sc_arr.null_count:
        raise EventTableError(f"bucket {b}: a row outside {SPLITS}")
    sc = np.asarray(sc_arr.to_numpy(zero_copy_only=False), dtype=np.int8)
    did = np.asarray(_col(t, "decision_id"), dtype=np.int64)
    lp = log_price_of(_col(t, "price_obs"))
    carry = stream_carry_in(uid, ts_us // 1_000_000, did, sc, lp)
    uniq, inv = np.unique(uid, return_inverse=True)
    hexes = hex_of(digests_of(uniq.tolist()))
    if want_hex is not None:
        want = set(str(h) for h in want_hex)
        keep_u = np.fromiter((h in want for h in hexes.tolist()), dtype=bool, count=uniq.size)
    else:
        keep_u = np.ones(uniq.size, dtype=bool)
    keep = keep_u[inv]
    seg = np.asarray(_col(t, "segment_index"), dtype=np.float64)
    pos = np.asarray(_col(t, "position_in_segment"), dtype=np.int64)
    ses = np.asarray(_col(t, "session_index"), dtype=np.float64)
    if np.any(~np.isfinite(seg)) or np.any(~np.isfinite(ses)):
        raise EventTableError(f"bucket {b}: NaN segment_index / session_index")
    _, item_all = np.unique(_col(t, "product_id").astype(object), return_inverse=True)
    rows = np.flatnonzero(keep)
    o = np.lexsort((pos[rows], seg[rows], uid[rows]))          # the store key (full_store.py L237)
    r = rows[o]
    ku = inv[r]
    newu = np.concatenate(([True], ku[1:] != ku[:-1])) if r.size else np.zeros(0, dtype=bool)
    starts = np.flatnonzero(newu)
    counts = np.diff(np.append(starts, r.size)).astype(np.int64)
    users = ku[starts]
    raw = {}
    for c in RAW_COLUMNS:
        raw[c] = np.asarray(_col(t, c))[r]
    # decisions_index metadata (the store's own join source; full_store.META_COLUMNS + support fields)
    parts = []
    for s in SPLITS:
        try:
            parts.append(read_index_projected(ds_root, "frozen", s, list(META_COLUMNS), buckets=[int(b)]))
        except FileNotFoundError:
            continue
    mt = pa.concat_tables(parts) if parts else None
    meta = _join_meta(did[r], mt)
    del t
    tab = EventTable(user_hex=hexes[users].astype(object), user_start=starts.astype(np.int64), user_count=counts,
                     user_bucket=np.full(users.size, int(b), dtype=np.int64),
                     carry={k: np.array([v.get(u, 0.0) for u in uniq[users].tolist()], dtype=np.float64)
                            for k, v in carry.items()},
                     row_user=np.repeat(np.arange(users.size, dtype=np.int64), counts), decision_id=did[r],
                     ts_us=ts_us[r], split_code=sc[r], segment=seg[r].astype(np.int64), position=pos[r],
                     session=ses[r].astype(np.int64), item=item_all[r].astype(np.int64),
                     main_code=identity_code(raw["main_category_idx"]), brand_code=identity_code(raw["brand_idx"]),
                     log_price=lp[r], raw=raw, meta=meta,
                     source={"bucket": int(b), "rows_in_bucket_non_test": int(n_all), "rows_kept": int(r.size)})
    # users absent from a stream have carry 0 by construction (they add nothing to it); mark presence for audits
    tab.check()
    return tab


def _join_meta(did: np.ndarray, mt: Optional[pa.Table]) -> dict:
    n = did.size
    out = {"client_idx": np.zeros(n, np.int64), "target_class": np.full(n, -1, np.int64),
           "loss_mask": np.zeros(n, bool), "eval_mask": np.zeros(n, bool), "target_ts_us": np.full(n, -1, np.int64),
           "has_obs": np.zeros(n, bool), "decision_ts_us": np.full(n, -1, np.int64), "in_index": np.zeros(n, bool)}
    if mt is None or mt.num_rows == 0 or n == 0:
        return out
    md = np.asarray(_col(mt, "decision_id"), dtype=np.int64)
    o = np.argsort(md, kind="stable")
    ms = md[o]
    p = np.clip(np.searchsorted(ms, did), 0, ms.size - 1)
    hit = ms[p] == did
    src = o[p[hit]]
    out["in_index"] = hit
    out["client_idx"][hit] = np.asarray(_col(mt, "client_idx"), dtype=np.int64)[src]
    out["target_class"][hit] = np.asarray(_col(mt, "target_class"), dtype=np.int64)[src]
    out["loss_mask"][hit] = np.asarray(_col(mt, "loss_mask"), dtype=bool)[src]
    out["eval_mask"][hit] = np.asarray(_col(mt, "eval_mask"), dtype=bool)[src]
    out["has_obs"][hit] = np.asarray(_col(mt, "has_observed_target"), dtype=bool)[src]
    out["target_ts_us"][hit] = _ts_us(mt["target_ts"])[src]
    out["decision_ts_us"][hit] = _ts_us(mt["decision_ts"])[src]
    return out


def load_users(ds_root: Path, want_hex: Iterable[str], *, buckets: Optional[Sequence[int]] = None,
               log=print) -> EventTable:
    """The event table of the users `want_hex` (client_key_hex) over `buckets` (default all 64). Every wanted user
    must be found exactly once (a user lives in one bucket)."""
    from v4data.common import N_BUCKETS
    want = [str(h) for h in want_hex]
    bl = list(range(N_BUCKETS)) if buckets is None else [int(b) for b in buckets]
    parts = []
    for b in bl:
        tb = load_bucket(ds_root, b, want_hex=want)
        log(f"[v4trunc.events] bucket {b}: {tb.n_users} users / {tb.n_rows} rows")
        if tb.n_users:
            parts.append(tb)
    tab = concat(parts)
    missing = sorted(set(want) - set(tab.user_hex.tolist()))
    if missing:
        raise EventTableError(f"{len(missing)} requested users were not found in buckets {bl[0]}..{bl[-1]}")
    return tab


def concat(parts: Sequence[EventTable]) -> EventTable:
    if not parts:
        z = np.zeros(0, np.int64)
        return EventTable(user_hex=np.zeros(0, object), user_start=z, user_count=z, user_bucket=z,
                          carry={s: np.zeros(0) for s in STREAMS}, row_user=z, decision_id=z, ts_us=z,
                          split_code=np.zeros(0, np.int8), segment=z, position=z, session=z, item=z, main_code=z,
                          brand_code=z, log_price=np.zeros(0), raw={c: np.zeros(0) for c in RAW_COLUMNS},
                          meta={k: np.zeros(0) for k in ("client_idx", "target_class", "loss_mask", "eval_mask",
                                                         "target_ts_us", "has_obs", "decision_ts_us", "in_index")})
    if len(parts) == 1:
        return parts[0]
    off_r, off_u = 0, 0
    item_off = 0
    ru, st, items = [], [], []
    for p in parts:
        ru.append(p.row_user + off_u)
        st.append(p.user_start + off_r)
        items.append(p.item + item_off)                      # item codes stay exact per bucket (a user is in one)
        item_off += int(p.item.max()) + 1 if p.n_rows else 0
        off_r += p.n_rows
        off_u += p.n_users
    cat = lambda f: np.concatenate([getattr(p, f) for p in parts])  # noqa: E731
    return EventTable(user_hex=cat("user_hex"), user_start=np.concatenate(st), user_count=cat("user_count"),
                      user_bucket=cat("user_bucket"),
                      carry={s: np.concatenate([p.carry[s] for p in parts]) for s in parts[0].carry},
                      row_user=np.concatenate(ru), decision_id=cat("decision_id"), ts_us=cat("ts_us"),
                      split_code=cat("split_code"), segment=cat("segment"), position=cat("position"),
                      session=cat("session"), item=np.concatenate(items), main_code=cat("main_code"),
                      brand_code=cat("brand_code"), log_price=cat("log_price"),
                      raw={c: np.concatenate([p.raw[c] for p in parts]) for c in parts[0].raw},
                      meta={k: np.concatenate([p.meta[k] for p in parts]) for k in parts[0].meta},
                      source={"parts": [p.source for p in parts]})


# ------------------------------------------------------------------------------------------------ synthetic
def from_arrays(*, user: np.ndarray, decision_id: np.ndarray, ts_s: np.ndarray, split: Sequence[str],
                segment: np.ndarray, session: np.ndarray, item: np.ndarray, main_token: np.ndarray,
                brand_token: np.ndarray, price_obs: np.ndarray, raw: Mapping[str, np.ndarray],
                meta: Mapping[str, np.ndarray], carry: Optional[Mapping[str, np.ndarray]] = None,
                user_hex: Optional[Sequence[str]] = None) -> EventTable:
    """A synthetic event table (tests): rows already in (user, chronological) order; positions are derived from the
    segments; `raw` must hold every RAW_COLUMNS column; carry defaults to 0."""
    u = np.asarray(user, dtype=np.int64)
    n = u.size
    newu = np.ones(n, dtype=bool)
    newu[1:] = u[1:] != u[:-1]
    starts = np.flatnonzero(newu)
    counts = np.diff(np.append(starts, n)).astype(np.int64)
    seg = np.asarray(segment, dtype=np.int64)
    newseg = newu.copy()
    newseg[1:] |= seg[1:] != seg[:-1]
    sst = np.flatnonzero(newseg)
    pos = (np.arange(n) - sst[np.cumsum(newseg) - 1] + 1).astype(np.int64)
    U = starts.size
    lp = log_price_of(price_obs)
    tab = EventTable(user_hex=np.array(list(user_hex) if user_hex is not None else [f"{i:064x}" for i in range(U)],
                                       dtype=object),
                     user_start=starts.astype(np.int64), user_count=counts, user_bucket=np.zeros(U, np.int64),
                     carry={s: (np.asarray(carry[s], dtype=np.float64) if carry is not None else np.zeros(U))
                            for s in STREAMS},
                     row_user=np.repeat(np.arange(U, dtype=np.int64), counts),
                     decision_id=np.asarray(decision_id, dtype=np.int64),
                     ts_us=np.asarray(ts_s, dtype=np.int64) * 1_000_000,
                     split_code=np.array([SPLIT_CODE[s] for s in split], dtype=np.int8), segment=seg, position=pos,
                     session=np.asarray(session, dtype=np.int64), item=np.asarray(item, dtype=np.int64),
                     main_code=identity_code(main_token), brand_code=identity_code(brand_token), log_price=lp,
                     raw={c: np.asarray(raw[c]).copy() for c in RAW_COLUMNS},
                     meta={k: np.asarray(v).copy() for k, v in meta.items()}, source={"synthetic": True})
    tab.raw["main_category_idx"] = np.asarray(main_token).astype(tab.raw["main_category_idx"].dtype)
    tab.raw["brand_idx"] = np.asarray(brand_token).astype(tab.raw["brand_idx"].dtype)
    tab.raw["prefix_representative_count"] = (pos - 1).astype(np.float32)
    return tab


def refuse_splits(splits: Iterable[str]) -> None:
    for s in splits:
        refuse_test_split(s)
