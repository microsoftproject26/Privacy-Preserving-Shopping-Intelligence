"""DATA-07 test kit: real-data helpers (frozen v1 bucket 0, read-only), synthetic event tables, leaky builder variants
for the negative controls."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
V4 = CODE.parent
if str(CODE / "tests_data") not in sys.path:
    sys.path.append(str(CODE / "tests_data"))

import v4train  # noqa: E402,F401  (ppsi_env first)

import numpy as np  # noqa: E402

from data_testkit import REAL_DS, assert_raises, nc  # noqa: E402,F401
from v4trunc import events as E  # noqa: E402
from v4trunc.builder import FULL, TruncatedHistoryBuilder, load_normalizers  # noqa: E402

STATE_SHA = "2ff0ff4ec1d27d254972c2f212e81f5fee6277dc9cced31a6a5baf5b4b002727"   # TRANSFORM_STATE_CLEAN_V2 self-hash
K = 158_486


def have_real() -> bool:
    return (Path(REAL_DS) / "model_ready" / "user_bucket=0" / "features.parquet").is_file()


def sample_users(table, n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    val_users = np.unique(table.row_user[(table.split_code > 0) & table.meta["eval_mask"]])
    pick = np.sort(rng.choice(val_users, min(n, val_users.size), replace=False))
    t = table.select_rows(np.isin(table.row_user, pick))
    keep_u = t.user_count > 0
    idx = np.flatnonzero(keep_u)
    return E.EventTable(user_hex=t.user_hex[idx], user_start=t.user_start[idx], user_count=t.user_count[idx],
                        user_bucket=t.user_bucket[idx], carry={k: v[idx] for k, v in t.carry.items()},
                        row_user=np.searchsorted(idx, t.row_user), decision_id=t.decision_id, ts_us=t.ts_us,
                        split_code=t.split_code, segment=t.segment, position=t.position, session=t.session,
                        item=t.item, main_code=t.main_code, brand_code=t.brand_code, log_price=t.log_price,
                        raw=t.raw, meta=t.meta, source=dict(t.source, sample=int(idx.size)))


def make_builder(table, cls=TruncatedHistoryBuilder):
    return cls(table, normalizers=load_normalizers(REAL_DS), state_sha256=STATE_SHA)


def val_query_rows(table) -> np.ndarray:
    return np.flatnonzero((table.split_code > 0) & table.meta["eval_mask"])


# ------------------------------------------------------------------------------------------------ leaky variants
class FutureEventBuilder(TruncatedHistoryBuilder):
    """Leaks ONE extra event: the event after e_q is appended to the listed rows (the window ends at e_{q+1})."""

    def block(self, q_row, h):
        b = super().block(q_row, h)
        t = self.table
        nxt = int(q_row) + 1
        if nxt < t.n_rows and int(t.row_user[nxt]) == int(t.row_user[q_row]):
            for c in b.raw:
                b.raw[c] = np.concatenate([b.raw[c], np.asarray(t.raw[c][nxt:nxt + 1]).astype(b.raw[c].dtype)])
            b.rows = np.append(b.rows, nxt)
            b.session = np.append(b.session, t.session[nxt])
            b.position = np.append(b.position, b.position[-1] + 1)
        return b


class ExtraPriorEventBuilder(TruncatedHistoryBuilder):
    """Leaks ONE extra event: H_{h+1}(q) instead of H_h(q) whenever it exists."""

    def listed_rows(self, q_row, h):
        if h is FULL:
            return super().listed_rows(q_row, h)
        avail = int(q_row) - int(self.table.user_start[int(self.table.row_user[q_row])])
        return super().listed_rows(q_row, min(int(h) + 1, avail))


class NoCarryBuilder(TruncatedHistoryBuilder):
    """Direct per-user window sums (carry-in 0): differs from v1 in the last float32 bit on ~1% of rows."""

    def user_context(self, rows):
        saved = {k: v.copy() for k, v in self.table.carry.items()}
        try:
            for k in self.table.carry:
                self.table.carry[k] = np.zeros_like(self.table.carry[k])
            return super().user_context(rows)
        finally:
            for k in saved:
                self.table.carry[k] = saved[k]


class MergedStreamBuilder(TruncatedHistoryBuilder):
    """Replays VALIDATION_SELECT events into a VALIDATION_CONFIRM query's user context (not the v1 stream rule)."""

    def user_context(self, rows):
        from v4trunc import recompute as R
        from v4trunc.rules import STREAMS
        merged = dict(STREAMS, VALIDATION_CONFIRM=("TRAIN", "VALIDATION_SELECT", "VALIDATION_CONFIRM"))
        orig = R.stream_rows

        def patched(sc, q, streams, names):
            return orig(sc, q, merged, names)
        import v4trunc.builder as B
        B.stream_rows = patched
        try:
            return super().user_context(rows)
        finally:
            B.stream_rows = orig


class FullHistoryUctxBuilder(TruncatedHistoryBuilder):
    """The E11.7 leaking builder: the query's user_context comes from the FULL history whatever h is."""

    def user_context(self, rows):
        t = self.table
        q = int(rows[-1])
        full = np.arange(int(t.user_start[int(t.row_user[q])]), q + 1, dtype=np.int64)
        return super().user_context(full)


# ------------------------------------------------------------------------------------------------ synthetic tables
def synthetic_table(spec, *, carry=None, seed: int = 0):
    """spec: list of users; each user a list of events (split, ts_s, segment, session, item, main_tok, brand_tok, price,
    target_class, eval_mask, loss_mask, target_ts_s or None). Raw stored columns get fixed marker values."""
    rows = []
    for u, evs in enumerate(spec):
        for e in evs:
            rows.append((u,) + tuple(e))
    n = len(rows)
    col = lambda i: [r[i] for r in rows]  # noqa: E731
    user = np.array(col(0))
    split = col(1)
    ts = np.array(col(2), dtype=np.int64)
    raw = {}
    rng = np.random.default_rng(seed)
    for c in E.RAW_COLUMNS:
        if c in ("product_idx",):
            raw[c] = np.array(col(5), dtype=np.int64) + 3
        elif c in ("category_idx", "daypart_idx", "local_weekday_idx"):
            raw[c] = np.full(n, 3, dtype=np.int64 if c == "category_idx" else np.int32)
        elif c in ("brand_idx", "main_category_idx"):
            raw[c] = np.zeros(n, dtype=np.int64)
        elif c in ("price_missing", "price_zero", "price_conflict", "brand_missing", "brand_recovered", "brand_conflict",
                   "category_missing", "category_recovered", "category_conflict", "category_code_missing",
                   "category_code_recovered", "main_category_missing", "prev_gap_unavailable",
                   "item_age_unavailable", "simultaneous_block"):
            raw[c] = np.zeros(n, dtype=bool)
        else:
            raw[c] = rng.random(n).astype(np.float32) + np.float32(0.5)      # "stored" values (e.g. full history)
    raw["price_missing"] = ~(np.array(col(8), dtype=np.float64) > 0)
    tt = np.array([(-1 if r[12] is None else int(r[12]) * 1_000_000) for r in rows], dtype=np.int64)
    meta = {"client_idx": user + 10, "target_class": np.array(col(9), dtype=np.int64),
            "loss_mask": np.array(col(11), dtype=bool), "eval_mask": np.array(col(10), dtype=bool),
            "target_ts_us": tt, "has_obs": tt >= 0, "decision_ts_us": ts * 1_000_000, "in_index": np.ones(n, bool)}
    return E.from_arrays(user=user, decision_id=np.arange(n, dtype=np.int64) * 7 + 100, ts_s=ts, split=split,
                         segment=np.array(col(3)), session=np.array(col(4)), item=np.array(col(5)),
                         main_token=np.array(col(6)), brand_token=np.array(col(7)), price_obs=np.array(col(8), float),
                         raw=raw, meta=meta, carry=carry)
