"""The data adapter: leave-one-out train.csv / holdout.csv -> per-user chronological sequences -> decision rows
(one row = one next-item decision with its right-padded window of the last <= 50 events).

Reading and ordering follow the reference benchmark code: pandas read_csv with parse_dates=[datetime], a
`_source_order` column = the file row order, sort by (user_id, datetime, _source_order) with a stable sort; the
catalogue = the sorted unique external item ids of train.csv (internal class j = the j-th catalogue item; model item
token = j + 3, the PAD / MISSING / OOV offset); holdout rows whose user has no train row are dropped and counted.

Splits (per user u with the FULL chronological train sequence f_0 .. f_{L-1}, ascending stable by (datetime, file row)):
  inner VALIDATION  built exactly as the reference trainer builds it: rank rows by datetime DESCENDING with a STABLE
                    sort; the rank-0 row of each user is the validation target. Ties at the maximum timestamp therefore
                    select the FIRST tied row in file order, which is NOT always the last element of the ascending
                    sequence. The inner sequence s = the full sequence with that row removed (length L-1).
                    validation: target = that row's item, context = the last <= 50 of s, seen = items of s.
  TRAIN decisions   from the inner sequence s_0 .. s_{L-2}: targets s_t for t = 1 .. L-2, context s_{max(0, t-50)} ..
                    s_{t-1}   (canonical order: user index, then t; one federated client = one user = exactly its rows)
  TEST (holdout)    only users with a holdout row; target = the holdout item; context = the last <= 50 of the FULL
                    sequence f (the validation item stays in its sorted position); seen = ALL items of f; candidates =
                    the train.csv catalogue; no retraining. Read only by `load_holdout_view` (holdout.py), after model
                    selection on the inner validation split.
The pretraining band: a seeded permutation (SeedSequence([BAND_TAG, 2026]), PCG64) of the sorted external user ids;
the first round(0.15 U) users. PRE trains on the band users' TRAIN decisions only (their validation / test items are
never training targets).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .common import (
    BAND_FRAC,
    BAND_SEED,
    BAND_TAG,
    MAX_LEN,
    BenchRefused,
    canon_sha256,
    check_dataset,
    sha256_file,
)

ITEM_OFFSET = 3                       # item tokens: 0 PAD, 1 MISSING, 2 OOV, classes from 3
COLUMNS = ("user_id", "item_id", "datetime")


def split_dir(root, dataset: str) -> Path:
    return Path(root) / check_dataset(dataset) / "leave_one_out"


def _read(path: Path):
    import pandas as pd
    frame = pd.read_csv(path, dtype={"user_id": "int64", "item_id": "int64"}, parse_dates=["datetime"])
    missing = set(COLUMNS) - set(frame.columns)
    if missing:
        raise BenchRefused(f"{path}: missing columns {sorted(missing)}")
    frame["_source_order"] = np.arange(len(frame), dtype=np.int64)
    return frame


def band_users(sorted_user_ids: np.ndarray, *, seed: int = BAND_SEED, frac: float = BAND_FRAC) -> np.ndarray:
    """The pretraining band (sorted external user ids)."""
    u = np.asarray(sorted_user_ids, dtype=np.int64)
    if u.size and not np.all(u[1:] > u[:-1]):
        raise BenchRefused("band_users needs strictly ascending user ids")
    perm = np.random.Generator(np.random.PCG64(np.random.SeedSequence([BAND_TAG, int(seed)]))).permutation(u.size)
    n = int(np.floor(frac * u.size + 0.5))
    return np.sort(u[perm[:n]])


@dataclass
class BenchData:
    dataset: str
    train_path: Path
    train_sha256: str
    catalog: np.ndarray               # [K] sorted external item ids
    user_ids: np.ndarray              # [U] sorted external user ids
    seq_offsets: np.ndarray           # [U+1] int64 INNER sequences (validation row removed): lengths L-1
    seq_items: np.ndarray             # [sum (L-1)] int64 internal class ids, chronological per user
    full_offsets: np.ndarray          # [U+1] int64 FULL train sequences (TEST context / seen)
    full_items: np.ndarray            # [sum L] int64
    val_pos: np.ndarray               # [U] int64 position of the validation row inside the full sequence
    val_item: np.ndarray              # [U] int64 internal class of the validation row
    band: np.ndarray                  # [U] bool
    dec_user: np.ndarray              # [N] int64 user index of each TRAIN decision (canonical order)
    dec_pos: np.ndarray               # [N] int64 target position t in the user's sequence
    user_dec_offsets: np.ndarray      # [U+1] int64 into the decision arrays
    max_len: int = MAX_LEN
    stats: dict = field(default_factory=dict)

    # ---------------------------------------------------------------------------------------------- basic facts
    @property
    def K(self) -> int:
        return int(self.catalog.size)

    @property
    def U(self) -> int:
        return int(self.user_ids.size)

    @property
    def N(self) -> int:
        return int(self.dec_user.size)

    @property
    def lengths(self) -> np.ndarray:
        """FULL train sequence lengths L (inner sequences have L - 1)."""
        return np.diff(self.full_offsets)

    @property
    def inner_lengths(self) -> np.ndarray:
        return np.diff(self.seq_offsets)

    def key(self, u: int) -> str:
        return f"u{int(u):07d}"

    def keys(self, users: np.ndarray | None = None) -> list:
        idx = np.arange(self.U) if users is None else np.asarray(users)
        return [self.key(u) for u in idx]

    def user_of_key(self, k: str) -> int:
        if not (isinstance(k, str) and k.startswith("u") and len(k) == 8):
            raise BenchRefused(f"not a benchmark client key: {k!r}")
        return int(k[1:])

    def n_valid(self, u: int) -> int:
        return int(self.user_dec_offsets[u + 1] - self.user_dec_offsets[u])

    def band_sha256(self) -> str:
        return canon_sha256(self.user_ids[self.band].tolist())

    def fingerprint(self) -> str:
        """The cohort / manifest hash of this data view (participation plans and checkpoints bind to it)."""
        return canon_sha256({"dataset": self.dataset, "train_sha256": self.train_sha256, "K": self.K, "U": self.U,
                             "N": self.N, "max_len": self.max_len, "band_sha256": self.band_sha256()})

    def band_item_mask(self) -> np.ndarray:
        """[K] bool: items present in the band users' inner-train sequences (what PRE / S8 can have seen)."""
        m = np.zeros(self.K, dtype=bool)
        for u in np.flatnonzero(self.band):
            m[self.seq_items[self.seq_offsets[u]:self.seq_offsets[u + 1]]] = True
        return m

    def describe(self) -> dict:
        lab = self.val_item
        band_items = np.zeros(self.K, dtype=bool)
        bu = np.flatnonzero(self.band)
        for u in bu:                                       # the band's training part: s_0 .. s_{L-2}
            band_items[self.seq_items[self.seq_offsets[u]:self.seq_offsets[u + 1]]] = True
        return {"dataset": self.dataset, "train_path": str(self.train_path), "train_sha256": self.train_sha256,
                "K": self.K, "U": self.U, "N_train_decisions": self.N, "max_len": self.max_len,
                "band": {"seed": BAND_SEED, "frac": BAND_FRAC, "n_users": int(self.band.sum()),
                         "sha256": self.band_sha256(),
                         "n_train_decisions": int(sum(self.n_valid(u) for u in bu)),
                         "items_seen_in_band_inputs_frac": float(band_items.mean()),
                         "validation_targets_covered_frac": float(band_items[lab].mean())},
                "fingerprint": self.fingerprint(), **self.stats}

    def dec_range(self, u: int) -> tuple:
        return int(self.user_dec_offsets[u]), int(self.user_dec_offsets[u + 1])

    # ---------------------------------------------------------------------------------------------- batches
    def _arrays(self, source: str) -> tuple:
        if source == "inner":
            return self.seq_offsets, self.seq_items
        if source == "full":
            return self.full_offsets, self.full_items
        raise BenchRefused(f"unknown sequence source {source!r}")

    def _window(self, users: np.ndarray, ends: np.ndarray, source: str = "inner") -> dict:
        """Right-padded windows of the last <= max_len items before position `ends` (exclusive) of each user."""
        offs, items = self._arrays(source)
        import torch
        users = np.asarray(users, dtype=np.int64)
        ends = np.asarray(ends, dtype=np.int64)
        start = np.maximum(0, ends - self.max_len)
        ln = ends - start
        if ln.size and int(ln.min()) < 1:
            raise BenchRefused("a decision without context (length 0) cannot be scored")
        width = int(ln.max()) if ln.size else 1
        ar = np.arange(width, dtype=np.int64)
        valid = ar[None, :] < ln[:, None]
        idx = offs[users][:, None] + start[:, None] + ar[None, :]
        tok = np.zeros(valid.shape, dtype=np.int64)
        tok[valid] = items[idx[valid]] + ITEM_OFFSET
        pos = np.where(valid, ar[None, :] + 1, 0)
        return {"item_tokens": torch.from_numpy(tok), "lengths": torch.from_numpy(ln.copy()),
                "attention_mask": torch.from_numpy(valid), "position_ids": torch.from_numpy(pos)}

    def batch(self, rows) -> dict:
        """TRAIN decision rows (canonical row ids) -> a model batch (target_class, loss_mask, target_ts)."""
        import torch
        rows = np.asarray(rows, dtype=np.int64)
        u, t = self.dec_user[rows], self.dec_pos[rows]
        b = self._window(u, t)
        b["target_class"] = torch.from_numpy(self.seq_items[self.seq_offsets[u] + t].astype(np.int64))
        b["loss_mask"] = torch.ones(rows.size, dtype=torch.bool)
        b["target_ts"] = torch.from_numpy(t.copy())        # causal order key (position in the user's sequence)
        return b

    def client_examples(self, u: int) -> dict:
        a, b = self.dec_range(u)
        return self.batch(np.arange(a, b, dtype=np.int64))

    def band_rows(self) -> np.ndarray:
        """Canonical TRAIN decision ids of the band users (PRE's training set)."""
        bu = np.flatnonzero(self.band)
        if not bu.size:
            return np.zeros(0, dtype=np.int64)
        return np.concatenate([np.arange(*self.dec_range(u), dtype=np.int64) for u in bu])

    # ---------------------------------------------------------------------------------------------- evaluation
    def validation_users(self, tail="auto") -> np.ndarray:
        """The inner-validation population of the reference pipeline: users with >= 2 inner-train events whose
        validation item occurs in the inner-train items kept by the reference trainer, which keeps the last
        session_max_len + 1 inner events per user: 51 for the three S3 sets (session_max_len 50), 201 for ML-1M
        (session_max_len 200)."""
        tail = (201 if self.dataset == "ml_1m" else 51) if tail == "auto" else tail
        keep = self.inner_lengths >= 2
        present = np.zeros(self.K, dtype=bool)
        for u in np.flatnonzero(keep):
            seq = self.seq_items[self.seq_offsets[u]:self.seq_offsets[u + 1]]
            present[seq[-tail:] if tail else seq] = True
        return np.flatnonzero(keep & present[self.val_item]).astype(np.int64)

    def validation_view(self) -> EvalView:
        us = self.validation_users()
        return EvalView(self, "VALIDATION", us, ends=self.inner_lengths[us].astype(np.int64),
                        targets=self.val_item[us].astype(np.int64), source="inner")


@dataclass
class EvalView:
    """The scored rows of one split: users, context end (exclusive; = the seen prefix), single target per user."""
    data: BenchData
    split: str                         # VALIDATION | TEST
    users: np.ndarray
    ends: np.ndarray
    targets: np.ndarray                # internal class of the target; -1 = a cold holdout item (counted as a MISS)
    source: str = "inner"              # "inner" (validation) | "full" (TEST)
    n_cold_items: int = 0              # holdout rows kept as misses because their item is not in the train catalogue
    n_unknown_users: int = 0           # holdout rows dropped because their user has no train row (cannot be scored)

    def batch(self, sel) -> dict:
        sel = np.asarray(sel, dtype=np.int64)
        return self.data._window(self.users[sel], self.ends[sel], self.source)

    def seen_pairs(self, sel) -> tuple:
        """(row-in-batch, internal item) pairs of the seen items (the items of the context prefix [0, end))."""
        sel = np.asarray(sel, dtype=np.int64)
        d = self.data
        offs, arr = d._arrays(self.source)
        n = self.ends[sel]
        rows = np.repeat(np.arange(sel.size, dtype=np.int64), n)
        starts = offs[self.users[sel]]
        off = np.arange(int(n.sum()), dtype=np.int64) - np.repeat(np.cumsum(n) - n, n)
        items = arr[np.repeat(starts, n) + off]
        return rows, items

    def digest(self) -> str:
        return canon_sha256({"split": self.split, "source": self.source, "users": self.users.tolist(),
                             "ends": self.ends.tolist(), "targets": self.targets.tolist()})


def validation_positions(full_offsets: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """Position, inside each user's FULL ascending-stable sequence, of the reference trainer's inner-validation row.

    Reference rule: rank = interactions.sort_values(datetime, ascending=False, kind="stable").groupby(user)
    .cumcount(); validation row = rank 0. A stable DESCENDING sort keeps tied rows in file order, so rank 0 is the FIRST
    row (in file order) among the user's rows with the maximum datetime. Our sequences are sorted ascending by
    (datetime, file row), in which those tied rows sit contiguously at the end in file order: the validation row is the
    first of that trailing run of equal datetimes."""
    U = full_offsets.size - 1
    out = np.empty(U, dtype=np.int64)
    for u in range(U):
        a, b = int(full_offsets[u]), int(full_offsets[u + 1])
        seg = dt[a:b]
        out[u] = int(np.searchsorted(seg, seg[-1], side="left"))
    return out


def load(dataset: str, root, *, max_len: int = MAX_LEN) -> BenchData:
    """train.csv -> BenchData (the holdout is NOT read here)."""
    path = split_dir(root, dataset) / "train.csv"
    if not path.is_file():
        raise BenchRefused(f"{path} is absent")
    fr = _read(path)
    fr = fr.sort_values(["user_id", "datetime", "_source_order"], kind="stable")
    catalog = np.sort(fr["item_id"].unique().astype(np.int64))
    users = fr["user_id"].to_numpy(dtype=np.int64)
    items = np.searchsorted(catalog, fr["item_id"].to_numpy(dtype=np.int64)).astype(np.int64)
    dt = fr["datetime"].to_numpy().astype("int64")
    uids, starts, counts = np.unique(users, return_index=True, return_counts=True)
    full_off = np.zeros(uids.size + 1, dtype=np.int64)
    full_off[1:] = np.cumsum(counts)
    if not np.array_equal(starts, full_off[:-1]):
        raise BenchRefused("the sorted frame is not grouped by user")
    vpos = validation_positions(full_off, dt)
    keep = np.ones(items.size, dtype=bool)
    keep[full_off[:-1] + vpos] = False
    val_item = items[full_off[:-1] + vpos]
    inner_items = items[keep]
    inner_off = np.zeros(uids.size + 1, dtype=np.int64)
    inner_off[1:] = np.cumsum(counts - 1)
    band_ids = band_users(uids)
    band = np.isin(uids, band_ids)
    ndec = np.maximum(counts - 2, 0).astype(np.int64)      # inner targets t = 1 .. L-2
    udo = np.zeros(uids.size + 1, dtype=np.int64)
    udo[1:] = np.cumsum(ndec)
    dec_user = np.repeat(np.arange(uids.size, dtype=np.int64), ndec)
    dec_pos = (np.arange(int(ndec.sum()), dtype=np.int64) - np.repeat(udo[:-1], ndec)) + 1
    stats = {"train_rows": len(fr), "n_train_users": int(uids.size),
             "n_val_row_not_last": int((vpos != counts - 1).sum())}
    return BenchData(dataset=dataset, train_path=path, train_sha256=sha256_file(path), catalog=catalog,
                     user_ids=uids, seq_offsets=inner_off, seq_items=inner_items, full_offsets=full_off,
                     full_items=items, val_pos=vpos, val_item=val_item, band=band, dec_user=dec_user,
                     dec_pos=dec_pos, user_dec_offsets=udo, max_len=int(max_len), stats=stats)


def load_holdout_view(data: BenchData, root) -> EvalView:
    """The TEST view (holdout.csv). Called by holdout.py only, after model selection on the inner validation split."""
    path = split_dir(root, data.dataset) / "holdout.csv"
    fr = _read(path).sort_values(["user_id", "datetime", "_source_order"], kind="stable")
    hu = fr["user_id"].to_numpy(dtype=np.int64)
    hi = fr["item_id"].to_numpy(dtype=np.int64)
    ui = np.searchsorted(data.user_ids, hu)
    ok_u = (ui < data.U) & (data.user_ids[np.minimum(ui, data.U - 1)] == hu)
    ci = np.searchsorted(data.catalog, hi)
    ok_i = (ci < data.K) & (data.catalog[np.minimum(ci, data.K - 1)] == hi)
    # Unknown users cannot be scored (no history) and are counted. A COLD holdout item (not in the train catalogue)
    # is KEPT as a miss (rank 0) in the denominator, the semantics of the reference evaluation (the user stays among the
    # holdout users, the item can never be recommended). Counted and reported (EvalView.n_cold_items); never dropped.
    n_unknown_users = int((~ok_u).sum())
    n_cold = int((ok_u & ~ok_i).sum())
    ci = np.where(ok_i, ci, -1)
    ui, ci = ui[ok_u], ci[ok_u]
    if np.unique(ui).size != ui.size:
        raise BenchRefused("leave-one-out holdout must hold one item per user")
    order = np.argsort(ui, kind="stable")
    ui, ci = ui[order].astype(np.int64), ci[order].astype(np.int64)
    return EvalView(data, "TEST", ui, ends=data.lengths[ui].astype(np.int64), targets=ci, source="full",
                    n_cold_items=n_cold, n_unknown_users=n_unknown_users)


def count_csv_rows(path) -> int:
    with open(path, "rb") as f:
        return sum(1 for _ in f) - 1


def statistics_check(dataset: str, root, data: BenchData | None = None) -> dict:
    """Counts of the adapter vs the release's statistics.csv (holdout: row count only, no item is read)."""
    import pandas as pd
    st = pd.read_csv(Path(root) / dataset / "statistics.csv").iloc[0].to_dict()
    d = data if data is not None else load(dataset, root)
    hrows = count_csv_rows(split_dir(root, dataset) / "holdout.csv")
    out = {"n_users": (int(st["n_users"]), d.U),
           "train_rows": (int(st["leave_one_out_n_train_interactions"]), int(d.full_items.size)),
           "holdout_rows": (int(st["leave_one_out_n_holdout_interactions"]), hrows),
           "train_decisions_eq_rows_minus_2U": (int(d.full_items.size) - 2 * d.U, d.N),
           "catalog_le_n_items": (True, d.K <= int(st["n_items"]))}
    out["ok"] = all(a == b for a, b in out.values())
    out["n_items_release"], out["K_train_catalog"] = int(st["n_items"]), d.K
    return out
