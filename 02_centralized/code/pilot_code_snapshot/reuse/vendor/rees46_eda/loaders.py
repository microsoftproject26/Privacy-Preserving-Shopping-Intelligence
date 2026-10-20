"""Engine-neutral NumPy/Arrow loaders for the shared decision ledger.

Both the enriched GRU and the enriched SASRec consume the SAME decisions, the
same bounded context window, the same features and the same targets. The only
difference is how a model consumes the batch:

* GRU  packs by ``lengths`` over a ZERO-INITIALISED bounded window - it may not
  carry hidden state from before the shared window.
* SASRec applies a causal mask plus the key-padding mask and gathers the query
  at ``last_valid_index``, not at an arbitrary final padded position.

Nothing here builds, trains or optimises a model.

IMPORTANT for consumers: this is a per-decision ledger, not one window per
user. Do NOT copy the original SASRec sampler's "last window per user"
behaviour onto it - that would silently drop most supervised decisions.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pyarrow.parquet as pq

from .config import (
    FEATURE_GROUPS, REQUIRED_FLAG_FOR,
    EVENT_NUMERIC_FEATURES, EVENT_QUALITY_FLAGS, NORMALIZED_EVENT_NUMERIC,
    NORMALIZED_SESSION_PREFIX, NORMALIZED_USER_CONTEXT, NO_TARGET_CLASS, OOV, PAD,
    SESSION_PREFIX_FEATURES, USER_CONTEXT_FEATURES, USER_CONTEXT_MASKS,
)

# Stored column name -> batch key. Storage uses the *_idx naming (contiguous
# TRAIN-fitted model indices); the batch keeps the documented contract names.
IDX_COLUMNS = {
    "product_idx": "item_tokens",
    "category_idx": "category_tokens",
    "brand_idx": "brand_tokens",
    "main_category_idx": "main_category_tokens",
    # Contextual local-time categoricals. Closed vocabularies, so these never
    # produce OOV in practice - but they carry the same reserved semantics as
    # every other categorical channel so a downstream embedding table can be
    # sized and masked identically.
    "daypart_idx": "daypart_tokens",
    "local_weekday_idx": "weekday_tokens",
}

# Per-event numeric block presented to the model, in a fixed documented order.
EVENT_NUMERIC_BLOCK = EVENT_NUMERIC_FEATURES + SESSION_PREFIX_FEATURES

# Raw numeric feature -> the boolean flag that forces it to normalized zero.
MISSING_FLAG_OF = {
    "log1p_price": "price_missing_or_conflict",
    "log1p_prev_raw_view_gap_s": "prev_gap_unavailable",
    "log1p_item_observed_age_s": "item_age_unavailable",
}
USER_MISSING_FLAG_OF = {
    "recent_mean_log_price": "recent_price_supported",
    "current_minus_recent_mean_log_price": "recent_price_supported",
}


@dataclass
class Normalizers:
    spec: dict[str, dict[str, float]]

    @classmethod
    def load(cls, path: Path) -> "Normalizers":
        with open(path, "r", encoding="utf-8") as fh:
            return cls(spec=json.load(fh)["features"])

    def apply(self, name: str, x: np.ndarray) -> np.ndarray:
        p = self.spec.get(name)
        if p is None:
            return x.astype(np.float32)
        if p.get("constant"):
            return np.zeros_like(x, dtype=np.float32)
        y = np.clip(x, p["clip_lo"], p["clip_hi"])
        return ((y - p["mean"]) / p["std"]).astype(np.float32)


class BucketStore:
    """All feature rows of one user hash bucket, indexed by logical key.

    Windows are resolved by the LOGICAL key (user_id, segment_index,
    position_in_segment), never by physical file order, so a repartition or a
    different write order cannot change a batch.
    """

    def __init__(self, path: Path, columns: Optional[list[str]] = None):
        tbl = pq.read_table(path, columns=columns)
        self.n = tbl.num_rows
        self.col: dict[str, np.ndarray] = {}
        for name in tbl.column_names:
            arr = tbl[name]
            try:
                self.col[name] = arr.to_numpy(zero_copy_only=False)
            except Exception:
                self.col[name] = np.asarray(arr.to_pylist())
        order = np.lexsort((self.col["position_in_segment"],
                            self.col["segment_index"], self.col["user_id"]))
        for k in self.col:
            self.col[k] = self.col[k][order]
        # segment start offsets, so a window is a contiguous slice
        u, s = self.col["user_id"], self.col["segment_index"]
        newseg = np.concatenate(([True], (u[1:] != u[:-1]) | (s[1:] != s[:-1])))
        self.seg_start = np.flatnonzero(newseg)
        self.seg_of_row = np.cumsum(newseg) - 1
        self.row_of_decision = {int(d): i for i, d in enumerate(self.col["decision_id"])}


# Directory layouts this loader can open. A live run dir and a frozen version
# dir hold the SAME artifacts under different names: the freeze renames them to
# read as a published dataset rather than as pipeline internals. Resolving the
# names here means the frozen version is directly train-loadable - the
# alternative was asking a consumer to rename folders, which makes the frozen
# copy the one artifact the production loader cannot read.
LAYOUTS: dict[str, dict[str, str]] = {
    "live":   {"features": "features",
               "index": "indices/decisions",
               "vocab": "vocabularies"},
    "frozen": {"features": "model_ready",
               "index": "decisions_index",
               "vocab": "mappings"},
}


def detect_layout(root: Path) -> str:
    """Which layout `root` holds, decided by what is actually on disk.

    A directory is only claimed when the vocabulary AND the feature root both
    exist under that naming, so a partially copied tree raises rather than
    resolving half its paths against the wrong layout.
    """
    root = Path(root)
    found = [name for name, m in LAYOUTS.items()
             if (root / m["vocab"]).is_dir() and (root / m["features"]).is_dir()]
    if len(found) == 1:
        return found[0]
    if len(found) > 1:                       # both namings present: ambiguous
        raise ValueError(
            f"{root} matches multiple layouts {found}; pass layout= explicitly")
    raise FileNotFoundError(
        f"{root} matches no known layout. Expected either "
        f"{LAYOUTS['live']['vocab']}/ + {LAYOUTS['live']['features']}/ (live run) or "
        f"{LAYOUTS['frozen']['vocab']}/ + {LAYOUTS['frozen']['features']}/ (frozen version)")


class DecisionWindowLoader:
    """Lazy, decision-level window loader shared by both backbones.

    `layout` selects the directory naming: "live" for a run dir, "frozen" for a
    frozen version dir, or None to detect it from disk.
    """

    def __init__(self, run_dir: Path, context_length: int,
                 normalizers: Optional[Normalizers] = None,
                 layout: Optional[str] = None):
        self.run_dir = Path(run_dir)
        self.L = int(context_length)
        self.layout = layout or detect_layout(self.run_dir)
        if self.layout not in LAYOUTS:
            raise ValueError(f"unknown layout {self.layout!r}; "
                             f"expected one of {sorted(LAYOUTS)}")
        m = LAYOUTS[self.layout]
        self.features_root = self.run_dir / m["features"]
        self.index_root = self.run_dir.joinpath(*m["index"].split("/"))
        self.vocab_root = self.run_dir / m["vocab"]
        self.normalizers = normalizers or Normalizers.load(
            self.vocab_root / "normalizers.json")
        self._buckets: dict[int, BucketStore] = {}
        cat = pq.read_table(self.vocab_root / "catalogue.parquet")
        self.catalogue_size = cat.num_rows
        self.product_idx_of_class = cat["product_idx"].to_numpy(zero_copy_only=False)
        # raw STRING product ids, so a class can always be translated back
        self.product_id_of_class = cat["product_id"].to_pylist()
        # `client_id` in the emitted batch is the PSEUDONYMOUS contiguous index,
        # never the raw user id string. Raw-user provenance stays in the private
        # index and the mapping file only.
        cl = pq.read_table(self.vocab_root / "clients.parquet")
        self.client_idx_of_user = dict(
            zip(cl["user_id"].to_pylist(), cl["client_idx"].to_pylist()))

    # -- plumbing ---------------------------------------------------------
    def bucket(self, b: int) -> BucketStore:
        if b not in self._buckets:
            p = self.features_root / f"user_bucket={b}" / "features.parquet"
            self._buckets[b] = BucketStore(p)
        return self._buckets[b]

    def index(self, split: str, buckets: Optional[Iterable[int]] = None):
        """Decision index for one partition.

        Files are addressed by explicit path rather than by a Hive partition
        filter, so the loader never depends on how the engine happened to infer
        a partition column's type.
        """
        import pyarrow as pa
        root = self.index_root / f"split={split}"
        if not root.exists():
            raise FileNotFoundError(f"no decision index for split {split}: {root}")
        if buckets is None:
            dirs = sorted(root.glob("user_bucket=*"))
        else:
            dirs = [root / f"user_bucket={int(b)}" for b in buckets]
        parts = []
        for d in dirs:
            files = sorted(d.glob("*.parquet"))
            if not files:
                continue
            b = int(d.name.split("=", 1)[1])
            t = pq.read_table(files)
            if "user_bucket" in t.column_names:
                t = t.drop(["user_bucket"])
            t = t.append_column("user_bucket",
                                pa.array(np.full(t.num_rows, b, np.int64)))
            parts.append(t)
        if not parts:
            any_file = next(iter(sorted(root.glob("user_bucket=*/*.parquet"))), None)
            if any_file is None:
                raise FileNotFoundError(f"no index files under {root}")
            # requested buckets are empty: return an EMPTY table with the real schema
            sch = pq.read_schema(any_file)
            if "user_bucket" in sch.names:
                sch = sch.remove(sch.get_field_index("user_bucket"))
            sch = sch.append(pa.field("user_bucket", pa.int64()))
            return sch.empty_table()
        return pa.concat_tables(parts)

    # -- batch assembly ---------------------------------------------------
    def collate(self, decisions: list[dict[str, Any]]) -> dict[str, np.ndarray]:
        B, L = len(decisions), self.L
        F = len(EVENT_NUMERIC_BLOCK)
        Q = len(EVENT_QUALITY_FLAGS)
        U = len(USER_CONTEXT_FEATURES)
        M = len(USER_CONTEXT_MASKS)

        item = np.zeros((B, L), np.int64)
        cat = np.zeros((B, L), np.int64)
        brand = np.zeros((B, L), np.int64)
        main = np.zeros((B, L), np.int64)
        daypart = np.zeros((B, L), np.int64)
        weekday = np.zeros((B, L), np.int64)
        num = np.zeros((B, L, F), np.float32)
        flags = np.zeros((B, L, Q), np.int8)
        attn = np.zeros((B, L), bool)
        pos = np.zeros((B, L), np.int64)
        lengths = np.zeros(B, np.int64)
        last_idx = np.zeros(B, np.int64)
        uctx = np.zeros((B, U), np.float32)
        umask = np.zeros((B, M), np.float32)
        tgt = np.full(B, NO_TARGET_CLASS, np.int64)
        loss = np.zeros(B, bool)
        ev = np.zeros(B, bool)
        dec_ids = np.zeros(B, np.int64)
        clients = np.zeros(B, np.int64)
        segs = np.zeros(B, np.int64)

        for bi, d in enumerate(decisions):
            store = self.bucket(int(d["user_bucket"]))
            row = store.row_of_decision[int(d["decision_id"])]
            seg = store.seg_of_row[row]
            seg_lo = store.seg_start[seg]
            p = int(store.col["position_in_segment"][row])      # 1-based
            take = min(L, p)
            lo = row - take + 1
            assert lo >= seg_lo, "window escaped its model segment"
            sl = slice(lo, row + 1)

            item[bi, :take] = store.col["product_idx"][sl]
            cat[bi, :take] = store.col["category_idx"][sl]
            brand[bi, :take] = store.col["brand_idx"][sl]
            main[bi, :take] = store.col["main_category_idx"][sl]
            daypart[bi, :take] = store.col["daypart_idx"][sl]
            weekday[bi, :take] = store.col["local_weekday_idx"][sl]
            attn[bi, :take] = True
            pos[bi, :take] = np.arange(1, take + 1)
            lengths[bi] = take
            last_idx[bi] = take - 1

            # `price_missing` already covers an invalid observed price: the block
            # aggregate is built from `price_valid`, so a negative or non-finite
            # price never becomes an eligible observation.
            pmiss = (store.col["price_missing"][sl] | store.col["price_conflict"][sl])
            for fi, name in enumerate(EVENT_NUMERIC_BLOCK):
                raw = store.col[name][sl].astype(np.float64)
                if name in NORMALIZED_EVENT_NUMERIC or name in NORMALIZED_SESSION_PREFIX:
                    v = self.normalizers.apply(name, raw)
                else:
                    v = raw.astype(np.float32)          # sine/cosine: never scaled
                flag = MISSING_FLAG_OF.get(name)
                if flag == "price_missing_or_conflict":
                    v = np.where(pmiss, 0.0, v).astype(np.float32)
                elif flag is not None:
                    v = np.where(store.col[flag][sl], 0.0, v).astype(np.float32)
                num[bi, :take, fi] = v
            for qi, name in enumerate(EVENT_QUALITY_FLAGS):
                flags[bi, :take, qi] = store.col[name][sl].astype(np.int8)

            for ui, name in enumerate(USER_CONTEXT_FEATURES):
                raw = np.float64(store.col[name][row])
                if name in NORMALIZED_USER_CONTEXT:
                    val = float(self.normalizers.apply(name, np.array([raw]))[0])
                else:
                    val = float(raw)
                sup = USER_MISSING_FLAG_OF.get(name)
                if sup is not None and float(store.col[sup][row]) == 0.0:
                    val = 0.0
                uctx[bi, ui] = val
            for mi, name in enumerate(USER_CONTEXT_MASKS):
                umask[bi, mi] = float(store.col[name][row])

            tgt[bi] = int(d["target_class"])
            loss[bi] = bool(d["loss_mask"])
            ev[bi] = bool(d["eval_mask"])
            dec_ids[bi] = int(d["decision_id"])
            clients[bi] = int(d.get("client_idx")
                              or self.client_idx_of_user.get(d["user_id"], OOV))
            segs[bi] = int(store.col["segment_index"][row])

        return {
            "item_tokens": item, "category_tokens": cat, "brand_tokens": brand,
            "main_category_tokens": main,
            "daypart_tokens": daypart, "weekday_tokens": weekday,
            "event_numeric_features": num,
            "event_quality_flags": flags, "user_context": uctx, "user_context_masks": umask,
            "attention_mask": attn, "position_ids": pos, "lengths": lengths,
            "last_valid_index": last_idx, "target_class": tgt,
            "loss_mask": loss, "eval_mask": ev,
            "decision_id": dec_ids, "client_id": clients, "model_segment_id": segs,
        }

    def batch_from_index(self, index_table, rows: Iterable[int]) -> dict[str, np.ndarray]:
        cols = index_table.column_names
        recs = []
        for r in rows:
            recs.append({c: index_table[c][r].as_py() for c in cols})
        return self.collate(recs)


# ---------------------------------------------------------------------------
# Adapters (data-only; no model code)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Ablation groups
# ---------------------------------------------------------------------------

def feature_group_view(batch: dict[str, np.ndarray], group: str) -> dict[str, np.ndarray]:
    """A VIEW of one batch restricted to an ablation group's channels.

    This slices; it never rebuilds. Every group therefore sees exactly the same
    decisions, the same window and the same targets, so an ablation compares
    information content and nothing else. Supervision keys are always present:
    a group without a target is not an ablation, it is a different task.
    """
    spec = FEATURE_GROUPS[group]
    block = EVENT_NUMERIC_BLOCK
    keep_num = block if spec["numeric"] == ["ALL"] else spec["numeric"]
    cols = [block.index(n) for n in keep_num]

    out: dict[str, np.ndarray] = {
        k: batch[k] for k in ("attention_mask", "position_ids", "lengths",
                              "last_valid_index", "target_class", "loss_mask",
                              "eval_mask", "decision_id", "client_id",
                              "model_segment_id") if k in batch}
    for ch in spec["categorical"]:
        out[ch] = batch[ch]
    out["event_numeric_features"] = (
        batch["event_numeric_features"][..., cols] if cols
        else np.zeros(batch["event_numeric_features"].shape[:2] + (0,), np.float32))
    out["event_numeric_feature_names"] = np.array(keep_num, dtype=object)

    # Quality flags travel with the numerics, always. Several numerics are
    # ZERO-FILLED when unavailable, so handing them over without their
    # indicator tells the model a missing price is a free item and a missing
    # gap is an instantaneous one. The protocol owner settled this (C5): the
    # neutral value reaches the model ONLY together with the missing
    # indicator. Enforced here rather than trusted, because this function is
    # the access path named in the frozen FEATURE_GROUPS.json.
    want_flags = spec.get("quality_flags", ["ALL"])
    if want_flags == ["ALL"] or want_flags:
        needed = set(EVENT_QUALITY_FLAGS if want_flags == ["ALL"] else want_flags)
        for n in keep_num:
            needed.update(REQUIRED_FLAG_FOR.get(n, []))
        fcols = [EVENT_QUALITY_FLAGS.index(f) for f in EVENT_QUALITY_FLAGS
                 if f in needed]
        out["event_quality_flags"] = batch["event_quality_flags"][..., fcols]
        out["event_quality_flag_names"] = np.array(
            [EVENT_QUALITY_FLAGS[i] for i in fcols], dtype=object)
    else:
        missing = {f for n in keep_num for f in REQUIRED_FLAG_FOR.get(n, [])}
        if missing:
            raise ValueError(
                f"group {group!r} selects zero-filled numerics {sorted(keep_num)} "
                f"but declares no quality flags; {sorted(missing)} are required "
                "so the model can tell a missing value from a real zero")
    # Only [] or ["ALL"] are defined. A future group naming individual
    # user-context features would otherwise silently receive NEITHER the
    # features nor their masks - the failure would look like an ablation
    # result rather than a bug.
    _uc = spec["user_context"]
    if _uc == ["ALL"]:
        out["user_context"] = batch["user_context"]
        out["user_context_masks"] = batch["user_context_masks"]
    elif _uc:
        raise ValueError(
            f"group {group!r} declares user_context={_uc!r}; only [] or ['ALL'] "
            "are supported, because a mask must always travel with its feature")
    out["feature_group"] = group
    return out


# Channels a backbone REQUIRES to be a valid next-item model at all. Absence is
# a contract violation, not an ablation.
REQUIRED_SHARED = ("item_tokens", "target_class", "loss_mask", "eval_mask",
                   "decision_id", "attention_mask")
# Channels an ablation group may legitimately withhold. A backbone must pass
# through whatever is present and must NOT invent what is not - reintroducing a
# channel F0 removed would silently turn every ablation into F4.
OPTIONAL_CHANNELS = (
    "category_tokens", "brand_tokens", "main_category_tokens",
    "daypart_tokens", "weekday_tokens",
    "event_numeric_features", "event_quality_flags",
    "event_numeric_feature_names", "event_quality_flag_names",
    "user_context", "user_context_masks",
    "client_id", "model_segment_id", "feature_group",
)


def _require(batch: dict[str, Any], keys, backbone: str) -> None:
    missing = [k for k in keys if k not in batch]
    if missing:
        raise KeyError(
            f"{backbone} requires {missing}; a feature group may withhold optional "
            f"feature channels but never the sequence, masks or supervision keys")


def _passthrough(batch: dict[str, Any]) -> dict[str, Any]:
    return {k: batch[k] for k in OPTIONAL_CHANNELS if k in batch}


def gru_inputs(batch: dict[str, Any]) -> dict[str, Any]:
    """GRU view: packed by `lengths` over a ZERO-INITIALISED bounded window.

    No hidden state may be carried in from before the shared window; a
    full-session stateful GRU is a separately labelled history-policy
    experiment, not this contract.

    Optional feature channels are passed through ONLY when present, so an
    ablation group that withholds them stays withheld. This is why the function
    cannot simply name every channel: doing so made F0 and F1 unusable.
    """
    _require(batch, REQUIRED_SHARED + ("lengths",), "GRU")
    out = _passthrough(batch)
    out.update({
        "item_tokens": batch["item_tokens"],
        "lengths": batch["lengths"],
        "initial_state": "zeros",
        "attention_mask": batch["attention_mask"],
        "target_class": batch["target_class"], "loss_mask": batch["loss_mask"],
        "eval_mask": batch["eval_mask"], "decision_id": batch["decision_id"],
    })
    return out


def sasrec_inputs(batch: dict[str, Any]) -> dict[str, Any]:
    """SASRec view: key-padding mask + causal mask, query at last_valid_index."""
    _require(batch, REQUIRED_SHARED + ("position_ids", "last_valid_index"), "SASRec")
    B, L = batch["item_tokens"].shape
    causal = np.tril(np.ones((L, L), bool))
    out = _passthrough(batch)
    out.update({
        "item_tokens": batch["item_tokens"],
        "key_padding_mask": ~batch["attention_mask"], "causal_mask": causal,
        "position_ids": batch["position_ids"], "query_index": batch["last_valid_index"],
        "target_class": batch["target_class"], "loss_mask": batch["loss_mask"],
        "eval_mask": batch["eval_mask"], "decision_id": batch["decision_id"],
    })
    return out


def model_inputs(batch: dict[str, Any], feature_group: Optional[str] = None,
                 backbone: str = "gru") -> dict[str, Any]:
    """The single training access path: pick an ablation group, pick a backbone.

    Group selection happens FIRST and backbone adaptation second, so the two
    concerns cannot contaminate each other: the group decides what information
    exists, the backbone decides only how it is shaped.
    """
    b = feature_group_view(batch, feature_group) if feature_group else batch
    bb = backbone.strip().lower()
    if bb == "gru":
        return gru_inputs(b)
    if bb == "sasrec":
        return sasrec_inputs(b)
    raise ValueError(f"unknown backbone {backbone!r}; expected 'gru' or 'sasrec'")


def negative_sampler(seed: int, decision_id: int, epoch: int, n: int,
                     catalogue_size: int, positive_class: int) -> np.ndarray:
    """Deterministic uniform candidate sampler over REAL catalogue classes.

    Excludes the current positive and, by construction, every special token
    (PAD / MISSING / OOV are not output classes at all). It does NOT exclude
    the user's future history.
    """
    out = np.empty(n, np.int64)
    filled = 0
    k = 0
    while filled < n:
        # A cryptographic digest, NOT Python's hash(): hash() is interpreter
        # width dependent and therefore not portable, and the specification
        # forbids it for any persistent or reproducible identifier.
        digest = hashlib.blake2b(
            f"{seed}:{int(decision_id)}:{int(epoch)}:{k}".encode(), digest_size=8).digest()
        rng = np.random.default_rng(int.from_bytes(digest, "big"))
        cand = rng.integers(0, catalogue_size, size=max(n * 2, 16))
        cand = cand[cand != positive_class]
        take = min(n - filled, cand.shape[0])
        out[filled:filled + take] = cand[:take]
        filled += take
        k += 1
    return out
