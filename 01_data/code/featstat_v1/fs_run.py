"""featstat_v1.fs_run - FEATSTAT-01 main job (statistics only; no model, no training, no GPU).

Populations (TRAIN only; n = supervised TRAIN examples under catalogue A = TRAIN decisions with loss_mask, the
CENSUS-02 definition):
  POOL = users with 5 <= n <= 100 (owner amendment section 1); FULL = all users with n >= 1 (comparison only).
Rows: every supervised TRAIN example of the population in the Clean V2 layout. For each user bucket the v1 rich
CompactStore is built through v4data.store.CleanV2Store (splits=("TRAIN",)) and v4data.transform.clean_v2_view is
applied to the decision rows (one row per supervised example, its own event = last_valid_index). A deterministic
sample of real CleanV2Store.batch windows per bucket is compared byte for byte with these rows, and the reference
encoder v4data.transform.encode_rows is compared on every row. Raw v1 columns are read (filtered to split == TRAIN at
read time) where a statistic needs the pre-transform value. VALIDATION and TEST are never requested.
No raw identifier is written anywhere: user_id and product_id strings live only in memory.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
import ppsi_env  # noqa: E402,F401  (import isolation first, T43)

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from v4data.common import (K, SCHEMA_SELF_SHA256, VERSION_HASH, dataset_root, load_schema,  # noqa: E402
                           sha256_file)
from v4data.store import CleanV2Store  # noqa: E402
from v4data.transform import V1_VIEW_TAG, build_spec, clean_v2_view, encode_rows  # noqa: E402
from v4data.transform_state import load_state  # noqa: E402

from featstat_v1 import fs_core as F  # noqa: E402
from featstat_v1 import fs_inventory as INV  # noqa: E402

ITEM_VOCAB = 158489
SQRT2 = math.sqrt(2.0)
EXPECT = {"pool_users": 1446986, "pool_examples": 28623386, "nge1_users": 2694998, "nge1_examples": 40180608,
          "B_users": 1197003, "B_examples": 2503629, "D_users": 51009, "D_examples": 9053593, "A1_users": 1509381}
CELLS = ("POOL_DEC", "POOL_NONDEC", "NGE1_NONPOOL_DEC", "NGE1_NONPOOL_NONDEC", "N0_ROWS")
CLASSES = ("V2_VALID", "V1_ONLY", "INVALID")
DROPPED_NUM = {"local_hour_sin": 5, "local_hour_cos": 6, "prefix_item_count": 14, "log1p_prefix_elapsed_s": 18}
DROPPED_MASK = {"has_history_mask": 0, "prev_user_event_available": 4}
TOKENS = (("item_tokens", "_item_tokens", "product_idx"), ("category_tokens", "_category_tokens", "category_idx"),
          ("brand_tokens", "_brand_tokens", "brand_idx"),
          ("main_category_tokens", "_main_category_tokens", "main_category_idx"),
          ("daypart_tokens", "_daypart_tokens", "daypart_idx"), ("weekday_tokens", "_weekday_tokens", "local_weekday_idx"))
AUX = "category_code_idx"
CYCLIC = ("utc_hour_sin", "utc_hour_cos", "utc_weekday_sin", "utc_weekday_cos", "local_weekday_sin",
          "local_weekday_cos", "local_hour_sin", "local_hour_cos")
GATED_V2 = ("log1p_price", "log1p_prev_raw_view_gap_s", "log1p_item_observed_age_s", "recent_mean_log_price",
            "current_minus_recent_mean_log_price", "log1p_time_since_prev_user_event_s",
            "log1p_time_since_prev_session_last_event_s")
DEAD_FLAGS = ("price_conflict", "brand_conflict", "category_missing", "category_recovered", "category_conflict",
              "category_code_recovered")
SOURCES = ("brand_source", "category_id_source", "category_code_source")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def utc_now():
    try:
        return subprocess.run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def release_memory():
    """gc + return freed heap pages to the OS (memory only; the pattern of snap_v1 full_store)."""
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:  # noqa: BLE001
        pass


def rss_mib():
    try:
        out = {}
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith(("VmRSS:", "VmHWM:")):
                    out[line.split(":")[0]] = int(line.split()[1]) / 1024.0
        return out
    except OSError:
        return {}


# ================================================================================ column registry
def build_registry(spec, v1norm):
    """The 59 decision-grain columns: 46 Clean V2 channels, the 6 dropped channels, 6 tokens, 1 aux."""
    reg = []

    def z0(name, p=None):
        lo, hi, mu, sd = p if p is not None else spec.norm_v2[name]
        return F.f32((min(max(0.0, lo), hi) - mu) / sd)

    for n in spec.v2_numeric:
        if n in CYCLIC:
            kind, dflt = "cyclic", None
        elif n == "is_weekend":
            kind, dflt = "binary", 0.0
        elif n in GATED_V2:
            kind, dflt = "normalized_gated", 0.0
        else:
            kind, dflt = "normalized", z0(n)
        reg.append({"name": n, "block": "v2_numeric", "kind": kind, "default": dflt})
    for n in spec.v2_flags:
        reg.append({"name": n, "block": "v2_flag", "kind": "binary", "default": 0.0})
    for n in spec.v2_uctx:
        if n == "has_history":
            kind, dflt = "binary", 0.0
        elif n in ("recent_category_match_share", "recent_brand_match_share"):
            kind, dflt = "share", 0.0
        elif n in GATED_V2:
            kind, dflt = "normalized_gated", 0.0
        else:
            kind, dflt = "normalized", z0(n)
        reg.append({"name": n, "block": "v2_uctx", "kind": kind, "default": dflt})
    for n in spec.v2_masks:
        reg.append({"name": n, "block": "v2_mask", "kind": "binary", "default": 0.0})
    for n in DROPPED_NUM:
        if n in CYCLIC:
            kind, dflt = "cyclic", None
        else:
            p = v1norm[n]
            raw0 = 1.0 if n == "prefix_item_count" else 0.0
            kind = "normalized"
            dflt = F.f32((min(max(raw0, p["clip_lo"]), p["clip_hi"]) - p["mean"]) / p["std"])
        reg.append({"name": n, "block": "dropped_numeric", "kind": kind, "default": dflt})
    for n in DROPPED_MASK:
        reg.append({"name": n, "block": "dropped_mask", "kind": "binary", "default": 0.0})
    for n, _, _ in TOKENS:
        reg.append({"name": n, "block": "token", "kind": "token", "default": None})
    reg.append({"name": AUX, "block": "aux_not_batched", "kind": "token", "default": None})
    for i, r in enumerate(reg):
        r["idx"] = i
    return reg


# ================================================================================ phase A (per bucket)
class Acc:
    def __init__(self, reg):
        self.reg = reg
        self.dicts = {r["name"]: [] for r in reg}
        self.codes = {r["name"]: [] for r in reg}
        self.pool = []
        self.n_dec = []
        self.nfreq = {}
        self.strata = {k: 0 for k in ("A1_users", "B_users", "B_examples", "pool_users", "pool_examples",
                                      "D_users", "D_examples", "nge1_users", "nge1_examples", "train_users",
                                      "train_rows")}
        self.verify = {"store_batch_windows_checked": 0, "store_batch_mismatches": 0, "pad_nonzero": 0,
                       "encode_rows_max_abs_diff": {"event_numeric_features": 0.0, "event_quality_flags": 0.0,
                                                    "user_context": 0.0, "user_context_masks": 0.0},
                       "alignment_checks": 0, "rc_counts_view": {}, "names_ok": True}
        self.ledger = {}
        self.ledger_max_rot = {}
        self.raw_zero = {}
        self.gate2x2 = {}
        self.win = {"FULL": {}, "POOL": {}}
        self.win_item = {"FULL": np.zeros(ITEM_VOCAB), "POOL": np.zeros(ITEM_VOCAB)}
        self.raw_tables = {}
        self.cell_bincount = {}
        self.targets = {"FULL": np.zeros(K, np.int64), "POOL": np.zeros(K, np.int64)}
        self.pdict: dict = {}
        self.fs_parts = {c: [] for c in range(5)}
        self.poolrows = {"pcode": [], "bts": [], "age": [], "age_unavail": [], "is_dec": []}
        self.nonwhole_second_ts = 0
        self.sources = {}
        self.bucket_log = []

    def add_count(self, d, key, grp, val):
        d.setdefault(key, {}).setdefault(grp, 0)
        d[key][grp] += int(val)


def _rawcols(spec):
    return (["user_id", "decision_id", "block_ts", "product_id", "segment_index", "position_in_segment",
             "product_idx", "category_idx", "brand_idx", "main_category_idx", "category_code_idx", "daypart_idx",
             "local_weekday_idx"] + list(spec.v1_numeric) + list(spec.v1_flags) + list(spec.v1_uctx)
            + list(spec.v1_masks) + list(SOURCES))


def process_bucket(b, ctx, acc):
    t0 = time.time()
    spec = ctx["spec"]
    store = CleanV2Store.build(splits=("TRAIN",), buckets=[b], ds_root=ctx["ds_root"],
                               state_sha256=ctx["state_sha"], n_threads=3, log=lambda *a, **k: None)
    cs = store.cs
    ds = store.decision_set("TRAIN", "loss")
    nrow = int(cs.n_rows)
    rows = ds.end_row.astype(np.int64)
    ndec = rows.shape[0]

    # ---------------- raw v1 columns, TRAIN rows only (filtered at read), in the store's row order
    path = Path(ctx["ds_root"]) / "model_ready" / f"user_bucket={b}" / "features.parquet"
    tbl = pq.read_table(path, columns=["split"] + _rawcols(spec), filters=[("split", "==", "TRAIN")])
    if tbl.num_rows != nrow:
        raise AssertionError(f"bucket {b}: raw TRAIN rows {tbl.num_rows} != store rows {nrow}")
    if pc.any(pc.not_equal(tbl["split"], "TRAIN")).as_py():
        raise AssertionError("non-TRAIN row in the filtered read")
    uid = tbl["user_id"].to_numpy(zero_copy_only=False)
    seg = tbl["segment_index"].to_numpy(zero_copy_only=False)
    pos = tbl["position_in_segment"].to_numpy(zero_copy_only=False)
    order = np.lexsort((pos, seg, uid))
    del seg, pos

    def col(name):
        return tbl[name].to_numpy(zero_copy_only=False)[order]

    raw = {n: col(n) for n in list(spec.v1_numeric) + list(spec.v1_flags) + list(spec.v1_uctx) + list(spec.v1_masks)}
    tok_raw = {n: col(n).astype(np.int64) for n in ("product_idx", "category_idx", "brand_idx", "main_category_idx",
                                                    "category_code_idx", "daypart_idx", "local_weekday_idx")}
    did = col("decision_id")
    # alignment with the store (every row's tokens; every decision's id)
    for tname, attr, rname in TOKENS:
        if not np.array_equal(getattr(cs, attr).astype(np.int64), tok_raw[rname]):
            raise AssertionError(f"bucket {b}: {tname} misaligned with the raw read")
    if not np.array_equal(did[rows], ds.decision_id):
        raise AssertionError(f"bucket {b}: decision ids misaligned")
    acc.verify["alignment_checks"] += 1
    uid_s = uid[order]
    del uid
    newu = np.empty(nrow, dtype=bool)
    newu[0] = True
    newu[1:] = uid_s[1:] != uid_s[:-1]
    del uid_s
    ugrp = np.cumsum(newu) - 1
    nu = int(ugrp[-1]) + 1
    dgrp = ugrp[rows]
    n_user = np.bincount(dgrp, minlength=nu)
    o = np.argsort(dgrp, kind="stable")
    gs, cls_ = dgrp[o], ds.client_idx[o]
    same = gs[1:] == gs[:-1]
    if np.any(cls_[1:][same] != cls_[:-1][same]) or len(np.unique(ds.client_idx)) != int(np.count_nonzero(n_user)):
        raise AssertionError(f"bucket {b}: client_idx is not one-to-one with the user groups")
    strat = F.stratum_of(n_user)
    s = acc.strata
    s["train_users"] += nu
    s["train_rows"] += nrow
    s["A1_users"] += int(np.count_nonzero(strat == 0))
    for code, key in ((1, "B"), (2, "pool"), (3, "D")):
        s[f"{key}_users"] += int(np.count_nonzero(strat == code))
        s[f"{key}_examples"] += int(n_user[strat == code].sum())
    s["nge1_users"] += int(np.count_nonzero(n_user >= 1))
    s["nge1_examples"] += int(n_user.sum())
    uv, uc = np.unique(n_user, return_counts=True)
    for v, c in zip(uv.tolist(), uc.tolist()):
        acc.nfreq[v] = acc.nfreq.get(v, 0) + c
    pool_user = strat == 2
    nge1_user = n_user >= 1
    row_pool = pool_user[ugrp]
    row_nge1 = nge1_user[ugrp]
    is_dec = np.zeros(nrow, dtype=bool)
    is_dec[rows] = True
    dec_pool = row_pool[rows]
    cell = np.full(nrow, 4, dtype=np.int8)
    cell[row_nge1 & ~row_pool & ~is_dec] = 3
    cell[row_nge1 & ~row_pool & is_dec] = 2
    cell[row_pool & ~is_dec] = 1
    cell[row_pool & is_dec] = 0
    if np.any(is_dec & ~row_nge1):
        raise AssertionError("a supervised decision belongs to an n=0 user")
    del ugrp, newu

    # ---------------- Clean V2 per decision row via the library (pseudo-batches of 1-event windows)
    v2 = {"event_numeric_features": [], "event_quality_flags": [], "user_context": [], "user_context_masks": []}
    for sl in F.iter_chunks(ndec, 131072):
        r = rows[sl]
        n = r.shape[0]
        batch = {"event_numeric_features": cs._event_numeric_features[r][:, None, :],
                 "event_quality_flags": cs._event_quality_flags[r][:, None, :],
                 "user_context": cs._user_context[r],
                 "user_context_masks": cs._user_context_masks[r].astype(np.float32),
                 "attention_mask": np.ones((n, 1), dtype=bool), "last_valid_index": np.zeros(n, dtype=np.int64),
                 "feature_view": V1_VIEW_TAG, "v1_input_schema_sha256": store.runtime_input_schema_sha256}
        out = clean_v2_view(batch, spec, encoded_with="V1_NORMALIZERS", state_sha256=store.state_sha256)
        if (tuple(out["event_numeric_feature_names"]) != spec.v2_numeric
                or tuple(out["user_context_names"]) != spec.v2_uctx):
            acc.verify["names_ok"] = False
        for k in v2:
            a = np.asarray(out[k])
            v2[k].append(a[:, 0, :] if a.ndim == 3 else a)
        for k2, v in out["clean_v2_rc_counts"].items():
            acc.verify["rc_counts_view"][k2] = acc.verify["rc_counts_view"].get(k2, 0) + int(v)
    v2 = {k: np.concatenate(v) for k, v in v2.items()}
    v2n, v2f, v2u, v2m = (v2["event_numeric_features"], v2["event_quality_flags"], v2["user_context"],
                          v2["user_context_masks"])

    # ---------------- reference encoder on every decision row (formula of record)
    raw_dec = {k: v[rows] for k, v in raw.items()}
    for sl in F.iter_chunks(ndec, 131072):
        ref = encode_rows({k: v[sl] for k, v in raw_dec.items()}, spec)
        for k, mine in (("event_numeric_features", v2n), ("event_quality_flags", v2f), ("user_context", v2u),
                        ("user_context_masks", v2m)):
            d = float(np.max(np.abs(ref[k].astype(np.float64) - mine[sl].astype(np.float64)), initial=0.0))
            acc.verify["encode_rows_max_abs_diff"][k] = max(acc.verify["encode_rows_max_abs_diff"][k], d)
    del raw_dec

    # ---------------- real CleanV2Store.batch windows vs the per-row values (deterministic sample)
    k = min(ctx["verify_per_bucket"], ndec)
    if k > 0:
        vpos = np.unique(np.linspace(0, ndec - 1, k).round().astype(np.int64))
        bb = store.batch(ds, vpos)
        g = {kk: (vv.numpy() if hasattr(vv, "numpy") else np.asarray(vv)) for kk, vv in bb.items()
             if kk in ("event_numeric_features", "event_quality_flags", "user_context", "user_context_masks",
                       "lengths", "item_tokens", "category_tokens", "brand_tokens", "main_category_tokens",
                       "daypart_tokens", "weekday_tokens")}
        num_src, flag_src = list(spec.num_src), list(spec.flag_src)
        for i, p in enumerate(vpos.tolist()):
            ln = int(g["lengths"][i])
            er = int(rows[p])
            win = np.arange(er - ln + 1, er + 1)
            bad = (not np.array_equal(g["event_numeric_features"][i, :ln], cs._event_numeric_features[win][:, num_src])
                   or not np.array_equal(g["event_quality_flags"][i, :ln], cs._event_quality_flags[win][:, flag_src])
                   or not np.array_equal(g["user_context"][i], v2u[p])
                   or not np.array_equal(g["user_context_masks"][i], v2m[p])
                   or not np.array_equal(g["event_numeric_features"][i, ln - 1], v2n[p])
                   or not np.array_equal(g["event_quality_flags"][i, ln - 1], v2f[p]))
            for tname, attr, _ in TOKENS:
                bad = bad or not np.array_equal(g[tname][i, :ln], getattr(cs, attr)[win].astype(np.int64))
            acc.verify["store_batch_mismatches"] += int(bad)
            acc.verify["pad_nonzero"] += int(np.count_nonzero(g["event_numeric_features"][i, ln:])
                                             + np.count_nonzero(g["event_quality_flags"][i, ln:])
                                             + np.count_nonzero(g["item_tokens"][i, ln:]))
        acc.verify["store_batch_windows_checked"] += len(vpos)
        del bb, g

    # ---------------- decision-grain column values -> local codes
    colvals = {}
    for j, n in enumerate(spec.v2_numeric):
        colvals[n] = v2n[:, j]
    for j, n in enumerate(spec.v2_flags):
        colvals[n] = v2f[:, j]
    for j, n in enumerate(spec.v2_uctx):
        colvals[n] = v2u[:, j]
    for j, n in enumerate(spec.v2_masks):
        colvals[n] = v2m[:, j]
    v1n_dec = cs._event_numeric_features[rows]
    v1m_dec = cs._user_context_masks[rows]
    for n, j in DROPPED_NUM.items():
        colvals[n] = v1n_dec[:, j]
    for n, j in DROPPED_MASK.items():
        colvals[n] = v1m_dec[:, j].astype(np.float32)
    for tname, attr, _ in TOKENS:
        colvals[tname] = getattr(cs, attr)[rows]
    colvals[AUX] = tok_raw["category_code_idx"][rows]
    for r_ in acc.reg:
        u, c = F.local_codes(np.ascontiguousarray(colvals[r_["name"]]))
        acc.dicts[r_["name"]].append(u)
        acc.codes[r_["name"]].append(c)
    acc.pool.append(dec_pool.copy())
    acc.n_dec.append(ndec)
    del colvals, v1n_dec, v1m_dec

    # ---------------- identity ledger over all TRAIN rows (raw v1 values; contract RC1-RC7 + extra relations)
    f64 = {n: raw[n].astype(np.float64) for n in ("log1p_prefix_elapsed_s", "log1p_segment_elapsed_s",
                                                   "prefix_item_count", "prefix_representative_count",
                                                   "local_hour_sin", "local_hour_cos", "utc_hour_sin",
                                                   "utc_hour_cos", "log1p_prev_view_count", "recent_history_length",
                                                   "current_known_category_run_length")}
    rot = np.maximum(np.abs(f64["local_hour_sin"] - (f64["utc_hour_sin"] + f64["utc_hour_cos"]) / SQRT2),
                     np.abs(f64["local_hour_cos"] - (f64["utc_hour_cos"] - f64["utc_hour_sin"]) / SQRT2))
    hh = raw["has_history"].astype(np.float64)
    viol = {
        "RC1_prefix_elapsed_eq_segment_elapsed": np.abs(f64["log1p_prefix_elapsed_s"] - f64["log1p_segment_elapsed_s"]) > 1e-6,
        "RC2_prefix_item_count_eq_rep_plus_1": np.abs(f64["prefix_item_count"] - (f64["prefix_representative_count"] + 1.0)) > 1e-6,
        "RC3_local_hour_rotation_le_1e-6": rot > 1e-6,
        "RC4_has_history_mask_eq_has_history": raw["has_history_mask"].astype(np.float64) != hh,
        "RC5_prev_user_event_available_eq_has_history": raw["prev_user_event_available"].astype(np.float64) != hh,
        "RC6_prev_session_implies_prev_user_event": (raw["prev_session_available"] > 0) & ~(raw["prev_user_event_available"] > 0),
        "RC7_shares_zero_when_unsupported": ((raw["recent_category_match_supported"] == 0) & (raw["recent_category_match_share"] != 0))
                                            | ((raw["recent_brand_match_supported"] == 0) & (raw["recent_brand_match_share"] != 0)),
        "E1_price_zero_eq_price_missing": raw["price_zero"] != raw["price_missing"],
        "E2_main_category_missing_eq_category_code_missing": raw["main_category_missing"] != raw["category_code_missing"],
        "E3_brand_missing_eq_brand_token_MISSING": raw["brand_missing"] != (tok_raw["brand_idx"] == 1),
        "E4_main_category_missing_eq_main_category_token_MISSING": raw["main_category_missing"] != (tok_raw["main_category_idx"] == 1),
        "E5_category_missing_eq_category_token_MISSING": raw["category_missing"] != (tok_raw["category_idx"] == 1),
        "E6_has_history_eq_prev_view_count_gt0": (hh > 0) != (f64["log1p_prev_view_count"] > 0),
        "E7_recent_history_length_eq_min_expm1_prev_view_count_100": np.abs(f64["recent_history_length"] - np.minimum(np.rint(np.expm1(f64["log1p_prev_view_count"])), 100.0)) > 0.5,
        "E8_run_length_zero_iff_main_category_missing": (f64["current_known_category_run_length"] == 0) != raw["main_category_missing"],
        "E9_prev_gap_unavailable_eq_not_has_history": raw["prev_gap_unavailable"] != (hh == 0),
        "E10_recent_price_supported_eq_has_history": raw["recent_price_supported"].astype(np.float64) != hh,
        "E11_category_code_missing_eq_category_code_token_MISSING": raw["category_code_missing"] != (tok_raw["category_code_idx"] == 1),
        "E12_prev_session_available_implies_has_history": (raw["prev_session_available"] > 0) & (hh == 0),
    }
    for f in DEAD_FLAGS:
        viol[f"DEAD_{f}_nonzero"] = raw[f].astype(bool)
    groups = {"ALL_TRAIN_ROWS": np.ones(nrow, dtype=bool), "POOL_TRAIN_ROWS": row_pool,
              "NGE1_DECISIONS": is_dec, "POOL_DECISIONS": is_dec & row_pool}
    for kk, vv in viol.items():
        for gname, gm in groups.items():
            acc.add_count(acc.ledger, kk, gname, np.count_nonzero(vv & gm))
    for gname, gm in groups.items():
        acc.ledger_max_rot[gname] = max(acc.ledger_max_rot.get(gname, 0.0), float(rot[gm].max(initial=0.0)))
    del f64, rot, viol

    # ---------------- raw-sentinel missingness at decision rows (gate vs raw == 0)
    pm = raw["price_missing"] | raw["price_conflict"]
    gates_raw = {
        "log1p_price": pm, "log1p_prev_raw_view_gap_s": raw["prev_gap_unavailable"].astype(bool),
        "log1p_item_observed_age_s": raw["item_age_unavailable"].astype(bool),
        "recent_mean_log_price": raw["recent_price_supported"] == 0,
        "current_minus_recent_mean_log_price": (raw["recent_price_supported"] == 0) | pm,
        "log1p_time_since_prev_user_event_s": raw["prev_user_event_available"] == 0,
        "log1p_time_since_prev_session_last_event_s": raw["prev_session_available"] == 0,
        "recent_category_match_share": raw["recent_category_match_supported"] == 0,
        "recent_brand_match_share": raw["recent_brand_match_supported"] == 0,
    }
    for gname, gm in (("FULL", is_dec), ("POOL", is_dec & row_pool)):
        for n in list(spec.v1_numeric) + list(spec.v1_uctx):
            acc.add_count(acc.raw_zero, n, gname, np.count_nonzero((raw[n] == 0) & gm))
        for n, gt in gates_raw.items():
            z = raw[n] == 0
            for a_, b_ in ((1, 1), (1, 0), (0, 1), (0, 0)):
                m_ = (gt if a_ else ~gt) & (z if b_ else ~z) & gm
                acc.add_count(acc.gate2x2, n, f"{gname}|unknown={a_}|raw0={b_}", np.count_nonzero(m_))

    # ---------------- window-exposure grain (every real position of every supervised window)
    lengths = ds.length.astype(np.int64)
    starts = rows - lengths + 1
    num_all = cs._event_numeric_features[:, list(spec.num_src)].astype(np.float64)
    flg_all = cs._event_quality_flags[:, list(spec.flag_src)].astype(np.float64)
    fi = {n: j for j, n in enumerate(spec.v2_flags)}
    avail_all = {"price_available": (flg_all[:, fi["price_missing"]] == 0) & (flg_all[:, fi["price_conflict"]] == 0),
                 "prev_gap_available": flg_all[:, fi["prev_gap_unavailable"]] == 0,
                 "item_age_available": flg_all[:, fi["item_age_unavailable"]] == 0,
                 "segment_prefix_available": raw["prefix_representative_count"] > 0}
    for gname, sel in (("FULL", np.ones(ndec, dtype=bool)), ("POOL", dec_pool)):
        d = (np.bincount(starts[sel], minlength=nrow + 1) - np.bincount(rows[sel] + 1, minlength=nrow + 1))
        w = np.cumsum(d[:-1]).astype(np.float64)
        if int(w.sum()) != int(lengths[sel].sum()):
            raise AssertionError("window weights do not sum to the window lengths")
        W = acc.win[gname]
        W["S0"] = W.get("S0", 0.0) + float(w.sum())
        W["n_windows"] = W.get("n_windows", 0) + int(sel.sum())
        s1 = w @ num_all
        s2 = w @ (num_all * num_all)
        for j, n in enumerate(spec.v2_numeric):
            dflt = acc.reg[j]["default"]
            e = W.setdefault(n, {"S1": 0.0, "S2": 0.0, "Sdef": 0.0, "Szero": 0.0})
            e["S1"] += float(s1[j])
            e["S2"] += float(s2[j])
            e["Szero"] += float(w @ (num_all[:, j] == 0.0))
            if dflt is not None:
                e["Sdef"] += float(w @ (num_all[:, j] == np.float32(dflt)))
        fsum = w @ flg_all
        for j, n in enumerate(spec.v2_flags):
            e = W.setdefault(n, {"S1": 0.0})
            e["S1"] += float(fsum[j])
        for tname, attr, _ in TOKENS:
            t = getattr(cs, attr)
            e = W.setdefault(tname, {"MISSING": 0.0, "OOV": 0.0, "KNOWN": 0.0})
            e["MISSING"] += float(w @ (t == 1))
            e["OOV"] += float(w @ (t == 2))
            e["KNOWN"] += float(w @ (t >= 3))
        for an, am in avail_all.items():
            W.setdefault("avail", {}).setdefault(an, 0.0)
            W["avail"][an] += float(w @ am)
        acc.win_item[gname] += np.bincount(cs._item_tokens, weights=w, minlength=ITEM_VOCAB)
    del num_all, flg_all, avail_all

    # ---------------- section 9 raw tables (17 normalized kept channels) per cell x validity class
    rps = raw["recent_price_supported"]
    for n in spec.norm_v2:
        x = raw[n]
        if not np.all(np.isfinite(x)):
            raise AssertionError(f"non-finite raw {n}")
        if n == "log1p_price":
            v2ok = ~pm
            v1ok = v2ok
        elif n == "log1p_prev_raw_view_gap_s":
            v2ok = ~raw["prev_gap_unavailable"].astype(bool)
            v1ok = v2ok
        elif n == "log1p_item_observed_age_s":
            v2ok = ~raw["item_age_unavailable"].astype(bool)
            v1ok = v2ok
        elif n == "recent_mean_log_price":
            v2ok = rps > 0
            v1ok = v2ok
        elif n == "current_minus_recent_mean_log_price":
            v2ok = (rps > 0) & ~pm
            v1ok = rps > 0
        elif n == "log1p_time_since_prev_user_event_s":
            v2ok = raw["prev_user_event_available"] > 0
            v1ok = np.ones(nrow, dtype=bool)
        elif n == "log1p_time_since_prev_session_last_event_s":
            v2ok = raw["prev_session_available"] > 0
            v1ok = np.ones(nrow, dtype=bool)
        else:
            v2ok = np.ones(nrow, dtype=bool)
            v1ok = v2ok
        klass = np.where(v2ok, 0, np.where(v1ok, 1, 2)).astype(np.int8)
        key = cell.astype(np.int16) * 3 + klass
        o2 = np.lexsort((x, key))
        xs, ks = x[o2], key[o2]
        st = np.flatnonzero(np.r_[True, (xs[1:] != xs[:-1]) | (ks[1:] != ks[:-1])])
        cnt = np.diff(np.r_[st, nrow])
        uvals, ukeys = xs[st], ks[st]
        for kv in np.unique(ukeys).tolist():
            m_ = ukeys == kv
            acc.raw_tables.setdefault(n, {}).setdefault(int(kv), F.TableAccumulator(8)).add(uvals[m_], cnt[m_])
        del o2, xs, ks

    # ---------------- catalogue support, vocab usage, targets per cell
    for rname, V in (("product_idx", ITEM_VOCAB), ("category_idx", INV.VOCAB_SIZES["category_tokens"]),
                     ("brand_idx", INV.VOCAB_SIZES["brand_tokens"]),
                     ("main_category_idx", INV.VOCAB_SIZES["main_category_tokens"]),
                     ("category_code_idx", ctx["catcode_vocab"])):
        t = tok_raw[rname]
        if t.min() < 0 or t.max() >= V:
            raise AssertionError(f"{rname} out of range")
        bc = np.bincount(cell.astype(np.int64) * V + t, minlength=5 * V).reshape(5, V)
        acc.cell_bincount[rname] = acc.cell_bincount.get(rname, 0) + bc
    acc.targets["FULL"] += np.bincount(ds.target_class, minlength=K)
    acc.targets["POOL"] += np.bincount(ds.target_class[dec_pool], minlength=K)

    # ---------------- pooled first-seen proxy: per cell, per product, min block_ts over TRAIN rows
    pid = col("product_id")
    u, inv = np.unique(pid, return_inverse=True)
    cu = np.fromiter((acc.pdict.setdefault(sv, len(acc.pdict)) for sv in u.tolist()), dtype=np.int64, count=len(u))
    pcode = cu[inv.reshape(-1)]
    del pid, u, inv, cu
    bts = col("block_ts").astype("datetime64[us]").view(np.int64)
    acc.nonwhole_second_ts += int(np.count_nonzero(bts % 1_000_000))
    o3 = np.lexsort((bts, pcode, cell))
    cs_, ps_, ts_ = cell[o3], pcode[o3], bts[o3]
    st = np.flatnonzero(np.r_[True, (cs_[1:] != cs_[:-1]) | (ps_[1:] != ps_[:-1])])
    for cc in range(5):
        m_ = cs_[st] == cc
        acc.fs_parts[cc].append((ps_[st][m_].copy(), ts_[st][m_].copy()))
    del o3, cs_, ps_, ts_
    pr = row_pool
    acc.poolrows["pcode"].append(pcode[pr].astype(np.int32))
    acc.poolrows["bts"].append(bts[pr])
    acc.poolrows["age"].append(raw["log1p_item_observed_age_s"][pr].astype(np.float32))
    acc.poolrows["age_unavail"].append(raw["item_age_unavailable"][pr].astype(bool))
    acc.poolrows["is_dec"].append(is_dec[pr])

    # ---------------- recovery-source exposure
    for sname in SOURCES:
        sv = col(sname)
        svs = np.where(pd_isnull(sv), "<NULL>", sv).astype(str)
        for gname, gm in (("ALL_TRAIN_ROWS", np.ones(nrow, dtype=bool)), ("POOL_TRAIN_ROWS", row_pool),
                          ("NGE1_DECISIONS", is_dec), ("POOL_DECISIONS", is_dec & row_pool)):
            uu, cc = np.unique(svs[gm], return_counts=True)
            for a_, b_ in zip(uu.tolist(), cc.tolist()):
                acc.add_count(acc.sources, f"{sname}={a_}", gname, b_)
    del tbl, raw, tok_raw
    acc.bucket_log.append({"bucket": b, "train_rows": nrow, "decisions": ndec, "pool_decisions": int(dec_pool.sum()),
                           "users": nu, "seconds": round(time.time() - t0, 1), "rss": rss_mib()})
    del store, cs, ds
    release_memory()


def pd_isnull(a):
    return np.array([x is None for x in a], dtype=bool) if a.dtype == object else np.zeros(a.shape, dtype=bool)


# ================================================================================ phase B
def build_codes(acc):
    N = int(sum(acc.n_dec))
    pool = np.concatenate(acc.pool)
    acc.pool = None
    G, C = {}, {}
    for r_ in acc.reg:
        n = r_["name"]
        g = np.unique(np.concatenate(acc.dicts[n]))
        dt = F.compact_uint(len(g))
        arr = np.empty(N, dtype=dt)
        off = 0
        for u, c in zip(acc.dicts[n], acc.codes[n]):
            m = np.searchsorted(g, u)
            if not np.array_equal(g[m], u):
                raise AssertionError(f"{n}: global dictionary lookup mismatch")
            arr[off:off + c.shape[0]] = m.astype(dt)[c]
            off += c.shape[0]
        if off != N:
            raise AssertionError("row count mismatch in code build")
        acc.dicts[n] = None
        acc.codes[n] = None
        G[n], C[n] = g, arr
        release_memory()
    return N, pool, G, C


def column_stats(reg, G, C, pool, N, Np):
    stats, counts = {}, {}
    for r_ in reg:
        n = r_["name"]
        g, c = G[n], C[n]
        cf = np.bincount(c, minlength=len(g)).astype(np.int32)
        cp = np.bincount(c[pool], minlength=len(g)).astype(np.int32)
        counts[n] = (cf, cp)
        out = {}
        for pop, cnt, NN in (("FULL", cf, N), ("POOL", cp, Np)):
            nz = cnt > 0
            vals = g[nz]
            cc = cnt[nz]
            mom = F.table_moments(vals, cc) if r_["kind"] != "token" else None
            imode = int(np.argmax(cc))
            e = {"n": int(NN), "distinct": int(nz.sum()), "min": vals[0].item(), "max": vals[-1].item(),
                 "mode_value": vals[imode].item(), "mode_share": float(cc[imode] / NN)}
            if mom is not None:
                e.update({"mean": mom["mean"], "std": mom["std"], "var": mom["var"]})
                e["quantiles"] = F.quantiles_table(vals, cc)
                e["share_exact_zero"] = float(cc[vals == 0].sum() / NN)
            if r_["default"] is not None:
                e["default_value"] = r_["default"]
                e["default_share"] = float(cc[vals == np.float32(r_["default"])].sum() / NN)
            if r_["kind"] == "binary":
                e["prevalence"] = float(cc[vals == 1].sum() / NN)
            if r_["kind"] == "token":
                e["MISSING_share"] = float(cc[vals == 1].sum() / NN)
                e["OOV_share"] = float(cc[vals == 2].sum() / NN)
                e["KNOWN_share"] = float(cc[vals >= 3].sum() / NN)
                e["distinct_known"] = int(np.count_nonzero(vals >= 3))
                e["top10_share"] = float(np.sort(cc)[::-1][:10].sum() / NN)
            e["rows_in_multirow_value_groups_share"] = float(cc[cc >= 2].sum() / NN)
            out[pop] = e
        stats[n] = out
    return stats, counts


def avail_indicators(getv, z0_prefix):
    return {
        "item_known": getv("item_tokens") >= 3, "category_known": getv("category_tokens") >= 3,
        "brand_known": getv("brand_tokens") >= 3, "main_category_known": getv("main_category_tokens") >= 3,
        "price_available": (getv("price_missing") == 0) & (getv("price_conflict") == 0),
        "prev_gap_available": getv("prev_gap_unavailable") == 0,
        "item_age_available": getv("item_age_unavailable") == 0,
        "brand_flag_available": getv("brand_missing") == 0,
        "category_code_available": getv("category_code_missing") == 0,
        "main_category_flag_available": getv("main_category_missing") == 0,
        "segment_prefix_available": getv("prefix_representative_count") != np.float32(z0_prefix),
        "history_available": getv("has_history") == 1,
        "category_match_supported": getv("recent_category_match_supported") == 1,
        "brand_match_supported": getv("recent_brand_match_supported") == 1,
        "recent_price_supported": getv("recent_price_supported") == 1,
        "f1_price_diff_available": (getv("recent_price_supported") == 1) & (getv("price_missing") == 0)
                                   & (getv("price_conflict") == 0),
        "prev_session_available": getv("prev_session_available") == 1,
    }


CHANNEL_AVAIL = {"log1p_price": "price_available", "log1p_prev_raw_view_gap_s": "prev_gap_available",
                 "log1p_item_observed_age_s": "item_age_available",
                 "recent_category_match_share": "category_match_supported",
                 "recent_brand_match_share": "brand_match_supported", "recent_mean_log_price": "recent_price_supported",
                 "current_minus_recent_mean_log_price": "f1_price_diff_available",
                 "log1p_time_since_prev_user_event_s": "history_available",
                 "log1p_time_since_prev_session_last_event_s": "prev_session_available",
                 "log1p_prev_view_count": "history_available", "log1p_prev_session_count": "history_available",
                 "recent_history_length": "history_available", "recent_distinct_items": "history_available",
                 "recent_distinct_known_main_categories": "history_available",
                 "item_tokens": "item_known", "category_tokens": "category_known", "brand_tokens": "brand_known",
                 "main_category_tokens": "main_category_known"}


def chunk_pass(reg, G, C, counts, pool, N, Np, stats, chunk):
    corr_cols = [r_["name"] for r_ in reg if r_["kind"] != "token"]
    m = len(corr_cols)
    mean_f = np.array([stats[c]["FULL"]["mean"] for c in corr_cols])
    mean_p = np.array([stats[c]["POOL"]["mean"] for c in corr_cols])
    rk_f = {c: F.midranks(counts[c][0]) for c in corr_cols}
    rk_p = {c: F.midranks(counts[c][1]) for c in corr_cols}
    Gf = {c: G[c].astype(np.float64) for c in corr_cols}
    Cf = np.zeros((m, m))
    Cp = np.zeros((m, m))
    Rf = np.zeros((m, m))
    Rp = np.zeros((m, m))
    z0p = next(r_["default"] for r_ in reg if r_["name"] == "prefix_representative_count")
    ind_names = None
    Af = Ap = None
    grp = {g: {"FULL": {"any_available": 0, "any_non_default": 0}, "POOL": {"any_available": 0, "any_non_default": 0}}
           for g in INV.GROUPS}
    regd = {r_["name"]: r_ for r_ in reg}
    for sl in F.iter_chunks(N, chunk):
        pm = pool[sl]
        X = np.empty((sl.stop - sl.start, m))
        R = np.empty_like(X)
        Rq = np.empty((int(pm.sum()), m))
        for j, c in enumerate(corr_cols):
            cc = C[c][sl]
            X[:, j] = Gf[c][cc]
            R[:, j] = rk_f[c][cc]
            Rq[:, j] = rk_p[c][cc[pm]]
        Xp = X[pm] - mean_p
        X -= mean_f
        Cf += X.T @ X
        Cp += Xp.T @ Xp
        R -= (N + 1) / 2.0
        Rq -= (Np + 1) / 2.0
        Rf += R.T @ R
        Rp += Rq.T @ Rq
        del X, Xp, R, Rq

        def getv(name, _sl=sl):
            return G[name][C[name][_sl]]
        ind = avail_indicators(getv, z0p)
        if ind_names is None:
            ind_names = list(ind)
            Af = np.zeros((len(ind_names), len(ind_names)))
            Ap = np.zeros_like(Af)
        A = np.stack([ind[k] for k in ind_names], axis=1).astype(np.float32)
        Aq = A[pm]
        Af += (A.T @ A).astype(np.float64)
        Ap += (Aq.T @ Aq).astype(np.float64)
        for g, mem in INV.GROUPS.items():
            av = np.zeros(sl.stop - sl.start, dtype=bool)
            nd = np.zeros_like(av)
            nd_defined = False
            for mname in mem:
                r_ = regd[mname]
                a_key = CHANNEL_AVAIL.get(mname)
                if a_key is not None:
                    av |= ind[a_key]
                else:
                    av |= True
                v = getv(mname)
                if r_["kind"] == "token":
                    nd |= v >= 3
                    nd_defined = True
                elif r_["default"] is not None:
                    nd |= v != np.float32(r_["default"])
                    nd_defined = True
            grp[g]["FULL"]["any_available"] += int(av.sum())
            grp[g]["POOL"]["any_available"] += int(av[pm].sum())
            if nd_defined:
                grp[g]["FULL"]["any_non_default"] += int(nd.sum())
                grp[g]["POOL"]["any_non_default"] += int(nd[pm].sum())
            else:
                grp[g]["FULL"]["any_non_default"] = None
                grp[g]["POOL"]["any_non_default"] = None
        del A, Aq, ind
    return {"corr_cols": corr_cols, "pearson": {"FULL": F.corr_from_comoment(Cf), "POOL": F.corr_from_comoment(Cp)},
            "spearman": {"FULL": F.corr_from_comoment(Rf), "POOL": F.corr_from_comoment(Rp)},
            "ind_names": ind_names, "ind_counts": {"FULL": np.rint(Af).astype(np.int64),
                                                    "POOL": np.rint(Ap).astype(np.int64)},
            "groups": grp}


def _sort_pair(cx, cy, dy, pool, n, want_pairs):
    """Exact pair counts for a high-cardinality pair by one quicksort of the int64 pair key (thread-safe; numpy
    sorting and gathers release the GIL). Returns {pop: (v_xy, v_yx, pairs|None)}."""
    key = cx.astype(np.int64) * dy + cy
    o = np.argsort(key)
    ks = key[o]
    del key
    flag = np.empty(n, dtype=bool)
    flag[0] = True
    np.not_equal(ks[1:], ks[:-1], out=flag[1:])
    st = np.flatnonzero(flag)
    del flag
    uk = ks[st]
    del ks
    cf = np.diff(np.append(st, n)).astype(np.int64)
    cp = np.add.reduceat(pool[o].astype(np.int32), st).astype(np.int64)
    del o, st
    px, py = uk // dy, uk % dy
    del uk
    out = {}
    for p, cnt in (("FULL", cf), ("POOL", cp)):
        vxy, vyx = F.fd_from_pairs(px, py, cnt)
        out[p] = (vxy, vyx, (px, py, cnt) if want_pairs else None)
    return out


def fd_pass(reg, G, C, counts, pool, N, Np, dense_max, workers=4, log_every=100):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    names = [r_["name"] for r_ in reg]
    D = {n: len(G[n]) for n in names}
    cat = {n for n in names if any(n == t[0] for t in TOKENS) or n == AUX}
    nn = len(names)
    V = {p: np.zeros((nn, nn), dtype=np.int64) for p in ("FULL", "POOL")}
    LAM = {p: np.full((nn, nn), np.nan) for p in ("FULL", "POOL")}
    cramer = {}
    t0 = time.time()
    npairs = 0
    path_n = {"constant": 0, "binary": 0, "dense": 0, "sort": 0}
    path_s = {k: 0.0 for k in path_n}
    maxm = {p: {n: int(counts[n][0 if p == "FULL" else 1].max()) for n in names} for p in ("FULL", "POOL")}
    NN = {"FULL": N, "POOL": Np}

    def record(i, j, res):
        xi, yj = names[i], names[j]
        for p in ("FULL", "POOL"):
            vxy, vyx, pairs = res[p]
            V[p][i, j], V[p][j, i] = vxy, vyx
            lxy = F.gk_lambda(vxy, NN[p], maxm[p][yj])
            lyx = F.gk_lambda(vyx, NN[p], maxm[p][xi])
            LAM[p][i, j] = np.nan if lxy is None else lxy
            LAM[p][j, i] = np.nan if lyx is None else lyx
            if pairs is not None:
                px, py, cnt = pairs
                cramer.setdefault(f"{xi}|{yj}", {})[p] = F.cramers_v(
                    px, py, cnt, counts[xi][0 if p == "FULL" else 1], counts[yj][0 if p == "FULL" else 1])

    deferred = []
    for i in range(nn):
        xi = names[i]
        cxp = C[xi][pool]
        for j in range(i + 1, nn):
            yj = names[j]
            npairs += 1
            t1 = time.time()
            res = {}
            if D[xi] == 1 or D[yj] == 1:
                for p in ("FULL", "POOL"):
                    # a constant column is determined by anything; X constant: X->Y violations = N - max_y n_y
                    vxy = NN[p] - maxm[p][yj] if D[xi] == 1 else 0
                    vyx = NN[p] - maxm[p][xi] if D[yj] == 1 else 0
                    if D[xi] == 1 and D[yj] == 1:
                        vxy = vyx = 0
                    res[p] = (vxy, vyx, None)
                kind = "constant"
            elif D[xi] == 2 or D[yj] == 2:
                cyp = C[yj][pool]
                for p in ("FULL", "POOL"):
                    cxx, cyy = (C[xi], C[yj]) if p == "FULL" else (cxp, cyp)
                    if D[xi] == 2:
                        bc, oc, oname, flip = cxx, cyy, yj, False
                    else:
                        bc, oc, oname, flip = cyy, cxx, xi, True
                    cnt_o = counts[oname][0 if p == "FULL" else 1]
                    v_bo, v_ob, n1 = F.fd_binary_any(bc, oc, D[oname], cnt_o)
                    pairs = None
                    if xi in cat and yj in cat:
                        pairs = _pairs_from_binary(cnt_o - n1, n1)
                    res[p] = (v_ob, v_bo, _swap(pairs)) if flip else (v_bo, v_ob, pairs)
                del cyp
                kind = "binary"
            elif D[xi] * D[yj] <= dense_max:
                cyp = C[yj][pool]
                for p in ("FULL", "POOL"):
                    ci, cj = (C[xi], C[yj]) if p == "FULL" else (cxp, cyp)
                    key = ci.astype(np.int64) * D[yj] + cj
                    t = np.bincount(key, minlength=D[xi] * D[yj]).reshape(D[xi], D[yj])
                    del key
                    vxy, vyx = F.fd_from_dense(t)
                    pairs = None
                    if xi in cat and yj in cat:
                        px, py = np.nonzero(t)
                        pairs = (px, py, t[px, py])
                    res[p] = (vxy, vyx, pairs)
                    del t
                del cyp
                kind = "dense"
            else:
                deferred.append((i, j))
                continue
            record(i, j, res)
            path_n[kind] += 1
            path_s[kind] += time.time() - t1
            if npairs % log_every == 0:
                log(f"fd pairs {npairs} ({time.time() - t0:.0f}s) last {xi}|{yj} deferred={len(deferred)}")
        del cxp
    release_memory()
    rss_now = rss_mib().get("VmRSS", 0.0)
    workers = max(1, min(int(workers), int((14500.0 - rss_now) // 2400.0))) if rss_now else max(1, int(workers))
    log(f"fd: {len(deferred)} high-cardinality pairs by sorting with {workers} threads (RSS {rss_now:.0f} MiB)")
    t2 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(_sort_pair, C[names[i]], C[names[j]], D[names[j]], pool, N,
                          names[i] in cat and names[j] in cat): (i, j) for i, j in deferred}
        done = 0
        for fu in as_completed(futs):
            i, j = futs[fu]
            record(i, j, fu.result())
            done += 1
            if done % 8 == 0:
                release_memory()
            if done % 20 == 0:
                log(f"fd sorted pairs {done}/{len(deferred)} ({time.time() - t2:.0f}s)")
    path_n["sort"] = len(deferred)
    path_s["sort"] = time.time() - t2
    return {"names": names, "D": D, "violations": V, "lambda": LAM, "cramer": cramer, "seconds": time.time() - t0,
            "pairs": npairs, "path_pairs": path_n, "path_seconds": path_s}


def _pairs_from_binary(n0, n1):
    k = np.arange(len(n0))
    px = np.r_[np.zeros(len(k), np.int64), np.ones(len(k), np.int64)]
    py = np.r_[k, k]
    cnt = np.r_[n0, n1]
    m = cnt > 0
    return px[m], py[m], cnt[m]


def _swap(pairs):
    if pairs is None:
        return None
    px, py, cnt = pairs
    o = np.lexsort((px, py))
    return py[o], px[o], cnt[o]


def multi_fd(G, C, pool, N, Np):
    """Violations of multi-column determinants for the relations named in the Clean V2 contract (4.3, 4.8)."""
    def comp(names, p):
        key = np.zeros(N if p == "FULL" else Np, dtype=np.int64)
        for n in names:
            c = C[n] if p == "FULL" else C[n][pool]
            key = key * len(G[n]) + c
        return np.unique(key, return_inverse=True)[1].reshape(-1)

    rel = [(("utc_hour_sin", "utc_hour_cos"), ("local_hour_sin", "local_hour_cos", "daypart_tokens")),
           (("utc_hour_sin", "utc_hour_cos", "utc_weekday_sin", "utc_weekday_cos"),
            ("local_weekday_sin", "local_weekday_cos", "is_weekend", "weekday_tokens", "daypart_tokens")),
           (("log1p_price", "recent_mean_log_price"), ("current_minus_recent_mean_log_price",)),
           (("log1p_price", "recent_mean_log_price", "price_missing", "recent_price_supported"),
            ("current_minus_recent_mean_log_price",)),
           (("item_tokens",), ("category_tokens", "brand_tokens", "main_category_tokens")),
           (("category_tokens",), ("main_category_tokens",))]
    out = []
    for xs, ys in rel:
        for p in ("FULL", "POOL"):
            cx = comp(xs, p)
            for y in ys:
                cy = C[y] if p == "FULL" else C[y][pool]
                key = cx.astype(np.int64) * len(G[y]) + cy
                uk, cnt = np.unique(key, return_counts=True)
                vxy, _ = F.fd_from_pairs(uk // len(G[y]), uk % len(G[y]), cnt)
                out.append({"determinant": list(xs), "dependent": y, "population": p, "violations": int(vxy),
                            "violation_rate": vxy / (N if p == "FULL" else Np)})
    return out


# ================================================================================ provenance (section 9)
def provenance(acc, spec, schema, v1norm, pool_counts, ds_root):
    out = {"normalizers": {}, "notes": []}
    rec = schema["normalized_channels_v2"]
    m1 = schema["m1_reencoding"]["features"]
    f1 = schema["f1_reencoding"]

    def tab(n, cells, classes):
        parts = acc.raw_tables.get(n, {})
        vs, cs_ = [], []
        for ce in cells:
            for kl in classes:
                a = parts.get(ce * 3 + kl)
                if a is not None:
                    v, c = a.result()
                    vs.append(v)
                    cs_.append(c)
        return F.merge_tables(vs, cs_)

    pops = {"FULL_TRAIN": (0, 1, 2, 3, 4), "NGE1_USERS": (0, 1, 2, 3), "POOL_5_100": (0, 1)}
    for n in spec.norm_v2:
        r = rec[n]
        reenc = n in m1 or n == f1["feature"]
        v1pred = (0, 1)                       # class V2_VALID + V1_ONLY = the v1 predicate
        e = {"recorded_v2": {"clip_lo": r["clip_lo"], "clip_hi": r["clip_hi"], "mean": r["mean"], "std": r["std"]},
             "fit_population_rule": r.get("fit_population"), "re_encoded_in_v2": reenc}
        if reenc:
            e["recorded_v1"] = {"clip_lo": v1norm[n]["clip_lo"], "clip_hi": v1norm[n]["clip_hi"],
                                "mean": v1norm[n]["mean"], "std": v1norm[n]["std"]}
        fits = {}
        for pname, cells in pops.items():
            tv, tc = tab(n, cells, (0,))
            t1v, t1c = tab(n, cells, v1pred)
            qlo = F.quantile_linear(t1v, t1c, 0.001)
            qhi = F.quantile_linear(t1v, t1c, 0.999)
            qlo_v = F.quantile_linear(tv, tc, 0.001)
            qhi_v = F.quantile_linear(tv, tc, 0.999)
            mom_rec = F.clipped_moments(tv, tc, r["clip_lo"], r["clip_hi"])
            mom_ref = F.clipped_moments(tv, tc, qlo, qhi) if qlo is not None else None
            mom_ref_valid = F.clipped_moments(tv, tc, qlo_v, qhi_v) if qlo_v is not None else None
            fe = {"n_v2_valid_rows": int(tc.sum()), "n_v1_predicate_rows": int(t1c.sum()),
                  "clip_quantiles_v1_predicate": [qlo, qhi], "clip_quantiles_v2_valid_only": [qlo_v, qhi_v],
                  "moments_at_recorded_clip": mom_rec, "moments_at_refit_clip_same_rule": mom_ref,
                  "moments_at_refit_clip_valid_only": mom_ref_valid}
            if reenc:
                fe["v1_all_rule_moments_at_recorded_v1_clip"] = F.clipped_moments(
                    t1v, t1c, v1norm[n]["clip_lo"], v1norm[n]["clip_hi"])
            fits[pname] = fe
        e["fits"] = fits
        full = fits["FULL_TRAIN"]
        e["reproduction_full_train"] = {
            "clip_lo_abs_diff_vs_recorded": _ad(full["clip_quantiles_v1_predicate"][0], r["clip_lo"]),
            "clip_hi_abs_diff_vs_recorded": _ad(full["clip_quantiles_v1_predicate"][1], r["clip_hi"]),
            "clip_lo_f32_equal": _f32eq(full["clip_quantiles_v1_predicate"][0], r["clip_lo"]),
            "clip_hi_f32_equal": _f32eq(full["clip_quantiles_v1_predicate"][1], r["clip_hi"]),
            "mean_rel_diff_vs_recorded": _rd(full["moments_at_recorded_clip"]["mean"], r["mean"]),
            "std_rel_diff_vs_recorded": _rd(full["moments_at_recorded_clip"]["std"], r["std"]),
        }
        if reenc:
            vv = full["v1_all_rule_moments_at_recorded_v1_clip"]
            e["reproduction_full_train"]["v1_mean_rel_diff"] = _rd(vv["mean"], v1norm[n]["mean"])
            e["reproduction_full_train"]["v1_std_rel_diff"] = _rd(vv["std"], v1norm[n]["std"])
        # the pool-only candidate normalizer: the same rule applied to the pool's own TRAIN rows
        pool = fits["POOL_5_100"]
        if reenc:
            cand = {"rule": "recorded-clip rule of the V2 re-encoding (clip bounds kept), mean/std refit on the pool's "
                            "valid rows", "clip_lo": r["clip_lo"], "clip_hi": r["clip_hi"],
                    "mean": pool["moments_at_recorded_clip"]["mean"], "std": pool["moments_at_recorded_clip"]["std"]}
            alt = {"rule": "clip bounds refit as the 0.001/0.999 quantiles of the pool's v1-predicate rows",
                   "clip_lo": pool["clip_quantiles_v1_predicate"][0], "clip_hi": pool["clip_quantiles_v1_predicate"][1],
                   "mean": (pool["moments_at_refit_clip_same_rule"] or {}).get("mean"),
                   "std": (pool["moments_at_refit_clip_same_rule"] or {}).get("std")}
        else:
            cand = {"rule": "v1 rule applied to the pool: 0.001/0.999 quantile clip + mean/std on the pool's valid rows",
                    "clip_lo": pool["clip_quantiles_v1_predicate"][0], "clip_hi": pool["clip_quantiles_v1_predicate"][1],
                    "mean": pool["moments_at_refit_clip_same_rule"]["mean"],
                    "std": pool["moments_at_refit_clip_same_rule"]["std"]}
            alt = {"rule": "recorded full-TRAIN clip bounds kept, mean/std refit on the pool",
                   "clip_lo": r["clip_lo"], "clip_hi": r["clip_hi"], "mean": pool["moments_at_recorded_clip"]["mean"],
                   "std": pool["moments_at_recorded_clip"]["std"]}
        e["pool_only_candidate"] = cand
        e["pool_only_alternative"] = alt
        e["diff_pool_minus_full_own_units"] = {k: _d(cand[k], r[k]) for k in ("clip_lo", "clip_hi", "mean", "std")}
        e["diff_pool_minus_full_relative"] = {k: _rd(cand[k], r[k]) for k in ("mean", "std")}
        # effect on transformed values
        eff = {}
        for grain, cells in (("POOL_SUPERVISED_EXAMPLES", (0,)), ("POOL_TRAIN_ROWS", (0, 1))):
            tv, tc = tab(n, cells, (0,))
            ninv = int(tab(n, cells, (1, 2))[1].sum())
            for vname, par in (("candidate", cand), ("alternative", alt)):
                if par["mean"] is None:
                    continue
                zf = (np.clip(tv.astype(np.float64), r["clip_lo"], r["clip_hi"]) - r["mean"]) / r["std"]
                zp = (np.clip(tv.astype(np.float64), par["clip_lo"], par["clip_hi"]) - par["mean"]) / par["std"]
                dd = np.abs(zp - zf)
                d32 = np.abs(zp.astype(np.float32).astype(np.float64) - zf.astype(np.float32).astype(np.float64))
                w = tc.astype(np.float64)
                eff[f"{grain}|{vname}"] = {
                    "valid_rows": int(tc.sum()), "rezeroed_rows_unchanged": ninv,
                    "max_abs_change_z": float(dd.max(initial=0.0)),
                    "mean_abs_change_z_valid_rows": float(np.dot(w, dd) / w.sum()) if w.sum() else None,
                    "mean_abs_change_z_all_rows": float(np.dot(w, dd) / (w.sum() + ninv)) if w.sum() else None,
                    "max_abs_change_float32": float(d32.max(initial=0.0)),
                    "rows_changed_float32_share": float(w[d32 > 0].sum() / w.sum()) if w.sum() else None}
        e["effect_on_transformed_values"] = eff
        lo_, hi_, mu_, sd_ = r["clip_lo"], r["clip_hi"], r["mean"], r["std"]
        if n not in GATED_V2:
            e["encoding_of_raw_zero"] = {"full": (min(max(0.0, lo_), hi_) - mu_) / sd_,
                                         "pool_candidate": (min(max(0.0, cand["clip_lo"]), cand["clip_hi"])
                                                            - cand["mean"]) / cand["std"]}
        out["normalizers"][n] = e
    return out


def _ad(a, b):
    return None if a is None or b is None else abs(float(a) - float(b))


def _d(a, b):
    return None if a is None or b is None else float(a) - float(b)


def _rd(a, b):
    return None if a is None or b is None or b == 0 else (float(a) - float(b)) / abs(float(b))


def _f32eq(a, b):
    return None if a is None else bool(np.float32(a) == np.float32(b))


def catalogue_vocab_firstseen(acc, ds_root, schema, v1norm, prov):
    out = {}
    cat = pq.read_table(Path(ds_root) / "mappings" / "catalogue.parquet",
                        columns=["product_idx", "representative_views", "output_class"])
    pidx = cat["product_idx"].to_numpy()
    rv = cat["representative_views"].to_numpy().astype(np.int64)
    oc = cat["output_class"].to_numpy()
    if not np.array_equal(pidx, oc + 3):
        raise AssertionError("catalogue product_idx != output_class + 3")
    bc = acc.cell_bincount["product_idx"]
    sup = {"FULL_TRAIN": bc.sum(0), "NGE1_USERS": bc[:4].sum(0), "POOL_5_100": bc[:2].sum(0)}
    mism = int(np.count_nonzero(sup["FULL_TRAIN"][pidx] != rv))
    inB = sup["POOL_5_100"][pidx] >= 5
    inN = sup["NGE1_USERS"][pidx] >= 5
    tgt_p = acc.targets["POOL"]
    tgt_f = acc.targets["FULL"]
    pool_dec_items = bc[0]
    Np = int(tgt_p.sum())
    wpool = acc.win_item["POOL"]
    notB_idx = pidx[~inB]
    head = int((~inB).sum())
    out["catalogue"] = {
        "support_definition": "TRAIN feature rows (= ledger TRAIN rows = representative views) per product_idx",
        "support_full_train_equals_catalogue_representative_views_mismatches": mism,
        "K_A_full_train_support_ge5": int(len(pidx)),
        "K_B_pool_only_support_ge5": int(inB.sum()), "K_nge1_only_support_ge5": int(inN.sum()),
        "A_minus_B": head, "B_minus_A": 0,
        "B_minus_A_basis": "DERIVED: support over a user subset is <= the full support, so pool support >= 5 implies "
                           "full support >= 5 (B is a subset of A)",
        "catalogue_items_with_zero_pool_support": int(np.count_nonzero(sup["POOL_5_100"][pidx] == 0)),
        "pool_supervised_examples": Np,
        "pool_examples_current_item_not_in_B_share": float(pool_dec_items[notB_idx].sum() / Np),
        "pool_window_positions_item_not_in_B_share": float(wpool[notB_idx].sum() / wpool.sum()),
        "pool_targets_class_not_in_B": int(tgt_p[~inB].sum()),
        "pool_targets_class_not_in_B_share": float(tgt_p[~inB].sum() / Np),
        "classes_with_ge1_pool_positive": int(np.count_nonzero(tgt_p >= 1)),
        "classes_with_ge5_pool_positives": int(np.count_nonzero(tgt_p >= 5)),
        "classes_with_zero_pool_positives": int(np.count_nonzero(tgt_p == 0)),
        "classes_with_ge1_full_positive": int(np.count_nonzero(tgt_f >= 1)),
        "head_param_change_if_B": {"gru_tied_rows_x128_plus_bias": -head * (128 + 1),
                                   "sasrec_item_embed_plus_output_embed_plus_bias": -head * (256 + 256 + 1)},
        "note": "effects are on the pool's own TRAIN rows; the circular effect on n (targets turning OOV changes "
                "eligibility) is CENSUS-02's CATALOGUE_OPTION_B",
    }
    voc = {}
    for rname, key in (("category_idx", "category"), ("brand_idx", "brand"), ("main_category_idx", "main_category"),
                       ("category_code_idx", "category_code")):
        b = acc.cell_bincount[rname]
        allc, poolc, ngc = b.sum(0), b[:2].sum(0), b[:4].sum(0)
        V = len(allc)
        known = np.arange(V) >= 3
        voc[key] = {"vocab_rows_known": int(known.sum()), "known_seen_in_full_train_rows": int(np.count_nonzero(allc[known])),
                    "known_seen_in_nge1_rows": int(np.count_nonzero(ngc[known])),
                    "known_seen_in_pool_rows": int(np.count_nonzero(poolc[known])),
                    "known_unseen_in_pool_rows": int(np.count_nonzero((allc > 0) & (poolc == 0) & known)),
                    "pool_rows_MISSING_share": float(poolc[1] / poolc.sum()), "pool_rows_OOV_share": float(poolc[2] / poolc.sum()),
                    "pool_train_rows_token_change_under_pool_vocab": 0,
                    "basis": "DERIVED: every value on a pool TRAIN row is by definition seen in the pool, so a pool-only "
                             "vocabulary changes no pool TRAIN token (only the dense index numbering); VALIDATION "
                             "effects are not measured here (VALIDATION is not read by FEATSTAT-01)"}
    out["vocabularies"] = voc
    # first-seen proxy
    G = len(acc.pdict)
    BIG = np.iinfo(np.int64).max
    fs = np.full((5, G), BIG, dtype=np.int64)
    for cc in range(5):
        for codes, mins in acc.fs_parts[cc]:
            np.minimum.at(fs[cc], codes, mins)
    fs_full = fs.min(0)
    fs_pool = fs[:2].min(0)
    pc_ = np.concatenate(acc.poolrows["pcode"]).astype(np.int64)
    bt = np.concatenate(acc.poolrows["bts"])
    age = np.concatenate(acc.poolrows["age"])
    un = np.concatenate(acc.poolrows["age_unavail"])
    isd = np.concatenate(acc.poolrows["is_dec"])
    acc.poolrows = None
    ff = fs_full[pc_]
    fp = fs_pool[pc_]
    av_full = ff < bt
    av_pool = fp < bt
    sec_full = bt // 1_000_000 - ff // 1_000_000
    sec_pool = bt // 1_000_000 - np.where(fp == BIG, 0, fp) // 1_000_000
    age_full_proxy = np.where(av_full, np.log1p(np.maximum(sec_full, 0).astype(np.float64)), 0.0).astype(np.float32)
    age_pool = np.where(av_pool, np.log1p(np.maximum(sec_pool, 0).astype(np.float64)), 0.0).astype(np.float32)
    agree_avail = av_full == ~un
    agree_val = (age_full_proxy == age) | ~av_full
    rp = prov["normalizers"]["log1p_item_observed_age_s"]
    r = rp["recorded_v2"]
    cand = rp["pool_only_candidate"]
    # fully pool-only pipeline: pool first-seen ages on all pool TRAIN rows, then the v1 rule refit on them
    va, vc = F.value_counts(age_pool[av_pool])
    qlo, qhi = F.quantile_linear(va, vc, 0.001), F.quantile_linear(va, vc, 0.999)
    mom = F.clipped_moments(va, vc, qlo, qhi)
    full_pool = {"clip_lo": qlo, "clip_hi": qhi, "mean": mom["mean"], "std": mom["std"]}

    def z(a, par, avail):
        return np.where(avail, (np.clip(a.astype(np.float64), par["clip_lo"], par["clip_hi"]) - par["mean"]) / par["std"], 0.0)

    z_cur = z(age, r, ~un)
    variants = {"normalizer_only_pool": z(age, cand, ~un), "fully_pool_first_seen_and_normalizer": z(age_pool, full_pool, av_pool),
                "pool_first_seen_full_normalizer": z(age_pool, r, av_pool)}
    eff = {}
    for vn, zz in variants.items():
        dd = np.abs(zz - z_cur)
        for grain, m_ in (("POOL_SUPERVISED_EXAMPLES", isd), ("POOL_TRAIN_ROWS", np.ones_like(isd))):
            eff[f"{grain}|{vn}"] = {"rows": int(m_.sum()), "max_abs_change_z": float(dd[m_].max(initial=0.0)),
                                    "mean_abs_change_z": float(dd[m_].mean()) if m_.any() else None,
                                    "rows_changed_share": float(np.count_nonzero(dd[m_] > 0) / max(1, m_.sum()))}
    flips = {}
    for grain, m_ in (("POOL_SUPERVISED_EXAMPLES", isd), ("POOL_TRAIN_ROWS", np.ones_like(isd))):
        flips[grain] = {"available_full_but_not_pool": int(np.count_nonzero(av_full & ~av_pool & m_)),
                        "available_pool_but_not_full": int(np.count_nonzero(~av_full & av_pool & m_)),
                        "rows": int(m_.sum()),
                        "available_full_share": float(av_full[m_].mean()), "available_pool_share": float(av_pool[m_].mean()),
                        "raw_log_age_mean_abs_change_both_available": float(np.abs(age_pool - age_full_proxy)[m_ & av_full & av_pool].mean())
                        if np.any(m_ & av_full & av_pool) else None,
                        "raw_log_age_max_abs_change_both_available": float(np.abs(age_pool - age_full_proxy)[m_ & av_full & av_pool].max(initial=0.0))}
    out["first_seen_item_age"] = {
        "definition": "first_seen_train = MIN(event_time_utc) per product over pooled non-barrier TRAIN raw views "
                      "(pipeline4.py:208-217); PROXY here = MIN(block_ts) per product over TRAIN feature rows of the "
                      "population (the raw view table is not read); validated against the stored feature on the pool rows",
        "products_seen_in_train_rows": G, "block_ts_not_whole_second_rows": acc.nonwhole_second_ts,
        "proxy_validation_on_pool_train_rows": {"rows": int(len(age)),
                                                "availability_agreement_share": float(agree_avail.mean()),
                                                "value_exact_float32_agreement_share_when_available": float(agree_val[av_full].mean()) if av_full.any() else None,
                                                "max_abs_value_diff_when_both_available": float(np.abs(age_full_proxy - age)[av_full & ~un].max(initial=0.0))},
        "availability_flips": flips, "pool_only_first_seen_normalizer_v1_rule": full_pool, "effect_vs_current": eff}
    return out


# ================================================================================ derived outputs
def summarize(reg, stats, cp, fd, N, Np, acc):
    names = cp["corr_cols"]
    res = {"clusters": {}, "status": {}, "transfer": {}}
    thr = {"pearson_ge_0.95": ("pearson", 0.95), "pearson_ge_0.99": ("pearson", 0.99),
           "spearman_ge_0.95": ("spearman", 0.95), "spearman_ge_0.99": ("spearman", 0.99)}
    cont = [r_["name"] for r_ in reg if r_["kind"] in ("cyclic", "normalized", "normalized_gated", "share")
            or r_["name"] in ("is_weekend", "has_history")]
    binr = [r_["name"] for r_ in reg if r_["kind"] == "binary"]
    for pop in ("FULL", "POOL"):
        for tname, (kind, t) in thr.items():
            idx = [names.index(c) for c in cont]
            M = cp[kind][pop][np.ix_(idx, idx)]
            res["clusters"].setdefault(pop, {})[tname] = F.clusters(cont, M, t)
        idx = [names.index(c) for c in binr]
        M = cp["pearson"][pop][np.ix_(idx, idx)]
        res["clusters"][pop]["phi_ge_0.95_binary"] = F.clusters(binr, M, 0.95)
        kept = [c for c in cont if not any(c == d for d in list(DROPPED_NUM) + list(DROPPED_MASK))]
        idx = [names.index(c) for c in kept]
        res["clusters"][pop]["pearson_ge_0.95_kept_only"] = F.clusters(kept, cp["pearson"][pop][np.ix_(idx, idx)], 0.95)
        res["clusters"][pop]["spearman_ge_0.95_kept_only"] = F.clusters(kept, cp["spearman"][pop][np.ix_(idx, idx)], 0.95)
    fdn = fd["names"] if fd else []
    dropped = set(DROPPED_NUM) | set(DROPPED_MASK)
    for r_ in reg:
        n = r_["name"]
        st = {}
        for pop in ("FULL", "POOL"):
            s = stats[n][pop]
            e = {"constant": s["distinct"] == 1, "near_constant_99": s["mode_share"] >= 0.99,
                 "near_constant_999": s["mode_share"] >= 0.999}
            cl = []
            for tname in ("pearson_ge_0.95", "spearman_ge_0.95", "phi_ge_0.95_binary"):
                for c in res["clusters"][pop][tname]:
                    if n in c["members"]:
                        cl.append({"rule": tname, "with": [m for m in c["members"] if m != n]})
            e["clustered_0.95"] = cl
            if fd:
                j = fdn.index(n)
                V = fd["violations"][pop]
                det_all, det_kept, bij = [], [], []
                for i, xn in enumerate(fdn):
                    if i == j or fd["D"][xn] == 1:
                        continue
                    if V[i, j] == 0:
                        informative = stats[xn][pop]["rows_in_multirow_value_groups_share"] >= 0.5
                        det_all.append({"by": xn, "informative": informative})
                        if xn not in dropped and xn != AUX:
                            det_kept.append(xn)
                        if V[j, i] == 0:
                            bij.append(xn)
                e["exactly_determined_by"] = det_all
                e["exactly_determined_by_kept_channel"] = det_kept
                e["exact_one_to_one_with"] = bij
                lam = fd["lambda"][pop][:, j]
                best = int(np.nanargmax(np.where(np.arange(len(fdn)) == j, np.nan, lam))) if np.any(np.isfinite(np.delete(lam, j))) else None
                e["best_single_predictor"] = None if best is None else {"by": fdn[best], "lambda": float(lam[best]),
                                                                        "violations": int(V[best, j])}
            st[pop] = e
        res["status"][n] = st
        a, bpop = stats[n]["FULL"], stats[n]["POOL"]
        tr = {}
        for key in ("mean", "std", "mode_share", "default_share", "prevalence", "share_exact_zero", "distinct",
                    "MISSING_share", "OOV_share", "KNOWN_share"):
            if key in a and a[key] is not None and bpop.get(key) is not None:
                tr[key] = {"pool": bpop[key], "full": a[key], "pool_minus_full": bpop[key] - a[key]}
        if a.get("std"):
            tr["mean_shift_in_full_sd"] = (bpop["mean"] - a["mean"]) / a["std"]
        sf, sp_ = st["FULL"], st["POOL"]
        changes = []
        for key in ("constant", "near_constant_99", "near_constant_999"):
            if sf[key] != sp_[key]:
                changes.append(key)
        if {tuple(x["with"]) for x in sf["clustered_0.95"]} != {tuple(x["with"]) for x in sp_["clustered_0.95"]}:
            changes.append("cluster_membership_0.95")
        if fd and set(sf["exactly_determined_by_kept_channel"]) != set(sp_["exactly_determined_by_kept_channel"]):
            changes.append("exact_determinants")
        tr["status_changes_pool_vs_full"] = changes
        res["transfer"][n] = tr
    # correlation transfer
    dP = np.abs(cp["pearson"]["POOL"] - cp["pearson"]["FULL"])
    dS = np.abs(cp["spearman"]["POOL"] - cp["spearman"]["FULL"])
    k = np.unravel_index(np.nanargmax(dP), dP.shape)
    k2 = np.unravel_index(np.nanargmax(dS), dS.shape)
    res["correlation_transfer"] = {"max_abs_pearson_diff": float(np.nanmax(dP)), "at": [names[k[0]], names[k[1]]],
                                   "max_abs_spearman_diff": float(np.nanmax(dS)), "at_spearman": [names[k2[0]], names[k2[1]]],
                                   "pairs_abs_pearson_diff_gt_0.05": int(np.nansum(np.triu(dP, 1) > 0.05)),
                                   "pairs_abs_spearman_diff_gt_0.05": int(np.nansum(np.triu(dS, 1) > 0.05))}
    if fd:
        VF, VP = fd["violations"]["FULL"], fd["violations"]["POOL"]
        diff = []
        for i, xn in enumerate(fdn):
            for j, yn in enumerate(fdn):
                if i != j and ((VF[i, j] == 0) != (VP[i, j] == 0)):
                    diff.append({"X": xn, "Y": yn, "violations_full": int(VF[i, j]), "violations_pool": int(VP[i, j])})
        res["fd_transfer"] = {"ordered_pairs_exact_status_differs": len(diff), "pairs": diff[:400]}
    return res


def matrix_json(names, M):
    return {"names": names, "matrix": F.jsafe(np.round(M, 6))}


# ================================================================================ main
def parse_buckets(s):
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--buckets", default="0-63")
    ap.add_argument("--state", required=True, help="TRANSFORM_STATE_CLEAN_V2.json (server copy)")
    ap.add_argument("--verify-per-bucket", type=int, default=512)
    ap.add_argument("--strict-counts", type=int, default=1)
    ap.add_argument("--skip-fd", type=int, default=0)
    ap.add_argument("--dense-max", type=int, default=1 << 24)
    ap.add_argument("--chunk", type=int, default=400000)
    ap.add_argument("--hash-inputs", type=int, default=1)
    ap.add_argument("--fd-workers", type=int, default=4)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t_utc0 = utc_now()
    T0 = time.time()
    ds_root = dataset_root()
    ver = json.loads((ds_root / "VERSION.json").read_text(encoding="utf-8"))["version_hash"]
    if ver != VERSION_HASH:
        raise AssertionError("dataset version hash mismatch")
    state = load_state(Path(a.state))
    schema = load_schema()
    spec = build_spec(schema)
    v1norm = json.loads((ds_root / "mappings" / "normalizers.json").read_text(encoding="utf-8"))["features"]
    catcode_vocab = pq.read_metadata(ds_root / "mappings" / "vocab_category_code.parquet").num_rows + 3
    reg = build_registry(spec, v1norm)
    ctx = {"spec": spec, "ds_root": ds_root, "state_sha": state["transform_state_sha256"],
           "verify_per_bucket": a.verify_per_bucket, "catcode_vocab": catcode_vocab}
    buckets = parse_buckets(a.buckets)
    acc = Acc(reg)
    log(f"FEATSTAT-01 start {t_utc0} buckets={len(buckets)} ds_root={ds_root}")
    for b in buckets:
        process_bucket(b, ctx, acc)
        bl = acc.bucket_log[-1]
        log(f"bucket {b} rows={bl['train_rows']} dec={bl['decisions']} pool_dec={bl['pool_decisions']} "
            f"{bl['seconds']}s rss={bl['rss']}")
    s = acc.strata
    repro = {k: {"measured": s[k], "expected": EXPECT[k], "equal": s[k] == EXPECT[k]} for k in EXPECT}
    all_equal = all(v["equal"] for v in repro.values())
    log("strata reproduction", json.dumps(repro))
    if a.strict_counts and not all_equal:
        (out / "FEATSTAT_ABORT.json").write_text(json.dumps({"reason": "population counts do not reproduce",
                                                             "reproduction": repro}, indent=1), encoding="utf-8")
        raise SystemExit(3)
    t_phaseA = time.time()

    N = int(sum(acc.n_dec))
    Np = int(sum(int(x.sum()) for x in acc.pool))
    src = {"task": "FEATSTAT-01", "evidence": "PROJECT_MEASURED (this run's code, TRAIN rows only)",
           "populations": {"POOL": "users with 5 <= n <= 100 (n = supervised TRAIN examples, catalogue A)",
                           "FULL": "all users with n >= 1 (comparison only)"},
           "grain": "one row per supervised TRAIN example (decision with loss_mask) in the Clean V2 layout; "
                    "event channels at the decision's own event (last_valid_index); window-exposure grain reported "
                    "separately for event channels",
           "n_rows": {"FULL": N, "POOL": Np}}
    inv = INV.build_inventory(schema)
    inv["parameter_formula_check"] = INV.check_formulas()
    _write(out / "FEATSTAT_INVENTORY.json", dict(src, **inv), indent=1)

    # ---------------- section 9 provenance first (frees the raw tables before the heavy passes)
    prov = provenance(acc, spec, schema, v1norm, None, ds_root)
    prov.update(catalogue_vocab_firstseen(acc, ds_root, schema, v1norm, prov))
    acc.raw_tables = None
    acc.fs_parts = None
    release_memory()
    _write(out / "FEATSTAT_PROVENANCE.json", dict(src, **prov), indent=1)
    log("provenance written", rss_mib())

    N2, pool, G, C = build_codes(acc)
    if N2 != N or int(pool.sum()) != Np:
        raise AssertionError("row counts changed during the code build")
    release_memory()
    log(f"codes built N={N} Np={Np}", rss_mib())
    stats, counts = column_stats(reg, G, C, pool, N, Np)
    log("column stats done")
    cpass = chunk_pass(reg, G, C, counts, pool, N, Np, stats, a.chunk)
    release_memory()
    log("chunk pass done", rss_mib())

    # ---------------- window grain summaries
    win = {}
    for pop in ("FULL", "POOL"):
        W = acc.win[pop]
        S0 = W["S0"]
        e = {"window_positions": int(S0), "windows": W["n_windows"], "mean_window_length": S0 / W["n_windows"]}
        for n in spec.v2_numeric:
            x = W[n]
            mu = x["S1"] / S0
            has_default = next(r_ for r_ in reg if r_["name"] == n)["default"] is not None
            e[n] = {"mean": mu, "std": math.sqrt(max(x["S2"] / S0 - mu * mu, 0.0)),
                    "share_exact_zero": x["Szero"] / S0, "default_share": (x["Sdef"] / S0) if has_default else None}
        for n in spec.v2_flags:
            e[n] = {"prevalence": W[n]["S1"] / S0}
        for tname, _, _ in TOKENS:
            e[tname] = {k2: W[tname][k2] / S0 for k2 in ("MISSING", "OOV", "KNOWN")}
        e["availability"] = {k2: v / S0 for k2, v in W["avail"].items()}
        win[pop] = e

    # ---------------- missingness / availability overlap
    ind = cpass["ind_names"]
    miss = {}
    for pop, NN in (("FULL", N), ("POOL", Np)):
        A = cpass["ind_counts"][pop]
        d = np.diag(A).astype(np.float64)
        jac = np.full(A.shape, np.nan)
        jmiss = np.full(A.shape, np.nan)
        for i in range(len(ind)):
            for j in range(len(ind)):
                un = d[i] + d[j] - A[i, j]
                jac[i, j] = A[i, j] / un if un > 0 else np.nan
                m11 = NN - d[i] - d[j] + A[i, j]
                unm = (NN - d[i]) + (NN - d[j]) - m11
                jmiss[i, j] = m11 / unm if unm > 0 else np.nan
        miss[pop] = {"indicator_available_share": {k2: float(d[i] / NN) for i, k2 in enumerate(ind)},
                     "jaccard_available": matrix_json(ind, jac), "jaccard_missing": matrix_json(ind, jmiss),
                     "cooccurrence_counts": {"names": ind, "matrix": F.jsafe(A)}}
    chan_miss = {}
    for n, key in CHANNEL_AVAIL.items():
        i = ind.index(key)
        chan_miss[n] = {"availability_indicator": key}
        for pop, NN in (("FULL", N), ("POOL", Np)):
            st_ = stats[n][pop]
            chan_miss[n][pop] = {
                "mask_based_unavailable_share": 1.0 - float(cpass["ind_counts"][pop][i, i] / NN),
                "sentinel_based_v2_exact_zero_share": st_.get("share_exact_zero"),
                "sentinel_based_default_share": st_.get("default_share"),
                "token_MISSING_or_OOV_share": (st_["MISSING_share"] + st_["OOV_share"]) if "MISSING_share" in st_
                else None}
    raw_sent = {}
    for n in acc.raw_zero:
        raw_sent[n] = {pop: acc.raw_zero[n][pop] / (N if pop == "FULL" else Np) for pop in ("FULL", "POOL")}
    groups = {}
    for g, v in cpass["groups"].items():
        groups[g] = {"members": INV.GROUPS[g]}
        for pop, NN in (("FULL", N), ("POOL", Np)):
            groups[g][pop] = {"any_available_share": v[pop]["any_available"] / NN,
                              "any_non_default_share": None if v[pop]["any_non_default"] is None
                              else v[pop]["any_non_default"] / NN}

    # ---------------- input hashes (once)
    files_json = json.loads((ds_root / "FILES.json").read_text(encoding="utf-8"))
    fmap = {e_["path"]: e_["sha256"] for e_ in files_json["files"]}
    read = [f"model_ready/user_bucket={b}/features.parquet" for b in buckets]
    read += [p_ for p_ in fmap if p_.startswith("decisions_index/split=TRAIN/") and
             any(f"user_bucket={b}/" in p_ for b in buckets)]
    read += ["mappings/catalogue.parquet", "mappings/normalizers.json", "mappings/vocab_category_code.parquet",
             "VERSION.json"]
    hashes, bad = {}, []
    if a.hash_inputs:
        for p_ in read:
            h = sha256_file(ds_root / p_)
            hashes[p_] = h
            if fmap.get(p_) is None or fmap[p_] != h:
                bad.append(p_)
    log(f"input hashes {len(hashes)} mismatches {len(bad)}")

    def emit(fd, mfd, stage):
        summ = summarize(reg, stats, cpass, fd, N, Np, acc)
        stats_doc = dict(src)
        stats_doc.update({
            "stage": stage,
            "registry": [{k2: r_[k2] for k2 in ("idx", "name", "block", "kind", "default")} for r_ in reg],
            "channel_stats": stats, "window_exposure_grain": win,
            "missingness": {"per_channel": chan_miss, "raw_sentinel_zero_share_at_decision_rows": raw_sent,
                            "gate_vs_raw_zero_2x2_counts": acc.gate2x2, "availability_overlap": miss},
            "feature_group_prevalence": groups,
            "correlation": {
                "pearson": {p: matrix_json(cpass["corr_cols"], cpass["pearson"][p]) for p in ("FULL", "POOL")},
                "spearman": {p: matrix_json(cpass["corr_cols"], cpass["spearman"][p]) for p in ("FULL", "POOL")},
                "rules": "Pearson over all 52 non-token columns (phi for binary pairs, point-biserial for numeric x "
                         "binary); Spearman = Pearson of mid-ranks (exact, all rows, ties averaged)"},
            "clusters": summ["clusters"],
            "cluster_rule": "single-linkage connected components of the graph with an edge i-j when |r_ij| >= "
                            "threshold (Pearson or Spearman, computed separately; phi for binary channels); members "
                            "in tensor order",
            "status": summ["status"], "transfer": summ["transfer"], "correlation_transfer": summ["correlation_transfer"],
            "identity_ledger": {"violation_counts": acc.ledger, "max_local_hour_rotation_residual": acc.ledger_max_rot},
            "recovery_source_counts": acc.sources,
            "n_frequency_table": sorted([[k2, v] for k2, v in acc.nfreq.items()]),
        })
        _write(out / "FEATSTAT_STATS.json", stats_doc)
        if fd:
            fd_doc = dict(src)
            fd_doc.update({"names": fd["names"], "distinct_full": fd["D"],
                           "violations": {p: F.jsafe(fd["violations"][p]) for p in ("FULL", "POOL")},
                           "lambda": {p: F.jsafe(np.round(fd["lambda"][p], 8)) for p in ("FULL", "POOL")},
                           "cramers_v_exact_all_rows": fd["cramer"], "multi_column": mfd,
                           "fd_transfer": summ.get("fd_transfer"), "path_pairs": fd["path_pairs"],
                           "path_seconds": fd["path_seconds"],
                           "rule": "violations[X][Y] = N - sum_x max_y n(x,y): rows that must change for X to "
                                   "determine Y exactly; 0 = exact FD on the population; lambda = Goodman-Kruskal "
                                   "lambda(Y|X); exact on all rows (no sampling)", "seconds": fd["seconds"]})
            _write(out / "FEATSTAT_FD.json", fd_doc)
        code_files = sorted(HERE.glob("*.py"))
        receipt = {
            "task": "FEATSTAT-01", "stage": stage, "date_u_start": t_utc0, "date_u_written": utc_now(),
            "wall_seconds": round(time.time() - T0, 1), "phaseA_seconds": round(t_phaseA - T0, 1),
            "peak_rss": rss_mib(), "buckets": buckets, "ds_root": str(ds_root), "version_hash": ver,
            "schema_self_sha256": SCHEMA_SELF_SHA256, "transform_state_sha256": state["transform_state_sha256"],
            "transform_state_file_sha256": sha256_file(Path(a.state)),
            "inputs_read_sha256_checked_against_FILES_json": len(hashes), "input_hash_mismatches": bad,
            "input_sha256": hashes,
            "code_sha256": {p.name: sha256_file(p) for p in code_files},
            "v4data_sha256": {p.name: sha256_file(p) for p in sorted((CODE / "v4data").glob("*.py"))},
            "population_reproduction": repro, "population_reproduction_all_equal": all_equal,
            "clean_v2_verification": acc.verify, "bucket_log": acc.bucket_log,
            "fd_path_pairs": fd["path_pairs"] if fd else None, "fd_path_seconds": fd["path_seconds"] if fd else None,
            "no_validation_or_test_rows_read": True,
            "reads": "features.parquet read with split == TRAIN filter at read; CleanV2Store built with "
                     "splits=('TRAIN',) (the snapshot CompactStore filters its own read to TRAIN); decisions_index "
                     "split=TRAIN only; mappings catalogue/normalizers/vocab_category_code",
            "no_raw_identifier_written": True,
        }
        _write(out / "FEATSTAT_RECEIPT.json", receipt, indent=1)
        log(f"emitted stage {stage}", rss_mib())

    emit(None, None, "PRE_FD")
    if not a.skip_fd:
        fd = fd_pass(reg, G, C, counts, pool, N, Np, a.dense_max, workers=a.fd_workers)
        log(f"fd pass done {fd['pairs']} pairs {fd['seconds']:.0f}s {fd['path_pairs']} {fd['path_seconds']}",
            rss_mib())
        mfd = multi_fd(G, C, pool, N, Np)
        log("multi-column fd done")
        emit(fd, mfd, "FINAL")
    log("DONE", utc_now(), round(time.time() - T0, 1), "s", rss_mib())


def _write(path: Path, obj, indent=None):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(F.jsafe(obj), indent=indent, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
