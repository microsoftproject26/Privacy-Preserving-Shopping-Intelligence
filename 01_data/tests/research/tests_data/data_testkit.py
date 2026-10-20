"""Shared helpers for the DATA-04 tests: negative-control marker, a synthetic frozen-layout dataset, raw-row fixtures.

Negative controls follow code/tests_fl/fl_testkit.py: a key-property check is run against a deliberately wrong
variant and MUST raise AssertionError (strict xfail; an XPASS fails the suite).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import ppsi_paths
from v4data.common import VERSION_HASH, utc_us

LOCAL_DS = Path("F:/Queens University/Microsoft Project/eda/rees46_nextitem_v1")
SERVER_DS = ppsi_paths.DATA_ROOT
REAL_DS = SERVER_DS if SERVER_DS.is_dir() else LOCAL_DS


def nc(reason: str):
    """Negative control: the wrapped key-property check must raise AssertionError (strict xfail)."""
    return pytest.mark.xfail(strict=True, raises=AssertionError, reason="NEGATIVE CONTROL: " + reason)


def assert_raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc:
        return
    raise AssertionError(f"{getattr(fn, '__name__', fn)} did not raise {exc}")


# ------------------------------------------------------------------------------------ synthetic dataset
def _ts(iso: str) -> int:
    return utc_us(iso)


def make_user(uid: str, events: list, tails: dict | None = None, overrides: dict | None = None) -> list:
    """events: list of (split, iso_ts, product, segment). Segments are consecutive; within a segment every event
    but the last is OBSERVED_IN_PARTITION with the next event as target. `tails` maps a segment id to
    (status, next_split, next_block_type) for its last event (default NO_OBSERVED_NEXT). `overrides` maps an event
    index to a dict of ledger fields to force (used to plant defects the builder must catch)."""
    tails = tails or {}
    overrides = overrides or {}
    rows = []
    pos = 0
    for i, (split, iso, prod, seg) in enumerate(events):
        pos = pos + 1 if i and events[i - 1][3] == seg else 1
        nxt = events[i + 1] if i + 1 < len(events) and events[i + 1][3] == seg else None
        r = {"user_id": uid, "split": split, "decision_ts": _ts(iso), "product_id": prod, "segment_index": float(seg),
             "session_index": float(seg), "position_in_segment": pos}
        if nxt is not None:
            r.update(target_product_id=nxt[2], target_ts=_ts(nxt[1]), target_status="OBSERVED_IN_PARTITION",
                     next_split=None, next_block_type=None)
        else:
            st, ns, nb = tails.get(seg, ("NO_OBSERVED_NEXT", None, None))
            r.update(target_product_id=None, target_ts=None, target_status=st, next_split=ns, next_block_type=nb)
        r.update(overrides.get(i, {}))
        rows.append(r)
    return rows


def write_dataset(root: Path, users: list, catalogue: list, bucket: int = 0) -> Path:
    """Write a frozen-layout dataset (decisions, model_ready, mappings, VERSION.json) for the given user rows."""
    root = Path(root)
    rows = [r for u in users for r in u]
    cls = {p: i for i, p in enumerate(catalogue)}
    did = 0
    for r in rows:
        did += 1
        r["decision_id"] = 100_000_000_000 + did
        t = r["target_product_id"]
        has = t is not None
        r["has_observed_target"] = has
        r["target_class"] = cls.get(t, -1) if has else -1
        r["target_oov"] = bool(has and t not in cls)
        r.setdefault("loss_mask", bool(r["split"] == "TRAIN" and has and r["target_class"] >= 0))
        r["eval_mask"] = bool(r["split"] in ("VALIDATION_SELECT", "VALIDATION_CONFIRM", "TEST") and has)
    tsu = pa.timestamp("us")
    led = pa.table({
        "user_id": pa.array([r["user_id"] for r in rows], pa.string()),
        "decision_id": pa.array([r["decision_id"] for r in rows], pa.int64()),
        "decision_ts": pa.array([r["decision_ts"] for r in rows], pa.int64()).cast(tsu),
        "split": pa.array([r["split"] for r in rows], pa.string()),
        "product_id": pa.array([r["product_id"] for r in rows], pa.string()),
        "session_index": pa.array([r["session_index"] for r in rows], pa.float64()),
        "segment_index": pa.array([r["segment_index"] for r in rows], pa.float64()),
        "position_in_segment": pa.array([r["position_in_segment"] for r in rows], pa.int64()),
        "target_product_id": pa.array([r["target_product_id"] for r in rows], pa.string()),
        "target_ts": pa.array([r["target_ts"] for r in rows], pa.int64()).cast(tsu),
        "target_status": pa.array([r["target_status"] for r in rows], pa.string()),
        "next_split": pa.array([r["next_split"] for r in rows], pa.string()),
        "next_block_type": pa.array([r["next_block_type"] for r in rows], pa.string()),
        "has_observed_target": pa.array([r["has_observed_target"] for r in rows], pa.bool_()),
        "loss_mask": pa.array([r["loss_mask"] for r in rows], pa.bool_()),
        "eval_mask": pa.array([r["eval_mask"] for r in rows], pa.bool_()),
        "target_class": pa.array([r["target_class"] for r in rows], pa.int64()),
        "target_oov": pa.array([r["target_oov"] for r in rows], pa.bool_()),
    })
    d = root / "decisions" / f"user_bucket={bucket}"
    d.mkdir(parents=True, exist_ok=True)
    pq.write_table(led, d / "ledger.parquet")
    feat = pa.table({
        "user_id": led["user_id"], "split": led["split"], "segment_index": led["segment_index"],
        "position_in_segment": led["position_in_segment"], "product_id": led["product_id"],
        "block_ts": led["decision_ts"], "decision_id": led["decision_id"],
        "category_idx": pa.array([2 if r["product_id"].startswith("rare") else 3 + len(r["product_id"]) % 2
                                  for r in rows], pa.int64()),
        "brand_idx": pa.array([1 if r["product_id"].endswith("m") else 3 for r in rows], pa.int64()),
        "main_category_idx": pa.array([2 if r["product_id"].endswith("o") else 4 for r in rows], pa.int64()),
    })
    f = root / "model_ready" / f"user_bucket={bucket}"
    f.mkdir(parents=True, exist_ok=True)
    pq.write_table(feat, f / "features.parquet")
    m = root / "mappings"
    m.mkdir(parents=True, exist_ok=True)
    sup = {}
    for r in rows:
        if r["split"] == "TRAIN":
            sup[r["product_id"]] = sup.get(r["product_id"], 0) + 1
    pq.write_table(pa.table({"product_id": pa.array(catalogue, pa.string()),
                             "output_class": pa.array(list(range(len(catalogue))), pa.int64()),
                             "product_idx": pa.array([3 + i for i in range(len(catalogue))], pa.int64()),
                             "representative_views": pa.array([sup.get(p, 0) for p in catalogue], pa.int64())}),
                   m / "catalogue.parquet")
    tr_users = sorted({r["user_id"] for r in rows if r["split"] == "TRAIN"})
    pq.write_table(pa.table({"user_id": pa.array(tr_users, pa.string()),
                             "client_idx": pa.array(list(range(3, 3 + len(tr_users))), pa.int64())}),
                   m / "clients.parquet")
    (root / "VERSION.json").write_text(json.dumps({"version_hash": VERSION_HASH}), encoding="utf-8")
    return root


def standard_users():
    """A small population exercising every rule the manifests encode."""
    cat = ["pA", "pB", "pC", "pD", "pE", "pFm", "pGo"]
    warm = make_user("warm-1", [
        ("TRAIN", "2019-10-02T10:00:00", "pA", 0), ("TRAIN", "2019-10-02T10:10:00", "pB", 0),
        ("TRAIN", "2019-10-02T10:20:00", "pA", 0),                     # A -> B -> A revisit
        ("TRAIN", "2019-11-15T23:50:00", "pC", 1), ("TRAIN", "2019-11-15T23:59:59", "pD", 1),
        ("VALIDATION_SELECT", "2019-11-16T00:00:00", "pA", 2), ("VALIDATION_SELECT", "2019-11-16T00:30:00", "rare-x", 2),
        ("VALIDATION_SELECT", "2019-11-16T00:40:00", "pC", 2),         # A -> rare-x (OOV) -> C
        ("VALIDATION_SELECT", "2019-11-19T23:00:00", "pE", 3),         # BU VS->VC (legacy barrier)
        ("VALIDATION_CONFIRM", "2019-11-20T00:00:00", "pB", 4), ("VALIDATION_CONFIRM", "2019-11-20T00:05:00", "pFm", 4),
        ("VALIDATION_CONFIRM", "2019-11-22T23:59:00", "pGo", 5),       # BU VC->TEST (right edge)
        ("TEST", "2019-11-24T10:00:00", "pA", 6), ("TEST", "2019-11-24T10:05:00", "pB", 6),
    ], tails={3: ("BOUNDARY_UNRESOLVED", "VALIDATION_CONFIRM", "SINGLE_ITEM"),
              5: ("BOUNDARY_UNRESOLVED", "TEST", "SINGLE_ITEM"),
              1: ("NO_OBSERVED_NEXT", None, None)})
    warm2 = make_user("warm-2", [
        ("TRAIN", "2019-10-05T09:00:00", "pB", 0), ("TRAIN", "2019-10-05T09:05:00", "pC", 0),
        ("TRAIN", "2019-10-05T09:10:00", "rare-y", 0),                # TRAIN OOV target (support 1: BELOW_SUPPORT)
        ("VALIDATION_CONFIRM", "2019-11-21T09:00:00", "pD", 1), ("VALIDATION_CONFIRM", "2019-11-21T09:05:00", "rare-y", 1),
        ("VALIDATION_CONFIRM", "2019-11-21T09:10:00", "pE", 1),
        ("VALIDATION_CONFIRM", "2019-11-21T12:00:00", "pA", 2),
    ], tails={2: ("AMBIGUOUS_BOUNDARY", "VALIDATION_CONFIRM", "AMBIGUOUS_TIME_BLOCK_BARRIER")})
    # a TRAIN row whose target lies at TRAIN_END but carries loss_mask=True (planted leak, must be excluded)
    leak = make_user("warm-leak", [
        ("TRAIN", "2019-10-07T09:00:00", "pA", 0), ("TRAIN", "2019-10-07T09:05:00", "pB", 0),
        ("TRAIN", "2019-11-15T23:59:00", "pC", 1), ("TRAIN", "2019-11-15T23:59:30", "pD", 1),
        ("VALIDATION_SELECT", "2019-11-17T09:00:00", "pE", 2), ("VALIDATION_SELECT", "2019-11-17T09:05:00", "pA", 2),
    ], overrides={2: {"target_ts": _ts("2019-11-16T00:00:00")}})
    cold_new = make_user("cold-new", [
        ("VALIDATION_SELECT", "2019-11-17T12:00:00", "pA", 0), ("VALIDATION_SELECT", "2019-11-17T12:03:00", "pB", 0),
        ("VALIDATION_SELECT", "2019-11-17T12:06:00", "pC", 0),
    ])
    cold_ev = make_user("cold-events", [
        ("TRAIN", "2019-10-10T08:00:00", "pA", 0),
        ("VALIDATION_CONFIRM", "2019-11-20T08:00:00", "pB", 1), ("VALIDATION_CONFIRM", "2019-11-20T08:05:00", "pC", 1),
    ])
    return [warm, warm2, leak, cold_new, cold_ev], cat


# ------------------------------------------------------------------------------------ raw-row fixtures
def raw_rows(spec, n: int = 6, seed: int = 0) -> dict:
    """A consistent v1 raw-row dict (every identity RC1-RC7 holds) for encode_rows."""
    rng = np.random.default_rng(seed)
    raw = {}
    for name in spec.v1_numeric:
        raw[name] = rng.uniform(0.5, 6.0, n).astype(np.float32)
    th = rng.uniform(0, 2 * np.pi, n)
    raw["utc_hour_sin"], raw["utc_hour_cos"] = np.sin(th).astype(np.float32), np.cos(th).astype(np.float32)
    raw["local_hour_sin"] = np.sin(th + np.pi / 4).astype(np.float32)
    raw["local_hour_cos"] = np.cos(th + np.pi / 4).astype(np.float32)
    raw["log1p_prefix_elapsed_s"] = raw["log1p_segment_elapsed_s"].copy()
    raw["prefix_representative_count"] = rng.integers(0, 30, n).astype(np.float32)
    raw["prefix_item_count"] = raw["prefix_representative_count"] + 1
    raw["is_weekend"] = rng.integers(0, 2, n).astype(np.float32)
    for f in spec.v1_flags:
        raw[f] = np.zeros(n, dtype=bool)
    for name in spec.v1_uctx:
        raw[name] = rng.uniform(0.0, 5.0, n).astype(np.float32)
    raw["current_minus_recent_mean_log_price"] = rng.uniform(-2, 2, n).astype(np.float32)
    for m in spec.v1_masks:
        raw[m] = np.ones(n, dtype=np.float32)
    raw["has_history"] = np.ones(n, dtype=np.float32)
    raw["recent_category_match_share"] = rng.uniform(0, 1, n).astype(np.float32)
    raw["recent_brand_match_share"] = rng.uniform(0, 1, n).astype(np.float32)
    return raw


def cold_raw(spec) -> dict:
    """One genuinely cold user row: every user-context raw value 0 and every mask 0."""
    raw = raw_rows(spec, n=1)
    for name in spec.v1_uctx:
        raw[name] = np.zeros(1, dtype=np.float32)
    for m in spec.v1_masks:
        raw[m] = np.zeros(1, dtype=np.float32)
    return raw
