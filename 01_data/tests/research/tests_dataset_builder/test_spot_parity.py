"""Real-data spot parity (DATA-04 packet): the first 1,024 TRAIN loss decisions of bucket 0 (ascending decision_id).

Proves: Clean V2 equals v1 byte-for-byte on every kept column of every window; the three re-encoded user-context
columns equal the frozen per-column formula applied to the stored raw values (EXACT_INVERSE path, tolerance 1e-5,
the observed maximum is printed); the reference encoder encode_rows agrees with the store view.
Does not prove: equality with any v1 model output (no weights are involved).
"""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from data_testkit import REAL_DS
from v4data.manifests import read_non_test
from v4data.transform import encode_rows, reference_formula

N = 1024
TOL_REENCODE = 1e-5


def _batches(real_store):
    ds = real_store.decision_set("TRAIN", "loss")
    pos = np.arange(N)
    return ds, real_store.v1_batch(ds, pos), real_store.batch(ds, pos)


def test_spot_parity_kept_columns_exact(real_store, spec):
    ds, v1, v2 = _batches(real_store)
    n1, n2 = v1["event_numeric_features"].numpy(), v2["event_numeric_features"].numpy()
    assert n2.shape[-1] == 15 and np.array_equal(n2, n1[..., list(spec.num_src)])
    f1, f2 = v1["event_quality_flags"].numpy(), v2["event_quality_flags"].numpy()
    assert f2.shape[-1] == 15 and np.array_equal(f2, f1[..., list(spec.flag_src)])
    m1, m2 = v1["user_context_masks"].numpy(), v2["user_context_masks"].numpy()
    assert m2.shape[-1] == 4 and np.array_equal(m2, m1[:, list(spec.mask_src)])
    u1, u2 = v1["user_context"].numpy(), v2["user_context"].numpy()
    re_idx = {e["uctx_index"] for e in spec.reencode.values()}
    keep = [i for i in range(12) if i not in re_idx]
    assert np.array_equal(u2[:, keep], u1[:, keep])
    for k in ("item_tokens", "category_tokens", "brand_tokens", "main_category_tokens", "daypart_tokens",
              "weekday_tokens", "attention_mask", "lengths", "target_class", "decision_id"):
        assert np.array_equal(v1[k].numpy(), v2[k].numpy()), k


def _raw_decision_rows(spec, decision_ids):
    names = list(spec.v1_numeric) + list(spec.v1_flags) + list(spec.v1_uctx) + list(spec.v1_masks)
    t = read_non_test(REAL_DS / "model_ready" / "user_bucket=0" / "features.parquet", names + ["decision_id"])
    t = t.filter(pc.is_in(t["decision_id"], value_set=pa.array(decision_ids)))
    o = np.argsort(t["decision_id"].to_numpy())
    got = t["decision_id"].to_numpy()[o]
    assert np.array_equal(got, np.sort(decision_ids))
    return {n: t[n].to_numpy(zero_copy_only=False)[o] for n in names}


def test_spot_parity_reencoded_columns_equal_formula(real_store, spec):
    ds, v1, v2 = _batches(real_store)
    dids = v2["decision_id"].numpy()
    assert np.all(np.diff(dids) > 0)
    raw = _raw_decision_rows(spec, dids)
    u2 = v2["user_context"].numpy()
    worst = {}
    for name, e in spec.reencode.items():
        if e["gate"] == "F1":
            gate = (raw["recent_price_supported"] != 0) & ~(raw["price_missing"].astype(bool)
                                                           | raw["price_conflict"].astype(bool))
        else:
            gate = raw[e["gate"]] != 0
        ref = reference_formula(name, raw[name], gate, spec)
        diff = np.abs(u2[:, spec.v2_uctx.index(name)].astype(np.float64) - ref.astype(np.float64))
        worst[name] = float(diff.max())
        assert worst[name] <= TOL_REENCODE, (name, worst[name])
        assert np.all(u2[~gate, spec.v2_uctx.index(name)] == 0.0)
    print("spot-parity max |view - formula| per re-encoded column:", worst)


def test_spot_parity_reference_encoder_agrees(real_store, spec):
    ds, v1, v2 = _batches(real_store)
    raw = _raw_decision_rows(spec, v2["decision_id"].numpy())
    ref = encode_rows(raw, spec)
    lvi = v2["last_valid_index"].numpy()
    rows = np.arange(N)
    n2 = v2["event_numeric_features"].numpy()[rows, lvi]
    assert np.array_equal(n2, ref["event_numeric_features"]), "decision-row numerics differ from the reference encoder"
    assert np.array_equal(v2["event_quality_flags"].numpy()[rows, lvi], ref["event_quality_flags"])
    assert np.array_equal(v2["user_context_masks"].numpy(), ref["user_context_masks"])
    re_idx = [spec.v2_uctx.index(n) for n in spec.reencode]
    keep = [i for i in range(12) if i not in re_idx]
    assert np.array_equal(v2["user_context"].numpy()[:, keep], ref["user_context"][:, keep])
    assert np.max(np.abs(v2["user_context"].numpy()[:, re_idx] - ref["user_context"][:, re_idx])) <= TOL_REENCODE


def test_iter_batches_bounded(real_store):
    ds = real_store.decision_set("TRAIN", "loss")
    sizes = [b["decision_id"].shape[0] for b in real_store.iter_batches(ds, np.arange(70), batch_size=32)]
    assert sizes == [32, 32, 6]
