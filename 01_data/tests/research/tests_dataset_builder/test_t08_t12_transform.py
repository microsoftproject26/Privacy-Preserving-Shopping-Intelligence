"""T08, T09, T11, T12 for the Clean V2 transform (PKG/12_ACCEPTANCE_TESTS.md L13-L17; DATA_CONTRACT_CLEAN_V2 section 10).

Proves: missing/invalid values re-zero and stay finite; non-finite input is refused; degenerate refits are refused;
the three re-encoded columns use the frozen valid-only values and re-zero exactly; the golden cold-user vector;
the state machine refuses double transforms and foreign layouts; every runtime identity check catches its defect;
future-relevant dead flags stay in the tensor.
Does not prove: any ranking effect, or that future data has no new defect class.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from data_testkit import REAL_DS, assert_raises, cold_raw, nc, raw_rows
from v4data.transform import (IdentityCheckError, SchemaError, StateError, V1_VIEW_TAG, build_spec, clean_v2_view,
                              encode_rows, identity_checks_raw, validate_schema)


def _enc(spec, raw):
    return encode_rows(raw, spec)


def _u(spec, name):
    return spec.v2_uctx.index(name)


def _n(spec, name):
    return spec.v2_numeric.index(name)


# ------------------------------------------------------------------------------------------------ T08
def test_t08_invalid_price_rezeroed_and_finite(spec):
    raw = raw_rows(spec, n=5)
    raw["price_missing"][:] = [True, False, False, True, False]
    raw["price_conflict"][:] = [False, True, False, False, False]
    raw["log1p_price"][[0, 1, 3]] = 0.0                      # stored sentinel for unavailable prices
    out = _enc(spec, raw)
    col = out["event_numeric_features"][:, _n(spec, "log1p_price")]
    assert col[0] == 0.0 and col[1] == 0.0 and col[3] == 0.0
    assert np.all(col[[2, 4]] != 0.0)
    for k in ("event_numeric_features", "user_context", "user_context_masks"):
        assert np.all(np.isfinite(out[k]))


def test_t08_all_missing_column_is_all_zero(spec):
    raw = raw_rows(spec, n=4)
    raw["prev_gap_unavailable"][:] = True
    raw["log1p_prev_raw_view_gap_s"][:] = 0.0
    out = _enc(spec, raw)
    assert np.all(out["event_numeric_features"][:, _n(spec, "log1p_prev_raw_view_gap_s")] == 0.0)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_t08_nonfinite_input_refused(spec, bad):
    raw = raw_rows(spec, n=3)
    raw["log1p_price"][1] = bad
    assert_raises(IdentityCheckError, encode_rows, raw, spec)


def test_t08_degenerate_refit_refused(schema):
    s = copy.deepcopy(schema)
    s["normalized_channels_v2"]["log1p_time_since_prev_user_event_s"]["std"] = 0.0
    s["m1_reencoding"]["features"]["log1p_time_since_prev_user_event_s"]["new_std"] = 0.0
    assert_raises(SchemaError, build_spec, s)


@nc("an encoder without the re-zero gate leaks z(0) for a missing price")
def test_nc_t08_no_rezero(spec):
    raw = raw_rows(spec, n=1)
    raw["price_missing"][:] = True
    raw["log1p_price"][:] = 0.0
    lo, hi, mu, sd = spec.norm_v2["log1p_price"]
    naive = (min(max(0.0, lo), hi) - mu) / sd
    assert naive == 0.0


# ------------------------------------------------------------------------------------------------ T09
def test_t09_reencoded_columns_use_frozen_values(spec):
    raw = raw_rows(spec, n=6, seed=3)
    out = _enc(spec, raw)
    for name, e in spec.reencode.items():
        lo, hi = e["clip"]
        mu, sd = e["new"]
        want = ((np.clip(raw[name].astype(np.float64), lo, hi) - mu) / sd).astype(np.float32)
        assert np.array_equal(out["user_context"][:, _u(spec, name)], want), name
    assert spec.reencode["current_minus_recent_mean_log_price"]["new"] == (0.002065693626402541, 0.9945344391661521)
    assert spec.reencode["log1p_time_since_prev_user_event_s"]["new"] == (5.3788876148270335, 3.178172706038617)
    assert spec.reencode["log1p_time_since_prev_session_last_event_s"]["new"] == (11.8168906913688, 1.6580237946197298)


def test_t09_gates_rezero_exactly(spec):
    raw = raw_rows(spec, n=4, seed=5)
    raw["prev_user_event_available"][:] = [0, 1, 1, 1]
    raw["has_history"][:] = raw["prev_user_event_available"]
    raw["has_history_mask"][:] = raw["prev_user_event_available"]
    raw["prev_session_available"][:] = [0, 0, 1, 1]
    raw["recent_price_supported"][:] = [1, 1, 0, 1]
    raw["price_missing"][:] = [False, True, False, False]       # F-1: recent supported, current price unknown
    raw["log1p_price"][1] = 0.0
    raw["recent_mean_log_price"][2] = 0.0
    raw["current_minus_recent_mean_log_price"][[1, 2]] = 0.0
    out = _enc(spec, raw)["user_context"]
    assert out[0, _u(spec, "log1p_time_since_prev_user_event_s")] == 0.0
    assert np.all(out[[0, 1], _u(spec, "log1p_time_since_prev_session_last_event_s")] == 0.0)
    assert out[1, _u(spec, "current_minus_recent_mean_log_price")] == 0.0, "F-1 row not re-zeroed"
    assert out[2, _u(spec, "current_minus_recent_mean_log_price")] == 0.0
    assert out[2, _u(spec, "recent_mean_log_price")] == 0.0
    assert out[3, _u(spec, "current_minus_recent_mean_log_price")] != 0.0


def test_t09_golden_cold_user_vector(spec, schema):
    out = _enc(spec, cold_raw(spec))
    gold = np.array(schema["tensor_layout_v2"]["cold_user_vector_v2_expected"]["user_context"], dtype=np.float32)
    assert np.allclose(out["user_context"][0], gold, atol=1e-6)
    assert np.all(out["user_context_masks"][0] == 0.0)


def test_t09_f1_fit_with_zeros_reproduces_the_old_normalizer(spec):
    """The valid-only F-1 fit plus the 117,315 raw zeros gives back the recorded all-row mean/std (G0B C2)."""
    nv, z = 48_434_230, 117_315
    mu, sd = spec.reencode["current_minus_recent_mean_log_price"]["new"]
    old_mu, old_sd = spec.reencode["current_minus_recent_mean_log_price"]["old"]
    n = nv + z
    m = nv * mu / n
    s = math.sqrt(nv * (sd * sd + mu * mu) / n - m * m)
    assert abs(m / old_mu - 1) < 1e-12 and abs(s / old_sd - 1) < 1e-12


@nc("a fit that includes the 117,315 sentinel zeros is the OLD normalizer, not the frozen valid-only one")
def test_nc_t09_fit_with_zeros_is_rejected(spec):
    nv, z = 48_434_230, 117_315
    mu, sd = spec.reencode["current_minus_recent_mean_log_price"]["new"]
    n = nv + z
    m = nv * mu / n
    s = math.sqrt(nv * (sd * sd + mu * mu) / n - m * m)
    assert abs(s - sd) < 1e-9


def test_t09_missing_gate_would_surface_documented_values(spec):
    for name, expect in (("log1p_time_since_prev_user_event_s", -1.692446607639351),
                         ("log1p_time_since_prev_session_last_event_s", -7.127093549389634)):
        lo, hi, mu, sd = spec.norm_v2[name]
        assert abs((min(max(0.0, lo), hi) - mu) / sd - expect) < 1e-12


@nc("removing an M1 re-zero entry leaves the unavailable row at z(0) != 0")
def test_nc_t09_missing_m1_gate(spec):
    lo, hi, mu, sd = spec.norm_v2["log1p_time_since_prev_session_last_event_s"]
    assert (min(max(0.0, lo), hi) - mu) / sd == 0.0


def test_t09_real_cold_rows_equal_golden(spec, schema):
    import pyarrow.compute as pc
    from v4data.manifests import read_non_test
    names = list(spec.v1_numeric) + list(spec.v1_flags) + list(spec.v1_uctx) + list(spec.v1_masks)
    t = read_non_test(REAL_DS / "model_ready" / "user_bucket=0" / "features.parquet", names)
    t = t.filter(pc.equal(t["has_history"], 0.0))
    assert t.num_rows > 100
    raw = {n: t[n].to_numpy(zero_copy_only=False) for n in names}
    out = encode_rows(raw, spec)
    gold = np.array(schema["tensor_layout_v2"]["cold_user_vector_v2_expected"]["user_context"], dtype=np.float32)
    assert np.allclose(out["user_context"], gold[None, :], atol=1e-6)


# ------------------------------------------------------------------------------------------------ T11
def _v1_batch(spec, B=2, L=3):
    rng = np.random.default_rng(0)
    raw = raw_rows(spec, n=B * L, seed=9)
    num = np.stack([raw[n] for n in spec.v1_numeric], axis=1).reshape(B, L, -1).astype(np.float32)
    num[..., 14] = num[..., 13]                    # in a v1-ENCODED batch the two count columns are equal z values
    flg = np.stack([raw[f].astype(np.int8) for f in spec.v1_flags], axis=1).reshape(B, L, -1)
    uc = np.stack([raw[n][:B] for n in spec.v1_uctx], axis=1).astype(np.float32)
    mk = np.stack([raw[m][:B] for m in spec.v1_masks], axis=1).astype(np.float32)
    return {"event_numeric_features": num, "event_quality_flags": flg, "user_context": uc, "user_context_masks": mk,
            "attention_mask": np.ones((B, L), bool), "last_valid_index": np.full(B, L - 1),
            "feature_view": V1_VIEW_TAG, "v1_input_schema_sha256": spec.input_schema_sha256}


def test_t11_view_shapes_and_state_tags(spec):
    out = clean_v2_view(_v1_batch(spec), spec)
    assert out["event_numeric_features"].shape[-1] == 15 and out["event_quality_flags"].shape[-1] == 15
    assert out["user_context"].shape[-1] == 12 and out["user_context_masks"].shape[-1] == 4
    assert out["feature_view"] == "clean_v2_rich"


def test_t11_double_transform_refused(spec):
    out = clean_v2_view(_v1_batch(spec), spec)
    assert_raises(StateError, clean_v2_view, out, spec)


def test_t11_untagged_or_foreign_layout_refused(spec):
    b = _v1_batch(spec)
    b.pop("feature_view")
    assert_raises(StateError, clean_v2_view, b, spec)
    b = _v1_batch(spec)
    b["v1_input_schema_sha256"] = "0" * 64
    assert_raises(StateError, clean_v2_view, b, spec)


def test_t11_input_not_mutated_and_stored_logs_not_relogged(spec):
    b = _v1_batch(spec)
    snap = {k: np.array(v, copy=True) for k, v in b.items() if isinstance(v, np.ndarray)}
    clean_v2_view(b, spec)
    for k, v in snap.items():
        assert np.array_equal(b[k], v), f"{k} was mutated"
    raw = raw_rows(spec, n=3)
    got = encode_rows(raw, spec)["event_numeric_features"][:, _n(spec, "log1p_segment_elapsed_s")]
    lo, hi, mu, sd = spec.norm_v2["log1p_segment_elapsed_s"]
    want = ((np.clip(raw["log1p_segment_elapsed_s"].astype(np.float64), lo, hi) - mu) / sd).astype(np.float32)
    assert np.array_equal(got, want)


def test_t11_valid_negative_z_accepted(spec):
    raw = raw_rows(spec, n=2)
    raw["current_minus_recent_mean_log_price"][:] = [-3.5, -0.2]
    out = encode_rows(raw, spec)["user_context"][:, _u(spec, "current_minus_recent_mean_log_price")]
    assert np.all(out < 0)


@nc("applying log1p to an already-logged stored value changes the encoding")
def test_nc_t11_double_log(spec):
    raw = raw_rows(spec, n=3)
    lo, hi, mu, sd = spec.norm_v2["log1p_segment_elapsed_s"]
    good = (np.clip(raw["log1p_segment_elapsed_s"].astype(np.float64), lo, hi) - mu) / sd
    bad = (np.clip(np.log1p(raw["log1p_segment_elapsed_s"].astype(np.float64)), lo, hi) - mu) / sd
    assert np.allclose(good, bad)


# ------------------------------------------------------------------------------------------------ T12
PERTURB = {
    "RC1": lambda r: r["log1p_prefix_elapsed_s"].__setitem__(0, r["log1p_prefix_elapsed_s"][0] + 1e-3),
    "RC2": lambda r: r["prefix_item_count"].__setitem__(0, r["prefix_item_count"][0] + 1),
    "RC3": lambda r: r["local_hour_sin"].__setitem__(0, np.float32(np.sin(np.arcsin(r["utc_hour_sin"][0]) + 0.5))),
    "RC4": lambda r: r["has_history_mask"].__setitem__(0, 1.0 - r["has_history_mask"][0]),
    "RC5": lambda r: r["prev_user_event_available"].__setitem__(0, 1.0 - r["prev_user_event_available"][0]),
    "RC7": lambda r: (r["recent_category_match_supported"].__setitem__(0, 0.0),
                      r["recent_category_match_share"].__setitem__(0, 0.5)),
}


def test_t12_consistent_rows_pass(spec):
    assert all(v == 0 for v in identity_checks_raw(raw_rows(spec, n=8)).values())


@pytest.mark.parametrize("rc", sorted(PERTURB))
def test_t12_each_identity_catches_its_defect(spec, rc):
    raw = raw_rows(spec, n=4, seed=11)
    PERTURB[rc](raw)
    with pytest.raises(IdentityCheckError) as ei:
        encode_rows(raw, spec)
    assert rc in str(ei.value)


def test_t12_rc6_session_implies_event(spec):
    raw = raw_rows(spec, n=3)
    raw["prev_user_event_available"][0] = 0.0
    raw["has_history"][0] = 0.0
    raw["has_history_mask"][0] = 0.0
    raw["prev_session_available"][0] = 1.0
    assert_raises(IdentityCheckError, encode_rows, raw, spec)


def test_t12_real_bucket0_identities_hold(spec):
    from v4data.manifests import read_non_test
    names = list(spec.v1_numeric) + list(spec.v1_flags) + list(spec.v1_uctx) + list(spec.v1_masks)
    t = read_non_test(REAL_DS / "model_ready" / "user_bucket=0" / "features.parquet", names)
    raw = {n: t[n].to_numpy(zero_copy_only=False) for n in names}
    rc = identity_checks_raw(raw)
    assert all(v == 0 for v in rc.values()) and t.num_rows > 1_000_000
    for f in spec.dead_flags:
        assert f in spec.v2_flags


def test_t12_schema_validator_keeps_dead_flags(schema):
    s = copy.deepcopy(schema)
    lay = s["tensor_layout_v2"]["event_quality_flags"]
    i = lay["names"].index("category_conflict")
    lay["names"].pop(i)
    lay["v1_source_indices"].pop(i)
    assert_raises(SchemaError, validate_schema, s)


@nc("an identity check with a loose 1e-2 tolerance misses a 1e-3 elapsed-time defect")
def test_nc_t12_loose_tolerance(spec):
    raw = raw_rows(spec, n=4)
    raw["log1p_prefix_elapsed_s"][0] += np.float32(1e-3)
    resid = np.abs(raw["log1p_prefix_elapsed_s"].astype(np.float64) - raw["log1p_segment_elapsed_s"])
    assert np.count_nonzero(resid > 1e-2) > 0
