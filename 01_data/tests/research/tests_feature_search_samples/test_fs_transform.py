"""FS-01 TRANSFORM_STATE_FS tests (optional sensitivity normalizers): the fit kernels against numpy, the per-channel
row predicates against the schema, the TRAIN-only bucket reader, the reproduction criterion, and (server) the state
file against TRANSFORM_STATE_CLEAN_V2 - with strict-xfail negative controls."""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from conftest import DATA_FS, V4, real_only
from data_testkit import nc
from fs_testkit import load_json
from featstat_v1 import fs_core as F
from v4data.common import digests_of, digest_rows, load_schema, sha256_file
from v4fs_data import fs_common as FC
from v4fs_data import fs_transform as T

CHANNELS = tuple(load_schema()["normalized_channels_v2"].keys())


def _sample(seed=0, n=20000):
    rng = np.random.default_rng(seed)
    x = rng.normal(5.0, 2.0, n).astype(np.float32)
    x[:5] = 100.0                      # 5 outliers per side: below the 0.1% tail (20 rows), so the clip binds
    x[5:10] = -50.0
    return x


def test_fit_kernels_match_numpy():
    x = _sample()
    t = F.value_counts(x)
    fit = T.fit_channel(t, t)
    lo, hi = np.percentile(x.astype(np.float64), [0.1, 99.9], method="linear")
    assert fit["clip_lo"] == float(np.float32(lo)) and fit["clip_hi"] == float(np.float32(hi))
    c = np.clip(x.astype(np.float64), fit["clip_lo"], fit["clip_hi"])
    assert abs(fit["mean"] - c.mean()) <= 1e-12 * abs(c.mean())
    assert abs(fit["std"] - c.std()) <= 1e-12 * c.std()
    assert fit["n_quantile_rows"] == fit["n_valid_rows"] == x.size


@nc("fitting without the clip must not reproduce the clipped moments")
def test_nc_unclipped_moments_differ():
    x = _sample()
    fit = T.fit_channel(F.value_counts(x), F.value_counts(x))
    assert abs(fit["mean"] - float(x.astype(np.float64).mean())) <= 1e-12 * abs(fit["mean"])


def _raw(n=8):
    b = lambda *v: np.array(v, dtype=bool)  # noqa: E731
    f = lambda *v: np.array(v, dtype=np.float32)  # noqa: E731
    return {"price_missing": b(0, 1, 0, 0, 0, 0, 0, 0), "price_conflict": b(0, 0, 1, 0, 0, 0, 0, 0),
            "prev_gap_unavailable": b(0, 0, 0, 1, 0, 0, 0, 0), "item_age_unavailable": b(0, 0, 0, 0, 1, 0, 0, 0),
            "recent_price_supported": f(1, 1, 1, 1, 1, 0, 1, 1), "prev_user_event_available": f(1, 1, 1, 1, 1, 1, 0, 1),
            "prev_session_available": f(1, 1, 1, 1, 1, 1, 1, 0)}


def test_masks_follow_the_schema_fit_population():
    raw, n = _raw(), 8
    exp = {"log1p_price": ([1, 0, 0, 1, 1, 1, 1, 1],) * 2,
           "log1p_prev_raw_view_gap_s": ([1, 1, 1, 0, 1, 1, 1, 1],) * 2,
           "log1p_item_observed_age_s": ([1, 1, 1, 1, 0, 1, 1, 1],) * 2,
           "recent_mean_log_price": ([1, 1, 1, 1, 1, 0, 1, 1],) * 2,
           "current_minus_recent_mean_log_price": ([1, 1, 1, 1, 1, 0, 1, 1], [1, 0, 0, 1, 1, 0, 1, 1]),
           "log1p_time_since_prev_user_event_s": ([1] * 8, [1, 1, 1, 1, 1, 1, 0, 1]),
           "log1p_time_since_prev_session_last_event_s": ([1] * 8, [1, 1, 1, 1, 1, 1, 1, 0])}
    for c in CHANNELS:
        q, v = T.masks(c, raw, n)
        eq, ev = exp.get(c, ([1] * 8, [1] * 8))
        assert q.tolist() == [bool(i) for i in eq] and v.tolist() == [bool(i) for i in ev], c


def _synthetic_features(root, users_train, users_test):
    rows = []
    for u, k in users_train:
        rows += [("TRAIN", u, float(k + j)) for j in range(5)]
    for u, k in users_test:
        rows += [("TEST", u, 1e6)]
    n = len(rows)
    cols = {"split": pa.array([r[0] for r in rows]), "user_id": pa.array([r[1] for r in rows])}
    for c in CHANNELS:
        cols[c] = pa.array(np.array([r[2] for r in rows], dtype=np.float32))
    for p in ("price_missing", "price_conflict", "prev_gap_unavailable", "item_age_unavailable"):
        cols[p] = pa.array(np.zeros(n, dtype=bool))
    for p in ("recent_price_supported", "prev_user_event_available", "prev_session_available"):
        cols[p] = pa.array(np.ones(n, dtype=np.float32))
    d = root / "model_ready" / "user_bucket=0"
    d.mkdir(parents=True)
    pq.write_table(pa.table(cols), d / "features.parquet")


def test_bucket_reader_is_train_only_and_splits_fs_rows(tmp_path):
    _synthetic_features(tmp_path, [("u1", 0), ("u2", 10), ("u3", 20)], [("u1", 0), ("u9", 0)])
    fs = frozenset(digest_rows(digests_of(["u2"])))
    r = T.bucket_tables(0, ds=tmp_path, fs_set=fs, channels=CHANNELS)
    assert r["rows"] == 15 and r["fs_rows"] == 5 and r["fs_users"] == 1
    for c in CHANNELS:
        (qa, _), (qf, _) = r["tables"][(c, "ALL")][0], r["tables"][(c, "FS")][0]
        assert qa.max() < 1e6                                        # the planted TEST values never enter
        assert sorted(qf.tolist()) == [10.0, 11.0, 12.0, 13.0, 14.0]


@nc("an unfiltered read of the synthetic features does contain the planted TEST rows")
def test_nc_unfiltered_features_contain_test(tmp_path):
    _synthetic_features(tmp_path, [("u1", 0)], [("u9", 0)])
    t = pq.read_table(tmp_path / "model_ready" / "user_bucket=0" / "features.parquet", columns=["split"])
    assert "TEST" not in set(t["split"].to_pylist())


def test_reproduction_criterion():
    x = _sample(1)
    t = F.value_counts(x)
    fit = T.fit_channel(t, t)
    rec = {k: fit[k] for k in ("clip_lo", "clip_hi", "mean", "std")}
    cmp = T.compare(fit, rec, t)
    assert cmp["reproduced"] and cmp["distinct_values_float32_z_differ"] == 0


@nc("a recorded mean off by 1e-9 relative is not a reproduction")
def test_nc_reproduction_criterion_rejects_small_drift():
    x = _sample(2)
    t = F.value_counts(x)
    fit = T.fit_channel(t, t)
    rec = {k: fit[k] for k in ("clip_lo", "clip_hi", "mean", "std")}
    rec["mean"] *= 1 + 1e-9
    assert T.compare(fit, rec, t)["reproduced"]


# ------------------------------------------------------------------------------------------ real (server)
def _state():
    return load_json(DATA_FS / "TRANSFORM_STATE_FS.json")


@real_only
def test_real_transform_state_fs_method_and_values():
    from v4data.transform_state import load_state
    st = _state()
    assert FC.verify_self_hash(st, "transform_state_fs_sha256")
    assert st["status"] == "OPTIONAL_SENSITIVITY_ONLY" and "never used by the screen" in st["use"]
    clean = load_state(V4 / "data_v2" / "TRANSFORM_STATE_CLEAN_V2.json")
    assert clean["transform_state_sha256"] == FC.TRANSFORM_STATE_SELF_SHA256
    assert set(st["normalized_channels_fs"]) == set(CHANNELS) == set(clean["normalized_channels_v2"])
    m = st["method_proof_all_train"]
    assert m["all_reproduced"] is True and m["all_train_rows"]["observed"] == T.FULL_TRAIN_ROWS
    for c in CHANNELS:
        rec = clean["normalized_channels_v2"][c]
        fit = m["per_channel"][c]["fit_all_train"]
        assert fit["clip_lo"] == rec["clip_lo"] and fit["clip_hi"] == rec["clip_hi"], c
        assert abs(fit["mean"] - rec["mean"]) <= T.REL_TOL * max(abs(rec["mean"]), 1e-300), c
        assert abs(fit["std"] - rec["std"]) <= T.REL_TOL * rec["std"], c
        fs = st["normalized_channels_fs"][c]
        assert fs["std"] > 0 and fs["clip_lo"] <= fs["clip_hi"] and fs["n_valid_rows"] > 0
    assert st["rezero_map"] == clean["rezero_map"]
    assert st["fit"]["fs_users_seen_in_train_rows"] == FC.FS_N
    h = load_json(DATA_FS / "manifests" / f"{FC.FS_TRAIN_ID}.header.json")
    assert st["fit"]["population"]["header_content_sha256"] == h["header_content_sha256"]
    reg = load_json(V4 / "contracts" / "manifest_registry" / "REGISTRY_FS.json")
    assert reg["transform_state"]["transform_state_sha256"] == FC.TRANSFORM_STATE_SELF_SHA256
    o = reg["transform_state_optional_sensitivity"]
    assert o["transform_state_fs_sha256"] == st["transform_state_fs_sha256"]
    assert sha256_file(V4 / o["path"]) == o["file_sha256"]


@real_only
@nc("the FS-fitted normalizers are not the full-TRAIN ones (the refit changes at least one value)")
def test_nc_real_fs_values_equal_clean_v2():
    st = _state()
    assert all(all(st["normalized_channels_fs"][c][k] == st["normalized_channels_fs"][c]["clean_v2_values"][k]
                   for k in ("clip_lo", "clip_hi", "mean", "std")) for c in CHANNELS)
