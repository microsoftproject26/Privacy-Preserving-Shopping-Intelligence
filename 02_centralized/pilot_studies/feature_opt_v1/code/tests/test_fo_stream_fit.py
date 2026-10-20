"""Package test list item 6 -- stream_fit.py (owner: Opus): the fit-on-VS trap. Never modifies
the code under test. Writes only under pytest tmp dirs (dataset root via a plain path argument,
outputs via FO_ROOT_OVERRIDE through the set_fo_root fixture).
"""
from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import conftest_fo  # noqa: F401
from conftest_fo import set_fo_root  # noqa: F401

import feature_view as fv
import stream_fit as sf

SPEC = fv.normalizers().spec


def _all_columns() -> list[str]:
    return sorted(set(
        ["split", "local_weekday_idx", "daypart_idx"] + list(fv.M1_FEATURES) + list(fv.M1_FEATURES.values())
        + [a for a, _, _ in sf.AFFINE] + [b for _, b, _ in sf.AFFINE]
        + ["utc_hour_sin", "utc_hour_cos", "local_hour_sin", "local_hour_cos"]
        + [a for a, b in sf.FLAG_DUPS + sf.MASK_DUPS] + [b for a, b in sf.FLAG_DUPS + sf.MASK_DUPS]
        + list(fv.DROP_FLAG_DEAD)))


def _write_bucket(dataset_root, *, train_vals, train_mask, vs_val: float):
    """One user_bucket=0/features.parquet with TRAIN rows for the M1 feature under test
    (log1p_time_since_prev_session_last_event_s / prev_session_available) plus a handful of
    filler VS rows whose M1 value is `vs_val` (used to probe the fit-on-VS trap)."""
    cols = _all_columns()
    n_train = len(train_vals)
    n_vs = 2
    n = n_train + n_vs
    split = np.array(["TRAIN"] * n_train + ["VALIDATION_SELECT"] * n_vs, dtype=object)

    data = {c: np.zeros(n, dtype=np.float64) for c in cols if c != "split"}
    data["split"] = split
    data["local_weekday_idx"] = np.full(n, 3, dtype=np.int64)
    data["daypart_idx"] = np.full(n, 4, dtype=np.int64)

    f1 = "log1p_time_since_prev_session_last_event_s"
    m1v = "prev_session_available"
    f2 = "log1p_time_since_prev_user_event_s"
    m2v = "prev_user_event_available"
    data[f1][:n_train] = train_vals
    data[m1v][:n_train] = train_mask
    data[f1][n_train:] = vs_val
    data[m1v][n_train:] = 1.0
    # feature 2: keep simple/constant/available so it never degenerates
    data[f2][:] = 6.0
    data[m2v][:] = 1.0
    # AFFINE pairs held exactly consistent (not under test here)
    data["log1p_prefix_elapsed_s"][:] = 2.0
    data["log1p_segment_elapsed_s"][:] = 2.0
    data["prefix_item_count"][:] = 5.0
    data["prefix_representative_count"][:] = 4.0
    # mask-dup columns consistent with each other and with feature 2's mask
    data["has_history_mask"][:] = 1.0
    data["has_history"][:] = 1.0
    data[m2v][:] = 1.0

    table = pa.table({c: pa.array(data[c]) if c != "split" else pa.array(list(split)) for c in cols})
    bdir = dataset_root / "model_ready" / "user_bucket=0"
    bdir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, bdir / "features.parquet")
    return f1, m1v


def test_fit_on_vs_trap_valid_only_excludes_unavailable_rows(tmp_path, set_fo_root):
    dataset_root = tmp_path / "dataset"
    root1 = tmp_path / "root1"
    root1.mkdir()
    set_fo_root(root1)

    train_vals = [5.0, 6.0, 7.0, 100.0, 100.0, 100.0]   # last 3 are "unavailable" (garbage raw)
    train_mask = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    f1, m1v = _write_bucket(dataset_root, train_vals=train_vals, train_mask=train_mask, vs_val=3.0)

    rc = sf.main(str(dataset_root), [0])
    assert rc == 0
    m1_fit = json.loads((root1 / "audit" / "M1_FIT.json").read_text())

    clip_hi = SPEC[f1]["clip_hi"]
    valid_clipped = np.clip(np.array(train_vals[:3]), SPEC[f1]["clip_lo"], clip_hi)
    expect_mean = float(valid_clipped.mean())
    expect_std = float(valid_clipped.std())
    got = m1_fit["features"][f1]
    assert got["valid_mean"] == pytest.approx(expect_mean, abs=1e-9)
    assert got["valid_std"] == pytest.approx(expect_std, abs=1e-9)

    # negative control: the "all rows" (including unavailable, clipped-garbage) statistic moves
    # far away from the valid-only mean -- proving valid-only really did exclude those rows.
    all_mean = got["train_all_rows"]["mean"]
    assert abs(all_mean - expect_mean) > 1.0, (
        f"negative control failed: all-rows mean {all_mean} did not move away from valid-only mean {expect_mean}")


def test_fit_on_vs_trap_extreme_vs_values_never_move_the_train_fit(tmp_path, set_fo_root):
    dataset_root = tmp_path / "dataset"
    train_vals = [5.0, 6.0, 7.0, 100.0, 100.0, 100.0]
    train_mask = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    f1, m1v = _write_bucket(dataset_root, train_vals=train_vals, train_mask=train_mask, vs_val=3.0)

    root1 = tmp_path / "root1"
    root1.mkdir()
    set_fo_root(root1)
    assert sf.main(str(dataset_root), [0]) == 0
    fit1 = json.loads((root1 / "audit" / "M1_FIT.json").read_text())["features"][f1]

    # Overwrite the SAME dataset with an EXTREME value in the VS rows only; TRAIN rows unchanged.
    _write_bucket(dataset_root, train_vals=train_vals, train_mask=train_mask, vs_val=9999.0)

    root2 = tmp_path / "root2"
    root2.mkdir()
    set_fo_root(root2)
    assert sf.main(str(dataset_root), [0]) == 0
    fit2 = json.loads((root2 / "audit" / "M1_FIT.json").read_text())["features"][f1]

    assert fit1["valid_mean"] == pytest.approx(fit2["valid_mean"], abs=1e-12)
    assert fit1["valid_std"] == pytest.approx(fit2["valid_std"], abs=1e-12)
    # VS drift-only stat, in contrast, DOES move: proves the VS extreme value was actually applied
    # and the fit's stability isn't just an accident of an unused column.
    vs1 = fit1["vs_valid_drift_only"]["mean"]
    vs2 = fit2["vs_valid_drift_only"]["mean"]
    assert abs(vs1 - vs2) > 1.0
