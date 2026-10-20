"""Synthetic unit test for profile_train_sample.py.

Builds a tiny fake `model_ready/user_bucket=<b>/features.parquet` partition (2 buckets)
with EXACTLY the column names the reused pipeline actually uses (see
`profile_train_sample.ALL_EXPECTED_COLUMNS`, sourced from full_store.py/config.py/
loaders.py), plus a matching `mappings/normalizers.json`, and checks:

  1. The script runs end-to-end (schema check passes, JSON is written, is valid JSON,
     contains no bare NaN/Infinity).
  2. Numbers are right: counts, exact-zero counts, min/max, and the redundancy-check
     outputs match hand-computed expected values.
  3. NEGATIVE CONTROL: one row deliberately violates the `price_zero -> price_missing`
     co-occurrence hypothesis, and the test asserts the checker correctly reports
     `exact_duplicate_on_sample: False` on the sample that contains it (proving the
     checker can detect a real break, not just rubber-stamp clean data).
  4. Structural counting (n_pad_rows) correctly flags an injected PAD-valued categorical
     cell even though real per-event storage should never contain one.
  5. `run_profile` refuses forbidden splits (VALIDATION_CONFIRM/TEST/LABEL_GRACE).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import profile_train_sample as pts  # noqa: E402


ALL_COLS = pts.ALL_EXPECTED_COLUMNS


_DEFAULT_U_S, _DEFAULT_U_C, _DEFAULT_L_S, _DEFAULT_L_C = None, None, None, None


def _row(**overrides) -> dict:
    """One well-formed synthetic event row; every expected column gets a safe default,
    then `overrides` replaces specific fields per test case."""
    r = {c: 0 for c in ALL_COLS}
    du_s, du_c, dl_s, dl_c = _apply_exact_hour_rotation(0.0)  # theta=0 -> consistent +3h pair
    r.update({
        "user_id": "u1", "segment_index": 0, "position_in_segment": 1, "decision_id": 1,
        "split": "TRAIN",
        "product_idx": 5, "category_idx": 5, "brand_idx": 5, "main_category_idx": 5,
        "daypart_idx": 3, "local_weekday_idx": 3,
        "log1p_price": 5.0, "price_missing": 0, "price_zero": 0, "price_conflict": 0,
        "brand_missing": 0, "brand_recovered": 0, "brand_conflict": 0,
        "category_missing": 0, "category_recovered": 0, "category_conflict": 0,
        "category_code_missing": 0, "category_code_recovered": 0, "main_category_missing": 0,
        "prev_gap_unavailable": 0, "item_age_unavailable": 0, "simultaneous_block": 0,
        "utc_hour_sin": du_s, "utc_hour_cos": du_c, "utc_weekday_sin": 0.0, "utc_weekday_cos": 1.0,
        "local_hour_sin": dl_s, "local_hour_cos": dl_c, "local_weekday_sin": 0.0, "local_weekday_cos": 1.0,
        "is_weekend": 0.0,
        "log1p_prev_raw_view_gap_s": 1.0, "log1p_segment_elapsed_s": 2.0,
        "log1p_item_observed_age_s": 3.0, "prefix_representative_count": 4.0,
        "prefix_item_count": 5.0, "prefix_known_category_count": 1.0,
        "prefix_category_switch_count": 0.0, "current_known_category_run_length": 2.0,
        "log1p_prefix_elapsed_s": 2.0,
        "log1p_prev_view_count": 1.0, "log1p_prev_session_count": 1.0,
        "recent_history_length": 10.0, "has_history": 1.0, "recent_distinct_items": 3.0,
        "recent_distinct_known_main_categories": 2.0, "recent_category_match_share": 0.5,
        "recent_brand_match_share": 0.5, "recent_mean_log_price": 4.5,
        "current_minus_recent_mean_log_price": 0.5, "log1p_time_since_prev_user_event_s": 1.0,
        "log1p_time_since_prev_session_last_event_s": 1.0,
        "has_history_mask": 1.0, "recent_category_match_supported": 1.0,
        "recent_brand_match_supported": 1.0, "recent_price_supported": 1.0,
        "prev_user_event_available": 1.0, "prev_session_available": 1.0,
    })
    r.update(overrides)
    return r


def _apply_exact_hour_rotation(local_hour_from_utc_hour_frac):
    """Given a UTC hour-of-day fraction in [0,24), returns (u_sin,u_cos,l_sin,l_cos) that
    are an EXACT +3h rotation (no boundary effects for the hour cycle -- it always wraps
    cleanly mod 24)."""
    theta = 2 * math.pi * local_hour_from_utc_hour_frac / 24
    phi = 2 * math.pi * 3 / 24
    return (math.sin(theta), math.cos(theta), math.sin(theta + phi), math.cos(theta + phi))


def build_dataset(root: Path) -> None:
    rows_b0 = []
    rows_b1 = []

    # --- bucket 0: TRAIN rows, including the negative control and the redundancy cases ---
    # Row A: perfectly clean row, all redundancy hypotheses hold.
    u_s, u_c, l_s, l_c = _apply_exact_hour_rotation(10.0)
    rows_b0.append(_row(decision_id=1, log1p_price=5.0, price_missing=0, price_zero=0,
                         log1p_segment_elapsed_s=2.0, log1p_prefix_elapsed_s=2.0,
                         prefix_representative_count=4.0, prefix_item_count=5.0,
                         utc_hour_sin=u_s, utc_hour_cos=u_c, local_hour_sin=l_s, local_hour_cos=l_c))

    # Row B: NEGATIVE CONTROL -- price_zero=1 but price_missing=0 (violates the documented
    # 100%-co-occurrence hypothesis). Everything else stays clean.
    u_s2, u_c2, l_s2, l_c2 = _apply_exact_hour_rotation(14.0)
    rows_b0.append(_row(decision_id=2, log1p_price=0.0, price_missing=0, price_zero=1,
                         log1p_segment_elapsed_s=3.0, log1p_prefix_elapsed_s=3.0,
                         prefix_representative_count=1.0, prefix_item_count=2.0,
                         utc_hour_sin=u_s2, utc_hour_cos=u_c2, local_hour_sin=l_s2, local_hour_cos=l_c2))

    # Row C: cold user (has_history=0) -- every user_context/mask value must read 0.
    rows_b0.append(_row(decision_id=3, has_history=0.0, has_history_mask=0.0,
                         log1p_prev_view_count=0.0, log1p_prev_session_count=0.0,
                         recent_history_length=0.0, recent_distinct_items=0.0,
                         recent_distinct_known_main_categories=0.0,
                         recent_category_match_share=0.0, recent_category_match_supported=0.0,
                         recent_brand_match_share=0.0, recent_brand_match_supported=0.0,
                         recent_mean_log_price=0.0, recent_price_supported=0.0,
                         current_minus_recent_mean_log_price=0.0,
                         log1p_time_since_prev_user_event_s=0.0, prev_user_event_available=0.0,
                         log1p_time_since_prev_session_last_event_s=0.0, prev_session_available=0.0))

    # Row D: unknown category (main_category_idx == MISSING=1) with a NONZERO
    # current_known_category_run_length raw value forced to 0 per the documented formula.
    rows_b0.append(_row(decision_id=4, main_category_idx=1, main_category_missing=1,
                         prefix_known_category_count=0.0, prefix_category_switch_count=0.0,
                         current_known_category_run_length=0.0))

    # Row E: OOV product/category/brand tokens, but main_category_idx stays KNOWN so this
    # row does NOT also count toward the "unknown category" subgroup (kept independent of
    # row D's unknown-category case).
    rows_b0.append(_row(decision_id=5, product_idx=2, category_idx=2, brand_idx=2,
                         main_category_idx=5))

    # Row F: injected PAD-valued categorical cell (structural negative control -- real
    # per-event storage should never contain this; the profiler must still count it
    # correctly rather than silently skip it).
    rows_b0.append(_row(decision_id=6, product_idx=0))

    # Row G: a VALIDATION_SELECT row (must be excluded when profiling split="TRAIN").
    rows_b0.append(_row(decision_id=7, split="VALIDATION_SELECT", log1p_price=6.0))

    # Row H: a TEST row -- must never be touched; included to prove split filtering keeps
    # it out even though it is physically present in the same file.
    rows_b0.append(_row(decision_id=8, split="TEST", log1p_price=999.0))

    # --- bucket 1: a couple more TRAIN rows so buckets=[0,1] aggregates across files ---
    u_s3, u_c3, l_s3, l_c3 = _apply_exact_hour_rotation(20.0)
    rows_b1.append(_row(decision_id=9, log1p_price=5.5, log1p_segment_elapsed_s=1.0,
                        log1p_prefix_elapsed_s=1.0, prefix_representative_count=0.0,
                        prefix_item_count=1.0, utc_hour_sin=u_s3, utc_hour_cos=u_c3,
                        local_hour_sin=l_s3, local_hour_cos=l_c3))
    rows_b1.append(_row(decision_id=10, log1p_price=4.5))

    for b, rows in ((0, rows_b0), (1, rows_b1)):
        d = root / "model_ready" / f"user_bucket={b}"
        d.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, d / "features.parquet")

    mappings = root / "mappings"
    mappings.mkdir(parents=True, exist_ok=True)
    normalizers = {
        "fit_scope": "TRAIN", "clip_quantiles": [0.001, 0.999],
        "features": {
            # mean/std chosen so z(0) is a nonzero, hand-checkable value for the
            # unknown-category test (current_known_category_run_length etc).
            "prefix_known_category_count": {"clip_lo": 0.0, "clip_hi": 5.0, "mean": 1.0, "std": 2.0, "constant": False},
            "prefix_category_switch_count": {"clip_lo": 0.0, "clip_hi": 7.0, "mean": 0.5, "std": 1.0, "constant": False},
            "current_known_category_run_length": {"clip_lo": 0.0, "clip_hi": 75.0, "mean": 3.0, "std": 2.0, "constant": False},
        },
    }
    (mappings / "normalizers.json").write_text(json.dumps(normalizers), encoding="utf-8")


@pytest.fixture()
def dataset_root(tmp_path: Path) -> Path:
    build_dataset(tmp_path)
    return tmp_path


def test_schema_check_passes(dataset_root: Path) -> None:
    report = pts.verify_schema(pts._bucket_path(dataset_root, 0))
    assert report["unused_extra_columns_present"] == []


def test_schema_mismatch_fails_loudly(tmp_path: Path) -> None:
    d = tmp_path / "model_ready" / "user_bucket=0"
    d.mkdir(parents=True)
    # Missing almost every expected column on purpose.
    pq.write_table(pa.table({"split": ["TRAIN"], "product_idx": [3]}), d / "features.parquet")
    with pytest.raises(pts.SchemaMismatchError):
        pts.verify_schema(pts._bucket_path(tmp_path, 0))


def test_forbidden_splits_refused(dataset_root: Path) -> None:
    for bad in ("TEST", "VALIDATION_CONFIRM", "LABEL_GRACE"):
        with pytest.raises(pts.ForbiddenSplitError):
            pts.run_profile(dataset_root, [0, 1], ["TRAIN", bad], distinct_cap=100)


def test_run_profile_end_to_end_and_json_strictness(dataset_root: Path, tmp_path: Path) -> None:
    result = pts.run_profile(dataset_root, [0, 1], ["TRAIN"], distinct_cap=100)
    out_path = tmp_path / "out.json"
    pts._dump_json(result, out_path)  # must not raise (allow_nan=False)
    text = out_path.read_text(encoding="utf-8")
    reloaded = json.loads(text)  # must be strictly valid JSON
    assert "NaN" not in text.split('"')[0::2]  # no bare (non-string) NaN token
    assert reloaded["label"] == "SAMPLE_NOT_POPULATION"

    train = result["per_split"]["TRAIN"]
    # 8 TRAIN rows total across both buckets (6 in bucket0 excluding the VS/TEST rows, +2 in bucket1)
    price_stat = next(f for f in train["numeric"] if f["feature"] == "log1p_price")
    assert price_stat["count"] == 8


def test_price_zero_negative_control_detected(dataset_root: Path) -> None:
    result = pts.run_profile(dataset_root, [0, 1], ["TRAIN"], distinct_cap=100)
    chk = result["per_split"]["TRAIN"]["redundancy_checks"]["price_zero_vs_missing"]
    # Row B set price_zero=1, price_missing=0 -> "either" > "both", hypothesis must FAIL.
    assert chk["n_either_flag"] == 1
    assert chk["n_both_flags"] == 0
    assert chk["exact_duplicate_on_sample"] is False


def test_segment_vs_prefix_elapsed_holds_on_clean_rows(dataset_root: Path) -> None:
    result = pts.run_profile(dataset_root, [0, 1], ["TRAIN"], distinct_cap=100)
    chk = result["per_split"]["TRAIN"]["redundancy_checks"]["segment_vs_prefix_elapsed"]
    # Every TRAIN row in the fixture sets these two fields equal.
    assert chk["n_rows_differing"] == 0
    assert chk["exact_duplicate_on_sample"] is True


def test_prefix_item_vs_representative_count_affine(dataset_root: Path) -> None:
    result = pts.run_profile(dataset_root, [0, 1], ["TRAIN"], distinct_cap=100)
    chk = result["per_split"]["TRAIN"]["redundancy_checks"]["prefix_item_vs_representative_count"]
    assert chk["n_rows_violating"] == 0
    assert chk["affine_duplicate_on_sample"] is True


def test_hour_rotation_exact_on_clean_rows(dataset_root: Path) -> None:
    result = pts.run_profile(dataset_root, [0, 1], ["TRAIN"], distinct_cap=100)
    chk = result["per_split"]["TRAIN"]["redundancy_checks"]["local_hour_rotation_of_utc_hour"]
    assert chk["exact_rotation_on_sample"] is True
    assert chk["max_residual"] < 1e-9


def test_categorical_pad_and_oov_counted(dataset_root: Path) -> None:
    result = pts.run_profile(dataset_root, [0, 1], ["TRAIN"], distinct_cap=100)
    cat = {c["feature"]: c for c in result["per_split"]["TRAIN"]["categorical"]}
    product = cat["product_idx"]
    assert product["n_pad_rows"] == 1   # row F
    assert product["n_oov"] == 1        # row E
    main_cat = cat["main_category_idx"]
    assert main_cat["n_missing"] == 1   # row D


def test_cold_user_vector_collapses_to_zero(dataset_root: Path) -> None:
    result = pts.run_profile(dataset_root, [0, 1], ["TRAIN"], distinct_cap=100)
    cold = result["per_split"]["TRAIN"]["cold_user"]
    assert cold["n_cold_rows"] == 1
    for feat, stat in cold["raw"].items():
        assert stat["max"] == 0.0, f"{feat} raw value not 0 for the cold-user row"


def test_unknown_category_encoded_is_nonzero(dataset_root: Path) -> None:
    result = pts.run_profile(dataset_root, [0, 1], ["TRAIN"], distinct_cap=100)
    unk = result["per_split"]["TRAIN"]["unknown_category"]
    assert unk["n_unknown_category_rows"] == 1
    raw = unk["raw"]["current_known_category_run_length"]
    enc = unk["encoded"]["current_known_category_run_length"]
    assert raw["max"] == 0.0
    # normalizers.json: mean=3.0, std=2.0 -> z(0) = (0-3)/2 = -1.5, NOT 0.
    assert enc["max"] == pytest.approx(-1.5)
    assert enc["max"] != 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
