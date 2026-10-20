"""profile_train_sample.py -- read-only, deterministic, bounded profiler for the A1
(rich_frozen_channels) SASRec feature set.

Owner: Sonnet A, SASREC_FEATURE_OPT_V1 audit (see
`review/packets/SONNET_A_INVENTORY.md`). Written to be RUN LATER, on the server, by the
study lead, against the real frozen dataset. It is NOT run against real data here -- only
against a tiny synthetic fixture in `tests/test_profile_train_sample.py`.

What it reads
-------------
`<dataset_root>/model_ready/user_bucket=<b>/features.parquet` for a small, DECLARED,
deterministic set of buckets (`--buckets`, default "0,17,42" = 3 buckets), for split
TRAIN and, for drift only, VALIDATION_SELECT. NEVER VALIDATION_CONFIRM or TEST -- those
splits are hard-refused (see `_FORBIDDEN_SPLITS`).

Column names and the frozen layout are NOT guessed. They are taken from the reused,
byte-identical-to-source code actually shipped to the trainer:
  - `evidence/snap/reuse/full_store.py:158-174` (`_needed_feature_columns`) for exactly
    which columns full_store.py itself projects for channels="F4".
  - `evidence/snap/reuse/vendor/rees46_eda/config.py:353-460` for the ordered feature-name
    lists (EVENT_NUMERIC_FEATURES, EVENT_QUALITY_FLAGS, USER_CONTEXT_FEATURES,
    USER_CONTEXT_MASKS, SESSION_PREFIX_FEATURES, NORMALIZED_*).
  - `evidence/snap/reuse/vendor/rees46_eda/loaders.py:38-63` (IDX_COLUMNS, MISSING_FLAG_OF,
    USER_MISSING_FLAG_OF).
This file was verified byte-identical (sha256) to `next_item_eda/src/rees46_eda/{loaders,
config}.py` during the audit (see `audit/FEATURE_INVENTORY.csv` notes column). If the
schema this script finds on disk does not match these names EXACTLY, it FAILS LOUDLY
(`SchemaMismatchError`) rather than silently skipping or guessing a column.

Design constraints (all from the task packet)
----------------------------------------------
- Single-threaded: OMP/MKL/OPENBLAS/NUMEXPR/POLARS thread env vars are pinned to "1"
  BEFORE numpy/pyarrow are imported (see the top of this file, ahead of any heavy import).
- <=400 MiB RSS target: every statistic is computed ONE (feature, mask) column pair at a
  time, projected directly off disk with pyarrow column projection, filtered to the
  declared split, reduced to scalars/small arrays, then discarded before the next column
  is read. No full multi-column table for all declared buckets is ever held at once.
- Every statistic is labelled SAMPLE, never population-exact, in the output JSON's
  top-level `"label"` field and per-block `"scope"` field.
- Output is strict JSON: `json.dump(..., allow_nan=False)` so a NaN/Infinity that slips
  through un-converted raises instead of producing invalid JSON; every float that CAN be
  non-finite is passed through `_finite_or_tag` first, turning NaN/+Inf/-Inf into an
  explicit string tag rather than a bare (invalid-JSON) literal.
"""
from __future__ import annotations

import os

# Must happen before numpy/pyarrow/polars are imported anywhere in the process.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "POLARS_MAX_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

try:
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
except Exception:
    pass

SAMPLE_LABEL = "SAMPLE_NOT_POPULATION"
_ALLOWED_SPLITS = ("TRAIN", "VALIDATION_SELECT")
_FORBIDDEN_SPLITS = ("VALIDATION_CONFIRM", "TEST", "LABEL_GRACE")

# ---------------------------------------------------------------------------------
# Frozen column layout, copied verbatim from the reused/verified source (see file
# docstring for exact citations). DO NOT hand-edit without re-citing the source line.
# ---------------------------------------------------------------------------------
EVENT_NUMERIC_FEATURES = [
    "log1p_price", "utc_hour_sin", "utc_hour_cos", "utc_weekday_sin", "utc_weekday_cos",
    "local_hour_sin", "local_hour_cos", "local_weekday_sin", "local_weekday_cos",
    "is_weekend", "log1p_prev_raw_view_gap_s", "log1p_segment_elapsed_s",
    "log1p_item_observed_age_s", "prefix_representative_count",
]  # config.py:353-371
SESSION_PREFIX_FEATURES = [
    "prefix_item_count", "prefix_known_category_count", "prefix_category_switch_count",
    "current_known_category_run_length", "log1p_prefix_elapsed_s",
]  # config.py:424-429
EVENT_NUMERIC_BLOCK = EVENT_NUMERIC_FEATURES + SESSION_PREFIX_FEATURES  # loaders.py:52 (19)

EVENT_QUALITY_FLAGS = [
    "price_missing", "price_zero", "price_conflict", "brand_missing", "brand_recovered",
    "brand_conflict", "category_missing", "category_recovered", "category_conflict",
    "category_code_missing", "category_code_recovered", "main_category_missing",
    "prev_gap_unavailable", "item_age_unavailable", "simultaneous_block",
]  # config.py:382-398 (15)

USER_CONTEXT_FEATURES = [
    "log1p_prev_view_count", "log1p_prev_session_count", "recent_history_length",
    "has_history", "recent_distinct_items", "recent_distinct_known_main_categories",
    "recent_category_match_share", "recent_brand_match_share", "recent_mean_log_price",
    "current_minus_recent_mean_log_price", "log1p_time_since_prev_user_event_s",
    "log1p_time_since_prev_session_last_event_s",
]  # config.py:400-413 (12)
USER_CONTEXT_MASKS = [
    "has_history_mask", "recent_category_match_supported", "recent_brand_match_supported",
    "recent_price_supported", "prev_user_event_available", "prev_session_available",
]  # config.py:415-422 (6)

CATEGORICAL_COLUMNS = [
    "product_idx", "category_idx", "brand_idx", "main_category_idx",
    "daypart_idx", "local_weekday_idx",
]  # loaders.py:38-49 IDX_COLUMNS keys (storage names)

NORMALIZED_EVENT_NUMERIC = [
    "log1p_price", "log1p_prev_raw_view_gap_s", "log1p_segment_elapsed_s",
    "log1p_item_observed_age_s", "prefix_representative_count",
]  # config.py:434-440
NORMALIZED_SESSION_PREFIX = [
    "prefix_item_count", "prefix_known_category_count", "prefix_category_switch_count",
    "current_known_category_run_length", "log1p_prefix_elapsed_s",
]  # config.py:454-460
NORMALIZED_USER_CONTEXT = [
    "log1p_prev_view_count", "log1p_prev_session_count", "recent_history_length",
    "recent_distinct_items", "recent_distinct_known_main_categories",
    "recent_mean_log_price", "current_minus_recent_mean_log_price",
    "log1p_time_since_prev_user_event_s", "log1p_time_since_prev_session_last_event_s",
]  # config.py:442-452

MISSING_FLAG_OF = {
    "log1p_price": "price_missing_or_conflict",  # special: OR of price_missing/price_conflict
    "log1p_prev_raw_view_gap_s": "prev_gap_unavailable",
    "log1p_item_observed_age_s": "item_age_unavailable",
}  # loaders.py:55-59
USER_MISSING_FLAG_OF = {
    "recent_mean_log_price": "recent_price_supported",
    "current_minus_recent_mean_log_price": "recent_price_supported",
}  # loaders.py:60-63

# Base columns full_store.py always projects (needed for split filter / joins / ordering);
# `split` is required here even though it is not itself a model feature.
BASE_COLUMNS = ["user_id", "segment_index", "position_in_segment", "decision_id", "split",
                "product_idx", "log1p_prev_view_count"]  # full_store.py:159-160

ALL_EXPECTED_COLUMNS = sorted(set(
    BASE_COLUMNS + ["category_idx", "brand_idx", "main_category_idx", "daypart_idx",
                    "local_weekday_idx"]
    + EVENT_NUMERIC_BLOCK + EVENT_QUALITY_FLAGS + USER_CONTEXT_FEATURES + USER_CONTEXT_MASKS
))  # == full_store.py:_needed_feature_columns("F4") deduplicated (37 columns)

RESERVED_PAD, RESERVED_MISSING, RESERVED_OOV, RESERVED_KNOWN_START = 0, 1, 2, 3


class SchemaMismatchError(RuntimeError):
    """Raised when the on-disk parquet schema does not exactly match the frozen,
    reused column layout this profiler was told to expect. Never silently skipped."""


class ForbiddenSplitError(RuntimeError):
    """Raised if a caller asks this profiler to touch VALIDATION_CONFIRM/TEST/LABEL_GRACE."""


# ---------------------------------------------------------------------------------
# JSON-safety helpers
# ---------------------------------------------------------------------------------

def _finite_or_tag(x: float) -> Any:
    """Map a possibly-nonfinite float to a strict-JSON-safe value: finite floats pass
    through as native floats (via round-trip through float()), non-finite values become
    an explicit string tag so `json.dump(..., allow_nan=False)` never has to reject a
    silently-produced NaN/Infinity literal."""
    x = float(x)
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "+Inf" if x > 0 else "-Inf"
    return x


def _dump_json(obj: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, allow_nan=False, sort_keys=False)


# ---------------------------------------------------------------------------------
# Schema verification
# ---------------------------------------------------------------------------------

def verify_schema(bucket_path: Path) -> dict:
    schema = pq.read_schema(bucket_path)
    names = set(schema.names)
    missing = [c for c in ALL_EXPECTED_COLUMNS if c not in names]
    if missing:
        raise SchemaMismatchError(
            f"{bucket_path}: missing expected column(s) {missing}. Refusing to guess a "
            "replacement name -- this profiler must be re-pointed at the correct frozen "
            "layout or the reused source's column lists must be re-checked."
        )
    extra = sorted(names - set(ALL_EXPECTED_COLUMNS))
    return {"checked_against_columns": ALL_EXPECTED_COLUMNS, "unused_extra_columns_present": extra}


def _bucket_path(dataset_root: Path, bucket: int) -> Path:
    return dataset_root / "model_ready" / f"user_bucket={bucket}" / "features.parquet"


# ---------------------------------------------------------------------------------
# Streaming accumulators (column-at-a-time; nothing multi-bucket is held in RAM)
# ---------------------------------------------------------------------------------

@dataclass
class NumericAccumulator:
    count: int = 0
    n_missing_by_mask: int = 0
    n_exact_zero: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    vmin: float = math.inf
    vmax: float = -math.inf
    values_for_quantiles: list = field(default_factory=list)  # bounded: <=3 buckets
    distinct: set = field(default_factory=set)
    distinct_capped: bool = False
    distinct_cap: int = 2000

    def add(self, valid_values: np.ndarray, missing_mask: Optional[np.ndarray]) -> None:
        self.count += int(valid_values.shape[0])
        if missing_mask is not None:
            self.n_missing_by_mask += int(missing_mask.sum())
        self.n_exact_zero += int((valid_values == 0.0).sum())
        if valid_values.size:
            self.total += float(valid_values.sum())
            self.total_sq += float(np.square(valid_values, dtype=np.float64).sum())
            self.vmin = min(self.vmin, float(valid_values.min()))
            self.vmax = max(self.vmax, float(valid_values.max()))
            self.values_for_quantiles.append(valid_values)
        if not self.distinct_capped:
            for v in np.unique(valid_values)[: max(0, self.distinct_cap - len(self.distinct) + 1)]:
                self.distinct.add(float(v))
                if len(self.distinct) > self.distinct_cap:
                    self.distinct_capped = True
                    break

    def finalize(self) -> dict:
        n = self.count
        mean = self.total / n if n else float("nan")
        var = (self.total_sq / n - mean * mean) if n else float("nan")
        std = math.sqrt(var) if (n and var >= 0) else float("nan")
        if self.values_for_quantiles:
            allv = np.concatenate(self.values_for_quantiles)
            qs = np.quantile(allv, [0.001, 0.01, 0.25, 0.5, 0.75, 0.99, 0.999])
        else:
            qs = [float("nan")] * 7
        n_distinct = len(self.distinct)
        return {
            "count": n,
            "missing_by_mask": self.n_missing_by_mask,
            "exact_zeros": self.n_exact_zero,
            "min": _finite_or_tag(self.vmin) if n else None,
            "q001": _finite_or_tag(qs[0]), "q01": _finite_or_tag(qs[1]),
            "q25": _finite_or_tag(qs[2]), "q50": _finite_or_tag(qs[3]),
            "q75": _finite_or_tag(qs[4]), "q99": _finite_or_tag(qs[5]),
            "q999": _finite_or_tag(qs[6]),
            "max": _finite_or_tag(self.vmax) if n else None,
            "mean": _finite_or_tag(mean), "std": _finite_or_tag(std),
            "n_distinct": (f">={self.distinct_cap}" if self.distinct_capped else n_distinct),
            "constant": bool(n and n_distinct <= 1),
        }


def _project_and_filter(bucket_path: Path, split: str, columns: list) -> pa.Table:
    cols = list(dict.fromkeys(columns + ["split"]))
    tbl = pq.read_table(bucket_path, columns=cols)
    keep = pc.equal(tbl["split"], split)
    return tbl.filter(keep)


def _missing_mask_for(tbl: pa.Table, feature_name: str, flag_map: dict) -> Optional[np.ndarray]:
    flag = flag_map.get(feature_name)
    if flag is None:
        return None
    if flag == "price_missing_or_conflict":
        pm = tbl["price_missing"].to_numpy(zero_copy_only=False).astype(bool)
        pc_ = tbl["price_conflict"].to_numpy(zero_copy_only=False).astype(bool)
        return pm | pc_
    return tbl[flag].to_numpy(zero_copy_only=False).astype(bool)


def profile_numeric_column(dataset_root: Path, buckets: list, split: str,
                            feature_name: str, flag_map: dict, group: str,
                            distinct_cap: int) -> dict:
    acc = NumericAccumulator(distinct_cap=distinct_cap)
    needed_flags = []
    fl = flag_map.get(feature_name)
    if fl == "price_missing_or_conflict":
        needed_flags = ["price_missing", "price_conflict"]
    elif fl is not None:
        needed_flags = [fl]
    for b in buckets:
        path = _bucket_path(dataset_root, b)
        tbl = _project_and_filter(path, split, [feature_name] + needed_flags)
        if tbl.num_rows == 0:
            continue
        vals = tbl[feature_name].to_numpy(zero_copy_only=False).astype(np.float64)
        mmask = _missing_mask_for(tbl, feature_name, flag_map)
        # "VALID values only": raw per-event rows never carry PAD (PAD is a batching-time
        # artifact of the windowed store, not a row in features.parquet -- see file
        # docstring), so every row read here already qualifies as VALID at this grain.
        acc.add(vals, mmask)
    out = acc.finalize()
    out["feature"] = feature_name
    out["batch_group"] = group
    out["scope"] = SAMPLE_LABEL
    return out


def profile_flag_or_mask_column(dataset_root: Path, buckets: list, split: str,
                                 name: str, group: str, distinct_cap: int) -> dict:
    acc = NumericAccumulator(distinct_cap=distinct_cap)
    for b in buckets:
        path = _bucket_path(dataset_root, b)
        tbl = _project_and_filter(path, split, [name])
        if tbl.num_rows == 0:
            continue
        vals = tbl[name].to_numpy(zero_copy_only=False).astype(np.float64)
        acc.add(vals, None)
    out = acc.finalize()
    out["feature"] = name
    out["batch_group"] = group
    out["scope"] = SAMPLE_LABEL
    out["rate"] = _finite_or_tag(out["mean"]) if isinstance(out["mean"], (int, float)) else out["mean"]
    return out


def profile_categorical_column(dataset_root: Path, buckets: list, split: str,
                                name: str) -> dict:
    total = 0
    n_pad = n_missing = n_oov = n_known = 0
    for b in buckets:
        path = _bucket_path(dataset_root, b)
        tbl = _project_and_filter(path, split, [name])
        if tbl.num_rows == 0:
            continue
        vals = tbl[name].to_numpy(zero_copy_only=False)
        total += vals.shape[0]
        n_pad += int((vals == RESERVED_PAD).sum())
        n_missing += int((vals == RESERVED_MISSING).sum())
        n_oov += int((vals == RESERVED_OOV).sum())
        n_known += int((vals >= RESERVED_KNOWN_START).sum())
    return {
        "feature": name, "scope": SAMPLE_LABEL, "count": total,
        "n_pad_rows": n_pad,  # expected 0: raw per-event rows are never PAD
        "n_missing": n_missing, "n_oov": n_oov, "n_known": n_known,
        "oov_rate": _finite_or_tag(n_oov / total) if total else None,
        "missing_rate": _finite_or_tag(n_missing / total) if total else None,
    }


# ---------------------------------------------------------------------------------
# Redundancy hypothesis checks (REDUNDANCY_FINDINGS.md items 1-5)
# ---------------------------------------------------------------------------------

def check_price_zero_vs_missing(dataset_root: Path, buckets: list, split: str) -> dict:
    n_either = n_both = 0
    for b in buckets:
        path = _bucket_path(dataset_root, b)
        tbl = _project_and_filter(path, split, ["price_zero", "price_missing"])
        if tbl.num_rows == 0:
            continue
        z = tbl["price_zero"].to_numpy(zero_copy_only=False).astype(bool)
        m = tbl["price_missing"].to_numpy(zero_copy_only=False).astype(bool)
        n_either += int((z | m).sum())
        n_both += int((z & m).sum())
    return {"hypothesis": "price_zero implies price_missing (and vice versa)",
            "scope": SAMPLE_LABEL, "n_either_flag": n_either, "n_both_flags": n_both,
            "exact_duplicate_on_sample": n_either == n_both}


def check_segment_vs_prefix_elapsed(dataset_root: Path, buckets: list, split: str) -> dict:
    n = 0
    max_abs_diff = 0.0
    n_diff = 0
    for b in buckets:
        path = _bucket_path(dataset_root, b)
        tbl = _project_and_filter(path, split, ["log1p_segment_elapsed_s", "log1p_prefix_elapsed_s"])
        if tbl.num_rows == 0:
            continue
        a = tbl["log1p_segment_elapsed_s"].to_numpy(zero_copy_only=False).astype(np.float64)
        c = tbl["log1p_prefix_elapsed_s"].to_numpy(zero_copy_only=False).astype(np.float64)
        d = np.abs(a - c)
        n += a.shape[0]
        max_abs_diff = max(max_abs_diff, float(d.max()) if d.size else 0.0)
        n_diff += int((d > 1e-9).sum())
    return {"hypothesis": "log1p_segment_elapsed_s == log1p_prefix_elapsed_s exactly",
            "scope": SAMPLE_LABEL, "count": n, "n_rows_differing": n_diff,
            "max_abs_diff": _finite_or_tag(max_abs_diff), "exact_duplicate_on_sample": n_diff == 0}


def check_prefix_item_vs_representative_count(dataset_root: Path, buckets: list, split: str) -> dict:
    n = 0
    n_not_plus_one = 0
    max_abs_diff = 0.0
    for b in buckets:
        path = _bucket_path(dataset_root, b)
        tbl = _project_and_filter(path, split, ["prefix_item_count", "prefix_representative_count"])
        if tbl.num_rows == 0:
            continue
        a = tbl["prefix_item_count"].to_numpy(zero_copy_only=False).astype(np.float64)
        r = tbl["prefix_representative_count"].to_numpy(zero_copy_only=False).astype(np.float64)
        d = np.abs((a - r) - 1.0)
        n += a.shape[0]
        max_abs_diff = max(max_abs_diff, float(d.max()) if d.size else 0.0)
        n_not_plus_one += int((d > 1e-9).sum())
    return {"hypothesis": "prefix_item_count == prefix_representative_count + 1 exactly",
            "scope": SAMPLE_LABEL, "count": n, "n_rows_violating": n_not_plus_one,
            "max_abs_diff_from_plus_one": _finite_or_tag(max_abs_diff),
            "affine_duplicate_on_sample": n_not_plus_one == 0}


def _rotation_residual(u_sin, u_cos, l_sin, l_cos, phi: float) -> np.ndarray:
    pred_sin = u_sin * math.cos(phi) + u_cos * math.sin(phi)
    pred_cos = u_cos * math.cos(phi) - u_sin * math.sin(phi)
    return np.sqrt((pred_sin - l_sin) ** 2 + (pred_cos - l_cos) ** 2)


def check_hour_rotation(dataset_root: Path, buckets: list, split: str) -> dict:
    phi = 2 * math.pi * 3 / 24  # CONTEXTUAL_TZ_OFFSET_HOURS = 3 (config.py:235)
    n = 0
    max_res = 0.0
    for b in buckets:
        path = _bucket_path(dataset_root, b)
        cols = ["utc_hour_sin", "utc_hour_cos", "local_hour_sin", "local_hour_cos"]
        tbl = _project_and_filter(path, split, cols)
        if tbl.num_rows == 0:
            continue
        arrs = [tbl[c].to_numpy(zero_copy_only=False).astype(np.float64) for c in cols]
        res = _rotation_residual(*arrs, phi=phi)
        n += res.shape[0]
        max_res = max(max_res, float(res.max()) if res.size else 0.0)
    return {"hypothesis": "local_hour_(sin,cos) is an exact fixed +3h rotation of utc_hour_(sin,cos)",
            "scope": SAMPLE_LABEL, "count": n, "max_residual": _finite_or_tag(max_res),
            "exact_rotation_on_sample": max_res < 1e-6}


def check_weekday_rotation(dataset_root: Path, buckets: list, split: str) -> dict:
    """local_weekday is computed from isodow(local_ts): on rows where the +3h shift does
    NOT cross local midnight, the local date equals the UTC date, so
    local_weekday_(sin,cos) is IDENTICAL to utc_weekday_(sin,cos) (zero rotation, an exact
    duplicate -- stronger than local_hour's fixed-phase rotation). On rows where the shift
    DOES cross local midnight (UTC hour in [21,24)), the local weekday is advanced by
    exactly one day, i.e. rotated by 2*pi/7 relative to the UTC pair. Both hypotheses are
    checked; every row is expected to satisfy exactly one of them."""
    phi_week = 2 * math.pi / 7
    n = 0
    n_identical = 0
    n_one_day_forward = 0
    for b in buckets:
        path = _bucket_path(dataset_root, b)
        cols = ["utc_weekday_sin", "utc_weekday_cos", "local_weekday_sin", "local_weekday_cos"]
        tbl = _project_and_filter(path, split, cols)
        if tbl.num_rows == 0:
            continue
        u_s, u_c, l_s, l_c = (tbl[c].to_numpy(zero_copy_only=False).astype(np.float64) for c in cols)
        res_identity = _rotation_residual(u_s, u_c, l_s, l_c, phi=0.0)
        res_plus_one = _rotation_residual(u_s, u_c, l_s, l_c, phi=phi_week)
        n += res_identity.shape[0]
        n_identical += int((res_identity < 1e-6).sum())
        n_one_day_forward += int(((res_identity >= 1e-6) & (res_plus_one < 1e-6)).sum())
    n_unexplained = n - n_identical - n_one_day_forward
    frac = (n_identical / n) if n else None
    return {"hypothesis": "local_weekday_(sin,cos) EQUALS utc_weekday_(sin,cos) exactly on rows "
                          "that do not cross local midnight, and is exactly one weekday-step "
                          "(2*pi/7) rotated forward on rows that do",
            "scope": SAMPLE_LABEL, "count": n, "n_identical_no_shift": n_identical,
            "n_one_day_forward_shift": n_one_day_forward, "n_unexplained": n_unexplained,
            "fraction_identical": _finite_or_tag(frac) if frac is not None else None,
            "expected_fraction_identical_approx": 21 / 24,
            "all_rows_explained": n_unexplained == 0,
            "note": "boundary rows (~3/24 of a day, UTC hour 21-24) are EXPECTED to be the "
                    "one-day-forward case by construction (FEATURE_DICTIONARY.md:33); this is "
                    "not a bug. n_unexplained>0 would indicate the hypothesis itself is wrong."}


# ---------------------------------------------------------------------------------
# Normalizer application (independent re-derivation from normalizers.json, not imported
# vendor code) -- used for the cold-user / unknown-category encoded-value tables.
# ---------------------------------------------------------------------------------

def load_normalizers(dataset_root: Path) -> dict:
    for rel in ("mappings/normalizers.json", "vocabularies/normalizers.json"):
        p = dataset_root / rel
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))["features"]
    raise FileNotFoundError(
        f"normalizers.json not found under {dataset_root}/mappings or /vocabularies -- "
        "refusing to guess fitted parameters."
    )


def _apply_normalizer(name: str, raw: np.ndarray, spec: dict) -> np.ndarray:
    p = spec.get(name)
    if p is None:
        return raw.astype(np.float64)
    if p.get("constant"):
        return np.zeros_like(raw, dtype=np.float64)
    y = np.clip(raw, p["clip_lo"], p["clip_hi"])
    return (y - p["mean"]) / p["std"]


def cold_user_encoded_vector(dataset_root: Path, buckets: list, split: str) -> dict:
    """Rows with has_history==0 (raw); reports the raw AND normalizer-encoded vector for
    every user_context feature + every user_context_masks entry, restricted to that
    subgroup, answering OPEN_QUESTIONS.md Q1."""
    spec = load_normalizers(dataset_root)
    cols = USER_CONTEXT_FEATURES + USER_CONTEXT_MASKS
    accs = {c: NumericAccumulator(distinct_cap=16) for c in cols}
    accs_encoded = {c: NumericAccumulator(distinct_cap=16) for c in cols}
    n_cold = 0
    for b in buckets:
        path = _bucket_path(dataset_root, b)
        tbl = _project_and_filter(path, split, cols)
        if tbl.num_rows == 0:
            continue
        has_hist = tbl["has_history"].to_numpy(zero_copy_only=False).astype(np.float64)
        cold = has_hist == 0.0
        if not cold.any():
            continue
        n_cold += int(cold.sum())
        for c in cols:
            raw = tbl[c].to_numpy(zero_copy_only=False).astype(np.float64)[cold]
            accs[c].add(raw, None)
            enc = raw if c in USER_CONTEXT_MASKS else _apply_normalizer(c, raw, spec)
            accs_encoded[c].add(enc, None)
    return {
        "scope": SAMPLE_LABEL, "n_cold_rows": n_cold,
        "raw": {c: accs[c].finalize() for c in cols},
        "encoded": {c: accs_encoded[c].finalize() for c in cols},
        "note": "encoded == normalizer(raw) for user_context features (masks pass through raw); "
                "a genuinely cold user is expected to collapse every column to a single constant "
                "(n_distinct<=1) both raw and encoded.",
    }


def unknown_category_encoded_values(dataset_root: Path, buckets: list, split: str) -> dict:
    """Rows where main_category_idx is MISSING(1) or OOV(2); reports raw+encoded values of
    the three category-derived counts that are NOT flag-gated, answering OPEN_QUESTIONS.md
    Q2 (raw 0 -> z(0), not literal 0.0, when the value is zero-but-not-flagged-missing)."""
    spec = load_normalizers(dataset_root)
    targets = ["prefix_known_category_count", "prefix_category_switch_count",
               "current_known_category_run_length"]
    accs_raw = {c: NumericAccumulator(distinct_cap=64) for c in targets}
    accs_enc = {c: NumericAccumulator(distinct_cap=64) for c in targets}
    n_unknown = 0
    total = 0
    for b in buckets:
        path = _bucket_path(dataset_root, b)
        tbl = _project_and_filter(path, split, ["main_category_idx"] + targets)
        if tbl.num_rows == 0:
            continue
        total += tbl.num_rows
        mc = tbl["main_category_idx"].to_numpy(zero_copy_only=False)
        unknown = (mc == RESERVED_MISSING) | (mc == RESERVED_OOV)
        if not unknown.any():
            continue
        n_unknown += int(unknown.sum())
        for c in targets:
            raw = tbl[c].to_numpy(zero_copy_only=False).astype(np.float64)[unknown]
            accs_raw[c].add(raw, None)
            accs_enc[c].add(_apply_normalizer(c, raw, spec), None)
    return {
        "scope": SAMPLE_LABEL, "n_unknown_category_rows": n_unknown, "n_total_rows": total,
        "unknown_category_rate": _finite_or_tag(n_unknown / total) if total else None,
        "raw": {c: accs_raw[c].finalize() for c in targets},
        "encoded": {c: accs_enc[c].finalize() for c in targets},
        "note": "z(0) = (0-mean)/std is expected to be NONZERO here -- these features are not "
                "flag-gated to zero-after-normalization (they are true counts, not missing "
                "sentinels; FEATURE_DICTIONARY.md:45-47).",
    }


# ---------------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------------

def run_profile(dataset_root: Path, buckets: list, splits: list, distinct_cap: int) -> dict:
    for s in splits:
        if s in _FORBIDDEN_SPLITS:
            raise ForbiddenSplitError(f"refusing to profile forbidden split {s!r}")
        if s not in _ALLOWED_SPLITS:
            raise ValueError(f"unknown split {s!r}; allowed: {_ALLOWED_SPLITS}")

    schema_report = verify_schema(_bucket_path(dataset_root, buckets[0]))
    for b in buckets[1:]:
        r = verify_schema(_bucket_path(dataset_root, b))
        if r["unused_extra_columns_present"] != schema_report["unused_extra_columns_present"]:
            schema_report["unused_extra_columns_present"] = sorted(
                set(schema_report["unused_extra_columns_present"])
                | set(r["unused_extra_columns_present"]))

    out: dict = {
        "label": SAMPLE_LABEL,
        "dataset_root": str(dataset_root),
        "buckets": buckets,
        "splits_profiled": splits,
        "schema_check": schema_report,
        "per_split": {},
    }

    for split in splits:
        block: dict = {"numeric": [], "quality_flags": [], "user_context": [],
                       "user_context_masks": [], "categorical": [],
                       "redundancy_checks": {}, "cold_user": None, "unknown_category": None}
        for name in EVENT_NUMERIC_BLOCK:
            block["numeric"].append(profile_numeric_column(
                dataset_root, buckets, split, name, MISSING_FLAG_OF, "event_numeric_features",
                distinct_cap))
        for name in EVENT_QUALITY_FLAGS:
            block["quality_flags"].append(profile_flag_or_mask_column(
                dataset_root, buckets, split, name, "event_quality_flags", distinct_cap))
        for name in USER_CONTEXT_FEATURES:
            block["user_context"].append(profile_numeric_column(
                dataset_root, buckets, split, name, USER_MISSING_FLAG_OF, "user_context",
                distinct_cap))
        for name in USER_CONTEXT_MASKS:
            block["user_context_masks"].append(profile_flag_or_mask_column(
                dataset_root, buckets, split, name, "user_context_masks", distinct_cap))
        for name in CATEGORICAL_COLUMNS:
            block["categorical"].append(profile_categorical_column(dataset_root, buckets, split, name))

        block["redundancy_checks"] = {
            "price_zero_vs_missing": check_price_zero_vs_missing(dataset_root, buckets, split),
            "segment_vs_prefix_elapsed": check_segment_vs_prefix_elapsed(dataset_root, buckets, split),
            "prefix_item_vs_representative_count": check_prefix_item_vs_representative_count(
                dataset_root, buckets, split),
            "local_hour_rotation_of_utc_hour": check_hour_rotation(dataset_root, buckets, split),
            "local_weekday_rotation_of_utc_weekday": check_weekday_rotation(dataset_root, buckets, split),
        }
        block["cold_user"] = cold_user_encoded_vector(dataset_root, buckets, split)
        block["unknown_category"] = unknown_category_encoded_values(dataset_root, buckets, split)
        out["per_split"][split] = block

    return out


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", required=True, type=Path,
                    help="Frozen dataset root containing model_ready/user_bucket=*/features.parquet "
                         "and mappings/normalizers.json.")
    ap.add_argument("--buckets", default="0,17,42",
                    help="Comma-separated deterministic bucket list (default 3 buckets).")
    ap.add_argument("--splits", default="TRAIN,VALIDATION_SELECT",
                    help="Comma-separated splits to profile. VALIDATION_CONFIRM/TEST/LABEL_GRACE "
                         "are hard-refused regardless of this argument.")
    ap.add_argument("--out", required=True, type=Path, help="Output JSON path.")
    ap.add_argument("--n-distinct-cap", type=int, default=2000,
                    help="Cap on exact distinct-value tracking per numeric column.")
    args = ap.parse_args(argv)

    buckets = [int(x) for x in args.buckets.split(",") if x.strip() != ""]
    splits = [s.strip() for s in args.splits.split(",") if s.strip() != ""]
    if not buckets:
        raise ValueError("--buckets must name at least one bucket")

    result = run_profile(args.dataset_root, buckets, splits, args.n_distinct_cap)
    _dump_json(result, args.out)
    print(f"[profile_train_sample] wrote {args.out} (label={SAMPLE_LABEL}, "
          f"buckets={buckets}, splits={splits})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
