"""T1 TaskExample adapter for the S1-PR-07 real-data Flower smoke.

This module is the only place that reads private REES46 task example files
for the federated smoke path. It produces opaque-client-only outputs; raw
user identities and raw label_value strings never leave this module.

Non-scientific: zero-history representation. Sequential GRU history is deferred
to the final model training lane.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import polars as pl
import torch

from ppsi.federated.clients import client_id_from_user
from ppsi.training.batch import (
    Phase1Batch,
    Phase1BatchSpec,
    validate_canonical_phase1_batch,
)
from ppsi.training.fixtures import default_batch_spec

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Required column names in T1 task example parquets.
_REQUIRED_COLUMNS = frozenset(
    {"client", "session", "decision_order", "label_value", "task_mask", "status", "cohort", "split"}
)

# Expected G1 SHA-256 prefixes (first 16 hex chars = 8 bytes).
_EXPECTED_SHA_PREFIX_COHORT = "32d4b8ce4bb84f78"
_EXPECTED_SHA_PREFIX_TRAIN = "42b1617c1d2f1b5a"
_EXPECTED_SHA_PREFIX_VALIDATION = "e5e8522510371741"

_EXPECTED_ROW_TRAIN = 3_113_814
_EXPECTED_ROW_VALIDATION = 438_185


# ---------------------------------------------------------------------------
# File verification
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_t1_task_example_file(
    path: Path | str,
    *,
    expected_sha_prefix: str,
    expected_row_count: int,
    expected_split: str,
    file_label: str,
    expected_category_count: int = 588,
) -> str:
    """Verify a private T1 task example parquet before use.

    Parameters
    ----------
    path
        Absolute or repo-relative path to the parquet file.
    expected_sha_prefix
        First 16 hex characters of the expected SHA-256 digest.
    expected_row_count
        Expected exact number of rows.
    expected_split
        Either ``"TRAIN"`` or ``"VALIDATION"``.
    file_label
        Human-readable label used in error messages.
    expected_category_count
        Expected category count for validating integral label codes.

    Returns
    -------
    str
        The full SHA-256 hex digest of the file.

    Raises
    ------
    FileNotFoundError
        If the file is missing.
    ValueError
        If the SHA-256 prefix, row count, or column/invariant checks fail.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Required T1 file missing: {path} ({file_label}). "
            "This file must exist before running the smoke."
        )

    digest = _sha256_file(path)
    if not digest.startswith(expected_sha_prefix):
        raise ValueError(
            f"{file_label}: SHA-256 prefix mismatch. "
            f"Expected prefix {expected_sha_prefix!r}, got {digest[: len(expected_sha_prefix)]!r} "
            f"(full: {digest})"
        )

    df = pl.read_parquet(path)
    actual_rows = len(df)
    if actual_rows != expected_row_count:
        raise ValueError(
            f"{file_label}: row count mismatch. "
            f"Expected {expected_row_count:,}, got {actual_rows:,}."
        )

    missing_cols = _REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"{file_label}: missing required columns: {sorted(missing_cols)}")

    # Explicit null checks on all required columns before equality checks
    required_null_check_cols = [
        "client",
        "session",
        "decision_order",
        "label_value",
        "task_mask",
        "status",
        "cohort",
        "split",
    ]
    for col_name in required_null_check_cols:
        null_count = df.filter(pl.col(col_name).is_null()).height
        if null_count > 0:
            raise ValueError(f"{file_label}: {null_count:,} rows have null {col_name}.")

    # Invariant assertions for this split (explicit rejection of mismatch and nulls)
    bad_split = df.filter((pl.col("split") != expected_split) | pl.col("split").is_null())
    if len(bad_split) > 0:
        raise ValueError(f"{file_label}: {len(bad_split):,} rows have split != {expected_split!r}.")
    bad_cohort = df.filter((pl.col("cohort") != "C1") | pl.col("cohort").is_null())
    if len(bad_cohort) > 0:
        raise ValueError(f"{file_label}: {len(bad_cohort):,} rows have cohort != 'C1'.")
    bad_mask = df.filter((pl.col("task_mask") != True) | pl.col("task_mask").is_null())
    if len(bad_mask) > 0:
        raise ValueError(f"{file_label}: {len(bad_mask):,} rows have task_mask != True.")
    bad_status = df.filter((pl.col("status") != "OBSERVED") | pl.col("status").is_null())
    if len(bad_status) > 0:
        raise ValueError(f"{file_label}: {len(bad_status):,} rows have status != 'OBSERVED'.")

    # Validate label_value values
    validate_and_convert_t1_labels(
        df,
        category_count=expected_category_count,
        split_label=file_label,
    )

    return digest


def load_t1_category_spec(vocab_path: Path | str) -> tuple[int, set[int]]:
    """Load categories from the frozen vocabulary JSON and validate dense codes.

    Validates:
        category_count = int(vocab["categories"]["count"]) == 588
        valid_codes = {int(v) for v in vocab["categories"]["code_of_category_id"].values()}
        valid_codes == set(range(category_count))

    Returns
    -------
    tuple[int, set[int]]
        (category_count, valid_codes)
    """
    vocab_path = Path(vocab_path)
    if not vocab_path.is_file():
        raise FileNotFoundError(f"Vocabulary file missing: {vocab_path}")
    with open(vocab_path, encoding="utf-8") as f:
        vocab = json.load(f)

    categories = vocab.get("categories", {})
    if "count" not in categories or "code_of_category_id" not in categories:
        raise ValueError(
            "Vocabulary JSON must contain categories.count and categories.code_of_category_id mapping."
        )

    category_count = int(categories["count"])
    if category_count != 588:
        raise ValueError(f"Expected category_count == 588, got {category_count}.")

    code_map = categories["code_of_category_id"]
    valid_codes = {int(v) for v in code_map.values()}
    if valid_codes != set(range(category_count)):
        raise ValueError(
            f"valid_codes != set(range({category_count})). Expected dense codes 0..{category_count - 1}."
        )

    return category_count, valid_codes


def verify_vocabulary(vocab_path: Path | str, *, expected_category_count: int = 588) -> str:
    """Verify the frozen vocabulary JSON and return its SHA-256."""
    vocab_path = Path(vocab_path)
    if not vocab_path.is_file():
        raise FileNotFoundError(f"Vocabulary file missing: {vocab_path}")
    digest = _sha256_file(vocab_path)
    cat_count, _ = load_t1_category_spec(vocab_path)
    if cat_count != expected_category_count:
        raise ValueError(
            f"Vocabulary category count mismatch: expected {expected_category_count}, got {cat_count}."
        )
    return digest


def validate_and_convert_t1_labels(
    df: pl.DataFrame,
    *,
    category_count: int = 588,
    valid_codes: set[int] | None = None,
    split_label: str = "T1",
) -> pl.DataFrame:
    """Validate T1 label_value column and return df with label_code: pl.Int64.

    Every T1 label_value:
    - must not be null
    - must be finite (not NaN, not Inf)
    - must be mathematically integral (val == floor(val))
    - converted with int(val)
    - require 0 <= code < category_count
    - require code in valid_codes

    Rejects: -1, 588, 60.5, NaN, Inf, null.
    No modulo. No fallback. No silent coercion. No label dropping.
    """
    if "label_value" not in df.columns:
        raise ValueError(f"{split_label}: missing required 'label_value' column.")

    # 1. Null check
    null_count = df.filter(pl.col("label_value").is_null()).height
    if null_count > 0:
        raise ValueError(f"{split_label}: {null_count:,} rows have null label_value.")

    # 2. Check on unique values (fast & thorough for any size dataset)
    unique_vals = df["label_value"].unique().to_list()
    expected_valid_codes = valid_codes if valid_codes is not None else set(range(category_count))

    for val in unique_vals:
        if val is None:
            raise ValueError(f"{split_label}: null label_value found.")
        try:
            f_val = float(val)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"{split_label}: cannot convert label_value {val!r} to float: {exc}"
            ) from exc

        if not math.isfinite(f_val):
            raise ValueError(f"{split_label}: non-finite label_value found: {val}")

        if f_val != math.floor(f_val):
            raise ValueError(
                f"{split_label}: non-integral label_value found: {val}. "
                "T1 label_value must be mathematically integral."
            )

        code = int(f_val)
        if not (0 <= code < category_count):
            raise ValueError(
                f"{split_label}: label code {code} out of range [0, {category_count - 1}]."
            )
        if code not in expected_valid_codes:
            raise ValueError(f"{split_label}: label code {code} not in valid_codes.")

    # 3. Add label_code column as Int64
    return df.with_columns(
        pl.col("label_value").cast(pl.Float64).cast(pl.Int64).alias("label_code")
    )


# ---------------------------------------------------------------------------
# Client mapping
# ---------------------------------------------------------------------------


def _build_client_id_map(df: pl.DataFrame) -> dict[Any, str]:
    """Map unique raw ``client`` values → opaque client IDs.

    Hashes only unique values, not all rows. Never persists the raw mapping.
    """
    unique_clients = df.select("client").unique()["client"].to_list()
    return {raw: client_id_from_user(str(raw)) for raw in unique_clients}


def load_t1_client_counts(
    train_path: Path | str,
    validation_path: Path | str,
    base_manifest_client_ids: set[str],
) -> pl.DataFrame:
    """Load T1 train and validation client example counts.

    Returns a DataFrame with columns:
        client_id, t1_train_example_count, t1_validation_example_count,
        eligible_for_t1_smoke

    Parameters
    ----------
    train_path
        Path to the T1 TRAIN parquet (already verified).
    validation_path
        Path to the T1 VALIDATION parquet (already verified).
    base_manifest_client_ids
        Set of opaque client IDs in the base #19 manifest.
        Every T1 client must be a member; fails if any is outside C1.

    Raises
    ------
    ValueError
        If any T1 client maps outside the base #19 manifest.
    """
    train_df = pl.read_parquet(train_path)
    val_df = pl.read_parquet(validation_path)

    # Build client maps (hash only unique values)
    train_map = _build_client_id_map(train_df)
    val_map = _build_client_id_map(val_df)

    # Merge maps and validate all clients are in base manifest
    all_raw = set(train_map) | set(val_map)
    combined_map = {}
    outside_c1 = []
    for raw in all_raw:
        cid = client_id_from_user(str(raw))
        combined_map[raw] = cid
        if cid not in base_manifest_client_ids:
            outside_c1.append(str(raw)[:8] + "...")  # truncated for safety

    if outside_c1:
        raise ValueError(
            f"{len(outside_c1)} T1 clients map outside the base C1 manifest. "
            "This violates the C1 scope invariant."
        )

    # Use str-keyed maps for replace_strict (column cast to String)
    combined_map_str: dict[str, str] = {str(k): v for k, v in combined_map.items()}

    # Train counts
    train_counts = (
        train_df.with_columns(
            pl.col("client")
            .cast(pl.String)
            .replace_strict(combined_map_str, return_dtype=pl.String)
            .alias("client_id")
        )
        .group_by("client_id")
        .agg(pl.len().alias("t1_train_example_count"))
    )

    # Validation counts
    val_counts = (
        val_df.with_columns(
            pl.col("client")
            .cast(pl.String)
            .replace_strict(combined_map_str, return_dtype=pl.String, default=None)
            .alias("client_id")
        )
        .filter(pl.col("client_id").is_not_null())
        .group_by("client_id")
        .agg(pl.len().alias("t1_validation_example_count"))
    )

    # Join to get full picture from base manifest
    all_cids = sorted(base_manifest_client_ids)
    result = (
        pl.DataFrame({"client_id": all_cids})
        .join(train_counts, on="client_id", how="left")
        .join(val_counts, on="client_id", how="left")
        .with_columns(
            pl.col("t1_train_example_count").fill_null(0).cast(pl.Int64),
            pl.col("t1_validation_example_count").fill_null(0).cast(pl.Int64),
        )
        .with_columns((pl.col("t1_train_example_count") > 0).alias("eligible_for_t1_smoke"))
        .sort("client_id")
    )
    return result


# ---------------------------------------------------------------------------
# Smoke slice materialization
# ---------------------------------------------------------------------------


def prepare_t1_smoke_slices(
    train_path: Path | str,
    validation_path: Path | str,
    selected_client_ids: list[str],
    *,
    category_count: int = 588,
    valid_codes: set[int] | None = None,
    max_train_examples_per_client: int = 32,
    max_validation_examples: int = 256,
    train_output_path: Path | str,
    validation_output_path: Path | str,
) -> tuple[str, str]:
    """Materialize tiny smoke slices for training and validation.

    Train:
        For union of selected clients, stable-sort by (client_id, session,
        decision_order), take first ``max_train_examples_per_client`` rows per
        client. Persist only (client_id, session, decision_order, label_code).
        Raw ``client`` and ``label_value`` are not persisted.

    Validation:
        Map all rows to opaque client IDs, stable-sort, take first
        ``max_validation_examples`` valid rows. Same four columns.

    Returns
    -------
    (train_sha256, validation_sha256)
    """
    train_path = Path(train_path)
    validation_path = Path(validation_path)
    train_output_path = Path(train_output_path)
    validation_output_path = Path(validation_output_path)

    selected_set = set(selected_client_ids)

    # --- TRAIN ---
    train_df = pl.read_parquet(train_path)
    train_with_labels = validate_and_convert_t1_labels(
        train_df,
        category_count=category_count,
        valid_codes=valid_codes,
        split_label="T1 TRAIN",
    )

    train_map = _build_client_id_map(train_with_labels)
    train_map_str: dict[str, str] = {str(k): v for k, v in train_map.items()}
    train_with_id = train_with_labels.with_columns(
        pl.col("client")
        .cast(pl.String)
        .replace_strict(train_map_str, return_dtype=pl.String)
        .alias("client_id"),
    ).filter(pl.col("client_id").is_in(selected_set))

    # Stable sort by (client_id, session, decision_order) then cap at 32 per client
    train_sorted = train_with_id.sort(["client_id", "session", "decision_order"])
    train_capped = (
        train_sorted.with_columns(
            pl.int_range(pl.len(), dtype=pl.Int64).over("client_id").alias("_row_within_client")
        )
        .filter(pl.col("_row_within_client") < max_train_examples_per_client)
        .drop("_row_within_client")
    )

    # Persist only the four required columns; no raw client or label_value
    train_slice = train_capped.select(["client_id", "session", "decision_order", "label_code"])
    train_output_path.parent.mkdir(parents=True, exist_ok=True)
    train_slice.write_parquet(train_output_path)
    train_sha256 = _sha256_file(train_output_path)

    # --- VALIDATION ---
    val_df = pl.read_parquet(validation_path)
    val_with_labels = validate_and_convert_t1_labels(
        val_df,
        category_count=category_count,
        valid_codes=valid_codes,
        split_label="T1 VALIDATION",
    )

    val_map = _build_client_id_map(val_with_labels)
    val_map_str: dict[str, str] = {str(k): v for k, v in val_map.items()}
    val_with_id = val_with_labels.with_columns(
        pl.col("client")
        .cast(pl.String)
        .replace_strict(val_map_str, return_dtype=pl.String)
        .alias("client_id"),
    )

    val_sorted = val_with_id.sort(["client_id", "session", "decision_order"])
    val_slice = val_sorted.select(["client_id", "session", "decision_order", "label_code"]).head(
        max_validation_examples
    )

    validation_output_path.parent.mkdir(parents=True, exist_ok=True)
    val_slice.write_parquet(validation_output_path)
    val_sha256 = _sha256_file(validation_output_path)

    return train_sha256, val_sha256


# ---------------------------------------------------------------------------
# Phase1Batch construction
# ---------------------------------------------------------------------------


def _row_to_query_features(decision_order: int) -> tuple[int, float, float]:
    """Compute query categorical and continuous features for one row.

    Returns
    -------
    (query_context_id, f0, f1)
    """
    do = max(decision_order, 0)
    query_context_id = 1 + (do % 7)  # in [1, 7], pad=0 is reserved
    f0 = float(math.log1p(do))
    f1 = float(min(do, 1000) / 1000.0)
    return query_context_id, f0, f1


def make_t1_smoke_batches(
    slice_df: pl.DataFrame,
    *,
    spec: Phase1BatchSpec | None = None,
    batch_size: int = 8,
) -> list[Phase1Batch]:
    """Build Phase1Batch list from a T1 smoke slice DataFrame.

    The DataFrame must have columns: client_id, session, decision_order, label_code.
    Uses zero-history representation (physical L=1, semantic length=0).
    Validates each batch with ``validate_canonical_phase1_batch`` before returning.

    Parameters
    ----------
    slice_df
        Rows for a single client (or arbitrary set — batching is over rows).
    spec
        Phase1BatchSpec to use. Defaults to ``default_batch_spec()``.
    batch_size
        Maximum rows per batch. Final short batch is kept.

    Returns
    -------
    list[Phase1Batch]
        One or more validated batches covering all rows.
    """
    if spec is None:
        spec = default_batch_spec()

    rows = slice_df.select(["decision_order", "label_code"]).to_dicts()
    if not rows:
        return []

    batches: list[Phase1Batch] = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        B = len(chunk)

        # History: physical L=1, semantic length=0 (zero-history)
        L = 1
        lengths = torch.zeros(B, dtype=torch.int64)  # semantic length = 0
        history_mask = torch.zeros(B, L, dtype=torch.bool)  # all padding

        # History categorical: all pad_id
        history_categorical_ids: dict[str, torch.Tensor] = {
            ch.name: torch.full((B, L), ch.pad_id, dtype=torch.int64)
            for ch in spec.history_categorical
        }

        # History continuous: zeros
        history_continuous = torch.zeros(B, L, spec.history_continuous_dim, dtype=torch.float32)

        # Query features
        query_context_ids = []
        f0_vals = []
        f1_vals = []
        for row in chunk:
            qcid, f0, f1 = _row_to_query_features(int(row["decision_order"]))
            query_context_ids.append(qcid)
            f0_vals.append(f0)
            f1_vals.append(f1)

        query_categorical_ids: dict[str, torch.Tensor] = {
            "query_context_id": torch.tensor(query_context_ids, dtype=torch.int64)
        }
        query_continuous = torch.tensor(
            [[f0, f1] for f0, f1 in zip(f0_vals, f1_vals)], dtype=torch.float32
        )

        # Candidate: physical K=1, all-pad
        K = 1
        candidate_ids = torch.full((B, K), spec.candidate_id_pad_id, dtype=torch.int64)
        candidate_categorical_ids: dict[str, torch.Tensor] = {
            ch.name: torch.full((B, K), ch.pad_id, dtype=torch.int64)
            for ch in spec.candidate_categorical
        }
        candidate_continuous = torch.zeros(B, K, spec.candidate_continuous_dim, dtype=torch.float32)
        candidate_mask = torch.zeros(B, K, dtype=torch.bool)

        # T1 targets
        t1_target = torch.tensor([int(row["label_code"]) for row in chunk], dtype=torch.int64)
        t1_present = torch.ones(B, dtype=torch.bool)

        # T2/T3 absent
        t2_target = torch.zeros(B, 1, dtype=torch.float32)
        t2_present = torch.zeros(B, dtype=torch.bool)
        t3_gains = torch.zeros(B, K, dtype=torch.float32)
        t3_present = torch.zeros(B, dtype=torch.bool)

        batch = Phase1Batch(
            history_categorical_ids=history_categorical_ids,
            history_continuous_features=history_continuous,
            lengths=lengths,
            history_mask=history_mask,
            query_categorical_ids=query_categorical_ids,
            query_continuous_features=query_continuous,
            candidate_ids=candidate_ids,
            candidate_categorical_ids=candidate_categorical_ids,
            candidate_continuous_features=candidate_continuous,
            candidate_mask=candidate_mask,
            t1_target=t1_target,
            t2_target=t2_target,
            t3_gains=t3_gains,
            t1_present=t1_present,
            t2_present=t2_present,
            t3_present=t3_present,
        )
        validate_canonical_phase1_batch(batch, spec)
        batches.append(batch)

    return batches
