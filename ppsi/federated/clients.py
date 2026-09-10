"""Stable user-to-client identity and manifest building.

One canonical user maps to one opaque versioned client ID.
Client IDs are deterministic, seed-independent, and collision-checked.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLIENT_IDENTITY_VERSION = "v1"
_IDENTITY_PREFIX = "ClientIdentity/v1"

# Default path for the frozen temporal split config.
_DEFAULT_TEMPORAL_CONFIG = "config/s1-ds-05-06.v1.json"


# ---------------------------------------------------------------------------
# Temporal boundaries — loaded from frozen config
# ---------------------------------------------------------------------------


def load_temporal_boundaries(
    config_path: str | Path = _DEFAULT_TEMPORAL_CONFIG,
) -> dict[str, tuple[datetime, datetime]]:
    """Load frozen temporal split boundaries from the protocol config.

    Returns a dict mapping split name to (start_utc, end_utc) half-open
    interval, e.g. ``{"TRAIN": (dt_start, dt_end), ...}``.

    Raises ``FileNotFoundError`` if the config is missing and
    ``ValueError`` if the schema is unexpected.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Temporal split config not found: {path}. "
            "Expected config/s1-ds-05-06.v1.json from the frozen data protocol."
        )
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    raw_splits = cfg.get("temporal_split")
    if not isinstance(raw_splits, dict):
        raise TypeError("temporal_split key missing or invalid in config")

    boundaries: dict[str, tuple[datetime, datetime]] = {}
    for name, pair in raw_splits.items():
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"Invalid temporal_split entry for {name}")
        start = datetime.fromisoformat(pair[0]).replace(tzinfo=UTC)
        end = datetime.fromisoformat(pair[1]).replace(tzinfo=UTC)
        boundaries[name] = (start, end)

    for required in ("TRAIN", "VALIDATION"):
        if required not in boundaries:
            raise ValueError(f"Missing required split '{required}' in temporal config")

    return boundaries


# ---------------------------------------------------------------------------
# Client identity
# ---------------------------------------------------------------------------


def client_id_from_user(user_id: str) -> str:
    """Return a deterministic opaque client ID for a canonical user identity.

    The mapping is::

        client_id = "client-v1-" + SHA256("ClientIdentity/v1\\n" + user_id)

    Properties:
        - deterministic (same user always produces the same ID)
        - seed-independent (no experiment seed parameter)
        - opaque (original user_id cannot be trivially recovered)
        - collision-checked at manifest-build time

    Does NOT use Python built-in ``hash()``.
    Does NOT claim anonymity or differential privacy.
    """
    if not isinstance(user_id, str) or not user_id:
        raise ValueError(f"user_id must be a non-empty string, got {user_id!r}")
    material = f"{_IDENTITY_PREFIX}\n{user_id}".encode()
    digest = hashlib.sha256(material).hexdigest()
    return f"client-{CLIENT_IDENTITY_VERSION}-{digest}"


# ---------------------------------------------------------------------------
# Manifest building
# ---------------------------------------------------------------------------


def _validate_cohort_manifest(df: pl.DataFrame) -> None:
    """Validate the upstream cohort manifest has expected columns."""
    required = {"user_id", "cohort"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Cohort manifest missing columns: {missing}")
    if df.is_empty():
        raise ValueError("Cohort manifest is empty")


def _compute_event_stats(
    c1_user_ids: pl.Series,
    events_path: str,
    excluded_sessions_path: str,
    temporal_config_path: str | Path = _DEFAULT_TEMPORAL_CONFIG,
) -> pl.DataFrame:
    """Compute per-user per-split event statistics from the raw parquet.

    Uses Polars lazy scan with predicate pushdown to avoid loading the full
    42M-row parquet into memory. Joins to C1 users, excludes boundary-crossing
    sessions, and assigns events to TRAIN/VALIDATION splits using the frozen
    temporal config.

    TEST events are never read or counted — they remain null/sealed.

    Parameters
    ----------
    c1_user_ids
        Series of C1 user IDs to filter to.
    events_path
        Path to the full raw events parquet.
    excluded_sessions_path
        Path to the excluded sessions parquet. REQUIRED — fails if missing.
    temporal_config_path
        Path to the frozen temporal split config.
    """
    # Load temporal boundaries from config (not hardcoded)
    boundaries = load_temporal_boundaries(temporal_config_path)
    train_start, train_end = boundaries["TRAIN"]
    val_start, val_end = boundaries["VALIDATION"]

    # Lazy scan with only required columns
    lf = pl.scan_parquet(events_path).select("user_id", "event_time", "event_type", "user_session")

    events_schema = lf.collect_schema()
    events_uid_type = events_schema["user_id"]

    # Cast c1 user_id to match events user_id type for join
    c1_df = c1_user_ids.to_frame("user_id").with_columns(pl.col("user_id").cast(events_uid_type))
    lf = lf.join(c1_df.lazy(), on="user_id", how="inner")

    # Exclude boundary-crossing sessions using upstream (user_id, user_session) key
    excluded = pl.read_parquet(excluded_sessions_path)
    if "user_id" not in excluded.columns or "user_session" not in excluded.columns:
        raise ValueError(
            "Excluded sessions parquet must contain 'user_id' and 'user_session' columns"
        )
    exc_keys = (
        excluded.select("user_id", "user_session")
        .with_columns(
            pl.col("user_id").cast(events_uid_type),
            pl.col("user_session").cast(events_schema["user_session"]),
        )
        .lazy()
    )
    lf = lf.join(exc_keys, on=["user_id", "user_session"], how="anti")

    # Ensure UTC timezone awareness for event_time if naive
    event_time_tz = events_schema["event_time"].time_zone
    if event_time_tz is None:
        lf = lf.with_columns(pl.col("event_time").dt.replace_time_zone("UTC"))

    # Assign split labels based on frozen temporal boundaries
    # Half-open intervals: [start, end)
    lf = lf.with_columns(
        pl.when((pl.col("event_time") >= train_start) & (pl.col("event_time") < train_end))
        .then(pl.lit("TRAIN"))
        .when((pl.col("event_time") >= val_start) & (pl.col("event_time") < val_end))
        .then(pl.lit("VALIDATION"))
        .otherwise(pl.lit("OTHER"))
        .alias("split")
    )

    # Filter to TRAIN and VALIDATION only (never touch TEST/GRACE)
    lf = lf.filter(pl.col("split").is_in(["TRAIN", "VALIDATION"]))

    # Aggregate per user per split in a single bounded-memory scan
    split_stats = (
        lf.group_by("user_id", "split")
        .agg(
            pl.len().alias("event_count"),
            pl.col("event_type").value_counts().alias("event_type_counts_raw"),
            pl.col("event_time").min().alias("min_time"),
            pl.col("event_time").max().alias("max_time"),
        )
        .collect()
    )

    # Pivot to get train/validation columns per user
    train = split_stats.filter(pl.col("split") == "TRAIN").select(
        "user_id",
        pl.col("event_count").cast(pl.UInt32).alias("train_event_count"),
        pl.col("event_type_counts_raw").alias("train_etcr"),
        pl.col("min_time").alias("_train_min"),
        pl.col("max_time").alias("_train_max"),
    )
    val = split_stats.filter(pl.col("split") == "VALIDATION").select(
        "user_id",
        pl.col("event_count").cast(pl.UInt32).alias("validation_event_count"),
        pl.col("event_type_counts_raw").alias("val_etcr"),
        pl.col("min_time").alias("_val_min"),
        pl.col("max_time").alias("_val_max"),
    )

    # Join train and validation stats
    result = c1_df.join(train, on="user_id", how="left").join(val, on="user_id", how="left")

    # Compute first_allowed_time and last_allowed_time
    result = result.with_columns(
        pl.min_horizontal("_train_min", "_val_min").alias("first_allowed_time"),
        pl.max_horizontal("_train_max", "_val_max").alias("last_allowed_time"),
    ).drop("_train_min", "_train_max", "_val_min", "_val_max")

    # Convert event_type_counts struct-list to JSON strings for serialization
    for col_name, raw_col in [
        ("train_event_type_counts", "train_etcr"),
        ("validation_event_type_counts", "val_etcr"),
    ]:
        if raw_col in result.columns:
            result = result.with_columns(
                pl.col(raw_col)
                .map_elements(_event_type_counts_to_json, return_dtype=pl.String)
                .alias(col_name)
            ).drop(raw_col)
        else:
            result = result.with_columns(pl.lit(None, dtype=pl.String).alias(col_name))

    return result


def _event_type_counts_to_json(counts_list: list[dict[str, Any]] | None) -> str | None:
    """Convert Polars value_counts struct list to a JSON string."""
    if counts_list is None:
        return None
    try:
        result = {}
        for entry in counts_list:
            if isinstance(entry, dict):
                event_type = entry.get("event_type", entry.get("value"))
                count = entry.get("count", entry.get("counts"))
            else:
                event_type = getattr(entry, "event_type", None) or entry["event_type"]
                count = getattr(entry, "count", None) or entry["count"]
            if event_type is not None and count is not None:
                result[str(event_type)] = int(count)
        return json.dumps(result, sort_keys=True) if result else None
    except (KeyError, ValueError, TypeError, AttributeError):
        return None


def build_client_manifest(
    cohort_manifest_path: str,
    *,
    events_parquet_path: str | None = None,
    excluded_sessions_path: str | None = None,
    temporal_config_path: str | Path = _DEFAULT_TEMPORAL_CONFIG,
) -> pl.DataFrame:
    """Build the ClientManifest v1 from the upstream cohort manifest.

    Parameters
    ----------
    cohort_manifest_path
        Path to the frozen cohort manifest parquet.
    events_parquet_path
        Optional path to the raw events parquet for computing per-split
        event statistics. If None, event count columns are left null.
    excluded_sessions_path
        Path to the excluded sessions parquet. REQUIRED when
        events_parquet_path is provided — fails if missing.
    temporal_config_path
        Path to the frozen temporal split config.

    Returns
    -------
    pl.DataFrame
        One row per C1 client, sorted by client_id.
    """
    # Load and validate cohort manifest
    cohort_df = pl.read_parquet(cohort_manifest_path)
    _validate_cohort_manifest(cohort_df)

    # Select C1 users only — consume as-is, NO extra 25% filter
    c1_users = cohort_df.filter(pl.col("cohort") == "C1")
    if c1_users.is_empty():
        raise ValueError("No C1 users found in cohort manifest")

    # Check for duplicate user_ids
    user_ids = c1_users["user_id"]
    if user_ids.n_unique() != len(user_ids):
        dup_count = len(user_ids) - user_ids.n_unique()
        raise ValueError(f"Cohort manifest contains {dup_count} duplicate C1 user_ids")

    # Generate client IDs
    client_ids = user_ids.map_elements(
        lambda uid: client_id_from_user(str(uid)),
        return_dtype=pl.String,
    )

    # Check for client_id collisions (SHA-256 collision = catastrophic)
    if client_ids.n_unique() != len(client_ids):
        collision_count = len(client_ids) - client_ids.n_unique()
        raise ValueError(
            f"Client ID collision detected: {collision_count} collisions among "
            f"{len(client_ids)} users. This indicates a SHA-256 hash collision."
        )

    # Build the base manifest
    manifest = pl.DataFrame(
        {
            "manifest_version": ["client_manifest_v1"] * len(c1_users),
            "client_id": client_ids,
            "cohort": ["C1"] * len(c1_users),
            "eligible": [True] * len(c1_users),
            "task_example_counts_status": ["not_measured"] * len(c1_users),
        }
    )

    # Add event statistics if raw events are available
    if events_parquet_path is not None:
        # excluded_sessions is REQUIRED when computing event stats
        if excluded_sessions_path is None:
            raise ValueError(
                "excluded_sessions_path is required when events_parquet_path is "
                "provided. Boundary-crossing sessions must be excluded per protocol."
            )
        if not Path(excluded_sessions_path).is_file():
            raise FileNotFoundError(
                f"Excluded sessions file not found: {excluded_sessions_path}. "
                "This file is required for correct event statistics."
            )

        event_stats = _compute_event_stats(
            user_ids,
            events_parquet_path,
            excluded_sessions_path,
            temporal_config_path,
        )
        # Build a mapping table: user_id -> client_id with matching types
        id_map = pl.DataFrame(
            {
                "user_id": user_ids.cast(event_stats["user_id"].dtype),
                "client_id": client_ids,
            }
        )
        event_with_client = event_stats.join(id_map, on="user_id", how="left").drop("user_id")
        manifest = manifest.join(event_with_client, on="client_id", how="left")
        manifest = manifest.with_columns(
            pl.lit("measured", dtype=pl.String).alias("event_statistics_status")
        )
    else:
        # Add null event columns when raw events not available
        manifest = manifest.with_columns(
            pl.lit(None, dtype=pl.UInt32).alias("train_event_count"),
            pl.lit(None, dtype=pl.UInt32).alias("validation_event_count"),
            pl.lit(None, dtype=pl.String).alias("train_event_type_counts"),
            pl.lit(None, dtype=pl.String).alias("validation_event_type_counts"),
            pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias("first_allowed_time"),
            pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias("last_allowed_time"),
            pl.lit("not_measured_raw_events_unavailable", dtype=pl.String).alias(
                "event_statistics_status"
            ),
        )

    # Sort by client_id for deterministic output
    manifest = manifest.sort("client_id")

    return manifest


# ---------------------------------------------------------------------------
# Manifest content hashing
# ---------------------------------------------------------------------------


def manifest_content_sha256(df: pl.DataFrame) -> str:
    """Compute a deterministic SHA-256 over the logical manifest content.

    The manifest must be sorted by client_id before calling this function.
    Hashes the canonical CSV text representation sorted by columns so that
    the identity is independent of physical serialization format (Parquet
    row-group layout, IPC metadata, etc.).
    """
    sorted_cols = sorted(df.columns)
    csv_text = df.select(sorted_cols).sort("client_id").write_csv()
    return hashlib.sha256(csv_text.encode("utf-8")).hexdigest()


manifest_sha256 = manifest_content_sha256


# ---------------------------------------------------------------------------
# Distribution statistics
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DistributionStats:
    """Aggregate distribution statistics for a numeric column."""

    client_count: int
    min: float
    max: float
    mean: float
    p50: float
    p90: float
    p95: float
    p99: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "client_count": self.client_count,
            "min": self.min,
            "max": self.max,
            "mean": self.mean,
            "p50": self.p50,
            "p90": self.p90,
            "p95": self.p95,
            "p99": self.p99,
        }


def compute_distribution_stats(
    values: pl.Series,
) -> DistributionStats:
    """Compute distribution statistics for a numeric series.

    Null values are excluded from computation.
    """
    non_null = values.drop_nulls()
    if len(non_null) == 0:
        raise ValueError("Cannot compute distribution stats on empty/all-null series")

    return DistributionStats(
        client_count=len(non_null),
        min=float(non_null.min()),  # type: ignore[arg-type]
        max=float(non_null.max()),  # type: ignore[arg-type]
        mean=float(non_null.mean()),  # type: ignore[arg-type]
        p50=float(non_null.quantile(0.50, interpolation="nearest")),
        p90=float(non_null.quantile(0.90, interpolation="nearest")),
        p95=float(non_null.quantile(0.95, interpolation="nearest")),
        p99=float(non_null.quantile(0.99, interpolation="nearest")),
    )
