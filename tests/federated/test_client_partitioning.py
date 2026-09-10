"""Tests for ppsi.federated.clients — client identity and manifest building."""

from __future__ import annotations

import inspect
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from ppsi.federated.clients import (
    CLIENT_IDENTITY_VERSION,
    build_client_manifest,
    client_id_from_user,
    compute_distribution_stats,
    load_temporal_boundaries,
    manifest_content_sha256,
    manifest_sha256,
)

# ---------------------------------------------------------------------------
# Client identity tests
# ---------------------------------------------------------------------------

_CLIENT_ID_RE = re.compile(r"^client-v1-[0-9a-f]{64}$")


class TestClientIdFromUser:
    """Tests for the client_id_from_user function."""

    def test_same_user_same_id(self) -> None:
        """Same user always produces the same client ID."""
        uid = "rees46:user:12345"
        assert client_id_from_user(uid) == client_id_from_user(uid)

    def test_seed_independence(self) -> None:
        """Client ID does not depend on any experiment seed.

        Function accepts only user_id and takes no seed argument.
        """
        sig = inspect.signature(client_id_from_user)
        assert list(sig.parameters.keys()) == ["user_id"]

        uid = "rees46:user:99999"
        id1 = client_id_from_user(uid)
        id2 = client_id_from_user(uid)
        assert id1 == id2

    def test_different_users_different_ids(self) -> None:
        """Different users produce different client IDs."""
        id1 = client_id_from_user("rees46:user:1")
        id2 = client_id_from_user("rees46:user:2")
        assert id1 != id2

    def test_client_id_format(self) -> None:
        """Client ID matches the expected format."""
        cid = client_id_from_user("rees46:user:42")
        assert _CLIENT_ID_RE.match(cid), f"Unexpected format: {cid}"

    def test_version_in_id(self) -> None:
        """Client ID contains the version string."""
        cid = client_id_from_user("rees46:user:1")
        assert f"client-{CLIENT_IDENTITY_VERSION}-" in cid

    def test_empty_user_id_raises(self) -> None:
        """Empty user_id raises ValueError."""
        with pytest.raises(ValueError):
            client_id_from_user("")

    def test_non_string_raises(self) -> None:
        """Non-string user_id raises ValueError."""
        with pytest.raises(ValueError):
            client_id_from_user(12345)  # type: ignore[arg-type]

    def test_many_users_no_collision(self) -> None:
        """Generate 10,000 IDs and verify uniqueness (collision check)."""
        ids = [client_id_from_user(f"user:{i}") for i in range(10_000)]
        assert len(set(ids)) == 10_000


# ---------------------------------------------------------------------------
# Manifest tests
# ---------------------------------------------------------------------------


def _make_toy_cohort(n: int = 5) -> pl.DataFrame:
    """Create a toy cohort manifest with C1 and C2 users."""
    return pl.DataFrame(
        {
            "user_id": list(range(100, 100 + n)) + [999, 998],
            "cohort": ["C1"] * n + ["C2", "C3_VAL"],
        }
    )


class TestBuildClientManifest:
    """Tests for build_client_manifest (without events parquet)."""

    def test_basic_build(self, tmp_path: str) -> None:
        """Build a manifest from a toy cohort."""
        cohort = _make_toy_cohort(5)
        path = f"{tmp_path}/cohort.parquet"
        cohort.write_parquet(path)

        manifest = build_client_manifest(path)

        assert len(manifest) == 5
        assert manifest["cohort"].unique().to_list() == ["C1"]
        assert manifest["eligible"].all()
        assert manifest["task_example_counts_status"].unique().to_list() == ["not_measured"]

    def test_no_duplicate_membership(self, tmp_path: str) -> None:
        """Duplicate user_ids in the C1 cohort raise ValueError."""
        cohort = pl.DataFrame({"user_id": [1, 1, 2], "cohort": ["C1", "C1", "C1"]})
        path = f"{tmp_path}/dup.parquet"
        cohort.write_parquet(path)

        with pytest.raises(ValueError, match="duplicate"):
            build_client_manifest(path)

    def test_deterministic_manifest_order(self, tmp_path: str) -> None:
        """Shuffled input produces the same sorted output and hash."""
        cohort1 = pl.DataFrame({"user_id": [10, 20, 30, 40, 50], "cohort": ["C1"] * 5})
        cohort2 = pl.DataFrame({"user_id": [50, 30, 10, 40, 20], "cohort": ["C1"] * 5})
        p1 = f"{tmp_path}/c1.parquet"
        p2 = f"{tmp_path}/c2.parquet"
        cohort1.write_parquet(p1)
        cohort2.write_parquet(p2)

        m1 = build_client_manifest(p1)
        m2 = build_client_manifest(p2)

        assert m1["client_id"].to_list() == m2["client_id"].to_list()
        assert manifest_sha256(m1) == manifest_sha256(m2)

    def test_empty_cohort_raises(self, tmp_path: str) -> None:
        """Empty cohort manifest raises ValueError."""
        cohort = pl.DataFrame({"user_id": pl.Series([], dtype=pl.Int64), "cohort": []})
        path = f"{tmp_path}/empty.parquet"
        cohort.write_parquet(path)

        with pytest.raises(ValueError):
            build_client_manifest(path)

    def test_no_c1_users_raises(self, tmp_path: str) -> None:
        """Cohort with no C1 users raises ValueError."""
        cohort = pl.DataFrame({"user_id": [1, 2], "cohort": ["C2", "C3_VAL"]})
        path = f"{tmp_path}/no_c1.parquet"
        cohort.write_parquet(path)

        with pytest.raises(ValueError, match="No C1"):
            build_client_manifest(path)

    def test_missing_columns_raises(self, tmp_path: str) -> None:
        """Cohort missing required columns raises ValueError."""
        bad = pl.DataFrame({"uid": [1], "label": ["C1"]})
        path = f"{tmp_path}/bad.parquet"
        bad.write_parquet(path)

        with pytest.raises(ValueError, match="missing columns"):
            build_client_manifest(path)

    def test_manifest_sorted_by_client_id(self, tmp_path: str) -> None:
        """Output manifest is sorted by client_id."""
        cohort = _make_toy_cohort(20)
        path = f"{tmp_path}/cohort.parquet"
        cohort.write_parquet(path)

        manifest = build_client_manifest(path)
        ids = manifest["client_id"].to_list()
        assert ids == sorted(ids)

    def test_null_event_columns_without_events(self, tmp_path: str) -> None:
        """Without events parquet, event columns are null."""
        cohort = _make_toy_cohort(3)
        path = f"{tmp_path}/cohort.parquet"
        cohort.write_parquet(path)

        manifest = build_client_manifest(path)
        assert manifest["train_event_count"].null_count() == 3
        assert manifest["validation_event_count"].null_count() == 3
        assert (
            manifest["event_statistics_status"].to_list()
            == ["not_measured_raw_events_unavailable"] * 3
        )

    def test_manifest_collision_rejection(
        self, tmp_path: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify that SHA-256 client ID collision raises ValueError with clear message."""
        import ppsi.federated.clients as clients_mod

        cohort = pl.DataFrame({"user_id": [1, 2], "cohort": ["C1", "C1"]})
        path = f"{tmp_path}/cohort_dup.parquet"
        cohort.write_parquet(path)

        # Force identical client_id for distinct users to trigger collision branch
        monkeypatch.setattr(
            clients_mod, "client_id_from_user", lambda uid: "client-v1-collidinghash"
        )

        with pytest.raises(ValueError, match="Client ID collision detected"):
            build_client_manifest(path)

    def test_manifest_content_sha256_canonical(self) -> None:
        """Manifest content SHA is independent of column ordering and based on logical content."""
        df1 = pl.DataFrame(
            {
                "client_id": ["client-v1-b", "client-v1-a"],
                "cohort": ["C1", "C1"],
                "val": [1, 2],
            }
        )
        df2 = pl.DataFrame(
            {
                "val": [2, 1],
                "cohort": ["C1", "C1"],
                "client_id": ["client-v1-a", "client-v1-b"],
            }
        )
        h1 = manifest_content_sha256(df1)
        h2 = manifest_content_sha256(df2)
        assert h1 == h2


# ---------------------------------------------------------------------------
# Data scope test — no extra quarter filter
# ---------------------------------------------------------------------------


class TestDataScope:
    """Verify that no code path applies user_id % 4 == 1 again."""

    def test_no_extra_filter_in_source(self) -> None:
        """Grep the clients module source for any modulo-4 filter."""
        source = Path(__file__).resolve().parents[1] / "ppsi" / "federated" / "clients.py"
        if not source.exists():
            import ppsi.federated.clients as mod

            source = Path(mod.__file__)

        content = source.read_text(encoding="utf-8")
        assert "% 4" not in content, "Source contains modulo-4 filter"
        assert "user_id % 4 == 1" not in content, "Source applies quarter filter"


# ---------------------------------------------------------------------------
# Temporal boundaries tests
# ---------------------------------------------------------------------------


class TestTemporalBoundaries:
    """Tests for loading frozen temporal boundaries."""

    def test_load_default_boundaries(self) -> None:
        """Loads boundaries from canonical config/s1-ds-05-06.v1.json."""
        bounds = load_temporal_boundaries()
        assert "TRAIN" in bounds
        assert "VALIDATION" in bounds
        train_start, train_end = bounds["TRAIN"]
        val_start, val_end = bounds["VALIDATION"]
        assert train_start < train_end
        assert val_start < val_end
        assert train_start.tzinfo is not None

    def test_missing_config_raises(self, tmp_path: str) -> None:
        """Non-existent config raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_temporal_boundaries(f"{tmp_path}/nonexistent.json")


# ---------------------------------------------------------------------------
# Event statistics tests
# ---------------------------------------------------------------------------


class TestEventStatistics:
    """Tests for raw events processing with excluded sessions and temporal boundaries."""

    def test_event_stats_with_excluded_sessions(self, tmp_path: str) -> None:
        """Verify train/val event aggregation and session exclusion using (user_id, user_session)."""
        cohort = pl.DataFrame({"user_id": [101, 102], "cohort": ["C1", "C1"]})
        cohort_path = f"{tmp_path}/cohort.parquet"
        cohort.write_parquet(cohort_path)

        cfg = {
            "temporal_split": {
                "TRAIN": ["2019-10-01T00:00:00Z", "2019-10-21T00:00:00Z"],
                "VALIDATION": ["2019-10-21T00:00:00Z", "2019-10-28T00:00:00Z"],
                "TEST": ["2019-10-28T00:00:00Z", "2019-11-01T00:00:00Z"],
            }
        }
        cfg_path = f"{tmp_path}/temp_cfg.json"
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f)

        events = pl.DataFrame(
            {
                "user_id": [101, 101, 101, 101, 102, 102, 999],
                "event_time": [
                    datetime(2019, 10, 5, 12, 0, tzinfo=UTC),
                    datetime(2019, 10, 6, 12, 0, tzinfo=UTC),
                    datetime(2019, 10, 22, 12, 0, tzinfo=UTC),
                    datetime(2019, 10, 29, 12, 0, tzinfo=UTC),
                    datetime(2019, 10, 10, 12, 0, tzinfo=UTC),
                    datetime(2019, 10, 11, 12, 0, tzinfo=UTC),
                    datetime(2019, 10, 10, 12, 0, tzinfo=UTC),
                ],
                "event_type": [
                    "view",
                    "cart",
                    "purchase",
                    "view",
                    "view",
                    "purchase",
                    "view",
                ],
                "user_session": ["s1", "s2", "s3", "s4", "s5", "s5", "s_other"],
            }
        )
        events_path = f"{tmp_path}/events.parquet"
        events.write_parquet(events_path)

        excluded = pl.DataFrame({"user_id": [101], "user_session": ["s2"]})
        exc_path = f"{tmp_path}/excluded.parquet"
        excluded.write_parquet(exc_path)

        manifest = build_client_manifest(
            cohort_path,
            events_parquet_path=events_path,
            excluded_sessions_path=exc_path,
            temporal_config_path=cfg_path,
        )

        assert len(manifest) == 2
        m101 = manifest.filter(pl.col("client_id") == client_id_from_user("101"))
        assert m101["train_event_count"].to_list() == [1]
        assert m101["validation_event_count"].to_list() == [1]
        assert json.loads(m101["train_event_type_counts"].to_list()[0]) == {"view": 1}
        assert json.loads(m101["validation_event_type_counts"].to_list()[0]) == {"purchase": 1}
        assert m101["first_allowed_time"].null_count() == 0
        assert m101["last_allowed_time"].null_count() == 0

        m102 = manifest.filter(pl.col("client_id") == client_id_from_user("102"))
        assert m102["train_event_count"].to_list() == [2]
        assert m102["validation_event_count"].to_list() == [None]
        assert json.loads(m102["train_event_type_counts"].to_list()[0]) == {
            "purchase": 1,
            "view": 1,
        }
        assert manifest["event_statistics_status"].to_list() == ["measured", "measured"]

    def test_missing_excluded_sessions_raises(self, tmp_path: str) -> None:
        """When events_parquet_path is provided, missing excluded_sessions raises."""
        cohort = pl.DataFrame({"user_id": [1], "cohort": ["C1"]})
        cpath = f"{tmp_path}/c.parquet"
        cohort.write_parquet(cpath)

        events = pl.DataFrame(
            {
                "user_id": [1],
                "event_time": [datetime(2019, 10, 1, tzinfo=UTC)],
                "event_type": ["view"],
                "user_session": ["s1"],
            }
        )
        epath = f"{tmp_path}/e.parquet"
        events.write_parquet(epath)

        with pytest.raises(ValueError, match="excluded_sessions_path is required"):
            build_client_manifest(cpath, events_parquet_path=epath, excluded_sessions_path=None)

        with pytest.raises(FileNotFoundError, match="Excluded sessions file not found"):
            build_client_manifest(
                cpath,
                events_parquet_path=epath,
                excluded_sessions_path=f"{tmp_path}/nonexistent.parquet",
            )


# ---------------------------------------------------------------------------
# Distribution statistics tests
# ---------------------------------------------------------------------------


class TestDistributionStats:
    """Verify distribution stats computation with hand-worked data."""

    def test_basic_stats(self) -> None:
        """Verify min/max/mean/p50/p90/p95/p99 on known data."""
        values = pl.Series("counts", list(range(1, 11)), dtype=pl.UInt32)
        stats = compute_distribution_stats(values)

        assert stats.client_count == 10
        assert stats.min == 1.0
        assert stats.max == 10.0
        assert stats.mean == 5.5
        assert stats.p50 == 5.0 or stats.p50 == 6.0
        assert stats.p90 >= 9.0
        assert stats.p95 >= 9.0
        assert stats.p99 >= 10.0

    def test_single_value(self) -> None:
        """All percentiles equal the single value."""
        values = pl.Series("x", [42], dtype=pl.UInt32)
        stats = compute_distribution_stats(values)

        assert stats.client_count == 1
        assert stats.min == 42.0
        assert stats.max == 42.0
        assert stats.mean == 42.0
        assert stats.p50 == 42.0

    def test_nulls_excluded(self) -> None:
        """Null values are excluded from computation."""
        values = pl.Series("x", [1, None, 3, None, 5], dtype=pl.UInt32)
        stats = compute_distribution_stats(values)

        assert stats.client_count == 3
        assert stats.min == 1.0
        assert stats.max == 5.0

    def test_all_null_raises(self) -> None:
        """All-null series raises ValueError."""
        values = pl.Series("x", [None, None], dtype=pl.UInt32)
        with pytest.raises(ValueError, match="empty"):
            compute_distribution_stats(values)
