from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from ppsi.features.collision_report import (
    build_collision_report,
    collision_metrics,
    write_collision_report,
)


def test_collision_metrics_has_known_aggregate_values() -> None:
    assert collision_metrics([1, 1, 2, 4, 4, 4], bucket_count=5) == {
        "bucket_count": 5,
        "usable_buckets": 4,
        "unique_product_count": 6,
        "occupied_buckets": 3,
        "collided_buckets": 2,
        "excess_collisions": 3,
        "collision_rate": 0.5,
        "maximum_bucket_load": 3,
    }


def test_report_uses_unique_catalog_items_and_is_deterministic(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.parquet"
    pl.DataFrame({"item": [4, 1, 2, 3, 2]}).write_parquet(catalog)

    first = build_collision_report(catalog, bucket_counts=[4], seed=13)
    second = build_collision_report(catalog, bucket_counts=[4], seed=13)

    assert first == second
    assert first["input"]["catalog_row_count"] == 5
    assert first["input"]["unique_product_count"] == 4
    assert first["hashing"] == {
        "algorithm": "blake2b-64-v1",
        "seed": 13,
        "padding_bucket": 0,
        "item_namespace": "rees46:item:<catalog item>",
        "candidate_bucket_counts": [4],
    }
    assert first["results"] == [
        {
            "bucket_count": 4,
            "usable_buckets": 3,
            "unique_product_count": 4,
            "occupied_buckets": 2,
            "collided_buckets": 1,
            "excess_collisions": 2,
            "collision_rate": 0.5,
            "maximum_bucket_load": 3,
        }
    ]

    output = tmp_path / "report.json"
    write_collision_report(first, output)
    first_bytes = output.read_bytes()
    write_collision_report(second, output)

    assert output.read_bytes() == first_bytes
    assert "rees46:item:1" not in json.dumps(first)


@pytest.mark.parametrize(
    ("bucket_ids", "bucket_count"),
    [([0], 4), ([4], 4), ([1], 1)],
)
def test_collision_metrics_rejects_invalid_buckets(
    bucket_ids: list[int], bucket_count: int
) -> None:
    with pytest.raises(ValueError):
        collision_metrics(bucket_ids, bucket_count=bucket_count)


def test_report_rejects_duplicate_candidate_counts(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.parquet"
    pl.DataFrame({"item": [1]}).write_parquet(catalog)

    with pytest.raises(ValueError, match="unique"):
        build_collision_report(catalog, bucket_counts=[8, 8], seed=13)
