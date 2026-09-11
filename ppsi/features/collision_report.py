"""Deterministic aggregate collision report for the product hashing feature."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import polars as pl

from ppsi.features.hashing import HASH_ALGORITHM, PAD_BUCKET, ProductHashConfig, hash_product_id

_ITEM_NAMESPACE = "rees46:item:"
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _repo_relative(path: Path) -> str:
    """Record the input path relative to the repository so the JSON stays machine-neutral."""

    try:
        return path.resolve().relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        return path.name


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collision_metrics(bucket_ids: Sequence[int], *, bucket_count: int) -> dict[str, int | float]:
    """Summarize one bucket ID per unique product without retaining product identifiers."""

    if bucket_count < 2:
        raise ValueError("bucket_count must leave bucket 0 for padding")
    if any(bucket_id <= PAD_BUCKET or bucket_id >= bucket_count for bucket_id in bucket_ids):
        raise ValueError("bucket IDs must be in the non-padding configured range")

    loads = Counter(bucket_ids)
    unique_product_count = len(bucket_ids)
    occupied_buckets = len(loads)
    excess_collisions = unique_product_count - occupied_buckets

    return {
        "bucket_count": bucket_count,
        "usable_buckets": bucket_count - 1,
        "unique_product_count": unique_product_count,
        "occupied_buckets": occupied_buckets,
        "collided_buckets": sum(load > 1 for load in loads.values()),
        "excess_collisions": excess_collisions,
        "collision_rate": round(
            excess_collisions / unique_product_count if unique_product_count else 0.0,
            6,
        ),
        "maximum_bucket_load": max(loads.values(), default=0),
    }


def build_collision_report(
    catalog_path: Path,
    *,
    bucket_counts: Sequence[int],
    seed: int,
) -> dict[str, Any]:
    """Build aggregate evidence from the catalogue using the production hash function."""

    catalog_path = Path(catalog_path)
    if not catalog_path.is_file():
        raise FileNotFoundError(catalog_path)

    candidates = tuple(bucket_counts)
    if not candidates:
        raise ValueError("at least one bucket count is required")
    if len(set(candidates)) != len(candidates):
        raise ValueError("bucket counts must be unique")

    catalog = pl.read_parquet(catalog_path, columns=["item"])
    if catalog.get_column("item").null_count():
        raise ValueError("catalogue item IDs must not be null")

    items = catalog.get_column("item").unique().sort().to_list()
    namespaced_items = tuple(f"{_ITEM_NAMESPACE}{item}" for item in items)

    results: list[dict[str, int | float]] = []
    for bucket_count in candidates:
        config = ProductHashConfig(bucket_count=bucket_count, seed=seed, residual_dim=1)
        bucket_ids = tuple(hash_product_id(item_id, config) for item_id in namespaced_items)
        results.append(collision_metrics(bucket_ids, bucket_count=bucket_count))

    return {
        "schema": "ProductHashCollisionReport",
        "version": 1,
        "task_id": "S2-SE-03",
        "input": {
            "file": _repo_relative(catalog_path),
            "sha256": _sha256(catalog_path),
            "catalog_row_count": catalog.height,
            "unique_product_count": len(items),
        },
        "hashing": {
            "algorithm": HASH_ALGORITHM,
            "seed": seed,
            "padding_bucket": PAD_BUCKET,
            "item_namespace": "rees46:item:<catalog item>",
            "candidate_bucket_counts": list(candidates),
        },
        "decision_status": "informational; final bucket count remains open for architecture review",
        "results": results,
    }


def write_collision_report(report: dict[str, Any], output_path: Path) -> None:
    """Write stable UTF-8 JSON with no timestamp or machine-specific metadata."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--bucket-counts", required=True, nargs="+", type=int)
    args = parser.parse_args(argv)

    report = build_collision_report(
        args.catalog,
        bucket_counts=args.bucket_counts,
        seed=args.seed,
    )
    write_collision_report(report, args.output)
    print(f"Wrote aggregate collision report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
