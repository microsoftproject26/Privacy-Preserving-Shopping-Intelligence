"""S1-DS-02: convert the REES46 October CSV to ProcessedRawParquet v1.

The file is intentionally self-contained. Read it from top to bottom:
configuration -> source checks -> one-batch transform -> artifact writing -> CLI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import psutil
import pyarrow
import pyarrow.parquet as pq

# 1) الـschema الثابت وقواعد الرفض
EXPECTED_COLUMNS = [
    "event_time", "event_type", "product_id", "category_id", "category_code",
    "brand", "price", "user_id", "user_session",
]

REJECTION_CODES = (
    "CSV_PARSE_ERROR", "MISSING_REQUIRED_ID", "INVALID_TIMESTAMP",
    "UNKNOWN_EVENT_TYPE", "INVALID_PRICE",
)

PROCESSED_SCHEMA_V1 = {
    "event_time": "Datetime(us, UTC)", "event_type": "String",
    "product_id": "String", "category_id": "String",
    "category_code": "String(nullable)", "brand": "String(nullable)",
    "price": "Float64", "user_id": "String",
    "user_session": "String(nullable)", "source": "String",
    "source_file": "String", "source_row_number": "Int64",
}


class ConversionError(RuntimeError):
    """A visible data or contract error."""


# 2) قراءة الـconfig والتحقق من ملف المصدر
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest().upper()


def write_json(path: Path, data: dict) -> None:
    text = json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2, default=str)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def load_config(path: Path) -> tuple[dict, Path]:
    path = path.resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("config_version") != 1:
        raise ConversionError("Only conversion config v1 is supported")
    if tuple(config.get("rejection_priority", [])) != REJECTION_CODES:
        raise ConversionError("The rejection enum/order is not v1")
    workspace = (path.parent / config.get("workspace", ".")).resolve()
    return config, workspace


def resolve_path(workspace: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (workspace / path).resolve()


def check_csv_header(csv_path: Path) -> None:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        header = stream.readline().rstrip("\r\n").split(",")
    if header != EXPECTED_COLUMNS:
        raise ConversionError(f"Unexpected CSV columns: {header}")


def verify_input(config: dict, workspace: Path, hash_file: bool = True) -> dict:
    csv_path = resolve_path(workspace, config["input_csv"])
    manifest_path = resolve_path(workspace, config["input_manifest"])
    if not csv_path.is_file() or not manifest_path.is_file():
        raise ConversionError("The CSV or DatasetManifest v1 is missing")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest["raw_file"]
    if csv_path.stat().st_size != expected["size_bytes"]:
        raise ConversionError("CSV size does not match DatasetManifest v1")

    observed_hash = sha256_file(csv_path) if hash_file else expected["sha256"].upper()
    if observed_hash != expected["sha256"].upper():
        raise ConversionError("CSV SHA-256 does not match DatasetManifest v1")
    check_csv_header(csv_path)

    return {
        "csv_path": csv_path,
        "manifest": manifest,
        "sha256": observed_hash,
        "expected_rows": expected["row_count"],
    }


def open_csv(csv_path: Path, batch_size: int):
    all_strings = {name: pl.String for name in EXPECTED_COLUMNS}
    return pl.scan_csv(
        csv_path,
        schema_overrides=all_strings,
        infer_schema_length=0,
        null_values=[""],
        ignore_errors=False,
        truncate_ragged_lines=False,
        low_memory=True,
        rechunk=False,
    ).collect_batches(chunk_size=batch_size, maintain_order=True)


# 3) فحص وتحويل batch واحدة
def timestamp_expr() -> pl.Expr:
    return pl.col("event_time").str.strptime(
        pl.Datetime("us", "UTC"),
        format="%Y-%m-%d %H:%M:%S UTC",
        strict=False,
    )


def price_expr() -> pl.Expr:
    return pl.col("price").cast(pl.Float64, strict=False)


def missing_id_expr(config: dict) -> pl.Expr:
    checks = [
        pl.col(name).is_null() | (pl.col(name).str.strip_chars() == "")
        for name in config["required_non_null_ids"]
    ]
    return pl.any_horizontal(checks)


def invalid_price_expr(config: dict) -> pl.Expr:
    price = price_expr()
    invalid = price.is_null() | price.is_nan() | price.is_infinite()
    if not config["price_policy"]["allow_negative"]:
        invalid = invalid | (price < 0)
    if not config["price_policy"]["allow_zero"]:
        invalid = invalid | (price == 0)
    return invalid


def rejection_expr(config: dict) -> pl.Expr:
    unknown_event = pl.col("event_type").is_null() | ~pl.col("event_type").is_in(config["allowed_event_types"])
    return (
        pl.when(missing_id_expr(config)).then(pl.lit("MISSING_REQUIRED_ID"))
        .when(timestamp_expr().is_null()).then(pl.lit("INVALID_TIMESTAMP"))
        .when(unknown_event).then(pl.lit("UNKNOWN_EVENT_TYPE"))
        .when(invalid_price_expr(config)).then(pl.lit("INVALID_PRICE"))
        .otherwise(pl.lit(None, dtype=pl.String))
        .alias("rejection_code")
    )


def diagnostic_expr() -> pl.Expr:
    return (
        pl.when(pl.col("rejection_code") == "MISSING_REQUIRED_ID").then(pl.lit("required ID is null/empty"))
        .when(pl.col("rejection_code") == "INVALID_TIMESTAMP").then(pl.lit("invalid UTC timestamp"))
        .when(pl.col("rejection_code") == "UNKNOWN_EVENT_TYPE").then(pl.lit("unknown event type"))
        .when(pl.col("rejection_code") == "INVALID_PRICE").then(pl.lit("invalid price"))
        .otherwise(pl.lit(None, dtype=pl.String))
        .alias("diagnostic")
    )


def classify_batch(raw: pl.DataFrame, first_row: int, config: dict) -> tuple[pl.DataFrame, pl.DataFrame]:
    numbered = pl.int_range(first_row, first_row + raw.height, dtype=pl.Int64)
    checked = raw.with_columns(
        numbered.alias("source_row_number"),
        timestamp_expr().alias("_event_time"),
        price_expr().alias("_price"),
        rejection_expr(config),
    ).with_columns(diagnostic_expr())
    return select_accepted(checked, config), select_rejected(checked)


def select_accepted(checked: pl.DataFrame, config: dict) -> pl.DataFrame:
    return checked.filter(pl.col("rejection_code").is_null()).select(
        pl.col("_event_time").alias("event_time"),
        "event_type", "product_id", "category_id", "category_code", "brand",
        pl.col("_price").alias("price"), "user_id", "user_session",
        pl.lit("REES46").alias("source"),
        pl.lit(Path(config["input_csv"]).name).alias("source_file"),
        "source_row_number",
    )


def select_rejected(checked: pl.DataFrame) -> pl.DataFrame:
    return checked.filter(pl.col("rejection_code").is_not_null()).select(
        "source_row_number", "rejection_code", "diagnostic", *EXPECTED_COLUMNS
    )


@dataclass
class Stats:
    source_rows: int = 0
    accepted_rows: int = 0
    rejected_rows: int = 0
    batches: int = 0
    peak_rss_bytes: int = 0
    min_time: datetime | None = None
    max_time: datetime | None = None
    min_price: float | None = None
    max_price: float | None = None
    previous_time: datetime | None = None
    events: Counter = field(default_factory=Counter)
    rejections: Counter = field(default_factory=lambda: Counter({code: 0 for code in REJECTION_CODES}))
    nulls: Counter = field(default_factory=lambda: Counter({name: 0 for name in EXPECTED_COLUMNS}))


# 4) تجميع الإحصائيات وكتابة Parquet على دفعات
def add_raw_stats(stats: Stats, raw: pl.DataFrame) -> int:
    first_row = stats.source_rows + 1
    stats.source_rows += raw.height
    stats.nulls.update({name: raw[name].null_count() for name in EXPECTED_COLUMNS})
    return first_row


def add_accepted_stats(stats: Stats, frame: pl.DataFrame) -> None:
    stats.accepted_rows += frame.height
    if frame.is_empty():
        return
    counts = frame.group_by("event_type").len()
    stats.events.update(dict(zip(counts["event_type"].to_list(), counts["len"].to_list())))
    update_ranges(stats, frame)


def update_ranges(stats: Stats, frame: pl.DataFrame) -> None:
    batch_min, batch_max = frame["event_time"].min(), frame["event_time"].max()
    if stats.previous_time is not None and batch_min < stats.previous_time:
        raise ConversionError("CSV is not ordered by (event_time, source_row_number)")
    stats.previous_time = batch_max
    stats.min_time = batch_min if stats.min_time is None else min(stats.min_time, batch_min)
    stats.max_time = batch_max if stats.max_time is None else max(stats.max_time, batch_max)
    price_min, price_max = frame["price"].min(), frame["price"].max()
    stats.min_price = price_min if stats.min_price is None else min(stats.min_price, price_min)
    stats.max_price = price_max if stats.max_price is None else max(stats.max_price, price_max)


def add_rejected_stats(stats: Stats, frame: pl.DataFrame) -> None:
    stats.rejected_rows += frame.height
    if frame.is_empty():
        return
    counts = frame.group_by("rejection_code").len()
    stats.rejections.update(dict(zip(counts["rejection_code"].to_list(), counts["len"].to_list())))


def write_parquet(frame: pl.DataFrame, path: Path, writers: dict, name: str, config: dict) -> None:
    if frame.is_empty():
        return
    table = frame.to_arrow()
    if name not in writers:
        writers[name] = pq.ParquetWriter(
            path, table.schema, compression=config["compression"],
            compression_level=config["compression_level"],
            write_statistics=config["statistics"], use_dictionary=True,
        )
    writers[name].write_table(table, row_group_size=frame.height)


def prepare_output(config: dict, workspace: Path) -> tuple[Path, Path]:
    output = resolve_path(workspace, config["output_directory"])
    staging = output.with_name(output.name + ".staging")
    if output.exists() or staging.exists():
        raise ConversionError("Output already exists; refusing to overwrite it")
    staging.mkdir(parents=True)
    return output, staging


def process_batch(raw: pl.DataFrame, stats: Stats, staging: Path, writers: dict, config: dict) -> None:
    first_row = add_raw_stats(stats, raw)
    accepted, rejected = classify_batch(raw, first_row, config)
    add_accepted_stats(stats, accepted)
    add_rejected_stats(stats, rejected)
    write_parquet(accepted, staging / "processed_raw_parquet_v1.parquet", writers, "accepted", config)
    write_parquet(rejected, staging / "rejected_rows_v1.parquet", writers, "rejected", config)
    stats.peak_rss_bytes = max(stats.peak_rss_bytes, psutil.Process(os.getpid()).memory_info().rss)
    stats.batches += 1


def process_csv(source: dict, staging: Path, config: dict) -> Stats:
    stats = Stats(peak_rss_bytes=psutil.Process(os.getpid()).memory_info().rss)
    writers = {}
    try:
        for raw in open_csv(source["csv_path"], int(config["batch_size"])):
            process_batch(raw, stats, staging, writers, config)
    finally:
        for writer in writers.values():
            writer.close()
    if stats.source_rows != stats.accepted_rows + stats.rejected_rows:
        raise ConversionError("Row reconciliation failed")
    return stats


# 5) كتابة التقرير والـmanifest وإغلاق التحويل
def parquet_manifest(path: Path, root: Path) -> list[dict]:
    if not path.exists():
        return []
    return [{
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }]


def make_report(stats: Stats, config: dict, status: str, started: str, duration: float) -> dict:
    return {
        "schema_version": 1, "contract": "ConversionReport v1", "status": status,
        "source_rows": stats.source_rows, "accepted_rows": stats.accepted_rows,
        "rejected_rows": stats.rejected_rows,
        "reconciliation_ok": stats.source_rows == stats.accepted_rows + stats.rejected_rows,
        "rejection_counts": dict(stats.rejections), "event_counts": dict(sorted(stats.events.items())),
        "raw_null_counts": dict(stats.nulls),
        "accepted_time_range_utc": {"min": stats.min_time, "max": stats.max_time},
        "accepted_price_range": {"min": stats.min_price, "max": stats.max_price},
        "deterministic_order": config["deterministic_tie_key"],
        "batches": stats.batches, "parquet_files": 1 if stats.accepted_rows else 0,
        "duration_seconds": round(duration, 3), "peak_rss_bytes": stats.peak_rss_bytes,
        "started_utc": started, "finished_utc": datetime.now(UTC).isoformat(),
    }


def make_manifest(source: dict, config: dict, stats: Stats, staging: Path, status: str) -> dict:
    processed = {
        "contract": "ProcessedRawParquet v1", "config_version": 1,
        "schema": PROCESSED_SCHEMA_V1,
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest().upper(),
        "input_sha256": source["sha256"],
        "accepted_parts": parquet_manifest(staging / "processed_raw_parquet_v1.parquet", staging),
        "rejected_parts": parquet_manifest(staging / "rejected_rows_v1.parquet", staging),
        "row_counts": {"source": stats.source_rows, "accepted": stats.accepted_rows, "rejected": stats.rejected_rows},
        "rejection_enum": list(REJECTION_CODES), "status": status,
    }
    runtime = {"python": platform.python_version(), "polars": pl.__version__, "pyarrow": pyarrow.__version__, "platform": platform.platform()}
    return {**source["manifest"], "processed_raw_parquet": processed, "runtime": runtime}


def save_metadata(staging: Path, report: dict, manifest: dict, config: dict) -> None:
    write_json(staging / "conversion_report.v1.json", report)
    write_json(staging / "dataset_manifest.v1.json", manifest)
    write_json(staging / "effective_conversion_config.v1.json", config)


def finalize(output: Path, staging: Path, source: dict, config: dict, stats: Stats, started: str, duration: float) -> dict:
    blocked = stats.rejected_rows > 0 and config["fail_if_any_rejected"]
    status = "BLOCKED_PENDING_MALFORMED_ROW_DECISION" if blocked else "COMPLETE"
    report = make_report(stats, config, status, started, duration)
    save_metadata(staging, report, make_manifest(source, config, stats, staging, status), config)
    if blocked:
        raise ConversionError(f"{stats.rejected_rows} rejected rows need the team decision")
    if stats.source_rows != source["expected_rows"]:
        raise ConversionError(f"Expected {source['expected_rows']} rows, got {stats.source_rows}")
    staging.rename(output)
    return report


def convert(config_path: Path) -> dict:
    config, workspace = load_config(config_path)
    source = verify_input(config, workspace)
    output, staging = prepare_output(config, workspace)
    started, clock = datetime.now(UTC).isoformat(), time.perf_counter()
    try:
        stats = process_csv(source, staging, config)
    except Exception as error:
        write_json(staging / "conversion_report.v1.json", {
            "schema_version": 1, "status": "FAILED", "failure_code": "CSV_PARSE_ERROR",
            "message": str(error), "started_utc": started,
        })
        raise ConversionError(f"Conversion failed; evidence is in {staging}") from error
    return finalize(output, staging, source, config, stats, started, time.perf_counter() - clock)


def validate_output(output: Path) -> dict:
    output = output.resolve()
    report = json.loads((output / "conversion_report.v1.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "dataset_manifest.v1.json").read_text(encoding="utf-8"))
    if report["status"] != "COMPLETE" or not report["reconciliation_ok"]:
        raise ConversionError("Output is not complete/reconciled")
    parts = manifest["processed_raw_parquet"]["accepted_parts"]
    for item in parts:
        path = output / item["path"]
        if path.stat().st_size != item["size_bytes"] or sha256_file(path) != item["sha256"]:
            raise ConversionError(f"Bad artifact hash: {path}")
    rows = pl.scan_parquet(output / "processed_raw_parquet_v1.parquet").select(
        pl.len(), pl.col("source_row_number").n_unique()
    ).collect().row(0)
    if rows != (report["accepted_rows"], report["accepted_rows"]):
        raise ConversionError("Parquet row reconciliation failed")
    return {"status": "PASS", "rows": rows[0], "parquet_files": len(parts)}


# 6) Self-test صغير لا يحتاج الداتا الأصلية
def tiny_config(folder: Path, csv_path: Path, rows: int) -> Path:
    manifest = {"manifest_version": 1, "raw_file": {
        "filename": csv_path.name, "size_bytes": csv_path.stat().st_size,
        "sha256": sha256_file(csv_path), "row_count": rows,
    }}
    write_json(folder / "manifest.json", manifest)
    config = {
        "config_version": 1, "contract": "ProcessedRawParquet v1", "workspace": ".",
        "input_csv": str(csv_path), "input_manifest": "manifest.json", "output_directory": "output",
        "batch_size": 2, "compression": "zstd", "compression_level": 3, "statistics": True,
        "deterministic_tie_key": ["event_time", "source_row_number"],
        "allowed_event_types": ["view", "cart", "purchase"],
        "required_non_null_ids": ["product_id", "category_id", "user_id"],
        "price_policy": {"allow_zero": True, "allow_negative": False},
        "rejection_priority": list(REJECTION_CODES), "fail_if_any_rejected": True,
    }
    write_json(folder / "config.json", config)
    return folder / "config.json"


def test_valid_fixture(folder: Path, header: str) -> None:
    csv_path = folder / "valid.csv"
    csv_path.write_text(
        header
        + "2019-10-01 00:00:00 UTC,view,100,200,shoes,nike,10.50,300,s-1\n"
        + "2019-10-01 00:00:01 UTC,cart,100,200,,,10.50,300,s-1\n",
        encoding="utf-8",
    )
    report = convert(tiny_config(folder, csv_path, 2))
    if report["event_counts"] != {"cart": 1, "view": 1}:
        raise ConversionError("Self-test event oracle failed")
    if validate_output(folder / "output")["rows"] != 2:
        raise ConversionError("Self-test output oracle failed")


def read_fixture_output(folder: Path) -> pl.DataFrame:
    return pl.read_parquet(folder / "output" / "processed_raw_parquet_v1.parquet")


def test_rejection_fixture(folder: Path, header: str) -> None:
    csv_path = folder / "rejections.csv"
    csv_path.write_text(
        header
        + "2019-10-01 00:00:00 UTC,view,,200,shoes,nike,10.50,300,s-1\n"
        + "bad-time,view,100,200,shoes,nike,10.50,300,s-1\n"
        + "2019-10-01 00:00:02 UTC,remove,100,200,shoes,nike,10.50,300,s-1\n"
        + "2019-10-01 00:00:03 UTC,view,100,200,shoes,nike,bad,300,s-1\n",
        encoding="utf-8",
    )
    config_path = tiny_config(folder, csv_path, 4)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    raw = next(open_csv(csv_path, 10))
    _, rejected = classify_batch(raw, 1, config)
    expected = ["MISSING_REQUIRED_ID", "INVALID_TIMESTAMP", "UNKNOWN_EVENT_TYPE", "INVALID_PRICE"]
    if rejected["rejection_code"].to_list() != expected:
        raise ConversionError("Self-test rejection priority failed")


def test_malformed_fixture(folder: Path, header: str) -> None:
    csv_path = folder / "malformed.csv"
    csv_path.write_text(
        header + "2019-10-01 00:00:00 UTC,view,100,200,shoes,nike,10.50,300,s-1,EXTRA\n",
        encoding="utf-8",
    )
    try:
        list(open_csv(csv_path, 10))
    except pl.exceptions.ComputeError:
        return
    raise ConversionError("Self-test malformed CSV must fail visibly")


def self_test() -> dict:
    header = ",".join(EXPECTED_COLUMNS) + "\n"
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        valid = folder / "valid-case"
        valid.mkdir()
        test_valid_fixture(valid, header)

        repeated = folder / "repeat-case"
        repeated.mkdir()
        test_valid_fixture(repeated, header)
        if not read_fixture_output(valid).equals(read_fixture_output(repeated)):
            raise ConversionError("Self-test repeatability oracle failed")

        rejected = folder / "rejection-case"
        rejected.mkdir()
        test_rejection_fixture(rejected, header)

        malformed = folder / "malformed-case"
        malformed.mkdir()
        test_malformed_fixture(malformed, header)
        return {
            "status": "PASS", "valid_rows": 2, "rejection_oracles": 4,
            "malformed_csv_oracle": "PASS", "repeatability_oracle": "PASS",
        }


# 7) أوامر التشغيل
def command_line() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("self-test")
    for name in ("verify-input", "convert"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
    validate = commands.add_parser("validate-output")
    validate.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = command_line()
    try:
        if args.command == "self-test":
            result = self_test()
        elif args.command == "validate-output":
            result = validate_output(args.output)
        else:
            config, workspace = load_config(args.config)
            result = verify_input(config, workspace) if args.command == "verify-input" else convert(args.config)
            result.pop("manifest", None)
            if isinstance(result.get("csv_path"), Path):
                result["csv_path"] = str(result["csv_path"])
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    except ConversionError as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
