# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["numpy==2.3.5", "pandas==2.2.3", "pyarrow==21.0.0", "psutil==7.2.2"]
# ///
"""One isolated, pinned pandas adapter for Eid's frozen session/order identity.

This script does not change the project's pyproject.toml, uv.lock or .venv.
Run with `uv run --no-project --python 3.11.14 --script <this-file>`.
It reads frozen labels; it never rebuilds TaskExamples or reads sealed_test.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

COUNTS = {
    "TRAIN": {"events": 4376137, "t1": 3113814, "t2": 2770471, "t2_observed": 2291753},
    "VALIDATION": {"events": 622013, "t1": 438185, "t2": 392554, "t2_observed": 322087},
}
PRIVATE = Path("artifacts/baselines/s2-pr-04-05/prepared")
PUBLIC = Path("docs/evidence/s2-pr-04-05/data_views.v1.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def session_keys(frame: pd.DataFrame, null_prefix: str) -> pd.Series:
    """Exactly the owning notebook's pandas hashing expression, no new identity."""
    users = frame["user_id"].astype(str)
    raw = frame["user_session"]
    provided = pd.util.hash_pandas_object(users + "|" + raw.astype(str), index=False)
    singleton = pd.util.hash_pandas_object(
        null_prefix + "|" + users + "|" + frame["source_row_number"].astype(str), index=False
    )
    return provided.where(raw.notna(), singleton)


def canonicalize(parts: list[pd.DataFrame], excluded: set[int]) -> pd.DataFrame:
    """Assemble complete sessions before sorting; order is GLOBAL inside the split."""
    if not parts:
        raise ValueError("no permitted raw events were loaded")
    frame = pd.concat(parts, ignore_index=True)
    frame = frame.loc[~frame["session"].isin(excluded)].copy()
    if frame["source_row_number"].duplicated().any():
        raise ValueError("duplicate raw provenance row")
    frame = frame.sort_values(
        ["session", "event_time", "source_row_number"], kind="stable"
    ).reset_index(drop=True)
    frame["order"] = np.arange(len(frame), dtype=np.int64)
    return frame


def dense_categories(raw_ids: pd.Series, mapping: dict[str, int]) -> pd.Series:
    """Preserve 64-bit identities before mapping; never accept a float round-trip."""
    if pd.api.types.is_float_dtype(raw_ids.dtype) or raw_ids.isna().any():
        raise ValueError("raw category IDs must remain non-null integers")
    if pd.api.types.is_integer_dtype(raw_ids.dtype):
        s = raw_ids.astype("int64").astype(str)
    elif pd.api.types.is_string_dtype(raw_ids.dtype) or pd.api.types.is_object_dtype(raw_ids.dtype):
        try:
            s = raw_ids.astype("int64").astype(str)
        except (ValueError, TypeError) as exc:
            raise ValueError("raw category IDs must remain non-null integers") from exc
    else:
        raise ValueError("raw category IDs must remain non-null integers")
    return s.map(mapping).fillna(-1).astype("int32")


def load_events(
    raw: Path,
    start,
    end,
    users: set[str],
    excluded: set[int],
    mapping: dict[str, int],
    null_prefix: str,
) -> pd.DataFrame:
    import pyarrow.dataset as ds

    columns = [
        "event_time",
        "user_id",
        "user_session",
        "source_row_number",
        "event_type",
        "product_id",
        "category_id",
        "price",
    ]
    source = ds.dataset(raw, format="parquet")
    predicate = (ds.field("event_time") >= pd.Timestamp(start).to_pydatetime()) & (
        ds.field("event_time") < pd.Timestamp(end).to_pydatetime()
    )
    parts = []
    for batch in source.to_batches(columns=columns, filter=predicate, batch_size=200000):
        chunk = batch.to_pandas()
        chunk = chunk.loc[chunk["user_id"].isin(users)].copy()
        if chunk.empty:
            continue
        if (
            chunk[["user_id", "source_row_number", "product_id", "category_id", "event_time"]]
            .isna()
            .any()
            .any()
        ):
            raise ValueError("raw required field contains null")
        chunk["session"] = session_keys(chunk, null_prefix)
        chunk["client"] = chunk["user_id"].astype("int64")
        chunk["item"] = chunk["product_id"].astype("int64")
        # Map before any shift: raw IDs can be larger than 2**53.
        chunk["category_code"] = dense_categories(chunk["category_id"], mapping)
        chunk["event_code"] = chunk["event_type"].map({"view": 0, "cart": 1, "purchase": 2})
        if chunk["event_code"].isna().any():
            raise ValueError("unknown event type")
        chunk["event_code"] = chunk["event_code"].astype("int8")
        chunk["event_time"] = pd.to_datetime(chunk["event_time"], utc=True).dt.as_unit("ns")
        parts.append(
            chunk[
                [
                    "session",
                    "client",
                    "item",
                    "category_code",
                    "event_code",
                    "event_time",
                    "source_row_number",
                    "price",
                ]
            ]
        )
    return canonicalize(parts, excluded)


def validate_examples(examples: pd.DataFrame, task: str, split: str) -> None:
    required = [
        "client",
        "session",
        "decision_order",
        "task_mask",
        "status",
        "cohort",
        "split",
        "label_value",
    ]
    required += ["current_category", "category_changed"] if task == "t1" else ["item", "category"]
    if any(name not in examples for name in required):
        raise ValueError("missing TaskExample contract column")
    if examples[[x for x in required if x != "label_value"]].isna().any().any():
        raise ValueError("null in required TaskExample field")
    if not examples["split"].eq(split).all() or not examples["cohort"].eq("C1").all():
        raise ValueError("wrong TaskExample split/cohort")
    if not pd.api.types.is_bool_dtype(examples["task_mask"]):
        raise ValueError("task_mask must be boolean")
    if examples[["client", "session", "decision_order"]].duplicated().any():
        raise ValueError("duplicate frozen decision key")
    observed = examples["task_mask"]
    if not examples.loc[observed, "status"].eq("OBSERVED").all():
        raise ValueError("observed/mask mismatch")
    y = examples.loc[observed, "label_value"].to_numpy(dtype=np.float64)
    maximum = 588 if task == "t1" else 2
    if (
        not np.isfinite(y).all()
        or not np.equal(y, np.floor(y)).all()
        or ((y < 0) | (y >= maximum)).any()
    ):
        raise ValueError("invalid observed label")
    if task == "t1":
        if not observed.all():
            raise ValueError("frozen T1 has no censored rows")
        if not pd.api.types.is_bool_dtype(examples["category_changed"]):
            raise ValueError("category_changed must be a boolean evaluation flag")
        if not np.array_equal(
            examples["category_changed"].to_numpy(), examples["current_category"].to_numpy() != y
        ):
            raise ValueError("frozen T1 category-change flag disagrees with its targets")
    if task == "t2" and (
        not examples.loc[~observed, "status"].eq("CENSORED").all()
        or examples.loc[~observed, "label_value"].notna().any()
    ):
        raise ValueError("censored labels must be null, never zero")


def bind_examples(events: pd.DataFrame, examples: pd.DataFrame, task: str) -> np.ndarray:
    values = examples["decision_order"].to_numpy()
    if values.dtype.kind not in "iu" or (values < 0).any() or (values >= len(events)).any():
        raise ValueError("decision_order must identify a global canonical event row")
    pos = values.astype(np.int64)
    rows = events.iloc[pos]
    for key in ("client", "session"):
        if not np.array_equal(rows[key].to_numpy(), examples[key].to_numpy()):
            raise ValueError(f"frozen-to-raw {key} binding failed")
    category = "current_category" if task == "t1" else "category"
    if not np.array_equal(rows["category_code"].to_numpy(), examples[category].to_numpy()):
        raise ValueError("frozen-to-raw category binding failed")
    if task == "t2":
        if not np.array_equal(rows["item"].to_numpy(), examples["item"].to_numpy()):
            raise ValueError("frozen-to-raw query item binding failed")
        if not rows["event_code"].eq(0).all():
            raise ValueError("T2 anchor is not a view")
    return pos


def t2_predictors(events: pd.DataFrame) -> pd.DataFrame:
    """All behavioral aggregations are STRICT prefixes before the current event."""
    sessions = events["session"]
    group = events.groupby("session", sort=False)
    output = pd.DataFrame(index=events.index)
    output["category_code"] = events["category_code"]
    price = events["price"].to_numpy(dtype=np.float64)
    if np.isinf(price).any() or (price < 0).any():
        raise ValueError("invalid raw price")
    output["log_query_price"] = np.log1p(price)
    output["log_prior_events"] = np.log1p(group.cumcount().to_numpy())
    for code, name in enumerate(("views", "carts", "purchases")):
        flag = events["event_code"].eq(code).astype("int32")
        output[f"log_prior_{name}"] = np.log1p(flag.groupby(sessions, sort=False).cumsum() - flag)
    first = ~events.duplicated(["session", "item"])
    output["log_prior_distinct_items"] = np.log1p(
        first.astype("int32").groupby(sessions, sort=False).cumsum() - first.astype("int32")
    )
    output["log_same_item_prior_events"] = np.log1p(
        events.groupby(["session", "item"], sort=False).cumcount()
    )
    elapsed = (events["event_time"] - group["event_time"].transform("first")).dt.total_seconds()
    gap = group["event_time"].diff().dt.total_seconds().fillna(0)
    if (gap < 0).any():
        raise ValueError("events are not canonically ordered")
    output["log_session_elapsed_seconds"] = np.log1p(elapsed)
    output["log_previous_gap_seconds"] = np.log1p(gap)
    hour = events["event_time"].dt.hour.to_numpy() + events["event_time"].dt.minute.to_numpy() / 60
    output["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    output["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    output["weekend"] = (events["event_time"].dt.dayofweek >= 5).astype("int8")
    return output


def main() -> None:
    import importlib.metadata
    import platform
    import runpy
    import subprocess

    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--raw", type=Path, default=Path("Dataset/processed_raw_parquet_v1.parquet")
    )
    args = parser.parse_args()
    root = args.repo.resolve()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=root, text=True
    ).strip()
    if branch != "s2-pr-04-05-classical-session-baselines":
        raise RuntimeError("wrong branch; the human must switch before preparation")
    owning_identity = runpy.run_path(str(root / "ppsi/training/identity.py"))
    text_sha = owning_identity["file_sha256"]
    raw = (root / args.raw).resolve()
    if "sealed_test" in raw.parts or not raw.is_file():
        raise FileNotFoundError("canonical raw parquet is required; sealed TEST is prohibited")
    if psutil.virtual_memory().available < 6 * 1024**3:
        raise RuntimeError("BLOCKED_CAPACITY: free at least 6 GiB before preparation")
    gate_path = root / "docs/evidence/s1-ds-09/g1_gate_v1.frozen.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate["verdict"] != "GO":
        raise ValueError("G1 is not GO")
    wanted = {x["file"]: x["sha256"] for x in gate["artifacts"]}
    inputs = {}
    files = {}
    for task in ("t1", "t2"):
        for split, suffix in (("TRAIN", "_train"), ("VALIDATION", "")):
            path = (
                root
                / f"data/examples/INTERNAL_DO_NOT_UPLOAD_task_examples_{task}{suffix}_v1.proposed.parquet"
            )
            files[(task, split)] = path
    for rel in (
        "data/protocol/INTERNAL_DO_NOT_UPLOAD_cohort_manifest_v1.proposed.parquet",
        "data/protocol/INTERNAL_DO_NOT_UPLOAD_excluded_sessions_v1.proposed.parquet",
        "docs/evidence/s1-d1-ds-07/vocabulary_v1.proposed.json",
    ):
        path = root / rel
        files[rel] = path
    for path in files.values():
        digest = sha256(path)
        expected = wanted[path.name]
        if not digest.startswith(expected):
            raise ValueError(f"frozen input digest mismatch: {path.name}")
        inputs[str(path.relative_to(root)).replace("\\", "/")] = {
            "sha256": digest,
            "expected_prefix": expected,
            "hash_convention": "raw_bytes",
        }
    raw_digest = sha256(raw)
    if pq.ParquetFile(raw).metadata.num_rows != 42448764:
        raise ValueError("unexpected canonical raw row count")
    vocab = json.loads(
        files["docs/evidence/s1-d1-ds-07/vocabulary_v1.proposed.json"].read_text(encoding="utf-8")
    )
    mapping = vocab["categories"]["code_of_category_id"]
    if vocab["categories"]["count"] != 588 or set(mapping.values()) != set(range(588)):
        raise ValueError("frozen category vocabulary mismatch")
    cohort = pd.read_parquet(
        files["data/protocol/INTERNAL_DO_NOT_UPLOAD_cohort_manifest_v1.proposed.parquet"]
    )
    c1 = set(cohort.loc[cohort["cohort"].eq("C1"), "user_id"].astype(str))
    # Use ALL rows, including censored T2 rows, to recover the producer's complete user slice.
    users = {
        str(u)
        for u in pq.read_table(files[("t2", "TRAIN")], columns=["client"])["client"]
        .unique()
        .to_pylist()
    }
    if len(c1) != 388789 or len(users) != 97279 or not users.issubset(c1):
        raise ValueError("frozen C1 example-slice membership mismatch; never reapply modulo")
    excluded = set(
        pd.read_parquet(
            files["data/protocol/INTERNAL_DO_NOT_UPLOAD_excluded_sessions_v1.proposed.parquet"]
        )["session_key"]
    )
    protocol = json.loads(
        (root / "docs/evidence/s1-ds-05-06/data_protocol_v1.proposed.json").read_text(
            encoding="utf-8"
        )
    )
    upstream = json.loads((root / "config/s1-ds-05-06.v1.json").read_text(encoding="utf-8"))
    prefix = upstream["session_policy"]["null_fallback_prefix"]
    out = root / PRIVATE
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(
            "prepared directory already exists; verify/reuse it, never overwrite blindly"
        )
    out.mkdir(parents=True, exist_ok=True)
    outputs = {}
    stats = {}
    history = None
    for split in ("TRAIN", "VALIDATION"):
        boundary = protocol["temporal_split"][split]
        events = load_events(
            raw, boundary["start"], boundary["end_exclusive"], users, excluded, mapping, prefix
        )
        if len(events) != COUNTS[split]["events"]:
            raise ValueError(f"{split} raw membership/order count differs from producer")
        if split == "TRAIN":
            if (events["category_code"] < 0).any():
                raise ValueError("TRAIN contains a category outside its frozen vocabulary")
            history = events.groupby("client", sort=False).size()
            sessions = (
                events.groupby("session", sort=True)
                .agg(
                    items=("item", lambda x: sorted(set(x))),
                    categories=("category_code", lambda x: sorted(set(x))),
                    end_time=("event_time", "max"),
                )
                .reset_index()
            )
            sessions["session_key"] = sessions["session"].map(lambda x: f"{int(x):020d}")
            sessions["end_time_ns"] = sessions["end_time"].astype("int64")
            path = out / "train_sessions.parquet"
            sessions[["session_key", "items", "categories", "end_time_ns"]].to_parquet(
                path, index=False
            )
            outputs[path.name] = sha256(path)
            del sessions
        predictor_rows = t2_predictors(events)
        for task in ("t1", "t2"):
            examples = pd.read_parquet(files[(task, split)])
            validate_examples(examples, task, split)
            if len(examples) != COUNTS[split][task]:
                raise ValueError("frozen TaskExample row count mismatch")
            positions = bind_examples(events, examples, task)
            stats[f"{task}_{split}"] = {
                "rows": len(examples),
                "observed": int(examples["task_mask"].sum()),
                "bound_rows": len(positions),
            }
            if task == "t2":
                if int(examples["task_mask"].sum()) != COUNTS[split]["t2_observed"]:
                    raise ValueError("observed T2 support changed")
                view = predictor_rows.iloc[positions].reset_index(drop=True)
                for name in (
                    "client",
                    "session",
                    "decision_order",
                    "label_value",
                    "task_mask",
                    "status",
                    "split",
                ):
                    view[name] = examples[name].to_numpy()
                view["row_ordinal"] = np.arange(len(examples), dtype=np.int64)
                view["train_history_count"] = examples["client"].map(history).to_numpy()
            elif split == "TRAIN":
                popularity = np.bincount(
                    examples["label_value"].astype("int64"), minlength=588
                ).astype(np.int64)
                path = out / "t1_popularity.npy"
                np.save(path, popularity, allow_pickle=False)
                outputs[path.name] = sha256(path)
                del examples
                continue
            else:
                view = examples[
                    ["client", "session", "decision_order", "label_value", "category_changed"]
                ].copy()
                starts = events.groupby("session", sort=False)["order"].transform("min").to_numpy()
                items = events["item"].to_numpy()
                view["query_items"] = [
                    items[max(int(starts[p]), int(p) - 19) : int(p) + 1].tolist() for p in positions
                ]
                view["row_ordinal"] = np.arange(len(examples), dtype=np.int64)
                view["train_history_count"] = examples["client"].map(history).to_numpy()
            if view["train_history_count"].isna().any():
                raise ValueError("missing authoritative TRAIN event history")
            path = out / f"{task}_{split.lower()}.parquet"
            view.to_parquet(path, index=False)
            outputs[path.name] = sha256(path)
            del view, examples
        del events, predictor_rows
        gc.collect()
    report = {
        "schema": "baseline_data_views_v1",
        "version": "1",
        "status": "PASS",
        "task_scope": {"S2-PR-04": "T2 C1", "S2-PR-05": "T1 C1"},
        "inputs": inputs,
        "outputs": {
            str(PRIVATE / k).replace("\\", "/"): {"sha256": v, "hash_convention": "raw_bytes"}
            for k, v in outputs.items()
        },
        "raw_sha256": raw_digest,
        "raw_identity_status": "LOCAL_FULL_HASH_RECORDED; NO_UPSTREAM_PARQUET_SHA_ASSUMED",
        "canonical_raw_source": "Drive file 1WOG1Rlb1JFp0vYjdutow5q56xxvZ3bAt",
        "base_c1_users": len(c1),
        "example_slice_users": len(users),
        "binding": stats,
        "history_policy": "whole canonical session; features strictly before anchor; query item included for T1 kNN",
        "test_consumed": False,
        "preparation_environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "pyarrow": importlib.metadata.version("pyarrow"),
            "psutil": psutil.__version__,
        },
    }
    report["preparation_source_ref"] = {
        "uri": "scripts/baselines/prepare_baseline_views.py",
        "sha256": text_sha(Path(__file__).resolve()),
        "hash_convention": "canonical_text",
    }
    report["upstream_config_refs"] = {
        rel: {"sha256": text_sha(root / rel), "hash_convention": "canonical_text"}
        for rel in (
            "config/s1-ds-05-06.v1.json",
            "docs/evidence/s1-ds-05-06/data_protocol_v1.proposed.json",
            "docs/evidence/s1-ds-09/g1_gate_v1.frozen.json",
        )
    }
    path = root / PUBLIC
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print("PREPARED_BASELINE_VIEWS_PASS", stats)


if __name__ == "__main__":
    main()
