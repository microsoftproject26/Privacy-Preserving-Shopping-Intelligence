"""Build the history windows the T2 and T3 heads will need, while the GPU is busy.

`S2-DS-06` and `S2-DS-07` put a new head on the encoder `S2-DS-01` trains, so neither can
start until that checkpoint exists. But their *windows* do not depend on the model at all,
and building them is CPU work - so it happens now rather than adding an hour to the front
of each task.

The two tasks do not share T1's decision points. T2 asks about the first view of a product
in a session; T1 asks about the first later event on a different item. Only 73.6% of T2's
VALIDATION decisions coincide with a T1 one, so 103,732 of them need a window that does not
exist yet.

T3 differs again: its examples carry one row per (query, eligible positive), so 8M rows
describe far fewer distinct decisions. Windows are built once per distinct decision and an
index maps each row to its window - otherwise the same session history would be stored
tens of millions of times.

Nothing here reads TEST.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# These scripts live in two places: beside the task folder on the machine that ran them,
# and in `scripts/model/` in the repository. Rather than hard-code either layout, walk up
# until the directory holding the `ppsi` package appears.
PROJECT = Path(__file__).resolve().parent.parent
for _candidate in Path(__file__).resolve().parents:
    if (_candidate / "ppsi" / "__init__.py").exists():
        sys.path.insert(0, str(_candidate))
        break
else:  # the task-folder layout, where the package sits in a sibling checkout
    sys.path.insert(0, str(PROJECT / "Repo_S2DS01"))

from ppsi.data.rees46 import load_events, read_json
from ppsi.data.sequences import build_windows

TASK = Path(__file__).resolve().parent

_ROOT = os.environ.get("PPSI_DATA_ROOT")
if _ROOT:
    _ROOT = Path(_ROOT)
    D1, PROTOCOL_DIR, PROTOCOL_OUT = _ROOT / "examples", _ROOT / "protocol", _ROOT / "protocol"
    RAW = _ROOT / "raw" / "processed_raw_parquet_v1.parquet"
else:
    D1 = PROJECT / "S1-D1-DS-07_T3_Protocol_and_Task_Examples" / "output"
    PROTOCOL_DIR = PROJECT / "S1-DS-05-06_Cohort_Temporal_Protocol"
    PROTOCOL_OUT = PROTOCOL_DIR / "output"
    RAW = PROJECT / "Dataset" / "processed_raw_parquet_v1.parquet"

CONFIG = read_json(TASK / "config.json")
HISTORY = CONFIG["model"]["history_length"]
CHUNK = 200_000

WINDOW_CHANNELS = {
    "category": "int32",
    "product": "int32",
    "event": "int8",
    "brand": "int32",
    "price_band": "int8",
    "gap": "float32",
}
SCALARS = {
    "lengths": "int64",
    "target": "int64",
    "query_category": "int32",
    "query_product": "int32",
    "query_brand": "int32",
    "query_price_band": "int8",
    "client": "int64",
}


def distinct_decisions(frame: pd.DataFrame, examples: pd.DataFrame) -> tuple:
    """Positions of each distinct decision, and each example row's index into them.

    T3 carries one row per (query, positive). Building a window per row would store the
    same session history for every positive of the same query - millions of duplicates of
    identical arrays.
    """
    keys = examples[["session", "decision_order"]].drop_duplicates().reset_index(drop=True)
    keys["window"] = np.arange(len(keys), dtype="int64")

    located = frame.reset_index(names="position").merge(
        keys, left_on=["session", "order"], right_on=["session", "decision_order"], how="inner"
    )
    assert len(located) == len(keys), (
        f"{len(located):,} of {len(keys):,} distinct decisions could not be located in the "
        "loaded events; the examples and the events do not describe the same rows"
    )

    located = located.sort_values("window")
    row_to_window = examples.merge(keys, on=["session", "decision_order"], how="left")["window"]
    assert row_to_window.notna().all(), "an example row has no window"
    return located["position"].to_numpy(), row_to_window.to_numpy().astype("int64")


def write_windows(cache: Path, split: str, frame, positions, targets) -> dict:
    total = len(positions)
    arrays = {}
    for name, dtype in WINDOW_CHANNELS.items():
        arrays[name] = np.lib.format.open_memmap(
            cache / f"{split}_{name}.npy", mode="w+", dtype=dtype, shape=(total, HISTORY)
        )
    for name, dtype in SCALARS.items():
        arrays[name] = np.lib.format.open_memmap(
            cache / f"{split}_{name}.npy", mode="w+", dtype=dtype, shape=(total,)
        )

    for start in range(0, total, CHUNK):
        stop = min(start + CHUNK, total)
        built = build_windows(
            frame, positions[start:stop], targets[start:stop], history_length=HISTORY
        )
        for name, array in arrays.items():
            array[start:stop] = getattr(built, name)
        del built
        print(f"    windows {stop:>9,} / {total:,}", end="\r")
    for array in arrays.values():
        array.flush()

    lengths = arrays["lengths"][:]
    print(f"\n    {total:,} windows, mean history {lengths.mean():.2f}")
    return {"windows": int(total), "mean_history": round(float(lengths.mean()), 2)}


def main() -> None:
    started = time.time()
    protocol = read_json(PROTOCOL_OUT / "data_protocol_v1.proposed.json")
    upstream = read_json(PROTOCOL_DIR / "config.json")
    vocabulary = read_json(D1 / "vocabulary_v1.proposed.json")
    category_code = {
        int(k): int(v) for k, v in vocabulary["categories"]["code_of_category_id"].items()
    }
    catalog = pd.read_parquet(
        D1 / "item_catalog_v1.proposed.parquet", columns=["item", "price_band"]
    )
    price_band = catalog.set_index("item")["price_band"]

    cohort = pd.read_parquet(
        PROTOCOL_OUT / "INTERNAL_DO_NOT_UPLOAD_cohort_manifest_v1.proposed.parquet"
    )
    c1 = cohort.loc[cohort["cohort"] == "C1", "user_id"].astype("int64")
    users = set(c1[c1 % 4 == 1])
    excluded = set(
        pd.read_parquet(
            PROTOCOL_OUT / "INTERNAL_DO_NOT_UPLOAD_excluded_sessions_v1.proposed.parquet"
        )["session_key"]
    )
    splits = {
        name: (pd.Timestamp(v["start"]), pd.Timestamp(v["end_exclusive"]))
        for name, v in protocol["temporal_split"].items()
    }
    null_prefix = upstream["session_policy"]["null_fallback_prefix"]

    summary = {}
    for task in ("t2", "t3"):
        cache = TASK / f"cache_{task}"
        cache.mkdir(exist_ok=True)
        summary[task] = {}
        for split, suffix in (("TRAIN", "_train"), ("VALIDATION", "")):
            print(f"\n=== {task.upper()} {split} " + "=" * 50)
            start, end = splits[split]
            frame = load_events(
                RAW,
                start=start,
                end=end,
                users=users,
                excluded=excluded,
                null_prefix=null_prefix,
                category_code=category_code,
                price_band=price_band,
            )

            path = D1 / f"INTERNAL_DO_NOT_UPLOAD_task_examples_{task}{suffix}_v1.proposed.parquet"
            examples = pd.read_parquet(path)
            print(f"  events {len(frame):,} | example rows {len(examples):,}")

            positions, row_to_window = distinct_decisions(frame, examples)
            print(
                f"  distinct decisions {len(positions):,} "
                f"({len(examples) / max(len(positions), 1):.1f} rows each)"
            )

            name = f"{task}_{split.lower()}"
            # The window target is a placeholder here: T2's label is a purchase flag and
            # T3's is a relevance gain, and both live in the label file rather than in the
            # window. Only the history is being cached.
            stats = write_windows(
                cache, name, frame, positions, np.zeros(len(positions), dtype="int64")
            )
            np.save(cache / f"{name}_row_to_window.npy", row_to_window)

            labels = {
                "label_value": examples["label_value"].to_numpy(),
                "task_mask": examples["task_mask"].to_numpy(),
            }
            for key, values in labels.items():
                np.save(cache / f"{name}_{key}.npy", values)

            masked = int((~examples["task_mask"]).sum())
            stats.update(
                {
                    "example_rows": len(examples),
                    "censored_rows": masked,
                    "censored_percent": round(masked / len(examples) * 100, 2),
                }
            )
            print(
                f"    example rows {len(examples):,} | censored {masked:,} "
                f"({stats['censored_percent']}%)"
            )
            summary[task][split] = stats
            del frame, examples

    (TASK / "cache_heads_summary.json").write_text(
        json.dumps({"history_length": HISTORY, "tasks": summary, "test_rows_read": 0}, indent=2),
        encoding="utf-8",
    )
    print("\n" + "=" * 70)
    for task, splits_done in summary.items():
        for split, stats in splits_done.items():
            print(
                f"  {task.upper():3} {split:11} {stats['windows']:>9,} windows  "
                f"{stats['example_rows']:>10,} rows  "
                f"{stats['censored_percent']:>5}% censored"
            )
    print(f"\ntotal {(time.time() - started) / 60:.1f} min   TEST was not opened.")


if __name__ == "__main__":
    main()
