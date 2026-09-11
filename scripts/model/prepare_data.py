"""Build the history windows S2-DS-01 trains on, once, and cache them to disk.

Reading 42.4M raw events and assembling 3.5M windows takes minutes; a tuning sweep runs
this many times. So it runs once here and every later run memory-maps the result.

Windows are written straight into memory-mapped arrays in chunks. Held in RAM the six
channels come to roughly 1.1 GB for TRAIN alone, and the temporary copies numpy makes
while building them multiply that several times over - which is exactly the ceiling that
forced the 25% slice in the first place. Chunking makes the peak independent of how many
decisions there are.
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

from ppsi.data.rees46 import load_events, locate_decisions, read_json
from ppsi.data.sequences import build_windows

TASK = Path(__file__).resolve().parent
CACHE = TASK / "cache"
CACHE.mkdir(exist_ok=True)

# Two layouts, one script. On this machine the inputs sit inside the project tree; in a
# cloud bundle they are copied into one flat `data/` folder. PPSI_DATA_ROOT picks the
# second without forking the code, because a second copy of a loader is a second place for
# the rules to drift.
_ROOT = os.environ.get("PPSI_DATA_ROOT")
if _ROOT:
    _ROOT = Path(_ROOT)
    D1 = _ROOT / "examples"
    PROTOCOL_DIR = _ROOT / "protocol"
    PROTOCOL_OUT = _ROOT / "protocol"
    RAW = _ROOT / "raw" / "processed_raw_parquet_v1.parquet"
else:
    D1 = PROJECT / "S1-D1-DS-07_T3_Protocol_and_Task_Examples" / "output"
    PROTOCOL_DIR = PROJECT / "S1-DS-05-06_Cohort_Temporal_Protocol"
    PROTOCOL_OUT = PROTOCOL_DIR / "output"
    RAW = PROJECT / "Dataset" / "processed_raw_parquet_v1.parquet"

config = read_json(TASK / "config.json")
HISTORY = config["model"]["history_length"]
CHUNK = 200_000

CHANNELS = {
    "category": ("int32", (HISTORY,)),
    "product": ("int32", (HISTORY,)),
    "event": ("int8", (HISTORY,)),
    "brand": ("int32", (HISTORY,)),
    "price_band": ("int8", (HISTORY,)),
    "gap": ("float32", (HISTORY,)),
    "lengths": ("int64", ()),
    "target": ("int64", ()),
    "query_category": ("int32", ()),
    "query_product": ("int32", ()),
    "query_brand": ("int32", ()),
    "query_price_band": ("int8", ()),
    "client": ("int64", ()),
}


def prepare(
    split: str,
    examples_path: Path,
    start,
    end,
    users,
    excluded,
    null_prefix,
    category_code,
    price_band,
) -> dict:
    started = time.time()
    print(f"\n=== {split} " + "=" * 60)
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
    print(f"  events loaded        : {len(frame):,}  ({time.time() - started:.0f}s)")
    print(f"  sessions             : {frame['session'].nunique():,}")
    print(f"  clients              : {frame['user'].nunique():,}")
    print(f"  categories unseen    : {int((frame['category'] < 0).sum()):,}")

    positions, labels = locate_decisions(frame, examples_path)
    total = len(positions)
    print(f"  frozen decisions     : {total:,}  - all located")

    out = {}
    for name, (dtype, shape) in CHANNELS.items():
        path = CACHE / f"{split.lower()}_{name}.npy"
        out[name] = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=(total, *shape))

    for start_row in range(0, total, CHUNK):
        stop = min(start_row + CHUNK, total)
        built = build_windows(
            frame, positions[start_row:stop], labels[start_row:stop], history_length=HISTORY
        )
        for name in CHANNELS:
            out[name][start_row:stop] = getattr(built, name)
        del built
        print(f"    windows {stop:>9,} / {total:,}", end="\r")

    for array in out.values():
        array.flush()
    print(f"\n  windows written      : {total:,}  ({time.time() - started:.0f}s total)")

    # The checks that would have caught the two defects this pipeline already survived.
    lengths = out["lengths"][:]
    category = out["category"]
    last = category[np.arange(min(total, 100_000)), lengths[: min(total, 100_000)] - 1]
    assert (last >= 0).all(), "a decision's own category is missing from its window"
    real = out["gap"][:100_000][np.arange(HISTORY)[None, :] < lengths[:100_000, None]]
    assert real.max() > 0, "every time gap is zero - the timestamp unit is wrong again"
    print(f"  mean real history    : {lengths.mean():.1f} events")
    print(f"  gap channel is alive : max {real.max():.3f}, distinct {len(np.unique(real)):,}")

    del frame
    return {
        "split": split,
        "decisions": int(total),
        "mean_history": float(lengths.mean()),
        "seconds": round(time.time() - started, 1),
    }


def main() -> None:
    protocol = read_json(PROTOCOL_OUT / "data_protocol_v1.proposed.json")
    upstream = read_json(PROTOCOL_DIR / "config.json")
    vocabulary = read_json(D1 / "vocabulary_v1.proposed.json")

    category_code = {
        int(k): int(v) for k, v in vocabulary["categories"]["code_of_category_id"].items()
    }
    print(f"frozen vocabulary      : {len(category_code):,} categories")

    catalog = pd.read_parquet(
        D1 / "item_catalog_v1.proposed.parquet", columns=["item", "price_band"]
    )
    price_band = catalog.set_index("item")["price_band"]
    print(f"frozen catalogue       : {len(price_band):,} items with a price band")

    cohort = pd.read_parquet(
        PROTOCOL_OUT / "INTERNAL_DO_NOT_UPLOAD_cohort_manifest_v1.proposed.parquet"
    )
    c1 = cohort.loc[cohort["cohort"] == "C1", "user_id"].astype("int64")
    modulo = config["scope"]["user_slice"]
    users = set(c1[c1 % 4 == 1])
    print(f"cohort C1              : {len(c1):,} clients")
    print(f"measured slice         : {len(users):,} clients  ({modulo})")

    excluded = set(
        pd.read_parquet(
            PROTOCOL_OUT / "INTERNAL_DO_NOT_UPLOAD_excluded_sessions_v1.proposed.parquet"
        )["session_key"]
    )
    print(f"excluded sessions      : {len(excluded):,}")

    splits = {
        name: (pd.Timestamp(v["start"]), pd.Timestamp(v["end_exclusive"]))
        for name, v in protocol["temporal_split"].items()
    }
    null_prefix = upstream["session_policy"]["null_fallback_prefix"]

    summary = []
    for split, examples in [
        ("TRAIN", D1 / "INTERNAL_DO_NOT_UPLOAD_task_examples_t1_train_v1.proposed.parquet"),
        ("VALIDATION", D1 / "INTERNAL_DO_NOT_UPLOAD_task_examples_t1_v1.proposed.parquet"),
    ]:
        start, end = splits[split]
        summary.append(
            prepare(
                split, examples, start, end, users, excluded, null_prefix, category_code, price_band
            )
        )

    (CACHE / "prepare_summary.json").write_text(
        json.dumps({"history_length": HISTORY, "splits": summary, "test_rows_read": 0}, indent=2),
        encoding="utf-8",
    )
    print("\n" + "=" * 70)
    print(pd.DataFrame(summary).to_string(index=False))
    print("\nTEST was not opened.  test_rows_read = 0")


if __name__ == "__main__":
    main()
