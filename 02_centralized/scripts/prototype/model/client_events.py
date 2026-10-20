"""Count TRAIN events per client, once, and cache them.

The strata in `ADR-001` bucket clients by **how many TRAIN events they have**, not by how
many decisions they produce - because events are what a device would actually have to learn
from, and a client with twenty events might yield only a dozen decisions. Bucketing by
decisions shifts every client's bucket and quietly changes what the stratified table means.

This reads the raw events the same way `prepare_data.py` does, but keeps only a count per
client, so it costs one pass and a few hundred kilobytes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

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

from ppsi.data.rees46 import read_json, session_key

TASK = Path(__file__).resolve().parent
CACHE = TASK / "cache"
PROTOCOL = PROJECT / "S1-DS-05-06_Cohort_Temporal_Protocol"
RAW = PROJECT / "Dataset" / "processed_raw_parquet_v1.parquet"


def main() -> None:
    protocol = read_json(PROTOCOL / "output" / "data_protocol_v1.proposed.json")
    upstream = read_json(PROTOCOL / "config.json")
    null_prefix = upstream["session_policy"]["null_fallback_prefix"]
    start = pd.Timestamp(protocol["temporal_split"]["TRAIN"]["start"])
    end = pd.Timestamp(protocol["temporal_split"]["TRAIN"]["end_exclusive"])

    cohort = pd.read_parquet(
        PROTOCOL / "output" / "INTERNAL_DO_NOT_UPLOAD_cohort_manifest_v1.proposed.parquet"
    )
    c1 = cohort.loc[cohort["cohort"] == "C1", "user_id"].astype("int64")
    users = set(c1[c1 % 4 == 1])

    excluded = set(
        pd.read_parquet(
            PROTOCOL / "output" / "INTERNAL_DO_NOT_UPLOAD_excluded_sessions_v1.proposed.parquet"
        )["session_key"]
    )

    parquet = pq.ParquetFile(str(RAW))
    counts = pd.Series(dtype="int64")
    for group in range(parquet.metadata.num_row_groups):
        chunk = parquet.read_row_group(
            group, columns=["event_time", "user_id", "user_session", "source_row_number"]
        ).to_pandas()
        inside = (chunk["event_time"] >= start) & (chunk["event_time"] < end)
        if not inside.any():
            continue
        rows = chunk[inside]
        rows = rows[rows["user_id"].astype("int64").isin(users)]
        if rows.empty:
            continue
        rows = rows.copy()
        # The same exclusion the windows apply, so the counts describe the same events.
        rows = rows[~session_key(rows, null_prefix).isin(excluded)]
        counts = counts.add(rows["user_id"].astype("int64").value_counts(), fill_value=0)

    counts = counts.astype("int64").sort_index()
    np.save(CACHE / "train_events_per_client_index.npy", counts.index.to_numpy())
    np.save(CACHE / "train_events_per_client_values.npy", counts.to_numpy())

    edges = [10, 20, 50, 100, np.inf]
    labels = ["10-19", "20-49", "50-99", "100+"]
    buckets = pd.cut(counts, bins=edges, labels=labels, right=False)
    summary = buckets.value_counts().reindex(labels)
    print(f"clients with TRAIN events : {len(counts):,}")
    print(f"total TRAIN events        : {int(counts.sum()):,}")
    print(f"median events per client  : {counts.median():.0f}")
    print()
    print(summary.rename("clients").to_string())

    (CACHE / "client_events_summary.json").write_text(
        json.dumps(
            {
                "clients": len(counts),
                "train_events": int(counts.sum()),
                "median_events_per_client": float(counts.median()),
                "buckets": {label: int(summary[label]) for label in labels},
                "bucketed_by": "TRAIN events, per ADR-001 - not decisions",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
