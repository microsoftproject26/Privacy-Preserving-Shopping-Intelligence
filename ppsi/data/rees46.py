"""Loading real REES46 events and locating the frozen decisions inside them.

Two files, two jobs, and confusing them is the defect this module exists to prevent:

* The **task example** files say *what* to predict - one row per decision, carrying its
  position and its label. They are read, never rebuilt. Rebuilding them is how this
  project once produced 263 decisions that the upstream notebook never had, caught only
  by an assertion.
* The **raw event** file supplies the *history* to predict from. A decision's position is
  meaningless without the events that precede it.

Everything derived here comes from a frozen S1 artifact: category codes from
`vocabulary_v1`, price bands from `item_catalog_v1`, the cohort from the manifest, the
split boundaries from `data_protocol_v1`. Nothing is refitted, because refitting on
different data produces different codes for the same category and silently invalidates
every published number.

No path here points at TEST. That is not an oversight; `test_rows_read == 0` is a headline
check of the whole project and this module is where it would be broken.

**And for a long time this module *was* where it was broken.** The loader looped over every
row group, decoded it, and only then applied the timestamp mask - so TEST rows were read
into memory on the way past, and the `test_rows_read: 0` written into each result JSON was
a literal, not a measurement. No TEST row ever reached a training array, so no number was
contaminated; but the provenance claim was stronger than the evidence, which for a project
whose central discipline is the TEST seal is its own kind of defect.

`load_events` now uses the per-row-group statistics Parquet already carries to skip any
group that cannot intersect the requested window, and it counts what it actually decoded.
`rows_read_outside_window` is returned rather than asserted, so the number in a result JSON
is measured. A group straddling a boundary still has to be decoded to be filtered, and that
is exactly what the count makes visible instead of hiding.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ppsi.data.sequences import (
    BRAND_HASH_SEED,
    BRAND_NAMESPACE,
    ITEM_NAMESPACE,
    PRODUCT_HASH_SEED,
    bucket_lookup,
)
from ppsi.models.batch_spec import BRAND_BUCKETS, EVENT_CODE, PRODUCT_BUCKETS

RAW_COLUMNS = [
    "event_time",
    "event_type",
    "product_id",
    "category_id",
    "brand",
    "user_id",
    "user_session",
    "source_row_number",
]


def read_json(path: Path | str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def session_key(frame: pd.DataFrame, null_prefix: str) -> pd.Series:
    """The frozen logical session key, including the null-session singleton rule.

    A row whose `user_session` is null gets a key of its own rather than being merged with
    every other null row of that user, which would fuse unrelated visits into one session.
    """
    user = frame["user_id"].astype(str)
    raw = frame["user_session"]
    provided = pd.util.hash_pandas_object(user + "|" + raw.astype(str), index=False)
    singleton = pd.util.hash_pandas_object(
        null_prefix + "|" + user + "|" + frame["source_row_number"].astype(str), index=False
    )
    return provided.where(raw.notna(), singleton)



def _as_timestamp(value) -> pd.Timestamp:
    """A row-group statistic as a UTC timestamp, whatever Arrow handed back.

    Statistics come back as a datetime, an int of some unit, or a string depending on how
    the file was written. Guessing wrong here would silently skip a group that does hold
    wanted rows, so an unrecognised type raises rather than defaults.
    """
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def load_events(
    raw_path: Path | str,
    *,
    start,
    end,
    users: set,
    excluded: set,
    null_prefix: str,
    category_code: dict,
    price_band: pd.Series,
) -> pd.DataFrame:
    """Every event of one split for these users, canonically ordered and encoded.

    Read row group by row group. The whole file is 42.4M events and only a fraction of
    them belong to this cohort and interval; assembling the file first and filtering
    afterwards is what turned a 2.3 GB dataset into a memory problem.
    """
    parquet = pq.ParquetFile(str(raw_path))
    time_column = RAW_COLUMNS.index("event_time")
    parts = []
    provenance = {"row_groups": parquet.metadata.num_row_groups, "groups_opened": 0,
                  "groups_skipped_by_statistics": 0, "rows_decoded": 0,
                  "rows_read_outside_window": 0}

    for group in range(parquet.metadata.num_row_groups):
        # The statistics are in the footer; consulting them costs no decode. A group whose
        # own minimum is at or past `end`, or whose maximum precedes `start`, cannot hold a
        # single row we want - so it is never opened, and its rows are never read.
        stats = parquet.metadata.row_group(group).column(time_column).statistics
        if (stats is not None and stats.has_min_max
                and (_as_timestamp(stats.min) >= end
                     or _as_timestamp(stats.max) < start)):
            provenance["groups_skipped_by_statistics"] += 1
            continue

        chunk = parquet.read_row_group(group, columns=RAW_COLUMNS).to_pandas()
        provenance["groups_opened"] += 1
        provenance["rows_decoded"] += len(chunk)
        inside = (chunk["event_time"] >= start) & (chunk["event_time"] < end)
        provenance["rows_read_outside_window"] += int((~inside).sum())
        if not inside.any():
            continue
        rows = chunk[inside]
        keep = rows["user_id"].astype("int64").isin(users)
        if not keep.any():
            continue
        rows = rows[keep].copy()
        rows["user"] = rows["user_id"].astype("int64")
        rows["session"] = session_key(rows, null_prefix)
        parts.append(rows.drop(columns=["user_id", "user_session"]))

    frame = pd.concat(parts, ignore_index=True)
    del parts
    frame = frame[~frame["session"].isin(excluded)]
    # Carried on the frame rather than returned, so the five existing callers keep working
    # and any of them can record measured provenance instead of writing a constant.
    frame.attrs["provenance"] = provenance

    # The canonical order. The time gap reads order-adjacency, so sorting differently here
    # produces a different feature for the same event.
    frame = frame.sort_values(["session", "event_time", "source_row_number"])
    frame = frame.drop(columns=["source_row_number"]).reset_index(drop=True)
    frame["order"] = np.arange(len(frame), dtype="int64")

    item = frame["product_id"].astype("int64")
    frame["item"] = item
    # A category absent from the frozen vocabulary becomes -1, exactly as upstream encodes
    # it. It is turned into OOV only when it enters the model as an input.
    frame["category"] = (
        frame["category_id"].astype("int64").map(category_code).fillna(-1).astype("int32")
    )
    frame["event_code"] = frame["event_type"].map(EVENT_CODE).astype("int8")
    assert frame["event_code"].notna().all(), "an event type outside view/cart/purchase"

    products = bucket_lookup(item.to_numpy(), ITEM_NAMESPACE, PRODUCT_BUCKETS, PRODUCT_HASH_SEED)
    frame["product_bucket"] = item.map(products).astype("int32")

    brand = frame["brand"].fillna("__UNK__").astype(str)
    brands = bucket_lookup(brand.to_numpy(), BRAND_NAMESPACE, BRAND_BUCKETS, BRAND_HASH_SEED)
    frame["brand_bucket"] = brand.map(brands).astype("int32")

    # Bands come from the frozen catalogue. An item the catalogue never saw gets band 0,
    # which already means "no usable TRAIN price" - the same statement, not a new one.
    frame["price_band"] = item.map(price_band).fillna(0).astype("int8")

    return frame.drop(columns=["product_id", "category_id", "brand", "event_type"])


def locate_decisions(frame: pd.DataFrame, examples_path: Path | str) -> tuple:
    """Find each frozen decision inside the loaded events and carry its label across.

    The join is on `(session, decision_order)`. Every frozen decision must be found: a
    missing one means the events were loaded differently from the way the examples were
    built, and the resulting model would be scored against a baseline measured on a
    different set of rows.
    """
    examples = pd.read_parquet(examples_path, columns=["session", "decision_order", "label_value"])
    located = frame.reset_index(names="position").merge(
        examples, left_on=["session", "order"], right_on=["session", "decision_order"], how="inner"
    )
    assert len(located) == len(examples), (
        f"{len(located):,} of {len(examples):,} frozen decisions could not be located in "
        "the loaded events; the two are not describing the same rows"
    )

    located = located.sort_values("position")
    positions = located["position"].to_numpy()
    labels = located["label_value"].to_numpy().astype("int64")
    assert labels.min() >= 0, "a T1 label must be a real category code"
    return positions, labels
