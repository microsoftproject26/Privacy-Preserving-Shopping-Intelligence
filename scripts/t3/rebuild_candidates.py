"""Extend the T3 candidate lists to every TRAIN query item, using only TRAIN.

The frozen candidate artifact builds lists for `47,948` anchors, chosen as the query items
that turned out to have a positive **in VALIDATION**:

```python
# Only anchors with a positive are ever scored, so only those need a list.
anchor_meta = queries.loc[queries["query_item"].isin(positives["query_item"].unique()), ...]
```

The reasoning is sound for *evaluation* - an anchor with no positive contributes nothing to
NDCG either way - and we checked that every one of the 18,814 evaluable queries has its
list, so the published comparison is fair. The cost is elsewhere: **263,112 of 1,740,433
TRAIN queries (15.12%) have no list at all**, and only `106,910` (6.14%) carry a retrieved
positive a listwise loss can learn from. A future VALIDATION outcome decides which TRAIN
rows are allowed to update the model, which is training-selection leakage and not merely a
deployment limit.

## Why this is safe to do in our lane

Each anchor's list depends only on that anchor: its category, its co-occurrence counts, and
the TRAIN popularity of the items in its pool. **Adding anchors cannot change an existing
anchor's list.** So:

| | effect |
|---|---|
| VALIDATION evaluable queries | unchanged - all 18,814 already had lists |
| their candidate lists | unchanged |
| the `0.2707` macro baseline | **still valid** |
| TRAIN queries with no list | 15.12% → near zero |

The gate below asserts the first two rather than assuming them: the 47,948 existing anchors
are rebuilt and compared row for row against the frozen file. **If a single list differs, the
reconstruction is wrong and nothing is written.** A rebuilt artifact that quietly disagreed
with the frozen one would invalidate every T3 number without any symptom.

Nothing upstream is modified. This writes a new file in this task's output, the way
`S2-DS-06/labels.py` corrects the T2 labels without touching their source.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
TASK = Path(__file__).resolve().parent
OUTPUT = TASK / "output"
sys.path.insert(0, str(PROJECT / "Repo_S2DS01"))

from ppsi.data.rees46 import load_events, read_json          # noqa: E402

EXAMPLES = PROJECT / "S1-D1-DS-07_T3_Protocol_and_Task_Examples/output"
PROTOCOL_OUT = PROJECT / "S1-DS-05-06_Cohort_Temporal_Protocol/output"
RAW = PROJECT / "Dataset/processed_raw_parquet_v1.parquet"
SELECTED = "cooccurrence_then_popularity"


def frozen_inputs() -> dict:
    """Everything the upstream build read, read the same way and never refitted."""
    catalog = pd.read_parquet(EXAMPLES / "item_catalog_v1.proposed.parquet")
    protocol = read_json(EXAMPLES / "t3_protocol_v1.proposed.json")
    upstream = read_json(PROTOCOL_OUT / "data_protocol_v1.proposed.json")

    vocabulary = read_json(EXAMPLES / "vocabulary_v1.proposed.json")
    category_code = {int(k): int(v)
                     for k, v in vocabulary["categories"]["code_of_category_id"].items()}
    price_band = catalog.set_index("item")["price_band"]

    # Read exactly the way `S2-DS-01/prepare_data.py` reads them. A second spelling of the
    # same cohort or the same window would build lists over a different population, and the
    # gate below would then fail for a reason that has nothing to do with the construction.
    cohort = pd.read_parquet(
        PROTOCOL_OUT / "INTERNAL_DO_NOT_UPLOAD_cohort_manifest_v1.proposed.parquet")
    c1 = cohort.loc[cohort["cohort"] == "C1", "user_id"].astype("int64")
    users = set(c1[c1 % 4 == 1])
    excluded = set(pd.read_parquet(
        PROTOCOL_OUT
        / "INTERNAL_DO_NOT_UPLOAD_excluded_sessions_v1.proposed.parquet")["session_key"])
    splits = {name: (pd.Timestamp(v["start"]), pd.Timestamp(v["end_exclusive"]))
              for name, v in upstream["temporal_split"].items()}
    session_policy = read_json(
        PROJECT / "S1-DS-05-06_Cohort_Temporal_Protocol" / "config.json")
    return {"catalog": catalog, "protocol": protocol, "users": users,
            "excluded": excluded, "train_window": splits["TRAIN"],
            "null_prefix": session_policy["session_policy"]["null_fallback_prefix"],
            "category_code": category_code, "price_band": price_band}


def build_lists(train: pd.DataFrame, anchor_meta: pd.DataFrame, catalog: pd.DataFrame,
                cap: int, require_band: bool) -> pd.DataFrame:
    """The frozen construction, verbatim, over whichever anchors it is handed.

    `anchor_meta` carries `query_item`, `category` and `price_band`, and it comes from the
    **query rows**, not from the catalogue. That is not a detail: **3,928 of the 47,948
    frozen anchors are not in the item catalogue at all**, and deriving anchor metadata from
    the catalogue silently drops every one of them. The first attempt did exactly that and
    rebuilt 4,182,599 rows against the frozen 4,554,279 - caught by the gate below, which is
    the only reason it is a paragraph here rather than a wrong artifact.
    """
    keys = ["category", "price_band"] if require_band else ["category"]
    pool = catalog[["item", "category", "price_band", "popularity"]].copy()
    pool = pool.sort_values(keys + ["popularity", "item"],
                            ascending=[True] * len(keys) + [False, True])
    pool["pop_rank"] = pool.groupby(keys).cumcount()

    joined = pool.loc[pool["pop_rank"] <= cap].merge(
        anchor_meta[["query_item"] + keys], on=keys)
    popularity_lists = joined.loc[joined["item"] != joined["query_item"]].sort_values(
        ["query_item", "pop_rank"])
    popularity_lists["rank"] = popularity_lists.groupby("query_item").cumcount()
    popularity_lists = popularity_lists.loc[popularity_lists["rank"] < cap,
                                            ["query_item", "item", "rank"]]
    del joined

    # Whole-session co-occurrence, restricted to the anchors that need a list.
    pairs_source = train[["session", "item", "category"]].drop_duplicates()
    anchored = pairs_source.loc[pairs_source["item"].isin(set(anchor_meta["query_item"]))]
    pairs = anchored.merge(pairs_source, on=["session", "category"], suffixes=("_q", "_c"))
    pairs = pairs.loc[pairs["item_q"] != pairs["item_c"]]
    cooccurrence = pairs.groupby(["item_q", "item_c"]).size().rename("count").reset_index()
    del pairs, anchored, pairs_source

    cooc = cooccurrence.rename(columns={"item_q": "query_item", "item_c": "item"}).merge(
        pool[["item", "category", "price_band", "popularity"]], on="item").merge(
        anchor_meta, on="query_item", suffixes=("", "_anchor"))
    eligible = ((cooc["category"] == cooc["category_anchor"])
                & (cooc["item"] != cooc["query_item"]))
    if require_band:
        eligible &= cooc["price_band"] == cooc["price_band_anchor"]
    cooc = cooc.loc[eligible].sort_values(["query_item", "count", "popularity", "item"],
                                          ascending=[True, False, False, True])
    cooc["rank"] = cooc.groupby("query_item").cumcount()
    head = cooc.loc[cooc["rank"] < cap, ["query_item", "item", "rank"]]
    del cooc, cooccurrence

    # Popularity fills whatever co-occurrence left empty, without repeating an item. The
    # tail's rank continues from where the head stopped - `cumcount() + offset`, exactly as
    # upstream wrote it - rather than being renumbered after a concat. The two give the same
    # ordering, and matching the original spelling is what keeps the gate meaningful.
    placed = pd.MultiIndex.from_frame(head[["query_item", "item"]])
    tail = popularity_lists[["query_item", "item"]]
    tail = tail.loc[~pd.MultiIndex.from_frame(tail).isin(placed)].copy()
    tail["offset"] = (tail["query_item"].map(head.groupby("query_item").size())
                      .fillna(0).astype("int64"))
    tail["rank"] = tail.groupby("query_item").cumcount() + tail["offset"]
    combined = pd.concat([head, tail[["query_item", "item", "rank"]]], ignore_index=True)
    return combined.loc[combined["rank"] < cap].reset_index(drop=True)


def main() -> None:
    started = time.time()
    inputs = frozen_inputs()
    cap = int(inputs["protocol"]["retrieval"]["max_candidates"])
    require_band = bool(inputs["protocol"].get("eligibility", {})
                        .get("price_band_required", False))
    print(f"  cap {cap}, same price band required: {require_band}")

    frozen = pd.read_parquet(EXAMPLES / "t3_candidate_lists_v1.proposed.parquet")
    frozen = frozen.loc[frozen["retrieval"] == SELECTED,
                        ["query_item", "item", "rank"]].reset_index(drop=True)
    frozen_anchors = np.sort(frozen["query_item"].unique())
    print(f"  frozen artifact: {len(frozen):,} rows over {len(frozen_anchors):,} anchors")

    # Anchor metadata comes from the query rows, which is where upstream took it from.
    # `price_band` is carried only because the merge keys name it; the frozen protocol sets
    # `price_band_required: false`, so it never affects eligibility.
    def anchors_from(path: Path) -> pd.DataFrame:
        rows = pd.read_parquet(path, columns=["query_item", "category"]).drop_duplicates()
        return rows.assign(price_band=0)

    train_meta = anchors_from(
        EXAMPLES / "INTERNAL_DO_NOT_UPLOAD_task_examples_t3_train_v1.proposed.parquet")
    validation_meta = anchors_from(
        EXAMPLES / "INTERNAL_DO_NOT_UPLOAD_task_examples_t3_v1.proposed.parquet")
    train_anchors = np.sort(train_meta["query_item"].unique())
    print(f"  TRAIN query items: {len(train_anchors):,}")
    print(f"  missing a list  : {len(set(train_anchors) - set(frozen_anchors)):,}")

    every_meta = (pd.concat([validation_meta, train_meta], ignore_index=True)
                  .drop_duplicates("query_item"))
    frozen_meta = every_meta.loc[every_meta["query_item"].isin(set(frozen_anchors))]
    assert len(frozen_meta) == len(frozen_anchors), (
        f"{len(frozen_meta):,} of {len(frozen_anchors):,} frozen anchors have query "
        "metadata; the rebuild cannot reproduce a list for an anchor it cannot describe")

    start, end = inputs["train_window"]
    print(f"\n  loading TRAIN events {start.date()} .. {end.date()} ...")
    train = load_events(RAW, start=start, end=end, users=inputs["users"],
                        excluded=inputs["excluded"], null_prefix=inputs["null_prefix"],
                        category_code=inputs["category_code"],
                        price_band=inputs["price_band"])
    print(f"  events {len(train):,}, sessions {train['session'].nunique():,} "
          f"({time.time() - started:.0f}s)")

    # --- the gate: rebuild what already exists and demand it match exactly -------------
    print("\n  gate: rebuilding the frozen anchors and comparing row for row")
    replica = build_lists(train, frozen_meta, inputs["catalog"], cap, require_band)
    left = frozen.sort_values(["query_item", "rank"]).reset_index(drop=True)
    right = replica.sort_values(["query_item", "rank"]).reset_index(drop=True)
    if len(left) != len(right) or not left.equals(right):
        differing = int((left["item"].to_numpy() != right["item"].to_numpy()).sum()) \
            if len(left) == len(right) else -1
        raise SystemExit(
            f"the rebuild does not reproduce the frozen lists ({len(left):,} rows against "
            f"{len(right):,}, {differing} differing items). The construction here is not the "
            "one upstream used, so an extended artifact built from it would silently change "
            "every T3 number. Nothing written.")
    print(f"  ok  {len(left):,} rows reproduce exactly")

    # --- the extension -----------------------------------------------------------------
    print(f"\n  building lists for {len(every_meta):,} anchors ...")
    extended = build_lists(train, every_meta, inputs["catalog"], cap, require_band)

    # And the same check again, now inside the bigger artifact.
    inside = extended.loc[extended["query_item"].isin(set(frozen_anchors))]
    inside = inside.sort_values(["query_item", "rank"]).reset_index(drop=True)
    assert inside.equals(left), (
        "adding anchors changed an existing anchor's list, which the construction should "
        "make impossible. The baseline would no longer be comparable.")
    print(f"  ok  the frozen anchors are byte-identical inside the extended artifact")

    covered = extended["query_item"].nunique()
    still_missing = len(set(train_anchors) - set(extended["query_item"].unique()))
    path = OUTPUT / "t3_candidate_lists_train_anchors_v1.parquet"
    extended.assign(retrieval=SELECTED).to_parquet(path, index=False)

    summary = {
        "task": "S2-DS-07", "artifact": path.name,
        "derived_from": "t3_candidate_lists_v1.proposed.parquet, same construction",
        "retrieval": SELECTED, "max_candidates": cap,
        "anchors_frozen": int(len(frozen_anchors)),
        "anchors_train": int(len(train_anchors)),
        "anchors_total": int(covered),
        "train_anchors_still_without_a_list": int(still_missing),
        "rows": int(len(extended)),
        "frozen_lists_reproduce_exactly": True,
        "validation_baseline_unchanged": True,
        "why": ("each anchor's list depends only on that anchor, so adding anchors cannot "
                "change an existing list. Asserted twice: once against the frozen file and "
                "once inside the extended artifact."),
        "test_seal": {"test_rows_used": 0, "measured": "_SEAL/test_seal_measured.json"},
    }
    (OUTPUT / "s2_ds_07_candidates_rebuild.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  wrote {len(extended):,} rows over {covered:,} anchors -> {path.name}")
    print(f"  TRAIN anchors still without a list: {still_missing:,}")
    print(f"\ntotal {(time.time() - started) / 60:.0f} min")


if __name__ == "__main__":
    main()
