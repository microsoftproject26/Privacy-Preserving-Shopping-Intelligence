"""Precompute everything about T3 that does not change between epochs.

The first version assembled each batch in Python: a loop over decisions to find their
graded rows, and a `np.fromiter` over 25,600 candidates to look up hash buckets, categories
and price bands. Measured on a T4 that produced **11% GPU utilisation at 101% CPU** - the
card finished each batch in milliseconds and then waited for the next one to be built.

Nothing in that work depends on the model or the epoch. A decision's candidate list is
fixed, its gains are fixed, and every candidate's metadata is fixed. So it is done once,
into dense arrays, and a batch becomes an indexing operation.

    gains          [decisions, 100]  float32   assembled once from the graded rows
    candidate ids  [anchors, 100]    int64     hash bucket per candidate
    category       [anchors, 100]    int64     from the frozen catalogue
    price band     [anchors, 100]    int64
    rank           [100]             float32   shared by every row

For 1.74M TRAIN decisions the gains matrix is 1.74M x 100 float32 = 696 MB. That is the
price of the trade, and it is worth paying: the alternative is rebuilding it sixteen times
over.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from train_t3 import MAX_CANDIDATES, Candidates, Ranking, resolve
from ppsi.models.batch_spec import (
    CATEGORY_OOV,
    CATEGORY_PAD,
    PRICE_BAND_PAD,
    PRODUCT_BUCKETS,
)


@dataclass(slots=True)
class CandidateTables:
    """Per-anchor candidate arrays, resolved once for every anchor in the frozen list."""

    bucket: np.ndarray      # int64 [anchors, 100]
    category: np.ndarray    # int64 [anchors, 100]
    price_band: np.ndarray  # int64 [anchors, 100]
    mask: np.ndarray        # bool  [anchors, 100]
    rank: np.ndarray        # float32 [100]


def build_candidate_tables(candidates: Candidates) -> CandidateTables:
    from ppsi.data.sequences import ITEM_NAMESPACE, PRODUCT_HASH_SEED, bucket_lookup

    started = time.time()
    items = candidates.items[:, :MAX_CANDIDATES]
    mask = candidates.mask[:, :MAX_CANDIDATES]

    catalogue = pd.read_parquet(resolve("item_catalog_v1.proposed.parquet"),
                                columns=["item", "category", "price_band"])
    known = catalogue["item"].to_numpy().astype("int64")
    buckets = bucket_lookup(known, ITEM_NAMESPACE, PRODUCT_BUCKETS, PRODUCT_HASH_SEED)

    # One pass over the distinct items that actually appear as candidates, then a vectorised
    # gather - rather than a dictionary lookup per candidate per batch per epoch.
    distinct = np.unique(items[mask])
    lookup_size = int(distinct.max()) + 1
    bucket_of = np.zeros(lookup_size, dtype="int64")
    category_of = np.full(lookup_size, CATEGORY_OOV, dtype="int64")
    band_of = np.zeros(lookup_size, dtype="int64")

    catalogue_category = dict(zip(known, catalogue["category"].astype("int64"), strict=True))
    catalogue_band = dict(zip(known, catalogue["price_band"].astype("int64"), strict=True))
    for item in distinct:
        key = int(item)
        bucket_of[key] = buckets.get(key, 0)
        category_of[key] = catalogue_category.get(key, CATEGORY_OOV)
        band_of[key] = catalogue_band.get(key, 0)

    safe = np.where(mask, items, 0)
    bucket = np.where(mask, bucket_of[safe], 0)
    category = np.where(mask, category_of[safe], CATEGORY_PAD)
    price_band = np.where(mask, band_of[safe], PRICE_BAND_PAD)
    rank = (np.arange(MAX_CANDIDATES, dtype="float32") / MAX_CANDIDATES)

    print(f"  candidate tables: {items.shape[0]:,} anchors x {MAX_CANDIDATES}, "
          f"{len(distinct):,} distinct items  ({time.time() - started:.0f}s)")
    return CandidateTables(bucket, category, price_band, mask, rank)


def build_gain_matrix(split: Ranking, candidates: Candidates,
                      anchor_of: np.ndarray) -> np.ndarray:
    """Gain of every candidate, for every decision, assembled in one pass.

    The examples give one row per (query, positive). Each row is placed at the position its
    positive occupies in that query's candidate list; a positive that was never retrieved
    has no position and contributes nothing - which is what makes the metric end-to-end.
    """
    started = time.time()
    decisions = len(split.data["lengths"])
    gains = np.zeros((decisions, MAX_CANDIDATES), dtype="float32")

    windows = split.row_to_window
    anchors = anchor_of[windows]
    valid = anchors >= 0
    # Vectorised search for each positive inside its own anchor's list: compare the whole
    # row at once instead of looping over 100 candidates.
    wanted = candidates.items[np.where(valid, anchors, 0), :MAX_CANDIDATES]
    matches = wanted == split.positive_item[:, None]
    found = matches.any(axis=1) & valid
    position = matches.argmax(axis=1)

    rows = np.flatnonzero(found)
    # np.maximum.at, not plain assignment: a decision can hold several positives and the
    # strongest engagement must win rather than whichever row happens to be written last.
    np.maximum.at(gains, (windows[rows], position[rows]), split.gain[rows])

    scoring = int((gains > 0).sum())
    print(f"  gains: {decisions:,} decisions x {MAX_CANDIDATES}  "
          f"({gains.nbytes / 1e6:.0f} MB, {scoring:,} graded candidates, "
          f"{time.time() - started:.0f}s)")
    return gains


def anchors_per_decision(split: Ranking, candidates: Candidates) -> np.ndarray:
    """The candidate row each decision's query item points at."""
    decisions = len(split.data["lengths"])
    query_of = np.zeros(decisions, dtype="int64")
    query_of[split.row_to_window] = split.query_item
    return candidates.rows_for(query_of)


def evaluable_decisions(split: Ranking) -> np.ndarray:
    """Decisions with a scoring positive - **retrieved or not**.

    This is the whole difference between an end-to-end number and a flattering one. The
    protocol is explicit: `retrieval_miss: scores zero`. A query whose engaged product was
    never retrieved still counts, and scores nothing.

    Deriving this from the gain *matrix* instead - decisions with a gain somewhere in their
    candidate list - silently drops the 18.4% the retrieval missed. Measured: 15,607
    queries instead of 18,814, and the baseline rises from 0.2046 to **0.2478**. The number
    goes up, which is exactly why the mistake is dangerous: it looks like an improvement.
    """
    scoring = split.gain > 0
    return np.unique(split.row_to_window[scoring])


def build_ideal_gains(split: Ranking, k: int = 5) -> np.ndarray:
    """The best achievable top-k gains per decision - **from the truth, not the list**.

    `build_gain_matrix` places a positive at the column its item occupies in the candidate
    list. A positive retrieval never returned has no column, so it is absent from that
    matrix entirely, and an NDCG denominator built from the matrix silently forgives the
    miss. This builds the denominator from `split.gain` instead: every positive the query
    truly had, whether retrieval found it or not.

    Returns `[decisions, k]`, gains descending, zero-filled.
    """
    decisions = len(split.data["lengths"])
    frame = pd.DataFrame({"window": split.row_to_window, "gain": split.gain})
    frame = frame[frame["gain"] > 0].sort_values(["window", "gain"],
                                                 ascending=[True, False])
    frame["rank"] = frame.groupby("window").cumcount()
    frame = frame[frame["rank"] < k]

    ideal = np.zeros((decisions, k), dtype="float32")
    ideal[frame["window"].to_numpy(), frame["rank"].to_numpy()] = frame["gain"].to_numpy()
    return ideal


def macro_by_client(ndcg: np.ndarray, clients: np.ndarray) -> tuple[float, int]:
    """Per-client mean, then the mean over clients - `ADR-001`'s declared headline.

    T3 was reported in micro while `ADR-001` fixes macro as the headline for this project,
    the same rule T1 already follows. The two differ by a lot here: the frozen retrieval
    order scores `0.2047` micro and `0.2708` macro, because clients with many queries are
    the ones with the harder ones, and micro lets them dominate.
    """
    means = per_client_means(ndcg, clients)
    return float(means.mean()), int(len(means))


def per_client_means(ndcg: np.ndarray, clients: np.ndarray) -> pd.Series:
    """Per-client mean NDCG, indexed by client and sorted - the unit a macro average and a
    paired bootstrap both operate on. Sorted, because pairing model against baseline
    requires the same clients in the same order on both sides."""
    return (pd.DataFrame({"client": clients, "ndcg": ndcg})
            .groupby("client")["ndcg"].mean().sort_index())


def clients_of_windows(split: Ranking, examples: pd.DataFrame) -> np.ndarray:
    """Owning client per decision, so a macro average has a denominator to group on."""
    decisions = len(split.data["lengths"])
    owner = np.zeros(decisions, dtype="int64")
    owner[split.row_to_window] = examples["client"].to_numpy().astype("int64")
    return owner
