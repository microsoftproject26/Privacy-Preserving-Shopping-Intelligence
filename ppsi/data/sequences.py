"""Session history windows - the sequential representation the GRU actually reads.

`ppsi.federated.task_examples` builds Phase1Batch objects whose history is entirely
padding, and says so: "zero-history representation. Sequential GRU history is deferred
to the final model training lane." This module is that deferral being paid.

Everything here reproduces the window semantics the S2-SMOKE notebook measured, because
its baseline comparison is only valid against windows built the same way. Three of those
semantics are load-bearing and each has already caused a wrong number once:

* The window **ends at and includes the decision event**, so `lengths` counts it.
* The window is **right-padded**: real events occupy columns `0 .. lengths-1` and the
  decision event sits at `lengths-1`. Built naively they come out left-padded, which fed
  the GRU nothing but padding and drove MRR@20 to 0.4509 - below the trivial baseline -
  while every shape and dtype check still passed.
* The window is **clipped at the session start**, so it never reaches into a previous
  session.

The frame must be sorted by (session, event_time, source_row_number) with
`order = arange(len(frame))`, because the time-gap feature reads order-adjacency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ppsi.features.hashing import ProductHashConfig, hash_product_id
from ppsi.models.batch_spec import (
    CATEGORY_OOV,
    CATEGORY_PAD,
    EVENT_PAD,
    PRICE_BAND_PAD,
)

# One day. Sessions do not span longer, and the cap stops a stale browser tab from
# dominating a feature whose whole job is to separate a fast scroll from a decision.
MAX_GAP_SECONDS = 86_400

# S1-SE-05 fixes the hash; this module does not invent one. The seed is part of the
# frozen representation, so changing it silently moves every product to a different
# embedding row and invalidates any exported model.
PRODUCT_HASH_SEED = 0
BRAND_HASH_SEED = 1

ITEM_NAMESPACE = "rees46:item:"
BRAND_NAMESPACE = "rees46:brand:"


def bucket_lookup(values, namespace: str, buckets: int, seed: int) -> dict:
    """Map each distinct raw value to its frozen hash bucket, once.

    `hash_product_id` is a per-string blake2b call. Applying it to 4.4M events directly
    would hash the same product thousands of times, so it is applied to the distinct
    values and the result is used as a lookup.
    """
    config = ProductHashConfig(bucket_count=buckets, seed=seed, residual_dim=1)
    return {value: hash_product_id(f"{namespace}{value}", config) for value in np.unique(values)}


def map_through(values, lookup: dict, missing: int):
    """Dictionary lookup over an array, with an explicit value for anything absent."""
    return np.array([lookup.get(value, missing) for value in values], dtype="int32")


@dataclass(frozen=True, slots=True)
class Windows:
    """One fixed-length history window per decision, right-padded.

    Column 0 is the oldest event in the window; column `lengths-1` is the decision event
    itself. Columns from `lengths` onward are padding and carry each channel's pad id.
    """

    category: np.ndarray  # int32   [N, L]
    product: np.ndarray  # int32   [N, L]
    event: np.ndarray  # int8    [N, L]
    brand: np.ndarray  # int32   [N, L]
    price_band: np.ndarray  # int8    [N, L]
    gap: np.ndarray  # float32 [N, L]
    lengths: np.ndarray  # int64   [N]
    target: np.ndarray  # int64   [N]
    query_category: np.ndarray  # int32   [N] - the decision event's own item
    query_product: np.ndarray  # int32   [N]
    query_brand: np.ndarray  # int32   [N]
    query_price_band: np.ndarray  # int8    [N]
    client: np.ndarray  # int64   [N]

    def __len__(self) -> int:
        return int(self.lengths.shape[0])

    @property
    def history_length(self) -> int:
        return int(self.category.shape[1])


def build_windows(frame, decisions, targets, *, history_length: int) -> Windows:
    """Build one right-padded history window per decision position.

    Parameters
    ----------
    frame
        Events for one split, sorted canonically, carrying the columns
        `session, order, category, product_bucket, event_code, brand_bucket,
        price_band, event_time, user`.
    decisions
        Positions in `frame` at which a decision is made - read from the frozen task
        examples, never rebuilt here.
    targets
        The frozen label for each decision, in the same order.
    history_length
        Number of window columns, L.
    """
    decisions = np.asarray(decisions)
    session_start = frame.groupby("session")["order"].transform("min").to_numpy()

    # Columns run oldest to newest, and the last one is the decision itself.
    offsets = np.arange(history_length - 1, -1, -1)
    index = decisions[:, None] - offsets[None, :]
    inside = index >= session_start[decisions][:, None]
    safe = np.where(inside, index, 0)

    category = frame["category"].to_numpy()
    product = frame["product_bucket"].to_numpy()
    event = frame["event_code"].to_numpy()
    brand = frame["brand_bucket"].to_numpy()
    band = frame["price_band"].to_numpy()

    # An unseen category is a real input the model must cope with, so it becomes OOV
    # here. It disqualifies a decision only as a *target*, and that happened upstream.
    category_window = np.where(inside, category[safe], CATEGORY_PAD)
    category_window = np.where((category_window < 0) & inside, CATEGORY_OOV, category_window)
    product_window = np.where(inside, product[safe], 0)
    event_window = np.where(inside, event[safe], EVENT_PAD)
    brand_window = np.where(inside, brand[safe], 0)
    band_window = np.where(inside, band[safe], PRICE_BAND_PAD)

    # `.astype("int64") // 10**9` is what S2-SMOKE used, and it is a trap. It assumes the
    # column is datetime64[ns]; pandas 3 stores this one as datetime64[us], so the divisor
    # is a thousand times too large and every gap collapses to zero. Nothing raises - the
    # channel simply goes dead, and an ablation would report "time does not help" about a
    # feature that never arrived. `as_unit("s")` states the unit instead of assuming it.
    seconds = frame["event_time"].dt.as_unit("s").astype("int64").to_numpy()
    gap = np.where(inside, seconds[safe] - seconds[np.maximum(safe - 1, 0)], 0)
    gap = np.where(inside & (index > session_start[decisions][:, None]), gap, 0)
    # A negative gap means the canonical order broke. Clipping it to zero would hide
    # that, so it is asserted instead.
    assert gap.min() >= 0, "time went backwards inside a session; the frame is not sorted"
    gap = np.log1p(np.minimum(gap, MAX_GAP_SECONDS)).astype("float32")

    lengths = inside.sum(axis=1)
    assert lengths.min() >= 1, "a decision is always inside its own session"

    # Built right-to-left the real events sit at the back. Roll each row left by its own
    # padding width so they sit at the front, which is what the readout expects.
    source = np.minimum(
        np.arange(history_length)[None, :] + (history_length - lengths)[:, None],
        history_length - 1,
    )
    keep = np.arange(history_length)[None, :] < lengths[:, None]

    def realign(window, pad):
        return np.where(keep, np.take_along_axis(window, source, axis=1), pad)

    category_window = realign(category_window, CATEGORY_PAD)
    product_window = realign(product_window, 0)
    event_window = realign(event_window, EVENT_PAD)
    brand_window = realign(brand_window, 0)
    band_window = realign(band_window, PRICE_BAND_PAD)
    gap = np.where(keep, np.take_along_axis(gap, source, axis=1), 0.0)

    # The query is the item the decision is about. T1 does not read it, but the contract
    # forbids a pad id anywhere in a query channel, and T2/T3 need it, so it is filled
    # from the decision event itself - which always exists.
    decision_category = category[decisions]
    return Windows(
        category=category_window.astype("int32"),
        product=product_window.astype("int32"),
        event=event_window.astype("int8"),
        brand=brand_window.astype("int32"),
        price_band=band_window.astype("int8"),
        gap=gap,
        lengths=lengths.astype("int64"),
        target=np.asarray(targets).astype("int64"),
        query_category=np.where(decision_category < 0, CATEGORY_OOV, decision_category).astype(
            "int32"
        ),
        query_product=product[decisions].astype("int32"),
        query_brand=brand[decisions].astype("int32"),
        query_price_band=band[decisions].astype("int8"),
        client=frame["user"].to_numpy()[decisions].astype("int64"),
    )
