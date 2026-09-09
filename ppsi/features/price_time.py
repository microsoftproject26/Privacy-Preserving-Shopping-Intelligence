"""Price banding and time-gap encoding for model inputs.

`docs/evidence/s1-d1-ds-07/price_transform_v1.proposed.json` names this module as
the owner of the production price transform.  The band edges were fitted on TRAIN
by that task and are frozen there; this module only *applies* them, so nothing
here fits, learns, or keeps split-specific state.
"""

from __future__ import annotations

import json
import math
from bisect import bisect_right
from collections.abc import Iterable
from datetime import datetime
from functools import cache
from itertools import pairwise
from numbers import Real
from pathlib import Path

TIME_GAP_CAP_SECONDS = 86_400.0
"""The 24-hour cap already used by the Phase 1 real-data sequence features."""

PRICE_TRANSFORM_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "evidence"
    / "s1-d1-ds-07"
    / "price_transform_v1.proposed.json"
)

NO_TRAIN_PRICE_BAND = 0
"""Band 0 already means "no usable TRAIN price"; it is not a separate missing flag."""

LOWEST_PRICE_BAND = 1
HIGHEST_PRICE_BAND = 4


@cache
def frozen_price_edges(path: str | None = None) -> tuple[float, ...]:
    """Return the frozen quartile price edges, ascending.

    Cached because the edges are frozen input, not state this module produces.
    """

    source = Path(path) if path is not None else PRICE_TRANSFORM_PATH
    transform = json.loads(source.read_text(encoding="utf-8"))
    edges = tuple(float(edge) for edge in transform["price_edges"])
    if list(edges) != sorted(edges):
        raise ValueError("frozen price edges must be ascending")
    return edges


def price_band(price: Real | None, *, edges: Iterable[float] | None = None) -> int:
    """Return the frozen band ``0..4`` for one item's median TRAIN price.

    ``None``, non-finite, negative, and zero prices all return
    :data:`NO_TRAIN_PRICE_BAND`, because the frozen statistic is the median TRAIN
    price *ignoring non-positive prices*.  A price sitting exactly on an edge
    belongs to the band above it.
    """

    if price is not None and (isinstance(price, bool) or not isinstance(price, Real)):
        raise TypeError("price must be a real number or None")

    if price is None:
        return NO_TRAIN_PRICE_BAND
    value = float(price)
    if not math.isfinite(value) or value <= 0:
        return NO_TRAIN_PRICE_BAND

    boundaries = frozen_price_edges() if edges is None else tuple(float(edge) for edge in edges)
    return LOWEST_PRICE_BAND + bisect_right(boundaries, value)


def encode_price_bands(
    prices: Iterable[Real | None], *, edges: Iterable[float] | None = None
) -> tuple[int, ...]:
    """Band every median price in order, reusing one frozen edge list."""

    boundaries = frozen_price_edges() if edges is None else tuple(float(edge) for edge in edges)
    return tuple(price_band(price, edges=boundaries) for price in prices)


def encode_time_gaps(event_times: Iterable[datetime]) -> tuple[float, ...]:
    """Encode gaps in an already ordered history as capped ``log1p(seconds)``.

    The first event and events with equal timestamps receive zero. A timestamp
    earlier than its predecessor is rejected instead of being silently fixed.
    """

    times = tuple(event_times)
    if any(not isinstance(event_time, datetime) for event_time in times):
        raise TypeError("event_times must contain datetime values")
    if not times:
        return ()

    encoded = [0.0]
    for previous, current in pairwise(times):
        try:
            gap_seconds = (current - previous).total_seconds()
        except TypeError as exc:
            raise ValueError("event_times must use compatible timezone information") from exc
        if gap_seconds < 0:
            raise ValueError("event_times must already be ordered")
        encoded.append(math.log1p(min(gap_seconds, TIME_GAP_CAP_SECONDS)))

    return tuple(encoded)
