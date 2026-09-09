from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from ppsi.features.price_time import (
    HIGHEST_PRICE_BAND,
    LOWEST_PRICE_BAND,
    NO_TRAIN_PRICE_BAND,
    PRICE_TRANSFORM_PATH,
    TIME_GAP_CAP_SECONDS,
    encode_price_bands,
    encode_time_gaps,
    frozen_price_edges,
    price_band,
)

CATALOG_PATH = (
    Path(__file__).resolve().parents[2] / "fixtures" / "reference" / "item_catalog_v1.proposed.parquet"
)

# One catalogue item is priced exactly at the top edge and was banded by the original
# rank-based quartile cut rather than by the edge comparison. Recording it keeps the
# reconciliation exact instead of loosening the assertion to a tolerance.
KNOWN_EDGE_TIE_PRICE = 194.17


def utc(minute: int, second: int = 0) -> datetime:
    return datetime(2019, 10, 23, 10, minute, second, tzinfo=UTC)


def test_frozen_edges_match_the_committed_transform() -> None:
    transform = json.loads(PRICE_TRANSFORM_PATH.read_text(encoding="utf-8"))

    assert frozen_price_edges() == tuple(transform["price_edges"])
    assert transform["fitted_on"] == "TRAIN only"


@pytest.mark.parametrize("missing", [None, float("nan"), float("inf"), float("-inf"), -1.0, -0.01, 0.0])
def test_unusable_prices_all_become_band_zero(missing) -> None:
    assert price_band(missing) == NO_TRAIN_PRICE_BAND


def test_bands_span_the_frozen_range_and_rise_with_price() -> None:
    edges = frozen_price_edges()
    prices = [edges[0] / 2, edges[0] + 1, edges[1] + 1, edges[2] + 1]

    bands = encode_price_bands(prices)

    assert bands == (LOWEST_PRICE_BAND, 2, 3, HIGHEST_PRICE_BAND)
    assert list(bands) == sorted(bands)


def test_a_price_exactly_on_an_edge_belongs_to_the_band_above() -> None:
    edges = frozen_price_edges()

    assert price_band(edges[0]) == 2
    assert price_band(math.nextafter(edges[0], 0.0)) == LOWEST_PRICE_BAND


def test_banding_is_deterministic_and_rejects_non_numbers() -> None:
    prices = [None, 5.0, 500.0, 0.0]

    assert encode_price_bands(prices) == encode_price_bands(prices)
    with pytest.raises(TypeError):
        price_band("29.58")
    with pytest.raises(TypeError):
        price_band(True)


def test_module_reproduces_the_frozen_catalogue_bands() -> None:
    catalog = pl.read_parquet(CATALOG_PATH, columns=["median_price", "price_band"])
    prices = catalog.get_column("median_price").to_list()
    expected = catalog.get_column("price_band").to_list()

    computed = encode_price_bands(prices)

    disagreements = [
        (price, want, got)
        for price, want, got in zip(prices, expected, computed, strict=True)
        if want != got
    ]
    assert [price for price, _, _ in disagreements] == [KNOWN_EDGE_TIE_PRICE]
    assert computed.count(NO_TRAIN_PRICE_BAND) == expected.count(NO_TRAIN_PRICE_BAND)


def test_first_event_and_equal_timestamps_encode_zero() -> None:
    gaps = encode_time_gaps([utc(0), utc(0), utc(0, 1)])

    assert gaps[0] == 0.0
    assert gaps[1] == 0.0
    assert gaps[2] == pytest.approx(math.log1p(1.0))


def test_large_gaps_are_capped_and_identical_beyond_the_cap() -> None:
    start = utc(0)
    far = start + timedelta(seconds=TIME_GAP_CAP_SECONDS * 5)
    at_cap = start + timedelta(seconds=TIME_GAP_CAP_SECONDS)

    assert encode_time_gaps([start, far])[1] == pytest.approx(math.log1p(TIME_GAP_CAP_SECONDS))
    assert encode_time_gaps([start, far]) == encode_time_gaps([start, at_cap])


def test_out_of_order_history_is_rejected_not_repaired() -> None:
    with pytest.raises(ValueError, match="already be ordered"):
        encode_time_gaps([utc(5), utc(0)])


def test_gap_encoding_is_deterministic_and_type_checked() -> None:
    history = [utc(0), utc(1), utc(30)]

    assert encode_time_gaps(history) == encode_time_gaps(history)
    assert encode_time_gaps([]) == ()
    with pytest.raises(TypeError):
        encode_time_gaps(["2019-10-23T10:00:00+00:00"])


def test_mixing_naive_and_aware_timestamps_is_rejected() -> None:
    with pytest.raises(ValueError, match="timezone"):
        naive = datetime(2019, 10, 23, 10, 0)  # noqa: DTZ001 - the point of the test
        encode_time_gaps([naive, utc(1)])
