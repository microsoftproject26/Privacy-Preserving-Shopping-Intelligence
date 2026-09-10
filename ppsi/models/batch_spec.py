"""The Phase1BatchSpec this project trains on, as opposed to the fixture one.

`ppsi.training.fixtures.default_batch_spec` describes a toy world - 64 items, 16
categories - built to exercise the contract. `S1-PR-07` uses it because its smoke
carries no history at all. Real training needs the real vocabularies, and once a
lane has exported or benchmarked against them they cannot move: renaming a channel
forces `S2-SE-01` to re-export and resizing one invalidates `S2-SE-02`'s latency
numbers.

So this file is a freeze point. Every constant below is either read from a frozen
S1 artifact or chosen here once.

Padding ids are not all 0, and that is deliberate. The canonical validator forbids a
pad id from appearing in a valid position, so a channel whose real values start at 0
cannot use 0 as padding. Categories are dense codes `0..587` from `vocabulary_v1`,
and price bands are `0..4` where band 0 already means "no usable TRAIN price" - both
therefore pad above their range instead of at zero.
"""

from __future__ import annotations

from ppsi.training.batch import CategoricalChannelSpec, Phase1BatchSpec

# --- categories -------------------------------------------------------------------
# 0..587 are the frozen vocabulary_v1 codes. A category absent from TRAIN is a real
# input the model must cope with, so it is mapped to OOV rather than dropped; it is
# only as a *target* that it disqualifies a decision, and that happened upstream.
CATEGORY_COUNT = 588
CATEGORY_OOV = 588
CATEGORY_PAD = 589
CATEGORY_VOCAB = 590

# --- products ---------------------------------------------------------------------
# Hashed rather than enumerated: the frozen catalogue holds 113,765 of the dataset's
# 166,794 items, so a dense vocabulary would have no row for a third of them. A hash
# has a row for everything. Bucket 0 is reserved for padding, so real ids land in
# 1..PRODUCT_BUCKETS-1.
PRODUCT_BUCKETS = 50_000

# --- brands -----------------------------------------------------------------------
# 3,445 distinct brands plus __UNK__. Hashed for the same reason and with the same
# reserved bucket 0, sized to keep collisions rare at this cardinality.
BRAND_BUCKETS = 5_000

# --- price bands ------------------------------------------------------------------
# 0..4 exactly as frozen in price_transform_v1, where 0 means "no usable TRAIN price"
# and 1..4 are the TRAIN quartiles at 29.58 / 77.20 / 194.17. Never refitted here.
PRICE_BAND_PAD = 5
PRICE_BAND_VOCAB = 6

# --- event types ------------------------------------------------------------------
# REES46 has exactly three: view, cart, purchase. There is no remove_from_cart.
# Shifted by one so that 0 can be padding without colliding with `view`.
EVENT_PAD = 0
EVENT_VIEW = 1
EVENT_CART = 2
EVENT_PURCHASE = 3
EVENT_VOCAB = 4
EVENT_CODE = {"view": EVENT_VIEW, "cart": EVENT_CART, "purchase": EVENT_PURCHASE}

# --- continuous -------------------------------------------------------------------
# One channel: log1p of the seconds since the previous event in the same session,
# clipped at one day. Two events a second apart and two an hour apart are different
# behaviour, and nothing else in the batch carries that.
HISTORY_CONTINUOUS_DIM = 1

HISTORY_CHANNELS = ("category_id", "product_bucket", "event_type_id", "brand_bucket", "price_band")
QUERY_CHANNELS = (
    "query_category_id",
    "query_product_bucket",
    "query_brand_bucket",
    "query_price_band",
)
CANDIDATE_CHANNELS = ("candidate_category_id", "candidate_price_band")


def phase1_batch_spec_v1() -> Phase1BatchSpec:
    """The frozen batch contract for real Phase 1 training.

    Every lane builds against this: `S2-DS-*` trains on it, `S2-PR-*` batches clients
    into it, `S2-SE-*` exports a model that consumes it.
    """
    return Phase1BatchSpec(
        history_categorical=(
            CategoricalChannelSpec("category_id", pad_id=CATEGORY_PAD, vocab_size=CATEGORY_VOCAB),
            CategoricalChannelSpec("product_bucket", pad_id=0, vocab_size=PRODUCT_BUCKETS),
            CategoricalChannelSpec("event_type_id", pad_id=EVENT_PAD, vocab_size=EVENT_VOCAB),
            CategoricalChannelSpec("brand_bucket", pad_id=0, vocab_size=BRAND_BUCKETS),
            CategoricalChannelSpec(
                "price_band", pad_id=PRICE_BAND_PAD, vocab_size=PRICE_BAND_VOCAB
            ),
        ),
        # The query is the item the decision is about. T1 does not need it, but T2 asks
        # "will *this* product be purchased" and T3 ranks candidates against it, so the
        # channel exists from the start rather than being added once the spec is frozen.
        query_categorical=(
            CategoricalChannelSpec(
                "query_category_id", pad_id=CATEGORY_PAD, vocab_size=CATEGORY_VOCAB
            ),
            CategoricalChannelSpec("query_product_bucket", pad_id=0, vocab_size=PRODUCT_BUCKETS),
            CategoricalChannelSpec("query_brand_bucket", pad_id=0, vocab_size=BRAND_BUCKETS),
            CategoricalChannelSpec(
                "query_price_band", pad_id=PRICE_BAND_PAD, vocab_size=PRICE_BAND_VOCAB
            ),
        ),
        candidate_categorical=(
            CategoricalChannelSpec(
                "candidate_category_id", pad_id=CATEGORY_PAD, vocab_size=CATEGORY_VOCAB
            ),
            CategoricalChannelSpec(
                "candidate_price_band", pad_id=PRICE_BAND_PAD, vocab_size=PRICE_BAND_VOCAB
            ),
        ),
        history_continuous_dim=HISTORY_CONTINUOUS_DIM,
        query_continuous_dim=0,
        candidate_continuous_dim=0,
        candidate_id_pad_id=0,
        candidate_id_vocab_size=PRODUCT_BUCKETS,
    )
