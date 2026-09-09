"""Feature representations shared by training and deployment."""

from ppsi.features.hashing import (
    HASH_ALGORITHM,
    PAD_BUCKET,
    ProductHashConfig,
    ProductHashEmbedding,
    hash_product_id,
    hash_product_ids,
)
from ppsi.features.price_time import (
    HIGHEST_PRICE_BAND,
    LOWEST_PRICE_BAND,
    NO_TRAIN_PRICE_BAND,
    TIME_GAP_CAP_SECONDS,
    encode_price_bands,
    encode_time_gaps,
    frozen_price_edges,
    price_band,
)

__all__ = [
    "HASH_ALGORITHM",
    "HIGHEST_PRICE_BAND",
    "LOWEST_PRICE_BAND",
    "NO_TRAIN_PRICE_BAND",
    "PAD_BUCKET",
    "TIME_GAP_CAP_SECONDS",
    "ProductHashConfig",
    "ProductHashEmbedding",
    "encode_price_bands",
    "encode_time_gaps",
    "frozen_price_edges",
    "hash_product_id",
    "hash_product_ids",
    "price_band",
]
