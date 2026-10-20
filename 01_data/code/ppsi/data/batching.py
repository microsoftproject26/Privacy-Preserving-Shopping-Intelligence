"""Turning numpy history windows into the Phase1Batch the trainer contract expects.

`ppsi.data.sequences` produces the windows as compact numpy arrays - int32 and int8, so
three million decisions fit. This module converts a slice of them into tensors of the
exact dtypes the contract demands, and hands the result to the *canonical* validator
rather than the runtime one.

That choice is deliberate. `LocalTrainerCore` applies `validate_phase1_batch`, which
checks semantics. `validate_canonical_phase1_batch` additionally checks storage: that a
pad id never appears in a valid position, that padded slots hold exactly the pad id, that
masked continuous features are exactly zero, and that absent targets carry the canonical
filler. A producer that satisfies only the runtime rules can still hand the federated
lane batches whose masked storage is garbage, and the failure would surface as a
mysterious disagreement between regimes rather than as an error here.
"""

from __future__ import annotations

import numpy as np
import torch

from ppsi.models.batch_spec import PRICE_BAND_PAD
from ppsi.training.batch import (
    Phase1Batch,
    Phase1BatchSpec,
    validate_canonical_phase1_batch,
)


def windows_to_batch(windows, rows, spec: Phase1BatchSpec, *, validate: bool = True) -> Phase1Batch:
    """Build one Phase1Batch from the given rows of a `Windows` object.

    Only T1 is present. T2 and T3 are absent, so their targets carry the spec's canonical
    fillers and the candidate axis is one physical column of pure padding - the contract
    requires K >= 1 even when there is nothing to rank.
    """
    rows = np.asarray(rows)
    size = len(rows)
    length = windows.history_length

    lengths = torch.from_numpy(windows.lengths[rows].astype("int64"))
    # Derived, never built independently: the validator checks this exact relation, and
    # a mask constructed separately is a second source of truth waiting to drift.
    history_mask = torch.arange(length).unsqueeze(0) < lengths.unsqueeze(1)

    history_categorical_ids = {
        "category_id": torch.from_numpy(windows.category[rows].astype("int64")),
        "product_bucket": torch.from_numpy(windows.product[rows].astype("int64")),
        "event_type_id": torch.from_numpy(windows.event[rows].astype("int64")),
        "brand_bucket": torch.from_numpy(windows.brand[rows].astype("int64")),
        "price_band": torch.from_numpy(windows.price_band[rows].astype("int64")),
    }
    history_continuous = torch.from_numpy(windows.gap[rows].astype("float32")).unsqueeze(-1)

    query_categorical_ids = {
        "query_category_id": torch.from_numpy(windows.query_category[rows].astype("int64")),
        "query_product_bucket": torch.from_numpy(windows.query_product[rows].astype("int64")),
        "query_brand_bucket": torch.from_numpy(windows.query_brand[rows].astype("int64")),
        "query_price_band": torch.from_numpy(windows.query_price_band[rows].astype("int64")),
    }
    query_continuous = torch.zeros(size, spec.query_continuous_dim, dtype=torch.float32)

    # A pad id is forbidden anywhere in a query channel. Bucket 0 is the product and
    # brand pad, and it can legitimately arise only if a raw value were missing; the
    # frozen examples guarantee the decision event exists, so this is a guard, not a fix.
    for name, values in query_categorical_ids.items():
        channel = next(c for c in spec.query_categorical if c.name == name)
        if bool((values == channel.pad_id).any()):
            raise ValueError(
                f"{name} contains the pad id {channel.pad_id} in a query position; "
                "the decision event must always supply a real value"
            )

    candidate_ids = torch.full((size, 1), spec.candidate_id_pad_id, dtype=torch.int64)
    candidate_categorical_ids = {
        "candidate_category_id": torch.full(
            (size, 1), _pad_of(spec, "candidate_category_id"), dtype=torch.int64
        ),
        "candidate_price_band": torch.full((size, 1), PRICE_BAND_PAD, dtype=torch.int64),
    }
    candidate_continuous = torch.zeros(size, 1, spec.candidate_continuous_dim, dtype=torch.float32)
    candidate_mask = torch.zeros(size, 1, dtype=torch.bool)

    batch = Phase1Batch(
        history_categorical_ids=history_categorical_ids,
        history_continuous_features=history_continuous,
        lengths=lengths,
        history_mask=history_mask,
        query_categorical_ids=query_categorical_ids,
        query_continuous_features=query_continuous,
        candidate_ids=candidate_ids,
        candidate_categorical_ids=candidate_categorical_ids,
        candidate_continuous_features=candidate_continuous,
        candidate_mask=candidate_mask,
        t1_target=torch.from_numpy(windows.target[rows].astype("int64")),
        t2_target=torch.zeros(size, 1, dtype=torch.float32),
        t3_gains=torch.zeros(size, 1, dtype=torch.float32),
        t1_present=torch.ones(size, dtype=torch.bool),
        t2_present=torch.zeros(size, dtype=torch.bool),
        t3_present=torch.zeros(size, dtype=torch.bool),
    )
    if validate:
        validate_canonical_phase1_batch(batch, spec)
    return batch


def _pad_of(spec: Phase1BatchSpec, name: str) -> int:
    for channel in spec.candidate_categorical:
        if channel.name == name:
            return channel.pad_id
    raise KeyError(name)


def iterate_batches(
    windows,
    spec: Phase1BatchSpec,
    *,
    batch_size: int,
    shuffle: bool,
    generator: np.random.Generator | None = None,
    validate: bool = False,
):
    """Yield Phase1Batch objects over every row of `windows`.

    Validation defaults off here and on in `windows_to_batch`: the canonical validator
    walks every tensor, which is worth paying once per fixture and not 6,081 times per
    epoch. The training notebook validates the first batch of every run and then trusts
    the loop that produced it.
    """
    order = np.arange(len(windows))
    if shuffle:
        (generator if generator is not None else np.random.default_rng()).shuffle(order)
    for start in range(0, len(order), batch_size):
        yield windows_to_batch(windows, order[start : start + batch_size], spec, validate=validate)
