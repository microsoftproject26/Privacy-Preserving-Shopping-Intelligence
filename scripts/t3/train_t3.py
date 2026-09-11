"""The T3 candidate-ranking head on the encoder S2-DS-01 trained.

**The task:** given a query item, order its candidate list so the products the user actually
engages with next come first.

T3 is a ranking problem, not a classification one, and three of its properties make it
unlike either head before it.

**1. The metric only counts a fraction of the rows.** Gains follow the frozen `INTENT`
rule - purchase 2, cart 1, **view 0** - and 98.1% of the 1,096,774 VALIDATION rows are
views. NDCG gives them no credit at all. So 21,284 scoring positives across **18,814
evaluable queries** are the entire measurement, and scoring over all rows would compute a
different quantity from the one `S1-D1-DS-07` published.

**2. The ceiling is 0.8201, not 1.0.** It was measured: only **81.59%** of scoring positives
appear in their own candidate list. If the right product was never retrieved, no reranking
can find it. Every T3 number has to be read against that ceiling.

**3. One window serves many rows.** The examples carry one row per (query, positive), so
8,088,359 TRAIN rows describe 1,740,433 distinct decisions - about 4.6 each. The session is
encoded once per decision and the candidates are scored against that single vector.

Nothing about the ranking loss is frozen upstream: `ContractSmokeObjective` uses squared
error and says outright that it is not the final choice. So the loss is chosen by
measurement, and the losing options stay in the report.

    the number to beat   0.2046   end-to-end NDCG@5
    the ceiling          0.8201
    noise floor          0.0048
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

PROJECT = Path(__file__).resolve().parent.parent
for _candidate in Path(__file__).resolve().parents:
    if (_candidate / "ppsi" / "__init__.py").exists():
        sys.path.insert(0, str(_candidate))
        break
else:
    sys.path.insert(0, str(PROJECT / "Repo_S2DS01"))

from ppsi.models.batch_spec import (  # noqa: E402
    CATEGORY_OOV,
    CATEGORY_PAD,
    PRICE_BAND_PAD,
    phase1_batch_spec_v1,
)
from ppsi.models.checkpoint import load_encoder as shared_load_encoder
from ppsi.models.session_gru import SessionGRU, SessionGRUConfig, build_model  # noqa: E402
from ppsi.training.batch import Phase1Batch  # noqa: E402

TASK = Path(__file__).resolve().parent
OUTPUT = TASK / "output"
OUTPUT.mkdir(exist_ok=True)

SPEC = phase1_batch_spec_v1()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The headline is MACRO - `ADR-001` fixes that for this project, and T1 already follows it.
# T3 was first reported in micro, which is a different and easier question: micro lets the
# clients with many queries dominate, and here that moves the frozen retrieval order from
# 0.2708 to 0.2047. Both are printed; only the macro one is the result.
#
# Every figure below is measured with the full-oracle NDCG denominator (see `ndcg_at_k`),
# so none of them match the pre-review numbers exactly.
BEST_SIMPLE_MACRO = 0.2707   # frozen retrieval order, macro over 4,591 clients
BEST_SIMPLE_MICRO = 0.2046   # the same ordering, micro over 18,814 queries
CEILING_MACRO = 0.8795       # a perfect reranking of what retrieval returned, macro
CEILING_MICRO = 0.8201       # the published ceiling - correct, but micro

# With the denominator fixed, micro reproduces the upstream `0.2046` **exactly**. Upstream
# had always computed it correctly; the drift to 0.2056 was ours alone. The old gate
# accepted anything within 0.005 of it, which is five times the size of the error, so it
# could not have caught this. TOLERANCE is now sized to the agreement we actually have.
TOLERANCE = 0.001

BEST_SIMPLE = BEST_SIMPLE_MICRO   # retained so the calibration gate keeps its meaning
CEILING = CEILING_MICRO
HALF_WIDTH = 0.0048
MAX_CANDIDATES = 100

ARRAYS = ("category", "product", "event", "brand", "price_band", "gap", "lengths",
          "query_category", "query_product", "query_brand", "query_price_band", "client")


def resolve(*names: str) -> Path:
    """Find an input wherever this is running.

    Two layouts, and neither is worth forking the code for: the task folder on the machine
    that produced the results, and a flat `data/` tree on a studio. `PPSI_DATA_ROOT` names
    the second explicitly; otherwise a short list of candidates is searched.
    """
    roots = []
    if os.environ.get("PPSI_DATA_ROOT"):
        root = Path(os.environ["PPSI_DATA_ROOT"])
        roots += [root, root / "protocol", root / "encoder", root / "cache_t3", root.parent]
    roots += [PROJECT / "S2-DS-01_GRU_T1_Model",
              PROJECT / "S2-DS-01_GRU_T1_Model" / "output",
              PROJECT / "data", PROJECT / "data" / "protocol",
              PROJECT / "data" / "encoder", Path("data"),
              PROJECT / "S1-D1-DS-07_T3_Protocol_and_Task_Examples" / "output"]
    for root in roots:
        for name in names:
            if (root / name).exists():
                return root / name
    looked = "; ".join(str(r) for r in roots)
    raise SystemExit(
        f"cannot find any of {names}. Looked in: {looked}. "
        "Set PPSI_DATA_ROOT to the directory holding protocol/ and cache_t3/."
    )


@dataclass(slots=True)
class Ranking:
    """One split of T3: windows per decision, and the graded rows that point at them.

    `row_to_window` is what keeps this affordable. Without it the same session history
    would be materialised once per positive - 8.1M copies instead of 1.74M.
    """

    name: str
    data: dict
    row_to_window: np.ndarray
    positive_item: np.ndarray
    query_item: np.ndarray
    gain: np.ndarray
    query_id: np.ndarray

    @classmethod
    def load(cls, split: str, examples: pd.DataFrame) -> "Ranking":
        cache = resolve("cache_t3")
        prefix = f"t3_{split}"
        data = {key: np.load(cache / f"{prefix}_{key}.npy") for key in ARRAYS}
        row_to_window = np.load(cache / f"{prefix}_row_to_window.npy")
        assert len(row_to_window) == len(examples), (
            f"{len(row_to_window):,} cached rows against {len(examples):,} example rows; "
            "the cache and the frozen examples are not describing the same thing")
        return cls(split, data, row_to_window,
                   examples["positive_item"].to_numpy().astype("int64"),
                   examples["query_item"].to_numpy().astype("int64"),
                   examples["label_value"].to_numpy().astype("float32"),
                   examples["query_id"].to_numpy().astype("int64"))

    def __len__(self) -> int:
        return len(self.row_to_window)


def candidate_table(protocol: dict) -> tuple:
    """The frozen candidate list for the selected retrieval, as dense arrays.

    The file holds **both** strategies side by side - 4,554,279 rows each - and the
    protocol selected `cooccurrence_then_popularity`. Which one to use is read from the
    protocol, never written here: restating a frozen rule in a second place is the defect
    that has cost this project three wrong numbers, and it fired `S1-DS-09`'s fence once.
    """
    selected = protocol["retrieval"]["selected"]
    frame = pd.read_parquet(resolve("t3_candidate_lists_v1.proposed.parquet"))
    frame = frame[frame["retrieval"] == selected]
    frame = frame.sort_values(["query_item", "rank"])

    anchors = frame["query_item"].to_numpy()
    unique, start = np.unique(anchors, return_index=True)
    counts = np.diff(np.append(start, len(anchors)))
    width = int(counts.max())

    items = np.zeros((len(unique), width), dtype="int64")
    mask = np.zeros((len(unique), width), dtype=bool)
    flat = frame["item"].to_numpy().astype("int64")
    for row, (begin, size) in enumerate(zip(start, counts, strict=True)):
        items[row, :size] = flat[begin:begin + size]
        mask[row, :size] = True

    index = {int(anchor): row for row, anchor in enumerate(unique)}
    print(f"  candidates: {selected}, {len(frame):,} rows over {len(unique):,} anchors, "
          f"width {width}")
    return items, mask, index


class Candidates:
    """Lookup from a query item to its frozen candidate row."""

    def __init__(self, items: np.ndarray, mask: np.ndarray, index: dict) -> None:
        self.items, self.mask, self.index = items, mask, index

    def rows_for(self, query_items: np.ndarray) -> np.ndarray:
        return np.array([self.index.get(int(q), -1) for q in query_items], dtype="int64")


def build_batch(split: Ranking, decisions: np.ndarray, gains: np.ndarray,
                anchor_of: np.ndarray, tables) -> Phase1Batch:
    """One Phase1Batch, built entirely by indexing precomputed arrays.

    The first version assembled candidate metadata with Python loops here and reached 11%
    GPU utilisation at 101% CPU - the card idle, waiting on batch construction. Everything
    that does not depend on the model now lives in `prepared.py` and this is a gather.
    """
    windows = split.data
    take = lambda key, dtype: torch.from_numpy(  # noqa: E731
        np.ascontiguousarray(windows[key][decisions]).astype(dtype))
    lengths = take("lengths", "int64")
    size, width = len(decisions), windows["category"].shape[1]

    anchor_rows = anchor_of[decisions]
    valid = anchor_rows >= 0
    safe = np.where(valid, anchor_rows, 0)

    candidate_mask = np.where(valid[:, None], tables.mask[safe], False)
    buckets = np.where(candidate_mask, tables.bucket[safe], 0)
    category = np.where(candidate_mask, tables.category[safe], CATEGORY_PAD)
    band = np.where(candidate_mask, tables.price_band[safe], PRICE_BAND_PAD)
    rank = np.where(candidate_mask, tables.rank[None, :], 0.0).astype("float32")

    graded = gains.copy()
    graded[~candidate_mask] = 0.0

    return Phase1Batch(
        history_categorical_ids={
            "category_id": take("category", "int64"),
            "product_bucket": take("product", "int64"),
            "event_type_id": take("event", "int64"),
            "brand_bucket": take("brand", "int64"),
            "price_band": take("price_band", "int64"),
        },
        history_continuous_features=take("gap", "float32").unsqueeze(-1),
        lengths=lengths,
        history_mask=torch.arange(width).unsqueeze(0) < lengths.unsqueeze(1),
        query_categorical_ids={
            "query_category_id": take("query_category", "int64"),
            "query_product_bucket": take("query_product", "int64"),
            "query_brand_bucket": take("query_brand", "int64"),
            "query_price_band": take("query_price_band", "int64"),
        },
        query_continuous_features=torch.zeros(size, 0, dtype=torch.float32),
        candidate_ids=torch.from_numpy(buckets),
        candidate_categorical_ids={
            "candidate_category_id": torch.from_numpy(category),
            "candidate_price_band": torch.from_numpy(band),
        },
        candidate_continuous_features=torch.from_numpy(rank).unsqueeze(-1),
        candidate_mask=torch.from_numpy(candidate_mask),
        t1_target=torch.zeros(size, dtype=torch.int64),
        t2_target=torch.zeros(size, 1, dtype=torch.float32),
        t3_gains=torch.from_numpy(graded),
        t1_present=torch.zeros(size, dtype=torch.bool),
        t2_present=torch.zeros(size, dtype=torch.bool),
        t3_present=torch.from_numpy(candidate_mask.any(axis=1)),
    )


def ndcg_at_k(scores: np.ndarray, gains: np.ndarray, mask: np.ndarray,
              ideal: np.ndarray, k: int = 5) -> np.ndarray:
    """Graded NDCG@k, end to end - a positive that was never retrieved scores zero.

    "End to end" is the whole point. The alternative, scoring only the positives that made
    it into the list, measures the reranker in isolation and quietly hides the 18.4% of
    scoring positives that retrieval missed. The protocol chose the honest one.

    **`ideal` is not optional, and deriving it from `gains` is the defect this signature
    exists to prevent.** `gains` holds one column per *retrieved* candidate, so an ideal
    taken from it describes the best ranking of what retrieval happened to return - not the
    best ranking of what the user actually engaged with. A query with two positives, one
    retrieved and one missed, then scores a perfect 1.0 for ranking the retrieved one first.

    Full misses were already handled: `idcg == 0` returns zero. **Partial** misses were not,
    and there are 485 of them. Measured on the frozen retrieval order, the flattery is
    `0.2056` against a true `0.2047`. Small, and in the direction that looks like success -
    which is the only reason it survived a review of the metric.

    `ideal` is `[B, k]`: the top-k gains the query truly had, retrieved or not, built once
    per split by `prepared.build_ideal_gains`.
    """
    ranked = np.where(mask, scores, -np.inf)
    order = np.argsort(-ranked, axis=1)[:, :k]
    top_gains = np.take_along_axis(gains, order, axis=1)
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = ((2**top_gains - 1) * discount).sum(axis=1)

    assert ideal.shape == (len(gains), k), (
        f"ideal is {ideal.shape}, expected {(len(gains), k)}; the denominator is not the "
        "full-oracle one and the metric is not end to end")
    idcg = ((2**ideal - 1) * discount).sum(axis=1)
    return np.where(idcg > 0, dcg / np.maximum(idcg, 1e-12), 0.0)


def listwise_loss(scores: torch.Tensor, gains: torch.Tensor,
                  mask: torch.Tensor) -> torch.Tensor:
    """Softmax cross-entropy over the candidate set, weighted by gain.

    A ranking loss rather than a pointwise one: it pushes a graded candidate above the
    others in its own list, which is what NDCG rewards, instead of pushing every score
    toward its gain in isolation.
    """
    scores = scores.masked_fill(~mask, -1e9)
    log_probability = torch.log_softmax(scores, dim=1)
    weight = gains * mask
    total = weight.sum(dim=1)
    per_row = -(weight * log_probability).sum(dim=1) / torch.clamp(total, min=1e-9)
    return per_row[total > 0].mean() if (total > 0).any() else scores.sum() * 0.0


def pointwise_loss(scores: torch.Tensor, gains: torch.Tensor,
                   mask: torch.Tensor) -> torch.Tensor:
    """Squared error against the gain, on valid candidates only."""
    difference = (scores - gains) ** 2
    return difference[mask].mean() if mask.any() else scores.sum() * 0.0


LOSSES = {"listwise": listwise_loss, "pointwise": pointwise_loss}


def load_encoder(model: SessionGRU, checkpoint: Path) -> dict:
    """Load the S2-DS-01 encoder, tolerating a reshaped T3 head.

    Adding the retrieval-rank channel widened `candidate_projection` from [128, 40] to
    [128, 41], so the S2-DS-01 checkpoint no longer matches it. Nothing is lost: the T3
    head was never trained in that task - it existed only because `RawModelOutput` requires
    all three tensors - so those weights were random anyway.

    The rule now lives in `ppsi.models.checkpoint`, shared with `S2-DS-06`, because keeping
    a private copy here is what let the two tasks disagree about the same checkpoint.
    """
    provenance = shared_load_encoder(model, checkpoint)
    print(f"    loaded {provenance['loaded']} tensors from {checkpoint.name}; "
          f"{len(provenance['dropped'])} T3-head tensors start fresh")
    return provenance
