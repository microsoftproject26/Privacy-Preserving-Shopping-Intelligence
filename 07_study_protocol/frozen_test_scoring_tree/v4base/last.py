"""v4base.last — LAST baseline (Amendment 02 §4): last-item repeat, scoring the items in the user's causal history
by recency; every other class is unscored.

Deviation from the task packet's literal wording ("others -inf"): `v4eval.ranking._refuse_nonfinite` refuses any
non-finite (inf/NaN) FLOATING score (`NonFiniteScoreError`, contract G1) — a real -inf dense row would be refused by
the frozen evaluator itself, and v4eval is never edited to accept it. LAST therefore scores in int64 and uses a
finite sentinel, SENTINEL = -1, which is strictly below every real recency value (recency in [0, L)); v4eval's
non-finite check does not even apply to an integer tensor (`Tensor.is_floating_point()` is False), so this is not a
weakening of the tie rule, only a type choice consistent with the frozen `rank_from_scores` docstring ("precomputed
scores, float or int64 ... never coerced"). See company/handoffs/BASE-01.md.

Recency: column j of item_tokens is more recent as j increases (v4hist.windows / v4models.synthetic: real events at
columns [0, length-1], column length-1 immediately precedes the decision). A repeated item keeps its MOST RECENT
occurrence's recency (`scatter_reduce_(..., reduce="amax")`).
"""
from __future__ import annotations

import torch

from .sessions import history_valid_mask, item_classes

SENTINEL = -1


class LastItemScorer:
    """Stateless (no fit): every eval batch is scored purely from its own causal-history columns."""

    def __init__(self, K: int):
        self.K = int(K)

    def score_batch(self, batch: dict) -> torch.Tensor:
        item_tokens = batch["item_tokens"]
        item_tokens = item_tokens if torch.is_tensor(item_tokens) else torch.as_tensor(item_tokens)
        B, L = int(item_tokens.shape[0]), int(item_tokens.shape[1])
        cls, valid_tok = item_classes(item_tokens, self.K)
        valid = valid_tok & history_valid_mask(batch["lengths"], L)
        recency = torch.arange(L, dtype=torch.int64).unsqueeze(0).expand(B, L)
        src = torch.where(valid, recency, torch.full_like(recency, SENTINEL))
        idx = torch.where(valid, cls, torch.zeros_like(cls)).clamp(0, self.K - 1)
        out = torch.full((B, self.K), SENTINEL, dtype=torch.int64)
        out.scatter_reduce_(1, idx, src, reduce="amax", include_self=True)
        return out


__all__ = ["LastItemScorer", "SENTINEL"]
