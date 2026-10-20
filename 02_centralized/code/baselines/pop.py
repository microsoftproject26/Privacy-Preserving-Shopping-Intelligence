"""v4base.pop — POP baseline (Amendment 02 §4): global popularity of the fitting cohort's TRAIN target items.

score_j = count(target_class == j) over every TRAIN decision the BatchProvider serves (`v4train.provider.BatchProvider`
guarantees every row is loss-eligible: loss_mask and target_class in [0, K)). This is the same row for every eval
decision, so `PopScorer.score_batch` returns a zero-copy broadcast view of the [K] counts vector (`.expand(B, K)`) —
no [B, K] materialisation, unlike LAST / SKNN which genuinely differ per row.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .guard import RssGuard

DEFAULT_FIT_CHUNK = 8192


class PopScorer:
    def __init__(self, counts: torch.Tensor):
        self.counts = counts.to(torch.int64)
        self.K = int(self.counts.shape[0])

    def score_batch(self, batch: dict) -> torch.Tensor:
        did = batch["decision_id"]
        b = int(did.shape[0] if torch.is_tensor(did) else np.asarray(did).shape[0])
        return self.counts.unsqueeze(0).expand(b, self.K)

    def as_dict(self) -> dict:
        return {"K": self.K, "total_train_targets": int(self.counts.sum()), "max_count": int(self.counts.max()),
                "n_items_seen": int((self.counts > 0).sum())}


def fit(provider, K: int, *, chunk: int = DEFAULT_FIT_CHUNK, guard: Optional[RssGuard] = None) -> PopScorer:
    """Scan `provider` (a BatchProvider: real StoreTrainProvider or a synthetic stand-in) once, in canonical order,
    accumulating a [K] int64 histogram of target_class. Strictly TRAIN-only by construction (the provider IS the
    TRAIN split; VALIDATION is never opened here)."""
    n = int(provider.n_decisions)
    counts = torch.zeros(int(K), dtype=torch.int64)
    for start in range(0, n, chunk):
        pos = np.arange(start, min(n, start + chunk), dtype=np.int64)
        b = provider.batch(pos)
        tgt = b["target_class"]
        tgt = (tgt if torch.is_tensor(tgt) else torch.as_tensor(tgt)).to(torch.int64)
        if tgt.numel() and (int(tgt.min()) < 0 or int(tgt.max()) >= K):
            raise ValueError(f"POP fit: a TRAIN batch target_class is outside [0, {K}); every TRAIN row must be "
                             "loss-eligible (BatchProvider contract)")
        counts += torch.bincount(tgt, minlength=int(K))[:int(K)]
        if guard is not None:
            guard.check()
    return PopScorer(counts)


__all__ = ["PopScorer", "fit", "DEFAULT_FIT_CHUNK"]
