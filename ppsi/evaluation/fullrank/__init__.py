"""Full-catalogue evaluator for next-item recommendation.

One evaluator for every training paradigm (central, federated, personalised, on-device fine-tuned, local-only):

* exact full-catalogue target ranks with a fixed tie rule, the query computed once and the item table streamed;
* end-to-end metrics (an out-of-vocabulary target scores 0) and rankable-only metrics, with coverage, over target
  sets that are identical for every method on the same manifest;
* micro and user-macro MRR / HR (= Recall) / NDCG at k in {1, 5, 10, 20}, accumulated in float64;
* strata from manifest fields only, and an explicit UNDEFINED value (never NaN or 0) for empty populations;
* the paired user bootstrap (one shared resample matrix, ratios bootstrapped directly) and retained quality;
* result rows with a full comparison key, and alignment of predictions by decision id and ordered manifest hash.

It depends only on numpy and torch.
"""
from __future__ import annotations

from pathlib import Path

from .hashing import code_sha256

EVALUATOR_ID = "fullrank-1.0.0"
_HERE = Path(__file__).resolve().parent


def evaluator_code_sha256() -> str:
    """sha256 over this package's .py files (name + bytes, sorted): the evaluator identity in every result row."""
    return code_sha256(sorted(_HERE.glob("*.py")))


__all__ = ["EVALUATOR_ID", "evaluator_code_sha256"]
