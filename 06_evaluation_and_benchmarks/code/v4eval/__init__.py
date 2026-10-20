"""v4eval — the PPSI V4 shared evaluator (implements contracts/EVALUATION_CONTRACT_V4.md R2; task EVAL-02).

One evaluator for every method (C, FA, FP, PF, CPF, LO): exact full-K target ranks with the quoted tie rule, streamed
over the item table with q computed once; E2E (OOV = 0) and RANKABLE metrics with coverage over identical target sets;
micro and client-macro MRR / HR (= Recall) / NDCG at k in {1, 5, 10, 20} in FP64; strata from manifest fields only;
UNDEFINED (never NaN or 0) for empty populations and zero denominators; the paired user bootstrap (1000 resamples,
seed 20260926, one shared W, ratios bootstrapped directly); retained / loss quality; result rows with the full
comparison key; alignment by decision_id and ordered manifest hash; and the ALL_METHODS_LOCK gate for TEST manifests.
It imports only numpy and torch (no snapshot module).
"""
from __future__ import annotations

from pathlib import Path

from .hashing import code_sha256

EVALUATOR_ID = "v4eval-1.0.0"
_HERE = Path(__file__).resolve().parent


def evaluator_code_sha256() -> str:
    """sha256 over this package's .py files (name + bytes, sorted) — the evaluator identity in every result row."""
    return code_sha256(sorted(_HERE.glob("*.py")))


__all__ = ["EVALUATOR_ID", "evaluator_code_sha256"]
