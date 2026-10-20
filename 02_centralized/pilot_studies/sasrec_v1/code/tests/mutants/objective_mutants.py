"""Deliberately WRONG loss-aggregation formulas -- negative controls proving
the uneven-microbatch-accumulation test actually detects a broken
normalisation (03 sec.6, 05 item 19/20, Fable #6)."""
from __future__ import annotations

from typing import Sequence


def mean_of_microbatch_means(loss_sums: Sequence[float], counts: Sequence[float]) -> float:
    """WRONG: averages per-microbatch means instead of dividing the pooled
    sum by the pooled count. Differs from sum/count whenever the
    contributing counts are uneven across microbatches -- exactly the bug
    the contract forbids ('Loss=sum/count, not averages of uneven microbatch
    means')."""
    means = [s / n for s, n in zip(loss_sums, counts) if n > 0]
    if not means:
        raise ValueError("no contributing microbatches")
    return sum(means) / len(means)
