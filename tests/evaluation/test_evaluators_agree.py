"""The two T1 evaluators must give the same answer, or the project has two sets of numbers.

`S2-DS-01` built `ppsi.models.evaluation` to score its own ladder, and `S2-PR-07` built
`ppsi.evaluation.t1` as the frozen harness every regime will be judged by. Both compute the
rank of the true category. Neither lane is wrong to have one - but if they drift apart, R1
and R2 end up reported on different metrics and the gap between them absorbs the difference.

This project has already paid for that mistake three times, each time inside a single
notebook. Across two lanes it would be harder to notice and more expensive to unwind.

The tie case is the one that matters. Any two implementations agree on distinct scores; they
diverge exactly where several categories share a score, which is common early in training
and universal for a model-free baseline. Both evaluators document the same rule - count the
strictly greater, then the equal-with-lower-code - and this asserts they implement it.
"""

from __future__ import annotations

import numpy as np
import pytest

from ppsi.evaluation.t1 import rank_targets_from_scores
from ppsi.models.evaluation import rank_of_truth

CATEGORY_COUNT = 588


def _targets(rows: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, CATEGORY_COUNT, rows)


@pytest.mark.parametrize(
    ("name", "maker"),
    [
        ("distinct scores", lambda rng, rows: rng.standard_normal((rows, CATEGORY_COUNT))),
        # Three distinct values across 588 classes: roughly 196 classes tie at every level.
        ("heavy ties", lambda rng, rows: rng.integers(0, 3, (rows, CATEGORY_COUNT)).astype(float)),
        # The degenerate case a no-signal baseline produces: everything ties with everything.
        ("all tied", lambda rng, rows: np.zeros((rows, CATEGORY_COUNT))),
    ],
)
def test_the_two_t1_evaluators_rank_identically(name: str, maker) -> None:
    rows = 512
    rng = np.random.default_rng(13)
    scores = np.asarray(maker(rng, rows), dtype="float64")
    targets = _targets(rows, seed=42)

    theirs = rank_targets_from_scores(scores, targets, category_count=CATEGORY_COUNT).numpy()
    ours = rank_of_truth(scores, targets, k=CATEGORY_COUNT)

    mismatched = int((theirs != ours).sum())
    assert mismatched == 0, (
        f"{name}: the two T1 evaluators disagree on {mismatched} of {rows} rows. "
        "Every published T1 number depends on which one was used, and the R1-vs-R2 gap "
        "would silently include the difference.")


def test_the_two_evaluators_agree_on_reciprocal_rank() -> None:
    """Equal ranks are necessary; equal MRR is what actually gets published."""
    rows = 512
    rng = np.random.default_rng(2026)
    scores = rng.standard_normal((rows, CATEGORY_COUNT))
    targets = _targets(rows, seed=13)

    theirs = rank_targets_from_scores(scores, targets, category_count=CATEGORY_COUNT).numpy()
    ours = rank_of_truth(scores, targets, k=CATEGORY_COUNT)

    # `rank_of_truth` returns 0 for a truth outside the top k; with k = C that cannot happen,
    # and asserting it here keeps the comparison honest rather than quietly dividing by zero.
    assert (ours > 0).all()
    assert np.isclose((1.0 / theirs).mean(), (1.0 / ours).mean(), rtol=0, atol=1e-12)
