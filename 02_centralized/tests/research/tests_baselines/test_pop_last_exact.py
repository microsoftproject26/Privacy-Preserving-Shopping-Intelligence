"""BASE-01: POP and LAST, exact values on a small, fully hand-built (non-random) fixture."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from basekit import nc
from v4base import last as base_last
from v4base import pop as base_pop

K = 6
L = 5
# row0: history classes [2, 0, 4] (length 3), target 4     tokens = class + 3
# row1: history classes [1, 1]    (length 2), target 1
# row2: history classes [3]       (length 1), target 3
ITEM_TOKENS = torch.tensor([[5, 3, 7, 0, 0],
                            [4, 4, 0, 0, 0],
                            [6, 0, 0, 0, 0]], dtype=torch.int64)
LENGTHS = torch.tensor([3, 2, 1], dtype=torch.int64)
TARGET_CLASS = torch.tensor([4, 1, 3], dtype=torch.int64)
DECISION_ID = torch.tensor([10, 20, 30], dtype=torch.int64)
BATCH = {"item_tokens": ITEM_TOKENS, "lengths": LENGTHS, "target_class": TARGET_CLASS, "decision_id": DECISION_ID}


class _OneShotProvider:
    """A BatchProvider serving exactly BATCH (one chunk)."""
    n_decisions = 3

    def batch(self, positions):
        idx = torch.as_tensor(positions, dtype=torch.long)
        return {k: v.index_select(0, idx) for k, v in BATCH.items()}


def test_pop_counts_are_the_exact_train_target_histogram():
    scorer = base_pop.fit(_OneShotProvider(), K)
    assert scorer.counts.tolist() == [0, 1, 0, 1, 1, 0]          # target_class = [4, 1, 3]
    row = scorer.score_batch(BATCH)
    assert row.shape == (3, K)
    for r in range(3):
        assert row[r].tolist() == [0, 1, 0, 1, 1, 0]              # the SAME row, broadcast, for every decision


def test_last_scores_are_exact_recency_by_column():
    scorer = base_last.LastItemScorer(K)
    out = scorer.score_batch(BATCH)
    S = base_last.SENTINEL
    # row0: col0->class2 (recency 0), col1->class0 (recency 1), col2->class4 (recency 2, most recent)
    assert out[0].tolist() == [1, S, 0, S, 2, S]
    # row1: col0/col1 both class1 (recency 0 and 1); the MOST RECENT occurrence (1) wins
    assert out[1].tolist() == [S, 1, S, S, S, S]
    # row2: col0->class3 (recency 0), nothing else valid
    assert out[2].tolist() == [S, S, S, 0, S, S]


@nc("without the range check, POP fit would silently accept an out-of-range TRAIN target_class")
def test_nc_pop_fit_would_accept_an_out_of_range_target_without_the_check():
    class _BadProvider:
        n_decisions = 1

        def batch(self, positions):
            return {"target_class": torch.tensor([K + 5], dtype=torch.int64)}

    accepted = False
    try:
        base_pop.fit(_BadProvider(), K)
        accepted = True
    except ValueError:
        pass
    assert accepted
