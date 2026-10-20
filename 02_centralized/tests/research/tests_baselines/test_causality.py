"""BASE-01: causality — no event at or after the decision is ever read by LAST or SKNN, even if one is physically
present in storage beyond the row's own `length` (an adversarial fixture: a real Clean V2 store never does this by
construction, but this baseline must not rely on that alone — `history_valid_mask`/`current_session_mask` must
themselves exclude it). POP reads only `target_class` (the decision's own label), which is never a "future event"
question, so its causality is structural and is not re-tested here.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from v4models.synthetic import synthetic_examples_v2

from basekit import K, L, WIDTHS, nc
from v4base import last as base_last
from v4base import sknn as base_sknn
from v4base.sessions import ITEM_TOKEN_OFFSET

# Classes reserved so real random draws never collide with the "poison" marker: item_tokens draws in [3, K - 3),
# i.e. classes in [0, K - 6), so POISON_CLASS = K - 1 is never a genuine event.
POISON_VOCAB = {"item_tokens": K - 3, "category_tokens": 40, "brand_tokens": 40, "main_category_tokens": 8,
                "daypart_tokens": 8, "weekday_tokens": 10}
POISON_CLASS = K - 1


def _poisoned_batch(seed: int = 5, n: int = 10) -> dict:
    """A TRAIN-shaped batch with every padded (>= length) column overwritten to look like a real, valid, current-
    session event of class POISON_CLASS — the adversarial case: if any code path forgot to mask by length, this
    poison would leak into LAST's recency scores or SKNN's session pool."""
    g = torch.Generator().manual_seed(seed)
    ex = synthetic_examples_v2(n, K, WIDTHS, g, max_len=L, window=L, with_session_offset=True, vocab=POISON_VOCAB)
    lengths = ex["lengths"]
    pos = torch.arange(L).unsqueeze(0)
    pad = pos >= lengths.unsqueeze(1)                                   # [n, L] True on padded columns
    ex["item_tokens"] = torch.where(pad, torch.full_like(ex["item_tokens"], POISON_CLASS + ITEM_TOKEN_OFFSET),
                                    ex["item_tokens"])
    ex["session_offset_tokens"] = torch.where(pad, torch.ones_like(ex["session_offset_tokens"]),
                                              ex["session_offset_tokens"])
    last = ex["item_tokens"].gather(1, (lengths - 1).clamp(min=0).view(-1, 1)).squeeze(1)
    ex["target_class"] = (last - ITEM_TOKEN_OFFSET).clamp(0, K - 1)
    ex["decision_id"] = torch.arange(2000, 2000 + 2 * n, 2, dtype=torch.int64)
    assert bool(pad.any()), "fixture must have at least one padded row to be a meaningful test"
    return ex


def test_last_ignores_padding_even_when_poisoned_to_look_like_the_current_event():
    ex = _poisoned_batch()
    scorer = base_last.LastItemScorer(K)
    out = scorer.score_batch(ex)
    lengths = ex["lengths"]
    for r in range(ex["item_tokens"].shape[0]):
        if int(L - lengths[r]) == 0:
            continue
        # POISON_CLASS can only be legitimately scored if it is ALSO the row's true last (real) item, which the
        # reserved vocab makes impossible (POISON_CLASS is never drawn as a real token).
        assert int(out[r, POISON_CLASS]) == base_last.SENTINEL, \
            f"row {r}: a padded column leaked into LAST's recency score for the poison class"


def test_sknn_pool_never_contains_a_padded_only_item():
    ex = _poisoned_batch(seed=7)

    class _Stub:
        n_decisions = int(ex["decision_id"].shape[0])

        def batch(self, positions):
            idx = torch.as_tensor(positions, dtype=torch.long)
            return {k: (v.index_select(0, idx) if torch.is_tensor(v) else v) for k, v in ex.items()}

    pool = base_sknn.fit(_Stub(), K, chunk=4)
    assert POISON_CLASS not in pool.item_index, "a padded-only column leaked into the SKNN past-session pool"
    for arr in pool.sessions:
        assert POISON_CLASS not in arr.tolist(), "a padded-only column leaked into a past session's item set"

    scorer = base_sknn.SessionKNNScorer(pool, k=5, sample_size=20)
    out = scorer.score_batch(ex)
    assert float(out[:, POISON_CLASS].abs().sum()) == 0.0, "SKNN scored the poison class from a padded column"


@nc("without the length mask, the poisoned padding (session_offset also poisoned to 1) would leak into LAST")
def test_nc_last_would_leak_padding_without_the_length_mask():
    ex = _poisoned_batch()
    scorer = base_last.LastItemScorer(K)
    out = scorer.score_batch(ex)
    leaked = any(int(out[r, POISON_CLASS]) != base_last.SENTINEL for r in range(ex["item_tokens"].shape[0])
                if int(L - ex["lengths"][r]) > 0)
    assert leaked


@nc("without the length mask, the poisoned padding would leak into the SKNN past-session pool")
def test_nc_sknn_pool_would_leak_padding_without_the_length_mask():
    ex = _poisoned_batch(seed=7)

    class _Stub:
        n_decisions = int(ex["decision_id"].shape[0])

        def batch(self, positions):
            idx = torch.as_tensor(positions, dtype=torch.long)
            return {k: (v.index_select(0, idx) if torch.is_tensor(v) else v) for k, v in ex.items()}

    pool = base_sknn.fit(_Stub(), K, chunk=4)
    assert POISON_CLASS in pool.item_index
