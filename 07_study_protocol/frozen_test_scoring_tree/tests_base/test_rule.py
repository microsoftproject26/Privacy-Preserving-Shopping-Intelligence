"""BASE-02: RULE — a fully hand-computed fixture for the MAIN-TRAIN category/brand map (including a mode tie,
smaller-token-wins), the popularity-within-category/brand shares, the final 0.5/0.5 blend, the "unknown
category/brand -> term 0" mask, and causality (padding never leaks into the permitted-history share even when
poisoned to look like a real, known category/brand view).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch

from basekit import nc
from v4base import pop as base_pop
from v4base import rule as base_rule

K = 4
# item2's category evidence deliberately ties (token4 vs token7, both count 1) -> smaller token (4) must win.
# item3 (token6) is never a WINDOW item anywhere (only ever a TARGET), so it is "unknown category/brand" by
# construction, exercising the "items with unknown category/brand: their term is 0" rule on a genuinely
# non-vacuous case (item3 still has real MAIN-TRAIN popularity, via its own target_class row). Each row is a
# DIFFERENT user (USER_KEYS below), so `scan_best_window_per_user`'s dedup is a no-op here (each user contributes
# their one and only window) -- the dedup mechanism ITSELF is exercised separately, below.
ROWS = [
    {"item_tokens": [3, 4, 0], "lengths": 2, "cats": [5, 5, 0], "brands": [10, 11, 0], "target": 1},
    {"item_tokens": [3, 3, 5], "lengths": 3, "cats": [6, 5, 7], "brands": [10, 12, 13], "target": 2},
    {"item_tokens": [5, 0, 0], "lengths": 1, "cats": [4, 0, 0], "brands": [13, 0, 0], "target": 0},
    {"item_tokens": [3, 0, 0], "lengths": 1, "cats": [5, 0, 0], "brands": [10, 0, 0], "target": 3},
]
USER_KEYS = [0, 1, 2, 3]


class _Provider:
    n_decisions = len(ROWS)

    def batch(self, positions):
        idx = [int(i) for i in positions]
        return {"item_tokens": torch.tensor([ROWS[i]["item_tokens"] for i in idx], dtype=torch.int64),
               "lengths": torch.tensor([ROWS[i]["lengths"] for i in idx], dtype=torch.int64),
               "decision_id": torch.tensor(idx, dtype=torch.int64),
               "main_category_tokens": torch.tensor([ROWS[i]["cats"] for i in idx], dtype=torch.int64),
               "brand_tokens": torch.tensor([ROWS[i]["brands"] for i in idx], dtype=torch.int64),
               "target_class": torch.tensor([ROWS[i]["target"] for i in idx], dtype=torch.int64)}


def _build_tables():
    provider = _Provider()
    pop_scorer = base_pop.fit(provider, K)
    cat_of, brand_of = base_rule.fit_category_brand_map(provider, K, user_keys=USER_KEYS)
    return base_rule.RuleTables(pop_scorer.counts, cat_of, brand_of)


def test_category_brand_map_with_a_mode_tie_broken_by_the_smaller_token():
    cat_of, brand_of = base_rule.fit_category_brand_map(_Provider(), K, user_keys=USER_KEYS)
    assert cat_of.tolist() == [5, 5, 4, base_rule.UNKNOWN_TOKEN]
    assert brand_of.tolist() == [10, 11, 13, base_rule.UNKNOWN_TOKEN]


def test_fit_category_brand_map_dedupes_overlapping_windows_per_user():
    # SAME user (user_key=0) has two TRAIN decisions of EQUAL length (a tie -> the larger decision_id wins, i.e.
    # row B, the later/more-complete window). Without the dedup, item0 (token3)'s category votes would combine
    # BOTH rows -- {99: 2, 8: 1} -- and wrongly mode to 99. Deduped, only row B is ever read, giving 8.
    rows = [
        {"item_tokens": [3, 3], "lengths": 2, "cats": [99, 99], "brands": [199, 199], "did": 100},   # row A (older)
        {"item_tokens": [3, 4], "lengths": 2, "cats": [8, 7], "brands": [88, 77], "did": 200},        # row B (newer)
    ]

    class _DedupProvider:
        n_decisions = len(rows)

        def batch(self, positions):
            idx = [int(i) for i in positions]
            return {"item_tokens": torch.tensor([rows[i]["item_tokens"] for i in idx], dtype=torch.int64),
                   "lengths": torch.tensor([rows[i]["lengths"] for i in idx], dtype=torch.int64),
                   "decision_id": torch.tensor([rows[i]["did"] for i in idx], dtype=torch.int64),
                   "main_category_tokens": torch.tensor([rows[i]["cats"] for i in idx], dtype=torch.int64),
                   "brand_tokens": torch.tensor([rows[i]["brands"] for i in idx], dtype=torch.int64)}

    cat_of, brand_of = base_rule.fit_category_brand_map(_DedupProvider(), 2, user_keys=[0, 0])
    assert cat_of.tolist() == [8, 7]
    assert brand_of.tolist() == [88, 77]


def test_pop_counts_and_within_category_brand_shares():
    tables = _build_tables()
    assert tables.pop_within_cat.tolist() == pytest.approx([0.5, 0.5, 1.0, 1.0], abs=1e-6)
    assert tables.pop_within_brand.tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0], abs=1e-6)
    assert tables.cat_known.tolist() == [True, True, True, False]
    assert tables.brand_known.tolist() == [True, True, True, False]


def _eval_batch():
    return {"item_tokens": torch.zeros((1, 3), dtype=torch.int64),   # irrelevant to RULE's own history reading
           "lengths": torch.tensor([3], dtype=torch.int64),
           "main_category_tokens": torch.tensor([[5, 4, 1]], dtype=torch.int64),
           "brand_tokens": torch.tensor([[10, 13, 1]], dtype=torch.int64)}


def test_hand_computed_final_score_including_the_unknown_item_masked_to_zero():
    tables = _build_tables()
    scorer = base_rule.RuleScorer(tables)
    out = scorer.score_batch(_eval_batch())[0]
    expected = [0.25, 1 / 12, 1 / 3, 0.0]
    assert out.tolist() == pytest.approx(expected, abs=1e-6)


def test_empty_permitted_history_scores_all_zero():
    tables = _build_tables()
    scorer = base_rule.RuleScorer(tables)
    batch = {"item_tokens": torch.zeros((1, 3), dtype=torch.int64), "lengths": torch.tensor([0], dtype=torch.int64),
            "main_category_tokens": torch.zeros((1, 3), dtype=torch.int64),
            "brand_tokens": torch.zeros((1, 3), dtype=torch.int64)}
    out = scorer.score_batch(batch)[0]
    assert out.tolist() == [0.0, 0.0, 0.0, 0.0]


def test_padding_never_leaks_into_the_permitted_history_share_even_when_poisoned():
    tables = _build_tables()
    scorer = base_rule.RuleScorer(tables)
    # length=1 (only one real view, token5=known category matching item0/item1); the padded column is poisoned to
    # look like three MORE views of a different (also known) category token7.
    batch = {"item_tokens": torch.zeros((1, 4), dtype=torch.int64), "lengths": torch.tensor([1], dtype=torch.int64),
            "main_category_tokens": torch.tensor([[5, 7, 7, 7]], dtype=torch.int64),
            "brand_tokens": torch.tensor([[10, 7, 7, 7]], dtype=torch.int64)}
    out = scorer.score_batch(batch)[0]
    # if padding leaked, pu_cat[5] would be 1/4 instead of 1/1 -- term_cat[0] would be 0.5*0.25=0.125, not 0.5
    assert out[0].item() == pytest.approx(0.5 * 1.0 * 0.5 + 0.5 * 1.0 * 1.0, abs=1e-6)


@nc("pop_within_cat is genuinely nonzero for the unknown-category item here -- the known>=3 mask in score_batch is "
   "doing real work, not guarding an already-zero (vacuous) case")
def test_nc_unknown_category_term_would_leak_without_the_mask():
    tables = _build_tables()
    n_valid = 3
    pu_cat = torch.bincount(torch.tensor([5, 4, 1]), minlength=tables.cat_vocab).to(torch.float32) / n_valid
    unmasked_term_item3 = float(pu_cat[int(tables.cat_of[3])] * tables.pop_within_cat[3])
    assert unmasked_term_item3 == 0.0
