"""Tests for the evaluator (Task T-C): T9 (OOV-merge macro fixture) and T8
(tiny exhaustive reference), per the brief's "Tests" section.

Run with the LOCAL python from `.local_ai\\gru_v1`:
    <local python> -m pytest code\\test_evaluator.py -v
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from evaluator import _ranks_chunked, evaluate

# ===========================================================================
# T9 -- OOV-merge macro fixture (FABLE_PLAN.md sec. E)
#
# Three users, K=30 (comfortably >= the largest requested rank, 25):
#   A unseen (client_id=2): two decisions, both rank 1.
#   B unseen (client_id=2): one decision, rank 25.
#   C warm   (client_id=7): one decision, rank 2.
# Distinct eval_user_key per user (A=0, B=1, C=2).
# ===========================================================================

T9_K = 30


def _rank_fixture_row(K: int, target: int, rank: int) -> np.ndarray:
    """A score row (no ties) whose exact rank for `target` is `rank`:
    target gets the lowest score (0.0); exactly (rank-1) other columns get a
    strictly higher score (1.0); everything else is strictly lower (-1.0).
    rank = 1 + #{score > target} + #{tie, lower id} = 1 + (rank-1) + 0.
    """
    scores = np.full(K, -1.0, dtype=np.float64)
    scores[target] = 0.0
    others = [i for i in range(K) if i != target]
    for i in others[: rank - 1]:
        scores[i] = 1.0
    return scores


def _make_t9_rows():
    # Row order: A-d1(rank1), A-d2(rank1), B-d1(rank25), C-d1(rank2)
    scores = np.stack(
        [
            _rank_fixture_row(T9_K, target=5, rank=1),
            _rank_fixture_row(T9_K, target=7, rank=1),
            _rank_fixture_row(T9_K, target=3, rank=25),
            _rank_fixture_row(T9_K, target=9, rank=2),
        ]
    )
    targets = np.array([5, 7, 3, 9], dtype=np.int64)
    eval_mask = np.array([True, True, True, True])
    eval_user_key = np.array([0, 0, 1, 2], dtype=np.int64)  # A, A, B, C
    client_id = np.array([2, 2, 2, 7], dtype=np.int64)      # A, A, B, C
    rows = {
        "target_class": targets,
        "eval_mask": eval_mask,
        "eval_user_key": eval_user_key,
        "client_id": client_id,
    }
    return scores, rows


def _scores_fn_factory(scores: np.ndarray, batch_size: int):
    scores_t = torch.as_tensor(scores, dtype=torch.float32)

    def scores_fn():
        for start in range(0, scores_t.shape[0], batch_size):
            yield scores_t[start : start + batch_size]

    return scores_fn


def test_t9_macro_and_micro_mrr():
    scores, rows = _make_t9_rows()
    scores_fn = _scores_fn_factory(scores, batch_size=2)  # exercise multi-batch scoring
    result = evaluate(scores_fn, rows, k=20, chunk_size=7)

    macro_mrr = result["end_to_end"]["macro"]["mrr@20"]
    micro_mrr = result["end_to_end"]["micro"]["mrr@20"]
    user_count = result["end_to_end"]["macro"]["user_count"]

    assert macro_mrr == pytest.approx(0.5, abs=1e-9)
    assert micro_mrr == pytest.approx(0.625, abs=1e-9)
    assert user_count == 3
    assert result["user_count"] == 3


@pytest.mark.xfail(strict=True, reason="grouping macro by client_id merges unseen users")
def test_t9_broken_group_by_client_id():
    scores, rows = _make_t9_rows()
    broken_rows = dict(rows)
    broken_rows["eval_user_key"] = rows["client_id"]  # the bug: group by client_id
    scores_fn = _scores_fn_factory(scores, batch_size=4)
    result = evaluate(scores_fn, broken_rows, k=20, chunk_size=7)

    macro_mrr = result["end_to_end"]["macro"]["mrr@20"]
    user_count = result["end_to_end"]["macro"]["user_count"]

    # The bug actually produces 0.5833.../2 (asserted for documentation);
    # the real assertions below (correct values) are what must fail.
    assert macro_mrr == pytest.approx(0.5833333333333334, abs=1e-6)
    assert user_count == 2

    assert macro_mrr == pytest.approx(0.5, abs=1e-9)
    assert user_count == 3


# --- OOV extension: give C an OOV-target row -------------------------------

def _make_t9_oov_extension_rows():
    scores, rows = _make_t9_rows()
    oov_row_scores = np.full(T9_K, 0.0, dtype=np.float64)  # irrelevant, target=-1
    scores = np.concatenate([scores, oov_row_scores[None, :]], axis=0)
    rows = {
        "target_class": np.concatenate([rows["target_class"], [-1]]),
        "eval_mask": np.concatenate([rows["eval_mask"], [True]]),
        "eval_user_key": np.concatenate([rows["eval_user_key"], [2]]),  # user C
        "client_id": np.concatenate([rows["client_id"], [7]]),
    }
    return scores, rows


def test_t9_oov_extension_end_to_end_vs_closed():
    scores, rows = _make_t9_oov_extension_rows()
    scores_fn = _scores_fn_factory(scores, batch_size=3)
    result = evaluate(scores_fn, rows, k=20, chunk_size=11)

    macro_mrr_e2e = result["end_to_end"]["macro"]["mrr@20"]
    macro_mrr_closed = result["closed"]["macro"]["mrr@20"]

    assert macro_mrr_e2e == pytest.approx(0.4166666666666667, abs=1e-6)
    assert macro_mrr_closed == pytest.approx(0.5, abs=1e-9)

    # OOV row is counted in the end-to-end denominator, dropped from closed.
    assert result["end_to_end"]["n"] == 5
    assert result["closed"]["n"] == 4


@pytest.mark.xfail(strict=True, reason="dropping target_class==-1 rows under eval_mask")
def test_t9_broken_drop_oov_rows():
    scores, rows = _make_t9_oov_extension_rows()
    # The bug: silently drop -1-under-eval_mask rows instead of counting them
    # with zero credit -- simulated by removing the row entirely before
    # calling the (correct) evaluator, so its end-to-end computation
    # degenerates to the closed-catalogue computation.
    keep = rows["target_class"] >= 0
    dropped_rows = {name: np.asarray(val)[keep] for name, val in rows.items()}
    dropped_scores = scores[keep]

    scores_fn = _scores_fn_factory(dropped_scores, batch_size=4)
    result = evaluate(scores_fn, dropped_rows, k=20, chunk_size=11)
    macro_mrr_e2e = result["end_to_end"]["macro"]["mrr@20"]

    assert macro_mrr_e2e == pytest.approx(0.4166666666666667, abs=1e-6)


# ===========================================================================
# T8 -- tiny exhaustive reference (FABLE_PLAN.md sec. E)
#
# K=11, 7 decisions, k=3. See the module docstring below each row for the
# property it exercises. Ranks and metric values are hand-derived directly
# from the rank definition:
#     rank = 1 + #{j: s_j > s_t} + #{j < t: s_j == s_t}
# ===========================================================================

T8_K = 11
T8_KCUT = 3

# fmt: off
_T8_ROWS = [
    # (name, user_key, client_id, eval_mask, target, scores)
    ("R1_tie_lower_id",  0, 7, True,  5, [3, 2, 8, 1, 0, 8, 4, 5, 6, 7, 9]),
    ("R2_tie_higher_id", 0, 7, True,  4, [9, 8, 7, 6, 5, 2, 5, 1, 0, 3, 4]),
    ("R3_oov",           1, 2, True, -1, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0]),
    ("R4_excluded",      1, 2, False, 0, [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 10]),
    ("R5_revisit",       2, 7, True,  7, [1, 2, 3, 4, 5, 6, 7, 10, 8, 9, 0]),
    ("R6_plain_B",       1, 2, True,  3, [10, 1, 2, 9, 3, 4, 5, 6, 7, 8, 0]),
    ("R7_plain_C_miss",  2, 7, True,  9, [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]),
]
# fmt: on


def _t8_rows_and_scores():
    scores = np.array([r[5] for r in _T8_ROWS], dtype=np.float64)
    rows = {
        "target_class": np.array([r[4] for r in _T8_ROWS], dtype=np.int64),
        "eval_mask": np.array([r[3] for r in _T8_ROWS], dtype=bool),
        "eval_user_key": np.array([r[1] for r in _T8_ROWS], dtype=np.int64),
        "client_id": np.array([r[2] for r in _T8_ROWS], dtype=np.int64),
    }
    return scores, rows


def _bruteforce_rank(scores_row: np.ndarray, target: int) -> int:
    """Independent reference: stable sort by (-score, class_id) ascending;
    1-based position of `target` in that order. Ascending id wins ties, so
    the target never benefits from one -- matches the rank definition
    exactly without using the chunked accumulation the implementation uses.
    """
    order = sorted(range(len(scores_row)), key=lambda j: (-scores_row[j], j))
    return order.index(target) + 1


def test_t8_bruteforce_ranks_match_hand_derivation():
    scores, rows = _t8_rows_and_scores()
    expected_ranks = {"R1_tie_lower_id": 3, "R2_tie_higher_id": 5, "R5_revisit": 1,
                       "R6_plain_B": 2, "R7_plain_C_miss": 10}
    by_name = {r[0]: i for i, r in enumerate(_T8_ROWS)}
    for name, expected in expected_ranks.items():
        i = by_name[name]
        got = _bruteforce_rank(scores[i], int(rows["target_class"][i]))
        assert got == expected, f"{name}: bruteforce rank {got} != hand-derived {expected}"


def test_t8_chunked_ranks_match_bruteforce_for_all_chunk_sizes():
    scores, rows = _t8_rows_and_scores()
    scores_t = torch.as_tensor(scores, dtype=torch.float32)
    target = torch.as_tensor(rows["target_class"], dtype=torch.long)
    eval_mask = rows["eval_mask"]
    sel = eval_mask & (rows["target_class"] >= 0)
    sel_idx = np.nonzero(sel)[0]

    bf_ranks = [
        _bruteforce_rank(scores[i], int(rows["target_class"][i])) for i in sel_idx
    ]

    for chunk_size in (1, 4, 11):
        got = _ranks_chunked(
            scores_t[sel_idx], target[sel_idx], chunk_size
        ).tolist()
        assert got == bf_ranks, f"chunk_size={chunk_size}: {got} != {bf_ranks}"


# --- hand-computed expected micro/macro Recall@3 / MRR@3 -------------------
#
# Per-row ranks (hand-derived, cross-checked by the bruteforce test above):
#   R1=3 (hit)  R2=5 (miss)  R3=OOV(0 credit)  R5=1 (hit)
#   R6=2 (hit)  R7=10 (miss)                    [R4 excluded: eval_mask=False]
#
# End-to-end population (eval_mask): R1,R2,R3,R5,R6,R7 (n=6)
# Closed population (eval_mask & target>=0): R1,R2,R5,R6,R7 (n=5)
# Users: A={R1,R2}  B={R3,R6} (end-to-end) / {R6} (closed)  C={R5,R7}

def _credit(rank, k=T8_KCUT):
    if rank is None or rank > k:
        return 0.0, 0.0, 0.0
    return 1.0, 1.0 / rank, 1.0 / math.log2(rank + 1)

R1, R2, R5, R6, R7 = 3, 5, 1, 2, 10  # ranks
c1, c2, c5, c6, c7 = (_credit(R1), _credit(R2), _credit(R5), _credit(R6), _credit(R7))
c3 = (0.0, 0.0, 0.0)  # OOV: zero credit, only present in end-to-end population


def _mean(*vals):
    return sum(vals) / len(vals)


EXPECTED = {
    "end_to_end": {
        "n": 6,
        "micro": {
            "recall@3": _mean(c1[0], c2[0], c3[0], c5[0], c6[0], c7[0]),
            "mrr@3": _mean(c1[1], c2[1], c3[1], c5[1], c6[1], c7[1]),
            "ndcg@3": _mean(c1[2], c2[2], c3[2], c5[2], c6[2], c7[2]),
        },
        "macro": {
            # A={c1,c2}  B={c3,c6}  C={c5,c7}
            "recall@3": _mean(_mean(c1[0], c2[0]), _mean(c3[0], c6[0]), _mean(c5[0], c7[0])),
            "mrr@3": _mean(_mean(c1[1], c2[1]), _mean(c3[1], c6[1]), _mean(c5[1], c7[1])),
            "ndcg@3": _mean(_mean(c1[2], c2[2]), _mean(c3[2], c6[2]), _mean(c5[2], c7[2])),
            "user_count": 3,
        },
    },
    "closed": {
        "n": 5,
        "micro": {
            "recall@3": _mean(c1[0], c2[0], c5[0], c6[0], c7[0]),
            "mrr@3": _mean(c1[1], c2[1], c5[1], c6[1], c7[1]),
            "ndcg@3": _mean(c1[2], c2[2], c5[2], c6[2], c7[2]),
        },
        "macro": {
            # A={c1,c2}  B={c6} (only, R3 dropped)  C={c5,c7}
            "recall@3": _mean(_mean(c1[0], c2[0]), c6[0], _mean(c5[0], c7[0])),
            "mrr@3": _mean(_mean(c1[1], c2[1]), c6[1], _mean(c5[1], c7[1])),
            "ndcg@3": _mean(_mean(c1[2], c2[2]), c6[2], _mean(c5[2], c7[2])),
            "user_count": 3,
        },
    },
}


def _run_t8(chunk_size: int):
    scores, rows = _t8_rows_and_scores()
    scores_fn = _scores_fn_factory(scores, batch_size=3)
    return evaluate(scores_fn, rows, k=T8_KCUT, chunk_size=chunk_size)


@pytest.mark.parametrize("chunk_size", [1, 4, 11])
def test_t8_micro_macro_recall_mrr_ndcg(chunk_size):
    result = _run_t8(chunk_size)

    for pop in ("end_to_end", "closed"):
        assert result[pop]["n"] == EXPECTED[pop]["n"], pop
        for scope in ("micro", "macro"):
            for name, expected_val in EXPECTED[pop][scope].items():
                got = result[pop][scope][name]
                assert got == pytest.approx(expected_val, abs=1e-6), f"{pop}.{scope}.{name}"


def test_t8_chunk_sizes_agree_exactly():
    r1 = _run_t8(1)
    r4 = _run_t8(4)
    r11 = _run_t8(11)

    def strip(d):
        return {
            pop: {"n": d[pop]["n"], "micro": d[pop]["micro"], "macro": d[pop]["macro"]}
            for pop in ("end_to_end", "closed")
        }

    assert strip(r1) == strip(r4) == strip(r11)


# --- broken variants (strict xfail) ----------------------------------------

@pytest.mark.xfail(strict=True, reason="grouping macro by client_id")
def test_t8_broken_group_by_client_id():
    scores, rows = _t8_rows_and_scores()
    broken_rows = dict(rows)
    broken_rows["eval_user_key"] = rows["client_id"]
    scores_fn = _scores_fn_factory(scores, batch_size=3)
    result = evaluate(scores_fn, broken_rows, k=T8_KCUT, chunk_size=4)

    assert result["end_to_end"]["macro"]["mrr@3"] == pytest.approx(
        EXPECTED["end_to_end"]["macro"]["mrr@3"], abs=1e-6
    )
    assert result["end_to_end"]["macro"]["user_count"] == 3


@pytest.mark.xfail(strict=True, reason="dropping target_class==-1 rows under eval_mask")
def test_t8_broken_drop_oov_rows():
    scores, rows = _t8_rows_and_scores()
    keep = rows["target_class"] >= 0
    dropped_rows = {name: np.asarray(val)[keep] for name, val in rows.items()}
    dropped_scores = scores[keep]
    scores_fn = _scores_fn_factory(dropped_scores, batch_size=3)
    result = evaluate(scores_fn, dropped_rows, k=T8_KCUT, chunk_size=4)

    # Correct end-to-end n is 6 (OOV row counted); the broken (dropped) setup
    # only has 5 eval_mask rows total (R4 was never eval_mask anyway), so
    # end-to-end degenerates to n=5 instead of 6.
    assert result["end_to_end"]["n"] == 6
    assert result["end_to_end"]["micro"]["mrr@3"] == pytest.approx(
        EXPECTED["end_to_end"]["micro"]["mrr@3"], abs=1e-6
    )


@pytest.mark.xfail(strict=True, reason='"target wins ties" rank rule')
def test_t8_broken_target_always_wins_ties():
    """Broken rank rule: rank = 1 + #{j: s_j > s_t} only (no eq_before term
    at all, so ties NEVER count against the target, even with a lower class
    id). Recomputed only for R1 (the tie-with-lower-id row), where this
    changes the rank from the correct 3 down to 2."""
    scores, rows = _t8_rows_and_scores()
    i = 0  # R1_tie_lower_id
    row_scores = scores[i]
    target = int(rows["target_class"][i])
    s_t = row_scores[target]
    broken_rank = 1 + int((row_scores > s_t).sum())  # no eq_before term
    assert broken_rank == 2  # documents the bug's effect

    correct_rank = _bruteforce_rank(row_scores, target)
    assert correct_rank == 3
    assert broken_rank == correct_rank


@pytest.mark.xfail(strict=True, reason="filtering out items seen in the context")
def test_t8_broken_filter_seen_items():
    """Broken candidate filtering: the revisited target item (R5, "seen in
    the context") gets filtered out of the candidate set, simulated here by
    setting its own score to -inf before ranking -- the worst possible
    outcome for a filtered-out item. The correct evaluator (no filtering,
    per FABLE_PLAN.md) must NOT do this and ranks R5 at 1 (a hit); the
    broken version buries it at the bottom (a miss)."""
    scores, rows = _t8_rows_and_scores()
    broken_scores = scores.copy()
    i = 4  # R5_revisit
    target = int(rows["target_class"][i])
    broken_scores[i, target] = float("-inf")

    scores_fn = _scores_fn_factory(broken_scores, batch_size=3)
    result = evaluate(scores_fn, rows, k=T8_KCUT, chunk_size=4)

    assert result["end_to_end"]["micro"]["recall@3"] == pytest.approx(
        EXPECTED["end_to_end"]["micro"]["recall@3"], abs=1e-6
    )
