"""BASE-01: SKNN on a hand-computed fixture — vanilla (idf-free) cosine similarity, top-k selection, and
most-recent sampling, checked against an INDEPENDENT oracle formula (not a re-invocation of `score_session`'s own
code path) written directly in this test.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from v4base.sknn import SessionPool, score_session

K = 10
SESSIONS = [
    [1, 2, 3],          # S0 (oldest)
    [2, 3],             # S1
    [1, 4],             # S2
    [3, 4, 5],          # S3
    [1, 2, 3, 4],       # S4 (most recent)
]
QUERY = [1, 2, 3]


def _oracle_cosine(a, b) -> float:
    a, b = set(a), set(b)
    inter = len(a & b)
    denom = math.sqrt(len(a) * len(b))
    return inter / denom if denom > 0 else 0.0


def _build_pool() -> SessionPool:
    pool = SessionPool(K)
    for s in SESSIONS:
        pool.add(s)
    pool.finalize()
    return pool


def test_top_k_scores_match_a_hand_written_oracle():
    pool = _build_pool()
    got = score_session(pool, QUERY, k=3, sample_size=10)

    sims = [_oracle_cosine(QUERY, s) for s in SESSIONS]                # every session shares >= 1 item with QUERY
    ranked = sorted(range(len(SESSIONS)), key=lambda i: -sims[i])
    top3 = ranked[:3]                                                  # S0 (1.0), S4 (~0.866), S1 (~0.8165)
    assert set(top3) == {0, 4, 1}, f"oracle top-3 changed unexpectedly: {top3} (sims={sims})"

    expected: dict = {}
    for i in top3:
        for it in SESSIONS[i]:
            expected[it] = expected.get(it, 0.0) + sims[i]

    assert set(got) == set(expected)
    for item, val in expected.items():
        assert got[item] == pytest.approx(val, abs=1e-9), f"item {item}: got {got.get(item)} expected {val}"
    # excluded candidates (S2, S3) must contribute nothing: item 5 only appears in S3
    assert 5 not in got


def test_most_recent_sampling_drops_the_oldest_shared_session():
    """sample_size=1 keeps only the single MOST RECENT session sharing an item with the query (S4); S0's exclusive
    contribution (none here, since S0 subset S4) is instead checked via a session with a UNIQUE item that only the
    oldest neighbour carries."""
    pool = SessionPool(K)
    pool.add([1, 9])          # S0: oldest, carries the unique item 9
    pool.add([1, 8])          # S1
    pool.add([1, 7])          # S2 (most recent)
    pool.finalize()

    full = score_session(pool, [1], k=3, sample_size=10)
    assert 9 in full and 8 in full and 7 in full, "with no sampling limit every neighbour must contribute"

    limited = score_session(pool, [1], k=3, sample_size=2)
    assert 9 not in limited, "most-recent sampling must drop the oldest session's exclusive item once sample_size < n"
    assert 8 in limited and 7 in limited


def test_no_shared_item_is_an_empty_score_not_an_error():
    pool = _build_pool()
    assert score_session(pool, [99], k=3, sample_size=10) == {}
    assert score_session(pool, [], k=3, sample_size=10) == {}
