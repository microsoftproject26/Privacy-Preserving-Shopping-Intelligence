"""Exact full-catalogue ranking: the tie rule, chunk-size invariance, non-finite refusal, sampled refusal.

Proves: rank = 1 + #{s_j > s_t} + #{j < t : s_j == s_t} on hand-checked ties (float and int64); ranks identical for
every class_chunk (integer counts, tolerance 0); the streaming path (q once, item table in score blocks) equals the
precomputed path bit for bit when one block covers K, and differs from it only at float-level near-ties otherwise;
cached and recomputed blocks agree and a nondeterministic kernel is caught; NaN / inf anywhere is refused (a naive
ranker ranks a NaN target 1); candidate subsets and wrong-K scores are refused.
Does not prove: GPU kernels are bit-identical to CPU, or anything about model quality.
"""
from __future__ import annotations

import torch
from fullrank_testkit import assert_raises, nc

from ppsi.evaluation.fullrank.errors import (
    NondeterministicScoreError,
    NonFiniteScoreError,
    SampledEvaluationRefused,
)
from ppsi.evaluation.fullrank.ranking import rank_from_query, rank_from_scores

TIES = torch.tensor([[0.5, 0.9, 0.5, 0.5, 0.2]] * 4)
TIE_TARGETS = torch.tensor([2, 0, 3, 4])
TIE_EXPECTED = torch.tensor([3, 2, 4, 5])   # t=2: 0.9 + tie(0); t=0: 0.9; t=3: 0.9 + ties(0,2); t=4: four greater


# ------------------------------------------------------------------------------------------------ checks
def check_tie_rule(rank_fn):
    assert torch.equal(rank_fn(TIES, TIE_TARGETS), TIE_EXPECTED)
    ints = (TIES * 10).to(torch.long)
    assert torch.equal(rank_fn(ints, TIE_TARGETS), TIE_EXPECTED)


def check_chunk_invariance(rank_fn_chunk):
    g = torch.Generator().manual_seed(7)
    s = torch.round(torch.randn(50, 301, generator=g) * 4) / 4           # many exact ties
    t = torch.randint(0, 301, (50,), generator=g)
    ref = rank_fn_chunk(s, t, 301)
    for c in (1, 2, 7, 64, 300, 32768):
        assert torch.equal(rank_fn_chunk(s, t, c), ref), c


def check_nonfinite_refused(rank_fn):
    s = torch.randn(3, 10)
    for bad in (float("nan"), float("inf"), float("-inf")):
        x = s.clone()
        x[1, 4] = bad                                                   # the target's own score
        assert_raises(NonFiniteScoreError, rank_fn, x, torch.tensor([0, 4, 2]))
        y = s.clone()
        y[0, 9] = bad                                                   # a competitor's score
        assert_raises(NonFiniteScoreError, rank_fn, y, torch.tensor([0, 4, 2]))


def check_sampled_refused(rank_fn_kw):
    s = torch.randn(4, 12)
    t = torch.tensor([1, 2, 3, 4])
    assert_raises(SampledEvaluationRefused, rank_fn_kw, s, t, candidates=torch.arange(6))
    assert_raises(SampledEvaluationRefused, rank_fn_kw, s, t, K_expected=24)


# ------------------------------------------------------------------------------------------------ variants
def naive_rank(s, t, class_chunk=None):
    """A naive comparison-based ranker (no finiteness check): 1 + #{s_j > s_t} + #{j < t : s_j == s_t}."""
    st = s.gather(1, t.view(-1, 1))
    idx = torch.arange(s.shape[1]).view(1, -1)
    return 1 + (s > st).sum(1) + ((s == st) & (idx < t.view(-1, 1))).sum(1)


def default_ranks(s, t):
    return rank_from_scores(s, t).ranks


def optimistic_ties(s, t):
    st = s.gather(1, t.view(-1, 1))
    return 1 + (s > st).sum(1)


def chunk_local_ties(s, t, c):
    """Bug: ties counted only inside the target's own chunk."""
    st = s.gather(1, t.view(-1, 1))
    idx = torch.arange(s.shape[1]).view(1, -1)
    same_chunk = (idx // c) == (t.view(-1, 1) // c)
    return 1 + (s > st).sum(1) + ((s == st) & (idx < t.view(-1, 1)) & same_chunk).sum(1)


def permissive_sampled(s, t, candidates=None, K_expected=None):
    if candidates is not None:
        s = s[:, candidates]
    return optimistic_ties(s, t.clamp(max=s.shape[1] - 1))


# ------------------------------------------------------------------------------------------------ tests
def test_tie_rule():
    check_tie_rule(default_ranks)


def test_tie_rule_streaming_duplicate_rows():
    W = torch.tensor([[1.0, 0.0], [2.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.5, 0.0]])
    q = torch.tensor([[0.5, 3.0]] * 4)
    r = rank_from_query(q, W, None, TIE_TARGETS, score_block=5)
    assert torch.equal(r.ranks, TIE_EXPECTED)
    assert torch.equal(r.n_tied, torch.tensor([2, 2, 2, 0]))
    assert torch.equal(r.pessimistic_ranks, torch.tensor([4, 4, 4, 5]))


@nc("optimistic tie rule (the target wins ties)")
def test_nc_optimistic_ties():
    check_tie_rule(optimistic_ties)


def test_class_chunk_invariance():
    check_chunk_invariance(lambda s, t, c: rank_from_scores(s, t, class_chunk=c).ranks)


def test_class_chunk_invariance_streaming():
    g = torch.Generator().manual_seed(3)
    q, W, b = torch.randn(40, 8, generator=g), torch.randn(300, 8, generator=g), torch.randn(300, generator=g)
    t = torch.randint(0, 300, (40,), generator=g)
    ref = rank_from_query(q, W, b, t, score_block=64, class_chunk=64).ranks
    for c in (1, 5, 17, 64):
        assert torch.equal(rank_from_query(q, W, b, t, score_block=64, class_chunk=c).ranks, ref)


@nc("ties counted only within the target's chunk (chunk-dependent ranks)")
def test_nc_chunk_local_ties():
    check_chunk_invariance(chunk_local_ties)


def test_single_block_equals_precomputed_and_naive():
    g = torch.Generator().manual_seed(11)
    q, W, b = torch.randn(64, 32, generator=g), torch.randn(500, 32, generator=g), torch.randn(500, generator=g)
    t = torch.randint(0, 500, (64,), generator=g)
    full = torch.addmm(b, q, W.t())                                   # the model's logits
    s = rank_from_query(q, W, b, t, score_block=500).ranks
    assert torch.equal(s, rank_from_scores(full, t).ranks)
    assert torch.equal(s, naive_rank(full, t))


def test_score_block_invariance_within_tolerance():
    """Different score blocks change float32 scores at the ulp level; ranks may move only at genuine near-ties."""
    g = torch.Generator().manual_seed(5)
    q, W, b = torch.randn(200, 128, generator=g), torch.randn(2000, 128, generator=g), torch.randn(2000, generator=g)
    t = torch.randint(0, 2000, (200,), generator=g)
    W_near = W.clone()
    W_near[1::2] = W_near[0::2] + 1e-7 * torch.randn(1000, 128, generator=g)   # near-duplicate items (sub-ulp)
    for W_case, b_case in ((W, b), (W_near, None)):
        full = torch.addmm(b_case, q, W_case.t()) if b_case is not None else q @ W_case.t()
        ref = rank_from_query(q, W_case, b_case, t, score_block=2000).ranks
        s_t = full.gather(1, t.view(-1, 1))
        tol = 1e-4 * float(full.abs().max())
        close = ((full - s_t).abs() <= tol).sum(1) - 1                 # competitors within tol of the target
        for blk in (1, 7, 100, 1024):
            r = rank_from_query(q, W_case, b_case, t, score_block=blk).ranks
            dr = (r - ref).abs()
            assert bool((dr <= close).all()), blk                        # a rank moves only by near-tied competitors


def test_cache_and_recompute_agree():
    g = torch.Generator().manual_seed(9)
    q, W, b = torch.randn(30, 16, generator=g), torch.randn(700, 16, generator=g), torch.randn(700, generator=g)
    t = torch.randint(0, 700, (30,), generator=g)
    a = rank_from_query(q, W, b, t, score_block=100, cache_bytes=0)
    c = rank_from_query(q, W, b, t, score_block=100, cache_bytes=1 << 30)
    assert torch.equal(a.ranks, c.ranks) and torch.equal(a.n_tied, c.n_tied)


def test_nondeterministic_kernel_refused(monkeypatch):
    g = torch.Generator().manual_seed(2)
    q, W, b = torch.randn(8, 4, generator=g), torch.randn(50, 4, generator=g), torch.randn(50, generator=g)
    t = torch.randint(0, 50, (8,), generator=g)
    real = torch.addmm
    calls = {"n": 0}

    def drifting(*a, **k):
        calls["n"] += 1
        out = real(*a, **k)
        return out + 1e-3 if calls["n"] > 5 else out                  # pass 2 differs from pass 1

    monkeypatch.setattr(torch, "addmm", drifting)
    assert_raises(NondeterministicScoreError, rank_from_query, q, W, b, t, score_block=10, cache_bytes=0)


def test_nonfinite_refused():
    check_nonfinite_refused(lambda s, t: rank_from_scores(s, t))
    g = torch.Generator().manual_seed(4)
    q, W, b = torch.randn(3, 4, generator=g), torch.randn(10, 4, generator=g), torch.randn(10, generator=g)
    t = torch.tensor([0, 4, 2])
    for name in ("q", "W", "b"):
        x = {"q": q.clone(), "W": W.clone(), "b": b.clone()}
        x[name].view(-1)[1] = float("nan")
        assert_raises(NonFiniteScoreError, rank_from_query, x["q"], x["W"], x["b"], t, score_block=3)
    big = torch.full((10, 4), 3e38)
    assert_raises(NonFiniteScoreError, rank_from_query, torch.full((3, 4), 3e38), big, None, t, score_block=3)


def test_naive_ranker_defect_nan_target_ranks_first():
    """Documents the defect the evaluator fixes: a NaN target score gets rank 1, silently."""
    s = torch.randn(1, 10)
    s[0, 3] = float("nan")
    assert int(naive_rank(s, torch.tensor([3]))[0]) == 1


@nc("a naive ranker accepts a NaN target score (rank 1)")
def test_nc_naive_accepts_nan():
    check_nonfinite_refused(naive_rank)


def test_sampled_and_wrong_k_refused():
    check_sampled_refused(lambda s, t, **kw: rank_from_scores(s, t, **kw))
    q, W = torch.randn(4, 3), torch.randn(12, 3)
    assert_raises(SampledEvaluationRefused, rank_from_query, q, W, None, torch.tensor([1, 2, 3, 4]), K_expected=24)


@nc("a sampled-candidate evaluator is accepted for a headline row")
def test_nc_sampled_accepted():
    check_sampled_refused(permissive_sampled)


def test_deterministic_repeat():
    g = torch.Generator().manual_seed(8)
    q, W, b = torch.randn(33, 8, generator=g), torch.randn(257, 8, generator=g), torch.randn(257, generator=g)
    t = torch.randint(0, 257, (33,), generator=g)
    a = rank_from_query(q, W, b, t, score_block=50)
    c = rank_from_query(q, W, b, t, score_block=50)
    assert torch.equal(a.ranks, c.ranks)
