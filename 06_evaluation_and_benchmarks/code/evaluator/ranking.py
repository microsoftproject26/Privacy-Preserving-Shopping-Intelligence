"""v4eval.ranking — exact full-K target ranks (EVALUATION_CONTRACT_V4 §6).

Rank rule, quoted from the snapshot evaluator (reuse/eval_full.py:L63; evaluator.py:L42-L48):
    rank = 1 + #{j : s_j > s_t} + #{j < t : s_j == s_t}
Ascending class id wins ties; the target never benefits from a tie. Every class of the full catalogue is a candidate
(no sampling, no subset, no top-k; E6.2, G2). The counts accumulate over an exact partition of the K axis, so the ranks
are identical for every `class_chunk` (integer counts, tolerance 0; G3).

Two entry points:
  rank_from_scores(scores [n, K], target)          precomputed scores, float or int64 (the snapshot's semantics).
  rank_from_query(q [n, d], W [K, d], b, target)   the V4 streaming path: q is computed once by the caller, the item
                                                   table is streamed in `score_block` rows; no [n, K] tensor exists.
Streaming and score blocks. A float32 GEMM is not bit-identical across block widths on CPU (measured in EVAL-02:
max |diff| ~5e-5 at d = 256), so the score of (i, j) depends on the registered `score_block`. `score_block` is therefore
a fixed part of the evaluator configuration (identical for every method, recorded in every result row). To make the
quoted tie rule apply to bit-equal scores, the target score s_t is taken from the SAME block computation as its
competitors: pass 1 computes each block and gathers s_t, pass 2 recomputes (or re-reads a cached) block and counts.
Pass 2 checks that the target's score reproduces bit-for-bit; a nondeterministic kernel is refused. With
score_block >= K the block is one addmm, bit-identical to BaseAdapter.logits(q) and to the snapshot evaluator's input.
Non-finite q, W, b or scores are refused (G1): the snapshot silently ranks a NaN target 1.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from .errors import NondeterministicScoreError, NonFiniteScoreError, SampledEvaluationRefused

DEFAULT_CLASS_CHUNK = 32768
DEFAULT_SCORE_BLOCK = 32768
DEFAULT_CACHE_BYTES = 256 * 1024 * 1024


@dataclass
class RankResult:
    ranks: Tensor        # int64 [n], in [1, K]: the quoted rule
    n_tied: Tensor       # int64 [n]: other classes whose score equals s_t exactly (tie diagnostic, E6.4)
    eq_before: Tensor    # int64 [n]: tied classes with a lower class id (the ones counted against the target)

    @property
    def pessimistic_ranks(self) -> Tensor:
        """1 + #{s_j > s_t} + #{j != t : s_j == s_t} (descriptive sensitivity only; the headline uses `ranks`)."""
        return self.ranks - self.eq_before + self.n_tied


def _refuse_nonfinite(t: Optional[Tensor], what: str) -> None:
    if t is None or not t.is_floating_point():
        return
    if not bool(torch.isfinite(t).all()):
        raise NonFiniteScoreError(f"non-finite values in {what}; the V4 evaluator refuses them (contract G1)")


def _check_target(target: Tensor, K: int, n: int) -> Tensor:
    if target.ndim != 1 or target.shape[0] != n:
        raise ValueError("target must be [n] and match the score rows")
    target = target.to(torch.long)
    if n and (int(target.min()) < 0 or int(target.max()) >= K):
        raise ValueError("every ranked target must be a class in [0, K); OOV and censored rows are never ranked")
    return target


def _refuse_sampling(K: int, K_expected: Optional[int], candidates) -> None:
    if candidates is not None:
        raise SampledEvaluationRefused("candidate subsets / sampled negatives are refused: V4 ranks the full K (G2)")
    if K_expected is not None and K != int(K_expected):
        raise SampledEvaluationRefused(f"scores cover {K} classes, the registered catalogue has {K_expected} (G2)")


def _accumulate(S: Tensor, start: int, s_t: Tensor, target: Tensor, class_chunk: int,
                gt: Tensor, eq_before: Tensor, eq_other: Tensor) -> None:
    width = S.shape[1]
    tcol = target.view(-1, 1)
    st = s_t.view(-1, 1)
    for a in range(0, width, class_chunk):
        c = S[:, a:a + class_chunk]
        idx = torch.arange(start + a, start + a + c.shape[1], device=S.device).view(1, -1)
        gt += (c > st).sum(1)
        eq = c == st
        eq_before += (eq & (idx < tcol)).sum(1)
        eq_other += (eq & (idx != tcol)).sum(1)


def rank_from_scores(scores: Tensor, target: Tensor, *, class_chunk: int = DEFAULT_CLASS_CHUNK,
                     K_expected: Optional[int] = None, candidates=None) -> RankResult:
    """Exact ranks from a precomputed [n, K] score matrix (float of any width, or int64; never coerced)."""
    if scores.ndim != 2:
        raise ValueError(f"scores must be [n, K], got {tuple(scores.shape)}")
    if class_chunk < 1:
        raise ValueError("class_chunk must be >= 1")
    n, K = scores.shape
    _refuse_sampling(K, K_expected, candidates)
    _refuse_nonfinite(scores, "scores")
    target = _check_target(target.to(scores.device), K, n)
    s_t = scores.gather(1, target.view(-1, 1)).squeeze(1)
    z = lambda: torch.zeros(n, dtype=torch.long, device=scores.device)  # noqa: E731
    gt, eqb, eqo = z(), z(), z()
    _accumulate(scores, 0, s_t, target, int(class_chunk), gt, eqb, eqo)
    return RankResult(ranks=1 + gt + eqb, n_tied=eqo, eq_before=eqb)


def block_scores(q: Tensor, W: Tensor, b: Optional[Tensor], s: int, e: int) -> Tensor:
    """Scores of classes [s, e): BaseAdapter.logits restricted to the block (addmm with a bias, matmul without).
    With s = 0 and e = K this is BaseAdapter.logits(q) op-for-op (the frozen T3 single block)."""
    S = q @ W[s:e].t() if b is None else torch.addmm(b[s:e], q, W[s:e].t())
    _refuse_nonfinite(S, f"scores of block [{s}, {e})")
    return S


@torch.no_grad()
def rank_from_query(q: Tensor, W: Tensor, b: Optional[Tensor], target: Tensor, *,
                    score_block: int = DEFAULT_SCORE_BLOCK, class_chunk: int = DEFAULT_CLASS_CHUNK,
                    K_expected: Optional[int] = None, cache_bytes: int = DEFAULT_CACHE_BYTES,
                    candidates=None) -> RankResult:
    """Exact ranks with q computed once and the item table streamed in `score_block` rows.

    Block scores are `torch.addmm(b_blk, q, W_blk.t())` (BaseAdapter.logits restricted to the block), or q @ W_blk.t()
    without a bias."""
    if q.ndim != 2 or W.ndim != 2 or q.shape[1] != W.shape[1]:
        raise ValueError(f"q [n, d] and W [K, d] required, got {tuple(q.shape)} and {tuple(W.shape)}")
    if score_block < 1 or class_chunk < 1:
        raise ValueError("score_block and class_chunk must be >= 1")
    n, K = q.shape[0], W.shape[0]
    if b is not None and tuple(b.shape) != (K,):
        raise ValueError("bias must be [K]")
    _refuse_sampling(K, K_expected, candidates)
    _refuse_nonfinite(q, "query q")
    _refuse_nonfinite(W, "head weight W")
    _refuse_nonfinite(b, "head bias b")
    target = _check_target(target.to(q.device), K, n)
    blocks = [(s, min(s + int(score_block), K)) for s in range(0, K, int(score_block))]

    def block(s: int, e: int) -> Tensor:
        return block_scores(q, W, b, s, e)

    cache = {}
    budget = int(cache_bytes)
    s_t = torch.empty(n, dtype=q.dtype, device=q.device)
    for s, e in blocks:                                             # pass 1: gather s_t from its own block
        S = block(s, e)
        sel = ((target >= s) & (target < e)).nonzero(as_tuple=True)[0]
        if sel.numel():
            s_t[sel] = S[sel, target[sel] - s]
        nbytes = S.numel() * S.element_size()
        if nbytes <= budget:
            cache[s] = S
            budget -= nbytes
    z = lambda: torch.zeros(n, dtype=torch.long, device=q.device)  # noqa: E731
    gt, eqb, eqo = z(), z(), z()
    for s, e in blocks:                                             # pass 2: count against the same bits
        S = cache.pop(s) if s in cache else block(s, e)
        sel = ((target >= s) & (target < e)).nonzero(as_tuple=True)[0]
        if sel.numel() and not torch.equal(S[sel, target[sel] - s], s_t[sel]):
            raise NondeterministicScoreError(f"block [{s}, {e}) did not reproduce the target scores bit-for-bit")
        _accumulate(S, s, s_t, target, int(class_chunk), gt, eqb, eqo)
    return RankResult(ranks=1 + gt + eqb, n_tied=eqo, eq_before=eqb)
