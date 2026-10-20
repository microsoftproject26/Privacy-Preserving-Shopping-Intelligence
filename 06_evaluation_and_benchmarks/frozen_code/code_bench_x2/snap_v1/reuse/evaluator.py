"""Evaluator -- Task T-C.

Implements the evaluation contract specified in ``FABLE_PLAN.md`` section E
(``.local_ai/gru_v1/FABLE_PLAN.md``): exact full-catalogue ranking (no
``torch.topk``, no candidate filtering), Recall/MRR/NDCG @k in micro and
macro form over two denominators (end-to-end and closed-catalogue), macro
grouping by ``eval_user_key`` (never ``client_id``), and warm/unseen cohort
cuts on ``client_id`` for reporting only.

Implementation only -- no reinterpretation of the semantics above. Whatever
is genuinely ambiguous is left unimplemented rather than guessed.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping

import torch
from torch import Tensor

__all__ = ["evaluate"]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _get(rows: Any, name: str):
    if isinstance(rows, Mapping):
        return rows[name]
    return getattr(rows, name)


def _as_tensor(x, dtype) -> Tensor:
    if isinstance(x, Tensor):
        return x.to(dtype)
    return torch.as_tensor(x, dtype=dtype)


# ---------------------------------------------------------------------------
# Rank computation -- class-chunked, no top-k materialisation.
#
# rank = 1 + #{j : s_j > s_t} + #{j < t : s_j == s_t}
#
# Ascending class id wins ties; the target NEVER benefits from a tie (only
# ties with a strictly LOWER class id count against it). Accumulating the two
# counts chunk-by-chunk over the K axis is an exact partition of the sum, so
# the result is identical for every chunk_size -- this is what "no top-k
# materialisation" buys: correctness is independent of how K is chunked.
# ---------------------------------------------------------------------------

def _ranks_chunked(scores: Tensor, target: Tensor, chunk_size: int) -> Tensor:
    """scores: [n, K] float; target: [n] int64 (all >= 0). Returns [n] int64."""
    n, K = scores.shape
    device = scores.device
    s_t = scores.gather(1, target.view(-1, 1)).squeeze(1)  # [n]
    gt = torch.zeros(n, dtype=torch.long, device=device)
    eq_before = torch.zeros(n, dtype=torch.long, device=device)
    target_col = target.view(-1, 1)
    for start in range(0, K, chunk_size):
        end = min(start + chunk_size, K)
        chunk = scores[:, start:end]                                   # [n, w]
        gt += (chunk > s_t.unsqueeze(1)).sum(dim=1)
        idx = torch.arange(start, end, device=device).unsqueeze(0)      # [1, w]
        eq_lower = (chunk == s_t.unsqueeze(1)) & (idx < target_col)
        eq_before += eq_lower.sum(dim=1)
    return 1 + gt + eq_before


# ---------------------------------------------------------------------------
# Metric credit vectors from ranks
# ---------------------------------------------------------------------------

def _credits_from_ranks(ranks: Tensor, k: int) -> dict[str, Tensor]:
    ranks_f = ranks.to(torch.float32)
    hit = (ranks >= 1) & (ranks <= k)
    recall = hit.to(torch.float32)
    zeros = torch.zeros_like(ranks_f)
    mrr = torch.where(hit, 1.0 / ranks_f.clamp(min=1), zeros)
    ndcg = torch.where(hit, 1.0 / torch.log2(ranks_f.clamp(min=1) + 1.0), zeros)
    return {f"recall@{k}": recall, f"mrr@{k}": mrr, f"ndcg@{k}": ndcg}


def _population_block(
    credits: dict[str, Tensor], pop_mask: Tensor, eval_user_key: Tensor
) -> dict[str, Any]:
    idx = pop_mask.nonzero(as_tuple=True)[0]
    n = int(idx.numel())
    micro: dict[str, float] = {}
    macro: dict[str, float] = {}

    if n == 0:
        for name in credits:
            micro[name] = float("nan")
            macro[name] = float("nan")
        macro["user_count"] = 0
        return {"n": 0, "micro": micro, "macro": macro}

    keys = eval_user_key[idx]
    unique_keys, inverse = torch.unique(keys, return_inverse=True)
    n_users = int(unique_keys.numel())

    for name, credit in credits.items():
        sub = credit[idx]
        micro[name] = float(sub.mean().item())

        sums = torch.zeros(n_users, dtype=torch.float32)
        counts = torch.zeros(n_users, dtype=torch.float32)
        sums.index_add_(0, inverse, sub)
        counts.index_add_(0, inverse, torch.ones_like(sub))
        per_user_mean = sums / counts
        macro[name] = float(per_user_mean.mean().item())

    macro["user_count"] = n_users
    return {"n": n, "micro": micro, "macro": macro}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate(
    scores_fn: Callable[[], Iterable],
    rows: Any,
    k: int = 20,
    chunk_size: int = 8192,
) -> dict[str, Any]:
    """Evaluate Recall/MRR/NDCG @k, micro and macro, over both denominators.

    Parameters
    ----------
    scores_fn:
        Zero-argument callable; ``scores_fn()`` returns an iterable of
        ``[B, K]`` score batches (torch tensor or array-like) covering every
        row of ``rows``, in order, batched however the caller likes. Full
        catalogue scores -- no candidate filtering, seen/revisited items
        included.
    rows:
        Provides (by ``rows[name]`` or ``rows.name``) four length-N arrays:
        ``target_class`` (int64, -1 = no target / OOV), ``eval_mask`` (bool),
        ``eval_user_key`` (int64, dense per-user id -- NEVER ``client_id``),
        ``client_id`` (int64; used only for the warm/unseen cohort cut).
    k:
        Cutoff for Recall/MRR/NDCG@k.
    chunk_size:
        Width of the class-chunks used to accumulate the two rank counts.
        Purely a compute/memory knob -- every chunk_size gives an identical
        result (no ``torch.topk``, no top-k materialisation).

    Returns
    -------
    dict with keys ``k``, ``end_to_end`` (denominator = eval_mask),
    ``closed`` (denominator = eval_mask & target_class>=0), each holding
    ``n``, ``micro`` (metric name -> float) and ``macro`` (metric name ->
    float, plus ``user_count``); top-level ``user_count`` mirrors
    ``end_to_end.macro.user_count``; and ``cohorts.warm`` /
    ``cohorts.unseen`` (client_id >= 3 / == 2 respectively), each with the
    same ``end_to_end`` / ``closed`` structure.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    if k < 1:
        raise ValueError("k must be >= 1")

    target_class = _as_tensor(_get(rows, "target_class"), torch.long)
    eval_mask = _as_tensor(_get(rows, "eval_mask"), torch.bool)
    eval_user_key = _as_tensor(_get(rows, "eval_user_key"), torch.long)
    client_id = _as_tensor(_get(rows, "client_id"), torch.long)

    N = target_class.shape[0]
    if not (eval_mask.shape[0] == N and eval_user_key.shape[0] == N and client_id.shape[0] == N):
        raise ValueError("rows.target_class/eval_mask/eval_user_key/client_id must share length")

    ranks = torch.full((N,), -1, dtype=torch.long)

    offset = 0
    for batch_scores in scores_fn():
        batch_scores = _as_tensor(batch_scores, torch.float32)
        if batch_scores.ndim != 2:
            raise ValueError("scores_fn must yield [B,K] score batches")
        B = batch_scores.shape[0]
        if offset + B > N:
            raise ValueError("scores_fn produced more rows than `rows` contains")

        row_idx = torch.arange(offset, offset + B)
        batch_target = target_class[row_idx]
        batch_eval = eval_mask[row_idx]
        need = batch_eval & (batch_target >= 0)
        if bool(need.any()):
            need_rows = row_idx[need]
            sub_scores = batch_scores[need]
            sub_target = batch_target[need]
            ranks[need_rows] = _ranks_chunked(sub_scores, sub_target, chunk_size)
        offset += B

    if offset != N:
        raise ValueError(f"scores_fn produced {offset} rows, expected {N} (len(rows))")

    credits = _credits_from_ranks(ranks, k)

    end_to_end_pop = eval_mask
    closed_pop = eval_mask & (target_class >= 0)

    def block(pop_mask: Tensor) -> dict[str, Any]:
        return _population_block(credits, pop_mask, eval_user_key)

    result: dict[str, Any] = {
        "k": k,
        "end_to_end": block(end_to_end_pop),
        "closed": block(closed_pop),
    }
    result["user_count"] = result["end_to_end"]["macro"]["user_count"]

    warm_mask = client_id >= 3
    unseen_mask = client_id == 2
    result["cohorts"] = {
        "warm": {
            "end_to_end": block(end_to_end_pop & warm_mask),
            "closed": block(closed_pop & warm_mask),
        },
        "unseen": {
            "end_to_end": block(end_to_end_pop & unseen_mask),
            "closed": block(closed_pop & unseen_mask),
        },
    }
    return result
