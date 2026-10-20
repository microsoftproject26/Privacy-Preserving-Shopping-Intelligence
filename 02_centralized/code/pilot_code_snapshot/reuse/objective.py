"""Training objective -- Task T-B.

Implements the objective specified in ``FABLE_PLAN.md`` section C
(``.local_ai/gru_v1/FABLE_PLAN.md``):

    exact-CE (PRIMARY): full-catalogue cross-entropy over rows with
    ``loss_mask & (target_class >= 0)``, summed (never averaged) per call so
    that gradient-accumulation callers can sum numerators and denominators
    across microbatches and divide exactly once.

    sampled-CE (FALLBACK, disabled by default): complete spec per section C,
    reachable only with an explicit ``enabled=True``. Never used unless the
    pre-registered E2 throughput decision rule (section C) says so -- that
    decision is out of scope for this module, which only implements the
    mechanics.

Implementation only -- no reinterpretation of the semantics above. Whatever
is genuinely ambiguous is left unimplemented rather than guessed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "exact_ce_loss",
    "sampled_ce_loss",
    "LossAccumulator",
]


# ---------------------------------------------------------------------------
# Exact full-catalogue cross-entropy (PRIMARY)
# ---------------------------------------------------------------------------

def exact_ce_loss(
    logits: Tensor,          # [B, K]
    target_class: Tensor,    # [B] int64, may contain -1 (no-target / OOV)
    loss_mask: Tensor,       # [B] bool
) -> tuple[Tensor, Tensor]:
    """Exact full-catalogue cross-entropy, summed over contributing rows.

    Selection: ``loss_mask & (target_class >= 0)``.

    Returns ``(loss_sum, n_contributing)`` -- the SUM of per-row losses and
    the COUNT of contributing rows, deliberately NOT divided here. Callers
    accumulate ``loss_sum``/``n_contributing`` across microbatches and divide
    exactly once at the end (see ``LossAccumulator``); a per-microbatch mean
    would silently mis-weight uneven microbatches.

    If no rows contribute, returns a zero loss that still participates in
    autograd (``logits.sum() * 0``) and a count of 0.

    No label smoothing, no class weights, no extra softmax -- raw logits are
    passed straight to ``F.cross_entropy``.
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must be [B,K], got shape {tuple(logits.shape)}")
    if target_class.shape != loss_mask.shape:
        raise ValueError("target_class and loss_mask must share shape [B]")
    if target_class.shape[0] != logits.shape[0]:
        raise ValueError("target_class/loss_mask must have the same B as logits")

    sel = loss_mask.to(torch.bool) & (target_class >= 0)
    n_contributing = sel.sum()

    if int(n_contributing.item()) == 0:
        zero_loss = logits.sum() * 0
        return zero_loss, n_contributing

    sel_logits = logits[sel]
    sel_target = target_class[sel].to(torch.long)
    loss_sum = F.cross_entropy(sel_logits, sel_target, reduction="sum")
    return loss_sum, n_contributing


@dataclass
class LossAccumulator:
    """Accumulates ``(loss_sum, n)`` pairs across microbatches and divides
    exactly once, at the end, via ``.mean()``.

    This is what makes gradient accumulation correct: summing numerators and
    denominators separately and dividing once, rather than averaging
    per-microbatch means (which mis-weights uneven microbatches).
    """

    _loss_sum: Tensor | float = field(default=0.0)
    _n: Tensor | float = field(default=0.0)

    def add(self, loss_sum: Tensor, n: Tensor) -> "LossAccumulator":
        self._loss_sum = self._loss_sum + loss_sum
        self._n = self._n + n
        return self

    def mean(self) -> Tensor:
        """Return ``loss_sum / n``, dividing exactly once.

        Raises if nothing has been accumulated (n == 0) -- callers that need
        to handle the zero-contributing-rows case explicitly should inspect
        ``.n`` before calling ``.mean()``.
        """
        n = self._n
        n_val = n.item() if isinstance(n, Tensor) else n
        if n_val == 0:
            raise ZeroDivisionError("LossAccumulator.mean() called with n_contributing == 0")
        return self._loss_sum / n

    @property
    def n(self) -> Tensor | float:
        return self._n

    @property
    def loss_sum(self) -> Tensor | float:
        return self._loss_sum


# ---------------------------------------------------------------------------
# Sampled cross-entropy (FALLBACK -- default-OFF, complete spec per sec. C)
# ---------------------------------------------------------------------------

def sampled_ce_loss(
    logits: Tensor,          # [B, K] -- full-catalogue logits (positive column selected by target_class)
    target_class: Tensor,    # [B] int64, may contain -1
    loss_mask: Tensor,       # [B] bool
    log_q: Tensor,           # [K] float, precomputed log q(j) over the K real classes
    *,
    n_negatives: int = 2048,
    enabled: bool = False,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Sampled-softmax cross-entropy fallback (FABLE_PLAN.md sec. C).

    Never reachable by default: raises ``RuntimeError`` unless
    ``enabled=True`` is passed explicitly. This mirrors the plan's
    requirement that the sampled objective is only used if the pre-registered
    E2 throughput rule says so -- a decision made by the caller, never by
    this function.

    Mechanics (complete spec):
      * proposal ``q(j) ~ representative_views(j) ** 0.5`` over the K real
        classes; ``log_q`` is precomputed by the caller and passed in.
      * ONE shared set of ``N = n_negatives`` negatives per microbatch, drawn
        via ``torch.multinomial(q, N, replacement=True)`` (duplicates kept).
      * candidates per row = ``[positive, n_1 .. n_N]``.
      * subtract ``log(N * q(j))`` from EVERY column, including the positive.
      * any negative column equal to that row's own ``target_class`` has its
        logit set to ``-inf`` before softmax (accidental-hit masking).
      * cross-entropy with target index 0 (the positive is always column 0).
      * same masking (``loss_mask & target_class >= 0``) and the same
        sum/count return contract as ``exact_ce_loss``.
    """
    if not enabled:
        raise RuntimeError(
            "sampled_ce_loss is a default-OFF fallback (FABLE_PLAN.md sec. C); "
            "call with enabled=True to use it."
        )
    if logits.ndim != 2:
        raise ValueError(f"logits must be [B,K], got shape {tuple(logits.shape)}")
    if target_class.shape != loss_mask.shape:
        raise ValueError("target_class and loss_mask must share shape [B]")
    B, K = logits.shape
    if log_q.shape != (K,):
        raise ValueError(f"log_q must be shape [K]=({K},), got {tuple(log_q.shape)}")

    sel = loss_mask.to(torch.bool) & (target_class >= 0)
    n_contributing = sel.sum()

    if int(n_contributing.item()) == 0:
        zero_loss = logits.sum() * 0
        return zero_loss, n_contributing

    sel_logits = logits[sel]                       # [n, K]
    sel_target = target_class[sel].to(torch.long)   # [n]
    n = sel_logits.shape[0]

    q = torch.exp(log_q)
    negatives = torch.multinomial(
        q, n_negatives, replacement=True, generator=generator
    )  # [N] -- ONE shared set for the whole microbatch, duplicates kept

    # candidates per row = [positive, n_1 .. n_N]
    negatives_b = negatives.unsqueeze(0).expand(n, -1)               # [n, N]
    candidates = torch.cat([sel_target.unsqueeze(1), negatives_b], dim=1)  # [n, N+1]

    cand_logits = torch.gather(sel_logits, 1, candidates)            # [n, N+1]

    # subtract log(N * q(j)) from EVERY column, including the positive
    log_nq = torch.log(torch.tensor(float(n_negatives), device=logits.device)) + log_q[candidates]
    cand_logits = cand_logits - log_nq

    # accidental-hit masking: negative columns (1..N) equal to this row's own
    # target -> -inf before softmax. Column 0 (the true positive) is exempt.
    neg_cols = candidates[:, 1:]                                     # [n, N]
    hit = neg_cols == sel_target.unsqueeze(1)                        # [n, N]
    neg_logits = cand_logits[:, 1:].masked_fill(hit, float("-inf"))
    cand_logits = torch.cat([cand_logits[:, :1], neg_logits], dim=1)

    target_idx = torch.zeros(n, dtype=torch.long, device=logits.device)
    loss_sum = F.cross_entropy(cand_logits, target_idx, reduction="sum")
    return loss_sum, n_contributing
