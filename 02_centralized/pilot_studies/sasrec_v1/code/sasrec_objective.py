"""SASRec-CE objective (owner: Opus). 03 section 6.

One exact full-catalogue cross-entropy term per frozen TRAIN decision, computed on a
single [B, K] logit tensor at the decision's readout position. No overlapping
all-position labels, no sampled negatives.

`ce_sum` returns the SUM of per-row losses and the COUNT of contributing rows; the
caller divides exactly once per optimizer update (correct under uneven microbatch
accumulation and the 128-decision epoch tail).

Label smoothing (the single registered C1 candidate) uses the full-catalogue formula
    loss_i = (1 - eps) * (-log p_i[y_i]) + eps * mean_j (-log p_i[j]),   j over all K
which is exactly torch's `F.cross_entropy(..., label_smoothing=eps)` for class-index
targets. It is not label corruption and no class is excluded.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["ce_sum", "ce_sum_reference"]


def _select(logits: Tensor, target_class: Tensor, loss_mask: Tensor):
    if logits.ndim != 2:
        raise ValueError(f"logits must be [B,K], got {tuple(logits.shape)}")
    if target_class.shape != loss_mask.shape or target_class.shape[0] != logits.shape[0]:
        raise ValueError("target_class/loss_mask must be [B] matching logits")
    sel = loss_mask.to(torch.bool) & (target_class >= 0)
    return sel


def ce_sum(logits: Tensor, target_class: Tensor, loss_mask: Tensor, *,
           label_smoothing: float = 0.0) -> tuple[Tensor, Tensor]:
    """(sum of per-row CE over rows with loss_mask & target>=0, count of such rows)."""
    if not (0.0 <= float(label_smoothing) < 1.0):
        raise ValueError("label_smoothing must be in [0, 1)")
    sel = _select(logits, target_class, loss_mask)
    n = sel.sum()
    if bool(sel.all()):
        sel_logits, sel_target = logits, target_class
    else:
        sel_logits, sel_target = logits[sel], target_class[sel]
    if sel_logits.shape[0] == 0:
        return logits.sum() * 0.0, n
    loss_sum = F.cross_entropy(sel_logits, sel_target.long(), reduction="sum",
                               label_smoothing=float(label_smoothing))
    return loss_sum, n


def ce_sum_reference(logits: Tensor, target_class: Tensor, loss_mask: Tensor, *,
                     label_smoothing: float = 0.0) -> tuple[Tensor, Tensor]:
    """Independent explicit formula (tests only): logsumexp-based, FP64."""
    sel = _select(logits, target_class, loss_mask)
    x = logits[sel].double()
    y = target_class[sel].long()
    if x.shape[0] == 0:
        return logits.sum() * 0.0, sel.sum()
    logp = x - torch.logsumexp(x, dim=1, keepdim=True)
    nll = -logp.gather(1, y.view(-1, 1)).squeeze(1)
    uniform = -logp.mean(dim=1)
    eps = float(label_smoothing)
    return ((1.0 - eps) * nll + eps * uniform).sum(), sel.sum()
