"""Exact full-catalogue cross-entropy as (sum, count) with no host sync, and exact microbatch accumulation.

`ce_sum_count` is the training objective: one [B, K] row per decision, rows with loss_mask AND target_class >= 0, the
SUM of per-row CE plus the COUNT, label smoothing 0, written without any `.item()` / `bool(tensor)` so it never forces
a device sync.

`accumulate_step` implements exact accumulation: an effective batch of n decisions is split into microbatches of
`micro` rows; each microbatch backpropagates sum_mb / n_eff, where n_eff (the effective batch's contributing count) is
known on the host BEFORE the forward, so the accumulated gradient is the gradient of (sum over the effective batch) /
n_eff: the effective batch and the loss sum / count are preserved exactly (up to FP32 summation order). With
micro = n it is the single-pass step.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

OBJECTIVE_ID = "EXACT_FULL_K_CE_SUM_OVER_LOSS_ELIGIBLE_DIVIDE_ONCE_LS0"


def ce_sum_count(logits: Tensor, target_class: Tensor, loss_mask: Tensor) -> tuple:
    if logits.ndim != 2:
        raise ValueError(f"logits must be [B, K], got {tuple(logits.shape)}")
    if target_class.shape != loss_mask.shape or target_class.shape[0] != logits.shape[0]:
        raise ValueError("target_class / loss_mask must be [B] matching logits")
    sel = loss_mask.to(torch.bool) & (target_class >= 0)
    t = torch.where(sel, target_class.to(torch.long), torch.zeros_like(target_class, dtype=torch.long))
    ce = F.cross_entropy(logits, t, reduction="none")
    return torch.where(sel, ce, torch.zeros_like(ce)).sum(), sel.sum()


def host_contributing(batch: Mapping[str, Tensor]) -> int:
    """Loss-contributing rows of a CPU batch (host arithmetic on CPU tensors: no device sync)."""
    t, m = batch["target_class"], batch["loss_mask"]
    if t.device.type != "cpu" or m.device.type != "cpu":
        raise ValueError("host_contributing needs the CPU batch (before the device move)")
    return int(np.count_nonzero((t.numpy() >= 0) & m.numpy().astype(bool)))


def split_rows(batch: Mapping[str, Tensor], start: int, stop: int) -> dict:
    return {k: (v[start:stop] if torch.is_tensor(v) and v.ndim >= 1 else v) for k, v in batch.items()}


def to_device(batch: Mapping, device: torch.device) -> dict:
    """Model inputs to the device; `lengths` stays on CPU (pack_padded_sequence wants it there)."""
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) and k != "lengths" else v)
            for k, v in batch.items()}


def accumulate_step(scores_fn: Callable[[Mapping], Tensor], batch_cpu: Mapping[str, Tensor], n_eff: int, micro: int,
                    device: torch.device) -> Tensor:
    """Forward + backward of one effective batch in microbatches; returns the (device) loss sum, grads accumulated."""
    if n_eff <= 0:
        raise ValueError("an effective batch needs >= 1 contributing decision")
    n_rows = int(batch_cpu["target_class"].shape[0])
    micro = int(micro) if micro else n_rows
    total = None
    for s in range(0, n_rows, micro):
        mb = to_device(split_rows(batch_cpu, s, min(n_rows, s + micro)), device)
        ls, _ = ce_sum_count(scores_fn(mb), mb["target_class"], mb["loss_mask"])
        (ls / n_eff).backward()
        total = ls.detach() if total is None else total + ls.detach()
    return total


__all__ = ["OBJECTIVE_ID", "accumulate_step", "ce_sum_count", "host_contributing", "split_rows", "to_device"]
