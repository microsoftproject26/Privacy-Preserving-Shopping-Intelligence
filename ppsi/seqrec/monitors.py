"""Gradient and common-mode drift monitors.

Gradient monitor (`grad_report`, after backward, before clipping):
  global L2 norm over the trainable parameters (aliases once: named_parameters() de-duplicates), per-parameter norms,
  parameters with no gradient / an exactly-zero gradient, non-finite gradients. Verdict:
    NONFINITE  any non-finite gradient element;
    ZERO       global norm exactly 0 (nothing would move: a dead objective or a detached graph);
    EXPLODING  global norm > explode_at (default 1e4; with clipping at 1.0 this is a runaway, not a clip event);
    OK         otherwise.
  `zero_grad_columns` names the input columns of a first-layer weight whose gradient is exactly zero, e.g. flag
  columns that never fire in the training data (their weights stay at init).

Common-mode drift monitor (`table_stats` / `common_mode_report`, any time):
  for an output table W [K, d] and bias b [K]: ||mean_rows(W)||, the median row norm, their ratio, mean(b), std(b).
  Under dense-softmax Adam the output rows of a SASRec head drift in their common mode; recenter_output_() after
  every step removes it, so the recentered ratio stays ~0. `drift_verdict` flags DRIFT when the ratio exceeds
  `max_ratio`. For the tied GRU head the same statistics are reported (descriptive only: the GRU is not recentered).
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import torch
from torch import Tensor

VERDICTS = ("OK", "NONFINITE", "ZERO", "EXPLODING")


def grad_report(named_parameters: Iterable, *, explode_at: float = 1e4) -> dict:
    total_sq = 0.0
    per, none, zero, nonfinite = {}, [], [], []
    for name, p in named_parameters:
        if not p.requires_grad:
            continue
        g = p.grad
        if g is None:
            none.append(name)
            continue
        if not bool(torch.isfinite(g).all()):
            nonfinite.append(name)
            continue
        n = float(g.detach().double().norm())
        per[name] = n
        total_sq += n * n
        if n == 0.0:
            zero.append(name)
    gn = math.sqrt(total_sq)
    if nonfinite:
        verdict = "NONFINITE"
    elif gn == 0.0:
        verdict = "ZERO"
    elif gn > explode_at:
        verdict = "EXPLODING"
    else:
        verdict = "OK"
    return {"verdict": verdict, "global_norm": gn, "n_params": len(per) + len(none) + len(nonfinite),
            "no_grad": none, "zero_grad": zero, "nonfinite": nonfinite, "per_param_norm": per}


def zero_grad_columns(weight_grad: Tensor, names: Sequence[str], offset: int = 0) -> list:
    """Names of the input columns [offset, offset + len(names)) of a [out, in] weight gradient that are exactly 0."""
    cols = weight_grad[:, offset:offset + len(names)]
    dead = (cols == 0).all(dim=0)
    return [n for n, d in zip(names, dead.tolist(), strict=True) if d]


@torch.no_grad()
def table_stats(W: Tensor, b: Tensor | None = None) -> dict:
    W = W.detach().double()
    cm = float(W.mean(0).norm())
    med = float(W.norm(dim=1).median())
    out = {"common_mode_norm": cm, "median_row_norm": med, "ratio": cm / med if med > 0 else float("inf")}
    if b is not None:
        bd = b.detach().double()
        out.update({"b_mean": float(bd.mean()), "b_std": float(bd.std())})
    return out


def common_mode_report(adapter) -> dict:
    """Output-head statistics (the head the family actually scores with) and, for SASRec, the input table too."""
    rep = {"head": table_stats(adapter.head_weight(), adapter.head_bias())}
    m = adapter.module
    if getattr(adapter, "family", "") == "SASREC":
        rep["input_table"] = table_stats(m.item_embed.weight[3:3 + m.K])
    return rep


def drift_verdict(stats: dict, *, max_ratio: float = 1e-4, max_abs_b_mean: float = 1e-5) -> str:
    """DRIFT if the head common mode (relative to the median row norm) or the bias mean exceeds the bound."""
    h = stats.get("head", stats)
    if h["ratio"] > max_ratio or abs(h.get("b_mean", 0.0)) > max_abs_b_mean:
        return "DRIFT"
    return "OK"


__all__ = ["VERDICTS", "common_mode_report", "drift_verdict", "grad_report", "table_stats", "zero_grad_columns"]
