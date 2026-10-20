"""Deliberately-corrupted-scores helper -- negative control proving int64
rank exactness matters (05 item 27) and that our fixed-point fingerprint
helper (test_eval_and_precision.py) fails closed on a changed population."""
from __future__ import annotations

import torch


def cast_to_float32_before_rank(scores: torch.Tensor) -> torch.Tensor:
    """Simulates the forbidden regression named in eval_full.py's own
    docstring: coercing int64 scores to float32 before ranking, which loses
    exactness above 2**24 and can silently manufacture ties. Used only to
    show what WOULD happen if `rank_targets` did this (it deliberately does
    not)."""
    return scores.float()
