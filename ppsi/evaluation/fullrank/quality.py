"""Retained quality against a matched central row.

retained_percent = 100 M / C; relative_loss_percent = 100 (C - M) / C; delta = M - C, on the SAME population, metric,
aggregation, denominator and seed pairing. No clamping: retained may exceed 100 and loss may be negative.
C = 0 gives UNDEFINED(ZERO_DENOMINATOR), never a fabricated 0; the delta stays defined. A negative or non-finite value
is impossible for these metrics and is refused.
"""
from __future__ import annotations

import math

from .values import ZERO_DENOMINATOR, Undefined


def _check(x, what: str) -> float:
    if isinstance(x, Undefined):
        raise ValueError(f"{what} is UNDEFINED ({x.reason}); retained quality needs defined values")  # noqa: TRY004
    f = float(x)
    if not math.isfinite(f) or f < 0:
        raise ValueError(f"{what} = {f!r}: metric values are finite and >= 0")
    return f


def retained_percent(M, C):
    m, c = _check(M, "M"), _check(C, "C")
    return Undefined(ZERO_DENOMINATOR) if c == 0.0 else 100.0 * m / c


def relative_loss_percent(M, C):
    m, c = _check(M, "M"), _check(C, "C")
    return Undefined(ZERO_DENOMINATOR) if c == 0.0 else 100.0 * (c - m) / c


def delta(M, C) -> float:
    return _check(M, "M") - _check(C, "C")
