"""The s(EFE) learning-rate shape over effective full-data epochs (EFE) and its FL / LO hooks.

s(e), the fraction of the peak LR at e EFE:
    [0, 0.02]    linear 0 -> 1          (warmup)
    (0.02, 2.0]  1                      (peak)
    (2.0, 3.0]   linear 1 -> 0.1        (decay)
    (3.0, 5.5]   0.1                    (low phase)
    (5.5, 6.0]   linear 0.1 -> 0.01     (cooldown; 6.0 = CONTROLLED_BUDGET endpoint)
  continuation (only when a run is still improving at 6.0, `continuation=True`): (6.0, 7.5] 0.1, (7.5, 8.0] linear
    0.1 -> 0.01; exactly 6.0 in the continuation phase (reachable only by the "start" point) is the main 6.0 value.
  The arithmetic mirrors the central trainer's schedule expression for expression — the same knot interpolation
  y0 + (y1 - y0)(e - x0)/(x1 - x0) and the same domains: main [0, 6.0] with a 1e-12 endpoint tolerance (an EFE in
  (6.0, 6.0 + 1e-12] is clamped to 6.0 and reads the 6.0 value; 6.0 + 2e-12 is refused), continuation (6.0, 8.0]
  with no tolerance above 8.0, so central and federated runs read bit-identical LR factors.
  Consequence of the interpolation: s(6.0) = 0.009999999999999995 (0.01 to 5e-16), so "0.01 x peak" pins hold to
  1e-12 relative, not as exact float equality.

Exposure point: exposure_point = "end" for every unit of work in the study.
  * FA / FP / PF: round r uses lr_peak_local * s((E_r + n_r_planned) / N), constant over the round's local steps,
    with the planned n_consumed enforced equal to the actual one (runtime.py); the PF personal LR = the shared peak
    x the same factor.
  * LO: step t of T uses lr_peak * s(6 * (t + 1) / T), t = 0..T-1 (disclosure: tiny LO clients skip the warmup
    segment; their first step is at s(6/T)).
  The point is an EXPLICIT argument everywhere (no default is relied on); "start" (s at the unit's first
  exposure: FL round 0 and each LO client's first step at LR 0) remains available only as an explicitly named option
  for regression tests.
"""
from __future__ import annotations

import itertools
import math
from collections.abc import Callable

WARMUP_END, PEAK_END, DECAY_END, LOW_END, MAIN_END = 0.02, 2.0, 3.0, 5.5, 6.0
CONT_FLAT_END, CEILING, LOW, FLOOR = 7.5, 8.0, 0.1, 0.01
EDGE_TOL = 1e-12                                           # endpoint tolerance (as the central schedule); main end only
KNOTS = ((0.0, 0.0), (0.02, 1.0), (2.0, 1.0), (3.0, 0.1), (5.5, 0.1), (6.0, 0.01))
CONT_KNOTS = ((7.5, 0.1), (8.0, 0.01))
ENDPOINT_EFE = MAIN_END
CONTINUATION_CEILING_EFE = CEILING
EXPOSURE_POINTS = ("start", "end")
LOCKED_EXPOSURE_POINT = "end"                              # the study setting (a reference, NOT a default)


class ScheduleError(ValueError):
    pass


def check_exposure_point(exposure_point) -> str:
    if exposure_point is None:
        raise ScheduleError("exposure_point must be given explicitly; the study value is 'end'")
    if exposure_point not in EXPOSURE_POINTS:
        raise ScheduleError(f"exposure_point must be one of {EXPOSURE_POINTS}, got {exposure_point!r}")
    return exposure_point


def _interp(e: float, knots) -> float:
    for (x0, y0), (x1, y1) in itertools.pairwise(knots):
        if x0 <= e <= x1:
            return y0 if x1 == x0 else y0 + (y1 - y0) * (e - x0) / (x1 - x0)
    raise ScheduleError(f"EFE {e} outside the knot range")


def _s_main(e: float) -> float:
    if not math.isfinite(e) or e < 0.0 or e > MAIN_END + EDGE_TOL:
        raise ScheduleError(f"main schedule defined on [0, {MAIN_END}] (+{EDGE_TOL:g}), got EFE {e}; use "
                            "continuation=True beyond it")
    return _interp(min(e, MAIN_END), KNOTS)                   # inside the tolerance: clamped to the 6.0 value


def _s_continuation(e: float) -> float:
    if not math.isfinite(e) or e <= MAIN_END or e > CEILING:  # no tolerance at 8.0
        raise ScheduleError(f"continuation schedule defined on ({MAIN_END}, {CEILING}], got EFE {e}")
    if e <= CONT_FLAT_END:
        return LOW
    return _interp(e, CONT_KNOTS)


def s_of_efe(e: float, *, continuation: bool = False) -> float:
    """Main phase (default) or continuation phase fraction of the peak LR."""
    e = float(e)
    if continuation:
        return _s_main(e) if e == MAIN_END else _s_continuation(e)
    return _s_main(e)


def round_efe_point(efe_at_round_start: float, efe_at_round_end: float | None, exposure_point: str) -> float:
    """The EFE at which a round's LR is read, per the (explicit) exposure point."""
    if check_exposure_point(exposure_point) == "start":
        return float(efe_at_round_start)
    if efe_at_round_end is None:
        raise ScheduleError("exposure_point='end' needs the round's end EFE (its planned n_consumed)")
    if efe_at_round_end < efe_at_round_start:
        raise ScheduleError("round end EFE before its start")
    return float(efe_at_round_end)


def round_lr(lr_peak_local: float, efe_at_round_start: float, *, exposure_point: str,
             efe_at_round_end: float | None = None, continuation: bool = False) -> float:
    """FA / FP / PF local LR for a round: lr_peak_local * s(EFE at the round's end (study) or start (regression))."""
    e = round_efe_point(efe_at_round_start, efe_at_round_end, exposure_point)
    return float(lr_peak_local) * s_of_efe(e, continuation=continuation)


def lo_lr_schedule(lr_peak: float, passes: int = 6, *, exposure_point: str) -> Callable[[int, int], float]:
    """LO: the same shape over the client's own passes; step t of T uses lr_peak * s(6 * (t + d) / T), d = 1 for the
    study's "end", 0 for the regression-only "start"."""
    if passes < 1:
        raise ScheduleError("passes must be >= 1")
    d = 1 if check_exposure_point(exposure_point) == "end" else 0

    def lr(step: int, total_steps: int) -> float:
        if total_steps < 1 or not 0 <= step < total_steps:
            raise ScheduleError(f"LO step {step} outside [0, {total_steps})")
        return float(lr_peak) * s_of_efe(ENDPOINT_EFE * (step + d) / total_steps)
    return lr
