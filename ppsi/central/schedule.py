"""The common exposure-based learning-rate schedule and the stopping rule of central training.

Time is measured in epochs of exposures (EFE = cumulative loss-contributing exposures / N). Schedule s(EFE), the
fraction of the peak LR, is piecewise linear through the knots (0, 0), (0.02, 1.0), (2.0, 1.0), (3.0, 0.1), (5.5, 0.1),
(6.0, 0.01):
    [0, 0.02] warmup 0 -> 1; [0.02, 2.0] peak; [2.0, 3.0] decay 1 -> 0.1; [3.0, 5.5] low 0.1; [5.5, 6.0] cooldown
    0.1 -> 0.01 (6.0 = the controlled-budget endpoint).
Continuation, only for a STILL_IMPROVING run with a budget recorded in advance:
    (6.0, 7.5] 0.1; (7.5, 8.0] linear 0.1 -> 0.01 (8.0 = the practical ceiling; never a target).
The interpolation is y0 + (y1 - y0) * (e - x0) / (x1 - x0) on the knot segment containing e (first match), the same
arithmetic as ppsi.fedsim.schedule.s_of_efe, so central and federated runs agree bit for bit. The 6.0 endpoint
carries a 1e-12 tolerance in both: s_main accepts EFE up to 6.0 + 1e-12 and reads the 6.0 value there;
s(6.0) = 0.009999999999999995, so every "0.01 x peak" check uses a 1e-12 tolerance, never exact equality.

LR discretisation: the exposure point is "end": the LR of an update is peak * s(EFE at the END of the exposures it
consumes); a central update uses peak * s((a + n) / N). `exposure_point` is a REQUIRED part of every run
configuration (LR_EXPOSURE_POINT is the default value, never a silent fallback); "start" (the EFE before the update)
is kept as a tested alternative. planned_lr_table gives the whole phase's LR sequence before training (hashed; its
minimum LR > 0; the final main update ends at 6N exposures with LR 0.01 * peak).

Stopping rule:
  * evaluation every 0.5 EFE; no stop before 6.0 (no early stop of any kind);
  * strictly-best: a value is a new best iff value > running_best + 0.0002 (the first evaluation always is);
  * PRACTICAL_BEST = the strictly-best evaluation in [0.5, 6.0]; CONTROLLED_BUDGET = the 6.0 evaluation;
  * STILL_IMPROVING at 6.0 iff the strictly-best is at 5.5 or 6.0 (patience 2);
  * the continuation stops after 2 consecutive non-improving evaluations, or at 8.0, whichever comes first; its rows
    are PRACTICAL_BEST_EXTENDED.

An alternative schedule S1 (warmup-stable-decay, linear) on the same 6-EFE clock,
    S1 = (0, 0), (0.02, 1.0), (4.2, 1.0), (6.0, 0.01),
is selected by name ("S0" | "S1"); S0 stays the default of every function. The same interpolation and endpoint
tolerance apply. The continuation exists for S0 only.
Phase TUNING: the main clock [0, 6.0], evaluations at 2 / 4 / 6 EFE only (TUNING_MARKS) and weights-only snapshots at
5.0 / 5.5 / 6.0 (SNAPSHOT_MARKS, for checkpoint averaging).
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np

WARMUP_END = 0.02
PEAK_END = 2.0
DECAY_END = 3.0
LOW_END = 5.5
MAIN_END = 6.0
CONT_FLAT_END = 7.5
CEILING = 8.0
LOW = 0.1
FLOOR = 0.01
EVAL_EVERY = 0.5
TOLERANCE = 0.0002
PATIENCE = 2
CALIBRATION_END = 2.0
KNOTS = ((0.0, 0.0), (0.02, 1.0), (2.0, 1.0), (3.0, 0.1), (5.5, 0.1), (6.0, 0.01))
END_TOL = 1e-12                    # the 6.0 endpoint tolerance (as ppsi.fedsim.schedule)
CONT_KNOTS = ((7.5, 0.1), (8.0, 0.01))
S1_KNOTS = ((0.0, 0.0), (0.02, 1.0), (4.2, 1.0), (6.0, 0.01))     # warmup-stable-decay, linear
SCHEDULES = {"S0": KNOTS, "S1": S1_KNOTS}
DEFAULT_SCHEDULE = "S0"
TUNING_MARKS = (2.0, 4.0, 6.0)          # evaluations at 2 / 4 / 6 EFE; the endpoint read at 6.0
SNAPSHOT_MARKS = (5.0, 5.5, 6.0)        # weights-only checkpoints for checkpoint averaging

PHASES = ("MAIN", "CALIBRATION", "CONTINUATION", "TUNING")
LR_EXPOSURE_POINTS = ("end", "start")
LR_EXPOSURE_POINT = "end"          # the default (a run configuration must still state it explicitly)


class ScheduleError(ValueError):
    pass


def _interp(e: float, knots) -> float:
    for (x0, y0), (x1, y1) in pairwise(knots):
        if x0 <= e <= x1:
            return y0 if x1 == x0 else y0 + (y1 - y0) * (e - x0) / (x1 - x0)
    raise ScheduleError(f"EFE {e} outside the knot range")


def check_schedule(schedule: str | None) -> str:
    if schedule not in SCHEDULES:
        raise ScheduleError(f"schedule must be one of {tuple(SCHEDULES)}, got {schedule!r}")
    return schedule


def s_main(efe: float, schedule: str = DEFAULT_SCHEDULE) -> float:
    """Fraction of the peak LR on [0, 6.0], with the 1e-12 endpoint tolerance (EFE in (6.0, 6.0 + 1e-12] reads the
    6.0 value). Anything else is refused (the cap is a ceiling, not a target). `schedule`: S0 (default) or S1."""
    e = float(efe)
    if not math.isfinite(e) or e < 0.0 or e > MAIN_END + END_TOL:
        raise ScheduleError(f"main schedule defined on [0, {MAIN_END}] (+{END_TOL:g}), got EFE {e}")
    return _interp(min(e, MAIN_END), SCHEDULES[check_schedule(schedule)])


def s_continuation(efe: float) -> float:
    """Fraction of the peak LR on (6.0, 8.0] (0.1 on (6.0, 7.5], then linear to 0.01 at 8.0)."""
    e = float(efe)
    if not math.isfinite(e) or e <= MAIN_END or e > CEILING:
        raise ScheduleError(f"continuation schedule defined on ({MAIN_END}, {CEILING}], got EFE {e}")
    if e <= CONT_FLAT_END:
        return LOW
    return _interp(e, CONT_KNOTS)


def s_of(efe: float, phase: str, schedule: str = DEFAULT_SCHEDULE) -> float:
    if phase in ("MAIN", "CALIBRATION", "TUNING"):
        if phase == "CALIBRATION" and efe > CALIBRATION_END:
            raise ScheduleError(f"calibration runs to {CALIBRATION_END} EFE only")
        return s_main(efe, schedule)
    if phase == "CONTINUATION":
        if check_schedule(schedule) != DEFAULT_SCHEDULE:
            raise ScheduleError("the continuation exists for the S0 schedule only")
        if efe == MAIN_END:                   # only reachable by the "start" point: the 6.0 value, as fedsim
            return s_main(efe)
        return s_continuation(efe)
    raise ScheduleError(f"unknown phase {phase!r}")


def check_point(point: str | None) -> str:
    if point not in LR_EXPOSURE_POINTS:
        raise ScheduleError(f"exposure_point must be one of {LR_EXPOSURE_POINTS} (required key), got {point!r}")
    return point


def lr_for_update(peak_lr: float, exposures_before: int, n_contributing: int, N: int, phase: str,
                  point: str = LR_EXPOSURE_POINT, schedule: str = DEFAULT_SCHEDULE) -> float:
    """LR of the update that consumes exposures (exposures_before, exposures_before + n_contributing]."""
    if N <= 0 or n_contributing <= 0:
        raise ScheduleError("N and the update's contributing count must be positive")
    at = exposures_before + n_contributing if check_point(point) == "end" else exposures_before
    return float(peak_lr) * s_of(at / N, phase, schedule)


def planned_lr_table(peak_lr: float, N: int, B: int, phase: str, point: str, *, start_exposures: int = 0,
                     end_efe: float | None = None, schedule: str = DEFAULT_SCHEDULE) -> dict:
    """The LR of every update of a phase, planned before training from (N, B, tail per pass, point).

    Batches are consecutive B-sized slices of each pass with the tail at its own size, exactly as the trainer takes
    them, so the table is the trainer's per-update LR sequence. Returns the table plus its sha256 (float64 LE bytes)."""
    end_efe = phase_end(phase) if end_efe is None else float(end_efe)
    end = mark_exposures(end_efe, N)
    lrs, exps, e = [], [], int(start_exposures)
    while e < end:
        pos = e % N
        n = min(B, N - pos)
        lrs.append(lr_for_update(peak_lr, e, n, N, phase, point, schedule))
        e += n
        exps.append(e)
    arr = np.asarray(lrs, dtype="<f8")
    return {"lr": lrs, "exposures_after": exps, "n_updates": len(lrs), "min_lr": float(arr.min()) if lrs else None,
            "last_lr": lrs[-1] if lrs else None, "final_exposures": e,
            "sha256": hashlib.sha256(arr.tobytes()).hexdigest(), "point": point, "phase": phase,
            "schedule": schedule}


def eval_marks(phase: str) -> list[float]:
    """The evaluation marks of a phase (multiples of 0.5 EFE)."""
    if phase == "MAIN":
        lo, hi = EVAL_EVERY, MAIN_END
    elif phase == "CALIBRATION":
        lo, hi = EVAL_EVERY, CALIBRATION_END
    elif phase == "CONTINUATION":
        lo, hi = MAIN_END + EVAL_EVERY, CEILING
    elif phase == "TUNING":
        return list(TUNING_MARKS)
    else:
        raise ScheduleError(f"unknown phase {phase!r}")
    n = round((hi - lo) / EVAL_EVERY) + 1
    return [round(lo + i * EVAL_EVERY, 6) for i in range(n)]


def snapshot_marks(phase: str) -> list[float]:
    """Weights-only snapshot marks of a phase (TUNING: 5.0 / 5.5 / 6.0; none elsewhere)."""
    return list(SNAPSHOT_MARKS) if phase == "TUNING" else []


def phase_end(phase: str) -> float:
    return {"MAIN": MAIN_END, "CALIBRATION": CALIBRATION_END, "CONTINUATION": CEILING, "TUNING": MAIN_END}[phase]


def mark_exposures(mark: float, N: int) -> int:
    """Exposures at which a mark is reached: the first update boundary with exposures >= ceil(mark * N)."""
    num = round(mark * 2)                       # marks are multiples of 0.5
    return -(-num * N // 2)


@dataclass
class StopTracker:
    """Running strictly-best tracker over one run's evaluation marks (main and continuation share the running best)."""
    tolerance: float = TOLERANCE
    history: list[dict] = field(default_factory=list)
    best_mark: float | None = None
    best_value: float | None = None
    main_best_mark: float | None = None
    main_best_value: float | None = None

    def update(self, mark: float, value: float, phase: str) -> bool:
        v = float(value)
        if self.history and mark <= self.history[-1]["mark"]:
            raise ScheduleError(f"marks must increase: {mark} after {self.history[-1]['mark']}")
        improved = self.best_value is None or v > self.best_value + self.tolerance
        if improved:
            self.best_mark, self.best_value = float(mark), v
            if phase in ("MAIN", "CALIBRATION"):
                self.main_best_mark, self.main_best_value = float(mark), v
        self.history.append({"mark": float(mark), "value": v, "improved": bool(improved), "phase": phase,
                             "best_mark_after": self.best_mark, "best_value_after": self.best_value})
        return improved

    def still_improving(self) -> bool:
        """At 6.0 (after the 6.0 evaluation): the strictly-best is at 5.5 or 6.0 (patience 2)."""
        marks = [h["mark"] for h in self.history if h["phase"] == "MAIN"]
        if not marks or abs(marks[-1] - MAIN_END) > 1e-9:
            raise ScheduleError("STILL_IMPROVING is decided only after the 6.0 evaluation")
        return self.main_best_mark is not None and self.main_best_mark >= LOW_END - 1e-9

    def continuation_should_stop(self) -> bool:
        """After a continuation evaluation: stop on the 2nd consecutive non-improving evaluation, or at the ceiling."""
        cont = [h for h in self.history if h["phase"] == "CONTINUATION"]
        if not cont:
            return False
        if cont[-1]["mark"] >= CEILING - 1e-9:
            return True
        run = 0
        for h in reversed(cont):
            if h["improved"]:
                break
            run += 1
        return run >= PATIENCE

    def state(self) -> dict:
        return {"tolerance": self.tolerance, "history": [dict(h) for h in self.history], "best_mark": self.best_mark,
                "best_value": self.best_value, "main_best_mark": self.main_best_mark,
                "main_best_value": self.main_best_value}

    @classmethod
    def from_state(cls, st: dict) -> StopTracker:
        t = cls(tolerance=float(st["tolerance"]))
        t.history = [dict(h) for h in st["history"]]
        t.best_mark, t.best_value = st["best_mark"], st["best_value"]
        t.main_best_mark, t.main_best_value = st["main_best_mark"], st["main_best_value"]
        return t


__all__ = [
    "CALIBRATION_END",
    "CEILING",
    "CONT_FLAT_END",
    "DECAY_END",
    "DEFAULT_SCHEDULE",
    "END_TOL",
    "EVAL_EVERY",
    "FLOOR",
    "KNOTS",
    "LOW",
    "LOW_END",
    "LR_EXPOSURE_POINT",
    "LR_EXPOSURE_POINTS",
    "MAIN_END",
    "PATIENCE",
    "PEAK_END",
    "PHASES",
    "S1_KNOTS",
    "SCHEDULES",
    "SNAPSHOT_MARKS",
    "TOLERANCE",
    "TUNING_MARKS",
    "WARMUP_END",
    "ScheduleError",
    "StopTracker",
    "check_point",
    "check_schedule",
    "eval_marks",
    "lr_for_update",
    "mark_exposures",
    "phase_end",
    "planned_lr_table",
    "s_continuation",
    "s_main",
    "s_of",
    "snapshot_marks",
]
