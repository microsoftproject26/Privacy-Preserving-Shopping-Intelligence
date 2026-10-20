"""Deliberately broken schedule S' variants -- negative controls (Gate P2:
review/FABLE_GATE_P2_RESPONSE.md 'Minimum tests to add' items 1/3/4).

No module-level dependency on old (pre-S') sasrec_schedule constants: this
module must still import cleanly even though the pre-S' patience-based
stopping constants (MIN_EPOCHS/MAX_EPOCHS/PATIENCE/FLOOR_START_EPOCH) may no
longer exist.
"""
from __future__ import annotations

import sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parent.parent.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import sasrec_schedule as sched  # noqa: E402


def epoch_end_floor_late(state, epoch, metric, *, floor_start_epoch=3, min_delta=2e-4):
    """WRONG: post-floor miss counting starts one epoch later than the
    schedule's real floor_start_epoch. Negative control for 'a mutant with
    the floor starting one epoch late'."""
    return sched.epoch_end(state, epoch, metric, floor_start_epoch=floor_start_epoch + 1,
                            min_delta=min_delta)


def epoch_cap_ignores_slow_arm(now_s, deadline_s, arms, **kw):
    """WRONG: projects from the FASTEST arm (min mean_epoch_seconds) instead
    of the slower one. Negative control for 'epoch_cap must use the slower
    arm, not the average/fastest'."""
    fastest = min(arms, key=lambda a: a["mean_train_epoch_seconds"])
    return sched.epoch_cap(now_s, deadline_s, [fastest], **kw)


def epoch_cap_no_safety_margin(now_s, deadline_s, arms, **kw):
    """WRONG: drops the safety_hours margin entirely (safety_hours=0).
    Negative control showing the cap changes once the deadline safety
    margin is removed."""
    kw = dict(kw)
    kw["safety_hours"] = 0.0
    return sched.epoch_cap(now_s, deadline_s, arms, **kw)
