"""The exposure-based LR schedule, the planned LR table and the stopping rule.

  * s(EFE) has the stated shape: warmup to 0.02, peak to 2.0, decay to 0.1 by 3.0, low 0.1 to 5.5, cooldown to 0.01
    at 6.0; the continuation is 0.1 on (6.0, 7.5] then linear to 0.01 at 8.0; nothing is defined past the cap;
  * low LR before the stop: every EFE in [3.0, 6.0] runs at <= 0.1 x peak, and the only main stop point (6.0) comes
    after it; the LR of an update uses the EFE at the end of the update (the first update is not at LR 0);
  * STILL_IMPROVING iff the strictly-best (> +0.0002) is at 5.5 or 6.0; a +0.0001 gain at 6.0 is not an improvement;
  * the continuation stops on the 2nd consecutive non-improving evaluation (not the 1st, not the 3rd) or at 8.0;
  * the planned LR table: min LR > 0, last update at 0.01 x peak, 6N exposures, hashed;
  * central and federated schedules agree bit for bit, with the same 1e-12 endpoint tolerance; S1 has its knots.
Negative controls: a constant-LR schedule; a patience-1 stop rule.
"""
from __future__ import annotations

import numpy as np
import pytest

from ppsi.central import schedule as S
from ppsi.central.schedule import (
    StopTracker,
    eval_marks,
    lr_for_update,
    mark_exposures,
    planned_lr_table,
    s_continuation,
    s_main,
    s_of,
)
from ppsi.fedsim import schedule as fls

MAIN_MARKS = eval_marks("MAIN")


def nc(reason: str):
    return pytest.mark.xfail(strict=True, raises=AssertionError, reason="NEGATIVE CONTROL: " + reason)


def assert_raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc:
        return
    raise AssertionError(f"{getattr(fn, '__name__', fn)} did not raise {exc}")


# ------------------------------------------------------------------------------------------------ the schedule
def test_schedule_shape_and_refusals():
    approx = pytest.approx
    assert s_main(0.0) == 0.0 and s_main(0.01) == approx(0.5) and s_main(0.02) == 1.0
    assert s_main(1.0) == 1.0 and s_main(2.0) == 1.0
    assert s_main(2.5) == approx(0.55) and s_main(3.0) == approx(0.1)
    assert s_main(4.25) == approx(0.1) and s_main(5.5) == approx(0.1)
    assert s_main(5.75) == approx(0.055) and s_main(6.0) == approx(0.01)
    assert s_continuation(6.01) == approx(0.1) and s_continuation(7.5) == approx(0.1)
    assert s_continuation(7.75) == approx(0.055) and s_continuation(8.0) == approx(0.01)
    for bad in (lambda: s_main(6.01), lambda: s_main(-0.1), lambda: s_continuation(6.0),
                lambda: s_continuation(8.01), lambda: s_of(2.01, "CALIBRATION"), lambda: s_of(1.0, "LO")):
        assert_raises(S.ScheduleError, bad)
    assert lr_for_update(1e-3, 0, 256, 3_906_929, "MAIN") > 0.0             # end-of-update EFE: never LR 0
    assert lr_for_update(1e-3, 6 * 1000 - 8, 8, 1000, "MAIN") == approx(1e-5)  # last main update at 0.01 x peak
    assert MAIN_MARKS == [0.5 * i for i in range(1, 13)]
    assert eval_marks("CALIBRATION") == [0.5, 1.0, 1.5, 2.0] and eval_marks("CONTINUATION") == [6.5, 7.0, 7.5, 8.0]
    assert eval_marks("TUNING") == [2.0, 4.0, 6.0] and S.snapshot_marks("TUNING") == [5.0, 5.5, 6.0]
    assert mark_exposures(0.5, 201) == 101 and mark_exposures(6.0, 201) == 1206


def _check_low_lr_before_stop(s_fn):
    grid = [3.0 + i * 0.01 for i in range(301)]
    assert all(s_fn(e) <= S.LOW + 1e-12 for e in grid), "the LR must be in the low phase before the 6.0 stop"
    assert all(abs(s_fn(e) - S.LOW) < 1e-12 for e in grid if e <= S.LOW_END), "a real low-LR interval [3.0, 5.5]"
    assert s_fn(6.0) == pytest.approx(S.FLOOR) and max(s_fn(e) for e in (0.5, 1.0, 1.5, 2.0)) == 1.0


def test_low_lr_before_stop():
    _check_low_lr_before_stop(s_main)


def test_exposure_point_is_required_and_end_is_the_default():
    assert S.LR_EXPOSURE_POINT == "end"
    assert lr_for_update(1e-3, 0, 256, 3_906_929, "MAIN", point="end") > 0.0
    assert lr_for_update(1e-3, 0, 256, 3_906_929, "MAIN", point="start") == 0.0
    assert_raises(S.ScheduleError, lr_for_update, 1e-3, 0, 8, 1000, "MAIN", point="END")
    assert_raises(S.ScheduleError, lr_for_update, 1e-3, 0, 0, 1000, "MAIN")


def test_planned_lr_table():
    for N, B in ((200, 32), (1000, 256), (40_180_608 // 64, 256)):
        t = planned_lr_table(1e-3, N, B, "MAIN", "end")
        assert t["min_lr"] > 0.0 and t["final_exposures"] == 6 * N and t["last_lr"] == pytest.approx(1e-5, abs=1e-15)
        assert t["n_updates"] == 6 * -(-N // B) and len(t["sha256"]) == 64
    st = planned_lr_table(1e-3, 200, 32, "MAIN", "start")
    assert st["min_lr"] == 0.0 and st["last_lr"] > 1e-5
    assert planned_lr_table(1e-3, 200, 32, "MAIN", "end")["sha256"] != st["sha256"]
    s1 = planned_lr_table(1e-3, 200, 32, "TUNING", "end", schedule="S1")
    assert s1["sha256"] != planned_lr_table(1e-3, 200, 32, "TUNING", "end")["sha256"]
    assert s1["last_lr"] == pytest.approx(1e-5, abs=1e-15)


def test_s1_schedule():
    assert S.SCHEDULES["S1"] == ((0.0, 0.0), (0.02, 1.0), (4.2, 1.0), (6.0, 0.01))
    assert s_main(3.0, "S1") == 1.0 and s_main(4.2, "S1") == 1.0 and abs(s_main(6.0, "S1") - 0.01) <= 1e-12
    assert_raises(S.ScheduleError, s_main, 1.0, "S2")
    assert_raises(S.ScheduleError, s_of, 7.0, "CONTINUATION", "S1")          # the continuation exists for S0 only


def test_matches_the_federated_schedule_bit_for_bit():
    grid = [k[0] for k in S.KNOTS] + list(np.linspace(0.0, 6.0, 10_000))
    for e in grid:
        assert s_main(e) == fls.s_of_efe(e), f"s differs at {e!r}"
    for e in list(np.linspace(6.0, 8.0, 10_001))[1:]:
        assert s_continuation(e) == fls.s_of_efe(e, continuation=True), f"continuation differs at {e!r}"
    assert abs(s_main(6.0) - 0.01) <= 1e-12 and abs(fls.s_of_efe(6.0) - 0.01) <= 1e-12


def test_endpoint_tolerance_1e12_in_both():
    assert S.END_TOL == 1e-12 == fls.EDGE_TOL
    assert s_main(6.0 + 1e-12) == s_main(6.0) == s_main(6.0 + 5e-13)
    assert_raises(S.ScheduleError, s_main, 6.0 + 2e-12)
    assert abs(s_main(6.0) - 0.01) <= 1e-12 and s_main(6.0) == 0.009999999999999995   # never exact 0.01
    for e in (6.0 + 1e-13, 6.0 + 1e-12):
        assert float.hex(s_main(e)) == float.hex(fls.s_of_efe(e))


# ------------------------------------------------------------------------------------------------ the tracker
def _track(values):
    t = StopTracker()
    for m, v in zip(MAIN_MARKS, values, strict=False):
        t.update(m, v, "MAIN")
    return t


def test_still_improving_rule_and_tolerance():
    base = [0.1 + 0.01 * i for i in range(12)]                       # strictly improving to 6.0
    assert _track(base).still_improving() and _track(base).main_best_mark == 6.0
    at55 = [*base[:11], base[10]]                                     # best at 5.5, flat at 6.0
    assert _track(at55).still_improving() and _track(at55).main_best_mark == 5.5
    at50 = [*base[:10], base[9], base[9]]                             # best at 5.0: two non-improving evaluations
    assert not _track(at50).still_improving() and _track(at50).main_best_mark == 5.0
    tiny = [*base[:10], base[9], base[9] + 0.0001]                    # +0.0001 <= tolerance: not an improvement
    assert not _track(tiny).still_improving()
    over = [*base[:10], base[9], base[9] + 0.00021]                   # > tolerance: improvement at 6.0
    assert _track(over).still_improving() and _track(over).main_best_mark == 6.0
    assert_raises(S.ScheduleError, _track(base[:11]).still_improving)  # decided only after the 6.0 evaluation


def test_tracker_refuses_non_increasing_marks_and_round_trips_its_state():
    t = _track([0.1, 0.2, 0.15])
    assert_raises(S.ScheduleError, t.update, 1.0, 0.3, "MAIN")
    back = StopTracker.from_state(t.state())
    assert back.state() == t.state() and back.best_mark == 1.0


def _stop_mark(improved_flags, should_stop=None):
    t = _track([0.1 + 0.01 * i for i in range(12)])
    v = t.best_value
    for m, imp in zip(eval_marks("CONTINUATION"), improved_flags, strict=False):
        v = v + 0.001 if imp else v
        t.update(m, v, "CONTINUATION")
        if (should_stop or StopTracker.continuation_should_stop)(t):
            return m
    return None


def _check_patience(should_stop=None):
    assert _stop_mark([False, False], should_stop) == 7.0, "stop on the 2nd consecutive non-improving evaluation"
    assert _stop_mark([True, False, False], should_stop) == 7.5
    assert _stop_mark([False, True, False, False], should_stop) == 8.0      # the ceiling comes first
    assert _stop_mark([True, True, True, True], should_stop) == 8.0


def test_patience_is_not_off_by_one():
    _check_patience()


# ------------------------------------------------------------------------------------------------ negative controls
@nc("a constant-LR schedule never enters a low-LR phase before the stop")
def test_nc_constant_schedule():
    _check_low_lr_before_stop(lambda e: 1.0)


@nc("a patience-1 continuation stop rule is off by one")
def test_nc_patience_one():
    def patience1(t):
        cont = [h for h in t.history if h["phase"] == "CONTINUATION"]
        return bool(cont) and (not cont[-1]["improved"] or cont[-1]["mark"] >= 8.0)
    _check_patience(patience1)
