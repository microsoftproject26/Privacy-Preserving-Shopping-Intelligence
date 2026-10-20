"""T21 — the schedule enters low LR before the stop; patience is not off-by-one; the cap is not a target; plus the
STILL_IMPROVING logic and the continuation rules (G0 D6; G0B C3; EVALUATION_CONTRACT_V4 E12.1).

  * s(EFE) has the registered shape: warmup to 0.02, peak to 2.0, decay to 0.1 by 3.0, low 0.1 to 5.5, cooldown to 0.01
    at 6.0; the continuation is 0.1 on (6.0, 7.5] then linear to 0.01 at 8.0; nothing is defined past the cap;
  * low LR before the stop: every EFE in [3.0, 6.0] runs at <= 0.1 x peak, and the only main stop point (6.0) comes
    after it; the LR of an update uses the EFE at the end of the update (the first update is not at LR 0);
  * no early stop: a run whose metric peaks at 1.0 EFE and then falls still trains and evaluates to 6.0;
  * STILL_IMPROVING iff the strictly-best (> +0.0002) is at 5.5 or 6.0; a +0.0001 gain at 6.0 is not an improvement;
  * the continuation stops on the 2nd consecutive non-improving evaluation (not the 1st, not the 3rd) or at 8.0, and
    is refused without a budget recorded at run start, with a changed budget, or for a run that is not STILL_IMPROVING.
Negative controls: a constant-LR schedule; a patience-1 stop rule; a continuation that runs to the cap regardless.
"""
from __future__ import annotations

import json

import pytest

from trainkit import art  # noqa: E402  (TRAIN-03: sealed artifact root)
from trainkit import B, N, assert_raises, cfg, nc, ramp, read, scripted, trainer
from v4train import schedule as S
from v4train.schedule import StopTracker, eval_marks, lr_for_update, mark_exposures, s_continuation, s_main, s_of
from v4train.trainer import BUDGET_SCHEMA, TrainerError

MAIN_MARKS = eval_marks("MAIN")


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
    assert mark_exposures(0.5, 201) == 101 and mark_exposures(6.0, 201) == 1206


def _check_low_lr_before_stop(s_fn):
    grid = [3.0 + i * 0.01 for i in range(301)]
    assert all(s_fn(e) <= S.LOW + 1e-12 for e in grid), "the LR must be in the low phase before the 6.0 stop"
    assert all(abs(s_fn(e) - S.LOW) < 1e-12 for e in grid if e <= S.LOW_END), "a real low-LR interval [3.0, 5.5]"
    assert s_fn(6.0) == pytest.approx(S.FLOOR) and max(s_fn(e) for e in (0.5, 1.0, 1.5, 2.0)) == 1.0


def test_t21_low_lr_before_stop():
    _check_low_lr_before_stop(s_main)


# ------------------------------------------------------------------------------------------------ the tracker
def _track(values):
    t = StopTracker()
    for m, v in zip(MAIN_MARKS, values):
        t.update(m, v, "MAIN")
    return t


def test_still_improving_rule_and_tolerance():
    base = [0.1 + 0.01 * i for i in range(12)]                       # strictly improving to 6.0
    assert _track(base).still_improving() and _track(base).main_best_mark == 6.0
    at55 = base[:11] + [base[10]]                                     # best at 5.5, flat at 6.0
    assert _track(at55).still_improving() and _track(at55).main_best_mark == 5.5
    at50 = base[:10] + [base[9], base[9]]                             # best at 5.0: two non-improving evaluations
    assert not _track(at50).still_improving() and _track(at50).main_best_mark == 5.0
    tiny = base[:10] + [base[9], base[9] + 0.0001]                    # +0.0001 <= tolerance: not an improvement
    assert not _track(tiny).still_improving()
    over = base[:10] + [base[9], base[9] + 0.00021]                   # > tolerance: improvement at 6.0
    assert _track(over).still_improving() and _track(over).main_best_mark == 6.0
    assert_raises(S.ScheduleError, _track(base[:11]).still_improving)  # decided only after the 6.0 evaluation


def _stop_mark(improved_flags, should_stop=None):
    t = _track([0.1 + 0.01 * i for i in range(12)])
    v = t.best_value
    for m, imp in zip(eval_marks("CONTINUATION"), improved_flags):
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


def test_t21_patience_is_not_off_by_one():
    _check_patience()


# ------------------------------------------------------------------------------------------------ trainer runs
def test_t21_no_early_stop_before_6(work):
    vals = {m: (0.5 if m == 1.0 else 0.2) for m in MAIN_MARKS}          # peaks at 1.0, then falls
    s = trainer(work / "r", cfg("noearly", "GRU"), eval_override=scripted(vals)).run()
    assert [e["mark"] for e in s["evals"]] == MAIN_MARKS and s["exposures"] == 6 * N
    assert s["practical_best"]["mark"] == 1.0 and s["controlled_budget"]["mark"] == 6.0
    assert not s["still_improving"] and s["updates"] == 6 * -(-N // B)


@pytest.mark.parametrize("peak_at,expect", [(6.0, True), (5.5, True), (5.0, False)])
def test_still_improving_in_the_trainer(work, peak_at, expect):
    s = trainer(work / "r", cfg(f"si{peak_at}", "GRU"), eval_override=scripted(ramp(MAIN_MARKS, peak_at))).run()
    assert s["still_improving"] is expect and s["practical_best"]["mark"] == peak_at
    assert s["label"] == ("BUDGET_LIMITED_STILL_IMPROVING" if expect else "STOPPED_AT_CONTROLLED_ENDPOINT")
    assert (art(work / "r") / "checkpoints" / "latest.pt").exists() is expect      # the 6.0 resumable only if improving


def _budget(path, run_id, max_efe=8.0):
    path.write_text(json.dumps({"schema": BUDGET_SCHEMA, "run_id": run_id, "max_efe": max_efe,
                                "approved_by": "synthetic-test", "recorded_utc": "2026-09-26T00:00:00Z"}),
                    encoding="utf-8")
    return str(path)


def _continued(work, cont_flags, monkey=None):
    run_id = "cont"
    vals = ramp(MAIN_MARKS, 6.0)
    v = vals[6.0]
    for m, imp in zip(eval_marks("CONTINUATION"), cont_flags):
        v = v + 0.001 if imp else v
        vals[m] = v
    c = cfg(run_id, "GRU", continuation_budget=_budget(work / "budget.json", run_id))
    tr = trainer(work / "r", c, eval_override=scripted(vals))
    s = tr.run()
    assert s["still_improving"]
    tr2 = trainer(work / "r", c, eval_override=scripted(vals))
    return tr2.continue_run()


def test_t21_cap_is_not_a_target(work):
    s = _continued(work, [True, False, False, True])
    assert s["status"] == "DONE_CONTINUATION" and s["continuation_stop_mark"] == 7.5, "stopped before the cap"
    assert s["exposures"] < 8 * N and s["practical_best_extended"]["mark"] == 6.5
    assert s["practical_best"]["mark"] == 6.0                                  # the [0, 6] best is unchanged
    ext = read(art(work / "r") / "RESULT_ROWS_EXTENDED.json")["rows"]
    assert {r["selection_rule"] for r in ext} == {"PRACTICAL_BEST_EXTENDED"}
    main = read(art(work / "r") / "RESULT_ROWS.json")["rows"]
    assert {r["selection_rule"] for r in main} == {"PRACTICAL_BEST", "CONTROLLED_BUDGET"}
    assert not (art(work / "r") / "checkpoints" / "latest.pt").exists()
    assert (art(work / "r") / "checkpoints" / "best_extended.pt").exists()


def test_continuation_refusals(work):
    vals = ramp(MAIN_MARKS, 6.0)
    c = cfg("nob", "GRU")                                                    # no budget recorded at start
    trainer(work / "a", c, eval_override=scripted(vals)).run()
    assert_raises(TrainerError, trainer(work / "a", c, eval_override=scripted(vals)).continue_run)
    c2 = cfg("chg", "GRU", continuation_budget=_budget(work / "b.json", "chg"))
    trainer(work / "b", c2, eval_override=scripted(vals)).run()
    _budget(work / "b.json", "chg", max_efe=7.5)                              # edited after the start
    assert_raises(TrainerError, trainer(work / "b", c2, eval_override=scripted(vals)).continue_run)
    flat = ramp(MAIN_MARKS, 4.0)
    c3 = cfg("flat", "GRU", continuation_budget=_budget(work / "c.json", "flat"))
    trainer(work / "c", c3, eval_override=scripted(flat)).run()
    assert_raises(TrainerError, trainer(work / "c", c3, eval_override=scripted(flat)).continue_run)


# ------------------------------------------------------------------------------------------------ negative controls
@nc("a constant-LR schedule never enters a low-LR phase before the stop")
def test_nc_t21_constant_schedule():
    _check_low_lr_before_stop(lambda e: 1.0)


@nc("a patience-1 continuation stop rule is off by one")
def test_nc_t21_patience_one():
    def patience1(t):
        cont = [h for h in t.history if h["phase"] == "CONTINUATION"]
        return bool(cont) and (not cont[-1]["improved"] or cont[-1]["mark"] >= 8.0)
    _check_patience(patience1)


@nc("a continuation that ignores patience runs to the cap")
def test_nc_t21_cap_as_target(work, monkeypatch):
    monkeypatch.setattr(StopTracker, "continuation_should_stop",
                        lambda self: bool(self.history) and self.history[-1]["mark"] >= 8.0)
    s = _continued(work, [True, False, False, True])
    assert s["continuation_stop_mark"] == 7.5, "stopped before the cap"
