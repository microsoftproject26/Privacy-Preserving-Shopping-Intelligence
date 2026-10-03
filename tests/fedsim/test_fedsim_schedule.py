"""The s(EFE) hook, with the exposure point as ONE named parameter that every config must give EXPLICITLY (the
study uses "end"; "start" is kept only as an explicitly named regression option).

Proves: the shape at every knot, refusal past 6.0 without continuation and past 8.0 with
it; for BOTH exposure points, FA / FP / PF use lr_peak_local * s(EFE at the round's start / end) from the one hook
(identical LR sequences across methods, EFE from the actual counters, planned n_consumed checked against the
actual; the PF personal LR follows the same shape and point); LO uses the same shape over its own 6 passes
(step t of T: peak * s(6 (t + d) / T), d = 0 for "start", 1 for "end"); a missing or unknown point is refused.
The run-loop contract (planned LR table, continuation) is in test_fedsim_exposure_point.py.
"""
from __future__ import annotations

import math

import pytest
from fedsim_testkit import assert_raises, clients, nc, one_client, solver, tiny

import ppsi.fedsim.client as flc
import ppsi.fedsim.runtime as flr
from ppsi.fedsim.client import LocalSolver, PFConfig
from ppsi.fedsim.local_only import LORecipe, LORunner
from ppsi.fedsim.numerics import state_digest
from ppsi.fedsim.participation import ParticipationPlan
from ppsi.fedsim.personal import PersonalStore
from ppsi.fedsim.runtime import FLRun, LRTableError, RunConfig
from ppsi.fedsim.schedule import (
    EXPOSURE_POINTS,
    LOCKED_EXPOSURE_POINT,
    ScheduleError,
    lo_lr_schedule,
    round_lr,
    s_of_efe,
)
from ppsi.fedsim.server import Server

PEAK = 0.05
POINTS = list(EXPOSURE_POINTS)


@pytest.mark.parametrize("e,s", [(0.0, 0.0), (0.01, 0.5), (0.02, 1.0), (1.0, 1.0), (2.0, 1.0), (2.5, 0.55),
                                 (3.0, 0.1), (4.0, 0.1), (5.5, 0.1), (5.75, 0.055), (6.0, 0.01)])
def test_shape_at_knots(e, s):
    assert math.isclose(s_of_efe(e), s, rel_tol=0, abs_tol=1e-12)


def test_continuation_and_refusals():
    assert_raises(ScheduleError, s_of_efe, 6.0001)
    assert s_of_efe(6.0 + 1e-13) == s_of_efe(6.0 + 1e-12) == s_of_efe(6.0)          # 1e-12 endpoint tolerance: clamped
    assert_raises(ScheduleError, s_of_efe, 6.0 + 2e-12)                              # outside the 1e-12 tolerance
    assert s_of_efe(6.0, continuation=True) == s_of_efe(6.0)                        # 6.0 itself: the main value
    assert math.isclose(s_of_efe(6.0 + 1e-9, continuation=True), 0.1)
    assert_raises(ScheduleError, s_of_efe, 5.9, continuation=True)
    assert math.isclose(s_of_efe(6.5, continuation=True), 0.1)
    assert math.isclose(s_of_efe(7.5, continuation=True), 0.1)
    assert math.isclose(s_of_efe(7.75, continuation=True), 0.055)
    assert math.isclose(s_of_efe(8.0, continuation=True), 0.01)
    for bad in (8.01, -1e-9, float("nan"), float("inf")):
        assert_raises(ScheduleError, s_of_efe, bad, continuation=True)
    assert round_lr(1e-3, 1.0, exposure_point="start") == 1e-3
    assert round_lr(1e-3, 0.0, exposure_point="start") == 0.0
    assert round_lr(1e-3, 0.0, efe_at_round_end=0.01, exposure_point="end") == 1e-3 * s_of_efe(0.01)
    assert_raises(ScheduleError, round_lr, 1e-3, 0.0, exposure_point="end")          # end EFE required
    assert_raises(ScheduleError, round_lr, 1e-3, 0.0, exposure_point="middle")


def test_exposure_point_has_no_default():
    """No code default; a missing key is refused everywhere."""
    assert LOCKED_EXPOSURE_POINT == "end"
    assert_raises(ScheduleError, RunConfig, "d", "FA", 1, 1e-3, LocalSolver(lr=1e-3))
    assert_raises(ScheduleError, RunConfig, "d", "FA", 1, 1e-3, LocalSolver(lr=1e-3), exposure_point="mid")
    assert_raises(TypeError, lo_lr_schedule, 1.0)
    assert_raises(TypeError, round_lr, 1e-3, 1.0)
    assert_raises(ScheduleError, LORecipe, "x" * 64, 1, lr_shape="s_efe")


def _run(method, point):
    cs = clients(24, seed=131, invalid_frac=0.1)
    by_key = {c.key: c for c in cs}
    n_dec = sum(int((c.examples["target_class"] >= 0).sum()) for c in cs)
    plan = ParticipationPlan.build(list(by_key), seed=2026, manifest_hash="sched", group_size=6)
    a = tiny(1)
    srv = Server(a, init_sha256=state_digest(a.broadcast_state()))
    kw = {"FA": {}, "FP": {"mu": 0.01}, "PF": {"pf": PFConfig(lr=PEAK)}}[method]
    cfg = RunConfig(f"sched-{method}", method, 2026, PEAK, solver(), exposure_point=point, **kw)
    store = PersonalStore(a.query_dim) if method == "PF" else None
    run = FLRun(cfg, plan, srv, by_key.get, n_dec, workers=[tiny(102)], store=store)
    starts, ends = [], []
    for _ in range(9):
        starts.append(run.counters.efe)
        run.step()
        ends.append(run.counters.efe)
    return run, starts, ends


def check_round_lrs(spy_records, point):
    per = {}
    for method in ("FA", "FP", "PF"):
        spy_records.clear()
        try:
            run, starts, ends = _run(method, point)
        except LRTableError as e:
            raise AssertionError(f"the planned LR table refuses this variant: {e}") from None
        efe = starts if point == "start" else ends
        expected = [PEAK * s_of_efe(e) for e in efe]
        assert [lr for _, lr in run.lr_log] == expected, f"{method}: lr != peak * s(EFE at round {point})"
        seen = {}
        for r, lr, plr in spy_records:
            seen.setdefault(r, set()).add((lr, plr))
        for r in range(len(efe)):
            want_p = expected[r] if method == "PF" else None
            assert seen[r] == {(expected[r], want_p)}, f"{method} round {r}: visits used {seen[r]}"
        per[method] = expected
    assert per["FA"] == per["FP"] == per["PF"], "FA / FP / PF must share one LR hook"
    assert max(per["FA"]) == PEAK
    assert (per["FA"][0] == 0.0) == (point == "start"), "round 0 runs at LR 0 exactly under the start convention"
    assert min(per["FA"]) > 0.0 or point == "start", "(end) every round with exposures has LR > 0"


@pytest.fixture
def lr_spy(monkeypatch):
    import ppsi.fedsim.server as fls
    orig = fls.client_update
    records = []

    def spy(adapter, theta_r, client, solver_, **kw):
        pf = kw.get("pf")
        records.append((kw["round_idx"], solver_.lr, pf.lr if pf is not None else None))
        return orig(adapter, theta_r, client, solver_, **kw)
    monkeypatch.setattr(fls, "client_update", spy)
    return records


@pytest.mark.parametrize("point", POINTS)
def test_fa_fp_pf_share_the_round_hook(lr_spy, point):
    check_round_lrs(lr_spy, point)


def check_lo_shape(schedule_fn, point):
    a = tiny(3, tied=False)
    theta0 = a.broadcast_state(clone=True)
    sol = LocalSolver(lr=PEAK, passes=6, batch_size=16)
    recipe = LORecipe(state_digest(theta0), 2026, passes=6, eval_at=(2, 4, 6), solver=sol)
    seen = []
    orig = flc._set_lr
    flc._set_lr = lambda opt, lr: (seen.append(lr), orig(opt, lr))
    try:
        r = LORunner(a, theta0, recipe, lr_schedule=schedule_fn).run(one_client("user-lo", 40))
    finally:
        flc._set_lr = orig
    T = r.steps
    d = 0 if point == "start" else 1
    assert T == 18
    assert seen == [PEAK * s_of_efe(6.0 * (t + d) / T) for t in range(T)], "LO must use the same shape"
    assert (seen[0] == 0.0) == (point == "start")


@pytest.mark.parametrize("point", POINTS)
def test_lo_uses_the_same_shape_over_its_passes(point):
    check_lo_shape(lo_lr_schedule(PEAK, exposure_point=point), point)


@nc("round LR from a shifted EFE (neither the round start nor its end)")
def test_nc_round_lr_shifted(monkeypatch, lr_spy):
    monkeypatch.setattr(flr, "s_of_efe", lambda e, **k: s_of_efe(min(e + 0.25, 6.0)))
    check_round_lrs(lr_spy, "start")


@nc("end-point LR computed from the start EFE")
def test_nc_end_point_uses_start(monkeypatch, lr_spy):
    monkeypatch.setattr(flr, "round_efe_point", lambda s0, e1, point: s0)
    check_round_lrs(lr_spy, "end")


@nc("LO trained at a constant LR instead of the shape")
def test_nc_lo_constant_lr():
    check_lo_shape(lambda step, total: PEAK, "start")
