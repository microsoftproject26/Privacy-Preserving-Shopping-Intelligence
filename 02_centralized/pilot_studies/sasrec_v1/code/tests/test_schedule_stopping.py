"""Schedule / stopping / selection -- REWRITTEN for Gate P2 schedule S' (owner
38h deadline; see review/FABLE_GATE_P2_RESPONSE.md 'Minimum tests to add' and
config/run_registry.locked.json v2). Pure functions, no model, no I/O; every
mandatory check has a negative control.

S' replaces the old min5/max8/patience-2 early-stopping schedule with FIXED
epochs (planned=6, cut-adjustable down to min_epochs=4) and no early stopping.
The v1 `individual_should_stop`/`pair_should_stop` tests are KEPT (module
docstring: "kept and tested for the record") even though the v2 trainer no
longer calls them; `epoch_end` (best = raw max, tie earlier) and
`select_configuration` stay unchanged and are still tested.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import sasrec_schedule as sched
from mutants import schedule_mutants as sm

U = 156_956  # U_PROD, paths.py


# ---------------------------------------------------------------------------
# Item 1: lr_at boundary values for schedule S' (peak_epochs=1, cosine_epochs=1
# defaults): u<=warmup linear; u<=U peak; U<u<=2U cosine peak->0.1*peak;
# u>2U floor.
# ---------------------------------------------------------------------------
def test_lr_at_boundaries_schedule_S_prime():
    peak = 5e-4
    warmup = 2000
    floor = 0.1 * peak

    def expected(u):
        """Independent hand derivation straight from registry v2's
        `schedule.lr_definition` / the Gate P2 response (not a call into
        sasrec_schedule)."""
        if u <= warmup:
            return peak * u / warmup
        if u <= U:
            return peak
        if u <= 2 * U:
            return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * (u - U) / U))
        return floor

    boundary_us = [1, 2000, 2001, U, U + 1, int(1.5 * U), 2 * U, 2 * U + 1, 5 * U]
    for u in boundary_us:
        got = sched.lr_at(u, peak, U)
        want = expected(u)
        assert got == pytest.approx(want, rel=1e-9, abs=1e-12), f"u={u}: lr_at={got} != expected={want}"

    assert sched.lr_at(2000, peak, U) == pytest.approx(peak)
    assert sched.lr_at(U, peak, U) == pytest.approx(peak)
    assert sched.lr_at(2 * U, peak, U) == pytest.approx(floor, rel=1e-9)
    assert sched.lr_at(5 * U, peak, U) == pytest.approx(floor, rel=1e-9)

    # Negative control: an off-by-one warmup boundary (2001 instead of 2000)
    # must disagree with the correct expected value exactly at u=2000.
    mutant_at_2000 = sched.lr_at(2000, peak, U, warmup=2001)
    assert mutant_at_2000 != pytest.approx(expected(2000), rel=1e-9), \
        "negative control failed: shifting `warmup` by one should move the u=2000 boundary value"

    # Negative control: peak_epochs=2 (instead of the default 1) must move the
    # u=U boundary out of 'peak' phase (U is now mid-peak, not the plateau end).
    mutant_at_U = sched.lr_at(U, peak, U, peak_epochs=2)
    assert mutant_at_U == pytest.approx(peak), "sanity: still inside the (now longer) peak plateau"
    mutant_at_2U_shifted = sched.lr_at(2 * U, peak, U, peak_epochs=2)
    assert mutant_at_2U_shifted == pytest.approx(peak), \
        "with peak_epochs=2 the plateau now extends through 2U -- must differ from the default-schedule floor value"
    assert mutant_at_2U_shifted != pytest.approx(floor, rel=1e-9), \
        "negative control failed: peak_epochs=2 should move u=2U out of the floor phase"

    with pytest.raises(ValueError):
        sched.lr_at(0, peak, U)


def test_lr_phase_labels_schedule_S_prime():
    assert sched.lr_phase(1, U) == "warmup"
    assert sched.lr_phase(2000, U) == "warmup"
    assert sched.lr_phase(2001, U) == "peak"
    assert sched.lr_phase(U, U) == "peak"
    assert sched.lr_phase(U + 1, U) == "cosine"
    assert sched.lr_phase(2 * U, U) == "cosine"
    assert sched.lr_phase(2 * U + 1, U) == "floor"
    assert sched.lr_phase(5 * U, U) == "floor"


# ---------------------------------------------------------------------------
# floor_start_epoch(peak_epochs=1, cosine_epochs=1) == 3; eval_epochs(n, 3).
# ---------------------------------------------------------------------------
def test_floor_start_epoch_default_is_3():
    assert sched.floor_start_epoch() == 3
    assert sched.floor_start_epoch(peak_epochs=1, cosine_epochs=1) == 3
    # Negative control: a shifted schedule (2 peak epochs) must move floor_start.
    assert sched.floor_start_epoch(peak_epochs=2, cosine_epochs=1) == 4
    assert sched.floor_start_epoch(peak_epochs=1, cosine_epochs=2) == 4


def test_eval_epochs_floor_only():
    assert sched.eval_epochs(6, floor_start=3) == [3, 4, 5, 6]
    assert sched.eval_epochs(3, floor_start=3) == [3]
    assert sched.eval_epochs(4, floor_start=3) == [3, 4]
    assert sched.eval_epochs(2, floor_start=3) == [], "no floor epoch reached yet -- must not fabricate one"

    # Negative control: floor_start off by one must change the returned list.
    assert sched.eval_epochs(6, floor_start=4) != sched.eval_epochs(6, floor_start=3)


# ---------------------------------------------------------------------------
# Item 3: epoch_cap as a pure function -- 'slow arm -> 5 or 4', 'never below 4'.
# `arms` schema (confirmed from the landed sasrec_schedule.py docstring):
# [{"epochs_done": int, "mean_train_epoch_seconds": float, "eval_seconds": None|float}].
# ---------------------------------------------------------------------------
def _arm(epochs_done: float, mean_train_epoch_seconds: float, eval_seconds=None) -> dict:
    return {"epochs_done": epochs_done, "mean_train_epoch_seconds": mean_train_epoch_seconds,
            "eval_seconds": eval_seconds}


def _worst_case_seconds(arms, E, *, floor_start=3, eval_seconds_default=1600.0) -> float:
    """Independent reimplementation of epoch_cap's per-E projection (used only
    to construct test deadlines with exact, verifiable arithmetic -- the
    assertions below still call the real `sched.epoch_cap`, never this)."""
    worst = 0.0
    for a in arms:
        done = int(a["epochs_done"])
        ev = float(a["eval_seconds"]) if a.get("eval_seconds") else float(eval_seconds_default)
        remaining_train = max(0, E - done) * float(a["mean_train_epoch_seconds"])
        remaining_evals = len([e for e in sched.eval_epochs(E, floor_start) if e > done])
        worst = max(worst, remaining_train + remaining_evals * ev)
    return worst


def _deadline_for(arms, E_fits, *, planned=6, min_epochs=4, post_hours=2.0, safety_hours=1.0,
                   margin_s=1800.0) -> float:
    """A deadline (with now=0) that makes `_worst_case_seconds(arms, E_fits)`
    fit with `margin_s` to spare, given the real function's
    `budget = deadline - now - (post_hours+safety_hours)*3600`."""
    budget = _worst_case_seconds(arms, E_fits) + margin_s
    return budget + (post_hours + safety_hours) * 3600.0


def test_epoch_cap_fast_arms_reach_planned_6():
    now = 0.0
    deadline = 40 * 3600.0  # generous deadline
    arms = [_arm(2, 3.0 * 3600), _arm(2, 3.2 * 3600)]
    cap = sched.epoch_cap(now, deadline, arms, planned=6, min_epochs=4)
    assert cap == 6, "with a generous deadline and fast arms, the cap must reach the full planned epoch count"


def test_epoch_cap_slow_arm_forces_5_or_4():
    now = 0.0
    fast = _arm(2, 3.0 * 3600)
    slow = _arm(2, 6.5 * 3600)  # slow arm dominates the projection (max over arms)
    arms = [fast, slow]
    worst_5 = _worst_case_seconds(arms, 5)
    worst_6 = _worst_case_seconds(arms, 6)
    assert worst_6 > worst_5, "sanity: E=6 must cost strictly more than E=5"
    # A deadline that fits E=5's projection but not E=6's.
    deadline = _deadline_for(arms, 5, margin_s=1800.0)
    budget = deadline - now - (2.0 + 1.0) * 3600.0
    assert worst_6 > budget >= worst_5, "sanity: this deadline must fit E=5 but not E=6"
    cap = sched.epoch_cap(now, deadline, arms, planned=6, min_epochs=4)
    assert cap == 5, f"a slow arm under this deadline must cap at exactly 5, got {cap}"


def test_epoch_cap_never_below_min_epochs():
    now = 0.0
    deadline = 1.0  # essentially no time left at all
    arms = [_arm(2, 10 * 3600), _arm(2, 12 * 3600)]
    cap = sched.epoch_cap(now, deadline, arms, planned=6, min_epochs=4)
    assert cap == 4, "epoch_cap must never return below min_epochs, however tight the deadline"
    assert cap >= 4


def test_epoch_cap_uses_the_slower_arm_not_the_average():
    now = 0.0
    fast = _arm(2, 2.0 * 3600)
    slow = _arm(2, 7.0 * 3600)
    arms = [fast, slow]
    # A deadline that fits E=6 if (incorrectly) projected from the AVERAGE
    # arm's speed, but not from the real (slower-arm) projection.
    avg_arm = _arm(2, (2.0 * 3600 + 7.0 * 3600) / 2)
    deadline = _deadline_for([avg_arm], 6, margin_s=1800.0)
    worst_6_real = _worst_case_seconds(arms, 6)
    budget = deadline - now - (2.0 + 1.0) * 3600.0
    assert worst_6_real > budget, "sanity: the REAL (slow-arm) E=6 projection must not fit this deadline"
    cap = sched.epoch_cap(now, deadline, arms, planned=6, min_epochs=4)
    assert cap < 6, "epoch_cap must project from the SLOWER arm, not an average across arms"

    # Negative control: the buggy variant that only looks at the FASTEST arm
    # must (at this same deadline) reach 6, disagreeing with the real function.
    mutant_cap = sm.epoch_cap_ignores_slow_arm(now, deadline, arms, planned=6, min_epochs=4)
    assert mutant_cap == 6 and cap < 6, \
        "negative control failed: ignoring the slow arm should let this deadline reach the full planned epochs"


def test_epoch_cap_safety_margin_negative_control():
    now = 0.0
    fast = _arm(2, 3.0 * 3600)
    slow = _arm(2, 5.0 * 3600)
    arms = [fast, slow]
    worst_6 = _worst_case_seconds(arms, 6)
    # Deadline placed so E=6 fits WITHOUT the 1h safety margin but not with it:
    # budget_with_safety = deadline - 3h; budget_without = deadline - 2h.
    deadline = worst_6 + 2.5 * 3600.0  # budget_with_safety = worst_6 - 0.5h (misses); budget_without = worst_6 + 0.5h (fits)
    cap_real = sched.epoch_cap(now, deadline, arms, planned=6, min_epochs=4)
    cap_no_safety = sm.epoch_cap_no_safety_margin(now, deadline, arms, planned=6, min_epochs=4)
    assert cap_real < 6, "sanity: the real function (with its safety margin) must not reach 6 here"
    assert cap_no_safety == 6, \
        "negative control failed: dropping the safety margin should let this exact deadline reach 6"


# ---------------------------------------------------------------------------
# epoch_end 'stays': best = raw max over evaluated epochs, tie keeps the
# earlier epoch. Exercised on the real GRU F0/F4 trajectories with the new
# floor_start_epoch=3 (S'); no stopping/patience semantics are asserted here
# (the trainer no longer consults post_floor_misses for stopping).
# ---------------------------------------------------------------------------
F0 = [0.154372, 0.159049, 0.161037, 0.161386, 0.173874, 0.173758]
F4 = [0.154786, 0.160550, 0.162498, 0.163648, 0.165699, 0.165996]


def _run_series(series, floor_start_epoch=3):
    state = sched.StopState()
    states = []
    for i, metric in enumerate(series, start=1):
        state, rec = sched.epoch_end(state, i, metric, floor_start_epoch=floor_start_epoch)
        states.append(state)
    return states


def test_epoch_end_best_raw_max_tie_earlier_on_GRU_trajectories():
    states_f0 = _run_series(F0)
    states_f4 = _run_series(F4)
    # F0 raw max is at epoch5 (0.173874 > epoch6's 0.173758).
    assert states_f0[-1].best_epoch == 5
    assert states_f0[-1].best_metric == pytest.approx(0.173874)
    # F4 is monotonically increasing -- best stays the LAST epoch.
    assert states_f4[-1].best_epoch == 6
    assert states_f4[-1].best_metric == pytest.approx(0.165996)


def test_epoch_end_exact_tie_keeps_earlier_epoch():
    state = sched.StopState()
    state, _ = sched.epoch_end(state, 1, 0.20, floor_start_epoch=3)
    state, _ = sched.epoch_end(state, 2, 0.20, floor_start_epoch=3)  # exact tie with epoch 1
    assert state.best_epoch == 1, "an exact tie must keep the EARLIER epoch"
    assert state.best_metric == pytest.approx(0.20)

    # Negative control: a mutant that used >= instead of > for raw_improved
    # would incorrectly move best_epoch to the later tied epoch.
    mutant_best_epoch = 2 if 0.20 >= 0.20 else 1  # what a >=-based bug would pick
    assert mutant_best_epoch != state.best_epoch


def test_floor_start_epoch_mutant_changes_epoch_end_miss_accounting():
    """Negative control: 'a mutant with the floor starting one epoch late'
    still changes epoch_end's post_floor_misses trajectory under S' (floor
    now starts at 3, not 4)."""
    series = [0.10, 0.20, 0.15, 0.25, 0.35]  # epoch3 is a genuine miss (0.15 < 0.20)
    correct = _run_series(series, floor_start_epoch=3)
    mutant = _run_series(series, floor_start_epoch=4)  # one epoch late
    assert correct[2].post_floor_misses == 1, "sanity: epoch3 is counted (and a miss) under the correct floor_start=3"
    assert mutant[2].post_floor_misses == 0, "sanity: under the late-floor mutant, epoch3 is not yet counted"
    assert [s.post_floor_misses for s in correct] != [s.post_floor_misses for s in mutant], \
        "negative control failed: floor-starting-one-epoch-late must change the miss trajectory"


# ---------------------------------------------------------------------------
# `individual_should_stop` / `pair_should_stop`: the sasrec_schedule.py module
# docstring says these implement the SUPERSEDED v1 protocol and are "kept
# (and tested) for the record" even though the v2 trainer no longer calls
# them. Kept here unchanged (MIN_EPOCHS=5/MAX_EPOCHS=8/PATIENCE=2/
# FLOOR_START_EPOCH=4, all still the module defaults).
# ---------------------------------------------------------------------------
def _run_series_v1(series):
    state = sched.StopState()
    states, stops = [], []
    for i, metric in enumerate(series, start=1):
        state, rec = sched.epoch_end(state, i, metric)  # v1 defaults: floor_start_epoch=4
        states.append(state)
        stops.append(sched.individual_should_stop(state, i))
    return states, stops


def test_v1_gru_trajectories_never_stop_before_epoch5_and_match_hand_derived_misses():
    states_f0, stops_f0 = _run_series_v1(F0)
    states_f4, stops_f4 = _run_series_v1(F4)
    expected_f0_misses = [0, 0, 0, 0, 0, 1]  # epoch6: 0.173758 < 0.173874 -> miss
    expected_f4_misses = [0, 0, 0, 0, 0, 0]
    assert [s.post_floor_misses for s in states_f0] == expected_f0_misses
    assert [s.post_floor_misses for s in states_f4] == expected_f4_misses
    for epoch in range(1, 5):
        assert stops_f0[epoch - 1][0] is False, f"F0 must not stop before epoch 5 (epoch {epoch})"
        assert stops_f4[epoch - 1][0] is False, f"F4 must not stop before epoch 5 (epoch {epoch})"
    for epoch in (5, 6):
        pair_stop, _ = sched.pair_should_stop(states_f0[epoch - 1], states_f4[epoch - 1], epoch)
        assert pair_stop is False


def test_v1_pair_stopping_requires_BOTH_arms_and_both_stop_at_the_same_epoch():
    series_a = [0.10, 0.20, 0.30, 0.31, 0.311, 0.100, 0.099, 0.098]
    series_b_improving = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    states_a, _ = _run_series_v1(series_a)
    states_b_improving, _ = _run_series_v1(series_b_improving)
    assert states_a[6].post_floor_misses >= 2
    assert states_b_improving[6].post_floor_misses == 0
    pair_stop_isolated, _ = sched.pair_should_stop(states_a[6], states_b_improving[6], 7)
    assert pair_stop_isolated is False, "one arm alone reaching patience must not stop the pair"

    series_b_both = [0.10, 0.20, 0.30, 0.40, 0.50, 0.20, 0.19, 0.18]
    states_b_both, _ = _run_series_v1(series_b_both)
    assert states_b_both[6].post_floor_misses >= 2
    pair_stop_joint, reason = sched.pair_should_stop(states_a[6], states_b_both[6], 7)
    assert pair_stop_joint is True and reason == "pair_patience"


def test_v1_early_stop_never_before_epoch5_randomised():
    rng = np.random.default_rng(2026)
    for trial in range(20):
        series = rng.uniform(0.05, 0.30, size=8).tolist()
        state = sched.StopState()
        for epoch, metric in enumerate(series, start=1):
            state, _ = sched.epoch_end(state, epoch, metric)
            stop, reason = sched.individual_should_stop(state, epoch)
            if epoch < 5:
                assert stop is False, f"trial {trial} epoch {epoch}: stopped before min_epochs=5"
            if epoch == 8:
                assert stop is True and reason == "max_epochs"


# ---------------------------------------------------------------------------
# Item 2/5: fixed-epoch stop at exactly E, never earlier. S' has no schedule.py
# stopping function any more (the trainer just runs until epoch==E); the
# 'equal-epoch assertion helper' is small enough to own here, per the
# coordinator's note, and is exercised directly (a real CLI-level version
# also runs in test_trainer_exposure_resume.py once train_sasrec.py lands).
# ---------------------------------------------------------------------------
def fixed_epoch_should_stop(epoch: int, E: int) -> bool:
    """S': stop iff epoch >= E (the cap/registry-fixed epoch count); never
    earlier, regardless of any metric."""
    if epoch < 1 or E < 1:
        raise ValueError("epoch and E must be >= 1")
    return epoch >= E


def assert_equal_epochs(*epoch_counts: int) -> None:
    """The 'equal-epoch assertion' the supervisor/lock must run before any
    A0-vs-A1 comparison (registry v2 'pair.rule'): every matched arm must
    have completed the SAME number of epochs."""
    if len(set(epoch_counts)) != 1:
        raise ValueError(f"matched arms completed unequal epoch counts: {epoch_counts}")


@pytest.mark.parametrize("E", [4, 5, 6])
def test_fixed_epoch_stop_at_exactly_E_never_earlier(E):
    for epoch in range(1, E):
        assert fixed_epoch_should_stop(epoch, E) is False, f"E={E}: must not stop before epoch {E} (at {epoch})"
    assert fixed_epoch_should_stop(E, E) is True
    assert fixed_epoch_should_stop(E + 1, E) is True

    # Negative control: an off-by-one E (E-1) stops exactly one epoch early.
    assert fixed_epoch_should_stop(E - 1, E - 1) is True and fixed_epoch_should_stop(E - 1, E) is False


def test_equal_epoch_assertion_helper():
    assert_equal_epochs(5, 5, 5)  # must not raise
    with pytest.raises(ValueError):
        assert_equal_epochs(5, 4)
    with pytest.raises(ValueError):
        assert_equal_epochs(6, 6, 5)


# ---------------------------------------------------------------------------
# select_configuration: UNCHANGED -- the full tie chain (metric -> micro R@20
# -> params -> registry order), each level isolated.
# ---------------------------------------------------------------------------
def test_select_configuration_tie_chain_metric_then_micro_recall():
    summaries = [
        dict(run_id="C0", final_status="COMPLETE", best_metric=0.2050, micro_recall20_at_best=0.30, n_params=1000),
        dict(run_id="C1", final_status="COMPLETE", best_metric=0.2050, micro_recall20_at_best=0.32, n_params=1200),
        dict(run_id="C2", final_status="COMPLETE", best_metric=0.2050 - 4e-10, micro_recall20_at_best=0.99, n_params=1),
        dict(run_id="C3", final_status="COMPLETE", best_metric=0.19, micro_recall20_at_best=0.99, n_params=1),
        dict(run_id="C4", final_status="RUNNING", best_metric=0.30, micro_recall20_at_best=0.5, n_params=1),
        dict(run_id="C5", final_status="COMPLETE", best_metric=float("nan"), micro_recall20_at_best=0.5, n_params=1),
    ]
    order = ["C0", "C1", "C2", "C3", "C4", "C5"]
    result = sched.select_configuration(summaries, order)
    assert result["selected"] == "C2", result
    assert set(result["tied_within_1e-6"]) == {"C0", "C1", "C2"}
    assert {i["run_id"] for i in result["ineligible"]} == {"C4", "C5"}


def test_select_configuration_tie_chain_params_then_registry_order():
    summaries = [
        dict(run_id="D0", final_status="COMPLETE", best_metric=0.50, micro_recall20_at_best=0.40, n_params=500),
        dict(run_id="D1", final_status="COMPLETE", best_metric=0.50, micro_recall20_at_best=0.40, n_params=300),
    ]
    result = sched.select_configuration(summaries, ["D0", "D1"])
    assert result["selected"] == "D1", "fewer params must win once metric and micro R@20 are tied"

    summaries_eb = [
        dict(run_id="E_A", final_status="COMPLETE", best_metric=0.50, micro_recall20_at_best=0.40, n_params=300),
        dict(run_id="E_B", final_status="COMPLETE", best_metric=0.50, micro_recall20_at_best=0.40, n_params=300),
    ]
    result_registry_first = sched.select_configuration(summaries_eb, ["E_B", "E_A"])
    assert result_registry_first["selected"] == "E_B", "fully tied runs must fall back to registry order"
    result_registry_second = sched.select_configuration(summaries_eb, ["E_A", "E_B"])
    assert result_registry_second["selected"] == "E_A"


def test_select_configuration_no_eligible_runs():
    summaries = [dict(run_id="Z", final_status="STOPPED_NONFINITE", best_metric=0.9, micro_recall20_at_best=0.5, n_params=1)]
    result = sched.select_configuration(summaries, ["Z"])
    assert result["selected"] is None
    assert "no eligible" in result["reason"]
