"""Lock PREG1_LOCK_VALUES.json T2 (exposure_point = "end") — the FL-owned blocking pins P1-P7.

P1 every FL / LO run config carries exposure_point explicitly; the launcher paths (RunConfig.from_dict,
   LORecipe.from_dict) refuse a missing key — no code default is relied on.
P2 the planned round-LR table is generated and hashed BEFORE training; min LR > 0 over rounds with exposures; the
   final main round ends at exactly 6N exposures with LR = peak * s(6.0) (0.01 x peak to 5e-16, see fl.schedule).
P3 fl.schedule.s_of_efe == v4train.schedule.s_main / s_continuation / s_of bit for bit (float.hex) on the knots,
   a 10,000-point grid per phase AND inside the lock's 1e-12 tolerance zone (6.0, 6.0 + 1e-12] (clamped to the 6.0
   value); the same refusals at the domain edges (6.0 + 2e-12, 8.0 + 1e-13, ...); EDGE_TOL == END_TOL == 1e-12;
   v4train is imported read-only (FL-02R1).
P4 one client, one round per update: the FL round LRs == v4train.schedule.lr_for_update(..., "END") exactly.
P5 LO for T in {1, 2, 6, 7, 300, 10000}: lr(0) > 0, lr(T-1) == peak * s(6.0) (0.01 peak to 1e-12), T effective
   updates; an actual LO run for T in {1, 2, 6, 7} counts T optimizer steps at LR > 0.
P6 T26 / T27 re-run under "end" through the run loop: FP(mu=0) == FA bitwise; mu > 0 changes the result;
   PF(lr_p=0, lambda=0) == FA bitwise; p = 0 reproduces the shared forward.
P7 continuation past 6.0 only after a STILL_IMPROVING receipt recomputed from the run's own evaluations; forged or
   NOT_IMPROVING receipts and receipt-less steps are refused; the continuation uses s_continuation and stops after 2
   non-improving evaluations or at 8.0.
"""
from __future__ import annotations

import math
import random

import pytest
import torch

import fl.client as flc
import fl.runtime as flr
import v4train.schedule as central                       # read-only (lock pin P3 / P4)
from fl.checkpoint import CheckpointError
from fl.client import LocalSolver, PFConfig
from fl.local_only import LORecipe, LORunner
from fl.numerics import state_digest
from fl.participation import ParticipationPlan
from fl.personal import PersonalStore
from fl.runtime import FLRun, LRTableError, RunConfig, build_lr_table
from fl.schedule import KNOTS, ScheduleError, lo_lr_schedule, s_of_efe
from fl.server import Server
from fl_testkit import assert_raises, clients, nc, one_client, solver, tiny

PEAK = 0.05
END = "end"


def _registered(seed=1, tied=True):
    a = tiny(seed, tied=tied)
    if not tied:
        a.server_post_aggregate()
    return Server(a, init_sha256=state_digest(a.broadcast_state()))


def _cohort(n=24, seed=151, g=6):
    cs = clients(n, seed=seed, invalid_frac=0.1)
    by_key = {c.key: c for c in cs}
    n_dec = sum(int((c.examples["target_class"] >= 0).sum()) for c in cs)
    plan = ParticipationPlan.build(list(by_key), seed=2026, manifest_hash="t2", group_size=g)
    return by_key, n_dec, plan


def _run(method, *, point=END, eval_fn=None, pf=None, mu=0.0, **kw):
    by_key, n_dec, plan = _cohort()
    srv = _registered()
    store = PersonalStore(srv.adapter.query_dim) if method == "PF" else None
    cfg = RunConfig(f"t2-{method}", method, 2026, PEAK, solver(), mu=mu, pf=pf, exposure_point=point, **kw)
    return FLRun(cfg, plan, srv, by_key.get, n_dec, workers=[tiny(103)], store=store, eval_fn=eval_fn)


# ------------------------------------------------------------------------------------------------ P1
def check_explicit_key(load_run, load_lo):
    base = {"run_id": "p1", "method": "FA", "seed": 1, "lr_peak_local": 1e-3, "solver": {"lr": 1e-3}}
    assert_raises(ValueError, load_run, dict(base))
    assert_raises(ValueError, load_run, {**base, "exposure_point": None})
    assert load_run({**base, "exposure_point": "end"}).exposure_point == "end"
    lo = {"theta0_digest": "0" * 64, "seed": 1, "passes": 6, "solver": {"lr": 1e-3, "passes": 6}}
    assert_raises(ValueError, load_lo, dict(lo))
    assert load_lo({**lo, "exposure_point": "end"}).exposure_point == "end"


def test_p1_missing_exposure_point_refused():
    check_explicit_key(RunConfig.from_dict, LORecipe.from_dict)
    assert_raises(ScheduleError, RunConfig, "p1", "FA", 1, 1e-3, LocalSolver(lr=1e-3))
    assert_raises(ScheduleError, LORecipe, "0" * 64, 1, lr_shape="s_efe")


@nc("a launcher that fills in a default exposure point")
def test_nc_p1_defaulting_launcher():
    def lenient(d):
        return RunConfig.from_dict({**d, "exposure_point": d.get("exposure_point") or "end"})

    def lenient_lo(d):
        return LORecipe.from_dict({**d, "exposure_point": d.get("exposure_point") or "end"})
    check_explicit_key(lenient, lenient_lo)


# ------------------------------------------------------------------------------------------------ P2
def check_lr_table(builder):
    by_key, n_dec, plan = _cohort()
    nv = {k: int((c.examples["target_class"] >= 0).sum()) for k, c in by_key.items()}
    cfg = RunConfig("p2", "FA", 1, PEAK, solver(), exposure_point=END)
    t = builder(plan, nv, n_dec, cfg)
    rows = t["rows"]
    assert rows[-1][3] == 6 * n_dec, "the final main unit must end at exactly 6N exposures"
    assert t["final_lr"] == PEAK * central.s_main(6.0) and math.isclose(t["final_lr"], 0.01 * PEAK, rel_tol=1e-12)
    assert t["min_lr"] > 0 and all(r[4] > 0 for r in rows if r[2] > 0)
    assert len(t["digest"]) == 64 and t["n_rounds"] == 3 * plan.rounds_per_sweep
    assert_raises(LRTableError, builder, plan, nv, n_dec + 1, cfg)           # 6N not on a round boundary: refused


def test_p2_planned_lr_table_hashed_before_training():
    check_lr_table(build_lr_table)
    run = _run("FA")
    assert run.state.cursor == 0 and run.lr_table["digest"]                  # built at construction, before any round
    d0 = run.lr_table["digest"]
    run.run(max_rounds=3)
    assert [lr for _, lr in run.lr_log] == [r[4] for r in run.lr_table["rows"][:3]] and run.lr_table["digest"] == d0
    assert run.state_dict()["lr_table_digest"] == d0


@nc("a table builder that does not check the 6N endpoint")
def test_nc_p2_unchecked_endpoint():
    def lax(plan, nv, N, cfg):
        try:
            return build_lr_table(plan, nv, N, cfg)
        except LRTableError:
            return {"rows": [[0, 0, 1, 1, 1.0, 1.0]], "final_lr": 0.0, "min_lr": 1.0, "digest": "x" * 64,
                    "n_rounds": 1}
    check_lr_table(lax)


# ------------------------------------------------------------------------------------------------ P3
def _grid():
    rng = random.Random(20260926)
    pts = [x for x, _ in KNOTS] + [i * 6.0 / 9999 for i in range(10_000)]
    pts += [rng.uniform(0.0, 6.0) for _ in range(1000)] + [0.02 + 1e-15, 2.0 - 1e-15, 5.5 + 1e-15, 6.0 - 1e-15]
    pts += TOL_ZONE
    cont = [6.0 + (i + 1) * 2.0 / 10_000 for i in range(10_000)] + [6.0 + 1e-13, 7.5, 7.5 + 1e-15, 8.0]
    return pts, cont


TOL_ZONE = [6.0 + 1e-15, 6.0 + 1e-13, 6.0 + 5e-13, 6.0 + 9e-13, 6.0 + 1e-12]    # (6.0, 6.0 + 1e-12]


def check_bitwise(fn_main, fn_cont):
    import fl.schedule as fls
    assert fls.EDGE_TOL == central.END_TOL == 1e-12, "lock T2 P3: the 6.0 endpoint tolerance is 1e-12 in both"
    for e in TOL_ZONE:                                                       # inside the tolerance: the 6.0 value
        try:
            got = fn_main(e)
        except (ScheduleError, central.ScheduleError) as err:
            raise AssertionError(f"EFE {e!r} refused inside the lock's 1e-12 tolerance: {err}") from None
        assert float.hex(got) == float.hex(central.s_main(e)) == float.hex(central.s_main(6.0)), e
    pts, cont = _grid()
    bad = [e for e in pts if float.hex(fn_main(e)) != float.hex(central.s_main(e))]
    assert not bad, f"{len(bad)} main-phase points differ from v4train.schedule.s_main, e.g. {bad[:3]}"
    bad = [e for e in cont if float.hex(fn_cont(e)) != float.hex(central.s_continuation(e))]
    assert not bad, f"{len(bad)} continuation points differ from v4train.schedule.s_continuation, e.g. {bad[:3]}"
    assert float.hex(fn_cont(6.0)) == float.hex(central.s_of(6.0, "CONTINUATION"))
    for e in (6.0 + 2e-12, 6.0 + 1e-9, -1e-9, float("nan")):             # outside the tolerance: refused
        assert_raises((ScheduleError, central.ScheduleError), fn_main, e)
        assert_raises(central.ScheduleError, central.s_main, e)
    for e in (8.0 + 1e-13, 5.9):
        assert_raises((ScheduleError, central.ScheduleError), fn_cont, e)
        assert_raises(central.ScheduleError, central.s_of, e, "CONTINUATION")


def test_p3_schedule_bitwise_equal_to_central():
    check_bitwise(lambda e: s_of_efe(e), lambda e: s_of_efe(e, continuation=True))


@nc("no endpoint tolerance: 6.0 + 1e-13 refused instead of clamped (the pre-FL-02R1 behaviour)")
def test_nc_p3_no_endpoint_tolerance():
    def strict_main(e):
        if e > 6.0:
            raise ScheduleError(f"strict main refuses {e}")
        return s_of_efe(e)
    check_bitwise(strict_main, lambda e: s_of_efe(e, continuation=True))


@nc("a tolerance without the clamp (extrapolating the cooldown past 6.0)")
def test_nc_p3_tolerance_without_clamp():
    def extrapolate(e):
        if 6.0 < e <= 6.0 + 1e-12:
            return 0.1 + (0.01 - 0.1) * (e - 5.5) / (6.0 - 5.5)
        return s_of_efe(e)
    check_bitwise(extrapolate, lambda e: s_of_efe(e, continuation=True))


@nc("a float32 re-implementation of the schedule")
def test_nc_p3_float32_schedule():
    import numpy as np
    check_bitwise(lambda e: float(np.float32(s_of_efe(e)) * np.float32(1.0)),
                  lambda e: s_of_efe(e, continuation=True))


# ------------------------------------------------------------------------------------------------ P4
def check_one_client_updates(point, central_point):
    c = one_client("user-solo", 10, seed=4)
    plan = ParticipationPlan.build([c.key], seed=1, manifest_hash="p4", group_size=1)
    N = 40                                                            # each round = one update = 0.25 EFE
    srv = _registered()
    sol = LocalSolver(lr=PEAK, passes=1, batch_size=16, clip=1.0)
    cfg = RunConfig("p4", "FA", 2026, PEAK, sol, exposure_point=point)
    run = FLRun(cfg, plan, srv, {c.key: c}.get, N, workers=[tiny(104)]).run()
    assert run.state.done and len(run.lr_log) == 24 and run.counters.steps == 24
    want = [central.lr_for_update(PEAK, 10 * r, 10, N, "MAIN", central_point) for r in range(24)]
    assert [lr for _, lr in run.lr_log] == want, "FL round LRs must equal the central lr_for_update exactly"


def test_p4_fl_round_lrs_equal_central_updates():
    check_one_client_updates("end", "end")


def test_p4_regression_start_matches_central_start():
    check_one_client_updates("start", "start")


@nc("FL at the round start compared with the central END convention")
def test_nc_p4_start_vs_central_end():
    check_one_client_updates("start", "end")


# ------------------------------------------------------------------------------------------------ P5
def check_lo_T(point):
    for T in (1, 2, 6, 7, 300, 10_000):
        lr = lo_lr_schedule(PEAK, 6, exposure_point=point)
        vals = [lr(t, T) for t in range(T)]
        assert vals[0] > 0, f"T={T}: the first LO step must have LR > 0"
        assert vals[-1] == PEAK * central.s_main(6.0) and math.isclose(vals[-1], 0.01 * PEAK, rel_tol=1e-12)
        assert sum(v > 0 for v in vals) == T, f"T={T}: every step must be an effective update"


def check_lo_runs(point):
    a = tiny(3, tied=False)
    theta0 = a.broadcast_state(clone=True)
    for T, passes, n in ((1, 1, 10), (2, 2, 10), (6, 6, 10), (7, 7, 10)):
        sol = LocalSolver(lr=PEAK, passes=passes, batch_size=16)
        recipe = LORecipe(state_digest(theta0), 2026, passes=passes, eval_at=(passes,), solver=sol,
                          lr_shape="s_efe", exposure_point=point)
        seen = []
        orig = flc._set_lr
        flc._set_lr = lambda opt, lr: (seen.append(lr), orig(opt, lr))
        try:
            r = LORunner(a, theta0, recipe).run(one_client("user-lo", n))
        finally:
            flc._set_lr = orig
        assert r.steps == T and len(seen) == T and all(v > 0 for v in seen), (T, seen)
        assert seen[-1] == PEAK * central.s_main(6.0)


def test_p5_lo_every_step_effective():
    check_lo_T(END)
    check_lo_runs(END)


@nc("LO at the start convention (first step at LR 0)")
def test_nc_p5_lo_start():
    check_lo_T("start")


# ------------------------------------------------------------------------------------------------ P6
def _final(run, rounds=5):
    run.run(max_rounds=rounds)
    return run.server.digest(), [lr for _, lr in run.lr_log]


def check_t26_t27_under_end(fp0_digest):
    fa = _final(_run("FA"))
    assert fp0_digest == fa, "T26 under 'end': FP(mu=0) must equal FA bitwise"
    fp = _final(_run("FP", mu=0.1))
    assert fp[1] == fa[1] and fp[0] != fa[0], "T26 under 'end': mu > 0 must change the result, same LRs"
    pf0 = _final(_run("PF", pf=PFConfig(lr=0.0, lam=0.0)))
    assert pf0 == fa, "T27 under 'end': PF(lr_p=0, lambda=0) must equal FA bitwise"


def test_p6_t26_t27_rerun_under_end():
    check_t26_t27_under_end(_final(_run("FP", mu=0.0)))
    a = tiny(1)
    a.module.eval()
    b = one_client("user-q", 9).examples
    with torch.no_grad():
        assert torch.equal(a.scores(b), a.scores(b, torch.zeros(a.query_dim)))   # T27: p = 0 recovers the forward


@nc("FP(mu=0) run with its own RNG stream under 'end'")
def test_nc_p6_fp_own_rng():
    run = _run("FP", mu=0.0)
    run.cfg = RunConfig("t2-FP", "FP", 2027, PEAK, solver(), mu=0.0, exposure_point=END)   # a different seed
    check_t26_t27_under_end(_final(run))


# ------------------------------------------------------------------------------------------------ P7
def _finished(metric_of_efe):
    """A finished main run whose evaluation metric is a function of the run's EFE at evaluation time."""
    holder = {}
    run = _run("FA", eval_fn=lambda server: metric_of_efe(holder["run"].counters.efe))
    holder["run"] = run
    run.run()
    assert run.state.done and run.counters.efe == 6.0
    return run


def check_continuation_gate(allow):
    improving = _finished(lambda e: e)                                  # best at 6.0 -> STILL_IMPROVING
    rec = improving.still_improving_receipt()
    assert rec["verdict"] == "STILL_IMPROVING" and rec["best_half_efe"] == 12
    assert_raises(RuntimeError, improving.step)                        # no receipt: no round past 6.0
    allow(improving, rec)
    n0 = len(improving.lr_log)
    improving.step()
    r, lr = improving.lr_log[n0]
    assert improving.state.phase == "CONTINUATION"
    assert lr == PEAK * central.s_of(improving.counters.efe, "CONTINUATION"), "continuation uses s_continuation"
    improving.run()
    assert improving.state.done and improving.counters.efe == 8.0      # never non-improving: runs to the ceiling
    flat = _finished(lambda e: min(e, 6.0))                             # improving to 6.0, then flat
    allow(flat, flat.still_improving_receipt())
    flat.run()
    cont = [x for x in flat.state.evals if x[4] == "CONTINUATION"]
    assert flat.state.done and len(cont) == 2 and not any(x[3] for x in cont) and flat.counters.efe < 8.0, \
        "the continuation stops after 2 non-improving evaluations"
    early = _finished(lambda e: 1.0 / (1.0 + e))                        # best at 0.5 -> NOT_IMPROVING
    rec2 = early.still_improving_receipt()
    assert rec2["verdict"] == "NOT_IMPROVING"
    assert_raises(CheckpointError, early.continue_after_endpoint, rec2)
    forged = dict(rec2, verdict="STILL_IMPROVING", best_half_efe=12)
    assert_raises(CheckpointError, early.continue_after_endpoint, forged)


def test_p7_continuation_only_after_still_improving_receipt():
    check_continuation_gate(lambda run, rec: run.continue_after_endpoint(rec))


@nc("continuation opened without checking the receipt")
def test_nc_p7_unchecked_continuation():
    def open_anyway(run, rec):
        run.state.phase, run.state.done = "CONTINUATION", False
        run.lr_table = build_lr_table(run.plan, run.n_valid_by_key, run.n_decisions_total, run.cfg,
                                      continuation=True, start_round=run.state.cursor,
                                      start_exposures=run.counters.exposures, end_efe=8.0)
    run = _finished(lambda e: 1.0 / (1.0 + e))                          # NOT_IMPROVING
    open_anyway(run, None)
    try:
        run.step()
    except Exception as e:                                             # noqa: BLE001
        raise AssertionError(f"unexpected failure {e}") from None
    raise AssertionError("a NOT_IMPROVING run was continued without a receipt")
