"""The participation plan and the exposure / coverage counters.

Proves: seeded permutation sweeps WITHOUT replacement over the cohort keys; fixed consecutive groups; the partial
last group handled explicitly (SMALLER_FINAL_GROUP) and disclosed; determinism from (seed, manifest hash) across
processes and independent of the caller's key order; EFE / coverage / visit / step counters from the ACTUAL
n_consumed of the round reports (no-op clients counted). Does not prove: anything about the real cohort manifest.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from fedsim_testkit import assert_raises, clients, nc, solver, tiny

import ppsi.fedsim.participation as flp
import ppsi.fedsim.runtime as flr
from ppsi.fedsim.participation import ExposureCounters, ParticipationPlan, PlanError, keys_digest
from ppsi.fedsim.runtime import FLRun, RunConfig
from ppsi.fedsim.server import Server

REPO_ROOT = Path(__file__).resolve().parents[2]
KEYS = [f"user-{i:05d}" for i in range(200)]
MH = "cohort-manifest-sha256-for-tests"


def _plan(keys=KEYS, seed=2026, mh=MH, g=64):
    return ParticipationPlan.build(keys, seed=seed, manifest_hash=mh, group_size=g)


def check_sweeps_without_replacement(p: ParticipationPlan, sweeps=3):
    R = p.rounds_per_sweep
    assert R == -(-p.n_clients // p.group_size)
    for sw in range(sweeps):
        rounds = [p.round_keys(sw * R + g) for g in range(R)]
        sizes = [len(r) for r in rounds]
        assert sizes[:-1] == [p.group_size] * (R - 1), sizes
        assert sizes[-1] == (p.partial_group_size or p.group_size), sizes
        flat = [k for r in rounds for k in r]
        assert Counter(flat) == Counter(p.keys), f"sweep {sw} is not a permutation without replacement"
    assert p.round_keys(0) != p.round_keys(R), "consecutive sweeps must use different permutations"


def check_seed_and_hash_sensitivity(build):
    a, b = build(KEYS, 2026, MH), build(KEYS, 2026, MH)
    assert a.round_keys(0) == b.round_keys(0) and a.digest() == b.digest()
    assert build(KEYS, 2027, MH).round_keys(0) != a.round_keys(0), "the plan must depend on the seed"
    assert build(KEYS, 2026, MH + "x").round_keys(0) != a.round_keys(0), "the plan must depend on the manifest hash"
    shuffled = list(np.random.default_rng(5).permutation(KEYS))
    assert build(shuffled, 2026, MH).round_keys(0) == a.round_keys(0), "the caller's key order must not matter"


def _build(keys, seed, mh):
    return ParticipationPlan.build(keys, seed=seed, manifest_hash=mh, group_size=64)


# ------------------------------------------------------------------------------------------------ tests
def test_plan_sweeps_without_replacement_fixed_groups():
    check_sweeps_without_replacement(_plan())
    check_sweeps_without_replacement(_plan(g=40))                  # 200 / 40: no partial group


def test_plan_partial_last_group_disclosed():
    d = _plan().describe()
    assert d["rounds_per_sweep"] == 4 and d["partial_group_size"] == 8
    assert d["partial_group_policy"] == "SMALLER_FINAL_GROUP" and "8 clients" in d["partial_group_note"]
    big = ParticipationPlan.build([f"u{i}" for i in range(262_144)], seed=2026, manifest_hash=MH)
    assert big.rounds_per_sweep == 4096 and big.partial_group_size == 0          # the study cohort: 3 visits = 12,288
    assert 3 * big.rounds_per_sweep == 12_288


def test_plan_deterministic_from_seed_and_manifest_hash():
    check_seed_and_hash_sensitivity(_build)


def test_plan_deterministic_across_processes():
    p = _plan()
    code = (f"import json,sys; sys.path.insert(0, r'{REPO_ROOT}'); from ppsi.fedsim.participation import ParticipationPlan as P; "
            f"p = P.build([f'user-{{i:05d}}' for i in range(200)], seed=2026, manifest_hash='{MH}', group_size=64); "
            "print(json.dumps([p.digest()] + [p.round_keys(r) for r in (0, 3, 4, 11)]))")
    env = dict(os.environ, PYTHONHASHSEED="4242", PYTHONDONTWRITEBYTECODE="1")
    out = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, env=env, timeout=120,
                         check=False)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout) == [p.digest()] + [p.round_keys(r) for r in (0, 3, 4, 11)]


def test_plan_refusals():
    assert_raises(PlanError, ParticipationPlan.build, [], seed=1, manifest_hash=MH)
    assert_raises(PlanError, ParticipationPlan.build, ["a", "b", "a"], seed=1, manifest_hash=MH)
    assert_raises(PlanError, ParticipationPlan.build, ["a", 2], seed=1, manifest_hash=MH)
    assert_raises(PlanError, ParticipationPlan.build, ["a", "b"], seed=1, manifest_hash="")
    assert_raises(PlanError, ParticipationPlan.build, KEYS, seed=1, manifest_hash=MH, expected_keys_digest="0" * 64)
    ParticipationPlan.build(KEYS, seed=1, manifest_hash=MH, expected_keys_digest=keys_digest(KEYS))


# ------------------------------------------------------------------------------------------------ counters
def _registered_tiny(seed):
    a = tiny(seed, tied=True)
    return a, Server(a, init_sha256=_digest(a))


def _digest(a):
    from ppsi.fedsim.numerics import state_digest
    return state_digest(a.broadcast_state())


def _run_two_sweeps():
    cs = clients(40, seed=121, invalid_frac=0.2)
    cs[3].examples["target_class"][:] = -1                                   # a no-op (no-label) client
    by_key = {c.key: c for c in cs}
    n_dec = sum(int((c.examples["target_class"] >= 0).sum()) for c in cs)
    plan = ParticipationPlan.build(list(by_key), seed=2026, manifest_hash=MH, group_size=16)
    _a, srv = _registered_tiny(1)
    cfg = RunConfig("counters", "FA", 2026, 0.05, solver(), ckpt_every=10 ** 9, exposure_point="end")
    run = FLRun(cfg, plan, srv, by_key.get, n_dec, workers=[tiny(101)])
    return run, plan, by_key, n_dec


def check_counters_from_actual_consumption():
    run, plan, by_key, _n_dec = _run_two_sweeps()
    R = plan.rounds_per_sweep
    total = 0
    for r in range(2 * R):
        run.step()
        total += 2 * sum(int((by_key[k].examples["target_class"] >= 0).sum()) for k in plan.round_keys(r))
        assert run.counters.exposures == total, "exposures must be the sum of actual n_consumed"
        if r == R - 1:
            assert run.counters.reached_efe(2.0) and run.counters.efe == 2.0, run.counters.efe
            assert run.counters.coverage_by_sweep["0"] == plan.n_clients
            assert (run.counters.visit_counts == 1).all()
    c = run.counters
    assert c.efe == 4.0 and (c.visit_counts == 2).all() and c.coverage_by_sweep["1"] == plan.n_clients
    n_noop_clients = sum(1 for cl in by_key.values() if int((cl.examples["target_class"] >= 0).sum()) == 0)
    assert n_noop_clients >= 1
    assert c.visits == 80 and c.noop_visits == 2 * n_noop_clients and sum(c.steps_hist.values()) == c.visits
    assert c.rounds == 2 * R and c.summary()["min_visits"] == c.summary()["max_visits"] == 2


class PlannedRowsCounters(ExposureCounters):
    """WRONG: counts planned rows x passes (invalid rows included) instead of the reports' actual n_consumed."""
    rows: dict = {}  # noqa: RUF012  (deliberately wrong shared state)

    def update(self, plan, round_idx, visits):
        fake = [type(v)(**{**v.__dict__, "n_consumed": 2 * self.rows[v.key]}) for v in visits]
        super().update(plan, round_idx, fake)


def test_counters_from_actual_n_consumed():
    check_counters_from_actual_consumption()


def test_counters_state_roundtrip():
    run, _unused_plan, *_ = _run_two_sweeps()
    run.run(max_rounds=5)
    d = run.counters.state_dict()
    back = ExposureCounters.from_state_dict(d)
    assert back.summary() == run.counters.summary() and back.half_efe_units() == run.counters.half_efe_units()


@nc("sampling with replacement inside a sweep")
def test_nc_plan_with_replacement(monkeypatch):
    def with_replacement(seed, mh, g, sweep, n):
        return np.random.default_rng(seed + sweep).integers(0, n, n)
    monkeypatch.setattr(flp, "_sweep_perm", with_replacement)
    check_sweeps_without_replacement(_plan())


@nc("the plan ignores the manifest hash")
def test_nc_plan_ignores_manifest_hash():
    check_seed_and_hash_sensitivity(lambda keys, seed, mh: ParticipationPlan.build(keys, seed=seed,
                                                                                   manifest_hash=MH, group_size=64))


@nc("the plan keeps the caller's key order")
def test_nc_plan_caller_order():
    def unsorted(keys, seed, mh):
        p = ParticipationPlan.build(keys, seed=seed, manifest_hash=mh, group_size=64)
        return ParticipationPlan(tuple(keys), p.seed, p.manifest_hash, p.group_size)
    check_seed_and_hash_sensitivity(unsorted)


@nc("EFE counted from planned rows instead of actual n_consumed")
def test_nc_counters_planned_rows(monkeypatch):
    cs = clients(40, seed=121, invalid_frac=0.2)
    PlannedRowsCounters.rows = {c.key: int(c.examples["target_class"].shape[0]) for c in cs}
    monkeypatch.setattr(flr, "ExposureCounters", PlannedRowsCounters)
    check_counters_from_actual_consumption()
