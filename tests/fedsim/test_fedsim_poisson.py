"""Poisson participation for FA_1024 / S_CAL / DP, fixed denominator, expected-exposure LR
schedule, empty rounds, and the DP-plan enforcement (tiny synthetic family, CPU).

Proves: each client joins round r independently with probability q = m / N (empirical rate, per-client counts and the
cohort-size variance within binomial tolerances; exact rational q at the study setting), from (seed, "poisson", r)
only (deterministic, no RNG state, identical after a cache flush and across a resume); the DP round divides by the
FIXED m whether |C_r| > m or < m (independent reference); the planned LR depends on the round index only
(s(6 (r + 1) / T), E_total = T q passes V, identical for two datasets with the same keys) while planned == actual
consumption is still enforced on the realised cohort; an empty round releases noise only for z > 0 and is a no-op for
z = 0 (S calibration logs an empty norm list); a DP-configured run with a uniform / sweep plan, or with group_size !=
m, is refused; a DP run resumes bitwise. Negative controls: denominator |C_r|; fixed-size sampling fails the binomial
cohort-size test; a DP run accepting a uniform plan; an LR schedule read from realised exposures is data-dependent.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest
import torch
from fedsim_testkit import assert_raises, clients, nc, solver, tiny

from ppsi.fedsim import participation as P
from ppsi.fedsim.checkpoint import CheckpointManager
from ppsi.fedsim.client import client_update, valid_rows
from ppsi.fedsim.dp import DPConfig, clip_factor, flat_delta_norm, noise_like
from ppsi.fedsim.numerics import state_digest
from ppsi.fedsim.participation import ParticipationPlan
from ppsi.fedsim.runtime import FLRun, RunConfig, build_lr_table
from ppsi.fedsim.schedule import s_of_efe
from ppsi.fedsim.server import Server, run_round

SEED = 2026
PEAK = 0.05
CS = clients(24, seed=707, invalid_frac=0.1)
BY_KEY = {c.key: c for c in CS}
N_DEC = sum(int((c.examples["target_class"] >= 0).sum()) for c in CS)
M = 6                                                          # q = 6 / 24 = 0.25


def theta0():
    a = tiny(1, tied=False)
    a.server_post_aggregate()
    return a.broadcast_state(clone=True), state_digest(a.broadcast_state())


THETA0, INIT = theta0()


def fresh_server():
    a = tiny(1, tied=False)
    a.load_state_(THETA0)
    return Server(a, init_sha256=INIT)


def pplan(keys=None, m=M, seed=SEED, sampling="poisson"):
    return ParticipationPlan.build(list(keys or BY_KEY), seed=seed, manifest_hash="poisson-tests", group_size=m,
                                   sampling=sampling)


def synthetic_keys(n):
    return [f"k{i:07d}" for i in range(n)]


# ------------------------------------------------------------------------------------------------ the plan
def test_poisson_plan_deterministic_seeded_and_stateless():
    p, q = pplan(), pplan()
    rounds = [p.round_keys(r) for r in range(40)]
    assert rounds == [q.round_keys(r) for r in range(40)]
    P._poisson_round.cache_clear()                            # no hidden state: a resumed process re-derives it
    assert rounds == [pplan().round_keys(r) for r in range(40)]
    assert len({tuple(x) for x in rounds}) > 30                # rounds differ
    assert rounds != [pplan(seed=SEED + 1).round_keys(r) for r in range(40)]
    for r in range(40):
        pos = p.round_positions(r)
        assert list(pos) == sorted(pos) and len(set(pos.tolist())) == len(pos)   # no repeats; sorted-key order
    d = p.describe()
    assert d["sampling_mode"] == "poisson" and d["inclusion_probability"] == 0.25
    assert p.digest() != pplan(sampling="uniform").digest() != pplan(sampling="sweep").digest()
    assert pplan(sampling="sweep").describe().get("sampling_mode") is None           # sweep description unchanged
    assert_raises(P.PlanError, p.round_positions, -1)


def test_poisson_stream_is_the_documented_one():
    """derive_seed(seed, "poisson", r) -> PCG64 -> u ~ Uniform{0..N-1} per sorted key, member iff u < m."""
    p = pplan()
    for r in (0, 7):
        rng = np.random.Generator(np.random.PCG64(P.derive_seed(SEED, "poisson", r)))
        u = rng.integers(0, p.n_clients, size=p.n_clients, dtype=np.int64)
        assert p.round_keys(r) == [p.keys[i] for i in np.flatnonzero(u < M)]


def test_empirical_inclusion_rate_matches_q():
    n, m, R = 200, 20, 2000                                    # q = 0.1
    p = pplan(synthetic_keys(n), m=m)
    q = p.inclusion_probability
    counts = np.zeros(n, dtype=np.int64)
    sizes = []
    for r in range(R):
        pos = p.round_positions(r)
        counts[pos] += 1
        sizes.append(len(pos))
    total, mean = counts.sum(), n * R * q
    assert abs(total - mean) <= 5 * math.sqrt(n * R * q * (1 - q))                    # overall rate
    assert np.all(np.abs(counts - R * q) <= 6 * math.sqrt(R * q * (1 - q)))           # every client
    var, want = float(np.var(sizes, ddof=1)), n * q * (1 - q)                         # |C_r| ~ Binomial(N, q)
    assert abs(var - want) <= 5 * want * math.sqrt(2 / (R - 1))
    assert min(sizes) < m < max(sizes)


def test_study_setting_q_exact_and_rate():
    n, m = 131072, 1024
    p = pplan(synthetic_keys(n), m=m)
    assert Fraction(p.inclusion_probability) == Fraction(1, 128)
    R = 48
    sizes = [len(p.round_positions(r)) for r in range(R)]
    assert abs(sum(sizes) - R * m) <= 5 * math.sqrt(R * m * (1 - 1 / 128))
    assert len(set(sizes)) > 1


def test_group_size_at_least_n_includes_everyone():
    p = pplan(m=100)
    assert p.inclusion_probability == 1.0 and all(p.round_keys(r) == list(p.keys) for r in range(5))


# ------------------------------------------------------------------------------------------------ the DP round
def reference_round(theta_r, keys, dp: DPConfig, round_idx, *, denominator=None, lr=PEAK):
    """Independent reference: single visits from theta_r, flat clip, sum, noise zS on the sum, / m."""
    w = tiny(55, tied=False)
    shared = w.manifest.shared_keys
    total = OrderedDict((k, torch.zeros_like(theta_r[k])) for k in shared)
    for k_ in keys:
        res = client_update(w, theta_r, BY_KEY[k_], replace(solver(), lr=lr), round_idx=round_idx, seed=SEED,
                            clone_upload=True)
        if res.n_consumed == 0:
            continue
        f = clip_factor(flat_delta_norm(res.upload, theta_r, shared), dp.clip_norm)
        for k in shared:
            total[k].add_(res.upload[k] - theta_r[k], alpha=f)
    m = dp.denominator if denominator is None else denominator
    noise = noise_like(shared, theta_r, seed=SEED, round_idx=round_idx, std=dp.noise_std) if dp.noise_std else None
    new = OrderedDict((k, theta_r[k] + ((total[k] + noise[k]) if noise is not None else total[k]) / float(m))
                      for k in shared)
    for k in w.manifest.buffer_keys:
        new[k] = theta_r[k]
    post = tiny(1, tied=False)
    post.load_state_(new)
    post.server_post_aggregate()
    return post.extract_shared(clone=True)


def _round_with(pred):
    p = pplan()
    return next(r for r in range(500) if pred(len(p.round_keys(r))))


def _dp_round(r, dp):
    keys = pplan().round_keys(r)
    srv = fresh_server()
    theta_r = srv.broadcast()
    rep = run_round(srv, [tiny(7, tied=False)], [BY_KEY[k] for k in keys], solver(lr=PEAK), round_idx=r, seed=SEED,
                    n_shards=2, dp=dp, n_sampled=len(keys))
    return srv, theta_r, rep, keys


def _close(a, b, keys, atol=1e-6):
    return all(torch.allclose(a[k], b[k], rtol=0, atol=atol) for k in keys)


@pytest.mark.parametrize("side", ["above", "below"])
def test_fixed_denominator_when_cohort_differs_from_m(side):
    r = _round_with((lambda n: n > M + 1) if side == "above" else (lambda n: 0 < n < M - 1))
    dp = DPConfig(0.35, 0.8, M)
    srv, theta_r, rep, keys = _dp_round(r, dp)
    assert rep.dp["denominator"] == M and rep.dp["n_sampled"] == len(keys) != M
    ref = reference_round(theta_r, keys, dp, r)
    assert _close(srv.adapter.extract_shared(), ref, srv.manifest.shared_keys)


# ------------------------------------------------------------------------------------------------ FLRun
def cfg(**kw):
    base = {"run_id": "poisson", "method": "FA", "seed": SEED, "lr_peak_local": PEAK, "solver": solver(), "n_shards": 2,
                "exposure_point": "end", "dropout_p": 0.0, "dp": DPConfig(0.3, 1.1, M), "endpoint_rounds": 12}
    base.update(kw)
    return RunConfig(**base)


def new_run(c, plan=None, ckpt=None, resume=False, get=BY_KEY.__getitem__, n_dec=N_DEC):
    args = (c, plan or pplan(), fresh_server(), get, n_dec)
    kw = {"workers": [tiny(7, tied=False)], "ckpt": ckpt}
    return FLRun.resume(*args, **kw) if resume else FLRun(*args, **kw)


def test_dp_requires_poisson_with_group_equal_to_m():
    for dp in (DPConfig(0.3, 1.1, M), DPConfig(0.3, 0.0, M), DPConfig(None, 0.0, M)):   # DP, FA_1024, S_CAL
        for sampling in ("uniform", "sweep"):
            with pytest.raises(ValueError, match="Poisson"):
                new_run(cfg(dp=dp), pplan(sampling=sampling))
        new_run(cfg(dp=dp))                                                            # poisson accepted
        for m in (M - 1, M + 2):                              # the denominator must be the plan's fixed m
            with pytest.raises(ValueError):
                new_run(cfg(dp=replace(dp, denominator=m)))
    for bad in (None, 6.5, "survivors"):                     # no variable / non-integer denominator
        with pytest.raises((ValueError, TypeError)):
            DPConfig(0.3, 1.0, bad)


def test_lr_schedule_is_expected_exposure_and_round_indexed():
    run = new_run(cfg())
    t = run.lr_table
    T, plan = 12, run.plan
    V = sum(int(valid_rows(c.examples).numel()) for c in CS)
    assert Fraction(*t["E_total_exact"]) == Fraction(T * M * 2 * V, len(CS)) and t["V_valid_rows"] == V
    assert t["E_total"] == float(Fraction(T * M * 2 * V, len(CS)))
    for r, row in enumerate(t["rows"]):
        realised = 2 * sum(int(valid_rows(BY_KEY[k].examples).numel()) for k in plan.round_keys(r))
        assert row[2] == realised and row[4] == PEAK * s_of_efe(6.0 * (r + 1) / T)
    assert t["E_realised_total"] == sum(x[2] for x in t["rows"]) and t["final_lr"] == PEAK * s_of_efe(6.0)
    assert t["cohort_sizes"] == [len(plan.round_keys(r)) for r in range(T)]
    run.run()
    assert run.counters.exposures == t["E_realised_total"]          # planned == actual enforced every round
    assert [lr for _, lr in run.lr_log] == [x[4] for x in t["rows"]]
    assert [e[0] for e in run.state.evals] == [] and run.state.cursor == T
    assert [run.units_after_row(i) for i in range(T)] == [(12 * (i + 1)) // T for i in range(T)]


def test_study_expected_exposure_is_six_efe():
    keys = synthetic_keys(131072)
    plan = pplan(keys, m=1024)
    nv = {k: 3 for k in keys}
    c = cfg(dp=DPConfig(1.0, 0.5642, 1024), endpoint_rounds=384)
    t = build_lr_table(plan, nv, 3 * len(keys), c)
    assert t["E_total"] == 6 * 3 * len(keys) and t["planned_efe"] == 6.0          # T q passes = 384 / 128 x 2
    assert t["final_lr"] == PEAK * s_of_efe(6.0) and t["n_rounds"] == 384
    assert abs(t["realised_planned_efe"] - 6.0) < 0.1


def test_lr_does_not_depend_on_the_clients_data():
    other = {c.key: c for c in clients(24, seed=708, invalid_frac=0.3)}
    other = {k: replace(c, key=kk) for kk, (k, c) in zip(sorted(BY_KEY), sorted(other.items()))}
    a, b = new_run(cfg()), new_run(cfg(), get=other.__getitem__,
                                   n_dec=sum(int(valid_rows(c.examples).numel()) for c in other.values()))
    assert [x[4] for x in a.lr_table["rows"]] == [x[4] for x in b.lr_table["rows"]]
    assert [x[2] for x in a.lr_table["rows"]] != [x[2] for x in b.lr_table["rows"]]


def test_flrun_poisson_dp_resume_bitwise(tmp_path):
    ref = new_run(cfg()).run()
    first = new_run(cfg(ckpt_every=4), ckpt=CheckpointManager(tmp_path, "poisson"))
    first.run(max_rounds=6)
    second = new_run(cfg(ckpt_every=4), ckpt=CheckpointManager(tmp_path, "poisson"), resume=True)
    assert second.state.cursor == 4
    second.run()
    assert second.server.digest() == ref.server.digest()
    assert second.participation_log == ref.participation_log and second.lr_log == ref.lr_log
    assert [x["n_sampled"] for x in ref.participation_log] == [len(pplan().round_keys(r)) for r in range(12)]
    s = ref.summary()["participation"]
    assert s["sampling"] == "poisson" and s["cohort_size_min"] < s["cohort_size_max"]


# ------------------------------------------------------------------------------------------------ empty rounds
SPARSE_M = 1                                                   # q = 1 / 24: P(empty) = (23/24)^24 ~ 0.36


def _empty_round(T=24):
    p = pplan(m=SPARSE_M)
    empties = [r for r in range(T) if not p.round_keys(r)]
    assert empties and empties[0] > 0
    return p, empties[0]


def test_empty_round_noise_only_for_z_and_noop_for_z0():
    p, r0 = _empty_round()
    for dp in (DPConfig(0.3, 1.1, SPARSE_M), DPConfig(0.3, 0.0, SPARSE_M), DPConfig(None, 0.0, SPARSE_M)):
        run = new_run(cfg(dp=dp, endpoint_rounds=24), plan=p)
        run.run(max_rounds=r0)
        before = run.server.broadcast()
        ledger = run.ledger.snapshot()
        out = run.step()
        after = run.server.broadcast()
        assert out["sampled"] == [] and run.ledger.snapshot() == ledger
        shared = run.server.manifest.shared_keys
        if dp.noise_std > 0:
            noise = noise_like(shared, before, seed=SEED, round_idx=r0, std=dp.noise_std)
            post = tiny(1, tied=False)
            post.load_state_(OrderedDict([(k, before[k] + noise[k] / float(SPARSE_M)) for k in shared] +
                                         [(k, before[k]) for k in run.server.manifest.buffer_keys]))
            post.server_post_aggregate()
            assert all(torch.equal(after[k], post.extract_shared()[k]) for k in shared)
            assert out["dp"]["noise_only"] and out["dp"]["denominator"] == SPARSE_M and not out["noop_round"]
            assert run.participation_log[-1]["noise_only"] is True
        else:
            assert all(torch.equal(after[k], before[k]) for k in before) and out["noop_round"]
            assert "noise_only" not in run.participation_log[-1]
        if dp.clip_norm is None:
            assert run.dp_norm_log[-1] == [r0, []]
        run.run()
        assert run.state.done and run.summary()["participation"]["empty_rounds"] >= 1


def test_empty_round_resume_bitwise(tmp_path):
    p, r0 = _empty_round()
    c = cfg(dp=DPConfig(0.3, 1.1, SPARSE_M), endpoint_rounds=24, ckpt_every=r0 + 1)
    ref = new_run(c, plan=p).run()
    a = new_run(c, plan=p, ckpt=CheckpointManager(tmp_path, "e"))
    a.run(max_rounds=r0 + 3)
    b = new_run(c, plan=p, ckpt=CheckpointManager(tmp_path, "e"), resume=True)
    assert b.state.cursor == r0 + 1
    b.run()
    assert b.server.digest() == ref.server.digest() and b.participation_log == ref.participation_log


# ------------------------------------------------------------------------------------------------ negative controls
@nc("dividing by the realised cohort size |C_r| instead of the fixed m")
def test_nc_variable_denominator():
    r = _round_with(lambda n: n > M + 1)
    dp = DPConfig(0.35, 0.0, M)
    srv, theta_r, _, keys = _dp_round(r, dp)
    bad = reference_round(theta_r, keys, dp, r, denominator=len(keys))
    assert _close(srv.adapter.extract_shared(), bad, srv.manifest.shared_keys)


@nc("fixed-size (uniform WOR) sampling has a constant cohort: it fails the binomial cohort-size test")
def test_nc_uniform_plan_fails_binomial_cohort_test():
    p = pplan(synthetic_keys(200), m=20, sampling="uniform")
    sizes = [len(p.round_positions(r)) for r in range(500)]
    want = 200 * 0.1 * 0.9
    assert abs(float(np.var(sizes, ddof=1)) - want) <= 5 * want * math.sqrt(2 / 499)


@nc("a DP-configured run with a uniform (WOR) plan is accepted")
def test_nc_dp_run_accepts_uniform_plan():
    try:
        new_run(cfg(), pplan(sampling="uniform"))
        accepted = True
    except ValueError:
        accepted = False
    assert accepted, "the WOR plan with z > 0 was refused (as it must be)"


@nc("an LR schedule read from the REALISED exposures depends on the clients' row counts")
def test_nc_realised_exposure_schedule_is_data_dependent():
    other = {c.key: c for c in clients(24, seed=708, invalid_frac=0.3)}
    other = {kk: replace(c, key=kk) for kk, (_k, c) in zip(sorted(BY_KEY), sorted(other.items()))}

    def realised_lrs(get):
        t = new_run(cfg(), get=get).lr_table                  # MUTANT: progress from realised E / E_realised_total
        E_tot = t["E_realised_total"]
        return [PEAK * s_of_efe(6.0 * x[3] / E_tot) for x in t["rows"]]
    assert realised_lrs(BY_KEY.__getitem__) == realised_lrs(other.__getitem__)
