"""Seeded client drop-out, uniform per-round sampling, the round-count endpoint; defaults unchanged.

Proves: drop-out depends only on (run seed, round) and is identical for FA / FP / PF; survivors are aggregated exactly
as the FedAvg driver aggregates them (bitwise); a 0-survivor round is a recorded no-op; the round-count LR table ends
at peak * s(6.0) with normalized progress; an interrupted drop-out run resumes bitwise (theta, counters, log); uniform
sampling is without replacement and deterministic; with every new option off the config digest, the plan description
and the checkpoint payload keys are exactly the plain ones. Negative controls: unseeded drop-out, aggregating the
dropped clients too.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

import numpy as np
import pytest
from fedsim_testkit import clients, nc, solver, tiny

from ppsi.fedsim.checkpoint import CheckpointManager
from ppsi.fedsim.client import PFConfig
from ppsi.fedsim.numerics import state_digest
from ppsi.fedsim.participation import ParticipationPlan, dropout_split
from ppsi.fedsim.personal import PersonalStore
from ppsi.fedsim.runtime import FLRun, RunConfig
from ppsi.fedsim.schedule import s_of_efe
from ppsi.fedsim.server import Server, run_round

SEED = 2026
PEAK = 0.05
CS = clients(24, seed=303, invalid_frac=0.1)
BY_KEY = {c.key: c for c in CS}
N_DEC = sum(int((c.examples["target_class"] >= 0).sum()) for c in CS)


def theta0():
    a = tiny(1, tied=False)
    a.server_post_aggregate()
    return a.broadcast_state(clone=True), state_digest(a.broadcast_state())


THETA0, INIT = theta0()


def fresh_server():
    a = tiny(1, tied=False)
    a.load_state_(THETA0)
    return Server(a, init_sha256=INIT)


def plan(sampling="sweep", group=6):
    return ParticipationPlan.build(list(BY_KEY), seed=SEED, manifest_hash="dropout-tests", group_size=group,
                                   sampling=sampling)


def cfg(method="FA", **kw):
    base = {"run_id": f"dropout-{method}", "method": method, "seed": SEED, "lr_peak_local": PEAK, "solver": solver(),
                "n_shards": 2, "exposure_point": "end", "dropout_p": 0.1, "endpoint_rounds": 12}
    if method == "FP":
        base["mu"] = 0.01
    if method == "PF":
        base["pf"] = PFConfig(lr=PEAK)
    base.update(kw)
    return RunConfig(**base)


def new_run(c, p, ckpt=None, resume=False):
    kw = {"workers": [tiny(7, tied=False)], "ckpt": ckpt, "store": PersonalStore(8) if c.method == "PF" else None}
    args = (c, p, fresh_server(), BY_KEY.__getitem__, N_DEC)
    return FLRun.resume(*args, **kw) if resume else FLRun(*args, **kw)


# ------------------------------------------------------------------------------------------------ drop-out draw
def test_dropout_seeded_by_seed_and_round_only():
    keys = [f"k{i}" for i in range(64)]
    a = dropout_split(keys, seed=SEED, round_idx=5, p=0.1)
    assert a == dropout_split(keys, seed=SEED, round_idx=5, p=0.1)
    assert a != dropout_split(keys, seed=SEED, round_idx=6, p=0.1)
    assert a != dropout_split(keys, seed=SEED + 1, round_idx=5, p=0.1)
    surv, drop = a
    assert surv + drop != [] and sorted(surv + drop) == sorted(keys) and not set(surv) & set(drop)
    assert [k for k in keys if k in set(surv)] == surv, "survivors keep the logical order"
    assert dropout_split(keys, seed=SEED, round_idx=5, p=0.0) == (keys, [])


def test_dropout_rate_and_independence():
    keys = [f"k{i}" for i in range(64)]
    n_drop = [len(dropout_split(keys, seed=SEED, round_idx=r, p=0.1)[1]) for r in range(4000)]
    rate = sum(n_drop) / (64 * 4000)
    assert abs(rate - 0.1) < 0.004                         # 5 sigma ~ 0.0024 * ... (sd of the mean ~ 0.00059)
    assert abs(np.var(n_drop) - 64 * 0.1 * 0.9) < 0.6       # binomial(64, 0.1) variance 5.76


def test_dropout_identical_across_methods():
    p = plan()
    logs = []
    for m in ("FA", "FP", "PF"):
        run = new_run(cfg(m), p).run(max_rounds=5)
        logs.append(run.participation_log)
    assert logs[0] == logs[1] == logs[2]
    assert sum(len(x["dropped"]) for x in logs[0]) > 0


# ------------------------------------------------------------------------------------------------ aggregation
def test_survivors_aggregated_exactly_as_fedavg():
    p = plan()
    run = new_run(cfg(), p)
    ref_srv, ref_workers = fresh_server(), [tiny(7, tied=False)]
    for r in range(4):
        out = run.step()
        assert out["sampled"] == p.round_keys(r)
        surv, drop = dropout_split(p.round_keys(r), seed=SEED, round_idx=r, p=0.1)
        assert out["survived"] == surv and out["dropped"] == drop
        lr = run.lr_log[-1][1]
        from dataclasses import replace
        run_round(ref_srv, ref_workers, [BY_KEY[k] for k in surv], replace(solver(), lr=lr), round_idx=r,
                  seed=SEED, n_shards=2)
        assert run.server.digest() == ref_srv.digest(), f"round {r}"


def _all_dropped_round(p, prob):
    for r in range(200):
        s, _d = dropout_split(p.round_keys(r), seed=SEED, round_idx=r, p=prob)
        if not s:
            return r
    raise AssertionError("no all-dropped round found")


def test_zero_survivor_round_is_recorded_noop():
    p = plan(group=2)
    c = cfg(dropout_p=0.6, endpoint_rounds=30)
    r0 = _all_dropped_round(p, 0.6)
    assert r0 < 30
    run = new_run(c, p)
    run.run(max_rounds=r0)
    before = (run.server.digest(), run.counters.exposures, run.server.round)
    out = run.step()
    assert out["noop_round"] and out["survived"] == [] and len(out["dropped"]) == 2 and out["weight"] == 0
    assert (run.server.digest(), run.counters.exposures, run.server.round) == before
    assert run.participation_log[-1]["noop"] is True


# ------------------------------------------------------------------------------------------------ LR table / endpoint
def test_round_count_table_normalized_endpoint():
    p = plan()
    run = new_run(cfg(), p)
    t = run.lr_table
    assert t["n_rounds"] == 12 and t["endpoint_rounds"] == 12
    assert t["E_total"] == sum(r[2] for r in t["rows"])
    assert t["final_lr"] == PEAK * s_of_efe(6.0)
    exp_n = [2 * sum(int((BY_KEY[k].examples["target_class"] >= 0).sum())
                     for k in dropout_split(p.round_keys(r), seed=SEED, round_idx=r, p=0.1)[0]) for r in range(12)]
    assert [r[2] for r in t["rows"]] == exp_n
    run.run()
    assert run.state.done and run.state.cursor == 12
    assert run.counters.exposures == t["E_total"]
    assert run.summary()["participation"]["rounds"] == 12


def test_resume_bitwise_with_dropout(tmp_path):
    p = plan()
    ref = new_run(cfg(), p).run()
    first = new_run(cfg(ckpt_every=3), p, CheckpointManager(tmp_path, "r"))
    first.run(max_rounds=7)                                   # latest.pt at round 6
    second = new_run(cfg(ckpt_every=3), p, CheckpointManager(tmp_path, "r"), resume=True)
    assert second.state.cursor == 6 and len(second.participation_log) == 6
    second.run()
    assert second.server.digest() == ref.server.digest()
    assert second.participation_log == ref.participation_log
    assert second.counters.summary() == ref.counters.summary()


# ------------------------------------------------------------------------------------------------ uniform sampling
def test_uniform_sampling_without_replacement_and_seeded():
    p = plan("uniform", group=8)
    q = plan("uniform", group=8)
    seen = set()
    for r in range(30):
        ks = p.round_keys(r)
        assert len(ks) == 8 and len(set(ks)) == 8 and ks == q.round_keys(r)
        seen.add(tuple(ks))
    assert len(seen) == 30, "rounds are independent draws"
    counts = np.zeros(24)
    for r in range(3000):
        for i in p.round_positions(r):
            counts[i] += 1
    assert abs(counts.mean() - 3000 * 8 / 24) < 1e-9 and counts.std() / counts.mean() < 0.05
    assert p.describe()["sampling_mode"] == "uniform" and p.digest() != plan("sweep", 8).digest()


def test_uniform_sampling_requires_round_endpoint():
    with pytest.raises(ValueError):
        new_run(cfg(dropout_p=0.0, endpoint_rounds=None), plan("uniform"))
    with pytest.raises(ValueError):
        cfg(endpoint_rounds=None)                              # drop-out needs the round-count endpoint


# ------------------------------------------------------------------------------------------------ defaults unchanged
def _old_digest(c: RunConfig) -> str:
    d = asdict(c)
    d.pop("ckpt_every")
    for k in ("dropout_p", "dp", "endpoint_rounds"):
        d.pop(k)
    for k in ("server_opt", "lr_shape"):              # default-valued fields, not part of the digest
        d.pop(k, None)
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()


def test_defaults_bit_identical_digests_and_payload():
    c = RunConfig(run_id="x", method="FA", seed=1, lr_peak_local=0.1, solver=solver(), exposure_point="end")
    assert c.digest() == _old_digest(c)
    assert "sampling_mode" not in plan().describe()
    assert plan().describe()["sampling"] == "seeded permutation sweeps without replacement (PCG64), fixed consecutive groups"
    run = new_run(RunConfig(run_id="x", method="FA", seed=SEED, lr_peak_local=PEAK, solver=solver(),
                            exposure_point="end"), ParticipationPlan.build(list(BY_KEY), seed=SEED,
                                                                          manifest_hash="dropout-tests",
                                                                          group_size=8))
    assert "round_count_state" not in run.state_dict() and "participation" not in run.summary()
    assert c.digest() != cfg().digest()


# ------------------------------------------------------------------------------------------------ negative controls
@nc("drop-out drawn from an unseeded RNG is not reproducible")
def test_nc_nondeterministic_dropout():
    keys = [f"k{i}" for i in range(64)]

    def bad(ks):
        u = np.random.default_rng().random(len(ks))
        return [k for k, x in zip(ks, u) if x >= 0.1]
    assert bad(keys) == bad(keys) == dropout_split(keys, seed=SEED, round_idx=0, p=0.1)[0]


@nc("aggregating the dropped clients too differs from the survivors-only reference")
def test_nc_dropped_clients_aggregated():
    from dataclasses import replace
    p = plan()
    r = next(r for r in range(50) if dropout_split(p.round_keys(r), seed=SEED, round_idx=r, p=0.1)[1])
    run = new_run(cfg(), p)
    run.run(max_rounds=r)
    bad_srv = fresh_server()
    bad_srv.adapter.load_state_(run.server.broadcast())
    run.step()
    run_round(bad_srv, [tiny(7, tied=False)], [BY_KEY[k] for k in p.round_keys(r)],
              replace(solver(), lr=run.lr_log[-1][1]), round_idx=r, seed=SEED, n_shards=2)
    assert bad_srv.digest() == run.server.digest()
