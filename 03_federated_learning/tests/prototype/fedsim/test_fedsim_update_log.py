"""update_log.py + the server.Server observer hook + the runtime `update_norms` step field (CPU, tiny synthetic
families; no data).

Proves: (1) bit-identity with the observer ON vs OFF over 8-round synthetic FLRuns (FedAvg on the tied family with
drop-out, FedAdam on the frozen-table untied family, FedAvgM, DP-FedAvg z > 0 on a Poisson plan): the per-round theta
digests, the comm ledger, every step record (minus `update_norms`) and the content digest of every checkpoint written
(latest / best / endpoint) are identical; (2) the logged norms equal a hand computation from the observed (theta_r,
aggregate, theta_{r+1}) and theta_pre (float64), per group and in total, and cos with the previous step; (3) the DP
path logs post-noise server-side quantities only: no n_clients, no per-client value, and the logged aggregate is the
released state (step == Delta_agg for FedAvg on frozen tables); (4) rounds without a server step give
{"server_step": False}; two steps without a pop are refused; group_of maps the SASRec keys.
NC (strict xfail on AssertionError): an observer that touches the server weights breaks the bit-identity check.
"""
from __future__ import annotations

import math
from collections import OrderedDict

import torch
from fedsim_testkit import assert_raises, clients, nc, solver, tiny

from ppsi.fedsim.checkpoint import CheckpointManager, load_checkpoint
from ppsi.fedsim.client import valid_rows
from ppsi.fedsim.dp import DPConfig
from ppsi.fedsim.numerics import state_digest
from ppsi.fedsim.participation import ParticipationPlan
from ppsi.fedsim.runtime import FROZEN_ITEM_KEYS, FLRun, RunConfig, freeze_item_tables
from ppsi.fedsim.server import Server, ServerOptConfig
from ppsi.fedsim.update_log import (
    ITEM_TABLE_KEYS,
    UpdateNormObserver,
    group_of,
    noop_round_record,
    round_update_record,
)

ROUNDS = 8


def _cfg(**kw):
    base = {"run_id": "p2ul", "method": "FA", "seed": 5, "lr_peak_local": 0.05, "solver": solver(), "exposure_point": "end",
                "dropout_p": 0.1, "endpoint_rounds": ROUNDS, "n_shards": 2, "ckpt_every": 3}
    base.update(kw)
    return RunConfig(**base)


def _adapters(kind: str):
    if kind == "tied":
        return tiny(1), tiny(9)
    s, w = tiny(1, tied=False), tiny(9, tied=False)
    if kind == "frozen":
        freeze_item_tables(s, FROZEN_ITEM_KEYS)
        freeze_item_tables(w, FROZEN_ITEM_KEYS, source=s.broadcast_state())
    return s, w


def _run(tmp_path, name, cfg, *, kind="tied", observer=False, plan_kw=None, spy=None, bad=False):
    """One 8-round FLRun; returns (run, per-round digests, step records, observer). The global RNGs are reset first
    (every checkpoint carries them, and earlier runs in the process advance them), so two runs are comparable."""
    import random

    import numpy as np
    torch.manual_seed(20261004)
    np.random.seed(20261004)
    random.seed(20261004)
    cl = {c.key: c for c in clients(8)}
    pk = {"group_size": 3, "sampling": "sweep"} if plan_kw is None else plan_kw
    plan = ParticipationPlan.build(sorted(cl), seed=cfg.seed, manifest_hash="m", **pk)
    s, w = _adapters(kind)
    srv = Server(s, init_sha256=state_digest(s.broadcast_state()))
    obs = None
    if observer:
        obs = UpdateNormObserver(srv.adapter.broadcast_state(clone=True), srv.manifest.shared_keys)
        srv.observer = obs
        if spy is not None or bad:
            inner = obs

            def wrapped(theta_r, agg, theta_new):
                if spy is not None:
                    spy.append(tuple(OrderedDict((k, v.detach().clone()) for k, v in d.items())
                                     for d in (theta_r, agg, theta_new)))
                if bad:                                    # NC: an observer that touches the server weights
                    with torch.no_grad():
                        srv.adapter.shared_parameters()[0].mul_(1.0 + 2.0 ** -10)
                inner(theta_r, agg, theta_new)
            wrapped.pop = inner.pop
            srv.observer = wrapped
    n_dec = sum(int(valid_rows(c.examples).numel()) for c in cl.values())
    ck = CheckpointManager(tmp_path / name, cfg.run_id)
    marks = iter(range(100))
    run = FLRun(cfg, plan, srv, cl.__getitem__, n_dec, workers=[w], ckpt=ck,
                eval_fn=lambda _s: float(next(marks)))     # strictly increasing -> best.pt at every mark
    digests, recs = [], []
    while not run.state.done:
        recs.append(run.step())
        digests.append(srv.digest())
    return run, digests, recs, obs


def _ckpt_digests(run) -> dict:
    out = {}
    for p in sorted(run.ckpt.dir.glob("*.pt")):
        out[p.name] = load_checkpoint(p)["__content_digest__"]
    return out


def _strip(recs):
    return [{k: v for k, v in r.items() if k != "update_norms"} for r in recs]


CASES = {
    "fedavg_tied_dropout": {"kind": "tied", "cfg": {}},
    "fedadam_frozen": {"kind": "frozen", "cfg": {"server_opt": ServerOptConfig("fedadam", lr=0.01)}},
    "fedavgm_untied": {"kind": "untied", "cfg": {"server_opt": ServerOptConfig("fedavgm", lr=0.3)}},
    "dp_poisson_frozen": {"kind": "frozen", "cfg": {"dropout_p": 0.0, "dp": DPConfig(0.5, 0.8, 3)},
                              "plan_kw": {"group_size": 3, "sampling": "poisson"}},
}


def _check_identical(tmp_path, name, case, bad=False):
    off = _run(tmp_path, f"{name}_off", _cfg(**case["cfg"]), kind=case["kind"], plan_kw=case.get("plan_kw"))
    on = _run(tmp_path, f"{name}_on", _cfg(**case["cfg"]), kind=case["kind"], plan_kw=case.get("plan_kw"),
              observer=True, bad=bad)
    assert off[1] == on[1], f"{name}: theta trajectory differs with the observer on"
    assert off[0].ledger.snapshot() == on[0].ledger.snapshot(), name
    assert _strip(off[2]) == _strip(on[2]), name
    assert _ckpt_digests(off[0]) == _ckpt_digests(on[0]) and len(_ckpt_digests(on[0])) >= 2, name
    assert all("update_norms" not in r for r in off[2]) and all("update_norms" in r for r in on[2]), name
    return off, on


# ------------------------------------------------------------------------------------------------ (1) bit identity
def test_observer_on_off_bit_identical(tmp_path):
    for name, case in CASES.items():
        _off, on = _check_identical(tmp_path, name, case)
        un = [r["update_norms"] for r in on[2]]
        assert len(un) == ROUNDS and any(u["server_step"] for u in un), name


# ------------------------------------------------------------------------------------------------ (2) hand computation
def _sq(t):
    return float((t.double() ** 2).sum())


def test_norms_equal_hand_computation(tmp_path):
    spy = []
    run, _d, recs, obs = _run(tmp_path, "hand", _cfg(server_opt=ServerOptConfig("fedadam", lr=0.01)), kind="frozen",
                              observer=True, spy=spy)
    keys = list(run.server.manifest.shared_keys)
    assert not set(FROZEN_ITEM_KEYS) & set(keys) and keys == list(obs.keys)
    pre = obs.theta_pre_T
    stepped = [r["update_norms"] for r in recs if r["update_norms"]["server_step"]]
    assert len(stepped) == len(spy) >= 3
    prev = None
    for rec, (th, agg, new) in zip(stepped, spy):
        groups = {}
        for k in keys:
            g = groups.setdefault(group_of(k), [0, 0.0, 0.0, 0.0, 0.0])
            g[0] += th[k].numel()
            g[1] += _sq(agg[k] - th[k])
            g[2] += _sq(new[k] - th[k])
            g[3] += _sq(new[k] - pre[k])
            g[4] += _sq(pre[k])
        assert sorted(groups) == sorted(rec["groups"])
        for name, (n, a, s, p, p0) in groups.items():
            got = rec["groups"][name]
            assert got["numel"] == n
            for f, want in (("agg_delta_l2", a), ("step_l2", s), ("pre_dist_l2", p), ("pre_l2", p0)):
                assert math.isclose(got[f], math.sqrt(want), rel_tol=1e-12, abs_tol=1e-15), (name, f)
        tot = rec["total_T"]
        ta = sum(g[1] for g in groups.values())
        assert math.isclose(tot["agg_delta_l2"], math.sqrt(ta), rel_tol=1e-12)
        assert math.isclose(tot["rel_pre_dist"], math.sqrt(sum(g[3] for g in groups.values())) /
                            math.sqrt(sum(g[4] for g in groups.values())), rel_tol=1e-12)
        d = {k: (agg[k] - th[k]).double() for k in keys}
        if prev is None:
            assert rec["cos_prev"] is None and rec["cos_prev_available"] is False
        else:
            dot = sum(float((d[k] * prev[k]).sum()) for k in keys)
            nn = math.sqrt(sum(float((d[k] ** 2).sum()) for k in keys) * sum(float((prev[k] ** 2).sum()) for k in keys))
            assert math.isclose(rec["cos_prev"], dot / nn, rel_tol=1e-9, abs_tol=1e-12)
        prev = d
        assert rec["n_clients"] >= 1 and rec["finite"] is True


# ------------------------------------------------------------------------------------------------ (3) DP: post-noise only
ALLOWED_DP = {"server_step", "total_T", "groups", "cos_prev", "cos_prev_available", "n_keys_T", "finite"}
GROUP_FIELDS = {"numel", "agg_delta_l2", "step_l2", "pre_dist_l2", "pre_l2", "rel_pre_dist"}


def test_dp_path_logs_post_noise_server_quantities_only(tmp_path):
    spy = []
    case = CASES["dp_poisson_frozen"]
    run, _d, recs, _obs = _run(tmp_path, "dp", _cfg(**case["cfg"]), kind="frozen", plan_kw=case["plan_kw"],
                               observer=True, spy=spy)
    assert run.cfg.dp.noise_multiplier > 0
    stepped = [r["update_norms"] for r in recs if r["update_norms"]["server_step"]]
    assert stepped and len(stepped) == len(spy)
    for r in recs:
        u = r["update_norms"]
        assert "n_clients" not in u, "z > 0: the cohort size is not logged (06L_E E1)"
        assert set(u) <= ALLOWED_DP | {"server_step"}
        for g in (u.get("groups") or {}).values():
            assert set(g) == GROUP_FIELDS
    for u, (th, agg, new) in zip(stepped, spy):
        # FedAvg on frozen tables: the released theta IS the aggregate the observer saw (post-noise), bit for bit
        assert all(torch.equal(agg[k], new[k]) for k in run.server.manifest.shared_keys)
        assert u["total_T"]["step_l2"] == u["total_T"]["agg_delta_l2"]
    # z = 0 runs (and FedAvg runs) do log the clients aggregated
    off = _run(tmp_path, "z0", _cfg(dropout_p=0.0, dp=DPConfig(0.5, 0.0, 3)), kind="frozen",
               plan_kw=case["plan_kw"], observer=True)
    assert all("n_clients" in r["update_norms"] for r in off[2])


# ------------------------------------------------------------------------------------------------ (4) records / refusals
def test_noop_rounds_pop_and_groups(tmp_path):
    _run_obj, _d, recs, obs = _run(tmp_path, "noop", _cfg(dropout_p=0.7), kind="tied", observer=True,
                              plan_kw={"group_size": 1, "sampling": "sweep"})
    un = [r["update_norms"] for r in recs]
    empty = [u for u, r in zip(un, recs) if not r["survived"]]
    assert empty and all(u == dict(noop_round_record(), n_clients=0) for u in empty)
    assert all(u["server_step"] for u, r in zip(un, recs) if r["survived"])
    assert obs.pop() is None
    s = tiny(1)
    o = UpdateNormObserver(s.broadcast_state(clone=True), s.manifest.shared_keys)
    st = s.broadcast_state(clone=True)
    o(st, st, st)
    assert_raises(RuntimeError, o, st, st, st)            # never two steps merged silently
    rec = o.pop()
    assert rec["total_T"]["agg_delta_l2"] == 0.0 and rec["total_T"]["rel_pre_dist"] == 0.0
    assert_raises(KeyError, UpdateNormObserver, {}, ["enc.weight"])
    want = {"blocks.0.qkv.weight": "block0.qkv", "blocks.1.out.bias": "block1.out", "blocks.0.ff1.weight": "block0.ff1",
            "blocks.1.ff2.weight": "block1.ff2", "blocks.0.ln1.weight": "layernorm", "blocks.1.ln2.bias": "layernorm",
            "final_norm.weight": "layernorm", "input_norm.bias": "layernorm", "side_norm.weight": "layernorm",
            "user_norm.bias": "layernorm", "side_gate": "gates", "user_gate": "gates", "pos_embed.weight": "pos_embed",
            "side_proj.weight": "side_proj", "user_mlp.weight": "user_mlp", "user_proj.bias": "user_proj",
            "side_embeds.brand.weight": "side_embeds", "numflag_proj.weight": "numflag_proj",
            "item_embed.weight": "item_tables", "output_embed": "item_tables", "output_bias": "item_tables"}
    assert {k: group_of(k) for k in want} == want and ITEM_TABLE_KEYS == FROZEN_ITEM_KEYS
    th = OrderedDict(w=torch.tensor([1.0, 2.0]))
    rec, delta = round_update_record(th, {"w": torch.tensor([1.0, 4.0])}, {"w": torch.tensor([1.0, 3.0])},
                                     {"w": torch.tensor([0.0, 2.0])})
    assert rec["total_T"]["agg_delta_l2"] == 2.0 and rec["total_T"]["step_l2"] == 1.0
    assert rec["total_T"]["pre_dist_l2"] == math.sqrt(2.0) and torch.equal(delta["w"], torch.tensor([0.0, 2.0]))
    rec2, _ = round_update_record(th, {"w": torch.tensor([1.0, 0.0])}, th, {"w": torch.tensor([0.0, 2.0])}, delta)
    assert rec2["cos_prev"] == -1.0


@nc("an observer that modifies the server weights must break the on/off bit-identity check")
def test_nc_observer_touching_weights_breaks_identity(tmp_path):
    _check_identical(tmp_path, "nc", CASES["fedavg_tied_dropout"], bad=True)
