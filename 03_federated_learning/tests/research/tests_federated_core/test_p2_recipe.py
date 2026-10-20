"""P2-02 (PREG2 §5.2-§5.3; PHASE2_IMPLEMENTATION_PLAN P2-02) — the fl-library side of the Phase-2 recipe plumbing (CPU,
tiny synthetic families; no data).

Proves: (1) server LR eta_s = 0 leaves theta bit-identical over 3 rounds for FedAvg (= FedAvgM beta 0 at LR 0, the
launch_fl.p2_server_opt mapping), FedAvgM (beta 0.9) and FedAdam (the C0 anchor; tied family = no recentering, and the
frozen-table untied family whose recentering hooks are no-ops); (2) FedAvg at eta_s 0.3 is exactly
theta_r + 0.3 (aggregate - theta_r) evaluated by the fl.server FedAvgM(beta 0) ops; (3) the item-LR multiplier
(fl.client.ItemLRSolver) puts the item tables into their own lr x m group(s) (tag lr_mult), _set_lr keeps the ratio, the
other parameters move exactly as with the plain solver, and a frozen-table adapter refuses a multiplier; a plain
LocalSolver builds the pre-P2 groups; (4) the Phase-1 fl RunConfig digests of FA / FP / PF / FA_Q8 / FA_1024 / DP8 /
S_CAL / FA_WARM / FA_WARM_FROZEN (MAIN + CALIBRATION) / DP_WARM_FROZEN equal the reference values MEASURED on the
pre-P2 code (launch_fl.py c9fdc8c0, runtime.py e627e943, client.py 8a41e6d6; FA / S_CAL default = the committed
tests_int/test_fl05_launch BASELINE 66770288 / 8fb83823), LocalSolver's field set is unchanged (so LORecipe digests are
too), RunConfig.from_dict / digest drop a None item_lr_mult (digest guard) and carry a real one.
NC (strict xfail): a digest that keeps a None item_lr_mult key changes every Phase-1 digest.
"""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import asdict, fields, replace

import torch

from fl.client import ITEM_LR_KEYS, ItemLRSolver, LocalSolver, _set_lr, client_update, make_optimizer, solver_from_dict
from fl.numerics import state_digest
from fl.runtime import FROZEN_ITEM_KEYS, RunConfig, freeze_item_tables
from fl.server import Server, ServerOptConfig, run_round
from fl_testkit import assert_raises, clients, nc, one_client, solver, tiny

# MEASURED on the pre-P2 code (scratchpad p2_pins.py; peak 0.000711, fused True, theta_pre identity = test_fl15.IDENT;
# s4 = the registered S4 point fedadam 0.01 / fl_const / sgd_m 0.1 / passes 1; sf 0.1 = the S8 incumbent cell)
PINS = [json.loads(x) for x in r"""
{"cfg":{"dp":null,"dropout_p":0.1,"endpoint_efe":6.0,"endpoint_rounds":6827,"exposure_point":"end","lr_peak_local":0.000711,"max_attempts":2,"method":"FA","mu":0.0,"n_shards":2,"pf":null,"seed":2026,"solver":{"batch_size":16,"clip":1.0,"fused":true,"lr":0.000711,"passes":2,"weight_decay":1e-05}},"digest":"66770288acf78c816e98bf65c78896acb4d16b41767ef6c7809af1fe13c5c74c","key":"FA|default|wf=None|sf=None|MAIN","run_id":"FL_FA_SMALL_s2026"}
{"cfg":{"dp":{"clip_norm":null,"denominator":1024,"noise_multiplier":0.0},"dropout_p":0.0,"endpoint_efe":6.0,"endpoint_rounds":384,"exposure_point":"end","lr_peak_local":0.000711,"max_attempts":2,"method":"FA","mu":0.0,"n_shards":2,"pf":null,"seed":2026,"solver":{"batch_size":16,"clip":1.0,"fused":true,"lr":0.000711,"passes":2,"weight_decay":1e-05}},"digest":"8fb83823caeb687137c9b221d50228366a3ad45c7564b44b2a5f403106e727b3","key":"S_CAL|default|wf=None|sf=None|MAIN","run_id":"FL_S_CAL_SMALL_s2026"}
{"cfg":{"dp":null,"dropout_p":0.1,"endpoint_efe":6.0,"endpoint_rounds":13654,"exposure_point":"end","lr_peak_local":0.1,"lr_shape":"fl_const","max_attempts":2,"method":"FA","mu":0.0,"n_shards":2,"pf":null,"seed":2026,"server_opt":{"beta1":0.9,"beta2":0.99,"lr":0.01,"name":"fedadam","tau":0.001,"v_init":"tau_squared"},"solver":{"batch_size":16,"clip":1.0,"fused":null,"lr":0.1,"optimizer":"sgd_m","passes":1,"weight_decay":0.0}},"digest":"7ba286d659a25f1eeaa744ff08bcd26f63f0023b98a13b51fd78022e63f056cb","key":"FA|s4|wf=None|sf=None|MAIN","run_id":"FL_FA_SMALL_s2026"}
{"cfg":{"dp":null,"dropout_p":0.1,"endpoint_efe":6.0,"endpoint_rounds":13654,"exposure_point":"end","lr_peak_local":0.1,"lr_shape":"fl_const","max_attempts":2,"method":"FP","mu":0.01,"n_shards":2,"pf":null,"seed":2026,"server_opt":{"beta1":0.9,"beta2":0.99,"lr":0.01,"name":"fedadam","tau":0.001,"v_init":"tau_squared"},"solver":{"batch_size":16,"clip":1.0,"fused":null,"lr":0.1,"optimizer":"sgd_m","passes":1,"weight_decay":0.0}},"digest":"1088b133fb3f8075b2f7921b7744577f531f95b2b483087d4fdb8d5579e8840e","key":"FP|s4|wf=None|sf=None|MAIN","run_id":"FL_FP_SMALL_s2026"}
{"cfg":{"dp":null,"dropout_p":0.1,"endpoint_efe":6.0,"endpoint_rounds":13654,"exposure_point":"end","lr_peak_local":0.1,"lr_shape":"fl_const","max_attempts":2,"method":"PF","mu":0.0,"n_shards":2,"pf":{"clip":1.0,"lam":0.0001,"lr":0.000711},"seed":2026,"server_opt":{"beta1":0.9,"beta2":0.99,"lr":0.01,"name":"fedadam","tau":0.001,"v_init":"tau_squared"},"solver":{"batch_size":16,"clip":1.0,"fused":null,"lr":0.1,"optimizer":"sgd_m","passes":1,"weight_decay":0.0}},"digest":"fcc4bd45864b546d625cef0c8d1ed2df27bacdfc7a1be555aa75406b82f1454d","key":"PF|s4|wf=None|sf=None|MAIN","run_id":"FL_PF_SMALL_s2026"}
{"cfg":{"dp":null,"dropout_p":0.1,"endpoint_efe":6.0,"endpoint_rounds":13654,"exposure_point":"end","lr_peak_local":0.1,"lr_shape":"fl_const","max_attempts":2,"method":"FA","mu":0.0,"n_shards":2,"pf":null,"seed":2026,"server_opt":{"beta1":0.9,"beta2":0.99,"lr":0.01,"name":"fedadam","tau":0.001,"v_init":"tau_squared"},"solver":{"batch_size":16,"clip":1.0,"fused":null,"lr":0.1,"optimizer":"sgd_m","passes":1,"weight_decay":0.0}},"digest":"ec672f06dbd75320a2d92d2996cc5aa415896fd1c639ad3ce38dce3b0d75cd32","key":"FA_Q8|s4|wf=None|sf=None|MAIN","run_id":"FL_FA_Q8_SMALL_s2026"}
{"cfg":{"dp":{"clip_norm":0.2843,"denominator":1024,"noise_multiplier":0.0},"dropout_p":0.0,"endpoint_efe":6.0,"endpoint_rounds":384,"exposure_point":"end","lr_peak_local":0.1,"lr_shape":"fl_const","max_attempts":2,"method":"FA","mu":0.0,"n_shards":2,"pf":null,"seed":2026,"server_opt":{"beta1":0.9,"beta2":0.99,"lr":0.01,"name":"fedadam","tau":0.001,"v_init":"tau_squared"},"solver":{"batch_size":16,"clip":1.0,"fused":null,"lr":0.1,"optimizer":"sgd_m","passes":2,"weight_decay":0.0}},"digest":"942b7bde7b1f1c2caf06dc6cfd8c07c396f009c712f06567096af5f25a1b9009","key":"FA_1024|s4|wf=None|sf=None|MAIN","run_id":"FL_FA_1024_SMALL_s2026"}
{"cfg":{"dp":{"clip_norm":0.2843,"denominator":1024,"noise_multiplier":0.5642},"dropout_p":0.0,"endpoint_efe":6.0,"endpoint_rounds":384,"exposure_point":"end","lr_peak_local":0.1,"lr_shape":"fl_const","max_attempts":2,"method":"FA","mu":0.0,"n_shards":2,"pf":null,"seed":2026,"server_opt":{"beta1":0.9,"beta2":0.99,"lr":0.01,"name":"fedadam","tau":0.001,"v_init":"tau_squared"},"solver":{"batch_size":16,"clip":1.0,"fused":null,"lr":0.1,"optimizer":"sgd_m","passes":2,"weight_decay":0.0}},"digest":"7f574bdd499ea583c221fa767b5bcc9c166840622eac0c35714aeb66103c12d3","key":"DP8|s4|wf=None|sf=None|MAIN","run_id":"FL_DP8_SMALL_s2026"}
{"cfg":{"dp":{"clip_norm":null,"denominator":1024,"noise_multiplier":0.0},"dropout_p":0.0,"endpoint_efe":6.0,"endpoint_rounds":384,"exposure_point":"end","lr_peak_local":0.1,"lr_shape":"fl_const","max_attempts":2,"method":"FA","mu":0.0,"n_shards":2,"pf":null,"seed":2026,"server_opt":{"beta1":0.9,"beta2":0.99,"lr":0.01,"name":"fedadam","tau":0.001,"v_init":"tau_squared"},"solver":{"batch_size":16,"clip":1.0,"fused":null,"lr":0.1,"optimizer":"sgd_m","passes":2,"weight_decay":0.0}},"digest":"19d315d9d4d942dc9965c232d6a60680cdf669dea97a144842241b6a3f545dbc","key":"S_CAL|s4|wf=None|sf=None|MAIN","run_id":"FL_S_CAL_SMALL_s2026"}
{"cfg":{"dp":null,"dropout_p":0.1,"endpoint_efe":6.0,"endpoint_rounds":13654,"exposure_point":"end","lr_peak_local":0.1,"lr_shape":"fl_const","max_attempts":2,"method":"FA","mu":0.0,"n_shards":2,"pf":null,"seed":2026,"server_opt":{"beta1":0.9,"beta2":0.99,"lr":0.001,"name":"fedadam","tau":0.001,"v_init":"tau_squared"},"solver":{"batch_size":16,"clip":1.0,"fused":null,"lr":0.1,"optimizer":"sgd_m","passes":1,"weight_decay":0.0}},"digest":"b1a605721308eebbd1ed6a06d6ac4808ad971d70556a65e0460d976ae1af1c0b","key":"FA_WARM|s4|wf=None|sf=0.1|MAIN","run_id":"FL_FA_WARM_SMALL_s2026"}
{"cfg":{"dp":null,"dropout_p":0.1,"endpoint_efe":6.0,"endpoint_rounds":13654,"exposure_point":"end","lr_peak_local":0.1,"lr_shape":"fl_const","max_attempts":2,"method":"FA","mu":0.0,"n_shards":2,"pf":null,"seed":2026,"server_opt":{"beta1":0.9,"beta2":0.99,"lr":0.001,"name":"fedadam","tau":0.001,"v_init":"tau_squared"},"solver":{"batch_size":16,"clip":1.0,"fused":null,"lr":0.1,"optimizer":"sgd_m","passes":1,"weight_decay":0.0}},"digest":"d52e93152724d633180d0dfae509ba773836ec45deeb1c3d5a9ed8fe747387ff","key":"FA_WARM_FROZEN|s4|wf=None|sf=0.1|MAIN","run_id":"FL_FA_WARM_FROZEN_SMALL_s2026"}
{"cfg":{"dp":null,"dropout_p":0.1,"endpoint_efe":6.0,"endpoint_rounds":3414,"exposure_point":"end","lr_peak_local":0.1,"lr_shape":"fl_const","max_attempts":2,"method":"FA","mu":0.0,"n_shards":2,"pf":null,"seed":2026,"server_opt":{"beta1":0.9,"beta2":0.99,"lr":0.001,"name":"fedadam","tau":0.001,"v_init":"tau_squared"},"solver":{"batch_size":16,"clip":1.0,"fused":null,"lr":0.1,"optimizer":"sgd_m","passes":1,"weight_decay":0.0}},"digest":"6f18a9efa46bf7a23ad0bef36e0738c0a6e6ada26d4d2a6959a234fe1b90a523","key":"FA_WARM_FROZEN|s4|wf=None|sf=0.1|CALIBRATION","run_id":"FL_FA_WARM_FROZEN_SMALL_s2026"}
{"cfg":{"dp":{"clip_norm":0.2843,"denominator":1024,"noise_multiplier":0.5642},"dropout_p":0.0,"endpoint_efe":6.0,"endpoint_rounds":384,"exposure_point":"end","lr_peak_local":0.1,"lr_shape":"fl_const","max_attempts":2,"method":"FA","mu":0.0,"n_shards":2,"pf":null,"seed":2026,"server_opt":{"beta1":0.9,"beta2":0.99,"lr":0.001,"name":"fedadam","tau":0.001,"v_init":"tau_squared"},"solver":{"batch_size":16,"clip":1.0,"fused":null,"lr":0.1,"optimizer":"sgd_m","passes":2,"weight_decay":0.0}},"digest":"ea605fd1a4f19cd92ef5245e2e0c1673b9671efd04eaeaad9f843f0e2f9211cb","key":"DP_WARM_FROZEN|s4|wf=None|sf=0.1|MAIN","run_id":"FL_DP_WARM_FROZEN_SMALL_s2026"}
""".strip().splitlines()]
LOCALSOLVER_FIELDS = ["lr", "betas", "eps", "weight_decay", "clip", "batch_size", "passes", "optimizer", "fused"]


def _server(kind):
    if kind == "tied":
        s, w = tiny(1), tiny(9)
    else:
        s, w = tiny(1, tied=False), tiny(9, tied=False)
        freeze_item_tables(s, FROZEN_ITEM_KEYS)
        freeze_item_tables(w, FROZEN_ITEM_KEYS, source=s.broadcast_state())
    return Server(s, init_sha256=state_digest(s.broadcast_state())), w


SERVER_ZERO = {"fedavg (fedavgm beta 0, lr 0)": ServerOptConfig("fedavgm", lr=0.0, beta1=0.0),
               "fedavgm beta 0.9, lr 0": ServerOptConfig("fedavgm", lr=0.0),
               "fedadam lr 0": ServerOptConfig("fedadam", lr=0.0)}


# ------------------------------------------------------------------------------------------------ (1) eta_s = 0
def test_server_lr_zero_leaves_theta_bit_identical():
    for kind in ("tied", "frozen"):
        for name, cfg in SERVER_ZERO.items():
            srv, w = _server(kind)
            srv.attach_server_opt(cfg)
            d0 = srv.digest()
            for r in range(3):
                rep = run_round(srv, [w], clients(6, seed=3 + r), solver(), round_idx=r, seed=5)
                assert rep.weight > 0
            assert srv.digest() == d0 and srv.round == 3, (kind, name)
            assert srv.server_opt.steps == 3


# ------------------------------------------------------------------------------------------------ (2) FedAvg eta_s 0.3
def test_fedavg_server_lr_03_is_theta_plus_03_delta_exactly():
    srv, w = _server("tied")
    srv.attach_server_opt(ServerOptConfig("fedavgm", lr=0.3, beta1=0.0))     # = launch_fl.p2_server_opt(FedAvg 0.3)
    seen = []

    def spy(th, agg, new):
        seen.append(tuple(OrderedDict((k, v.clone()) for k, v in d.items()) for d in (th, agg, new)))
    srv.observer = spy
    for r in range(2):
        run_round(srv, [w], clients(6, seed=3 + r), solver(), round_idx=r, seed=5)
    assert len(seen) == 2
    for th, agg, new in seen:
        for k in srv.manifest.shared_keys:
            delta = torch.sub(agg[k], th[k])
            want = torch.add(th[k], torch.mul(torch.add(torch.mul(torch.zeros_like(delta), 0.0), delta), 0.3))
            assert torch.equal(new[k], want), k
            assert torch.allclose(new[k], th[k] + 0.3 * (agg[k] - th[k]), rtol=0, atol=1e-7)
        assert any(not torch.equal(new[k], agg[k]) for k in srv.manifest.shared_keys)   # not plain FedAvg


# ------------------------------------------------------------------------------------------------ (3) item-LR multiplier
def _ids(groups):
    return [sorted(id(p) for p in g["params"]) for g in groups]


def test_item_lr_multiplier_groups_ratio_and_effect():
    ad = tiny(1, tied=False)
    named = dict(zip(ad.manifest.shared_keys, ad.shared_parameters()))
    item_ids = {id(named[k]) for k in ITEM_LR_KEYS}
    for opt in ("adamw", "sgd_m", "sgd"):
        plain = make_optimizer(ad, LocalSolver(lr=0.05, optimizer=opt))
        assert _ids(plain.param_groups) == _ids(ad.param_groups(0.0))
        assert all("lr_mult" not in g for g in plain.param_groups)
        o = make_optimizer(ad, ItemLRSolver(lr=0.05, optimizer=opt, item_lr_mult=0.1))
        tagged = [g for g in o.param_groups if "lr_mult" in g]
        rest = [g for g in o.param_groups if "lr_mult" not in g]
        assert {id(p) for g in tagged for p in g["params"]} == item_ids
        assert not item_ids & {id(p) for g in rest for p in g["params"]}
        assert sorted(id(p) for g in o.param_groups for p in g["params"]) == sorted(id(p) for p in ad.shared_parameters())
        assert all(g["lr"] == 0.05 * 0.1 and g["lr_mult"] == 0.1 for g in tagged)
        assert all(g["lr"] == 0.05 for g in rest)
        if opt == "adamw":                                 # the ndim weight-decay rule is kept inside the item groups
            assert {g["weight_decay"] for g in tagged} == {1e-5, 0.0}
        _set_lr(o, 0.2)
        assert all(g["lr"] == 0.2 * 0.1 for g in tagged) and all(g["lr"] == 0.2 for g in rest)
        _set_lr(plain, 0.2)
        assert all(g["lr"] == 0.2 for g in plain.param_groups)
    # effect: one plain SGD step, no clip -> the non-item tensors move exactly as with the plain solver
    c = one_client("u-1", 12, seed=4)
    kw = dict(lr=0.5, optimizer="sgd", passes=1, batch_size=64, clip=None)
    outs = {}
    for name, sv in (("plain", LocalSolver(**kw)), ("mult", ItemLRSolver(**kw, item_lr_mult=0.25))):
        a = tiny(1, tied=False, recenter=False)            # no post-step recentering: the raw optimizer step
        theta = a.broadcast_state(clone=True)
        res = client_update(a, theta, c, sv, round_idx=0, seed=5)
        outs[name] = (theta, res.upload)
    th = outs["plain"][0]
    moved = 0
    for k in outs["plain"][1]:
        dp, dm = outs["plain"][1][k] - th[k], outs["mult"][1][k] - th[k]
        if k in ITEM_LR_KEYS:
            assert torch.allclose(dm, 0.25 * dp, rtol=1e-4, atol=1e-7), k
            moved += int(dp.abs().max() > 0)
        else:
            assert torch.equal(dm, dp), k
    assert moved >= 2
    f = tiny(1, tied=False)
    freeze_item_tables(f, FROZEN_ITEM_KEYS)
    assert_raises(ValueError, make_optimizer, f, ItemLRSolver(lr=0.05, item_lr_mult=0.1))
    for bad in (None, 0.0, -1.0, float("nan"), True):
        assert_raises(ValueError, ItemLRSolver, lr=0.05, item_lr_mult=bad)


# ------------------------------------------------------------------------------------------------ (4) digests
def test_phase1_runconfig_digests_equal_the_reference_values():
    seen = set()
    for p in PINS:
        cfg = RunConfig.from_dict(dict(p["cfg"], run_id=p["run_id"], ckpt_every=64))
        assert type(cfg.solver) is LocalSolver, p["key"]
        assert cfg.digest() == p["digest"], p["key"]
        assert RunConfig.from_dict(dict(p["cfg"], run_id=p["run_id"], ckpt_every=7)).digest() == p["digest"]
        seen.add(p["key"].split("|")[0])
    assert seen == {"FA", "FP", "PF", "FA_Q8", "FA_1024", "DP8", "S_CAL", "FA_WARM", "FA_WARM_FROZEN", "DP_WARM_FROZEN"}
    assert {p["key"]: p["digest"][:8] for p in PINS if "default" in p["key"]} == {
        "FA|default|wf=None|sf=None|MAIN": "66770288", "S_CAL|default|wf=None|sf=None|MAIN": "8fb83823"}


def test_solver_fields_from_dict_and_digest_guard():
    assert [f.name for f in fields(LocalSolver)] == LOCALSOLVER_FIELDS
    assert [f.name for f in fields(ItemLRSolver)] == LOCALSOLVER_FIELDS + ["item_lr_mult"]
    assert ITEM_LR_KEYS == FROZEN_ITEM_KEYS
    p = next(x for x in PINS if x["key"].startswith("FA_WARM_FROZEN|s4") and x["key"].endswith("MAIN"))
    base = dict(p["cfg"], run_id=p["run_id"], ckpt_every=64)
    none = dict(base, solver=dict(base["solver"], item_lr_mult=None))
    assert type(solver_from_dict(none["solver"])) is LocalSolver
    assert RunConfig.from_dict(none).digest() == p["digest"]
    real = dict(base, solver=dict(base["solver"], item_lr_mult=0.1))
    rc = RunConfig.from_dict(real)
    assert type(rc.solver) is ItemLRSolver and rc.solver.item_lr_mult == 0.1 and rc.digest() != p["digest"]
    assert RunConfig.from_dict(dict(real, solver=dict(real["solver"], item_lr_mult=0.01))).digest() != rc.digest()
    assert type(replace(rc.solver, lr=0.5)) is ItemLRSolver   # the per-round LR replace keeps the multiplier


@nc("a digest that keeps the None item_lr_mult key must change the pinned Phase-1 digests")
def test_nc_digest_without_guard_changes_phase1_digests():
    p = PINS[0]
    rc = RunConfig.from_dict(dict(p["cfg"], run_id=p["run_id"], ckpt_every=64))
    d = asdict(rc)
    d.pop("ckpt_every")
    for k, default in (("dropout_p", 0.0), ("dp", None), ("endpoint_rounds", None), ("server_opt", None),
                       ("lr_shape", "central")):
        if d[k] == default:
            d.pop(k)
    d["solver"]["item_lr_mult"] = None                    # what a LocalSolver FIELD (the rejected design) would add
    naive = hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()
    assert naive == p["digest"]
