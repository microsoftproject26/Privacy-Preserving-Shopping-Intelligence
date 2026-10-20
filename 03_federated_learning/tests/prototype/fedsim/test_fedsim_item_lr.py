"""Server learning rate, the item-table LR multiplier and config-digest stability (CPU, tiny synthetic families; no
data).

Proves: (1) server LR eta_s = 0 leaves theta bit-identical over 3 rounds for FedAvg (= FedAvgM beta 0 at LR 0, the
server-LR mapping), FedAvgM (beta 0.9) and FedAdam (tied family = no recentering, and the
frozen-table untied family whose recentering hooks are no-ops); (2) FedAvg at eta_s 0.3 is exactly
theta_r + 0.3 (aggregate - theta_r) evaluated by the server.py FedAvgM(beta 0) ops; (3) the item-LR multiplier
(client.ItemLRSolver) puts the item tables into their own lr x m group(s) (tag lr_mult), _set_lr keeps the ratio, the
other parameters move exactly as with the plain solver, and a frozen-table adapter refuses a multiplier; a plain
LocalSolver builds the plain groups; (4) the RunConfig digests of FA / FP / PF / FA_Q8 / FA_1024 / DP8 / S_CAL /
FA_WARM / FA_WARM_FROZEN (MAIN + CALIBRATION) / DP_WARM_FROZEN style configs equal pinned reference values,
LocalSolver's field set is unchanged (so LORecipe digests are too), RunConfig.from_dict / digest drop a None
item_lr_mult (digest guard) and carry a real one.
NC (strict xfail): a digest that keeps a None item_lr_mult key changes every pinned digest.
"""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import asdict, fields, replace

import torch
from fedsim_testkit import assert_raises, clients, nc, one_client, solver, tiny

from ppsi.fedsim.client import (
    ITEM_LR_KEYS,
    ItemLRSolver,
    LocalSolver,
    _set_lr,
    client_update,
    make_optimizer,
    solver_from_dict,
)
from ppsi.fedsim.numerics import state_digest
from ppsi.fedsim.runtime import FROZEN_ITEM_KEYS, RunConfig, freeze_item_tables
from ppsi.fedsim.server import Server, ServerOptConfig, run_round

# Pinned config digests (generic values; "fedadam" = FedAdam server optimiser, fl_const LR shape, client SGD-m)
PINS = [json.loads(x) for x in r"""
{"cfg":{"dp":null,"dropout_p":0.1,"endpoint_efe":6.0,"endpoint_rounds":120,"exposure_point":"end","lr_peak_local":0.001,"max_attempts":2,"method":"FA","mu":0.0,"n_shards":2,"pf":null,"seed":2026,"solver":{"batch_size":16,"clip":1.0,"fused":true,"lr":0.001,"passes":2,"weight_decay":1e-05}},"digest":"f17c1617970d1fc0fbea2ef295351ba79847ca242039ea12c490fc99962fca4a","key":"FA|default|MAIN","run_id":"fa-s2026"}
{"cfg":{"dp":{"clip_norm":null,"denominator":64,"noise_multiplier":0.0},"dropout_p":0.0,"endpoint_efe":6.0,"endpoint_rounds":48,"exposure_point":"end","lr_peak_local":0.001,"max_attempts":2,"method":"FA","mu":0.0,"n_shards":2,"pf":null,"seed":2026,"solver":{"batch_size":16,"clip":1.0,"fused":true,"lr":0.001,"passes":2,"weight_decay":1e-05}},"digest":"fd0a576875eec1761ed18fb994e32b169f7e612e16bcc24f3e4915825861b8a2","key":"S_CAL|default|MAIN","run_id":"s_cal-s2026"}
{"cfg":{"dp":null,"dropout_p":0.1,"endpoint_efe":6.0,"endpoint_rounds":240,"exposure_point":"end","lr_peak_local":0.1,"lr_shape":"fl_const","max_attempts":2,"method":"FA","mu":0.0,"n_shards":2,"pf":null,"seed":2026,"server_opt":{"beta1":0.9,"beta2":0.99,"lr":0.01,"name":"fedadam","tau":0.001,"v_init":"tau_squared"},"solver":{"batch_size":16,"clip":1.0,"fused":null,"lr":0.1,"optimizer":"sgd_m","passes":1,"weight_decay":0.0}},"digest":"dc6949a52ffd09b1d930e7593ed5f653c62e9c0c3c904fdf1199a66428c96181","key":"FA|fedadam|MAIN","run_id":"fa-s2026"}
{"cfg":{"dp":null,"dropout_p":0.1,"endpoint_efe":6.0,"endpoint_rounds":240,"exposure_point":"end","lr_peak_local":0.1,"lr_shape":"fl_const","max_attempts":2,"method":"FP","mu":0.01,"n_shards":2,"pf":null,"seed":2026,"server_opt":{"beta1":0.9,"beta2":0.99,"lr":0.01,"name":"fedadam","tau":0.001,"v_init":"tau_squared"},"solver":{"batch_size":16,"clip":1.0,"fused":null,"lr":0.1,"optimizer":"sgd_m","passes":1,"weight_decay":0.0}},"digest":"58b26b06ff416bc72472d06e1ed594c34a80ded54deeb6b8f2dce1be4dd3d0fb","key":"FP|fedadam|MAIN","run_id":"fp-s2026"}
{"cfg":{"dp":null,"dropout_p":0.1,"endpoint_efe":6.0,"endpoint_rounds":240,"exposure_point":"end","lr_peak_local":0.1,"lr_shape":"fl_const","max_attempts":2,"method":"PF","mu":0.0,"n_shards":2,"pf":{"clip":1.0,"lam":0.0001,"lr":0.001},"seed":2026,"server_opt":{"beta1":0.9,"beta2":0.99,"lr":0.01,"name":"fedadam","tau":0.001,"v_init":"tau_squared"},"solver":{"batch_size":16,"clip":1.0,"fused":null,"lr":0.1,"optimizer":"sgd_m","passes":1,"weight_decay":0.0}},"digest":"106cd6d9dcd35f2db872141860b2b8eeb3bccb7582ca883822edaae8561a8fbb","key":"PF|fedadam|MAIN","run_id":"pf-s2026"}
{"cfg":{"dp":null,"dropout_p":0.1,"endpoint_efe":6.0,"endpoint_rounds":240,"exposure_point":"end","lr_peak_local":0.1,"lr_shape":"fl_const","max_attempts":2,"method":"FA","mu":0.0,"n_shards":2,"pf":null,"seed":2026,"server_opt":{"beta1":0.9,"beta2":0.99,"lr":0.01,"name":"fedadam","tau":0.001,"v_init":"tau_squared"},"solver":{"batch_size":16,"clip":1.0,"fused":null,"lr":0.1,"optimizer":"sgd_m","passes":1,"weight_decay":0.0}},"digest":"1910723df64329a8568ccfd410a7ed4bf44e76da516bbdc9476407b42a3853c9","key":"FA_Q8|fedadam|MAIN","run_id":"fa_q8-s2026"}
{"cfg":{"dp":{"clip_norm":0.5,"denominator":64,"noise_multiplier":0.0},"dropout_p":0.0,"endpoint_efe":6.0,"endpoint_rounds":48,"exposure_point":"end","lr_peak_local":0.1,"lr_shape":"fl_const","max_attempts":2,"method":"FA","mu":0.0,"n_shards":2,"pf":null,"seed":2026,"server_opt":{"beta1":0.9,"beta2":0.99,"lr":0.01,"name":"fedadam","tau":0.001,"v_init":"tau_squared"},"solver":{"batch_size":16,"clip":1.0,"fused":null,"lr":0.1,"optimizer":"sgd_m","passes":2,"weight_decay":0.0}},"digest":"3011d4d80f30fe0e732c0bc725d62399b56a2ac84f874e44740c2c7b098a1e13","key":"FA_1024|fedadam|MAIN","run_id":"fa_1024-s2026"}
{"cfg":{"dp":{"clip_norm":0.5,"denominator":64,"noise_multiplier":1.0},"dropout_p":0.0,"endpoint_efe":6.0,"endpoint_rounds":48,"exposure_point":"end","lr_peak_local":0.1,"lr_shape":"fl_const","max_attempts":2,"method":"FA","mu":0.0,"n_shards":2,"pf":null,"seed":2026,"server_opt":{"beta1":0.9,"beta2":0.99,"lr":0.01,"name":"fedadam","tau":0.001,"v_init":"tau_squared"},"solver":{"batch_size":16,"clip":1.0,"fused":null,"lr":0.1,"optimizer":"sgd_m","passes":2,"weight_decay":0.0}},"digest":"df693411c5e1d2f595b237b9aa48ee5e0d7e236f40173df58b4c403a3174f805","key":"DP8|fedadam|MAIN","run_id":"dp8-s2026"}
{"cfg":{"dp":{"clip_norm":null,"denominator":64,"noise_multiplier":0.0},"dropout_p":0.0,"endpoint_efe":6.0,"endpoint_rounds":48,"exposure_point":"end","lr_peak_local":0.1,"lr_shape":"fl_const","max_attempts":2,"method":"FA","mu":0.0,"n_shards":2,"pf":null,"seed":2026,"server_opt":{"beta1":0.9,"beta2":0.99,"lr":0.01,"name":"fedadam","tau":0.001,"v_init":"tau_squared"},"solver":{"batch_size":16,"clip":1.0,"fused":null,"lr":0.1,"optimizer":"sgd_m","passes":2,"weight_decay":0.0}},"digest":"f358cce9520aefaf0489d4d0907ce9240ad80e095ffb7d4f21370539f482278d","key":"S_CAL|fedadam|MAIN","run_id":"s_cal-s2026"}
{"cfg":{"dp":null,"dropout_p":0.1,"endpoint_efe":6.0,"endpoint_rounds":240,"exposure_point":"end","lr_peak_local":0.1,"lr_shape":"fl_const","max_attempts":2,"method":"FA","mu":0.0,"n_shards":2,"pf":null,"seed":2026,"server_opt":{"beta1":0.9,"beta2":0.99,"lr":0.005,"name":"fedadam","tau":0.001,"v_init":"tau_squared"},"solver":{"batch_size":16,"clip":1.0,"fused":null,"lr":0.1,"optimizer":"sgd_m","passes":1,"weight_decay":0.0}},"digest":"487ae3a947985d831f00c071f7d041845cb4ddc0151518defb4b90f74096a03e","key":"FA_WARM|fedadam|MAIN","run_id":"fa_warm-s2026"}
{"cfg":{"dp":null,"dropout_p":0.1,"endpoint_efe":6.0,"endpoint_rounds":240,"exposure_point":"end","lr_peak_local":0.1,"lr_shape":"fl_const","max_attempts":2,"method":"FA","mu":0.0,"n_shards":2,"pf":null,"seed":2026,"server_opt":{"beta1":0.9,"beta2":0.99,"lr":0.005,"name":"fedadam","tau":0.001,"v_init":"tau_squared"},"solver":{"batch_size":16,"clip":1.0,"fused":null,"lr":0.1,"optimizer":"sgd_m","passes":1,"weight_decay":0.0}},"digest":"545b481d57fffdc8e5383a6912f0e9b6f4e68e285ab3c8c5c002b0feb093fc9c","key":"FA_WARM_FROZEN|fedadam|MAIN","run_id":"fa_warm_frozen-s2026"}
{"cfg":{"dp":null,"dropout_p":0.1,"endpoint_efe":6.0,"endpoint_rounds":60,"exposure_point":"end","lr_peak_local":0.1,"lr_shape":"fl_const","max_attempts":2,"method":"FA","mu":0.0,"n_shards":2,"pf":null,"seed":2026,"server_opt":{"beta1":0.9,"beta2":0.99,"lr":0.005,"name":"fedadam","tau":0.001,"v_init":"tau_squared"},"solver":{"batch_size":16,"clip":1.0,"fused":null,"lr":0.1,"optimizer":"sgd_m","passes":1,"weight_decay":0.0}},"digest":"37055f1532a79e158b802a8721d6c98bd14e5ee4e5f6780315b59f39a892c889","key":"FA_WARM_FROZEN|fedadam|CALIBRATION","run_id":"fa_warm_frozen-s2026"}
{"cfg":{"dp":{"clip_norm":0.5,"denominator":64,"noise_multiplier":1.0},"dropout_p":0.0,"endpoint_efe":6.0,"endpoint_rounds":48,"exposure_point":"end","lr_peak_local":0.1,"lr_shape":"fl_const","max_attempts":2,"method":"FA","mu":0.0,"n_shards":2,"pf":null,"seed":2026,"server_opt":{"beta1":0.9,"beta2":0.99,"lr":0.005,"name":"fedadam","tau":0.001,"v_init":"tau_squared"},"solver":{"batch_size":16,"clip":1.0,"fused":null,"lr":0.1,"optimizer":"sgd_m","passes":2,"weight_decay":0.0}},"digest":"5ea25d2e1a85f402a3c3625dd38ec92bcca2204a15fbf6032e32fefe57b5cf97","key":"DP_WARM_FROZEN|fedadam|MAIN","run_id":"dp_warm_frozen-s2026"}
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
    srv.attach_server_opt(ServerOptConfig("fedavgm", lr=0.3, beta1=0.0))     # = FedAvg with server LR 0.3
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
    kw = {"lr": 0.5, "optimizer": "sgd", "passes": 1, "batch_size": 64, "clip": None}
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
def test_runconfig_digests_equal_the_pinned_values():
    seen = set()
    for p in PINS:
        cfg = RunConfig.from_dict(dict(p["cfg"], run_id=p["run_id"], ckpt_every=64))
        assert type(cfg.solver) is LocalSolver, p["key"]
        assert cfg.digest() == p["digest"], p["key"]
        assert RunConfig.from_dict(dict(p["cfg"], run_id=p["run_id"], ckpt_every=7)).digest() == p["digest"]
        seen.add(p["key"].split("|")[0])
    assert seen == {"FA", "FP", "PF", "FA_Q8", "FA_1024", "DP8", "S_CAL", "FA_WARM", "FA_WARM_FROZEN", "DP_WARM_FROZEN"}
    assert {p["key"]: p["digest"][:8] for p in PINS if "default" in p["key"]} == {
        "FA|default|MAIN": "f17c1617", "S_CAL|default|MAIN": "fd0a5768"}


def test_solver_fields_from_dict_and_digest_guard():
    assert [f.name for f in fields(LocalSolver)] == LOCALSOLVER_FIELDS
    assert [f.name for f in fields(ItemLRSolver)] == LOCALSOLVER_FIELDS + ["item_lr_mult"]
    assert ITEM_LR_KEYS == FROZEN_ITEM_KEYS
    p = next(x for x in PINS if x["key"].startswith("FA_WARM_FROZEN|fedadam") and x["key"].endswith("MAIN"))
    base = dict(p["cfg"], run_id=p["run_id"], ckpt_every=64)
    none = dict(base, solver=dict(base["solver"], item_lr_mult=None))
    assert type(solver_from_dict(none["solver"])) is LocalSolver
    assert RunConfig.from_dict(none).digest() == p["digest"]
    real = dict(base, solver=dict(base["solver"], item_lr_mult=0.1))
    rc = RunConfig.from_dict(real)
    assert type(rc.solver) is ItemLRSolver and rc.solver.item_lr_mult == 0.1 and rc.digest() != p["digest"]
    assert RunConfig.from_dict(dict(real, solver=dict(real["solver"], item_lr_mult=0.01))).digest() != rc.digest()
    assert type(replace(rc.solver, lr=0.5)) is ItemLRSolver   # the per-round LR replace keeps the multiplier


@nc("a digest that keeps the None item_lr_mult key must change the pinned digests")
def test_nc_digest_without_guard_changes_pinned_digests():
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
