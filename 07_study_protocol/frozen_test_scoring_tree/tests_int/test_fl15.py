"""FL-15 (PREG1_AMENDMENT_06M_B B2 / B4; REAL_WORLD_FL_PRACTICE_AUDIT R5) — the 2-D warm-start screens S7 / S8 in
lr_screen / launch_fl, the FT_FA_WARM / FT_FA_WARM_FROZEN device start kind in launch_device, and the per-round client
training loss log (CPU, tiny synthetic worlds of test_fl05 / test_fl12 / test_int04; no real data).

B2 proves: --warm-server-lr-factor at its default (absent / 1) leaves every warm arm spec and RunConfig digest, the
FL-10 224-pair pin and the pinned default trajectory byte-identical; FedAdam f 0.1 = eta* / 10 exactly (a warm run at
(eta 0.01, f 0.1) is bitwise the run at eta 0.001), FedAvgM f 0.25 = eta* / 4, FedAvg f 0.25 = FedAvgM beta 0 at LR 0.25
= theta + 0.25 (aggregate - theta) bitwise; unregistered factors / axes / cold arms / MAIN CLI use refuse; the 2-D rule
(0.0005 band, >= 0.002 over the incumbent (m*, eta*), ties -> (m*, eta*)) with its tie order; `select-warm` needs the 4
registered runs (the 06K 2-run screen refuses), records excluded cells and refuses an excluded incumbent; the 2-D file
is verified (tampering refused) and a MAIN warm run takes BOTH factors from the registered file (bind / resolve_spec),
the 1-D file and a server axis that is not the bound tuning winner's are refused.
B4 proves: the LO / FT_C / FT_FA / FT_PRE device spec digests are unchanged (pinned before FL-15); FT_FA_WARM(_FROZEN)
applies the registered SELECTION_FT_C cell with the FT_C procedure; its base is a verified FA_WARM / FA_WARM_FROZEN MAIN
checkpoint only (another arm, a CALIBRATION run, another seed, an unfinished run, a tampered spec, a MAIN-trained init,
a C_BIG run id, a config-digest mismatch, a wrong layout / sha pin refuse); the FT cell equals the FT_C procedure on the
same start bitwise and its rows carry the base identity.
R5 proves: TRAIN_LOSS.jsonl has one line per round equal to the round's own VisitRecords, resumes byte-identically,
never exists for z > 0 (DP), and every pinned trajectory is unchanged.
NCs (strict xfail on AssertionError): each new refusal removed / mutated in the real wiring.
REV-FL-15 fixes: m2 select-warm refuses a non-S4 tuning file; m3 FT_FA / FT_C refuse an FA_WARM(_FROZEN) base; m4 a MAIN
warm --dry-run runs bind_warm_selection against its --fl-tuning file; m1 pinned as a unit case (disclosure). One NC each.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import v4int.launch_device as D
import v4int.launch_fl as F
import v4int.lr_screen as L
from fl.numerics import state_digest
from fl.synthetic import make_tiny_adapter
from intkit import assert_raises, nc
from test_dev_ftpre import PRE_DEVFTPRE_DIGESTS, SEL_C, _args, _selection_file
from test_fl05_launch import run_tiny, tiny_spec
from test_fl10_fedadam_launch import CALIB_EVAL, PRE_FL10_SHA, PRE_FL10_TRAJ, _tiny, all_grid_digests, ladder  # noqa: F401
from test_fl12_launch import pre, s4, s4_spec, untied_world, warm_runs, warm_sel, warm_world  # noqa: F401
from test_int04_device import K, PEAK, _frozen, _plans, _world

T = F.tuning_config
VAR = "SASREC_D64_B2"
# a synthetic theta_pre identity (the digest table below was MEASURED with it on the pre-FL-15 launch_fl 5652e4c7)
IDENT = {"kind": "PRETRAIN_FOR_FL", "run_id": "PRETRAIN_FOR_FL_SMALL_SASREC_D64_B2_s2026", "run_label": "PRETRAIN_FOR_FL",
         "train_manifest": "PRETRAIN_TRAIN_5_100_NONMAIN", "fl_init_kind": "endpoint_6.0", "checkpoint_sha256": "a" * 64,
         "checkpoint_bytes": 123, "state_digest": "b" * 64, "model_variant": VAR, "family": "SASREC", "seed": 2026}
TK = ({}, dict(server_opt="fedadam", server_lr=0.01, lr_mult=1.0, lr_shape="fl_const"),
      dict(server_opt="fedavgm", lr_mult=2.0),
      dict(server_opt="fedadam", server_lr=0.001, client_opt="sgd_m", client_lr=0.1, lr_shape="fl_const", local_passes=4))
PRE_FL15_WARM_SHA = "d718f42c727788dee42d4cf7cd1fafb17c92f19fd5dd3ee995546670f13faae5"   # 32 (spec, RunConfig) pairs
# the FT_PRE spec digests (MEASURED on launch_device 0ae48c72 before FL-15): (role, selection cell) -> digest
PRE_FL15_FT_PRE = {("MAIN", "FT_C_f0.3_P2"): "ad3d29bb45e24e322f362f67417b5f1467be1583dcb083b53d719105b33f3bd2",
                   ("MAIN", "FT_C_f0.3_P4"): "1e40b56719005f123f3ca613a5ce7eba1ac3c071c2968cf7248e87b9c24cbfa4",
                   ("UNSEEN_HOLDOUT", "FT_C_f0.3_P2"): "2efd7e2fe31d808a9f3bca9dcf0efe770eb2e520051a6457f69b71ff26c88276",
                   ("UNSEEN_HOLDOUT", "FT_C_f0.3_P4"): "6b4bcaac35e9fa1ced94b3e392f0feec3fd54226c6264558caa6aeb5e1661c66",
                   ("SPARSE_PANEL", "FT_C_f0.3_P2"): "038180a36fa84694b15991282f7d7e7ebdba025ea312931da2d8543b6fcf0dc7",
                   ("SPARSE_PANEL", "FT_C_f0.3_P4"): "ff29aac1c40d08d201a40bfc5693d094863d2855a5982c64dd1ce38d502a22cc"}
PRE_FL15_FT_FA = "0189ad23515791612951421f554bc566ab014741f4f9762eae4f6c8afbc6ffcf"      # FT_FA MAIN, FT_FA_f0.1_P2


@pytest.fixture(scope="module", autouse=True)
def _strict_numerics_module():
    """conftest's strict-FP32 fixture is function-scoped; the module fixtures (ladder / warm runs) set up before it
    when a selection (e.g. -k test_nc_) starts with them, so enforce strict FP32 at module scope first."""
    from fl.numerics import enforce_strict_fp32
    torch.set_num_threads(1)
    enforce_strict_fp32()
    yield


def _json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def warm_digest_table(**extra) -> dict:
    out = {}
    for arm, fs in (("FA_WARM", (None, 0.25)), ("FA_WARM_FROZEN", (None, 4.0))):
        for f in fs:
            for tk in TK:
                for role in ("MAIN", "CALIBRATION"):
                    s = F.arm_spec(arm, seed=2026, peak_lr=0.000711, init_from=IDENT, warm_lr_factor=f, role=role,
                                   **tk, **extra)
                    out[f"{arm}|{f}|{F.tuning_label(F.spec_tuning(s))}|{role}"] = (
                        F.spec_digest(s), F.fl_run_config(s, "x", 64).digest())
    return out


# ================================================================================================ B2: the server axis
def test_default_warm_digests_and_trajectories_unchanged(tmp_path):
    for extra in ({}, {"warm_server_lr_factor": None}, {"warm_server_lr_factor": 1.0}):
        d = warm_digest_table(**extra)
        assert len(d) == 32 and hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest() == PRE_FL15_WARM_SHA
    g = all_grid_digests()
    assert len(g) == 224 and hashlib.sha256(json.dumps(g, sort_keys=True).encode()).hexdigest() == PRE_FL10_SHA
    assert run_tiny(tmp_path, _tiny("FA"), "FA")["summary"]["server_digest"] == PRE_FL10_TRAJ["FA"]
    for arm in F.ARMS:
        kw = {"clip_norm": 1.0} if arm in ("FA_1024", "DP") else {}
        kw.update({"z": 0.5642} if arm == "DP" else {})
        assert "warm_server_lr_factor" not in F.arm_spec(arm, seed=2026, peak_lr=0.000711, **kw)


def test_warm_server_factor_spec_rows_and_refusals():
    s = F.arm_spec("FA_WARM", seed=2026, peak_lr=0.000711, init_from=IDENT, server_opt="fedadam", server_lr=0.01,
                   lr_mult=4.0, warm_server_lr_factor=0.1)
    assert s["warm_server_lr_factor"] == 0.1 and s["server_lr"] == 0.01           # the tuning point keeps eta*
    assert s["cfg"]["server_opt"] == {"name": "fedadam", "lr": 0.001, "beta1": 0.9, "beta2": 0.99, "tau": 1e-3,
                                      "v_init": "tau_squared"}
    assert F.spec_tuning(s) == T(optimizer="fedadam", server_lr=0.01, lr_mult=4.0)
    F.check_warm_spec(s)
    lab = F.warm_row_labels(s)
    assert lab["fl_warm_server_lr_factor"] == 0.1 and lab["fl_warm_server_lr"] == 0.001
    assert F.fl_run_config(s, "x", 64).server_opt.lr == 0.001
    m = F.arm_spec("FA_WARM_FROZEN", seed=2026, peak_lr=0.000711, init_from=IDENT, server_opt="fedavgm", lr_mult=2.0,
                   warm_lr_factor=4.0, warm_server_lr_factor=0.25)
    assert m["cfg"]["server_opt"] == {"name": "fedavgm", "lr": 0.25, "beta1": 0.9} and F.spec_tuning(m)["server_lr"] == 1.0
    a = F.arm_spec("FA_WARM", seed=2026, peak_lr=0.000711, init_from=IDENT, warm_server_lr_factor=0.25)   # FedAvg
    assert a["cfg"]["server_opt"] == {"name": "fedavgm", "lr": 0.25, "beta1": 0.0}
    assert F.spec_tuning(a)["optimizer"] == "fedavg" and F.warm_row_labels(a)["fl_warm_server_lr"] == 0.25
    c = F.fl_run_config(a, "x", 64).server_opt
    assert (c.name, c.lr, c.beta1) == ("fedavgm", 0.25, 0.0)
    assert F.warm_grid("FA_WARM", "fedadam") == [(1.0, 1.0), (1.0, 0.1), (0.25, 1.0), (0.25, 0.1)]
    assert F.warm_grid("FA_WARM_FROZEN", "fedavg") == [(1.0, 1.0), (1.0, 0.25), (4.0, 1.0), (4.0, 0.25)]
    for arm, kw in (("FA_WARM", dict(server_opt="fedadam", server_lr=0.01, warm_server_lr_factor=0.25)),
                    ("FA_WARM", dict(warm_server_lr_factor=0.1)),                   # FedAvg axis is {1, 0.25}
                    ("FA_WARM", dict(server_opt="fedavgm", warm_server_lr_factor=0.1)),
                    ("FA_WARM", dict(warm_server_lr_factor=0.5)),
                    ("FA", dict(server_opt="fedadam", server_lr=0.01, warm_server_lr_factor=0.1))):
        init = {} if arm == "FA" else {"init_from": IDENT}
        assert_raises(F.LaunchRefused, F.arm_spec, arm, seed=2026, peak_lr=0.000711, **init, **kw)
    for bad in (dict(s, warm_server_lr_factor=1.0),                               # a default recorded
                dict(s, cfg=dict(s["cfg"], server_opt=dict(s["cfg"]["server_opt"], lr=0.01))),   # step not scaled
                {k: v for k, v in s.items() if k != "warm_server_lr_factor"},    # scaled without the key
                dict(s, warm_server_lr_factor=0.25)):
        assert_raises(F.LaunchRefused, F.spec_tuning, bad)
    cold = F.arm_spec("FA", seed=2026, peak_lr=0.000711, server_opt="fedadam", server_lr=0.01)
    assert_raises(F.LaunchRefused, F.check_warm_spec, dict(cold, warm_server_lr_factor=0.1))


def test_fedavg_server_factor_is_scaled_fedavg():
    from fl.server import ServerOptConfig, make_server_opt
    ad = make_tiny_adapter(seed=3, tied=False)
    theta = ad.broadcast_state(clone=True)
    g = torch.Generator().manual_seed(0)
    shared = set(ad.manifest.shared_keys)
    new = {k: (v + 0.01 * torch.randn(v.shape, generator=g) if k in shared else v.clone()) for k, v in theta.items()}
    opt = make_server_opt(ServerOptConfig(**F.warm_server_opt(T(), 0.25)), ad.manifest, theta)
    for _ in range(3):                                          # beta 0: no momentum carries between rounds
        out = opt.step(theta, new)
        for k in shared:
            assert torch.equal(out[k], torch.add(theta[k], torch.mul(torch.sub(new[k], theta[k]), 0.25)))


def test_warm_server_factor_runs_exact_and_resume(pre, tmp_path):
    ident = pre["ident"]
    a = run_tiny(tmp_path, tiny_spec("FA_WARM", init_from=ident, server_opt="fedadam", server_lr=0.01,
                                     warm_server_lr_factor=0.1), "a", world=warm_world(pre))
    b = run_tiny(tmp_path, tiny_spec("FA_WARM", init_from=ident, server_opt="fedadam", server_lr=0.001), "b",
                 world=warm_world(pre))
    c = run_tiny(tmp_path, tiny_spec("FA_WARM", init_from=ident, server_opt="fedadam", server_lr=0.01), "c",
                 world=warm_world(pre))
    assert a["status"] == b["status"] == "DONE_MAIN"
    assert a["summary"]["server_digest"] == b["summary"]["server_digest"] != c["summary"]["server_digest"]   # eta*/10
    rows = _json(tmp_path / "a" / "RESULT_ROWS.json")["rows"]
    assert rows and all(r["fl_warm_server_lr_factor"] == 0.1 and r["fl_warm_server_lr"] == 0.001
                        and r["fl_server_lr"] == 0.01 for r in rows)
    spec = tiny_spec("FA_WARM", init_from=ident, warm_server_lr_factor=0.25)       # FedAvg at 1/4
    s = run_tiny(tmp_path, spec, "s", world=warm_world(pre))
    assert run_tiny(tmp_path, spec, "p", world=warm_world(pre), interrupt_at=29)["status"] == "INTERRUPTED"
    r = run_tiny(tmp_path, spec, "p", world=warm_world(pre), resume=True)
    assert r["summary"]["server_digest"] == s["summary"]["server_digest"]
    plain = run_tiny(tmp_path, tiny_spec("FA_WARM", init_from=ident), "plain", world=warm_world(pre))
    assert plain["summary"]["server_digest"] != s["summary"]["server_digest"]
    assert s["summary"]["server_opt"]["cfg"]["lr"] == 0.25 and "server_opt" not in plain["summary"]


def test_warm_server_preflight_and_run_ids(pre, tmp_path):
    base = ["--seed", "2026", "--gpu", "0", "--run-dir", str(tmp_path / "r"), "--history-policy", "USER_HISTORY_L50"]
    init = ["--init-from", str(pre["path"])]
    adam = ["--server-opt", "fedadam", "--server-lr", "0.01"]
    calib = ["--arm", "FA_WARM", "--role", "CALIBRATION", "--smoke"] + base + init + adam
    a = F.parse_args(calib + ["--warm-server-lr-factor", "0.1", "--warm-lr-factor", "0.25"])
    F.fl05_preflight(a)
    assert F.default_run_id(a).endswith("FL_FA_WARM_LRSCREEN_FEDADAM_eta0.01_x1_wf0.25_sf0.1_CALIB_s2026")
    b = F.parse_args(["--arm", "FA_WARM_FROZEN", "--freeze-item-tables", "--role", "CALIBRATION", "--smoke"] + base + init
                     + ["--warm-server-lr-factor", "0.25"])                       # FedAvg axis
    F.fl05_preflight(b)
    assert F.default_run_id(b).endswith("FL_FA_WARM_FROZEN_LRSCREEN_x1_sf0.25_CALIB_s2026")
    assert F.default_run_id(F.parse_args(calib)).endswith("FL_FA_WARM_LRSCREEN_FEDADAM_eta0.01_x1_CALIB_s2026")
    wsel = ["--warm-selection", str(tmp_path / "w.json")]
    for argv in (calib + ["--warm-server-lr-factor", "0.25"],                     # FedAdam axis {1, 0.1}
                 ["--arm", "FA_WARM", "--smoke"] + base + init + ["--warm-server-lr-factor", "0.1"],   # FedAvg {1, 0.25}
                 ["--arm", "FA", "--smoke"] + base + adam + ["--warm-server-lr-factor", "0.1"],        # cold arm
                 ["--arm", "FA_WARM"] + base + init + wsel + ["--warm-server-lr-factor", "0.25"],      # MAIN: file only
                 ["--arm", "FA_WARM"] + base + init + ["--warm-server-lr-factor", "0.25"],
                 ["--arm", "FA_WARM", "--smoke"] + base + init + wsel + ["--warm-server-lr-factor", "0.25"]):
        assert_raises(F.LaunchRefused, F.fl05_preflight, F.parse_args(argv))
    F.fl05_preflight(F.parse_args(["--arm", "FA_WARM"] + base + init + wsel))  # MAIN with the file: accepted here


# ================================================================================================ B2: the 2-D rule
G7 = F.warm_grid("FA_WARM", "fedadam")                          # (1, 1), (1, 0.1), (0.25, 1), (0.25, 0.1)


def _v(*vals, grid=G7):
    return dict(zip(grid, vals))


def test_warm_decision_rule():
    d = L.warm_decision("FA_WARM", "fedadam", _v(0.100, 0.1015, 0.1019, 0.099))       # gain 0.0019 < 0.002
    assert d["winner"] == (1.0, 1.0) and d["no_measurable_gain"] and d["tie_set"] == ["f=1|sf=1"]
    assert d["best_candidate"] == "f=0.25|sf=1" and abs(d["gain"] - 0.0019) < 1e-12 and d["min_gain"] == 0.002
    assert d["tie_band_set"] == ["f=1|sf=0.1", "f=0.25|sf=1"] and d["incumbent"] == "f=1|sf=1"
    d = L.warm_decision("FA_WARM", "fedadam", _v(0.100, 0.1021, 0.1023, 0.1020))      # gain 0.0023, band of 3
    assert d["winner"] == (1.0, 0.1) and not d["no_measurable_gain"] and d["best_candidate"] == "f=0.25|sf=1"
    assert d["tie_set"] == d["tie_band_set"] == ["f=1|sf=0.1", "f=0.25|sf=1", "f=0.25|sf=0.1"]   # fewer axes first
    d = L.warm_decision("FA_WARM", "fedadam", _v(0.100, 0.090, 0.090, 0.1030))        # a clear 2-axis winner
    assert d["winner"] == (0.25, 0.1) and d["tie_set"] == ["f=0.25|sf=0.1"]
    d = L.warm_decision("FA_WARM", "fedadam", {(1.0, 1.0): 0.1, (0.25, 1.0): 0.2})    # excluded cells: not ranked
    assert d["winner"] == (0.25, 1.0)
    assert L.warm_decision("FA_WARM", "fedadam", {(1.0, 1.0): 0.1})["no_measurable_gain"]   # no candidate: gain None
    d = L.warm_decision("FA_WARM", "fedadam", _v(0.1000, 0.1017, 0.1021, 0.09))       # REV-FL-15 m1 (disclosed):
    assert d["winner"] == (1.0, 0.1) and d["best_candidate"] == "f=0.25|sf=1"          # gain = the best candidate's
    assert abs(d["gain"] - 0.0021) < 1e-12                                             # (0.0021); the winner's own
    g8 = F.warm_grid("FA_WARM_FROZEN", "fedavgm")                                      # is 0.0017
    d = L.warm_decision("FA_WARM_FROZEN", "fedavgm", _v(0.1, 0.1003, 0.1026, 0.1027, grid=g8))
    assert d["winner"] == (4.0, 1.0) and d["tie_set"] == ["f=4|sf=1", "f=4|sf=0.25"]
    for arm, opt, vals in (("FA_WARM", "fedadam", {(0.25, 1.0): 0.2}),              # no incumbent
                           ("FA_WARM", "fedadam", {(1.0, 1.0): 0.1, (1.0, 0.25): 0.2}),   # FedAvg cell on FedAdam
                           ("FA_WARM", "fedadam", {(1.0, 1.0): 0.1, (4.0, 1.0): 0.2}),    # an S8 factor
                           ("FA_WARM", "fedadam", {(1.0, 1.0): float("nan")}),
                           ("FA", "fedadam", {(1.0, 1.0): 0.1}), ("FA_WARM", "sgd", {(1.0, 1.0): 0.1})):
        assert_raises(L.SelectionRefused, L.warm_decision, arm, opt, vals)


def _copy_run(src: Path, dst: Path, value) -> Path:
    """A copy of a finished warm CALIB run whose PRACTICAL_BEST primary row value is replaced (None = non-finite)."""
    shutil.copytree(src, dst)
    p = dst / "RESULT_ROWS.json"
    rr = _json(p)
    for r in rr["rows"]:
        if r.get("selection_rule") == "PRACTICAL_BEST" and (r["aggregation"], r["target_denominator"], r["metric"],
                                                             int(r["k"])) == ("macro", "E2E", "mrr", 20):
            r["value"] = value
    p.write_text(json.dumps(rr), encoding="utf-8")
    return dst


def test_select_warm_2d_excluded_cells(warm_sel, tmp_path):
    runs = warm_runs(warm_sel, "FA_WARM")
    bad = _copy_run(runs[3], tmp_path / "nan3", None)
    obj = L.selection_warm("S7", runs[:3] + [bad], warm_sel["s4_file"])
    assert [e["label"] for e in obj["excluded"]] == ["f=0.25|sf=0.1"] and len(obj["values"]) == 3
    assert sorted(obj["screened"]) == sorted(F.warm_cell_label(*c) for c in G7)
    out = tmp_path / "W.json"
    out.write_text(json.dumps(obj), encoding="utf-8")
    assert F.load_warm_selection(out)["winner"] == obj["winner"]
    inc_bad = _copy_run(runs[0], tmp_path / "nan0", None)
    with pytest.raises(L.SelectionRefused, match="incumbent"):
        L.selection_warm("S7", [inc_bad] + runs[1:], warm_sel["s4_file"])


def _cells(obj):
    return {F.warm_cell_label(*c): c for c in F.warm_grid(obj["arm"], obj["server_axis"]["optimizer"])}


def resign_2d(src: dict, path: Path, *, values=None, eta=None, decide=True, raw=None, **changes) -> Path:
    """A re-signed copy of a 2-D warm selection (values / eta* changed, the decision recomputed unless decide=False)."""
    o = json.loads(json.dumps(src))
    opt = o["server_axis"]["optimizer"]
    if eta is not None:
        o["server_axis"]["eta_star"] = eta
        o["server_axis"]["server_lrs"] = {F.num_tag(f): (eta if f == 1.0 else eta / F.WARM_SERVER_DIVISORS[f])
                                          for f in F.WARM_SERVER_FACTORS[opt]}
    if values is not None:
        o["values"], o["excluded"] = {F.warm_cell_label(*c): v for c, v in values.items()}, []
    if raw is not None:                                          # values keyed by (possibly unregistered) labels
        o["values"] = dict(raw)
    if decide:
        dec = L.warm_decision(o["arm"], opt, {_cells(o)[k]: v for k, v in o["values"].items()})
        o.update({k: dec[k] for k in F.WARM_2D_DECISION_KEYS})
        wf, sf = dec["winner"]
        o["winner"] = {"warm_lr_factor": wf, "warm_server_lr_factor": sf, "label": F.warm_cell_label(wf, sf),
                       "server_lr": o["server_axis"]["server_lrs"][F.num_tag(sf)]}
        o["winner_factor"], o["winner_server_lr_factor"] = wf, sf
    o.update(changes)
    o["self_sha256"] = F._canon(o)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(o), encoding="utf-8")
    return path


def legacy_1d(obj2: dict, path: Path) -> Path:
    """A verified PPSI_V4_FL_WARM_SELECTION_1 file (the superseded {m*, m*/4} screen) for the same stage / theta_pre."""
    o = {"format": F.WARM_SELECTION_FORMAT, "stage": obj2["stage"], "arm": obj2["arm"], "values": {"1": 0.1, "0.25": 0.2},
         "screened": [0.25, 1.0], "excluded": [], "winner_factor": 0.25, "tie_set": [0.25],
         "init_from": obj2["init_from"], "tuning_selection": obj2["tuning_selection"]}
    o["self_sha256"] = F._canon(o)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(o), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def s7_file(warm_sel, tmp_path_factory):
    root = tmp_path_factory.mktemp("s7file")
    out = root / "FL_WARM_SELECTION_S7.json"
    L.write_selection_warm("S7", warm_runs(warm_sel, "FA_WARM"), out, warm_sel["s4_file"])
    return {"path": out, "obj": _json(out), "root": root}


def test_load_warm_selection_2d_tamper_refusals(s7_file, tmp_path):
    obj = s7_file["obj"]
    assert F.load_warm_selection(s7_file["path"])["format"] == F.WARM_SELECTION_FORMAT_2D
    other = next(k for k in obj["values"] if k != obj["winner"]["label"])
    wf, sf = _cells(obj)[other]
    cases = {"winner": dict(decide=False, winner=dict(obj["winner"], label=other, warm_lr_factor=wf,
                                                      warm_server_lr_factor=sf), winner_factor=wf,
                            winner_server_lr_factor=sf),
             "gain": dict(decide=False, gain=0.5), "band": dict(decide=False, tie_band_set=[]),
             "min_gain": dict(decide=False, min_gain=0.001),
             "axis": dict(decide=False, server_axis=dict(obj["server_axis"], factors=[1.0, 0.25])),
             "screened": dict(decide=False, screened=obj["screened"][:3]),
             "label": dict(decide=False, raw={**{k: v for k, v in obj["values"].items() if k != other},
                                              "f=0.5|sf=1": 0.3}),
             "eta": dict(decide=False, server_axis=dict(obj["server_axis"], eta_star=0.02)),
             "tuning": dict(decide=False, tuning_selection=dict(obj["tuning_selection"], winner="fedx|m=4")),
             "init": dict(decide=False, init_from=dict(obj["init_from"], train_manifest="MAIN_TRAIN_5_100_131072"))}
    for name, kw in cases.items():
        p = resign_2d(obj, tmp_path / f"{name}.json", **kw)
        assert_raises(F.LaunchRefused, F.load_warm_selection, p)
    raw = dict(obj, gain=0.5)                                    # not re-signed
    (tmp_path / "raw.json").write_text(json.dumps(raw), encoding="utf-8")
    assert_raises(F.LaunchRefused, F.load_warm_selection, tmp_path / "raw.json")
    ok = resign_2d(obj, tmp_path / "ok.json")                    # a re-signed copy with the same values verifies
    assert F.load_warm_selection(ok)["winner"] == obj["winner"]
    leg = legacy_1d(obj, tmp_path / "legacy.json")
    lo = F.load_warm_selection(leg)                              # the 1-D record still verifies ...
    with pytest.raises(F.LaunchRefused, match="2-D"):            # ... but never configures a run
        F.warm_selection_winner(lo, leg)


def _main_args(tmp_path, pre, warm_sel, reg, dry=False):
    a = F.parse_args(["--arm", "FA_WARM", "--seed", "2027", "--gpu", "0", "--run-dir", str(tmp_path / "r"),
                      "--history-policy", "USER_HISTORY_L50", "--init-from", str(pre["path"]),
                      "--warm-selection", str(reg), "--fl-tuning", str(warm_sel["s4_file"])]
                     + (["--dry-run"] if dry else []))
    return a


def _registered_s7(tmp_path, monkeypatch, obj, **kw) -> Path:
    reg = resign_2d(obj, tmp_path / "contracts_fl" / "FL_WARM_SELECTION_S7.json", **kw)
    monkeypatch.setitem(F.WARM_SELECTION_PATH, "S7", reg)
    return reg


def _bind_main(monkeypatch, warm_sel) -> dict:
    mb = {"sha256": _sha(warm_sel["s4_file"]), "path": str(warm_sel["s4_file"]), "kind": "fl_tuning"}
    monkeypatch.setattr(F, "bind_main_selection", lambda args, peak_lr=None: None if args.smoke else mb)
    return mb


WIN_025_01 = {(1.0, 1.0): 0.100, (1.0, 0.1): 0.100, (0.25, 1.0): 0.100, (0.25, 0.1): 0.105}


def test_main_warm_run_takes_both_factors(warm_sel, s7_file, pre, tmp_path, monkeypatch):
    reg = _registered_s7(tmp_path, monkeypatch, s7_file["obj"], values=WIN_025_01)
    mb = _bind_main(monkeypatch, warm_sel)
    eta = warm_sel["w4"]["server_lr"]
    b = F.bind_warm_selection(_main_args(tmp_path, pre, warm_sel, reg), pre["ident"], mb)
    assert (b["winner_factor"], b["winner_server_lr_factor"], b["winner_server_lr"]) == (0.25, 0.1, eta / 10)
    assert b["format"] == F.WARM_SELECTION_FORMAT_2D and b["server_optimizer"] == "fedadam"
    args = _main_args(tmp_path, pre, warm_sel, reg)
    F.fl05_preflight(args)
    spec, extra = F.resolve_spec(args, 0.000711, fused=True, init_identity=pre["ident"])
    assert spec["warm_lr_factor"] == 0.25 and spec["warm_server_lr_factor"] == 0.1
    assert spec["cfg"]["server_opt"]["lr"] == eta / 10 and spec["server_lr"] == eta
    assert extra["warm_selection"]["sha256"] == _sha(reg) and extra["main_selection"] == mb
    assert F.spec_tuning(spec) == warm_sel["w4"]
    dspec, _ = F.resolve_spec(_main_args(tmp_path, pre, warm_sel, reg, dry=True), 0.000711, fused=True,
                              init_identity=pre["ident"])
    assert F.spec_digest(dspec) == F.spec_digest(spec)          # the dry run shows the same MAIN spec
    main_bind_none = F.bind_warm_selection
    assert_raises(F.LaunchRefused, main_bind_none, args, pre["ident"], dict(mb, path=None))
    assert_raises(F.LaunchRefused, main_bind_none, args, pre["ident"], dict(mb, sha256="0" * 64))


def test_main_warm_refuses_1d_file_and_foreign_server_axis(warm_sel, s7_file, pre, tmp_path, monkeypatch):
    mb = _bind_main(monkeypatch, warm_sel)
    reg = tmp_path / "contracts_fl" / "FL_WARM_SELECTION_S7.json"
    legacy_1d(s7_file["obj"], reg)
    monkeypatch.setitem(F.WARM_SELECTION_PATH, "S7", reg)
    args = _main_args(tmp_path, pre, warm_sel, reg)
    with pytest.raises(F.LaunchRefused, match="2-D"):
        F.bind_warm_selection(args, pre["ident"], mb)
    with pytest.raises(F.LaunchRefused, match="2-D"):
        F.resolve_spec(args, 0.000711, fused=True, init_identity=pre["ident"])
    with pytest.raises(F.LaunchRefused, match="2-D"):
        F.resolve_spec(_main_args(tmp_path, pre, warm_sel, reg, dry=True), 0.000711, fused=True,
                       init_identity=pre["ident"])
    foreign = 0.1 if warm_sel["w4"]["server_lr"] != 0.1 else 0.01  # a registered FedAdam eta that is not eta*
    reg2 = _registered_s7(tmp_path / "f", monkeypatch, s7_file["obj"], eta=foreign)
    F.load_warm_selection(reg2)                                  # a valid file on its own ...
    with pytest.raises(F.LaunchRefused, match="server axis"):   # ... but not the bound winner's server axis
        F.bind_warm_selection(_main_args(tmp_path, pre, warm_sel, reg2), pre["ident"], mb)


# ================================================================================================ REV-FL-15 m2 / m4
@pytest.fixture(scope="module")
def warm_s3(s4, pre, tmp_path_factory):
    """The 4 FA_WARM CALIB runs of the 2-D grid launched at the S3 winner (the point of an S3 stage file)."""
    root = tmp_path_factory.mktemp("warms3")
    dirs = []
    for f, sf in F.warm_grid("FA_WARM", s4["w3"]["optimizer"]):
        spec = s4_spec("FA_WARM", role="CALIBRATION", init_from=pre["ident"], warm_lr_factor=None if f == 1.0 else f,
                       warm_server_lr_factor=None if sf == 1.0 else sf, **s4["kw"])
        name = f"FA_WARM_{f}_{sf}"
        assert run_tiny(root, spec, name, world=warm_world(pre, eval_id=CALIB_EVAL))["status"] == "DONE_MAIN"
        dirs.append(root / name)
    return {"dirs": dirs, "s3_file": s4["s3_file"]}


def test_select_warm_refuses_a_non_s4_tuning_file(warm_s3, warm_sel):
    """REV-FL-15 m2: 4 valid warm runs at an S3 winner with the S3 stage file: refused (the S7 / S8 screens are bound
    to the FINAL stage-S4 selection); the S4 file with its runs is accepted."""
    with pytest.raises(L.SelectionRefused, match="stage 'S3'"):
        L.selection_warm("S7", warm_s3["dirs"], warm_s3["s3_file"])
    assert L.selection_warm("S7", warm_runs(warm_sel, "FA_WARM"), warm_sel["s4_file"])["tuning_selection"]["stage"] == "S4"


def test_main_warm_dry_run_checks_the_same_binding(warm_sel, warm_s3, s7_file, pre, tmp_path, monkeypatch):
    """REV-FL-15 m4: a MAIN warm --dry-run goes through bind_warm_selection against its --fl-tuning file: a foreign
    server axis, a warm file bound to another tuning file, or no --fl-tuning refuse, exactly as the real launch."""
    foreign = 0.1 if warm_sel["w4"]["server_lr"] != 0.1 else 0.01
    reg = _registered_s7(tmp_path, monkeypatch, s7_file["obj"], eta=foreign)
    with pytest.raises(F.LaunchRefused, match="server axis"):
        F.resolve_spec(_main_args(tmp_path, pre, warm_sel, reg, dry=True), 0.000711, fused=True,
                       init_identity=pre["ident"])
    reg = _registered_s7(tmp_path / "ok", monkeypatch, s7_file["obj"], values=WIN_025_01)
    args = _main_args(tmp_path, pre, warm_sel, reg, dry=True)
    spec, extra = F.resolve_spec(args, 0.000711, fused=True, init_identity=pre["ident"])
    assert extra["warm_selection"]["dry_run"] and extra["warm_selection"]["sha256"] == _sha(reg)
    assert (spec["warm_lr_factor"], spec["warm_server_lr_factor"]) == (0.25, 0.1)
    args.fl_tuning = warm_s3["s3_file"]                                               # another tuning file
    with pytest.raises(F.LaunchRefused, match="tuning selection sha"):
        F.resolve_spec(args, 0.000711, fused=True, init_identity=pre["ident"])
    args.fl_tuning = None
    with pytest.raises(F.LaunchRefused, match="needs the --fl-tuning"):
        F.resolve_spec(args, 0.000711, fused=True, init_identity=pre["ident"])


# ================================================================================================ B4: FT_FA_WARM
def test_existing_device_spec_digests_unchanged():
    sel = {"LO": {"arm": "LO", "cell": "LO_P4"}, "FT_C": {"arm": "FT_C", "cell": "FT_C_f0.3_P4"},
           "FT_FA": {"arm": "FT_FA", "cell": "FT_FA_f0.1_P2"}}
    for (arm, role), want in PRE_DEVFTPRE_DIGESTS.items():
        kw = dict(select=True) if role == "UNSEEN_VAL" else dict(selection=sel[arm])
        assert D.spec_digest(D.device_spec(arm, seed=2026, peak_lr=0.000711, role=role, start_digest="s" * 64,
                                           fused=True, **kw)) == want, (arm, role)
    reg = D.load_selection(D.FT_PRE_SELECTION)
    for (role, cell), want in PRE_FL15_FT_PRE.items():
        s = SEL_C if cell == SEL_C["cell"] else reg
        assert D.spec_digest(D.device_spec("FT_PRE", seed=2026, peak_lr=0.000711, role=role, start_digest="s" * 64,
                                           fused=True, selection=s)) == want, (role, cell)
    assert D.spec_digest(D.device_spec("FT_FA", seed=2026, peak_lr=0.000711, role="MAIN", start_digest="s" * 64,
                                       fused=True, selection=sel["FT_FA"])) == PRE_FL15_FT_FA
    assert D.FT_PRE_SELECTION_SHA256.startswith("e47965d2") and D.FT_PRE_SELECTION_SHA256 == _sha(D.FT_PRE_SELECTION)


def test_ft_fa_warm_spec_applies_the_registered_ft_c_selection():
    reg = D.load_selection(D.FT_PRE_SELECTION)
    for arm, base in (("FT_FA_WARM", "FA_WARM"), ("FT_FA_WARM_FROZEN", "FA_WARM_FROZEN")):
        s = D.device_spec(arm, seed=2026, peak_lr=PEAK, role="MAIN", start_digest="x", selection=reg)
        assert s["selected_cell"] == f"{arm}_f0.3_P4" and s["cells"] == [{"cell": f"{arm}_f0.3_P4", "passes": 4,
                                                                            "lr_frac": 0.3}]
        assert s["ft_procedure"] == "FT_C" and s["procedure_selection_cell"] == "FT_C_f0.3_P4"
        assert s["procedure_selection_arm"] == "FT_C" and s["base_arm"] == base
        assert D.LOCK_CODE[arm] == "FA" and D.FT_PROCEDURE[arm] == "FT_C"
        for kw, msg in ((dict(selection=dict(SEL_C, arm=arm)), "not FT_C"),
                        (dict(select=True, role="UNSEEN_VAL"), "--select refused"),
                        (dict(selection=SEL_C, role="UNSEEN_VAL"), "never run on the UNSEEN_VAL"),
                        (dict(), "needs --selection")):
            args = dict(seed=2026, peak_lr=PEAK, role="MAIN", start_digest="x")
            args.update(kw)
            with pytest.raises(D.LaunchRefused, match=msg):
                D.device_spec(arm, **args)


@pytest.fixture(scope="module")
def warm_base(pre, tmp_path_factory):
    """Finished tiny FL runs of the K = 24 untied model: FA_WARM MAIN (seed 2026), FA_WARM_FROZEN MAIN, an FA_WARM
    CALIBRATION run, a cold FA MAIN run and an FA_WARM MAIN run whose id carries the C_BIG marker."""
    root = tmp_path_factory.mktemp("warmbase")
    ident = pre["ident"]
    runs = {"warm": ("FL_FA_WARM_SMALL_s2026", tiny_spec("FA_WARM", init_from=ident), warm_world(pre)),
            "frozen": ("FL_FA_WARM_FROZEN_SMALL_s2026", tiny_spec("FA_WARM_FROZEN", init_from=ident), warm_world(pre)),
            "calib": ("FL_FA_WARM_LRSCREEN_x1_CALIB_s2026", tiny_spec("FA_WARM", init_from=ident, role="CALIBRATION"),
                      warm_world(pre, eval_id=CALIB_EVAL)),
            "cold": ("FL_FA_SMALL_s2026", tiny_spec("FA"), untied_world()),
            "cbig": ("C_BIG_SASREC_s2026", tiny_spec("FA_WARM", init_from=ident), warm_world(pre))}
    out = {"root": root}
    for key, (rid, spec, world) in runs.items():
        res = F.execute(spec=spec, run_id=rid, run_dir=root / key, say=lambda *_: None, **world)
        assert res["status"] == "DONE_MAIN", key
        out[key] = root / key / "ckpt" / rid / "best.pt"
        out[key + "_end"] = root / key / "ckpt" / rid / "endpoint_6.0.pt"
        out[key + "_server"] = world["server_adapter"]
    return out


def _worker():
    return make_tiny_adapter(K=K, seed=1, tied=False)


def _start_args(**kw):
    return _args(**dict(dict(arm="FT_FA_WARM", base_kind=D.WARM_BASE_KIND, dry_run=False), **kw))


def test_load_warm_base_is_the_fa_warm_checkpoint(warm_base, pre):
    from fl.checkpoint import load_checkpoint
    for key, arm in (("warm", "FA_WARM"), ("warm_end", "FA_WARM"), ("frozen", "FA_WARM_FROZEN")):
        w = _worker()
        theta0 = state_digest(w.broadcast_state())
        b = D.load_warm_base(w, warm_base[key], base_arm=arm, seed=2026)
        ck = load_checkpoint(warm_base[key])
        assert b["digest"] == state_digest(ck["theta"]) == state_digest(b["state"]) != theta0
        assert b["kind"] == arm and b["base_kind"] == "fa_warm" and b["file_sha256"] == _sha(warm_base[key])
        assert b["init_from"] == pre["ident"] and b["checkpoint"] == warm_base[key].name
        assert b["run_id"] == warm_base[key].parent.name
    fz = D.load_warm_base(_worker(), warm_base["frozen"], base_arm="FA_WARM_FROZEN", seed=2026)
    t_pre = pre["adapter"].broadcast_state()
    assert all(torch.equal(fz["state"][k], t_pre[k]) for k in F.FROZEN_ITEM_KEYS)    # the tables are theta_pre's
    assert fz["frozen_item_tables"] == list(F.FROZEN_ITEM_KEYS)
    end = warm_base["warm_server"].broadcast_state()                                # the run's final server state
    assert D.load_warm_base(_worker(), warm_base["warm_end"], base_arm="FA_WARM", seed=2026)["digest"] == \
        state_digest(end)
    st, base, rec = D.resolve_start(_start_args(base=warm_base["warm"]), None, _worker(), theta0="t0", model_variant=VAR)
    assert rec["kind"] == "FA_WARM" and "state" not in rec and base["digest"] == state_digest(st)
    lab = D.ft_warm_row_labels(rec, {"self_sha256": "q" * 64})
    assert lab["base_source"] == "FA_WARM" and lab["base_run_id"] == "FL_FA_WARM_SMALL_s2026"
    assert lab["base_checkpoint_sha256"] == _sha(warm_base["warm"]) and lab["base_init_run_id"] == pre["ident"]["run_id"]
    assert lab["base_init_train_manifest_id"] == "PRETRAIN_TRAIN_5_100_NONMAIN" and lab["ft_selection_self_sha256"] == "q" * 64
    st2, _b, rec2 = D.resolve_start(_start_args(arm="FT_FA_WARM_FROZEN", base=warm_base["frozen"]), None, _worker(),
                                    theta0="t0", model_variant=VAR)
    assert rec2["kind"] == "FA_WARM_FROZEN"
    D.load_warm_base(_worker(), warm_base["warm"], base_arm="FA_WARM", seed=2026, want_sha256=_sha(warm_base["warm"]))


def _forge(src_run: Path, dst: Path, edit) -> Path:
    """A copy of a finished FL run whose recorded arm spec is edited (`edit(spec)`) and consistently re-hashed in
    ARM_SPEC.json and RUN_SUMMARY.json; returns its best.pt."""
    shutil.copytree(src_run, dst)
    side, summ = _json(dst / "ARM_SPEC.json"), _json(dst / "RUN_SUMMARY.json")
    spec = summ["arm_spec"]
    edit(spec)
    dg = F.spec_digest(spec)
    side.update(arm_spec=spec, arm_spec_sha256=dg)
    summ.update(arm_spec=spec, arm_spec_sha256=dg)
    (dst / "ARM_SPEC.json").write_text(json.dumps(side), encoding="utf-8")
    (dst / "RUN_SUMMARY.json").write_text(json.dumps(summ), encoding="utf-8")
    return next((dst / "ckpt").glob("*/best.pt"))


def test_load_warm_base_refusals(warm_base, tmp_path):
    LW = D.load_warm_base
    run_warm = warm_base["warm"].parents[2]
    cases = [(warm_base["frozen"], "FA_WARM", 2026, "is not FA_WARM"),               # another warm arm
             (warm_base["warm"], "FA_WARM_FROZEN", 2026, "is not FA_WARM_FROZEN"),
             (warm_base["cold"], "FA_WARM", 2026, "is not FA_WARM"),                  # a cold-start FA run
             (warm_base["calib"], "FA_WARM", 2026, "role"),                           # a CALIBRATION screen run
             (warm_base["warm"], "FA_WARM", 2027, "seed"),                            # FT_FA_WARM - FA_WARM per seed
             (warm_base["cbig"], "FA_WARM", 2026, "MAIN users")]                      # a C_BIG run id
    for path, arm, seed, msg in cases:
        with pytest.raises(D.LaunchRefused, match=msg):
            LW(_worker(), path, base_arm=arm, seed=seed)
    stray = tmp_path / "stray" / "best.pt"                                            # outside <run>/ckpt/<id>/
    stray.parent.mkdir()
    stray.write_bytes(warm_base["warm"].read_bytes())
    with pytest.raises(D.LaunchRefused, match="not <run_dir>/ckpt"):
        LW(_worker(), stray, base_arm="FA_WARM", seed=2026)
    with pytest.raises(D.LaunchRefused, match="sha256"):
        LW(_worker(), warm_base["warm"], base_arm="FA_WARM", seed=2026, want_sha256="0" * 64)
    shutil.copytree(run_warm, tmp_path / "tamper")                                  # spec edited, digest not
    side = _json(tmp_path / "tamper" / "ARM_SPEC.json")
    side["arm_spec"]["cfg"]["max_attempts"] = 3
    (tmp_path / "tamper" / "ARM_SPEC.json").write_text(json.dumps(side), encoding="utf-8")
    with pytest.raises(D.LaunchRefused, match="arm_spec_sha256"):
        LW(_worker(), next((tmp_path / "tamper" / "ckpt").glob("*/best.pt")), base_arm="FA_WARM", seed=2026)
    cfgd = _forge(run_warm, tmp_path / "cfg", lambda s: s["cfg"].update(max_attempts=3))   # re-hashed: config differs
    with pytest.raises(D.LaunchRefused, match="config_digest"):
        LW(_worker(), cfgd, base_arm="FA_WARM", seed=2026)
    mainp = _forge(run_warm, tmp_path / "main", lambda s: s["init_from"].update(train_manifest="MAIN_TRAIN_5_100_131072"))
    with pytest.raises(D.LaunchRefused, match="MAIN users"):
        LW(_worker(), mainp, base_arm="FA_WARM", seed=2026)
    shutil.copytree(run_warm, tmp_path / "unfinished")                              # not DONE_MAIN
    summ = _json(tmp_path / "unfinished" / "RUN_SUMMARY.json")
    summ["status"] = "SMOKE_DONE_NOT_EVIDENCE"
    (tmp_path / "unfinished" / "RUN_SUMMARY.json").write_text(json.dumps(summ), encoding="utf-8")
    with pytest.raises(D.LaunchRefused, match="DONE_MAIN"):
        LW(_worker(), next((tmp_path / "unfinished" / "ckpt").glob("*/best.pt")), base_arm="FA_WARM", seed=2026)
    (tmp_path / "unfinished" / "RUN_SUMMARY.json").unlink()
    with pytest.raises(D.LaunchRefused, match="unreadable"):
        LW(_worker(), next((tmp_path / "unfinished" / "ckpt").glob("*/best.pt")), base_arm="FA_WARM", seed=2026)
    for arm in ("FT_FA_WARM", "FT_FA_WARM_FROZEN"):                                 # through run()'s start helper
        with pytest.raises(D.LaunchRefused):
            D.resolve_start(_start_args(arm=arm, base=warm_base["cold"]), None, _worker(), theta0="t0",
                            model_variant=VAR)
    with pytest.raises(D.LaunchRefused, match="not a verified FA_WARM"):             # a start of another kind
        D.refuse_theta_pre_base("FT_FA_WARM", {"kind": "fl", "run_id": "FL_FA_SMALL_s2026"})


def test_ft_fa_and_ft_c_refuse_a_warm_base(warm_base, tmp_path):
    """REV-FL-15 m3: FT_FA (--base-kind fl) refuses an FA_WARM / FA_WARM_FROZEN checkpoint (by run id FL_FA_WARM* and
    by the run's recorded arm, e.g. the C_BIG-named FA_WARM run); a cold-start FA base still loads for FT_FA."""
    for key in ("warm", "warm_end", "frozen", "calib", "cbig"):
        with pytest.raises(D.LaunchRefused, match="FT_FA_WARM"):
            D.resolve_start(_start_args(arm="FT_FA", base=warm_base[key], base_kind="fl"), None, _worker(), theta0="t0",
                            model_variant=VAR)
    st, base, rec = D.resolve_start(_start_args(arm="FT_FA", base=warm_base["cold"], base_kind="fl"), None, _worker(),
                                    theta0="t0", model_variant=VAR)
    assert rec["kind"] == "fl" and rec["run_id"] == "FL_FA_SMALL_s2026"
    with pytest.raises(D.LaunchRefused, match="warm FL checkpoint"):                 # run id alone (no FL layout)
        D.refuse_warm_base("FT_C", tmp_path / "x.pt", {"run_id": "FL_FA_WARM_FROZEN_SMALL_s2027"})
    D.refuse_warm_base("FT_C", tmp_path / "x.pt", {"run_id": "02B_SASREC_MAIN_s2026"})


def test_ft_fa_warm_preflight(warm_base, tmp_path):
    gate = tmp_path / "G1.json"
    gate.write_text("{}", encoding="utf-8")
    b = warm_base["warm"]
    pre = D.preflight(_args(arm="FT_FA_WARM", base=b, base_kind="fa_warm"), environ={}, gate_path=gate)
    assert pre["ft_fa_warm"]["selection"]["is_registered_file"] and pre["ft_fa_warm"]["base_arm"] == "FA_WARM"
    assert pre["ft_fa_warm"]["selection"]["file_sha256"] == D.FT_PRE_SELECTION_SHA256
    other = _selection_file(tmp_path)
    cases = [(dict(arm="FT_FA_WARM"), "needs --base"),
             (dict(arm="FT_FA_WARM", base=b, base_kind="fl"), "needs --base"),
             (dict(arm="FT_FA_WARM", base=tmp_path / "nope.pt", base_kind="fa_warm"), "does not exist"),
             (dict(arm="FT_FA_WARM", base=b, base_kind="fa_warm", init_from=tmp_path / "i.json"), "--init-from"),
             (dict(arm="FT_FA_WARM", base=b, base_kind="fa_warm", select=True, role="UNSEEN_VAL"), "--select"),
             (dict(arm="FT_FA_WARM", base=b, base_kind="fa_warm", role="UNSEEN_VAL"), "never run on the UNSEEN_VAL"),
             (dict(arm="FT_FA_WARM", base=b, base_kind="fa_warm", selection=other), "registered"),
             (dict(arm="FT_FA_WARM_FROZEN", base=b, base_kind="fa_warm",
                   selection=_selection_file(tmp_path, arm="FT_FA", cell="FT_FA_f0.3_P4", name="FA.json")),
              "registered FT_C selection"),
             (dict(arm="FT_FA", base=b, base_kind="fa_warm", selection=other), "fa_warm"),
             (dict(arm="LO", base_kind="fa_warm"), "fa_warm")]
    for kw, msg in cases:
        with pytest.raises(D.LaunchRefused, match=msg):
            D.preflight(_args(**kw), environ={}, gate_path=gate)
    smoke = D.preflight(_args(arm="FT_FA_WARM", base=b, base_kind="fa_warm", selection=other, smoke=True), environ={},
                        gate_path=gate)
    assert smoke["ft_fa_warm"]["selection"]["is_registered_file"] is False
    a = D.parse_args(["--arm", "FT_FA_WARM_FROZEN", "--seed", "2026", "--gpu", "0", "--run-dir", "r", "--history-policy",
                      "USER_HISTORY_L50", "--base", str(b), "--base-kind", "fa_warm"])
    assert a.base_kind == "fa_warm" and a.arm == "FT_FA_WARM_FROZEN"


def _row_fn(spec, over):
    from v4eval.rows import COMPARISON_KEY
    kb = {f: "x" for f in COMPARISON_KEY}
    kb.update(aggregation="macro", target_denominator="E2E")
    return lambda res, cell, selected: D.device_rows(res, kb, {"run_label": "PRODUCTION"}, spec=spec, cell=cell,
                                                     selected=selected, down_bytes=7, overrides=over)


def test_ft_fa_warm_cell_is_the_ft_c_procedure_on_the_base(warm_base, tmp_path):
    w = _world()
    st, base, rec = D.resolve_start(_start_args(base=warm_base["warm"]), None, _worker(), theta0="t0",
                                    model_variant=VAR)
    out = {}
    for arm in ("FT_FA_WARM", "FT_C"):
        spec = D.device_spec(arm, seed=2026, peak_lr=PEAK, role="MAIN", start_digest=state_digest(st), selection=SEL_C)
        over = D.ft_warm_row_labels(rec, SEL_C) if arm == "FT_FA_WARM" else None
        out[arm] = D.execute(spec=spec, run_dir=tmp_path / arm, worker=make_tiny_adapter(K=K, seed=3, tied=False),
                             start_state=st, plans=_plans(w), get_support=lambda q: w.sup[q.key],
                             get_eval_batch=w.get_eval_batch, manifest=w.m, frozen=_frozen(), row_fn=_row_fn(spec, over),
                             say=lambda *_: None, n_shards=2)
    a = np.load(tmp_path / "FT_FA_WARM" / "cells" / "FT_FA_WARM_f0.3_P2.ranks_int32.npy")
    b = np.load(tmp_path / "FT_C" / "cells" / "FT_C_f0.3_P2.ranks_int32.npy")
    assert np.array_equal(a, b) and out["FT_FA_WARM"]["values"]["FT_FA_WARM_f0.3_P2"] == out["FT_C"]["values"][
        "FT_C_f0.3_P2"]
    rows = _json(tmp_path / "FT_FA_WARM" / "RESULT_ROWS.json")["rows"]
    assert rows and {r["method"] for r in rows} == {"FT_FA_WARM"} and {r["ft_procedure"] for r in rows} == {"FT_C"}
    assert {r["base_source"] for r in rows} == {"FA_WARM"} and {r["base_run_id"] for r in rows} == \
        {"FL_FA_WARM_SMALL_s2026"} and {r["training_paradigm"] for r in rows} == {"CLIENT_FINETUNE"}
    assert {r["procedure_selection_cell"] for r in rows} == {"FT_C_f0.3_P2"}
    assert _json(tmp_path / "FT_FA_WARM" / "RUN_SUMMARY.json")["spec"]["base_arm"] == "FA_WARM"


# ================================================================================================ R5: training loss
def test_round_loss_record_units():
    from fl.runtime import round_loss_record
    V = lambda steps, loss: SimpleNamespace(steps=steps, loss_sum=loss)  # noqa: E731
    r = round_loss_record([V(2, 3.0), V(4, 4.0), V(0, 0.0)])
    assert r == {"n_clients": 2, "n_steps": 6, "loss_sum": 7.0, "mean_step_loss": 7.0 / 6,
                 "client_mean_loss": (1.5 + 1.0) / 2, "finite": True}
    e = round_loss_record([])
    assert e["n_clients"] == 0 and e["loss_sum"] is None and e["mean_step_loss"] is None and e["client_mean_loss"] is None
    n = round_loss_record([V(2, float("nan"))])
    assert n["finite"] is False and n["loss_sum"] is None and n["client_mean_loss"] is None


def _loss_lines(d: Path) -> list:
    return F.read_train_loss(d)


def test_train_loss_log_equals_the_visit_records_and_resumes(tmp_path, monkeypatch):
    import fl.runtime
    from fl.runtime import round_loss_record
    seen = []
    real = fl.runtime.run_round

    def spy(*a, **kw):
        rep = real(*a, **kw)
        seen.append((rep.round_idx, round_loss_record(rep.visits), [(v.steps, v.loss_sum) for v in rep.visits]))
        return rep
    monkeypatch.setattr(fl.runtime, "run_round", spy)
    a = run_tiny(tmp_path, _tiny("FA"), "a")
    assert a["summary"]["server_digest"] == PRE_FL10_TRAJ["FA"]                    # numerics unchanged
    lines = _loss_lines(tmp_path / "a")
    assert [r["round"] for r in lines] == list(range(a["summary"]["rounds"]))
    by = {i: (want, visits) for i, want, visits in seen}
    assert len(by) >= len(lines) - 2                                                 # (a round may have 0 survivors)
    for r in lines:
        if r["round"] not in by:
            assert r["n_clients"] == 0 and r["loss_sum"] is None
            continue
        want, visits = by[r["round"]]
        assert {k: r[k] for k in want} == want and r["n_steps"] == sum(s for s, _ in visits if s > 0)
        assert r["mean_step_loss"] == r["loss_sum"] / r["n_steps"] and r["finite"] and r["client_mean_loss"] > 0
    tl = a["summary"]["train_loss_log"]
    assert tl["complete"] and tl["rounds_logged"] == len(lines) and tl["sha256"] == _sha(tmp_path / "a" / F.TRAIN_LOSS_FILE)
    assert _json(tmp_path / "a" / "RUN_SUMMARY.json")["train_loss_log"]["sha256"] == tl["sha256"]
    k = a["summary"]["rounds"] // 2
    part = run_tiny(tmp_path, _tiny("FA"), "p", interrupt_at=k)
    assert part["status"] == "INTERRUPTED" and [r["round"] for r in _loss_lines(tmp_path / "p")] == list(range(k))
    with open(tmp_path / "p" / F.TRAIN_LOSS_FILE, "a", encoding="utf-8") as f:     # a stale line past the checkpoint
        f.write(json.dumps({"round": k + 1, "loss_sum": 9.9}) + "\n")
    res = run_tiny(tmp_path, _tiny("FA"), "p", resume=True)
    assert res["summary"]["server_digest"] == a["summary"]["server_digest"]
    assert (tmp_path / "p" / F.TRAIN_LOSS_FILE).read_bytes() == (tmp_path / "a" / F.TRAIN_LOSS_FILE).read_bytes()
    d = F.train_loss_summary(tmp_path / "a", len(lines) + 1)
    assert d["complete"] is False and d["missing_rounds"] == [len(lines)]


def test_train_loss_never_for_dp_and_pinned_trajectories(tmp_path):
    for arm in ("DP", "FA_1024", "FP", "FA_Q8"):
        out = run_tiny(tmp_path, _tiny(arm), arm)
        assert out["status"] == "DONE_MAIN" and out["summary"]["server_digest"] == PRE_FL10_TRAJ[arm]
        has = (tmp_path / arm / F.TRAIN_LOSS_FILE).exists()
        assert has == (arm != "DP") and ("train_loss_log" in out["summary"]) == (arm != "DP")
    assert not F.train_loss_applies(_tiny("DP")) and F.train_loss_applies(_tiny("FA_1024"))
    assert "06L_E E1" in F.TRAIN_LOSS_AUTHORITY


# ================================================================================================ negative controls
@nc("select-warm with the grid cut to 2 cells accepts the superseded 06K 2-run screen")
def test_nc_2run_screen_accepted(warm_sel, monkeypatch):
    runs = warm_runs(warm_sel, "FA_WARM")[:2]
    two = F.warm_grid("FA_WARM", warm_sel["w4"]["optimizer"])[:2]
    monkeypatch.setattr(F, "warm_grid", lambda arm, opt: list(two))                   # MUTANT
    assert_raises(L.SelectionRefused, L.selection_warm, "S7", runs, warm_sel["s4_file"])


@nc("the 1-D warm selection file configures a (dry-run) MAIN warm spec when both 2-D guards (warm_selection_winner "
    "and the server-axis check) are removed")
def test_nc_1d_file_configures_main(warm_sel, s7_file, pre, tmp_path, monkeypatch):
    reg = legacy_1d(s7_file["obj"], tmp_path / "contracts_fl" / "FL_WARM_SELECTION_S7.json")
    monkeypatch.setitem(F.WARM_SELECTION_PATH, "S7", reg)
    monkeypatch.setattr(F, "warm_selection_winner", lambda obj, path: (float(obj["winner_factor"]), 1.0))   # MUTANT
    monkeypatch.setattr(F, "check_warm_server_axis", lambda obj, tw, path: None)     # (defence in depth: both)
    assert_raises(F.LaunchRefused, F.resolve_spec, _main_args(tmp_path, pre, warm_sel, reg, dry=True), 0.000711,
                  fused=True, init_identity=pre["ident"])


@nc("an unregistered server-LR factor (FedAdam x 0.25) accepted by arm_spec")
def test_nc_unregistered_server_factor(monkeypatch):
    monkeypatch.setattr(F, "warm_server_factor_value", lambda arm, opt, f: 1.0 if f is None else float(f))   # MUTANT
    monkeypatch.setitem(F.WARM_SERVER_DIVISORS, 0.25, 4)
    assert_raises(F.LaunchRefused, F.arm_spec, "FA_WARM", seed=2026, peak_lr=0.000711, init_from=IDENT,
                  server_opt="fedadam", server_lr=0.01, warm_server_lr_factor=0.25)


@nc("a non-smoke MAIN warm run takes --warm-server-lr-factor from the CLI (the server preflight removed)")
def test_nc_main_cli_server_factor(pre, tmp_path, monkeypatch):
    monkeypatch.setattr(F, "warm_server_preflight", lambda args, main_run: None)      # MUTANT
    argv = ["--arm", "FA_WARM", "--seed", "2026", "--gpu", "0", "--run-dir", str(tmp_path / "r"), "--history-policy",
            "USER_HISTORY_L50", "--init-from", str(pre["path"]), "--warm-selection", str(tmp_path / "w.json"),
            "--warm-server-lr-factor", "0.25"]
    assert_raises(F.LaunchRefused, F.fl05_preflight, F.parse_args(argv))


@nc("a warm file whose server axis is not the bound tuning winner's binds (the axis check removed)")
def test_nc_foreign_server_axis_binds(warm_sel, s7_file, pre, tmp_path, monkeypatch):
    mb = _bind_main(monkeypatch, warm_sel)
    foreign = 0.1 if warm_sel["w4"]["server_lr"] != 0.1 else 0.01
    reg = _registered_s7(tmp_path, monkeypatch, s7_file["obj"], eta=foreign)
    monkeypatch.setattr(F, "check_warm_server_axis", lambda obj, tw, path: None)     # MUTANT
    assert_raises(F.LaunchRefused, F.bind_warm_selection, _main_args(tmp_path, pre, warm_sel, reg), pre["ident"], mb)


@nc("the 2-D rule without 06J §4(ii) (MIN_GAIN 0) lets a candidate with gain 0.0019 replace (m*, eta*)")
def test_nc_min_gain_dropped(monkeypatch):
    monkeypatch.setattr(L, "MIN_GAIN", 0.0)                                           # MUTANT
    assert L.warm_decision("FA_WARM", "fedadam", _v(0.100, 0.1015, 0.1019, 0.099))["winner"] == (1.0, 1.0)


@nc("an excluded incumbent (m*, eta*) is replaced silently by the best remaining cell")
def test_nc_excluded_incumbent_accepted(monkeypatch):
    monkeypatch.setattr(L, "warm_incumbent", lambda vals: max(vals, key=vals.get))   # MUTANT
    assert_raises(L.SelectionRefused, L.warm_decision, "FA_WARM", "fedadam", {(0.25, 1.0): 0.2, (1.0, 0.1): 0.1})


@nc("a MAIN warm run whose resolve_spec drops the selected server-LR factor keeps eta* (bind returns factor 1)")
def test_nc_main_run_ignores_selected_server_lr(warm_sel, s7_file, pre, tmp_path, monkeypatch):
    reg = _registered_s7(tmp_path, monkeypatch, s7_file["obj"], values=WIN_025_01)
    _bind_main(monkeypatch, warm_sel)
    real = F.bind_warm_selection
    monkeypatch.setattr(F, "bind_warm_selection",                                     # MUTANT
                        lambda *a, **k: dict(real(*a, **k), winner_server_lr_factor=1.0))
    spec, _ = F.resolve_spec(_main_args(tmp_path, pre, warm_sel, reg), 0.000711, fused=True, init_identity=pre["ident"])
    assert spec.get("warm_server_lr_factor") == 0.1


@nc("FT_FA_WARM's base loader replaced by the plain FL loader accepts a cold-start FA checkpoint")
def test_nc_warm_base_unchecked(warm_base, monkeypatch):
    def plain(adapter, path, *, base_arm, seed, want_sha256=None):                   # MUTANT
        b = D.load_base(adapter, path, "fl", want_sha256)
        return dict(b, kind=base_arm, base_kind=D.WARM_BASE_KIND)
    monkeypatch.setattr(D, "load_warm_base", plain)
    assert_raises(D.LaunchRefused, D.resolve_start, _start_args(base=warm_base["cold"]), None, _worker(), theta0="t0",
                  model_variant=VAR)


@nc("a warm-base guard without the MAIN checks accepts a base whose init was trained on MAIN users")
def test_nc_warm_base_main_trained(warm_base, tmp_path, monkeypatch):
    p = _forge(warm_base["warm"].parents[2], tmp_path / "main",
               lambda s: s["init_from"].update(train_manifest="MAIN_TRAIN_5_100_131072"))
    monkeypatch.setattr(F, "MAIN_TRAIN_MANIFESTS", ())                               # MUTANT (deny-list emptied,
    monkeypatch.setattr(F, "MAIN_RUN_MARKERS", ())                                   # allow-list pointed at MAIN)
    monkeypatch.setitem(F.PRETRAIN, "train_manifest", "MAIN_TRAIN_5_100_131072")
    assert_raises(D.LaunchRefused, D.load_warm_base, _worker(), p, base_arm="FA_WARM", seed=2026)


@nc("an FT_FA_WARM argument guard that is not called lets a real run apply a non-registered selection")
def test_nc_ft_warm_args_skipped(warm_base, tmp_path, monkeypatch):
    gate = tmp_path / "G1.json"
    gate.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(D, "check_ft_warm_args", lambda args: {})                    # MUTANT
    assert_raises(D.LaunchRefused, D.preflight, _args(arm="FT_FA_WARM", base=warm_base["warm"], base_kind="fa_warm",
                                                      selection=_selection_file(tmp_path)), environ={}, gate_path=gate)


@nc("REV-FL-15 m2: select-warm without the S4 stage check writes a warm file against an S3 stage file")
def test_nc_select_warm_non_s4_tuning(warm_s3, monkeypatch):
    monkeypatch.setattr(L, "check_warm_tuning_stage", lambda tobj, tuning: None)     # MUTANT
    assert_raises(L.SelectionRefused, L.selection_warm, "S7", warm_s3["dirs"], warm_s3["s3_file"])


@nc("REV-FL-15 m3: FT_FA without the warm-base guard runs from an FA_WARM checkpoint (--base-kind fl)")
def test_nc_ft_fa_from_warm_base(warm_base, monkeypatch):
    monkeypatch.setattr(D, "refuse_warm_base", lambda arm, path, rec: None)          # MUTANT
    assert_raises(D.LaunchRefused, D.resolve_start, _start_args(arm="FT_FA", base=warm_base["warm"], base_kind="fl"),
                  None, _worker(), theta0="t0", model_variant=VAR)


@nc("REV-FL-15 m4: a MAIN warm --dry-run that only loads the file (no binding) accepts a foreign server axis")
def test_nc_dry_run_without_binding(warm_sel, s7_file, pre, tmp_path, monkeypatch):
    foreign = 0.1 if warm_sel["w4"]["server_lr"] != 0.1 else 0.01
    reg = _registered_s7(tmp_path, monkeypatch, s7_file["obj"], eta=foreign)

    def load_only(args, init_identity, binding):                                      # MUTANT (the pre-fix dry branch)
        wf, wsf = F.warm_selection_winner(F.load_warm_selection(args.warm_selection), args.warm_selection)
        return {"winner_factor": wf, "winner_server_lr_factor": wsf}
    monkeypatch.setattr(F, "bind_warm_selection", load_only)
    assert_raises(F.LaunchRefused, F.resolve_spec, _main_args(tmp_path, pre, warm_sel, reg, dry=True), 0.000711,
                  fused=True, init_identity=pre["ident"])


@nc("TRAIN_LOSS.jsonl not trimmed on resume keeps a stale line (the resumed log differs from the straight run)")
def test_nc_train_loss_not_trimmed(tmp_path, monkeypatch):
    a = run_tiny(tmp_path, _tiny("FA"), "a")
    k = a["summary"]["rounds"] // 2
    part = run_tiny(tmp_path, _tiny("FA"), "p", interrupt_at=k)
    if part["status"] != "INTERRUPTED":                                               # a setup error, not the NC
        raise RuntimeError("the interrupted leg did not interrupt")
    with open(tmp_path / "p" / F.TRAIN_LOSS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"round": k + 1, "loss_sum": 9.9}) + "\n")
    monkeypatch.setattr(F, "trim_train_loss", lambda run_dir, cursor: None)          # MUTANT
    try:
        run_tiny(tmp_path, _tiny("FA"), "p", resume=True)
    except RuntimeError as e:                                                         # duplicate rounds detected
        raise AssertionError(str(e)) from e
    assert (tmp_path / "p" / F.TRAIN_LOSS_FILE).read_bytes() == (tmp_path / "a" / F.TRAIN_LOSS_FILE).read_bytes()
    assert a["status"] == "DONE_MAIN"


@nc("a z > 0 (DP) run writes TRAIN_LOSS.jsonl when the DP gate is removed")
def test_nc_dp_train_loss(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "train_loss_applies", lambda spec: True)                  # MUTANT
    run_tiny(tmp_path, _tiny("DP"), "dp")
    assert not (tmp_path / "dp" / F.TRAIN_LOSS_FILE).exists()
