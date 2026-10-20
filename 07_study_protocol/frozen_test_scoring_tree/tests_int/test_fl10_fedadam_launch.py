"""FL-10 (PREG1_AMENDMENT_06G §1-§3, 06I §3, 06J §1-§2) — FedAdam / FedAvgM / FL_CONST / client SGD-m in launch_fl and the
staged FL tuning selection (CPU, the tiny FL world of test_fl05_launch).

Proves: every (arm spec sha256, fl RunConfig digest) pair at the defaults over all 7 arms x seeds x fused modes x DP z x
all 7 multipliers + the CALIB screen (224 pairs) is byte-identical to the PRE-FL-10 code (aggregate below, computed
2026-09-29 from launch_fl.py ca93ceea… / fl/runtime.py d51e2dc0… / fl/server.py 63ccf395… / fl/client.py 8aabe73c…), and
the tiny-world trajectories of all 7 arms are bit-identical (server digests below, same pre-FL-10 code); every tuning
value is registered (refusals); spec / cfg / rows carry only the non-default dimensions and spec_tuning re-verifies
them; each option runs deterministically, differs from the default and resumes bitwise; DP / FA_1024 run with a server
optimizer; `lr_screen select-joint --stage S2 | S3` writes O_EXCL self-hashed stage files with the 06J rules (required
runs, edge conditions, tie orders) and prints only the winner + sha; launch_fl verifies them (--fl-tuning) and, once
FL_TUNING_SELECTION.json exists, a non-smoke MAIN run refuses the LR_SELECTION files. NCs: fedadam without --server-lr;
--server-lr with fedavg; a MAIN run without the tuning file once it exists; the S2 selection with a missing (FedAvgM)
run; an unregistered client LR; an unregistered LR shape.
FIX-DELTA-02 (REV-DELTA-02 B1 / M3): the stage fixtures screen the registered edge runs that select-joint names
(EdgeRunsMissing) before writing a stage file; the 06J §4(ii) and edge-requirement tests live in
tests_int/test_fix_delta02.py.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

import v4int.launch_fl as F
import v4int.lr_screen as L
from intkit import assert_raises, nc
from test_fl05_launch import _args, fl05_world, run_tiny, tiny_spec

PEAK_REG = 0.000711
CALIB_EVAL = "M06_CALIB_VALIDATION_WARM"
PRE_FL10_SHA = "1b0a98fb8cb790fafdc52f8c99129672b3445d0ce7eff0b861a45cab7e2de61a"   # 224 (spec, RunConfig) pairs
PRE_FL10_TRAJ = {
    "DP": "3660db83812a69e3d36277e2fba368410338a8e7ff09dab537453a43725e8bfb",
    "FA": "1ed4b587e9b01652ff11033fe420daadbdd7dc5f67230d97cdc49f5b55b01f95",
    "FA_1024": "c7fb6298b329b18d18d961944b583f646059385e8c9acbff4d7ffc56c19baa2c",
    "FA_Q8": "279af659d4f8fa0d3f2a8c597263dfcc0b2d90e896721223be25d926caefd75e",
    "FP": "f727265af7caaec7b95a75577ab71ceb8c1d738111453f6e4b44edc0ce192e12",
    "PF": "83f2fac7cea2ea279afda3ae164fc1048b3c6d1f82e74976bd0d0ca7e2550720",
    "S_CAL": "8b64c614b0c5ebf2a74ab7a57aa6249b22acc367d314110910ee06dcbd4697aa"}
ARM_KW = {"FA": {}, "FP": {}, "PF": {}, "FA_Q8": {}, "FA_1024": {"clip_norm": 0.5},
          "DP": {"clip_norm": 0.5, "z": 0.5642}, "S_CAL": {}}
T = F.tuning_config
OPTIONS = {"fedadam": dict(server_opt="fedadam", server_lr=0.1), "fedavgm": dict(server_opt="fedavgm"),
           "fl_const": dict(lr_shape="fl_const"), "sgd_m": dict(client_opt="sgd_m", client_lr=0.1, lr_shape="fl_const"),
           "fedadam_tau": dict(server_opt="fedadam", server_lr=0.01, server_tau=1e-5)}


def _json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def all_grid_digests() -> dict:
    out = {}
    for arm in F.ARMS:
        for seed in (2026, 2027):
            if arm == "S_CAL" and seed != 2026:
                continue
            for fused in (True, None):
                for z in ((0.5642, 1.3094) if arm == "DP" else (None,)):
                    for m in F.LR_MULTS:
                        kw = {}
                        if arm in ("FA_1024", "DP"):
                            kw["clip_norm"] = 1.2345
                        if arm == "DP":
                            kw["z"] = z
                        s = F.arm_spec(arm, seed=seed, peak_lr=PEAK_REG, fused=fused, lr_mult=m, **kw)
                        out[f"{arm}|{seed}|{fused}|{z}|{m}|MAIN"] = [
                            F.spec_digest(s), F.fl_run_config(s, f"FL_{arm}_SMALL_s{seed}", 64).digest()]
    for m in F.LR_MULTS:
        for fused in (True, None):
            s = F.arm_spec("FA", seed=2026, peak_lr=PEAK_REG, fused=fused, lr_mult=m, role="CALIBRATION")
            out[f"FA|2026|{fused}|None|{m}|CALIBRATION"] = [F.spec_digest(s),
                                                            F.fl_run_config(s, "FL_FA_LRSCREEN", 64).digest()]
    return out


def _tiny(arm, **kw):
    rounds = 12 if arm in ("FA", "FP", "PF", "FA_Q8") else (20 if arm == "S_CAL" else 8)
    s = tiny_spec(arm, rounds=rounds, **ARM_KW[arm], **kw)
    if arm == "S_CAL":
        s["max_rounds"] = rounds
    return s


# ------------------------------------------------------------------------------------------------ digest invariance
def test_default_digests_byte_identical_to_pre_fl10_code():
    d = all_grid_digests()
    assert len(d) == 224
    assert hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest() == PRE_FL10_SHA
    for arm in F.ARMS:                                   # explicit defaults == the default: no spec / cfg key added
        s = F.arm_spec(arm, seed=2026, peak_lr=PEAK_REG, server_opt="fedavg", lr_shape="central",
                       client_opt="adamw_fresh", **({"clip_norm": 1.0} if arm in ("FA_1024", "DP") else {}),
                       **({"z": 0.5642} if arm == "DP" else {}))
        assert s == F.arm_spec(arm, seed=2026, peak_lr=PEAK_REG, **({"clip_norm": 1.0} if arm in ("FA_1024", "DP")
                                                                     else {}), **({"z": 0.5642} if arm == "DP" else {}))
        assert F.spec_tuning(s) == F.TUNING_DEFAULTS


def test_default_trajectories_bit_identical_to_pre_fl10_code(tmp_path):
    got = {arm: run_tiny(tmp_path, _tiny(arm), arm)["summary"]["server_digest"] for arm in F.ARMS}
    assert got == PRE_FL10_TRAJ
    rows = _json(tmp_path / "FA" / "RESULT_ROWS.json")["rows"]
    assert rows and all(not any(k.startswith(("fl_server", "fl_client", "fl_lr_shape")) for k in r) for r in rows)
    assert "server_opt" not in _json(tmp_path / "FA" / "RUN_SUMMARY.json")


# ------------------------------------------------------------------------------------------------ values / spec
def test_tuning_values_and_refusals():
    from fl.client import SGD_M_MOMENTUM
    from fl.server import FEDADAM_DEFAULTS, FEDAVGM_DEFAULTS, V_INIT
    assert F.FEDADAM_HYPER == dict(FEDADAM_DEFAULTS, v_init=V_INIT) and F.FEDAVGM_HYPER == FEDAVGM_DEFAULTS
    assert SGD_M_MOMENTUM == 0.9
    assert F.server_opt_value(None, None) == ("fedavg", None) == F.server_opt_value("fedavg", None)
    assert F.server_opt_value("fedavgm", None) == ("fedavgm", 1.0) == F.server_opt_value("fedavgm", 1.0)
    for lr in (0.01, 0.1, 0.003, 0.001):
        assert F.server_opt_value("FedAdam", str(lr)) == ("fedadam", lr)
    for opt, lr in (("fedadam", None), ("fedadam", 0.05), ("fedadam", 1.0), ("fedavg", 0.1), (None, 0.01),
                    ("fedavgm", 0.1), ("fedyogi", 0.1), ("fedadam", "x")):
        assert_raises(F.LaunchRefused, F.server_opt_value, opt, lr)
    assert T() == F.TUNING_DEFAULTS and T(optimizer="fedadam", server_lr=0.1)["server_tau"] == 1e-3
    for kw in (dict(server_tau=1e-5), dict(optimizer="fedavgm", server_tau=1e-5),
               dict(optimizer="fedadam", server_lr=0.1, server_tau=1e-4), dict(lr_shape="cosine"),
               dict(client_opt="sgd"), dict(client_lr=0.1), dict(client_opt="sgd_m"),
               dict(client_opt="sgd_m", client_lr=0.5), dict(client_opt="sgd_m", client_lr=0.1, lr_mult=2.0),
               dict(lr_mult=3.0)):
        assert_raises(F.LaunchRefused, T, **kw)
    for lr in F.CLIENT_LRS + F.CLIENT_EDGE_LRS:
        assert T(client_opt="sgd_m", client_lr=lr)["client_lr"] == lr
    assert F.tuning_label(T(lr_mult=2.0)) == "fedavg|m=2"
    assert F.tuning_label(T(optimizer="fedadam", server_lr=0.01, server_tau=1e-5, lr_mult=4.0)) == \
        "fedadam|eta=0.01|tau=1e-05|m=4"
    assert F.tuning_label(T(optimizer="fedavgm", lr_shape="fl_const", client_opt="sgd_m", client_lr=10.0)) == \
        "fedavgm|m=1|shape=fl_const|client=sgd_m|clr=10"


@pytest.mark.parametrize("name", sorted(OPTIONS))
def test_spec_cfg_rows_for_each_option(name):
    kw = OPTIONS[name]
    base = F.arm_spec("FA", seed=2026, peak_lr=PEAK_REG, role="CALIBRATION", fused=True)
    s = F.arm_spec("FA", seed=2026, peak_lr=PEAK_REG, role="CALIBRATION", fused=True, **kw)
    t = F.spec_tuning(s)
    assert F.spec_digest(s) != F.spec_digest(base) and L._strip_lr(s) == L._strip_lr(base)
    cfg = F.fl_run_config(s, "x", 64)
    assert cfg.digest() != F.fl_run_config(base, "x", 64).digest()
    if t["optimizer"] == "fedadam":
        assert s["cfg"]["server_opt"] == dict(name="fedadam", lr=t["server_lr"], **dict(F.FEDADAM_HYPER,
                                                                                       tau=t["server_tau"]))
        assert cfg.server_opt.tau == t["server_tau"] and ("server_tau" in s) == (t["server_tau"] != 1e-3)
    if t["optimizer"] == "fedavgm":
        assert s["cfg"]["server_opt"] == {"name": "fedavgm", "lr": 1.0, "beta1": 0.9} and s["server_lr"] == 1.0
        assert cfg.server_opt.beta2 is None
    if t["lr_shape"] == "fl_const":
        assert s["cfg"]["lr_shape"] == cfg.lr_shape == "fl_const"
    if t["client_opt"] == "sgd_m":
        assert s["cfg"]["solver"] == dict(base["cfg"]["solver"], lr=0.1, optimizer="sgd_m", weight_decay=0.0,
                                          fused=None)
        assert s["cfg"]["lr_peak_local"] == 0.1 and cfg.solver.optimizer == "sgd_m"
    labels = F.row_labels(t)
    assert labels and all(k.startswith("fl_") for k in labels)
    bad = json.loads(json.dumps(s))
    bad["cfg"].pop("server_opt", None)
    bad["cfg"].pop("lr_shape", None)
    bad["cfg"]["solver"]["optimizer"] = "adamw"
    assert_raises(F.LaunchRefused, F.spec_tuning, bad)


def test_pf_sgd_keeps_personal_lr_and_run_ids():
    b = F.arm_spec("PF", seed=2026, peak_lr=PEAK_REG)
    s = F.arm_spec("PF", seed=2026, peak_lr=PEAK_REG, client_opt="sgd_m", client_lr=1.0)
    assert s["cfg"]["pf"] == b["cfg"]["pf"] and s["cfg"]["lr_peak_local"] == 1.0
    ids = {
        ("--server-opt", "fedadam", "--server-lr", "0.1", "--lr-mult", "4"): "FL_FA_LRSCREEN_FEDADAM_eta0.1_x4_CALIB_s2026",
        ("--server-opt", "fedadam", "--server-lr", "0.01", "--server-tau", "1e-5", "--lr-mult", "4"):
            "FL_FA_LRSCREEN_FEDADAM_eta0.01_tau1e-05_x4_CALIB_s2026",
        ("--server-opt", "fedavgm", "--lr-mult", "8"): "FL_FA_LRSCREEN_FEDAVGM_x8_CALIB_s2026",
        ("--server-opt", "fedavgm", "--lr-shape", "fl_const", "--lr-mult", "8"):
            "FL_FA_LRSCREEN_FEDAVGM_FLCONST_x8_CALIB_s2026",
        ("--server-opt", "fedavgm", "--lr-shape", "fl_const", "--client-opt", "sgd_m", "--client-lr", "0.1"):
            "FL_FA_LRSCREEN_FEDAVGM_FLCONST_SGDM_lr0.1_x1_CALIB_s2026",
        ("--lr-mult", "4"): "FL_FA_LRSCREEN_x4_CALIB_s2026"}
    for extra, want in ids.items():
        a = _args(Path("."), "--role", "CALIBRATION", *extra, smoke=False)
        F.fl05_preflight(a)
        assert F.default_run_id(a) == want
    a = _args(Path("."), "--role", "CALIBRATION", "--server-opt", "fedavgm", "--lr-shape", "fl_const", "--client-opt",
              "sgd_m", "--client-lr", "0.1", smoke=False)
    spec, extra = F.resolve_spec(a, PEAK_REG, fused=True)
    assert F.spec_tuning(spec) == T(optimizer="fedavgm", lr_shape="fl_const", client_opt="sgd_m", client_lr=0.1)


def test_cli_refusals(tmp_path):
    sel = tmp_path / "LR_SELECTION_EXT.json"
    tun = tmp_path / "FL_TUNING_SELECTION.json"
    for extra in (("--server-opt", "fedadam"), ("--server-lr", "0.1"), ("--server-opt", "fedavg", "--server-lr", "0.1"),
                  ("--server-opt", "fedadam", "--server-lr", "0.5"), ("--server-opt", "fedavgm", "--server-lr", "0.1"),
                  ("--server-tau", "1e-5"), ("--client-lr", "0.1"), ("--client-opt", "sgd_m"),
                  ("--client-opt", "sgd_m", "--client-lr", "0.1", "--lr-mult", "2"),
                  ("--server-opt", "fedadam", "--server-lr", "0.1", "--lr-selection", str(sel)),
                  ("--lr-shape", "fl_const", "--lr-selection", str(sel)),
                  ("--fl-tuning", str(tun), "--lr-mult", "2"), ("--fl-tuning", str(tun), "--lr-selection", str(sel)),
                  ("--fl-tuning", str(tun), "--server-opt", "fedavgm"), ("--fl-tuning", str(tun), "--lr-shape", "fl_const"),
                  ("--fl-tuning", str(tun), "--client-opt", "sgd_m", "--client-lr", "0.1")):
        assert_raises(F.LaunchRefused, F.fl05_preflight, _args(tmp_path, *extra))
    assert_raises(F.LaunchRefused, F.fl05_preflight, _args(tmp_path, "--role", "CALIBRATION", "--fl-tuning", str(tun)))
    for bad in (("--server-opt", "fedyogi"), ("--lr-shape", "cosine"), ("--client-opt", "sgd")):
        assert_raises(SystemExit, F.parse_args, ["--arm", "FA", "--seed", "2026", "--gpu", "0", "--run-dir", "r",
                                                 "--history-policy", "USER_HISTORY_L50", *bad])


# ------------------------------------------------------------------------------------------------ engine
def _fa(**kw):
    return tiny_spec("FA", rounds=12, **kw)


@pytest.mark.parametrize("name", sorted(OPTIONS))
def test_engine_determinism_rows_and_resume(tmp_path, name):
    kw = OPTIONS[name]
    a = run_tiny(tmp_path, _fa(**kw), "a")
    b = run_tiny(tmp_path, _fa(**kw), "b")
    assert a["status"] == "DONE_MAIN"
    sa = a["summary"]
    assert sa["server_digest"] == b["summary"]["server_digest"] != PRE_FL10_TRAJ["FA"]
    t = F.spec_tuning(sa["arm_spec"])
    rows = _json(tmp_path / "a" / "RESULT_ROWS.json")["rows"]
    assert rows and all({k: r[k] for k in F.row_labels(t)} == F.row_labels(t) for r in rows)
    if t["optimizer"] != "fedavg":
        assert sa["server_opt"]["steps"] > 0 and sa["server_opt"] == b["summary"]["server_opt"]
    part = run_tiny(tmp_path, _fa(**kw), "p", interrupt_at=5)
    assert part["status"] == "INTERRUPTED"
    assert_raises(F.LaunchRefused, run_tiny, tmp_path, _fa(), "p", resume=True)              # other spec refused
    res = run_tiny(tmp_path, _fa(**kw), "p", resume=True)
    for k in ("server_digest", "evals", "ledger", "counters", "lr_table") + (("server_opt",) if "server_opt" in sa else ()):
        assert sa[k] == res["summary"][k], k


def test_dp_and_fa1024_with_server_optimizers(tmp_path):
    for arm in ("DP", "FA_1024"):
        for name, kw in (("adam", dict(server_opt="fedadam", server_lr=0.01)), ("avgm", dict(server_opt="fedavgm"))):
            on = run_tiny(tmp_path, _tiny(arm, **kw), f"{arm}_{name}")
            assert on["status"] == "DONE_MAIN" and on["summary"]["server_opt"]["steps"] >= 1
            assert on["summary"]["server_digest"] != PRE_FL10_TRAJ[arm]


# ------------------------------------------------------------------------------------------------ stage rules
def _c(value, incumbent=False, **kw):
    return dict(T(**kw), value=value, incumbent=incumbent)


def test_apply_rule_stage_tie_orders():
    R = L.apply_rule_stage
    inc = _c(0.1000, True, lr_mult=4.0)
    w, tie = R("S2", [inc, _c(0.1004, optimizer="fedavgm", lr_mult=4.0), _c(0.1003, optimizer="fedadam", server_lr=0.01,
                                                                             lr_mult=1.0)])
    assert w == T(lr_mult=4.0) and tie == ["fedavg|m=4"]                  # FIX-DELTA-02 B1: gain 0.0004 < 0.002
    d = L.stage_decision("S2", [inc, _c(0.1004, optimizer="fedavgm", lr_mult=4.0),
                                _c(0.1003, optimizer="fedadam", server_lr=0.01, lr_mult=1.0)])
    assert d["tie_band_set"] == ["fedavg|m=4", "fedavgm|m=4", "fedadam|eta=0.01|m=1"] and d["no_measurable_gain"]
    w, tie = R("S2", [inc, _c(0.2, optimizer="fedavgm", lr_mult=4.0), _c(0.2003, optimizer="fedadam", server_lr=0.01,
                                                                          lr_mult=1.0)])
    assert w["optimizer"] == "fedavgm"                                                         # FedAvgM before FedAdam
    w, _ = R("S2", [inc, _c(0.3, optimizer="fedadam", server_lr=0.1, lr_mult=1.0),
                    _c(0.3, optimizer="fedadam", server_lr=0.01, lr_mult=4.0)])
    assert (w["server_lr"], w["lr_mult"]) == (0.1, 1.0)                                         # |log2 m| before eta
    w, _ = R("S2", [inc, _c(0.3, optimizer="fedadam", server_lr=0.01, lr_mult=4.0),
                    _c(0.3, optimizer="fedadam", server_lr=0.01, server_tau=1e-5, lr_mult=4.0)])
    assert w["server_tau"] == 1e-3                                                              # larger tau first
    s3inc = _c(0.1, True, optimizer="fedavgm", lr_mult=4.0)
    s3inc_label = F.tuning_label(s3inc)
    w, tie = R("S3", [s3inc, _c(0.1002, optimizer="fedavgm", lr_mult=4.0, lr_shape="fl_const"),
                      _c(0.1004, optimizer="fedavgm", lr_shape="fl_const", client_opt="sgd_m", client_lr=0.1)])
    assert w == T(optimizer="fedavgm", lr_mult=4.0) and len(tie) == 1                          # incumbent (B1)
    assert L.stage_decision("S3", [s3inc, _c(0.1002, optimizer="fedavgm", lr_mult=4.0, lr_shape="fl_const"),
                                   _c(0.1004, optimizer="fedavgm", lr_shape="fl_const", client_opt="sgd_m",
                                      client_lr=0.1)])["tie_band_set"][0] == s3inc_label          # incumbent first
    w, _ = R("S3", [s3inc, _c(0.2, optimizer="fedavgm", lr_shape="fl_const", client_opt="sgd_m", client_lr=1.0),
                    _c(0.2, optimizer="fedavgm", lr_shape="fl_const", client_opt="sgd_m", client_lr=0.1),
                    _c(0.2001, optimizer="fedavgm", lr_mult=4.0, lr_shape="fl_const")])
    assert w["client_opt"] == "adamw_fresh" and w["lr_shape"] == "fl_const"                  # AdamW before SGD
    w, _ = R("S3", [s3inc, _c(0.2, optimizer="fedavgm", lr_shape="fl_const", client_opt="sgd_m", client_lr=1.0),
                    _c(0.2, optimizer="fedavgm", lr_shape="fl_const", client_opt="sgd_m", client_lr=0.1)])
    assert w["client_lr"] == 0.1                                                                # smaller SGD lr
    for bad in ([], [dict(inc, value=float("nan"))], [inc, dict(inc)], [inc, dict(_c(0.1, True, lr_mult=2.0))],
                [dict(inc, server_lr=0.05)]):
        assert_raises(L.SelectionRefused, R, "S2", bad)
    assert_raises(L.SelectionRefused, R, "S5", [inc])


# ------------------------------------------------------------------------------------------------ staged selection
def _screen(root, name, **kw):
    out = run_tiny(root, tiny_spec("FA", role="CALIBRATION", **kw), name, world=fl05_world(eval_id=CALIB_EVAL))
    assert out["status"] == "DONE_MAIN", name
    return root / name


def _label_dirs(dirs):
    return {F.tuning_label(L.read_run(d)["tuning"]): d for d in dirs}


N_TINY = 1152                                     # fa_rounds(1152, 2) = 60 = the tiny screens' round count


def screen_point(root, name, t):
    """A tiny CALIBRATION screen run at tuning point t (a non-default pass count gets the exposure budget of N_TINY, as
    test_fl12_launch.s4_spec); an existing finished run dir of that name is reused."""
    d = root / name
    if (d / "RUN_SUMMARY.json").is_file():
        return d
    s = tiny_spec("FA", role="CALIBRATION", **F.tuning_kwargs(t))
    p = s["cfg"]["solver"]["passes"]
    if p != F.KNOBS["passes"]:
        eb = F.exposure_budget(N_TINY, p)
        s["exposure_budget"] = eb
        s["cfg"]["endpoint_rounds"] = s["max_rounds"] = eb["rounds"]
    out = run_tiny(root, s, name, world=fl05_world(eval_id=CALIB_EVAL))
    assert out["status"] == "DONE_MAIN", name
    return d


def complete_edges(root, stage, runs, first, prefix="edge"):
    """FIX-DELTA-02 M3: screen the registered 06J edge runs that select-joint names (EdgeRunsMissing) until the run set
    of the stage is complete; returns (runs incl. the edge runs, the edge run dirs)."""
    runs, edges = list(runs), []
    for _ in range(4):
        try:
            L.selection_stage(stage, runs, first)
            return runs, edges
        except L.EdgeRunsMissing as e:
            assert e.needed and str(e).count("launch_fl --role CALIBRATION") == len(e.needed)
            for t in e.needed:
                d = screen_point(root, f"{prefix}_" + "".join(ch if ch.isalnum() else "_" for ch in t["label"]),
                                 {k: t[k] for k in F.TUNING_KEYS})
                runs.append(d)
                edges.append(d)
    raise AssertionError(f"{stage}: the edge runs did not complete")


@pytest.fixture(scope="module")
def ladder(tmp_path_factory):
    root = tmp_path_factory.mktemp("fl10_ladder")
    avg = {m: _screen(root, f"x{F.LR_MULT_TAG[m]}", lr_mult=m) for m in (0.5, 1.0, 2.0, 4.0)}
    ext = root / "contracts_fl" / F.LR_SELECTION_EXT_NAME
    ext.parent.mkdir()
    L.write_selection_ext(list(avg.values()), ext)
    m_star = _json(ext)["selected_lr_mult"]
    s2 = [_screen(root, f"adam_{lr}_{m}", lr_mult=m, server_opt="fedadam", server_lr=lr) for lr, m in F.fedadam_grid(m_star)]
    s2.append(_screen(root, "avgm", lr_mult=m_star, server_opt="fedavgm"))
    s2 += [screen_point(root, f"avgm_a1_{F.num_tag(t['lr_mult'])}", t)                  # FL-14: 06L A1 FedAvgM m {1, 2}
           for t in L.stage_required("S2", T(lr_mult=m_star), m_star)[5:]]
    s2_base = list(s2)
    runs2, s2_edges = complete_edges(root, "S2", [avg[m_star]] + s2, ext, "s2")    # FIX-DELTA-02 M3
    s2 = runs2[1:]
    s2_file = root / "S2.json"
    L.write_selection_stage("S2", [avg[m_star]] + s2, s2_file, ext)
    w2 = _json(s2_file)["winner"]
    srv = dict(server_opt=w2["optimizer"], server_lr=w2["server_lr"] if w2["optimizer"] == "fedadam" else None,
               server_tau=w2["server_tau"] if w2["optimizer"] == "fedadam" and w2["server_tau"] != 1e-3 else None)
    w2_dir = _label_dirs([avg[m_star]] + s2)[w2["label"]]
    s3 = [_screen(root, "s3_adamw", lr_mult=w2["lr_mult"], lr_shape="fl_const", **srv)]
    s3 += [_screen(root, f"s3_sgd_{lr}", lr_shape="fl_const", client_opt="sgd_m", client_lr=lr, **srv)
           for lr in F.CLIENT_LRS]
    s3 += [screen_point(root, f"s3_a2_{F.num_tag(t['client_lr'])}", t)                  # FL-14: 06L A2 (deduped)
           for t in L.stage_required("S3", T(**{k: w2[k] for k in F.TUNING_KEYS}), m_star)[4:]]
    s3_base = list(s3)
    runs3, s3_edges = complete_edges(root, "S3", [w2_dir] + s3, s2_file, "s3")       # FIX-DELTA-02 M3
    s3 = runs3[1:]
    return {"root": root, "avg": avg, "ext": ext, "m_star": m_star, "s2": s2, "s2_file": s2_file, "w2_dir": w2_dir,
            "s3": s3, "srv": srv, "w2": w2, "s2_base": s2_base, "s2_edges": s2_edges, "s3_base": s3_base,
            "s3_edges": s3_edges}


def test_stage_s2_file(ladder, tmp_path):
    lad = ladder
    obj = _json(lad["s2_file"])
    assert obj["format"] == F.FL_TUNING_FORMAT and obj["stage"] == "S2" and obj["self_sha256"] == F._canon(obj)
    assert len(obj["configs"]) == 1 + len(obj["required"]) + len(lad["s2_edges"])        # FL-14: + 06L A1 cells
    assert obj["incumbent"] == f"fedavg|m={F.num_tag(lad['m_star'])}"
    assert sum(c["incumbent"] for c in obj["configs"]) == 1 and obj["m_star"] == lad["m_star"]
    win, tie = L.apply_rule_stage("S2", [c for c in obj["configs"] if not c["excluded"]])
    assert {k: obj["winner"][k] for k in F.TUNING_KEYS} == win and obj["tie_set"] == tie
    assert obj["min_gain"] == 0.002 and obj["no_measurable_gain"] == (obj["gain"] < 0.002)             # 06J §4(ii)
    assert [lab for e in obj["edges"] for lab in e["labels"]] == [F.tuning_label(L.read_run(d)["tuning"])
                                                                   for d in lad["s2_edges"]]
    assert obj["stage_history"][0]["sha256"] == hashlib.sha256(lad["ext"].read_bytes()).hexdigest()
    assert F.load_fl_tuning(lad["s2_file"])["stage"] == "S2"
    # CLI: prints only the winner + sha; O_EXCL
    out = tmp_path / "S2_cli.json"
    said = []
    argv = ["select-joint", "--stage", "S2", "--runs", str(lad["avg"][lad["m_star"]]), *map(str, lad["s2"]),
            "--first-selection", str(lad["ext"]), "--out", str(out)]
    assert L.main(argv, say=said.append) == 0
    rec = json.loads(said[0])
    assert set(rec) == {"path", "sha256", "self_sha256", "stage", "winner"} and set(rec["winner"]) == set(F.TUNING_KEYS)
    assert rec["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert L.main(argv, say=said.append) == 2


def test_stage_s2_refusals_and_edges(ladder, tmp_path):
    lad = ladder
    inc = lad["avg"][lad["m_star"]]
    S = L.selection_stage
    base, edges = lad["s2_base"], lad["s2_edges"]
    by = _label_dirs(base)
    avgm = [d for lab, d in by.items() if lab.startswith("fedavgm")]
    adam = [d for lab, d in by.items() if lab.startswith("fedadam")]
    assert_raises(L.SelectionRefused, S, "S2", base + edges, lad["ext"])                       # no incumbent
    assert_raises(L.SelectionRefused, S, "S2", [inc] + adam + edges, lad["ext"])               # FedAvgM missing
    assert_raises(L.SelectionRefused, S, "S2", [inc] + adam[1:] + avgm + edges, lad["ext"])    # one FedAdam missing
    other = next(d for m, d in lad["avg"].items() if m != lad["m_star"])
    assert_raises(L.SelectionRefused, S, "S2", [inc, other] + lad["s2"], lad["ext"])           # non-incumbent FedAvg
    assert_raises(L.SelectionRefused, S, "S2", [inc] + lad["s2"] + [lad["s2"][0]], lad["ext"])  # twice
    assert_raises(L.SelectionRefused, S, "S2", [inc] + lad["s2"], lad["s2_file"])             # not the EXT file
    assert_raises(L.SelectionRefused, L.selection_ext, [lad["avg"][1.0], lad["avg"][2.0], adam[0]])
    if edges:                                                  # FIX-DELTA-02 M3: a condition-true edge is REQUIRED
        assert_raises(L.EdgeRunsMissing, S, "S2", [inc] + base, lad["ext"])
        if len(edges) >= 2:
            assert_raises(L.SelectionRefused, S, "S2", [inc] + base + edges[:1], lad["ext"])    # both or none
    else:
        S("S2", [inc] + base, lad["ext"])
    # an edge run whose condition does not hold is refused
    obj = _json(lad["s2_file"])
    base_adam = [c for c in obj["configs"] if c["optimizer"] == "fedadam" and c["server_tau"] == 1e-3
                 and c["server_lr"] in F.FEDADAM_LRS and not c["excluded"]]
    b = max(base_adam, key=lambda c: c["value"])
    if b["server_lr"] != 0.01:                                 # the eta edge is not allowed
        e = [screen_point(tmp_path, f"eta_{lr}", T(optimizer="fedadam", server_lr=lr, lr_mult=b["lr_mult"]))
             for lr in F.FEDADAM_EDGE_LRS]
        assert_raises(L.SelectionRefused, S, "S2", [inc] + lad["s2"] + e, lad["ext"])
    if obj["winner"]["optimizer"] != "fedadam" and not any(
            c["server_tau"] == 1e-5 for c in obj["configs"] if c["optimizer"] == "fedadam"):
        tau = screen_point(tmp_path, "tau", T(optimizer="fedadam", server_lr=b["server_lr"], server_tau=1e-5,
                                              lr_mult=b["lr_mult"]))
        assert_raises(L.SelectionRefused, S, "S2", [inc] + lad["s2"] + [tau], lad["ext"])   # no FedAdam winner
    t = tmp_path / "nonfinite"                                                                # excluded + recorded
    shutil.copytree(avgm[0], t)
    rr = _json(t / "RESULT_ROWS.json")
    for r in rr["rows"]:
        if r["selection_rule"] == "PRACTICAL_BEST":
            r["value"] = None
    (t / "RESULT_ROWS.json").write_text(json.dumps(rr), encoding="utf-8")
    runs, _e = complete_edges(tmp_path, "S2", [inc] + adam + avgm[1:] + [t], lad["ext"], "nf")   # FL-14: + A1
    o = S("S2", runs, lad["ext"])
    assert [c["label"] for c in o["excluded"]] == [f"fedavgm|m={F.num_tag(lad['m_star'])}"]
    assert len(o["values"]) == len(o["required"]) + len(_e)


def test_stage_s3_file_and_refusals(ladder, tmp_path):
    lad = ladder
    S = L.selection_stage
    runs = [lad["w2_dir"]] + lad["s3"]
    out = tmp_path / "S3.json"
    rec = L.write_selection_stage("S3", runs, out, lad["s2_file"])
    obj = _json(out)
    assert obj["stage"] == "S3" and len(obj["configs"]) == 1 + len(obj["required"]) + len(lad["s3_edges"])
    assert obj["incumbent"] == lad["w2"]["label"]
    assert [h["stage"] for h in obj["stage_history"]] == ["S1b", "S2"]
    assert obj["stage_history"][1]["sha256"] == hashlib.sha256(lad["s2_file"].read_bytes()).hexdigest()
    loaded = F.load_fl_tuning(out)
    assert {k: loaded["winner"][k] for k in F.TUNING_KEYS} == rec["winner"]
    assert_raises(L.SelectionRefused, S, "S3", lad["s3"], lad["s2_file"])                     # no incumbent
    sgd1 = _label_dirs(lad["s3_base"])
    no_sgd1 = [d for lab, d in sgd1.items() if not lab.endswith("clr=1")]
    assert_raises(L.SelectionRefused, S, "S3", [lad["w2_dir"]] + no_sgd1 + lad["s3_edges"], lad["s2_file"])  # SGD 1.0
    assert_raises(L.SelectionRefused, S, "S3", runs, lad["ext"])                              # not an S2 file
    assert_raises(L.SelectionRefused, S, "S3", runs + [lad["avg"][0.5]], lad["s2_file"])      # unregistered here
    if lad["s3_edges"]:                                                                       # M3: required
        assert_raises(L.EdgeRunsMissing, S, "S3", [lad["w2_dir"]] + lad["s3_base"], lad["s2_file"])
    sgd = [c for c in obj["configs"] if c["client_opt"] == "sgd_m" and c["client_lr"] in F.CLIENT_LRS
           and not c["excluded"] and L._server_point(c) == L._server_point(lad["w2"])]   # FL-14: 06J cells only
    b = max(sgd, key=lambda c: c["value"]) if sgd else None
    if b is None or b["client_lr"] != 1.0:
        e = _screen(tmp_path, "sgd_3", lr_shape="fl_const", client_opt="sgd_m", client_lr=3.0, **lad["srv"])
        assert_raises(L.SelectionRefused, S, "S3", runs + [e], lad["s2_file"])                # edge not allowed
    # the tuning file drives a MAIN spec (--fl-tuning)
    argv = ["--arm", "FP", "--seed", "2027", "--gpu", "0", "--run-dir", str(tmp_path / "r"),
            "--history-policy", "USER_HISTORY_L50", "--fl-tuning", str(out)]
    args = F.parse_args(argv)
    assert_raises(F.LaunchRefused, F.require_lr_selection, args)          # FIX-DELTA-01 F2: not the registered file
    t, prov = F.resolve_tuning(args)
    assert t == rec["winner"] and prov["source"] == "fl_tuning" and prov["stage"] == "S3"
    spec, extra = F.resolve_spec(F.parse_args(argv + ["--smoke"]), PEAK_REG, fused=True)     # the wiring (smoke)
    assert F.spec_tuning(spec) == t and extra["fl_tuning"]["self_sha256"] == obj["self_sha256"]


def test_tuning_file_tamper_refused(ladder, tmp_path):
    obj = _json(ladder["s2_file"])

    def write(o, name):
        p = tmp_path / name
        p.write_text(json.dumps(o), encoding="utf-8")
        return p
    F.load_fl_tuning(write(obj, "ok.json"))
    other = [c for c in obj["configs"] if c["label"] != obj["winner"]["label"]][0]
    for field, val in (("winner", {k: other[k] for k in F.TUNING_KEYS}), ("tie_set", obj["tie_set"] + ["fedavg|m=32"]),
                       ("configs", [c for c in obj["configs"] if c["optimizer"] != "fedadam"]), ("m_star", 32.0),
                       ("stage", "S4"), ("configs", [dict(c, label="fedavg|m=32") if c is obj["configs"][0] else c
                                                     for c in obj["configs"]])):
        bad = dict(obj, **{field: val})
        bad["self_sha256"] = F._canon(bad)
        assert_raises(F.LaunchRefused, F.load_fl_tuning, write(bad, f"t_{field}.json"))
    assert_raises(F.LaunchRefused, F.load_fl_tuning, write(dict(obj, m_star=64.0), "stale.json"))


def test_main_refuses_lr_selection_once_tuning_exists(ladder, tmp_path, monkeypatch):
    monkeypatch.setattr(F, "FL_TUNING_PATH", tmp_path / "registered" / F.FL_TUNING_NAME)
    monkeypatch.setattr(F, "LR_SELECTION_EXT_PATH", tmp_path / "registered" / F.LR_SELECTION_EXT_NAME)
    folder = tmp_path / "c"
    folder.mkdir()
    ext = folder / F.LR_SELECTION_EXT_NAME
    shutil.copy(ladder["ext"], ext)

    def a(*extra, smoke=False):
        return _args(tmp_path, *extra, smoke=smoke, arm="FP")
    # FIX-DELTA-01 F2: files outside the registered contracts/fl path never satisfy a non-smoke MAIN run (the positive
    # registered-path cases are in tests_int/test_fix_delta01_fl_binding.py)
    assert_raises(F.LaunchRefused, F.require_lr_selection, a("--lr-selection", str(ext)))
    tun = folder / F.FL_TUNING_NAME
    shutil.copy(ladder["s2_file"], tun)
    assert_raises(F.LaunchRefused, F.require_lr_selection, a("--lr-selection", str(ext)))     # sibling exists
    assert_raises(F.LaunchRefused, F.require_lr_selection, a("--fl-tuning", str(tun)))
    F.require_lr_selection(a("--lr-selection", str(ext), smoke=True))                          # smoke: not refused
    assert_raises(F.LaunchRefused, F.require_lr_selection, a())
    other = tmp_path / "o"
    other.mkdir()
    shutil.copy(ladder["ext"], other / F.LR_SELECTION_EXT_NAME)
    assert_raises(F.LaunchRefused, F.require_lr_selection, a("--lr-selection", str(other / F.LR_SELECTION_EXT_NAME)))
    F.FL_TUNING_PATH.parent.mkdir()
    shutil.copy(tun, F.FL_TUNING_PATH)                                                         # the registered path
    assert_raises(F.LaunchRefused, F.require_lr_selection, a("--lr-selection", str(other / F.LR_SELECTION_EXT_NAME)))
    assert_raises(F.LaunchRefused, F.run, a("--lr-selection", str(ext)), {}, {})              # before any import


# ------------------------------------------------------------------------------------------------ negative controls
def _accepted(fn, *a, exc=F.LaunchRefused, **kw):
    try:
        fn(*a, **kw)
        return True
    except exc:
        return False


@nc("--server-opt fedadam without --server-lr is accepted")
def test_nc_fedadam_without_server_lr_accepted(tmp_path):
    assert _accepted(F.fl05_preflight, _args(tmp_path, "--server-opt", "fedadam")), "refused (as it must be)"


@nc("--server-lr with --server-opt fedavg is accepted")
def test_nc_server_lr_with_fedavg_accepted(tmp_path):
    assert _accepted(F.fl05_preflight, _args(tmp_path, "--server-opt", "fedavg", "--server-lr", "0.1")), \
        "refused (as it must be)"


@nc("a non-smoke MAIN run with the EXT LR selection is accepted while FL_TUNING_SELECTION.json exists")
def test_nc_main_without_tuning_file_accepted(ladder, tmp_path, monkeypatch):
    folder = tmp_path / "c"                                        # FIX-DELTA-01: both files at the registered paths
    folder.mkdir()
    monkeypatch.setattr(F, "LR_SELECTION_EXT_PATH", folder / F.LR_SELECTION_EXT_NAME)
    monkeypatch.setattr(F, "FL_TUNING_PATH", folder / F.FL_TUNING_NAME)
    ext = folder / F.LR_SELECTION_EXT_NAME
    shutil.copy(ladder["ext"], ext)
    shutil.copy(ladder["s2_file"], folder / F.FL_TUNING_NAME)
    assert _accepted(F.require_lr_selection, _args(tmp_path, "--lr-selection", str(ext), smoke=False, arm="FA")), \
        "refused (as it must be)"


@nc("the S2 selection accepts a run set missing the registered FedAvgM run")
def test_nc_joint_selection_missing_run_accepted(ladder):
    runs = [ladder["avg"][ladder["m_star"]]] + [d for d in ladder["s2"] if d.name != "avgm"]
    assert _accepted(L.selection_stage, "S2", runs, ladder["ext"], exc=L.SelectionRefused), "refused (as it must be)"


@nc("an unregistered client SGD LR (0.5) is accepted")
def test_nc_unregistered_client_lr_accepted():
    assert _accepted(T, client_opt="sgd_m", client_lr=0.5), "refused (as it must be)"


@nc("an unregistered LR shape (cosine) is accepted")
def test_nc_unregistered_lr_shape_accepted():
    assert _accepted(T, lr_shape="cosine"), "refused (as it must be)"
