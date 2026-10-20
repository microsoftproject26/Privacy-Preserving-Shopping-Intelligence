"""TRAIN-02 item 6 — the P2 driver (v4train.p2_driver): the pure selection functions per PREG1_AMENDMENT_02R1 §3 with
fixtures, the launcher command lines of a stage, the stage files, and an end-to-end stage 1 -> stage 2 with a stand-in
runner (no training: the runner writes the RUN_SUMMARY a finished TUNING run would write).

  * exclusions: missing / failed / non-finite, and < 0.5 x the family best (strict), per run;
  * stage 1 -> 2: the best valid trial of each variant level; an exact tie -> the lower LR; a level with no valid run ->
    FABLE_TRIGGER;
  * stage 2: the 2-seed mean; |delta| <= 0.0005 -> fewer parameters, then the lower LR; > 0.0005 -> the higher mean; a
    config with an excluded seed is invalid (FABLE_TRIGGER for its level);
  * CKPT_AVG carried iff the 2-seed gain is > 0.0005 (exactly 0.0005 is not carried; a missing value is not carried);
  * the command lines use launch_central.py --role TUNING --hp-id --phase CALIBRATION --seed 2026 | 2027; a dry run
    writes nothing; a stage file is never rewritten; stage 2 needs a verified stage-1 file; the child environment drops
    the CUDA_VISIBLE_DEVICES default that importing the snapshot sets.
Negative control: a tie rule that ignores the 0.0005 band.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import v4train.p2_driver as D
from trainkit import assert_raises, nc, raises_with
from v4train import hp_registry as H

REG = H.load_hp_registry()
GRU = REG["families"]["GRU"]
SAS = REG["families"]["SASREC"]
LEVELS = H.VARIANT_LEVELS


def test_register_02b_script_refuses_without_pythonpath(tmp_path):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    p = subprocess.run([sys.executable, "-B", str(Path(D.__file__)), "register-02b", "--p2-dir",
                        str(tmp_path / "missing"), "--out", str(tmp_path / "unused.json")],
                       cwd=tmp_path, env=env, capture_output=True, text=True)
    assert p.returncode == 2
    assert "REGISTER-02B REFUSED:" in p.stderr
    assert "ModuleNotFoundError" not in p.stderr and "Traceback" not in p.stderr
NP = {"GRU_D128": 20_962_210, "GRU_D256": 41_500_000, "SASREC_D256_B2": 83_092_004, "SASREC_D256_B3": 83_900_000,
      "SASREC_D384_B2": 126_000_000}


def summ(t, value, ck=None, status="DONE_TUNING", seed=2026):
    return {"status": status, "endpoint": {"value": value}, "ckpt_avg": {"value": value if ck is None else ck},
            "shared_numel": NP[t["model_variant"]], "values_at_marks": {"6.0": value}, "peak_lr": t["peak_lr"],
            "model_variant": t["model_variant"], "seed": seed}


def vals(trials, values, **kw):
    return {t["hp_id"]: D.run_values(summ(t, v, **kw)) for t, v in zip(trials, values)}


# ------------------------------------------------------------------------------------------------ pure functions
def test_run_values_and_exclusions():
    t = GRU[0]
    ok = D.run_values(summ(t, 0.3))
    assert ok["finite"] and ok["endpoint"] == 0.3 and ok["shared_numel"] == NP["GRU_D128"]
    assert not D.run_values(None)["finite"] and D.run_values(None)["status"] == "NO_SUMMARY"
    assert not D.run_values(summ(t, 0.3, status="FAILED"))["finite"]
    assert not D.run_values(summ(t, float("nan")))["finite"]
    v = {"a": {"finite": True, "endpoint": 0.30}, "b": {"finite": True, "endpoint": 0.15},
         "c": {"finite": True, "endpoint": 0.1499999}, "d": {"finite": False, "endpoint": None, "status": "FAILED"}}
    best = D.family_best(list(v.values()))
    ex = D.exclusions(v, best)
    assert best == 0.30 and ex["a"] is None and ex["b"] is None, "exactly 0.5 x best is kept (< excludes)"
    assert ex["c"].startswith("EXCLUDED_BELOW_HALF") and ex["d"].startswith("EXCLUDED_NON_FINITE")


def test_stage1_plan_best_per_level_and_triggers():
    values = [0.30, 0.31, 0.29, 0.28, 0.305, 0.20, 0.32, 0.33, 0.10, float("nan"), 0.25, 0.26]
    p = D.stage1_plan(GRU, vals(GRU, values), LEVELS["GRU"])
    assert p["status"] == "OK" and p["stage2_plan"] == {"GRU_D128": GRU[1]["hp_id"], "GRU_D256": GRU[7]["hp_id"]}
    assert p["excluded"][GRU[8]["hp_id"]].startswith("EXCLUDED_BELOW_HALF") and \
        p["excluded"][GRU[9]["hp_id"]].startswith("EXCLUDED_NON_FINITE")
    tie = [0.30] * 6 + [0.2] * 6                                                  # exact tie in GRU_D128 -> lower LR
    p = D.stage1_plan(GRU, vals(GRU, tie), LEVELS["GRU"])
    d128 = [t for t in GRU if t["model_variant"] == "GRU_D128"]
    assert p["stage2_plan"]["GRU_D128"] == min(d128, key=lambda t: t["peak_lr"])["hp_id"]
    dead = [0.3] * 6 + [float("nan")] * 6
    p = D.stage1_plan(GRU, vals(GRU, dead), LEVELS["GRU"])
    assert p["status"] == "FABLE_TRIGGER" and p["stage2_plan"]["GRU_D256"] is None and "GRU_D256" in p["triggers"][0]
    sv = [0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30, 0.31]
    p = D.stage1_plan(SAS, vals(SAS, sv), LEVELS["SASREC"])
    assert p["stage2_plan"] == {"SASREC_D256_B2": SAS[3]["hp_id"], "SASREC_D256_B3": SAS[7]["hp_id"],
                                "SASREC_D384_B2": SAS[11]["hp_id"]}


def _cands():
    return [{"hp_id": "A", "mean": 0.3000, "n_params": 100, "peak_lr": 1e-3},
            {"hp_id": "B", "mean": 0.3005, "n_params": 200, "peak_lr": 5e-4},
            {"hp_id": "C", "mean": 0.2990, "n_params": 50, "peak_lr": 3e-4}]


def _check_tie_rule():
    ch = D.choose_with_tie(_cands())
    assert ch["tuned"] == "A" and ch["tie_set"] == ["A", "B"] and ch["tie_rule_applied"], \
        "|delta| = 0.0005 is a tie -> fewer parameters"
    c = _cands()
    c[1]["mean"] = 0.30051
    assert D.choose_with_tie(c)["tuned"] == "B", "|delta| > 0.0005 -> the higher mean"
    same = [{"hp_id": "X", "mean": 0.3, "n_params": 100, "peak_lr": 2e-3},
            {"hp_id": "Y", "mean": 0.3002, "n_params": 100, "peak_lr": 1e-3}]
    assert D.choose_with_tie(same)["tuned"] == "Y", "same size -> the lower LR"


def test_choose_with_tie():
    _check_tie_rule()
    assert_raises(D.P2Error, D.choose_with_tie, [])


@nc("a tie rule that ignores the 0.0005 band (always the higher mean)")
def test_nc_tie_rule_without_band(monkeypatch):
    monkeypatch.setattr(D, "TIE_ABS", 0.0)                                                      # MUTANT
    monkeypatch.setattr(D, "EPS", 0.0)
    _check_tie_rule()


def test_ckpt_avg_carry():
    assert D.ckpt_avg_carry([0.30, 0.31], [0.3006, 0.3106])["carry"] is True
    assert D.ckpt_avg_carry([0.30, 0.31], [0.3005, 0.3105])["carry"] is False, "exactly +0.0005 is not carried"
    assert D.ckpt_avg_carry([0.30, 0.31], [0.30, None])["carry"] is False
    assert D.ckpt_avg_carry([0.30, 0.31], [0.29, 0.33])["carry"] is True             # the 2-seed mean decides


def test_stage2_select():
    A, B, C = "P2_GRU_D128_S0_s3", "P2_GRU_D256_S0_s3", "P2_GRU_D128_S0_s0"
    plan = {"GRU_D128": A, "GRU_D256": B}
    t = {x["hp_id"]: x for x in GRU}
    s1 = {h: D.run_values(summ(t[h], v)) for h, v in ((A, 0.3000), (B, 0.3004),
                                                      (C, 0.29))}
    s2 = {A: D.run_values(summ(t[A], 0.3002, ck=0.3012, seed=2027)),
          B: D.run_values(summ(t[B], 0.3006, seed=2027))}
    s1[A] = D.run_values(summ(t[A], 0.3000, ck=0.3002))
    out = D.stage2_select(GRU, plan, s1, s2, LEVELS["GRU"])
    assert out["status"] == "OK" and out["choice"]["tie_set"] == sorted([A, B])
    assert out["tuned"]["hp_id"] == A and out["tuned"]["model_variant"] == "GRU_D128", \
        "0.3005 - 0.3001 <= 0.0005 -> the smaller model"
    assert out["configs"]["GRU_D128"]["mean"] == pytest.approx(0.3001)
    assert out["ckpt_avg_carried_to_p4"] is True and out["ckpt_avg_rule"]["gain"] == pytest.approx(0.0006, abs=1e-9)
    s2b = dict(s2, **{B: D.run_values(summ(t[B], 0.1, seed=2027))})       # < 0.5 x best: excluded
    out = D.stage2_select(GRU, plan, s1, s2b, LEVELS["GRU"])
    assert out["status"] == "FABLE_TRIGGER" and not out["configs"]["GRU_D256"]["valid"]
    assert out["tuned"]["hp_id"] == A
    out = D.stage2_select(GRU, plan, s1, {}, LEVELS["GRU"])
    assert out["status"] == "FABLE_TRIGGER" and "tuned" not in out


# ------------------------------------------------------------------------------------------------ driver plumbing
def test_launch_cmd_and_child_env(monkeypatch):
    cmd = D.launch_cmd("SASREC", "P2_SASREC_D256_B2_S1_s1", 2027, 1, Path("/r/x"), history_policy="USER_HISTORY_L50",
                       python="python")
    assert cmd[:3] == ["python", "-B", str(D.LAUNCHER)] and Path(cmd[2]).name == "launch_central.py"
    joined = " ".join(cmd)
    for part in ("--family SASREC", "--seed 2027", "--phase CALIBRATION", "--role TUNING", "--hp-id P2_SASREC_D256_B2_S1_s1",
                 "--gpu 1", "--history-policy USER_HISTORY_L50"):
        assert part in joined
    assert "--peak-lr" not in joined, "the LR comes from the HP trial (the launcher reads it)"
    assert "--resume" in D.launch_cmd("GRU", "P2_GRU_D128_S0_s0", 2026, 0, Path("x"), history_policy="U", resume=True)
    monkeypatch.setattr(D, "_CVD0", None)
    monkeypatch.setitem(D._ENV0, "CUDA_VISIBLE_DEVICES", "0")
    assert "CUDA_VISIBLE_DEVICES" not in D.child_env() and D.child_env()["PYTHONDONTWRITEBYTECODE"] == "1"
    monkeypatch.setattr(D, "_CVD0", "1")
    assert D.child_env()["CUDA_VISIBLE_DEVICES"] == "1"


def _args(tmp_path, fam, stage, dry=False):
    return argparse.Namespace(family=fam, stage=stage, gpu=0, out_dir=tmp_path, history_policy="USER_HISTORY_L50",
                              hp_registry=None, theta0_variants=None, dry_run=dry)


def test_dry_run_prints_the_rung_and_writes_nothing(tmp_path, capsys):
    assert D.run_stage(_args(tmp_path, "GRU", 1, dry=True), runner=None) == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if "launch_central.py" in ln]
    assert len(lines) == 12 and all("--seed 2026" in ln and "--role TUNING" in ln for ln in lines)
    assert [ln.split("--hp-id ")[1].split()[0] for ln in lines] == [t["hp_id"] for t in GRU]
    assert list(tmp_path.iterdir()) == []


def _fake_runner(values):
    """Stands in for launch_central.py: writes the DONE_TUNING RUN_SUMMARY of the trial named in the command."""
    def run(cmd, **kw):
        hp = cmd[cmd.index("--hp-id") + 1]
        seed = int(cmd[cmd.index("--seed") + 1])
        rd = Path(cmd[cmd.index("--run-dir") + 1])
        rd.mkdir(parents=True, exist_ok=True)
        t = [x for f in REG["families"].values() for x in f if x["hp_id"] == hp][0]
        v, ck = values[(hp, seed)]
        (rd / "RUN_SUMMARY.json").write_text(json.dumps(summ(t, v, ck=ck, seed=seed)), encoding="utf-8")
        return argparse.Namespace(returncode=0)
    return run


def test_stage1_then_stage2_end_to_end(tmp_path):
    vals1 = {(t["hp_id"], 2026): (0.20 + 0.01 * i, None) for i, t in enumerate(SAS)}
    assert D.run_stage(_args(tmp_path, "SASREC", 1), runner=_fake_runner(vals1), say=lambda *_: None) == 0
    s1 = D.load_stage(tmp_path / "P2_SASREC_STAGE1.json")
    assert s1["selection"]["stage2_plan"] == {"SASREC_D256_B2": SAS[3]["hp_id"], "SASREC_D256_B3": SAS[7]["hp_id"],
                                              "SASREC_D384_B2": SAS[11]["hp_id"]}
    assert s1["hp_registry"]["self_sha256"] == REG["self_sha256"] and len(s1["runs"]) == 12
    raises_with(D.P2Error, "never re-run", D.run_stage, _args(tmp_path, "SASREC", 1), runner=_fake_runner(vals1))
    vals2 = {(SAS[3]["hp_id"], 2027): (0.23, 0.2310), (SAS[7]["hp_id"], 2027): (0.27, None),
             (SAS[11]["hp_id"], 2027): (0.3095, None)}
    assert D.run_stage(_args(tmp_path, "SASREC", 2), runner=_fake_runner(vals2), say=lambda *_: None) == 0
    s2 = D.load_stage(tmp_path / "P2_SASREC_STAGE2.json")
    sel = s2["selection"]
    assert sel["status"] == "OK" and sel["tuned"]["hp_id"] == SAS[11]["hp_id"]
    assert sel["configs"]["SASREC_D384_B2"]["mean"] == pytest.approx((0.31 + 0.3095) / 2)
    assert s2["stage1_file"]["self_sha256"] == s1["self_sha256"] and s2["seed"] == 2027


def test_stage2_needs_a_verified_stage1(tmp_path):
    raises_with(D.P2Error, "does not exist", D.run_stage, _args(tmp_path, "GRU", 2), runner=None)
    vals1 = {(t["hp_id"], 2026): (0.25, None) for t in GRU}
    D.run_stage(_args(tmp_path, "GRU", 1), runner=_fake_runner(vals1), say=lambda *_: None)
    p = tmp_path / "P2_GRU_STAGE1.json"
    obj = json.loads(p.read_text(encoding="utf-8"))
    obj["selection"]["stage2_plan"]["GRU_D256"] = GRU[0]["hp_id"]                   # edited plan, stale self-hash
    p.write_text(json.dumps(obj), encoding="utf-8")
    raises_with(D.P2Error, "self-hash", D.run_stage, _args(tmp_path, "GRU", 2), runner=None)
