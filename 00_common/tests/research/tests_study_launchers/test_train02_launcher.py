"""TRAIN-02 — the launcher flags of a P2 (TUNING) run: --role TUNING --hp-id [--hp-registry] [--theta0-variants].

  * preflight (torch-free): --role TUNING needs --hp-id and --phase CALIBRATION, and refuses --calibration-record /
    --continue-run / --continuation-budget; --hp-id without --role TUNING is refused; --peak-lr stays required for every
    other launch; a TUNING launch needs no --peak-lr (the HP trial gives it);
  * after the environment export: the trial comes from the hash-checked HP registry (a tampered copy is refused; an
    explicit --peak-lr must equal the trial's; another family's trial is refused) and gives the peak LR, dropout,
    schedule and model variant; theta0_init_sha256 comes from THETA0_VARIANTS.json (self-hash verified); a missing file
    refuses a non-smoke TUNING launch and gives None for a smoke;
  * the P2 driver's command line parses and passes the preflight.
TRAIN-02R1: F2 (a TUNING launch refuses --train-manifest / --eval-manifest), F6 (--recipe-id for a MAIN run: preflight
and the hash-checked recipe values; an HP-trial id is not a recipe), the P2R2 default (the stale P2 file refused).
Negative controls: a launcher that takes --peak-lr over the HP trial; a TUNING preflight that accepts manifest flags.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import v4int.launch_central as L
from intkit import assert_raises, nc
from test_launcher import fake_runner, gate


def targs(tmp_path, *extra, smoke=False, phase="CALIBRATION", hp="P2_GRU_D256_S0_s3", family="GRU", role=True):
    a = ["--family", family, "--seed", "2026", "--phase", phase, "--gpu", "0", "--run-dir", str(tmp_path / "r"),
         "--history-policy", "USER_HISTORY_L50"]
    if role:
        a += ["--role", "TUNING"]
    if hp:
        a += ["--hp-id", hp]
    if smoke:
        a += ["--smoke", "--checkpoint-every", "300"]
    return L.parse_args(a + list(extra))


def test_tuning_preflight(tmp_path):
    g = gate(tmp_path)
    rec = L.preflight(targs(tmp_path), runner=fake_runner(), environ={}, gate_path=g)
    assert rec["tuning"] == {"role": "TUNING", "hp_id": "P2_GRU_D256_S0_s3"}
    for a in (targs(tmp_path, hp=None), targs(tmp_path, phase="MAIN"),
              targs(tmp_path, "--calibration-record", "c.json"), targs(tmp_path, "--continue-run"),
              targs(tmp_path, "--continuation-budget", "b.json"), targs(tmp_path, role=False)):
        assert_raises(L.LaunchRefused, L.preflight, a, runner=fake_runner(), environ={}, gate_path=g)
    no_lr = L.parse_args(["--family", "GRU", "--seed", "2026", "--phase", "CALIBRATION", "--gpu", "0", "--run-dir",
                          str(tmp_path / "x"), "--history-policy", "USER_HISTORY_L50"])
    assert_raises(L.LaunchRefused, L.preflight, no_lr, runner=fake_runner(), environ={}, gate_path=g)
    with pytest.raises(SystemExit):
        targs(tmp_path, "--role", "MAIN")                                      # only TUNING is a --role value


def test_driver_command_line_parses(tmp_path):
    from v4train.p2_driver import launch_cmd
    cmd = launch_cmd("SASREC", "P2_SASREC_D384_B2_S1_s1", 2027, 0, tmp_path / "r", history_policy="USER_HISTORY_L50")
    a = L.parse_args(cmd[3:])
    assert (a.role, a.hp_id, a.phase, a.seed, a.peak_lr) == ("TUNING", "P2_SASREC_D384_B2_S1_s1", "CALIBRATION", 2027, None)
    L.preflight(a, runner=fake_runner(), environ={}, gate_path=gate(tmp_path))


def _check_trial_values(tmp_path):
    from v4train.hp_registry import load_hp_registry, trial
    t = trial(load_hp_registry(), "P2_GRU_D256_S0_s3")
    v = L._tuning_values(targs(tmp_path), "GRU")
    assert (v["peak_lr"], v["dropout"], v["schedule"], v["model_variant"]) == \
        (t["peak_lr"], t["dropout"], t["schedule"], t["model_variant"]) and v["hp_sha256"] == t["sha256"]
    assert L._tuning_values(targs(tmp_path, "--peak-lr", repr(t["peak_lr"])), "GRU")["peak_lr"] == t["peak_lr"]
    assert_raises(L.LaunchRefused, L._tuning_values, targs(tmp_path, "--peak-lr", "1e-3"), "GRU")
    assert_raises(L.LaunchRefused, L._tuning_values, targs(tmp_path), "SASREC")
    obj = json.loads(Path(v["hp_registry"]).read_text(encoding="utf-8"))
    obj["families"]["GRU"][7]["peak_lr"] = 1e-3                                   # tampered copy
    bad = tmp_path / "HP.json"
    bad.write_text(json.dumps(obj), encoding="utf-8")
    assert_raises(L.LaunchRefused, L._tuning_values, targs(tmp_path, "--hp-registry", str(bad)), "GRU")


def test_trial_values_from_the_hp_registry(tmp_path):
    _check_trial_values(tmp_path)


@nc("a launcher that takes --peak-lr over the HP trial")
def test_nc_peak_lr_over_trial(tmp_path, monkeypatch):
    real = L._tuning_values

    def mutant(args, fam):
        v = real(L.parse_args([x for x in _argv_without_lr(args)]), fam)
        if args.peak_lr is not None:
            v["peak_lr"] = float(args.peak_lr)                                    # MUTANT
        return v
    monkeypatch.setattr(L, "_tuning_values", mutant)
    _check_trial_values(tmp_path)


def _argv_without_lr(args):
    a = ["--family", args.family, "--seed", str(args.seed), "--phase", args.phase, "--gpu", str(args.gpu),
         "--run-dir", str(args.run_dir), "--history-policy", args.history_policy, "--role", "TUNING",
         "--hp-id", args.hp_id]
    if args.hp_registry:
        a += ["--hp-registry", str(args.hp_registry)]
    return a


def test_theta0_from_theta0_variants(tmp_path):
    tun = {"model_variant": "GRU_D256", "hp_id": "P2_GRU_D256_S0_s3"}
    missing = tmp_path / "none.json"
    assert L._theta0_for(targs(tmp_path, "--theta0-variants", str(missing), smoke=True), "S", dict(tun), 2026) is None
    assert_raises(L.LaunchRefused, L._theta0_for, targs(tmp_path, "--theta0-variants", str(missing)), "S", dict(tun),
                  2026)
    import torch
    from v4models.variants import self_sha256
    obj = {"format": "PPSI_V4_THETA0_VARIANTS_1", "K_override": None,
           "schemas": {"S": {"builds": {torch.__version__: {"GRU_D256": {"2026": {"init_sha256": "c" * 64}}}}}}}
    obj["self_sha256"] = self_sha256(obj)
    p = tmp_path / "T0.json"
    p.write_text(json.dumps(obj), encoding="utf-8")
    t2 = dict(tun)
    assert L._theta0_for(targs(tmp_path, "--theta0-variants", str(p)), "S", t2, 2026) == "c" * 64
    assert t2["theta0_variants"]["self_sha256"] == obj["self_sha256"]
    assert_raises(L.LaunchRefused, L._theta0_for, targs(tmp_path, "--theta0-variants", str(p)), "S", dict(tun), 2027)
    obj["schemas"]["S"]["builds"][torch.__version__]["GRU_D256"]["2026"]["init_sha256"] = "d" * 64   # stale self-hash
    p.write_text(json.dumps(obj), encoding="utf-8")
    assert_raises(L.LaunchRefused, L._theta0_for, targs(tmp_path, "--theta0-variants", str(p)), "S", dict(tun), 2026)


# ------------------------------------------------------------------------------------------------ TRAIN-02R1
def _check_manifest_flags_refused(tmp_path):
    g = gate(tmp_path)
    for extra in (("--eval-manifest", "MAIN_VALIDATION_5_100_131072"), ("--train-manifest", "MAIN_TRAIN_5_100_131072")):
        assert_raises(L.LaunchRefused, L.preflight, targs(tmp_path, *extra), runner=fake_runner(), environ={},
                      gate_path=g)
        assert_raises(L.LaunchRefused, L.preflight, targs(tmp_path, *extra, smoke=True), runner=fake_runner(),
                      environ={})


def test_f2_tuning_refuses_manifest_overrides(tmp_path):
    _check_manifest_flags_refused(tmp_path)


@nc("a TUNING preflight that accepts --train-manifest / --eval-manifest (MAIN_VALIDATION could be loaded)")
def test_nc_tuning_accepts_manifest_overrides(tmp_path, monkeypatch):
    real = L._role_preflight

    def lax(args, rec):
        a = L.parse_args(_argv_without_lr(args) + (["--smoke", "--checkpoint-every", "300"] if args.smoke else []))
        real(a, rec)                                                              # MUTANT: the manifest flags dropped
    monkeypatch.setattr(L, "_role_preflight", lax)
    _check_manifest_flags_refused(tmp_path)


@pytest.fixture(autouse=True)
def _p2_cleanup():
    yield
    import p2kit_int
    p2kit_int.cleanup()                               # the V4-internal P2 fixture dirs (TRAIN-02R3)


def margs(tmp_path, *extra, phase="MAIN", smoke=False):
    a = ["--family", "GRU", "--seed", "2026", "--phase", phase, "--gpu", "0", "--run-dir", str(tmp_path / "m"),
         "--history-policy", "USER_HISTORY_L50", "--recipe-id", "02B_GRU"]
    if smoke:
        a += ["--smoke", "--checkpoint-every", "300"]
    return L.parse_args(a + list(extra))


def test_f6_recipe_preflight(tmp_path):
    g = gate(tmp_path)
    rec = L.preflight(margs(tmp_path), runner=fake_runner(), environ={}, gate_path=g)
    assert rec["tuned_recipe"] == {"recipe_id": "02B_GRU"}, "a tuned-recipe MAIN run needs no --calibration-record"
    for a in (margs(tmp_path, phase="CALIBRATION"), margs(tmp_path, "--calibration-record", "c.json"),
              targs(tmp_path, "--recipe-id", "02B_GRU")):
        assert_raises(L.LaunchRefused, L.preflight, a, runner=fake_runner(), environ={}, gate_path=g)


def test_f6_recipe_values(tmp_path):
    from p2kit_int import register                                            # vendored (TRAIN-02R2 (b))
    reg, _ = register(tmp_path)
    v = L._recipe_values(margs(tmp_path, "--tuned-recipes", str(reg), smoke=True), "GRU")
    assert (v["recipe_id"], v["model_variant"], v["schedule"]) == ("02B_GRU", "GRU_D256", "S1")
    assert_raises(L.LaunchRefused, L._recipe_values, margs(tmp_path, "--tuned-recipes", str(reg), smoke=True), "SASREC")
    assert_raises(L.LaunchRefused, L._recipe_values,
                  margs(tmp_path, "--tuned-recipes", str(reg), "--peak-lr", "1e-3", smoke=True), "GRU")
    assert_raises(L.LaunchRefused, L._recipe_values,
                  margs(tmp_path, "--tuned-recipes", str(tmp_path / "no.json"), smoke=True), "GRU")
    bad = L.parse_args(["--family", "GRU", "--seed", "2026", "--phase", "MAIN", "--gpu", "0", "--run-dir", "x",
                        "--history-policy", "USER_HISTORY_L50", "--recipe-id", "P2_GRU_D256_S1_s4",
                        "--tuned-recipes", str(reg), "--smoke", "--checkpoint-every", "300"])
    assert_raises(L.LaunchRefused, L._recipe_values, bad, "GRU")                   # an HP-trial id is not a recipe
    # TRAIN-02R2 F2: a non-smoke launch accepts only the production file pinned at G1b
    assert_raises(L.LaunchRefused, L._recipe_values, margs(tmp_path, "--tuned-recipes", str(reg)), "GRU")


def test_hp_registry_default_is_p2r2(tmp_path):
    from v4train.hp_registry import REGISTRY_PATH
    assert REGISTRY_PATH.name == "HP_REGISTRY_P2R2.json"
    stale = REGISTRY_PATH.parent / "HP_REGISTRY_P2.json"
    assert_raises(L.LaunchRefused, L._tuning_values, targs(tmp_path, "--hp-registry", str(stale)), "GRU")
