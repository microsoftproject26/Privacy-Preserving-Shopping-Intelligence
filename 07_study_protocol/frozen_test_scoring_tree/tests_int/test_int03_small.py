"""INT-03 item 3 — the SMALL (SASREC_D64_B2) mini-ladder and SMALL recipe (v4int.small_recipe).

  * the ladder = the 4 SASRec LR midpoints of the VERIFIED HP_REGISTRY_P2R2.json, S0 / [0.1], SASREC_D64_B2; written
    O_EXCL, self-hashed; a tampered copy is refused;
  * install() routes ONLY the SMALL formats through v4int.small_recipe: the P2R2 default and every other file load
    exactly as before; RunConfig.validate accepts a ladder trial (TUNING) and the SMALL recipe (MAIN, C_SMALL);
  * launch_central --role TUNING --hp-registry <ladder> and --recipe-id SMALL_SASREC_D64_B2 --tuned-recipes <file>
    resolve the values; a non-smoke SMALL recipe must be the production path;
  * the ladder driver (fake runner) writes stage 1 / stage 2 files with the 02R1 §3 rule and `register` the recipe.
Negative controls: a ladder at other LRs would be loaded without the design check; a routing hook that diverted every
file would change the P2R2 load.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

import v4int.launch_central as L
import v4int.small_recipe as SR
from intkit import assert_raises, nc

sys.path.append(str(Path(__file__).resolve().parents[1] / "tests_train"))
from trainkit import KEY, registry_keys  # noqa: E402  (read-only import)


@pytest.fixture(autouse=True)
def _hook():
    SR.install()
    yield
    SR.uninstall()


def _ladder(tmp_path) -> Path:
    p = tmp_path / "HP_REGISTRY_SMALL_LADDER.json"
    if not p.exists():
        SR.write_ladder(p)
    return p


def test_ladder_is_the_p2r2_sasrec_midpoints(tmp_path):
    import v4train.hp_registry as H
    p2 = SR._ORIG["load_hp_registry"](H.REGISTRY_PATH)
    lad = SR.load_ladder(_ladder(tmp_path))
    ts = lad["families"]["SASREC"]
    assert [t["peak_lr"] for t in ts] == [s["peak_lr"] for s in p2["raw"]["lr_grid"]["SASREC"]["strata"]]
    assert {(t["model_variant"], t["schedule"], tuple(t["dropout"])) for t in ts} == {("SASREC_D64_B2", "S0", (0.1,))}
    assert [t["hp_id"] for t in ts] == [f"P2S_SASREC_D64_B2_S0_s{k}" for k in range(4)]
    assert_raises(FileExistsError, SR.write_ladder, _ladder(tmp_path))                       # O_EXCL


def test_tampered_ladder_refused(tmp_path):
    obj = json.loads(_ladder(tmp_path).read_text(encoding="utf-8"))
    obj["families"]["SASREC"][1]["peak_lr"] = 1e-3
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(obj), encoding="utf-8")
    assert_raises(SR.SmallRecipeError, SR.load_ladder, bad)
    import v4train.hp_registry as H
    assert_raises(H.HPRegistryError, H.load_hp_registry, bad)


def test_hook_leaves_every_other_file_unchanged(tmp_path):
    import v4train.hp_registry as H
    a = H.load_hp_registry()
    b = SR._ORIG["load_hp_registry"](H.REGISTRY_PATH)
    assert a == b and H.load_hp_registry(_ladder(tmp_path))["raw"]["format"] == SR.HP_FORMAT


def _targs(tmp_path, *extra, smoke=True):
    a = ["--family", "SASREC", "--seed", "2026", "--gpu", "0", "--run-dir", str(tmp_path / "r"),
         "--history-policy", "USER_HISTORY_L50"] + list(extra)
    if smoke:
        a += ["--smoke", "--checkpoint-every", "300"]
    return L.parse_args(a)


def test_tuning_values_and_runconfig_accept_a_ladder_trial(tmp_path):
    from v4train.recipe import RunConfig
    lp = _ladder(tmp_path)
    v = L._tuning_values(_targs(tmp_path, "--phase", "CALIBRATION", "--role", "TUNING", "--hp-id",
                                "P2S_SASREC_D64_B2_S0_s2", "--hp-registry", str(lp)), "SASREC")
    assert v["model_variant"] == "SASREC_D64_B2" and v["dropout"] == [0.1] and v["schedule"] == "S0"
    base = dict(run_id="r", exposure_point="end", cuda_visible_devices="0", determinism_mode="STRICT", key=dict(KEY),
                **registry_keys())
    c = RunConfig(family="SASREC", seed=2026, role="TUNING", phase="CALIBRATION", hp_id="P2S_SASREC_D64_B2_S0_s2",
                  hp_registry=str(lp), peak_lr=v["peak_lr"], model_variant="SASREC_D64_B2", dropout=(0.1,),
                  schedule="S0", theta0_init_sha256="a" * 64, **base)
    c.validate()
    import dataclasses
    from v4train.recipe import RecipeError
    assert_raises(RecipeError, dataclasses.replace(c, peak_lr=1e-3).validate)


def _fake_run(values):
    """A fake launch_central child: writes a DONE_TUNING summary with the given endpoint value."""
    class P:
        returncode = 0

    def runner(cmd, **kw):
        rd = Path(cmd[cmd.index("--run-dir") + 1])
        hp = cmd[cmd.index("--hp-id") + 1]
        seed = int(cmd[cmd.index("--seed") + 1])
        rd.mkdir(parents=True, exist_ok=True)
        (rd / "RUN_SUMMARY.json").write_text(json.dumps({
            "status": "DONE_TUNING", "endpoint": {"value": values[(hp, seed)]}, "ckpt_avg": {"value": None},
            "shared_numel": 20_710_048, "seed": seed}), encoding="utf-8")
        return P()
    return runner


def _stages(tmp_path) -> Path:
    lp = _ladder(tmp_path)
    out = tmp_path / "ladder"
    vals = {(f"P2S_SASREC_D64_B2_S0_s{k}", 2026): 0.10 + 0.01 * k for k in range(4)}
    vals[("P2S_SASREC_D64_B2_S0_s3", 2027)] = 0.125
    ns = argparse.Namespace(stage=1, gpu=0, out_dir=out, hp_registry=lp, theta0_variants=None,
                            history_policy="USER_HISTORY_L50", dry_run=False, allow_shared_gpu=True)
    assert SR.run_stage(ns, runner=_fake_run(vals), say=lambda *_: None) == 0
    s1 = SR.load_stage(out / "SMALL_LADDER_STAGE1.json")
    assert s1["selection"]["stage2_plan"] == {"SASREC_D64_B2": "P2S_SASREC_D64_B2_S0_s3"}
    assert "--allow-shared-gpu" in s1["runs"][0]["cmd"] and str(lp) in s1["runs"][0]["cmd"]
    ns.stage = 2
    assert SR.run_stage(ns, runner=_fake_run(vals), say=lambda *_: None) == 0
    assert_raises(SR.SmallRecipeError, SR.run_stage, ns, runner=_fake_run(vals), say=lambda *_: None)   # no re-run
    return out


def test_ladder_driver_register_and_recipe_values(tmp_path):
    out = _stages(tmp_path)
    rp = tmp_path / "SMALL_RECIPE.json"
    reg = SR.register(out, rp, _ladder(tmp_path))
    assert reg["recipe"]["source_hp_id"] == "P2S_SASREC_D64_B2_S0_s3" and reg["recipe"]["mean_6.0"] == 0.1275
    a = _targs(tmp_path, "--phase", "MAIN", "--recipe-id", SR.RECIPE_ID, "--tuned-recipes", str(rp))
    v = L._recipe_values(a, "SASREC")
    assert v["model_variant"] == "SASREC_D64_B2" and v["peak_lr"] == reg["recipe"]["peak_lr"]
    assert_raises(L.LaunchRefused, L._recipe_values, _targs(tmp_path, "--phase", "MAIN", "--recipe-id", SR.RECIPE_ID,
                                                            "--tuned-recipes", str(rp), smoke=False), "SASREC")
    assert SR.check_small_production(SR.RECIPE_PATH, {"self_sha256": "x"})["pinned"] == "x"
    import v4int.launch_fl as F                                                   # the FL launcher's SMALL recipe
    fa = F.parse_args(["--arm", "FA", "--seed", "2026", "--gpu", "0", "--run-dir", str(tmp_path / "f"),
                       "--history-policy", "USER_HISTORY_L50", "--smoke", "--dry-run", "--small-recipe", str(rp)])
    assert F._small_recipe(fa)["peak_lr"] == reg["recipe"]["peak_lr"]
    said = []
    assert F.dry_run(fa, say=said.append) == 0 and json.loads(said[0])["arm_spec"]["cfg"]["lr_peak_local"] == \
        reg["recipe"]["peak_lr"]
    fa.smoke = False
    assert_raises(F.LaunchRefused, F._small_recipe, fa)                           # non-smoke: production path only
    assert_raises(SR.SmallRecipeError, SR.check_small_production, rp, {"self_sha256": "x"})


def test_launch_central_shared_gpu_flag(tmp_path):
    from test_launcher import fake_runner, gate
    g = gate(tmp_path)
    busy = fake_runner(apps=[("00000000:46:00.0", "4242")], mem={"0": 3000.0, "1": 0.0})
    tun = ["--phase", "CALIBRATION", "--role", "TUNING", "--hp-id", "P2S_SASREC_D64_B2_S0_s0"]
    assert_raises(L.LaunchRefused, L.preflight, _targs(tmp_path, *tun, smoke=False), runner=busy, environ={},
                  gate_path=g)
    rec = L.preflight(_targs(tmp_path, *tun, "--allow-shared-gpu", smoke=False), runner=busy, environ={}, gate_path=g)
    assert rec["gpu_check"]["shared_gpu"]["allowed"] is True
    main = ["--phase", "MAIN", "--recipe-id", SR.RECIPE_ID, "--allow-shared-gpu"]
    assert_raises(L.LaunchRefused, L.preflight, _targs(tmp_path, *main, smoke=False), runner=fake_runner(),
                  environ={}, gate_path=g)
    assert_raises(L.LaunchRefused, L.preflight, _targs(tmp_path, *tun, "--allow-shared-gpu", smoke=False),
                  runner=fake_runner(fail=True), environ={}, gate_path=g)             # nvidia-smi failure: refused


@nc("a ladder at other LRs would be accepted without the design check against P2R2")
def test_nc_ladder_without_design_check(tmp_path):
    obj = json.loads(_ladder(tmp_path).read_text(encoding="utf-8"))
    obj["families"]["SASREC"][0]["peak_lr"] = 5e-3
    obj["families"]["SASREC"][0]["sha256"] = SR.canonical_sha256(obj["families"]["SASREC"][0], exclude=("sha256",))
    obj["self_sha256"] = SR.canonical_sha256(obj)                                  # self-consistent forgery
    fresh = SR.ladder_obj()
    assert obj["families"] == fresh["families"], "the design check is what refuses the forged ladder"


@nc("a routing hook that diverted EVERY file would change the P2R2 load")
def test_nc_hook_diverting_every_file(tmp_path):
    import v4train.hp_registry as H
    lad = SR.load_ladder(_ladder(tmp_path))                                        # MUTANT: ignores the path
    assert lad == SR._ORIG["load_hp_registry"](H.REGISTRY_PATH)
