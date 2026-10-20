"""P2-05 C2 (v4int.c_full_small; PREG2 §4.3) - the launch contract (SMALL_SASREC_D64_B2, seed 2026, run
id prefix P2_C2_CFULL_SMALL_, production SMALL recipe, no peak / manifest / lock overrides), the launch_central wiring on
the REAL REGISTRY_V2_FULL_EXT.json / REGISTRY_V2.json (read only): the trainer's MAIN role = (FULL_TRAIN_5_100_TIEBREAK,
MAIN_VALIDATION_5_100_131072); the dry run binds the TIEBREAK manifest and the D64 variant; the labels
(P2_C2_CFULL_SMALL / P2_CONTROL; C_FULL untouched); v4int.c_full is unchanged by importing / installing this shim.
NCs: C_FULL's own run id accepted; the 02B (D256) recipe passing the plan check."""
from __future__ import annotations

import hashlib
import json
import types
from pathlib import Path

import pytest

import v4int.c_full_small as CS
from intkit import assert_raises, nc

CODE = Path(__file__).resolve().parent.parent
V4 = CODE.parent
EXT = V4 / "contracts" / "manifest_registry" / "REGISTRY_V2_FULL_EXT.json"
BASE_REL = "contracts/manifest_registry/REGISTRY_V2.json"
BASE = V4 / BASE_REL
ARGV = ["--family", "SASREC", "--seed", "2026", "--phase", "MAIN", "--recipe-id", "SMALL_SASREC_D64_B2", "--gpu", "0",
        "--run-dir", "RUNDIR", "--history-policy", "USER_HISTORY_L50"]
D64_RECIPE = {"recipe_id": "SMALL_SASREC_D64_B2", "model_variant": "SASREC_D64_B2", "peak_lr": 7.5e-4,
              "recipe_registry": "contracts/hp_registry/SMALL_RECIPE_SASREC_D64_B2.json", "dropout": [0.1],
              "schedule": "S0", "recipe_sha256": "r"}


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _args(extra=(), argv=ARGV):
    from v4int.launch_central import parse_args
    return parse_args(list(argv) + list(extra))


def _swap(old, new):
    return [new if x == old else x for x in ARGV]


def test_ids_match_d8_and_central_ft():
    from v4data08 import d8_common as D8
    from v4int import central_ft as CF
    assert (CS.TRAIN_ID, CS.EVAL_ID) == (D8.TRAIN_ID, D8.EVAL_ID)
    assert CF.p2_label_of_run_id(f"{CS.RUN_ID_PREFIX}SASREC_D64_B2_s2026") == CS.RUN_LABEL == CF.C2_ARM


def test_launch_contract_default_run_id_and_refusals():
    assert CS.check_launch_args(_args()) == "P2_C2_CFULL_SMALL_SASREC_D64_B2_s2026"
    assert CS.check_launch_args(_args(["--allow-shared-gpu", "--coschedule-06g"])) == \
        "P2_C2_CFULL_SMALL_SASREC_D64_B2_s2026"
    for extra in (["--train-manifest", "MAIN_TRAIN_5_100_131072"], ["--eval-manifest", "X"], ["--peak-lr", "1e-3"],
                  ["--run-id", "C_FULL_02B_SASREC_s2026"], ["--run-id", "SMALL_SASREC_D64_B2_MAIN_s2026"],
                  ["--calibration-record", "c.json"], ["--continuation-budget", "b.json"], ["--role", "TUNING"],
                  ["--transform-state", "t.json"], ["--allow-shared-gpu"], ["--coschedule-06g"],
                  ["--manifest-dir", "m"], ["--manifest-hashes", "h.json"], ["--header-list", "l.json"],
                  ["--lock-path", "lock.json"], ["--theta0-variants", "t0.json"], ["--hp-registry", "hp.json"],
                  ["--tuned-recipes", "other.json"]):
        assert_raises(CS.CFullSmallRefused, CS.check_launch_args, _args(extra))
    for old, new in (("SMALL_SASREC_D64_B2", "02B_SASREC"), ("2026", "2027"), ("SASREC", "GRU"),
                     ("USER_HISTORY_L50", "SESSION_L50")):
        assert_raises(CS.CFullSmallRefused, CS.check_launch_args, _args(argv=_swap(old, new)))
    for extra in (["--smoke"], ["--smoke", "--tuned-recipes", "other.json"]):        # S8: no MAIN_VALIDATION smoke
        assert_raises(CS.CFullSmallRefused, CS.check_launch_args, _args(extra))


@nc("a C2 smoke (MAIN_VALIDATION into SMOKE_EVAL.json) must be refused (PREG2 §6.3 S8)")
def test_nc_c2_smoke_accepted():
    try:
        CS.check_launch_args(_args(["--smoke"]))
        accepted = True
    except CS.CFullSmallRefused:
        accepted = False
    assert accepted


@nc("C_FULL's own run id must be refused by the C2 launch contract")
def test_nc_c_full_run_id_accepted():
    try:
        CS.check_launch_args(_args(["--run-id", "C_FULL_02B_SASREC_s2026"]))
        accepted = True
    except CS.CFullSmallRefused:
        accepted = False
    assert accepted


def test_normalise_adds_run_id_and_the_production_small_recipe():
    from v4int.small_recipe import RECIPE_PATH
    a = _args()
    out = CS.normalise_argv(list(ARGV), a, "P2_C2_CFULL_SMALL_SASREC_D64_B2_s2026")
    assert out[-4:] == ["--run-id", "P2_C2_CFULL_SMALL_SASREC_D64_B2_s2026", "--tuned-recipes", str(RECIPE_PATH)]
    assert Path(a.tuned_recipes) == Path(RECIPE_PATH)


def _lc(tmp_path, recipe=D64_RECIPE):
    class Refused(RuntimeError):
        pass

    base_ctx = {"manifest_registry_path": BASE, "manifest_registry_rel_path": BASE_REL,
                "manifest_registry_file_sha256": _sha(BASE),
                "manifest_registry_self_sha256": json.loads(BASE.read_text(encoding="utf-8"))["self_sha256"],
                "data_version": "CLEAN_V2_POP5100_V2"}
    return types.SimpleNamespace(G1_GATE=tmp_path / "gate.json", V4=V4, LaunchRefused=Refused,
                                 bind_data_protocol_lock_v2=lambda paths, *, smoke, gate_path=None: dict(base_ctx),
                                 build_trainer_registry_view=lambda *a, **k: pytest.fail("original view called"),
                                 export_environment=lambda gpu: {"CUDA_VISIBLE_DEVICES": str(gpu)},
                                 _recipe_values=lambda args, fam: dict(recipe), data_paths=lambda args: None,
                                 utc=lambda: "2026-10-03T00:00:00Z")


@pytest.fixture
def lc_env(tmp_path, monkeypatch):
    import v4int.verify as VF
    monkeypatch.setattr(VF, "verify_registry", VF.verify_registry)      # restored after the test
    c_full_sha = _sha(CODE / "v4int" / "c_full.py")
    yield _lc(tmp_path)
    assert _sha(CODE / "v4int" / "c_full.py") == c_full_sha               # c_full.py untouched


def test_install_rewires_launch_central_on_the_real_extension(lc_env, tmp_path):
    import v4int.verify as VF
    from v4train.registry import load_manifest_registry
    labelled = []
    record = {}
    CS.install(lc_env, EXT, record, labels=lambda: labelled.append(1) or {"stub": True})
    ctx = lc_env.bind_data_protocol_lock_v2(None, smoke=True)
    assert Path(ctx["manifest_registry_path"]) == EXT.resolve() and labelled == [1]
    assert record["ext_checks"]["all_ok"] and ctx["c_full_base_registry"]["file_sha256"] == _sha(BASE)
    assert getattr(VF.verify_registry, "__c_full_small__", False)        # wrapper 3 (c_full's factory) installed
    rd = tmp_path / "run"
    v = lc_env.build_trainer_registry_view(ctx["manifest_registry_path"], rd,
                                           registry_rel_path=ctx["manifest_registry_rel_path"],
                                           registry_file_sha256=ctx["manifest_registry_file_sha256"])
    assert load_manifest_registry(v["path"]).role("MAIN") == (CS.TRAIN_ID, CS.EVAL_ID)
    assert [p.name for p in rd.glob("P2_C2_SHIM_*.json")] == ["P2_C2_SHIM_0.json"]


def test_install_refuses_a_missing_extension(lc_env, tmp_path):
    CS.install(lc_env, tmp_path / "absent.json", {}, labels=lambda: pytest.fail("labels before the extension check"))
    assert_raises(lc_env.LaunchRefused, lc_env.bind_data_protocol_lock_v2, None, smoke=True)


def test_dry_run_binds_the_tiebreak_manifest_and_the_d64_variant(lc_env, tmp_path, monkeypatch):
    verified = []
    monkeypatch.setattr(CS, "_verify_manifests", lambda LC, lockctx, paths: verified.append(
        Path(lockctx["manifest_registry_path"]).name) or {"stub": True})
    shim, largv = CS.split_argv(["--no-providers", "--"] + [str(tmp_path / "run") if x == "RUNDIR" else x
                                                            for x in ARGV])
    args = _args(argv=largv)
    rid = CS.check_launch_args(args)
    largv = CS.normalise_argv(largv, args, rid)
    record = {"dry_run": True}
    CS.install(lc_env, EXT, record, labels=lambda: {"stub": True})
    out = CS.dry_run(lc_env, args, shim, record, small_install=False)
    assert out["plan"] == {"train_manifest": "FULL_TRAIN_5_100_TIEBREAK", "eval_manifest": "MAIN_VALIDATION_5_100_131072",
                           "recipe_id": "SMALL_SASREC_D64_B2", "model_variant": "SASREC_D64_B2", "peak_lr": 7.5e-4}
    assert out["trainer_view"]["expected_manifests"] == ["FULL_TRAIN_5_100_TIEBREAK", "MAIN_VALIDATION_5_100_131072"]
    assert verified == ["REGISTRY_V2_FULL_EXT.json"] and not (tmp_path / "run").exists()   # nothing written
    assert args.run_id == "P2_C2_CFULL_SMALL_SASREC_D64_B2_s2026"


def test_plan_check_refusals():
    ok = CS.check_plan((CS.TRAIN_ID, CS.EVAL_ID), D64_RECIPE)
    assert ok["model_variant"] == "SASREC_D64_B2"
    for want, tuned in (((CS.TRAIN_ID, CS.EVAL_ID), dict(D64_RECIPE, recipe_id="02B_SASREC", model_variant="SASREC_D256_B3")),
                        (("MAIN_TRAIN_5_100_131072", CS.EVAL_ID), D64_RECIPE),
                        ((CS.TRAIN_ID, CS.EVAL_ID), dict(D64_RECIPE, model_variant="SASREC_D256_B2"))):
        assert_raises(CS.CFullSmallRefused, CS.check_plan, want, tuned)


@nc("the 02B (SASREC_D256_B3) recipe must fail the C2 plan check")
def test_nc_02b_recipe_passes_plan():
    try:
        CS.check_plan((CS.TRAIN_ID, CS.EVAL_ID), dict(D64_RECIPE, recipe_id="02B_SASREC", model_variant="SASREC_D256_B3"))
        accepted = True
    except CS.CFullSmallRefused:
        accepted = False
    assert accepted


def test_labels_on_the_real_runconfig(monkeypatch):
    import v4train.recipe as RC
    import v4train.trainer as TR
    from v4int import central_ft as CF
    monkeypatch.setattr(RC.RunConfig, "run_label", RC.RunConfig.__dict__["run_label"])
    monkeypatch.setattr(TR.CentralTrainer, "_row_labels", TR.CentralTrainer.__dict__["_row_labels"])
    for cls in (RC.RunConfig, TR.CentralTrainer):
        monkeypatch.setattr(cls, "__p2_label__", False, raising=False)
    CF.install_p2_label_wrappers(RC, TR)
    prop = RC.RunConfig.__dict__["run_label"]
    fake = lambda rid, t=False: types.SimpleNamespace(run_id=rid, test_overrides=t, role=None, manifest_role="MAIN",  # noqa: E731
                                                      population="COHORT", data_version="CLEAN_V2_POP5100_V2")
    assert prop.fget(fake("P2_C2_CFULL_SMALL_SASREC_D64_B2_s2026")) == "P2_C2_CFULL_SMALL"
    assert prop.fget(fake("C_FULL_02B_SASREC_s2026")) == "PRODUCTION"               # C_FULL untouched
    assert prop.fget(fake("SMOKE_NOT_EVIDENCE__P2_C2_CFULL_SMALL_SASREC_D64_B2_s2026", True)) == "SMOKE_NOT_EVIDENCE"
    base = {"row_status": "CENTRAL_CONTEST", "comparison_status": "SAME_CONDITIONS_BLOCK_CANDIDATE",
            "c_matched_eligible": True, "downstream_seed_allowed": True}
    stub = type("StubTrainer", (), {"_row_labels": lambda self: dict(base)})        # rows: on a stub trainer class
    CF.install_p2_label_wrappers(types.SimpleNamespace(RunConfig=type("R", (), {"run_label": property(
        lambda self: "PRODUCTION")})), types.SimpleNamespace(CentralTrainer=stub))
    s = stub()
    s.cfg = fake("P2_C2_CFULL_SMALL_SASREC_D64_B2_s2026")
    got = s._row_labels()
    assert got["row_status"] == got["comparison_status"] == "P2_CONTROL" and got["method"] == "P2_C2_CFULL_SMALL"
    assert got["c_matched_eligible"] is False and got["downstream_seed_allowed"] is False
    s.cfg = fake("C_FULL_02B_SASREC_s2026")
    assert s._row_labels() == base


def test_main_refusals(tmp_path):
    assert CS.main(["--family", "SASREC"]) == 2                                      # no '--'
    rd = str(tmp_path / "run")
    base = [rd if x == "RUNDIR" else x for x in ARGV]
    assert CS.main(["--"] + [("02B_SASREC" if x == "SMALL_SASREC_D64_B2" else x) for x in base]) == 2
    assert CS.main(["--"] + base + ["--peak-lr", "1e-3"]) == 2
    assert not Path(rd).exists()
