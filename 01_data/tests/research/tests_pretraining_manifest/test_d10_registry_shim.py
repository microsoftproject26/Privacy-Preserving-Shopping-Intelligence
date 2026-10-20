"""DATA-10 registry extension + pretraining launcher shim tests: REGISTRY_V2 entries byte-identical in the extension,
the trainer view (the unedited v4train loader resolves role MAIN to the PRETRAIN pair), the v4int.verify wrapper
routing, the PRETRAIN_FOR_FL labels (wrapper 4) against the real v4train selection rules, the endpoint_6.0 FL-init
record, the launch-argument contract and the launch_central wiring - with strict-xfail negative controls. The real
REGISTRY_V2 / DATA_PROTOCOL_LOCK_V2 files are only READ (copied into pytest's temporary directory where a write is
needed)."""
from __future__ import annotations

import copy
import inspect
import json
import shutil
import types
from pathlib import Path

import pytest

from conftest import SYN_PRETRAIN, V4
from data_testkit import assert_raises, nc
from v4data.common import sha256_file
from v4data10 import d10_common as D10
from v4data10 import d10_finalize as F
from v4data10 import d10_registry as R

BASE_REL = F.BASE_REL
BASE_PATH = V4 / BASE_REL


def _base():
    return json.loads(BASE_PATH.read_text(encoding="utf-8"))


def _entry(synth_build, status=D10.ROLE):
    e = F.registry_entry(synth_build["out"].parent, synth_build["out"].name)
    return dict(e, status=status, role=status)


_SYNTH_EXPECT: dict = {}


@pytest.fixture(autouse=True)
def _synthetic_population_expectation(monkeypatch):
    """The synthetic builds carry a toy population; the population content checks are pointed at it while
    `_SYNTH_EXPECT` is set (the tests that prove the REAL values are enforced clear it)."""
    _SYNTH_EXPECT.clear()
    real = R.expected_population
    monkeypatch.setattr(R, "expected_population", lambda: dict(_SYNTH_EXPECT) if _SYNTH_EXPECT else real())
    yield
    _SYNTH_EXPECT.clear()


def _ext(synth_build, base=None, **kw):
    base = _base() if base is None else base
    entry = _entry(synth_build, **kw)
    psha = entry["population_sha256_selection_key_v2"]
    _SYNTH_EXPECT.update(rank_range=list(SYN_PRETRAIN), n_users=entry["n_users"], n_decisions=entry["n_decisions"],
                         population_sha256_selection_key_v2=psha)
    return R.build_ext_registry(base, base_rel_path=BASE_REL, base_file_sha256=sha256_file(BASE_PATH), entry=entry,
                                band={"rank_range": list(SYN_PRETRAIN), "n_users": entry["n_users"],
                                      "population_sha256_selection_key_v2": psha},
                                build_receipt={"receipt_sha256": "x"}, code={"d10_code_sha256": "y"})


def _ok(ext, base=None):
    base = _base() if base is None else base
    return R.verify_ext_registry(ext, base, sha256_file(BASE_PATH))["all_ok"]


# ------------------------------------------------------------------------------------------ extension rule
def test_real_registry_v2_is_the_lock_v2_pinned_registry():
    from v4data08.d8_finalize import base_pin_checks
    pins = base_pin_checks(V4)
    assert all(v is True for k, v in pins.items() if not k.startswith("_")), pins


def test_extension_keeps_every_registry_v2_entry_byte_identical(synth_build):
    base = _base()
    ext = _ext(synth_build, base)
    chk = R.verify_ext_registry(ext, base, sha256_file(BASE_PATH))
    assert chk["all_ok"], {k: v for k, v in chk.items() if not v}
    for k in ("roles", "manifests", "population_bands"):
        for sk, v in base[k].items():
            assert R.canon(ext[k][sk]) == R.canon(v)
    assert ext["roles"]["PRETRAIN"] == {"train": D10.TRAIN_ID, "eval": "UNEW_VAL_VALIDATION_32768"}
    assert R.canon(ext["manifests"][D10.EVAL_ID]) == R.canon(base["manifests"]["UNEW_VAL_VALIDATION_32768"])
    assert ext["extends"]["self_sha256"] == base["self_sha256"] and ext["data_version"] == base["data_version"]
    assert set(ext) - set(base) == set(R.ADDED_TOP_KEYS)


def test_partial_build_entry_cannot_enter_the_registry(synth_build):
    assert_raises(R.ExtRegistryError, _ext, synth_build, status="PARTIAL_TEST_BUILD")


def test_write_extension_rereads_and_refuses_overwrite(synth_build, tmp_path):
    (tmp_path / "contracts" / "manifest_registry").mkdir(parents=True)
    shutil.copyfile(BASE_PATH, tmp_path / BASE_REL)
    ext = _ext(synth_build)
    out = F.write_extension(tmp_path, ext)
    assert out["checks"]["all_ok"] and out["self_sha256"] == ext["self_sha256"]
    assert out["registry"] == "contracts/manifest_registry/REGISTRY_V2_PRETRAIN_EXT.json"
    assert F.write_extension(tmp_path, ext)["file_sha256"] == out["file_sha256"]      # identical bytes: idempotent
    assert_raises(SystemExit, F.write_extension, tmp_path, R.with_self(dict(ext, extension_task="DATA-10b")))


def test_population_content_checks_use_the_registered_values(synth_build):
    """With the REAL expectation a toy-population extension is refused on every content check; the registered values
    are the band arithmetic of TIEBREAK - MAIN and the locally MEASURED band hash."""
    base = _base()
    ext = _ext(synth_build, base)
    _SYNTH_EXPECT.clear()
    chk = R.verify_ext_registry(ext, base, sha256_file(BASE_PATH))
    assert not chk["all_ok"]
    for k in ("population_bands.PRETRAIN_NONMAIN.rank_range", "population_bands.PRETRAIN_NONMAIN.n_users",
              "manifests.PRETRAIN_train.n_users", "manifests.PRETRAIN_train.n_decisions",
              "population_sha256.eq_registered"):
        assert chk[k] is False, k
    exp = R.expected_population()
    assert exp == {"rank_range": [131_072, 1_348_682], "n_users": 1_217_610, "n_decisions": 24_089_824,
                   "population_sha256_selection_key_v2": D10.PRETRAIN_SHA_V2}


@nc("a REGISTRY_V2 entry altered inside the extension (re-signed) must fail byte-identity")
def test_nc_tampered_base_entry(synth_build):
    ext = copy.deepcopy(_ext(synth_build))
    ext["manifests"]["MAIN_TRAIN_5_100_131072"]["n_decisions"] += 1
    assert _ok(R.with_self(ext))


@nc("a PRETRAIN role evaluated on MAIN_VALIDATION must fail the extension rule")
def test_nc_pretrain_eval_on_main_validation(synth_build):
    ext = copy.deepcopy(_ext(synth_build))
    ext["roles"]["PRETRAIN"]["eval"] = "MAIN_VALIDATION_5_100_131072"
    assert _ok(R.with_self(ext))


@nc("an extra, unregistered top-level key must fail the extension rule")
def test_nc_extra_top_key(synth_build):
    assert _ok(R.with_self(dict(_ext(synth_build), surprise=1)))


@nc("an extension that names another base file sha256 must fail")
def test_nc_wrong_base_file_sha(synth_build):
    base = _base()
    assert R.verify_ext_registry(_ext(synth_build, base), base, "0" * 64)["all_ok"]


@nc("a band whose rank_range starts at 0 (i.e. includes MAIN) must fail the extension rule")
def test_nc_band_including_main(synth_build):
    ext = copy.deepcopy(_ext(synth_build))
    ext["population_bands"]["PRETRAIN_NONMAIN"]["rank_range"] = [0, SYN_PRETRAIN[1]]
    assert _ok(R.with_self(ext))


@nc("an entry with a wrong n_decisions must fail the extension rule")
def test_nc_wrong_entry_n_decisions(synth_build):
    ext = copy.deepcopy(_ext(synth_build))
    ext["manifests"][D10.TRAIN_ID]["n_decisions"] += 1
    assert _ok(R.with_self(ext))


@nc("an entry whose population hash is the MAIN band's must fail the extension rule")
def test_nc_population_hash_of_main(synth_build):
    ext = copy.deepcopy(_ext(synth_build))
    main_sha = _base()["population_bands"]["MAIN"]["population_sha256_selection_key_v2"]
    ext["manifests"][D10.TRAIN_ID]["population_sha256_selection_key_v2"] = main_sha
    ext["population_bands"]["PRETRAIN_NONMAIN"]["population_sha256_selection_key_v2"] = main_sha
    _SYNTH_EXPECT["population_sha256_selection_key_v2"] = main_sha
    assert _ok(R.with_self(ext))


# ------------------------------------------------------------------------------------------ trainer view
def test_trainer_view_resolves_main_to_the_pretrain_pair(synth_build, tmp_path):
    from v4train.provider import expected_manifests
    from v4train.registry import load_manifest_registry
    ext = _ext(synth_build)
    view = R.trainer_view(ext, ext_rel_path=F.EXT_REL, ext_file_sha256="a" * 64)
    assert R.check_trainer_view(view, ext)["all_ok"]
    p = tmp_path / R.VIEW_FILE
    p.write_text(json.dumps(view, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    reg = load_manifest_registry(p)                             # the trainer's own, unedited loader
    assert reg.role("MAIN") == reg.role("CONTINUATION") == (D10.TRAIN_ID, D10.EVAL_ID)
    assert expected_manifests("COHORT", "MAIN", registry=str(p)) == (D10.TRAIN_ID, D10.EVAL_ID)


@nc("the launcher's standard REGISTRY_V2 view (MAIN -> MAIN_TRAIN) is not a PRETRAIN view")
def test_nc_standard_view_is_not_pretrain(synth_build, tmp_path):
    from v4int.launch_central import build_trainer_registry_view
    ext = _ext(synth_build)
    v = build_trainer_registry_view(BASE_PATH, tmp_path, registry_rel_path=BASE_REL,
                                    registry_file_sha256=sha256_file(BASE_PATH))
    assert R.check_trainer_view(json.loads(Path(v["path"]).read_text(encoding="utf-8")), ext)["all_ok"]


# ------------------------------------------------------------------------------------------ verify wrapper
def _stub_orig(calls):
    def orig(registry_path, lock_self, lock_file, *, lock=None, root=None):
        calls.append(Path(registry_path))
        reg = json.loads(Path(registry_path).read_text(encoding="utf-8"))
        return reg, sha256_file(registry_path), reg["self_sha256"], "V1CTX"
    return orig


def test_verify_wrapper_routes_extension_through_the_base(synth_build, tmp_path):
    from v4int.pretrain_small import make_verify_wrapper
    ext_p = tmp_path / "EXT.json"
    ext_p.write_text(json.dumps(_ext(synth_build)), encoding="utf-8")
    calls = []
    w = make_verify_wrapper(_stub_orig(calls), ext_p, BASE_PATH)
    reg, fsha, self_, ctx = w(ext_p, "L", "F")
    assert calls == [BASE_PATH] and reg["roles"]["PRETRAIN"] and fsha == sha256_file(ext_p) and ctx == "V1CTX"
    reg2 = w(BASE_PATH, "L", "F")[0]                            # any other path: delegated unchanged
    assert calls[-1] == BASE_PATH and "PRETRAIN" not in reg2["roles"] and self_ == reg["self_sha256"]


@nc("the verify wrapper must refuse a tampered extension")
def test_nc_verify_wrapper_refuses_tampered(synth_build, tmp_path):
    from v4int.pretrain_small import make_verify_wrapper
    ext = copy.deepcopy(_ext(synth_build))
    ext["manifests"]["UNEW_VAL_VALIDATION_32768"]["n_eval_rows"] -= 1
    ext_p = tmp_path / "EXT.json"
    ext_p.write_text(json.dumps(R.with_self(ext)), encoding="utf-8")
    try:
        make_verify_wrapper(_stub_orig([]), ext_p, BASE_PATH)(ext_p, "L", "F")
        accepted = True
    except ValueError:
        accepted = False
    assert accepted


# ------------------------------------------------------------------------------------------ wrapper 4: labels
def _fake_cfg(run_id, test_overrides=False):
    return types.SimpleNamespace(run_id=run_id, test_overrides=test_overrides, role=None, manifest_role="MAIN",
                                 population="COHORT", data_version="CLEAN_V2_POP5100_V2")


@pytest.fixture
def real_labels(monkeypatch):
    """Wrapper 4 installed on the REAL v4train.recipe.RunConfig / v4train.trainer.CentralTrainer, restored after."""
    import v4train.recipe as RC
    import v4train.trainer as TR
    from v4int.pretrain_small import install_label_wrappers
    monkeypatch.setattr(RC.RunConfig, "run_label", RC.RunConfig.__dict__["run_label"])
    monkeypatch.setattr(TR.CentralTrainer, "_row_labels", TR.CentralTrainer.__dict__["_row_labels"])
    for cls in (RC.RunConfig, TR.CentralTrainer):
        monkeypatch.setattr(cls, "__pretrain_label__", False, raising=False)
    install_label_wrappers(RC, TR)
    return RC, TR


def test_label_wrapper_on_the_real_runconfig_property(real_labels):
    RC, _ = real_labels
    prop = RC.RunConfig.__dict__["run_label"]
    assert prop.fget(_fake_cfg("PRETRAIN_FOR_FL_SMALL_SASREC_D64_B2_s2026")) == "PRETRAIN_FOR_FL"
    assert prop.fget(_fake_cfg("SMALL_SASREC_D64_B2_MAIN_s2026")) == "PRODUCTION"          # C_SMALL untouched
    assert prop.fget(_fake_cfg("SMOKE_NOT_EVIDENCE__PRETRAIN_FOR_FL_x", True)) == "SMOKE_NOT_EVIDENCE"
    assert "run_label" not in [f for f in RC.RunConfig.__dataclass_fields__]              # not in the config hash


def _stub_modules(base_rows):
    class RunConfig:
        def __init__(self, run_id, test_overrides=False):
            self.run_id, self.test_overrides = run_id, test_overrides

        @property
        def run_label(self):
            return "SMOKE_NOT_EVIDENCE" if self.test_overrides else "PRODUCTION"

    class CentralTrainer:
        def __init__(self, cfg):
            self.run_label = cfg.run_label

        def _row_labels(self):
            return dict(base_rows)

    return types.SimpleNamespace(RunConfig=RunConfig), types.SimpleNamespace(CentralTrainer=CentralTrainer)


BASE_ROW = {"row_status": "CENTRAL_CONTEST", "comparison_status": "SAME_CONDITIONS_BLOCK_CANDIDATE",
            "c_matched_eligible": True, "downstream_seed_allowed": True}


def test_label_wrapper_rows_and_real_selection_refusals():
    from v4int.pretrain_small import install_label_wrappers
    from v4train.selection import SelectionError, assert_c_matched_eligible, refuse_run_labels
    RC, TR = _stub_modules(BASE_ROW)
    install_label_wrappers(RC, TR)
    install_label_wrappers(RC, TR)                                      # idempotent
    cfg = RC.RunConfig("PRETRAIN_FOR_FL_SMALL_SASREC_D64_B2_s2026")
    tr = TR.CentralTrainer(cfg)
    row = dict({"method": "C", "family": "SASREC", "selection_rule": "CONTROLLED_BUDGET"}, **tr._row_labels(),
               run_label=tr.run_label)
    assert cfg.run_label == row["run_label"] == row["method"] == row["row_status"] == "PRETRAIN_FOR_FL"
    assert row["comparison_status"] == "PRETRAIN_FOR_FL_DESCRIPTIVE" and row["fl_init_rule"] == "endpoint_6.0"
    assert row["c_matched_eligible"] is False and row["downstream_seed_allowed"] is False
    for stage in ("COHORT", "TIEBREAK", "CALIBRATION", "C_MATCHED"):
        assert_raises(SelectionError, refuse_run_labels, [row], stage)
    assert_raises(SelectionError, assert_c_matched_eligible, row)


def test_label_wrapper_leaves_other_runs_untouched():
    from v4int.pretrain_small import install_label_wrappers
    RC, TR = _stub_modules(BASE_ROW)
    install_label_wrappers(RC, TR)
    for rid, smoke, lab in (("SMALL_SASREC_D64_B2_MAIN_s2026", False, "PRODUCTION"),
                            ("SMOKE_NOT_EVIDENCE__PRETRAIN_FOR_FL_x", True, "SMOKE_NOT_EVIDENCE")):
        cfg = RC.RunConfig(rid, smoke)
        assert cfg.run_label == lab and TR.CentralTrainer(cfg)._row_labels() == BASE_ROW


@nc("without wrapper 4 the pretraining run would carry the PRODUCTION label")
def test_nc_unwrapped_label_is_production():
    import v4train.recipe as RC
    prop = RC.RunConfig.__dict__["run_label"]
    fget = getattr(prop.fget, "__wrapped__", prop).fget
    assert fget(_fake_cfg("PRETRAIN_FOR_FL_SMALL_SASREC_D64_B2_s2026")) == "PRETRAIN_FOR_FL"


# ------------------------------------------------------------------------------------------ FL init record
def test_fl_init_record_is_the_endpoint_file_without_reading_values(tmp_path):
    from v4int.pretrain_small import FL_INIT_FILE, fl_init_record
    ck = tmp_path / "sealed" / "checkpoints"
    ck.mkdir(parents=True)
    (ck / "endpoint_6.0.pt").write_bytes(b"endpoint-weights")
    (ck / "best.pt").write_bytes(b"best-weights")
    (tmp_path / "sealed" / "CHECKPOINTS.json").write_text("NOT JSON: must never be read", encoding="utf-8")
    rec = fl_init_record(tmp_path, "PRETRAIN_FOR_FL_SMALL_SASREC_D64_B2_s2026")
    assert rec["fl_init"]["path"] == "sealed/checkpoints/endpoint_6.0.pt"
    assert rec["fl_init"]["sha256"] == sha256_file(ck / "endpoint_6.0.pt") != sha256_file(ck / "best.pt")
    assert json.loads((tmp_path / FL_INIT_FILE).read_text(encoding="utf-8")) == rec


@nc("a run with only a best checkpoint must not yield an FL init record")
def test_nc_fl_init_needs_the_endpoint(tmp_path):
    from v4int.pretrain_small import PretrainRefused, fl_init_record
    ck = tmp_path / "sealed" / "checkpoints"
    ck.mkdir(parents=True)
    (ck / "best.pt").write_bytes(b"best-weights")
    try:
        fl_init_record(tmp_path, "PRETRAIN_FOR_FL_x")
        accepted = True
    except PretrainRefused:
        accepted = False
    assert accepted


# ------------------------------------------------------------------------------------------ launch contract / wiring
ARGV = ["--family", "SASREC", "--seed", "2026", "--phase", "MAIN", "--recipe-id", "SMALL_SASREC_D64_B2", "--gpu", "0",
        "--run-dir", "RUNDIR", "--history-policy", "USER_HISTORY_L50"]


def _args(extra=(), argv=ARGV):
    from v4int.launch_central import parse_args
    return parse_args(list(argv) + list(extra))


def _swap(old, new):
    return [new if x == old else x for x in ARGV]


def test_launch_contract_default_run_id_recipe_and_refusals():
    from v4int.pretrain_small import PretrainRefused, check_launch_args, normalise_argv, recipe_path
    assert check_launch_args(_args()) == "PRETRAIN_FOR_FL_SMALL_SASREC_D64_B2_s2026"
    assert check_launch_args(_args(["--smoke"])) == "SMOKE_NOT_EVIDENCE__PRETRAIN_FOR_FL_SMALL_SASREC_D64_B2_s2026"
    a = _args()
    largv = normalise_argv(list(ARGV), a, check_launch_args(a))
    assert largv[-4:] == ["--run-id", "PRETRAIN_FOR_FL_SMALL_SASREC_D64_B2_s2026", "--tuned-recipes",
                          str(recipe_path())]
    assert recipe_path().name == "SMALL_RECIPE_SASREC_D64_B2.json"
    assert check_launch_args(_args(["--tuned-recipes", str(recipe_path())]))
    for extra in (["--train-manifest", "MAIN_TRAIN_5_100_131072"], ["--eval-manifest", "X"],
                  ["--run-id", "SMALL_SASREC_D64_B2_MAIN_s2026"], ["--calibration-record", "c.json"],
                  ["--transform-state", "t.json"], ["--allow-shared-gpu"], ["--manifest-dir", "m"],
                  ["--peak-lr", "0.001"], ["--tuned-recipes", "other.json"], ["--lock-path", "lock.json"],
                  ["--theta0-variants", "t0.json"], ["--hp-registry", "hp.json"], ["--continuation-budget", "b"]):
        assert_raises(PretrainRefused, check_launch_args, _args(extra))
    for argv in (_swap("SASREC", "GRU"), _swap("SMALL_SASREC_D64_B2", "02B_SASREC"), _swap("2026", "2027"),
                 _swap("USER_HISTORY_L50", "SESSION_L50")):
        assert_raises(PretrainRefused, check_launch_args, _args(argv=argv))


@nc("the C_SMALL run id must be refused by the PRETRAIN launch contract")
def test_nc_c_small_run_id_refused():
    from v4int.pretrain_small import PretrainRefused, check_launch_args
    try:
        check_launch_args(_args(["--run-id", "SMALL_SASREC_D64_B2_MAIN_s2026"]))
        accepted = True
    except PretrainRefused:
        accepted = False
    assert accepted


def test_install_rewires_launch_central(synth_build, tmp_path, monkeypatch):
    import v4int.verify as VF
    from v4int import pretrain_small as PS
    from v4train.registry import load_manifest_registry
    monkeypatch.setattr(VF, "verify_registry", VF.verify_registry)      # restored after the test
    ext_p = tmp_path / "REGISTRY_V2_PRETRAIN_EXT.json"
    ext_p.write_text(json.dumps(_ext(synth_build)), encoding="utf-8")

    class Refused(RuntimeError):
        pass

    base_ctx = {"manifest_registry_path": BASE_PATH, "manifest_registry_rel_path": BASE_REL,
                "manifest_registry_file_sha256": sha256_file(BASE_PATH),
                "manifest_registry_self_sha256": _base()["self_sha256"], "data_version": "CLEAN_V2_POP5100_V2"}
    LC = types.SimpleNamespace(G1_GATE=tmp_path / "gate.json", V4=V4, LaunchRefused=Refused,
                               bind_data_protocol_lock_v2=lambda paths, *, smoke, gate_path=None: dict(base_ctx),
                               build_trainer_registry_view=lambda *a, **k: pytest.fail("original view called"))
    record, labelled = {}, []
    PS.install(LC, ext_p, record, labels=lambda: labelled.append(1) or {"stub": True})
    ctx = LC.bind_data_protocol_lock_v2(None, smoke=True)
    assert Path(ctx["manifest_registry_path"]) == ext_p.resolve() and labelled == [1]
    assert ctx["manifest_registry_self_sha256"] == json.loads(ext_p.read_text())["self_sha256"]
    assert getattr(VF.verify_registry, "__pretrain__", False)           # wrapper 3 installed lazily
    run_dir = tmp_path / "run"
    v = LC.build_trainer_registry_view(ctx["manifest_registry_path"], run_dir,
                                       registry_rel_path=ctx["manifest_registry_rel_path"],
                                       registry_file_sha256=ctx["manifest_registry_file_sha256"])
    assert Path(v["path"]).name == "REGISTRY_V2_TRAINER_VIEW.json"
    assert load_manifest_registry(v["path"]).role("MAIN") == (D10.TRAIN_ID, D10.EVAL_ID)
    assert list(run_dir.glob("PRETRAIN_SHIM_*.json")) and record["ext_checks"]["all_ok"]


def test_install_refuses_a_missing_extension(tmp_path, monkeypatch):
    import v4int.verify as VF
    from v4int import pretrain_small as PS
    monkeypatch.setattr(VF, "verify_registry", VF.verify_registry)

    class Refused(RuntimeError):
        pass

    LC = types.SimpleNamespace(G1_GATE=None, V4=V4, LaunchRefused=Refused,
                               bind_data_protocol_lock_v2=lambda paths, *, smoke, gate_path=None: {
                                   "manifest_registry_path": BASE_PATH, "manifest_registry_rel_path": BASE_REL,
                                   "manifest_registry_file_sha256": sha256_file(BASE_PATH),
                                   "manifest_registry_self_sha256": "s"},
                               build_trainer_registry_view=None)
    PS.install(LC, tmp_path / "absent.json", {}, labels=lambda: pytest.fail("labels before the extension check"))
    assert_raises(Refused, LC.bind_data_protocol_lock_v2, None, smoke=True)


def test_launch_central_order_installs_wrappers_before_the_runconfig():
    """The unedited launcher calls small_recipe.install, then the (wrapped) lock binding, before RunConfig(...): so
    the SMALL recipe loads and wrapper 4 is in force when the config and the trainer are built."""
    from v4int import launch_central as LC
    src = inspect.getsource(LC.run)
    i_small, i_recipe = src.index("_small_install()"), src.index("_recipe_values(args")
    i_bind, i_view, i_cfg = (src.index("bind_data_protocol_lock_v2(paths"), src.index("build_trainer_registry_view("),
                             src.index("cfg = RunConfig("))
    assert i_small < i_recipe < i_bind < i_view < i_cfg < src.index("CentralTrainer(cfg")
    for name in ("_recipe_values", "_theta0_for", "export_environment", "data_paths", "DETERMINISM_MODE", "utc"):
        assert hasattr(LC, name), name


def test_main_refusals_and_fl_init_only(tmp_path, capsys):
    from v4int import pretrain_small as PS
    assert PS.main(["--family", "SASREC"]) == 2                          # no '--'
    assert PS.main(["--"] + _swap("SMALL_SASREC_D64_B2", "02B_SASREC")) == 2
    rd = tmp_path / "run"
    argv = ["--fl-init-only", "--"] + [str(rd) if x == "RUNDIR" else x for x in ARGV]
    assert PS.main(argv) == 2                                           # no endpoint yet
    (rd / "sealed" / "checkpoints").mkdir(parents=True)
    (rd / "sealed" / "checkpoints" / "endpoint_6.0.pt").write_bytes(b"w")
    assert PS.main(argv) == 0 and (rd / PS.FL_INIT_FILE).is_file()
