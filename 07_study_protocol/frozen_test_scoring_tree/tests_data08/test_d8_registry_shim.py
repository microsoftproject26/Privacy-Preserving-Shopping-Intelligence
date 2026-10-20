"""DATA-08 registry extension + C_FULL launcher shim tests: REGISTRY_V2 entries byte-identical in the extension, the
trainer view (the unedited v4train loader resolves role MAIN to the C_FULL pair), the v4int.verify wrapper routing, the
launch-argument contract and the launch_central wiring - with strict-xfail negative controls. The real REGISTRY_V2 /
DATA_PROTOCOL_LOCK_V2 files are only READ (copied into pytest's temporary directory where a write is needed)."""
from __future__ import annotations

import copy
import json
import shutil
import types
from pathlib import Path

import pytest

from conftest import V4
from data_testkit import assert_raises, nc
from v4data.common import sha256_file
from v4data08 import d8_common as D8
from v4data08 import d8_finalize as F
from v4data08 import d8_registry as R

BASE_REL = F.BASE_REL
BASE_PATH = V4 / BASE_REL


def _base():
    return json.loads(BASE_PATH.read_text(encoding="utf-8"))


def _entry(synth_build, status=D8.ROLE):
    e = F.registry_entry(synth_build["out"].parent, synth_build["out"].name)
    e = dict(e, status=status, role=status)
    return e


_SYNTH_EXPECT: dict = {}


@pytest.fixture(autouse=True)
def _synthetic_population_expectation(monkeypatch):
    """The synthetic builds carry a toy population; C-F1's TIEBREAK content check is pointed at that toy population
    while `_SYNTH_EXPECT` is set (the tests that prove the REAL values are enforced clear it)."""
    _SYNTH_EXPECT.clear()
    real = R.expected_population
    monkeypatch.setattr(R, "expected_population", lambda: dict(_SYNTH_EXPECT) if _SYNTH_EXPECT else real())
    yield
    _SYNTH_EXPECT.clear()


def _ext(synth_build, base=None, **kw):
    base = _base() if base is None else base
    entry = _entry(synth_build, **kw)
    _SYNTH_EXPECT.update(rank_range=[0, 8], n_users=entry["n_users"], n_decisions=entry["n_decisions"],
                         population_sha256_selection_key_v2=entry["population_sha256_selection_key_v2"])
    return R.build_ext_registry(base, base_rel_path=BASE_REL, base_file_sha256=sha256_file(BASE_PATH),
                                entry=entry, band={"rank_range": [0, 8]},
                                build_receipt={"receipt_sha256": "x"}, code={"d8_code_sha256": "y"})


def _resign(d):
    return R.with_self(d)


# ------------------------------------------------------------------------------------------ extension rule
def test_real_registry_v2_is_the_lock_v2_pinned_registry():
    pins = F.base_pin_checks(V4)
    assert all(v is True for k, v in pins.items() if not k.startswith("_")), pins


def test_extension_keeps_every_registry_v2_entry_byte_identical(synth_build):
    base = _base()
    ext = _ext(synth_build, base)
    chk = R.verify_ext_registry(ext, base, sha256_file(BASE_PATH))
    assert chk["all_ok"], {k: v for k, v in chk.items() if not v}
    for k in ("roles", "manifests", "population_bands"):
        for sk, v in base[k].items():
            assert R.canon(ext[k][sk]) == R.canon(v)
    assert ext["roles"]["C_FULL"] == {"train": D8.TRAIN_ID, "eval": "MAIN_VALIDATION_5_100_131072"}
    assert ext["extends"]["self_sha256"] == base["self_sha256"] and ext["data_version"] == base["data_version"]


def test_partial_build_entry_cannot_enter_the_registry(synth_build):
    assert_raises(R.ExtRegistryError, _ext, synth_build, status="PARTIAL_TEST_BUILD")


def test_write_extension_rereads_and_refuses_overwrite(synth_build, tmp_path):
    (tmp_path / "contracts" / "manifest_registry").mkdir(parents=True)
    shutil.copyfile(BASE_PATH, tmp_path / BASE_REL)
    ext = _ext(synth_build)
    out = F.write_extension(tmp_path, ext)
    assert out["checks"]["all_ok"] and out["self_sha256"] == ext["self_sha256"]
    assert F.write_extension(tmp_path, ext)["file_sha256"] == out["file_sha256"]      # identical bytes: idempotent
    other = _resign(dict(ext, extension_task="DATA-08b"))
    assert_raises(SystemExit, F.write_extension, tmp_path, other)


@nc("a REGISTRY_V2 entry altered inside the extension (re-signed) must fail byte-identity")
def test_nc_tampered_base_entry(synth_build):
    base = _base()
    ext = copy.deepcopy(_ext(synth_build, base))
    ext["manifests"]["MAIN_TRAIN_5_100_131072"]["n_decisions"] += 1
    assert R.verify_ext_registry(_resign(ext), base, sha256_file(BASE_PATH))["all_ok"]


@nc("a C_FULL role pointing at another eval manifest must fail the extension rule")
def test_nc_wrong_c_full_eval(synth_build):
    base = _base()
    ext = copy.deepcopy(_ext(synth_build, base))
    ext["roles"]["C_FULL"]["eval"] = "CALIB_VALIDATION_5_100_32768"
    assert R.verify_ext_registry(_resign(ext), base, sha256_file(BASE_PATH))["all_ok"]


@nc("an extra, unregistered top-level key must fail the extension rule")
def test_nc_extra_top_key(synth_build):
    base = _base()
    ext = dict(_ext(synth_build, base), surprise=1)
    assert R.verify_ext_registry(_resign(ext), base, sha256_file(BASE_PATH))["all_ok"]


@nc("an extension that names another base file sha256 must fail")
def test_nc_wrong_base_file_sha(synth_build):
    base = _base()
    assert R.verify_ext_registry(_ext(synth_build, base), base, "0" * 64)["all_ok"]


def test_population_content_checks_use_the_locked_tiebreak_values(synth_build):
    """C-F1: with the REAL expectation, a well-formed but toy-population extension is refused on every content check;
    with matching values it passes (the checks are the locked TB rank range / sha / n_users / n_decisions)."""
    base = _base()
    ext = _ext(synth_build, base)
    _SYNTH_EXPECT.clear()                                          # -> real D8 values
    chk = R.verify_ext_registry(ext, base, sha256_file(BASE_PATH))
    assert not chk["all_ok"]
    for k in ("population_bands.TIEBREAK.rank_range", "manifests.C_FULL_train.n_users",
              "manifests.C_FULL_train.n_decisions", "manifests.C_FULL_train.population_sha256_selection_key_v2"):
        assert chk[k] is False, k
    exp = R.expected_population()
    assert exp["rank_range"] == list(D8.BANDS["TIEBREAK"]) and exp["n_users"] == 1_348_682
    assert exp["n_decisions"] == 26_677_600 and exp["population_sha256_selection_key_v2"] == D8.TB_SHA_V2


@nc("a TIEBREAK band whose rank_range differs from BANDS['TIEBREAK'] must fail the extension rule")
def test_nc_wrong_band_rank_range(synth_build):
    base = _base()
    ext = copy.deepcopy(_ext(synth_build, base))
    ext["population_bands"]["TIEBREAK"]["rank_range"] = [0, 9]
    assert R.verify_ext_registry(_resign(ext), base, sha256_file(BASE_PATH))["all_ok"]


@nc("a C_FULL entry with a wrong n_decisions must fail the extension rule")
def test_nc_wrong_entry_n_decisions(synth_build):
    base = _base()
    ext = copy.deepcopy(_ext(synth_build, base))
    ext["manifests"][D8.TRAIN_ID]["n_decisions"] += 1
    assert R.verify_ext_registry(_resign(ext), base, sha256_file(BASE_PATH))["all_ok"]


# ------------------------------------------------------------------------------------------ trainer view
def test_trainer_view_resolves_main_to_the_c_full_pair(synth_build, tmp_path):
    from v4train.provider import expected_manifests
    from v4train.registry import load_manifest_registry
    ext = _ext(synth_build)
    view = R.trainer_view(ext, ext_rel_path="contracts/manifest_registry/REGISTRY_V2_FULL_EXT.json",
                          ext_file_sha256="a" * 64)
    assert R.check_trainer_view(view, ext)["all_ok"]
    p = tmp_path / R.VIEW_FILE
    p.write_text(json.dumps(view, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    reg = load_manifest_registry(p)                             # the trainer's own, unedited loader
    assert reg.role("MAIN") == reg.role("CONTINUATION") == (D8.TRAIN_ID, D8.EVAL_ID)
    assert expected_manifests("COHORT", "MAIN", registry=str(p)) == (D8.TRAIN_ID, D8.EVAL_ID)


@nc("the launcher's standard REGISTRY_V2 view (MAIN -> MAIN_TRAIN) is not a C_FULL view")
def test_nc_standard_view_is_not_c_full(synth_build, tmp_path):
    from v4int.launch_central import build_trainer_registry_view
    ext = _ext(synth_build)
    v = build_trainer_registry_view(BASE_PATH, tmp_path, registry_rel_path=BASE_REL,
                                    registry_file_sha256=sha256_file(BASE_PATH))
    view = json.loads(Path(v["path"]).read_text(encoding="utf-8"))
    assert R.check_trainer_view(view, ext)["all_ok"]


# ------------------------------------------------------------------------------------------ verify wrapper
def _stub_orig(calls):
    def orig(registry_path, lock_self, lock_file, *, lock=None, root=None):
        calls.append(Path(registry_path))
        reg = json.loads(Path(registry_path).read_text(encoding="utf-8"))
        return reg, sha256_file(registry_path), reg["self_sha256"], "V1CTX"
    return orig


def test_verify_wrapper_routes_extension_through_the_base(synth_build, tmp_path):
    from v4int.c_full import make_verify_wrapper
    ext_p = tmp_path / "EXT.json"
    ext_p.write_text(json.dumps(_ext(synth_build)), encoding="utf-8")
    calls = []
    w = make_verify_wrapper(_stub_orig(calls), ext_p, BASE_PATH)
    reg, fsha, self_, ctx = w(ext_p, "L", "F")
    assert calls == [BASE_PATH] and reg["roles"]["C_FULL"] and fsha == sha256_file(ext_p) and ctx == "V1CTX"
    assert self_ == reg["self_sha256"]
    reg2 = w(BASE_PATH, "L", "F")[0]                            # any other path: delegated unchanged
    assert calls[-1] == BASE_PATH and "C_FULL" not in reg2["roles"]


@nc("the verify wrapper must refuse a tampered extension")
def test_nc_verify_wrapper_refuses_tampered(synth_build, tmp_path):
    from v4int.c_full import make_verify_wrapper
    ext = copy.deepcopy(_ext(synth_build))
    ext["manifests"]["MAIN_VALIDATION_5_100_131072"]["n_eval_rows"] -= 1
    ext_p = tmp_path / "EXT.json"
    ext_p.write_text(json.dumps(_resign(ext)), encoding="utf-8")
    w = make_verify_wrapper(_stub_orig([]), ext_p, BASE_PATH)
    try:
        w(ext_p, "L", "F")
        accepted = True
    except ValueError:
        accepted = False
    assert accepted


# ------------------------------------------------------------------------------------------ launch contract / wiring
ARGV = ["--family", "SASREC", "--seed", "2026", "--phase", "MAIN", "--recipe-id", "02B_SASREC", "--gpu", "0",
        "--run-dir", "RUNDIR", "--history-policy", "USER_HISTORY_L50"]


def _args(extra=()):
    from v4int.launch_central import parse_args
    return parse_args(ARGV + list(extra))


def test_launch_contract_default_run_id_and_refusals():
    from v4int.c_full import CFullRefused, check_launch_args
    assert check_launch_args(_args()) == "C_FULL_02B_SASREC_s2026"
    assert check_launch_args(_args(["--smoke"])) == "SMOKE_NOT_EVIDENCE__C_FULL_02B_SASREC_s2026"
    for extra in (["--train-manifest", "MAIN_TRAIN_5_100_131072"], ["--eval-manifest", "X"],
                  ["--run-id", "02B_SASREC_MAIN_s2026"], ["--calibration-record", "c.json"],
                  ["--transform-state", "t.json"], ["--allow-shared-gpu"], ["--manifest-dir", "m"],
                  ["--manifest-hashes", "h.json"], ["--header-list", "l.json"], ["--lock-path", "lock.json"],
                  ["--theta0-variants", "t0.json"], ["--hp-registry", "hp.json"]):
        assert_raises(CFullRefused, check_launch_args, _args(extra))
    from v4int.launch_central import parse_args
    gru = parse_args([x if x != "SASREC" else "GRU" for x in ARGV])
    assert_raises(CFullRefused, check_launch_args, gru)
    other = parse_args([x if x != "02B_SASREC" else "02B_SASREC_C2" for x in ARGV])
    assert_raises(CFullRefused, check_launch_args, other)


@nc("the C_BIG run id must be refused by the C_FULL launch contract")
def test_nc_cbig_run_id_refused():
    from v4int.c_full import CFullRefused, check_launch_args
    try:
        check_launch_args(_args(["--run-id", "02B_SASREC_MAIN_s2026"]))
        accepted = True
    except CFullRefused:
        accepted = False
    assert accepted


def test_install_rewires_launch_central(synth_build, tmp_path, monkeypatch):
    import v4int.verify as VF
    from v4int import c_full as CF
    from v4train.registry import load_manifest_registry
    monkeypatch.setattr(VF, "verify_registry", VF.verify_registry)      # restored after the test
    ext_p = tmp_path / "REGISTRY_V2_FULL_EXT.json"
    ext_p.write_text(json.dumps(_ext(synth_build)), encoding="utf-8")

    class Refused(RuntimeError):
        pass

    base_ctx = {"manifest_registry_path": BASE_PATH, "manifest_registry_rel_path": BASE_REL,
                "manifest_registry_file_sha256": sha256_file(BASE_PATH),
                "manifest_registry_self_sha256": _base()["self_sha256"], "data_version": "CLEAN_V2_POP5100_V2"}
    LC = types.SimpleNamespace(G1_GATE=tmp_path / "gate.json", V4=V4, LaunchRefused=Refused,
                               bind_data_protocol_lock_v2=lambda paths, *, smoke, gate_path=None: dict(base_ctx),
                               build_trainer_registry_view=lambda *a, **k: pytest.fail("original view called"))
    record = {}
    CF.install(LC, ext_p, record)
    ctx = LC.bind_data_protocol_lock_v2(None, smoke=True)
    assert Path(ctx["manifest_registry_path"]) == ext_p.resolve()
    assert ctx["manifest_registry_self_sha256"] == json.loads(ext_p.read_text())["self_sha256"]
    assert getattr(VF.verify_registry, "__c_full__", False)             # wrapper 3 installed lazily
    run_dir = tmp_path / "run"
    v = LC.build_trainer_registry_view(ctx["manifest_registry_path"], run_dir,
                                       registry_rel_path=ctx["manifest_registry_rel_path"],
                                       registry_file_sha256=ctx["manifest_registry_file_sha256"])
    assert Path(v["path"]).name == "REGISTRY_V2_TRAINER_VIEW.json"
    assert load_manifest_registry(v["path"]).role("MAIN") == (D8.TRAIN_ID, D8.EVAL_ID)
    assert list(run_dir.glob("C_FULL_SHIM_*.json")) and record["ext_checks"]["all_ok"]


def test_install_refuses_a_missing_extension(tmp_path, monkeypatch):
    import v4int.verify as VF
    from v4int import c_full as CF
    monkeypatch.setattr(VF, "verify_registry", VF.verify_registry)

    class Refused(RuntimeError):
        pass

    LC = types.SimpleNamespace(G1_GATE=None, V4=V4, LaunchRefused=Refused,
                               bind_data_protocol_lock_v2=lambda paths, *, smoke, gate_path=None: {
                                   "manifest_registry_path": BASE_PATH, "manifest_registry_rel_path": BASE_REL,
                                   "manifest_registry_file_sha256": sha256_file(BASE_PATH),
                                   "manifest_registry_self_sha256": "s"},
                               build_trainer_registry_view=None)
    CF.install(LC, tmp_path / "absent.json", {})
    assert_raises(Refused, LC.bind_data_protocol_lock_v2, None, smoke=True)
