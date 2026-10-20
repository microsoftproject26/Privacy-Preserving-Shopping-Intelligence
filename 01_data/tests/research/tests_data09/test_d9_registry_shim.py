"""DATA-09 registry extension + loader-gate shim tests: REGISTRY_V2 entries byte-identical in REGISTRY_V2_TEST_EXT (only
the five TEST roles / RETRO_TEST manifests added), O_EXCL writing, role resolution ONLY under the gate, the in-process
wrappers (launch_central bind / view, v4int.verify, v4int.providers, v4train.registry) and the gated verification path
of a RETRO_TEST manifest down to the v4eval EvalManifest - with strict-xfail negative controls.
The real REGISTRY_V2 / DATA_PROTOCOL_LOCK(_V2) files are only READ (copied into pytest's temporary workspace).
Test scaffolding (not product code): the synthetic partial build is re-labelled as a full build (header status = the
TEST role, header content hash, .OK marker and build receipt recomputed) so the registry rule can be exercised."""
from __future__ import annotations

import copy
import json
import shutil
import types
from pathlib import Path

import pytest

from conftest import V4, closed_gate_only, make_gate_files, run_synthetic_build
from data_testkit import assert_raises, nc
from v4data.common import TestGateError, atomic_write_bytes, json_bytes_strict, sha256_file, write_marker
from v4data09 import d9_common as D9
from v4data09 import d9_finalize as F
from v4data09 import d9_registry as R
from v4data09.d9_gate import TestGateRefused, check_test_gate
from v4fs_data import fs_common as FC
from v4int import test_roles as TR

BASE_REL = F.BASE_REL
EXT_REL = F.EXT_REL


def _relabel_full(root: Path, out_rel: str) -> None:
    """Scaffolding: status = role in every RETRO_TEST header (as a full build writes it); marker + receipt re-signed."""
    out = root / out_rel
    rp = out / "receipts" / "DATA09_BUILD_RECEIPT.json"
    rec = json.loads(rp.read_text(encoding="utf-8"))
    for mid in D9.TEST_IDS:
        hp = out / "manifests" / f"{mid}.header.json"
        h = json.loads(hp.read_text(encoding="utf-8"))
        h["status"] = D9.ROLE_OF_ID[mid]
        h.pop("header_content_sha256")
        h["header_content_sha256"] = FC.header_content_sha256(h)
        atomic_write_bytes(out, f"manifests/{mid}.header.json", json_bytes_strict(h))
        files = {f"manifests/{mid}.{t['table']}.parquet": t["file_sha256"] for t in h["tables"]}
        files[f"manifests/{mid}.header.json"] = sha256_file(hp)
        write_marker(out, f"manifests/{mid}.OK", files)
        rec["manifests"][mid] = FC.hashes_entry(h, h["tables"], sha256_file(hp))
    rec["full_build"] = True
    FC.write_json_self_hashed(out, "receipts/DATA09_BUILD_RECEIPT.json", {k: v for k, v in rec.items()
                                                                           if k != "receipt_sha256"})


@pytest.fixture(scope="module")
def ws(synth_ds, gate_files, synth_gate, tmp_path_factory):
    """A temporary ppsi_v4-shaped workspace: copied contracts, the relabelled synthetic RETRO_TEST build, the extension."""
    root = tmp_path_factory.mktemp("d9_ws")
    for rel in (BASE_REL, "contracts/DATA_PROTOCOL_LOCK_V2.json", "contracts/DATA_PROTOCOL_LOCK.json"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(V4 / rel, root / rel)
    res = run_synthetic_build(synth_ds, root / D9.OUT_REL, synth_gate)
    _relabel_full(root, D9.OUT_REL)
    ext = F.build_extension(root, synth_gate, D9.OUT_REL)
    out = F.write_extension(root, ext)
    return {"root": root, "ext": ext, "ext_path": root / EXT_REL, "base_path": root / BASE_REL, "write": out,
            "lock": gate_files[0], "register": gate_files[1], "gate": synth_gate, "build": res}


def _base(ws):
    return json.loads(ws["base_path"].read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------ extension rule
def test_extension_keeps_every_registry_v2_entry_byte_identical(ws):
    base, ext = _base(ws), ws["ext"]
    chk = R.verify_test_ext_registry(ext, base, sha256_file(ws["base_path"]))
    assert chk["all_ok"], {k: v for k, v in chk.items() if not v}
    for k in ("roles", "manifests"):
        for sk, v in base[k].items():
            assert R.canon(ext[k][sk]) == R.canon(v)
    assert set(ext["roles"]) - set(base["roles"]) == set(D9.TEST_ROLES)
    assert set(ext["manifests"]) - set(base["manifests"]) == set(D9.TEST_IDS)
    assert ext["roles"]["MAIN_TEST"] == {"train": "MAIN_TRAIN_5_100_131072", "eval": D9.MAIN_TEST_ID}
    assert ext["roles"]["TRUE_COLD_TEST"] == {"eval": D9.TRUE_COLD_TEST_ID}
    assert ext["extension_test_gate"]["test_exposure_register_entry_id"] == ws["gate"].entry_id
    assert ext["extends"] == dict(ext["extends"], path=BASE_REL, file_sha256=sha256_file(ws["base_path"]),
                                  self_sha256=base["self_sha256"])


def test_real_registry_v2_is_the_lock_v2_pinned_registry():
    from v4data08.d8_finalize import base_pin_checks
    pins = base_pin_checks(V4)
    assert all(v is True for k, v in pins.items() if not k.startswith("_")), pins


def test_write_extension_is_o_excl_and_never_overwrites(ws):
    again = F.write_extension(ws["root"], ws["ext"])                 # identical bytes: accepted, unchanged
    assert again["file_sha256"] == ws["write"]["file_sha256"] and again["checks"]["all_ok"]
    other = R.with_self(dict(ws["ext"], extension_task="DATA-09b"))
    assert_raises(SystemExit, F.write_extension, ws["root"], other)
    assert sha256_file(ws["ext_path"]) == ws["write"]["file_sha256"]


def test_build_extension_refuses_under_another_gate(ws, tmp_path):
    other = check_test_gate(*make_gate_files(tmp_path, lock_text='{"document": "ALL_METHODS_LOCK", "v": "other"}'))
    assert_raises(SystemExit, F.build_extension, ws["root"], other, D9.OUT_REL)


def test_partial_entry_cannot_enter_the_extension(ws):
    e = {m: dict(ws["ext"]["manifests"][m]) for m in D9.TEST_IDS}
    e[D9.MAIN_TEST_ID]["status"] = "PARTIAL_TEST_BUILD"
    assert_raises(R.ExtRegistryError, R.build_test_ext_registry, _base(ws), base_rel_path=BASE_REL,
                  base_file_sha256=sha256_file(ws["base_path"]), entries=e, gate_binding=ws["gate"].binding(),
                  build_receipt={}, code={})


@nc("a REGISTRY_V2 entry altered inside the extension (re-signed) must fail byte-identity")
def test_nc_tampered_base_entry(ws):
    ext = copy.deepcopy(ws["ext"])
    ext["manifests"]["MAIN_VALIDATION_5_100_131072"]["n_eval_rows"] += 1
    assert R.verify_test_ext_registry(R.with_self(ext), _base(ws), sha256_file(ws["base_path"]))["all_ok"]


@nc("MAIN_TEST pointing at a VALIDATION manifest must fail the extension rule")
def test_nc_main_test_points_at_validation(ws):
    ext = copy.deepcopy(ws["ext"])
    ext["roles"]["MAIN_TEST"]["eval"] = "MAIN_VALIDATION_5_100_131072"
    assert R.verify_test_ext_registry(R.with_self(ext), _base(ws), sha256_file(ws["base_path"]))["all_ok"]


@nc("an extra, unregistered top-level key must fail the extension rule")
def test_nc_extra_top_key(ws):
    ext = dict(ws["ext"], surprise=1)
    assert R.verify_test_ext_registry(R.with_self(ext), _base(ws), sha256_file(ws["base_path"]))["all_ok"]


@nc("a RETRO_TEST entry whose gate fields differ from the extension's gate must fail")
def test_nc_entry_gate_differs(ws):
    ext = copy.deepcopy(ws["ext"])
    ext["manifests"][D9.HEAVY_TEST_ID]["test_exposure_register_entry_id"] = 99
    assert R.verify_test_ext_registry(R.with_self(ext), _base(ws), sha256_file(ws["base_path"]))["all_ok"]


@nc("an extension naming another base file sha256 must fail")
def test_nc_wrong_base_file_sha(ws):
    assert R.verify_test_ext_registry(ws["ext"], _base(ws), "0" * 64)["all_ok"]


# ------------------------------------------------------------------------------------------ resolution under the gate
@closed_gate_only
def test_resolve_role_only_under_the_gate(ws):
    kw = dict(ext_path=ws["ext_path"], base_path=ws["base_path"])
    r = TR.resolve_role("MAIN_TEST", lock_path=ws["lock"], register_path=ws["register"], **kw)
    assert (r["train"], r["eval"]) == ("MAIN_TRAIN_5_100_131072", D9.MAIN_TEST_ID) and r["gate"].entry_id == 2
    assert TR.resolve_role("TRUE_COLD_TEST", lock_path=ws["lock"], register_path=ws["register"], **kw)["train"] is None
    assert_raises(TestGateRefused, TR.resolve_role, "MAIN_TEST", **kw)                 # the real gate: closed
    assert_raises(TestGateRefused, TR.resolve_role, "OTHER_TEST", lock_path=ws["lock"], register_path=ws["register"],
                  **kw)
    m = TR.resolve_role("MAIN", **kw)                                                    # non-TEST: no gate needed
    assert (m["train"], m["eval"], m["gate"]) == ("MAIN_TRAIN_5_100_131072", "MAIN_VALIDATION_5_100_131072", None)


@closed_gate_only
def test_resolution_refuses_before_opening_the_extension_when_the_gate_is_closed(tmp_path):
    try:
        TR.resolve_role("MAIN_TEST", ext_path=tmp_path / "absent.json")
        raise AssertionError("resolved")
    except TestGateRefused as e:
        assert "ALL_METHODS_LOCK" in str(e)                  # the gate refused, not the missing extension


@closed_gate_only
def test_test_gate_ok_hook(ws, tmp_path):
    assert TR.test_gate_ok("MAIN") is True and TR.test_gate_ok("UNSEEN_HOLDOUT") is True
    assert TR.test_gate_ok("MAIN_TEST") is False                                         # real gate closed
    kw = dict(lock_path=ws["lock"], register_path=ws["register"])
    assert TR.test_gate_ok("MAIN_TEST", **kw) is True
    assert TR.test_gate_ok("MAIN_TEST", ext_path=ws["ext_path"], **kw) is True
    assert TR.test_gate_ok("NOT_A_ROLE_TEST", **kw) is False
    lock2, reg2 = make_gate_files(tmp_path, lock_text='{"document": "ALL_METHODS_LOCK", "v": 3}')
    assert TR.test_gate_ok("MAIN_TEST", lock_path=lock2, register_path=reg2, ext_path=ws["ext_path"]) is False


@nc("resolving a TEST role with the real (closed) gate must refuse")
@closed_gate_only
def test_nc_resolve_without_gate(ws):
    try:
        TR.resolve_role("MAIN_TEST", ext_path=ws["ext_path"], base_path=ws["base_path"])
        resolved = True
    except TestGateError:
        resolved = False
    assert resolved


# ------------------------------------------------------------------------------------------ wrappers
@pytest.fixture
def lc_stub(ws, monkeypatch):
    import v4int.providers as PV
    import v4int.verify as VF
    import v4train.registry as TRG
    for mod, name in ((VF, "verify_registry"), (VF, "verify_manifest_by_registry"), (PV, "verify_manifest_by_registry"),
                      (PV, "refuse_test_manifest"), (PV, "build_eval_manifest"), (TRG, "load_manifest_registry")):
        monkeypatch.setattr(mod, name, getattr(mod, name))                   # restored after the test

    class Refused(RuntimeError):
        pass

    base_ctx = {"manifest_registry_path": ws["base_path"], "manifest_registry_rel_path": BASE_REL,
                "manifest_registry_file_sha256": sha256_file(ws["base_path"]),
                "manifest_registry_self_sha256": _base(ws)["self_sha256"], "data_version": "CLEAN_V2_POP5100_V2"}
    LC = types.SimpleNamespace(G1_GATE=None, V4=ws["root"], LaunchRefused=Refused,
                               bind_data_protocol_lock_v2=lambda paths, *, smoke, gate_path=None: dict(base_ctx),
                               build_trainer_registry_view=lambda *a, **k: {"original_view": True})
    yield LC
    TR.uninstall()


def test_install_rewires_launch_central_under_the_gate(ws, lc_stub, tmp_path):
    from v4train.provider import expected_manifests
    import v4train.registry as TRG
    LC = lc_stub
    TR.install(ws["ext_path"], role="MAIN_TEST", lock_path=ws["lock"], register_path=ws["register"], LC=LC, eager=False)
    ctx = LC.bind_data_protocol_lock_v2(None, smoke=True)
    assert Path(ctx["manifest_registry_path"]) == ws["ext_path"].resolve() and ctx["test_role"] == "MAIN_TEST"
    assert ctx["manifest_registry_self_sha256"] == ws["ext"]["self_sha256"]
    assert getattr(TRG.load_manifest_registry, TR.MARK, False)               # W3-W5 installed lazily by W1
    v = LC.build_trainer_registry_view(ctx["manifest_registry_path"], tmp_path / "run",
                                       registry_rel_path=ctx["manifest_registry_rel_path"],
                                       registry_file_sha256=ctx["manifest_registry_file_sha256"])
    assert Path(v["path"]).name == R.VIEW_FILE and v["test_role"] == "MAIN_TEST"
    want = ("MAIN_TRAIN_5_100_131072", D9.MAIN_TEST_ID)
    assert expected_manifests("COHORT", "MAIN", registry=str(v["path"])) == want
    assert TRG.load_manifest_registry(v["path"]).role("CONTINUATION") == want
    orig = TR._orig(TRG, "load_manifest_registry")                            # the unedited trainer loader refuses
    assert_raises(TRG.RegistryError, orig, v["path"])
    other = LC.build_trainer_registry_view(ws["base_path"], tmp_path / "run2", registry_rel_path=BASE_REL,
                                           registry_file_sha256="x")
    assert other == {"original_view": True}                                   # any other registry: unchanged


def test_wrappers_refuse_when_the_gate_closes(ws, lc_stub, tmp_path):
    from v4train.provider import expected_manifests
    lock2 = tmp_path / "ALL_METHODS_LOCK.json"
    shutil.copyfile(ws["lock"], lock2)
    reg2 = tmp_path / "REG.json"
    shutil.copyfile(ws["register"], reg2)
    LC = lc_stub
    TR.install(ws["ext_path"], role="MAIN_TEST", lock_path=lock2, register_path=reg2, LC=LC, eager=True)
    ctx = LC.bind_data_protocol_lock_v2(None, smoke=True)
    v = LC.build_trainer_registry_view(ctx["manifest_registry_path"], tmp_path / "run",
                                       registry_rel_path=ctx["manifest_registry_rel_path"],
                                       registry_file_sha256=ctx["manifest_registry_file_sha256"])
    lock2.unlink()                                                            # the gate closes
    assert_raises(TestGateError, expected_manifests, "COHORT", "MAIN", registry=str(v["path"]))
    assert_raises(LC.LaunchRefused, LC.bind_data_protocol_lock_v2, None, smoke=True)


def test_eval_only_role_gets_no_trainer_view(ws, lc_stub, tmp_path):
    LC = lc_stub
    TR.install(ws["ext_path"], role="TRUE_COLD_TEST", lock_path=ws["lock"], register_path=ws["register"], LC=LC)
    ctx = LC.bind_data_protocol_lock_v2(None, smoke=True)
    assert_raises(LC.LaunchRefused, LC.build_trainer_registry_view, ctx["manifest_registry_path"], tmp_path,
                  registry_rel_path=ctx["manifest_registry_rel_path"],
                  registry_file_sha256=ctx["manifest_registry_file_sha256"])


def test_provider_names_route_only_registered_test_ids(ws, lc_stub, monkeypatch):
    import v4int.providers as PV
    import v4int.verify as VF
    calls = []
    monkeypatch.setattr(VF, "verify_manifest_by_registry", lambda *a, **k: calls.append(("VF", a[0])) or "ORIG")
    monkeypatch.setattr(PV, "verify_manifest_by_registry", lambda *a, **k: calls.append(("PV", a[0])) or "ORIG")
    monkeypatch.setattr(TR, "verify_test_manifest", lambda mid, **k: calls.append(("TR", mid)) or "GATED")
    TR.install(ws["ext_path"], role="MAIN_TEST", lock_path=ws["lock"], register_path=ws["register"], LC=lc_stub)
    assert PV.verify_manifest_by_registry("MAIN_TRAIN_5_100_131072", ws["ext_path"]) == "ORIG"
    assert PV.verify_manifest_by_registry(D9.MAIN_TEST_ID, ws["ext_path"]) == "GATED"
    assert VF.verify_manifest_by_registry(D9.UNEW_TEST_ID, ws["ext_path"]) == "GATED"
    assert_raises(TestGateRefused, PV.verify_manifest_by_registry, D9.MAIN_TEST_ID, ws["base_path"])
    assert calls == [("PV", "MAIN_TRAIN_5_100_131072"), ("TR", D9.MAIN_TEST_ID), ("TR", D9.UNEW_TEST_ID)]
    PV.refuse_test_manifest(D9.MAIN_TEST_ID)                                  # gate open, registered id: served
    assert_raises(TestGateError, PV.refuse_test_manifest, "MAIN_TEST_SOMETHING_ELSE")
    assert_raises(TestGateError, PV.refuse_test_manifest, "MAIN_VALIDATION_5_100_131072",
                  header={"all_methods_lock_sha256": "x"})                    # the original header rule still applies


# ------------------------------------------------------------------------------------------ gated verification path
@pytest.fixture
def verified(ws, monkeypatch):
    """verify_test_manifest over the synthetic workspace. The base verification (REGISTRY_V2 under LOCK_V2 + the B1
    v1 lock) and the TRANSFORM_STATE check are stubbed with the REAL registry / lock objects, except the two catalogue
    values the synthetic dataset cannot share (candidate_order_hash, K); everything TEST-specific runs unstubbed."""
    import v4int.verify as VF
    base = _base(ws)
    v1 = json.loads((V4 / "contracts" / "DATA_PROTOCOL_LOCK.json").read_text(encoding="utf-8"))
    h = json.loads((ws["root"] / D9.OUT_REL / "manifests" / f"{D9.MAIN_TEST_ID}.header.json").read_text(encoding="utf-8"))
    v1s = copy.deepcopy(v1)
    v1s["catalogue"]["K"] = h["catalogue_K"]
    v1s["population_and_manifests"]["manifest_hashes"]["candidate_order_hash"] = h["candidate_order_hash"]
    monkeypatch.setattr(VF, "verify_registry", lambda p, ls, lf, *, lock=None, root=None: (
        base, sha256_file(ws["base_path"]), base["self_sha256"], (v1s, FC.LOCK_SELF_SHA256, FC.LOCK_FILE_SHA256)))
    ts = base["transform_state"]
    monkeypatch.setattr(VF, "verify_transform_state", lambda paths, lock: (ts["transform_state_sha256"],
                                                                           ts["file_sha256"]))
    return dict(ext_path=ws["ext_path"], root=ws["root"], lock_path=ws["lock"], register_path=ws["register"])


@pytest.mark.parametrize("mid", D9.TEST_IDS)
def test_verify_test_manifest_and_eval_manifest(ws, verified, mid):
    vm = TR.verify_test_manifest(mid, **verified)
    assert all(v is True for v in vm.checks.values()) and len(vm.checks) >= 40
    assert vm.kind == "EVAL" and vm.registry["test_gate"]["test_exposure_register_entry_id"] == 2
    em = TR.test_eval_manifest(vm, lock_path=verified["lock_path"], register_path=verified["register_path"])
    assert em.n == vm.canonical_decision_id.size == ws["ext"]["manifests"][mid]["n_decisions"]
    assert int(em.rankable.sum()) == ws["ext"]["manifests"][mid]["n_rankable_rows"]


def test_verify_test_manifest_refuses_without_the_gate_and_for_foreign_ids(ws, verified):
    kw = dict(verified)
    assert_raises(TestGateRefused, TR.verify_test_manifest, D9.MAIN_TEST_ID,
                  **dict(kw, lock_path=None, register_path=None))
    assert_raises(TestGateRefused, TR.verify_test_manifest, "MAIN_VALIDATION_5_100_131072", **kw)


def test_v4eval_refuses_a_retro_test_header_without_the_lock_path(ws, verified):
    from v4eval.errors import LockGateError
    from v4eval.manifest import build_eval_manifest
    from v4int.providers import EVAL_MANIFEST_COLUMNS
    vm = TR.verify_test_manifest(D9.MAIN_TEST_ID, **verified)
    cols = {c: vm.columns[c] for c in EVAL_MANIFEST_COLUMNS if c in vm.columns}
    assert_raises(LockGateError, build_eval_manifest, vm.header, cols)                     # v4eval's own E10.1 gate
    em = build_eval_manifest(vm.header, cols, all_methods_lock_path=verified["lock_path"])
    assert em.n == vm.canonical_decision_id.size


def test_build_eval_manifest_wrapper_hands_v4eval_the_lock(ws, verified, lc_stub):
    import v4int.providers as PV
    from v4int.providers import EVAL_MANIFEST_COLUMNS
    vm = TR.verify_test_manifest(D9.SPARSE_TEST_ID, **verified)
    cols = {c: vm.columns[c] for c in EVAL_MANIFEST_COLUMNS if c in vm.columns}
    TR.install(ws["ext_path"], role="SPARSE_PANEL_TEST", lock_path=ws["lock"], register_path=ws["register"],
               LC=lc_stub)
    em = PV.build_eval_manifest(vm.header, cols)
    assert em.n == vm.canonical_decision_id.size and em.header["v4_split"] == "RETRO_TEST"
    bad = dict(vm.header, test_exposure_register_entry_id=7)
    assert_raises(TestGateRefused, PV.build_eval_manifest, bad, cols)


@nc("a RETRO_TEST table file edited on disk must fail the gated verification")
def test_nc_tampered_table_fails_verification(ws, verified, tmp_path):
    import v4int.verify as VF
    root2 = tmp_path / "ws2"
    shutil.copytree(ws["root"], root2)
    f = root2 / D9.OUT_REL / "manifests" / f"{D9.HEAVY_TEST_ID}.decisions.parquet"
    f.write_bytes(f.read_bytes() + b"\0")
    try:
        TR.verify_test_manifest(D9.HEAVY_TEST_ID, **dict(verified, root=root2, ext_path=root2 / EXT_REL))
        ok = True
    except VF.ManifestVerificationError:
        ok = False
    assert ok


@nc("a RETRO_TEST manifest built under another gate must not be served")
def test_nc_foreign_gate_manifest_not_served(ws, verified, tmp_path):
    lock2, reg2 = make_gate_files(tmp_path, lock_text='{"document": "ALL_METHODS_LOCK", "v": 4}')
    try:
        TR.verify_test_manifest(D9.MAIN_TEST_ID, **dict(verified, lock_path=lock2, register_path=reg2))
        ok = True
    except TestGateError:
        ok = False
    assert ok


@closed_gate_only
def test_reeval_hook_contract():
    assert callable(TR.test_gate_ok) and TR.test_gate_ok("MAIN") and not TR.test_gate_ok("MAIN_TEST")
