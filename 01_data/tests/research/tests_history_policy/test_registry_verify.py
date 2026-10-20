"""The manifest-registry verification path (v4int.verify.verify_manifest_by_registry; Ops / SCHEMA-01 finding): a
manifest that DATA_PROTOCOL_LOCK does not list (FS-01's FEATURE_SEARCH_SCREEN manifests, DATA-06's v2 bands) is served
when a registry verifies it. The lock path for M01-M06 is unchanged (tests_int covers it).

* A registry-verified FS fixture (real bucket-0 rows, FS-01's registry shape, the real lock and Clean V2 state) passes;
  both providers build through it and their manifest_binding equals the registry entry.
* Refused: an edited registry (self-hash), a tampered header file, a header status that is neither
  FEATURE_SEARCH_SCREEN nor a registry role, another data_version, another transform state.
* On a host with FS-01's real files (the server): the two FS manifests verify through REGISTRY_FS.
* Negative controls (each must fail the 'registry accepts it' check): a tampered content hash, a wrong lock hash, a TEST
  id.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from histkit import REAL_DS, Q, assert_raises, nc
from v4int.verify import (DataPaths, ManifestVerificationError, TestManifestRefused, manifest_content_sha256,
                          registry_self_sha256, verify_lock, verify_manifest_by_registry)
import ppsi_paths

V4 = ppsi_paths.PROJECT_ROOT.resolve()       # the ppsi_v4 workspace (contracts, data)
REG_FS = V4 / "contracts" / "manifest_registry" / "REGISTRY_FS.json"
TR, EV = "FSX_TRAIN_FIXTURE", "FSX_VALIDATION_FIXTURE"
USER = "USER_HISTORY_L50"


def _paths() -> DataPaths:
    """The REAL v1 lock (HISTORICAL; INT-02 B2 made DataPaths.default() bind DATA_PROTOCOL_LOCK_V2, but REGISTRY_FS
    and every registry this module tests are bound to the v1 lock directly — the explicit historical route, per
    INT-02R1 guidance) and the locked TRANSFORM_STATE (a byte copy on this host); the manifest fields are unused here."""
    from intkit import transform_state_source
    d = DataPaths.historical()
    ts = d.transform_state_path if d.transform_state_path.is_file() else transform_state_source()
    return DataPaths(manifest_dir=d.manifest_dir, lock_path=d.lock_path, manifest_hashes_path=d.manifest_hashes_path,
                     header_list_path=d.header_list_path, transform_state_path=ts, production=False)


def _write_registry(path: Path, reg: dict) -> Path:
    reg = {k: v for k, v in reg.items() if k != "self_sha256"}
    reg["self_sha256"] = registry_self_sha256(reg)
    path.write_text(json.dumps(reg, indent=1), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def fsx(tmp_path_factory, real_train_store, real_val_store):
    """FS-01's layout under a temporary workspace root: data_fs/manifests/<id>.{tables,header,OK} + a registry."""
    from intkit import real_eval_tables, real_train_tables, write_manifest_files, write_r1_and_markers
    root = tmp_path_factory.mktemp("fsx_root")
    d2 = root / "data_fs"
    (d2 / "receipts").mkdir(parents=True)
    md = d2 / "manifests"
    lock, lself, lfile = verify_lock(_paths())
    extra = {"status": "FEATURE_SEARCH_SCREEN", "run_label": "FEATURE_SEARCH_SCREEN", "data_version": "DATA05_FSX",
             "data_protocol_lock_sha256": lself, "data_protocol_lock_file_sha256": lfile}
    tabs = {TR: real_train_tables(real_train_store, n_clients=60, seed=21),
            EV: real_eval_tables(real_val_store, n_users=80, seed=22)}
    ents = {}
    for mid, t in tabs.items():
        e = write_manifest_files(md, mid, t, header_extra=extra)
        h = json.loads((md / f"{mid}.header.json").read_text(encoding="utf-8"))
        ents[mid] = {"path": f"data_fs/manifests/{mid}.header.json", "manifest_dir": "data_fs/manifests",
                     "kind": "TRAIN" if mid == TR else "EVAL", "status": "FEATURE_SEARCH_SCREEN",
                     "content_sha256": manifest_content_sha256(h["tables"], e["tables"]),
                     "header_sha256": e["header_file_sha256"], "header_content_sha256": e["header_content_sha256"],
                     **{k: e[k] for k in ("n_users", "n_decisions", "n_eval_rows", "n_rankable_rows", "population_sha256",
                                          "ordered_decision_id_sha256", "eval_ordered_decision_id_sha256",
                                          "rankable_ordered_decision_id_sha256")},
                     "tables": {tn: {"file": f"data_fs/manifests/{mid}.{tn}.parquet", **v} for tn, v in e["tables"].items()},
                     "marker": f"data_fs/manifests/{mid}.OK"}
    write_r1_and_markers(d2, ents)
    ft = lock["features_and_transforms"]
    reg = {"registry": "MANIFEST_ROLE_REGISTRY", "registry_version": 1, "task": "DATA-05 test fixture",
           "data_version": "DATA05_FSX", "run_label": "FEATURE_SEARCH_SCREEN",
           "roles": {"FEATURE_SEARCH_SCREEN": {"train": TR, "eval": EV}}, "manifests": ents,
           "transform_state": {"path": "data_v2/TRANSFORM_STATE_CLEAN_V2.json",
                               "transform_state_sha256": ft["transform_state_self_sha256"],
                               "file_sha256": ft["transform_state_file_sha256"]},
           "path_root": str(root), "source_data_protocol_lock": {"lock_self_sha256": lself, "file_sha256": lfile}}
    return {"root": root, "reg": reg, "path": _write_registry(root / "REGISTRY_FSX.json", reg), "tabs": tabs}


def _variant(fsx, name, mutate) -> Path:
    reg = copy.deepcopy(fsx["reg"])
    mutate(reg)
    return _write_registry(fsx["root"] / name, reg)


def registry_accepts(mid, reg_path, root, kind) -> None:
    """The key property: the registry path verifies the manifest (refusals -> AssertionError)."""
    try:
        verify_manifest_by_registry(mid, reg_path, _paths(), root=root, expect_kind=kind)
    except (ManifestVerificationError, TestManifestRefused) as e:
        raise AssertionError(f"registry verification refused: {e}") from None


def test_registry_verified_fixture_passes_and_serves(fsx):
    from v4int.providers import StoreEvalProvider, StoreTrainProvider, comparison_key_base
    for mid, kind in ((TR, "TRAIN"), (EV, "EVAL")):
        vm = verify_manifest_by_registry(mid, fsx["path"], _paths(), root=fsx["root"], expect_kind=kind)
        assert vm.registry["self_sha256"] == json.loads(fsx["path"].read_text())["self_sha256"]
        assert vm.record()["verified_by"] == "MANIFEST_REGISTRY" and all(v is True for v in vm.checks.values())
        assert int(vm.canonical_decision_id.size) == fsx["reg"]["manifests"][mid]["n_decisions"]
    kw = dict(history_policy=USER, paths=_paths(), manifest_registry=fsx["path"], registry_root=fsx["root"],
              ds_root=REAL_DS, buckets=[0], log=Q)
    tp = StoreTrainProvider(TR, **kw)
    ep = StoreEvalProvider(EV, **kw)
    for p, mid in ((tp, TR), (ep, EV)):
        e = fsx["reg"]["manifests"][mid]
        b = p.manifest_binding
        assert (b["manifest_id"], b["content_sha256"], b["header_sha256"]) == (mid, e["content_sha256"], e["header_sha256"])
        assert p.describe()["verified"]["verified_by"] == "MANIFEST_REGISTRY"
    import numpy as np
    assert bool(tp.batch(np.arange(min(64, tp.n_decisions)))["loss_mask"].all())
    assert sum(int(x["decision_id"].shape[0]) for x in ep.batches(ep.eval_batch)) == ep.n_stream_rows
    key = comparison_key_base(_paths(), tp, ep)
    assert key["history_policy"] == USER and key["population_hash"] == tp.verified.header["population_sha256"]
    try:
        from v4train.registry import check_provider, load_manifest_registry     # TRAIN-01R2 (absent on some hosts)
    except ImportError:
        return
    reg = load_manifest_registry(fsx["path"])
    check_provider(reg, "FEATURE_SEARCH_SCREEN", "train", tp)
    check_provider(reg, "FEATURE_SEARCH_SCREEN", "eval", ep)


def test_registry_refusals(fsx):
    root, paths = fsx["root"], _paths()
    edited = fsx["root"] / "EDITED.json"
    obj = json.loads(fsx["path"].read_text())
    obj["data_version"] = "EDITED_WITHOUT_REHASH"
    edited.write_text(json.dumps(obj), encoding="utf-8")
    other_status = _variant(fsx, "STATUS.json", lambda r: r["manifests"][TR].update(status="BUILT_PRE_LOCK"))
    other_dv = _variant(fsx, "DV.json", lambda r: r.update(data_version="ANOTHER_VERSION"))
    other_ts = _variant(fsx, "TS.json", lambda r: r["transform_state"].update(transform_state_sha256="0" * 64))
    no_entry = _variant(fsx, "NOENTRY.json", lambda r: r["manifests"].pop(EV))
    for p in (edited, other_status, other_dv, other_ts):
        assert_raises(ManifestVerificationError, verify_manifest_by_registry, TR, p, paths, root=root)
    assert_raises(ManifestVerificationError, verify_manifest_by_registry, EV, no_entry, paths, root=root)
    assert_raises(ManifestVerificationError, verify_manifest_by_registry, TR, fsx["path"], paths, root=root,
                  expect_kind="EVAL")
    hp = root / fsx["reg"]["manifests"][EV]["path"]
    orig = hp.read_bytes()
    try:
        hp.write_bytes(orig.replace(b'"n_users"', b'"n_users" ', 1))           # same JSON, other bytes
        assert_raises(ManifestVerificationError, verify_manifest_by_registry, EV, fsx["path"], paths, root=root)
    finally:
        hp.write_bytes(orig)
    verify_manifest_by_registry(EV, fsx["path"], paths, root=root)                 # restored: accepted again


def test_real_fs_manifests_verify_through_registry_fs():
    """Where FS-01's files exist (the server), both FS manifests verify through the real REGISTRY_FS and the binding
    equals its entries."""
    if not REG_FS.is_file():
        pytest.skip("REGISTRY_FS.json is not present")
    reg = json.loads(REG_FS.read_text(encoding="utf-8"))
    need = [V4 / t["file"] for e in reg["manifests"].values() for t in e["tables"].values()]
    if not all(p.is_file() for p in need):
        pytest.skip("FS-01's manifest tables are not on this host")
    from v4int.providers import manifest_binding_of
    for mid, e in reg["manifests"].items():
        vm = verify_manifest_by_registry(mid, REG_FS, _paths())
        b = manifest_binding_of(vm)
        assert (b["content_sha256"], b["header_sha256"]) == (e["content_sha256"], e["header_sha256"]), mid
        assert int(vm.canonical_decision_id.size) == int(e["n_decisions"]), mid


@nc("a registry whose entry carries a tampered content_sha256 must be refused")
def test_nc_tampered_content_hash_is_refused(fsx):
    p = _variant(fsx, "NC_CONTENT.json", lambda r: r["manifests"][TR].update(content_sha256="1" * 64))
    registry_accepts(TR, p, fsx["root"], "TRAIN")


@nc("a registry built on another DATA_PROTOCOL_LOCK (wrong lock hash) must be refused")
def test_nc_wrong_lock_hash_is_refused(fsx):
    p = _variant(fsx, "NC_LOCK.json", lambda r: r["source_data_protocol_lock"].update(lock_self_sha256="2" * 64))
    registry_accepts(EV, p, fsx["root"], "EVAL")


@nc("a TEST manifest id is refused by the registry path before any file is opened")
def test_nc_test_id_is_refused(fsx):
    def add_test(r):
        r["manifests"]["FSX_TEST_FIXTURE"] = copy.deepcopy(r["manifests"][EV])
    p = _variant(fsx, "NC_TEST.json", add_test)
    registry_accepts("FSX_TEST_FIXTURE", p, fsx["root"], "EVAL")
