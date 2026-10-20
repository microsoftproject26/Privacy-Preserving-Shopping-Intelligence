"""TRAIN-01R2 `manifest_binding` of the v4int providers: {"manifest_id", "content_sha256", "header_sha256"} computed
from the manifest the provider VERIFIED (content_sha256 by the REGISTRY_FS rule over the verified tables; header_sha256
= the verified header file bytes). It equals a registry entry written by FS-01's own rule for a fixture and is accepted
by the (unmodified) v4train.registry.check_provider; the rule reproduces REGISTRY_FS's recorded values from FS-01's
header files. Negative control: a tampered header hash is caught."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from histkit import assert_raises, nc
from v4int.providers import MANIFEST_CONTENT_RULE, manifest_binding_of

# the registry checker is TRAIN-01R2's new module; a host without it (the server before R2 is synced) skips this file
pytest.importorskip("v4train.registry", reason="TRAIN-01R2 v4train.registry is not on this host")

V4 = Path(__file__).resolve().parents[2]
REG_FS = V4 / "contracts" / "manifest_registry" / "REGISTRY_FS.json"


def _fs_rule_content(header: dict) -> str:
    from v4fs_data.fs_common import manifest_content_sha256        # FS-01's own implementation (read-only import)
    return manifest_content_sha256(header["tables"])


def _fixture_registry(tmp_path, real_env, tp, ep, tamper_header: bool = False):
    from v4train.registry import load_manifest_registry, with_self_sha256
    md = real_env["paths"].manifest_dir
    ents = {}
    for mid, n in (("M05_CALIB_TRAIN", tp.n_decisions), ("M06_CALIB_VALIDATION_WARM", ep.manifest.n)):
        hp = md / f"{mid}.header.json"
        h = json.loads(hp.read_text(encoding="utf-8"))
        hsha = hashlib.sha256(hp.read_bytes()).hexdigest()
        ents[mid] = {"n_decisions": int(n), "content_sha256": _fs_rule_content(h),
                     "header_sha256": ("0" * 64 if tamper_header else hsha)}
    reg = with_self_sha256({"data_version": "DATA05_FIXTURE", "roles": {"CALIBRATION": {
        "train": "M05_CALIB_TRAIN", "eval": "M06_CALIB_VALIDATION_WARM"}}, "manifests": ents})
    p = tmp_path / ("REG_TAMPERED.json" if tamper_header else "REG.json")
    p.write_text(json.dumps(reg, indent=1), encoding="utf-8")
    return load_manifest_registry(p), ents


def binding_accepted(reg, tp, ep) -> None:
    """The key property: TRAIN-01R2's checker accepts both providers' bindings (RegistryError -> AssertionError)."""
    from v4train.registry import RegistryError, check_provider
    try:
        check_provider(reg, "CALIBRATION", "train", tp)
        check_provider(reg, "CALIBRATION", "eval", ep)
    except RegistryError as e:
        raise AssertionError(f"registry binding refused: {e}") from None


def test_binding_equals_the_registry_entry_for_a_fixture(tmp_path, real_env, providers_user, providers_session):
    for tp, ep in (providers_user, providers_session):
        reg, ents = _fixture_registry(tmp_path, real_env, tp, ep)
        for p, mid in ((tp, "M05_CALIB_TRAIN"), (ep, "M06_CALIB_VALIDATION_WARM")):
            b = p.manifest_binding
            assert b["manifest_id"] == mid
            assert b["content_sha256"] == ents[mid]["content_sha256"] and b["header_sha256"] == ents[mid]["header_sha256"]
            assert b["header_sha256"] == p.verified.header_file_sha256
            assert b["header_content_sha256"] == p.verified.header["header_content_sha256"]
            assert b["content_sha256_rule"] == MANIFEST_CONTENT_RULE and p.describe()["manifest_binding"] == b
        binding_accepted(reg, tp, ep)
    assert providers_user[0].manifest_binding == providers_session[0].manifest_binding    # manifest identity only


def test_a_tampered_registry_header_hash_is_refused(tmp_path, real_env, providers_user):
    from v4train.registry import RegistryError, check_provider
    tp, ep = providers_user
    reg, _ = _fixture_registry(tmp_path, real_env, tp, ep, tamper_header=True)
    assert_raises(RegistryError, check_provider, reg, "CALIBRATION", "train", tp)
    assert_raises(RegistryError, check_provider, reg, "CALIBRATION", "eval", ep)


def test_the_rule_reproduces_registry_fs_from_fs01_headers():
    """REGISTRY_FS's recorded content_sha256 / header_sha256 are reproduced by manifest_binding_of from FS-01's header
    files (the header's own table identities; verify_manifest checks them against the tables before a real use)."""
    reg = json.loads(REG_FS.read_text(encoding="utf-8")) if REG_FS.is_file() else None
    if reg is None:
        pytest.skip("REGISTRY_FS.json is not present")
    seen = 0
    for mid, e in reg["manifests"].items():
        cands = [V4 / e["path"], V4 / "data_fs" / "server" / "manifests" / Path(e["path"]).name]
        hp = next((c for c in cands if c.is_file()), None)
        if hp is None:
            continue
        h = json.loads(hp.read_text(encoding="utf-8"))
        vm = SimpleNamespace(manifest_id=mid, header=h, header_file_sha256=hashlib.sha256(hp.read_bytes()).hexdigest(),
                             tables={t["table"]: {k: t[k] for k in ("row_count", "table_content_sha256")}
                                     for t in h["tables"]})
        b = manifest_binding_of(vm)
        assert b["content_sha256"] == e["content_sha256"] == _fs_rule_content(h), mid
        assert b["header_sha256"] == e["header_sha256"], mid
        assert b["header_content_sha256"] == e["header_content_sha256"], mid
        seen += 1
    if not seen:
        pytest.skip("no FS-01 header file on this host")


@nc("a provider whose binding carries a tampered header hash must be refused by the registry check")
def test_nc_tampered_header_hash_is_caught(tmp_path, real_env, providers_user):
    tp, ep = providers_user
    reg, _ = _fixture_registry(tmp_path, real_env, tp, ep)
    bad = copy.copy(tp)
    bad.manifest_binding = dict(tp.manifest_binding, header_sha256="f" * 64)     # the deliberately wrong provider
    binding_accepted(reg, bad, ep)
