"""INT-02 / INT-02R1 items 2-3 (G1 combined gate, conditions B1 / B2 / R3 / R4): DataPaths defaults to
DATA_PROTOCOL_LOCK_V2 (B2, the v1 lock stays reachable HISTORICAL ONLY through DataPaths.historical());
v4int.verify.verify_registry accepts REGISTRY_V2 under LOCK_V2 through the B1 one-way superseded-lock exception, with
one negative control per clause (F3 too: the lock's pinned registry self-hash is enforced on BOTH routes); a full
synthetic end-to-end verify_manifest_by_registry run over a v1-shaped lock + registry + manifest superseded by a
v2-shaped lock (F1); and v4int.providers.comparison_key_base emits EXACTLY the R4 closed comparison-key field list,
values read from LOCK_V2 and the two providers (no code default). Everything reading the REAL contracts/*.json does
so read-only; nothing is written under contracts/. The full end-to-end comparison_key_base over REAL data_v3
manifests (built only on the server) is listed in the INT-02 handoff's R5 server-only cases.
"""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

import v4int.verify as V
from intkit import assert_raises, nc, real_lock, transform_state_source, write_manifest_files, write_r1_and_markers
from v4eval import frozen_config as fc
from v4int.providers import ProviderError, R4_COMPARISON_KEY_FIELDS, comparison_key_base
from v4int.verify import (DataPaths, ManifestVerificationError, lock_self_sha256, manifest_content_sha256,
                          registry_self_sha256, sha256_file, strict_json, verify_lock, verify_manifest_by_registry,
                          verify_registry)
import ppsi_paths

V4 = ppsi_paths.PROJECT_ROOT.resolve()       # the ppsi_v4 workspace (contracts, company records)
LOCK_V2_PATH = V4 / "contracts" / "DATA_PROTOCOL_LOCK_V2.json"
REGISTRY_V2_PATH = V4 / "contracts" / "manifest_registry" / "REGISTRY_V2.json"


def _lock_v2() -> dict:
    return json.loads(LOCK_V2_PATH.read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------------ B2
def test_default_data_paths_bind_lock_v2():
    p = DataPaths.default()
    assert p.lock_path == LOCK_V2_PATH
    assert p.manifest_dir == V4 / "data_v3" / "manifests"
    lock, lself, lfile = verify_lock(p)
    assert lock["document"] == "DATA_PROTOCOL_LOCK" and lock["version"] == "2.0.0"
    assert lself == lock["lock_self_sha256"] == _lock_v2()["lock_self_sha256"]


def test_historical_data_paths_bind_the_v1_lock():
    p = DataPaths.historical()
    assert p.lock_path == V4 / "contracts" / "DATA_PROTOCOL_LOCK.json"
    assert p.manifest_dir == V4 / "data_v2" / "manifests"
    lock, lself, _ = verify_lock(p)
    assert lock["lock_self_sha256"] == lself == _lock_v2()["supersedes"]["lock_self_sha256"]


@nc("a verifier that ignored B2 and always bound the v1 lock would never see LOCK_V2's own document/version")
def test_nc_default_would_be_the_v1_lock_without_b2():
    lock, _, _ = verify_lock(DataPaths.default())
    assert lock.get("version") == "1.0.0"                      # the v1 DATA_PROTOCOL_LOCK's version


# ------------------------------------------------------------------------------------------------ B1 (real REGISTRY_V2 / LOCK_V2)
def test_registry_v2_accepted_under_lock_v2_by_the_b1_exception():
    """REGISTRY_V2's OWN source_data_protocol_lock names the superseded v1 lock (not LOCK_V2's self-hash): refused by
    the plain rule, accepted only through the B1 exception (both clauses read from LOCK_V2 itself); F1: the exception
    also loads and independently verifies the REAL v1 lock file (never just its declared hashes)."""
    lock, lself, lfile = verify_lock(DataPaths.default())
    src = json.loads(REGISTRY_V2_PATH.read_text(encoding="utf-8"))["source_data_protocol_lock"]
    assert src["lock_self_sha256"] != lself, "the direct (same-lock) rule must NOT already accept this pairing"
    assert src["lock_self_sha256"] == lock["supersedes"]["lock_self_sha256"]
    reg, reg_file, reg_self, v1_ctx = verify_registry(REGISTRY_V2_PATH, lself, lfile, lock=lock)
    assert reg_self == lock["population_and_manifests"]["manifest_registry"]["self_sha256"]
    assert reg["data_version"] == lock["data_version"]["name"]
    assert v1_ctx is not None
    v1_lock, v1_self, v1_file = v1_ctx
    assert v1_self == v1_lock["lock_self_sha256"] == lock["supersedes"]["lock_self_sha256"]
    assert v1_file == lock["supersedes"]["file_sha256"]
    assert v1_lock["code"]["v4data_builder_code_sha256"] and v1_lock["catalogue"]["K"] == 158486


def test_b1_clause1_negative_control_wrong_supersedes():
    """Clause 1 tampered (LOCK_V2's declared `supersedes` no longer matches REGISTRY_V2's declared source): refused,
    even though clause 2 (registry.self_sha256 == the pinned value) still holds."""
    lock, lself, lfile = verify_lock(DataPaths.default())
    bad = copy.deepcopy(lock)
    bad["supersedes"]["lock_self_sha256"] = "0" * 64
    assert_raises(ManifestVerificationError, verify_registry, REGISTRY_V2_PATH, lself, lfile, lock=bad)


def test_b1_clause2_negative_control_wrong_pinned_self_hash():
    """Clause 2 tampered (LOCK_V2 no longer pins REGISTRY_V2's actual self-hash): refused, even though clause 1
    (registry.source == lock.supersedes) still holds."""
    lock, lself, lfile = verify_lock(DataPaths.default())
    bad = copy.deepcopy(lock)
    bad["population_and_manifests"]["manifest_registry"]["self_sha256"] = "0" * 64
    assert_raises(ManifestVerificationError, verify_registry, REGISTRY_V2_PATH, lself, lfile, lock=bad)


def test_b1_exception_never_applies_without_the_lock_argument():
    """`lock=None` (a caller that does not pass the full lock object): the exception cannot even be evaluated, so a
    registry built on a superseded lock is refused exactly as the plain (pre-B1) rule would refuse it."""
    _, lself, lfile = verify_lock(DataPaths.default())
    assert_raises(ManifestVerificationError, verify_registry, REGISTRY_V2_PATH, lself, lfile)


def test_b1_any_other_pairing_refused():
    """A registry that matches NEITHER the direct rule NOR the B1 exception (built on a third, unrelated lock)."""
    lock, lself, lfile = verify_lock(DataPaths.default())
    assert_raises(ManifestVerificationError, verify_registry, REGISTRY_V2_PATH, "1" * 64, "2" * 64, lock=lock)


# ------------------------------------------------------------------------------------------------ F3 (both routes)
def test_f3_direct_route_also_enforces_the_pinned_registry_self_hash():
    """A registry that DOES claim the current lock as its direct source (no B1 needed) but carries a different
    self-hash than the one the lock pins is refused too — F3 is not only clause 2 of the B1 exception."""
    lock, lself, lfile = verify_lock(DataPaths.default())
    bad_reg_path = REGISTRY_V2_PATH.parent / "_int02r1_f3_variant.json"
    reg = json.loads(REGISTRY_V2_PATH.read_text(encoding="utf-8"))
    reg["source_data_protocol_lock"] = {"lock_self_sha256": lself, "file_sha256": lfile}  # claims LOCK_V2 directly
    reg["self_sha256"] = registry_self_sha256({k: v for k, v in reg.items() if k != "self_sha256"})
    try:
        bad_reg_path.write_text(json.dumps(reg), encoding="utf-8")
        assert reg["self_sha256"] != lock["population_and_manifests"]["manifest_registry"]["self_sha256"]
        assert_raises(ManifestVerificationError, verify_registry, bad_reg_path, lself, lfile, lock=lock)
    finally:
        bad_reg_path.unlink(missing_ok=True)


# ------------------------------------------------------------------------------------------------ F1 (synthetic e2e)
def _build_superseded_env(tmp_path, real_train_store, *, header_over=None, v1lock_over=None):
    """A v1-shaped TEST DATA_PROTOCOL_LOCK + ONE real-row TRAIN manifest (registry-served, not lock-served) + a
    registry bound to that v1 lock, plus a v2-shaped TEST lock that `supersedes` the v1 lock and pins the registry's
    self-hash — mirroring LOCK_V2 / REGISTRY_V2's real relationship exactly, so `verify_manifest_by_registry` can be
    proven end to end without the real (server-only) data_v3 tree. `header_over` / `v1lock_over` mutate the header /
    v1 lock dict just before it is hashed and written (for the negative controls)."""
    from v4data.common import K, SCHEMA_SELF_SHA256, VERSION_HASH
    root = tmp_path
    v1dir = root / "v1"
    md = v1dir / "manifests"
    mid = "SUPERSEDED_ENV_TRAIN"
    tabs = {"clients": None, "decisions": None, "central_order": None}      # placeholder; filled below
    from intkit import real_train_tables
    tabs = real_train_tables(real_train_store, n_clients=40, seed=91)

    real = real_lock()
    cand = real["population_and_manifests"]["manifest_hashes"]["candidate_order_hash"]
    ts_local = v1dir / "TRANSFORM_STATE_CLEAN_V2.json"
    v1dir.mkdir(parents=True, exist_ok=True)
    ts_local.write_bytes(transform_state_source().read_bytes())
    st = json.loads(ts_local.read_text(encoding="utf-8"))

    v1lock = {"document": "DATA_PROTOCOL_LOCK", "version": "1.0.0-TEST",
             "data_version": {"base_version_hash": VERSION_HASH},
             "catalogue": {"K": int(K)},
             "features_and_transforms": {"schema_self_sha256": SCHEMA_SELF_SHA256,
                                         "transform_state_self_sha256": st["transform_state_sha256"],
                                         "transform_state_file_sha256": sha256_file(ts_local)},
             "population_and_manifests": {"manifest_hashes": {"candidate_order_hash": cand}},
             "code": {"v4data_builder_code_sha256": real["code"]["v4data_builder_code_sha256"]}}
    if v1lock_over is not None:
        v1lock_over(v1lock)
    v1lock["lock_self_sha256"] = lock_self_sha256(v1lock)
    v1lock_path = v1dir / "V1_LOCK.json"
    v1lock_path.write_text(json.dumps(v1lock, indent=1, sort_keys=True), encoding="utf-8")
    v1_self, v1_file = v1lock["lock_self_sha256"], sha256_file(v1lock_path)

    header_extra = {"status": "MAIN", "run_label": "MAIN", "data_version": "SUPERSEDED_ENV_V1",
                    "data_protocol_lock_sha256": v1_self, "data_protocol_lock_file_sha256": v1_file}
    if header_over is not None:
        header_over(header_extra)
    (v1dir / "receipts").mkdir(parents=True, exist_ok=True)
    e = write_manifest_files(md, mid, tabs, header_extra=header_extra, K_=int(K), cand=cand)
    write_r1_and_markers(v1dir, {mid: None})
    h = strict_json(md / f"{mid}.header.json")

    reg_entry = {"path": f"v1/manifests/{mid}.header.json", "kind": "TRAIN", "status": h["status"],
                "content_sha256": manifest_content_sha256(h["tables"], e["tables"]),
                "header_sha256": e["header_file_sha256"], "header_content_sha256": e["header_content_sha256"],
                "n_decisions": e["n_decisions"],
                "tables": {tn: {"file": f"v1/manifests/{mid}.{tn}.parquet", **v} for tn, v in e["tables"].items()},
                "marker": f"v1/manifests/{mid}.OK"}
    reg = {"data_version": "SUPERSEDED_ENV_V1", "roles": {"MAIN": {"train": mid, "eval": mid}},
          "manifests": {mid: reg_entry},
          "transform_state": {"transform_state_sha256": st["transform_state_sha256"], "file_sha256": sha256_file(ts_local)},
          "source_data_protocol_lock": {"lock_self_sha256": v1_self, "file_sha256": v1_file}}
    reg["self_sha256"] = registry_self_sha256(reg)
    reg_path = root / "REGISTRY_SUPERSEDED_ENV.json"
    reg_path.write_text(json.dumps(reg, indent=1, sort_keys=True), encoding="utf-8")

    v2lock = {"document": "DATA_PROTOCOL_LOCK", "version": "2.0.0-TEST",
             "supersedes": {"file": "v1/V1_LOCK.json", "lock_self_sha256": v1_self, "file_sha256": v1_file},
             "data_version": {"name": "SUPERSEDED_ENV_V2"},
             "features_and_transforms": {"production_schema": {"self_sha256": "S" * 64},
                                         "shared_statistics": {"transform_state_sha256": st["transform_state_sha256"],
                                                               "file_sha256": sha256_file(ts_local)}},
             "population_and_manifests": {"manifest_registry": {"self_sha256": reg["self_sha256"]}}}
    v2lock["lock_self_sha256"] = lock_self_sha256(v2lock)
    v2lock_path = root / "V2_LOCK.json"
    v2lock_path.write_text(json.dumps(v2lock, indent=1, sort_keys=True), encoding="utf-8")

    paths = DataPaths(manifest_dir=md, lock_path=v2lock_path, manifest_hashes_path=root / "unused_mh.json",
                      header_list_path=root / "unused_r1.txt", transform_state_path=ts_local, production=False)
    return reg_path, paths, root, mid


def test_f1_superseded_lock_end_to_end(tmp_path, real_train_store):
    reg_path, paths, root, mid = _build_superseded_env(tmp_path, real_train_store)
    vm = verify_manifest_by_registry(mid, reg_path, paths, root=root, expect_kind="TRAIN")
    assert all(v is True for v in vm.checks.values()), {k: v for k, v in vm.checks.items() if v is not True}
    assert vm.registry["self_sha256"] == json.loads(reg_path.read_text())["self_sha256"]


@pytest.mark.parametrize("field,bad", [
    ("builder_code_sha256", "0" * 64),
    ("feature_schema_self_sha256", "1" * 64),
    ("candidate_order_hash", "2" * 64),
    ("catalogue_K", 1),
])
def test_f1_header_field_wrong_vs_v1_lock_refused(tmp_path, real_train_store, field, bad):
    """Each header field F1 now checks against the v1 (superseded) lock, tampered one at a time: refused."""
    reg_path, paths, root, mid = _build_superseded_env(tmp_path, real_train_store, header_over=lambda h: h.update({field: bad}))
    assert_raises(ManifestVerificationError, verify_manifest_by_registry, mid, reg_path, paths, root=root,
                  expect_kind="TRAIN")


def test_f1_header_lock_binding_checked_against_v1_not_v2(tmp_path, real_train_store):
    """The header's data_protocol_lock_sha256 binds the v1 lock (as DATA-06 really builds it); binding it to the v2
    lock's self-hash instead (the pre-F1 bug) is refused."""
    reg_path, paths, root, mid = _build_superseded_env(tmp_path, real_train_store)
    v2lock = json.loads(paths.lock_path.read_text(encoding="utf-8"))
    reg_path2, paths2, root2, mid2 = _build_superseded_env(
        tmp_path / "wrong_binding", real_train_store,
        header_over=lambda h: h.update(data_protocol_lock_sha256=v2lock["lock_self_sha256"]))
    assert_raises(ManifestVerificationError, verify_manifest_by_registry, mid2, reg_path2, paths2, root=root2,
                  expect_kind="TRAIN")


def test_f1_transform_state_read_from_shared_statistics_wrong_value_refused(tmp_path, real_train_store):
    reg_path, paths, root, mid = _build_superseded_env(
        tmp_path, real_train_store,
        v1lock_over=lambda lk: lk.setdefault("_unused", None))  # baseline positive control below tampers v2, not v1
    v2 = json.loads(paths.lock_path.read_text(encoding="utf-8"))
    v2["features_and_transforms"]["shared_statistics"]["transform_state_sha256"] = "3" * 64
    paths.lock_path.write_text(json.dumps(v2), encoding="utf-8")
    assert_raises(ManifestVerificationError, verify_manifest_by_registry, mid, reg_path, paths, root=root,
                  expect_kind="TRAIN")


def test_f1_forged_supersedes_file_refused_even_with_matching_declared_hashes():
    """A `lock.supersedes` whose declared hashes still match REGISTRY_V2's declared source (clause 1 and 2 both look
    satisfied on paper) but whose `file` names a path that does not actually hash to those values: F1 loads the REAL
    file and refuses, rather than trusting the declared hashes alone."""
    lock, _, _ = verify_lock(DataPaths.default())
    forged = copy.deepcopy(lock)
    forged["supersedes"]["file"] = "contracts/DOES_NOT_EXIST.json"     # the file this would load does not exist
    forged_self = lock_self_sha256(forged)                             # an internally-consistent (self-hashing) lock
    assert forged["supersedes"]["lock_self_sha256"] == lock["supersedes"]["lock_self_sha256"]  # clauses still "hold"
    assert_raises(ManifestVerificationError, verify_registry, REGISTRY_V2_PATH, forged_self,
                  V.sha256_file(LOCK_V2_PATH), lock=forged)


@nc("without loading the REAL v1 lock (trusting the declared hashes alone), the forged-file case above would not be "
    "refused")
def test_nc_without_loading_the_real_v1_lock_a_forged_supersedes_would_be_trusted():
    lock, _, _ = verify_lock(DataPaths.default())
    forged = copy.deepcopy(lock)
    forged["supersedes"]["file"] = "contracts/DOES_NOT_EXIST.json"
    forged_self = lock_self_sha256(forged)

    def buggy_exception(src, stated, lk, lock_self):           # the pre-F1 behaviour: declared hashes only, no load
        if not isinstance(lk, dict) or lock_self_sha256(lk) != lock_self:
            return False
        sup = lk.get("supersedes") or {}
        pinned = ((lk.get("population_and_manifests") or {}).get("manifest_registry") or {})
        c1 = sup.get("lock_self_sha256") == src.get("lock_self_sha256") and sup.get("file_sha256") == src.get("file_sha256")
        c2 = stated == pinned.get("self_sha256")
        return bool(c1 and c2)
    reg = json.loads(REGISTRY_V2_PATH.read_text(encoding="utf-8"))
    accepted = buggy_exception(reg["source_data_protocol_lock"], reg["self_sha256"], forged, forged_self)
    assert not accepted           # FAILS: the buggy (pre-F1) exception would have accepted a forged, unloaded file


# ------------------------------------------------------------------------------------------------ R4
def _fake_provider(*, manifest_id, header, lself, mr_self, data_version, tst_sha, is_eval=False, frozen=None):
    verified = SimpleNamespace(lock_self_sha256=lself, registry={"self_sha256": mr_self, "data_version": data_version},
                              header=header)
    p = SimpleNamespace(history_policy="USER_HISTORY_L50", transform_state={"transform_state_sha256": tst_sha},
                       verified=verified, manifest_id=manifest_id)
    if is_eval:
        p.manifest = SimpleNamespace(header=header)
        p.frozen = frozen
    return p


def _r4_providers():
    lock, lself, _ = verify_lock(DataPaths.default())
    mr_self = lock["population_and_manifests"]["manifest_registry"]["self_sha256"]
    dv = lock["data_version"]["name"]
    tst_sha = lock["features_and_transforms"]["shared_statistics"]["transform_state_sha256"]
    kw = dict(lself=lself, mr_self=mr_self, data_version=dv, tst_sha=tst_sha)
    tp = _fake_provider(manifest_id="CALIB_TRAIN_5_100_32768",
                        header={"population_sha256_selection_key_v2": "POPHASH_R4_TEST"}, **kw)
    ep = _fake_provider(manifest_id="CALIB_VALIDATION_5_100_32768", header={"candidate_order_hash": "CANDHASH_R4_TEST"},
                        is_eval=True, frozen=fc.load(), **kw)
    return lock, tp, ep


def test_r4_comparison_key_is_the_closed_field_list_with_explicit_values():
    from v4data.common import canonical_json_bytes, sha256_bytes
    from v4hist.windows import input_contract_sha256
    lock, tp, ep = _r4_providers()
    key = comparison_key_base(DataPaths.default(), tp, ep)
    assert sorted(key) == sorted(R4_COMPARISON_KEY_FIELDS)
    assert key["data_protocol_hash"] == lock["lock_self_sha256"]
    assert key["data_version"] == lock["data_version"]["name"] == "CLEAN_V2_POP5100_V2"
    assert key["manifest_registry_self_sha256"] == lock["population_and_manifests"]["manifest_registry"]["self_sha256"]
    assert key["feature_schema_self_sha256"] == lock["features_and_transforms"]["production_schema"]["self_sha256"]
    assert key["transform_state_sha256"] == lock["features_and_transforms"]["shared_statistics"]["transform_state_sha256"]
    # F7: the VALUES of label_policy_hash and feature_contract_hash, computed with the same real functions/inputs
    want_label = sha256_bytes(canonical_json_bytes(lock["splits_and_labels"]))
    assert key["label_policy_hash"] == want_label
    want_contract = input_contract_sha256(key["feature_schema_self_sha256"], "USER_HISTORY_L50", None)
    assert key["feature_contract_hash"] == want_contract
    assert key["history_policy"] == "USER_HISTORY_L50"
    assert key["split"] == "VALIDATION"
    assert key["population_hash"] == "POPHASH_R4_TEST"
    assert key["candidate_order_hash"] == "CANDHASH_R4_TEST"
    assert key["train_manifest_id"] == "CALIB_TRAIN_5_100_32768"
    assert key["eval_manifest_id"] == "CALIB_VALIDATION_5_100_32768"
    assert key["determinism_mode"] == "STRICT"
    assert key["eval_stream"] == "EVAL_ROWS_ONLY"
    assert key["exposure_point"] == "end"
    assert (key["eval_score_block"], key["eval_batch"], key["eval_device_class"]) == (158486, 1024, "A16-FP32-DET")


def test_r4_requires_registry_verified_providers():
    """A provider verified without a manifest registry (`.verified.registry` None) cannot fill
    manifest_registry_self_sha256: refused, never silently defaulted."""
    _, tp, ep = _r4_providers()
    tp.verified.registry = None
    assert_raises(ProviderError, comparison_key_base, DataPaths.default(), tp, ep)


def test_r4_refuses_a_frozen_config_reporting_all_rows():
    """R2: ALL_ROWS is refused, even if some future frozen config were to carry it."""
    _, tp, ep = _r4_providers()
    ep.frozen = SimpleNamespace(eval_stream="ALL_ROWS", comparison_key_additions=ep.frozen.comparison_key_additions)
    assert_raises(ProviderError, comparison_key_base, DataPaths.default(), tp, ep)


@nc("without the R2 guard, comparison_key_base would silently accept and record ALL_ROWS (F7: this monkeypatches the "
    "refusal OUT of the real comparison_key_base, so the control fails only if the real code path itself changes — "
    "not a hand-written stand-in the guard could drift away from unnoticed)")
def test_nc_without_the_refusal_all_rows_would_be_accepted(monkeypatch):
    import v4int.providers as P
    _, tp, ep = _r4_providers()
    ep.frozen = SimpleNamespace(eval_stream="ALL_ROWS", comparison_key_additions=ep.frozen.comparison_key_additions)
    monkeypatch.setattr(P, "_refuse_non_frozen_eval_stream", lambda eval_stream: None)  # remove exactly this guard
    key = P.comparison_key_base(DataPaths.default(), tp, ep)   # the REAL function, guard neutralised
    assert key["eval_stream"] == "EVAL_ROWS_ONLY"          # FAILS: the un-guarded function actually recorded ALL_ROWS
