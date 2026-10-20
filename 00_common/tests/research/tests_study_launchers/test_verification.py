"""Fail-closed manifest verification: hash / lock / header / transform-state mismatches refused, TEST manifests refused,
compatibility with the real v4data ManifestBuilder output, and fingerprints that change with the manifest."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

import v4int.verify as V
from intkit import (REAL_DS, assert_raises, lock_and_reissue, nc, write_env, write_manifest_files,
                    write_r1_and_markers)
from v4int.providers import StoreEvalProvider, StoreTrainProvider
from v4int.verify import (DataPaths, ManifestVerificationError, TestManifestRefused, header_content_sha256,
                          verify_manifest)

Q = lambda *_: None  # noqa: E731
ALL4 = ("M05_CALIB_TRAIN", "M01_COHORT_TRAIN", "M06_CALIB_VALIDATION_WARM", "M02_VALIDATION_WARM")


def _copy_env(real_env, tmp: Path) -> DataPaths:
    src = Path(real_env["root"]) / "data_v2"
    dst = Path(tmp) / "data_v2"
    shutil.copytree(src, dst)
    p = real_env["paths"]
    rel = lambda q: dst / Path(q).relative_to(src)  # noqa: E731
    return DataPaths(manifest_dir=rel(p.manifest_dir), lock_path=rel(p.lock_path),
                     manifest_hashes_path=rel(p.manifest_hashes_path), header_list_path=rel(p.header_list_path),
                     transform_state_path=rel(p.transform_state_path), production=False)


def _failed(paths, mid) -> list:
    try:
        verify_manifest(mid, paths)
    except ManifestVerificationError as e:
        msg = str(e)
        return sorted(x.strip(" '") for x in msg[msg.index("[") + 1:msg.rindex("]")].split(",")) if "[" in msg else [msg]
    return []


def test_all_four_manifests_verify(real_env):
    for mid in ALL4:
        vm = verify_manifest(mid, real_env["paths"])
        assert all(v is True for v in vm.checks.values()) and len(vm.checks) >= 35
        assert np.all(vm.canonical_decision_id[1:] > vm.canonical_decision_id[:-1])
    assert vm.checks["n_decisions_eq_lock"] and vm.checks["n_decisions_eq_header"]     # counts from header + lock


def test_table_file_hash_mismatch_refused(real_env, tmp_path):
    paths = _copy_env(real_env, tmp_path)
    f = paths.manifest_dir / "M05_CALIB_TRAIN.decisions.parquet"
    pq.write_table(pq.read_table(f), f, compression="gzip")            # same content, other bytes
    bad = _failed(paths, "M05_CALIB_TRAIN")
    assert "decisions.file_sha256_eq_lock" in bad and "decisions.file_sha256_listed_r1" in bad
    assert_raises(ManifestVerificationError, StoreTrainProvider, "M05_CALIB_TRAIN", history_policy="SESSION_L50", paths=paths, ds_root=REAL_DS,
                  buckets=[0], log=Q)


def test_lock_tampering_refused(real_env, tmp_path):
    paths = _copy_env(real_env, tmp_path)
    lock = json.loads(paths.lock_path.read_text(encoding="utf-8"))
    lock["catalogue"]["K"] = 158485
    paths.lock_path.write_text(json.dumps(lock, indent=1), encoding="utf-8")
    assert_raises(ManifestVerificationError, verify_manifest, "M05_CALIB_TRAIN", paths)


def test_manifest_hashes_tampering_refused(real_env, tmp_path):
    paths = _copy_env(real_env, tmp_path)
    mh = json.loads(paths.manifest_hashes_path.read_text(encoding="utf-8"))
    mh["manifests"]["M05_CALIB_TRAIN"]["n_decisions"] += 1
    paths.manifest_hashes_path.write_text(json.dumps(mh, indent=1), encoding="utf-8")
    bad = _failed(paths, "M05_CALIB_TRAIN")
    assert "manifest_hashes_file_sha256_eq_lock" in bad and "manifest_hashes_entry_eq_lock" in bad


def _rewrite_header(paths, mid, fn, recompute=True):
    hp = paths.manifest_dir / f"{mid}.header.json"
    h = json.loads(hp.read_text(encoding="utf-8"))
    fn(h)
    if recompute:
        h["header_content_sha256"] = header_content_sha256(h)
    hp.write_text(json.dumps(h, indent=1), encoding="utf-8")
    write_r1_and_markers(paths.manifest_dir.parent, [m for m in ALL4])


def test_header_status_and_lock_binding_refused(real_env, tmp_path):
    paths = _copy_env(real_env, tmp_path)
    _rewrite_header(paths, "M05_CALIB_TRAIN", lambda h: h.update(status="PARTIAL_TEST_BUILD"))
    assert _failed(paths, "M05_CALIB_TRAIN") == ["header.status"]
    _rewrite_header(paths, "M05_CALIB_TRAIN", lambda h: h.update(status="BUILT_PRE_LOCK"))
    assert _failed(paths, "M05_CALIB_TRAIN") == []
    _rewrite_header(paths, "M06_CALIB_VALIDATION_WARM", lambda h: h.update(data_protocol_lock_sha256="0" * 64))
    assert _failed(paths, "M06_CALIB_VALIDATION_WARM") == ["header.data_protocol_lock_sha256_eq_lock_self"]
    _rewrite_header(paths, "M01_COHORT_TRAIN", lambda h: h.update(n_users=h["n_users"] + 1), recompute=False)
    assert set(_failed(paths, "M01_COHORT_TRAIN")) == {"header.content_self_hash", "header.n_users_eq_lock"}


def test_transform_state_and_marker_tampering_refused(real_env, tmp_path):
    paths = _copy_env(real_env, tmp_path)
    mk = paths.manifest_dir / "M06_CALIB_VALIDATION_WARM.OK"
    obj = json.loads(mk.read_text(encoding="utf-8"))
    k0 = sorted(obj["verified_sha256"])[0]
    obj["verified_sha256"][k0] = "0" * 64
    mk.write_text(json.dumps(obj), encoding="utf-8")
    assert _failed(paths, "M06_CALIB_VALIDATION_WARM") == ["ok_marker_matches_disk"]
    st = json.loads(paths.transform_state_path.read_text(encoding="utf-8"))
    st["no_test_rows_read"] = False
    paths.transform_state_path.write_text(json.dumps(st, indent=1), encoding="utf-8")
    assert_raises(ManifestVerificationError, verify_manifest, "M05_CALIB_TRAIN", paths)


def test_central_order_must_be_ascending_even_with_consistent_hashes(real_env, tmp_path):
    tabs = {"M05_CALIB_TRAIN": {t: dict(v) for t, v in real_env["tables"]["M05_CALIB_TRAIN"].items()}}
    n = len(tabs["M05_CALIB_TRAIN"]["decisions"]["decision_id"])
    tabs["M05_CALIB_TRAIN"]["central_order"] = {"row": np.arange(n, dtype=np.int64)}       # the table order
    paths = write_env(tmp_path, tabs)
    assert _failed(paths, "M05_CALIB_TRAIN") == ["central_order.strictly_ascending"]


def test_test_manifests_refused_before_any_file(tmp_path):
    nowhere = DataPaths(*(tmp_path / "missing" / x for x in ("m", "l", "h", "r", "t")))
    for mid in ("M07_TEST_WARM", "M08_TEST_COLD_PANEL", "M09_TEST_COLD_H_PANEL", "M02_VALIDATION_WARM_TEST"):
        assert_raises(TestManifestRefused, verify_manifest, mid, nowhere)
        assert_raises(TestManifestRefused, StoreTrainProvider, mid, history_policy="SESSION_L50", paths=nowhere, log=Q)
        assert_raises(TestManifestRefused, StoreEvalProvider, mid, history_policy="SESSION_L50", paths=nowhere, log=Q)
    assert not (tmp_path / "missing").exists()


def test_test_rows_and_test_headers_refused(real_env, tmp_path):
    tabs = {"M05_CALIB_TRAIN": {t: dict(v) for t, v in real_env["tables"]["M05_CALIB_TRAIN"].items()},
            "M06_CALIB_VALIDATION_WARM": {t: dict(v) for t, v in real_env["tables"]["M06_CALIB_VALIDATION_WARM"].items()}}
    sp = np.asarray(tabs["M05_CALIB_TRAIN"]["decisions"]["old_split"], dtype=object).copy()
    sp[2] = "TEST"
    tabs["M05_CALIB_TRAIN"]["decisions"]["old_split"] = sp
    v4 = np.asarray(tabs["M06_CALIB_VALIDATION_WARM"]["decisions"]["v4_split"], dtype=object).copy()
    v4[1] = "RETRO_TEST"
    tabs["M06_CALIB_VALIDATION_WARM"]["decisions"]["v4_split"] = v4
    paths = write_env(tmp_path / "rows", tabs)
    assert_raises(TestManifestRefused, verify_manifest, "M05_CALIB_TRAIN", paths)
    assert_raises(TestManifestRefused, verify_manifest, "M06_CALIB_VALIDATION_WARM", paths)
    paths2 = _copy_env(real_env, tmp_path / "hdr")
    _rewrite_header(paths2, "M02_VALIDATION_WARM", lambda h: h.update(v4_split="RETRO_TEST"))
    assert_raises(TestManifestRefused, verify_manifest, "M02_VALIDATION_WARM", paths2)


def test_real_builder_output_format_is_accepted(tmp_path):
    """The real v4data ManifestBuilder on the synthetic frozen-layout dataset (data_testkit), locked and re-issued the
    DATA-04R1 way: every check passes except header.status, which a partial test build must fail."""
    from data_testkit import standard_users, write_dataset
    from v4data.manifests import ManifestBuilder
    users, cat = standard_users()
    root = write_dataset(tmp_path / "ds", users, cat)
    out = tmp_path / "d2"
    b = ManifestBuilder(ds_root=root, out_root=out, buckets=[0], expect=False, log=Q)
    b.pass_support()
    b.pass_aggregate()
    b.select_populations()
    b.pass_extract()
    hashes = b.write(b.assemble())
    hdr = json.loads((out / "manifests" / "M01_COHORT_TRAIN.header.json").read_text(encoding="utf-8"))
    keep = {m: hashes[m] for m in ALL4}
    paths = lock_and_reissue(out, keep, K_=hdr["catalogue_K"], cand=hdr["candidate_order_hash"])
    for mid in ALL4:
        assert _failed(paths, mid) == ["header.status"], mid


def test_fingerprint_changes_with_the_manifest(real_env, providers, tmp_path):
    tp, ep = providers
    same = StoreTrainProvider("M05_CALIB_TRAIN", history_policy="SESSION_L50", paths=real_env["paths"], ds_root=REAL_DS, buckets=[0], log=Q)
    assert same.fingerprint == tp.fingerprint                                  # deterministic
    tabs = {t: dict(v) for t, v in real_env["tables"]["M05_CALIB_TRAIN"].items()}
    dec = tabs["decisions"]
    keep = np.ones(len(dec["decision_id"]), dtype=bool)
    keep[-1] = False                                                           # the last client's last row
    tabs["decisions"] = {k: np.asarray(v)[keep] for k, v in dec.items()}
    cli = {k: np.asarray(v).copy() for k, v in tabs["clients"].items()}
    cli["row_count"][-1] -= 1
    cli["n_eligible_train_examples"][-1] -= 1
    tabs["clients"] = cli
    tabs["central_order"] = {"row": np.argsort(tabs["decisions"]["decision_id"], kind="stable").astype(np.int64)}
    p1 = write_env(tmp_path / "one_less", {"M05_CALIB_TRAIN": tabs})
    fewer = StoreTrainProvider("M05_CALIB_TRAIN", history_policy="SESSION_L50", paths=p1, ds_root=REAL_DS, buckets=[0], log=Q)
    assert fewer.n_decisions == tp.n_decisions - 1 and fewer.fingerprint != tp.fingerprint
    d2 = tmp_path / "hdr" / "data_v2"
    e = {"M05_CALIB_TRAIN": write_manifest_files(d2 / "manifests", "M05_CALIB_TRAIN",
                                                 real_env["tables"]["M05_CALIB_TRAIN"],
                                                 header_extra={"note": "same tables, another header"})}
    p2 = lock_and_reissue(d2, e)
    other = StoreTrainProvider("M05_CALIB_TRAIN", history_policy="SESSION_L50", paths=p2, ds_root=REAL_DS, buckets=[0], log=Q)
    assert np.array_equal(other.decision_ids(), tp.decision_ids()) and other.fingerprint != tp.fingerprint
    assert ep.fingerprint != tp.fingerprint


@nc("a verifier without the file-hash checks would accept a re-encoded (tampered) manifest file")
def test_nc_file_hash_check_is_what_refuses(real_env, tmp_path, monkeypatch):
    paths = _copy_env(real_env, tmp_path)
    f = paths.manifest_dir / "M05_CALIB_TRAIN.decisions.parquet"
    pq.write_table(pq.read_table(f), f, compression="gzip")
    real_sha = V.sha256_file
    lock = json.loads(paths.lock_path.read_text(encoding="utf-8"))
    want = lock["population_and_manifests"]["manifest_hashes"]["manifests"]["M05_CALIB_TRAIN"]["tables"]["decisions"]
    listed = V.parse_sha_list(paths.header_list_path)
    marker = json.loads((paths.manifest_dir / "M05_CALIB_TRAIN.OK").read_text(encoding="utf-8"))["verified_sha256"]

    def fake_sha(p):                                                          # the deliberately wrong verifier
        return want["file_sha256"] if Path(p).name == f.name else real_sha(p)
    monkeypatch.setattr(V, "sha256_file", fake_sha)
    monkeypatch.setattr(V, "parse_sha_list", lambda _p: dict(listed))
    assert marker and listed
    assert_raises(ManifestVerificationError, verify_manifest, "M05_CALIB_TRAIN", paths)
