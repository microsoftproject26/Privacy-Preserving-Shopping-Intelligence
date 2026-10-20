"""DATA-09 TEST gate tests: the gate opens only with ALL_METHODS_LOCK + the first MANIFEST_BUILD register entry binding its
sha256 and a verifying hash chain; read_with_test refuses without a valid GateRecord; the REAL gate is closed on this host
and every DATA-09 entry point refuses before opening anything - with strict-xfail negative controls."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pyarrow.compute as pc

from conftest import V4, closed_gate_only, make_gate_files, make_register
from data_testkit import assert_raises, nc
from v4data.common import TestGateError
from v4data09 import d9_gate as G
from v4data09.d9_read import read_with_test


def _opens(lock, reg) -> bool:
    try:
        G.check_test_gate(lock, reg)
        return True
    except TestGateError:
        return False


def _ledger(ds: Path) -> Path:
    return ds / "decisions" / "user_bucket=0" / "ledger.parquet"


# ------------------------------------------------------------------------------------------ positive
def test_gate_opens_with_lock_and_manifest_build_entry(gate_files):
    lock, reg = gate_files
    g = G.check_test_gate(lock, reg)
    assert g.lock_sha256 == hashlib.sha256(lock.read_bytes()).hexdigest()
    assert g.entry_id == 2 and g.n_register_entries == 2
    assert g.header_fields() == {"all_methods_lock_sha256": g.lock_sha256, "test_exposure_register_entry_id": 2}
    assert G.revalidate(g).entry_sha256 == g.entry_sha256


def test_first_manifest_build_entry_binds(tmp_path):
    lock, reg = make_gate_files(tmp_path)
    sha = hashlib.sha256(lock.read_bytes()).hexdigest()
    make_register(reg, [{"access_purpose": "STRUCTURAL_COUNT"}, {"access_purpose": "MANIFEST_BUILD",
                                                                 "all_methods_lock_sha256": sha},
                        {"access_purpose": "LOCKED_EVALUATION", "all_methods_lock_sha256": sha},
                        {"access_purpose": "MANIFEST_BUILD", "all_methods_lock_sha256": sha}])
    assert G.check_test_gate(lock, reg).entry_id == 2            # a later append never changes the header id


def test_read_with_test_requires_a_valid_gate(synth_ds, synth_gate):
    t = read_with_test(_ledger(synth_ds), ["split", "decision_id"], synth_gate)
    assert pc.sum(pc.cast(pc.equal(t["split"], "TEST"), "int64")).as_py() > 0
    assert_raises(TestGateError, read_with_test, _ledger(synth_ds), ["split"], None)
    forged = dataclasses.replace(synth_gate, lock_sha256="0" * 64)
    assert_raises(TestGateError, read_with_test, _ledger(synth_ds), ["split"], forged)
    assert_raises(TestGateError, read_with_test, _ledger(synth_ds), ["split"], dataclasses.asdict(synth_gate))


def test_gate_refuses_after_the_lock_changes(tmp_path):
    lock, reg = make_gate_files(tmp_path)
    g = G.check_test_gate(lock, reg)
    lock.write_text('{"document": "ALL_METHODS_LOCK", "edited": true}', encoding="utf-8")
    assert_raises(TestGateError, G.revalidate, g)


# ------------------------------------------------------------------------------------------ the real gate is closed
@closed_gate_only
def test_real_gate_is_closed_on_this_host():
    assert not (V4 / "company" / "gates" / "ALL_METHODS_LOCK.json").exists()
    assert_raises(G.TestGateRefused, G.check_test_gate)


@closed_gate_only
def test_every_entry_point_refuses_while_the_real_gate_is_closed(tmp_path, capsys):
    from v4data09 import d9_build, d9_finalize
    assert d9_gate_main() == 2
    missing_ds = tmp_path / "no_dataset_here"                    # never opened: the gate refuses first
    assert d9_build.main(["--out", str(tmp_path / "out"), "--dataset", str(missing_ds)]) == 2
    assert not (tmp_path / "out").exists()
    for step in ("registry", "verify", "g0", "final"):
        assert d9_finalize.main([step, "--v4", str(tmp_path)]) == 2
    assert_raises(G.TestGateRefused, d9_build.D9Builder, ds_root=missing_ds, out_root=tmp_path, buckets=[0], gate=None)


def d9_gate_main():
    return G.main([])


# ------------------------------------------------------------------------------------------ negative controls
@nc("no ALL_METHODS_LOCK.json: the gate must refuse")
def test_nc_gate_no_lock(tmp_path):
    lock, reg = make_gate_files(tmp_path)
    lock.unlink()
    assert _opens(lock, reg)


@nc("the lock's sha256 differs from the register entry's: the gate must refuse")
def test_nc_gate_lock_sha_mismatch(tmp_path):
    lock, reg = make_gate_files(tmp_path)
    lock.write_text('{"document": "ALL_METHODS_LOCK", "v": 2}', encoding="utf-8")
    assert _opens(lock, reg)


@nc("no register entry binds the lock: the gate must refuse")
def test_nc_gate_register_entry_missing(tmp_path):
    lock, reg = make_gate_files(tmp_path)
    make_register(reg, [{"access_purpose": "STRUCTURAL_COUNT"}, {"access_purpose": "HISTORICAL_PRIOR_EXPOSURE"}])
    assert _opens(lock, reg)


@nc("an entry of the wrong purpose (LOCKED_EVALUATION) carrying the lock sha256 must not open the gate")
def test_nc_gate_wrong_purpose(tmp_path):
    lock, reg = make_gate_files(tmp_path, purpose="LOCKED_EVALUATION")
    assert _opens(lock, reg)


@nc("a MANIFEST_BUILD entry of split LEGACY_TEST must not open the RETRO_TEST gate")
def test_nc_gate_wrong_split(tmp_path):
    lock, reg = make_gate_files(tmp_path)
    sha = hashlib.sha256(lock.read_bytes()).hexdigest()
    make_register(reg, [{"access_purpose": "MANIFEST_BUILD", "split": "LEGACY_TEST", "all_methods_lock_sha256": sha}])
    assert _opens(lock, reg)


@nc("a tampered register entry (chain broken, not re-hashed) must not open the gate")
def test_nc_gate_broken_chain(tmp_path):
    lock, reg = make_gate_files(tmp_path)
    d = json.loads(reg.read_text(encoding="utf-8"))
    d["entries"][0]["observations_description"] = "edited after the fact"
    reg.write_text(json.dumps(d), encoding="utf-8")
    assert _opens(lock, reg)


@nc("a re-hashed but re-linked chain (entry 1 rewritten, entry 2 still points at the old hash) must not verify")
def test_nc_gate_relinked_chain(tmp_path):
    lock, reg = make_gate_files(tmp_path)
    d = json.loads(reg.read_text(encoding="utf-8"))
    e = d["entries"][0]
    e["source"] = "rewritten"
    body = {k: v for k, v in e.items() if k != "entry_sha256"}
    e["entry_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
                                       .encode()).hexdigest()
    reg.write_text(json.dumps(d), encoding="utf-8")
    assert _opens(lock, reg)


@nc("read_with_test without a GateRecord must refuse")
def test_nc_read_with_test_without_gate(synth_ds):
    try:
        read_with_test(_ledger(synth_ds), ["split"], None)
        read = True
    except TestGateError:
        read = False
    assert read
