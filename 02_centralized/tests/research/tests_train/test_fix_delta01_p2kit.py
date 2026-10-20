"""FIX-DELTA-01 F5 (REV-DELTA-01) — p2kit's V4-internal P2 work dir is per pytest session (keyed by --basetemp, set by
conftest.pytest_configure; the process id without one) and cleanup() removes only this session's dir, so concurrent
suites never delete each other's P2 fixtures (the 13 E + 1 F of test_train05 in the review's concurrent run).
Negative control: the pre-F5 cleanup (rmtree of the shared _p2work root) deletes another session's dir.
"""
from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

import p2kit
from trainkit import nc


def _other_session():
    other = p2kit.P2_BASE / "0123456789ab_other_session"
    other.mkdir(parents=True, exist_ok=True)
    (other / "P2_GRU_STAGE1.json").write_text("{}", encoding="utf-8")
    return other


def _drop(other: Path) -> None:
    shutil.rmtree(other, ignore_errors=True)
    try:
        p2kit.P2_BASE.rmdir()
    except OSError:
        pass


def test_p2_work_dir_is_per_session(request, tmp_path):
    bt = request.config.getoption("basetemp", None)
    key_src = os.environ[p2kit.SESSION_ENV]
    assert key_src == (str(Path(bt).resolve()) if bt else f"pid{os.getpid()}")
    assert p2kit.P2_ROOT.parent == p2kit.P2_BASE
    assert p2kit.P2_ROOT.name == p2kit.session_key() == hashlib.sha256(key_src.encode()).hexdigest()[:12]
    assert p2kit.under_v4(tmp_path / "p2").parent == p2kit.P2_ROOT                     # outside V4 -> this session
    assert p2kit.contest_root().name == p2kit.session_key()
    other = _other_session()
    try:
        p2kit.P2_ROOT.mkdir(parents=True, exist_ok=True)
        (p2kit.P2_ROOT / "mine.json").write_text("{}", encoding="utf-8")
        p2kit.cleanup()
        assert not p2kit.P2_ROOT.exists()
        assert (other / "P2_GRU_STAGE1.json").is_file()                                  # the other session survives
    finally:
        _drop(other)


def _sandbox(tmp_path, monkeypatch):
    base = tmp_path / "_p2work"
    mine, other = base / "mine", base / "other"
    for d in (mine, other):
        d.mkdir(parents=True)
        (d / "P2_GRU_STAGE1.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(p2kit, "P2_BASE", base)
    monkeypatch.setattr(p2kit, "P2_ROOT", mine)
    return mine, other


def test_cleanup_spares_other_sessions_sandbox(tmp_path, monkeypatch):
    mine, other = _sandbox(tmp_path, monkeypatch)
    p2kit.cleanup()
    assert not mine.exists() and (other / "P2_GRU_STAGE1.json").is_file()


@nc("the pre-F5 cleanup (rmtree of the shared _p2work root) spares another session's P2 dir")
def test_nc_shared_root_cleanup_spares_other_session(tmp_path, monkeypatch):
    mine, other = _sandbox(tmp_path, monkeypatch)                  # a sandbox base: never the real shared root
    shutil.rmtree(p2kit.P2_BASE, ignore_errors=True)               # MUTANT: the pre-F5 cleanup
    assert (other / "P2_GRU_STAGE1.json").is_file(), "the other session's P2 dir was deleted"
