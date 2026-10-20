"""Package test list items 22/24: fo_env isolation and path guards.

Independent test of fo_env.py (owner: Opus). Never modifies the code under test.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

import conftest_fo  # noqa: F401  (puts code/ on sys.path, sets FO_NORMALIZERS)
from conftest_fo import set_fo_root  # noqa: F401  (registers the fixture for this module)
import fo_env


# --------------------------------------------------------------------- (22) refuse_split
@pytest.mark.parametrize("split", ["VALIDATION_CONFIRM", "TEST", "LABEL_GRACE"])
def test_refuse_split_raises_for_forbidden_splits(split):
    with pytest.raises(PermissionError):
        fo_env.refuse_split(split)


def test_refuse_split_allows_train_and_validation_select():
    # negative control for the guard itself: these must NOT raise, or refuse_split
    # would be blocking legitimate splits instead of only the forbidden ones.
    fo_env.refuse_split("TRAIN")
    fo_env.refuse_split("VALIDATION_SELECT")


# --------------------------------------------------------------------- (22) study_path
def test_study_path_stays_inside_root(tmp_path, set_fo_root):
    set_fo_root(tmp_path)
    p = fo_env.study_path("audit", "thing.json")
    assert str(p).startswith(str(tmp_path.resolve()))


def test_study_path_refuses_escaping_paths(tmp_path, set_fo_root):
    set_fo_root(tmp_path)
    with pytest.raises(PermissionError):
        fo_env.study_path("..", "..", "etc", "passwd")


def test_study_path_rejects_sibling_name_prefix(tmp_path, set_fo_root):
    """Regression test (2026-09-25: converted from defect-documenting test after lead fix).
    study_path() used to guard escapes with a bare string
    ``str(p).startswith(str(FO_ROOT.resolve()))``, which let a sibling directory whose NAME merely
    starts with the root directory's name (e.g. root `.../tmp_root`, sibling `.../tmp_root_evil`)
    through unguarded. The lead fixed this with ``Path.is_relative_to`` (component-wise). This test
    now asserts the escape is REJECTED, with two negative controls proving genuine in-root paths
    (including one that literally contains the string "..") are still accepted."""
    set_fo_root(tmp_path)
    root_name = tmp_path.name
    sibling = tmp_path.parent / (root_name + "_evil")
    sibling.mkdir(parents=True, exist_ok=True)
    with pytest.raises(PermissionError):
        fo_env.study_path("..", root_name + "_evil", "x.txt")

    # Negative control 1: a genuine in-root path must still be accepted (the fix must not be
    # overly strict and reject legitimate paths).
    p_ok = fo_env.study_path("audit", "thing.json")
    assert str(p_ok).startswith(str(tmp_path.resolve()))

    # Negative control 2: an in-root path whose FINAL component merely contains the substring
    # ".." (but does not escape via a real ".." path segment) must also still be accepted.
    (tmp_path / "audit").mkdir(exist_ok=True)
    p_ok2 = fo_env.study_path("audit", "not..an..escape.json")
    assert str(p_ok2).startswith(str(tmp_path.resolve()))


# --------------------------------------------------------------------- (22) write_once semantics reused via fo_env.sha256_file
def test_sha256_file_matches_hashlib(tmp_path):
    import hashlib
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello world" * 100)
    expect = hashlib.sha256(p.read_bytes()).hexdigest()
    assert fo_env.sha256_file(p) == expect


# --------------------------------------------------------------------- (24) assert_isolated
def test_assert_isolated_passes_for_real_snapshot_modules():
    # feature_view.py already imported synthetic_store/sasrec_model/etc from the snapshot
    # (conftest_fo imports them); assert_isolated must accept them silently.
    import synthetic_store  # noqa: F401  (loaded from snapshot by conftest_fo)
    seen = fo_env.assert_isolated()
    assert "synthetic_store" in seen
    snap = str(fo_env.SNAP.resolve())
    assert seen["synthetic_store"].startswith(snap)


def test_assert_isolated_negative_control_foreign_module_raises(monkeypatch, tmp_path):
    """Negative control: inject a fake module object for one of the ISOLATED_MODULES names,
    with a __file__ OUTSIDE the snapshot, and confirm assert_isolated raises instead of
    silently accepting it."""
    foreign_file = tmp_path / "evil_paths.py"
    foreign_file.write_text("# not the snapshot\n")
    fake = types.ModuleType("paths")
    fake.__file__ = str(foreign_file)
    monkeypatch.setitem(sys.modules, "paths", fake)
    with pytest.raises(RuntimeError, match="isolation breach"):
        fo_env.assert_isolated()


def test_fo02_reuse_modules_preimported_from_snapshot():
    """Regression FO-02 (server probes failure 2026-09-26T07:35Z): the reused GRU train.py inserts
    gru_v1/code at sys.path[0]; later lazy imports (full_store; train's model/objective/evaluator/
    window_cache) then resolved in the GRU tree. After `import fo_env` every reuse module must already
    be loaded from the snapshot."""
    snap = fo_env.SNAP.resolve()
    for name in fo_env.REUSE_PREIMPORT:
        mod = sys.modules.get(name)
        assert mod is not None, f"{name} not pre-imported"
        assert Path(mod.__file__).resolve().is_relative_to(snap), f"{name} loaded from {mod.__file__}"


def test_fo02_foreign_code_dirs_scrubbed_from_sys_path():
    """train.py inserts the (server) gru_v1/code string into sys.path even where it does not exist, so
    this is a real local check: fo_env must have removed it, and a late lazy import stays in the snapshot."""
    assert not [p for p in sys.path if fo_env._is_foreign(p)]
    import importlib
    fs = importlib.import_module("full_store")
    assert Path(fs.__file__).resolve().is_relative_to(fo_env.SNAP.resolve())
    fo_env.assert_isolated()


def test_fo02_negative_control_foreign_sys_path_entry_raises(monkeypatch):
    """Negative control: a foreign code directory re-appearing on sys.path must make assert_isolated
    raise, with both path spellings (posix and Windows separators)."""
    for entry in (str(fo_env.FOREIGN_CODE_DIRS[0]), fo_env.FOREIGN_CODE_DIRS[0].as_posix() + "/vendor",
                  fo_env.FOREIGN_CODE_DIRS[1].as_posix().replace("/", "\\")):
        monkeypatch.setattr(sys, "path", [entry] + [p for p in sys.path])
        with pytest.raises(RuntimeError, match="foreign code directories"):
            fo_env.assert_isolated()
    # negative control for the matcher: a sibling whose name merely starts with "code" is not foreign
    assert not fo_env._is_foreign(fo_env.FOREIGN_CODE_DIRS[0].as_posix() + "_evil")


def test_verify_snapshot_passes_on_untouched_snapshot():
    man = fo_env.verify_snapshot()
    assert "files" in man and len(man["files"]) > 0


def test_verify_snapshot_negative_control_detects_drift(tmp_path, monkeypatch):
    # Negative control: point SNAP at a scratch copy of the manifest whose recorded hash for
    # one file no longer matches the file on disk, and confirm verify_snapshot raises.
    import json
    fake_snap = tmp_path / "snap"
    fake_snap.mkdir()
    victim = fake_snap / "victim.py"
    victim.write_text("original content\n")
    real_hash = fo_env.sha256_file(victim)
    manifest = {"files": {"victim.py": real_hash}}
    (fake_snap / "SNAPSHOT_MANIFEST.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(fo_env, "SNAP", fake_snap)
    fo_env.verify_snapshot()  # sanity: passes before drift
    victim.write_text("DRIFTED content\n")
    with pytest.raises(RuntimeError, match="snapshot drift"):
        fo_env.verify_snapshot()
