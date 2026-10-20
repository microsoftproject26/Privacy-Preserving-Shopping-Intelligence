"""T44: output paths cannot escape the declared workspace root -- neither via a relative '../'
segment, nor an absolute path elsewhere, nor a sibling directory that merely shares the
workspace's name as a string prefix (the classic naive-prefix-check bug this module exists to
avoid), nor a symlink planted inside the workspace that points outside it.
"""
from __future__ import annotations

import os

import pytest

from v4sup.errors import WorkspaceEscapeError
from v4sup.paths import safe_path


def test_relative_path_stays_inside(workspace):
    p = safe_path(workspace, "out/result.json")
    assert p == (workspace / "out" / "result.json").resolve()


def test_dotdot_escape_refused(workspace):
    with pytest.raises(WorkspaceEscapeError):
        safe_path(workspace, "../outside.json")


def test_nested_dotdot_escape_refused(workspace):
    with pytest.raises(WorkspaceEscapeError):
        safe_path(workspace, "a/b/../../../outside.json")


def test_absolute_outside_path_refused(workspace, tmp_path):
    outside = tmp_path / "elsewhere" / "f.json"
    with pytest.raises(WorkspaceEscapeError):
        safe_path(workspace, str(outside))


def test_workspace_root_itself_refused(workspace):
    with pytest.raises(WorkspaceEscapeError):
        safe_path(workspace, ".")


def test_sibling_prefix_path_refused(tmp_path):
    """The bug a naive `str(candidate).startswith(str(root))` check would miss: a sibling
    directory whose name merely starts with the workspace's name as a string
    ('.../workspace_evil') is not inside '.../workspace'."""
    root = tmp_path / "workspace"
    root.mkdir()
    sibling_evil = tmp_path / "workspace_evil"
    sibling_evil.mkdir()
    (sibling_evil / "f.json").write_text("{}")
    with pytest.raises(WorkspaceEscapeError):
        safe_path(root, str(sibling_evil / "f.json"))


def test_symlink_escape_refused(workspace, tmp_path):
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    (outside_dir / "secret.json").write_text("{}")
    link = workspace / "link_out"
    try:
        os.symlink(outside_dir, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted in this environment")
    with pytest.raises(WorkspaceEscapeError):
        safe_path(workspace, "link_out/secret.json")
