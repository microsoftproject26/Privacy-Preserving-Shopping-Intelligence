"""Workspace-containment helper (T44: "output paths cannot escape the workspace").

Every path the supervisor treats as an *output* -- an expected artifact, a status file, a log --
must resolve inside a declared workspace root before it is read or written. A naive
`str(path).startswith(str(root))` prefix check is wrong: it treats a sibling directory that merely
shares the workspace's name as a string prefix (".../workspace_evil/...") as if it were inside
".../workspace/...". This module resolves both paths (following symlinks) and checks containment
on path *parts*, via `Path.relative_to`, never on strings.
"""
from __future__ import annotations

import os
from pathlib import Path

from .errors import WorkspaceEscapeError


def safe_path(root: "os.PathLike | str", candidate: "os.PathLike | str") -> Path:
    """Resolve `candidate` (absolute, or relative to `root`) and require it to land inside `root`.

    Resolution follows symlinks (`Path.resolve()`), so a symlink planted inside the workspace that
    points outside it is caught too. `strict=False` is used so this also validates an output path
    *before* the phase has written it (the common case: we declare an expected artifact path, then
    check it after the subprocess exits).

    Raises `WorkspaceEscapeError` if the resolved path is not inside the resolved root (including
    the degenerate case where it equals the root itself, which is never a valid file path).
    """
    root_r = Path(root).resolve()
    cand = Path(candidate)
    full = cand if cand.is_absolute() else root_r / cand
    full_r = full.resolve(strict=False)
    if full_r == root_r:
        raise WorkspaceEscapeError(f"path {candidate!r} resolves to the workspace root itself, not a file in it")
    try:
        full_r.relative_to(root_r)
    except ValueError:
        raise WorkspaceEscapeError(f"path {candidate!r} resolves to {full_r}, outside workspace root {root_r}") from None
    return full_r
