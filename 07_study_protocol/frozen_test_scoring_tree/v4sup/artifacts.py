"""Expected-artifact validation (package 10 S4; T44/T45).

A phase is DONE only on exit code 0 AND every expected artifact:
  - resolves inside the workspace (T44, via `paths.safe_path`);
  - exists and is non-empty (>= min_bytes);
  - for a JSON artifact, parses as *strict* JSON.

Python's `json` module is, by default, permissive: it accepts the non-standard `NaN`,
`Infinity` and `-Infinity` literals and happily returns `float("nan")`. We refuse them
explicitly (T45) via `parse_constant` -- a NaN metric silently accepted as a number is worse than
a loud failure, and a phase that produced one must never be reported DONE.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .errors import WorkspaceEscapeError
from .paths import safe_path
from .registry import ExpectedArtifact


def _reject_constant(token: str):
    raise ValueError(f"non-finite JSON constant {token!r} is not allowed (strict JSON)")


def parse_strict_json(text: str):
    return json.loads(text, parse_constant=_reject_constant)


@dataclass(frozen=True)
class ArtifactCheck:
    path: str
    ok: bool
    reason: str


def validate_artifacts(workspace_root: "Path | str",
                        expected: "tuple[ExpectedArtifact, ...]") -> "tuple[bool, tuple[ArtifactCheck, ...]]":
    results = []
    for a in expected:
        try:
            full = safe_path(workspace_root, a.path)
        except WorkspaceEscapeError as e:
            results.append(ArtifactCheck(a.path, False, f"workspace escape refused: {e}"))
            continue
        if not full.is_file():
            results.append(ArtifactCheck(a.path, False, "missing"))
            continue
        size = full.stat().st_size
        if size < a.min_bytes:
            results.append(ArtifactCheck(a.path, False, f"empty or truncated ({size} bytes < {a.min_bytes})"))
            continue
        if a.wants_json_strict():
            try:
                parse_strict_json(full.read_text(encoding="utf-8"))
            except (ValueError, UnicodeDecodeError) as e:
                results.append(ArtifactCheck(a.path, False, f"not strict JSON: {e}"))
                continue
        results.append(ArtifactCheck(a.path, True, "ok"))
    ok = all(r.ok for r in results)
    return ok, tuple(results)
