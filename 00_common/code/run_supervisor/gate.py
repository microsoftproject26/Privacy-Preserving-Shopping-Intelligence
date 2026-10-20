"""Prerequisite gate-file check (package 10 S4: "no later stage starts if its prerequisite gate
file is missing/stale"; package 08 S7). Read-only: gate files may live anywhere -- they are
typically another phase's or another workspace's output -- so this module does not enforce
workspace containment on them; only `artifacts.py` does, for paths this supervisor itself owns.

Fail-closed: a missing file, an unreadable file, or a hash mismatch all mean "not ready" -- the
phase never starts. A hash mismatch is reported distinctly from "missing" because it usually means
something more concerning than "not produced yet" (staleness, corruption or tampering), even though
neither case is treated as fatal by itself here; `runner.py` decides what to do with the result.
"""
from __future__ import annotations

from dataclasses import dataclass

from .hashing import sha256_file
from .registry import GateFile


@dataclass(frozen=True)
class GateResult:
    ok: bool
    missing: "tuple[str, ...]"
    mismatched: "tuple[str, ...]"
    reason: str


def check_gates(prerequisites: "tuple[GateFile, ...]") -> GateResult:
    from pathlib import Path

    missing, mismatched = [], []
    for g in prerequisites:
        p = Path(g.path)
        if not p.is_file():
            missing.append(g.path)
            continue
        try:
            actual = sha256_file(p)
        except OSError:
            missing.append(g.path)
            continue
        if actual != g.sha256:
            mismatched.append(g.path)
    ok = not missing and not mismatched
    reason = "ok" if ok else f"missing={list(missing)} mismatched={list(mismatched)}"
    return GateResult(ok, tuple(missing), tuple(mismatched), reason)
