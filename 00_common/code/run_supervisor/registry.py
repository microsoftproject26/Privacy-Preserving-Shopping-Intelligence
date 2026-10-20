"""Phase registry: a JSON file naming every phase the supervisor knows how to run.

Top level: {"phases": [ ... ]}. Each entry:

  id                  str, unique within the registry.
  command             list[str]  -- argv, executed directly with no shell.  \
  shell_command       str        -- executed via a shell with pipefail semantics. / exactly one
                                      of the two is required.
  prerequisites       list of {"path": str, "sha256": str} -- gate files, checked (existence +
                      hash) before the phase may start. Read-only; may point anywhere on disk
                      (typically another phase's or another workspace's output), so *this* module
                      does not enforce workspace containment on them -- only on this phase's own
                      declared outputs (see `expected_artifacts` and `artifacts.py`).
  expected_artifacts  list of {"path": str, "min_bytes": int = 1, "json_strict": bool | null}.
                      Relative paths are resolved against the workspace root and MUST stay inside
                      it (T44). `json_strict` defaults to true for a path ending in ".json". At
                      least one entry is required: a phase with no declared output can never be
                      verified DONE (an empty/no-op phase must not be reportable as success).
  gpus                list[int] = []      -- GPU indices this phase needs exclusive use of.
  planned_peak_gib    float = 0.0         -- planned peak disk usage while this phase runs.
  max_attempts        int = 2             -- package 10 S4: "a second failure sets FAILED".
  depends_on          list[str] = []      -- phase ids that must be DONE before this one starts;
                      if one of them FAILs, this phase is stopped (BLOCKED), never silently run.
  timeout_s           float | null = None -- subprocess wall-clock timeout.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .errors import RegistryError


@dataclass(frozen=True)
class GateFile:
    path: str
    sha256: str


@dataclass(frozen=True)
class ExpectedArtifact:
    path: str
    min_bytes: int = 1
    json_strict: "bool | None" = None  # None = auto (True iff path ends with ".json")

    def wants_json_strict(self) -> bool:
        return self.json_strict if self.json_strict is not None else self.path.endswith(".json")


@dataclass(frozen=True)
class Phase:
    id: str
    command: "tuple[str, ...] | None"
    shell_command: "str | None"
    prerequisites: "tuple[GateFile, ...]"
    expected_artifacts: "tuple[ExpectedArtifact, ...]"
    gpus: "tuple[int, ...]"
    planned_peak_gib: float
    max_attempts: int
    depends_on: "tuple[str, ...]"
    timeout_s: "float | None"


def _phase_from_dict(d: dict) -> Phase:
    pid = d.get("id")
    if not pid or not isinstance(pid, str):
        raise RegistryError(f"phase entry missing a non-empty string 'id': {d!r}")
    command = d.get("command")
    shell_command = d.get("shell_command")
    if command is not None and not (isinstance(command, list) and all(isinstance(a, str) for a in command)):
        raise RegistryError(f"phase {pid}: 'command' must be a list of argv strings")
    if bool(command) == bool(shell_command):
        raise RegistryError(f"phase {pid}: exactly one of 'command' or 'shell_command' is required")
    prereqs = []
    for g in d.get("prerequisites", []):
        sha = g.get("sha256")
        if not isinstance(sha, str) or len(sha) != 64:
            raise RegistryError(f"phase {pid}: prerequisite {g.get('path')!r} has a malformed sha256")
        prereqs.append(GateFile(path=g["path"], sha256=sha))
    arts_raw = d.get("expected_artifacts", [])
    if not arts_raw:
        raise RegistryError(f"phase {pid}: at least one expected_artifacts entry is required "
                            f"(a phase with no declared output can never be verified DONE)")
    arts = tuple(ExpectedArtifact(path=a["path"], min_bytes=int(a.get("min_bytes", 1)),
                                   json_strict=a.get("json_strict")) for a in arts_raw)
    max_attempts = int(d.get("max_attempts", 2))
    if max_attempts < 1:
        raise RegistryError(f"phase {pid}: max_attempts must be >= 1")
    gpus = tuple(int(g) for g in d.get("gpus", []))
    return Phase(id=pid, command=tuple(command) if command else None, shell_command=shell_command,
                 prerequisites=tuple(prereqs), expected_artifacts=arts, gpus=gpus,
                 planned_peak_gib=float(d.get("planned_peak_gib", 0.0)), max_attempts=max_attempts,
                 depends_on=tuple(d.get("depends_on", [])), timeout_s=d.get("timeout_s"))


def load_registry(path: "Path | str") -> "dict[str, Phase]":
    raw = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RegistryError(f"registry {path}: invalid JSON: {e}") from e
    phases_raw = data.get("phases") if isinstance(data, dict) else None
    if not isinstance(phases_raw, list) or not phases_raw:
        raise RegistryError(f"registry {path}: 'phases' must be a non-empty list")
    phases: "dict[str, Phase]" = {}
    for d in phases_raw:
        ph = _phase_from_dict(d)
        if ph.id in phases:
            raise RegistryError(f"registry {path}: duplicate phase id {ph.id!r}")
        phases[ph.id] = ph
    for ph in phases.values():
        for dep in ph.depends_on:
            if dep not in phases:
                raise RegistryError(f"phase {ph.id}: depends_on unknown phase {dep!r}")
    return phases
