"""The skip / resume / start decision and the ``--force`` move-aside.

The decision reads only the run directory of a unit (its curated copy under
``<stage>/results/runs/...``, where the published checkpoints are extracted):

* **skip**    - a complete ``RUN_SUMMARY.json`` exists: it parses and its ``status`` is one of
  the launcher completion statuses (``DONE*``; ``PENDING_CKPT_AVG`` is not complete).
* **resume**  - the run is not complete, a resumable checkpoint matches one of the unit's globs
  (``ckpt/*/latest.pt`` for FL, ``checkpoints/latest.pt`` for the central trainer, shard
  markers for the device launcher) and the unit's launcher really supports resuming.
* **start**   - anything else (no run directory, or leftovers no launcher can continue).

``--force`` moves an existing run directory aside to ``<run dir>.bak-<timestamp>`` before the
decision, so the unit starts fresh; nothing is ever deleted.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .units import Unit

# The statuses the launchers write into RUN_SUMMARY.json when a run reached its end
# (launch_fl.py: DONE_MAIN / DONE_S_CAL; launch_device.py: DONE; v4train/trainer.py:
# DONE_MAIN / DONE_CALIBRATION / DONE_TUNING / DONE_CONTINUATION). PENDING_CKPT_AVG,
# RUNNING, INTERRUDED and PARTIAL mean the run still needs work.
COMPLETE_STATUSES = frozenset({
    "DONE", "DONE_MAIN", "DONE_S_CAL", "DONE_CALIBRATION", "DONE_TUNING", "DONE_CONTINUATION",
})


class Action(Enum):
    SKIP = "skip"
    RESUME = "resume"
    START = "start"


@dataclass(frozen=True)
class RunState:
    """What the run directory of a unit currently holds."""

    run_dir: Path
    exists: bool
    summary_path: Path | None   # the RUN_SUMMARY.json that was found (None when absent)
    summary: dict | None        # parsed summary (None when missing or unusable)
    problem: str | None         # why no complete summary could be read
    status: str | None
    complete: bool
    resumable: Path | None      # the checkpoint / shard marker that allows a resume


def read_run_state(run_dir: Path, summary_names: tuple[str, ...],
                   resumable: tuple[str, ...]) -> RunState:
    """Inspect one run directory: summary, completion and resumable checkpoint."""
    summary_path = summary = None
    problem = status = None
    if run_dir.is_dir():
        for name in summary_names:
            candidate = run_dir / name
            if candidate.is_file():
                summary_path = candidate
                try:
                    loaded = json.loads(candidate.read_text(encoding="utf-8"))
                except (OSError, ValueError) as exc:
                    problem = f"{name} does not parse ({exc.__class__.__name__}: {exc})"
                    break
                if not isinstance(loaded, dict):
                    problem = f"{name} is not a JSON object"
                    break
                summary = loaded
                status = loaded.get("status") if isinstance(loaded.get("status"), str) else None
                if status is None:
                    problem = f"{name} has no status field"
                break
        if summary_path is None:
            problem = "no RUN_SUMMARY.json"
    else:
        problem = "no run directory"
    complete = status in COMPLETE_STATUSES
    checkpoint = None
    for pattern in resumable:
        matches = sorted(run_dir.glob(pattern))
        if matches:
            checkpoint = matches[0]
            break
    return RunState(run_dir=run_dir, exists=run_dir.is_dir(), summary_path=summary_path,
                    summary=summary, problem=None if complete else problem, status=status,
                    complete=complete, resumable=checkpoint)


def decide(state: RunState, resume_flag: str) -> tuple[Action, str]:
    """The skip / resume / start decision for one unit and the reason shown in the plan."""
    if state.complete:
        return Action.SKIP, f"RUN_SUMMARY status {state.status}"
    if state.exists and state.resumable is not None and resume_flag != "none":
        where = state.resumable.relative_to(state.run_dir).as_posix() \
            if state.resumable.is_relative_to(state.run_dir) else state.resumable.name
        return Action.RESUME, f"{where} and the launcher supports {resume_flag}"
    if not state.exists:
        return Action.START, "no run directory"
    why = state.problem or f"status {state.status}"
    if state.resumable is not None:
        return (Action.START,
                f"{why}; a resumable checkpoint exists but the launcher cannot resume "
                "(--force restarts from scratch)")
    return Action.START, f"{why}; no resumable checkpoint (existing leftovers block a fresh " \
                         "start until --force moves them aside)"


def move_aside(run_dir: Path, timestamp: str | None = None) -> Path | None:
    """``--force``: move an existing run directory to ``<run dir>.bak-<timestamp>``.

    Returns the backup path, or None when there was nothing to move. Never deletes anything;
    a name collision (two forces within one second) gets a ``-2``, ``-3``... suffix.
    """
    if not run_dir.exists():
        return None
    stamp = timestamp or time.strftime("%Y%m%d-%H%M%S")
    backup = Path(f"{run_dir}.bak-{stamp}")
    suffix = 1
    while backup.exists():
        suffix += 1
        backup = Path(f"{run_dir}.bak-{stamp}-{suffix}")
    shutil.move(os.fspath(run_dir), os.fspath(backup))
    return backup


def results_root() -> Path:
    """``PPSI_RESULTS_ROOT`` re-read per call (tests point it at a temporary tree)."""
    import ppsi_paths

    value = os.environ.get("PPSI_RESULTS_ROOT")
    return Path(value) if value else ppsi_paths.RESULTS_ROOT


def runs_root() -> Path:
    """``PPSI_RUNS_ROOT`` re-read per call."""
    import ppsi_paths

    value = os.environ.get("PPSI_RUNS_ROOT")
    return Path(value) if value else ppsi_paths.RUNS_ROOT


def unit_run_dir(unit: Unit) -> Path:
    """The unit's run directory in the results tree (where the curated files and the
    extracted checkpoints live)."""
    return results_root() / unit.stage / "results" / "runs" / unit.run
