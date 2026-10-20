"""ppsi_pipeline - the run orchestrator behind ``run_pipeline.py`` and each stage's ``run.py``.

The command line is the same at the repository root and in every stage folder; the root runs
stages 01..06 in order while a stage's ``run.py`` pins itself to its own stage:

* ``--mode auto`` (the default) plans every training unit of the selected stages as **skip**
  (a complete ``RUN_SUMMARY.json``), **resume** (an incomplete run with a resumable checkpoint
  and a launcher that supports it) or **start**, prints the plan as a table, and then executes
  it. ``--dry-run`` prints the plan and exits.
* ``--force`` moves the existing outputs of the selected units aside to
  ``<run dir>.bak-<timestamp>`` and starts fresh; nothing is deleted.
* ``--from-results`` rebuilds the stage's summary tables (CSV + Markdown) and figures
  (PNG + PDF) from the files under ``<stage>/results/`` only - no GPU, no data, no training.

The main-study launchers are hash-pinned to the code trees recorded in ``07_study_protocol``
(the flat ``ppsi_v4`` tree; the closure-phase extensions tree). When a unit would start or
resume outside its recorded tree, the orchestrator says so and moves on instead of launching a
run that the launcher itself would refuse.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .decide import (Action, RunState, decide, move_aside, read_run_state, results_root,
                     runs_root, unit_run_dir)
from .report import run_from_results
from .units import REGISTRY, STAGE_NOTES, STAGE_ORDER, TREE_EXTENSIONS, Unit

PIN_HELP = ("07_study_protocol/deployment_manifests/ (frozen server trees) and "
            "07_study_protocol/handover/07_SERVER_AND_OPERATIONS.md (which tree ran what)")


@dataclass(frozen=True)
class PlannedUnit:
    """One unit of the plan: the decision and, with ``--force``, the moved-aside backup."""

    unit: Unit
    state: RunState
    action: Action
    reason: str
    backup: Path | None = None


def tree_root(tree: str) -> Path:
    """The root of the code tree a unit's launcher is pinned to (``PPSI_PROJECT_ROOT`` or
    ``PPSI_EXTENSIONS_ROOT``, re-read per call)."""
    import ppsi_paths

    env = "PPSI_EXTENSIONS_ROOT" if tree == TREE_EXTENSIONS else "PPSI_PROJECT_ROOT"
    default = (ppsi_paths.EXTENSIONS_ROOT if tree == TREE_EXTENSIONS
               else ppsi_paths.PROJECT_ROOT)
    value = os.environ.get(env)
    return Path(value) if value else default


def render_command(unit: Unit, resume: bool) -> str:
    """The unit's command with the server-side run directory filled in; ``--resume`` is
    appended exactly when the plan says resume (the flag exists in every resumable launcher,
    except the lane supervisors, whose per-job argv already carries it)."""
    run_dir = (runs_root() / unit.server_run).as_posix() if unit.server_run else None
    command = unit.command.replace("{run_dir}", run_dir or "<run dir>")
    if (resume and unit.resume != "none" and "--resume" not in command
            and not command.startswith("bash ")):
        command += f" {unit.resume}"
    return command


def _launcher_rel(command: str) -> str | None:
    """The script (or module) path inside the command template: ``code/v4int/launch_fl.py``,
    ``laneX_COLD.sh``, or ``experiment_extensions/pretraining/pretrain_resume.py`` for -m."""
    tokens = shlex.split(command)
    if "-m" in tokens:
        return tokens[tokens.index("-m") + 1].replace(".", "/") + ".py"
    scripts = [token for token in tokens if token.endswith((".py", ".sh"))]
    return scripts[0] if scripts else None


def build_plan(stages: Sequence[str], force: bool = False) -> list[PlannedUnit]:
    """Decide skip / resume / start for every unit of `stages` (applying ``--force`` first)."""
    plan: list[PlannedUnit] = []
    for stage in stages:
        for unit in REGISTRY.get(stage, ()):
            run_dir = unit_run_dir(unit)
            backup = None
            if force:
                backup = move_aside(run_dir)
            state = read_run_state(run_dir, unit.summary_names, unit.resumable)
            action, reason = decide(state, unit.resume)
            if backup is not None:
                reason = f"forced fresh start (previous outputs in {backup.name})"
                action = Action.START
            plan.append(PlannedUnit(unit=unit, state=state, action=action, reason=reason,
                                    backup=backup))
    return plan


def print_plan(plan: Sequence[PlannedUnit], stages: Sequence[str]) -> None:
    """The plan as a fixed-width table (stage, unit, tree, action, why)."""
    print(f"results root: {results_root()}")
    print(f"runs root:    {runs_root()}")
    counts = {action.value: 0 for action in Action}
    for item in plan:
        counts[item.action.value] += 1
    summary = ", ".join(f"{count} {name}" for name, count in counts.items())
    print(f"plan for {len(stages)} stage(s), {len(plan)} unit(s): {summary}")
    if not plan:
        for stage in stages:
            if stage in STAGE_NOTES:
                print(f"{stage}: {STAGE_NOTES[stage]}")
        return
    rows = [(item.unit.stage, item.unit.id.split("/", 1)[1], item.unit.tree,
             item.action.value.upper(), item.reason) for item in plan]
    widths = [max(len(str(cell)) for cell in column) for column in zip(*rows)]
    header = ("stage", "unit", "tree", "action", "why")
    widths = [max(w, len(h)) for w, h in zip(widths, header)]
    line = "  ".join(cell.ljust(w) for cell, w in zip(header, widths))
    print(line)
    print("-" * len(line))
    for row in rows:
        print("  ".join(str(cell).ljust(w) for cell, w in zip(row, widths)))


def _blocked(message: str) -> None:
    print(f"    not launched: {message}")


def execute(plan: Sequence[PlannedUnit]) -> int:
    """Run the START / RESUME units of an approved plan. Units pinned to a code tree this copy
    does not have are reported and skipped, never crashed; returns 1 only when a launched
    process itself fails."""
    failures = 0
    for item in plan:
        if item.action is Action.SKIP:
            continue
        unit, resume = item.unit, item.action is Action.RESUME
        command = render_command(unit, resume)
        root = tree_root(unit.tree)
        print(f"{unit.id}: {item.action.value} -> {command}")
        launcher = _launcher_rel(unit.command)
        if launcher is None or not (root / launcher).exists():
            which = "original flat ppsi_v4" if unit.tree != TREE_EXTENSIONS else "extensions"
            _blocked(f"its launcher is hash-pinned to the {which} code tree, and "
                     f"{launcher or 'the launcher script'} is not under {root}. Run it from "
                     f"the tree recorded in {PIN_HELP} - a moved or edited tree is refused by "
                     "the launcher's own SHA-256 pins")
            continue
        if "<" in command:
            _blocked("the arm-specific flag values are <...> placeholders pinned in the run's "
                     "sealed records (LAUNCH_*.json / ARM_SPEC.json); fill them in and rerun")
            continue
        result = subprocess.run(shlex.split(command), cwd=root)
        if result.returncode != 0:
            failures += 1
            print(f"    launcher exited {result.returncode}")
    return 1 if failures else 0


def _select_stages(values: Sequence[str]) -> list[str]:
    """Map stage arguments (``03``, ``3``, ``03_federated_learning``) to stage folder names."""
    selected: list[str] = []
    for value in values:
        key = value if len(value) > 2 else value.zfill(2)
        matches = [s for s in STAGE_ORDER if s == value or s.startswith(key + "_")]
        if not matches:
            raise SystemExit(f"unknown stage {value!r} (expected one of {', '.join(STAGE_ORDER)})")
        selected += matches
    return list(dict.fromkeys(selected)) or list(STAGE_ORDER)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_pipeline", description=__doc__.splitlines()[0],
        epilog="Skip / resume / start is decided on the results tree (PPSI_RESULTS_ROOT); "
               "launchers run from the code trees recorded in 07_study_protocol.")
    parser.add_argument("--stages", nargs="+", metavar="STAGE",
                        help="stage folders to run (01..06 by name or number); default: all, "
                             "in order - a stage's run.py only manages its own stage")
    parser.add_argument("--mode", choices=("auto",), default="auto",
                        help="auto: skip complete units, resume where the launcher supports "
                             "it, start the rest (the only mode)")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    parser.add_argument("--force", action="store_true",
                        help="move existing outputs of the selected units aside to "
                             "<run dir>.bak-<timestamp> and start fresh")
    parser.add_argument("--from-results", action="store_true",
                        help="rebuild the stage's tables (CSV+MD) and figures (PNG+PDF) from "
                             "<stage>/results/ only; exit 0 with a clear message when a stage "
                             "has no results")
    return parser


def main(argv: Sequence[str] | None = None, only_stage: str | None = None) -> int:
    """The entry point of ``run_pipeline.py`` and of every stage ``run.py`` (``only_stage``
    pins the stage). Returns 0 on success and on every nothing-to-do path."""
    args = build_parser().parse_args(argv)
    stages = _select_stages(args.stages) if args.stages else (
        [only_stage] if only_stage else list(STAGE_ORDER))
    if only_stage and stages != [only_stage]:
        raise SystemExit(f"this run.py manages {only_stage} only (got --stages "
                         f"{' '.join(args.stages)})")
    if args.from_results:
        return run_from_results(stages, dry_run=args.dry_run)
    plan = build_plan(stages, force=args.force)
    print_plan(plan, stages)
    if args.dry_run:
        return 0
    return execute(plan)


__all__ = ["PlannedUnit", "REGISTRY", "build_parser", "build_plan", "execute", "main",
           "print_plan", "render_command", "run_from_results", "tree_root"]
