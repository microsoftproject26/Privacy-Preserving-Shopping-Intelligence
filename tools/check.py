"""Run exactly what CI runs, in the same order, before you push.

    uv run python tools/check.py

Ordered cheapest first, so the mistake that costs the least to fix is also the one you
hear about first. Checking the lockfile takes about a quarter of a second; discovering
the same problem from a red X takes a push, a queue and three minutes, and it does not
say what to do about it.

Add `--fast` to stop before the test suite when you only want the quick gates.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Ruff runs over every stage's code, scripts and tests folders and the repository tools. The
# research trees among them are byte-identical to the reviewed server copies, so pyproject.toml
# limits ruff to syntax errors there (see [tool.ruff.lint.per-file-ignores]).
LINT_GLOBS = ("0*/code", "0*/scripts", "0*/tests", "tools")


def lint_paths() -> tuple[str, ...]:
    return tuple(
        path.relative_to(REPO_ROOT).as_posix()
        for pattern in LINT_GLOBS
        for path in sorted(REPO_ROOT.glob(pattern))
        if path.is_dir()
    )


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    command: tuple[str, ...]
    on_failure: str
    slow: bool = False


CHECKS: tuple[Check, ...] = (
    Check(
        name="lockfile is current",
        command=("uv", "lock", "--check"),
        on_failure=(
            "pyproject.toml changed but uv.lock was not regenerated.\n"
            "  Run:  uv lock\n"
            "  Then commit pyproject.toml and uv.lock together. A lock that arrives in a\n"
            "  later commit leaves every commit in between uninstallable."
        ),
    ),
    Check(
        name="package map is current",
        command=(sys.executable, "tools/list_packages.py", "--check"),
        on_failure=(
            "A package folder was added, moved or removed, and the package map in\n"
            "  pyproject.toml no longer matches the stage folders.\n"
            "  Run:  uv run python tools/list_packages.py --write\n"
            "  Then: uv lock, and commit pyproject.toml and uv.lock together."
        ),
    ),
    Check(
        name="environment matches the lock",
        command=("uv", "sync", "--locked", "--group", "dev"),
        on_failure=(
            "The locked environment could not be installed.\n"
            "  Run:  uv python install\n"
            "  Then re-run this script."
        ),
    ),
    Check(
        name="lint",
        command=("uv", "run", "--locked", "ruff", "check", *lint_paths()),
        on_failure=(
            "Ruff found problems. Most are mechanical.\n"
            "  Run:  uv run --locked ruff check 0*/code 0*/scripts 0*/tests tools --fix\n"
            "  Then look at whatever is left. Ruff enforces Python 3.11.14, so syntax from a\n"
            "  newer Python parses on your interpreter and still fails here."
        ),
    ),
    Check(
        name="every package imports",
        command=("uv", "run", "--locked", "python", "tools/list_packages.py", "--import"),
        on_failure=(
            "A package in the package map does not import from the installed environment.\n"
            "  The lines above name it and the error. Run tools/list_packages.py --check\n"
            "  first: a package missing from pyproject.toml imports from nowhere."
        ),
    ),
    Check(
        name="contract smoke",
        command=("uv", "run", "--locked", "python", "tools/prototype_view.py", "contract-smoke"),
        on_failure=(
            "A frozen schema or identity no longer validates.\n"
            "  Read the contract it names before changing anything: a frozen contract that\n"
            "  stopped validating is usually the tripwire working, not a broken check."
        ),
    ),
    Check(
        name="environment smoke",
        command=("uv", "run", "--locked", "python", "tools/env_smoke.py"),
        on_failure=(
            "The pinned interpreter or Torch is not usable.\n"
            "  Run:  uv python install\n"
            "  Then re-run this script."
        ),
        slow=True,
    ),
    Check(
        name="tests",
        command=("uv", "run", "--locked", "python", "-m", "pytest", "-q", "-n", "auto"),
        on_failure=(
            "A prototype test failed (testpaths in pyproject.toml lists the stage folders).\n"
            "  Re-run just the failing file to iterate:\n"
            "  uv run --locked python -m pytest -q <path>"
        ),
        slow=True,
    ),
)


def run(check: Check) -> tuple[bool, float]:
    started = time.perf_counter()
    completed = subprocess.run(check.command, cwd=REPO_ROOT, check=False)
    return completed.returncode == 0, time.perf_counter() - started


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fast", action="store_true", help="stop before the slow checks"
    )
    args = parser.parse_args(argv)

    selected = [check for check in CHECKS if not (args.fast and check.slow)]

    for position, check in enumerate(selected, start=1):
        print(f"\n[{position}/{len(selected)}] {check.name}", flush=True)
        passed, seconds = run(check)
        if not passed:
            print(f"\nFAILED: {check.name}  ({seconds:.1f}s)")
            print(f"\n{check.on_failure}\n")
            print("CI runs these in this order and stops at the first failure, so a later")
            print("check being untouched does not mean it would pass.")
            return 1
        print(f"      ok ({seconds:.1f}s)")

    print("\nAll checks passed." if not args.fast else "\nFast checks passed. Tests not run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
