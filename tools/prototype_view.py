"""Run prototype code that must stay byte-identical against a view of the old repository layout.

The S1-PR-03 contract validator (`scripts/experiments/validate_experiment_contracts.py`, now in
`06_evaluation_and_benchmarks/scripts/prototype/experiments/`) finds the repository root from its own
location, and its SHA-256 is one of the hashes in the frozen artifact manifest it verifies. Changing it
to find the stage layout would make it fail its own check. So it runs unchanged against a temporary
copy of the current files laid out the way the prototype had them (`ppsi_paths.prototype_view`).

    uv run python tools/prototype_view.py contract-smoke
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from ppsi_paths import prototype_view

# Everything the validator reads: the frozen schemas and records, the fixtures it validates, and every
# path in config/experiments/experiment_artifact_map.v1.json.
CONTRACT_SMOKE = {
    "script": "scripts/experiments/validate_experiment_contracts.py",
    "paths": (
        "config/experiments",
        "fixtures/experiments",
        "scripts/experiments",
        "docs/experiment-contracts.md",
        "docs/decisions.md",
        "docs/evidence/s1-pr-03",
    ),
}
PRESETS = {"contract-smoke": CONTRACT_SMOKE}


def run(script: str, paths: tuple[str, ...]) -> int:
    with tempfile.TemporaryDirectory(prefix="ppsi_prototype_view_") as tmp:
        view = prototype_view(Path(tmp), paths)
        return subprocess.run(
            [sys.executable, "-X", "utf8", str(view / script)], cwd=view, check=False
        ).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("preset", choices=sorted(PRESETS))
    args = parser.parse_args(argv)
    preset = PRESETS[args.preset]
    return run(preset["script"], preset["paths"])


if __name__ == "__main__":
    raise SystemExit(main())
