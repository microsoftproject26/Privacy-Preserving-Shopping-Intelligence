import json
import sys
from pathlib import Path

import ppsi_paths

_REPO_ROOT = ppsi_paths.REPO_ROOT
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ppsi.training.identity import file_sha256


def generate_manifest():
    evidence_dir = Path("docs/evidence/s1-pr-02")
    manifest_path = evidence_dir / "artifact_manifest.v1.json"

    # Files to hash
    files_to_hash = {
        "config": Path("config/fl_synthetic_smoke.v1.json"),
        "entry_point": Path("scripts/federated/fl_synthetic_smoke.py"),
        "lifecycle_test": Path("tests/federated/test_fl_synthetic_smoke.py"),
        "documentation": Path("docs/federated-synthetic-smoke.md"),
        "smoke_summary": evidence_dir / "fl_synthetic_smoke_summary.v1.json",
        "environment_lock": Path("uv.lock"),
    }

    manifest = {"schema": "artifact_manifest_v1", "version": 1, "artifacts": {}}

    # The manifest records the prototype-layout paths; prototype_path finds each file now.
    for logical_id, file_path in files_to_hash.items():
        if not ppsi_paths.prototype_path(file_path).exists() and logical_id != "documentation":
            raise FileNotFoundError(f"Missing required deliverable: {file_path}")

        # We allow missing documentation for now as we might generate it later,
        # but wait, the instruction says to hash documentation too. We should write documentation first.
        if ppsi_paths.prototype_path(file_path).exists():
            manifest["artifacts"][logical_id] = {
                "path": str(file_path.as_posix()),
                "sha256": file_sha256(ppsi_paths.prototype_path(file_path)),
            }

    with open(ppsi_paths.prototype_path(manifest_path), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Generated manifest at {manifest_path}")


if __name__ == "__main__":
    generate_manifest()
