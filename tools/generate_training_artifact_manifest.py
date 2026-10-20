"""Generate or verify the S1-PR-05 trainer artifact manifests.

Generation must happen only after code, tests, documentation and the canonical
Flower smoke summary are final. Verification is read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import ppsi_paths

# Kept from the flat layout; the installed package map makes ppsi and scripts importable.
_REPO_ROOT = ppsi_paths.REPO_ROOT
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ppsi.training.identity import (
    DEFAULT_TRAINER_CORE_FILES,
    build_trainer_core_manifest,
    canonical_manifest_bytes,
    file_sha256,
    verify_trainer_core_manifest,
)

REPO_ROOT = ppsi_paths.REPO_ROOT
EVIDENCE_DIR = ppsi_paths.prototype_path("docs/evidence/s1-pr-05", REPO_ROOT)
TRAINER_MANIFEST_PATH = EVIDENCE_DIR / "shared_trainer_core_manifest.v1.json"
ARTIFACT_MANIFEST_PATH = EVIDENCE_DIR / "artifact_manifest.v1.json"
SMOKE_SUMMARY_PATH = EVIDENCE_DIR / "unified_trainer_smoke_summary.v1.json"

ARTIFACTS: dict[str, str] = {
    "phase1_batch_v1": "ppsi/training/batch.py",
    "raw_output_contract": "ppsi/training/outputs.py",
    "phase1_model_protocol": "ppsi/training/protocol.py",
    "shared_state_spec": "ppsi/training/state.py",
    "unified_trainer_core": "ppsi/training/core.py",
    "trainer_identity": "ppsi/training/identity.py",
    "centralized_adapter": "ppsi/training/centralized.py",
    "flower_adapter": "ppsi/training/flower.py",
    "checkpoint_v1": "ppsi/training/checkpoint.py",
    "deterministic_sampler": "ppsi/training/sampler.py",
    "stub_initialization_bridge": "ppsi/training/initialization.py",
    "trainer_result_bridge": "ppsi/training/result.py",
    "contract_smoke_objective": "ppsi/training/objective.py",
    "contract_smoke_fixtures": "ppsi/training/fixtures.py",
    "contract_stub_model": "ppsi/training/stub_model.py",
    "federated_smoke_runtime": "scripts/federated/fl_synthetic_smoke.py",
    "unified_trainer_smoke_config": "config/unified_trainer_smoke.v1.json",
    "unified_trainer_smoke_entrypoint": "scripts/training_smoke.py",
    "unified_trainer_smoke_summary": (
        "docs/evidence/s1-pr-05/unified_trainer_smoke_summary.v1.json"
    ),
    "shared_trainer_core_manifest": ("docs/evidence/s1-pr-05/shared_trainer_core_manifest.v1.json"),
    "unified_trainer_docs": "docs/training-interface.md",
    "unified_trainer_tests": "tests/training",
}

_IGNORED_TREE_PARTS = frozenset({"__pycache__", ".pytest_cache"})
_IGNORED_SUFFIXES = frozenset({".pyc", ".pyo"})
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_PATHS = tuple(
    sorted(
        {path for path in ARTIFACTS.values() if not path.startswith("docs/evidence/")} | {"uv.lock"}
    )
)


def _is_hashable_file(path: Path) -> bool:
    return (
        path.is_file()
        and not any(part in _IGNORED_TREE_PARTS for part in path.parts)
        and path.suffix not in _IGNORED_SUFFIXES
    )


def trainer_core_manifest() -> dict[str, object]:
    """The shared trainer core manifest of the current source.

    `build_trainer_core_manifest` joins the prototype-layout names (`ppsi/training/core.py`) onto a
    root, and `identity.py` is itself one of the hashed files, so it stays unchanged and runs on a
    view of those files laid out the old way. The manifest is the same as over the real files.
    """
    with ppsi_paths.prototype_root(DEFAULT_TRAINER_CORE_FILES) as root:
        return build_trainer_core_manifest(root)


def sha256_tree(path: Path, *, root: Path = REPO_ROOT) -> str:
    """Hash a tracked-style tree while excluding interpreter/test caches.

    Files are named by their prototype-layout path (`tests/training/...`), as the manifest records them.
    """

    digest = hashlib.sha256()
    if root.resolve() == REPO_ROOT.resolve():
        relative_tree = path.resolve().relative_to(root.resolve()).as_posix()
        tracked = subprocess.check_output(
            ["git", "ls-files", "-z", "--", relative_tree],
            cwd=root,
            text=True,
            encoding="utf-8",
        )
        files = sorted(
            root / relative
            for relative in tracked.split("\0")
            if relative and _is_hashable_file(root / relative)
        )
    else:
        files = sorted(item for item in path.rglob("*") if _is_hashable_file(item))
    for file_path in files:
        relative = ppsi_paths.prototype_name(file_path, root).as_posix().encode("utf-8")
        content_hash = file_sha256(file_path).encode("ascii")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(content_hash)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _load_json(path: Path) -> dict:
    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"{path}: duplicate JSON key {key!r}")
            value[key] = item
        return value

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate_keys,
    )
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


def _canonical_json_sha256(value: dict[str, object]) -> str:
    return hashlib.sha256(canonical_manifest_bytes(value)).hexdigest()


def _require_source_git_sha(source_git_sha: object) -> str:
    if not isinstance(source_git_sha, str) or not _GIT_SHA_RE.fullmatch(source_git_sha):
        raise ValueError("source_git_sha must be 40 lowercase hex characters")
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{source_git_sha}^{{commit}}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError("source_git_sha does not identify an available Git commit")
    source_diff = subprocess.run(
        ["git", "diff", "--quiet", source_git_sha, "--", *_SOURCE_PATHS],
        cwd=REPO_ROOT,
        check=False,
    )
    if source_diff.returncode == 1:
        raise ValueError("source_git_sha does not contain the current smoke source")
    if source_diff.returncode != 0:
        raise RuntimeError("Unable to compare source_git_sha with the current smoke source")
    return source_git_sha


def validate_artifact_records(records: object) -> dict[str, dict[str, str]]:
    if not isinstance(records, dict):
        raise TypeError("Artifact manifest records must be an object")
    if set(records) != set(ARTIFACTS):
        missing = sorted(set(ARTIFACTS) - set(records))
        unknown = sorted(set(records) - set(ARTIFACTS))
        raise ValueError(f"Artifact logical ID set mismatch: missing={missing}, unknown={unknown}")

    paths: list[str] = []
    validated: dict[str, dict[str, str]] = {}
    for logical_id, expected_path in sorted(ARTIFACTS.items()):
        record = records[logical_id]
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise ValueError(f"Malformed artifact record: {logical_id}")
        if not isinstance(record["path"], str):
            raise TypeError(f"Artifact path must be a string for {logical_id}")
        if not isinstance(record["sha256"], str) or len(record["sha256"]) != 64:
            raise ValueError(f"Invalid artifact SHA-256 for {logical_id}")
        int(record["sha256"], 16)
        paths.append(record["path"])
        validated[logical_id] = record
    if len(paths) != len(set(paths)):
        raise ValueError("Artifact manifest contains duplicate canonical paths")
    for logical_id, expected_path in sorted(ARTIFACTS.items()):
        if validated[logical_id]["path"] != expected_path:
            raise ValueError(
                f"Artifact path mismatch for {logical_id}: "
                f"{validated[logical_id]['path']} != {expected_path}"
            )
    return validated


def validate_smoke_summary(summary: dict) -> None:
    if summary.get("schema") != "unified_trainer_smoke_summary_v1":
        raise ValueError("Smoke summary schema mismatch")
    if summary.get("version") != "1":
        raise ValueError("Smoke summary version mismatch")
    if summary.get("status") != "PASS":
        raise ValueError("Smoke summary status is not PASS")
    if summary.get("seed") != 13:
        raise ValueError("Smoke summary seed must be 13")

    identities = summary.get("identities")
    if not isinstance(identities, dict):
        raise TypeError("Smoke summary identities must be an object")
    for field in (
        "config_sha256",
        "input_fixture_sha256",
        "objective_config_sha256",
        "shared_trainer_core_sha256",
    ):
        value = identities.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"Smoke summary missing/invalid identities.{field}")
        int(value, 16)
    _require_source_git_sha(identities.get("source_git_sha"))

    config = _load_json(REPO_ROOT / "config/unified_trainer_smoke.v1.json")
    expected_identities = {
        "config_sha256": _canonical_json_sha256(config),
        "input_fixture_sha256": _canonical_json_sha256(
            {
                "schema": "deterministic_phase1_fixture_v1",
                "version": "1",
                "seed": 13,
                "client_batch_sizes": {"0": 2, "1": 4},
            }
        ),
        "objective_config_sha256": _canonical_json_sha256(
            {
                "schema": "contract_smoke_objective_config_v1",
                "version": "1",
                "objective_id": "contract_smoke_objective_v1_NON_SCIENTIFIC",
                "scientific": False,
            }
        ),
        "shared_trainer_core_sha256": trainer_core_manifest()["sha256"],
    }
    for field, expected in expected_identities.items():
        if identities[field] != expected:
            raise ValueError(f"Smoke summary identities.{field} does not match current source")

    initialization = summary.get("initialization")
    if not isinstance(initialization, dict):
        raise TypeError("Smoke summary initialization must be an object")
    if initialization.get("artifact_kind") != "FIXTURE_PROOF":
        raise ValueError("Smoke initialization must be FIXTURE_PROOF")
    for field in ("state_sha256", "model_config_sha256"):
        value = initialization.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"Smoke summary missing/invalid initialization.{field}")
        int(value, 16)
    if (
        not isinstance(initialization.get("state_size_bytes"), int)
        or initialization["state_size_bytes"] <= 0
    ):
        raise ValueError("Smoke initialization state_size_bytes must be positive")

    environment = summary.get("environment")
    if not isinstance(environment, dict):
        raise TypeError("Smoke summary environment must be an object")
    for field in ("python", "torch", "flower", "ray", "platform", "uv_lock_sha256"):
        if not environment.get(field):
            raise ValueError(f"Smoke summary missing environment.{field}")
    lock_sha = environment["uv_lock_sha256"]
    if not isinstance(lock_sha, str) or len(lock_sha) != 64:
        raise ValueError("Smoke summary uv_lock_sha256 must be 64 hex characters")
    int(lock_sha, 16)
    if lock_sha != file_sha256(REPO_ROOT / "uv.lock"):
        raise ValueError("Smoke summary uv_lock_sha256 does not match uv.lock")

    parity = summary.get("aggregation_parity")
    if not isinstance(parity, dict):
        raise TypeError("Smoke summary aggregation_parity must be an object")
    if parity.get("atol") != 1e-6 or parity.get("rtol") != 0.0:
        raise ValueError("Smoke summary aggregation tolerance must be atol=1e-6, rtol=0.0")
    max_abs_diff = parity.get("max_abs_diff")
    if (
        not isinstance(max_abs_diff, (int, float))
        or isinstance(max_abs_diff, bool)
        or not math.isfinite(float(max_abs_diff))
        or not 0 <= float(max_abs_diff) <= 1e-6
    ):
        raise ValueError("Smoke summary max_abs_diff must be finite and within tolerance")

    if config.get("num_clients") != 2:
        raise ValueError("Smoke config must use exactly clients 0 and 1")

    repetitions = summary.get("repetitions")
    if not isinstance(repetitions, list) or len(repetitions) != config["repeat_runs"]:
        raise ValueError("Smoke summary repetition count does not match config")

    expected_client_ids = [0, 1]
    expected_selection_trace = [expected_client_ids for _ in range(config["num_rounds"])]
    round_max_abs_diffs = []
    reference_digest: str | None = None
    reference_client_ids: object | None = None
    for run_index, repetition in enumerate(repetitions):
        if not isinstance(repetition, dict):
            raise TypeError("Smoke repetition must be an object")
        if repetition.get("run") != run_index:
            raise ValueError("Smoke repetition run indices must be consecutive from zero")
        trace = repetition.get("tracing_log")
        if not isinstance(trace, list) or len(trace) != config["num_rounds"]:
            raise ValueError("Smoke trace round count does not match config")
        if repetition.get("selected_client_ids") != expected_selection_trace:
            raise ValueError("Selected client trace must contain exactly clients 0 and 1")

        for expected_round, record in enumerate(trace, start=1):
            if not isinstance(record, dict):
                raise TypeError("Smoke trace record must be an object")
            if record.get("round") != expected_round:
                raise ValueError("Smoke trace rounds must be consecutive from one")
            server_input_digest = record.get("server_input_digest")
            aggregated_digest = record.get("aggregated_digest")
            for field, digest_value in (
                ("server_input_digest", server_input_digest),
                ("aggregated_digest", aggregated_digest),
            ):
                if not isinstance(digest_value, str) or not _SHA256_RE.fullmatch(digest_value):
                    raise ValueError(f"Round {field} must be 64 lowercase hex characters")
            if expected_round > 1 and server_input_digest != trace[expected_round - 2].get(
                "aggregated_digest"
            ):
                raise ValueError("Smoke trace breaks the aggregated-state redistribution chain")
            if record.get("aggregation_oracle_pass") is not True:
                raise ValueError("FedAvg aggregation oracle did not pass every round")
            if record.get("selected_client_count") != config["num_clients"]:
                raise ValueError("Each round must select exactly two clients")
            if record.get("selected_logical_ids") != expected_client_ids:
                raise ValueError("Each round must select exactly clients 0 and 1")
            clients = record.get("clients")
            if not isinstance(clients, list) or len(clients) != 2:
                raise ValueError("Each round must contain exactly two client records")
            if not all(isinstance(client, dict) for client in clients):
                raise TypeError("Smoke client record must be an object")
            client_record_ids = [client.get("logical_client_id") for client in clients]
            if any(type(client_id) is not int for client_id in client_record_ids):
                raise ValueError("Client records must contain integer logical client IDs")
            if client_record_ids != expected_client_ids:
                raise ValueError("Client records must contain clients 0 and 1 in order")
            for client in clients:
                received_digest = client.get("received_digest")
                for field in ("received_digest", "updated_digest"):
                    digest_value = client.get(field)
                    if not isinstance(digest_value, str) or not _SHA256_RE.fullmatch(digest_value):
                        raise ValueError(f"Client {field} must be 64 lowercase hex characters")
                if received_digest != server_input_digest:
                    raise ValueError("Client received_digest must match the round server input")

            round_max_abs_diff = record.get("max_abs_diff")
            if (
                not isinstance(round_max_abs_diff, (int, float))
                or isinstance(round_max_abs_diff, bool)
                or not math.isfinite(float(round_max_abs_diff))
                or not 0 <= float(round_max_abs_diff) <= 1e-6
            ):
                raise ValueError("Round max_abs_diff must be finite and within tolerance")
            round_max_abs_diffs.append(float(round_max_abs_diff))

        if not all(record.get("redistribution_pass") is True for record in trace[1:]):
            raise ValueError("Aggregated state was not redistributed after the first round")
        loss_history = repetition.get("global_loss_history")
        if not isinstance(loss_history, list) or len(loss_history) != config["num_rounds"] + 1:
            raise ValueError("Each repetition requires initial + one loss per round")
        loss_rounds = []
        for item in loss_history:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError("Malformed global loss history entry")
            if not isinstance(item[1], (int, float)) or isinstance(item[1], bool):
                raise TypeError("Global loss must be numeric")
            if not math.isfinite(float(item[1])):
                raise ValueError("Global loss must be finite")
            loss_rounds.append(item[0])
        if any(type(round_id) is not int for round_id in loss_rounds) or loss_rounds != list(
            range(config["num_rounds"] + 1)
        ):
            raise ValueError("Global loss round IDs must run from zero through num_rounds")

        initial_digest = trace[0].get("server_input_digest")
        final_digest = repetition.get("final_model_digest")
        if not isinstance(final_digest, str) or not _SHA256_RE.fullmatch(final_digest):
            raise ValueError("Final model digest must be 64 lowercase hex characters")
        if final_digest != trace[-1].get("aggregated_digest"):
            raise ValueError("Final model digest must match the final aggregated digest")
        if initial_digest == final_digest:
            raise ValueError("Smoke run did not produce a non-zero global model change")
        client_ids = repetition.get("selected_client_ids")
        if reference_digest is None:
            reference_digest = str(final_digest)
            reference_client_ids = client_ids
        else:
            if final_digest != reference_digest:
                raise ValueError("Final model digest differs across repetitions")
            if client_ids != reference_client_ids:
                raise ValueError("Selected logical client order differs across repetitions")

    if float(max_abs_diff) != max(round_max_abs_diffs):
        raise ValueError("Smoke summary max_abs_diff does not match measured round maximum")


def generate() -> None:
    if not SMOKE_SUMMARY_PATH.is_file():
        raise FileNotFoundError(ppsi_paths.prototype_name(SMOKE_SUMMARY_PATH, REPO_ROOT))
    validate_smoke_summary(_load_json(SMOKE_SUMMARY_PATH))

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    trainer_manifest = trainer_core_manifest()
    write_json(TRAINER_MANIFEST_PATH, trainer_manifest)

    records = {}
    for logical_id, relative_path in sorted(ARTIFACTS.items()):
        path = ppsi_paths.prototype_path(relative_path, REPO_ROOT)
        if not path.exists():
            raise FileNotFoundError(relative_path)
        records[logical_id] = {
            "path": relative_path,
            "sha256": sha256_tree(path) if path.is_dir() else file_sha256(path),
        }

    manifest = {
        "schema": "unified_trainer_artifact_manifest_v1",
        "version": "1",
        "task_id": "S1-PR-05",
        "artifacts": records,
    }
    write_json(ARTIFACT_MANIFEST_PATH, manifest)
    print(f"Generated {ppsi_paths.prototype_name(TRAINER_MANIFEST_PATH, REPO_ROOT)}")
    print(f"Generated {ppsi_paths.prototype_name(ARTIFACT_MANIFEST_PATH, REPO_ROOT)}")


def verify() -> None:
    trainer_manifest = _load_json(TRAINER_MANIFEST_PATH)
    with ppsi_paths.prototype_root(DEFAULT_TRAINER_CORE_FILES) as root:
        verify_trainer_core_manifest(root, trainer_manifest)

    manifest = _load_json(ARTIFACT_MANIFEST_PATH)
    if manifest.get("schema") != "unified_trainer_artifact_manifest_v1":
        raise ValueError("Artifact manifest schema mismatch")
    if manifest.get("version") != "1" or manifest.get("task_id") != "S1-PR-05":
        raise ValueError("Artifact manifest identity mismatch")
    records = validate_artifact_records(manifest.get("artifacts"))

    for logical_id, record in sorted(records.items()):
        path = ppsi_paths.prototype_path(str(record["path"]), REPO_ROOT)
        if not path.exists():
            raise FileNotFoundError(record["path"])
        actual = sha256_tree(path) if path.is_dir() else file_sha256(path)
        if actual != record["sha256"]:
            raise ValueError(
                f"Artifact SHA-256 mismatch for {logical_id}: {actual} != {record['sha256']}"
            )

    validate_smoke_summary(_load_json(SMOKE_SUMMARY_PATH))
    print("S1-PR-05 MANIFEST VERIFICATION: PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Read-only verification; do not generate or rewrite evidence.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.verify:
        verify()
    else:
        generate()


if __name__ == "__main__":
    main()
