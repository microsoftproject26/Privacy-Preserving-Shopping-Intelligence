"""Canonical identity for behavior-defining shared trainer files."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

# Objective implementations and adapters are deliberately excluded.
DEFAULT_TRAINER_CORE_FILES: tuple[str, ...] = (
    "ppsi/training/batch.py",
    "ppsi/training/core.py",
    "ppsi/training/identity.py",
    "ppsi/training/outputs.py",
    "ppsi/training/protocol.py",
    "ppsi/training/state.py",
)

_TEXT_SUFFIXES = frozenset({".py", ".md", ".json", ".toml", ".txt", ".yaml", ".yml"})
_TEXT_FILENAMES = frozenset({"uv.lock"})


def canonical_file_bytes(path: Path) -> bytes:
    """Return platform-independent bytes for tracked UTF-8 text artifacts.

    Git autocrlf must not change scientific trainer identity. Known text files
    are decoded as UTF-8 (accepting an optional BOM), normalized to LF, and
    encoded as plain UTF-8. Unknown/binary files retain exact bytes.
    """

    data = path.read_bytes()
    if path.name not in _TEXT_FILENAMES and path.suffix.lower() not in _TEXT_SUFFIXES:
        return data
    text = data.decode("utf-8-sig")
    return text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(canonical_file_bytes(path)).hexdigest()


def canonical_manifest_bytes(manifest: dict[str, object]) -> bytes:
    import json

    return json.dumps(
        manifest,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def build_trainer_core_manifest(
    repo_root: Path,
    *,
    relative_files: Iterable[str] = DEFAULT_TRAINER_CORE_FILES,
) -> dict[str, object]:
    files = []
    for relative in sorted(relative_files):
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        files.append({"path": relative, "sha256": file_sha256(path)})
    manifest: dict[str, object] = {
        "schema": "shared_trainer_core_manifest_v1",
        "version": "1",
        "text_normalization": "UTF8_LF_NO_BOM",
        "files": files,
    }
    manifest["sha256"] = hashlib.sha256(canonical_manifest_bytes(manifest)).hexdigest()
    return manifest


def verify_trainer_core_manifest(repo_root: Path, manifest: dict[str, object]) -> None:
    expected_fields = {"schema", "version", "text_normalization", "files", "sha256"}
    if set(manifest) != expected_fields:
        raise ValueError("Trainer core manifest fields do not match v1")
    if manifest.get("schema") != "shared_trainer_core_manifest_v1":
        raise ValueError("Trainer core manifest schema mismatch")
    if manifest.get("version") != "1":
        raise ValueError("Trainer core manifest version mismatch")
    stored_sha = manifest.get("sha256")
    without_sha = {key: value for key, value in manifest.items() if key != "sha256"}
    actual_manifest_sha = hashlib.sha256(canonical_manifest_bytes(without_sha)).hexdigest()
    if stored_sha != actual_manifest_sha:
        raise ValueError("Trainer core manifest SHA-256 mismatch")
    if manifest.get("text_normalization") != "UTF8_LF_NO_BOM":
        raise ValueError("Unsupported trainer text-normalization rule")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise TypeError("Trainer core manifest files must be a list")
    paths: list[str] = []
    for record in files:
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise ValueError("Malformed trainer core manifest record")
        relative_path = record["path"]
        if not isinstance(relative_path, str):
            raise TypeError("Trainer core manifest path must be a string")
        paths.append(relative_path)
        path = repo_root / relative_path
        if file_sha256(path) != record["sha256"]:
            raise ValueError(f"Trainer core file SHA mismatch: {record['path']}")
    if len(paths) != len(set(paths)):
        raise ValueError("Trainer core manifest contains duplicate paths")
    if tuple(sorted(paths)) != tuple(sorted(DEFAULT_TRAINER_CORE_FILES)):
        raise ValueError("Trainer core manifest file set does not match v1")
