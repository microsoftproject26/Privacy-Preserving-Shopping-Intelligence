from __future__ import annotations

import json

import pytest

from v4sup.errors import RegistryError
from v4sup.registry import load_registry

GOOD_SHA = "a" * 64


def write_registry(tmp_path, phases: list) -> "object":
    p = tmp_path / "registry.json"
    p.write_text(json.dumps({"phases": phases}))
    return p


def _basic_phase(**overrides) -> dict:
    d = {"id": "p1", "command": ["python", "-c", "pass"],
         "expected_artifacts": [{"path": "out.json"}], "gpus": [], "planned_peak_gib": 0.1,
         "max_attempts": 2}
    d.update(overrides)
    return d


def test_load_valid_registry(tmp_path):
    reg_path = write_registry(tmp_path, [_basic_phase()])
    reg = load_registry(reg_path)
    assert set(reg) == {"p1"}
    ph = reg["p1"]
    assert ph.command == ("python", "-c", "pass")
    assert ph.max_attempts == 2
    assert ph.expected_artifacts[0].path == "out.json"
    assert ph.expected_artifacts[0].wants_json_strict() is True


def test_missing_id_refused(tmp_path):
    reg_path = write_registry(tmp_path, [_basic_phase(id="")])
    with pytest.raises(RegistryError):
        load_registry(reg_path)


def test_duplicate_id_refused(tmp_path):
    reg_path = write_registry(tmp_path, [_basic_phase(), _basic_phase()])
    with pytest.raises(RegistryError):
        load_registry(reg_path)


def test_both_command_and_shell_command_refused(tmp_path):
    reg_path = write_registry(tmp_path, [_basic_phase(shell_command="echo hi")])
    with pytest.raises(RegistryError):
        load_registry(reg_path)


def test_neither_command_nor_shell_command_refused(tmp_path):
    d = _basic_phase()
    d.pop("command")
    reg_path = write_registry(tmp_path, [d])
    with pytest.raises(RegistryError):
        load_registry(reg_path)


def test_no_expected_artifacts_refused(tmp_path):
    """A phase that declares no output can never be verified DONE, so it must be refused at
    load time rather than silently trusted at run time."""
    d = _basic_phase()
    d.pop("expected_artifacts")
    reg_path = write_registry(tmp_path, [d])
    with pytest.raises(RegistryError):
        load_registry(reg_path)


def test_unknown_depends_on_refused(tmp_path):
    reg_path = write_registry(tmp_path, [_basic_phase(depends_on=["ghost"])])
    with pytest.raises(RegistryError):
        load_registry(reg_path)


def test_malformed_prerequisite_sha_refused(tmp_path):
    d = _basic_phase(prerequisites=[{"path": "gate.json", "sha256": "not-64-hex"}])
    reg_path = write_registry(tmp_path, [d])
    with pytest.raises(RegistryError):
        load_registry(reg_path)


def test_invalid_json_refused(tmp_path):
    p = tmp_path / "registry.json"
    p.write_text("{not json")
    with pytest.raises(RegistryError):
        load_registry(p)


def test_empty_phases_refused(tmp_path):
    reg_path = write_registry(tmp_path, [])
    with pytest.raises(RegistryError):
        load_registry(reg_path)


def test_valid_prerequisite_sha_loads(tmp_path):
    d = _basic_phase(prerequisites=[{"path": "gate.json", "sha256": GOOD_SHA}])
    reg_path = write_registry(tmp_path, [d])
    reg = load_registry(reg_path)
    assert reg["p1"].prerequisites[0].sha256 == GOOD_SHA
