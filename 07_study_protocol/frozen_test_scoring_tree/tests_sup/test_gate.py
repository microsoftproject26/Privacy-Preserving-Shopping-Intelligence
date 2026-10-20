"""A phase whose prerequisite hash is wrong never starts -- this is checked at two levels:
`check_gates` itself (unit level) and, in test_runner_retry.py, through `Supervisor.run_phase`
(no subprocess ever spawned, no log file ever created) when a gate is missing or mismatched.
"""
from __future__ import annotations

import hashlib

from v4sup.gate import check_gates
from v4sup.registry import GateFile


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_all_gates_present_and_matching_is_ok(tmp_path):
    f = tmp_path / "gate.json"
    f.write_bytes(b'{"ok": true}')
    res = check_gates((GateFile(path=str(f), sha256=_sha(f.read_bytes())),))
    assert res.ok
    assert res.missing == () and res.mismatched == ()


def test_missing_gate_file_never_ready(tmp_path):
    f = tmp_path / "absent.json"
    res = check_gates((GateFile(path=str(f), sha256="a" * 64),))
    assert not res.ok
    assert str(f) in res.missing


def test_wrong_hash_never_ready(tmp_path):
    f = tmp_path / "gate.json"
    f.write_bytes(b'{"ok": true}')
    wrong = "0" * 64
    res = check_gates((GateFile(path=str(f), sha256=wrong),))
    assert not res.ok
    assert str(f) in res.mismatched
    assert str(f) not in res.missing


def test_one_bad_gate_among_several_fails_all(tmp_path):
    good = tmp_path / "good.json"
    good.write_bytes(b"{}")
    bad = tmp_path / "bad.json"
    bad.write_bytes(b"{}")
    res = check_gates((GateFile(path=str(good), sha256=_sha(good.read_bytes())),
                       GateFile(path=str(bad), sha256="f" * 64)))
    assert not res.ok
    assert str(bad) in res.mismatched
