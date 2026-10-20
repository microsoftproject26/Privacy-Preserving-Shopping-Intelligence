"""A read-only probe CLI for the lead: one line of JSON, and it never writes anything."""
from __future__ import annotations

import json

from v4sup.probe import probe_line
from v4sup.status import write_status_atomic


def test_probe_is_one_json_line_and_never_writes(tmp_path):
    status_path = tmp_path / "status.json"
    write_status_atomic(status_path, {"phases": {"p1": {"state": "DONE"}, "p2": {"state": "RUNNING"}},
                                      "heartbeat_utc": "2026-01-01T00:00:00Z"})
    before = status_path.read_bytes()
    line = probe_line(status_path)
    assert "\n" not in line
    parsed = json.loads(line)  # must be valid single-line JSON
    assert parsed["overall"] == "IN_PROGRESS"
    assert parsed["phases"] == {"p1": "DONE", "p2": "RUNNING"}
    assert status_path.read_bytes() == before  # probe wrote nothing


def test_probe_not_started(tmp_path):
    line = probe_line(tmp_path / "never_written.json")
    parsed = json.loads(line)
    assert parsed["overall"] == "NOT_STARTED"


def test_probe_all_done(tmp_path):
    status_path = tmp_path / "status.json"
    write_status_atomic(status_path, {"phases": {"p1": {"state": "DONE"}, "p2": {"state": "DONE"}}})
    parsed = json.loads(probe_line(status_path))
    assert parsed["overall"] == "DONE"


def test_probe_any_failed_is_failed_overall(tmp_path):
    status_path = tmp_path / "status.json"
    write_status_atomic(status_path, {"phases": {"p1": {"state": "DONE"}, "p2": {"state": "FAILED"}}})
    parsed = json.loads(probe_line(status_path))
    assert parsed["overall"] == "FAILED"
