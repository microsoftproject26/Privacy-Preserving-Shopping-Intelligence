"""T46: an interrupted status write leaves the previous valid state. `write_status_atomic` writes
a full payload to a temp file, flushes and fsyncs it, then calls `os.replace` -- if the process (or
in this test, a monkeypatched `os.replace`) is interrupted before that final rename, the real
status file must be exactly what it was before the write was attempted, never a half-written file.
"""
from __future__ import annotations

import json

import pytest

from v4sup.status import append_event, merge_status, read_status, write_status_atomic


def test_write_then_read_roundtrips(tmp_path):
    p = tmp_path / "status.json"
    write_status_atomic(p, {"state": "RUNNING", "n": 1})
    assert read_status(p) == {"state": "RUNNING", "n": 1}


def test_read_missing_status_returns_none(tmp_path):
    assert read_status(tmp_path / "nope.json") is None


def test_interrupted_write_leaves_previous_valid_state(tmp_path, monkeypatch):
    p = tmp_path / "status.json"
    write_status_atomic(p, {"state": "DONE", "phase": "p1"})
    before = read_status(p)

    import v4sup.status as status_mod

    def boom(*a, **kw):
        raise OSError("simulated interruption before atomic replace")

    monkeypatch.setattr(status_mod.os, "replace", boom)
    with pytest.raises(OSError):
        write_status_atomic(p, {"state": "CORRUPTED_IF_THIS_WERE_VISIBLE"})

    after = read_status(p)
    assert after == before  # previous valid state preserved, no half-written file observed
    # no stray temp file left behind either
    leftovers = list(tmp_path.glob("status.json.tmp.*"))
    assert leftovers == []


def test_interrupted_write_to_a_fresh_path_leaves_no_file(tmp_path, monkeypatch):
    """Same interruption, but there was no previous status at all: the real path must still not
    exist afterward -- an interrupted write must not fabricate a partial success marker."""
    p = tmp_path / "status.json"
    import v4sup.status as status_mod

    def boom(*a, **kw):
        raise OSError("simulated interruption before atomic replace")

    monkeypatch.setattr(status_mod.os, "replace", boom)
    with pytest.raises(OSError):
        write_status_atomic(p, {"state": "SHOULD_NOT_APPEAR"})
    assert read_status(p) is None


def test_merge_status_is_read_modify_write_and_does_not_assert_completion(tmp_path):
    p = tmp_path / "status.json"
    write_status_atomic(p, {"state": "RUNNING", "phases": {"p1": {"state": "RUNNING"}}})
    merge_status(p, heartbeat_utc="2026-01-01T00:00:00Z")
    s = read_status(p)
    assert s["heartbeat_utc"] == "2026-01-01T00:00:00Z"
    assert s["state"] == "RUNNING"          # untouched
    assert s["phases"]["p1"]["state"] == "RUNNING"  # heartbeat never marks a phase DONE


def test_append_event_is_append_only_and_line_delimited(tmp_path):
    log = tmp_path / "events.jsonl"
    append_event(log, {"phase": "p1", "event": "attempt_start"})
    append_event(log, {"phase": "p1", "event": "attempt_end", "rc": 0})
    lines = log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    ev0, ev1 = (json.loads(ln) for ln in lines)
    assert ev0["event"] == "attempt_start"
    assert ev1["event"] == "attempt_end" and ev1["rc"] == 0
