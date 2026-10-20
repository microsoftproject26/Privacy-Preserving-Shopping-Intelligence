"""End-to-end Supervisor behavior over real (but trivial, fast, CPU-only) subprocesses:
  - a fake phase that exits 0 but writes nothing must fail;
  - a phase whose prerequisite hash is wrong never starts (no subprocess spawned at all);
  - GPU-busy and disk-REFUSE both mean "never starts" the same way;
  - retry logs are kept, one file per attempt, never overwritten;
  - a second (final) failure sets FAILED and stops phases that depend on it;
  - the heartbeat never marks a phase DONE by itself.
nvidia-smi and disk usage are mocked throughout (SYS-03 requirement).
"""
from __future__ import annotations

import json
import sys

from suptestkit import fake_disk_usage_fn, fake_nvidia_smi_runner

from v4sup.registry import ExpectedArtifact, GateFile, Phase
from v4sup.runner import Supervisor, SupervisorConfig

PY = sys.executable


def make_supervisor(tmp_path, registry, disk_free_gib=100.0, gpu_runner=None):
    ws = tmp_path / "workspace"
    ws.mkdir()
    cfg = SupervisorConfig(workspace_root=ws, registry=registry, status_path=ws / "state" / "status.json",
                           event_log_path=ws / "logs" / "events.jsonl", logs_dir=ws / "logs",
                           disk_usage_fn=fake_disk_usage_fn(disk_free_gib),
                           gpu_runner=gpu_runner or fake_nvidia_smi_runner(), sleep_fn=lambda s: None,
                           poll_interval_s=0.0)
    return Supervisor(cfg), ws


def _phase(id_, command, max_attempts=2, expected=("out.json",), gpus=(), planned_peak_gib=0.1,
           prerequisites=(), depends_on=()):
    return Phase(id=id_, command=tuple(command), shell_command=None, prerequisites=prerequisites,
                 expected_artifacts=tuple(ExpectedArtifact(path=p) for p in expected), gpus=gpus,
                 planned_peak_gib=planned_peak_gib, max_attempts=max_attempts, depends_on=depends_on, timeout_s=30)


def test_exits_zero_but_writes_nothing_fails(tmp_path):
    ph = _phase("p1", [PY, "-B", "-c", "pass"], max_attempts=1)
    sup, ws = make_supervisor(tmp_path, {"p1": ph})
    state = sup.run_phase("p1")
    assert state == "FAILED"
    assert sup.phase_state("p1") == "FAILED"
    assert not (ws / "out.json").exists()
    log = ws / "logs" / "p1" / "attempt_1.log"
    assert log.is_file() and log.stat().st_size > 0  # the subprocess did run


def test_retry_logs_are_kept_one_per_attempt(tmp_path):
    ph = _phase("p1", [PY, "-B", "-c", "pass"], max_attempts=2)
    sup, ws = make_supervisor(tmp_path, {"p1": ph})
    state1 = sup.run_phase("p1")
    assert state1 == "ATTEMPT_FAILED"
    state2 = sup.run_phase("p1")
    assert state2 == "FAILED"
    log_dir = ws / "logs" / "p1"
    attempt1, attempt2 = log_dir / "attempt_1.log", log_dir / "attempt_2.log"
    assert attempt1.is_file() and attempt1.stat().st_size > 0
    assert attempt2.is_file() and attempt2.stat().st_size > 0
    assert attempt1.read_text() != ""  # first attempt's log was not overwritten by the second


def test_happy_path_reaches_done(tmp_path):
    write_cmd = "import json,pathlib; pathlib.Path('out.json').write_text(json.dumps({'ok': True}))"
    ph = _phase("p1", [PY, "-B", "-c", write_cmd], max_attempts=2)
    sup, ws = make_supervisor(tmp_path, {"p1": ph})
    state = sup.run_phase("p1")
    assert state == "DONE"
    assert (ws / "out.json").is_file()
    assert json.loads((ws / "out.json").read_text()) == {"ok": True}


def test_gate_with_wrong_hash_never_starts(tmp_path):
    gate = tmp_path / "gate.json"
    gate.write_text("{}")
    ph = _phase("p1", [PY, "-B", "-c", "pass"], prerequisites=(GateFile(path=str(gate), sha256="0" * 64),))
    sup, ws = make_supervisor(tmp_path, {"p1": ph})
    state = sup.run_phase("p1")
    assert state == "WAITING_GATE"
    assert not (ws / "logs" / "p1").exists()  # no subprocess was ever spawned


def test_gate_missing_file_never_starts(tmp_path):
    ph = _phase("p1", [PY, "-B", "-c", "pass"],
               prerequisites=(GateFile(path=str(tmp_path / "absent.json"), sha256="0" * 64),))
    sup, ws = make_supervisor(tmp_path, {"p1": ph})
    state = sup.run_phase("p1")
    assert state == "WAITING_GATE"
    assert not (ws / "logs" / "p1").exists()


def test_gpu_busy_never_starts(tmp_path):
    busy_runner = fake_nvidia_smi_runner(gpu_csv="0, 00000000:01:00.0", apps_csv="00000000:01:00.0, 999")
    ph = _phase("p1", [PY, "-B", "-c", "pass"], gpus=(0,))
    sup, ws = make_supervisor(tmp_path, {"p1": ph}, gpu_runner=busy_runner)
    state = sup.run_phase("p1")
    assert state == "QUEUED_RESOURCE"
    assert not (ws / "logs" / "p1").exists()


def test_disk_refuse_never_starts(tmp_path):
    ph = _phase("p1", [PY, "-B", "-c", "pass"], planned_peak_gib=5.0)
    sup, ws = make_supervisor(tmp_path, {"p1": ph}, disk_free_gib=9.0)  # headroom 4 < floor 8
    state = sup.run_phase("p1")
    assert state == "QUEUED_RESOURCE"
    assert "REFUSE" in sup.phase_status("p1")["reason"]
    assert not (ws / "logs" / "p1").exists()


def test_second_failure_sets_failed_and_stops_dependents(tmp_path):
    a = _phase("A", [PY, "-B", "-c", "pass"], max_attempts=2)  # always writes nothing -> always fails
    write_cmd = "import pathlib; pathlib.Path('out.json').write_text('{}')"
    b = _phase("B", [PY, "-B", "-c", write_cmd], max_attempts=2, depends_on=("A",))
    sup, ws = make_supervisor(tmp_path, {"A": a, "B": b})

    assert sup.run_phase("A") == "ATTEMPT_FAILED"
    assert sup.run_phase("A") == "FAILED"          # second failure -> FAILED
    assert sup.phase_state("A") == "FAILED"

    state_b = sup.run_phase("B")
    assert state_b == "BLOCKED"                    # dependent phase stopped, never attempted
    assert sup.phase_state("B") == "BLOCKED"
    assert not (ws / "logs" / "B").exists()
    assert not (ws / "out.json").exists()


def test_waiting_dependency_before_upstream_done(tmp_path):
    a = _phase("A", [PY, "-B", "-c", "pass"], max_attempts=3)  # not yet run
    b = _phase("B", [PY, "-B", "-c", "pass"], depends_on=("A",))
    sup, ws = make_supervisor(tmp_path, {"A": a, "B": b})
    state_b = sup.run_phase("B")
    assert state_b == "WAITING_DEPENDENCY"
    assert not (ws / "logs" / "B").exists()


def test_run_all_drives_a_simple_chain_to_done(tmp_path):
    write_a = "import pathlib; pathlib.Path('a.json').write_text('{}')"
    write_b = "import pathlib; pathlib.Path('b.json').write_text('{}')"
    a = _phase("A", [PY, "-B", "-c", write_a], expected=("a.json",))
    b = _phase("B", [PY, "-B", "-c", write_b], expected=("b.json",), depends_on=("A",))
    sup, ws = make_supervisor(tmp_path, {"A": a, "B": b})
    status = sup.run_all(order=["A", "B"], max_rounds=10)
    assert status["phases"]["A"]["state"] == "DONE"
    assert status["phases"]["B"]["state"] == "DONE"


def test_heartbeat_never_marks_a_phase_done(tmp_path):
    ph = _phase("p1", [PY, "-B", "-c", "pass"])
    sup, ws = make_supervisor(tmp_path, {"p1": ph})
    sup.heartbeat(current_phase="p1")
    assert sup.phase_state("p1") == "PENDING"  # unaffected by the heartbeat
    status = json.loads((ws / "state" / "status.json").read_text())
    assert "heartbeat_utc" in status
    assert status.get("phases", {}) == {}
