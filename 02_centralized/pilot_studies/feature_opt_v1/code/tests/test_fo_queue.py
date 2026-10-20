"""fo_queue.py (owner: Opus) -- gpu_busy fail-closed behavior and ready() gating. Never modifies
the code under test. fo_queue.py has hardcoded server paths (F, CAMPAIGN) that are never touched
here: every test monkeypatches subprocess.run / the module's own path constants / its collaborator
functions instead of letting any real subprocess or filesystem path from the module execute.
"""
from __future__ import annotations

import types

import pytest

import conftest_fo  # noqa: F401
import fo_queue as fq


def _run(returncode=0, stdout="", stderr=""):
    return types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


# --------------------------------------------------------------------- gpu_busy fail-closed
def test_gpu_busy_nonzero_returncode_marks_every_gpu_busy(monkeypatch):
    calls = {"n": 0}

    def fake_run(argv, **kw):
        calls["n"] += 1
        if "--query-gpu=index,pci.bus_id" in argv:
            return _run(returncode=1)   # nvidia-smi query failed
        return _run(returncode=0, stdout="")

    monkeypatch.setattr(fq.subprocess, "run", fake_run)
    busy = fq.gpu_busy()
    assert busy == {"0": ["unknown"], "1": ["unknown"]}
    assert calls["n"] >= 1


def test_gpu_busy_subprocess_exception_marks_every_gpu_busy(monkeypatch):
    def fake_run(argv, **kw):
        raise OSError("nvidia-smi not found")

    monkeypatch.setattr(fq.subprocess, "run", fake_run)
    assert fq.gpu_busy() == {"0": ["unknown"], "1": ["unknown"]}


def test_gpu_busy_unparsable_query_line_marks_every_gpu_busy(monkeypatch):
    def fake_run(argv, **kw):
        if "--query-gpu=index,pci.bus_id" in argv:
            return _run(returncode=0, stdout="this line has no comma at all\n")
        return _run(returncode=0, stdout="")

    monkeypatch.setattr(fq.subprocess, "run", fake_run)
    assert fq.gpu_busy() == {"0": ["unknown"], "1": ["unknown"]}


def test_gpu_busy_app_on_unmapped_bus_marks_every_gpu_busy(monkeypatch):
    def fake_run(argv, **kw):
        if "--query-gpu=index,pci.bus_id" in argv:
            return _run(returncode=0, stdout="0,0000:01:00.0\n1,0000:02:00.0\n")
        return _run(returncode=0, stdout="0000:FF:00.0,12345\n")   # bus id not in the query map

    monkeypatch.setattr(fq.subprocess, "run", fake_run)
    busy = fq.gpu_busy()
    assert busy == {i: ["unmapped"] for i in ("0", "1")}


def test_gpu_busy_normal_mapping_only_the_right_gpu_busy(monkeypatch):
    """Negative control for all four fail-closed cases above: a clean, well-formed nvidia-smi
    output must narrow busy-ness down to exactly the GPU with a real compute process, not mark
    every GPU busy."""
    def fake_run(argv, **kw):
        if "--query-gpu=index,pci.bus_id" in argv:
            return _run(returncode=0, stdout="0,0000:01:00.0\n1,0000:02:00.0\n")
        return _run(returncode=0, stdout="0000:01:00.0,777\n")   # only GPU 0 has a process

    monkeypatch.setattr(fq.subprocess, "run", fake_run)
    busy = fq.gpu_busy()
    assert busy == {"0": ["777"], "1": []}
    assert busy.get("1") == []   # falsy -> ready() would treat GPU 1 as free
    assert bool(busy.get("0"))   # truthy -> ready() would treat GPU 0 as taken


# --------------------------------------------------------------------- ready()
def test_ready_false_when_final_stage_done_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(fq, "CAMPAIGN", tmp_path)   # no state/FINAL_STAGE_DONE under here
    monkeypatch.setattr(fq, "campaign_busy", lambda: [])
    monkeypatch.setattr(fq, "gpu_busy", lambda: {})
    monkeypatch.setattr(fq, "free_gib", lambda: 1000.0)
    ok, why = fq.ready((0,), 0.1)
    assert ok is False and "FINAL_STAGE_DONE" in why


def test_ready_false_when_campaign_process_alive(tmp_path, monkeypatch):
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "FINAL_STAGE_DONE").write_text("x")
    monkeypatch.setattr(fq, "CAMPAIGN", tmp_path)
    monkeypatch.setattr(fq, "campaign_busy", lambda: ["python train_sasrec.py ..."])
    monkeypatch.setattr(fq, "gpu_busy", lambda: {})
    monkeypatch.setattr(fq, "free_gib", lambda: 1000.0)
    ok, why = fq.ready((0,), 0.1)
    assert ok is False and "campaign processes alive" in why


def test_ready_false_when_disk_below_floor_plus_peak(tmp_path, monkeypatch):
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "FINAL_STAGE_DONE").write_text("x")
    monkeypatch.setattr(fq, "CAMPAIGN", tmp_path)
    monkeypatch.setattr(fq, "campaign_busy", lambda: [])
    monkeypatch.setattr(fq, "gpu_busy", lambda: {})
    monkeypatch.setattr(fq, "free_gib", lambda: fq.FLOOR_GIB + 0.05)   # below floor + peak(0.5)
    ok, why = fq.ready((0,), 0.5)
    assert ok is False and "disk free" in why


def test_ready_true_when_every_condition_satisfied(tmp_path, monkeypatch):
    """Negative control for the three false cases above: with everything satisfied, ready() must
    actually return True, proving the checks above are each individually necessary rather than
    the function being vacuously False."""
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "FINAL_STAGE_DONE").write_text("x")
    monkeypatch.setattr(fq, "CAMPAIGN", tmp_path)
    monkeypatch.setattr(fq, "campaign_busy", lambda: [])
    monkeypatch.setattr(fq, "gpu_busy", lambda: {"0": [], "1": ["999"]})
    monkeypatch.setattr(fq, "free_gib", lambda: fq.FLOOR_GIB + 10.0)
    ok, why = fq.ready((0,), 0.5)   # only asking for GPU 0, which is free
    assert ok is True and why == "ok"


def test_ready_false_when_the_requested_gpu_itself_is_busy(tmp_path, monkeypatch):
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "FINAL_STAGE_DONE").write_text("x")
    monkeypatch.setattr(fq, "CAMPAIGN", tmp_path)
    monkeypatch.setattr(fq, "campaign_busy", lambda: [])
    monkeypatch.setattr(fq, "gpu_busy", lambda: {"0": [], "1": ["999"]})
    monkeypatch.setattr(fq, "free_gib", lambda: fq.FLOOR_GIB + 10.0)
    ok, why = fq.ready((1,), 0.5)   # GPU 1 IS busy
    assert ok is False and "compute processes" in why
