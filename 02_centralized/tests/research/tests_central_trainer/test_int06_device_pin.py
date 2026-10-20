"""INT-06 (PREG1_AMENDMENT_06D §1) — the trainer's acceptance of the REGISTERED second device class (CPU tests; the
runtime device class and the pin lookup are patched on v4train.trainer, exactly like the frozen-class patch of
test_registry_schema_labels.py).

  * no pin: a non-A16 device class is refused for a non-test run (the pre-INT-06 behaviour, unchanged);
  * a pin whose device_name differs from the live GPU: refused at construction (v4eval.device_pin raises);
  * a pinned class: accepted; device_pin_sha256 recorded in RUN_START, every checkpoint meta, RUN_SUMMARY and every
    result row (via the trainer's row labels);
  * a resume under another pin (or none) is refused; the A16 path records no pin field (byte-identical records).
"""
from __future__ import annotations

import dataclasses
import json

import pytest
import torch

import v4train.trainer as T
from trainkit import (NEW_DATA_VERSION, art, cfg, interrupt_after, nc, providers, raises_with, ramp, read, scripted,
                      trainer, write_registry)
from v4eval import device_pin as dp
from v4train.evalconfig import frozen
from v4train.schedule import eval_marks
from v4train.trainer import InjectedInterruption, TrainerError

A40 = "NVIDIA_A40-FP32-DET"
MAIN = eval_marks("MAIN")


def _pin(name="NVIDIA A40", tag="a"):
    obj = {"format": dp.FORMAT, "device_class": A40, "device_name": name, "driver": "test-driver",
           "torch": torch.__version__, "cuda": "test-cuda",
           "receipts": {"t3_gpu_receipt_sha256": tag * 64, "t3_verdict": "PASS",
                        "strict_determinism_receipt_sha256": "b" * 64},
           "authority": "PREG1_AMENDMENT_06D §1", "registered_utc": "2026-09-29T00:00:00Z"}
    obj["self_sha256"] = dp._digest(obj)
    return obj


def _non_test(name):
    tp, ep = providers(n=1024)
    reg = write_registry({"MAIN": (tp, ep)}, data_version=NEW_DATA_VERSION)
    c = cfg(name, "GRU", manifest_registry=reg, data_version=NEW_DATA_VERSION, effective_batch=256)
    return dataclasses.replace(c, test_overrides=False), tp, ep


def test_no_pin_refuses_a_non_a16_class(work, monkeypatch):
    monkeypatch.setattr(T, "runtime_device_class", lambda device: A40)
    monkeypatch.setattr(T, "pin_for_device", lambda device: None)
    c, tp, ep = _non_test("nopin")
    raises_with(TrainerError, "evaluation device class", trainer, work / "nopin", c, tp, ep)


def test_pin_with_another_gpu_name_refused(work, monkeypatch, tmp_path):
    p = tmp_path / "pin.json"
    p.write_text(json.dumps(_pin(name="NVIDIA A40-OTHER")), encoding="utf-8")
    monkeypatch.setattr(T, "runtime_device_class", lambda device: A40)
    monkeypatch.setattr(dp.torch.cuda, "get_device_name", lambda _d: "NVIDIA A40")
    monkeypatch.setattr(T, "pin_for_device", lambda device: dp.pin_for_device("cuda:0", p))
    c, tp, ep = _non_test("badname")
    raises_with(dp.DevicePinError, "name differs", trainer, work / "badname", c, tp, ep)


def test_pinned_class_accepted_and_recorded_everywhere(work, monkeypatch):
    pin = _pin()
    monkeypatch.setattr(T, "runtime_device_class", lambda device: A40)
    monkeypatch.setattr(T, "pin_for_device", lambda device: pin)
    c, tp, ep = _non_test("pinned")
    tr = trainer(work / "pinned", c, tp, ep, eval_override=scripted(ramp(MAIN, 3.0)))
    assert tr.eval_device_class == A40 and tr.device_pin_sha256 == pin["self_sha256"]
    summary = tr.run()
    root = art(work / "pinned")
    assert read(work / "pinned" / "RUN_START.json")["device_pin_sha256"] == pin["self_sha256"]
    assert read(work / "pinned" / "RUN_START.json")["eval_device_class"] == A40
    assert summary["device_pin_sha256"] == pin["self_sha256"]
    rows = read(root / "RESULT_ROWS.json")["rows"]
    assert rows and all(r["device_pin_sha256"] == pin["self_sha256"] and r["eval_device_class"] == A40 for r in rows)
    ck = sorted((root / "checkpoints").glob("*.pt"))
    assert ck and all(torch.load(p, map_location="cpu", weights_only=False)["meta"]["device_pin_sha256"] ==
                      pin["self_sha256"] for p in ck)


def test_a16_path_records_no_pin_field(work, monkeypatch):
    monkeypatch.setattr(T, "runtime_device_class", lambda device: frozen().device_class)
    monkeypatch.setattr(T, "pin_for_device", lambda device: None)
    c, tp, ep = _non_test("a16")
    tr = trainer(work / "a16", c, tp, ep, eval_override=scripted(ramp(MAIN, 3.0)))
    tr.run()
    root = art(work / "a16")
    assert "device_pin_sha256" not in read(work / "a16" / "RUN_START.json")
    assert all("device_pin_sha256" not in r for r in read(root / "RESULT_ROWS.json")["rows"])


def test_resume_under_another_pin_refused(work, monkeypatch):
    pin_a, pin_b = _pin(tag="a"), _pin(tag="c")
    c = cfg("rpin", "GRU", checkpoint_every=4)
    monkeypatch.setattr(T, "pin_for_device", lambda device: pin_a)
    with pytest.raises(InjectedInterruption):
        trainer(work / "R", c, fault=interrupt_after(9), eval_override=scripted(ramp(MAIN, 3.0))).run()
    for other in (pin_b, None):
        monkeypatch.setattr(T, "pin_for_device", lambda device, o=other: o)
        raises_with(TrainerError, "device_pin_sha256", trainer(work / "R", c,
                                                               eval_override=scripted(ramp(MAIN, 3.0))).resume)
    monkeypatch.setattr(T, "pin_for_device", lambda device: pin_a)
    assert trainer(work / "R", c, eval_override=scripted(ramp(MAIN, 3.0))).resume()["status"] == "DONE_MAIN"


@nc("without a pin the A40 class is an accepted evaluation class")
def test_nc_a40_accepted_without_pin():
    assert A40 in dp.accepted_classes(frozen().device_class, None)


@nc("a pin registered for another GPU name is accepted for the live GPU")
def test_nc_pin_for_another_gpu_accepted(tmp_path, monkeypatch):
    p = tmp_path / "pin.json"
    p.write_text(json.dumps(_pin(name="NVIDIA A40-OTHER")), encoding="utf-8")
    monkeypatch.setattr(dp.torch.cuda, "get_device_name", lambda _d: "NVIDIA A40")
    try:
        dp.pin_for_device("cuda:0", p)
        accepted = True
    except dp.DevicePinError:
        accepted = False
    assert accepted
