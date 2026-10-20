"""GPU visibility (FL-02 finding, binding for TRAIN-01): snap_v1/reuse/train.py runs
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0") when ppsi_env pre-imports it, so any central process that imports
the model stack without exporting the variable silently sees only GPU 0 (and so do its children).

  * `cuda_visible_devices` is a REQUIRED run-config key; a config without it is refused;
  * the trainer refuses to start when CUDA_VISIBLE_DEVICES is unset, when the launcher did not export it (v4train
    captures the value before ppsi_env's import; a subprocess reproduces the real trap), or when it differs from the
    run config; the value (config, environment, launch capture) is recorded in RUN_START;
  * a launcher that exports it explicitly starts normally.
"""
from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import pytest

import v4train
from trainkit import assert_raises, cfg, read, scripted, trainer
from v4train.recipe import RecipeError
from v4train.schedule import eval_marks
from v4train.trainer import TrainerError

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
VALS = {m: 0.1 for m in eval_marks("MAIN")}

PROBE = r'''
import os, sys
sys.path.insert(0, {code!r}); sys.path.insert(0, {here!r})
import v4train                                   # captures CUDA_VISIBLE_DEVICES BEFORE ppsi_env's import
from trainkit import cfg, trainer
from v4train.trainer import TrainerError
print("AT_IMPORT", repr(v4train.LAUNCH_CUDA_VISIBLE_DEVICES), "ENV", repr(os.environ.get("CUDA_VISIBLE_DEVICES")))
try:
    trainer(__import__("pathlib").Path({work!r}), cfg("probe", "GRU", cuda_visible_devices="0"))
    print("STARTED")
except TrainerError as e:
    print("REFUSED", e)
'''


def _probe(work: Path, env_value):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env.pop("CUDA_VISIBLE_DEVICES", None)
    if env_value is not None:
        env["CUDA_VISIBLE_DEVICES"] = env_value
    code = PROBE.format(code=str(CODE), here=str(HERE), work=str(work))
    p = subprocess.run([sys.executable, "-B", "-c", code], cwd=str(CODE), env=env, capture_output=True, text=True,
                       timeout=300)
    assert p.returncode == 0, p.stderr[-2000:]
    return p.stdout


def test_required_key_and_mismatch_refused(work, monkeypatch):
    c = cfg("k", "GRU")
    assert_raises(RecipeError, dataclasses.replace(c, cuda_visible_devices=None).validate)
    assert_raises(RecipeError, dataclasses.replace(c, cuda_visible_devices="").validate)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert_raises(TrainerError, trainer, work / "m", cfg("m", "GRU", cuda_visible_devices="1"))      # mismatch
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert_raises(TrainerError, trainer, work / "u", cfg("u", "GRU", cuda_visible_devices="0"))      # unset
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(v4train, "LAUNCH_CUDA_VISIBLE_DEVICES", None)                               # not exported
    assert_raises(TrainerError, trainer, work / "n", cfg("n", "GRU", cuda_visible_devices="0"))


def test_recorded_in_run_start(work, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    trainer(work / "r", cfg("r", "GRU", cuda_visible_devices="0"), eval_override=scripted(VALS)).run()
    rec = read(work / "r" / "RUN_START.json")
    assert rec["cuda_visible_devices"]["config"] == rec["cuda_visible_devices"]["env"] == "0"
    assert rec["config"]["cuda_visible_devices"] == "0"


def test_subprocess_launcher_trap(work):
    out = _probe(work / "p1", None)                               # launcher forgot to export it
    assert "AT_IMPORT None" in out and "ENV '0'" in out, out      # train.py silently set "0" after the capture
    assert "REFUSED" in out and "not exported by the launcher" in out, out
    out = _probe(work / "p2", "0")                                # explicit export: starts
    assert "AT_IMPORT '0'" in out and "STARTED" in out, out
