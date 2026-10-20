"""BM-05: the TEST look (`final`) on the dataset's ONE device class (Fable §11 E1: training AND the TEST look).

  * CPU byte-identity: code_bench4's final and this tree's final (--device cpu) on byte-identical copies of one prepared tiny
    tree (PRE, FA, PF, FT_FA + LOCK) write byte-identical TEST outputs (ranks, TEST_*.json, TEST_RESULTS, TEST_LOOK, attempts);
  * NCs: a LOCK (or a locked run) of another device class is refused before the attempt is logged and before the holdout is
    read; CLI flags;
  * GPU two-run test (skipped unless BM_GPU_TEST=<physical GPU id>): prepare on the GPU class, then final --device cuda on
    two copies -> identical TEST digests, every TEST output carries the class (Ops, A40 server).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from conftest import make_release
from test_bm03_device import CODE, DS, GPU, CPU, _bench4_roots, _gpu_id, fake_run
from v4bench import accel, final
from v4bench import run as R
from v4bench.common import BenchRefused, canon_sha256, read_json

HELPER = Path(__file__).resolve().parent / "bm05_final_job.py"


def _job(stage, code_root, lib_root, release, runs, device, out=None, gpu=None):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONPATH", None)
    cmd = [sys.executable, "-B", str(HELPER), "--stage", stage, "--code-root", str(code_root), "--lib-root",
           str(lib_root), "--release", str(release), "--runs", str(runs), "--device", device]
    if out is not None:
        cmd += ["--out", str(out)]
    if gpu is not None:
        cmd += ["--gpu", str(gpu)]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    return json.loads(Path(out).read_text(encoding="utf-8")) if out is not None else p.stdout


def _diff(a, b):
    return sorted(k for k in set(a["files"]) | set(b["files"]) if a["files"].get(k) != b["files"].get(k))


# ------------------------------------------------------------------------------------------- CPU byte identity
def test_final_cpu_byte_identical_to_bench4(tmp_path):
    old_code, old_lib = _bench4_roots()
    if old_code is None:
        pytest.skip("no code_bench4 reference (code/_archive_bench4 or $BM_BENCH4_ROOT)")
    rel = make_release(tmp_path / "processed", n_users=40, n_items=25)
    prep = tmp_path / "prep"
    out = _job("prepare", CODE, CODE, rel["root"], prep, "cpu")
    assert '"lock_device_class": "CPU-FP32"' in out
    shutil.copytree(prep, tmp_path / "A")
    shutil.copytree(prep, tmp_path / "B")
    a = _job("final", old_code, old_lib, rel["root"], tmp_path / "A", "legacy", out=tmp_path / "a.json")
    b = _job("final", CODE, CODE, rel["root"], tmp_path / "B", "cpu", out=tmp_path / "b.json")
    assert a["n_test_users"] == b["n_test_users"] > 0 and b["device_class"] is None
    tests = [k for k in b["files"] if "/test/" in k or k.split("/")[-1].startswith("TEST_")]
    assert sum(k.endswith(".npy") for k in tests) == 8 and any(k.endswith("TEST_RESULTS.json") for k in tests)
    assert _diff(a, b) == []
    for p in (tmp_path / "B" / DS).rglob("TEST_*.json"):                   # the CPU outputs carry no new field
        assert "device_class" not in read_json(p)
    assert "device_class" not in read_json(tmp_path / "B" / DS / "TEST_RESULTS.json")


# ------------------------------------------------------------------------------------------- NCs
def test_is_cuda_is_strict_and_unknown_devices_refused():
    """Regression (found by the byte-identity test while writing BM-05): in final.score_run the name `device` is also the
    FT module; the old permissive is_cuda took a module for a GPU. Only "cuda[:i]" / a cuda torch.device are the GPU."""
    import types
    mod = types.ModuleType("device")
    assert not any(accel.is_cuda(x) for x in (None, "cpu", torch.device("cpu"), mod))
    assert all(accel.is_cuda(x) for x in ("cuda", "cuda:0", torch.device("cuda")))
    for bad in ("gpu", "tpu", mod):
        with pytest.raises(BenchRefused, match="unknown device"):
            accel.torch_device(bad)


def _fake_locked(runs, dclass):
    fake_run(runs, "C_FULL_s2026", "C_FULL", 2026, 0.1, kind="central", dclass=dclass)
    fake_run(runs, "FA_s2026", "FA", 2026, 0.1, dclass=dclass)
    return final.make_lock(runs, DS, ["C_FULL_s2026", "FA_s2026"], data_fingerprint="x", strict=False)


def _no_look(runs):
    assert not final.attempts_path(runs, DS).exists() and not final.look_path(runs, DS).exists()


def _as_cuda(monkeypatch):
    monkeypatch.setattr(accel, "torch_device", lambda device=None: torch.device("cuda") if accel.is_cuda(device)
                        else torch.device("cpu"))
    monkeypatch.setattr(accel, "device_class", lambda dev: GPU if accel.is_cuda(dev) else CPU)


def test_nc_gpu_lock_refused_on_cpu(tmp_path):
    assert _fake_locked(tmp_path, GPU)["device_class"] == GPU
    for dev in (None, "cpu"):
        with pytest.raises(BenchRefused, match="LOCK's device class is NVIDIA_A40-FP32-DET, this look runs on CPU-FP32"):
            final.final_eval(tmp_path, DS, None, tmp_path, device=dev)
    _no_look(tmp_path)


def test_nc_cpu_lock_refused_on_the_gpu_class(tmp_path, monkeypatch):
    _as_cuda(monkeypatch)
    assert _fake_locked(tmp_path, None)["device_class"] == CPU
    with pytest.raises(BenchRefused, match="LOCK's device class is CPU-FP32, this look runs on NVIDIA_A40-FP32-DET"):
        final.final_eval(tmp_path, DS, None, tmp_path, device="cuda")
    _no_look(tmp_path)
    lp = final.lock_path(tmp_path, DS)                                     # a bench4 LOCK (no field) = CPU-FP32
    lock = {k: v for k, v in read_json(lp).items() if k not in ("device_class", "self_sha256")}
    lock["self_sha256"] = canon_sha256(lock)
    os.chmod(lp, 0o644)
    lp.write_text(json.dumps(lock))
    with pytest.raises(BenchRefused, match="LOCK's device class is CPU-FP32"):
        final.final_eval(tmp_path, DS, None, tmp_path, device="cuda")
    _no_look(tmp_path)


def test_nc_locked_run_of_another_class_refused(tmp_path, monkeypatch):
    _as_cuda(monkeypatch)
    _fake_locked(tmp_path, GPU)
    rp = tmp_path / DS / "FA_s2026" / "RESULT.json"
    res = read_json(rp)
    res.pop("device_class")
    rp.write_text(json.dumps(res))
    with pytest.raises(BenchRefused, match="locked run FA_s2026 is not NVIDIA_A40-FP32-DET"):
        final.final_eval(tmp_path, DS, None, tmp_path, device="cuda")
    _no_look(tmp_path)


def test_cli_final_device_flags_and_refusal(release, tmp_path, capsys):
    runs = tmp_path / "runs"
    _fake_locked(runs, GPU)
    base = ["final", "--dataset", DS, "--data-root", str(release["root"]), "--runs-root", str(runs), "--threads", "1"]
    assert R.main([*base, "--gpu", "0"]) == 2 and "--gpu needs --device cuda" in capsys.readouterr().err
    assert R.main([*base, "--device", "cuda"]) == 2 and "--gpu" in capsys.readouterr().err
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert R.main(base) == 2 and "LOCK's device class is NVIDIA_A40-FP32-DET" in capsys.readouterr().err
    assert R.main([*base, "--device", "cpu"]) == 2 and "this look runs on CPU-FP32" in capsys.readouterr().err
    _no_look(runs)
    with pytest.raises(SystemExit):
        R.main([*base, "--device", "tpu"])


# ------------------------------------------------------------------------------------------- GPU (Ops, A40 server)
def test_gpu_final_two_runs_identical(tmp_path):
    """Prepare (PRE, FA, PF, FT_FA, LOCK) on the GPU class once; final --device cuda on two byte-identical copies:
    every TEST digest identical; TEST_RESULTS / TEST_LOOK / TEST_*.json / the attempt log carry the class."""
    g = _gpu_id()
    rel = make_release(tmp_path / "processed", n_users=40, n_items=25)
    prep = tmp_path / "prep"
    want = os.environ.get("BM_GPU_CLASS", GPU)
    assert f'"lock_device_class": "{want}"' in _job("prepare", CODE, CODE, rel["root"], prep, "cuda", gpu=g)
    shutil.copytree(prep, tmp_path / "A")
    shutil.copytree(prep, tmp_path / "B")
    a = _job("final", CODE, CODE, rel["root"], tmp_path / "A", "cuda", out=tmp_path / "a.json", gpu=g)
    b = _job("final", CODE, CODE, rel["root"], tmp_path / "B", "cuda", out=tmp_path / "b.json", gpu=g)
    assert a["device_class"] == b["device_class"] == want
    assert _diff(a, b) == []
    base = tmp_path / "A" / DS
    assert read_json(base / "TEST_RESULTS.json")["device"]["cublas_workspace_config"] == accel.CUBLAS_CFG
    assert read_json(base / "TEST_LOOK.json")["device_class"] == want
    assert all(read_json(p)["device_class"] == want for p in base.rglob("TEST_*_*.json") if p.parent.name == "test")
    rows = [json.loads(x) for x in final.attempts_path(tmp_path / "A", DS).read_text().splitlines()]
    assert rows[0]["event"] == "start" and rows[0]["device_class"] == want and rows[-1]["event"] == "end"
    p = subprocess.run([sys.executable, "-B", str(HELPER), "--stage", "final", "--code-root", str(CODE), "--lib-root",
                        str(CODE), "--release", str(rel["root"]), "--runs", str(prep), "--device", "cpu", "--out",
                        str(tmp_path / "c.json")], capture_output=True, text=True, timeout=600,
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    assert p.returncode != 0 and "LOCK's device class is" in p.stderr               # NC: the GPU LOCK on CPU
    assert not final.look_path(prep, DS).exists()
