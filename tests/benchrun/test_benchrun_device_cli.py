"""The device classes (CPU-FP32 and the deterministic GPU class) and the command line."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from benchrun_testkit import make_release

from ppsi.benchrun import accel, central
from ppsi.benchrun import data as D
from ppsi.benchrun import run as R
from ppsi.benchrun.common import BenchRefused

CPU, GPU = accel.DEVICE_CLASS_CPU, "CUDA_SOME_CARD-FP32-DET"


def test_device_class_of_and_one_class():
    assert accel.device_class_of(None) == CPU and accel.device_class_of({"spec": {}}) == CPU
    assert accel.device_class_of({"device_class": GPU}) == GPU
    assert accel.device_class_of({"spec": {"device_class": GPU}}) == GPU
    assert accel.device_class_of({"arm_spec": {"device_class": GPU}}) == GPU
    assert accel.one_class({"a": CPU, "b": CPU}, "x") == CPU
    with pytest.raises(BenchRefused, match="mixed"):
        accel.one_class({"a": CPU, "b": GPU}, "x")
    assert accel.torch_device(None).type == "cpu" and accel.torch_device("cpu").type == "cpu"
    assert accel.device_record(torch.device("cpu")) is None
    with pytest.raises(BenchRefused):
        accel.torch_device("tpu")


def test_nc_tf32_and_non_strict_numerics_refused(monkeypatch):
    cuda = torch.device("cuda")                                  # a device object only: no CUDA call is made
    monkeypatch.setattr(accel, "device_class", lambda dev: GPU if accel.is_cuda(dev) else CPU)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", accel.CUBLAS_CFG)
    from ppsi.fedsim.numerics import enforce_strict_fp32
    enforce_strict_fp32(deterministic=True, warn_only=False)
    accel.assert_numerics(cuda)                                  # the strict GPU numerics pass
    torch.backends.cuda.matmul.allow_tf32 = True
    with pytest.raises(BenchRefused, match="allow_tf32"):
        accel.assert_numerics(cuda)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    with pytest.raises(BenchRefused, match="cudnn.allow_tf32"):
        accel.assert_numerics(cuda)
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("high")
    with pytest.raises(BenchRefused, match="precision"):
        accel.assert_numerics(cuda)
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True, warn_only=True)
    with pytest.raises(BenchRefused, match="not strict"):
        accel.assert_numerics(cuda)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = True
    with pytest.raises(BenchRefused, match="benchmark"):
        accel.assert_numerics(cuda)
    torch.backends.cudnn.benchmark = False
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG")
    with pytest.raises(BenchRefused, match="CUBLAS"):
        accel.assert_numerics(cuda)
    accel.assert_numerics(torch.device("cpu"))                   # the CPU class is not affected by this guard


def test_nc_cuda_refused_without_cublas_or_gpu(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(BenchRefused, match="CUBLAS"):
        accel.torch_device("cuda")
    if not torch.cuda.is_available():
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", accel.CUBLAS_CFG)
        with pytest.raises(BenchRefused, match="one visible CUDA device"):
            accel.torch_device("cuda")


def test_nc_resume_refuses_another_device_class(tmp_path, monkeypatch):
    monkeypatch.setattr(central, "EFFECTIVE_BATCH", 8)
    rel = make_release(tmp_path / "p", n_users=40, n_items=30)
    d = D.load(rel["name"], rel["root"])
    run_dir = tmp_path / "runs" / "PRE_s2026"
    run_dir.mkdir(parents=True)
    (run_dir / "ARM_SPEC.json").write_text(json.dumps({"arm": "PRE", "device_class": GPU}))
    with pytest.raises(BenchRefused, match="device class"):
        central.run_central(data=d, arm="PRE", variant="SASREC_D64_B2", peak_lr=1e-3, dropout=(0.1,), seed=2026,
                            rows=d.band_rows(), select_users=np.flatnonzero(d.band), run_dir=run_dir, run_id="p",
                            resume=True, device="cpu")


def test_nc_resume_cpu_run_on_the_gpu_class(monkeypatch):
    cuda = torch.device("cuda")
    rec = {"device_class": GPU, "gpu_name": "SOME CARD", "torch": "x", "cuda": "y", "cudnn": 1}
    monkeypatch.setattr(accel, "device_class", lambda dev: GPU if accel.is_cuda(dev) else CPU)
    monkeypatch.setattr(accel, "device_record", lambda dev: dict(rec) if accel.is_cuda(dev) else None)
    with pytest.raises(BenchRefused, match="device class CPU-FP32"):
        accel.check_resume({"arm": "PRE", "spec_sha256": "x"}, cuda)            # a CPU ARM_SPEC (no field)
    accel.check_resume({"device_class": GPU, "device": dict(rec)}, cuda)       # same class + build: allowed
    with pytest.raises(BenchRefused, match="torch"):
        accel.check_resume({"device_class": GPU, "device": dict(rec, torch="other")}, cuda)
    accel.check_resume({"arm": "PRE"}, torch.device("cpu"))


# ------------------------------------------------------------------------------------------------ command line
def test_cli_refusals(release, tmp_path, capsys):
    base = ["--dataset", release["name"], "--data-root", str(release["root"]), "--runs-root", str(tmp_path / "r"),
            "--threads", "1", "--seed", "2026"]
    assert R.main(["run", "--arm", "NOT_AN_ARM", *base]) == 2
    assert R.main(["run", "--arm", "FT_FA", *base]) == 2                       # FT needs --base-run
    assert R.main(["run", "--arm", "DP8", "--smoke", *base]) == 2              # DP8 needs an S record
    assert R.main(["run", "--arm", "FA", "--budget-efe", "24", *base]) == 2    # the budget option is central-only
    assert R.main(["run", "--arm", "PRE", "--gpu", "0", *base]) == 2
    assert "--gpu needs --device cuda" in capsys.readouterr().err
    assert R.main(["run", "--arm", "PRE", "--device", "cuda", *base]) == 2
    assert "--gpu" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        R.main(["run", "--arm", "PRE", "--device", "tpu", *base])
    assert R.main(["describe", "--dataset", "not_a_dataset", "--data-root", str(release["root"])]) == 2
    assert R.main(["test", "--dataset", release["name"], "--data-root", str(release["root"]), "--runs-root",
                   str(tmp_path / "r"), "--threads", "1", "--runs", "C_FULL_s2026"]) == 2   # no such run


def test_cli_describe_reads_train_only(release, capsys):
    (release["root"] / release["name"] / "leave_one_out" / "holdout.csv").unlink()
    assert R.main(["describe", "--dataset", release["name"], "--data-root", str(release["root"]),
                   "--threads", "1"]) == 0
    assert '"K"' in capsys.readouterr().out


def test_cli_central_smoke_run(release, tmp_path, monkeypatch):
    monkeypatch.setattr(central, "EFFECTIVE_BATCH", 8)
    assert R.main(["run", "--dataset", release["name"], "--data-root", str(release["root"]), "--runs-root",
                   str(tmp_path / "r"), "--threads", "1", "--seed", "2026", "--arm", "PRE", "--smoke",
                   "--smoke-epochs", "0.5"]) == 0
    res = json.loads((tmp_path / "r" / release["name"] / "SMOKE" / "PRE_s2026" / "RESULT.json").read_text())
    assert res["status"] == "SMOKE_DONE_NOT_EVIDENCE" and res["spec"]["provenance"]["recipe_sha256"]
