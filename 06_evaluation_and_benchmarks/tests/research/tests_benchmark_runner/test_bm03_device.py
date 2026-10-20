"""BM-03: the opt-in deterministic GPU device class (FABLE_GATE_ENDGAME_SCRIPTS_20261003 §8).

  * CPU byte-identity guard: code_bench4 (code/_archive_bench4, or $BM_BENCH4_ROOT = a full code_bench4 tree) and this tree
    with --device cpu produce byte-identical checkpoints / ranks / records on a tiny fixed job set (central, FL, FT);
  * the device class is recorded and every mix is refused (NCs): resume, theta_pre, S record, FT base, LOCK, trackb-select,
    decide-warm, seed3; TF32-on / non-strict numerics refused; CLI flags;
  * the GPU determinism test (skipped unless BM_GPU_TEST=<physical GPU id>): the same job set twice on the GPU class ->
    identical digests (Ops runs it on the A40 server).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import make_release
from v4bench import accel, central, data as D, device as FT, final, fl_engine, selection
from v4bench import run as R
from v4bench.common import BenchRefused, canon_sha256, read_json
import ppsi_paths

CODE = ppsi_paths.code_dir("v4bench").parent   # the code folder that holds v4bench
HELPER = Path(__file__).resolve().parent / "bm03_digest_job.py"
DS = "s3_beauty"
GPU = accel.DEVICE_CLASS_A40
CPU = accel.DEVICE_CLASS_CPU


def _bench4_roots():
    env = os.environ.get("BM_BENCH4_ROOT")
    if env:
        return Path(env), Path(env)                              # a full code_bench4 tree (server)
    arch = ppsi_paths.code_dir("_archive_bench4")
    return (arch, CODE) if (arch / "v4bench" / "run.py").is_file() else (None, None)


def _helper(code_root, lib_root, release, runs, out, device, gpu=None, jobs="central,fl,ft", env_extra=None):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONPATH", None)
    env.update(env_extra or {})
    cmd = [sys.executable, "-B", str(HELPER), "--code-root", str(code_root), "--lib-root", str(lib_root),
           "--release", str(release), "--runs", str(runs), "--out", str(out), "--device", device, "--jobs", jobs]
    if gpu is not None:
        cmd += ["--gpu", str(gpu)]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    return json.loads(Path(out).read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------- CPU byte identity
def test_cpu_path_byte_identical_to_bench4(tmp_path):
    old_code, old_lib = _bench4_roots()
    if old_code is None:
        pytest.skip("no code_bench4 reference (code/_archive_bench4 or $BM_BENCH4_ROOT)")
    rel = make_release(tmp_path / "processed", n_users=60, n_items=30)
    a = _helper(old_code, old_lib, rel["root"], tmp_path / "runs_b4", tmp_path / "b4.json", "legacy")
    b = _helper(CODE, CODE, rel["root"], tmp_path / "runs_b5", tmp_path / "b5.json", "cpu")
    assert a["status"] == b["status"] and set(a["status"].values()) <= {"DONE_MAIN", "PARTIAL_LIMIT"}
    assert len(a["files"]) > 100 and sum(k.endswith(".pt") for k in a["files"]) >= 10
    diff = sorted(k for k in set(a["files"]) | set(b["files"]) if a["files"].get(k) != b["files"].get(k))
    assert diff == [], diff
    for res in (tmp_path / "runs_b5").rglob("RESULT.json"):     # the CPU records carry no new field
        assert "device_class" not in read_json(res) and "device_class" not in read_json(res).get("spec", {})


def test_cpu_default_equals_explicit_cpu_on_the_cli(release, tmp_path):
    base = ["run", "--dataset", DS, "--data-root", str(release["root"]), "--threads", "1", "--seed", "2026",
            "--arm", "PRE", "--smoke", "--smoke-epochs", "1"]
    assert R.main([*base, "--runs-root", str(tmp_path / "a")]) == 0
    assert R.main([*base, "--runs-root", str(tmp_path / "b"), "--device", "cpu"]) == 0
    ra = read_json(tmp_path / "a" / DS / "SMOKE" / "PRE_s2026" / "RESULT.json")
    rb = read_json(tmp_path / "b" / DS / "SMOKE" / "PRE_s2026" / "RESULT.json")
    assert ra == rb and "device_class" not in ra
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""


# ------------------------------------------------------------------------------------------- records / helpers
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


def test_nc_tf32_and_non_strict_numerics_refused(monkeypatch):
    cuda = torch.device("cuda")                                  # a device object only: no CUDA call is made
    saved = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
             torch.get_float32_matmul_precision(), torch.are_deterministic_algorithms_enabled(),
             torch.is_deterministic_algorithms_warn_only_enabled(), torch.backends.cudnn.deterministic,
             torch.backends.cudnn.benchmark)
    try:
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", accel.CUBLAS_CFG)
        from fl.numerics import enforce_strict_fp32
        enforce_strict_fp32(deterministic=True, warn_only=False)
        accel.assert_numerics(cuda)                              # the registered GPU numerics pass
        torch.backends.cuda.matmul.allow_tf32 = True             # NC: a TF32-on run
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
        accel.assert_numerics(torch.device("cpu"))               # the CPU class is not affected by this guard
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved[0], saved[1]
        torch.set_float32_matmul_precision(saved[2])
        torch.use_deterministic_algorithms(saved[3], warn_only=saved[4])
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = saved[5], saved[6]


def test_nc_cuda_refused_without_cublas_or_gpu(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(BenchRefused, match="CUBLAS"):
        accel.torch_device("cuda")
    if not torch.cuda.is_available():
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", accel.CUBLAS_CFG)
        with pytest.raises(BenchRefused, match="one visible CUDA device"):
            accel.torch_device("cuda")


def test_cli_device_flags(release, tmp_path, capsys):
    base = ["run", "--dataset", DS, "--data-root", str(release["root"]), "--runs-root", str(tmp_path / "r"),
            "--threads", "1", "--seed", "2026", "--arm", "PRE", "--smoke"]
    assert R.main([*base, "--gpu", "0"]) == 2 and "--gpu needs --device cuda" in capsys.readouterr().err
    assert R.main([*base, "--device", "cuda"]) == 2 and "--gpu" in capsys.readouterr().err
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""             # a refused cuda request never touched the env
    with pytest.raises(SystemExit):
        R.main([*base, "--device", "tpu"])
    # BM-05: `final` has --device too (Fable §11 E1); cuda without --gpu is refused before anything is touched
    assert R.main(["final", "--dataset", DS, "--threads", "1", "--device", "cuda"]) == 2
    assert "--gpu" in capsys.readouterr().err and os.environ["CUDA_VISIBLE_DEVICES"] == ""


def test_cli_cuda_refused_cleanly_without_a_gpu(release, tmp_path):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", BM_ALLOW_THREADS="1")
    env.pop("CUDA_VISIBLE_DEVICES", None)
    probe = subprocess.run([sys.executable, "-c", "import torch; print(torch.cuda.is_available())"], env=env,
                           capture_output=True, text=True)
    if probe.stdout.strip() == "True":
        pytest.skip("a CUDA device is visible: the GPU path is covered by test_gpu_two_runs_identical")
    p = subprocess.run([sys.executable, "-B", "-m", "v4bench.run", "run", "--dataset", DS, "--data-root",
                        str(release["root"]), "--runs-root", str(tmp_path / "r"), "--threads", "1", "--seed", "2026",
                        "--arm", "PRE", "--smoke", "--device", "cuda", "--gpu", "0"], cwd=str(CODE), env=env,
                       capture_output=True, text=True, timeout=600)
    assert p.returncode == 2 and "REFUSED" in p.stderr and "CUDA" in p.stderr
    assert not (tmp_path / "r" / DS / "SMOKE" / "PRE_s2026" / "RESULT.json").exists()


# ------------------------------------------------------------------------------------------- run-level NCs
@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setattr(central, "EFFECTIVE_BATCH", 8)
    base = tmp_path_factory.mktemp("bm03")
    rel = make_release(base / "processed", n_users=40, n_items=25)
    d = D.load(DS, rel["root"])
    runs = base / "runs"
    rd = lambda n: runs / DS / n                                 # noqa: E731
    central.run_central(data=d, arm="PRE", variant="SASREC_D64_B2", peak_lr=7.114e-4, dropout=(0.1,), seed=2026,
                        rows=d.band_rows(), select_users=np.flatnonzero(d.band), run_dir=rd("PRE_s2026"), run_id="p",
                        smoke_epochs=1.0, device="cpu")
    fl_engine.run_fl(data=d, arm="FA", seed=2026, run_dir=rd("FA_s2026"), run_id="f", smoke_rounds=2, device="cpu")
    yield {"d": d, "root": rel["root"], "runs": runs, "rd": rd}
    mp.undo()


def _as_gpu_copy(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst)
    res = read_json(dst / "RESULT.json")
    res["device_class"] = GPU
    (dst / "RESULT.json").write_text(json.dumps(res))
    return dst


def test_nc_resume_refuses_another_device_class(tiny):
    d, rd = tiny["d"], tiny["rd"]
    run_dir = rd("C_RESUME_s2026")
    run_dir.mkdir(parents=True)
    (run_dir / "ARM_SPEC.json").write_text(json.dumps({"arm": "PRE", "device_class": GPU}))
    with pytest.raises(BenchRefused, match="device class"):
        central.run_central(data=d, arm="PRE", variant="SASREC_D64_B2", peak_lr=7.114e-4, dropout=(0.1,), seed=2026,
                            rows=d.band_rows(), select_users=np.flatnonzero(d.band), run_dir=run_dir, run_id="p",
                            resume=True, device="cpu")
    fl_dir = rd("FA_RESUME_s2026")
    fl_dir.mkdir(parents=True)
    (fl_dir / "ARM_SPEC.json").write_text(json.dumps({"arm": "FA", "arm_spec": {"device_class": GPU}}))
    with pytest.raises(BenchRefused, match="device class"):
        fl_engine.run_fl(data=d, arm="FA", seed=2026, run_dir=fl_dir, run_id="f", smoke_rounds=2, resume=True,
                         device="cpu")


def test_nc_resume_cpu_run_on_the_gpu_class(monkeypatch):
    cuda = torch.device("cuda")
    rec = {"device_class": GPU, "gpu_name": "NVIDIA A40", "torch": "x", "cuda": "y", "cudnn": 1}
    monkeypatch.setattr(accel, "device_class", lambda dev: GPU if accel.is_cuda(dev) else CPU)
    monkeypatch.setattr(accel, "device_record", lambda dev: dict(rec) if accel.is_cuda(dev) else None)
    with pytest.raises(BenchRefused, match="device class CPU-FP32"):
        accel.check_resume({"arm": "PRE", "spec_sha256": "x"}, cuda)            # a CPU ARM_SPEC (no field)
    accel.check_resume({"device_class": GPU, "device": dict(rec)}, cuda)       # same class + build: allowed
    with pytest.raises(BenchRefused, match="torch"):
        accel.check_resume({"device_class": GPU, "device": dict(rec, torch="other")}, cuda)
    accel.check_resume({"arm": "PRE"}, torch.device("cpu"))


def test_nc_theta_pre_s_record_and_ft_base_of_another_class(tiny, tmp_path):
    d, rd = tiny["d"], tiny["rd"]
    gpre = _as_gpu_copy(rd("PRE_s2026"), tmp_path / "PRE_gpu")
    with pytest.raises(BenchRefused, match="theta_pre is a NVIDIA_A40-FP32-DET run"):
        fl_engine.run_fl(data=d, arm="FA_WARM", seed=2026, run_dir=tmp_path / "w", run_id="w", pre_run_dir=gpre,
                         smoke_rounds=2, device="cpu")
    n_clients = len(fl_engine.client_keys(d)[1])
    so = {"format": fl_engine.S_FORMAT, "S": 1.0, "n_norms": 1, "dataset": DS, "variant": "COLD", "n_clients": n_clients,
          "m": 1, "init_digest": None, "label": "SMOKE_NOT_EVIDENCE", "rounds_0based": [0], "rule": "x", "run_id": "s",
          "spec_sha256": "x", "seed": 2026, "device_class": GPU}
    so["self_sha256"] = canon_sha256(so)
    sp = tmp_path / "S_RECORD_COLD_s2026.json"
    sp.write_text(json.dumps(so))
    with pytest.raises(BenchRefused, match="S record is NVIDIA_A40-FP32-DET"):
        fl_engine.run_fl(data=d, arm="FA_1024", seed=2026, run_dir=tmp_path / "f1024", run_id="x", s_record=sp,
                         smoke_rounds=2, device="cpu")
    gfa = _as_gpu_copy(rd("FA_s2026"), tmp_path / "FA_gpu")
    with pytest.raises(BenchRefused, match="the FL base is a NVIDIA_A40-FP32-DET run"):
        FT.run_ft(data=d, data_root=tiny["root"], arm="FT_FA", seed=2026, base_run_dir=gfa, run_dir=tmp_path / "ft",
                  run_id="ft", workers=1, threads=1, limit_users=2, device="cpu")


# ------------------------------------------------------------------------------------------- decisions / LOCK NCs
def fake_run(runs, name, arm, seed, value, *, budget=6, max_len=50, dropout=0.1, kind="fl", dclass=None):
    d = runs / DS / name
    d.mkdir(parents=True)
    spec = {"seed": seed, "cfg": {"seed": seed}, "budget_efe": budget, "max_len": max_len, "dropout": [dropout],
            "select_population": "ALL"}
    summ = {g: {"ndcg@10": value} for g in ("ALL", "BAND", "OTHERS")}
    res = {"status": "DONE_MAIN", "arm": arm, "kind": kind, "spec": spec, "validation": {"PRACTICAL_BEST": summ},
           "checkpoints": {}}
    if dclass:
        res["device_class"] = dclass
    (d / "RESULT.json").write_text(json.dumps(res))


def test_nc_trackb_select_refuses_mixed_devices(tmp_path):
    fake_run(tmp_path, "C_FULL_s2026", "C_FULL", 2026, 0.10, kind="central")
    fake_run(tmp_path, "C_FULL__B24_L200_D0.1_s2026", "C_FULL", 2026, 0.12, budget=24, max_len=200, kind="central",
             dclass=GPU)
    with pytest.raises(BenchRefused, match="mixed"):
        selection.trackb_select(tmp_path, DS, "C_FULL", ["C_FULL_s2026", "C_FULL__B24_L200_D0.1_s2026"])
    assert not (tmp_path / DS / "trackb").exists()                         # refused before any WINNER file
    t2 = tmp_path / "g"
    fake_run(t2, "C_FULL_s2026", "C_FULL", 2026, 0.10, kind="central", dclass=GPU)
    fake_run(t2, "C_FULL__B24_L200_D0.1_s2026", "C_FULL", 2026, 0.12, budget=24, max_len=200, kind="central",
             dclass=GPU)
    obj = selection.trackb_select(t2, DS, "C_FULL", ["C_FULL_s2026", "C_FULL__B24_L200_D0.1_s2026"])
    assert obj["device_class"] == GPU


def test_nc_decide_warm_and_seed3_refuse_mixed_devices(tmp_path):
    for sd in (2026, 2027):
        fake_run(tmp_path, f"FA_WARM_s{sd}", "FA_WARM", sd, 0.10, dclass=GPU)
        fake_run(tmp_path, f"FA_WARM_FROZEN_s{sd}", "FA_WARM_FROZEN", sd, 0.10, dclass=GPU if sd == 2026 else None)
    with pytest.raises(BenchRefused, match="mixed"):
        selection.decide_warm(tmp_path, DS, ["FA_WARM_s2026", "FA_WARM_s2027"],
                              ["FA_WARM_FROZEN_s2026", "FA_WARM_FROZEN_s2027"])
    assert not (tmp_path / DS / "FA_WARM_DECISION.json").exists()
    for sd in (2026, 2027):
        fake_run(tmp_path, f"C_FULL_s{sd}", "C_FULL", sd, 0.10, kind="central", dclass=GPU)
    (tmp_path / DS / "FA_WARM_DECISION.json").write_text(json.dumps({"FA_WARM_STAR": "FA_WARM", "device_class": CPU}))
    with pytest.raises(BenchRefused, match="mixed"):                       # GPU runs vs a CPU-class decision
        selection.seed3_check(tmp_path, DS, {"C_FULL": ["C_FULL_s2026", "C_FULL_s2027"],
                                             "FA_WARM": ["FA_WARM_s2026", "FA_WARM_s2027"]})
    (tmp_path / DS / "FA_WARM_DECISION.json").write_text(json.dumps({"FA_WARM_STAR": "FA_WARM", "device_class": GPU}))
    obj = selection.seed3_check(tmp_path, DS, {"C_FULL": ["C_FULL_s2026", "C_FULL_s2027"],
                                               "FA_WARM": ["FA_WARM_s2026", "FA_WARM_s2027"]})
    assert obj["device_class"] == GPU


def test_nc_lock_refuses_mixed_devices(tmp_path):
    fake_run(tmp_path, "C_FULL_s2026", "C_FULL", 2026, 0.1, kind="central", dclass=GPU)
    fake_run(tmp_path, "FA_s2026", "FA", 2026, 0.1)                       # CPU-FP32 (no field)
    with pytest.raises(BenchRefused, match="LOCK refused: device classes are mixed"):
        final.make_lock(tmp_path, DS, ["C_FULL_s2026", "FA_s2026"], data_fingerprint="x", strict=False)
    assert not final.lock_path(tmp_path, DS).exists()
    t2 = tmp_path / "g"
    fake_run(t2, "C_FULL_s2026", "C_FULL", 2026, 0.1, kind="central", dclass=GPU)
    fake_run(t2, "FA_s2026", "FA", 2026, 0.1, dclass=GPU)
    (t2 / DS / "MANIFEST.json").write_text("{}")                          # data only: no device class, ignored
    (t2 / DS / "dp").mkdir()
    (t2 / DS / "dp" / "S_RECORD_COLD_s2026.json").write_text(json.dumps({"S": 1.0}))     # a CPU-class S record
    with pytest.raises(BenchRefused, match="dp/S_RECORD_COLD_s2026.json"):
        final.make_lock(t2, DS, ["C_FULL_s2026", "FA_s2026"], data_fingerprint="x", strict=False)
    (t2 / DS / "dp" / "S_RECORD_COLD_s2026.json").write_text(json.dumps({"S": 1.0, "device_class": GPU}))
    lock = final.make_lock(t2, DS, ["C_FULL_s2026", "FA_s2026"], data_fingerprint="x", strict=False)
    assert lock["device_class"] == GPU and read_json(final.lock_path(t2, DS))["device_class"] == GPU
    t3 = tmp_path / "c"
    fake_run(t3, "FA_s2026", "FA", 2026, 0.1)
    assert final.make_lock(t3, DS, ["FA_s2026"], data_fingerprint="x", strict=False)["device_class"] == CPU


# ------------------------------------------------------------------------------------------- GPU (Ops, A40 server)
def _gpu_id():
    g = os.environ.get("BM_GPU_TEST")
    if g is None:
        pytest.skip("GPU determinism test: set BM_GPU_TEST=<physical GPU id> (Ops, A40 server)")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
    p = subprocess.run([sys.executable, "-c", "import torch; print(torch.cuda.is_available())"], env=env,
                       capture_output=True, text=True)
    if p.stdout.strip() != "True":
        pytest.skip(f"no CUDA device visible as GPU {g}")
    return int(g)


def test_gpu_two_runs_identical(tmp_path):
    """The same tiny job set (PRE, C_FULL budget 24, FA, FA_WARM, FA_Q8, PF, FT_FA) twice on the GPU class: every
    checkpoint / ranks / record digest identical; every record carries the GPU class; theta0 equals the CPU theta0."""
    g = _gpu_id()
    rel = make_release(tmp_path / "processed", n_users=60, n_items=30)
    a = _helper(CODE, CODE, rel["root"], tmp_path / "g1", tmp_path / "g1.json", "cuda", gpu=g)
    b = _helper(CODE, CODE, rel["root"], tmp_path / "g2", tmp_path / "g2.json", "cuda", gpu=g)
    assert a["status"] == b["status"] and set(a["status"].values()) <= {"DONE_MAIN", "PARTIAL_LIMIT"}
    diff = sorted(k for k in set(a["files"]) | set(b["files"]) if a["files"].get(k) != b["files"].get(k))
    assert diff == [], diff
    want = os.environ.get("BM_GPU_CLASS", GPU)
    for res in (tmp_path / "g1").rglob("RESULT.json"):
        assert accel.device_class_of(read_json(res)) == want, res
    for st in (tmp_path / "g1").rglob("STATUS.json"):
        assert read_json(st).get("device_class") == want, st
    c = _helper(CODE, CODE, rel["root"], tmp_path / "c1", tmp_path / "c1.json", "cpu", jobs="central")
    assert c["status"]["PRE"] == "DONE_MAIN"
    spec_g = read_json(tmp_path / "g1" / DS / "PRE_s2026" / "ARM_SPEC.json")
    spec_c = read_json(tmp_path / "c1" / DS / "PRE_s2026" / "ARM_SPEC.json")
    assert spec_g["init_sha256"] == spec_c["init_sha256"]                  # theta0 is device-independent
    assert spec_g["device"]["cublas_workspace_config"] == accel.CUBLAS_CFG
    assert spec_g["device"]["numerics"]["cuda_matmul_allow_tf32"] is False
