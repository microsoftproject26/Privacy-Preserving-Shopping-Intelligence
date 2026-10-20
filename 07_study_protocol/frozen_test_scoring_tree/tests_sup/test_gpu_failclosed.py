"""The GPU check is fail-closed when nvidia-smi is missing, errors out, or returns something this
module cannot fully parse and map -- every such case must mark every requested GPU busy, never
free. All nvidia-smi calls are mocked (SYS-03 requirement); no test here depends on a real GPU or
a real nvidia-smi binary.
"""
from __future__ import annotations

from suptestkit import fake_nvidia_smi_runner

from v4sup.gpu import gpu_busy_map, gpu_ready


def test_no_gpu_needed_always_ready():
    runner = fake_nvidia_smi_runner(raise_exc=FileNotFoundError("no nvidia-smi"))
    ok, why = gpu_ready([], runner)
    assert ok


def test_missing_binary_fails_closed():
    runner = fake_nvidia_smi_runner(raise_exc=FileNotFoundError("[Errno 2] No such file or directory: 'nvidia-smi'"))
    busy = gpu_busy_map(runner)
    assert "unknown" in busy
    ok, why = gpu_ready([0], runner)
    assert not ok
    assert "fail-closed" in why


def test_nonzero_exit_fails_closed():
    runner = fake_nvidia_smi_runner(rc_gpu=1)
    busy = gpu_busy_map(runner)
    assert "unknown" in busy
    ok, _ = gpu_ready([0], runner)
    assert not ok


def test_unmapped_compute_app_fails_closed_for_every_gpu():
    runner = fake_nvidia_smi_runner(gpu_csv="0, 00000000:01:00.0", apps_csv="00000000:FF:00.0, 12345")
    busy = gpu_busy_map(runner)
    assert busy == {"0": ["unmapped compute app"]}
    ok, _ = gpu_ready([0], runner)
    assert not ok


def test_no_gpus_enumerated_fails_closed():
    runner = fake_nvidia_smi_runner(gpu_csv="")
    busy = gpu_busy_map(runner)
    assert "unknown" in busy


def test_clean_idle_gpu_is_ready():
    runner = fake_nvidia_smi_runner(gpu_csv="0, 00000000:01:00.0\n1, 00000000:02:00.0", apps_csv="")
    ok, why = gpu_ready([0, 1], runner)
    assert ok, why


def test_clean_busy_gpu_is_not_ready():
    runner = fake_nvidia_smi_runner(gpu_csv="0, 00000000:01:00.0\n1, 00000000:02:00.0",
                                    apps_csv="00000000:01:00.0, 54321")
    ok, why = gpu_ready([0], runner)
    assert not ok
    assert "busy" in why
    # the OTHER gpu, with no compute app, is still free
    ok1, _ = gpu_ready([1], runner)
    assert ok1
