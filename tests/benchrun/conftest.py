"""Benchmark-runner fixtures: a synthetic leave-one-out release (no real data), and the global torch settings (which
the runner sets for strict numerics) restored after each test module."""
from __future__ import annotations

import pytest
import torch
from benchrun_testkit import make_release


@pytest.fixture()
def release(tmp_path):
    return make_release(tmp_path / "processed")


@pytest.fixture(autouse=True, scope="module")
def _restore_torch_settings():
    saved = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(),
             torch.is_deterministic_algorithms_warn_only_enabled(), torch.get_float32_matmul_precision(),
             torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
             torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    yield
    threads, det, warn_only, precision, cudnn_det, cudnn_bench, tf32_mm, tf32_cudnn = saved
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(det, warn_only=warn_only)
    torch.set_float32_matmul_precision(precision)
    torch.backends.cudnn.deterministic = cudnn_det
    torch.backends.cudnn.benchmark = cudnn_bench
    torch.backends.cuda.matmul.allow_tf32 = tf32_mm
    torch.backends.cudnn.allow_tf32 = tf32_cudnn
