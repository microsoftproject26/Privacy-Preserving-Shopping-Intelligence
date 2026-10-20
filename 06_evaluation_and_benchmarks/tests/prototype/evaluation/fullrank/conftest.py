"""Test configuration for the full-catalogue evaluator: one CPU thread, strict FP32 and the evaluation config's
runtime flags (deterministic algorithms with warn_only=False) for every test in this directory (re-applied before
each test, so a test that changes a flag cannot leak it); the previous global settings are restored after each test
module.

Negative controls are `xfail(strict=True, raises=AssertionError)` tests named test_nc_*: each runs a key-property
check against a deliberately wrong variant and MUST fail (an XPASS fails the suite). All data are synthetic.
"""
from __future__ import annotations

import pytest
import torch

from ppsi.evaluation.fullrank.config import EvalConfig, apply_runtime_flags


def _apply():
    torch.set_num_threads(1)
    apply_runtime_flags(EvalConfig())


@pytest.fixture(autouse=True, scope="module")
def _strict_numerics():
    saved = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(),
             torch.is_deterministic_algorithms_warn_only_enabled(), torch.get_float32_matmul_precision(),
             torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
             torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    _apply()
    yield
    threads, det, warn_only, precision, cudnn_det, cudnn_bench, tf32_mm, tf32_cudnn = saved
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(det, warn_only=warn_only)
    torch.set_float32_matmul_precision(precision)
    torch.backends.cudnn.deterministic = cudnn_det
    torch.backends.cudnn.benchmark = cudnn_bench
    torch.backends.cuda.matmul.allow_tf32 = tf32_mm
    torch.backends.cudnn.allow_tf32 = tf32_cudnn


@pytest.fixture(autouse=True)
def _reapply_before_each_test(_strict_numerics):
    _apply()
