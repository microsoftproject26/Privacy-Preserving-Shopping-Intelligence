"""Test configuration for the sequence models: strict FP32, deterministic algorithms and one CPU thread for every test
in this directory; the previous global settings are restored after each test.

Negative controls are `xfail(strict=True)` tests named test_nc_*: each runs a key-property check against a
deliberately wrong variant and MUST fail with AssertionError (an XPASS fails the suite).
"""
from __future__ import annotations

import pytest
import torch

from ppsi.fedsim.numerics import enforce_strict_fp32

THREADS = 1


@pytest.fixture(autouse=True)
def _strict_numerics():
    saved = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(),
             torch.is_deterministic_algorithms_warn_only_enabled(), torch.get_float32_matmul_precision(),
             torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    torch.set_num_threads(THREADS)
    enforce_strict_fp32()
    yield
    threads, det, warn_only, precision, cudnn_det, cudnn_bench = saved
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(det, warn_only=warn_only)
    torch.set_float32_matmul_precision(precision)
    torch.backends.cudnn.deterministic = cudnn_det
    torch.backends.cudnn.benchmark = cudnn_bench
