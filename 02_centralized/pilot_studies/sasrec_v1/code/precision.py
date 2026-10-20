"""Precision / determinism enforcement (owner: Opus). Gate P fix 2, registry v2 precision.

Registry v2 (owner-approved, parity-gated): TRAINING steps use TF32 matmul
(`training_precision(True)`); EVERY evaluation runs strictly in FP32 inside `strict_eval()`,
which forces allow_tf32=False / matmul precision "highest" and restores the training flags.

A16 is Ampere: cuDNN TF32 is on by default, so "strict FP32" must be set explicitly,
asserted, and recorded. `tf32_canary` measures an FP32 GEMM against FP64 on the GPU:
true FP32 gives a relative error around 1e-7..1e-6, TF32 (10-bit mantissa) around
1e-4..1e-3, so the threshold 2e-5 separates them with a wide margin.
"""
from __future__ import annotations

import random

import numpy as np
import torch

CANARY_THRESHOLD = 2e-5


def configure_strict_fp32(seed: int) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    return assert_strict_fp32()


def precision_record() -> dict:
    return {
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "deterministic_warn_only": bool(torch.is_deterministic_algorithms_warn_only_enabled()),
        "reduced_precision_fp16_reduction": bool(torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction),
        "reduced_precision_bf16_reduction": bool(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction),
    }


def assert_strict_fp32() -> dict:
    rec = precision_record()
    problems = []
    if rec["cuda_matmul_allow_tf32"]:
        problems.append("cuda.matmul.allow_tf32 is True")
    if rec["cudnn_allow_tf32"]:
        problems.append("cudnn.allow_tf32 is True")
    if rec["float32_matmul_precision"] != "highest":
        problems.append(f"float32_matmul_precision={rec['float32_matmul_precision']}")
    if not rec["deterministic_algorithms"]:
        problems.append("deterministic algorithms disabled")
    if not rec["cudnn_deterministic"] or rec["cudnn_benchmark"]:
        problems.append("cudnn nondeterministic settings")
    if problems:
        raise RuntimeError("strict FP32 violated: " + "; ".join(problems))
    return rec


def tf32_canary(device, n: int = 1024, seed: int = 0) -> dict:
    """Relative Frobenius error of an FP32 GEMM vs FP64 on `device` under CURRENT flags."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    a = torch.randn(n, n, generator=g, dtype=torch.float64)
    b = torch.randn(n, n, generator=g, dtype=torch.float64)
    ref = a @ b
    got = (a.float().to(device) @ b.float().to(device)).double().cpu()
    rel = float((got - ref).norm() / ref.norm())
    return {"n": n, "relative_error": rel, "threshold": CANARY_THRESHOLD,
            "looks_strict_fp32": rel < CANARY_THRESHOLD}


def training_precision(tf32: bool) -> dict:
    """Set the training-phase matmul precision: TF32 (owner-approved) or strict FP32."""
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.set_float32_matmul_precision("high" if tf32 else "highest")
    return assert_training_precision(tf32)


def assert_training_precision(tf32: bool) -> dict:
    rec = precision_record()
    want_prec = "high" if tf32 else "highest"
    problems = []
    if rec["cuda_matmul_allow_tf32"] != bool(tf32) or rec["cudnn_allow_tf32"] != bool(tf32):
        problems.append(f"allow_tf32 flags {rec['cuda_matmul_allow_tf32']}/{rec['cudnn_allow_tf32']} != {tf32}")
    if rec["float32_matmul_precision"] != want_prec:
        problems.append(f"float32_matmul_precision={rec['float32_matmul_precision']} != {want_prec}")
    if not rec["deterministic_algorithms"] or not rec["cudnn_deterministic"] or rec["cudnn_benchmark"]:
        problems.append("determinism settings violated")
    if problems:
        raise RuntimeError("training precision violated: " + "; ".join(problems))
    return rec


class strict_eval:
    """Context manager: strict FP32 for an evaluation, then restore the previous flags."""

    def __enter__(self):
        self._saved = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                       torch.get_float32_matmul_precision())
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        assert_strict_fp32()
        return self

    def __exit__(self, *exc):
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, prec = self._saved
        torch.set_float32_matmul_precision(prec)
        return False
