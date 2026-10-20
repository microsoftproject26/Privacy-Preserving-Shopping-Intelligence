"""v4bench.accel — the opt-in deterministic GPU device class (BM-03; FABLE_GATE_ENDGAME_SCRIPTS_20261003 §8, addendum B16).

Two device classes exist; a run, a selection decision and a LOCK never mix them inside one dataset:
  CPU-FP32             the registered default (amendment 07 B6 / R9): torch deterministic algorithms, 16 threads, nice 15.
                       A record WITHOUT a "device_class" field is CPU-FP32: the CPU path writes exactly the bytes it wrote
                       before BM-03 (code_bench4), so its specs, spec hashes, resumes and records are unchanged.
  NVIDIA_A40-FP32-DET  --device cuda --gpu <physical id>: CUDA_VISIBLE_DEVICES = that id and CUBLAS_WORKSPACE_CONFIG = :4096:8
                       exported BEFORE CUDA initialises; torch.use_deterministic_algorithms(True) (strict, warn_only off),
                       cudnn.deterministic on, cudnn.benchmark off, TF32 off (matmul + cudnn), float32 matmul precision
                       "highest" (fl.numerics.enforce_strict_fp32). Every record the runner writes carries the class.
RNG (identical on both devices): data order (v4train.order, numpy PCG64), FL participation / drop-out / Poisson cohorts
(numpy), client batch order (CPU torch.Generator), DP noise (CPU generator), the model init (built and recentred on CPU, then
moved: init_sha256 is the same on both devices). Device-specific: dropout masks (the CUDA generator, seeded with the same
seed / the same derived per-visit seed) and the FA_Q8 stochastic-rounding stream (a generator on the delta's device), and every
floating-point reduction order. So a GPU run is reproducible on the GPU class, never bit-equal to its CPU twin.
No torch import at module level (common.py rule); the CLI calls prepare_cuda_env before any torch import.
"""
from __future__ import annotations

import os
from typing import Mapping, Optional

from .common import BenchRefused

DEVICE_CLASS_CPU = "CPU-FP32"                    # amendment 07 B6 / R9 (registered string)
DEVICE_CLASS_A40 = "NVIDIA_A40-FP32-DET"         # FABLE_GATE_ENDGAME_SCRIPTS_20261003 §8 (to be registered in B16)
CUBLAS_CFG = ":4096:8"                           # v4train.recipe.CUBLAS_WORKSPACE_CONFIGS_STRICT[0]
DEVICES = ("cpu", "cuda")


def device_class_of(rec: Optional[Mapping]) -> str:
    """The device class of a RESULT / ARM_SPEC / S record / decision: its own field, else its spec's, else CPU-FP32."""
    if not rec:
        return DEVICE_CLASS_CPU
    for src in (rec, rec.get("spec") or {}, rec.get("arm_spec") or {}):
        if isinstance(src, Mapping) and src.get("device_class"):
            return str(src["device_class"])
    return DEVICE_CLASS_CPU


def is_cuda(device) -> bool:
    """True only for "cuda[:i]" or a torch.device of type cuda (BM-05: anything else, e.g. a module object, is not)."""
    if device is None:
        return False
    if isinstance(device, str):
        return device.split(":")[0] == "cuda"
    return getattr(device, "type", None) == "cuda"


def _check_device_arg(device) -> None:
    """BM-05: refuse anything but None, "cpu", "cuda[:i]" or a torch.device of those types (never a silent CPU)."""
    if device is None or getattr(device, "type", None) in DEVICES:
        return
    if isinstance(device, str) and device.split(":")[0] in DEVICES:
        return
    raise BenchRefused(f"unknown device {device!r}: {DEVICES}")


def prepare_cuda_env(gpu) -> None:
    """--device cuda: export CUDA_VISIBLE_DEVICES = <gpu> and CUBLAS_WORKSPACE_CONFIG before CUDA starts (run.py blanks
    CUDA_VISIBLE_DEVICES at import for the CPU class; snap_v1/reuse/train.py only setdefaults it)."""
    if gpu is None or int(gpu) < 0:
        raise BenchRefused("--device cuda needs an explicit physical GPU id (--gpu >= 0)")
    import sys
    if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized():
        raise BenchRefused("CUDA was initialised before the device class could be set up (CUBLAS_WORKSPACE_CONFIG)")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(gpu))
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_CFG


def torch_device(device=None):
    """torch.device for a run: None / "cpu" -> cpu (no side effect, the CPU path is untouched); "cuda" -> the single visible
    GPU, after the strict deterministic numerics are enforced and verified."""
    import torch
    _check_device_arg(device)
    if not is_cuda(device):
        return torch.device("cpu")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS_CFG:
        raise BenchRefused(f"--device cuda: CUBLAS_WORKSPACE_CONFIG must be {CUBLAS_CFG} before CUDA starts")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise BenchRefused("--device cuda: expected exactly one visible CUDA device (CUDA_VISIBLE_DEVICES = --gpu)")
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    enforce(dev)
    device_class(dev)                            # refuses a non-A40 card (unless BM_ALLOW_ANY_GPU=1, tests only)
    return dev


def enforce(dev) -> dict:
    """Strict FP32 + deterministic algorithms; on CUDA warn_only is OFF (an op without a deterministic kernel raises)."""
    from fl.numerics import enforce_strict_fp32
    rec = enforce_strict_fp32(deterministic=True, warn_only=not is_cuda(dev))
    assert_numerics(dev)
    return rec


def assert_numerics(dev) -> None:
    """NC guard of the GPU class: refuses TF32 on, a lower matmul precision, non-strict determinism, cudnn benchmark or a
    missing CUBLAS workspace config. Called at the start of a cuda run and at every evaluation mark."""
    if not is_cuda(dev):
        return
    import torch
    bad = []
    if torch.backends.cuda.matmul.allow_tf32:
        bad.append("cuda.matmul.allow_tf32=True")
    if torch.backends.cudnn.allow_tf32:
        bad.append("cudnn.allow_tf32=True")
    if torch.get_float32_matmul_precision() != "highest":
        bad.append(f"float32_matmul_precision={torch.get_float32_matmul_precision()!r}")
    if not torch.are_deterministic_algorithms_enabled() or torch.is_deterministic_algorithms_warn_only_enabled():
        bad.append("deterministic algorithms not strict")
    if not torch.backends.cudnn.deterministic or torch.backends.cudnn.benchmark:
        bad.append("cudnn not deterministic / benchmark on")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS_CFG:
        bad.append(f"CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG')!r}")
    if bad:
        raise BenchRefused(f"{DEVICE_CLASS_A40}: numerics refused ({', '.join(bad)})")


def device_class(dev) -> str:
    if not is_cuda(dev):
        return DEVICE_CLASS_CPU
    import torch
    name = torch.cuda.get_device_name(dev)
    if "A40" in name:
        return DEVICE_CLASS_A40
    if os.environ.get("BM_ALLOW_ANY_GPU") == "1":            # tests only: a class that can never equal the A40 one
        return "CUDA_" + "".join(c if c.isalnum() else "_" for c in name) + "-FP32-DET"
    raise BenchRefused(f"--device cuda: {name!r} is not an A40 ({DEVICE_CLASS_A40} is the registered GPU class)")


def device_record(dev) -> Optional[dict]:
    """The GPU leg's identity (None on CPU: the CPU records stay byte-identical). Written into ARM_SPEC / RESULT."""
    if not is_cuda(dev):
        return None
    import torch
    from fl.numerics import numerics_receipt
    return {"device_class": device_class(dev), "gpu_name": torch.cuda.get_device_name(dev),
            "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "torch": torch.__version__,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"), "numerics": numerics_receipt()}


RESUME_KEYS = ("device_class", "gpu_name", "torch", "cuda", "cudnn")


def check_resume(prev: Optional[Mapping], dev) -> None:
    """--resume refuses a device-class mismatch (and, on the GPU class, another card model / torch / CUDA / cuDNN build)."""
    want = device_class(dev)
    got = device_class_of(prev)
    if got != want:
        raise BenchRefused(f"--resume: the run was started on device class {got}, this leg is {want} "
                           "(a run never mixes device classes)")
    if is_cuda(dev):
        old, new = (prev or {}).get("device") or {}, device_record(dev)
        diff = [k for k in RESUME_KEYS if old.get(k) != new.get(k)]
        if diff:
            raise BenchRefused(f"--resume: the GPU leg differs from the run's first leg in {diff}")


def one_class(classes: Mapping[str, str], what: str) -> str:
    """The single device class of a set of runs / records of ONE dataset; refuses a mix (B16: one device per dataset)."""
    got = sorted(set(classes.values()))
    if len(got) > 1:
        by = {c: sorted(k for k, v in classes.items() if v == c) for c in got}
        raise BenchRefused(f"{what} refused: device classes are mixed within one dataset {by}")
    return got[0] if got else DEVICE_CLASS_CPU
