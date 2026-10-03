"""The opt-in deterministic GPU device class.

Two kinds of device class exist; a run never mixes them, and every comparison should use one class per dataset:
  CPU-FP32                the default: torch deterministic algorithms on the CPU. A record WITHOUT a "device_class"
                          field is CPU-FP32: the CPU path writes exactly the records it would write without this module.
  CUDA_<card>-FP32-DET    --device cuda --gpu <physical id>: CUDA_VISIBLE_DEVICES = that id and CUBLAS_WORKSPACE_CONFIG =
                          :4096:8 exported BEFORE CUDA initialises; torch.use_deterministic_algorithms(True) (strict,
                          warn_only off), cudnn.deterministic on, cudnn.benchmark off, TF32 off (matmul + cudnn),
                          float32 matmul precision "highest". The class is named after the card model; every record the
                          runner writes carries it.
RNG (identical on both devices): data order (numpy PCG64), federated participation / drop-out / Poisson cohorts
(numpy), client batch order (CPU torch.Generator), DP noise (CPU generator), the model init (built and recentred on
CPU, then moved: init_sha256 is the same on both devices). Device-specific: dropout masks (the CUDA generator, seeded
with the same seed / the same derived per-visit seed), the 8-bit upload's stochastic-rounding stream, and every
floating-point reduction order. So a GPU run is reproducible on its device class, never bit-equal to its CPU twin.
No torch import at module level; the CLI calls prepare_cuda_env before any torch import.
"""
from __future__ import annotations

import os
from collections.abc import Mapping

from .common import BenchRefused

DEVICE_CLASS_CPU = "CPU-FP32"
CUBLAS_CFG = ":4096:8"
DEVICES = ("cpu", "cuda")


def device_class_of(rec: Mapping | None) -> str:
    """The device class of a RESULT / ARM_SPEC / S record: its own field, else its spec's, else CPU-FP32."""
    if not rec:
        return DEVICE_CLASS_CPU
    for src in (rec, rec.get("spec") or {}, rec.get("arm_spec") or {}):
        if isinstance(src, Mapping) and src.get("device_class"):
            return str(src["device_class"])
    return DEVICE_CLASS_CPU


def is_cuda(device) -> bool:
    """True only for "cuda[:i]" or a torch.device of type cuda (anything else, e.g. a module object, is not)."""
    if device is None:
        return False
    if isinstance(device, str):
        return device.split(":")[0] == "cuda"
    return getattr(device, "type", None) == "cuda"


def _check_device_arg(device) -> None:
    """Refuse anything but None, "cpu", "cuda[:i]" or a torch.device of those types (never a silent CPU)."""
    if device is None or getattr(device, "type", None) in DEVICES:
        return
    if isinstance(device, str) and device.split(":")[0] in DEVICES:
        return
    raise BenchRefused(f"unknown device {device!r}: {DEVICES}")


def prepare_cuda_env(gpu) -> None:
    """--device cuda: export CUDA_VISIBLE_DEVICES = <gpu> and CUBLAS_WORKSPACE_CONFIG before CUDA starts."""
    if gpu is None or int(gpu) < 0:
        raise BenchRefused("--device cuda needs an explicit physical GPU id (--gpu >= 0)")
    import sys
    if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized():
        raise BenchRefused("CUDA was initialised before the device class could be set up (CUBLAS_WORKSPACE_CONFIG)")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(gpu))
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_CFG


def torch_device(device=None):
    """torch.device for a run: None / "cpu" -> cpu (no side effect, the CPU path is untouched); "cuda" -> the single
    visible GPU, after the strict deterministic numerics are enforced and verified."""
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
    return dev


def enforce(dev) -> dict:
    """Strict FP32 + deterministic algorithms; on CUDA warn_only is OFF (an op without a deterministic kernel raises)."""
    from ppsi.fedsim.numerics import enforce_strict_fp32
    rec = enforce_strict_fp32(deterministic=True, warn_only=not is_cuda(dev))
    assert_numerics(dev)
    return rec


def assert_numerics(dev) -> None:
    """Guard of the GPU class: refuses TF32 on, a lower matmul precision, non-strict determinism, cudnn benchmark or a
    missing CUBLAS workspace config. Called at the start of a CUDA run and at every evaluation mark."""
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
        raise BenchRefused(f"{device_class(dev)}: numerics refused ({', '.join(bad)})")


def device_class(dev) -> str:
    if not is_cuda(dev):
        return DEVICE_CLASS_CPU
    import torch
    name = torch.cuda.get_device_name(dev)
    return "CUDA_" + "".join(c if c.isalnum() else "_" for c in name) + "-FP32-DET"


def device_record(dev) -> dict | None:
    """The GPU leg's identity (None on CPU: the CPU records stay unchanged). Written into ARM_SPEC / RESULT."""
    if not is_cuda(dev):
        return None
    import torch

    from ppsi.fedsim.numerics import numerics_record
    return {"device_class": device_class(dev), "gpu_name": torch.cuda.get_device_name(dev),
            "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "torch": torch.__version__,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"), "numerics": numerics_record()}


RESUME_KEYS = ("device_class", "gpu_name", "torch", "cuda", "cudnn")


def check_resume(prev: Mapping | None, dev) -> None:
    """--resume refuses a device-class mismatch (and, on a GPU class, another card model / torch / CUDA / cuDNN)."""
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
    """The single device class of a set of runs / records of ONE dataset; refuses a mix."""
    got = sorted(set(classes.values()))
    if len(got) > 1:
        by = {c: sorted(k for k, v in classes.items() if v == c) for c in got}
        raise BenchRefused(f"{what} refused: device classes are mixed within one dataset {by}")
    return got[0] if got else DEVICE_CLASS_CPU
