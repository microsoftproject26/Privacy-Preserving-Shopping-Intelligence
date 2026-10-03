"""The fixed evaluation numerics shared by every method.

float32 block scores are not bit-identical across block widths or batch shapes, so the scoring numerics are part of
the comparison key: every method of a comparison must be scored under one `EvalConfig`.

  score_block   rows of the item table scored at once; at least the catalogue size, so each batch is ONE block and
                the scores are the model's logits bit for bit
  class_chunk   width of the tie-counting chunks (does not change any rank)
  eval_batch    rows per prediction batch, in manifest order (strictly ascending decision_id, tail batch unpadded)
  cache_bytes   memory for cached score blocks between the two ranking passes
  flags         strict FP32 and deterministic algorithms (TF32 off, cuDNN deterministic, no benchmark mode)

`EvalConfig` is immutable. The public prediction and evaluation paths take a config (default `EvalConfig()`), and a
numeric passed next to it must EQUAL the config's value: changing the numerics means building another config, which
changes `sha256` and therefore every row's comparison key. `evaluate()` refuses an artifact produced under a
different config.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import torch

from .errors import EvalConfigError

CPU_DEVICE_CLASS = "CPU-FP32-DET"
DEFAULT_FLAGS = MappingProxyType({
    "matmul.allow_tf32": False, "cudnn.allow_tf32": False, "cudnn.benchmark": False,
    "cudnn.deterministic": True, "use_deterministic_algorithms": True, "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
FIXED_FIELDS = ("frozen", "score_block", "class_chunk", "eval_batch", "cache_bytes", "metric_accumulation",
                "config_sha256")


@dataclass(frozen=True)
class EvalConfig:
    score_block: int = 262144
    class_chunk: int = 32768
    eval_batch: int = 1024
    cache_bytes: int = 1024 * 1024 * 1024
    metric_accumulation: str = "FP64"
    reference_device_class: str | None = None   # rows from another device class are labelled as such
    flags: Mapping = field(default_factory=lambda: DEFAULT_FLAGS)

    def __post_init__(self):
        for name in ("score_block", "class_chunk", "eval_batch", "cache_bytes"):
            v = getattr(self, name)
            if not isinstance(v, int) or isinstance(v, bool) or v < (0 if name == "cache_bytes" else 1):
                raise EvalConfigError(f"{name} must be a positive int, got {v!r}")
        if self.metric_accumulation != "FP64":
            raise EvalConfigError("metrics are accumulated in FP64")
        object.__setattr__(self, "flags", MappingProxyType(dict(self.flags)))

    @property
    def comparison_key_additions(self) -> dict:
        return {"eval_score_block": self.score_block, "eval_batch": self.eval_batch}

    @property
    def sha256(self) -> str:
        doc = {"score_block": self.score_block, "class_chunk": self.class_chunk, "eval_batch": self.eval_batch,
               "cache_bytes": self.cache_bytes, "metric_accumulation": self.metric_accumulation,
               "reference_device_class": self.reference_device_class, "flags": dict(self.flags)}
        return hashlib.sha256(json.dumps(doc, sort_keys=True).encode("utf-8")).hexdigest()

    def eval_config(self, path: str, device_class: str) -> dict:
        """The artifact / result eval_config for this configuration. `eval_stream` is filled in by the prediction
        path with the observed stream form (E2E_ROWS / ALL_ROWS)."""
        return {"frozen": True, "path": path, "score_block": self.score_block, "class_chunk": self.class_chunk,
                "eval_batch": self.eval_batch, "cache_bytes": self.cache_bytes,
                "metric_accumulation": self.metric_accumulation, "eval_device_class": device_class,
                "reference_device_class": self.reference_device_class, "config_sha256": self.sha256}


def resolve(config: EvalConfig | None = None, **given) -> EvalConfig:
    """The config of a public evaluation path: `config` or, if None, EvalConfig(). Any numeric a caller passes
    (score_block, class_chunk, cache_bytes, eval_batch) must EQUAL the config's value; otherwise it is an override
    and is refused."""
    if config is None:
        cfg = EvalConfig()
    elif isinstance(config, EvalConfig):
        cfg = config
    else:
        raise EvalConfigError(f"an EvalConfig is required, not {type(config).__name__}")
    bad = {k: v for k, v in given.items() if v is not None and int(v) != int(getattr(cfg, k))}
    if bad:
        raise EvalConfigError(f"override refused: {bad} differ from the configured values "
                              f"{ {k: getattr(cfg, k) for k in bad} }")
    return cfg


def apply_runtime_flags(cfg: EvalConfig) -> None:
    """Set the configured flags (for launchers; CUBLAS_WORKSPACE_CONFIG must be set before CUDA initialises)."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", cfg.flags["CUBLAS_WORKSPACE_CONFIG"])
    torch.backends.cuda.matmul.allow_tf32 = cfg.flags["matmul.allow_tf32"]
    torch.backends.cudnn.allow_tf32 = cfg.flags["cudnn.allow_tf32"]
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = cfg.flags["cudnn.benchmark"]
    torch.backends.cudnn.deterministic = cfg.flags["cudnn.deterministic"]
    torch.use_deterministic_algorithms(cfg.flags["use_deterministic_algorithms"], warn_only=False)


def check_runtime_flags(cfg: EvalConfig, device: torch.device) -> None:
    bad = []
    f = cfg.flags
    if bool(torch.backends.cuda.matmul.allow_tf32) != f["matmul.allow_tf32"]:
        bad.append("torch.backends.cuda.matmul.allow_tf32")
    if bool(torch.backends.cudnn.allow_tf32) != f["cudnn.allow_tf32"]:
        bad.append("torch.backends.cudnn.allow_tf32")
    if torch.get_float32_matmul_precision() != "highest":
        bad.append("float32_matmul_precision != highest")
    if bool(torch.backends.cudnn.benchmark) != f["cudnn.benchmark"]:
        bad.append("torch.backends.cudnn.benchmark")
    if bool(torch.backends.cudnn.deterministic) != f["cudnn.deterministic"]:
        bad.append("torch.backends.cudnn.deterministic")
    if not torch.are_deterministic_algorithms_enabled() or torch.is_deterministic_algorithms_warn_only_enabled():
        bad.append("use_deterministic_algorithms(True, warn_only=False)")
    if (torch.device(device).type == "cuda"
            and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != f["CUBLAS_WORKSPACE_CONFIG"]):
        bad.append("CUBLAS_WORKSPACE_CONFIG")
    if bad:
        raise EvalConfigError("runtime flags differ from the configured values: " + ", ".join(bad))


def eval_device_class(device: torch.device) -> str:
    """A device class per GPU model (and one for the CPU), so rows never mix devices silently."""
    d = torch.device(device)
    if d.type == "cuda":
        name = torch.cuda.get_device_name(d)
        return "".join(c if c.isalnum() else "_" for c in name) + "-FP32-DET"
    return CPU_DEVICE_CLASS if d.type == "cpu" else f"{d.type.upper()}-FP32-DET"


def check_artifact_config(cfg: EvalConfig, eval_config: Mapping | None) -> None:
    """The artifact must have been produced under exactly this configuration."""
    want = cfg.eval_config(path="*", device_class="*")
    diff = [k for k in FIXED_FIELDS if (eval_config or {}).get(k) != want[k]]
    if diff:
        raise EvalConfigError(f"artifact eval_config differs from the evaluation config in {diff}")
