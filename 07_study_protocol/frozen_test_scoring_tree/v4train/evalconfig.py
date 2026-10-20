"""v4train.evalconfig — the ONE frozen evaluation configuration every run is evaluated with (PREG1 T3, binding).

The evaluation numerics come only from `v4eval.frozen_config.load()` (EVAL-03): score_block 158,486, class_chunk
32,768, eval_batch 1,024, cache_bytes 1 GiB, manifest order, FP64 metrics, the deterministic flags, and the
comparison-key additions eval_score_block / eval_batch / eval_device_class ("A16-FP32-DET"). This module holds no
value: `frozen()` returns the loader's FrozenEvalConfig (the object v4eval.predict / evaluate require), and
`EvalConfig.from_frozen` mirrors its values in a plain, hashable record for RunConfig / RUN_START. A run whose
EvalConfig differs from the loader's current values is refused (there is no test override: evaluate() accepts only
artifacts produced under the frozen config, so a non-frozen evaluation could never become a result row).

Flags: training keeps the G0 D5 strict-FP32 flags (fl.numerics.enforce_strict_fp32, deterministic algorithms in
warn-only mode, as FL-01); every evaluation switches to the frozen T3 flags (v4eval.frozen_config.apply_runtime_flags:
deterministic algorithms warn_only=False, cuDNN deterministic, benchmark off, TF32 off) and switches back afterwards.
The rows carry the RUNTIME device class; a non-test run on a device other than the frozen class is refused.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional


class EvalConfigError(ValueError):
    pass


def _get(obj: Any, *names, default=None):
    for n in names:
        if isinstance(obj, Mapping) and n in obj:
            return obj[n]
        if not isinstance(obj, Mapping) and hasattr(obj, n):
            return getattr(obj, n)
    return default


@dataclass(frozen=True)
class EvalConfig:
    score_block: int
    class_chunk: int
    eval_batch_size: int
    cache_bytes: int
    eval_device_class: str
    flags: dict = field(default_factory=dict)
    order: str = ""
    metric_accumulation: str = "FP64"
    lock_sha256: Optional[str] = None
    lock_version: Optional[str] = None
    source: str = "v4eval.frozen_config.load()"

    def validate(self) -> None:
        for f in ("score_block", "class_chunk", "eval_batch_size", "cache_bytes"):
            v = getattr(self, f)
            if not isinstance(v, int) or isinstance(v, bool) or v < 1:
                raise EvalConfigError(f"{f} must be a positive int, got {v!r}")
        if not self.eval_device_class:
            raise EvalConfigError("eval_device_class is required")
        if self.metric_accumulation != "FP64":
            raise EvalConfigError("metrics accumulate in FP64")

    def as_json(self) -> dict:
        return asdict(self)

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.as_json(), sort_keys=True).encode("utf-8")).hexdigest()

    @classmethod
    def from_frozen(cls, obj: Any) -> "EvalConfig":
        add = _get(obj, "comparison_key_additions", default={}) or {}
        ec = cls(score_block=int(_get(obj, "score_block")), class_chunk=int(_get(obj, "class_chunk")),
                 eval_batch_size=int(_get(obj, "eval_batch", "eval_batch_size")),
                 cache_bytes=int(_get(obj, "cache_bytes")),
                 eval_device_class=str(_get(obj, "device_class", "eval_device_class", default=None)
                                       or _get(add, "eval_device_class", default="")),
                 flags=dict(_get(obj, "flags", default={}) or {}), order=str(_get(obj, "order", default="")),
                 metric_accumulation=str(_get(obj, "metric_accumulation", default="FP64")),
                 lock_sha256=_get(obj, "lock_sha256", default=None), lock_version=_get(obj, "lock_version", default=None))
        ec.validate()
        return ec


def frozen():
    """The FrozenEvalConfig object from EVAL-03's loader (required by v4eval.predict_adapter / evaluate)."""
    from v4eval import frozen_config
    return frozen_config.load()


def load_frozen_eval_config() -> EvalConfig:
    return EvalConfig.from_frozen(frozen())


def runtime_device_class(device) -> str:
    from v4eval.frozen_config import eval_device_class
    return eval_device_class(device)


__all__ = ["EvalConfig", "EvalConfigError", "frozen", "load_frozen_eval_config", "runtime_device_class"]
