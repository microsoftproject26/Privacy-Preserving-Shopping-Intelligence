"""The explicit UNDEFINED value and strict-JSON serialisation.

An empty population or a zero denominator is UNDEFINED, never NaN and never 0. `Undefined` refuses to act as a number
or a truth value, so it cannot leak silently into arithmetic or a comparison. Serialisation is strict JSON
(allow_nan=False); a non-finite float anywhere is refused.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

EMPTY_POPULATION = "EMPTY_POPULATION"
ZERO_DENOMINATOR = "ZERO_DENOMINATOR"
ZERO_DENOMINATOR_IN_RESAMPLES = "ZERO_DENOMINATOR_IN_RESAMPLES"
MISSING_EVIDENCE = "MISSING_EVIDENCE"


@dataclass(frozen=True)
class Undefined:
    reason: str

    def __bool__(self):
        raise TypeError(f"UNDEFINED ({self.reason}) has no truth value")

    def __float__(self):
        raise TypeError(f"UNDEFINED ({self.reason}) is not a number")

    def __repr__(self) -> str:
        return f"UNDEFINED({self.reason})"


def is_undefined(x: Any) -> bool:
    return isinstance(x, Undefined)


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, Undefined):
        return {"value": None, "status": "UNDEFINED", "reason": obj.reason}
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        if not math.isfinite(f):
            raise ValueError(f"non-finite float {f!r} cannot be serialized (use UNDEFINED)")
        return f
    if isinstance(obj, np.ndarray):
        return [to_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if obj is None or isinstance(obj, str):
        return obj
    raise TypeError(f"cannot serialize {type(obj).__name__}")


def dumps(obj: Any) -> str:
    return json.dumps(to_jsonable(obj), allow_nan=False, sort_keys=True)
