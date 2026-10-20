"""Fixed TRAIN-only T1 scores; the existing frozen evaluator owns all metrics.

Counts are over frozen T1 decision rows, not raw events. No smoothing search,
validation fitting, current-category suppression, or second metric implementation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

import numpy as np

Variant = Literal["popularity", "markov", "last_category"]


def _codes(values, *, count: int, name: str, allow_unknown: bool = False) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a one-dimensional integer array")
    lower = -1 if allow_unknown else 0
    upper = count if allow_unknown else count - 1
    if array.size and (np.any(array < lower) or np.any(array > upper)):
        raise ValueError(f"{name} has a code outside the declared vocabulary")
    return array.astype(np.int64, copy=False)


@dataclass(frozen=True)
class T1CountBaselines:
    """Immutable counts with explicit all-zero Markov/OOV backoff and stable ties."""

    target_counts: np.ndarray
    transition_counts: np.ndarray
    train_decisions: int

    def __post_init__(self) -> None:
        p, t = np.asarray(self.target_counts), np.asarray(self.transition_counts)
        n = len(p) if p.ndim == 1 else 0
        if n < 2 or t.shape != (n, n):
            raise ValueError("count shapes must be [C] and [C,C], with C >= 2")
        if p.dtype.kind not in "iu" or t.dtype.kind not in "iu":
            raise ValueError("counts must be integers")
        if np.any(p < 0) or np.any(t < 0):
            raise ValueError("counts must be non-negative")
        if (
            isinstance(self.train_decisions, bool)
            or not isinstance(self.train_decisions, (int, np.integer))
            or self.train_decisions <= 0
        ):
            raise ValueError("train_decisions must be a positive integer")
        if int(p.sum()) != self.train_decisions or int(t.sum()) != self.train_decisions:
            raise ValueError("count totals must match the frozen TRAIN decision count")
        p, t = p.astype(np.int64, copy=True), t.astype(np.int64, copy=True)
        p.flags.writeable = False
        t.flags.writeable = False
        object.__setattr__(self, "target_counts", p)
        object.__setattr__(self, "transition_counts", t)

    @classmethod
    def fit(cls, current, targets, *, split: str, category_count: int = 588):
        if split != "TRAIN":
            raise ValueError("baseline statistics may be fitted on TRAIN only")
        if (
            isinstance(category_count, bool)
            or not isinstance(category_count, int)
            or category_count < 2
        ):
            raise ValueError("category_count must be an integer >= 2")
        x = _codes(current, count=category_count, name="current")
        y = _codes(targets, count=category_count, name="targets")
        if len(x) == 0 or len(x) != len(y):
            raise ValueError("nonempty, equally sized TRAIN decision arrays are required")
        p = np.bincount(y, minlength=category_count).astype(np.int64)
        t = np.zeros((category_count, category_count), dtype=np.int64)
        np.add.at(t, (x, y), 1)
        return cls(p, t, len(y))

    @property
    def category_count(self) -> int:
        return len(self.target_counts)

    def scores(self, variant: Variant, current) -> np.ndarray:
        """Score a caller-sized chunk; unknown -1/C inputs yield zero Markov/last rows."""
        x = _codes(current, count=self.category_count, name="current", allow_unknown=True)
        if variant == "popularity":
            return np.broadcast_to(self.target_counts, (len(x), self.category_count)).astype(
                np.float64, copy=True
            )
        scores = np.zeros((len(x), self.category_count), dtype=np.float64)
        valid = (x >= 0) & (x < self.category_count)
        if variant == "markov":
            scores[valid] = self.transition_counts[x[valid]]
        elif variant == "last_category":
            rows = np.flatnonzero(valid)
            scores[rows, x[valid]] = 1.0
        else:
            raise ValueError(f"unknown T1 baseline variant: {variant!r}")
        return scores

    def content_sha256(self) -> str:
        """Versioned logical digest, independent of NPZ timestamps/compression."""
        digest = hashlib.sha256(b"t1-count-baselines-v1\0")
        digest.update(str(self.category_count).encode("ascii") + b"\0")
        digest.update(str(self.train_decisions).encode("ascii") + b"\0")
        digest.update(self.target_counts.astype("<i8").tobytes(order="C"))
        digest.update(self.transition_counts.astype("<i8").tobytes(order="C"))
        return digest.hexdigest()
