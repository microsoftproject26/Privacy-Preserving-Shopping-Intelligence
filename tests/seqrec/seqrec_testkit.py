"""Shared helpers for the sequence-model tests (synthetic fixtures only; no data rows)."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from ppsi.fedsim.numerics import seeded_global_rng
from ppsi.seqrec.adapters import make_adapter
from ppsi.seqrec.build import construct_module, default_vocab
from ppsi.seqrec.features import InputWidths, gru_kwargs, sasrec_kwargs
from ppsi.seqrec.gru import ContextGRU
from ppsi.seqrec.synthetic import synthetic_batch

# a rich layout: 15 numeric features, 15 quality flags, 12 user-context values and 4 user-context masks
W = InputWidths(15, 15, 12, 4, view_id="rich",
                numeric_names=tuple(f"num_{i}" for i in range(15)),
                flag_names=tuple(f"flag_{i}" for i in range(15)),
                user_context_names=tuple(f"user_{i}" for i in range(12)),
                user_mask_names=tuple(f"user_mask_{i}" for i in range(4)))
DEAD_FLAGS = (2, 5, 7, 9, 11, 14)   # flag columns that never fire in the synthetic training data
FAMS = ["GRU", "SASREC"]
K_SMALL = 64


def nc(reason: str):
    """Negative control: the wrapped key-property check must raise AssertionError (strict xfail)."""
    return pytest.mark.xfail(strict=True, raises=AssertionError, reason="NEGATIVE CONTROL: " + reason)


def assert_raises(exc, fn, *a, **kw):
    """Like pytest.raises, but a missing exception is an AssertionError (so negative controls can use it)."""
    try:
        fn(*a, **kw)
    except exc:
        return
    raise AssertionError(f"{getattr(fn, '__name__', fn)} did not raise {exc}")


def vocab(K: int) -> dict:
    return default_vocab(K)


def poc(K: int) -> np.ndarray:
    return np.arange(3, 3 + int(K), dtype=np.int64)


def batch(n: int, K: int = K_SMALL, widths=W, seed: int = 0, **kw) -> dict:
    return synthetic_batch(n, K, widths, torch.Generator().manual_seed(seed), vocab=vocab(K), **kw)


def model(family: str, seed: int, K: int = K_SMALL, widths=W, impl: str = "unpadded"):
    """The module with its raw (un-recentered) init, on CPU."""
    return construct_module(family, widths, seed, vocab_sizes=vocab(K), poc=poc(K), impl=impl)


def adapter(family: str, seed: int = 2026, K: int = K_SMALL, widths=W, impl: str = "unpadded"):
    return make_adapter(model(family, seed, K, widths, impl), widths)


def is_gru(module) -> bool:
    return isinstance(module, ContextGRU)


def kwargs_for(module, b: dict) -> dict:
    return gru_kwargs(b, module) if is_gru(module) else sasrec_kwargs(b)


def forward(module, b: dict) -> torch.Tensor:
    return module(**kwargs_for(module, b))


def query(module, b: dict) -> torch.Tensor:
    return module.query(**kwargs_for(module, b))


class KShapeRecorder(TorchDispatchMode):
    """Records the shape of every tensor produced by every aten op (forward AND backward) while active.

    `violations(K, allowed_rows)`: tensors with a dimension equal to K whose remaining size (numel // K) is not an
    allowed row count. For the one-decision objective the only K-sized tensors are the bias [K], the head table [K, d]
    and its gradient, and the [B, K] logits / softmax / gradients (rows 1, d or B); a [B, L, K] or [B*L, K]
    all-position logit tensor has rows B*L (or the number of real tokens) and is flagged."""

    def __init__(self):
        super().__init__()
        self.records = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for t in tree_flatten(out)[0]:
            if isinstance(t, torch.Tensor):
                self.records.append((str(func), tuple(t.shape)))
        return out

    def k_tensors(self, K: int) -> list:
        return [(f, s) for f, s in self.records if K in s]

    def violations(self, K: int, allowed_rows) -> list:
        allowed = {int(r) for r in allowed_rows}
        bad = []
        for f, s in self.k_tensors(K):
            numel = int(np.prod(s)) if s else 1
            if numel // K not in allowed:
                bad.append((f, s))
        return bad

    def peak_k_rows(self, K: int) -> int:
        return max((int(np.prod(s)) // K for _, s in self.k_tensors(K)), default=0)


def train_step_under(recorder: KShapeRecorder, fn: Callable[[], torch.Tensor]) -> None:
    with recorder:
        loss = fn()
        loss.backward()


def ce_mean(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, target)


def seeded(seed: int):
    return seeded_global_rng(seed, torch.device("cpu"))


def maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double() - b.double()).abs().max())
