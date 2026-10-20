"""Shared helpers for the MODEL-01 tests (synthetic fixtures only; no dataset rows)."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

import model as snap_gru_mod                    # snapshot copy: snap_v1/reuse/model.py
import sasrec_model as snap_sasrec_mod          # snapshot copy: snap_v1/sasrec_model.py
import train as reuse_train                     # snapshot copy: snap_v1/reuse/train.py
from fl.families import SASREC_ENTRY, SASREC_KEYS
from fl.numerics import seeded_global_rng
from v4models import gru_v2, sasrec_v2
from v4models.build import construct_module, make_adapter
from v4models.schema import load_schema, v1_reference_widths
from v4models.synthetic import synthetic_examples_v2

SCHEMA = load_schema()
V2 = SCHEMA.widths
V1 = v1_reference_widths(SCHEMA)
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
    v = dict(reuse_train.VOCAB_SIZES)
    v["item_tokens"] = int(K) + 3
    return v


def poc(K: int) -> np.ndarray:
    return np.arange(3, 3 + int(K), dtype=np.int64)


def batch(n: int, K: int = K_SMALL, widths=V2, seed: int = 0, **kw) -> dict:
    return synthetic_examples_v2(n, K, widths, torch.Generator().manual_seed(seed), vocab=vocab(K), **kw)


# ------------------------------------------------------------------------------------------------ model builders
def snap_model(family: str, seed: int, K: int = K_SMALL, impl: str = "unpadded"):
    """The SNAPSHOT model (v1 widths 19/15/12/6), built exactly as fl.families.build_family does."""
    if family == "GRU":
        with seeded_global_rng(seed, torch.device("cpu")):
            return snap_gru_mod.build_model("F4_rich_side_information", vocab(K), product_idx_of_class=poc(K))
    cfg = snap_sasrec_mod.SASRecConfig.from_registry(SASREC_ENTRY)
    return snap_sasrec_mod.SASRecCE(cfg, vocab_sizes=vocab(K), product_idx_of_class=poc(K), seed=seed, impl=impl)


def v2_model(family: str, seed: int, K: int = K_SMALL, widths=V2, impl: str = "unpadded"):
    """The V2 module, raw (un-recentered) init, on CPU."""
    return construct_module(family, widths, seed, vocab_sizes=vocab(K), poc=poc(K), impl=impl)


def v2_adapter(family: str, seed: int = 2026, K: int = K_SMALL, widths=V2, impl: str = "unpadded"):
    return make_adapter(v2_model(family, seed, K, widths, impl), widths)


def kwargs_for(module, b: dict) -> dict:
    from v4models.build import extension_tokens_of
    ext = {c: b[c] for c in extension_tokens_of(module)}          # SCHEMA-01R1 extension tokens (L0 / L3: none)
    if isinstance(module, (gru_v2.ContextGRU, snap_gru_mod.ContextGRU)):
        return dict(reuse_train.build_named_kwargs(b, module), **ext)
    return dict({k: b[k] for k in SASREC_KEYS if k in b}, **ext)


def forward(module, b: dict) -> torch.Tensor:
    return module(**kwargs_for(module, b))


def query(module, b: dict) -> torch.Tensor:
    return module.query(**kwargs_for(module, b))


def is_gru(module) -> bool:
    return isinstance(module, (gru_v2.ContextGRU, snap_gru_mod.ContextGRU))


# ------------------------------------------------------------------------------------------------ [B, L, K] guard
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
        allowed = set(int(r) for r in allowed_rows)
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


__all__ = ["SCHEMA", "V1", "V2", "FAMS", "K_SMALL", "nc", "assert_raises", "vocab", "poc", "batch", "snap_model",
           "v2_model", "v2_adapter", "kwargs_for", "forward", "query", "is_gru", "KShapeRecorder",
           "train_step_under", "ce_mean", "seeded", "maxdiff"]
