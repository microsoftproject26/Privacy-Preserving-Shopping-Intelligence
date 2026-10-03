"""Shared helpers for the on-device tests (synthetic tiny models only; no data rows)."""
from __future__ import annotations

import pytest
import torch

from ppsi.ondevice.io import build_scoring_module

FAMS = ["GRU", "SASREC"]
K_SMALL = 64


def nc(reason: str):
    """Negative control: the wrapped key-property check must raise AssertionError (strict xfail)."""
    return pytest.mark.xfail(strict=True, raises=AssertionError, reason="NEGATIVE CONTROL: " + reason)


def assert_raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc:
        return
    raise AssertionError(f"{getattr(fn, '__name__', fn)} did not raise {exc}")


def tiny_scoring(family: str, *, seed: int = 2026, k: int = K_SMALL) -> dict:
    """A fresh-init (no checkpoint) tiny scoring module of `family`'s default size, the default rich widths and a
    synthetic K-class catalogue."""
    return build_scoring_module(family, ckpt_path=None, seed=seed, k=k)


def save_state_dict(module: torch.nn.Module, path) -> None:
    """A minimal checkpoint file `ppsi.ondevice.io.load_state_dict` reads back (a bare state dict)."""
    torch.save(dict(module.state_dict()), str(path))
