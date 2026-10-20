"""Shared helpers for the DEV-01 tests (synthetic tiny models only; no dataset rows, no onnxruntime/onnx import
here — individual tests importorskip those). Pattern of code/tests_models/models_testkit.py."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from v4dev.checkpointio import build_scoring_module
from v4models.schema import load_schema

SCHEMA = load_schema()
V2 = SCHEMA.widths
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
    """A fresh-init (no checkpoint) tiny scoring module of `family`'s current build, the real schema widths, a
    synthetic K-class catalogue."""
    return build_scoring_module(family, ckpt_path=None, seed=seed, k=k)


def save_state_dict(module: torch.nn.Module, path) -> None:
    """A minimal checkpoint file `v4dev.checkpointio.load_state_dict` reads back (a bare state dict)."""
    torch.save(dict(module.state_dict()), str(path))


__all__ = ["SCHEMA", "V2", "FAMS", "K_SMALL", "nc", "assert_raises", "tiny_scoring", "save_state_dict"]
