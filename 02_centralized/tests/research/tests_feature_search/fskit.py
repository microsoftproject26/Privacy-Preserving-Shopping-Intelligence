"""Shared helpers for the SCHEMA-01 tests_fs suite (synthetic in-memory providers only; no dataset rows)."""
from __future__ import annotations

import collections
import subprocess
from typing import Iterator, Optional

import pytest
import torch

import v4train  # noqa: F401  (ppsi_env first)
from v4fs import schemas as S
from v4models.build import build
from v4models.schema import load_schema
from v4train.provider import SyntheticEvalProvider

BASE = load_schema()
K = 64
L3 = S.load_registered("L3_NEAR_DUPLICATE")
L0 = S.load_registered("L0_CLEAN_V2")
L3S = S.load_registered("L3S_NEAR_DUPLICATE_SESSION")          # SCHEMA-01R1: the FS candidate base
L0S = S.load_registered("L0S_CLEAN_V2_SESSION")                # SCHEMA-01R1: the FS reference
_Usage = collections.namedtuple("_Usage", "total used free")
GIB = 2 ** 30


def nc(reason: str):
    return pytest.mark.xfail(strict=True, raises=AssertionError, reason="NEGATIVE CONTROL: " + reason)


def assert_raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc:
        return
    raise AssertionError(f"{getattr(fn, '__name__', fn)} did not raise {exc}")


def disk(free_gib: float):
    return lambda _path: _Usage(int(2000 * GIB), int(2000 * GIB - free_gib * GIB), int(free_gib * GIB))


BIG_DISK = disk(1000.0)


def eval_provider(n: int = 150, seed: int = 4, max_len: int = 8, session: bool = False):
    """A synthetic EvalProvider; session=True adds the synthetic DATA-05 keys (session offsets + G13) of run_fs."""
    ep = SyntheticEvalProvider(n, K, BASE.widths, seed=seed, max_len=max_len, manifest_id="FS_SYN_VALIDATION")
    if session:
        from v4fs.run_fs import SessionOffsetEval
        ep = SessionOffsetEval(ep)
    return ep


def model(family: str, schema=L3, seed: int = 2026):
    bm = build(family, schema, seed, "cpu", K=K)
    bm.module.eval()
    return bm


class ConstantChannelProvider:
    """Wraps an EvalProvider; sets one channel to a constant: per-event at every real position (0 on PAD, the Clean V2
    PAD rule) or, with pad_too, at every position; decision-level for every row."""

    def __init__(self, inner, key: str, col: Optional[int], value: float, pad_too: bool = False):
        self.inner, self.key, self.col, self.value, self.pad_too = inner, key, col, value, pad_too
        self.manifest = inner.manifest
        self.fingerprint = inner.fingerprint + "|const"

    def batches(self, bs: int) -> Iterator[dict]:
        for b in self.inner.batches(bs):
            b = dict(b)
            t = b[self.key].clone()
            if t.ndim >= 2 and self.key not in ("user_context", "user_context_masks"):
                am = b["attention_mask"].bool() | bool(self.pad_too)
                v = torch.where(am, torch.as_tensor(self.value, dtype=t.dtype), torch.zeros((), dtype=t.dtype))
                if self.col is None:
                    t = v.to(t.dtype)
                else:
                    t[..., self.col] = v.to(t.dtype)
            else:
                t[:, self.col] = self.value
            b[self.key] = t
            yield b


def smi_runner(apps: str = "", mem_used: float = 10.0, fail: bool = False):
    """A fake nvidia-smi: two GPUs (0 and 1)."""
    def run(argv, **kw):
        if fail:
            return subprocess.CompletedProcess(argv, 9, "", "boom")
        q = " ".join(argv)
        if "--query-gpu=index,pci.bus_id" in q:
            out = "0, 00000000:01:00.0\n1, 00000000:02:00.0\n"
        elif "--query-compute-apps" in q:
            out = apps
        else:
            out = f"0, GPU-a, NVIDIA A16, 5.0, 15000, 0\n1, GPU-b, NVIDIA A16, {mem_used}, 15000, 0\n"
        return subprocess.CompletedProcess(argv, 0, out, "")
    return run


__all__ = ["BASE", "K", "L3", "L0", "L3S", "L0S", "nc", "assert_raises", "disk", "BIG_DISK", "eval_provider", "model",
           "ConstantChannelProvider", "smi_runner"]
