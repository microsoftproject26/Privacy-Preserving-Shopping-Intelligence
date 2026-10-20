"""Shared fixtures/builders for the SASRec Gate P adversarial test suite.

Owner: Sonnet-tests (independent of Opus). No real data, no network. Adds the
`code` directory (this file's grandparent) to `sys.path` so `import paths`
resolves exactly the way production code expects; `paths` then puts
`code/reuse` (+ vendor) on `sys.path` for the audited reused modules.

All builders here make TINY synthetic tensors (small K, small d) -- never the
real catalogue, never real decisions. `synthetic_store.SyntheticStore` (05
contract sec.10, Sonnet-data) is not required: these are our own minimal
fixtures, built to the same right-padded layout contract documented in
SASREC_INTERFACES.md sec.0 and 03_ARCHITECTURE_CONTRACT.md sec.2.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import paths  # noqa: E402  (adds reuse + vendor to sys.path; never edited)
from sasrec_model import SASRecCE, SASRecConfig, SIDE_CHANNELS, SIDE_DIMS  # noqa: E402

# ---------------------------------------------------------------------------
# Small, fixed catalogue sizes for tests (never the real K=158486 catalogue).
# ---------------------------------------------------------------------------
TINY_K = 37                 # default full-catalogue size for exact/hand tests
SIDE_VOCAB = {"category_tokens": 9, "brand_tokens": 8, "main_category_tokens": 6,
              "daypart_tokens": 5, "weekday_tokens": 7}   # each >= 5 rows (contract sec.10)

HAS_CUDA = torch.cuda.is_available()
cuda_only = pytest.mark.skipif(not HAS_CUDA, reason="requires a visible CUDA device (GPU 0)")


# ---------------------------------------------------------------------------
# Config / vocab / model builders
# ---------------------------------------------------------------------------

def make_vocab_sizes(K: int, rich: bool) -> dict:
    sizes = {"item_tokens": K + 3}
    if rich:
        sizes.update(SIDE_VOCAB)
    return sizes


def make_poc(K: int) -> np.ndarray:
    """product_idx_of_class == arange(K)+3 -- the only value the known-class gate accepts."""
    return np.arange(K, dtype=np.int64) + 3


def make_config(*, group: str = "item_only", d_model: int = 16, blocks: int = 2,
                 heads: int = 2, ff_multiplier: int = 4, dropout: float = 0.0,
                 max_len: int = 50, ln_eps: float = 1e-5, run_id: str = "test") -> SASRecConfig:
    return SASRecConfig(run_id=run_id, group=group, d_model=d_model, blocks=blocks,
                         heads=heads, ff_multiplier=ff_multiplier, dropout=dropout,
                         max_len=max_len, ln_eps=ln_eps)


def make_model(cfg: SASRecConfig, K: int = TINY_K, *, seed: int = 0,
                impl: str = "unpadded", vocab_sizes: dict | None = None,
                poc: np.ndarray | None = None) -> SASRecCE:
    vocab_sizes = make_vocab_sizes(K, cfg.rich) if vocab_sizes is None else vocab_sizes
    poc = make_poc(K) if poc is None else poc
    return SASRecCE(cfg, vocab_sizes=vocab_sizes, product_idx_of_class=poc, seed=seed, impl=impl)


# ---------------------------------------------------------------------------
# Batch builders -- right-padded layout: real events at columns 0..len-1,
# PAD after; position_ids = 1..len on real events, 0 on PAD (store convention,
# SASREC_INTERFACES.md sec.0 / Gate P fix 1).
# ---------------------------------------------------------------------------

def make_batch(lengths, *, T: int | None = None, K: int = TINY_K, rich: bool = False,
               seed: int = 0, target_mode: str = "valid", pad_item_value: int | None = 0,
               pad_side_value: int = 0) -> dict:
    """Builds a synthetic right-padded batch dict, CPU tensors, matching the
    CompactStore.batch() key/dtype/layout contract (sec.0).

    `pad_item_value`: token id written into PAD columns of item_tokens (and,
    if rich, into PAD columns of every side-event tensor); `None` leaves PAD
    columns at their natural embedding-lookup-safe value 0 (PAD). Tests that
    check PAD invariance pass a nonzero (even out-of-range-looking-but-valid)
    value here to prove the model never reads it.
    """
    rng = np.random.default_rng(seed)
    lengths = np.asarray(lengths, dtype=np.int64)
    B = lengths.shape[0]
    T = int(lengths.max()) if T is None else T
    assert T >= int(lengths.max())

    item_tokens = np.zeros((B, T), dtype=np.int64)
    attention_mask = np.zeros((B, T), dtype=bool)
    position_ids = np.zeros((B, T), dtype=np.int64)
    for b in range(B):
        L = int(lengths[b])
        item_tokens[b, :L] = rng.integers(3, 3 + K, size=L)
        attention_mask[b, :L] = True
        position_ids[b, :L] = np.arange(1, L + 1)
        if pad_item_value is not None and L < T:
            item_tokens[b, L:] = pad_item_value

    batch = dict(
        item_tokens=torch.from_numpy(item_tokens),
        attention_mask=torch.from_numpy(attention_mask),
        position_ids=torch.from_numpy(position_ids),
        lengths=torch.from_numpy(lengths),
        last_valid_index=torch.from_numpy(lengths - 1),
    )

    if target_mode == "valid":
        target_class = rng.integers(0, K, size=B).astype(np.int64)
        loss_mask = np.ones(B, dtype=bool)
        eval_mask = np.ones(B, dtype=bool)
    elif target_mode == "none":
        target_class = np.full(B, -1, dtype=np.int64)
        loss_mask = np.zeros(B, dtype=bool)
        eval_mask = np.zeros(B, dtype=bool)
    else:
        raise ValueError(target_mode)
    batch["target_class"] = torch.from_numpy(target_class)
    batch["loss_mask"] = torch.from_numpy(loss_mask)
    batch["eval_mask"] = torch.from_numpy(eval_mask)
    batch["decision_id"] = torch.arange(B, dtype=torch.int64)
    batch["client_id"] = torch.arange(B, dtype=torch.int64) + 3

    if rich:
        for ch in SIDE_CHANNELS:
            vocab = SIDE_VOCAB[ch]
            tok = np.zeros((B, T), dtype=np.int64)
            for b in range(B):
                L = int(lengths[b])
                tok[b, :L] = rng.integers(1, vocab, size=L)  # avoid 0 = PAD-only value
                if L < T:
                    tok[b, L:] = pad_side_value
            batch[ch] = torch.from_numpy(tok)
        numf = np.zeros((B, T, 19), dtype=np.float32)
        flags = np.zeros((B, T, 15), dtype=np.int8)
        for b in range(B):
            L = int(lengths[b])
            numf[b, :L] = rng.normal(size=(L, 19)).astype(np.float32)
            flags[b, :L] = rng.integers(0, 2, size=(L, 15)).astype(np.int8)
        batch["event_numeric_features"] = torch.from_numpy(numf)
        batch["event_quality_flags"] = torch.from_numpy(flags)
        batch["user_context"] = torch.from_numpy(rng.normal(size=(B, 12)).astype(np.float32))
        batch["user_context_masks"] = torch.from_numpy(rng.integers(0, 2, size=(B, 6)).astype(np.float32))
    return batch


def forward_kwargs(batch: dict, *, with_position_ids: bool = True) -> dict:
    """The model.forward()-compatible subset of a batch dict (mirrors the
    reused train.build_named_kwargs selection, written independently here so
    tests do not depend on reuse/train.py's generic dispatch for every case)."""
    kw = dict(item_tokens=batch["item_tokens"], lengths=batch["lengths"],
              attention_mask=batch["attention_mask"])
    if with_position_ids:
        kw["position_ids"] = batch["position_ids"]
    for ch in SIDE_CHANNELS:
        if ch in batch:
            kw[ch] = batch[ch]
    for k in ("event_numeric_features", "event_quality_flags", "user_context", "user_context_masks"):
        if k in batch:
            kw[k] = batch[k]
    return kw


@pytest.fixture
def rng():
    return np.random.default_rng(12345)
