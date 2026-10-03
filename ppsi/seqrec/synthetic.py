"""Synthetic decisions in the model input layout (tests and benchmarks only; no data rows).

Right-padded windows (real events at columns 0..len-1, PAD after), position_ids 1..len on real events and 0 on PAD,
every [B, L, *] event tensor exactly 0 on PAD positions, event flags int8, and optionally some flag columns forced to
0 (flags that never fire in the training data).
"""
from __future__ import annotations

from collections.abc import Sequence

import torch

from .features import InputWidths, names_sequence

L_MAX = 50


def synthetic_batch(n: int, K: int, widths: InputWidths, gen: torch.Generator, *, vocab: dict | None = None,
                    max_len: int = L_MAX, min_len: int = 1, window: int = L_MAX,
                    zero_flag_columns: Sequence[int] = (), with_names: bool = False) -> dict:
    """n synthetic decisions; `window` is the stored width L (>= max_len), `lengths` in [min_len, max_len]."""
    if not (1 <= min_len <= max_len <= window):
        raise ValueError("need 1 <= min_len <= max_len <= window")
    V = dict(vocab) if vocab is not None else {"item_tokens": K + 3, "category_tokens": 663, "brand_tokens": 3956,
                                                 "main_category_tokens": 16, "daypart_tokens": 8,
                                                 "weekday_tokens": 10}
    lengths = torch.randint(min_len, max_len + 1, (n,), generator=gen)
    am = torch.arange(window)[None, :] < lengths[:, None]
    amf = am[..., None].to(torch.float32)

    def tok(v: int) -> torch.Tensor:
        return torch.randint(3, v, (n, window), generator=gen) * am

    flags = ((torch.rand(n, window, widths.flags_dim, generator=gen) < 0.05) & am[..., None]).to(torch.int8)
    for c in zero_flag_columns:
        flags[..., int(c)] = 0
    out = {"item_tokens": tok(V["item_tokens"]), "category_tokens": tok(V["category_tokens"]),
           "brand_tokens": tok(V["brand_tokens"]), "main_category_tokens": tok(V["main_category_tokens"]),
           "daypart_tokens": tok(V["daypart_tokens"]), "weekday_tokens": tok(V["weekday_tokens"]),
           "lengths": lengths, "attention_mask": am,
           "event_numeric_features": torch.randn(n, window, widths.numeric_dim, generator=gen) * amf,
           "event_quality_flags": flags,
           "user_context": torch.randn(n, widths.user_context_dim, generator=gen),
           "user_context_masks": (torch.rand(n, widths.user_mask_dim, generator=gen) < 0.5).to(torch.float32),
           "position_ids": torch.arange(1, window + 1)[None, :] * am,
           "target_class": torch.randint(0, K, (n,), generator=gen),
           "loss_mask": torch.ones(n, dtype=torch.bool)}
    if with_names:
        out.update(names_sequence(widths))
        out["feature_view"] = widths.view_id
    return out


def row_keys(batch: dict) -> list:
    """The per-row tensor keys of a batch (excludes name arrays / view metadata)."""
    return [k for k, v in batch.items() if isinstance(v, torch.Tensor)]


def take(batch: dict, idx) -> dict:
    """Row subset of the per-row tensors; metadata keys are carried through unchanged."""
    idx = torch.as_tensor(idx, dtype=torch.long)
    return {k: (v.index_select(0, idx) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def crop(batch: dict, width: int) -> dict:
    """The same decisions stored with a narrower window (width >= max length): pure right-padding change."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor) and v.ndim >= 2 and k not in ("user_context", "user_context_masks"):
            out[k] = v[:, :width].contiguous()
        else:
            out[k] = v
    return out


def pad_to(batch: dict, width: int) -> dict:
    """The same decisions stored with a wider window, zero right padding (PAD token 0, features 0, position 0)."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor) and v.ndim >= 2 and k not in ("user_context", "user_context_masks"):
            extra = width - v.shape[1]
            if extra < 0:
                raise ValueError("pad_to: width smaller than the stored window")
            shape = (v.shape[0], extra, *v.shape[2:])
            out[k] = torch.cat([v, torch.zeros(shape, dtype=v.dtype)], dim=1)
        else:
            out[k] = v
    return out


__all__ = ["L_MAX", "crop", "pad_to", "row_keys", "synthetic_batch", "take"]
