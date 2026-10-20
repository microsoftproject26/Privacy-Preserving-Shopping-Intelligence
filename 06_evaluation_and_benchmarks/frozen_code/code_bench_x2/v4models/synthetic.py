"""v4models.synthetic — synthetic decisions in the Clean V2 tensor layout (tests / benchmarks only; no data rows).

Right-padded windows (real events at columns 0..len-1, PAD after), store position_ids 1..len on real events and 0 on
PAD, every [B, L, *] event tensor exactly 0 on PAD positions (DATA_CONTRACT_CLEAN_V2 rezero map, PAD row), event
flags int8, the six DEAD_IN_TRAIN flag columns optionally forced to 0 (as on TRAIN/VS/VC).
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch

from .schema import InputWidths, names_sequence

L_MAX = 50


def synthetic_examples_v2(n: int, K: int, widths: InputWidths, gen: torch.Generator, *, vocab: Optional[dict] = None,
                          max_len: int = L_MAX, min_len: int = 1, window: int = L_MAX,
                          zero_flag_columns: Sequence[int] = (), with_names: bool = False,
                          with_session_offset: bool = False, with_session_intent: bool = False) -> dict:
    """n synthetic decisions; `window` is the stored width L (>= max_len), `lengths` in [min_len, max_len].
    with_session_offset (SCHEMA-01R1): also a `session_offset_tokens` [n, L] key (0 on PAD; 1..8 on valid positions,
    non-increasing towards the current event, which is 1), drawn AFTER every other tensor so the rest is unchanged.
    with_session_intent (SCHEMA-01R1, DL 48): also the G13 keys (session_mode_main_category_tokens [n, L]; the [n]
    floats current_matches_session_mode, its mask and session_mode_share), a deterministic function of the offsets."""
    if not (1 <= min_len <= max_len <= window):
        raise ValueError("need 1 <= min_len <= max_len <= window")
    V = dict(vocab) if vocab is not None else {"item_tokens": K + 3, "category_tokens": 663, "brand_tokens": 3956,
                                                 "main_category_tokens": 16, "daypart_tokens": 8, "weekday_tokens": 10}
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
    if with_session_offset or with_session_intent:
        out["session_offset_tokens"] = synthetic_session_offsets(lengths, window, gen)
    if with_session_intent:
        out.update(synthetic_session_intent(out["session_offset_tokens"], lengths))
    if with_names:
        out.update(names_sequence(widths))
        out["feature_view"] = widths.view_id
    return out


def synthetic_session_offsets(lengths: torch.Tensor, window: int, gen: Optional[torch.Generator] = None,
                              seg: Optional[torch.Tensor] = None) -> torch.Tensor:
    """[n, window] int64 session-offset tokens: position j of a window of length len is (len - 1 - j) events before
    the current one; with per-row session length s (drawn from `gen`, or `seg`), token = min((len-1-j) // s, 7) + 1;
    PAD = 0."""
    n = int(lengths.shape[0])
    if seg is None:
        seg = torch.randint(1, 9, (n,), generator=gen)
    j = torch.arange(window)[None, :]
    back = (lengths[:, None] - 1 - j).clamp(min=0)
    tok = torch.clamp(back // seg[:, None], max=7) + 1
    return torch.where(j < lengths[:, None], tok, torch.zeros_like(tok)).to(torch.int64)


def synthetic_session_intent(offsets: torch.Tensor, lengths: torch.Tensor, vocab: int = 17) -> dict:
    """Synthetic G13 keys from session-offset tokens: the per-event session mode token is constant per session
    (1 + (7 * session + row) % (vocab - 1), 0 on PAD); the current event matches the mode on even rows; the mask is 1
    unless the window holds one event; the share is 1 / (1 + session length % 4)."""
    n, L = offsets.shape
    row = torch.arange(n)[:, None]
    mode = 1 + (7 * offsets + row) % (vocab - 1)
    mode = torch.where(offsets > 0, mode, torch.zeros_like(mode)).to(torch.int64)
    avail = (lengths >= 2).to(torch.float32)
    match = ((torch.arange(n) % 2) == 0).to(torch.float32) * avail
    cur = (offsets == 1).sum(1)
    share = (1.0 / (1.0 + (cur % 4).to(torch.float32))) * avail
    return {"session_mode_main_category_tokens": mode, "current_matches_session_mode": match,
            "current_matches_session_mode_available": avail, "session_mode_share": share}


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
            shape = (v.shape[0], extra) + tuple(v.shape[2:])
            out[k] = torch.cat([v, torch.zeros(shape, dtype=v.dtype)], dim=1)
        else:
            out[k] = v
    return out


__all__ = ["L_MAX", "synthetic_examples_v2", "synthetic_session_offsets", "synthetic_session_intent", "row_keys", "take", "crop", "pad_to"]
