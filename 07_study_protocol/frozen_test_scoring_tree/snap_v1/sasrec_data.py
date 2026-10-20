"""sasrec_data.py (owner: Sonnet-data). SASREC_INTERFACES.md section 6.

Predictor-kwarg allowlisting, right-padded layout checks/repack, and the two
store factories (training: TRAIN+VALIDATION_SELECT; retrospective eval: VC/TEST
behind an existing lock). No model code, no training loop.

Import order matters: `import paths` first (puts `code/reuse` + its vendor on
sys.path), then the reused modules.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

import paths  # noqa: F401  -- sys.path side effect (code/reuse + vendor)
from full_store import CompactStore  # noqa: E402

__all__ = [
    "PREDICTOR_ALLOWLIST_ID", "PREDICTOR_ALLOWLIST_RICH", "FORBIDDEN_PREDICTOR_KEYS",
    "SIDE_CHANNELS", "predictor_kwargs", "check_right_padded", "repack_right",
    "build_training_store", "build_eval_store",
]

# ---------------------------------------------------------------------------
# Allowlists (SASREC_INTERFACES.md sec 6, Fable must-fail #28 / 05.A.28)
# ---------------------------------------------------------------------------

SIDE_CHANNELS = ("category_tokens", "brand_tokens", "main_category_tokens",
                 "daypart_tokens", "weekday_tokens")

# The forward kwargs `predictor_kwargs` may ever emit for an item-only model.
PREDICTOR_ALLOWLIST_ID = ("item_tokens", "lengths", "attention_mask")

# Item-only kwargs plus every rich side/numeric/user-context channel. Note
# `position_ids` is deliberately NOT here: the model derives its own positions
# and the trainer passes `position_ids` itself, outside `predictor_kwargs`
# (Gate P sec 9: "the reused evaluator does not [pass position_ids]").
PREDICTOR_ALLOWLIST_RICH = PREDICTOR_ALLOWLIST_ID + SIDE_CHANNELS + (
    "event_numeric_features", "event_quality_flags", "user_context", "user_context_masks",
)

# Keys that must NEVER appear in a forward kwarg dict: labels, identities, split
# names, retrospective-only bookkeeping, and positions (the model derives its
# own; `position_ids` travels outside `predictor_kwargs`, see Gate P sec 9).
FORBIDDEN_PREDICTOR_KEYS = frozenset({
    "target_class", "loss_mask", "eval_mask", "decision_id", "client_id",
    "eval_user_key", "user_bucket", "split", "prefix_position", "prior_view_count",
    "model_segment_id", "last_valid_index", "position_ids",
})


def _validate_allowlist(allowlist) -> None:
    bad = set(allowlist) & FORBIDDEN_PREDICTOR_KEYS
    if bad:
        raise ValueError(
            f"predictor allowlist is corrupted: it intersects FORBIDDEN_PREDICTOR_KEYS "
            f"on {sorted(bad)} -- refusing to build any forward kwargs from it"
        )


def predictor_kwargs(batch: dict, model) -> dict:
    """Exactly the allowlisted keys `model` declares (via `_categorical_channels`,
    `numeric_dim`, `flags_dim`, `use_user_context`), gathered from `batch`.

    Mirrors the reused `reuse/train.build_named_kwargs` selection logic exactly
    (same four model attributes decide inclusion), so the audited evaluator and
    this adapter always agree on what a given model receives. Raises if the
    allowlist it selects from intersects `FORBIDDEN_PREDICTOR_KEYS` (defensive:
    re-checked on every call, not just at import time, so a monkeypatched/
    corrupted module-level allowlist is caught here too).
    """
    categorical = tuple(getattr(model, "_categorical_channels"))
    numeric_dim = int(getattr(model, "numeric_dim"))
    flags_dim = int(getattr(model, "flags_dim"))
    use_user_context = bool(getattr(model, "use_user_context"))
    rich = numeric_dim > 0 or flags_dim > 0 or use_user_context or any(
        ch in categorical for ch in SIDE_CHANNELS)

    allowlist = PREDICTOR_ALLOWLIST_RICH if rich else PREDICTOR_ALLOWLIST_ID
    _validate_allowlist(allowlist)

    missing = [k for k in ("item_tokens", "lengths", "attention_mask") if k not in batch]
    if missing:
        raise KeyError(f"batch is missing required keys {missing}")

    kw: dict = {
        "item_tokens": batch["item_tokens"],
        "lengths": batch["lengths"],
        "attention_mask": batch["attention_mask"],
    }
    for ch in SIDE_CHANNELS:
        if ch in categorical:
            if ch not in allowlist:
                raise ValueError(f"model declares categorical channel {ch!r} not in the allowlist")
            if ch not in batch:
                raise KeyError(f"model requires categorical channel {ch!r} missing from batch")
            kw[ch] = batch[ch]
    if numeric_dim > 0:
        if "event_numeric_features" not in batch:
            raise KeyError("model requires event_numeric_features missing from batch")
        kw["event_numeric_features"] = batch["event_numeric_features"]
    if flags_dim > 0:
        if "event_quality_flags" not in batch:
            raise KeyError("model requires event_quality_flags missing from batch")
        kw["event_quality_flags"] = batch["event_quality_flags"]
    if use_user_context:
        for ch in ("user_context", "user_context_masks"):
            if ch not in batch:
                raise KeyError(f"model requires {ch!r} missing from batch")
            kw[ch] = batch[ch]

    bad = set(kw) & FORBIDDEN_PREDICTOR_KEYS
    if bad:
        raise ValueError(f"predictor_kwargs about to emit forbidden keys {sorted(bad)}")
    extra = set(kw) - set(allowlist)
    if extra:
        raise ValueError(f"predictor_kwargs selected keys {sorted(extra)} outside its own allowlist")
    return kw


# ---------------------------------------------------------------------------
# Layout checks / generic repack
# ---------------------------------------------------------------------------

# Per-event tensors that must read zero on PAD columns when present in a batch.
_PER_EVENT_KEYS = ("item_tokens",) + SIDE_CHANNELS + (
    "event_numeric_features", "event_quality_flags",
)


def _as_tensor(x):
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(np.asarray(x))


def check_right_padded(batch: dict) -> None:
    """Raise unless `batch` is a valid right-padded window batch:

    - each row's `attention_mask` is a contiguous prefix of length `lengths[b]`
      (True for columns < lengths[b], False after);
    - `position_ids` == 1..len on valid cells and 0 on PAD (when present);
    - `last_valid_index` == lengths - 1 (when present);
    - every per-event tensor present in `batch` (`_PER_EVENT_KEYS`) is exactly
      zero on every PAD column of every row.
    """
    am = _as_tensor(batch["attention_mask"]).to(torch.bool)
    lengths = _as_tensor(batch["lengths"]).long()
    if am.ndim != 2:
        raise ValueError(f"attention_mask must be [B,T], got shape {tuple(am.shape)}")
    B, T = am.shape
    if lengths.shape != (B,):
        raise ValueError(f"lengths must be [B], got shape {tuple(lengths.shape)}")
    if bool((lengths < 1).any()) or bool((lengths > T).any()):
        raise ValueError("lengths must satisfy 1 <= length <= T")

    ar = torch.arange(T)
    expect_valid = ar.unsqueeze(0) < lengths.unsqueeze(1)
    if not torch.equal(am, expect_valid):
        raise ValueError("attention_mask is not a contiguous right-aligned prefix of `lengths`")

    if "position_ids" in batch:
        pos = _as_tensor(batch["position_ids"]).long()
        if pos.shape != (B, T):
            raise ValueError(f"position_ids must be [B,T], got shape {tuple(pos.shape)}")
        expect_pos = torch.where(expect_valid, ar.unsqueeze(0) + 1, torch.zeros_like(ar).unsqueeze(0))
        if not torch.equal(pos, expect_pos):
            raise ValueError("position_ids must equal 1..len on valid cells and 0 on PAD")

    if "last_valid_index" in batch:
        lvi = _as_tensor(batch["last_valid_index"]).long()
        if not torch.equal(lvi, lengths - 1):
            raise ValueError("last_valid_index must equal lengths - 1")

    pad_mask = ~expect_valid  # [B,T]
    for key in _PER_EVENT_KEYS:
        if key not in batch:
            continue
        t = _as_tensor(batch[key])
        if t.shape[:2] != (B, T):
            raise ValueError(f"{key} must have shape [B,T,...], got {tuple(t.shape)}")
        if t.ndim > 2:
            bad = pad_mask.view(B, T, *([1] * (t.ndim - 2))).expand_as(t) & (t != 0)
        else:
            bad = pad_mask & (t != 0)
        if bool(bad.any()):
            raise ValueError(f"{key} has non-zero value(s) on a PAD column")


def repack_right(batch: dict) -> dict:
    """Generic left/scattered -> right repack.

    Moves each row's valid columns (per `attention_mask`, taken in ascending
    column order -- i.e. whatever order they already sit in, which is assumed
    chronological) into a contiguous 0..len-1 prefix, applies the SAME
    permutation to every per-event tensor present, and rebuilds
    `attention_mask`/`position_ids`/`lengths`/`last_valid_index`. Per-decision
    tensors (ndim==1, or ndim>=2 whose 2nd dim isn't T) pass through unchanged.
    """
    am = _as_tensor(batch["attention_mask"]).to(torch.bool)
    B, T = am.shape
    lengths = am.sum(dim=1).long()
    max_len = int(lengths.max().item()) if B else 0

    gather_idx = torch.zeros(B, max(max_len, 1), dtype=torch.long)
    valid_out = torch.zeros(B, max(max_len, 1), dtype=torch.bool)
    for b in range(B):
        cols = torch.nonzero(am[b], as_tuple=True)[0]
        L = int(cols.numel())
        if L:
            gather_idx[b, :L] = cols
            valid_out[b, :L] = True
    if max_len == 0:
        gather_idx = gather_idx[:, :0]
        valid_out = valid_out[:, :0]

    out: dict = {}
    for key, val in batch.items():
        if key in ("attention_mask", "position_ids", "lengths", "last_valid_index"):
            continue
        t = _as_tensor(val) if isinstance(val, (torch.Tensor, np.ndarray)) else val
        is_per_event = (isinstance(t, torch.Tensor) and t.ndim >= 2 and t.shape[0] == B and t.shape[1] == T)
        if not is_per_event:
            out[key] = val
            continue
        extra = t.shape[2:]
        gi = gather_idx
        if extra:
            gi = gi.view(B, max_len, *([1] * len(extra))).expand(B, max_len, *extra)
        gathered = torch.gather(t, 1, gi)
        zero_mask = ~valid_out
        if extra:
            zero_mask = zero_mask.view(B, max_len, *([1] * len(extra))).expand_as(gathered)
        gathered = torch.where(zero_mask, torch.zeros_like(gathered), gathered)
        out[key] = gathered

    ar = torch.arange(max_len)
    out["lengths"] = lengths
    out["attention_mask"] = valid_out
    out["position_ids"] = torch.where(valid_out, ar.unsqueeze(0) + 1, torch.zeros_like(ar).unsqueeze(0))
    out["last_valid_index"] = lengths - 1
    return out


# ---------------------------------------------------------------------------
# Store factories
# ---------------------------------------------------------------------------

def build_training_store(channels: str) -> CompactStore:
    """TRAIN + VALIDATION_SELECT only. Refuses to build anything else -- there
    is no `splits`/`split` parameter here on purpose."""
    if channels not in ("F0", "F4"):
        raise ValueError(f"channels must be 'F0' or 'F4', got {channels!r}")
    return CompactStore.build(paths.DATASET_ROOT, splits=("TRAIN", "VALIDATION_SELECT"),
                              channels=channels)


def build_eval_store(channels: str, split: str, *, lock_path) -> CompactStore:
    """Retrospective-only: VALIDATION_CONFIRM or TEST, and only if `lock_path`
    already exists on disk (the new SASRec lock -- never delete/replace an old
    GRU lock to satisfy this). Never call this from a test with a real lock."""
    if split not in ("VALIDATION_CONFIRM", "TEST"):
        raise ValueError(f"build_eval_store only serves VALIDATION_CONFIRM/TEST, got {split!r}")
    if channels not in ("F0", "F4"):
        raise ValueError(f"channels must be 'F0' or 'F4', got {channels!r}")
    lock_path = Path(lock_path)
    if not lock_path.exists():
        raise FileNotFoundError(
            f"refusing to build a {split} eval store: lock file {lock_path} does not exist "
            "(retrospective evaluation requires a NEW SASRec lock, see SASREC_INTERFACES.md sec 8)"
        )
    return CompactStore.build(paths.DATASET_ROOT, splits=(split,), channels=channels)
