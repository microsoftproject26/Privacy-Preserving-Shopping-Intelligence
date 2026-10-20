"""v4base.knn_emb — BASELINE_KNN_EMB (contracts/PREG1_AMENDMENT_04.md §2 arms table; DOWNSTREAM_FINAL_PLAN_FABLE.md
(2)): score(i) = cosine(mean embedding of the permitted history's last 10 items, item i's embedding), using a
CACHED item-embedding table read straight out of a trained checkpoint's own state dict — no model is built, no
training code beyond the two documented parameter names is imported (task packet's own rule). Primary = the SMALL
checkpoint (device-realistic); BIG is a reference point: both are just different `--item-table` paths through this
SAME loader/scorer.

Parameter names (read from `code/v4models/gru_v2.py` / `code/v4models/sasrec_v2.py`, never imported):
  GRU (ContextGRU): the tied item table is `item_embed.weight` [item_vocab_size, d] (PAD/MISSING/OOV rows included,
    ITEM_TOKEN_OFFSET=3 ahead of class 0); `product_idx_of_class` [K] int64 is a REGISTERED BUFFER (persistent,
    saved in the checkpoint) mapping class j -> its row of `item_embed.weight`. `index_select(0, product_idx_of_class)`
    is used unconditionally (not just in the non-contiguous case) -- it is exactly correct for the common contiguous
    case too (poc == arange(3, 3+K)) and needs no separate contiguity branch, unlike `gru_v2.ContextGRU.forward`'s
    own micro-optimisation (a slice) for that case.
  SASRec (SASRecCE): the untied output table is `output_embed`, an `nn.Parameter` of shape [K, d] -- ALREADY
    class-ordered (no offset, no PAD/MISSING/OOV rows: row j is class j). Its own `product_idx_of_class` buffer is
    registered non-persistent (`persistent=False`), so it is never in a SASRec checkpoint and is not needed here.

Checkpoint format: `code/v4train/checkpoint.py` saves a wrapping dict with the model's `state_dict()` under key
"model" (`torch.load(..., weights_only=True)`); a bare state dict (no wrapper) is also accepted, for a test fixture
or a hand-exported table, without assuming which. FL checkpoints ("theta" key, format PPSI_FL_CKPT_V1) and the
legacy snap_v1 "model_state_dict" key are read defensively too, in that documented order, so the ONE loader here
never needs to know which pipeline produced a given `--item-table` path.

Empty-history fallback (task packet: "Empty history -> all -inf except a registered fallback (popularity) — state
the choice"): the same sentinel correction as BASE-01/scoring.py applies -- a real -inf FLOATING row is refused by
v4eval's non-finite check (contract G1), and there is no mean embedding to take a cosine of in the first place when
the permitted history's last-10 window has zero valid items. The registered fallback IS the whole row, not a
sentinel: it is the SAME MAIN-TRAIN popularity counts table `v4base.pop.fit` already builds (cast to float32),
reused verbatim, never rebuilt here. A catalogue item whose OWN embedding row has exactly zero norm (a degenerate,
untouched row) scores 0.0 against every query, for the same reason SKNN's "no support" floor is 0.0: a defined,
finite value, not a sentinel.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from .sessions import history_valid_mask, item_classes

HISTORY_WINDOW = 10
FAMILIES = ("GRU", "SASREC")
GRU_TABLE_KEY = "item_embed.weight"
GRU_CLASS_MAP_KEY = "product_idx_of_class"
SASREC_TABLE_KEY = "output_embed"


def _state_dict_from_checkpoint(obj):
    """Unwrap the documented checkpoint container formats (see module docstring); a bare state dict (a plain
    mapping of str -> Tensor with no wrapper) is returned unchanged."""
    if isinstance(obj, dict):
        for key in ("model", "theta", "state_dict", "model_state_dict"):
            v = obj.get(key)
            if isinstance(v, dict):
                return v
    return obj


def load_checkpoint_state_dict(path) -> dict:
    obj = torch.load(str(path), map_location="cpu", weights_only=True)
    sd = _state_dict_from_checkpoint(obj)
    if not isinstance(sd, dict):
        raise ValueError(f"{path}: could not find a state dict (checked the top level, then 'model'/'theta'/"
                         "'state_dict'/'model_state_dict')")
    return sd


def item_table_from_state_dict(sd: dict, family: str, K: int) -> torch.Tensor:
    """[K, d] float32, class-ordered 0..K-1. Raises KeyError/ValueError with the exact missing key or shape
    mismatch rather than silently returning a wrong-shaped table."""
    fam = family.upper()
    if fam == "GRU":
        if GRU_TABLE_KEY not in sd or GRU_CLASS_MAP_KEY not in sd:
            raise KeyError(f"GRU checkpoint missing {GRU_TABLE_KEY!r} / {GRU_CLASS_MAP_KEY!r} "
                           "(v4models.gru_v2.ContextGRU's tied item table + class map)")
        W = sd[GRU_TABLE_KEY]
        poc = sd[GRU_CLASS_MAP_KEY].to(torch.long)
        if poc.ndim != 1 or int(poc.shape[0]) != int(K):
            raise ValueError(f"{GRU_CLASS_MAP_KEY} has shape {tuple(poc.shape)}, expected ({K},)")
        if int(poc.min()) < 0 or int(poc.max()) >= int(W.shape[0]):
            raise IndexError(f"{GRU_CLASS_MAP_KEY} references rows outside {GRU_TABLE_KEY} "
                             f"(table has {W.shape[0]} rows)")
        table = W.index_select(0, poc)
    elif fam == "SASREC":
        if SASREC_TABLE_KEY not in sd:
            raise KeyError(f"SASRec checkpoint missing {SASREC_TABLE_KEY!r} (v4models.sasrec_v2.SASRecCE's "
                           "untied output table)")
        table = sd[SASREC_TABLE_KEY]
        if table.ndim != 2 or int(table.shape[0]) != int(K):
            raise ValueError(f"{SASREC_TABLE_KEY} has shape {tuple(table.shape)}, expected ({K}, d)")
    else:
        raise ValueError(f"unknown family {family!r} (expected {FAMILIES})")
    return table.to(torch.float32).contiguous()


def load_item_table(path, family: str, K: int) -> torch.Tensor:
    return item_table_from_state_dict(load_checkpoint_state_dict(path), family, K)


class KnnEmbScorer:
    """Stateless given the cached table and the fallback counts: no fit() beyond loading the checkpoint (task
    packet: "No training")."""

    def __init__(self, item_table: torch.Tensor, fallback_counts: torch.Tensor, *,
                history_window: int = HISTORY_WINDOW):
        self.table = item_table.to(torch.float32)
        self.K = int(self.table.shape[0])
        norm = self.table.norm(dim=1)
        self._zero_row = norm == 0
        self._safe_norm = torch.where(self._zero_row, torch.ones_like(norm), norm)
        fb = fallback_counts.to(torch.float32)
        if int(fb.shape[0]) != self.K:
            raise ValueError(f"fallback_counts has {fb.shape[0]} entries, expected K={self.K}")
        self.fallback = fb
        self.history_window = int(history_window)

    def score_batch(self, batch: dict) -> torch.Tensor:
        item_tokens = batch["item_tokens"]
        item_tokens = item_tokens if torch.is_tensor(item_tokens) else torch.as_tensor(item_tokens)
        B, L = int(item_tokens.shape[0]), int(item_tokens.shape[1])
        cls, valid_tok = item_classes(item_tokens, self.K)
        valid = valid_tok & history_valid_mask(batch["lengths"], L)
        # the permitted history's last `history_window` VALID items (review fix, LOW: NOT the last `history_window`
        # COLUMNS — a column can be invalid, e.g. an OOV event, token==2, which is real history but no catalogue
        # embedding; skipping it must not shrink the window below `history_window` real items when older valid
        # items are available). `valid_rank[l]` = how many valid columns lie in [l, L-1] (inclusive): 1 for the
        # most recent valid item, 2 for the second-most-recent, etc — a reverse-inclusive cumulative sum.
        valid_rank = torch.flip(torch.cumsum(torch.flip(valid.to(torch.int64), dims=[1]), dim=1), dims=[1])
        in_window = valid & (valid_rank <= self.history_window)
        out = torch.empty((B, self.K), dtype=torch.float32)
        for r in range(B):
            idx = cls[r][in_window[r]]
            if idx.numel() == 0:
                out[r] = self.fallback
                continue
            mean_emb = self.table.index_select(0, idx).mean(dim=0)
            mnorm = float(mean_emb.norm())
            if mnorm == 0.0:
                out[r] = self.fallback
                continue
            sims = (self.table @ mean_emb) / (self._safe_norm * mnorm)
            out[r] = torch.where(self._zero_row, torch.zeros_like(sims), sims)
        return out


__all__ = ["HISTORY_WINDOW", "FAMILIES", "GRU_TABLE_KEY", "GRU_CLASS_MAP_KEY", "SASREC_TABLE_KEY",
          "load_checkpoint_state_dict", "item_table_from_state_dict", "load_item_table", "KnnEmbScorer"]
