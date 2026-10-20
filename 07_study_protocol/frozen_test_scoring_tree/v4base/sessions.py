"""v4base.sessions — shared readers over a Clean V2 batch dict (v4hist.windows / v4models.synthetic layout).

Read-only reading of the DOCUMENTED batch layout, nothing imported from v4hist/v4models is edited:
  item_tokens [B, L] int, right-padded: real events at columns [0, length-1] in ascending recency (column
    length-1 is the event immediately before the decision, i.e. the most recent); tokens 0..2 are PAD/BOS/OOV
    (never a catalogue item); a token >= 3 maps to class = token - ITEM_TOKEN_OFFSET in [0, K).
  lengths [B] int: the number of real (non-PAD) columns of that row.
  session_offset_tokens [B, L] int64 (USER_HISTORY_L50 only, DECISION_LOG 47; v4hist.windows.session_offset_tokens):
    0 on PAD, 1 on every event of the CURRENT session (the one the decision itself belongs to), an increasing
    integer further back in time. `current_session_mask` reads this column; LAST does not need it (LAST scores the
    whole causal history, not just the current session).
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import torch

ITEM_TOKEN_OFFSET = 3
SESSION_OFFSET_KEY = "session_offset_tokens"
CURRENT_SESSION_OFFSET = 1


def _as_tensor(x) -> torch.Tensor:
    return x if torch.is_tensor(x) else torch.as_tensor(x)


def item_classes(item_tokens, K: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """(classes, valid): classes = item_tokens - ITEM_TOKEN_OFFSET (same shape); valid = a real catalogue item
    (token >= ITEM_TOKEN_OFFSET and the resulting class is in [0, K)). Invalid positions carry an arbitrary class
    value that the caller must never read without checking `valid`."""
    tok = _as_tensor(item_tokens).to(torch.int64)
    cls = tok - ITEM_TOKEN_OFFSET
    valid = (tok >= ITEM_TOKEN_OFFSET) & (cls >= 0) & (cls < int(K))
    return cls, valid


def history_valid_mask(lengths, L: int) -> torch.Tensor:
    """[B, L] bool: column j is a real (non-PAD) event of that row iff j < lengths[row]."""
    ln = _as_tensor(lengths).to(torch.int64)
    pos = torch.arange(int(L), dtype=torch.int64).unsqueeze(0)
    return pos < ln.unsqueeze(1)


def current_session_mask(batch: dict) -> torch.Tensor:
    """[B, L] bool: column j is part of the decision's OWN (current) session. Raises KeyError with a clear message
    if the batch carries no session_offset_tokens (SESSION_L50 batches, or a provider that predates DL 47) — SKNN
    needs USER_HISTORY_L50; refuse rather than silently treat every column as one session."""
    if SESSION_OFFSET_KEY not in batch:
        raise KeyError(f"SKNN needs {SESSION_OFFSET_KEY!r} in the batch (USER_HISTORY_L50 session offsets, "
                       "DECISION_LOG 47); this provider/policy does not carry it")
    tok = _as_tensor(batch[SESSION_OFFSET_KEY]).to(torch.int64)
    return tok == CURRENT_SESSION_OFFSET


def row_current_session_items(item_tokens_row, K: int, session_mask_row: torch.Tensor) -> list:
    """Sorted distinct item classes of one 1-D row's current session (session_offset == 1 AND a real catalogue
    item). `item_tokens_row` and `session_mask_row` are both [L] (one decision's own history)."""
    cls, valid = item_classes(item_tokens_row, K)
    m = valid & session_mask_row
    return sorted(set(cls[m].tolist()))


def _key(v):
    return v.item() if isinstance(v, np.generic) else v


def scan_best_window_per_user(provider, *, user_keys, extra_fields: Sequence[str] = (), chunk: int = 2048,
                              guard=None) -> Dict:
    """One MAIN-TRAIN scan: for each user (identified by the CALLER-supplied `user_keys`, one entry per canonical
    position, aligned exactly like `provider.batch(positions)`'s own indexing — this function knows nothing about
    client_ord/client_key_hex and does no manifest-table reading itself; see `v4base.identity` for how a real
    caller resolves the correct key), keep ONLY their single longest (`lengths`) TRAIN decision's window (tie -> the
    larger decision_id) — a superset of every other TRAIN decision of theirs, up to the L-cap — plus their total
    TRAIN decision count (`n_train`, a count of DECISIONS, never de-duplicated).

    This is the de-duplication primitive shared by `v4base.rule.fit_category_brand_map` and
    `v4base.strata.scan_user_train_stats`: without it, the SAME real historical event is counted once per LATER
    decision whose window still covers it (up to ~L times), biasing any per-event share/vote statistic (a real
    per-user brand-loyalty share, or an item's category/brand mode vote).

    Returns {user_key: {"n_train": int, "lengths": int, "decision_id": int, "item_tokens": [int,...],
    <extra_field>: [int,...], ...}} — the stored per-field lists are the window's FULL L columns (including PAD);
    callers apply their own valid/causal masks (`item_classes`, `history_valid_mask`) when reading them back, same
    as any other batch."""
    n = int(provider.n_decisions)
    keys = np.asarray(user_keys)
    if keys.shape[0] != n:
        raise ValueError(f"user_keys has {keys.shape[0]} entries, expected {n} (one per TRAIN decision, aligned to "
                         "canonical position)")
    fields = ("item_tokens",) + tuple(extra_fields)
    n_train: Dict = {}
    best: Dict = {}
    for start in range(0, n, chunk):
        pos = np.arange(start, min(n, start + chunk), dtype=np.int64)
        b = provider.batch(pos)
        lengths = _as_tensor(b["lengths"]).to(torch.int64)
        dids = _as_tensor(b["decision_id"]).to(torch.int64)
        cols = {f: _as_tensor(b[f]) for f in fields}
        keys_here = keys[pos]
        for i in range(pos.shape[0]):
            u = _key(keys_here[i])
            L, d = int(lengths[i]), int(dids[i])
            n_train[u] = n_train.get(u, 0) + 1
            cur = best.get(u)
            if cur is None or L > cur["lengths"] or (L == cur["lengths"] and d > cur["decision_id"]):
                best[u] = {"lengths": L, "decision_id": d, **{f: cols[f][i].tolist() for f in fields}}
        if guard is not None:
            guard.check()
    for u, row in best.items():
        row["n_train"] = n_train[u]
    return best


__all__ = ["ITEM_TOKEN_OFFSET", "SESSION_OFFSET_KEY", "CURRENT_SESSION_OFFSET", "item_classes", "history_valid_mask",
           "current_session_mask", "row_current_session_items", "scan_best_window_per_user"]
