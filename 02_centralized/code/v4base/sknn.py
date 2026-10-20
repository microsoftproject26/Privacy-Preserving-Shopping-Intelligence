"""v4base.sknn — SKNN baseline (Amendment 02 §4): session item-kNN, Ludewig & Jannach (2018) style, vanilla
(idf-free) cosine similarity, most-recent sampling; registered grid k in {50, 100, 200, 500} x sample_size in
{500, 1000, 5000}.

Past-session pool (fit-time, TRAIN only; this is the one judgment call this baseline needed and it is written up
here in full, per BASE-01's "read v4int.providers" instruction and its own hard rule against inventing hidden
assumptions): the provider exposes each TRAIN decision's causal history as a window (item_tokens/session_offset
per USER_HISTORY_L50), never a raw "list of all sessions" object — v4int/v4hist do not materialise one. Ludewig &
Jannach's own past-session pool is "every session in the training data, as a finished item set". The closest
faithful reconstruction from what the provider actually serves is, per TRAIN decision d:

    past_session(d) = {items with session_offset_tokens == 1 in d's own window} UNION {target_class(d)}

i.e. the current session's items strictly before d, plus the actually-realized next item of that same session
(TRAIN's own label for d) — the causal completion of that session prefix. This is built ENTIRELY from TRAIN rows
(their own history window and their own TRAIN label), so it never reads a VALIDATION/TEST row and stays strictly
causal; a session that recurs across several TRAIN decisions contributes one (growing) pool entry per decision,
which is the standard behaviour of a session-based kNN index over a stream of session-prefixes, not an inflation
bug. `session.py`/BASE-01.md name where this could be swapped for a table of complete raw sessions if one becomes
available without editing v4int/v4hist.

Similarity: vanilla/idf-free cosine of two binary item-membership vectors, |A n B| / sqrt(|A| |B|) (no IDF weight,
no position/recency decay inside the similarity itself — the "vanilla vsknn" reading of the grid the task packet
registers). Sampling: "most-recent sampling" restricts the candidate neighbour pool, before similarity ranking, to
the `sample_size` most recent past-sessions among those sharing >= 1 item with the query (recency = TRAIN scan
order, which is canonical ascending decision_id — a global, causal recency proxy). Final: keep the top `k` candidates
by similarity, sum each kept neighbour's similarity onto every item it contains (dense; untouched items score 0.0,
the standard "no support" SKNN value — no sentinel needed, 0.0 is finite and already the correct floor).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .guard import RssGuard
from .sessions import current_session_mask, history_valid_mask, item_classes

DEFAULT_FIT_CHUNK = 512


class SessionPool:
    """The past-session pool built from TRAIN decisions only, in TRAIN canonical (= causal recency) order."""

    def __init__(self, K: int):
        self.K = int(K)
        self.sessions: List[np.ndarray] = []            # sorted distinct item classes, one array per past session
        self._sizes: List[int] = []
        self._item_index: Dict[int, list] = {}           # item class -> [session_idx, ...] in append (recency) order
        self.sizes: Optional[np.ndarray] = None          # set by finalize()
        self.item_index: Optional[Dict[int, np.ndarray]] = None

    def add(self, items: Sequence[int]) -> None:
        arr = np.asarray(sorted(set(int(i) for i in items)), dtype=np.int64)
        sidx = len(self.sessions)
        self.sessions.append(arr)
        self._sizes.append(int(arr.size))
        for it in arr.tolist():
            self._item_index.setdefault(it, []).append(sidx)

    def finalize(self) -> None:
        self.sizes = np.asarray(self._sizes, dtype=np.int64)
        self.item_index = {k: np.asarray(v, dtype=np.int64) for k, v in self._item_index.items()}

    @property
    def n_sessions(self) -> int:
        return len(self.sessions)


def fit(provider, K: int, *, chunk: int = DEFAULT_FIT_CHUNK, guard: Optional[RssGuard] = None) -> SessionPool:
    """Scan TRAIN (BatchProvider) in canonical order; see the module docstring for the exact past_session(d) rule."""
    pool = SessionPool(K)
    n = int(provider.n_decisions)
    for start in range(0, n, chunk):
        pos = np.arange(start, min(n, start + chunk), dtype=np.int64)
        b = provider.batch(pos)
        item_tokens = b["item_tokens"]
        item_tokens = item_tokens if torch.is_tensor(item_tokens) else torch.as_tensor(item_tokens)
        cls, valid_tok = item_classes(item_tokens, K)
        csm = current_session_mask(b)
        # defense in depth (causality): session_offset_tokens is contractually 0 (never 1) on a PAD column, but the
        # explicit length mask is ANDed in too so a malformed/adversarial batch can never leak a column at or past
        # the row's own length into the pool, even if its session offset were wrong.
        mask = valid_tok & csm & history_valid_mask(b["lengths"], item_tokens.shape[1])
        tgt = b["target_class"]
        tgt = (tgt if torch.is_tensor(tgt) else torch.as_tensor(tgt)).to(torch.int64)
        for r in range(item_tokens.shape[0]):
            t = int(tgt[r])
            if t < 0 or t >= K:
                raise ValueError(f"SKNN fit: a TRAIN batch target_class is outside [0, {K})")
            items = cls[r][mask[r]].tolist()
            items.append(t)
            pool.add(items)
        if guard is not None:
            guard.check()
    pool.finalize()
    return pool


def _recent_candidates(pool: SessionPool, query_items: Sequence[int], sample_size: int) -> np.ndarray:
    """The most-recent `sample_size` past sessions among those sharing >= 1 item with the query: each item's own
    inverted list is already ascending by recency, so a tail slice (cheap: no copy of the full list) followed by a
    union and one more tail slice bounds both the per-item and the combined work to O(sample_size)."""
    tails = []
    for it in query_items:
        lst = pool.item_index.get(int(it)) if pool.item_index is not None else None
        if lst is not None and lst.size:
            tails.append(lst[-sample_size:] if lst.size > sample_size else lst)
    if not tails:
        return np.empty(0, dtype=np.int64)
    merged = np.unique(np.concatenate(tails))
    if merged.size > sample_size:
        merged = merged[-sample_size:]
    return merged


def score_session(pool: SessionPool, query_items: Sequence[int], *, k: int, sample_size: int) -> Dict[int, float]:
    """Item-class -> pooled similarity score for one query (current) session. Empty dict if the query has no items
    or no TRAIN session shares any item with it (a legitimate all-zero SKNN row, not an error)."""
    q = sorted(set(int(i) for i in query_items))
    if not q:
        return {}
    cand = _recent_candidates(pool, q, int(sample_size))
    if cand.size == 0:
        return {}
    qset = set(q)
    qnorm = math.sqrt(len(q))
    sims = np.empty(cand.size, dtype=np.float64)
    for i, s in enumerate(cand.tolist()):
        sess = pool.sessions[s]
        inter = len(qset.intersection(sess.tolist()))
        denom = qnorm * math.sqrt(int(pool.sizes[s])) if pool.sizes[s] else 0.0
        sims[i] = (inter / denom) if denom > 0 else 0.0
    kk = min(int(k), cand.size)
    top_idx = np.argpartition(-sims, kk - 1)[:kk] if cand.size > kk else np.arange(cand.size)
    scores: Dict[int, float] = {}
    for j in top_idx.tolist():
        sim = float(sims[j])
        if sim <= 0.0:
            continue
        for it in pool.sessions[cand[j]].tolist():
            scores[it] = scores.get(it, 0.0) + sim
    return scores


class SessionKNNScorer:
    def __init__(self, pool: SessionPool, *, k: int, sample_size: int):
        self.pool = pool
        self.k = int(k)
        self.sample_size = int(sample_size)

    def score_batch(self, batch: dict) -> torch.Tensor:
        item_tokens = batch["item_tokens"]
        item_tokens = item_tokens if torch.is_tensor(item_tokens) else torch.as_tensor(item_tokens)
        B = int(item_tokens.shape[0])
        K = self.pool.K
        cls, valid_tok = item_classes(item_tokens, K)
        csm = current_session_mask(batch)
        mask = valid_tok & csm & history_valid_mask(batch["lengths"], item_tokens.shape[1])
        out = torch.zeros((B, K), dtype=torch.float32)
        for r in range(B):
            items = cls[r][mask[r]].tolist()
            scores = score_session(self.pool, items, k=self.k, sample_size=self.sample_size)
            if scores:
                idx = torch.as_tensor(list(scores.keys()), dtype=torch.int64)
                val = torch.as_tensor(list(scores.values()), dtype=torch.float32)
                out[r, idx] = val
        return out


__all__ = ["SessionPool", "fit", "score_session", "SessionKNNScorer", "DEFAULT_FIT_CHUNK"]
