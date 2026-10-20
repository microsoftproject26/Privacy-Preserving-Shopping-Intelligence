"""Full-scale evaluator -- Wave-1 Sonnet C (owner of this file).

Implements `FULL_INTERFACES.md` section 2: exact full-catalogue ranking
(class-chunked, no top-k, no candidate filtering) and the aggregate metric
structure, generalised from the pilot's `evaluator.py` (frozen, imported
here, never edited) to:

  * GPU-resident ranking (`rank_targets`) that works for float32 AND int64
    scores (int64 is required at full scale for the TRANS-full composite
    score, which exceeds 2^24 and is therefore no longer float32-exact --
    see `baselines_full.py`).
  * `aggregate`, built directly on `evaluator._credits_from_ranks` and
    `evaluator._population_block` (imported, never re-derived) so every
    number matches `evaluator.evaluate()` bit-for-bit, plus a reporting-only
    `"slices"` block (prior_view_count / prefix_position buckets).
  * `evaluate_model`, the training-time / locked-eval driver over a
    `CompactStore`/`DecisionSet` pair (Sonnet A's `full_store.py`).
  * `per_user_metric` / `paired_bootstrap`, the pre-registered paired
    bootstrap machinery (`FULL_RUN_PROTOCOL.md` sec. 9).

This module never loads VALIDATION_CONFIRM or TEST decisions itself -- that
is entirely a caller concern (`lock_and_eval.py`).
"""
from __future__ import annotations

import queue
import random
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterator, Optional

CODE_DIR = Path(__file__).resolve().parent
VENDOR_DIR = CODE_DIR / "vendor"
for _p in (str(CODE_DIR), str(VENDOR_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import train as train_mod  # noqa: E402  -- build_named_kwargs (frozen, never edited)
from evaluator import _credits_from_ranks, _population_block  # noqa: E402  -- frozen helpers

__all__ = [
    "rank_targets",
    "aggregate",
    "evaluate_model",
    "per_user_metric",
    "paired_bootstrap",
]

DEFAULT_K = 20
DEFAULT_CLASS_CHUNK = 32768
DEFAULT_EVAL_BATCH = 1024


# ---------------------------------------------------------------------------
# rank_targets -- GPU (or CPU) exact rank, dtype-agnostic (float32 OR int64).
#
# Identical formula and identical class-chunked accumulation to
# `evaluator._ranks_chunked`: rank = 1 + #{s_j > s_t} + #{j < t: s_j == s_t}.
# The only generalisation is that `scores` is never coerced to float32 here
# (evaluator.evaluate() does that internally for the GRU's own logits, which
# is harmless for float32-native scores but would silently destroy exactness
# for the int64 baseline composite scores at full scale -- see
# `baselines_full.py`'s module docstring). Never `torch.topk`, never
# candidate filtering; correctness is independent of `class_chunk` by
# construction (an exact partition of the K axis).
# ---------------------------------------------------------------------------

def rank_targets(scores: torch.Tensor, target: torch.Tensor,
                  class_chunk: int = DEFAULT_CLASS_CHUNK) -> torch.Tensor:
    """scores: [B, K] float32 or int64 on any device; target: [B] int64, all
    entries >= 0 (caller pre-filters to rankable rows). Returns [B] int64
    ranks on the same device as `scores`."""
    if scores.ndim != 2:
        raise ValueError(f"scores must be [B, K], got shape {tuple(scores.shape)}")
    if target.ndim != 1 or target.shape[0] != scores.shape[0]:
        raise ValueError("target must be [B], matching scores.shape[0]")
    if class_chunk < 1:
        raise ValueError("class_chunk must be >= 1")

    n, K = scores.shape
    device = scores.device
    target = target.to(device=device, dtype=torch.long)

    s_t = scores.gather(1, target.view(-1, 1)).squeeze(1)  # [n], same dtype as scores
    gt = torch.zeros(n, dtype=torch.long, device=device)
    eq_before = torch.zeros(n, dtype=torch.long, device=device)
    target_col = target.view(-1, 1)

    step = int(class_chunk)
    for start in range(0, K, step):
        end = min(start + step, K)
        chunk = scores[:, start:end]                                   # [n, w]
        gt += (chunk > s_t.unsqueeze(1)).sum(dim=1)
        idx = torch.arange(start, end, device=device).unsqueeze(0)      # [1, w]
        eq_lower = (chunk == s_t.unsqueeze(1)) & (idx < target_col)
        eq_before += eq_lower.sum(dim=1)

    return 1 + gt + eq_before


# ---------------------------------------------------------------------------
# aggregate -- same numbers as evaluator.evaluate(), built on evaluator's OWN
# helpers, plus a reporting-only "slices" diagnostic block.
# ---------------------------------------------------------------------------

def _get(ds: Any, name: str) -> np.ndarray:
    val = ds[name] if isinstance(ds, dict) else getattr(ds, name)
    arr = np.asarray(val)
    if not arr.flags.writeable:
        arr = arr.copy()  # e.g. a read-only memmap-backed array -- avoid torch's non-writable warning
    return arr


def _slice_block(credits: dict, pop_mask: torch.Tensor, eval_user_key: torch.Tensor,
                  k: int) -> dict:
    b = _population_block(credits, pop_mask, eval_user_key)
    names = (f"mrr@{k}", f"recall@{k}")
    return {
        "n": b["n"],
        "user_count": b["macro"].get("user_count", 0),
        "micro": {name: b["micro"][name] for name in names},
        "macro": {name: b["macro"][name] for name in names},
    }


def aggregate(ranks: np.ndarray, ds: Any, k: int = DEFAULT_K) -> dict:
    """ranks: int64 per ds row, canonical order, -1 where no rankable target
    (not eval_mask, or target_class < 0). Returns the SAME structure as
    `evaluator.evaluate()` (k, end_to_end, closed, user_count,
    cohorts.warm/unseen), plus a reporting-only "slices" block."""
    ranks_np = np.asarray(ranks)
    target_class = _get(ds, "target_class").astype(np.int64)
    eval_mask = _get(ds, "eval_mask").astype(bool)
    eval_user_key_np = _get(ds, "eval_user_key")
    client_idx = _get(ds, "client_idx").astype(np.int64)

    N = target_class.shape[0]
    if not (ranks_np.shape[0] == N and eval_mask.shape[0] == N
            and eval_user_key_np.shape[0] == N and client_idx.shape[0] == N):
        raise ValueError(
            "ranks/ds.target_class/eval_mask/eval_user_key/client_idx must share length "
            f"(got ranks={ranks_np.shape[0]}, target_class={N}, eval_mask={eval_mask.shape[0]}, "
            f"eval_user_key={eval_user_key_np.shape[0]}, client_idx={client_idx.shape[0]})"
        )
    if k < 1:
        raise ValueError("k must be >= 1")

    ranks_t = torch.as_tensor(ranks_np, dtype=torch.long)
    target_class_t = torch.as_tensor(target_class, dtype=torch.long)
    eval_mask_t = torch.as_tensor(eval_mask, dtype=torch.bool)
    eval_user_key_t = torch.as_tensor(eval_user_key_np, dtype=torch.long)
    client_idx_t = torch.as_tensor(client_idx, dtype=torch.long)

    credits = _credits_from_ranks(ranks_t, k)

    end_to_end_pop = eval_mask_t
    closed_pop = eval_mask_t & (target_class_t >= 0)

    def block(pop_mask: torch.Tensor) -> dict:
        return _population_block(credits, pop_mask, eval_user_key_t)

    result: dict = {
        "k": k,
        "end_to_end": block(end_to_end_pop),
        "closed": block(closed_pop),
    }
    result["user_count"] = result["end_to_end"]["macro"]["user_count"]

    warm_mask = client_idx_t >= 3
    unseen_mask = client_idx_t == 2
    result["cohorts"] = {
        "warm": {
            "end_to_end": block(end_to_end_pop & warm_mask),
            "closed": block(closed_pop & warm_mask),
        },
        "unseen": {
            "end_to_end": block(end_to_end_pop & unseen_mask),
            "closed": block(closed_pop & unseen_mask),
        },
    }

    # -- reporting-only diagnostic slices ------------------------------------
    prior_view_count = torch.as_tensor(
        _get(ds, "prior_view_count").astype(np.int64), dtype=torch.long)
    prefix_position = torch.as_tensor(
        _get(ds, "prefix_position").astype(np.int64), dtype=torch.long)

    prior_bins = [
        ("0", prior_view_count == 0),
        ("1", prior_view_count == 1),
        ("2-4", (prior_view_count >= 2) & (prior_view_count <= 4)),
        ("5-9", (prior_view_count >= 5) & (prior_view_count <= 9)),
        ("10+", prior_view_count >= 10),
    ]
    prefix_bins = [
        ("1", prefix_position == 1),
        ("2", prefix_position == 2),
        ("3-4", (prefix_position >= 3) & (prefix_position <= 4)),
        ("5-9", (prefix_position >= 5) & (prefix_position <= 9)),
        ("10-19", (prefix_position >= 10) & (prefix_position <= 19)),
        ("20-49", (prefix_position >= 20) & (prefix_position <= 49)),
        ("50+", prefix_position >= 50),
    ]
    result["slices"] = {
        "prior_views": {
            name: _slice_block(credits, end_to_end_pop & mask, eval_user_key_t, k)
            for name, mask in prior_bins
        },
        "prefix_len": {
            name: _slice_block(credits, end_to_end_pop & mask, eval_user_key_t, k)
            for name, mask in prefix_bins
        },
    }
    return result


# ---------------------------------------------------------------------------
# Background prefetch (host-side batch build overlapped with GPU compute).
# ---------------------------------------------------------------------------

class _Prefetcher:
    """Wraps a generator in a single background thread with a small bounded
    queue, so the NEXT host batch (store.batch(), CPU/IO-bound) is built
    while the CURRENT batch's forward+rank runs on the GPU. Pure throughput
    optimisation -- iteration order and results are unaffected."""

    _SENTINEL_DONE = object()

    def __init__(self, gen: Iterator, maxsize: int = 2) -> None:
        self._q: "queue.Queue" = queue.Queue(maxsize=maxsize)
        self._thread = threading.Thread(target=self._worker, args=(gen,), daemon=True)
        self._thread.start()

    def _worker(self, gen: Iterator) -> None:
        try:
            for item in gen:
                self._q.put(("ok", item))
        except Exception as exc:  # pragma: no cover -- surfaced to consumer
            self._q.put(("err", exc))
        finally:
            self._q.put(("done", None))

    def __iter__(self) -> "_Prefetcher":
        return self

    def __next__(self):
        tag, item = self._q.get()
        if tag == "done":
            raise StopIteration
        if tag == "err":
            raise item
        return item


# ---------------------------------------------------------------------------
# evaluate_model
# ---------------------------------------------------------------------------

def evaluate_model(model, store, ds, device, *, eval_batch: int = DEFAULT_EVAL_BATCH,
                    k: int = DEFAULT_K, return_ranks: bool = False,
                    class_chunk: int = DEFAULT_CLASS_CHUNK, prefetch: bool = True) -> dict:
    """model.eval() + torch.no_grad(); CPU/CUDA RNG state saved and restored;
    restores model.training; iterates `ds` in canonical order in chunks of
    `eval_batch` via `store.batch(ds, positions)`; forward via
    `train.build_named_kwargs`; logits stay on `device`, ranks computed on
    `device`; only rows with eval_mask & target_class >= 0 are ranked
    (others get rank -1). Returns `aggregate(...)` (+ "ranks" if requested)
    plus timing/throughput/peak-memory diagnostics.

    `class_chunk` and `prefetch` are additive, defaulted keyword-only
    knobs beyond the FULL_INTERFACES.md signature (both optional, both
    backward compatible with a caller that only passes the documented
    kwargs) -- see the Sonnet C report for why they were added.
    """
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_available = torch.cuda.is_available()
    cuda_state = torch.cuda.get_rng_state_all() if cuda_available else None
    was_training = model.training

    N = len(ds)
    ranks = torch.full((N,), -1, dtype=torch.long)

    def batches():
        for start in range(0, N, eval_batch):
            end = min(start + eval_batch, N)
            positions = np.arange(start, end)
            yield start, end, store.batch(ds, positions)

    iterator: Iterator = _Prefetcher(batches()) if prefetch else batches()

    if cuda_available and device is not None and torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model.eval()
    t0 = time.perf_counter()
    n_scored = 0
    try:
        with torch.no_grad():
            for start, end, batch in iterator:
                batch_gpu = {name: v.to(device, non_blocking=True) for name, v in batch.items()}
                kw = train_mod.build_named_kwargs(batch_gpu, model)
                logits = model(**kw)  # [B, K] on device

                target = batch_gpu["target_class"]
                eval_mask_b = batch_gpu["eval_mask"]
                need = eval_mask_b & (target >= 0)
                if bool(need.any()):
                    local_idx = need.nonzero(as_tuple=True)[0]
                    sub_scores = logits.index_select(0, local_idx)
                    sub_target = target.index_select(0, local_idx)
                    r = rank_targets(sub_scores, sub_target, class_chunk=class_chunk)
                    global_idx = (local_idx.to("cpu") + start)
                    ranks[global_idx] = r.to("cpu")
                n_scored += (end - start)
    finally:
        if was_training:
            model.train()
        else:
            model.eval()
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)

    if cuda_available and device is not None and torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0

    peak_mem_bytes = None
    if cuda_available and device is not None and torch.device(device).type == "cuda":
        peak_mem_bytes = int(torch.cuda.max_memory_allocated(device))

    if n_scored != N:
        raise RuntimeError(f"evaluate_model scored {n_scored} rows, expected {N} = len(ds)")

    result = aggregate(ranks.numpy(), ds, k=k)
    result["_elapsed_s"] = elapsed
    result["_rows_per_s"] = (N / elapsed) if elapsed > 0 else None
    result["_peak_gpu_mem_bytes"] = peak_mem_bytes
    result["_n_rows"] = N
    if return_ranks:
        result["ranks"] = ranks.numpy()
    return result


# ---------------------------------------------------------------------------
# per_user_metric / paired_bootstrap -- FULL_RUN_PROTOCOL.md sec. 9.
# ---------------------------------------------------------------------------

_VALID_METRICS = ("recall", "mrr", "ndcg")
_VALID_DENOMINATORS = ("end_to_end", "closed")


def per_user_metric(ranks: np.ndarray, ds: Any, metric: str = "mrr", k: int = DEFAULT_K,
                     denominator: str = "end_to_end") -> tuple[np.ndarray, np.ndarray]:
    """Per-user mean credit for `metric`@`k` under `denominator`, computed the
    same way `evaluator._population_block` groups macro credit (grouped by
    `eval_user_key`, never `client_id`). Returns (user_keys, per_user_mean),
    both length = number of DISTINCT users present in the population."""
    if metric not in _VALID_METRICS:
        raise ValueError(f"metric must be one of {_VALID_METRICS}, got {metric!r}")
    if denominator not in _VALID_DENOMINATORS:
        raise ValueError(f"denominator must be one of {_VALID_DENOMINATORS}, got {denominator!r}")

    ranks_t = torch.as_tensor(np.asarray(ranks), dtype=torch.long)
    target_class = torch.as_tensor(_get(ds, "target_class").astype(np.int64), dtype=torch.long)
    eval_mask = torch.as_tensor(_get(ds, "eval_mask").astype(bool), dtype=torch.bool)
    eval_user_key = torch.as_tensor(_get(ds, "eval_user_key"), dtype=torch.long)

    if denominator == "end_to_end":
        pop_mask = eval_mask
    else:
        pop_mask = eval_mask & (target_class >= 0)

    credits = _credits_from_ranks(ranks_t, k)
    credit = credits[f"{metric}@{k}"]

    idx = pop_mask.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)

    keys = eval_user_key[idx]
    sub = credit[idx].to(torch.float64)
    unique_keys, inverse = torch.unique(keys, return_inverse=True)
    n_users = int(unique_keys.numel())

    sums = torch.zeros(n_users, dtype=torch.float64)
    counts = torch.zeros(n_users, dtype=torch.float64)
    sums.index_add_(0, inverse, sub)
    counts.index_add_(0, inverse, torch.ones_like(sub))
    per_user_mean = (sums / counts).numpy()

    return unique_keys.numpy().astype(np.int64), per_user_mean


def paired_bootstrap(user_keys_a, vals_a, user_keys_b, vals_b, n: int = 1000,
                      seed: int = 2026) -> dict:
    """Paired user-level bootstrap (FULL_RUN_PROTOCOL.md sec. 9): resample
    users with replacement, `n` resamples, `np.random.default_rng(seed)`,
    95% percentile CI of the macro-metric difference (a - b). Both inputs
    must cover the IDENTICAL user set (asserted); values are aligned by
    user_keys before differencing."""
    ka = np.asarray(user_keys_a)
    kb = np.asarray(user_keys_b)
    va = np.asarray(vals_a, dtype=np.float64)
    vb = np.asarray(vals_b, dtype=np.float64)
    if ka.shape[0] != va.shape[0] or kb.shape[0] != vb.shape[0]:
        raise ValueError("user_keys/vals length mismatch")

    order_a = np.argsort(ka, kind="stable")
    order_b = np.argsort(kb, kind="stable")
    ka_sorted, va_sorted = ka[order_a], va[order_a]
    kb_sorted, vb_sorted = kb[order_b], vb[order_b]

    if ka_sorted.shape[0] != kb_sorted.shape[0] or not np.array_equal(ka_sorted, kb_sorted):
        raise ValueError(
            "paired_bootstrap requires the identical user set for both models "
            f"(n_a={ka_sorted.shape[0]}, n_b={kb_sorted.shape[0]}, "
            f"equal_sets={ka_sorted.shape[0] == kb_sorted.shape[0] and np.array_equal(ka_sorted, kb_sorted)})"
        )

    n_users = int(ka_sorted.shape[0])
    point_diff = float(va_sorted.mean() - vb_sorted.mean()) if n_users else float("nan")

    rng = np.random.default_rng(seed)
    diffs = np.empty(int(n), dtype=np.float64)
    for r in range(int(n)):
        sample = rng.integers(0, n_users, size=n_users) if n_users else np.empty(0, dtype=np.int64)
        diffs[r] = va_sorted[sample].mean() - vb_sorted[sample].mean() if n_users else float("nan")

    lo, hi = (np.percentile(diffs, [2.5, 97.5]) if n_users else (float("nan"), float("nan")))
    return {
        "diff": point_diff,
        "ci95": [float(lo), float(hi)],
        "n_users": n_users,
        "n_resamples": int(n),
        "seed": int(seed),
    }
