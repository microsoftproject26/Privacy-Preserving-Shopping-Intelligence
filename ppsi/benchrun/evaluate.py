"""Score an EvalView with a model adapter (full catalogue, seen-item filtering) and summarise.

Eval mode, no grad; the model runs on its own device (CPU, or the GPU device class), the ranking always on CPU.
"""
from __future__ import annotations

import numpy as np

from .common import GROUPS, METRICS, PRIMARY_METRIC
from .metrics import grouped_metrics, metrics_from_ranks, ranks_from_scores


def rank_chunk(adapter, view, sel, P=None) -> np.ndarray:
    """Ranks of the targets of view rows `sel` under `adapter` (the module must already be in eval mode).
    P = a dense [U, d] personal matrix (PF: shared + p_u); None for every other arm."""
    import torch
    sel = np.asarray(sel, dtype=np.int64)
    batch = view.batch(sel)
    dev = getattr(adapter, "device", None)
    if dev is not None and dev.type != "cpu":                  # GPU device class: inputs to the model's device
        from ppsi.central.objective import to_device
        batch = to_device(batch, dev)
    with torch.no_grad():
        q = adapter.query(batch)
        if P is not None:
            q = q + P.index_select(0, torch.as_tensor(view.users[sel], dtype=torch.long)).to(q.device)
        scores = adapter.logits(q).to(torch.float32).contiguous()
    if scores.device.type != "cpu":                            # the (exact, integer) ranking stays on CPU
        scores = scores.cpu()
    rows, items = view.seen_pairs(sel)
    return ranks_from_scores(scores, view.targets[sel], rows, items)


def evaluate_view(adapter, view, *, chunk: int = 512, subset: np.ndarray | None = None, P=None) -> np.ndarray:
    """int64 ranks [len(view.users)] (0 = unreachable / seen target). `subset` = row positions to score (others -1)."""
    import torch
    n = len(view.users)
    ranks = np.full(n, -1, dtype=np.int64)
    rows = np.arange(n, dtype=np.int64) if subset is None else np.asarray(subset, dtype=np.int64)
    # length-sorted chunks keep the padded width small; the result is written back by row position
    order = rows[np.argsort(view.ends[rows], kind="stable")]
    was = adapter.module.training
    adapter.module.eval()
    try:
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            for s in range(0, order.size, chunk):
                sel = order[s:s + chunk]
                ranks[sel] = rank_chunk(adapter, view, sel, P)
    finally:
        adapter.module.train(was)
    return ranks


def summarize(ranks: np.ndarray, view, band: np.ndarray) -> dict:
    """ALL / BAND / OTHERS plus two descriptive rows: users whose target item is / is not covered by the band's
    inner-train items (BAND_COVERED_TARGET / BAND_UNCOVERED_TARGET)."""
    r = np.asarray(ranks)
    if (r < 0).any():
        raise RuntimeError("unscored rows in the rank vector")
    out = grouped_metrics(r, view.users, band)
    cov = view.data.band_item_mask()[view.targets]
    cov = cov & (view.targets >= 0)
    out["BAND_COVERED_TARGET"] = metrics_from_ranks(r[cov])
    out["BAND_UNCOVERED_TARGET"] = metrics_from_ranks(r[~cov])
    return out


def primary(summary: dict, group: str = "ALL") -> float:
    v = summary[group][PRIMARY_METRIC]
    if v is None:
        raise RuntimeError(f"{group} has no users")
    return float(v)


def flat_row(summary: dict) -> dict:
    """{ALL_ndcg@10: ..., BAND_...} for jsonl logging."""
    out = {}
    for g in GROUPS:
        for m in METRICS:
            out[f"{g}_{m}"] = summary[g][m]
        out[f"{g}_n_users"] = summary[g]["n_users"]
    return out
