"""T2 AP and the existing upstream client top-3 companion, not macro AP."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score


def evaluate_t2(labels, probabilities, clients, row_ordinals) -> dict:
    y0 = np.asarray(labels)
    p = np.asarray(probabilities, dtype=np.float64)
    ids = np.asarray(clients)
    order = np.asarray(row_ordinals)
    n = len(y0)
    if n == 0 or any(x.ndim != 1 or len(x) != n for x in (y0, p, ids, order)):
        raise ValueError("nonempty aligned 1D inputs required")
    if not np.isin(y0, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("T2 requires observed binary labels and finite probabilities")
    if order.dtype.kind not in "iu" or (order < 0).any() or len(np.unique(order)) != n:
        raise ValueError("immutable source row ordinals must be unique nonnegative integers")
    if any(x is None or (isinstance(x, float) and not np.isfinite(x)) for x in ids):
        raise ValueError("client identity cannot be missing")
    y = y0.astype(np.int8)
    _, inv = np.unique(ids, return_inverse=True)
    counts = np.bincount(inv)
    positives = np.bincount(inv, weights=y)
    # Exact score ties follow original frozen artifact row order, as upstream method='first'.
    perm = np.lexsort((order, -p, inv))
    sorted_clients = inv[perm]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_clients)) + 1]
    local_position = np.arange(n) - np.repeat(starts, np.diff(np.r_[starts, n]))
    caught = np.bincount(
        inv[perm[local_position < 3]], weights=y[perm[local_position < 3]], minlength=len(counts)
    )
    buyers = positives > 0
    # Upstream feeds one recall scalar per buyer to summarise: both reported
    # averages then use the client as the unit. Do not relabel this decision-micro.
    recall = caught[buyers] / positives[buyers]
    both_classes = np.unique(y).size == 2
    return {
        "observed_decisions": n,
        "clients_with_observed_decisions": len(counts),
        "positive_decisions": int(y.sum()),
        "pr_auc_status": "AVAILABLE" if both_classes else "UNDEFINED_SINGLE_CLASS",
        "pr_auc_micro": float(average_precision_score(y, p)) if both_classes else None,
        "pr_auc_definition": "sklearn.average_precision_score; not trapezoidal PR area",
        "macro_pr_auc": None,
        "macro_pr_auc_reason": "not used by the upstream T2 protocol",
        "top3_status": "AVAILABLE" if buyers.any() else "ZERO_POSITIVE_CLIENT_SUPPORT",
        "client_recall_at_3_macro": float(recall.mean()) if buyers.any() else None,
        "client_recall_at_3_client_unit_mean": float(recall.mean()) if buyers.any() else None,
        "positive_client_support": int(buyers.sum()),
        "clients_without_purchase": int((~buyers).sum()),
        "top3_tie_break": "ascending original frozen artifact row ordinal",
    }
