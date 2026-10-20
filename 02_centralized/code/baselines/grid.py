"""v4base.grid — the registered SKNN grid (Amendment 02 §4): k in {50, 100, 200, 500} x sample_size in
{500, 1000, 5000}, tuned on CALIB_VALIDATION (fit on CALIB_TRAIN); selection metric = macro MRR@20, E2E denominator
(the same headline the trainer's own T1 selection reads, `v4eval` result[E2E][macro][mrr@20]`)."""
from __future__ import annotations

from typing import Iterable, List, Tuple

K_GRID: Tuple[int, ...] = (50, 100, 200, 500)
SAMPLE_SIZE_GRID: Tuple[int, ...] = (500, 1000, 5000)
SELECTION_METRIC = "mrr@20"
SELECTION_AGGREGATION = "macro"
SELECTION_DENOMINATOR = "E2E"


def grid_points() -> List[Tuple[int, int]]:
    return [(k, s) for k in K_GRID for s in SAMPLE_SIZE_GRID]


def metric_value(result: dict) -> float:
    """The registered selection metric of one evaluate() result."""
    return float(result[SELECTION_DENOMINATOR][SELECTION_AGGREGATION][SELECTION_METRIC])


def select_best(points: Iterable[dict]) -> dict:
    """`points`: [{"k": int, "sample_size": int, "value": float}, ...]. Picks the max `value`; a tie (|Δ| == 0,
    the grid has no registered tie-break tolerance of its own) prefers the smaller (k, sample_size) — fewer
    resources, matching the general spirit of Amendment 02's own tie rule ("fewer deviations")."""
    pts = list(points)
    if not pts:
        raise ValueError("select_best: no grid points to select from")
    best = None
    for p in pts:
        key = (-float(p["value"]), int(p["k"]), int(p["sample_size"]))
        if best is None or key < best[0]:
            best = (key, p)
    return best[1]


__all__ = ["K_GRID", "SAMPLE_SIZE_GRID", "SELECTION_METRIC", "SELECTION_AGGREGATION", "SELECTION_DENOMINATOR",
           "grid_points", "metric_value", "select_best"]
