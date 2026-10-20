"""Learning-rate schedule, epoch accounting, deadline cut rule and configuration selection
(owner: Opus). Pure functions, no I/O.

Registry v2 (owner 38 h deadline, Fable Gate P2) schedule S_prime. Update index u is
1-based and never resets on resume; U = ceil(N_train / 256) = 156,956 updates per epoch.
    u <= warmup:                       lr = peak * u / warmup
    warmup < u <= P*U:                 lr = peak                       (P = peak_epochs = 1)
    P*U < u <= (P+C)*U:                cosine from peak to floor        (C = cosine_epochs = 1)
    u > (P+C)*U:                       lr = floor = floor_mult * peak   (floor_mult = 0.1)
So epochs 3.. (1-based) run entirely at the floor rate: with the planned 6 epochs, four
full low-LR epochs. No early stopping: training runs a fixed number of epochs (planned 6,
or the one-time systems-only deadline cap in [4..6], identical for both anchors).
Full-VS evaluation happens only at floor epochs; best = raw maximum over the evaluated
epochs, an exact tie keeps the earlier epoch. The schedule never consults validation
results or the wall clock.

`individual_should_stop` / `pair_should_stop` implement the superseded v1 early-stopping
protocol; they are kept (and tested) for the record but are not used by the v2 trainer.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Optional

WARMUP_UPDATES = 2000
FLOOR_MULT = 0.1
FLOOR_START_EPOCH = 4
MIN_EPOCHS = 5
MAX_EPOCHS = 8
PATIENCE = 2
MIN_DELTA = 2e-4
SELECTION_TIE = 1e-6

assert MIN_EPOCHS - (FLOOR_START_EPOCH - 1) >= 2, "need >= 2 complete floor-LR epochs before any early stop"


def updates_per_epoch(n_train: int, effective_batch: int = 256) -> int:
    return -(-int(n_train) // int(effective_batch))


PEAK_EPOCHS = 1
COSINE_EPOCHS = 1
PLANNED_EPOCHS = 6
MIN_EPOCHS_UNDER_CUT = 4


def lr_at(u: int, peak: float, U: int, warmup: int = WARMUP_UPDATES, floor_mult: float = FLOOR_MULT,
          peak_epochs: int = PEAK_EPOCHS, cosine_epochs: int = COSINE_EPOCHS) -> float:
    if u < 1:
        raise ValueError("update index u is 1-based")
    if U < 1 or warmup < 1 or peak_epochs < 1 or cosine_epochs < 1:
        raise ValueError("U, warmup, peak_epochs and cosine_epochs must be >= 1")
    if warmup > peak_epochs * U:
        raise ValueError("warmup must end inside the peak phase")
    floor = floor_mult * peak
    p_end, c_end = peak_epochs * U, (peak_epochs + cosine_epochs) * U
    if u <= warmup:
        return peak * u / warmup
    if u <= p_end:
        return peak
    if u <= c_end:
        return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * (u - p_end) / (c_end - p_end)))
    return floor


def lr_phase(u: int, U: int, warmup: int = WARMUP_UPDATES, peak_epochs: int = PEAK_EPOCHS,
             cosine_epochs: int = COSINE_EPOCHS) -> str:
    if u <= warmup:
        return "warmup"
    if u <= peak_epochs * U:
        return "peak"
    if u <= (peak_epochs + cosine_epochs) * U:
        return "cosine"
    return "floor"


def floor_start_epoch(peak_epochs: int = PEAK_EPOCHS, cosine_epochs: int = COSINE_EPOCHS) -> int:
    """First 1-based epoch whose every update runs at the floor LR."""
    return peak_epochs + cosine_epochs + 1


def eval_epochs(n_epochs: int, floor_start: int = 3) -> list:
    """Epochs with a full VALIDATION_SELECT evaluation: the floor epochs only."""
    return list(range(int(floor_start), int(n_epochs) + 1))


def epoch_cap(now_s: float, deadline_s: float, arms: list, *, planned: int = PLANNED_EPOCHS,
              min_epochs: int = MIN_EPOCHS_UNDER_CUT, post_hours: float = 2.0, safety_hours: float = 1.0,
              floor_start: int = 3, eval_seconds_default: float = 1600.0) -> int:
    """One-time, systems-only deadline cut (registry v2 cut_rule), same value for both anchors.

    arms: [{epochs_done, mean_train_epoch_seconds, eval_seconds (None if unknown)}].
    Returns the largest E in [min_epochs..planned] whose projected finish (slower arm) plus
    post-training time is <= deadline - safety; min_epochs if none fits. Never reads scores."""
    if not arms or min_epochs > planned or min_epochs < floor_start + 1:
        raise ValueError("need arms and floor_start < min_epochs <= planned")
    budget = float(deadline_s) - float(now_s) - (post_hours + safety_hours) * 3600.0
    for E in range(int(planned), int(min_epochs) - 1, -1):
        worst = 0.0
        for a in arms:
            done = int(a["epochs_done"])
            ev = a.get("eval_seconds")
            ev = float(ev) if ev else float(eval_seconds_default)
            remaining_train = max(0, E - done) * float(a["mean_train_epoch_seconds"])
            remaining_evals = len([e for e in eval_epochs(E, floor_start) if e > done])
            worst = max(worst, remaining_train + remaining_evals * ev)
        if worst <= budget:
            return E
    return int(min_epochs)


def fixed_epoch_should_stop(epoch: int, n_epochs: int) -> bool:
    if epoch > n_epochs:
        raise ValueError(f"epoch {epoch} exceeds the fixed budget {n_epochs}")
    return epoch == n_epochs


@dataclass
class StopState:
    best_metric: Optional[float] = None
    best_epoch: Optional[int] = None
    post_floor_misses: int = 0
    last_epoch: int = 0
    history: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "StopState":
        return cls(best_metric=d.get("best_metric"), best_epoch=d.get("best_epoch"),
                   post_floor_misses=int(d.get("post_floor_misses", 0)),
                   last_epoch=int(d.get("last_epoch", 0)), history=list(d.get("history", [])))


def epoch_end(state: StopState, epoch: int, metric: float, *,
              floor_start_epoch: int = FLOOR_START_EPOCH,
              min_delta: float = MIN_DELTA) -> tuple[StopState, dict]:
    """Fold one epoch-end full-VS evaluation into a NEW state (input not mutated)."""
    if epoch != state.last_epoch + 1:
        raise ValueError(f"epoch {epoch} does not follow last evaluated epoch {state.last_epoch}")
    metric = float(metric)
    if not math.isfinite(metric):
        raise ValueError("non-finite selection metric")
    prev_best = state.best_metric
    raw_improved = prev_best is None or metric > prev_best
    hit = prev_best is None or (metric - prev_best) >= min_delta
    counted = epoch >= floor_start_epoch
    if not counted:
        misses = 0
    elif epoch == floor_start_epoch:
        misses = 0 if hit else 1              # counting starts fresh at the first floor epoch
    else:
        misses = 0 if hit else state.post_floor_misses + 1
    new = StopState(
        best_metric=metric if raw_improved else prev_best,
        best_epoch=epoch if raw_improved else state.best_epoch,
        post_floor_misses=misses,
        last_epoch=epoch,
        history=list(state.history),
    )
    rec = {"epoch": epoch, "metric": metric, "prev_best": prev_best,
           "delta": None if prev_best is None else metric - prev_best,
           "raw_improved": raw_improved, "meaningful_hit": hit, "post_floor_counted": counted,
           "post_floor_misses": new.post_floor_misses, "best_metric": new.best_metric,
           "best_epoch": new.best_epoch}
    new.history.append(rec)
    return new, rec


def individual_should_stop(state: StopState, epoch: int, *, min_epochs: int = MIN_EPOCHS,
                           max_epochs: int = MAX_EPOCHS, patience: int = PATIENCE) -> tuple[bool, Optional[str]]:
    if state.last_epoch != epoch:
        raise ValueError("stop decision must use the state of this epoch")
    if epoch >= max_epochs:
        return True, "max_epochs"
    if epoch >= min_epochs and state.post_floor_misses >= patience:
        return True, "patience"
    return False, None


def pair_should_stop(state_a: StopState, state_b: StopState, epoch: int, *,
                     min_epochs: int = MIN_EPOCHS, max_epochs: int = MAX_EPOCHS,
                     patience: int = PATIENCE) -> tuple[bool, Optional[str]]:
    """Joint rule for the matched anchors; both arms call it with the same inputs."""
    if state_a.last_epoch != epoch or state_b.last_epoch != epoch:
        raise ValueError("pair decision needs both arms evaluated at the same epoch")
    if epoch >= max_epochs:
        return True, "max_epochs"
    if (epoch >= min_epochs and state_a.post_floor_misses >= patience
            and state_b.post_floor_misses >= patience):
        return True, "pair_patience"
    return False, None


def select_configuration(summaries: list[dict], registry_order: list[str]) -> dict:
    """Choose ONE configuration from the primary-seed search runs (04 Phase 4).

    Each summary: run_id, seed, final_status, best_metric (full-VS end-to-end macro
    MRR@20), micro_recall20_at_best, n_params. Rule: highest best_metric; within 1e-6
    -> higher micro Recall@20 -> smaller parameter count -> registry order.
    Only COMPLETE runs are eligible; every other run is reported as ineligible.
    """
    order = {rid: i for i, rid in enumerate(registry_order)}
    eligible = [s for s in summaries if s.get("final_status") == "COMPLETE"
                and s.get("best_metric") is not None and math.isfinite(float(s["best_metric"]))]
    ineligible = [{"run_id": s.get("run_id"), "final_status": s.get("final_status")}
                  for s in summaries if s not in eligible]
    if not eligible:
        return {"selected": None, "reason": "no eligible COMPLETE run", "ineligible": ineligible}
    top = max(float(s["best_metric"]) for s in eligible)
    tied = [s for s in eligible if float(s["best_metric"]) >= top - SELECTION_TIE]
    tied.sort(key=lambda s: (-float(s.get("micro_recall20_at_best") or 0.0),
                             int(s.get("n_params") or 0), order.get(s["run_id"], 10**9)))
    chosen = tied[0]
    return {"selected": chosen["run_id"], "selected_best_metric": float(chosen["best_metric"]),
            "tied_within_1e-6": [s["run_id"] for s in tied],
            "ranking": sorted(((s["run_id"], float(s["best_metric"])) for s in eligible),
                              key=lambda t: -t[1]),
            "ineligible": ineligible,
            "rule": "max full-VS macro MRR@20; tie<=1e-6 -> micro R@20 -> fewer params -> registry order"}
