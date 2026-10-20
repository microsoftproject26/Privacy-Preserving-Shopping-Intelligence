"""v4train.bootstrap_ci — the paired user bootstrap CIs the T1 selection needs (T1d and tie-break stage 1).

Computed only through v4eval.bootstrap (EVALUATION_CONTRACT_V4 §7): unit = user (eval_user_key) of the manifest's
E2E user set, 1000 resamples, seed 20260926, ONE shared resample matrix W for every family and seed, statistic
macro_e2e of the MRR@20 credit, 95% percentile CI. The inputs are the exact int64 ranks of each run's PRACTICAL_BEST
evaluation (the rank file the trainer retains per evaluation, DISK rule 4).
  delta_bar_ci  seed-mean contrast GRU - SASREC over seeds 2026 / 2027 on M02   (T1d)
  delta_ci      single-seed contrast GRU - SASREC (seed 2026) on the full warm manifest (tie-break stage 1)
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from v4eval.bootstrap import contrast, make_plan, seed_contrast, user_stats
from v4eval.manifest import EvalManifest
from v4eval.metrics import credits_from_ranks

from .checkpoint import read_json


def practical_best_ranks(run_dir, token=None) -> np.ndarray:
    """The int64 ranks of a finished run's PRACTICAL_BEST evaluation (from its retained int32 rank file). TRAIN-03: a
    sealed contest run is read only with the UnsealToken of the G2 unseal gate (v4train.unseal.sealed_root)."""
    from .unseal import sealed_root
    root = sealed_root(run_dir, token)
    s = read_json(root / "RUN_SUMMARY.json")
    mark = float(s["practical_best"]["mark"])
    return np.load(root / "evals" / f"eval_{mark:.1f}.ranks_int32.npy").astype(np.int64)


def _stats(manifest: EvalManifest, ranks: np.ndarray, label: str):
    r = np.asarray(ranks, dtype=np.int64)
    if r.shape != (manifest.n,):
        raise ValueError(f"{label}: ranks must have one entry per manifest row")
    credit = credits_from_ranks(r, (20,))["mrr@20"]
    return user_stats(credit, manifest.eval_mask, manifest.rankable, manifest.eval_user_key, label)


def _plan(manifest: EvalManifest):
    return make_plan(np.unique(manifest.eval_user_key[manifest.eval_mask]))


def delta_bar_ci(manifest: EvalManifest, ranks: Mapping) -> dict:
    """ranks[(family, seed)] -> int ranks on `manifest`; returns the seed-mean GRU - SASREC contrast (T1d)."""
    plan = _plan(manifest)
    a = {s: _stats(manifest, ranks[("GRU", s)], f"GRU:{s}") for s in (2026, 2027)}
    b = {s: _stats(manifest, ranks[("SASREC", s)], f"SASREC:{s}") for s in (2026, 2027)}
    return seed_contrast(plan, a, b, "macro_e2e", "delta_bar_GRU_minus_SASREC")


def delta_ci(manifest: EvalManifest, ranks_gru: np.ndarray, ranks_sasrec: np.ndarray, *, seed: int = 2026) -> dict:
    plan = _plan(manifest)
    return contrast(plan, _stats(manifest, ranks_gru, f"GRU:{seed}"), _stats(manifest, ranks_sasrec, f"SASREC:{seed}"),
                    "macro_e2e", f"delta_full_{seed}_GRU_minus_SASREC")


__all__ = ["practical_best_ranks", "delta_bar_ci", "delta_ci"]
