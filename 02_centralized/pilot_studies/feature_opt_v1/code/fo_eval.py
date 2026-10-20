"""fo_eval.py (owner: Opus) — evaluation helpers for the feature study (VALIDATION_SELECT only).

All scoring is done by the audited reused evaluator (snapshot `eval_full.evaluate_model` /
`aggregate` / `per_user_metric` / `paired_bootstrap`): exhaustive K ranking, strict FP32, OOV
targets kept with zero credit, and macro grouping by the distinct `eval_user_key`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import fo_env
import feature_view as fv
import eval_full
import precision

KS = (1, 5, 20)
BOOT_N, BOOT_SEED = 1000, 20260926          # registered in FEATURE_EXPERIMENT_PROTOCOL.md


def build_vs_store():
    from full_store import CompactStore
    return CompactStore.build(fo_env.DATASET_ROOT, splits=("VALIDATION_SELECT",), channels="F4")


def load_cohort(store, cohort_parquet=None):
    import pyarrow.parquet as pq
    ds_vs = store.decisions("VALIDATION_SELECT", "eval")
    ids = pq.read_table(cohort_parquet or fo_env.paths.COHORT_PARQUET, columns=["decision_id"])["decision_id"].to_numpy()
    ids = np.ascontiguousarray(ids).astype(np.int64)
    if ids.shape[0] != fo_env.N_COHORT:
        raise RuntimeError(f"cohort size {ids.shape[0]} != {fo_env.N_COHORT}")
    pos = np.searchsorted(ds_vs.decision_id, ids)
    if not bool((ds_vs.decision_id[np.clip(pos, 0, len(ds_vs) - 1)] == ids).all()):
        raise RuntimeError("cohort decision missing from VALIDATION_SELECT")
    return ds_vs.take(pos)


def metrics(ranks: np.ndarray, ds) -> dict:
    out = {}
    for k in KS:
        a = eval_full.aggregate(ranks, ds, k=k)["end_to_end"]
        out[f"macro_mrr@{k}"], out[f"micro_mrr@{k}"] = a["macro"][f"mrr@{k}"], a["micro"][f"mrr@{k}"]
        out[f"macro_recall@{k}"], out[f"micro_recall@{k}"] = a["macro"][f"recall@{k}"], a["micro"][f"recall@{k}"]
    out["users"] = int(eval_full.aggregate(ranks, ds, k=20)["end_to_end"]["macro"]["user_count"])
    out["n"] = int(len(ranks))
    return out


def evaluate(model, store, ds, device) -> tuple[np.ndarray, dict, float]:
    with precision.strict_eval():
        r = eval_full.evaluate_model(model, store, ds, device, eval_batch=1024, k=20, return_ranks=True)
    ranks = np.asarray(r["ranks"]).astype(np.int64)
    return ranks, metrics(ranks, ds), float(r["_elapsed_s"])


def paired(ranks_a, ranks_b, ds, metric="mrr", k=20) -> dict:
    ka, va = eval_full.per_user_metric(ranks_a, ds, metric, k)
    kb, vb = eval_full.per_user_metric(ranks_b, ds, metric, k)
    return eval_full.paired_bootstrap(ka, va, kb, vb, n=BOOT_N, seed=BOOT_SEED)


def decision_is_weekend(store, ds, chunk: int = 4096) -> np.ndarray:
    """Local is_weekend of each decision's CURRENT event (reporting slice only)."""
    out = np.zeros(len(ds), dtype=bool)
    j = fv.NUM["is_weekend"]
    for s in range(0, len(ds), chunk):
        pos = np.arange(s, min(s + chunk, len(ds)))
        b = store.batch(ds, pos)
        L = torch.as_tensor(b["lengths"]).long()
        num = torch.as_tensor(b["event_numeric_features"])
        out[pos] = (num[torch.arange(len(pos)), L - 1, j] > 0.5).numpy()
    return out


def sliced(ranks_a, ranks_b, ds, mask: np.ndarray) -> dict:
    idx = np.nonzero(mask)[0]
    sub = ds.take(idx)
    ma, mb = metrics(ranks_a[idx], sub), metrics(ranks_b[idx], sub)
    return {"n": int(idx.size), "users": ma["users"], "delta_macro_mrr@20": ma["macro_mrr@20"] - mb["macro_mrr@20"],
            "a": ma, "b": mb}


def save_ranks(path: Path, ranks: np.ndarray, ds) -> str:
    np.savez_compressed(path, ranks=ranks.astype(np.int32), decision_id=np.asarray(ds.decision_id, dtype=np.int64))
    return fo_env.sha256_file(path)


def write_once(path: Path, obj) -> None:
    fo_env.write_json_once(path, obj)
