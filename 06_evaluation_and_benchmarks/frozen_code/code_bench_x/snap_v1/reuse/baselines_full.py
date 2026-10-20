"""POP-full and TRANS-full baselines -- Wave-1 Sonnet C (owner of this file).

Full-scale generalisation of the pilot's `baselines.py` (frozen, imported
nowhere here except `window_cache.CacheReader` for the regression path):

  POP-full:   score_j = count(target_class == j) over the FIT population.
  TRANS-full: first-order transition counts from the fit decision's own-event
              item TOKEN (`product_idx` from `model_ready`, == the last
              context item -- see `decision_item_tokens` in
              `FULL_INTERFACES.md`) to `target_class`, excluding sources
              PAD(0)/OOV(2); composite score
              `trans_count * (max_pop + 1) + pop_count`, exactly as the pilot
              -- unseen sources fall back to popularity automatically (an
              absent trans_matrix row is all-zero).

**Exactness at full scale**: the pilot's `evaluator.evaluate()` casts scores
to float32 internally, which is exact only up to 2**24 (16,777,216). The
pilot's own max_composite_score (6,362,279 over a 1,000,000-row TRAIN
manifest) was comfortably under that limit; at full scale (40,180,608 TRAIN
rows) the composite can exceed it, so scoring and ranking here are int64
throughout -- via `eval_full.rank_targets`, which (unlike
`evaluator.evaluate()`) never coerces its input to float32. This module
never calls `evaluator.evaluate()`.

Three swappable readers per side (fit / eval), per FULL_INTERFACES.md and
the Sonnet C task brief:
  (a) `fit_from_store` / `EvalRows` built from a real `CompactStore` +
      `DecisionSet` (Sonnet A's `full_store.py`) -- for later integration;
      written against the documented interface, not exercised by this
      module's own tests (the store does not exist yet).
  (b) `fit_from_parquet` / `eval_rows_from_parquet` -- a direct, projected
      Parquet reader (this module's main contribution): decisions_index for
      index columns, `model_ready` for `product_idx` (the TRANS token) and
      `log1p_prev_view_count` (the `prior_view_count` diagnostic), joined on
      `decision_id`; `eval_user_key` computed exactly as
      `build_manifests.build_vs_cohort`'s definition, but over the FULL
      split's eval_mask rows (no 10k-user sampling).
  (c) `fit_from_cache` + the pilot `cache/vs` path -- for the regression gate
      against `artifacts/BASELINES.json` (reuses `window_cache.CacheReader`,
      never edits it).

CLI:
  python baselines_full.py regression-gate [--cache-train ...] [--cache-vs ...]
      [--cohort ...] [--baselines-json ...] [--out ...]
  python baselines_full.py full-vs [--out ...] [--eval-batch-pop N]
      [--eval-batch-trans N] [--class-chunk N] [--skip-count-assertions]

Implementation only. Never write inside `/home/jovyan/shared_data/`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

CODE_DIR = Path(__file__).resolve().parent
VENDOR_DIR = CODE_DIR / "vendor"
for _p in (str(CODE_DIR), str(VENDOR_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import torch  # noqa: E402

from rees46_eda.loaders import detect_layout  # noqa: E402
from index_utils import read_index_projected  # noqa: E402
from window_cache import CacheReader  # noqa: E402
from eval_full import aggregate, rank_targets, DEFAULT_K  # noqa: E402

WORKSPACE = Path(
    "/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/gru_v1"
)
DATASET_ROOT = Path("/home/jovyan/shared_data/rees46_nextitem_v1")
VERSION_HASH = "860e5a1c3a81b8e4bbfeff4f245fa37638a43e4f260df179585964d2d4da8dbd"

ITEM_VOCAB_SIZE = 158489
EXPECTED_K = 158486
OOV_TOKEN = 2  # rees46_eda.config.OOV
PAD_TOKEN = 0  # rees46_eda.config.PAD
EVAL_K = DEFAULT_K

# FULL_RUN_PROTOCOL.md sec. 2 / sec. 6 -- gate 3 population counts.
EXPECTED_TRAIN_LOSS_ROWS = 40_180_608
EXPECTED_VS_EVAL_ROWS = 8_826_488
EXPECTED_VS_EVAL_USERS = 790_459


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    pop_count: np.ndarray             # int64 [K]
    trans_matrix: "sp.csr_matrix"     # int64 [ITEM_VOCAB_SIZE, K]
    max_pop: int
    max_trans: int
    max_composite_score: int
    n_fit_rows: int
    n_trans_pairs_fit: int
    n_trans_nonzero_src_tgt: int


@dataclass
class EvalRows:
    """Minimal DecisionSet-compatible container (duck-types against
    `eval_full.aggregate`'s field access: target_class, eval_mask,
    eval_user_key, client_idx, prior_view_count, prefix_position) plus the
    TRANS conditioning token `last_item`. Rows are in ascending decision_id
    (canonical) order."""
    decision_id: np.ndarray
    target_class: np.ndarray
    eval_mask: np.ndarray
    client_idx: np.ndarray
    eval_user_key: np.ndarray
    prior_view_count: np.ndarray
    prefix_position: np.ndarray
    last_item: np.ndarray

    def __len__(self) -> int:
        return int(self.decision_id.shape[0])


# ---------------------------------------------------------------------------
# fit() -- pure numpy/scipy, int64 throughout, no float32 anywhere.
# ---------------------------------------------------------------------------

def fit_from_arrays(target_class: np.ndarray, last_item: np.ndarray, K: int = EXPECTED_K) -> FitResult:
    target_class = np.asarray(target_class, dtype=np.int64)
    last_item = np.asarray(last_item, dtype=np.int64)
    if target_class.shape[0] != last_item.shape[0]:
        raise ValueError("target_class/last_item length mismatch")
    n = int(target_class.shape[0])

    valid_target = target_class >= 0
    pop_count = np.bincount(target_class[valid_target], minlength=K).astype(np.int64)
    if pop_count.shape[0] != K:
        pop_count = pop_count[:K]

    trans_valid = valid_target & (last_item != OOV_TOKEN) & (last_item != PAD_TOKEN)
    src = last_item[trans_valid]
    tgt = target_class[trans_valid]
    data = np.ones(src.shape[0], dtype=np.int64)
    trans_matrix = sp.coo_matrix((data, (src, tgt)), shape=(ITEM_VOCAB_SIZE, K)).tocsr()

    max_pop = int(pop_count.max()) if pop_count.size else 0
    max_trans = int(trans_matrix.data.max()) if trans_matrix.nnz else 0
    max_composite = max_trans * (max_pop + 1) + max_pop

    return FitResult(
        pop_count=pop_count, trans_matrix=trans_matrix, max_pop=max_pop, max_trans=max_trans,
        max_composite_score=int(max_composite), n_fit_rows=n,
        n_trans_pairs_fit=int(trans_valid.sum()), n_trans_nonzero_src_tgt=int(trans_matrix.nnz),
    )


def fit_from_cache(cache_dir: Path, K: int = EXPECTED_K) -> FitResult:
    """Reader (c) -- pilot cache/train, byte-identical to `baselines.py:fit`."""
    reader = CacheReader(cache_dir)
    n = reader.header["row_count"]
    raw = reader.read(np.arange(n))
    item_tokens = raw["item_tokens"].numpy()
    last_valid_index = raw["last_valid_index"].numpy()
    target_class = raw["target_class"].numpy()
    loss_mask = raw["loss_mask"].numpy()
    if not bool(loss_mask.all()):
        raise AssertionError("fit_from_cache: every row of a TRAIN fit cache must be loss_mask==true")
    last_item = item_tokens[np.arange(n), last_valid_index]
    return fit_from_arrays(target_class, last_item, K)


def fit_from_store(store, ds_loss, K: int = EXPECTED_K) -> FitResult:
    """Reader (a) -- a real CompactStore + the TRAIN loss_mask DecisionSet
    (`store.decisions("TRAIN", "loss")`). Written against
    FULL_INTERFACES.md's documented `decision_item_tokens` contract; not
    exercised by this module's own tests because `full_store.py` does not
    exist yet (Sonnet A, in progress) -- for Opus's later integration."""
    loss_mask = np.asarray(ds_loss.loss_mask)
    if not bool(loss_mask.all()):
        raise AssertionError("fit_from_store: ds_loss must be the loss_mask==true DecisionSet")
    last_item = np.asarray(store.decision_item_tokens(ds_loss))
    target_class = np.asarray(ds_loss.target_class)
    return fit_from_arrays(target_class, last_item, K)


def load_model_ready_columns(
    root: Path, columns=("decision_id", "product_idx", "log1p_prev_view_count")
) -> pa.Table:
    """decision_id -> (product_idx, log1p_prev_view_count, ...) for EVERY
    decision in the frozen dataset (every split, every bucket) -- read ONCE
    and reused for both the TRAIN fit and the VS eval read, so
    `model_ready/user_bucket=*/features.parquet` is only opened once per
    `full-vs` invocation."""
    files = sorted(root.glob("model_ready/user_bucket=*/features.parquet"))
    if not files:
        raise FileNotFoundError(f"no model_ready feature files under {root}/model_ready")
    return pq.read_table(files, columns=list(columns))


def fit_from_parquet(
    root: Path, layout: str, K: int = EXPECTED_K, split: str = "TRAIN",
    product_idx_table: Optional[pa.Table] = None, buckets=None,
    expect_n_rows: Optional[int] = None, log=log,
) -> FitResult:
    """Reader (b) -- direct projected-Parquet fit over an entire split's
    loss_mask==true population, via `index_utils.read_index_projected`
    (already streams/concatenates per-bucket, column-projected -- see
    `index_utils.py`), joined against `model_ready`'s `product_idx` (the
    TRANS conditioning token)."""
    t0 = time.time()
    idx = read_index_projected(root, layout, split, ["decision_id", "target_class", "loss_mask"],
                                buckets=buckets)
    idx = idx.filter(idx["loss_mask"])
    log(f"fit_from_parquet: {split} loss_mask rows = {idx.num_rows} (read in {time.time()-t0:.1f}s)")
    if expect_n_rows is not None and idx.num_rows != expect_n_rows:
        raise RuntimeError(f"fit_from_parquet: rows {idx.num_rows} != expected {expect_n_rows}")

    poc = product_idx_table if product_idx_table is not None else load_model_ready_columns(root)
    joined = idx.select(["decision_id", "target_class"]).join(poc, keys="decision_id", join_type="inner")
    if joined.num_rows != idx.num_rows:
        raise RuntimeError(
            f"fit_from_parquet: join dropped rows ({joined.num_rows} != {idx.num_rows}) -- "
            "every loss_mask decision must have a model_ready feature row"
        )

    target_class = joined["target_class"].to_numpy(zero_copy_only=False).astype(np.int64)
    if not bool((target_class >= 0).all()):
        raise AssertionError(
            "fit_from_parquet: loss_mask==true row(s) with target_class < 0 -- "
            "violates the frozen data contract (FULL_RUN_PROTOCOL.md sec. 2)"
        )
    last_item = joined["product_idx"].to_numpy(zero_copy_only=False).astype(np.int64)
    log(f"fit_from_parquet: joined with model_ready product_idx (total {time.time()-t0:.1f}s)")
    return fit_from_arrays(target_class, last_item, K)


# ---------------------------------------------------------------------------
# eval rows -- direct projected-Parquet reader (b).
# ---------------------------------------------------------------------------

def eval_rows_from_parquet(
    root: Path, layout: str, split: str = "VALIDATION_SELECT",
    model_ready_table: Optional[pa.Table] = None, buckets=None,
    expect_n_rows: Optional[int] = None, expect_n_users: Optional[int] = None, log=log,
) -> EvalRows:
    """Reader (b), eval side. `eval_user_key` = dense position of the row's
    user_id in the sorted distinct user_id list over THIS split's eval_mask
    rows -- identical definition to `build_manifests.build_vs_cohort`, just
    without the 10k-user sample (every eval_mask row is kept). Uses
    `pyarrow.compute.index_in` against the sorted distinct set (a single
    vectorised Arrow call) rather than boxing every row's user_id into a
    Python string (which `build_manifests.py`'s docstring measured as
    prohibitively slow over the full ~8-9M row population)."""
    t0 = time.time()
    idx = read_index_projected(
        root, layout, split,
        ["decision_id", "user_id", "client_idx", "target_class", "eval_mask", "position_in_segment"],
        buckets=buckets,
    )
    idx = idx.filter(idx["eval_mask"])
    n = idx.num_rows
    log(f"eval_rows_from_parquet: {split} eval_mask rows = {n} (read in {time.time()-t0:.1f}s)")
    if expect_n_rows is not None and n != expect_n_rows:
        raise RuntimeError(f"eval_rows_from_parquet: rows {n} != expected {expect_n_rows}")

    mr = model_ready_table if model_ready_table is not None else load_model_ready_columns(root)
    joined = idx.join(mr, keys="decision_id", join_type="inner")
    if joined.num_rows != n:
        raise RuntimeError(f"eval_rows_from_parquet: join dropped rows ({joined.num_rows} != {n})")

    order = pc.sort_indices(joined["decision_id"])  # canonical order: ascending decision_id
    joined = joined.take(order)

    distinct_arrow = pc.unique(joined["user_id"])
    distinct_sorted = distinct_arrow.take(pc.sort_indices(distinct_arrow))
    n_distinct = len(distinct_sorted)
    if expect_n_users is not None and n_distinct != expect_n_users:
        raise RuntimeError(f"eval_rows_from_parquet: distinct users {n_distinct} != expected {expect_n_users}")
    key_arr = pc.index_in(joined["user_id"], value_set=distinct_sorted)
    if key_arr.null_count > 0:
        raise RuntimeError("eval_rows_from_parquet: index_in produced null(s) (should be impossible)")
    eval_user_key = key_arr.to_numpy(zero_copy_only=False).astype(np.int64)

    decision_id = joined["decision_id"].to_numpy(zero_copy_only=False).astype(np.int64)
    target_class = joined["target_class"].to_numpy(zero_copy_only=False).astype(np.int64)
    client_idx = joined["client_idx"].to_numpy(zero_copy_only=False).astype(np.int64)
    eval_mask = np.ones(n, dtype=bool)  # already filtered to eval_mask == true
    prefix_position = joined["position_in_segment"].to_numpy(zero_copy_only=False).astype(np.int64)
    last_item = joined["product_idx"].to_numpy(zero_copy_only=False).astype(np.int64)

    raw_log1p_prev_view = joined["log1p_prev_view_count"].to_numpy(zero_copy_only=False).astype(np.float64)
    prior_view_count = np.rint(np.expm1(np.nan_to_num(raw_log1p_prev_view, nan=0.0)))
    prior_view_count = np.clip(prior_view_count, 0, None).astype(np.int64)

    log(f"eval_rows_from_parquet: done in {time.time()-t0:.1f}s n_distinct_users={n_distinct}")
    return EvalRows(
        decision_id=decision_id, target_class=target_class, eval_mask=eval_mask,
        client_idx=client_idx, eval_user_key=eval_user_key, prior_view_count=prior_view_count,
        prefix_position=prefix_position, last_item=last_item,
    )


def eval_rows_from_store(store, ds_eval) -> tuple:
    """Reader (a), eval side -- returns (ds_eval, last_item) for later
    integration; `ds_eval` itself already satisfies `aggregate`'s field
    access, so callers can pass it straight to `score_and_rank_*`."""
    last_item = np.asarray(store.decision_item_tokens(ds_eval))
    return ds_eval, last_item


# ---------------------------------------------------------------------------
# GPU scoring + ranking.
# ---------------------------------------------------------------------------

def score_and_rank_pop(fit: FitResult, target_class: np.ndarray, need_mask: np.ndarray, device,
                        k: int = EVAL_K, eval_batch: int = 16384,
                        class_chunk: int = 32768) -> np.ndarray:
    """POP score is the SAME row (global popularity) for every decision --
    `pop_gpu.expand(b, K)` keeps the BASE tensor a zero-copy broadcast view
    (no [B, K] copy at rest), but each `rank_targets` K-chunk still allocates
    a real [B, class_chunk] comparison-result buffer (unavoidable: the
    output of `>`/`==` cannot itself be a broadcast view). Choose `eval_batch`
    with `eval_batch * class_chunk` in mind -- e.g. 16384 * 32768 int64-sized
    intermediates is safe on a ~15 GB card; 65536 * 32768 is NOT (measured:
    OOM, since several such [B, class_chunk] buffers are live at once inside
    one `rank_targets` chunk iteration)."""
    K = fit.pop_count.shape[0]
    pop_gpu = torch.as_tensor(fit.pop_count, dtype=torch.int64, device=device)
    n = int(target_class.shape[0])
    ranks = np.full(n, -1, dtype=np.int64)
    target_t_all = torch.as_tensor(np.asarray(target_class), dtype=torch.long)
    need_idx_all = np.nonzero(np.asarray(need_mask))[0]

    for start in range(0, need_idx_all.shape[0], eval_batch):
        end = min(start + eval_batch, need_idx_all.shape[0])
        rows = need_idx_all[start:end]
        b = int(rows.shape[0])
        scores = pop_gpu.unsqueeze(0).expand(b, K)
        tgt = target_t_all[rows].to(device)
        r = rank_targets(scores, tgt, class_chunk=class_chunk)
        ranks[rows] = r.to("cpu").numpy()
    return ranks


def score_and_rank_trans(fit: FitResult, last_item: np.ndarray, target_class: np.ndarray,
                          need_mask: np.ndarray, device, k: int = EVAL_K,
                          eval_batch: int = 2048, class_chunk: int = 32768) -> np.ndarray:
    """Composite score `trans_count * (max_pop + 1) + pop_count`, computed
    entirely as int64: a GPU-resident dense `pop_count` broadcast (cloned,
    per chunk) is corrected in place by a sparse scatter-add of that chunk's
    CSR transition entries scaled by `(max_pop + 1)` -- never a CPU dense
    `.toarray()` over `[B, K]` (measured too slow at full scale; see the
    module docstring)."""
    K = fit.pop_count.shape[0]
    pop_gpu = torch.as_tensor(fit.pop_count, dtype=torch.int64, device=device)
    trans = fit.trans_matrix
    scale = fit.max_pop + 1
    n = int(target_class.shape[0])
    ranks = np.full(n, -1, dtype=np.int64)
    target_t_all = torch.as_tensor(np.asarray(target_class), dtype=torch.long)
    last_item = np.asarray(last_item, dtype=np.int64)
    need_idx_all = np.nonzero(np.asarray(need_mask))[0]

    for start in range(0, need_idx_all.shape[0], eval_batch):
        end = min(start + eval_batch, need_idx_all.shape[0])
        rows = need_idx_all[start:end]
        b = int(rows.shape[0])
        src = last_item[rows]
        sub_coo = trans[src].tocoo()  # CSR row-fancy-index (dup rows OK), then COO -- nnz-bound, not B*K
        dense = pop_gpu.unsqueeze(0).expand(b, K).clone()  # [b, K] int64, writable
        if sub_coo.nnz:
            row_idx = torch.as_tensor(sub_coo.row, dtype=torch.long, device=device)
            col_idx = torch.as_tensor(sub_coo.col, dtype=torch.long, device=device)
            vals = torch.as_tensor(sub_coo.data.astype(np.int64) * scale, dtype=torch.long, device=device)
            dense.index_put_((row_idx, col_idx), vals, accumulate=True)
        tgt = target_t_all[rows].to(device)
        r = rank_targets(dense, tgt, class_chunk=class_chunk)
        ranks[rows] = r.to("cpu").numpy()
    return ranks


def run_pop_baseline(fit: FitResult, eval_rows, device, k: int = EVAL_K,
                      eval_batch: int = 16384, class_chunk: int = 32768,
                      return_ranks: bool = False) -> dict:
    target_class = np.asarray(eval_rows.target_class)
    eval_mask = np.asarray(eval_rows.eval_mask).astype(bool)
    need = eval_mask & (target_class >= 0)
    t0 = time.time()
    ranks = score_and_rank_pop(fit, target_class, need, device, k=k, eval_batch=eval_batch,
                                class_chunk=class_chunk)
    elapsed = time.time() - t0
    result = aggregate(ranks, eval_rows, k=k)
    result["_elapsed_s"] = elapsed
    result["_rows_per_s"] = (len(eval_rows) / elapsed) if elapsed > 0 else None
    if return_ranks:
        result["ranks"] = ranks
    return result


def run_trans_baseline(fit: FitResult, eval_rows, device, k: int = EVAL_K,
                        eval_batch: int = 2048, class_chunk: int = 32768,
                        return_ranks: bool = False, last_item: "np.ndarray | None" = None) -> dict:
    """`last_item` is optional: `EvalRows` (readers b/c) already carries it as
    a field, so it is read from `eval_rows.last_item` by default. A real
    `DecisionSet` (reader a) has no such field -- its last-item token comes
    from `store.decision_item_tokens(ds)` instead, so callers using a real
    store must pass it explicitly here (see `eval_rows_from_store`)."""
    target_class = np.asarray(eval_rows.target_class)
    eval_mask = np.asarray(eval_rows.eval_mask).astype(bool)
    need = eval_mask & (target_class >= 0)
    li = last_item if last_item is not None else eval_rows.last_item
    t0 = time.time()
    ranks = score_and_rank_trans(fit, li, target_class, need, device, k=k,
                                  eval_batch=eval_batch, class_chunk=class_chunk)
    elapsed = time.time() - t0
    result = aggregate(ranks, eval_rows, k=k)
    result["_elapsed_s"] = elapsed
    result["_rows_per_s"] = (len(eval_rows) / elapsed) if elapsed > 0 else None
    if return_ranks:
        result["ranks"] = ranks
    return result


# ---------------------------------------------------------------------------
# Regression gate (protocol sec. 10.10 / gate iii): reader (c), pilot TRAIN
# manifest fit -> pilot 10k VS cohort eval, must reproduce BASELINES.json.
# ---------------------------------------------------------------------------

def regression_gate(cache_train_dir: Path, cache_vs_dir: Path, cohort_parquet: Path,
                     baselines_json: Path, device, k: int = EVAL_K, tol: float = 1e-6) -> dict:
    K = EXPECTED_K
    fit = fit_from_cache(cache_train_dir, K)

    reader = CacheReader(cache_vs_dir)
    n = reader.header["row_count"]
    raw = reader.read(np.arange(n))
    item_tokens = raw["item_tokens"].numpy()
    last_valid_index = raw["last_valid_index"].numpy()
    last_item = item_tokens[np.arange(n), last_valid_index]

    cohort = pq.read_table(cohort_parquet, columns=["eval_user_key"])
    eval_user_key = np.asarray(cohort["eval_user_key"].to_numpy(zero_copy_only=False), dtype=np.int64)
    if eval_user_key.shape[0] != n:
        raise ValueError("regression_gate: cohort parquet row count != cache/vs row count")

    eval_rows = EvalRows(
        decision_id=raw["decision_id"].numpy(),
        target_class=raw["target_class"].numpy(),
        eval_mask=raw["eval_mask"].numpy().astype(bool),
        client_idx=raw["client_id"].numpy(),
        eval_user_key=eval_user_key,
        prior_view_count=np.zeros(n, dtype=np.int64),   # not exercised by this gate (reporting-only)
        prefix_position=np.zeros(n, dtype=np.int64),    # not exercised by this gate (reporting-only)
        last_item=last_item,
    )

    pop_result = run_pop_baseline(fit, eval_rows, device, k=k, eval_batch=16384)
    trans_result = run_trans_baseline(fit, eval_rows, device, k=k, eval_batch=2048)

    expected = json.loads(Path(baselines_json).read_text(encoding="utf-8"))
    diffs: dict = {}

    def _check(name: str, got, exp) -> None:
        d = abs(float(got) - float(exp))
        diffs[name] = d
        if d > tol:
            raise AssertionError(f"{name}: got {got} expected {exp} |diff|={d} > tol {tol}")

    for name, got_block, exp_block in (("POP", pop_result, expected["POP"]),
                                        ("TRANS", trans_result, expected["TRANS"])):
        for pop_key in ("end_to_end", "closed"):
            _check(f"{name}.{pop_key}.n", got_block[pop_key]["n"], exp_block[pop_key]["n"])
            for scope in ("micro", "macro"):
                for metric in ("recall@20", "mrr@20", "ndcg@20"):
                    _check(f"{name}.{pop_key}.{scope}.{metric}",
                           got_block[pop_key][scope][metric], exp_block[pop_key][scope][metric])
            _check(f"{name}.{pop_key}.macro.user_count",
                   got_block[pop_key]["macro"]["user_count"], exp_block[pop_key]["macro"]["user_count"])

    return {
        "pass": True, "max_abs_diff": max(diffs.values()) if diffs else 0.0, "diffs": diffs,
        "pop_result": pop_result, "trans_result": trans_result,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _configure_threads() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "3")
    os.environ.setdefault("MKL_NUM_THREADS", "3")
    torch.set_num_threads(3)
    try:
        pa.set_cpu_count(3)
        pa.set_io_thread_count(3)
    except Exception:  # pragma: no cover -- defensive only
        pass


def main() -> int:
    _configure_threads()

    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)

    g = sub.add_parser("regression-gate")
    g.add_argument("--cache-train", default=str(WORKSPACE / "cache" / "train"))
    g.add_argument("--cache-vs", default=str(WORKSPACE / "cache" / "vs"))
    g.add_argument("--cohort", default=str(WORKSPACE / "artifacts" / "validation_select_cohort.parquet"))
    g.add_argument("--baselines-json", default=str(WORKSPACE / "artifacts" / "BASELINES.json"))
    g.add_argument("--out", default=str(WORKSPACE / "tests_full" / "C" / "baselines_full_regression.json"))

    f = sub.add_parser("full-vs")
    f.add_argument("--out", default=str(WORKSPACE / "tests_full" / "C" / "baselines_full_vs.json"))
    f.add_argument("--eval-batch-pop", type=int, default=16384)
    f.add_argument("--eval-batch-trans", type=int, default=2048)
    f.add_argument("--class-chunk", type=int, default=32768)
    f.add_argument("--skip-count-assertions", action="store_true")

    args = ap.parse_args()

    layout = detect_layout(DATASET_ROOT)
    if layout != "frozen":
        raise RuntimeError(f"expected frozen layout, got {layout!r}")
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    log(f"device={device} cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    if args.mode == "regression-gate":
        result = regression_gate(Path(args.cache_train), Path(args.cache_vs), Path(args.cohort),
                                  Path(args.baselines_json), device)
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        slim = {"pass": result["pass"], "max_abs_diff": result["max_abs_diff"], "diffs": result["diffs"]}
        out_path.write_text(json.dumps(slim, indent=2, default=str), encoding="utf-8")
        log(f"REGRESSION GATE PASS max_abs_diff={result['max_abs_diff']:.3e} wrote {out_path}")
        return 0

    if args.mode == "full-vs":
        t_start = time.time()
        expect_n = None if args.skip_count_assertions else EXPECTED_TRAIN_LOSS_ROWS
        expect_vs_n = None if args.skip_count_assertions else EXPECTED_VS_EVAL_ROWS
        expect_vs_users = None if args.skip_count_assertions else EXPECTED_VS_EVAL_USERS

        mr_table = load_model_ready_columns(DATASET_ROOT)
        log(f"model_ready product_idx table loaded: {mr_table.num_rows} rows")

        fit = fit_from_parquet(DATASET_ROOT, layout, EXPECTED_K, split="TRAIN",
                                product_idx_table=mr_table, expect_n_rows=expect_n, log=log)
        log(f"fit: n_fit_rows={fit.n_fit_rows} max_pop={fit.max_pop} max_trans={fit.max_trans} "
            f"max_composite_score={fit.max_composite_score} "
            f"n_trans_pairs_fit={fit.n_trans_pairs_fit} nnz={fit.n_trans_nonzero_src_tgt}")

        eval_rows = eval_rows_from_parquet(DATASET_ROOT, layout, split="VALIDATION_SELECT",
                                            model_ready_table=mr_table, expect_n_rows=expect_vs_n,
                                            expect_n_users=expect_vs_users, log=log)
        log(f"eval rows: n={len(eval_rows)}")

        pop_result = run_pop_baseline(fit, eval_rows, device, k=EVAL_K,
                                       eval_batch=args.eval_batch_pop, class_chunk=args.class_chunk)
        log(f"POP-full: macro_mrr@20={pop_result['end_to_end']['macro']['mrr@20']:.6f} "
            f"elapsed={pop_result['_elapsed_s']:.1f}s rows/s={pop_result['_rows_per_s']:.0f}")

        trans_result = run_trans_baseline(fit, eval_rows, device, k=EVAL_K,
                                           eval_batch=args.eval_batch_trans, class_chunk=args.class_chunk)
        log(f"TRANS-full: macro_mrr@20={trans_result['end_to_end']['macro']['mrr@20']:.6f} "
            f"elapsed={trans_result['_elapsed_s']:.1f}s rows/s={trans_result['_rows_per_s']:.0f}")

        output = {
            "version_hash": VERSION_HASH,
            "K": EXPECTED_K,
            "n_fit_rows": fit.n_fit_rows,
            "n_eval_rows": len(eval_rows),
            "max_pop": fit.max_pop,
            "max_trans": fit.max_trans,
            "max_composite_score": fit.max_composite_score,
            "n_trans_pairs_fit": fit.n_trans_pairs_fit,
            "n_trans_nonzero_src_tgt": fit.n_trans_nonzero_src_tgt,
            "eval_k": EVAL_K,
            "POP": pop_result,
            "TRANS": trans_result,
            "total_elapsed_s": time.time() - t_start,
        }
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
        log(f"wrote {out_path}; total_elapsed_s={output['total_elapsed_s']:.1f}")
        return 0

    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        log("=== FATAL ERROR ===")
        traceback.print_exc()
        raise
