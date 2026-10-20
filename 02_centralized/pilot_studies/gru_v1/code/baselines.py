"""POP and TRANS baselines -- Task T-L.

Both fitted ONLY on the 1,000,000-row TRAIN pilot manifest (cache/train --
never validation), evaluated through the SAME `evaluator.evaluate` and the
SAME tie rule as the GRU (ascending class id wins ties -- enforced simply by
reusing `evaluator.evaluate` unmodified):

  POP:   score_j = count(target_class == j) over the manifest.
  TRANS: first-order transition from the last context item TOKEN to the next
         class; popularity is both the tie-breaker and the fallback when the
         last item was unseen (never appeared as a fit-time transition
         source) or is OOV (token id 2). Score is the int64 composite
         `trans_count * (max_pop + 1) + pop_count`, so trans-tier ordering
         strictly dominates pop-tier ordering and ties resolve by ascending
         class id in the evaluator.

Emits artifacts/BASELINES.json with Micro/Macro Recall/MRR/NDCG@20 on both
denominators (end-to-end, closed) plus user_count, for POP and TRANS.

Implementation only. Run on Lobot only (reads cache/train, cache/vs, and
the frozen dataset's catalogue.parquet for K); does not need a GPU.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

WORKSPACE = Path(
    "/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/gru_v1"
)
CODE_DIR = WORKSPACE / "code"
VENDOR_DIR = CODE_DIR / "vendor"
ARTIFACTS = WORKSPACE / "artifacts"
CACHE_TRAIN = WORKSPACE / "cache" / "train"
CACHE_VS = WORKSPACE / "cache" / "vs"
DATASET_ROOT = Path("/home/jovyan/shared_data/rees46_nextitem_v1")

for _p in (str(CODE_DIR), str(VENDOR_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import torch  # noqa: E402

from evaluator import evaluate  # noqa: E402
from window_cache import CacheReader  # noqa: E402

VERSION_HASH = "860e5a1c3a81b8e4bbfeff4f245fa37638a43e4f260df179585964d2d4da8dbd"
ITEM_VOCAB_SIZE = 158489
EXPECTED_K = 158486
OOV_TOKEN = 2  # rees46_eda.config.OOV
PAD_TOKEN = 0  # rees46_eda.config.PAD
EVAL_K = 20
EVAL_CHUNK = 512
FLOAT32_EXACT_INT_LIMIT = 1 << 24  # 16,777,216 -- evaluator casts scores to float32


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_catalogue_k() -> int:
    import pyarrow.parquet as pq
    from rees46_eda.loaders import detect_layout

    layout = detect_layout(DATASET_ROOT)
    if layout != "frozen":
        raise RuntimeError(f"expected frozen layout, got {layout!r}")
    cat = pq.read_table(DATASET_ROOT / "mappings" / "catalogue.parquet")
    k = cat.num_rows
    if k != EXPECTED_K:
        raise RuntimeError(f"K mismatch: catalogue.parquet has {k} rows, expected {EXPECTED_K}")
    return k


def load_manifest_meta() -> dict:
    return json.loads((ARTIFACTS / "PILOT_MANIFEST.json").read_text(encoding="utf-8"))


def last_context_item(item_tokens: np.ndarray, last_valid_index: np.ndarray) -> np.ndarray:
    """The last context item TOKEN (not class) for every row: item_tokens[i,
    last_valid_index[i]]."""
    n = item_tokens.shape[0]
    return item_tokens[np.arange(n), last_valid_index]


# ---------------------------------------------------------------------------
# Fit (TRAIN pilot manifest ONLY)
# ---------------------------------------------------------------------------

def fit(cache_train_dir: Path, K: int) -> dict:
    reader = CacheReader(cache_train_dir)
    n = reader.header["row_count"]
    raw = reader.read(np.arange(n))
    item_tokens = raw["item_tokens"].numpy()
    last_valid_index = raw["last_valid_index"].numpy()
    target_class = raw["target_class"].numpy()
    loss_mask = raw["loss_mask"].numpy()
    if not bool(loss_mask.all()):
        raise AssertionError(
            "TRAIN pilot manifest row with loss_mask==False -- manifest is supposed to be "
            "pre-filtered to loss_mask==true rows; this is a pipeline bug upstream"
        )

    last_item = last_context_item(item_tokens, last_valid_index)

    valid_target = target_class >= 0
    pop_count = np.bincount(target_class[valid_target], minlength=K).astype(np.int64)
    if pop_count.shape[0] != K:
        raise RuntimeError(f"pop_count length {pop_count.shape[0]} != K {K}")

    # TRANS fit: exclude rows whose last item is OOV(2) or PAD(0) (defensive
    # -- PAD shouldn't occur given lengths>=1, but excluding it costs
    # nothing) and rows with no valid target. Excluded source items are
    # simply never populated as a trans_matrix row, so at eval time they
    # fall back to pure popularity automatically (an absent/all-zero row
    # makes the composite score collapse to `pop_count` alone) -- this IS
    # the "fallback when the last item was unseen or is OOV(2)" rule, not a
    # separate code path.
    trans_valid = valid_target & (last_item != OOV_TOKEN) & (last_item != PAD_TOKEN)
    src = last_item[trans_valid].astype(np.int64)
    tgt = target_class[trans_valid].astype(np.int64)
    data = np.ones(src.shape[0], dtype=np.int64)
    # coo_matrix(...).tocsr() sums duplicate (src, tgt) entries automatically.
    trans_matrix = sp.coo_matrix((data, (src, tgt)), shape=(ITEM_VOCAB_SIZE, K)).tocsr()

    max_pop = int(pop_count.max()) if pop_count.size else 0
    max_trans = int(trans_matrix.data.max()) if trans_matrix.nnz else 0
    max_composite_score = max_trans * (max_pop + 1) + max_pop

    return {
        "pop_count": pop_count,
        "trans_matrix": trans_matrix,
        "max_pop": max_pop,
        "max_trans": max_trans,
        "max_composite_score": int(max_composite_score),
        "n_fit_rows": int(n),
        "n_trans_pairs_fit": int(trans_valid.sum()),
        "n_trans_nonzero_src_tgt": int(trans_matrix.nnz),
    }


# ---------------------------------------------------------------------------
# VS cohort rows + scoring generators
# ---------------------------------------------------------------------------

def eval_user_key_array() -> np.ndarray:
    import pyarrow.parquet as pq
    t = pq.read_table(ARTIFACTS / "validation_select_cohort.parquet", columns=["eval_user_key"])
    return np.array(t["eval_user_key"].to_numpy(zero_copy_only=False), copy=True)


def load_vs_rows() -> tuple[dict, np.ndarray, int]:
    reader = CacheReader(CACHE_VS)
    n = reader.header["row_count"]
    raw = reader.read(np.arange(n))
    eval_user_key = eval_user_key_array()
    if eval_user_key.shape[0] != n:
        raise ValueError(
            "validation_select_cohort.parquet row count != cache/vs row count "
            "(cache row order must equal manifest row order)"
        )
    item_tokens = raw["item_tokens"].numpy()
    last_valid_index = raw["last_valid_index"].numpy()
    last_item = last_context_item(item_tokens, last_valid_index)
    rows = {
        "target_class": raw["target_class"],
        "eval_mask": raw["eval_mask"],
        "eval_user_key": torch.from_numpy(eval_user_key),
        "client_id": raw["client_id"],
    }
    return rows, last_item, n


def pop_scores_fn(pop_count: np.ndarray, n: int, chunk_size: int = EVAL_CHUNK):
    """Same score row (global popularity) for every decision -- yielded as a
    zero-copy expanded view, never materialised as an [n, K] array."""
    K = pop_count.shape[0]
    base = torch.from_numpy(pop_count.astype(np.float32))

    def gen():
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            yield base.unsqueeze(0).expand(end - start, K)

    return gen


def trans_scores_fn(fit_result: dict, last_item: np.ndarray, n: int, chunk_size: int = EVAL_CHUNK):
    pop_count = fit_result["pop_count"]
    trans_matrix = fit_result["trans_matrix"]
    max_pop = fit_result["max_pop"]
    K = pop_count.shape[0]
    pop_i64 = pop_count.astype(np.int64)

    def gen():
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            src = last_item[start:end]
            trans_dense = np.asarray(trans_matrix[src].todense(), dtype=np.int64)  # [b, K]
            composite = trans_dense * (max_pop + 1) + pop_i64[None, :]
            yield torch.from_numpy(composite.astype(np.float32))

    return gen


def run_baseline(name: str, scores_fn, rows: dict) -> dict:
    t0 = time.time()
    result = evaluate(scores_fn, rows, k=EVAL_K, chunk_size=8192)
    result["_elapsed_s"] = time.time() - t0
    log(f"{name}: macro_mrr@20(end_to_end)={result['end_to_end']['macro']['mrr@20']:.6f} "
        f"micro_recall@20(end_to_end)={result['end_to_end']['micro']['recall@20']:.6f} "
        f"user_count={result['user_count']} elapsed={result['_elapsed_s']:.1f}s")
    return result


def main() -> int:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    log("=== Task T-L baselines starting ===")

    manifest_meta = load_manifest_meta()
    if manifest_meta["version_hash"] != VERSION_HASH:
        raise RuntimeError(
            f"PILOT_MANIFEST.json version_hash {manifest_meta['version_hash']!r} != "
            f"expected {VERSION_HASH!r}"
        )

    K = load_catalogue_k()
    log(f"K={K}")

    fit_result = fit(CACHE_TRAIN, K)
    log(f"fit: n_fit_rows={fit_result['n_fit_rows']} max_pop={fit_result['max_pop']} "
        f"max_trans={fit_result['max_trans']} max_composite_score={fit_result['max_composite_score']} "
        f"n_trans_pairs_fit={fit_result['n_trans_pairs_fit']} "
        f"n_trans_nonzero_src_tgt={fit_result['n_trans_nonzero_src_tgt']}")
    composite_float32_exact = fit_result["max_composite_score"] <= FLOAT32_EXACT_INT_LIMIT
    log(f"composite score float32-exact (<= 2^24 = {FLOAT32_EXACT_INT_LIMIT}): "
        f"{composite_float32_exact}")

    rows, last_item, n_vs = load_vs_rows()
    log(f"VS cohort rows: {n_vs}")

    pop_result = run_baseline("POP", pop_scores_fn(fit_result["pop_count"], n_vs), rows)
    trans_result = run_baseline("TRANS", trans_scores_fn(fit_result, last_item, n_vs), rows)

    output = {
        "task": "T-L",
        "version_hash": VERSION_HASH,
        "train_manifest_sha256": manifest_meta["train_manifest"]["sha256"],
        "validation_select_cohort_sha256": manifest_meta["validation_select_cohort"]["sha256"],
        "K": K,
        "n_fit_rows": fit_result["n_fit_rows"],
        "n_vs_rows": n_vs,
        "max_pop": fit_result["max_pop"],
        "max_trans": fit_result["max_trans"],
        "max_composite_score": fit_result["max_composite_score"],
        "composite_score_float32_exact": composite_float32_exact,
        "n_trans_pairs_fit": fit_result["n_trans_pairs_fit"],
        "n_trans_nonzero_src_tgt": fit_result["n_trans_nonzero_src_tgt"],
        "eval_k": EVAL_K,
        "POP": pop_result,
        "TRANS": trans_result,
    }
    out_path = ARTIFACTS / "BASELINES.json"
    out_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    log(f"wrote {out_path}")
    log("=== Task T-L baselines DONE ===")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        log("=== FATAL ERROR ===")
        traceback.print_exc()
        raise
