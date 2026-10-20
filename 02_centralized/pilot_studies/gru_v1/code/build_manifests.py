"""Pilot manifest construction (T-G).

Produces, from the frozen decision index (read via ``index_utils`` column
projection only -- never ``DecisionWindowLoader.index()``):

1. TRAIN pilot manifest: `loss_mask == true` rows only, sorted by
   `decision_id`, then `numpy.random.default_rng(2026).choice(n, 1_000_000,
   replace=False)`. Persists `decision_id, user_bucket` as parquet.

2. VALIDATION_SELECT cohort: over `eval_mask == true` rows, the sorted
   distinct `user_id` list; 10,000 users sampled with
   `default_rng(2026).choice(n_distinct, 10_000, replace=False)` (the chosen
   positions in the sorted distinct list ARE `eval_user_key` by
   construction); ALL eval_mask decisions of those users. Persists
   `decision_id, user_bucket, eval_user_key, client_is_oov, target_oov`.
   `user_id` is used only in memory to build the cohort and is NEVER
   persisted.

3. `PILOT_MANIFEST.json`: sha256 of both parquet files, dataset
   `version_hash`, measured counts, and the RNG seed.

Data-only. No model code. Intended to run on LOBOT against the full frozen
dataset (TRAIN alone is 52.76M raw rows).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
VENDOR_DIR = CODE_DIR / "vendor"
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from rees46_eda.loaders import detect_layout  # noqa: E402
from rees46_eda.config import OOV  # noqa: E402
from index_utils import read_index_projected  # noqa: E402

SEED = 2026
TRAIN_MANIFEST_N = 1_000_000
COHORT_N_USERS = 10_000


def peak_rss_mb() -> float | None:
    try:
        import resource
        # ru_maxrss is KB on Linux, bytes on macOS; Lobot is Linux.
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_train_manifest(root: Path, layout: str) -> tuple[np.ndarray, np.ndarray, int]:
    t = read_index_projected(root, layout, "TRAIN", ["decision_id", "loss_mask"], buckets=None)
    raw_n = t.num_rows
    decision_id = t["decision_id"].to_numpy(zero_copy_only=False)
    user_bucket = t["user_bucket"].to_numpy(zero_copy_only=False)
    loss_mask = t["loss_mask"].to_numpy(zero_copy_only=False)
    del t

    keep = loss_mask
    dec_f = decision_id[keep]
    buck_f = user_bucket[keep]
    del decision_id, user_bucket, loss_mask, keep

    order = np.argsort(dec_f, kind="stable")
    dec_sorted = dec_f[order]
    buck_sorted = buck_f[order]
    del dec_f, buck_f, order

    n = dec_sorted.shape[0]
    rng = np.random.default_rng(SEED)
    idx = rng.choice(n, TRAIN_MANIFEST_N, replace=False)
    idx.sort()  # deterministic output row order; does not change WHICH rows were chosen

    sel_dec = dec_sorted[idx]
    sel_buck = buck_sorted[idx]
    return sel_dec, sel_buck, raw_n


def build_vs_cohort(root: Path, layout: str):
    """Uses pyarrow.compute (hash-based `unique`/`is_in`, C++, no per-row
    Python object boxing) for every operation over the FULL eval_mask==true
    population (~8-9M rows). `numpy.unique`/`numpy.isin` over a numpy object
    array of that many boxed Python strings was measured to take >15 minutes
    (sort-based dedup over the whole population, not just the distinct
    output) and was abandoned in favour of this. Only the small distinct
    list (~n_distinct, sorted once) and the small cohort output are ever
    converted to numpy object arrays.
    """
    t = read_index_projected(
        root, layout, "VALIDATION_SELECT",
        ["decision_id", "user_id", "client_idx", "target_oov", "eval_mask"],
        buckets=None,
    )
    t_ev = t.filter(t["eval_mask"])
    del t

    uid_col = t_ev["user_id"]  # Arrow ChunkedArray, still columnar

    # Hash-based distinct (Arrow, C++): O(n) over ~8-9M rows, not a sort.
    distinct_arrow = pc.unique(uid_col)
    sort_idx = pc.sort_indices(distinct_arrow)
    distinct_sorted_arrow = distinct_arrow.take(sort_idx)
    # Only the DISTINCT users (far fewer than the full eval population) are
    # boxed into Python strings here.
    distinct_users = distinct_sorted_arrow.to_numpy(zero_copy_only=False)
    n_distinct = distinct_users.shape[0]

    rng = np.random.default_rng(SEED)
    # positions chosen in the sorted distinct list ARE eval_user_key by
    # construction: the position sampled here is exactly the dense position
    # of that user in `distinct_users`.
    sampled_positions = rng.choice(n_distinct, COHORT_N_USERS, replace=False)
    sampled_users = distinct_users[sampled_positions]

    # Hash-based membership test (Arrow, C++) over the full eval population.
    sampled_users_arrow = pa.array(sampled_users, type=pa.string())
    keep_mask = pc.is_in(uid_col, value_set=sampled_users_arrow)
    t_cohort = t_ev.filter(keep_mask)
    del t_ev, keep_mask, uid_col

    dec_c = t_cohort["decision_id"].to_numpy(zero_copy_only=False)
    buck_c = t_cohort["user_bucket"].to_numpy(zero_copy_only=False)
    uid_c = t_cohort["user_id"].to_numpy(zero_copy_only=False)  # small: only the cohort's rows
    cli_c = t_cohort["client_idx"].to_numpy(zero_copy_only=False)
    oov_c = t_cohort["target_oov"].to_numpy(zero_copy_only=False)
    del t_cohort

    # dense position in the FULL sorted distinct list (not renumbered 0..9999)
    eval_user_key = np.searchsorted(distinct_users, uid_c).astype(np.int64)
    # sanity: every looked-up position must actually match (uid is a member)
    assert np.array_equal(distinct_users[eval_user_key], uid_c)

    client_is_oov = (cli_c == OOV)

    return {
        "decision_id": dec_c,
        "user_bucket": buck_c,
        "eval_user_key": eval_user_key,
        "client_is_oov": client_is_oov,
        "target_oov": oov_c,
        "n_distinct_eval_users": n_distinct,
        "n_sampled_users": sampled_users.shape[0],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True, help="artifacts directory")
    ap.add_argument("--version-hash", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    layout = detect_layout(root)

    t0 = time.time()
    sel_dec, sel_buck, raw_train_n = build_train_manifest(root, layout)
    t1 = time.time()
    print(f"TRAIN manifest: raw_rows={raw_train_n} filtered/sorted, sampled={sel_dec.shape[0]} "
          f"in {t1 - t0:.1f}s peak_rss_mb={peak_rss_mb()}")

    train_table = pa.table({
        "decision_id": pa.array(sel_dec, type=pa.int64()),
        "user_bucket": pa.array(sel_buck, type=pa.int64()),
    })
    train_path = out_dir / "train_pilot_manifest.parquet"
    pq.write_table(train_table, train_path)

    t2 = time.time()
    cohort = build_vs_cohort(root, layout)
    t3 = time.time()
    print(f"VS cohort: distinct_eval_users={cohort['n_distinct_eval_users']} "
          f"sampled_users={cohort['n_sampled_users']} cohort_rows={cohort['decision_id'].shape[0]} "
          f"in {t3 - t2:.1f}s peak_rss_mb={peak_rss_mb()}")

    vs_table = pa.table({
        "decision_id": pa.array(cohort["decision_id"], type=pa.int64()),
        "user_bucket": pa.array(cohort["user_bucket"], type=pa.int64()),
        "eval_user_key": pa.array(cohort["eval_user_key"], type=pa.int64()),
        "client_is_oov": pa.array(cohort["client_is_oov"], type=pa.bool_()),
        "target_oov": pa.array(cohort["target_oov"], type=pa.bool_()),
    })
    assert "user_id" not in vs_table.column_names
    vs_path = out_dir / "validation_select_cohort.parquet"
    pq.write_table(vs_table, vs_path)

    cohort_n = int(cohort["decision_id"].shape[0])
    distinct_users_in_cohort = int(np.unique(cohort["eval_user_key"]).shape[0])
    oov_target_rows = int(cohort["target_oov"].sum())
    unseen_user_rows = int(cohort["client_is_oov"].sum())
    history_available_rows = cohort_n - unseen_user_rows

    manifest = {
        "seed": SEED,
        "dataset_root": str(root),
        "layout": layout,
        "version_hash": args.version_hash,
        "train_manifest": {
            "path": str(train_path),
            "sha256": sha256_file(train_path),
            "raw_train_index_rows": int(raw_train_n),
            "rows": int(sel_dec.shape[0]),
        },
        "validation_select_cohort": {
            "path": str(vs_path),
            "sha256": sha256_file(vs_path),
            "cohort_decisions": cohort_n,
            "distinct_eval_users_total": int(cohort["n_distinct_eval_users"]),
            "sampled_users": int(cohort["n_sampled_users"]),
            "distinct_users_in_cohort": distinct_users_in_cohort,
            "oov_target_rows": oov_target_rows,
            "unseen_user_rows_client_idx_2": unseen_user_rows,
            "history_available_rows": history_available_rows,
        },
        "peak_rss_mb_observed": peak_rss_mb(),
        "elapsed_s": {"train_manifest": t1 - t0, "vs_cohort": t3 - t2},
    }
    manifest_path = out_dir / "PILOT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
