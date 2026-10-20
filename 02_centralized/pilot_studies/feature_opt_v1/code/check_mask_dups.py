"""check_mask_dups.py (owner: Opus): bounded SAMPLE check (3 buckets, TRAIN + VALIDATION_SELECT only).

It checks exact duplicates among flags and masks, and the valid-only statistics of the two user time
features against their fitted normalizer parameters. Read-only. Uses one column projection per bucket.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
import json, sys
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
pa.set_cpu_count(1); pa.set_io_thread_count(1)

D = Path("/home/jovyan/shared_data/rees46_nextitem_v1")
BUCKETS = (0, 17, 42)
PAIRS = [("category_code_missing", "main_category_missing"), ("has_history", "has_history_mask"),
         ("has_history_mask", "prev_user_event_available"), ("has_history_mask", "recent_price_supported"),
         ("price_zero", "price_missing")]
TIMES = [("log1p_time_since_prev_user_event_s", "prev_user_event_available"),
         ("log1p_time_since_prev_session_last_event_s", "prev_session_available"),
         ("log1p_prev_raw_view_gap_s", "prev_gap_unavailable")]
norm = json.loads((D / "mappings" / "normalizers.json").read_text())
out = {"label": "SAMPLE_NOT_POPULATION", "buckets": BUCKETS, "per_split": {}}
for split in ("TRAIN", "VALIDATION_SELECT"):
    res = {"pairs": {}, "times": {}}
    for a, b in PAIRS:
        n = diff = 0
        for bk in BUCKETS:
            t = pq.read_table(D / "model_ready" / f"user_bucket={bk}" / "features.parquet", columns=[a, b, "split"])
            m = np.asarray(t["split"].to_numpy(zero_copy_only=False) == split)
            x = np.asarray(t[a].to_numpy(zero_copy_only=False), dtype=np.float64)[m]
            y = np.asarray(t[b].to_numpy(zero_copy_only=False), dtype=np.float64)[m]
            n += x.size; diff += int((x != y).sum())
        res["pairs"][f"{a}=={b}"] = {"n": n, "n_differ": diff}
    for f, m_name in TIMES:
        vals_ok, vals_bad = [], []
        for bk in BUCKETS:
            t = pq.read_table(D / "model_ready" / f"user_bucket={bk}" / "features.parquet", columns=[f, m_name, "split"])
            m = np.asarray(t["split"].to_numpy(zero_copy_only=False) == split)
            x = np.asarray(t[f].to_numpy(zero_copy_only=False), dtype=np.float64)[m]
            k = np.asarray(t[m_name].to_numpy(zero_copy_only=False)).astype(bool)[m]
            avail = ~k if m_name.endswith("unavailable") else k
            vals_ok.append(x[avail]); vals_bad.append(x[~avail])
        ok, bad = np.concatenate(vals_ok), np.concatenate(vals_bad)
        spec = norm.get(f) or norm.get("features", {}).get(f) or {}
        res["times"][f] = {"n_available": int(ok.size), "n_unavailable": int(bad.size),
                           "available_mean": float(ok.mean()) if ok.size else None,
                           "available_std": float(ok.std()) if ok.size else None,
                           "available_q01_q50_q99": [float(q) for q in np.quantile(ok, [0.01, 0.5, 0.99])] if ok.size else None,
                           "unavailable_raw_distinct": [float(v) for v in np.unique(bad)[:5]],
                           "all_rows_mean": float(np.concatenate([ok, bad]).mean()),
                           "all_rows_std": float(np.concatenate([ok, bad]).std()),
                           "fitted_spec": spec}
    out["per_split"][split] = res
Path(sys.argv[1]).write_text(json.dumps(out, indent=1, allow_nan=False))
print(json.dumps(out)[:3000])
