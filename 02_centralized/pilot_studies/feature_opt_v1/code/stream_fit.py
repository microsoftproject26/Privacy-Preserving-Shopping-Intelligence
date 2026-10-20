"""stream_fit.py (owner: Opus) — ONE exact streamed pass over TRAIN (all 64 user buckets) + VS drift.

Run only after the campaign's final stage (heavy scan). It reads `model_ready/user_bucket=*/features.parquet`
with column projection, filtered to split == TRAIN (fits) and split == VALIDATION_SELECT (drift and
duplicate checks only; never used in a fit). VC/TEST are refused.

Outputs (never overwritten):
- audit/M1_FIT.json: for each M1 feature, the valid-only mean/std (values clipped by the RECORDED clip
  bounds, float64 accumulation) plus the all-rows mean/std, reproducing normalizers.json and so
  proving the defect at population level.
- config/P8_CALENDAR_CELL.json: the modal (local_weekday_idx, daypart_idx) over TRAIN rows.
- audit/DUPLICATES_FULL.json: the population check of every M2 duplicate and M3 constant flag.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import fo_env
import feature_view as fv

pa.set_cpu_count(1)
pa.set_io_thread_count(1)
SPLITS = ("TRAIN", "VALIDATION_SELECT")
AFFINE = [("log1p_prefix_elapsed_s", "log1p_segment_elapsed_s", 0.0), ("prefix_item_count", "prefix_representative_count", 1.0)]
FLAG_DUPS = [("price_zero", "price_missing"), ("main_category_missing", "category_code_missing")]
MASK_DUPS = [("has_history_mask", "has_history"), ("prev_user_event_available", "has_history")]


class Acc:
    def __init__(self):
        self.n = 0; self.s = 0.0; self.ss = 0.0

    def add(self, x: np.ndarray):
        x = x.astype(np.float64); self.n += x.size; self.s += float(x.sum()); self.ss += float((x * x).sum())

    def out(self):
        if self.n == 0:
            return {"n": 0}
        m = self.s / self.n
        return {"n": self.n, "mean": m, "std": float(np.sqrt(max(self.ss / self.n - m * m, 0.0)))}


def main(root: str, buckets: list[int]) -> int:
    root_p = fo_env.DATASET_ROOT if root == "default" else __import__("pathlib").Path(root)
    spec = fv.normalizers().spec
    cols = sorted(set(["split", "local_weekday_idx", "daypart_idx"] + list(fv.M1_FEATURES) + list(fv.M1_FEATURES.values())
                      + [a for a, _, _ in AFFINE] + [b for _, b, _ in AFFINE] + ["utc_hour_sin", "utc_hour_cos",
                      "local_hour_sin", "local_hour_cos"] + [a for a, b in FLAG_DUPS + MASK_DUPS] + [b for a, b in FLAG_DUPS + MASK_DUPS]
                      + list(fv.DROP_FLAG_DEAD)))
    acc = {sp: {f: {"valid": Acc(), "all": Acc()} for f in fv.M1_FEATURES} for sp in SPLITS}
    cal = np.zeros((16, 16), dtype=np.int64)
    dup = {sp: {} for sp in SPLITS}
    rows = {sp: 0 for sp in SPLITS}
    phi = 2 * np.pi * fv.LOCAL_SHIFT_H / 24.0
    t0 = time.time()
    for b in buckets:
        tbl = pq.read_table(root_p / "model_ready" / f"user_bucket={b}" / "features.parquet", columns=cols)
        split = tbl["split"].to_numpy(zero_copy_only=False)
        for sp in SPLITS:
            m = split == sp
            if not m.any():
                continue
            rows[sp] += int(m.sum())
            col = lambda n: np.asarray(tbl[n].to_numpy(zero_copy_only=False))[m]
            for f, mask in fv.M1_FEATURES.items():
                x = np.clip(col(f).astype(np.float64), spec[f]["clip_lo"], spec[f]["clip_hi"])
                av = col(mask).astype(bool)
                acc[sp][f]["all"].add(x)
                acc[sp][f]["valid"].add(x[av])
            if sp == "TRAIN":
                np.add.at(cal, (col("local_weekday_idx").astype(np.int64), col("daypart_idx").astype(np.int64)), 1)
            d = dup[sp]
            for a, k, shift in AFFINE:
                bad = int((col(a).astype(np.float64) != col(k).astype(np.float64) + shift).sum())
                d[f"{a}=={k}+{shift}"] = d.get(f"{a}=={k}+{shift}", 0) + bad
            us, uc, ls, lc = (col(n).astype(np.float64) for n in ("utc_hour_sin", "utc_hour_cos", "local_hour_sin", "local_hour_cos"))
            res = np.maximum(np.abs(ls - (np.cos(phi) * us + np.sin(phi) * uc)), np.abs(lc - (-np.sin(phi) * us + np.cos(phi) * uc)))
            d["local_hour_rotation_max_residual"] = max(d.get("local_hour_rotation_max_residual", 0.0), float(res.max()))
            for a, k in FLAG_DUPS + MASK_DUPS:
                d[f"{a}=={k}"] = d.get(f"{a}=={k}", 0) + int((col(a).astype(np.float64) != col(k).astype(np.float64)).sum())
            for n in fv.DROP_FLAG_DEAD:
                d[f"{n}_nonzero"] = d.get(f"{n}_nonzero", 0) + int((col(n) != 0).sum())
        del tbl
    wd, dp = np.unravel_index(int(cal.argmax()), cal.shape)
    m1 = {"label": "FULL_POPULATION" if len(buckets) == 64 else "SAMPLE_NOT_POPULATION", "buckets": buckets,
          "rows": rows, "fit_split": "TRAIN", "clip": "recorded normalizers.json bounds",
          "features": {f: {"mask": fv.M1_FEATURES[f], "fitted_normalizer": spec[f],
                           "train_valid": acc["TRAIN"][f]["valid"].out(), "train_all_rows": acc["TRAIN"][f]["all"].out(),
                           "vs_valid_drift_only": acc["VALIDATION_SELECT"][f]["valid"].out()} for f in fv.M1_FEATURES},
          "seconds": round(time.time() - t0, 1)}
    for f, r in m1["features"].items():
        r["valid_mean"], r["valid_std"] = r["train_valid"]["mean"], r["train_valid"]["std"]
        r["normalizer_reproduced_by_all_rows"] = bool(abs(r["train_all_rows"]["mean"] - spec[f]["mean"]) < 1e-3
                                                      and abs(r["train_all_rows"]["std"] - spec[f]["std"]) < 1e-3)
    cell = {"rule": "modal (local_weekday_idx, daypart_idx) over TRAIN feature rows (Fable decision item 3, P8)",
            "label": m1["label"], "local_weekday_idx": int(wd), "daypart_idx": int(dp), "count": int(cal[wd, dp]),
            "n_rows": int(cal.sum()), "instant": fv.calendar_instant(int(wd), int(dp))}
    outs = ((fo_env.study_path("audit", "M1_FIT.json"), m1), (fo_env.study_path("config", "P8_CALENDAR_CELL.json"), cell),
            (fo_env.study_path("audit", "DUPLICATES_FULL.json"), {"label": m1["label"], "rows": rows, "checks": dup}))
    for p, obj in outs:
        fo_env.write_json_once(p, obj)
    print(json.dumps({"rows": rows, "cell": [int(wd), int(dp)],
                      "m1": {f: (r["valid_mean"], r["valid_std"], r["normalizer_reproduced_by_all_rows"]) for f, r in m1["features"].items()},
                      "dup_nonzero": {sp: {k: v for k, v in d.items() if v} for sp, d in dup.items()}}))
    return 0


if __name__ == "__main__":
    bk = sys.argv[2] if len(sys.argv) > 2 else "all"
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "default",
                  list(range(64)) if bk == "all" else [int(x) for x in bk.split(",")]))
