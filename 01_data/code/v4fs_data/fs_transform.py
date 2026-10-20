"""v4fs_data.fs_transform (owner: Data engineer, FS-01) - TRANSFORM_STATE_FS: the 17 Clean V2 z-normalizers refit on the
FS users' TRAIN rows. OPTIONAL post-lock sensitivity analysis ONLY: the FS screen uses TRANSFORM_STATE_CLEAN_V2
(full-TRAIN statistics; DECISION_LOG 42 / 43). Nothing here is used by any screen run.

Rule (identical to TRANSFORM_STATE_CLEAN_V2 / FEATURE_SCHEMA_CLEAN_V2 2.1.0 normalized_channels_v2; the SAME code is
first run on ALL TRAIN rows and must reproduce every Clean V2 value before the FS values are accepted):
  grain     one unweighted row per TRAIN feature row (model_ready/user_bucket=*/features.parquet, split == TRAIN only)
  clip      clip_lo / clip_hi = float32 of the exact linear (numpy 'linear') 0.001 / 0.999 quantiles of the stored raw
            value over the channel's QUANTILE rows: the valid rows for the 14 v1-rule channels, ALL rows for M1 x 2 and
            recent_price_supported > 0 for F-1 (the v1 clips that the v2 re-encodings keep)
  moments   arithmetic mean and population std (ddof 0) of clip(x, clip_lo, clip_hi) over the channel's VALID rows
            (the schema fit_population; M1 x 2 and F-1: the valid-only convention of the v2 re-encoding)
  kernels   featstat_v1.fs_core (value tables, quantile_linear, clipped_moments) imported unchanged - the method with
            which FEATSTAT-01 reproduced all 17 recorded normalizers exactly
NOT refit (item side, full TRAIN, unchanged): catalogue, vocabularies, recovery tables, first_seen_train, the rezero rule
and the constants. log1p_item_observed_age_s keeps its stored raw value (full-TRAIN first_seen); only its z-normalizer
(clip, mean, std) is refit on FS.
Re-normalization of a channel c with the FS state: z = float32((min(max(float64(raw_c), clip_lo_c), clip_hi_c) - mean_c)
/ std_c), then z = 0.0 where the rezero gate of c holds (rezero_map unchanged; ungated channels are never re-zeroed).
TEST is never read: the scan filter is split == TRAIN and the result is asserted TRAIN-only.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
import ppsi_env  # noqa: E402,F401  (import isolation first, T43)

import numpy as np  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.dataset as pads  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from featstat_v1 import fs_core as F  # noqa: E402
from v4data.common import (N_BUCKETS, SCHEMA_SELF_SHA256, VERSION_HASH, TestGateError, dataset_root, digest_rows,  # noqa: E402
                           digests_of, load_schema, sha256_file, table_content_sha256, utc_now)
from v4data.manifests import reread_columns  # noqa: E402
from v4data.transform_state import load_state  # noqa: E402
from v4fs_data import fs_common as FC  # noqa: E402

V4 = FC.V4
STATE_REL = "TRANSFORM_STATE_FS.json"
PRED_COLS = ("price_missing", "price_conflict", "prev_gap_unavailable", "item_age_unavailable",
             "recent_price_supported", "prev_user_event_available", "prev_session_available")
FULL_TRAIN_ROWS = 52_760_314                      # SPLIT_COUNTS TRAIN.decisions = TRAIN feature rows
REL_TOL = 1e-12
RULES = {
    "log1p_price": ("VALID", "NOT price_missing AND NOT price_conflict", "price_missing OR price_conflict"),
    "log1p_prev_raw_view_gap_s": ("VALID", "NOT prev_gap_unavailable", "prev_gap_unavailable"),
    "log1p_item_observed_age_s": ("VALID", "NOT item_age_unavailable", "item_age_unavailable"),
    "recent_mean_log_price": ("VALID", "recent_price_supported > 0", "recent_price_supported == 0"),
    "current_minus_recent_mean_log_price": ("recent_price_supported > 0",
                                            "recent_price_supported > 0 AND NOT price_missing AND NOT price_conflict",
                                            "recent_price_supported == 0 OR price_missing OR price_conflict (the "
                                            "decision's own event)"),
    "log1p_time_since_prev_user_event_s": ("ALL", "prev_user_event_available > 0", "prev_user_event_available == 0"),
    "log1p_time_since_prev_session_last_event_s": ("ALL", "prev_session_available > 0", "prev_session_available == 0"),
}
_FS_SET: frozenset = frozenset()
_DS: Path = Path(".")
_CHANNELS: tuple = ()


def masks(name: str, raw: dict, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(quantile rows, valid rows) of a channel, exactly as FEATSTAT-01 (fs_run.process_bucket section 9)."""
    pm = raw["price_missing"].astype(bool) | raw["price_conflict"].astype(bool)
    rps = np.asarray(raw["recent_price_supported"]) > 0
    if name == "log1p_price":
        v = ~pm
        return v, v
    if name == "log1p_prev_raw_view_gap_s":
        v = ~raw["prev_gap_unavailable"].astype(bool)
        return v, v
    if name == "log1p_item_observed_age_s":
        v = ~raw["item_age_unavailable"].astype(bool)
        return v, v
    if name == "recent_mean_log_price":
        return rps, rps
    if name == "current_minus_recent_mean_log_price":
        return rps, rps & ~pm
    if name == "log1p_time_since_prev_user_event_s":
        return np.ones(n, dtype=bool), np.asarray(raw["prev_user_event_available"]) > 0
    if name == "log1p_time_since_prev_session_last_event_s":
        return np.ones(n, dtype=bool), np.asarray(raw["prev_session_available"]) > 0
    a = np.ones(n, dtype=bool)
    return a, a


def read_train(path: Path, columns) -> "pa.Table":  # noqa: F821
    t = pads.dataset(str(path), format="parquet", partitioning=None).to_table(
        columns=["split"] + list(columns), filter=pc.field("split") == "TRAIN")
    seen = set(pc.unique(t["split"]).to_pylist())
    if not seen <= {"TRAIN"}:
        raise TestGateError(f"{path}: the TRAIN scan returned {sorted(seen)}")
    return t


def bucket_tables(b: int, ds: Path | None = None, fs_set: frozenset | None = None, channels: tuple | None = None) -> dict:
    ds = _DS if ds is None else ds
    fs_set = _FS_SET if fs_set is None else fs_set
    channels = _CHANNELS if channels is None else channels
    t = read_train(Path(ds) / "model_ready" / f"user_bucket={b}" / "features.parquet",
                   ["user_id"] + list(channels) + list(PRED_COLS))
    n = t.num_rows
    uniq = pc.unique(t["user_id"])
    ui = pc.index_in(t["user_id"], value_set=uniq).to_numpy(zero_copy_only=False).astype(np.int64)
    in_fs_u = np.fromiter((d in fs_set for d in digest_rows(digests_of(uniq.to_pylist()))), dtype=bool, count=len(uniq))
    fsrow = in_fs_u[ui]
    raw = {c: t[c].to_numpy(zero_copy_only=False) for c in list(channels) + list(PRED_COLS)}
    del t
    out = {}
    for name in channels:
        x = raw[name]
        if not np.all(np.isfinite(x)):
            raise AssertionError(f"bucket {b}: non-finite raw {name}")
        q, v = masks(name, raw, n)
        same = q is v or np.array_equal(q, v)
        for pop, pm in (("ALL", None), ("FS", fsrow)):
            qm = q if pm is None else (q & pm)
            tq = F.value_counts(x[qm])
            tv = tq if same else F.value_counts(x[(v if pm is None else (v & pm))])
            out[(name, pop)] = (tq, None if same else tv)
    return {"bucket": b, "rows": int(n), "fs_rows": int(fsrow.sum()), "fs_users": int(in_fs_u.sum()), "tables": out}


def _worker(b: int) -> dict:
    return bucket_tables(b)


def fit_channel(q: tuple, v: tuple) -> dict:
    qv, qc = q
    vv, vc = v
    lo, hi = F.quantile_linear(qv, qc, 0.001), F.quantile_linear(qv, qc, 0.999)
    clo, chi = float(np.float32(lo)), float(np.float32(hi))
    mom = F.clipped_moments(vv, vc, clo, chi)
    return {"clip_lo": clo, "clip_hi": chi, "mean": mom["mean"], "std": mom["std"],
            "clip_quantiles_float64": [lo, hi], "n_quantile_rows": int(qc.sum()), "n_valid_rows": int(vc.sum()),
            "clipped_low_rate": mom["clipped_low_rate"], "clipped_high_rate": mom["clipped_high_rate"]}


def _z(x: np.ndarray, p) -> np.ndarray:
    lo, hi, mu, sd = p
    return ((np.clip(np.asarray(x, dtype=np.float64), lo, hi) - mu) / sd).astype(np.float32)


def compare(fit: dict, rec: dict, v: tuple) -> dict:
    """Fitted vs recorded parameters, and the float32 encodings of every distinct valid value under both."""
    rel = lambda a, b: abs(a - b) / abs(b) if b else abs(a - b)  # noqa: E731
    vv, vc = v
    zf = _z(vv, (fit["clip_lo"], fit["clip_hi"], fit["mean"], fit["std"]))
    zr = _z(vv, (rec["clip_lo"], rec["clip_hi"], rec["mean"], rec["std"]))
    diff = zf != zr
    out = {"clip_lo_equal": fit["clip_lo"] == rec["clip_lo"], "clip_hi_equal": fit["clip_hi"] == rec["clip_hi"],
           "mean_rel_diff": rel(fit["mean"], rec["mean"]), "std_rel_diff": rel(fit["std"], rec["std"]),
           "mean_float32_equal": bool(np.float32(fit["mean"]) == np.float32(rec["mean"])),
           "std_float32_equal": bool(np.float32(fit["std"]) == np.float32(rec["std"])),
           "distinct_valid_values": int(vv.size), "distinct_values_float32_z_differ": int(diff.sum()),
           "valid_rows_float32_z_differ": int(vc[diff].sum()),
           "max_abs_z_diff": float(np.max(np.abs(zf.astype(np.float64) - zr.astype(np.float64)), initial=0.0))}
    out["reproduced"] = bool(out["clip_lo_equal"] and out["clip_hi_equal"] and out["mean_rel_diff"] <= REL_TOL
                             and out["std_rel_diff"] <= REL_TOL)
    return out


def effect(fs_fit: dict, rec: dict, v: tuple) -> dict:
    vv, vc = v
    w = vc.astype(np.float64)
    zf = _z(vv, (fs_fit["clip_lo"], fs_fit["clip_hi"], fs_fit["mean"], fs_fit["std"])).astype(np.float64)
    zr = _z(vv, (rec["clip_lo"], rec["clip_hi"], rec["mean"], rec["std"])).astype(np.float64)
    d = np.abs(zf - zr)
    return {"fs_valid_rows": int(vc.sum()), "max_abs_change_z": float(d.max(initial=0.0)),
            "mean_abs_change_z": float(np.dot(w, d) / w.sum()) if w.sum() else None}


def fs_digests(v4: Path, out_rel: str = "data_fs") -> tuple[frozenset, dict]:
    mdir = Path(v4) / out_rel / "manifests"
    h = json.loads((mdir / f"{FC.FS_TRAIN_ID}.header.json").read_text(encoding="utf-8"))
    if h["header_content_sha256"] != FC.header_content_sha256(h):
        raise AssertionError("FS_TRAIN header content hash does not verify")
    t = next(x for x in h["tables"] if x["table"] == "clients")
    p = mdir / f"{FC.FS_TRAIN_ID}.clients.parquet"
    tbl = pq.ParquetFile(p).read()
    if sha256_file(p) != t["file_sha256"] or table_content_sha256(reread_columns(tbl)) != t["table_content_sha256"]:
        raise AssertionError("FS_TRAIN clients table does not verify against its header")
    hexs = tbl["client_key_hex"].to_pylist()
    return frozenset(bytes.fromhex(x) for x in hexs), {
        "manifest_id": FC.FS_TRAIN_ID, "header_content_sha256": h["header_content_sha256"], "n_users": len(hexs),
        "population_sha256": h["population_sha256"],
        "population_sha256_selection_key_v2": h["population_sha256_selection_key_v2"]}


def build_state(v4: Path, ds: Path, buckets, workers: int, fs_set: frozenset, fs_info: dict, *, full: bool,
                clean_state_path: Path) -> dict:
    global _FS_SET, _DS, _CHANNELS
    t0 = time.time()
    schema = load_schema()
    channels = tuple(schema["normalized_channels_v2"].keys())
    if len(channels) != 17:
        raise AssertionError(f"expected 17 normalized channels, got {len(channels)}")
    clean = load_state(clean_state_path)
    if full and clean["transform_state_sha256"] != FC.TRANSFORM_STATE_SELF_SHA256:
        raise AssertionError("TRANSFORM_STATE_CLEAN_V2 self-hash changed")
    rec_all = clean["normalized_channels_v2"]
    _FS_SET, _DS, _CHANNELS = fs_set, Path(ds), channels
    acc = {(c, pop, k): F.TableAccumulator(8) for c in channels for pop in ("ALL", "FS") for k in ("Q", "V")}
    same = {}
    tot = {"rows": 0, "fs_rows": 0, "fs_users": 0}

    def merge(r: dict) -> None:
        for k in tot:
            tot[k] += r[k]
        for (c, pop), (tq, tv) in r["tables"].items():
            acc[(c, pop, "Q")].add(*tq)
            same[c] = tv is None
            if tv is not None:
                acc[(c, pop, "V")].add(*tv)

    if workers > 1:
        with mp.get_context("fork").Pool(workers) as pool:          # fork: the workers inherit the FS set
            for r in pool.imap(_worker, list(buckets)):
                merge(r)
    else:
        for b in buckets:
            merge(_worker(b))
    rows, fs_rows, fs_users = tot["rows"], tot["fs_rows"], tot["fs_users"]
    t_pass = time.time()
    channels_fs, proof, eff = {}, {}, {}
    for c in channels:
        res = {}
        for pop in ("ALL", "FS"):
            q = acc[(c, pop, "Q")].result()
            v = q if same[c] else acc[(c, pop, "V")].result()
            res[pop] = (fit_channel(q, v), v)
        rec = {k: float(rec_all[c][k]) for k in ("clip_lo", "clip_hi", "mean", "std")}
        proof[c] = dict(compare(res["ALL"][0], rec, res["ALL"][1]), fit_all_train=res["ALL"][0])
        fsf = res["FS"][0]
        rq, rv, rz = RULES.get(c, ("ALL", "ALL", None))
        channels_fs[c] = {**{k: fsf[k] for k in ("clip_lo", "clip_hi", "mean", "std")},
                          "n_quantile_rows": fsf["n_quantile_rows"], "n_valid_rows": fsf["n_valid_rows"],
                          "clip_quantiles_float64": fsf["clip_quantiles_float64"],
                          "clipped_low_rate": fsf["clipped_low_rate"], "clipped_high_rate": fsf["clipped_high_rate"],
                          "quantile_rows": "VALID rows" if rq == "VALID" else rq, "valid_rows": rv,
                          "rezero_gate": rz, "fit_population_clean_v2": rec_all[c].get("fit_population"),
                          "clean_v2_values": rec,
                          "delta_vs_clean_v2": {k: fsf[k] - rec[k] for k in ("clip_lo", "clip_hi", "mean", "std")}}
        eff[c] = effect(fsf, rec, res["FS"][1])
    m1 = clean["m1"]
    f1 = clean["f1"]
    n_valid_checks = {**{f"M1.{n}.n_valid": (proof[n]["fit_all_train"]["n_valid_rows"], m1[n]["n_valid"]) for n in m1},
                      f"F1.{f1['feature']}.n_valid": (proof[f1["feature"]]["fit_all_train"]["n_valid_rows"], f1["n_valid"])}
    method = {"per_channel": proof,
              "all_train_rows": {"observed": rows, "expected": FULL_TRAIN_ROWS, "match": rows == FULL_TRAIN_ROWS},
              "n_valid_checks": {k: {"observed": a, "expected": b, "match": a == b} for k, (a, b) in n_valid_checks.items()},
              "criterion": f"clip_lo and clip_hi equal exactly (float32 of the quantile); |relative difference| of mean "
                           f"and std <= {REL_TOL} (float64 summation order); float32 z identity reported per channel",
              "reference": {"path": "data_v2/TRANSFORM_STATE_CLEAN_V2.json",
                            "transform_state_sha256": clean["transform_state_sha256"],
                            "file_sha256": sha256_file(clean_state_path)}}
    method["all_reproduced"] = bool(all(p["reproduced"] for p in proof.values())
                                    and (method["all_train_rows"]["match"] or not full)
                                    and all(x["match"] for x in method["n_valid_checks"].values()))
    per_channel_formula = {c: (f"z = float32((min(max(float64(raw), {channels_fs[c]['clip_lo']!r}), "
                               f"{channels_fs[c]['clip_hi']!r}) - {channels_fs[c]['mean']!r}) / "
                               f"{channels_fs[c]['std']!r})" + (f"; z = 0.0 when {channels_fs[c]['rezero_gate']}"
                                                                if channels_fs[c]["rezero_gate"] else "; never re-zeroed"))
                           for c in channels}
    state = {
        "state": "TRANSFORM_STATE_FS", "schema_version": schema["schema_version"], "view_id": "clean_v2_rich",
        "status": "OPTIONAL_SENSITIVITY_ONLY" if full else "PARTIAL_TEST_BUILD",
        "use": ("OPTIONAL post-lock sensitivity analysis only. The FS screen uses TRANSFORM_STATE_CLEAN_V2 (full-TRAIN "
                "statistics, DECISION_LOG 42 / 43); this state is never used by the screen."),
        "run_label": FC.FS_STATUS, "created_by_task": FC.TASK, "data_version": FC.DATA_VERSION,
        "v1_version_hash": VERSION_HASH, "schema_self_sha256": SCHEMA_SELF_SHA256,
        "base_transform_state": method["reference"],
        "fit": {"population": fs_info, "split": "TRAIN", "buckets": len(list(buckets)),
                "grain": "one unweighted row per TRAIN feature row of the FS users (canonical representative = decision)",
                "rows_all_train": rows, "rows_fs_train": fs_rows, "fs_users_seen_in_train_rows": fs_users,
                "estimator": "clip = float32(linear quantiles 0.001 / 0.999 over the quantile rows); mean and population "
                             "std (ddof 0) of clip(x) over the valid rows",
                "kernels": "featstat_v1.fs_core (value tables, quantile_linear, clipped_moments), unchanged"},
        "normalized_channels_fs": channels_fs,
        "renormalization": {"formula": "raw -> clip to [clip_lo, clip_hi] -> (x - mean) / std in float64 -> float32 -> "
                                       "rezero (0.0 where the channel's rezero gate holds; rezero_map unchanged)",
                            "per_channel": per_channel_formula},
        "rezero_map": clean["rezero_map"],
        "not_refit_item_side_full_train": ["catalogue (K, support, order)", "attribute vocabularies", "recovery tables",
                                           "first_seen_train (item age: the stored raw log1p age is used as is)",
                                           "rezero rule", "clip quantile levels and constants (+3h clock, calendars)"],
        "effect_on_fs_valid_rows_vs_clean_v2": eff,
        "method_proof_all_train": method,
        "code": {"fs_transform_sha256": sha256_file(HERE / "fs_transform.py"),
                 "featstat_fs_core_sha256": sha256_file(CODE / "featstat_v1" / "fs_core.py"),
                 "fs_common_sha256": sha256_file(HERE / "fs_common.py"),
                 "v4data_builder_code_sha256": FC.builder_code_sha256()},
        "timings_s": {"pass": round(t_pass - t0, 1), "total": round(time.time() - t0, 1)},
        "no_test_rows_read": True,
    }
    return FC.with_self_hash(state, "transform_state_fs_sha256")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="FS-01 TRANSFORM_STATE_FS (optional sensitivity normalizers)")
    ap.add_argument("--v4", default=str(V4))
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--buckets", default="all")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out-rel", default="data_fs")
    a = ap.parse_args(argv)
    v4 = Path(a.v4)
    ds = Path(a.dataset) if a.dataset else dataset_root()
    buckets = list(range(N_BUCKETS)) if a.buckets == "all" else [int(x) for x in a.buckets.split(",")]
    full = buckets == list(range(N_BUCKETS))
    fs_set, fs_info = fs_digests(v4, a.out_rel)
    if full and len(fs_set) != FC.FS_N:
        raise AssertionError(f"FS_TRAIN has {len(fs_set)} users, expected {FC.FS_N}")
    st = build_state(v4, ds, buckets, a.workers, fs_set, fs_info, full=full,
                     clean_state_path=v4 / "data_v2" / "TRANSFORM_STATE_CLEAN_V2.json")
    if full and st["fit"]["fs_users_seen_in_train_rows"] != FC.FS_N:
        raise AssertionError("not every FS user has TRAIN feature rows")
    out = v4 / a.out_rel
    p = FC.atomic_write_bytes(out, STATE_REL, FC.json_bytes_strict(st))
    back = json.loads(p.read_text(encoding="utf-8"))
    ok = FC.verify_self_hash(back, "transform_state_fs_sha256") and st["method_proof_all_train"]["all_reproduced"]
    print(json.dumps({"state": str(p), "file_sha256": sha256_file(p),
                      "transform_state_fs_sha256": st["transform_state_fs_sha256"],
                      "method_all_reproduced": st["method_proof_all_train"]["all_reproduced"],
                      "rows_all_train": st["fit"]["rows_all_train"], "rows_fs_train": st["fit"]["rows_fs_train"],
                      "python_utc_end": utc_now()}))
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
