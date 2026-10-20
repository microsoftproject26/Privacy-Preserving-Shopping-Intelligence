"""DATA-05 server measurements (Lobot only; physical GPU 0 only for the <= 300-update benchmark; no training).

  python -B code/tests_hist/server/data05_measure.py <step> [...]      (each step in a fresh process, from job.sh)

Steps (each writes runs_g1/data05/parts/<name>.json; every timestamp from `date -u`; resources recorded):
  census          the 5-100 pool (CENSUS-02 definitions via v4hist.census, n under catalogue A), all 64 buckets in
                  groups: per group ONE CompactStore over TRAIN + VALIDATION_SELECT + VALIDATION_CONFIRM, its RowIndex
                  proven aligned; for EVERY pool supervised TRAIN decision and EVERY pool VALIDATION eval row:
                  the window lengths under both policies, the exhaustive leakage / structure check (check_windows,
                  both policies), monotonicity, the window composition (segments spanned, TRAIN-period rows);
                  served-batch tie-in on 64 random batches of 256 per split and group (check_served_batch, both
                  policies, and only-the-window-changes); USER == SESSION bitwise for decisions without an earlier
                  segment; group 0 also: SESSION == the INT-01 decision-split stores bitwise and USER TRAIN windows
                  from a TRAIN-only store == from the full store. Counts only; no model outcome.
  provider        a manifest-scoped cache (v4int.cache.build_subset_store, the providers' data path) over the pool
                  decisions of a bucket range (--source pool) or a real provider over a registered HISTORICAL
                  manifest (--source manifest): build time, RSS, cache bytes, bytes per row, rows/s at B = 256
                  (random positions, the trainer's pass order) and, for evaluation, at 1,024 in order.
  bench           GPU 0: GRU and SASRec full-K training step (fused AdamW, clip, full-K CE, post_step, STRICT) on real
                  pool TRAIN batches: 15 warm-up + 95 USER (pre-loaded) + 95 SESSION (pre-loaded, same positions) + 95
                  USER (cache in the loop) = 300 updates per family. BENCH_NOT_EVIDENCE: weights never saved, no row.
  assemble        DATA05_SERVER_RECEIPT.json from the parts, with code hashes.
TEST is never read: ledgers and the index go through read_non_test; the store is the locked CompactStore path.
No raw user id is printed or saved.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
V4 = CODE.parent
OUT = Path(os.environ["DATA05_OUT"]) if os.environ.get("DATA05_OUT") else V4 / "runs_g1" / "data05"
PARTS = OUT / "parts"
for _p in (str(CODE), str(CODE / "tests_hist"), str(CODE / "tests_int"), str(CODE / "tests_data")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
GPU = 0
B = 256
POOL = {"users": 1_446_986, "train_examples": 28_623_386, "val_eval_rows": 5_823_491}     # CENSUS-02 (C MAIN)
ALL = {"elig": 40_180_608, "val_eval": 11_288_056}                                       # CENSUS-02 TOTAL


def utc() -> str:
    return subprocess.run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], capture_output=True, text=True).stdout.strip()


def sh(cmd: str) -> str:
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (p.stdout + p.stderr).strip()


def resources() -> dict:
    cg = None
    try:
        cg = int(Path("/sys/fs/cgroup/memory.current").read_text().strip())
    except OSError:
        pass
    return {"utc": utc(), "nvidia_smi": sh("nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu "
                                            "--format=csv,noheader"),
            "nvidia_smi_compute_apps": sh("nvidia-smi --query-compute-apps=gpu_bus_id,pid,used_memory "
                                          "--format=csv,noheader"),
            "free_g": sh("free -g"), "df_home": sh("df -B1 /home/jovyan | tail -1"), "loadavg": (list(os.getloadavg()) if hasattr(os, "getloadavg") else None),
            "cgroup_memory_current_bytes": cg}


def write_part(name: str, obj: dict) -> Path:
    PARTS.mkdir(parents=True, exist_ok=True)
    p = PARTS / f"{name}.json"
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, allow_nan=False, default=str), encoding="utf-8")
    os.replace(tmp, p)
    return p


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S', time.gmtime())}Z] {msg}", flush=True)


def batches_equal(a, b) -> tuple:
    import torch
    if set(a) != set(b):
        return False, f"key sets differ: {sorted(set(a) ^ set(b))}"
    for k in a:
        x, y = a[k], b[k]
        if torch.is_tensor(x) or torch.is_tensor(y):
            if not (torch.is_tensor(x) and torch.is_tensor(y) and x.dtype == y.dtype and x.shape == y.shape
                    and torch.equal(x, y)):
                return False, f"tensor {k} differs"
        elif x != y:
            return False, f"field {k} differs"
    return True, "equal"


def _stats(ts) -> dict:
    import numpy as np
    a = np.asarray(ts, dtype=np.float64)
    return {"n": int(a.size), "total_s": round(float(a.sum()), 3), "median_ms": round(float(np.median(a)) * 1e3, 3),
            "p90_ms": round(float(np.percentile(a, 90)) * 1e3, 3), "max_ms": round(float(a.max()) * 1e3, 3)}


def _buckets(spec: str) -> list:
    lo, _, hi = spec.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def _lock_state_sha() -> str:
    lock = json.loads((V4 / "contracts" / "DATA_PROTOCOL_LOCK.json").read_text(encoding="utf-8"))
    return lock["features_and_transforms"]["transform_state_self_sha256"]


def _trim() -> None:
    gc.collect()
    try:
        import pyarrow as pa
        pa.default_memory_pool().release_unused()
    except Exception:
        pass
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def pool_ids(defs, root, buckets) -> tuple:
    import numpy as np
    from v4hist.census import pool_bucket
    parts = [pool_bucket(defs, root, b) for b in buckets]
    tid = np.sort(np.concatenate([p["train_ids"] for p in parts]))
    vid = np.sort(np.concatenate([p["val_ids"] for p in parts]))
    cnt = {k: int(sum(p[k] for p in parts)) for k in ("users", "pool_users", "pool_train_examples",
                                                      "pool_val_eval_rows", "all_elig", "all_val_eval")}
    return tid, vid, cnt


# ------------------------------------------------------------------------------------------------ census
def cmd_census(a) -> int:
    import ppsi_env
    import v4train  # noqa: F401
    import numpy as np
    import torch
    from v4data.common import VALIDATION_SPLITS, dataset_root
    from v4data.store import CleanV2Store
    from v4hist.census import composition_stats, length_stats, row_defs, window_composition
    from v4hist.policy import CONTEXT_LENGTH as L, SESSION_L50, USER_HISTORY_L50
    from v4hist.stream import HistoryStore
    from v4hist.windows import check_served_batch, check_windows, window_rows
    from v4int.cache import merged_decision_set, peak_rss_bytes, rss_bytes
    torch.set_num_threads(4)
    ppsi_env.verify_snapshot()
    ppsi_env.assert_isolated()
    root = dataset_root()
    rec = {"step": "census", "label": "COUNTS_ONLY (no model outcome; TEST never read)", "utc_start": utc(),
           "resources_before": resources(), "dataset_root": str(root), "group_size": int(a.group_size)}
    defs = row_defs()
    allb = _buckets(a.buckets)
    groups = [allb[i:i + a.group_size] for i in range(0, len(allb), a.group_size)]
    rng = np.random.default_rng(20260926)
    acc = {k: [] for k in ("uT", "sT", "uV", "sV", "vcode", "segT", "segV", "trT", "trV", "stV")}
    tot = {"pool_users": 0, "pool_train_examples": 0, "pool_val_eval_rows": 0, "users": 0, "all_elig": 0,
           "all_val_eval": 0}
    leak = {f"{s}|{p}": {"n_decisions": 0, "n_window_rows": 0, "rows_from_earlier_segments": 0,
                         "rows_from_earlier_partitions": 0, "decisions_with_earlier_partition_rows": 0}
            for s in ("TRAIN", "VALIDATION") for p in (USER_HISTORY_L50, SESSION_L50)}
    mono = {s: {"checked": 0, "no_earlier_segment": 0, "session_at_cap": 0, "user_longer": 0}
            for s in ("TRAIN", "VALIDATION")}
    served = {s: {"batches": 0, "rows": 0, "window_rows_user": 0, "window_rows_session": 0}
              for s in ("TRAIN", "VALIDATION")}
    equal_no_earlier = {s: {"batches": 0, "rows": 0} for s in ("TRAIN", "VALIDATION")}
    order = {"rows": 0, "users": 0, "segments": 0, "user_row_pairs_checked": 0, "ts_decreasing": 0,
             "partition_decreasing": 0, "segment_spans_partitions": 0}
    align_checked = 0
    glog, extras = [], None
    CH = 200_000
    for gi, g in enumerate(groups):
        tg = time.time()
        tid, vid, cnt = pool_ids(defs, root, g)
        for k in tot:
            tot[k] += cnt[k]
        hsV = HistoryStore.build(USER_HISTORY_L50, VALIDATION_SPLITS, buckets=g, ds_root=root, n_threads=3,
                                 log=lambda *_: None)
        hsT = HistoryStore(hsV.store, hsV.index, USER_HISTORY_L50, ("TRAIN",))
        ix, cs, spec = hsV.index, hsV.store.cs, hsV.store.spec
        for k in order:
            order[k] += int(ix.order_checks[k])
        align_checked += int(sum(hsV.alignment["decisions_checked"].values()))
        for name, hs, kind, ids in (("TRAIN", hsT, "loss", tid), ("VALIDATION", hsV, "eval", vid)):
            du, dsess = hs.decision_set(kind), hs.raw_decision_set(kind)
            pos = hs.positions_for(dsess, ids)                          # every pool decision must be served
            rows = ix.rows_of(ids)                                       # from the decision identity
            if not np.array_equal(rows, dsess.end_row[pos]) or not np.array_equal(rows, du.end_row[pos]):
                raise AssertionError(f"group {g[0]}: end_row != the decision's own index row")
            lu, ls = du.length[pos].astype(np.int8), dsess.length[pos].astype(np.int8)
            for pol, d in ((USER_HISTORY_L50, du), (SESSION_L50, dsess)):
                for s in range(0, pos.size, CH):
                    p = pos[s:s + CH]
                    idx, valid = window_rows(d.end_row[p], d.length[p])
                    rep = check_windows(ix, rows[s:s + CH], idx, valid, policy=pol)
                    for k in leak[f"{name}|{pol}"]:
                        leak[f"{name}|{pol}"][k] += int(rep[k])
                    del idx, valid
            u64, s64 = lu.astype(np.int64), ls.astype(np.int64)
            no_earlier = (ix.user_rank[rows].astype(np.int64) + 1) == ix.position[rows]
            full = ix.position[rows] >= L
            if not (np.all(u64 >= s64) and np.array_equal(u64 == s64, no_earlier | full)):
                raise AssertionError(f"group {g[0]} {name}: monotonicity violated")
            m = mono[name]
            m["checked"] += int(pos.size)
            m["no_earlier_segment"] += int(no_earlier.sum())
            m["session_at_cap"] += int(full.sum())
            m["user_longer"] += int((u64 > s64).sum())
            comp = window_composition(ix, rows, u64)
            if name == "TRAIN":
                acc["uT"].append(lu); acc["sT"].append(ls)                      # noqa: E702
                acc["segT"].append(comp["n_segments"].astype(np.int8))
                acc["trT"].append(comp["n_train_rows"].astype(np.int8))
            else:
                acc["uV"].append(lu); acc["sV"].append(ls)                      # noqa: E702
                acc["vcode"].append(ix.split_code[rows].astype(np.int8))
                acc["segV"].append(comp["n_segments"].astype(np.int8))
                acc["trV"].append(comp["n_train_rows"].astype(np.int8))
                acc["stV"].append(comp["starts_in_train"])
            # served bytes == the checked rows, both policies; only the window changes
            for _ in range(int(a.served_batches)):
                p = rng.choice(pos, B, replace=False)
                bu, bs = hs.batch(du, p), hs.batch(dsess, p)
                for pol, d, bb in ((USER_HISTORY_L50, du, bu), (SESSION_L50, dsess, bs)):
                    idx, valid = window_rows(d.end_row[p], d.length[p])
                    check_windows(ix, ix.rows_of(d.decision_id[p]), idx, valid, policy=pol)
                    r_ = check_served_batch(bb, cs, spec, idx, valid)
                    served[name]["window_rows_user" if pol == USER_HISTORY_L50 else "window_rows_session"] += \
                        r_["window_rows"]
                for k in ("user_context", "user_context_masks", "target_class", "loss_mask", "eval_mask",
                          "decision_id", "client_id"):
                    if not torch.equal(bu[k], bs[k]):
                        raise AssertionError(f"{name}: {k} differs between the policies")
                served[name]["batches"] += 1
                served[name]["rows"] += B
            er = dsess.end_row                                           # ALL decisions of the group (any user)
            cand = np.flatnonzero((ix.user_rank[er].astype(np.int64) + 1) == ix.position[er])
            equal_no_earlier[name]["candidates_all_users"] = equal_no_earlier[name].get("candidates_all_users", 0)                 + int(cand.size)
            equal_no_earlier[name]["candidates_pool"] = equal_no_earlier[name].get("candidates_pool", 0)                 + int(no_earlier.sum())
            for _ in range(int(a.equal_batches)):
                if cand.size < B:
                    break
                p = rng.choice(cand, B, replace=False)
                ok, why = batches_equal(hs.batch(du, p), hs.batch(dsess, p))
                if not ok:
                    raise AssertionError(f"{name}: USER != SESSION for decisions without an earlier segment ({why})")
                equal_no_earlier[name]["batches"] += 1
                equal_no_earlier[name]["rows"] += B
            del du, dsess, pos, rows, comp
        if gi == 0:
            extras = _census_extras(g, root, hsV, hsT, tid, vid, rng, batches_equal, merged_decision_set,
                                    CleanV2Store, HistoryStore, VALIDATION_SPLITS, SESSION_L50, USER_HISTORY_L50)
        glog.append({"buckets": [g[0], g[-1]], "store_rows": int(cs.n_rows), "pool_train": int(tid.size),
                     "pool_val": int(vid.size), "s": round(time.time() - tg, 1), "rss_bytes": rss_bytes()})
        log(f"group {g[0]}-{g[-1]}: rows {cs.n_rows} pool TRAIN {tid.size} VAL {vid.size} "
            f"({glog[-1]['s']} s, rss {glog[-1]['rss_bytes']})")
        del hsV, hsT, ix, cs
        _trim()
    cat = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in acc.items()}
    uT, sT, uV, sV = (cat[k].astype(np.int64) for k in ("uT", "sT", "uV", "sV"))
    rec["reproduction"] = {
        "pool_users": [tot["pool_users"], POOL["users"], tot["pool_users"] == POOL["users"]],
        "pool_train_examples": [tot["pool_train_examples"], POOL["train_examples"],
                                tot["pool_train_examples"] == POOL["train_examples"]],
        "pool_val_eval_rows": [tot["pool_val_eval_rows"], POOL["val_eval_rows"],
                               tot["pool_val_eval_rows"] == POOL["val_eval_rows"]],
        "all_elig": [tot["all_elig"], ALL["elig"], tot["all_elig"] == ALL["elig"]],
        "all_val_eval": [tot["all_val_eval"], ALL["val_eval"], tot["all_val_eval"] == ALL["val_eval"]],
        "users_with_a_non_test_decision": tot["users"]}
    vcode = cat["vcode"].astype(np.int64)
    rec["window_length_census"] = {
        "TRAIN_pool_supervised": {"SESSION_L50": length_stats(sT), "USER_HISTORY_L50": length_stats(uT)},
        "VALIDATION_pool_eval_rows": {"SESSION_L50": length_stats(sV), "USER_HISTORY_L50": length_stats(uV)},
        "VALIDATION_by_subperiod": {
            "VAL_A_VALIDATION_SELECT": {"SESSION_L50": length_stats(sV[vcode == 1]),
                                        "USER_HISTORY_L50": length_stats(uV[vcode == 1])},
            "VAL_B_VALIDATION_CONFIRM": {"SESSION_L50": length_stats(sV[vcode == 2]),
                                         "USER_HISTORY_L50": length_stats(uV[vcode == 2])}}}
    trV = cat["trV"].astype(np.int64)
    rec["validation_train_period"] = {
        "share_of_VALIDATION_eval_rows_whose_USER_window_includes_TRAIN_events":
            float(np.count_nonzero(trV > 0) / trV.size),
        "check_first_window_row_is_TRAIN_iff_window_has_TRAIN_rows": bool(np.array_equal(cat["stV"], trV > 0)),
        "share_of_USER_window_positions_that_are_TRAIN_events": float(trV.sum() / uV.sum()),
        "SESSION_L50_windows_with_TRAIN_events": int(leak["VALIDATION|SESSION_L50"]["rows_from_earlier_partitions"])}
    rec["window_composition_USER_HISTORY_L50"] = {
        "TRAIN": composition_stats({"n_segments": cat["segT"].astype(np.int64),
                                    "n_train_rows": cat["trT"].astype(np.int64)}, uT),
        "VALIDATION": composition_stats({"n_segments": cat["segV"].astype(np.int64), "n_train_rows": trV}, uV)}
    rec["leakage_exhaustive"] = leak
    rec["monotonicity_exhaustive"] = mono
    rec["served_batch_tie_in"] = served
    rec["user_equals_session_without_earlier_segment"] = equal_no_earlier
    rec["ordering_proof"] = order
    rec["index_alignment_decisions_checked"] = align_checked
    rec["group0_extras"] = extras
    rec["group_log"] = glog
    rec["rss_bytes_end"], rec["peak_rss_bytes"] = rss_bytes(), peak_rss_bytes()
    rec["utc_end"] = utc()
    ok = all(v[2] for k, v in rec["reproduction"].items() if isinstance(v, list))
    rec["status"] = "PASS" if ok else "REPRODUCTION_MISMATCH"
    write_part("census", rec)
    log(f"census {rec['status']}")
    return 0 if ok else 3


def _census_extras(g, root, hsV, hsT, tid, vid, rng, batches_equal, merged_decision_set, CleanV2Store, HistoryStore,
                   VALIDATION_SPLITS, SESSION_L50, USER_HISTORY_L50) -> dict:
    """Group 0: SESSION on the full store == the INT-01 decision-split stores; USER TRAIN windows from a TRAIN-only
    store == from the full store (20,480 random pool rows each, bitwise)."""
    out = {}
    trs = CleanV2Store.build(splits=("TRAIN",), buckets=g, ds_root=root, n_threads=3, log=lambda *_: None)
    vas = CleanV2Store.build(splits=VALIDATION_SPLITS, buckets=g, ds_root=root, n_threads=3, log=lambda *_: None)
    for name, hs, ref, splits, kind, ids in (("TRAIN", hsT, trs, ("TRAIN",), "loss", tid),
                                             ("VALIDATION", hsV, vas, VALIDATION_SPLITS, "eval", vid)):
        d = hs.raw_decision_set(kind)
        rd = merged_decision_set(ref, splits, kind)
        p1, p2 = hs.positions_for(d, ids), ref.positions_for(rd, ids)
        n = 0
        for _ in range(80):
            k = rng.choice(p1.size, B, replace=False)
            ok, why = batches_equal(hs.batch(d, p1[k]), ref.batch(rd, p2[k]))
            if not ok:
                raise AssertionError(f"{name}: SESSION_L50 != the INT-01 decision-split store ({why})")
            n += B
        out[f"SESSION_equals_INT01_store_{name}_rows"] = n
    from v4hist.windows import build_row_index
    ht = HistoryStore(trs, build_row_index(root, trs.cs.buckets, ("TRAIN",)), USER_HISTORY_L50, ("TRAIN",))
    a_, b_ = ht.decision_set("loss"), hsT.decision_set("loss")
    p1, p2 = ht.positions_for(a_, tid), hsT.positions_for(b_, tid)
    n = 0
    for _ in range(80):
        k = rng.choice(p1.size, B, replace=False)
        ok, why = batches_equal(ht.batch(a_, p1[k]), hsT.batch(b_, p2[k]))
        if not ok:
            raise AssertionError(f"USER TRAIN windows depend on the store's later partitions ({why})")
        n += B
    out["USER_TRAIN_train_only_store_equals_full_store_rows"] = n
    del trs, vas, ht
    return out


# ------------------------------------------------------------------------------------------------ session features
def cmd_session_features(a) -> int:
    """DECISION_LOG 47/48 on the server: every pool VALIDATION eval decision of the buckets (and the pool TRAIN
    decisions of the first group): session_offset_tokens with the exhaustive proof, the session-mode features with a
    naive-reference sample, their distributions, and a cache-vs-stream tie-in of the served keys. Counts only."""
    import ppsi_env
    import v4train  # noqa: F401
    import numpy as np
    import torch
    sys.path.insert(0, str(CODE / "tests_hist"))
    from test_session_modes import check_against_naive                   # the naive reference (tests_hist)
    from v4data.common import VALIDATION_SPLITS, dataset_root
    from v4hist.census import row_defs
    from v4hist.stream import HistoryStore
    from v4hist.windows import (MAIN_UNKNOWN, MATCH_KEY, MATCH_MASK_KEY, SESSION_MODE_KEY, SHARE_KEY, USER_ONLY_KEYS,
                                check_session_offsets, session_mode_features, session_offset_tokens, window_rows)
    from v4int.cache import build_subset_store, peak_rss_bytes, rss_bytes
    from v4int.providers import add_session_offsets
    torch.set_num_threads(4)
    ppsi_env.verify_snapshot()
    root = dataset_root()
    rec = {"step": "session_features", "label": "COUNTS_ONLY (no model outcome; TEST never read)", "utc_start": utc(),
           "resources_before": resources()}
    allb = _buckets(a.buckets)
    groups = [allb[i:i + 4] for i in range(0, len(allb), 4)]
    defs = row_defs()
    rng = np.random.default_rng(47)
    acc = {s: {"tok_counts": np.zeros(8, np.int64), "n": 0, "with_earlier": 0, "sessions_in_window": [],
               "avail": [], "match": [], "share": [], "unk_pos": 0, "pos": 0, "unk_cur": 0, "naive_checked": 0,
               "checked": 0} for s in ("TRAIN", "VALIDATION")}
    CH = 100_000
    for gi, g in enumerate(groups):
        tid, vid, _cnt = pool_ids(defs, root, g)
        hsV = HistoryStore.build("USER_HISTORY_L50", VALIDATION_SPLITS, buckets=g, ds_root=root, n_threads=3,
                                 log=lambda *_: None)
        hsT = HistoryStore(hsV.store, hsV.index, "USER_HISTORY_L50", ("TRAIN",))
        ix = hsV.index
        todo = [("VALIDATION", hsV, "eval", vid)] + ([("TRAIN", hsT, "loss", tid)] if gi == 0 else [])
        for name, hs, kind, ids in todo:
            ds = hs.decision_set(kind)
            pos = hs.positions_for(ds, ids)
            A = acc[name]
            for s0 in range(0, pos.size, CH):
                p = pos[s0:s0 + CH]
                end, ln = ds.end_row[p], ds.length[p]
                tok = session_offset_tokens(ix.session_id, end, ln)
                idx, valid = window_rows(end, ln)
                rep = check_session_offsets(ix, ix.rows_of(ds.decision_id[p]), idx, valid, tok)
                A["tok_counts"] += np.asarray(rep["token_counts_1_to_8"], dtype=np.int64)
                A["n"] += int(p.size)
                A["with_earlier"] += rep["decisions_with_an_earlier_session"]
                A["sessions_in_window"].append(tok.max(axis=1).astype(np.int8))
                ic = np.where(valid, idx, 0)
                mt = np.where(valid, hs.store.cs._main_category_tokens[ic].astype(np.int64), 0)
                sp = ix.session_id.astype(np.int64)[ic]
                f = session_mode_features(mt, valid, sp)
                A["avail"].append(f[MATCH_MASK_KEY] > 0)
                A["match"].append(f[MATCH_KEY])
                A["share"].append(f[SHARE_KEY])
                A["unk_pos"] += int(np.count_nonzero(f[SESSION_MODE_KEY][valid] == MAIN_UNKNOWN))
                A["pos"] += int(valid.sum())
                lp = valid.sum(axis=1) - 1
                A["unk_cur"] += int(np.count_nonzero(f[SESSION_MODE_KEY][np.arange(p.size), lp] == MAIN_UNKNOWN))
                A["checked"] += int(p.size)
                k = rng.choice(p.size, min(512, p.size), replace=False)
                check_against_naive(mt[k], valid[k], sp[k], {kk: vv[k] for kk, vv in f.items()})
                A["naive_checked"] += int(k.size)
        if gi == 0:                                      # served keys: the provider cache == the stream, bitwise
            store, cds, _info = build_subset_store(vid, splits=VALIDATION_SPLITS, kind="eval", state_sha256=None,
                                                   history_policy="USER_HISTORY_L50", buckets=g, group_size=4,
                                                   n_threads=3, log=lambda *_: None)
            sds = hsV.decision_set("eval")
            n_eq = 0
            for _ in range(64):
                q = rng.choice(len(cds), B, replace=False)
                bc = add_session_offsets(store.batch(cds, q), store, cds, q)
                bs = hsV.batch(sds, hsV.positions_for(sds, cds.decision_id[q]))
                for kk in USER_ONLY_KEYS:
                    if not torch.equal(bc[kk], bs[kk]):
                        raise AssertionError(f"served {kk}: cache != stream")
                n_eq += B
            rec["cache_equals_stream_rows"] = n_eq
            del store, cds
        log(f"group {g[0]}-{g[-1]}: VAL {vid.size} done (rss {rss_bytes()})")
        del hsV, hsT, ix
        _trim()
    out = {}
    for name, A in acc.items():
        if not A["n"]:
            continue
        av = np.concatenate(A["avail"])
        sh = np.concatenate(A["share"])[av]
        swin = np.concatenate(A["sessions_in_window"]).astype(np.int64)
        tc = A["tok_counts"]
        out[name] = {
            "n_decisions": A["n"], "exhaustive_offset_proof_decisions": A["checked"],
            "naive_reference_checked_decisions": A["naive_checked"],
            "offset_token_share_of_positions_1_to_8": (tc / tc.sum()).round(6).tolist(),
            "share_of_decisions_with_an_earlier_session": A["with_earlier"] / A["n"],
            "sessions_in_window_capped_at_8_quantiles": {q: float(np.percentile(swin, q)) for q in (10, 25, 50, 75, 90)},
            "match_available_rate": float(av.mean()),
            "match_rate_among_available": float(np.concatenate(A["match"])[av].mean()) if av.any() else None,
            "share_among_available": ({"mean": float(sh.mean()), **{f"p{q}": float(np.percentile(sh, q))
                                                                   for q in (10, 25, 50, 75, 90)}} if sh.size else None),
            "unknown_rate_positions": A["unk_pos"] / A["pos"],
            "unknown_rate_current_mode_at_decision": A["unk_cur"] / A["n"]}
    rec["distributions"] = out
    rec["rss_bytes_end"], rec["peak_rss_bytes"] = rss_bytes(), peak_rss_bytes()
    rec["utc_end"] = utc()
    write_part("session_features", rec)
    return 0


# ------------------------------------------------------------------------------------------------ provider
def cmd_provider(a) -> int:
    import ppsi_env
    import v4train  # noqa: F401
    import numpy as np
    import torch
    from v4data.common import VALIDATION_SPLITS, dataset_root
    from v4hist.census import row_defs
    from v4int.cache import build_subset_store, peak_rss_bytes, rss_bytes
    from v4train.order import pass_permutation
    torch.set_num_threads(4)
    ppsi_env.verify_snapshot()
    name = f"provider_{a.source}_{a.kind}_{a.policy}" + (f"_{a.manifest}" if a.source == "manifest" else "")         + ("_with_transform_state" if a.transform_state else "") + (f"_{a.tag}" if a.tag else "")
    rec = {"step": name, "label": ("HISTORICAL_POPULATION (data-path evidence only)" if a.source == "manifest"
                                   else "5-100 POOL sample (pool users of the given buckets)"),
           "utc_start": utc(), "resources_before": resources(), "rss_bytes_start": rss_bytes(),
           "history_policy": a.policy, "kind": a.kind}
    t0 = time.time()
    if a.source == "pool":
        bl = _buckets(a.buckets)
        tid, vid, cnt = pool_ids(row_defs(), dataset_root(), bl)
        rec["pool_counts"], rec["buckets"] = cnt, [bl[0], bl[-1]]
        ids = tid if a.kind == "train" else vid
        t1 = time.time()
        store, ds, info = build_subset_store(ids, splits=("TRAIN",) if a.kind == "train" else VALIDATION_SPLITS,
                                             kind="loss" if a.kind == "train" else "eval",
                                             state_sha256=_lock_state_sha(), history_policy=a.policy, buckets=bl,
                                             group_size=8, n_threads=3, log=log)
        rec["pool_scan_s"], rec["build_s"] = round(t1 - t0, 2), round(time.time() - t1, 2)
        fn = lambda pos: store.batch(ds, pos)  # noqa: E731
        N, cinfo = len(ds), info.as_json()
        rec["describe"] = None
    else:
        from v4int.providers import StoreEvalProvider, StoreTrainProvider
        if a.kind == "train":
            p = StoreTrainProvider(a.manifest, history_policy=a.policy, log=log,
                                   transform_state=Path(a.transform_state) if a.transform_state else None)
            fn, N = p.batch, p.n_decisions
        else:
            p = StoreEvalProvider(a.manifest, history_policy=a.policy, prefetch=0, log=log,
                                  transform_state=Path(a.transform_state) if a.transform_state else None)
            fn, N = ((lambda pos: p.renorm.apply(p.store.batch(p.ds, pos))) if p.renorm is not None
                     else (lambda pos: p.store.batch(p.ds, pos))), p.n_stream_rows
        rec["build_s"] = round(time.time() - t0, 2)
        cinfo, rec["describe"] = p.cache_info.as_json(), p.describe()
    rec["rss_bytes_after_build"], rec["peak_rss_bytes_after_build"] = rss_bytes(), peak_rss_bytes()
    rec["n_decisions"], rec["cache"] = int(N), cinfo
    rec["bytes_per_row"] = round(cinfo["cache_nbytes"] / max(N, 1), 1)
    rec["event_rows_per_decision"] = round(cinfo["n_event_rows"] / max(N, 1), 3)
    perm = pass_permutation(2026, 0, N)
    nb = min(int(a.batches), N // B - 20)
    for i in range(20):
        fn(perm[i * B:(i + 1) * B])
    ts, lens = [], []
    for i in range(20, 20 + nb):
        pos = perm[i * B:(i + 1) * B]
        t = time.perf_counter()
        b = fn(pos)
        ts.append(time.perf_counter() - t)
        lens.append(float(b["lengths"].float().mean()))
    st = _stats(ts)
    rec["B256_random_trainer_order"] = dict(st, rows=nb * B, rows_per_s=round(nb * B / st["total_s"], 1),
                                            mean_window_length=round(float(np.mean(lens)), 3))
    if a.kind == "eval":
        t = time.perf_counter()
        n = 0
        for s in range(0, N, 1024):
            b = fn(np.arange(s, min(N, s + 1024), dtype=np.int64))
            n += int(b["decision_id"].shape[0])
        wall = time.perf_counter() - t
        rec["B1024_in_order_all_rows"] = {"rows": n, "wall_s": round(wall, 3), "rows_per_s": round(n / wall, 1)}
    rec["rss_bytes_end"], rec["peak_rss_bytes_end"] = rss_bytes(), peak_rss_bytes()
    rec["utc_end"] = utc()
    write_part(name, rec)
    log(f"{name}: {rec['B256_random_trainer_order']['rows_per_s']} rows/s, {rec['bytes_per_row']} B/row")
    return 0


# ------------------------------------------------------------------------------------------------ bench (GPU)
def _gpu_free() -> tuple:
    import v4int.launch_central as L
    from v4sup.gpu import default_runner, gpu_ready
    ok, why = gpu_ready([GPU])
    if not ok:
        return False, why
    try:
        mem = L._gpu_memory_used(GPU, default_runner)
    except Exception as e:
        return False, f"nvidia-smi: {e}"
    used = mem[str(GPU)]["memory_used_mib"]
    return used <= L.GPU_MEM_FOREIGN_MIB, f"memory.used {used} MiB"


def cmd_bench(a) -> int:
    import v4int.launch_central as L                                   # no torch / snapshot import
    if a.local_dry_run:
        return _bench_body(a, L, 0, "cpu")
    want = {}
    L.export_environment(GPU, want)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != want["CUDA_VISIBLE_DEVICES"] or \
            os.environ.get("CUBLAS_WORKSPACE_CONFIG") != want["CUBLAS_WORKSPACE_CONFIG"]:
        env = {k: v for k, v in os.environ.items() if k not in ("CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG")}
        env.update(want)
        os.execve(sys.executable, [sys.executable, "-B", str(Path(__file__).resolve()), *sys.argv[1:]], env)
    waited = 0
    while True:
        ok, why = _gpu_free()
        if ok:
            break
        if waited >= int(a.max_wait_min):
            write_part("bench", {"step": "bench", "skipped": f"GPU {GPU} not free after {waited} min: {why}",
                                 "utc": utc()})
            return 4
        log(f"GPU {GPU} busy ({why}); waiting (never touching a foreign process)")
        time.sleep(60)
        waited += 1
    return _bench_body(a, L, waited, "cuda")


def _bench_body(a, L, waited: int, kind: str) -> int:
    import v4train
    import numpy as np
    import torch
    from fl.numerics import enforce_strict_fp32
    from v4data.common import dataset_root
    from v4hist.census import row_defs
    from v4int.cache import build_subset_store, peak_rss_bytes, rss_bytes
    from v4models.build import build
    from v4models.schema import load_schema
    from v4train.objective import accumulate_step, host_contributing
    from v4train.order import pass_permutation
    from v4train.recipe import RECIPE
    if kind == "cuda":
        if v4train.LAUNCH_CUDA_VISIBLE_DEVICES != str(GPU) or torch.cuda.device_count() != 1:
            raise RuntimeError("CUDA_VISIBLE_DEVICES was not exported before the import, or more than one GPU is "
                               "visible")
        dev = torch.device("cuda:0")
        torch.cuda.set_device(dev)
    else:
        dev = torch.device("cpu")
        cu = types.SimpleNamespace(synchronize=lambda *_: None, reset_peak_memory_stats=lambda *_: None,
                                   max_memory_allocated=lambda *_: 0, max_memory_reserved=lambda *_: 0,
                                   empty_cache=lambda: None, get_device_name=lambda *_: "cpu")
        torch = types.SimpleNamespace(**{k: getattr(__import__("torch"), k) for k in ("device", "manual_seed",
                                                                                        "is_tensor", "optim", "nn")},
                                      cuda=cu)
    rec = {"step": "bench", "label": "BENCH_NOT_EVIDENCE (no weights saved, no result row; <= 300 updates/family)",
           "utc_start": utc(), "resources_before": resources(), "gpu_wait_min": waited,
           "device": {"physical_gpu": GPU, "name": torch.cuda.get_device_name(dev),
                      "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                      "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}}
    bl = _buckets(a.buckets)
    tid, _vid, cnt = pool_ids(row_defs(), dataset_root(), bl)
    rec["pool_counts"], rec["buckets"] = cnt, [bl[0], bl[-1]]
    caches = {}
    for pol in ("USER_HISTORY_L50", "SESSION_L50"):
        st, ds, info = build_subset_store(tid, splits=("TRAIN",), kind="loss", state_sha256=_lock_state_sha(),
                                          history_policy=pol, buckets=bl, group_size=8, n_threads=3, log=log)
        caches[pol] = (st, ds)
    N = int(tid.size)
    perm = pass_permutation(2026, 0, N)
    NB = 64
    pos_b = [perm[i * B:(i + 1) * B] for i in range(NB)]
    cpu = {pol: [caches[pol][0].batch(caches[pol][1], p) for p in pos_b] for pol in caches}
    rec["mean_window_length"] = {pol: round(float(np.mean([float(b["lengths"].float().mean()) for b in cpu[pol]])), 3)
                                 for pol in cpu}
    to_gpu = lambda b: {k: (v.to(dev) if torch.is_tensor(v) and k != "lengths" else v) for k, v in b.items()}  # noqa
    gpu = {pol: [to_gpu(b) for b in cpu[pol]] for pol in cpu}
    n_eff = [host_contributing(b) for b in cpu["USER_HISTORY_L50"]]
    schema = load_schema(CODE / "v4data" / "frozen_inputs" / "FEATURE_SCHEMA_CLEAN_V2.json")
    r = RECIPE
    W, T = (15, 95) if kind == "cuda" else (1, 2)
    plan = [("warmup_USER", W), ("USER_HISTORY_L50_preloaded", T), ("SESSION_L50_preloaded", T),
            ("USER_HISTORY_L50_cache_in_loop", T)]
    rec["updates_per_family"] = sum(n for _, n in plan)
    if rec["updates_per_family"] > 300:
        raise RuntimeError("the benchmark cap is 300 updates per family")
    rec["families"] = {}
    ust, uds = caches["USER_HISTORY_L50"]
    for fam in ("GRU", "SASREC"):
        bm = build(fam, schema, 2026, dev, v1_root=dataset_root())
        ad = bm.adapter
        opt = torch.optim.AdamW(ad.param_groups(r.weight_decay), lr=1e-3, betas=r.betas, eps=r.eps, fused=r.fused)
        shared = ad.shared_parameters()
        enforce_strict_fp32(deterministic=True, warn_only=False)            # STRICT (the proposed lock)
        bm.module.train()
        torch.manual_seed(2026)

        def step(batch, n):
            opt.zero_grad(set_to_none=True)
            ls = accumulate_step(ad.scores, batch, n, 0, dev)
            torch.nn.utils.clip_grad_norm_(shared, r.grad_clip)
            opt.step()
            ad.post_step()
            return ls

        fr = {"determinism": L.determinism_mode_in_effect()}
        done = 0
        for mode, n_steps in plan:
            torch.cuda.synchronize(dev)
            torch.cuda.reset_peak_memory_stats(dev)
            t0 = time.perf_counter()
            for i in range(n_steps):
                j = (done + i) % NB
                if mode in ("warmup_USER", "USER_HISTORY_L50_preloaded"):
                    ls = step(gpu["USER_HISTORY_L50"][j], n_eff[j])
                elif mode == "SESSION_L50_preloaded":
                    ls = step(gpu["SESSION_L50"][j], host_contributing(cpu["SESSION_L50"][j]))
                else:
                    k = (NB + done + i) % (N // B)
                    b = ust.batch(uds, perm[k * B:(k + 1) * B])
                    ls = step(b, host_contributing(b))
                if (i + 1) % 50 == 0:
                    float(ls.item())
            torch.cuda.synchronize(dev)
            wall = time.perf_counter() - t0
            done += n_steps
            fr[mode] = {"updates": n_steps, "wall_s": round(wall, 3), "updates_per_s": round(n_steps / wall, 3),
                        "decisions_per_s": round(n_steps * B / wall, 1),
                        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(dev)),
                        "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(dev))}
            log(f"{fam} {mode}: {fr[mode]['updates_per_s']} upd/s")
        fr["updates_total"] = done
        u, s_ = fr["USER_HISTORY_L50_preloaded"]["updates_per_s"], fr["SESSION_L50_preloaded"]["updates_per_s"]
        fr["user_over_session_rate"] = round(u / s_, 4)
        fr["cache_in_loop_over_preloaded"] = round(fr["USER_HISTORY_L50_cache_in_loop"]["updates_per_s"] / u, 4)
        rec["families"][fam] = fr
        del bm, ad, opt, shared
        torch.cuda.empty_cache()
    rec["weights_saved"] = False
    rec["host_rss_bytes"], rec["host_peak_rss_bytes"] = rss_bytes(), peak_rss_bytes()
    rec["resources_after"] = resources()
    rec["utc_end"] = utc()
    write_part("bench" if kind == "cuda" else "bench_local_dry_run", rec)
    return 0


# ------------------------------------------------------------------------------------------------ assemble
def cmd_assemble(a) -> int:
    parts = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(PARTS.glob("*.json"))}
    steps = []
    sp = OUT / "steps.jsonl"
    if sp.exists():
        steps = [json.loads(x) for x in sp.read_text(encoding="utf-8").splitlines() if x.strip()]
    tests = OUT / "tests" / "TEST_RUN.json"
    rec = {"receipt": "DATA05_SERVER_RECEIPT", "task": "DATA-05", "host": platform.node(), "utc_assembled": utc(),
           "tags": "PROJECT_MEASURED (server, this run)", "gpu": f"physical GPU {GPU} only (bench)",
           "labels": {"census": "COUNTS_ONLY on the 5-100 pool; no model outcome; TEST never read",
                      "bench": "BENCH_NOT_EVIDENCE", "provider_manifest": "HISTORICAL_POPULATION data-path evidence"},
           "code_sha256": {str(p.relative_to(CODE)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
                           for d in ("v4hist", "tests_hist", "v4int", "tests_int") for p in sorted((CODE / d).rglob("*"))
                           if p.is_file() and p.suffix in (".py", ".sh", ".ini") and "__pycache__" not in p.parts},
           "server_tests": (json.loads(tests.read_text(encoding="utf-8")) if tests.exists() else None),
           "steps": steps, "parts": parts,
           "pt_or_npy_files_left_under_data05": [str(p) for p in OUT.rglob("*") if p.suffix in (".pt", ".npy")]}
    tmp = OUT / "DATA05_SERVER_RECEIPT.json.tmp"
    tmp.write_text(json.dumps(rec, indent=1, allow_nan=False, default=str), encoding="utf-8")
    os.replace(tmp, OUT / "DATA05_SERVER_RECEIPT.json")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("census")
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--buckets", default="0-63")
    p.add_argument("--served-batches", type=int, default=64)
    p.add_argument("--equal-batches", type=int, default=16)
    p = sub.add_parser("provider")
    p.add_argument("--source", choices=("pool", "manifest"), required=True)
    p.add_argument("--kind", choices=("train", "eval"), required=True)
    p.add_argument("--policy", choices=("USER_HISTORY_L50", "SESSION_L50"), required=True)
    p.add_argument("--buckets", default="0-7")
    p.add_argument("--manifest", default=None)
    p.add_argument("--batches", type=int, default=2000)
    p.add_argument("--transform-state", default=None, help="manifest source only: a TRANSFORM_STATE path")
    p.add_argument("--tag", default=None, help="appended to the part name")
    p = sub.add_parser("bench")
    p.add_argument("--buckets", default="0-3")
    p.add_argument("--max-wait-min", type=int, default=60)
    p.add_argument("--local-dry-run", action="store_true", help="CPU, 2 steps per mode (code check only)")
    sub.add_parser("assemble")
    p = sub.add_parser("session-features")
    p.add_argument("--buckets", default="0-7")
    a = ap.parse_args(argv)
    return {"census": cmd_census, "provider": cmd_provider, "bench": cmd_bench, "assemble": cmd_assemble,
            "session-features": cmd_session_features}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
