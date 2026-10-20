"""test_full_store.py — launch-gate suite for CompactStore (Sonnet A, Wave 1).

Runnable as a script (writes a summary JSON to `$W/tests_full/A/gates_A.json` and stdout) or
under pytest (`test_all_gates_smoke`, which just runs the same thing and asserts overall pass).
Never loads VALIDATION_CONFIRM or TEST decisions or rows.

Gates implemented (see FULL_RUN_PROTOCOL.md #10 and the Sonnet-A task brief):
  - class map (`catalogue.product_idx == arange(158486)+3`)
  - population counts (TRAIN loss / VS eval / VS distinct eval users) + zero VC/TEST rows
  - gate (i): byte-exact parity vs the pilot window cache, over the full 1,000,000-row TRAIN
    manifest and 112,365-row VALIDATION_SELECT cohort (F4), plus an F0 subset check
  - gate (ii): byte-exact parity vs a fresh `DecisionWindowLoader.collate()`, >=4096 random
    decisions from >=8 buckets plus targeted edge cases
  - negative controls (shifted window, perturbed normaliser) that must be DETECTED as mismatches
  - T11: real cold-user fixture replacing the vacuous `post_freeze_smoke.py:101` check
  - feature invariants on a 100,000-decision TRAIN sample (report-only)
  - normaliser recomputation over the full TRAIN population (report-only, hard-fail > 1e-2)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

CODE_DIR = Path(__file__).resolve().parent
VENDOR_DIR = CODE_DIR / "vendor"
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import full_store as fs  # noqa: E402
from full_store import CompactStore, configure_threads  # noqa: E402
from index_utils import read_index_projected  # noqa: E402
from rees46_eda.loaders import DecisionWindowLoader, detect_layout  # noqa: E402
from rees46_eda.config import OOV, DAYPARTS, daypart_index  # noqa: E402
from window_cache import ARRAY_NAMES, CacheReader, collate_rows  # noqa: E402

configure_threads(3)


# ---------------------------------------------------------------------------
# comparison helpers
# ---------------------------------------------------------------------------

def arrays_bit_equal(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        return False
    if np.issubdtype(a.dtype, np.floating) or np.issubdtype(b.dtype, np.floating):
        af = a.astype(np.float32, copy=False)
        bf = b.astype(np.float32, copy=False)
        return np.array_equal(af.view(np.uint32), bf.view(np.uint32))
    return np.array_equal(a, b)


def _to_numpy(v):
    return v.numpy() if hasattr(v, "numpy") else np.asarray(v)


def compare_batch_dicts(cached: dict, computed: dict, array_names, int32_cast_names) -> list:
    mismatches = []
    for name in array_names:
        if name not in cached or name not in computed:
            mismatches.append(f"{name}: missing from one side (cached={name in cached}, "
                               f"computed={name in computed})")
            continue
        c = _to_numpy(cached[name])
        d = _to_numpy(computed[name])
        if name in int32_cast_names:
            if c.dtype != np.int64:
                c = c.astype(np.int64)
            if d.dtype != np.int64:
                d = d.astype(np.int64)
        if c.shape != d.shape:
            mismatches.append(f"{name}: shape {c.shape} != {d.shape}")
            continue
        if c.dtype != d.dtype:
            mismatches.append(f"{name}: dtype {c.dtype} != {d.dtype}")
            continue
        if not arrays_bit_equal(c, d):
            mismatches.append(f"{name}: values differ")
    return mismatches


# ---------------------------------------------------------------------------
# gate: class map
# ---------------------------------------------------------------------------

def gate_class_map(dataset_root: Path) -> dict:
    cat = pq.read_table(Path(dataset_root) / "mappings" / "catalogue.parquet", columns=["product_idx"])
    pidx = cat["product_idx"].to_numpy(zero_copy_only=False)
    expected = np.arange(pidx.shape[0]) + 3
    ok = np.array_equal(pidx, expected)
    return {"n": int(pidx.shape[0]), "expected_n": 158486, "matches_arange_plus_3": bool(ok),
            "pass": bool(ok) and pidx.shape[0] == 158486}


# ---------------------------------------------------------------------------
# gate: population counts
# ---------------------------------------------------------------------------

def gate_counts(store_f4: CompactStore) -> dict:
    ds_train = store_f4.decisions("TRAIN", "loss")
    ds_vs = store_f4.decisions("VALIDATION_SELECT", "eval")
    distinct_users = int(np.unique(ds_vs.eval_user_key).shape[0]) if len(ds_vs) else 0

    rejects_others = True
    for sp in ("VALIDATION_CONFIRM", "TEST"):
        try:
            store_f4.decisions(sp, "loss")
            rejects_others = False
        except ValueError:
            pass

    return {
        "train_loss_n": len(ds_train), "expected_train_loss_n": 40180608,
        "vs_eval_n": len(ds_vs), "expected_vs_eval_n": 8826488,
        "vs_distinct_eval_users": distinct_users, "expected_vs_distinct_eval_users": 790459,
        "store_rejects_vc_test_splits": rejects_others,
        "pass": (len(ds_train) == 40180608 and len(ds_vs) == 8826488
                 and distinct_users == 790459 and rejects_others),
    }


# ---------------------------------------------------------------------------
# gate (i): parity vs the pilot window cache
# ---------------------------------------------------------------------------

def gate_parity_i(store_f4: CompactStore, manifest_path: Path, cohort_path: Path,
                   cache_train_dir: Path, cache_vs_dir: Path, chunk: int = 50_000) -> dict:
    reader_train = CacheReader(cache_train_dir)
    reader_vs = CacheReader(cache_vs_dir)

    man = pq.read_table(manifest_path, columns=["decision_id"])
    man_dec = man["decision_id"].to_numpy(zero_copy_only=False)
    coh = pq.read_table(cohort_path, columns=["decision_id", "eval_user_key"])
    coh_dec = coh["decision_id"].to_numpy(zero_copy_only=False)
    coh_key = coh["eval_user_key"].to_numpy(zero_copy_only=False)

    ds_train = store_f4.decisions("TRAIN", "loss")
    ds_vs = store_f4.decisions("VALIDATION_SELECT", "eval")

    pos_train = np.searchsorted(ds_train.decision_id, man_dec)
    assert np.array_equal(ds_train.decision_id[pos_train], man_dec), (
        "TRAIN manifest decision_id not found in store's TRAIN loss DecisionSet")
    pos_vs = np.searchsorted(ds_vs.decision_id, coh_dec)
    assert np.array_equal(ds_vs.decision_id[pos_vs], coh_dec), (
        "VS cohort decision_id not found in store's VS eval DecisionSet")

    store_keys = ds_vs.eval_user_key[pos_vs]
    key_mismatches = int(np.sum(store_keys != coh_key))

    train_mismatches = []
    n_compared_train = 0
    for lo in range(0, man_dec.shape[0], chunk):
        hi = min(lo + chunk, man_dec.shape[0])
        rows = np.arange(lo, hi)
        cached = reader_train.read(rows)
        computed = store_f4.batch(ds_train, pos_train[lo:hi])
        mism = compare_batch_dicts(cached, computed, ARRAY_NAMES,
                                    reader_train.header.get("int32_cast_names", []))
        if mism:
            train_mismatches.append({"chunk": [int(lo), int(hi)], "mismatches": mism})
        n_compared_train += (hi - lo)

    vs_mismatches = []
    n_compared_vs = 0
    for lo in range(0, coh_dec.shape[0], chunk):
        hi = min(lo + chunk, coh_dec.shape[0])
        rows = np.arange(lo, hi)
        cached = reader_vs.read(rows)
        computed = store_f4.batch(ds_vs, pos_vs[lo:hi])
        mism = compare_batch_dicts(cached, computed, ARRAY_NAMES,
                                    reader_vs.header.get("int32_cast_names", []))
        if mism:
            vs_mismatches.append({"chunk": [int(lo), int(hi)], "mismatches": mism})
        n_compared_vs += (hi - lo)

    return {
        "n_compared_train": n_compared_train, "expected_n_train": 1_000_000,
        "n_compared_vs": n_compared_vs, "expected_n_vs": 112_365,
        "n_arrays_checked": len(ARRAY_NAMES),
        "train_mismatches": train_mismatches, "vs_mismatches": vs_mismatches,
        "eval_user_key_mismatches": key_mismatches,
        "pass": (not train_mismatches and not vs_mismatches and key_mismatches == 0
                 and n_compared_train == 1_000_000 and n_compared_vs == 112_365),
    }


def gate_parity_i_f0_subset(store_f0: CompactStore, manifest_path: Path, cohort_path: Path,
                             cache_train_dir: Path, cache_vs_dir: Path,
                             n_sample: int = 20_000, seed: int = 20260923) -> dict:
    reader_train = CacheReader(cache_train_dir)
    reader_vs = CacheReader(cache_vs_dir)
    keys = ["item_tokens", "attention_mask", "position_ids", "lengths", "last_valid_index",
            "target_class", "loss_mask", "eval_mask", "decision_id", "client_id"]

    man_dec = pq.read_table(manifest_path, columns=["decision_id"])["decision_id"].to_numpy(zero_copy_only=False)
    coh_dec = pq.read_table(cohort_path, columns=["decision_id"])["decision_id"].to_numpy(zero_copy_only=False)

    rng = np.random.default_rng(seed)
    n_t = min(n_sample, man_dec.shape[0])
    n_v = min(n_sample, coh_dec.shape[0])
    rows_t = np.sort(rng.choice(man_dec.shape[0], n_t, replace=False))
    rows_v = np.sort(rng.choice(coh_dec.shape[0], n_v, replace=False))

    ds_train = store_f0.decisions("TRAIN", "loss")
    ds_vs = store_f0.decisions("VALIDATION_SELECT", "eval")
    pos_t = np.searchsorted(ds_train.decision_id, man_dec[rows_t])
    assert np.array_equal(ds_train.decision_id[pos_t], man_dec[rows_t])
    pos_v = np.searchsorted(ds_vs.decision_id, coh_dec[rows_v])
    assert np.array_equal(ds_vs.decision_id[pos_v], coh_dec[rows_v])

    cached_t = reader_train.read(rows_t)
    computed_t = store_f0.batch(ds_train, pos_t)
    mism_t = compare_batch_dicts(cached_t, computed_t, keys, reader_train.header.get("int32_cast_names", []))

    cached_v = reader_vs.read(rows_v)
    computed_v = store_f0.batch(ds_vs, pos_v)
    mism_v = compare_batch_dicts(cached_v, computed_v, keys, reader_vs.header.get("int32_cast_names", []))

    return {"n_sample_train": int(n_t), "n_sample_vs": int(n_v),
            "mismatches_train": mism_t, "mismatches_vs": mism_v,
            "pass": (not mism_t) and (not mism_v)}


# ---------------------------------------------------------------------------
# gate (ii): parity vs authoritative DecisionWindowLoader.collate()
# ---------------------------------------------------------------------------

def _find_edge_cases(dataset_root: Path, layout: str, max_buckets: int = 20) -> dict:
    found = {}
    cases = []
    remaining = {"pos1", "pos49", "pos50", "pos51plus", "seg_first", "seg_last",
                 "crossing_train_vs", "loss_false_context", "unseen_user"}

    for b in range(max_buckets):
        if not remaining:
            break
        path = Path(dataset_root) / "model_ready" / f"user_bucket={b}" / "features.parquet"
        if not path.exists():
            continue
        tbl = pq.read_table(path, columns=["user_id", "segment_index", "position_in_segment",
                                            "decision_id", "split"])
        user_id = tbl["user_id"].to_numpy(zero_copy_only=False)
        segment_index = tbl["segment_index"].to_numpy(zero_copy_only=False)
        position_in_segment = tbl["position_in_segment"].to_numpy(zero_copy_only=False)
        decision_id = tbl["decision_id"].to_numpy(zero_copy_only=False)
        split = tbl["split"].to_numpy(zero_copy_only=False)
        n = decision_id.shape[0]
        if n == 0:
            continue
        order = np.lexsort((position_in_segment, segment_index, user_id))
        user_id_o = user_id[order]
        seg_o = segment_index[order]
        pos_o = position_in_segment[order]
        dec_o = decision_id[order]
        split_o = split[order]
        newseg = np.empty(n, dtype=bool)
        newseg[0] = True
        if n > 1:
            newseg[1:] = (user_id_o[1:] != user_id_o[:-1]) | (seg_o[1:] != seg_o[:-1])
        seg_start = np.flatnonzero(newseg)
        seg_end = np.concatenate([seg_start[1:], [n]]) - 1

        idx_by_split = {}
        for sp in ("TRAIN", "VALIDATION_SELECT"):
            t = read_index_projected(dataset_root, layout, sp,
                                      ["decision_id", "loss_mask", "eval_mask", "client_idx"], buckets=[b])
            idx_by_split[sp] = {
                "decision_id": t["decision_id"].to_numpy(zero_copy_only=False),
                "loss_mask": t["loss_mask"].to_numpy(zero_copy_only=False),
                "eval_mask": t["eval_mask"].to_numpy(zero_copy_only=False),
                "client_idx": t["client_idx"].to_numpy(zero_copy_only=False),
            }

        # Vectorised, whole-bucket "is this row a servable decision (loss_mask|eval_mask)?"
        # and "what is its loss_mask?", computed ONCE via sort+searchsorted per split (no
        # per-row Python dict / linear scan -- this is what made the naive version too slow
        # to find rarer edge cases within a bounded bucket budget).
        is_decision_row = np.zeros(n, dtype=bool)
        loss_mask_row = np.zeros(n, dtype=bool)
        for sp in ("TRAIN", "VALIDATION_SELECT"):
            arr = idx_by_split[sp]
            rows_here = np.flatnonzero(split_o == sp)
            if rows_here.size == 0 or arr["decision_id"].shape[0] == 0:
                continue
            order2 = np.argsort(arr["decision_id"], kind="stable")
            sd = arr["decision_id"][order2]
            sl = arr["loss_mask"][order2]
            se = arr["eval_mask"][order2]
            q = dec_o[rows_here]
            p = np.searchsorted(sd, q)
            p = np.clip(p, 0, sd.shape[0] - 1)
            matched = sd[p] == q
            good_rows = rows_here[matched]
            good_p = p[matched]
            is_decision_row[good_rows] = sl[good_p] | se[good_p]
            loss_mask_row[good_rows] = sl[good_p]

        def is_decision(dec_id, sp):  # kept for the (cheap) unseen_user path below
            arr = idx_by_split.get(sp)
            if arr is None or arr["decision_id"].shape[0] == 0:
                return False
            hit = np.flatnonzero(arr["decision_id"] == dec_id)
            if hit.size == 0:
                return False
            i = hit[0]
            return bool(arr["loss_mask"][i] or arr["eval_mask"][i])

        split_ok = np.isin(split_o, ["TRAIN", "VALIDATION_SELECT"])
        servable = is_decision_row & split_ok

        def try_positions(key, mask):
            if key not in remaining:
                return
            hit = np.flatnonzero(mask & servable)
            if hit.size:
                i = int(hit[0])
                cases.append((int(dec_o[i]), b, str(split_o[i])))
                found[key] = [int(dec_o[i]), b, str(split_o[i])]
                remaining.discard(key)

        try_positions("pos1", pos_o == 1)
        try_positions("pos49", pos_o == 49)
        try_positions("pos50", pos_o == 50)
        try_positions("pos51plus", pos_o > 50)
        try_positions("seg_first", np.isin(np.arange(n), seg_start))
        try_positions("seg_last", np.isin(np.arange(n), seg_end))

        if "crossing_train_vs" in remaining:
            is_train_row = split_o == "TRAIN"
            is_vs_row = split_o == "VALIDATION_SELECT"
            train_ct = np.add.reduceat(is_train_row.astype(np.int64), seg_start)
            vs_ct = np.add.reduceat(is_vs_row.astype(np.int64), seg_start)
            crossing = np.flatnonzero((train_ct > 0) & (vs_ct > 0))
            for cs in crossing:
                s0, s1 = seg_start[cs], seg_end[cs]
                vs_local = np.flatnonzero(is_vs_row[s0:s1 + 1])
                if vs_local.size == 0:
                    continue
                i = s0 + int(vs_local[0])
                if servable[i]:
                    cases.append((int(dec_o[i]), b, "VALIDATION_SELECT"))
                    found["crossing_train_vs"] = [int(dec_o[i]), b, "VALIDATION_SELECT"]
                    remaining.discard("crossing_train_vs")
                    break

        if "loss_false_context" in remaining:
            any_false_before = np.zeros(n, dtype=bool)
            for s0, s1 in zip(seg_start, seg_end):
                if s1 - s0 < 1:
                    continue
                seg_lm = loss_mask_row[s0:s1 + 1]
                cum_false = np.cumsum(~seg_lm)
                # position j (0-indexed) qualifies if some EARLIER row (< j) had loss_mask False
                any_false_before[s0 + 1:s1 + 1] = cum_false[:-1] > 0
            hit = np.flatnonzero(any_false_before & servable)
            if hit.size:
                i = int(hit[0])
                cases.append((int(dec_o[i]), b, str(split_o[i])))
                found["loss_false_context"] = [int(dec_o[i]), b, str(split_o[i])]
                remaining.discard("loss_false_context")

        if "unseen_user" in remaining:
            for sp in ("TRAIN", "VALIDATION_SELECT"):
                arr = idx_by_split[sp]
                hit = np.flatnonzero((arr["client_idx"] == OOV) & (arr["loss_mask"] | arr["eval_mask"]))
                if hit.size:
                    cases.append((int(arr["decision_id"][hit[0]]), b, sp))
                    found["unseen_user"] = [int(arr["decision_id"][hit[0]]), b, sp]
                    remaining.discard("unseen_user")
                    break

    return {"cases": cases, "found": found, "missing": sorted(remaining)}


def gate_parity_ii(store_f4: CompactStore, dataset_root: Path, layout: str,
                    seed: int = 20260923, n_random: int = 4200, n_buckets_random: int = 10) -> dict:
    rng = np.random.default_rng(seed)
    bucket_ids = list(range(n_buckets_random))

    candidates = []
    for b in bucket_ids:
        for split in ("TRAIN", "VALIDATION_SELECT"):
            t = read_index_projected(dataset_root, layout, split,
                                      ["decision_id", "loss_mask", "eval_mask"], buckets=[b])
            lm = t["loss_mask"].to_numpy(zero_copy_only=False)
            em = t["eval_mask"].to_numpy(zero_copy_only=False)
            keep = lm | em
            dec = t["decision_id"].to_numpy(zero_copy_only=False)[keep]
            for d in dec:
                candidates.append((int(d), b, split))

    idx = rng.choice(len(candidates), size=min(n_random, len(candidates)), replace=False)
    sampled = [candidates[i] for i in idx]

    edge = _find_edge_cases(dataset_root, layout, max_buckets=64)
    all_cases = sampled + edge["cases"]

    by_bucket: dict = {}
    for dec_id, b, split in all_cases:
        by_bucket.setdefault(b, []).append((dec_id, split))

    loader = DecisionWindowLoader(dataset_root, store_f4.context_length, layout=layout)
    mismatches = []
    n_compared = 0
    buckets_used = set()

    for b, items in by_bucket.items():
        buckets_used.add(b)
        by_split: dict = {}
        for dec_id, split in items:
            by_split.setdefault(split, set()).add(dec_id)
        for split, dec_id_set in by_split.items():
            dec_ids = np.array(sorted(dec_id_set), dtype=np.int64)
            ref = collate_rows(loader, dataset_root, layout, split, b, dec_ids)

            primary_kind = "loss" if split == "TRAIN" else "eval"
            other_kind = "eval" if split == "TRAIN" else "loss"
            ds_primary = store_f4.decisions(split, primary_kind)
            ds_other = store_f4.decisions(split, other_kind)
            if len(ds_primary) == 0:
                pos_p = np.zeros(dec_ids.shape[0], dtype=np.int64)
                found_p = np.zeros(dec_ids.shape[0], dtype=bool)
            else:
                pos_p = np.searchsorted(ds_primary.decision_id, dec_ids)
                pos_p_c = np.clip(pos_p, 0, len(ds_primary) - 1)
                found_p = ds_primary.decision_id[pos_p_c] == dec_ids
            if len(ds_other) == 0:
                pos_o = np.zeros(dec_ids.shape[0], dtype=np.int64)
                found_o = np.zeros(dec_ids.shape[0], dtype=bool)
            else:
                pos_o = np.searchsorted(ds_other.decision_id, dec_ids)
                pos_o_c = np.clip(pos_o, 0, len(ds_other) - 1)
                found_o = ds_other.decision_id[pos_o_c] == dec_ids

            for i, dec_id in enumerate(dec_ids):
                if found_p[i]:
                    computed = store_f4.batch(ds_primary, np.array([pos_p[i]]))
                elif found_o[i]:
                    computed = store_f4.batch(ds_other, np.array([pos_o[i]]))
                else:
                    mismatches.append(f"decision {dec_id} (bucket {b}, split {split}) has neither "
                                       "loss_mask nor eval_mask true -- not servable via decisions()")
                    continue
                ref_i = {k: ref[k][i:i + 1] for k in ARRAY_NAMES if k in ref}
                computed_np = {k: v for k, v in computed.items()}
                mism = compare_batch_dicts(ref_i, computed_np, list(ref_i.keys()), [])
                if mism:
                    mismatches.append(f"decision {dec_id} bucket {b} split {split}: {mism}")
                n_compared += 1
        loader._buckets.pop(b, None)

    # `crossing_train_vs` and `seg_last` are exhaustively verified ABSENT from the frozen
    # dataset (searched across all 64 model_ready buckets, not just the ones sampled here --
    # see the Sonnet-A report). Reasoning: (1) a segment's last event structurally can never
    # carry a valid target under this dataset's target rule (the target is the next event
    # WITHIN the same (user_id, segment_index) group -- collate()'s own BucketStore grouping
    # key), so it is never loss_mask/eval_mask true, by construction, for ANY segment, not
    # just ones ending near a split boundary; (2) no (user_id, segment_index) group was found
    # (over 64 buckets) containing rows from both TRAIN and VALIDATION_SELECT, so no window a
    # store could ever build ever crosses the split fence -- both are properties of the frozen
    # index itself, not of this test's search, and are therefore expected/allowed misses.
    STRUCTURALLY_ABSENT_EDGE_CASES = {"crossing_train_vs", "seg_last"}
    unexpected_missing = [m for m in edge["missing"] if m not in STRUCTURALLY_ABSENT_EDGE_CASES]

    return {
        "n_compared": n_compared,
        "n_random_requested": n_random,
        "n_buckets_used": len(buckets_used),
        "edge_cases_found": edge["found"],
        "edge_cases_missing": edge["missing"],
        "edge_cases_structurally_absent_allowed": sorted(STRUCTURALLY_ABSENT_EDGE_CASES),
        "edge_cases_unexpectedly_missing": unexpected_missing,
        "mismatches": mismatches,
        "pass": (not mismatches) and n_compared >= 4096 and len(buckets_used) >= 8
                 and not unexpected_missing,
    }


# ---------------------------------------------------------------------------
# negative controls
# ---------------------------------------------------------------------------

def gate_negative_controls(store_f4: CompactStore, dataset_root: Path) -> dict:
    ds_train = store_f4.decisions("TRAIN", "loss")
    n = len(ds_train)
    m = min(2000, n)
    positions = np.arange(m)
    computed_ok = store_f4.batch(ds_train, positions)
    shifted_positions = (positions + 1) % n
    computed_shifted = store_f4.batch(ds_train, shifted_positions)
    mism_shift = compare_batch_dicts(computed_ok, computed_shifted, ["item_tokens", "decision_id"], [])
    shift_detected = len(mism_shift) > 0

    orig_load = fs.Normalizers.load

    def perturbed_load(path):
        nz = orig_load(path)
        nz.spec = {k: dict(v) for k, v in nz.spec.items()}
        nz.spec["log1p_price"]["mean"] = nz.spec["log1p_price"]["mean"] + 5.0
        return nz

    fs.Normalizers.load = staticmethod(perturbed_load)
    try:
        small_store = fs.CompactStore.build(dataset_root, splits=("TRAIN",), channels="F4",
                                             buckets=[0, 1], context_length=50, n_threads=1,
                                             log=lambda *a, **k: None)
    finally:
        fs.Normalizers.load = orig_load

    correct_store = fs.CompactStore.build(dataset_root, splits=("TRAIN",), channels="F4",
                                           buckets=[0, 1], context_length=50, n_threads=1,
                                           log=lambda *a, **k: None)
    ds_small = small_store.decisions("TRAIN", "loss")
    ds_correct = correct_store.decisions("TRAIN", "loss")
    mm = min(len(ds_small), len(ds_correct), 200)
    pos = np.arange(mm)
    b_small = small_store.batch(ds_small, pos)
    b_correct = correct_store.batch(ds_correct, pos)
    mism_norm = compare_batch_dicts(b_correct, b_small, ["event_numeric_features"], [])
    norm_detected = len(mism_norm) > 0

    return {
        "shift_by_one_detected": shift_detected,
        "perturbed_normaliser_detected": norm_detected,
        "pass": shift_detected and norm_detected,
    }


# ---------------------------------------------------------------------------
# T11: real cold-user fixture
# ---------------------------------------------------------------------------

def gate_t11(dataset_root: Path, layout: str, store_f4: CompactStore) -> dict:
    t = read_index_projected(dataset_root, layout, "VALIDATION_SELECT",
                              ["decision_id", "client_idx"], buckets=None)
    dec = t["decision_id"].to_numpy(zero_copy_only=False)
    cli = t["client_idx"].to_numpy(zero_copy_only=False)
    unseen = np.sort(dec[cli == OOV])[:8]
    seen = np.sort(dec[cli >= 3])[:8]

    fake_client_id = np.array([3, 4, 5, 6], dtype=np.int64)  # no OOV present: simulates the bug
    original_vacuous_check = bool(np.any(fake_client_id == OOV)) or True
    real_check = bool(np.any(fake_client_id == OOV))

    ds = store_f4.decisions("VALIDATION_SELECT", "eval")
    transition_found = None
    if len(ds):
        order = np.argsort(ds.eval_user_key, kind="stable")
        key_sorted = ds.eval_user_key[order]
        boundaries = np.flatnonzero(np.concatenate([[True], key_sorted[1:] != key_sorted[:-1]]))
        group_ends = np.concatenate([boundaries[1:], [len(order)]])
        for s0, s1 in zip(boundaries, group_ends):
            if s1 - s0 < 2:
                continue
            members = order[s0:s1]
            if not np.all(ds.client_idx[members] == OOV):
                continue
            by_pos = members[np.argsort(ds.prefix_position[members], kind="stable")]
            b = store_f4.batch(ds, by_pos)
            has_hist = b["user_context_masks"][:, 0].numpy()
            if has_hist[0] == 0.0 and np.any(has_hist[1:] == 1.0):
                transition_found = {
                    "eval_user_key": int(ds.eval_user_key[by_pos[0]]),
                    "n_decisions": int(len(by_pos)),
                    "has_history_mask_sequence": has_hist.tolist(),
                }
                break

    return {
        "unseen_client_smallest8_decision_ids": unseen.tolist(),
        "seen_client_smallest8_decision_ids": seen.tolist(),
        "n_unseen_found": int(unseen.shape[0]), "n_seen_found": int(seen.shape[0]),
        "original_vacuous_check_result": original_vacuous_check,
        "real_check_result_on_broken_input": real_check,
        "vacuous_check_masks_the_bug": (original_vacuous_check is True and real_check is False),
        "cold_user_transition_example": transition_found,
        "pass": (unseen.shape[0] == 8 and seen.shape[0] == 8 and transition_found is not None
                 and original_vacuous_check is True and real_check is False),
    }


# ---------------------------------------------------------------------------
# feature invariants (report-only)
# ---------------------------------------------------------------------------

def gate_feature_invariants(store_f4: CompactStore, dataset_root: Path,
                             n_sample: int = 100_000, seed: int = 20260923) -> dict:
    ds = store_f4.decisions("TRAIN", "loss")
    rng = np.random.default_rng(seed)
    n = min(n_sample, len(ds))
    sample_pos = rng.choice(len(ds), n, replace=False)
    dec_ids = ds.decision_id[sample_pos]
    buckets = ds.user_bucket[sample_pos]

    cols = ["decision_id", "position_in_segment", "prefix_item_count", "prefix_representative_count",
            "log1p_prefix_elapsed_s", "log1p_segment_elapsed_s", "is_weekend", "local_weekday_idx",
            "local_hour", "daypart_idx", "price_missing", "log1p_price"]

    by_bucket: dict = {}
    for d, b in zip(dec_ids.tolist(), buckets.tolist()):
        by_bucket.setdefault(int(b), []).append(int(d))

    rows = {c: [] for c in cols}
    for b, ids in by_bucket.items():
        path = Path(dataset_root) / "model_ready" / f"user_bucket={b}" / "features.parquet"
        tbl = pq.read_table(path, columns=cols)
        dec_col = tbl["decision_id"].to_numpy(zero_copy_only=False)
        order = np.argsort(dec_col, kind="stable")
        dec_sorted = dec_col[order]
        ids_arr = np.array(sorted(ids), dtype=np.int64)
        pos = np.searchsorted(dec_sorted, ids_arr)
        assert np.array_equal(dec_sorted[pos], ids_arr)
        sel = order[pos]
        for c in cols:
            rows[c].append(tbl[c].to_numpy(zero_copy_only=False)[sel])

    data = {c: np.concatenate(rows[c]) for c in cols}
    out: dict = {}

    out["prefix_item_count_eq_position"] = {
        "holds": bool(np.array_equal(data["prefix_item_count"],
                                      data["position_in_segment"].astype(np.float64))),
        "violations": int(np.sum(data["prefix_item_count"] != data["position_in_segment"])),
    }
    v2_expected = (data["position_in_segment"] - 1).astype(np.float64)
    out["prefix_representative_count_eq_position_minus_1"] = {
        "holds": bool(np.array_equal(data["prefix_representative_count"], v2_expected)),
        "violations": int(np.sum(data["prefix_representative_count"] != v2_expected)),
    }
    out["log1p_prefix_elapsed_s_eq_log1p_segment_elapsed_s"] = {
        "holds": bool(np.array_equal(data["log1p_prefix_elapsed_s"], data["log1p_segment_elapsed_s"])),
        "violations": int(np.sum(data["log1p_prefix_elapsed_s"] != data["log1p_segment_elapsed_s"])),
    }
    expected_weekend = np.isin(data["local_weekday_idx"], [8, 9])
    out["is_weekend_eq_weekday_in_8_9"] = {
        "holds": bool(np.array_equal(data["is_weekend"] == 1.0, expected_weekend)),
        "violations": int(np.sum((data["is_weekend"] == 1.0) != expected_weekend)),
    }
    dpi = daypart_index()

    def expected_daypart_idx(hour):
        for name, lo, hi in DAYPARTS:
            if lo <= hour < hi:
                return dpi[name]
        return -1

    expected_dp = np.array([expected_daypart_idx(int(h)) for h in data["local_hour"]])
    out["daypart_consistent_with_local_hour"] = {
        "holds": bool(np.array_equal(data["daypart_idx"], expected_dp)),
        "violations": int(np.sum(data["daypart_idx"] != expected_dp)),
    }
    pm = data["price_missing"]
    out["price_missing_implies_log1p_price_zero"] = {
        "holds": bool(np.all(data["log1p_price"][pm] == 0.0)) if np.any(pm) else True,
        "violations": int(np.sum(data["log1p_price"][pm] != 0.0)) if np.any(pm) else 0,
        "n_price_missing_in_sample": int(np.sum(pm)),
    }
    out["n_sample"] = n
    out["all_hold"] = all(v.get("holds", True) for v in out.values() if isinstance(v, dict))
    return out


# ---------------------------------------------------------------------------
# normaliser recomputation (report-only, hard-fail only if > 1e-2)
# ---------------------------------------------------------------------------

def gate_normalizer_check(dataset_root: Path) -> dict:
    from rees46_eda.config import NORMALIZED_EVENT_NUMERIC, NORMALIZED_SESSION_PREFIX, NORMALIZED_USER_CONTEXT
    from rees46_eda.loaders import USER_MISSING_FLAG_OF, MISSING_FLAG_OF

    nz_path = Path(dataset_root) / "mappings" / "normalizers.json"
    nz = json.loads(nz_path.read_text(encoding="utf-8"))
    fit_scope = nz.get("fit_scope")
    spec = nz["features"]
    names = list(NORMALIZED_EVENT_NUMERIC) + list(NORMALIZED_SESSION_PREFIX) + list(NORMALIZED_USER_CONTEXT)
    # Every normalized feature that collate() zero-fills based on a flag/mask (MISSING_FLAG_OF
    # for event numerics, USER_MISSING_FLAG_OF for user-context numerics) was, empirically,
    # ALSO fit excluding those rows (see Sonnet-A report): recomputing over ALL TRAIN rows
    # deviates by several percent on exactly these 5 features and only these; restricting to
    # the "valid" subset below brings every one of them back to ~1e-14 relative deviation, the
    # same floating-point-exact match every unrestricted feature already shows.
    #   log1p_price:                        exclude where price_missing | price_conflict
    #   log1p_prev_raw_view_gap_s:           exclude where prev_gap_unavailable
    #   log1p_item_observed_age_s:           exclude where item_age_unavailable
    #   recent_mean_log_price /
    #   current_minus_recent_mean_log_price: exclude where recent_price_supported == 0
    needed_flag_cols = {"price_missing", "price_conflict", "prev_gap_unavailable", "item_age_unavailable"}

    def valid_mask(tbl, n, x):
        if n == "log1p_price":
            pmiss = tbl["price_missing"].to_numpy(zero_copy_only=False)
            pconf = tbl["price_conflict"].to_numpy(zero_copy_only=False)
            return ~(pmiss | pconf)
        flag = MISSING_FLAG_OF.get(n)
        if flag is not None:
            return ~tbl[flag].to_numpy(zero_copy_only=False)
        sup_col = USER_MISSING_FLAG_OF.get(n)
        if sup_col is not None:
            return tbl[sup_col].to_numpy(zero_copy_only=False) == 1.0
        return np.ones(x.shape[0], dtype=bool)

    sums = {n: 0.0 for n in names}
    sumsq = {n: 0.0 for n in names}
    counts = {n: 0 for n in names}

    buckets = fs._all_feature_buckets(dataset_root)
    for b in buckets:
        path = Path(dataset_root) / "model_ready" / f"user_bucket={b}" / "features.parquet"
        needed = set(names) | needed_flag_cols | set(USER_MISSING_FLAG_OF.values())
        cols = ["split"] + sorted(needed)
        tbl = pq.read_table(path, columns=cols)
        keep = pc.equal(tbl["split"], "TRAIN")
        tbl = tbl.filter(keep)
        if tbl.num_rows == 0:
            continue
        for n in names:
            p = spec[n]
            if p.get("constant"):
                continue
            x = tbl[n].to_numpy(zero_copy_only=False).astype(np.float64)
            x = x[valid_mask(tbl, n, x)]
            xc = np.clip(x, p["clip_lo"], p["clip_hi"])
            sums[n] += xc.sum()
            sumsq[n] += float((xc * xc).sum())
            counts[n] += xc.shape[0]

    out: dict = {"fit_scope": fit_scope,
                 "population": "TRAIN rows (split=='TRAIN'), restricted per-feature to rows NOT "
                                "zero-filled by MISSING_FLAG_OF / USER_MISSING_FLAG_OF (empirically "
                                "confirmed fit population -- see docstring above)",
                 "features": {}}
    max_rel_dev = 0.0
    for n in names:
        p = spec[n]
        if p.get("constant") or counts[n] == 0:
            out["features"][n] = {"skipped": True}
            continue
        mean = sums[n] / counts[n]
        var = sumsq[n] / counts[n] - mean * mean
        std = float(np.sqrt(max(var, 0.0)))
        rec_mean, rec_std = p["mean"], p["std"]
        rel_mean = abs(mean - rec_mean) / max(abs(rec_mean), 1e-9)
        rel_std = abs(std - rec_std) / max(abs(rec_std), 1e-9)
        max_rel_dev = max(max_rel_dev, rel_mean, rel_std)
        out["features"][n] = {
            "recomputed_mean": mean, "recorded_mean": rec_mean, "rel_dev_mean": rel_mean,
            "recomputed_std": std, "recorded_std": rec_std, "rel_dev_std": rel_std,
            "n": counts[n],
        }
    out["max_rel_deviation"] = max_rel_dev
    out["hard_fail_threshold"] = 1e-2
    out["pass"] = max_rel_dev <= 1e-2
    return out


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def run_all_gates(root: Path, manifest: Path, cohort: Path, cache_train: Path, cache_vs: Path,
                   skip_f0: bool = False) -> dict:
    layout = detect_layout(root)
    summary: dict = {"dataset_root": str(root)}

    print("[gates] class map check", flush=True)
    summary["class_map"] = gate_class_map(root)

    print("[gates] building F4 store (TRAIN+VALIDATION_SELECT, all buckets)", flush=True)
    t0 = time.time()
    store_f4 = CompactStore.build(root, splits=("TRAIN", "VALIDATION_SELECT"), channels="F4",
                                   buckets=None, context_length=50, n_threads=3, log=print)
    summary["f4_build_seconds"] = time.time() - t0
    summary["f4_n_rows"] = store_f4.n_rows
    summary["f4_mask_dtype"] = store_f4._masks_dtype

    print("[gates] counts", flush=True)
    summary["counts"] = gate_counts(store_f4)

    print("[gates] gate (i): parity vs pilot cache (F4)", flush=True)
    summary["gate_i_f4"] = gate_parity_i(store_f4, manifest, cohort, cache_train, cache_vs)

    if not skip_f0:
        print("[gates] building F0 store", flush=True)
        t0 = time.time()
        store_f0 = CompactStore.build(root, splits=("TRAIN", "VALIDATION_SELECT"), channels="F0",
                                       buckets=None, context_length=50, n_threads=3, log=print)
        summary["f0_build_seconds"] = time.time() - t0
        summary["f0_n_rows"] = store_f0.n_rows
        print("[gates] gate (i): parity vs pilot cache (F0 subset)", flush=True)
        summary["gate_i_f0_subset"] = gate_parity_i_f0_subset(store_f0, manifest, cohort,
                                                               cache_train, cache_vs)
        del store_f0

    print("[gates] gate (ii): parity vs collate()", flush=True)
    summary["gate_ii"] = gate_parity_ii(store_f4, root, layout)

    print("[gates] negative controls", flush=True)
    summary["negative_controls"] = gate_negative_controls(store_f4, root)

    print("[gates] T11 cold-user fixture", flush=True)
    summary["t11"] = gate_t11(root, layout, store_f4)

    print("[gates] feature invariants", flush=True)
    summary["feature_invariants"] = gate_feature_invariants(store_f4, root)

    print("[gates] normaliser check (full TRAIN scan)", flush=True)
    summary["normalizer_check"] = gate_normalizer_check(root)

    hard_gates = ["class_map", "counts", "gate_i_f4", "gate_ii", "negative_controls", "t11",
                  "normalizer_check"]
    if not skip_f0:
        hard_gates.append("gate_i_f0_subset")
    summary["overall_pass"] = all(summary[g].get("pass", False) for g in hard_gates)
    summary["hard_gates_checked"] = hard_gates
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="/home/jovyan/shared_data/rees46_nextitem_v1")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--cohort", default=None)
    ap.add_argument("--cache-train", default=None)
    ap.add_argument("--cache-vs", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-f0", action="store_true")
    args = ap.parse_args()

    W = Path(__file__).resolve().parents[1]
    manifest = Path(args.manifest) if args.manifest else W / "artifacts" / "train_pilot_manifest.parquet"
    cohort = Path(args.cohort) if args.cohort else W / "artifacts" / "validation_select_cohort.parquet"
    cache_train = Path(args.cache_train) if args.cache_train else W / "cache" / "train"
    cache_vs = Path(args.cache_vs) if args.cache_vs else W / "cache" / "vs"
    root = Path(args.root)

    summary = run_all_gates(root, manifest, cohort, cache_train, cache_vs, skip_f0=args.skip_f0)

    out_path = Path(args.out) if args.out else W / "tests_full" / "A" / "gates_A.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    brief = {k: v for k, v in summary.items()
             if k not in ("gate_i_f4", "gate_ii")}
    brief["gate_i_f4_pass"] = summary["gate_i_f4"]["pass"]
    brief["gate_ii_pass"] = summary["gate_ii"]["pass"]
    print(json.dumps(brief, indent=2, default=str))
    print(f"wrote {out_path}")
    print("GATES_DONE", flush=True)
    return 0 if summary["overall_pass"] else 1


def test_all_gates_smoke():
    """pytest entry point. Rebuilds the full F4(+F0) store and re-runs every gate -- prefer
    `python test_full_store.py` directly for iterative work; inspect gates_A.json for detail."""
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
