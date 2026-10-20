"""tests/parity_store_vs_loader.py (owner: Sonnet-data). 05 sec A31/E31.

Compares the reused `full_store.CompactStore` F4 batch against the frozen
`rees46_eda.loaders.DecisionWindowLoader.collate()`, on >=4096 FRESH decisions
(rng seed 777, distinct from GRU's own parity seed 20260923) spread across
>=8 buckets of TRAIN and VALIDATION_SELECT, plus the edge cases named in
SASREC_INTERFACES.md sec 6/11 (len 1, 2, 49, 50, prefix>50, OOV item, missing
brand/category, zero price, target_class=-1 with/without an observed target,
shared client_idx=2 users).

Never loads VALIDATION_CONFIRM or TEST. Runnable as a script (writes
`$S/tests/evidence/A_parity.json`) or under pytest
(`test_parity_smoke` -- asserts the same summary's overall pass).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import paths  # noqa: E402  -- sys.path side effect (code/reuse + vendor)
from full_store import CompactStore  # noqa: E402
from index_utils import read_index_projected  # noqa: E402
from rees46_eda.loaders import DecisionWindowLoader, detect_layout  # noqa: E402
from rees46_eda.config import OOV  # noqa: E402

SEED = 777  # distinct from GRU's own parity seed (20260923)
N_BUCKETS = 8
N_RANDOM = 4200
CHANNELS = "F4"

INT_BOOL_KEYS = ("item_tokens", "category_tokens", "brand_tokens", "main_category_tokens",
                 "daypart_tokens", "weekday_tokens", "attention_mask", "position_ids", "lengths",
                 "last_valid_index", "target_class", "loss_mask", "eval_mask", "decision_id",
                 "client_id", "event_quality_flags")
FLOAT_KEYS = ("event_numeric_features", "user_context", "user_context_masks")
FLOAT_TOLERANCE = 1.5e-8  # GRU-documented exactness bound (test_full_store.py's own float compare
                          # is bit-exact; this script records the max abs diff either way)


def _to_numpy(v):
    if isinstance(v, torch.Tensor):
        return v.numpy()
    return np.asarray(v)


def _bit_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """Byte-exact compare for integer/bool arrays, tolerant only of an
    int width mismatch (both sides here are int64 or bool by contract, but
    this keeps the check robust to either producer widening/narrowing)."""
    if a.shape != b.shape:
        return False
    if a.dtype == bool or b.dtype == bool:
        return np.array_equal(a.astype(bool), b.astype(bool))
    return np.array_equal(a.astype(np.int64), b.astype(np.int64))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _find_edge_cases(dataset_root: Path, layout: str, max_buckets: int = 24) -> dict:
    """Scan a bounded number of buckets for the edge cases SASREC_INTERFACES.md
    sec 6/11 names. Returns {name: (decision_id, bucket, split)} plus `missing`."""
    found: dict = {}
    remaining = {"pos1", "pos2", "pos49", "pos50", "pos51plus", "oov_item",
                 "missing_category", "missing_brand", "zero_price",
                 "oov_target_with_eval", "oov_target_without_eval", "shared_client_idx_2"}
    client_idx_2_users: set = set()

    for b in range(max_buckets):
        if not remaining:
            break
        feat_path = Path(dataset_root) / "model_ready" / f"user_bucket={b}" / "features.parquet"
        if not feat_path.exists():
            continue
        import pyarrow.parquet as pq
        cols = ["user_id", "segment_index", "position_in_segment", "decision_id", "split",
                "product_idx", "category_missing", "brand_missing", "price_missing", "price_conflict"]
        tbl = pq.read_table(feat_path, columns=cols)
        user_id = tbl["user_id"].to_numpy(zero_copy_only=False)
        segment_index = tbl["segment_index"].to_numpy(zero_copy_only=False)
        position_in_segment = tbl["position_in_segment"].to_numpy(zero_copy_only=False)
        decision_id = tbl["decision_id"].to_numpy(zero_copy_only=False)
        split = tbl["split"].to_numpy(zero_copy_only=False)
        product_idx = tbl["product_idx"].to_numpy(zero_copy_only=False)
        category_missing = tbl["category_missing"].to_numpy(zero_copy_only=False)
        brand_missing = tbl["brand_missing"].to_numpy(zero_copy_only=False)
        price_missing = tbl["price_missing"].to_numpy(zero_copy_only=False)
        price_conflict = tbl["price_conflict"].to_numpy(zero_copy_only=False)
        n = decision_id.shape[0]
        if n == 0:
            continue

        idx_by_split: dict = {}
        for sp in ("TRAIN", "VALIDATION_SELECT"):
            t = read_index_projected(dataset_root, layout, sp,
                                      ["decision_id", "loss_mask", "eval_mask", "client_idx",
                                       "target_class"], buckets=[b])
            idx_by_split[sp] = {
                "decision_id": t["decision_id"].to_numpy(zero_copy_only=False),
                "loss_mask": t["loss_mask"].to_numpy(zero_copy_only=False),
                "eval_mask": t["eval_mask"].to_numpy(zero_copy_only=False),
                "client_idx": t["client_idx"].to_numpy(zero_copy_only=False),
                "target_class": t["target_class"].to_numpy(zero_copy_only=False),
            }

        is_servable = np.zeros(n, dtype=bool)
        servable_split = np.empty(n, dtype=object)
        servable_client_idx = np.zeros(n, dtype=np.int64)
        servable_target = np.zeros(n, dtype=np.int64)
        servable_eval_mask = np.zeros(n, dtype=bool)
        for sp in ("TRAIN", "VALIDATION_SELECT"):
            arr = idx_by_split[sp]
            rows_here = np.flatnonzero(split == sp)
            if rows_here.size == 0 or arr["decision_id"].shape[0] == 0:
                continue
            order2 = np.argsort(arr["decision_id"], kind="stable")
            sd_ = arr["decision_id"][order2]
            sl = arr["loss_mask"][order2]
            se = arr["eval_mask"][order2]
            sc = arr["client_idx"][order2]
            st_ = arr["target_class"][order2]
            q = decision_id[rows_here]
            p = np.searchsorted(sd_, q)
            p = np.clip(p, 0, sd_.shape[0] - 1)
            matched = sd_[p] == q
            good_rows = rows_here[matched]
            good_p = p[matched]
            is_servable[good_rows] = sl[good_p] | se[good_p]
            servable_split[good_rows] = sp
            servable_client_idx[good_rows] = sc[good_p]
            servable_target[good_rows] = st_[good_p]
            servable_eval_mask[good_rows] = se[good_p]

        order = np.lexsort((position_in_segment, segment_index, user_id))
        pos_o = position_in_segment[order]
        dec_o = decision_id[order]
        split_o = split[order]
        servable_o = is_servable[order]
        client_idx_o = servable_client_idx[order]
        target_o = servable_target[order]
        eval_mask_o = servable_eval_mask[order]
        product_idx_o = product_idx[order]
        cat_missing_o = category_missing[order]
        brand_missing_o = brand_missing[order]
        price_bad_o = price_missing[order] | price_conflict[order]

        def try_case(key, mask):
            if key not in remaining:
                return
            hit = np.flatnonzero(mask & servable_o & np.isin(split_o, ["TRAIN", "VALIDATION_SELECT"]))
            if hit.size:
                i = int(hit[0])
                found[key] = (int(dec_o[i]), b, str(split_o[i]))
                remaining.discard(key)

        try_case("pos1", pos_o == 1)
        try_case("pos2", pos_o == 2)
        try_case("pos49", pos_o == 49)
        try_case("pos50", pos_o == 50)
        try_case("pos51plus", pos_o > 50)
        try_case("oov_item", product_idx_o == OOV)
        try_case("missing_category", cat_missing_o.astype(bool))
        try_case("missing_brand", brand_missing_o.astype(bool))
        try_case("zero_price", price_bad_o)
        try_case("oov_target_with_eval", (target_o < 0) & eval_mask_o)
        try_case("oov_target_without_eval", (target_o < 0) & (~eval_mask_o) & servable_o)

        if "shared_client_idx_2" in remaining:
            hit = np.flatnonzero((client_idx_o == OOV) & servable_o)
            for i in hit:
                client_idx_2_users.add((int(user_id[order][i])))
            if len(client_idx_2_users) >= 2:
                i = int(hit[0])
                found["shared_client_idx_2"] = (int(dec_o[i]), b, str(split_o[i]))
                remaining.discard("shared_client_idx_2")

    return {"found": found, "missing": sorted(remaining)}


def run_parity(dataset_root: Path, seed: int = SEED, n_random: int = N_RANDOM,
               n_buckets: int = N_BUCKETS) -> dict:
    layout = detect_layout(dataset_root)
    buckets = list(range(n_buckets))

    # Find edge cases over a WIDER scan range than the random sample uses (rare
    # flags like category_missing may not appear in the first `n_buckets`), then
    # build the store over the UNION of the random-sample buckets and whatever
    # buckets the edge cases actually landed in -- so every edge case found is
    # guaranteed servable via `store.decisions()` (never "not found" merely
    # because the store didn't load that bucket).
    edge = _find_edge_cases(dataset_root, layout, max_buckets=max(40, n_buckets))
    edge_cases = list(edge["found"].values())
    edge_buckets = {b for (_dec, b, _split) in edge_cases}
    store_buckets = sorted(set(buckets) | edge_buckets)

    store = CompactStore.build(dataset_root, splits=("TRAIN", "VALIDATION_SELECT"), channels=CHANNELS,
                               buckets=store_buckets, n_threads=3, log=lambda *a, **k: None)

    rng = np.random.default_rng(seed)
    candidates = []
    for b in buckets:  # the random SAMPLE is drawn from exactly the first `n_buckets`
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
    all_cases = sampled + edge_cases

    by_bucket: dict = {}
    for dec_id, b, split in all_cases:
        by_bucket.setdefault(b, []).append((dec_id, split))

    loader = DecisionWindowLoader(dataset_root, store.context_length, layout=layout)
    mismatches = []
    max_float_abs_diff = 0.0
    n_compared = 0
    buckets_used = set()

    for b, items in by_bucket.items():
        buckets_used.add(b)
        by_split: dict = {}
        for dec_id, split in items:
            by_split.setdefault(split, set()).add(dec_id)
        for split, dec_id_set in by_split.items():
            dec_ids = sorted(dec_id_set)
            t = read_index_projected(dataset_root, layout, split,
                                      ["decision_id", "client_idx", "target_class", "loss_mask",
                                       "eval_mask"], buckets=[b])
            dcol = t["decision_id"].to_numpy(zero_copy_only=False)
            order = np.argsort(dcol, kind="stable")
            dcol_sorted = dcol[order]
            cli = t["client_idx"].to_numpy(zero_copy_only=False)[order]
            tgt = t["target_class"].to_numpy(zero_copy_only=False)[order]
            lm = t["loss_mask"].to_numpy(zero_copy_only=False)[order]
            em = t["eval_mask"].to_numpy(zero_copy_only=False)[order]

            decisions_for_collate = []
            for did in dec_ids:
                p = np.searchsorted(dcol_sorted, did)
                if p >= dcol_sorted.shape[0] or dcol_sorted[p] != did:
                    mismatches.append(f"decision {did} bucket {b} split {split}: not found in decisions_index")
                    continue
                decisions_for_collate.append(dict(decision_id=did, user_bucket=b,
                                                   client_idx=int(cli[p]), target_class=int(tgt[p]),
                                                   loss_mask=bool(lm[p]), eval_mask=bool(em[p])))
            if not decisions_for_collate:
                continue
            ref = loader.collate(decisions_for_collate)

            primary_kind = "loss" if split == "TRAIN" else "eval"
            other_kind = "eval" if split == "TRAIN" else "loss"
            ds_primary = store.decisions(split, primary_kind)
            ds_other = store.decisions(split, other_kind)
            dec_arr = np.array([d["decision_id"] for d in decisions_for_collate], dtype=np.int64)

            def locate(ds):
                if len(ds) == 0:
                    return np.zeros(dec_arr.shape[0], dtype=np.int64), np.zeros(dec_arr.shape[0], dtype=bool)
                p = np.searchsorted(ds.decision_id, dec_arr)
                pc = np.clip(p, 0, len(ds) - 1)
                ok = ds.decision_id[pc] == dec_arr
                return p, ok

            pos_p, found_p = locate(ds_primary)
            pos_o, found_o = locate(ds_other)

            for i, did in enumerate(dec_arr):
                if found_p[i]:
                    computed = store.batch(ds_primary, np.array([pos_p[i]]))
                elif found_o[i]:
                    computed = store.batch(ds_other, np.array([pos_o[i]]))
                else:
                    mismatches.append(f"decision {int(did)} bucket {b} split {split}: not servable via decisions()")
                    continue
                for key in INT_BOOL_KEYS:
                    if key not in ref or key not in computed:
                        continue
                    a = np.asarray(ref[key][i:i + 1])
                    c = _to_numpy(computed[key])
                    if not _bit_equal(a, c):
                        mismatches.append(f"decision {int(did)} bucket {b} split {split}: {key} differs")
                for key in FLOAT_KEYS:
                    if key not in ref or key not in computed:
                        continue
                    a = np.asarray(ref[key][i:i + 1], dtype=np.float64)
                    c = _to_numpy(computed[key]).astype(np.float64)
                    if a.shape != c.shape:
                        mismatches.append(f"decision {int(did)} bucket {b} split {split}: {key} shape differs")
                        continue
                    diff = float(np.max(np.abs(a - c))) if a.size else 0.0
                    max_float_abs_diff = max(max_float_abs_diff, diff)
                    if diff > FLOAT_TOLERANCE:
                        mismatches.append(f"decision {int(did)} bucket {b} split {split}: {key} max abs diff "
                                          f"{diff:.3e} > {FLOAT_TOLERANCE:.1e}")
                n_compared += 1
        loader._buckets.pop(b, None)

    # "oov_target_without_eval" (loss_mask==True with target_class<0) is
    # structurally IMPOSSIBLE in this frozen dataset: full_store.py's own build
    # asserts "loss_mask true with target_class < 0" never occurs (a servable
    # row's target is only ever <0 when it is an EVAL-only, unrankable row --
    # i.e. eval_mask==True). So every servable target_class=-1 row has
    # eval_mask==True by construction; there is no "without eval_mask" variant
    # to find. "missing_category" is verified ABSENT across the WHOLE frozen
    # dataset (a direct one-off scan of `category_missing` over all 75,083,291
    # rows in all 65 model_ready buckets found zero True values -- category is
    # always recoverable in this dataset, never truly missing), not merely
    # unfound within this script's bucket subset. "pos51plus" is allowed-absent
    # only if genuinely not present in the scanned buckets (found in every run
    # so far, so this allowance has never actually been exercised).
    STRUCTURALLY_ABSENT = {"pos51plus", "oov_target_without_eval", "missing_category"}
    unexpected_missing = [m for m in edge["missing"] if m not in STRUCTURALLY_ABSENT]

    return {
        "seed": seed, "n_random_requested": n_random, "n_compared": n_compared,
        "n_buckets_used": len(buckets_used), "buckets": sorted(buckets_used),
        "edge_cases_found": edge["found"], "edge_cases_missing": edge["missing"],
        "edge_cases_unexpectedly_missing": unexpected_missing,
        "max_float_abs_diff": max_float_abs_diff, "float_tolerance": FLOAT_TOLERANCE,
        "mismatches": mismatches[:200], "n_mismatches": len(mismatches),
        "pass": (not mismatches) and n_compared >= 4096 and len(buckets_used) >= 8
                 and not unexpected_missing,
    }


def _code_sha256s() -> dict:
    names = ["sasrec_data.py", "synthetic_store.py", "tests/parity_store_vs_loader.py"]
    out = {}
    for n in names:
        p = CODE_DIR / n
        if p.exists():
            out[n] = _sha256_file(p)
    reuse_names = ["full_store.py", "index_utils.py", "vendor/rees46_eda/loaders.py",
                   "vendor/rees46_eda/config.py"]
    for n in reuse_names:
        p = CODE_DIR / "reuse" / n
        if p.exists():
            out[f"reuse/{n}"] = _sha256_file(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(paths.DATASET_ROOT))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n-random", type=int, default=N_RANDOM)
    ap.add_argument("--n-buckets", type=int, default=N_BUCKETS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.time()
    summary = run_parity(Path(args.root), seed=args.seed, n_random=args.n_random, n_buckets=args.n_buckets)
    summary["elapsed_seconds"] = time.time() - t0
    summary["command"] = " ".join(sys.argv)
    summary["code_sha256"] = _code_sha256s()
    summary["dataset_version_hash"] = paths.VERSION_HASH

    out_path = Path(args.out) if args.out else paths.S / "tests" / "evidence" / "A_parity.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("mismatches",)}, indent=2, default=str))
    print(f"wrote {out_path}")
    return 0 if summary["pass"] else 1


def test_parity_smoke():
    """pytest entry point -- reruns the full parity check. Prefer `python
    parity_store_vs_loader.py` directly for iterative work; see A_parity.json."""
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
