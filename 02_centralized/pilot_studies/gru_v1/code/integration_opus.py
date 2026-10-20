"""Wave-2 integration gates run by Opus through the REAL modules (owner: Opus).

The Wave-1 agents proved their pieces against stand-ins; this script proves the
assembled production path: `full_store.CompactStore` + `eval_full` + `baselines_full`.

  G-iii-store : pilot best checkpoints (F0, F4) evaluated through CompactStore + GPU ranking
                on the 10k cohort (eval_batch 512, pilot chunking) reproduce the recorded
                pilot eval records within 1e-6 (FULL_RUN_PROTOCOL.md sec. 10.10).
  G-base-store: POP/TRANS fitted through the store on the pilot TRAIN manifest are
                IDENTICAL to the pilot-cache fit, and reproduce BASELINES.json within 1e-6
                on the cohort.
  X-fit       : full-TRAIN POP/TRANS fit via the store == fit via the independent
                projected-Parquet reader (baselines_full reader b), exactly.
  X-eval-rows : full-VS eval rows via the store == reader b (decision order, targets,
                clients, eval_user_key, last item, prefix position, prior views), exactly.
  T-vs-eval   : timing of one full-VS GRU evaluation at eval_batch 1024 (the epoch-end
                cost), on the pilot F0 best checkpoint. Informational only.

VALIDATION_CONFIRM / TEST are never touched. Writes only its own JSON/log.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("MKL_NUM_THREADS", "3")

CODE_DIR = Path(__file__).resolve().parent
for _p in (str(CODE_DIR), str(CODE_DIR / "vendor")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(3)

import train as pilot_train  # noqa: E402
import model as model_mod  # noqa: E402
import baselines_full as bf  # noqa: E402
import eval_full  # noqa: E402
from full_store import CompactStore  # noqa: E402
from rees46_eda.loaders import detect_layout  # noqa: E402

W = CODE_DIR.parent
ROOT = Path("/home/jovyan/shared_data/rees46_nextitem_v1")
ART = W / "artifacts"
OUT = W / "tests_full" / "opus" / "integration.json"
TOL = 1e-6
PILOT_RUNS = {
    "F0_item_only": W / "experiments/gru/T-M_F0_item_only",
    "F4_rich_side_information": W / "experiments/gru/T-M_F4_rich_side_information",
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rss_gib():
    import psutil
    return psutil.Process().memory_info().rss / 2**30


def recorded_eval(run_dir: Path, update: int) -> dict:
    for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if "eval_full" in rec and rec.get("update") == update:
            return rec["eval_full"]
    raise KeyError(f"no eval record at update {update} in {run_dir}")


def compare_blocks(got: dict, exp: dict, metrics=("mrr@20", "recall@20", "ndcg@20")) -> dict:
    worst = 0.0
    problems = []
    for den in ("end_to_end", "closed"):
        for agg in ("micro", "macro"):
            for m in metrics:
                g, e = got[den][agg][m], exp[den][agg][m]
                d = abs(g - e)
                worst = max(worst, d)
                if d > TOL:
                    problems.append(f"{den}.{agg}.{m}: got {g} exp {e} diff {d:.3g}")
        if got[den]["n"] != exp[den]["n"]:
            problems.append(f"{den}.n {got[den]['n']} != {exp[den]['n']}")
        if got[den]["macro"]["user_count"] != exp[den]["macro"]["user_count"]:
            problems.append(f"{den}.user_count mismatch")
    return {"pass": not problems, "max_abs_diff": worst, "problems": problems}


def main() -> int:
    pilot_train.set_determinism(pilot_train.SEED)   # strict FP32, TF32 off (parity needs it)
    device = torch.device("cuda:0")
    out: dict = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    t0 = time.time()
    store = CompactStore.build(ROOT, splits=("TRAIN", "VALIDATION_SELECT"), channels="F4", log=log)
    out["store"] = {"n_rows": store.n_rows, "build_s": round(time.time() - t0, 1),
                    "rss_gib_after_build": round(rss_gib(), 2)}
    ds_train = store.decisions("TRAIN", "loss")
    ds_vs = store.decisions("VALIDATION_SELECT", "eval")
    out["store"]["rss_gib_after_decisions"] = round(rss_gib(), 2)
    out["counts"] = {"train_loss": len(ds_train), "vs_eval": len(ds_vs),
                     "vs_users": int(np.unique(ds_vs.eval_user_key).shape[0])}
    assert out["counts"] == {"train_loss": 40_180_608, "vs_eval": 8_826_488, "vs_users": 790_459}, out["counts"]
    log(f"store ok {out['store']} {out['counts']}")

    # ---- cohort DecisionSet in pilot cohort order ---------------------------------------
    coh = pq.read_table(ART / "validation_select_cohort.parquet", columns=["decision_id", "eval_user_key"])
    coh_ids = coh["decision_id"].to_numpy().astype(np.int64)
    pos = np.searchsorted(ds_vs.decision_id, coh_ids)
    assert np.array_equal(ds_vs.decision_id[pos], coh_ids)
    ds_coh = ds_vs.take(pos)
    assert np.array_equal(ds_coh.eval_user_key, coh["eval_user_key"].to_numpy().astype(np.int64))

    # ---- G-iii-store ---------------------------------------------------------------------
    poc = pilot_train.load_product_idx_of_class()
    out["G_iii_store"] = {}
    for group, run_dir in PILOT_RUNS.items():
        ck = torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
        m = model_mod.build_model(group, pilot_train.VOCAB_SIZES, product_idx_of_class=poc).to(device)
        m.load_state_dict(ck["model_state_dict"])
        res1 = eval_full.evaluate_model(m, store, ds_coh, device, eval_batch=512, return_ranks=True)
        res2 = eval_full.evaluate_model(m, store, ds_coh, device, eval_batch=512, return_ranks=True)
        cmp = compare_blocks(res1, recorded_eval(run_dir, int(ck["update_number"])))
        cmp["bit_identical_repeat"] = bool(np.array_equal(res1["ranks"], res2["ranks"]))
        cmp["pass"] = cmp["pass"] and cmp["bit_identical_repeat"]
        cmp["update"] = int(ck["update_number"])
        cmp["macro_mrr20"] = res1["end_to_end"]["macro"]["mrr@20"]
        out["G_iii_store"][group] = cmp
        log(f"G-iii-store {group}: {cmp}")
        del m

    # ---- G-base-store --------------------------------------------------------------------
    man = pq.read_table(ART / "train_pilot_manifest.parquet", columns=["decision_id"])
    man_ids = man["decision_id"].to_numpy().astype(np.int64)
    mpos = np.searchsorted(ds_train.decision_id, man_ids)
    assert np.array_equal(ds_train.decision_id[mpos], man_ids)
    fit_store_pilot = bf.fit_from_store(store, ds_train.take(mpos))
    fit_cache_pilot = bf.fit_from_cache(W / "cache" / "train")
    same_fit_pilot = (np.array_equal(fit_store_pilot.pop_count, fit_cache_pilot.pop_count)
                      and (fit_store_pilot.trans_matrix != fit_cache_pilot.trans_matrix).nnz == 0)
    base = json.loads((ART / "BASELINES.json").read_text(encoding="utf-8"))
    li_coh = store.decision_item_tokens(ds_coh)
    pop_res = bf.run_pop_baseline(fit_store_pilot, ds_coh, device)
    trans_res = bf.run_trans_baseline(fit_store_pilot, ds_coh, device, last_item=li_coh)
    out["G_base_store"] = {
        "fit_identical_to_pilot_cache_fit": bool(same_fit_pilot),
        "POP": compare_blocks(pop_res, base["POP"]),
        "TRANS": compare_blocks(trans_res, base["TRANS"]),
    }
    out["G_base_store"]["pass"] = (same_fit_pilot and out["G_base_store"]["POP"]["pass"]
                                   and out["G_base_store"]["TRANS"]["pass"])
    log(f"G-base-store: {out['G_base_store']}")

    # ---- X-fit and X-eval-rows (store vs independent reader b) ---------------------------
    layout = detect_layout(ROOT)
    mr = bf.load_model_ready_columns(ROOT)
    fit_store_full = bf.fit_from_store(store, ds_train)
    fit_parq_full = bf.fit_from_parquet(ROOT, layout, product_idx_table=mr, expect_n_rows=40_180_608, log=log)
    out["X_fit"] = {
        "pop_equal": bool(np.array_equal(fit_store_full.pop_count, fit_parq_full.pop_count)),
        "trans_equal": bool((fit_store_full.trans_matrix != fit_parq_full.trans_matrix).nnz == 0),
        "max_pop": fit_store_full.max_pop, "max_trans": fit_store_full.max_trans,
        "max_composite": fit_store_full.max_composite_score,
        "nnz": fit_store_full.n_trans_nonzero_src_tgt,
    }
    out["X_fit"]["pass"] = out["X_fit"]["pop_equal"] and out["X_fit"]["trans_equal"]
    log(f"X-fit: {out['X_fit']}")
    del fit_parq_full, fit_store_pilot, fit_cache_pilot

    rows_b = bf.eval_rows_from_parquet(ROOT, layout, "VALIDATION_SELECT", model_ready_table=mr,
                                       expect_n_rows=8_826_488, expect_n_users=790_459, log=log)
    del mr
    li_vs = store.decision_item_tokens(ds_vs)
    checks = {
        "decision_id": np.array_equal(ds_vs.decision_id, rows_b.decision_id),
        "target_class": np.array_equal(ds_vs.target_class, rows_b.target_class),
        "client_idx": np.array_equal(ds_vs.client_idx, rows_b.client_idx),
        "eval_user_key": np.array_equal(ds_vs.eval_user_key, rows_b.eval_user_key),
        "last_item": np.array_equal(li_vs, rows_b.last_item),
        "prefix_position": np.array_equal(ds_vs.prefix_position.astype(np.int64), rows_b.prefix_position),
        "prior_view_count": np.array_equal(ds_vs.prior_view_count.astype(np.int64), rows_b.prior_view_count),
        "eval_mask_all_true": bool(ds_vs.eval_mask.all()),
    }
    out["X_eval_rows"] = {k: bool(v) for k, v in checks.items()}
    out["X_eval_rows"]["pass"] = all(out["X_eval_rows"].values())
    log(f"X-eval-rows: {out['X_eval_rows']}")
    del rows_b

    # ---- T-vs-eval -----------------------------------------------------------------------
    ck = torch.load(PILOT_RUNS["F0_item_only"] / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    m = model_mod.build_model("F0_item_only", pilot_train.VOCAB_SIZES, product_idx_of_class=poc).to(device)
    m.load_state_dict(ck["model_state_dict"])
    t1 = time.time()
    res = eval_full.evaluate_model(m, store, ds_vs, device, eval_batch=1024)
    out["T_vs_eval"] = {"seconds": round(time.time() - t1, 1), "rows_per_s": res["_rows_per_s"],
                        "peak_gpu_mem_gib": (res["_peak_gpu_mem_bytes"] or 0) / 2**30,
                        "note": "pilot F0 checkpoint on full VS; informational timing only"}
    log(f"T-vs-eval: {out['T_vs_eval']}")

    out["overall_pass"] = all(v["pass"] for v in out["G_iii_store"].values()) and \
        out["G_base_store"]["pass"] and out["X_fit"]["pass"] and out["X_eval_rows"]["pass"]
    out["peak_rss_gib"] = round(__import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss / 2**20, 2)
    out["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    log(f"INTEGRATION_DONE overall_pass={out['overall_pass']}")
    return 0 if out["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
