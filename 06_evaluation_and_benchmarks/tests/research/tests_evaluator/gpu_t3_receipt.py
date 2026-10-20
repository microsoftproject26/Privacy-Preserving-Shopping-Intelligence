"""EVAL-03 Part B — the blocking T3 GPU receipt (contracts/PREG1_LOCK_VALUES.json T3 g1_gpu_receipt_required).

Runs on the server, one A16, under the FROZEN T3 configuration (v4eval.frozen_config.load(); all deterministic flags,
CUBLAS_WORKSPACE_CONFIG set before CUDA initializes), on the real M02_VALIDATION_WARM manifest and the Clean V2 store
(v4data.store over the snapshot CompactStore, VALIDATION splits only), with one random-init v4models checkpoint per
family (seed 2026, v4models.build with the real v1 class map). Required items:
  1. pass-2 bitwise reproduction: a full M02 pass with cache_bytes = 0, so pass 2 RECOMPUTES every score block and the
     evaluator's built-in check (target scores reproduce bit-for-bit, else NondeterministicScoreError) runs on every
     batch; its ranks must equal the frozen (cached) run's ranks;
  2. single block == BaseAdapter.logits bitwise, on sampled M02 batches (the rankable rows and the full batch);
  3. two invocations of the full M02 rank vector for one checkpoint identical (ranks, n_tied, eq_before);
  4. wall time and peak memory at K = 158,486 (GPU allocated / reserved per phase, host RSS).
Random-init metric values are recorded only as SANITY_RANDOM_INIT_NOT_A_RESULT. No TEST row is read. Output:
<receipt_root>/out/ (T3_GPU_RECEIPT.json, one int32 rank vector per family, run log).

Layout on the server (all writes under ppsi_v4/runs_g1/eval_receipt/):
  <E>/code/{v4eval,v4models,fl}/  the bundle (byte copies of the local code, hash-checked against BUNDLE_MANIFEST.json)
  <E>/code/gpu_t3_receipt.py      this script
  <E>/contracts/PREG1_LOCK_VALUES.json
  ppsi_env.py, snap_v1/ and v4data/ are imported from the server's ppsi_v4/code (DATA-04's final build).
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")       # T3 flag; must precede CUDA initialization
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent                     # <E>/code
E = HERE.parent                                            # receipt bundle root
V4 = next(p for p in E.parents if (p / "contracts" / "PREG1_LOCK_VALUES.json").is_file())
SERVER_CODE = V4 / "code"
for p in (str(SERVER_CODE), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
import ppsi_env  # noqa: E402,F401  (server copy: snapshot isolation, before any snapshot import)

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402

from v4eval import evaluator_code_sha256  # noqa: E402
from v4eval import frozen_config as fc  # noqa: E402
from v4eval.evaluate import evaluate  # noqa: E402
from v4eval.manifest import build_eval_manifest  # noqa: E402
from v4eval.predict import _research_predict_adapter, predict_adapter, stream_rows  # noqa: E402
from v4eval.ranking import block_scores  # noqa: E402

M02_COLS = ("decision_id", "eval_user_key", "eval_mask", "target_class", "target_oov", "censor_class", "subperiod",
            "old_split", "v4_split", "oov_cause", "rankable")


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(p: Path) -> str:
    return sha_bytes(Path(p).read_bytes())


def ranks_sha(a: np.ndarray) -> str:
    return sha_bytes(np.ascontiguousarray(a, dtype="<i8").tobytes())


def frozen_a16_reference_check(fam: str, previous: np.ndarray) -> dict:
    """INT-06: does the A16 reference rank vector equal the frozen A16 receipt's (code/tests_eval/receipts/
    T3_GPU_RECEIPT.json, device class A16-FP32-DET) run1 ranks_sha256? None when that receipt is not on this host."""
    for p in (HERE / "receipts" / "T3_GPU_RECEIPT.json", SERVER_CODE / "tests_eval" / "receipts" / "T3_GPU_RECEIPT.json"):
        if p.is_file():
            frozen_rec = json.loads(p.read_text(encoding="utf-8"))
            want = (((frozen_rec.get("families") or {}).get(fam) or {}).get("run1") or {}).get("ranks_sha256")
            ok = (frozen_rec.get("device") or {}).get("eval_device_class") == "A16-FP32-DET" and want is not None
            return {"frozen_a16_receipt": str(p), "frozen_a16_receipt_sha256": sha_file(p),
                    "reference_ranks_sha256": ranks_sha(previous),
                    "matches_frozen_a16_receipt": bool(ok and ranks_sha(previous) == want)}
    return {"frozen_a16_receipt": None, "matches_frozen_a16_receipt": None}


def rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0      # Linux: KiB


def log(msg: str, fh) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def verify_bundle() -> dict:
    man = json.loads((HERE / "BUNDLE_MANIFEST.json").read_text(encoding="utf-8"))
    bad = {rel: h for rel, h in man["files"].items() if sha_file(E / rel) != h}
    if bad:
        raise RuntimeError(f"bundle drift: {sorted(bad)}")
    return man


def load_m02(man_dir: Path):
    header = json.loads((man_dir / "M02_VALIDATION_WARM.header.json").read_text(encoding="utf-8"))
    tbl = pq.read_table(man_dir / "M02_VALIDATION_WARM.decisions.parquet", columns=list(M02_COLS))
    cols = {c: tbl[c].to_numpy(zero_copy_only=False) for c in M02_COLS}
    m = build_eval_manifest(header, cols)                    # verifies the three ordered hashes against the header
    return header, m


def merged_eval_set(store):
    from full_store import DecisionSet                       # snapshot copy (ppsi_env)
    from v4data.common import VALIDATION_SPLITS
    parts = [store.decision_set(s, "eval") for s in VALIDATION_SPLITS]
    cat = lambda f: np.concatenate([getattr(p, f) for p in parts])  # noqa: E731
    did = cat("decision_id")
    o = np.argsort(did, kind="stable")
    ds = DecisionSet(split="VALIDATION", kind="eval", decision_id=did[o], end_row=cat("end_row")[o],
                     length=cat("length")[o], target_class=cat("target_class")[o], loss_mask=cat("loss_mask")[o],
                     eval_mask=cat("eval_mask")[o], client_idx=cat("client_idx")[o], user_bucket=cat("user_bucket")[o],
                     eval_user_key=None, prior_view_count=cat("prior_view_count")[o],
                     prefix_position=cat("prefix_position")[o])
    if ds.decision_id.size > 1 and not np.all(np.diff(ds.decision_id) > 0):
        raise AssertionError("merged VALIDATION decision ids are not strictly ascending")
    return ds, {s: len(p) for s, p in zip(VALIDATION_SPLITS, parts)}


def single_block_check(adapter, batches, m, dev, idxs) -> dict:
    module = adapter.module
    was = module.training
    module.eval()
    W, b = adapter.head_weight(), adapter.head_bias()
    K = int(W.shape[0])
    rk = m.rankable
    stream = stream_rows(m)
    out = {"batches_checked": [], "rankable_rows_equal": True, "full_batch_equal": True, "n_rows_compared": 0}
    try:
        with torch.no_grad():
            for i in idxs:
                bt = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batches[i].items()}
                q = adapter.query(bt)
                rows = stream[i * 1024:i * 1024 + q.shape[0]]
                sel = torch.as_tensor(np.flatnonzero(rk[rows]), dtype=torch.long, device=dev)
                qs = q.index_select(0, sel)
                eq_r = torch.equal(block_scores(qs, W, b, 0, K), adapter.logits(qs))
                eq_f = torch.equal(block_scores(q, W, b, 0, K), adapter.logits(q))
                out["batches_checked"].append({"batch": int(i), "rows": int(q.shape[0]), "rankable_rows": int(sel.numel()),
                                               "rankable_rows_equal": bool(eq_r), "full_batch_equal": bool(eq_f)})
                out["rankable_rows_equal"] &= bool(eq_r)
                out["full_batch_equal"] &= bool(eq_f)
                out["n_rows_compared"] += int(q.shape[0])
    finally:
        module.train(was)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--families", default="GRU,SASREC")
    ap.add_argument("--n-threads", type=int, default=4)
    ap.add_argument("--sample-batches", type=int, default=24)
    args = ap.parse_args()
    out_dir = E / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    fh = open(out_dir / "run.log", "a", encoding="utf-8")
    torch.set_num_threads(int(args.n_threads))
    rec = {"task": "EVAL-03 Part B", "receipt": "T3 GPU receipt", "utc_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "host": platform.node(), "python": sys.version.split()[0], "torch": torch.__version__,
           "cuda": torch.version.cuda, "numpy": np.__version__, "status": "RUNNING"}
    try:
        rec["bundle"] = verify_bundle()
        rec["snapshot"] = {"verified_files": len(ppsi_env.verify_snapshot()["files"]),
                           "isolated_modules": len(ppsi_env.assert_isolated())}
        cfg = fc.load()
        fc.apply_runtime_flags(cfg)
        dev = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(dev)
        fc.check_runtime_flags(cfg, dev)
        props = torch.cuda.get_device_properties(dev)
        rec["device"] = {"index_in_process": args.gpu, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                         "uuid": str(getattr(props, "uuid", "")), "name": torch.cuda.get_device_name(dev),
                         "eval_device_class": fc.eval_device_class(dev), "total_memory_bytes": int(props.total_memory),
                         "note": "the snapshot's reuse/train.py setdefaults CUDA_VISIBLE_DEVICES=0 on import, so the "
                                 "launcher exports the physical GPU explicitly and the process uses cuda:0"}
        rec["frozen_config"] = dict(cfg.eval_config("adapter_stream", fc.eval_device_class(dev)),
                                    lock_file_sha256=cfg.lock_sha256, t3_sha256=cfg.sha256,
                                    flags=dict(cfg.flags))
        rec["runtime_flags"] = {"matmul.allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                                "cudnn.allow_tf32": bool(torch.backends.cudnn.allow_tf32),
                                "cudnn.benchmark": bool(torch.backends.cudnn.benchmark),
                                "cudnn.deterministic": bool(torch.backends.cudnn.deterministic),
                                "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
                                "deterministic_warn_only": bool(torch.is_deterministic_algorithms_warn_only_enabled()),
                                "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                                "float32_matmul_precision": torch.get_float32_matmul_precision()}
        rec["evaluator_code_sha256"] = evaluator_code_sha256()
        log(f"frozen config ok; device {rec['device']}", fh)

        t0 = time.time()
        header, m = load_m02(V4 / "data_v2" / "manifests")
        stream = stream_rows(m)
        rec["manifest"] = {"manifest_id": m.manifest_id, "n_rows": m.n, "n_eval_rows": int(stream.size),
                           "n_rankable_rows": int(m.rankable.sum()), "catalogue_K": m.catalogue_K,
                           "hashes_verified_against_header": dict(m.hashes),
                           "header_content_sha256": header.get("header_content_sha256"),
                           "data_protocol_lock_sha256": header.get("data_protocol_lock_sha256"),
                           "load_s": round(time.time() - t0, 2)}
        log(f"M02 loaded: {rec['manifest']}", fh)

        from v4data.common import VALIDATION_SPLITS, dataset_root
        from v4data.store import CleanV2Store
        t0 = time.time()
        store = CleanV2Store.build(splits=VALIDATION_SPLITS, n_threads=int(args.n_threads), log=lambda s: log(s, fh))
        ds, split_sizes = merged_eval_set(store)
        pos = store.positions_for(ds, m.decision_id[stream])
        if not np.array_equal(ds.target_class[pos], m.target_class[stream]):
            raise AssertionError("store target_class differs from M02 on the evaluation stream")
        rec["store"] = {"build_s": round(time.time() - t0, 1), "eval_decisions_by_split": split_sizes,
                        "rss_mib_after_build": round(rss_mib(), 1)}
        t0 = time.time()
        B = cfg.eval_batch
        batches = [store.batch(ds, pos[a:a + B]) for a in range(0, pos.size, B)]
        rec["batches"] = {"n_batches": len(batches), "eval_batch": B, "tail_rows": int(pos.size - B * (len(batches) - 1)),
                          "prebuild_s": round(time.time() - t0, 1), "rss_mib_after_prebuild": round(rss_mib(), 1)}
        log(f"store + {len(batches)} batches ready: {rec['store']} {rec['batches']}", fh)

        from v4models.build import build
        from v4models.schema import load_schema
        schema = load_schema(SERVER_CODE / "v4data" / "frozen_inputs" / "FEATURE_SCHEMA_CLEAN_V2.json")
        rec["schema"] = {"version": schema.schema_version, "self_sha256": schema.self_sha256}
        rng = np.random.default_rng(20260926)
        sample = sorted(set(list(range(8)) + rng.choice(len(batches), size=max(0, args.sample_batches - 8),
                                                         replace=False).tolist()))
        rec["families"] = {}
        for fam in [f.strip().upper() for f in args.families.split(",") if f.strip()]:
            fr = {}
            t0 = time.time()
            bm = build(fam, schema, seed=2026, device=dev, v1_root=dataset_root())
            fr["model"] = {k: v for k, v in bm.record().items() if k != "inventory"}
            fr["build_s"] = round(time.time() - t0, 1)
            runs = {}
            for tag in ("run1", "run2"):
                torch.cuda.synchronize(dev)
                torch.cuda.reset_peak_memory_stats(dev)
                t0 = time.time()
                art = predict_adapter(bm.adapter, batches, m, config=cfg, model_sha256=bm.init_sha256, device=dev)
                torch.cuda.synchronize(dev)
                runs[tag] = art
                fr[tag] = {"wall_s": round(time.time() - t0, 2),
                           "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated(dev)),
                           "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved(dev)),
                           "ranks_sha256": ranks_sha(art.ranks), "n_tied_sha256": ranks_sha(art.n_tied),
                           "eq_before_sha256": ranks_sha(art.eq_before),
                           "eval_config": art.header["eval_config"]}
                log(f"{fam} {tag}: {fr[tag]['wall_s']} s, peak alloc {fr[tag]['peak_gpu_allocated_bytes']}", fh)
            a1, a2 = runs["run1"], runs["run2"]
            fr["two_invocations_identical"] = bool(np.array_equal(a1.ranks, a2.ranks) and
                                                   np.array_equal(a1.n_tied, a2.n_tied) and
                                                   np.array_equal(a1.eq_before, a2.eq_before))
            torch.cuda.synchronize(dev)
            torch.cuda.reset_peak_memory_stats(dev)
            t0 = time.time()
            a3 = _research_predict_adapter(bm.adapter, batches, m, model_sha256=bm.init_sha256,
                                           score_block=cfg.score_block, class_chunk=cfg.class_chunk, cache_bytes=0,
                                           device=dev)
            torch.cuda.synchronize(dev)
            fr["pass2_recompute"] = {"wall_s": round(time.time() - t0, 2), "cache_bytes": 0,
                                     "nondeterministic_score_error_raised": False,
                                     "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated(dev)),
                                     "ranks_equal_frozen_run": bool(np.array_equal(a3.ranks, a1.ranks)),
                                     "n_tied_equal_frozen_run": bool(np.array_equal(a3.n_tied, a1.n_tied)),
                                     "ranks_sha256": ranks_sha(a3.ranks)}
            log(f"{fam} pass-2 recompute: {fr['pass2_recompute']}", fh)
            fr["single_block_equals_logits"] = single_block_check(bm.adapter, batches, m, dev, sample)
            log(f"{fam} single block == logits: rankable {fr['single_block_equals_logits']['rankable_rows_equal']}, "
                f"full {fr['single_block_equals_logits']['full_batch_equal']}", fh)
            res = evaluate(m, a1, config=cfg, evaluator_code_sha256=rec["evaluator_code_sha256"])
            fr["sanity_random_init_not_a_result"] = {
                "E2E_macro_mrr@20": res["E2E"]["macro"]["mrr@20"], "E2E_macro_ndcg@10": res["E2E"]["macro"]["ndcg@10"],
                "E2E_macro_hr@10": res["E2E"]["macro"]["hr@10"], "E2E_n_decisions": res["E2E"]["n_decisions"],
                "E2E_n_users": res["E2E"]["n_users"], "coverage": res["coverage"],
                "n_target_tied": res["tie_diagnostic"]["n_target_tied"], "max_n_tied": res["tie_diagnostic"]["max_n_tied"],
                "eval_stream": a1.header["eval_config"]["eval_stream"]}
            np.save(out_dir / f"ranks_{fam}_seed2026_run1_int32.npy", a1.ranks.astype(np.int32))
            reference = V4 / "runs_g1" / "r5_t3_receipt" / "out" / f"ranks_{fam}_seed2026_run1_int32.npy"
            if reference.is_file():
                previous = np.load(reference)
                fr["a16_reference"] = {"path": str(reference), "sha256": sha_file(reference),
                                       "same_shape": bool(previous.shape == a1.ranks.shape),
                                       "bitwise_equal": bool(np.array_equal(previous, a1.ranks)),
                                       "diff_count": int(np.count_nonzero(previous != a1.ranks))
                                       if previous.shape == a1.ranks.shape else None}
                # INT-06 review: bind the reference vector to the frozen A16 receipt's run1 ranks_sha256 (a False here
                # means the diff count was taken against some other vector and the family check fails).
                fr["a16_reference"].update(frozen_a16_reference_check(fam, previous))
            else:
                fr["a16_reference"] = {"path": str(reference), "missing": True}
            fr["checks_pass"] = bool(fr["two_invocations_identical"] and fr["pass2_recompute"]["ranks_equal_frozen_run"]
                                     and fr["pass2_recompute"]["n_tied_equal_frozen_run"]
                                     and fr["single_block_equals_logits"]["rankable_rows_equal"]
                                     and fr["single_block_equals_logits"]["full_batch_equal"]
                                     and a1.header["eval_config"]["eval_device_class"] in
                                     ("A16-FP32-DET", "NVIDIA_A40-FP32-DET") and
                                     not fr["a16_reference"].get("missing") and
                                     fr["a16_reference"].get("matches_frozen_a16_receipt") is not False)
            rec["families"][fam] = fr
            del runs, a1, a2, a3, bm
            torch.cuda.empty_cache()
        rec["host_peak_rss_mib"] = round(rss_mib(), 1)
        rec["all_checks_pass"] = all(f["checks_pass"] for f in rec["families"].values())
        rec["status"] = "COMPLETE"
    except Exception as e:                                        # the receipt records the failure; never a success
        rec["status"] = "FAILED"
        rec["error"] = f"{type(e).__name__}: {e}"
        if type(e).__name__ == "NondeterministicScoreError":
            rec["pass2_nondeterministic_score_error"] = True
        log(f"FAILED: {rec['error']}", fh)
    rec["utc_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tmp = out_dir / "T3_GPU_RECEIPT.json.tmp"
    tmp.write_text(json.dumps(rec, indent=1, allow_nan=False, default=str), encoding="utf-8")
    os.replace(tmp, out_dir / "T3_GPU_RECEIPT.json")
    fh.close()
    return 0 if rec["status"] == "COMPLETE" and rec.get("all_checks_pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
