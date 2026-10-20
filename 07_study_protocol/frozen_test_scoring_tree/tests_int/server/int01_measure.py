"""INT-01 server measurements (Lobot only; physical GPU 0 only; nothing is trained for production).

  python -B code/tests_int/server/int01_measure.py run-all          # the orchestrator (run detached)
  python -B code/tests_int/server/int01_measure.py <step> [...]      # one step in a fresh process

Steps (each writes runs_g1/int01/parts/<step>.json; every timestamp from `date -u`; `nvidia-smi` and `free -g` are
recorded before each measurement):
  provider --manifest M      M01 / M05 / M02 / M06 provider: build time (verify + cache), RSS, peak RSS, cache size;
                             sustained rows/s at B = 256 over the trainer's pass order (training) or at 1,024 in
                             manifest order (evaluation; prefetch off and on)
  proof-train                the streaming path (CleanV2Store over ALL TRAIN buckets): build time, RSS, rows/s; the
                             M01 and M05 caches bitwise equal to it on 20,480 random rows + whole trainer batches
  proof-eval                 the same for M02 / M06 against the full VALIDATION streaming store, every batch
  bench-step                 GPU 0: the trainer's step (AdamW fused, clip 1.0, full-K CE, post_step) with pre-loaded GPU
                             tensors vs CPU batches + H2D vs the real provider in the loop (BENCH_NOT_EVIDENCE)
  theta0                     GPU 0: seed-2026 theta0 of both families evaluated on M02 through StoreEvalProvider (the
                             frozen predict path and the trainer's own _evaluate) vs the T3 GPU receipt
  smoke                      GPU 0: launch_central --smoke legs A (600 straight), B1 (300 + interrupt), B2 (resume to
                             600) for GRU then SASRec; digests compared; weights deleted by the launcher
  postupdate-check           CPU: after the population-agnostic update (OWNER_AMENDMENT_POP_5_100), the new code verifies
                             the four measured manifests and rebuilds the M05 / M06 providers with fingerprints equal
                             to the measured ones; the launcher CLI still refuses a non-smoke run without G1
  assemble                   INT01_SERVER_RECEIPT.json from the parts
Labels (binding for every receipt of this run): SMOKE_NOT_EVIDENCE and HISTORICAL_POPULATION — M01 / M05 / M02 / M06
are superseded by the owner amendment; these are data-path / infrastructure measurements only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
V4 = CODE.parent
OUT = V4 / "runs_g1" / "int01"
PARTS = OUT / "parts"
LOGS = OUT / "logs"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
T3_SERVER = V4 / "runs_g1" / "eval_receipt" / "out" / "T3_GPU_RECEIPT.json"
GPU = 0
FAMILIES = ("GRU", "SASREC")


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
            "free_g": sh("free -g"), "df_home": sh("df -h /home/jovyan | tail -1"), "loadavg": list(os.getloadavg()),
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


def batch_digest(b) -> str:
    import torch
    h = hashlib.sha256()
    for k in sorted(b):
        v = b[k]
        if torch.is_tensor(v):
            t = v.contiguous()
            h.update(f"{k}|{t.dtype}|{tuple(t.shape)}".encode())
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        else:
            h.update(f"{k}|{v!r}".encode())
    return h.hexdigest()


def _stats(ts) -> dict:
    import numpy as np
    a = np.asarray(ts, dtype=np.float64)
    return {"n": int(a.size), "total_s": round(float(a.sum()), 3), "median_ms": round(float(np.median(a)) * 1e3, 3),
            "p90_ms": round(float(np.percentile(a, 90)) * 1e3, 3), "max_ms": round(float(a.max()) * 1e3, 3)}


# ------------------------------------------------------------------------------------------------ CPU steps
def cmd_provider(a) -> int:
    import v4train  # noqa: F401  (ppsi_env first)
    import numpy as np
    from v4int.cache import peak_rss_bytes, rss_bytes
    from v4int.providers import StoreEvalProvider, StoreTrainProvider
    from v4train.order import pass_permutation
    mid = a.manifest
    rec = {"step": f"provider_{mid}", "utc_start": utc(), "resources_before": resources(),
           "rss_bytes_start": rss_bytes()}
    t0 = time.time()
    if mid in ("M01_COHORT_TRAIN", "M05_CALIB_TRAIN"):
        p = StoreTrainProvider(mid, log=log)
        rec["build_s"] = round(time.time() - t0, 2)
        rec["rss_bytes_after_build"], rec["peak_rss_bytes_after_build"] = rss_bytes(), peak_rss_bytes()
        N = p.n_decisions
        perm = pass_permutation(2026, 0, N)
        nb = min(int(a.batches), N // 256)
        for i in range(20):                                               # warm-up (not timed)
            p.batch(perm[i * 256:(i + 1) * 256])
        ts = []
        for i in range(20, 20 + nb):
            pos = perm[i * 256:(i + 1) * 256]
            t = time.perf_counter()
            p.batch(pos)
            ts.append(time.perf_counter() - t)
        st = _stats(ts)
        rec["train_B256_trainer_order"] = dict(st, rows=nb * 256, rows_per_s=round(nb * 256 / st["total_s"], 1),
                                               access="seed-2026 pass-0 permutation slices (random positions)")
    else:
        p = StoreEvalProvider(mid, prefetch=0, log=log)
        rec["build_s"] = round(time.time() - t0, 2)
        rec["rss_bytes_after_build"], rec["peak_rss_bytes_after_build"] = rss_bytes(), peak_rss_bytes()
        for depth in (0, 2):
            p.prefetch = depth
            ts, n, t = [], 0, time.perf_counter()
            last = t
            for b in p.batches(p.eval_batch):
                now = time.perf_counter()
                ts.append(now - last)
                last = now
                n += int(b["decision_id"].shape[0])
            wall = time.perf_counter() - t
            rec[f"eval_B1024_in_order_prefetch{depth}"] = dict(_stats(ts), rows=n, wall_s=round(wall, 3),
                                                               rows_per_s=round(n / wall, 1))
    rec["describe"] = p.describe()
    rec["rss_bytes_end"], rec["peak_rss_bytes_end"] = rss_bytes(), peak_rss_bytes()
    rec["utc_end"] = utc()
    write_part(rec["step"], rec)
    return 0


def cmd_proof_train(a) -> int:
    import v4train  # noqa: F401
    import numpy as np
    from v4data.store import CleanV2Store
    from v4int.cache import peak_rss_bytes, rss_bytes
    from v4int.providers import StoreTrainProvider
    from v4train.order import pass_permutation
    rec = {"step": "proof_train", "utc_start": utc(), "resources_before": resources()}
    provs = {m: StoreTrainProvider(m, log=log) for m in ("M01_COHORT_TRAIN", "M05_CALIB_TRAIN")}
    rec["rss_bytes_after_caches"], rec["peak_rss_bytes_after_caches"] = rss_bytes(), peak_rss_bytes()
    state = provs["M01_COHORT_TRAIN"].verified.transform_state_self_sha256
    t0 = time.time()
    full = CleanV2Store.build(splits=("TRAIN",), n_threads=3, state_sha256=state, log=log)
    rec["stream_store"] = {"build_s": round(time.time() - t0, 2), "n_event_rows": int(full.cs.n_rows),
                           "event_array_nbytes": int(sum(getattr(full.cs, x).nbytes for x in (
                               "_item_tokens", "_category_tokens", "_brand_tokens", "_main_category_tokens",
                               "_daypart_tokens", "_weekday_tokens", "_event_numeric_features",
                               "_event_quality_flags", "_user_context", "_user_context_masks"))),
                           "rss_bytes_after_build": rss_bytes(), "peak_rss_bytes_after_build": peak_rss_bytes(),
                           "resources_after_build": resources()}
    ds = full.decision_set("TRAIN", "loss")
    rec["stream_store"]["n_loss_decisions"] = int(len(ds))
    for mid, tp in provs.items():
        N = tp.n_decisions
        spos = full.positions_for(ds, tp.decision_ids())
        perm = pass_permutation(2026, 0, N)
        nb = min(int(a.batches), N // 256)
        tc, tsr = [], []
        for i in range(nb):
            pos = perm[i * 256:(i + 1) * 256]
            t = time.perf_counter()
            full.batch(ds, spos[pos])
            tsr.append(time.perf_counter() - t)
            t = time.perf_counter()
            tp.batch(pos)
            tc.append(time.perf_counter() - t)
        s_st, c_st = _stats(tsr[20:]), _stats(tc[20:])
        m = {"stream_B256": dict(s_st, rows_per_s=round((nb - 20) * 256 / s_st["total_s"], 1)),
             "cache_B256": dict(c_st, rows_per_s=round((nb - 20) * 256 / c_st["total_s"], 1))}
        sample = pass_permutation(20260926, 0, N)[:80 * 256]            # 20,480 distinct random rows
        eq, rows, first_bad, dig = True, 0, None, hashlib.sha256()
        groups = [("random", sample[i * 256:(i + 1) * 256]) for i in range(80)]
        tail = (N // 256) * 256
        for s in (0, 256, 512, tail):                                    # 3 whole trainer batches + the tail
            groups.append((f"trainer_batch@{s}", perm[s:min(N, s + 256)]))
        for tag, pos in groups:
            c, s_ = tp.batch(pos), full.batch(ds, spos[pos])
            ok, why = batches_equal(c, s_)
            dig.update(batch_digest(s_).encode())
            rows += int(pos.size)
            if not ok and first_bad is None:
                first_bad = {"group": tag, "why": why}
            eq &= ok
        m["bitwise_proof"] = {"all_equal": bool(eq), "rows_compared": rows, "random_rows": 80 * 256,
                              "whole_trainer_batches": ["0", "256", "512", f"tail@{tail}"],
                              "first_mismatch": first_bad, "stream_batch_digests_sha256": dig.hexdigest(),
                              "note": "every tensor (dtype, shape, torch.equal) and every non-tensor field compared"}
        rec[mid] = m
        log(f"{mid}: proof all_equal={eq} rows={rows}; stream {m['stream_B256']['rows_per_s']} rows/s, "
            f"cache {m['cache_B256']['rows_per_s']} rows/s")
    rec["rss_bytes_end"], rec["peak_rss_bytes_end"] = rss_bytes(), peak_rss_bytes()
    rec["utc_end"] = utc()
    write_part("proof_train", rec)
    return 0 if all(rec[m]["bitwise_proof"]["all_equal"] for m in provs) else 1


def cmd_proof_eval(a) -> int:
    import v4train  # noqa: F401
    from v4data.common import VALIDATION_SPLITS
    from v4data.store import CleanV2Store
    from v4eval.predict import stream_rows
    from v4int.cache import merged_decision_set, peak_rss_bytes, rss_bytes
    from v4int.providers import StoreEvalProvider
    rec = {"step": "proof_eval", "utc_start": utc(), "resources_before": resources()}
    provs = {m: StoreEvalProvider(m, prefetch=0, log=log) for m in ("M02_VALIDATION_WARM", "M06_CALIB_VALIDATION_WARM")}
    state = provs["M02_VALIDATION_WARM"].verified.transform_state_self_sha256
    t0 = time.time()
    full = CleanV2Store.build(splits=VALIDATION_SPLITS, n_threads=3, state_sha256=state, log=log)
    ds = merged_decision_set(full, VALIDATION_SPLITS, "eval")
    rec["stream_store"] = {"build_s": round(time.time() - t0, 2), "n_event_rows": int(full.cs.n_rows),
                           "n_eval_decisions": int(len(ds)), "rss_bytes_after_build": rss_bytes(),
                           "peak_rss_bytes_after_build": peak_rss_bytes()}
    for mid, ep in provs.items():
        pos = full.positions_for(ds, ep.manifest.decision_id[stream_rows(ep.manifest)])
        B = ep.eval_batch
        eq, nb, rows, first_bad, tsr, tc = True, 0, 0, None, [], []
        dig = hashlib.sha256()
        it = ep.batches(B)
        for k in range(0, pos.size, B):
            t = time.perf_counter()
            s_ = full.batch(ds, pos[k:k + B])
            tsr.append(time.perf_counter() - t)
            t = time.perf_counter()
            c = next(it)
            tc.append(time.perf_counter() - t)
            ok, why = batches_equal(c, s_)
            dig.update(batch_digest(s_).encode())
            if not ok and first_bad is None:
                first_bad = {"batch": nb, "why": why}
            eq &= ok
            nb += 1
            rows += int(c["decision_id"].shape[0])
        s_st, c_st = _stats(tsr), _stats(tc)
        rec[mid] = {"bitwise_proof": {"all_equal": bool(eq), "batches_compared": nb, "rows_compared": rows,
                                      "first_mismatch": first_bad, "stream_batch_digests_sha256": dig.hexdigest()},
                    "stream_B1024": dict(s_st, rows_per_s=round(rows / s_st["total_s"], 1)),
                    "cache_B1024": dict(c_st, rows_per_s=round(rows / c_st["total_s"], 1))}
        log(f"{mid}: proof all_equal={eq} batches={nb} rows={rows}")
    rec["rss_bytes_end"], rec["peak_rss_bytes_end"] = rss_bytes(), peak_rss_bytes()
    rec["utc_end"] = utc()
    write_part("proof_eval", rec)
    return 0 if all(rec[m]["bitwise_proof"]["all_equal"] for m in provs) else 1


# ------------------------------------------------------------------------------------------------ GPU steps
def _gpu_env() -> dict:
    """The launcher's export (CUDA_VISIBLE_DEVICES=0, the T3 CUBLAS value) for a GPU step's fresh interpreter."""
    import v4int.launch_central as L
    env = {k: v for k, v in os.environ.items() if k not in ("CUDA_VISIBLE_DEVICES",)}
    L.export_environment(GPU, env)
    return env


def _gpu_free() -> tuple:
    import v4int.launch_central as L
    from v4sup.gpu import gpu_ready
    ok, why = gpu_ready([GPU])
    if not ok:
        return False, why
    try:
        mem = L._gpu_memory_used(GPU, __import__("v4sup.gpu", fromlist=["default_runner"]).default_runner)
    except Exception as e:
        return False, f"nvidia-smi: {e}"
    used = mem[str(GPU)]["memory_used_mib"]
    return (used <= L.GPU_MEM_FOREIGN_MIB), f"memory.used {used} MiB"


def _in_gpu_process_checks() -> dict:
    import v4train
    import torch
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(GPU) or v4train.LAUNCH_CUDA_VISIBLE_DEVICES != str(GPU):
        raise RuntimeError("CUDA_VISIBLE_DEVICES was not exported before the import")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected one visible GPU, got {torch.cuda.device_count()}")
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    return {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "device": torch.cuda.get_device_name(dev),
            "uuid": str(getattr(torch.cuda.get_device_properties(dev), "uuid", ""))}


def cmd_bench_step(a) -> int:
    import v4train  # noqa: F401  (captures CUDA_VISIBLE_DEVICES, then ppsi_env)
    import numpy as np
    import torch
    from fl.numerics import enforce_strict_fp32
    from v4data.common import dataset_root
    from v4int.cache import peak_rss_bytes, rss_bytes
    from v4int.launch_central import determinism_mode_in_effect
    from v4int.providers import StoreTrainProvider
    from v4models.build import build
    from v4models.schema import load_schema
    from v4train.objective import accumulate_step, host_contributing
    from v4train.order import pass_permutation
    from v4train.recipe import RECIPE
    rec = {"step": "bench_step", "label": "BENCH_NOT_EVIDENCE (no weights saved, no row written)", "utc_start": utc(),
           "resources_before": resources(), "process": _in_gpu_process_checks()}
    dev = torch.device("cuda:0")
    tp = StoreTrainProvider("M05_CALIB_TRAIN", log=log)
    N = tp.n_decisions
    perm = pass_permutation(2026, 0, N)
    NB = 64
    cpu_b = [tp.batch(perm[i * 256:(i + 1) * 256]) for i in range(NB)]
    n_eff = [host_contributing(b) for b in cpu_b]
    to_gpu = lambda b: {k: (v.to(dev) if torch.is_tensor(v) and k != "lengths" else v) for k, v in b.items()}  # noqa: E731
    gpu_b = [to_gpu(b) for b in cpu_b]
    schema = load_schema(CODE / "v4data" / "frozen_inputs" / "FEATURE_SCHEMA_CLEAN_V2.json")
    r = RECIPE
    steps, warm = int(a.steps), 20
    rec["families"] = {}
    for fam in FAMILIES:
        bm = build(fam, schema, 2026, dev, v1_root=dataset_root())
        ad = bm.adapter
        opt = torch.optim.AdamW(ad.param_groups(r.weight_decay), lr=1e-3, betas=r.betas, eps=r.eps, fused=r.fused)
        shared = ad.shared_parameters()
        enforce_strict_fp32(deterministic=True)                         # the trainer's training flags (pre-R1 default)
        bm.module.train()
        torch.manual_seed(2026)

        def step(batch, n):
            opt.zero_grad(set_to_none=True)
            ls = accumulate_step(ad.scores, batch, n, 0, dev)
            torch.nn.utils.clip_grad_norm_(shared, r.grad_clip)
            opt.step()
            ad.post_step()
            return ls

        fr = {"determinism": determinism_mode_in_effect()}
        for mode in ("pure_gpu_preloaded", "cpu_batches_h2d", "provider_in_loop"):
            torch.cuda.synchronize(dev)
            torch.cuda.reset_peak_memory_stats(dev)
            t0 = None
            for i in range(warm + steps):
                if i == warm:
                    torch.cuda.synchronize(dev)
                    t0 = time.perf_counter()
                j = i % NB
                if mode == "pure_gpu_preloaded":
                    ls = step(gpu_b[j], n_eff[j])
                elif mode == "cpu_batches_h2d":
                    ls = step(cpu_b[j], n_eff[j])
                else:
                    k = (NB + i) % (N // 256)
                    b = tp.batch(perm[k * 256:(k + 1) * 256])
                    ls = step(b, host_contributing(b))
                if (i + 1) % 50 == 0:
                    float(ls.item())                                     # the trainer's monitor sync cadence
            torch.cuda.synchronize(dev)
            wall = time.perf_counter() - t0
            fr[mode] = {"steps_timed": steps, "wall_s": round(wall, 3), "updates_per_s": round(steps / wall, 3),
                        "decisions_per_s": round(steps * 256 / wall, 1),
                        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(dev)),
                        "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(dev))}
            log(f"{fam} {mode}: {fr[mode]['updates_per_s']} updates/s")
        pg, pv = fr["pure_gpu_preloaded"]["updates_per_s"], fr["provider_in_loop"]["updates_per_s"]
        fr["provider_vs_pure_gpu_ratio"] = round(pv / pg, 4)
        fr["data_bound"] = bool(pv < 0.95 * pg)
        rec["families"][fam] = fr
        del bm, ad, opt, shared
        torch.cuda.empty_cache()
    rec["host_rss_bytes"], rec["host_peak_rss_bytes"] = rss_bytes(), peak_rss_bytes()
    rec["utc_end"] = utc()
    write_part("bench_step", rec)
    return 0


def cmd_theta0(a) -> int:
    import v4train  # noqa: F401
    import numpy as np
    import torch
    from v4data.common import dataset_root
    from v4eval import evaluator_code_sha256
    from v4eval import frozen_config as fc
    from v4eval.evaluate import evaluate
    from v4eval.predict import predict_adapter
    from v4int.cache import peak_rss_bytes, rss_bytes
    from v4int.launch_central import DETERMINISM_MODE
    from v4int.providers import StoreEvalProvider, StoreTrainProvider, comparison_key_base
    from v4int.verify import DataPaths
    from v4models.schema import load_schema
    from v4train.recipe import RunConfig
    from v4train.trainer import CentralTrainer
    rec = {"step": "theta0", "label": "SANITY_RANDOM_INIT_NOT_A_RESULT", "utc_start": utc(),
           "resources_before": resources(), "process": _in_gpu_process_checks()}
    t3 = json.loads(T3_SERVER.read_text(encoding="utf-8"))
    rec["t3_receipt"] = {"path": str(T3_SERVER), "sha256": hashlib.sha256(T3_SERVER.read_bytes()).hexdigest(),
                         "evaluator_code_sha256": t3["evaluator_code_sha256"], "torch": t3["torch"]}
    dev = torch.device("cuda:0")
    paths = DataPaths.default()
    tp = StoreTrainProvider("M01_COHORT_TRAIN", paths=paths, log=log)
    ep = StoreEvalProvider("M02_VALIDATION_WARM", paths=paths, log=log)          # production prefetch depth
    rec["providers"] = {"rss_bytes_after": rss_bytes(), "peak_rss_bytes_after": peak_rss_bytes(),
                        "train_fingerprint": tp.fingerprint, "eval_fingerprint": ep.fingerprint,
                        "eval_stream": ep.eval_stream, "eval_stream_source": ep.eval_stream_source,
                        "manifest_hashes": ep.manifest.hashes}
    key = comparison_key_base(paths, tp, ep)
    schema = load_schema(CODE / "v4data" / "frozen_inputs" / "FEATURE_SCHEMA_CLEAN_V2.json")
    sha = lambda x: hashlib.sha256(np.ascontiguousarray(x, dtype="<i8").tobytes()).hexdigest()  # noqa: E731
    fz = fc.load()
    rec["families"] = {}
    for fam in FAMILIES:
        extra = {"determinism_mode": DETERMINISM_MODE} if "determinism_mode" in RunConfig.__dataclass_fields__ else {}
        cfg = RunConfig(run_id=f"INT01_THETA0_{fam}", family=fam, seed=2026, exposure_point="end",
                        cuda_visible_devices=str(GPU), phase="MAIN", population="COHORT", peak_lr=1e-3, key=key,
                        test_overrides=True, **extra)
        rd = OUT / "theta0_tmp" / fam
        tr = CentralTrainer(cfg, schema, tp, ep, rd, device="cuda:0", v1_root=dataset_root())
        want = t3["families"][fam]
        fr = {"init_sha256": tr.built.init_sha256, "t3_init_sha256": want["model"]["init_sha256"]}
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)
        t0 = time.time()
        fc.apply_runtime_flags(fz)
        art = predict_adapter(tr.adapter, ep.batches(fz.eval_batch), ep.manifest, config=fz,
                              model_sha256=tr.built.init_sha256)
        res = evaluate(ep.manifest, art, config=fz, evaluator_code_sha256=evaluator_code_sha256())
        torch.cuda.synchronize(dev)
        fr["direct"] = {"wall_s": round(time.time() - t0, 2), "ranks_sha256": sha(art.ranks),
                        "n_tied_sha256": sha(art.n_tied), "eq_before_sha256": sha(art.eq_before),
                        "E2E_macro_mrr@20": res["E2E"]["macro"]["mrr@20"],
                        "E2E_macro_ndcg@10": res["E2E"]["macro"]["ndcg@10"],
                        "E2E_macro_hr@10": res["E2E"]["macro"]["hr@10"], "E2E_n_decisions": res["E2E"]["n_decisions"],
                        "E2E_n_users": res["E2E"]["n_users"], "coverage": res["coverage"],
                        "n_target_tied": res["tie_diagnostic"]["n_target_tied"],
                        "max_n_tied": res["tie_diagnostic"]["max_n_tied"],
                        "eval_stream": art.header["eval_config"]["eval_stream"],
                        "eval_device_class": art.header["eval_config"]["eval_device_class"],
                        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(dev)),
                        "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(dev))}
        del art
        t0 = time.time()
        out = tr._evaluate(0.0)                                          # the trainer's production evaluation call
        fr["trainer_evaluate"] = {"wall_s": round(time.time() - t0, 2), "ranks_sha256": sha(out.ranks),
                                  "value_E2E_macro_mrr@20": out.value,
                                  "eval_stream": out.result["eval_config"]["eval_stream"]}
        t3s = want["sanity_random_init_not_a_result"]
        d = fr["direct"]
        fr["equal_to_t3"] = {
            "init_sha256": fr["init_sha256"] == fr["t3_init_sha256"],
            "ranks_sha256": d["ranks_sha256"] == want["run1"]["ranks_sha256"],
            "n_tied_sha256": d["n_tied_sha256"] == want["run1"]["n_tied_sha256"],
            "eq_before_sha256": d["eq_before_sha256"] == want["run1"]["eq_before_sha256"],
            "trainer_ranks_sha256": fr["trainer_evaluate"]["ranks_sha256"] == want["run1"]["ranks_sha256"],
            **{k: d[k] == t3s[k] for k in ("E2E_macro_mrr@20", "E2E_macro_ndcg@10", "E2E_macro_hr@10",
                                           "E2E_n_decisions", "E2E_n_users", "coverage", "n_target_tied", "max_n_tied",
                                           "eval_stream")},
            "trainer_value": fr["trainer_evaluate"]["value_E2E_macro_mrr@20"] == t3s["E2E_macro_mrr@20"]}
        fr["all_equal"] = all(fr["equal_to_t3"].values())
        rec["families"][fam] = fr
        log(f"{fam}: theta0 equal to T3: {fr['all_equal']} ({fr['equal_to_t3']})")
        del tr, out
        torch.cuda.empty_cache()
        shutil.rmtree(rd, ignore_errors=True)
    shutil.rmtree(OUT / "theta0_tmp", ignore_errors=True)
    rec["theta0_tmp_removed"] = not (OUT / "theta0_tmp").exists()
    rec["host_rss_bytes"], rec["host_peak_rss_bytes"] = rss_bytes(), peak_rss_bytes()
    rec["all_equal"] = all(f["all_equal"] for f in rec["families"].values())
    rec["utc_end"] = utc()
    write_part("theta0", rec)
    return 0 if rec["all_equal"] else 1


# ------------------------------------------------------------------------------------------------ smoke
def smoke_legs(fam: str) -> list:
    base = [sys.executable, "-B", str(CODE / "v4int" / "launch_central.py"), "--family", fam, "--seed", "2026",
            "--phase", "CALIBRATION", "--peak-lr", "1e-3", "--gpu", str(GPU), "--smoke", "--smoke-updates", "600",
            "--checkpoint-every", "300"]
    d = OUT / "smoke" / fam
    return [("A_straight_600", base + ["--run-dir", str(d / "A")]),
            ("B1_interrupt_300", base + ["--run-dir", str(d / "B"), "--smoke-interrupt-at", "300"]),
            ("B2_resume_to_600", base + ["--run-dir", str(d / "B"), "--resume"])]


def cmd_postupdate_check(a) -> int:
    import v4train  # noqa: F401
    from v4int.providers import StoreEvalProvider, StoreTrainProvider
    from v4int.verify import verify_manifest
    rec = {"step": "postupdate_check", "labels": ["HISTORICAL_POPULATION"], "utc_start": utc(),
           "resources_before": resources(), "disk_before": _free_bytes()}
    rec["verify"] = {}
    for mid in ("M01_COHORT_TRAIN", "M05_CALIB_TRAIN", "M02_VALIDATION_WARM", "M06_CALIB_VALIDATION_WARM"):
        vm = verify_manifest(mid, log=log)
        rec["verify"][mid] = {"kind": vm.kind, "checks_passed": len(vm.checks),
                              "all_true": all(v is True for v in vm.checks.values()),
                              "n_decisions": int(vm.canonical_decision_id.size)}
    fps = {}
    for mid, cls in (("M05_CALIB_TRAIN", StoreTrainProvider), ("M06_CALIB_VALIDATION_WARM", StoreEvalProvider)):
        p = cls(mid, log=log)
        was = json.loads((PARTS / f"provider_{mid}.json").read_text(encoding="utf-8"))["describe"]["fingerprint"]
        fps[mid] = {"new_code": p.fingerprint, "measured": was, "equal": p.fingerprint == was}
        del p
    rec["fingerprints"] = fps
    q = subprocess.run([sys.executable, "-B", str(CODE / "v4int" / "launch_central.py"), "--family", "GRU", "--seed",
                        "2026", "--phase", "CALIBRATION", "--peak-lr", "1e-3", "--gpu", str(GPU), "--run-dir",
                        str(OUT / "postupdate_nonsmoke_probe")], cwd=str(CODE), capture_output=True, text=True,
                       env={k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"})
    rec["launcher_non_smoke_without_g1"] = {"exit_code": q.returncode, "stderr_tail": q.stderr.strip()[-300:],
                                           "run_dir_created": (OUT / "postupdate_nonsmoke_probe").exists()}
    rec["all_ok"] = (all(v["all_true"] for v in rec["verify"].values()) and all(v["equal"] for v in fps.values())
                     and q.returncode == 2 and not rec["launcher_non_smoke_without_g1"]["run_dir_created"])
    rec["utc_end"], rec["disk_after"] = utc(), _free_bytes()
    write_part("postupdate_check", rec)
    return 0 if rec["all_ok"] else 1


def cmd_smoke_compare(a) -> int:
    rec = {"step": "smoke_compare", "utc": utc(), "families": {}}
    for fam in FAMILIES:
        d = OUT / "smoke" / fam
        try:
            A = json.loads((d / "A" / "SMOKE_SUMMARY.json").read_text(encoding="utf-8"))
            B1 = json.loads((d / "B" / "SMOKE_LEG_INTERRUPTED.json").read_text(encoding="utf-8"))
            B = json.loads((d / "B" / "SMOKE_SUMMARY.json").read_text(encoding="utf-8"))
        except OSError as e:
            rec["families"][fam] = {"error": f"missing smoke summary: {e}"}
            continue
        keys = ("model_state_dict", "optimizer_state_dict", "rng_torch_cpu", "rng_cuda", "train_state", "stop_state",
                "updates", "exposures")
        eq = {k: A["digests"][k] == B["digests"][k] for k in keys}
        pts = [p for p in (OUT / "smoke" / fam).rglob("*.pt")]
        rec["families"][fam] = {
            "digests_equal": eq, "resume_bitwise": all(eq.values()), "updates": [A["digests"]["updates"],
                                                                                   B1["digests"]["updates"],
                                                                                   B["digests"]["updates"]],
            "eval_ranks_equal": A["eval"]["ranks_sha256"] == B["eval"]["ranks_sha256"],
            "trainer_code_sha256": sorted({A.get("trainer_code_sha256"), B1.get("trainer_code_sha256"),
                                           B.get("trainer_code_sha256")} - {None}),
            "A": {"updates_per_s": A["timing"]["updates_per_s_after_warmup_excl_checkpoints"],
                  "data_wait_s": A["timing"]["monitor_data_wait_s"], "gpu_memory_train": A["gpu_memory_train"],
                  "eval": A["eval"], "host_peak_rss_bytes": A["host_peak_rss_bytes"],
                  "weights_deleted": A["weights_deleted"], "determinism": A.get("determinism"),
                  "disk": {k: A.get(k) for k in ("disk_at_start", "disk_at_stop", "disk_at_end")}},
            "B1": {"disk": {k: B1.get(k) for k in ("disk_at_start", "disk_at_stop", "disk_at_end")},
                   "weights_retained_for_the_resume_leg": B1.get("weights_retained_for_the_resume_leg")},
            "B2": {"updates_per_s": B["timing"]["updates_per_s_after_warmup_excl_checkpoints"],
                   "host_peak_rss_bytes": B["host_peak_rss_bytes"], "weights_deleted": B["weights_deleted"],
                   "eval_wall_s": B["eval"]["eval_wall_s"],
                   "disk": {k: B.get(k) for k in ("disk_at_start", "disk_at_stop", "disk_at_end")}},
            "weight_files_left": [str(p) for p in pts]}
    write_part("smoke_compare", rec)
    ok = all(f.get("resume_bitwise") and not f.get("weight_files_left") for f in rec["families"].values())
    return 0 if ok else 1


# ------------------------------------------------------------------------------------------------ orchestration
def _free_bytes() -> dict:
    return {"utc": utc(), "df_B1": sh("df -B1 /home/jovyan | tail -1"),
            "free_bytes": int(shutil.disk_usage("/home/jovyan").free)}


def _step(name: str, argv: list, env=None, timeout=7200) -> dict:
    LOGS.mkdir(parents=True, exist_ok=True)
    r = {"step": name, "utc_start": utc(), "resources_before": resources(), "disk_before": _free_bytes(),
         "argv": argv[1:]}
    with open(LOGS / f"{name}.log", "w", encoding="utf-8") as fh:
        p = subprocess.run(argv, cwd=str(CODE), env=env or dict(os.environ), stdout=fh, stderr=subprocess.STDOUT,
                           timeout=timeout)
    r.update(exit_code=p.returncode, utc_end=utc(), disk_after=_free_bytes())
    with open(OUT / "steps.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(r, default=str) + "\n")
    log(f"step {name}: exit {p.returncode}")
    return r


def _wait_gpu(max_min=90) -> tuple:
    for i in range(max_min):
        ok, why = _gpu_free()
        if ok:
            return True, why
        log(f"GPU {GPU} busy ({why}); waiting (never killing a foreign process)")
        time.sleep(60)
    return False, why


def cmd_run_all(a) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    me = [sys.executable, "-B", str(Path(__file__).resolve())]
    cpu_env = {k: v for k, v in os.environ.items()}
    cpu_env["CUDA_VISIBLE_DEVICES"] = ""                               # CPU steps see no GPU
    cpu_env["PYTHONDONTWRITEBYTECODE"] = "1"
    only = set(a.only.split(",")) if a.only else None
    want = lambda s: only is None or s in only  # noqa: E731
    with open(OUT / "RUN_ALL_STARTED.json", "w", encoding="utf-8") as fh:
        json.dump({"utc": utc(), "host": platform.node(), "only": a.only}, fh)
    for mid in ("M01_COHORT_TRAIN", "M05_CALIB_TRAIN", "M02_VALIDATION_WARM", "M06_CALIB_VALIDATION_WARM"):
        if want(f"provider_{mid}"):
            _step(f"provider_{mid}", me + ["provider", "--manifest", mid], env=cpu_env)
    if want("proof_train"):
        _step("proof_train", me + ["proof-train"], env=cpu_env)
    if want("proof_eval"):
        _step("proof_eval", me + ["proof-eval"], env=cpu_env)
    for name, argv in (("bench_step", me + ["bench-step"]), ("theta0", me + ["theta0"])):
        if want(name):
            ok, why = _wait_gpu()
            if not ok:
                _record_skip(name, f"GPU {GPU} not free: {why}")
                continue
            _step(name, argv, env=_gpu_env())
    if want("smoke"):
        launcher_env = {k: v for k, v in os.environ.items() if k not in ("CUDA_VISIBLE_DEVICES",
                                                                         "CUBLAS_WORKSPACE_CONFIG")}
        stop = False
        for fam in FAMILIES:
            for leg, argv in smoke_legs(fam):
                ok, why = _wait_gpu()
                if not ok:
                    _record_skip(f"smoke_{fam}_{leg}", f"GPU not available: {why}")
                    stop = True
                    break
                # the launcher refuses (exit 2) when free - planned peak < 8 GiB: stop and report, never wait/run
                r = _step(f"smoke_{fam}_{leg}", argv, env=launcher_env)
                if r["exit_code"] != 0:
                    _record_skip(f"smoke_{fam}_remaining_legs", f"leg {leg} exit {r['exit_code']}: smoke stopped")
                    stop = True
                    break
            if stop:
                break
        guard = []                                          # end of the smoke job: no weight may outlive it
        for q in sorted((OUT / "smoke").rglob("*.pt*")) if (OUT / "smoke").exists() else []:
            guard.append({"file": str(q), "bytes": q.stat().st_size,
                          "sha256": hashlib.sha256(q.read_bytes()).hexdigest()})
            q.unlink()
        with open(OUT / "steps.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"step": "smoke_weight_guard", "utc": utc(), "leftover_deleted": guard,
                                 "disk_after": _free_bytes()}) + "\n")
        _step("smoke_compare", me + ["smoke-compare"], env=cpu_env)
    _step("assemble", me + ["assemble"], env=cpu_env)
    (OUT / "RUN_ALL_DONE").write_text(utc(), encoding="utf-8")
    return 0


def _record_skip(name, why):
    with open(OUT / "steps.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"step": name, "skipped": why, "utc": utc()}) + "\n")
    log(f"step {name} SKIPPED: {why}")


def cmd_assemble(a) -> int:
    parts = {}
    for p in sorted(PARTS.glob("*.json")):
        parts[p.stem] = json.loads(p.read_text(encoding="utf-8"))
    steps = []
    sp = OUT / "steps.jsonl"
    if sp.exists():
        steps = [json.loads(x) for x in sp.read_text(encoding="utf-8").splitlines() if x.strip()]
    labels = {"labels": ["SMOKE_NOT_EVIDENCE", "HISTORICAL_POPULATION"],
              "label_notes": "every smoke row / result is SMOKE_NOT_EVIDENCE (never valid for any selection); M01 / "
                             "M05 / M02 / M06 are HISTORICAL after OWNER_AMENDMENT_POP_5_100 (DECISION_LOG 31): these "
                             "are data-path and infrastructure measurements (throughput, RSS, bytes per row, bitwise "
                             "cache = stream, resume, theta0 consistency), never population evidence"}
    rec = {"receipt": "INT01_SERVER_RECEIPT", "task": "INT-01", **labels, "host": platform.node(),
           "utc_assembled": utc(), "tags": "PROJECT_MEASURED (server, this run)", "gpu": f"physical GPU {GPU} only",
           "code_sha256_at_measurement": parts.get("code_at_measurement", {}).get("code_sha256"),
           "code_sha256_at_assembly": {str(p.relative_to(CODE)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for d in ("v4int", "tests_int") for p in sorted((CODE / d).rglob("*.py"))
                           if "__pycache__" not in p.parts},
           "steps": steps, "parts": parts,
           "pt_files_left_under_int01": [str(p) for p in OUT.rglob("*.pt")]}
    tmp = OUT / "INT01_SERVER_RECEIPT.json.tmp"
    tmp.write_text(json.dumps(rec, indent=1, allow_nan=False, default=str), encoding="utf-8")
    os.replace(tmp, OUT / "INT01_SERVER_RECEIPT.json")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("postupdate-check")
    p = sub.add_parser("provider")
    p.add_argument("--manifest", required=True)
    p.add_argument("--batches", type=int, default=2000)
    p = sub.add_parser("proof-train")
    p.add_argument("--batches", type=int, default=600)
    sub.add_parser("proof-eval")
    p = sub.add_parser("bench-step")
    p.add_argument("--steps", type=int, default=200)
    sub.add_parser("theta0")
    sub.add_parser("smoke-compare")
    sub.add_parser("assemble")
    p = sub.add_parser("run-all")
    p.add_argument("--only", default=None)
    a = ap.parse_args(argv)
    return {"provider": cmd_provider, "proof-train": cmd_proof_train, "proof-eval": cmd_proof_eval,
            "bench-step": cmd_bench_step, "theta0": cmd_theta0, "smoke-compare": cmd_smoke_compare,
            "postupdate-check": cmd_postupdate_check,
            "assemble": cmd_assemble, "run-all": cmd_run_all}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
