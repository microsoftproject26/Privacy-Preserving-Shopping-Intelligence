"""fl.gpu_receipt — the GPU determinism receipt on the 2 x A16 (FL-02 item 4; Fable G0(b) BLOCKERS B).

Every check uses the CLEAN V2 families (v4models via fl.families.build_v2) at the full K = 158,486, synthetic inputs,
strict FP32 (matmul and cuDNN TF32 off), CUBLAS_WORKSPACE_CONFIG=:4096:8 and torch.use_deterministic_algorithms(True)
WITHOUT warn_only. If an op has no deterministic implementation, the error text is recorded verbatim and the check is
re-run in warn_only mode (its warnings recorded verbatim) so the bitwise verdict is still measured and labelled.

  init    per family and seed 2026 / 2027: the registered theta0 hash (and the raw pre-recentering hash) measured on
          this host, compared with MODEL-01's inventory; plus a CPU -> GPU -> CPU round trip of theta0.
  t34     2 worker processes, one per GPU, n_shards = 2; FP (mu 0.01) and PF rounds; three repeats: shard->GPU
          rotation 0, 0 (repeat) and 1 (swap). Verdicts: repeat bitwise and swap bitwise (server state + PF store).
  lo      the procedural LO recipe (6 passes, persistent fused AdamW, s(EFE)-shaped LR, eval at 2/4/6) over 1,024
          synthetic clients on one GPU; per-client on-device fingerprint + eval floats (+ sha256 for every 64th).
          Two runs (cuda:0 forward order, cuda:1 reverse order) are compared by `summarize`.
  resume  GRU FA over a plan of 8 clients in groups of 3 (3 rounds per sweep, a partial last group of 2): 3 rounds
          uninterrupted vs 2 rounds -> ONE checkpoint -> fresh process -> resume -> 1 round; the temporary checkpoint
          is deleted at the end with its sha256 logged. (Receipt RUN 1, 2026-09-26 16:15Z, predates lock T2 and used
          the round-start LR point, so its round 0 ran at LR 0; this code states exposure_point="end" explicitly.)
  summarize  merges the per-check JSONs into GPU_DETERMINISM_RECEIPT.json with one bitwise verdict per check.

Synthetic data only; nothing here reads a dataset row. `--device cpu` / `--devices cpu,cpu` / `--K` give the local
dry run used by tests_fl/test_gpu_receipt_plumbing.py.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

from .client import ClientData, LocalSolver, PFConfig
from .local_only import LORecipe, LORunner
from .numerics import enforce_strict_fp32, fingerprint, numerics_receipt, state_digest
from .personal import PersonalStore
from .server import RoundAborted, Server
from .synthetic import census_like_sizes

K_FULL = 158_486
SEED = 2026
PEAK = 1e-3                                                 # executed peak LR (G0(b) C8), both families
THREADS = 4
EXPOSURE_POINT = "end"                                      # PREG1_LOCK_VALUES.json T2 (explicit in every config)


def _solver(passes: int = 2) -> LocalSolver:
    return LocalSolver(lr=PEAK, passes=passes, batch_size=16, clip=1.0, weight_decay=1e-5,
                       fused=True if torch.cuda.is_available() else None)


def _setup(strict: bool) -> dict:
    torch.set_num_threads(THREADS)
    if torch.cuda.is_available() and strict and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG=:4096:8 must be exported before the process starts")
    return enforce_strict_fp32(warn_only=not strict)


def _is_determinism_error(e: BaseException) -> bool:
    t = str(e)
    return "deterministic" in t or "CUBLAS_WORKSPACE_CONFIG" in t


def worker_adapter(family: str, K: int, device: str = "cpu", seed: int = SEED + 1):
    from .families import build_v2
    return build_v2(family, seed=seed, K=K, device=device).adapter


def _clients(family: str, K: int, sizes, seed: int, prefix: str) -> list:
    from .families import synthetic_examples_v2
    g = torch.Generator().manual_seed(seed)
    out = []
    for i, n in enumerate(sizes):
        ex = synthetic_examples_v2(int(max(n, 1)), K, g)
        if n == 0:
            ex["target_class"][:] = -1                    # a no-label client (recorded no-op)
        out.append(ClientData(f"{prefix}{i:05d}", ex))
    return out


def _with_fallback(fn, *a, **kw) -> dict:
    """Run fn(strict=True); on a determinism error record it verbatim and re-run with warn_only."""
    try:
        return {"mode": "strict", **fn(*a, strict=True, **kw)}
    except (RuntimeError, RoundAborted) as e:
        if not _is_determinism_error(e):
            raise
        err = f"{type(e).__name__}: {e}"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = fn(*a, strict=False, **kw)
    res["warnings_warn_only"] = sorted({f"{w.category.__name__}: {w.message}" for w in caught} |
                                       set(res.get("warnings", [])))
    return {"mode": "warn_only_fallback", "strict_error_verbatim": err, **res}


# ------------------------------------------------------------------------------------------------ init
def check_init(K: int, device: str) -> dict:
    from .families import build_v2
    num = _setup(strict=True)
    inv_dir = Path(__file__).resolve().parents[1] / "v4models" / "inventory"
    out = {"numerics": num, "families": {}}
    for fam in ("GRU", "SASREC"):
        inv = None
        p = inv_dir / f"{fam}_CLEAN_V2_MANIFEST_INVENTORY.json"
        if p.exists() and K == K_FULL:
            inv = json.loads(p.read_text())["init_state"]
        rec = {}
        for seed in (2026, 2027):
            bm = build_v2(fam, seed=seed, K=K, device=device)
            on_dev = state_digest(bm.adapter.broadcast_state())          # device -> host copy, then sha256
            rec[str(seed)] = {"init_sha256": bm.init_sha256, "raw_init_sha256": bm.raw_init_sha256,
                              "after_device_round_trip": on_dev, "round_trip_bitwise": on_dev == bm.init_sha256,
                              "inventory_init_sha256": inv[str(seed)]["init_sha256"] if inv else None,
                              "equals_inventory": (inv[str(seed)]["init_sha256"] == bm.init_sha256) if inv else None,
                              "shared_numel": bm.manifest.shared_numel}
            del bm
        out["families"][fam] = rec
    return out


# ------------------------------------------------------------------------------------------------ t34
def _t34_once(family: str, K: int, devices: list, strict: bool, n_rounds: int) -> dict:
    from .families import build_v2
    from .pool import ProcessPool, WorkerSpec
    num = _setup(strict)
    spec = WorkerSpec("fl.gpu_receipt:worker_adapter", (("family", family), ("K", K)), threads=THREADS,
                      strict_deterministic=strict)
    clients = _clients(family, K, [3, 5, 16, 20, 1, 7, 33, 0], seed=91, prefix="t34-user-")
    bm = build_v2(family, seed=SEED, K=K, device="cpu")
    theta0 = bm.theta0(clone=True)
    methods = {"FP": dict(mu=0.01, pf=None), "PF": dict(mu=0.0, pf=PFConfig(lr=PEAK, lam=1e-4))}
    out = {"numerics_main": num, "devices": devices, "n_rounds": n_rounds, "init_sha256": bm.init_sha256,
           "clients": [int(c.examples["target_class"].ge(0).sum()) for c in clients], "methods": {}}
    t0 = time.perf_counter()
    with ProcessPool(spec, devices=devices) as pool:
        for mname, kw in methods.items():
            runs = []
            for rotation in (0, 0, 1):
                bm.adapter.load_state_(theta0)
                srv = Server(bm.adapter, init_sha256=bm.init_sha256)
                store = PersonalStore(bm.adapter.query_dim)
                rounds = []
                for r in range(n_rounds):
                    rep = pool.run_round(srv, clients, _solver(), round_idx=r, seed=SEED, store=store, n_shards=2,
                                         rotation=rotation, **kw)
                    rounds.append({"server_sha256": srv.digest(), "store_digest": store.digest(),
                                   "weight": rep.weight, "shard_devices": pool.job_devices[-1]})
                runs.append({"rotation": rotation, "rounds": rounds})
            key = lambda run: [(x["server_sha256"], x["store_digest"]) for x in run["rounds"]]  # noqa: E731
            out["methods"][mname] = {"runs": runs, "repeat_bitwise": key(runs[0]) == key(runs[1]),
                                     "swap_bitwise": key(runs[0]) == key(runs[2])}
        out["worker_pids"] = sorted(pool.pids_seen)
        out["warnings"] = sorted(pool.warnings)
    out["wall_s"] = time.perf_counter() - t0
    return out


def check_t34(family: str, K: int, devices: list, n_rounds: int = 2) -> dict:
    return _with_fallback(_t34_once, family, K, devices, n_rounds=n_rounds)


# ------------------------------------------------------------------------------------------------ lo
def _lo_hook(q: dict):
    def hook(adapter, budget: int) -> dict:
        from .client import select_rows
        qq = select_rows(q, torch.arange(q["target_class"].shape[0]), adapter.device)
        logits = adapter.scores(qq)
        tgt = qq["target_class"]
        rank = (logits > logits.gather(1, tgt[:, None])).sum(1) + 1
        mrr = torch.where(rank <= 20, 1.0 / rank.to(torch.float64), torch.zeros_like(rank, dtype=torch.float64))
        ce = torch.nn.functional.cross_entropy(logits, tgt, reduction="sum")
        return {"mrr20": float(mrr.mean()), "ce_sum": float(ce)}
    return hook


def _lo_once(family: str, K: int, device: str, order: str, n: int, strict: bool) -> dict:
    from .families import build_v2
    num = _setup(strict)
    if device.startswith("cuda"):
        from .pool import require_visible
        require_visible(device)
        torch.cuda.set_device(torch.device(device))
    rng = np.random.default_rng(SEED + 1)
    sizes = census_like_sizes(n, rng)
    sizes[15::16] = 0                                                  # every 16th client has no label
    clients = _clients(family, K, sizes, seed=SEED + 2, prefix="lo-user-")
    queries = _clients(family, K, [4] * n, seed=SEED + 3, prefix="lo-query-")
    bm = build_v2(family, seed=SEED, K=K, device=device)
    recipe = LORecipe(theta0_digest=bm.init_sha256, seed=SEED, passes=6, eval_at=(2, 4, 6), solver=_solver(6),
                      lr_shape="s_efe", exposure_point=EXPOSURE_POINT)
    runner = LORunner(bm.adapter, bm.theta0(clone=True), recipe, digest_fn=fingerprint)
    idx = list(range(n)) if order == "forward" else list(reversed(range(n)))
    per, t0, steps = {}, time.perf_counter(), 0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for i in idx:
            r = runner.run(clients[i], eval_hook=_lo_hook(queries[i].examples))
            steps += r.steps
            rec = {"fingerprint": r.final_digest, "n_valid": r.n_valid, "steps": r.steps, "no_label": r.no_label,
                   "evals": {str(k): v for k, v in sorted(r.evals.items())}}
            if i % 64 == 0:
                rec["sha256"] = state_digest(bm.adapter.extract_shared())
            per[clients[i].key] = rec
    return {"numerics": num, "family": family, "device": device, "order": order, "n_clients": n,
            "init_sha256": bm.init_sha256, "recipe_digest": recipe.digest(), "steps_total": steps,
            "n_no_label": int(sum(1 for v in per.values() if v["no_label"])), "wall_s": time.perf_counter() - t0,
            "warnings": sorted({f"{w.category.__name__}: {w.message}" for w in caught}),
            "gpu_name": torch.cuda.get_device_name(torch.device(device)) if device.startswith("cuda") else "cpu",
            "per_client": per}


def check_lo(family: str, K: int, device: str, order: str, n: int) -> dict:
    return _with_fallback(_lo_once, family, K, device, order, n)


# ------------------------------------------------------------------------------------------------ resume
def _resume_once(family: str, K: int, devices: list, phase: str, ckpt_root: str, strict: bool) -> dict:
    from .checkpoint import CheckpointManager, sha256_file
    from .families import build_v2
    from .participation import ParticipationPlan
    from .pool import ProcessPool, WorkerSpec
    from .runtime import FLRun, RunConfig
    num = _setup(strict)
    spec = WorkerSpec("fl.gpu_receipt:worker_adapter", (("family", family), ("K", K)), threads=THREADS,
                      strict_deterministic=strict)
    clients = _clients(family, K, [3, 9, 16, 2, 20, 5, 1, 12], seed=77, prefix="rs-user-")
    by_key = {c.key: c for c in clients}
    plan = ParticipationPlan.build(list(by_key), seed=SEED, manifest_hash="fl02-gpu-receipt-synthetic", group_size=3)
    n_dec = sum(int(c.examples["target_class"].ge(0).sum()) for c in clients)
    # phase "first" checkpoints once, after its 2 rounds; "second" never re-writes (one temporary checkpoint in total)
    cfg = RunConfig(f"resume-{family}", "FA", SEED, PEAK, _solver(), n_shards=2,
                    ckpt_every=2 if phase == "first" else 10 ** 9, exposure_point=EXPOSURE_POINT)
    bm = build_v2(family, seed=SEED, K=K, device="cpu")
    srv = Server(bm.adapter, init_sha256=bm.init_sha256)
    ckpt = CheckpointManager(Path(ckpt_root), cfg.run_id) if phase != "full" else None
    with ProcessPool(spec, devices=devices) as pool:
        if phase == "full":
            run = FLRun(cfg, plan, srv, by_key.get, n_dec, pool=pool).run(max_rounds=3)
        elif phase == "first":
            run = FLRun(cfg, plan, srv, by_key.get, n_dec, pool=pool, ckpt=ckpt).run(max_rounds=2)
        else:
            run = FLRun.resume(cfg, plan, srv, by_key.get, n_dec, pool=pool, ckpt=ckpt)
            start = run.state.cursor
            run.run(max_rounds=3 - start)
    out = {"numerics": num, "phase": phase, "rounds": run.state.cursor, "server_sha256": srv.digest(),
           "counters": run.counters.summary(), "lr_log": [list(x) for x in run.lr_log]}
    if phase == "second":
        out["resumed_from_cursor"] = start
        latest = ckpt.dir / ckpt.LATEST
        out["checkpoint_sha256"] = sha256_file(latest) if latest.exists() else None
        out["checkpoint_bytes"] = latest.stat().st_size if latest.exists() else 0
        ckpt.complete()                                                   # deletes latest.pt, sha256-logged
        out["files_after_cleanup"] = ckpt.files()
        out["removal_log"] = [json.loads(x) for x in (ckpt.removals.read_text().splitlines()
                                                      if ckpt.removals.exists() else [])]
    return out


def check_resume(family: str, K: int, devices: list, phase: str, ckpt_root: str) -> dict:
    return _with_fallback(_resume_once, family, K, devices, phase, ckpt_root)


# ------------------------------------------------------------------------------------------------ summarize
def summarize(d: Path) -> dict:
    j = lambda name: json.loads((d / name).read_text()) if (d / name).exists() else None  # noqa: E731
    out = {"what": "FL-02 GPU determinism receipt (Clean V2 families, full K, synthetic inputs)",
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "checks": {}}
    init = j("init.json")
    if init:
        out["init_hashes"] = init["families"]
        out["checks"]["init_hash_round_trip"] = all(v["round_trip_bitwise"] for f in init["families"].values()
                                                    for v in f.values())
        eq = [v["equals_inventory"] for f in init["families"].values() for v in f.values()]
        if all(x is not None for x in eq):          # a cross-host comparison, not a determinism verdict
            out["comparisons"] = {"init_hash_equals_local_inventory": all(eq),
                                  "note": "server-measured registered theta0 hashes vs MODEL-01's local inventory "
                                          "(torch builds differ: MODEL-01 risk 4); the lock records the server hashes"}
    for fam in ("GRU", "SASREC"):
        t = j(f"t34_{fam}.json")
        if t:
            for m, v in t["methods"].items():
                out["checks"][f"t34_{fam}_{m}_repeat_bitwise"] = v["repeat_bitwise"]
                out["checks"][f"t34_{fam}_{m}_gpu_swap_bitwise"] = v["swap_bitwise"]
            out.setdefault("modes", {})[f"t34_{fam}"] = t["mode"]
        a, b = j(f"lo_{fam}_A.json"), j(f"lo_{fam}_B.json")
        if a and b:
            mism = [k for k in a["per_client"] if a["per_client"][k] != b["per_client"].get(k)]
            out["checks"][f"lo_{fam}_1024_bitwise_A_vs_B"] = (not mism and len(a["per_client"]) == len(b["per_client"])
                                                             == a["n_clients"])
            out.setdefault("lo", {})[fam] = {"A": {k: a[k] for k in ("device", "order", "gpu_name", "wall_s",
                                                                      "steps_total", "n_no_label", "mode")},
                                             "B": {k: b[k] for k in ("device", "order", "gpu_name", "wall_s",
                                                                      "steps_total", "n_no_label", "mode")},
                                             "mismatched_clients": len(mism),
                                             "warnings": sorted(set(a.get("warnings", [])) | set(b.get("warnings", [])))}
            out.setdefault("modes", {})[f"lo_{fam}"] = [a["mode"], b["mode"]]
    full, second = j("resume_full.json"), j("resume_second.json")
    if full and second:
        out["checks"]["resume_GRU_bitwise"] = (full["server_sha256"] == second["server_sha256"]
                                               and full["counters"] == second["counters"]
                                               and full["lr_log"] == second["lr_log"])
        out["resume"] = {"checkpoint_sha256_deleted": second.get("checkpoint_sha256"),
                         "checkpoint_bytes": second.get("checkpoint_bytes"),
                         "removal_log": second.get("removal_log"), "files_after_cleanup": second.get("files_after_cleanup")}
    errors = {}
    for p in sorted(d.glob("*.json")):
        try:
            x = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            continue
        if isinstance(x, dict) and x.get("strict_error_verbatim"):
            errors[p.name] = x["strict_error_verbatim"]
        if isinstance(x, dict) and x.get("warnings_warn_only"):
            errors[p.name + ":warnings"] = x["warnings_warn_only"]
    out["nondeterministic_ops_verbatim"] = errors
    out["all_bitwise"] = bool(out["checks"]) and all(bool(v) for v in out["checks"].values())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("init", "t34", "lo", "resume"):
        a = sub.add_parser(name)
        a.add_argument("--K", type=int, default=K_FULL)
        a.add_argument("--out", required=True)
        a.add_argument("--family", default="GRU")
        a.add_argument("--device", default="cuda:0")
        a.add_argument("--devices", default="cuda:0,cuda:1")
        a.add_argument("--order", default="forward")
        a.add_argument("--n", type=int, default=1024)
        a.add_argument("--phase", default="full")
        a.add_argument("--ckpt-root", default="")
    a = sub.add_parser("summarize")
    a.add_argument("--dir", required=True)
    a.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "summarize":
        res = summarize(Path(args.dir))
    elif args.cmd == "init":
        res = check_init(args.K, args.device)
    elif args.cmd == "t34":
        res = check_t34(args.family.upper(), args.K, args.devices.split(","))
    elif args.cmd == "lo":
        res = check_lo(args.family.upper(), args.K, args.device, args.order, args.n)
    else:
        res = check_resume(args.family.upper(), args.K, args.devices.split(","), args.phase, args.ckpt_root)
    res.setdefault("env", {"CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                           "torch": torch.__version__, "cuda": torch.version.cuda,
                           "python": sys.version.split()[0]})
    tmp = Path(args.out).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(res, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, args.out)
    print(json.dumps({k: res[k] for k in res if k in ("mode", "checks", "all_bitwise", "wall_s")}, default=str)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
