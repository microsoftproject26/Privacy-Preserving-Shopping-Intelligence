"""The benchmark command line (CPU by default; `--device cuda --gpu N` selects the deterministic GPU class).

  python -m ppsi.benchrun.run describe --dataset s3_beauty --data-root DATA
  python -m ppsi.benchrun.run run --dataset s3_beauty --data-root DATA --runs-root RUNS --arm C_FULL --seed 2026
         [--recipe recipe.json] [--pre-run PRE_RUN_DIR] [--base-run FL_BASE_RUN_DIR] [--s-record S_RECORD.json]
         [--smoke] [--smoke-rounds 50] [--smoke-epochs 1.0] [--resume] [--threads 16] [--workers 1] [--max-len 50]
         [--dropout 0.1] [--budget-efe 6|24|96] [--wall-cap-hours H] [--device cpu|cuda] [--gpu ID]
  python -m ppsi.benchrun.run test --dataset s3_beauty --data-root DATA --runs-root RUNS --runs C_FULL_s2026 ...
         [--recipe recipe.json] [--workers 4] [--threads 8] [--device cpu|cuda] [--gpu ID]

DATA holds <dataset>/leave_one_out/{train.csv, holdout.csv} (the layout ppsi.benchmarks writes). With --device cuda,
CUDA_VISIBLE_DEVICES = --gpu and CUBLAS_WORKSPACE_CONFIG = :4096:8 are exported before torch is imported, then
accel.py enforces deterministic strict-FP32 numerics. Without it the run is CPU only (CUDA_VISIBLE_DEVICES is blanked
before torch is imported).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _parse(argv=None):
    ap = argparse.ArgumentParser(prog="ppsi.benchrun.run")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("describe", "run", "test"):
        p = sub.add_parser(name)
        p.add_argument("--dataset", required=True)
        p.add_argument("--data-root", required=True)
        p.add_argument("--threads", type=int, default=16)
        if name in ("run", "test"):
            p.add_argument("--runs-root", required=True)
            p.add_argument("--recipe", default=None, help="recipe JSON merged over recipe.DEFAULT_RECIPE")
            p.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
            p.add_argument("--gpu", type=int, default=None, help="--device cuda: the physical GPU id")
            p.add_argument("--workers", type=int, default=1)
        if name == "run":
            p.add_argument("--arm", required=True)
            p.add_argument("--seed", type=int, required=True)
            p.add_argument("--pre-run", default=None)
            p.add_argument("--base-run", default=None)
            p.add_argument("--smoke", action="store_true")
            p.add_argument("--smoke-rounds", type=int, default=50)
            p.add_argument("--smoke-epochs", type=float, default=1.0)
            p.add_argument("--limit-users", type=int, default=None, help="device-arm smoke only")
            p.add_argument("--resume", action="store_true")
            p.add_argument("--max-len", type=int, default=50)
            p.add_argument("--dropout", type=float, default=None)
            p.add_argument("--budget-efe", type=int, default=6, help="central arms only: 6 | 24 | 96")
            p.add_argument("--wall-cap-hours", type=float, default=None,
                           help="a run that hits the cap is INCOMPLETE and ineligible")
            p.add_argument("--s-record", default=None, help="FA_1024 / DP8: the S record of the matching S_CAL run")
        if name == "test":
            p.add_argument("--runs", nargs="+", required=True, help="run directory names under RUNS/<dataset>/")
    return ap.parse_args(argv)


def _threads(n: int) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    os.environ.setdefault("MKL_NUM_THREADS", str(n))
    import torch
    torch.set_num_threads(int(n))


def main(argv=None) -> int:
    a = _parse(argv)
    from .common import BenchRefused
    try:
        if getattr(a, "device", "cpu") == "cuda":            # before torch is imported (CUBLAS / visibility)
            from .accel import prepare_cuda_env
            prepare_cuda_env(a.gpu)
        else:
            if getattr(a, "gpu", None) is not None:
                raise BenchRefused("--gpu needs --device cuda")
            if "torch" not in sys.modules:
                os.environ["CUDA_VISIBLE_DEVICES"] = ""      # CPU only, never a GPU
        _threads(a.threads)
        from . import common
        from . import data as D
        common.check_dataset(a.dataset)
        root = Path(a.data_root)
        if a.cmd == "describe":
            print(json.dumps(D.load(a.dataset, root).describe(), indent=1, sort_keys=True, default=str))
            return 0
        from . import recipe
        rcp = recipe.load_recipe(a.recipe)
        runs_root = Path(a.runs_root)
        if a.cmd == "test":
            from .holdout import evaluate_holdout
            if a.workers * a.threads > 96:
                raise BenchRefused("at most 96 CPU threads in total (workers x threads)")
            d = D.load(a.dataset, root)
            out = evaluate_holdout(runs_root, a.dataset, d, root, a.runs, rcp=rcp, workers=a.workers,
                                   threads=a.threads, **({"device": a.device} if a.device == "cuda" else {}))
            print(json.dumps({"dataset": out["dataset"], "n_test_users": out["n_test_users"],
                              "runs": sorted(out["runs"])}, indent=1))
            return 0
        return _run(a, root, runs_root, rcp)
    except BenchRefused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2


def _run(a, root, runs_root, rcp) -> int:
    import numpy as np

    from . import central, common, device, fl_engine, recipe
    from . import data as D
    from .common import BenchRefused
    arm = a.arm
    if arm not in recipe.ALL_ARMS:
        raise BenchRefused(f"unknown arm {arm!r}: {recipe.ALL_ARMS}")
    if a.budget_efe != 6 and arm not in recipe.CENTRAL_ARMS:
        raise BenchRefused("the exposure-budget option applies to C_FULL and PRE only")
    if a.budget_efe not in (6, 24, 96):
        raise BenchRefused("budget must be 6, 24 or 96 EFE")
    label = arm
    if a.max_len != 50 or a.dropout is not None or a.budget_efe != 6:
        label = f"{arm}__B{a.budget_efe}_L{a.max_len}_D{a.dropout if a.dropout is not None else 'default'}"
    run_dir = common.run_dir_for(runs_root, a.dataset, label, a.seed, smoke=a.smoke)
    run_id = f"BENCH_{a.dataset}_{label}_s{a.seed}" + ("_SMOKE" if a.smoke else "")
    d = D.load(a.dataset, root, max_len=a.max_len)
    print(f"[bench] {run_id}: K={d.K} U={d.U} N={d.N} threads={a.threads} -> {run_dir}"
          + (f" device=cuda gpu={a.gpu}" if a.device == "cuda" else ""), flush=True)
    provenance = {"recipe_sha256": rcp["recipe_sha256"]}
    if arm in recipe.CENTRAL_ARMS:
        rc = recipe.central_recipe(arm, rcp)
        drop = (a.dropout,) if a.dropout is not None else tuple(rc["dropout"])
        if arm == "C_FULL":
            rows, sel = np.arange(d.N, dtype="int64"), None
        else:
            rows, sel = d.band_rows(), np.flatnonzero(d.band)
        out = central.run_central(data=d, arm=arm, variant=rc["model_variant"], peak_lr=rc["peak_lr"], dropout=drop,
                                  seed=a.seed, rows=rows, select_users=sel, run_dir=run_dir, run_id=run_id,
                                  smoke_epochs=a.smoke_epochs if a.smoke else None, resume=a.resume,
                                  max_len=a.max_len if a.max_len != 50 else None, budget_efe=a.budget_efe,
                                  early_stop_marks=6 if (arm == "C_FULL" and a.budget_efe != 6) else None,
                                  wall_cap_hours=a.wall_cap_hours, provenance=provenance, device=a.device)
    elif arm in recipe.FL_ARMS:
        out = fl_engine.run_fl(data=d, arm=arm, seed=a.seed, run_dir=run_dir, run_id=run_id, rcp=rcp,
                               pre_run_dir=Path(a.pre_run) if a.pre_run else None,
                               smoke_rounds=a.smoke_rounds if a.smoke else None, resume=a.resume,
                               dropout=(a.dropout,) if a.dropout is not None else None,
                               max_len=a.max_len if a.max_len != 50 else None,
                               s_record=Path(a.s_record) if a.s_record else None,
                               dp_dir=runs_root / a.dataset / "dp", wall_cap_hours=a.wall_cap_hours,
                               provenance=provenance, device=a.device)
    else:
        if not a.base_run:
            raise BenchRefused(f"{arm}: --base-run (the finished federated base run directory) is required")
        out = device.run_ft(data=d, data_root=root, arm=arm, seed=a.seed, base_run_dir=Path(a.base_run),
                            run_dir=run_dir, run_id=run_id, workers=a.workers, threads=a.threads, rcp=rcp,
                            limit_users=a.limit_users if a.smoke else None, device=a.device)
        out = {"status": out["status"], "timing": out["timing"]}
    print(f"[bench] {run_id}: {out['status']} timing={out.get('timing')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
