"""v4bench.run — the benchmark CLI (CPU by default; BM-03 adds the opt-in deterministic GPU class to `run`).

  python -B -m v4bench.run run   --dataset s3_beauty --arm C_FULL|PRE|FA|FA_WARM|FA_WARM_FROZEN|FP*|FA_Q8*|FT_* --seed 2026
                                 [--pre-run <PRE run dir>] [--base-run <FL base run dir>] [--smoke] [--smoke-rounds 50]
                                 [--resume] [--threads 16] [--workers 1] [--max-len 50] [--dropout 0.1]
                                 [--device cpu|cuda] [--gpu <physical id>]      (BM-03; cuda = NVIDIA_A40-FP32-DET)
  python -B -m v4bench.run describe --dataset s3_beauty          (counts, band coverage; reads train.csv only)
  python -B -m v4bench.run lock  --dataset s3_beauty --runs C_FULL_s2026 PRE_s2026 ...
  python -B -m v4bench.run final --dataset s3_beauty [--workers 4 --threads 8]   (the ONE TEST look; needs LOCK.json)

Run it with CUDA_VISIBLE_DEVICES="" under nice -n 15; the module sets both defensively before importing torch.
--device cuda --gpu N (run only): CUDA_VISIBLE_DEVICES=N and CUBLAS_WORKSPACE_CONFIG=:4096:8 are exported before torch is
imported, then v4bench.accel enforces the deterministic strict-FP32 numerics. --device cpu (the default) is the registered
CPU-FP32 class, unchanged. lock / final / trackb-select / decide-warm / seed3 refuse to mix device classes in one dataset.
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""                      # CPU only, never a GPU (set before any torch import)


def _parse(argv=None):
    ap = argparse.ArgumentParser(prog="v4bench.run")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "describe", "lock", "final", "manifest", "trackb-select", "decide-warm", "seed3"):
        p = sub.add_parser(name)
        p.add_argument("--dataset", required=True)
        p.add_argument("--data-root", default=None)
        p.add_argument("--runs-root", default=None)
        p.add_argument("--threads", type=int, default=16)
        if name == "run":
            p.add_argument("--arm", required=True)
            p.add_argument("--seed", type=int, required=True)
            p.add_argument("--pre-run", default=None)
            p.add_argument("--base-run", default=None)
            p.add_argument("--smoke", action="store_true")
            p.add_argument("--smoke-rounds", type=int, default=50)
            p.add_argument("--smoke-epochs", type=float, default=1.0)
            p.add_argument("--limit-users", type=int, default=None, help="FT smoke only")
            p.add_argument("--resume", action="store_true")
            p.add_argument("--workers", type=int, default=1)
            p.add_argument("--max-len", type=int, default=50)
            p.add_argument("--dropout", type=float, default=None)
            p.add_argument("--budget-efe", type=int, default=6, help="central arms only: 6 (Track A) | 24 | 96 (B8 O1)")
            p.add_argument("--wall-cap-hours", type=float, default=None,
                           help="a run that hits the cap is INCOMPLETE and ineligible (B10 Q4)")
            p.add_argument("--s-record", default=None, help="FA_1024 / DP8: the S_RECORD of the matching S_CAL run")
            p.add_argument("--device", default="cpu", choices=("cpu", "cuda"),
                           help="BM-03: cpu = CPU-FP32 (registered default); cuda = NVIDIA_A40-FP32-DET (deterministic)")
            p.add_argument("--gpu", type=int, default=None, help="--device cuda: the physical GPU id (CUDA_VISIBLE_DEVICES)")
        if name == "lock":
            p.add_argument("--runs", nargs="+", required=True)
            p.add_argument("--not-strict", action="store_true", help="tests only")
        if name == "trackb-select":
            p.add_argument("--arm", required=True)
            p.add_argument("--runs", nargs="+", required=True)
        if name == "decide-warm":
            p.add_argument("--s7-runs", nargs=2, required=True)
            p.add_argument("--s8-runs", nargs=2, required=True)
        if name == "seed3":
            p.add_argument("--cfull-runs", nargs=2, required=True)
            p.add_argument("--warm-runs", nargs=2, required=True)
        if name == "final":
            p.add_argument("--workers", type=int, default=1)
    return ap.parse_args(argv)


THREADS_FIXED = 16                                           # R9: the reduction order depends on the thread count


def _threads(n: int) -> None:
    if int(n) != THREADS_FIXED and os.environ.get("BM_ALLOW_THREADS") != "1":
        from .common import BenchRefused
        raise BenchRefused(f"CPU-FP32 device class: threads are fixed at {THREADS_FIXED} (amendment 07 R9), not {n}")
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    os.environ.setdefault("MKL_NUM_THREADS", str(n))
    import torch
    torch.set_num_threads(int(n))
    try:
        os.nice(15) if os.nice(0) < 15 else None
    except (AttributeError, OSError):
        pass


def main(argv=None) -> int:
    a = _parse(argv)
    from .common import BenchRefused
    try:
        if getattr(a, "device", "cpu") == "cuda":            # BM-03: before torch is imported (CUBLAS / visibility)
            from .accel import prepare_cuda_env
            prepare_cuda_env(a.gpu)
        elif getattr(a, "gpu", None) is not None:
            raise BenchRefused("--gpu needs --device cuda")
        _threads(a.threads)
    except BenchRefused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    from . import common, data as D, final, recipe, selection
    runs_root = common.Path(a.runs_root) if a.runs_root else common.RUNS_ROOT_DEFAULT
    try:
        common.check_dataset(a.dataset)
    except BenchRefused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    root = common.Path(a.data_root) if a.data_root else common.default_data_root(a.dataset)
    try:
        if a.cmd == "describe":
            d = D.load(a.dataset, root)
            print(common.json.dumps(d.describe(), indent=1, sort_keys=True, default=str))
            return 0
        if a.cmd == "lock":
            d = D.load(a.dataset, root)
            lock = final.make_lock(runs_root, a.dataset, a.runs, data_fingerprint=d.fingerprint(),
                                   strict=not a.not_strict)
            print(f"LOCK written: {final.lock_path(runs_root, a.dataset)} ({len(lock['runs'])} runs)")
            return 0
        if a.cmd == "final":
            final.verify_lock(runs_root, a.dataset)            # refuses before the data are even loaded
            d = D.load(a.dataset, root)
            if a.workers * a.threads > 96:
                raise BenchRefused("at most 96 CPU threads in total (workers x threads)")
            out = final.final_eval(runs_root, a.dataset, d, root, workers=a.workers, threads=a.threads)
            first = next(iter(out["runs"].values()))["PRACTICAL_BEST"]
            print(f"TEST done: {len(out['runs'])} runs, {out['n_test_users']} holdout users; populations "
                  f"ALL {first['ALL']['n_users']} / BAND {first['BAND']['n_users']} / OTHERS {first['OTHERS']['n_users']}")
            return 0
        if a.cmd == "manifest":
            return _manifest(a, root, runs_root)
        if a.cmd == "trackb-select":
            print(common.json.dumps(selection.trackb_select(runs_root, a.dataset, a.arm, a.runs), indent=1, default=str))
            return 0
        if a.cmd == "decide-warm":
            print(common.json.dumps(selection.decide_warm(runs_root, a.dataset, a.s7_runs, a.s8_runs), indent=1))
            return 0
        if a.cmd == "seed3":
            dec = common.read_json(runs_root / a.dataset / "FA_WARM_DECISION.json")
            print(common.json.dumps(selection.seed3_check(runs_root, a.dataset, {"C_FULL": a.cfull_runs,
                                                                                  dec["FA_WARM_STAR"]: a.warm_runs}), indent=1))
            return 0
        return _run(a, root, runs_root)
    except BenchRefused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2


def _manifest(a, root, runs_root) -> int:
    """The per-dataset manifest (R1 / R2 / R9): counts, T, m, band coverage and the eSASRec validation-row equality."""
    import numpy as np
    import pandas as pd
    from . import common, data as D, recipe, selection
    d = D.load(a.dataset, root)
    chk = D.statistics_check(a.dataset, root, d)
    fr = pd.read_csv(D.split_dir(root, a.dataset) / "train.csv", parse_dates=["datetime"])
    users = fr["user_id"].unique()                    # VERBATIM trainer.py:217-229 (get_val_mask_func_all)
    rank = fr.sort_values("datetime", ascending=False, kind="stable").groupby("user_id", sort=False).cumcount()
    mask = ((fr["user_id"].isin(users)) & (rank == 0)).values
    want = fr[mask].set_index("user_id")["item_id"].sort_index()
    got = pd.Series(d.catalog[d.val_item], index=d.user_ids).sort_index()
    equal = bool(len(want) == d.U and (want.to_numpy() == got.to_numpy()).all())
    ho = D.count_csv_rows(D.split_dir(root, a.dataset) / "holdout.csv")
    ncl = int((np.diff(d.user_dec_offsets) >= 1).sum())
    obj = {"format": "BM02_MANIFEST_1", "dataset": a.dataset, "utc": common.utc(), "describe": d.describe(),
           "statistics_check": chk, "esasrec_val_rows_equal_ours": equal,
           "n_val_row_not_last": d.stats["n_val_row_not_last"], "holdout_users": ho,
           "inner_validation_users": int(len(d.validation_users())), "fl_clients": ncl,
           "T_rounds_P1": recipe.fa_rounds(ncl, 1),
           "dp": {"q": 2.0 ** -7, "T": recipe.DP["T"], "z": recipe.DP["z"], "delta": recipe.DP["delta"],
                  "m": recipe.dp_cohort(ncl)},
           "pins": recipe.check_pins(required=False), "code_tree_digest": selection.tree_digest()}
    common.write_exclusive(runs_root / a.dataset / "MANIFEST.json", obj)
    print(common.json.dumps({k: obj[k] for k in ("dataset", "esasrec_val_rows_equal_ours", "holdout_users",
                                                  "fl_clients", "T_rounds_P1", "dp")}, default=str))
    return 0 if equal and chk["ok"] else 1


def _run(a, root, runs_root) -> int:
    from . import central, common, data as D, device, fl_engine, recipe
    from .common import BenchRefused
    arm = recipe.ALIASES.get(a.arm, a.arm)
    if arm in recipe.NOT_IMPLEMENTED:
        raise BenchRefused(f"{arm}: not implemented in this BM-02 build (see the handoff)")
    if arm not in recipe.ALL_ARMS:
        raise BenchRefused(f"unknown arm {arm!r}: {recipe.ALL_ARMS}")
    if not a.smoke:
        from . import selection
        selection.check_dp_variant(runs_root, a.dataset, arm)
    default_drop = a.dropout is None
    if a.budget_efe != 6 and arm not in recipe.CENTRAL_ARMS:
        raise BenchRefused("the exposure-budget axis applies to C_FULL and PRE only (B8 O1); the FL arms keep the R1 rule")
    if a.budget_efe not in (6, 24, 96):
        raise BenchRefused("budget must be 6, 24 or 96 EFE")
    if arm == "C_FULL" and a.dataset == "ml_1m" and a.budget_efe == 96:       # B10 Q4
        raise BenchRefused("the ML-1M C_FULL budget axis is {6, 24} EFE only (B10 Q4)")
    provenance = None
    if not a.smoke:                                         # M6: the mirrored constants are proven against the pinned files
        provenance = recipe.check_mirrors()
    label = arm
    if a.max_len != 50 or not default_drop or a.budget_efe != 6:
        label = f"{arm}__B{a.budget_efe}_L{a.max_len}_D{a.dropout if a.dropout is not None else 0.1}"
    run_dir = common.run_dir_for(runs_root, a.dataset, label, a.seed, smoke=a.smoke)
    run_id = f"BM_{a.dataset}_{label}_s{a.seed}" + ("_SMOKE" if a.smoke else "")
    d = D.load(a.dataset, root, max_len=a.max_len)
    print(f"[bm] {run_id}: K={d.K} U={d.U} N={d.N} threads={a.threads} -> {run_dir}"
          + (f" device=cuda gpu={a.gpu}" if a.device == "cuda" else ""), flush=True)
    if arm in recipe.CENTRAL_ARMS:
        rcp = recipe.central_recipe(arm)
        drop = (a.dropout,) if a.dropout is not None else tuple(rcp["dropout"])
        if arm == "C_FULL":
            rows, sel = __import__("numpy").arange(d.N, dtype="int64"), None
        else:
            rows, sel = d.band_rows(), __import__("numpy").flatnonzero(d.band)
        out = central.run_central(data=d, arm=arm, variant=rcp["model_variant"], peak_lr=rcp["peak_lr"], dropout=drop,
                                  seed=a.seed, rows=rows, select_users=sel, run_dir=run_dir, run_id=run_id,
                                  smoke_epochs=a.smoke_epochs if a.smoke else None, resume=a.resume,
                                  max_len=a.max_len if a.max_len != 50 else None, budget_efe=a.budget_efe,
                                  early_stop_marks=6 if (arm == "C_FULL" and a.budget_efe != 6) else None,
                                  wall_cap_hours=a.wall_cap_hours, provenance=provenance, device=a.device)
    elif arm in recipe.FL_ARMS:
        out = fl_engine.run_fl(data=d, arm=arm, seed=a.seed, run_dir=run_dir, run_id=run_id,
                               pre_run_dir=common.Path(a.pre_run) if a.pre_run else None,
                               smoke_rounds=a.smoke_rounds if a.smoke else None, resume=a.resume,
                               dropout=(a.dropout,) if a.dropout is not None else None,
                               max_len=a.max_len if a.max_len != 50 else None,
                               s_record=common.Path(a.s_record) if a.s_record else None,
                               dp_dir=runs_root / a.dataset / "dp", wall_cap_hours=a.wall_cap_hours,
                               provenance=provenance, device=a.device)
    else:
        if not a.base_run:
            raise BenchRefused(f"{arm}: --base-run (the finished FL base run directory) is required")
        out = device.run_ft(data=d, data_root=root, arm=arm, seed=a.seed, base_run_dir=common.Path(a.base_run),
                            run_dir=run_dir, run_id=run_id, workers=a.workers, threads=a.threads,
                            limit_users=a.limit_users if a.smoke else None, device=a.device)
        out = {"status": out["status"], "timing": out["timing"]}
    print(f"[bm] {run_id}: {out['status']} timing={out.get('timing')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
