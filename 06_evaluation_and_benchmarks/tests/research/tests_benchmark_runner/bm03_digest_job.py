"""BM-03 helper (not a test module): run a tiny fixed benchmark job set with ONE code tree and write the digests of every
deterministic output. Used by test_bm03_device.py for
  * the CPU byte-identity guard: code_bench4 (code/_archive_bench4, or $BM_BENCH4_ROOT) vs this tree with --device cpu;
  * the GPU determinism test: this tree with --device cuda, twice (Ops runs it on the A40 server).

  python -B tests_bench/bm03_digest_job.py --code-root <dir containing v4bench> --lib-root <dir containing fl, v4models, ...>
         --release <synthetic release root> --runs <empty dir> --out digests.json [--device cpu|cuda|legacy] [--gpu 0]
         [--jobs central,fl,ft]

--device legacy calls the runner WITHOUT the device argument (the only way to drive code_bench4); cpu passes device="cpu".
Volatile fields (utc, wall-clock timings) and the absolute runs root are removed before hashing; everything else, including
every checkpoint / ranks file byte, is hashed as written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

VOLATILE = {"utc", "timing", "eval_wall_s", "wall_s", "step_s", "mean_round_s_last50", "wall_s_this_leg", "s_per_user",
            "eval_s"}
DS = "s3_beauty"
SEED = 2026


def _clean(o, roots):
    if isinstance(o, dict):
        return {k: _clean(v, roots) for k, v in o.items() if k not in VOLATILE}
    if isinstance(o, list):
        return [_clean(v, roots) for v in o]
    if isinstance(o, str):
        for r in roots:
            o = o.replace(r, "<RUNS>")
        return o.replace("\\", "/")
    return o


def _drop_latest(row):
    """fl.checkpoint's latest.pt (the FL resumable, deleted at completion) also stores the process-global numpy / Python
    `random` states, which are entropy-seeded per process and never used by the run: its bytes differ between any two
    processes, even of the same code. Only its size / digest fields in the checkpoint event logs are dropped."""
    if isinstance(row, dict) and (row.get("saved") == "latest.pt" or str(row.get("path", "")).endswith("latest.pt")):
        return {k: v for k, v in row.items() if k not in ("bytes", "content_digest", "sha256")}
    return row


def digest_tree(runs: Path) -> dict:
    roots = sorted({str(runs), runs.as_posix(), str(runs.resolve()), runs.resolve().as_posix()}, key=len, reverse=True)
    out = {}
    for p in sorted(runs.rglob("*")):
        if not p.is_file() or ".tmp" in p.name:
            continue
        rel = p.relative_to(runs).as_posix()
        if p.suffix == ".json":
            out[rel] = hashlib.sha256(json.dumps(_clean(json.loads(p.read_text(encoding="utf-8")), roots),
                                                 sort_keys=True).encode()).hexdigest()
        elif p.suffix == ".jsonl":
            rows = [_clean(json.loads(x), roots) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
            rows = [_drop_latest(r) for r in rows]
            out[rel] = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
        else:                                               # .pt / .npy: every byte as written
            out[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-root", required=True)
    ap.add_argument("--lib-root", required=True)
    ap.add_argument("--release", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda", "legacy"))
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--jobs", default="central,fl,ft")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args(argv)
    os.environ["BM_ALLOW_THREADS"] = "1"
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    if a.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(a.gpu if a.gpu is not None else 0))
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(a.lib_root).resolve()))
    sys.path.insert(0, str(Path(a.code_root).resolve()))       # v4bench resolves here first
    import numpy as np
    import torch
    torch.set_num_threads(int(a.threads))
    from v4bench import central, data as D, device as FT, fl_engine
    assert Path(central.__file__).resolve().parent.parent == Path(a.code_root).resolve(), central.__file__
    central.EFFECTIVE_BATCH = 8                                # as test_pipeline: several updates per mark on a tiny set
    kw = {} if a.device == "legacy" else {"device": a.device}
    runs = Path(a.runs)
    d = D.load(DS, Path(a.release))
    jobs = set(a.jobs.split(","))
    rd = lambda n: runs / DS / n                               # noqa: E731
    status = {}
    if jobs & {"central", "fl", "ft"}:
        status["PRE"] = central.run_central(
            data=d, arm="PRE", variant="SASREC_D64_B2", peak_lr=7.114e-4, dropout=(0.1,), seed=SEED, rows=d.band_rows(),
            select_users=np.flatnonzero(d.band), run_dir=rd("PRE_s2026"), run_id="t_pre", **kw)["status"]
    if "central" in jobs:
        status["C_FULL"] = central.run_central(
            data=d, arm="C_FULL", variant="SASREC_D256_B3", peak_lr=4.0e-4, dropout=(0.1,), seed=SEED,
            rows=np.arange(d.N), select_users=None, run_dir=rd("C_FULL__B24_L50_D0.1_s2026"), run_id="t_cfull",
            budget_efe=24, early_stop_marks=6, **kw)["status"]
    if jobs & {"fl", "ft"}:
        for arm in ("FA", "FA_WARM", "FA_Q8", "PF"):
            if arm != "FA" and "fl" not in jobs:
                continue
            status[arm] = fl_engine.run_fl(data=d, arm=arm, seed=SEED, run_dir=rd(f"{arm}_s2026"), run_id=f"t_{arm}",
                                           pre_run_dir=rd("PRE_s2026") if arm.endswith("WARM") else None,
                                           **kw)["status"]
    if "ft" in jobs:
        status["FT_FA"] = FT.run_ft(data=d, data_root=Path(a.release), arm="FT_FA", seed=SEED,
                                    base_run_dir=rd("FA_s2026"), run_dir=rd("FT_FA_s2026"), run_id="t_ft", workers=1,
                                    threads=int(a.threads), limit_users=6, say=lambda *_: None, **kw)["status"]
    obj = {"status": status, "device": a.device, "torch": torch.__version__, "files": digest_tree(runs)}
    Path(a.out).write_text(json.dumps(obj, indent=1, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
