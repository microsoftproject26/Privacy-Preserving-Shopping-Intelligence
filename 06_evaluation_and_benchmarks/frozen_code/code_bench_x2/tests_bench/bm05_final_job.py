"""BM-05 helper (not a test module): the TEST look (`final`) on a tiny synthetic fixture with ONE code tree.

  --stage prepare   train PRE, FA, PF and FT_FA (all users) on --device, then write a non-strict LOCK of the four runs
                    (this tree's code; the LOCK carries the device class);
  --stage final     run v4bench.final.final_eval on a prepared runs tree and write the digests of the whole tree.

Used by test_bm05_final.py for
  * the CPU byte-identity of `final`: code_bench4's final (--device legacy: no device argument) vs this tree's final with
    --device cpu, each on its own byte-identical copy of one prepared tree;
  * the GPU two-run test: prepare on the GPU class once, then final --device cuda on two copies (Ops, A40 server).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

DS = "s3_beauty"
SEED = 2026
RUNS = ("PRE_s2026", "FA_s2026", "PF_s2026", "FT_FA_s2026")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("prepare", "final"))
    ap.add_argument("--code-root", required=True)
    ap.add_argument("--lib-root", required=True)
    ap.add_argument("--release", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda", "legacy"))
    ap.add_argument("--gpu", type=int, default=None)
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
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(Path(a.lib_root).resolve()))
    sys.path.insert(0, str(Path(a.code_root).resolve()))
    sys.path.append(str(here))
    import numpy as np
    import torch
    torch.set_num_threads(int(a.threads))
    from v4bench import central, data as D, device as FT, final, fl_engine
    assert Path(final.__file__).resolve().parent.parent == Path(a.code_root).resolve(), final.__file__
    from bm03_digest_job import digest_tree
    kw = {} if a.device == "legacy" else {"device": a.device}
    runs = Path(a.runs)
    rel = Path(a.release)
    d = D.load(DS, rel)
    rd = lambda n: runs / DS / n                                # noqa: E731
    if a.stage == "prepare":
        central.EFFECTIVE_BATCH = 8
        central.run_central(data=d, arm="PRE", variant="SASREC_D64_B2", peak_lr=7.114e-4, dropout=(0.1,), seed=SEED,
                            rows=d.band_rows(), select_users=np.flatnonzero(d.band), run_dir=rd("PRE_s2026"),
                            run_id="t_pre", **kw)
        for arm in ("FA", "PF"):
            fl_engine.run_fl(data=d, arm=arm, seed=SEED, run_dir=rd(f"{arm}_s2026"), run_id=f"t_{arm}", **kw)
        FT.run_ft(data=d, data_root=rel, arm="FT_FA", seed=SEED, base_run_dir=rd("FA_s2026"), run_dir=rd("FT_FA_s2026"),
                  run_id="t_ft", workers=1, threads=int(a.threads), say=lambda *_: None, **kw)
        lock = final.make_lock(runs, DS, list(RUNS), data_fingerprint=d.fingerprint(), strict=False)
        print(json.dumps({"lock_device_class": lock.get("device_class"), "n_runs": len(lock["runs"])}))
        return 0
    out = final.final_eval(runs, DS, d, rel, workers=1, threads=int(a.threads), say=lambda *_: None, **kw)
    obj = {"device": a.device, "n_test_users": out["n_test_users"], "device_class": out.get("device_class"),
           "files": digest_tree(runs)}
    Path(a.out).write_text(json.dumps(obj, indent=1, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
