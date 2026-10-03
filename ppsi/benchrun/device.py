"""The device arms (FT_FA, FT_FA_WARM, FT_FA_WARM_FROZEN): per-user on-device fine-tuning of a federated base model.

Per evaluation user u (ppsi.fedsim.finetune.FTRunner): copy the federated base checkpoint (sha256 verified against
the base run's RESULT.json), fine-tune ALL trainable parameters on the user's OWN support = its TRAIN decisions
t = 1 .. L-2 (target_ts = t, support cutoff = L-1: strictly before the inner validation target) with the recipe's FT
cell (LR = lr_frac x lr_peak_base, `passes` passes, fresh AdamW, batch 16, clip 1.0, wd 1e-5), then score the user's
one target against the full catalogue (seen items filtered); the copy is discarded. The TEST evaluation fine-tunes on
the same support (the inner-validation item is NOT added to the support, so the model and the protocol are exactly
those of the validation look).
Parallel: users are split over W processes (spawn); each process uses T torch threads.
Device: `device` None / "cpu" = the CPU-FP32 path; "cuda" = the deterministic GPU class (accel.py; every worker
process re-enforces it on the one visible GPU). The federated base must be of the same class.
"""
from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from . import recipe
from .common import BenchRefused, read_json, sha256_file, utc, write_json
from .evaluate import rank_chunk, summarize

_G: dict = {}


def load_base(base_run_dir: Path, rule: str, allow_smoke: bool = False) -> tuple:
    """(theta state, checkpoint record, base RESULT) of one rule of a finished FL base run."""
    from .central import load_weights
    base_run_dir = Path(base_run_dir)
    res = read_json(base_run_dir / "RESULT.json")
    ok = ("DONE_MAIN", "SMOKE_DONE_NOT_EVIDENCE") if allow_smoke else ("DONE_MAIN",)
    if res.get("status") not in ok or res.get("kind") != "fl":
        raise BenchRefused(f"{base_run_dir} is not a finished FL run")
    ck = res["checkpoints"][rule]
    p = base_run_dir / ck["file"]
    if sha256_file(p) != ck["sha256"]:
        raise BenchRefused(f"{p}: sha256 differs from the base RESULT.json")
    return load_weights(p)["theta"], dict(ck, abs_path=str(p)), res


def _init_worker(dataset: str, data_root: str, theta_path: str, seed: int, threads: int, max_len: int,
                 device=None, rcp=None):
    import torch
    torch.set_num_threads(int(threads))
    from ppsi.fedsim.numerics import enforce_strict_fp32, state_digest

    from . import accel
    from . import data as D
    from .central import load_weights
    from .model import build_model
    dev = accel.torch_device(device)                             # cpu (default, untouched) or the GPU class
    enforce_strict_fp32(deterministic=True, warn_only=not accel.is_cuda(dev))
    d = D.load(dataset, data_root, max_len=max_len)
    theta = load_weights(theta_path)["theta"]
    _G.update(data=d, theta=theta, digest=state_digest(theta), seed=int(seed), recipe=rcp,
              adapter=build_model(rcp["fl"]["model_variant"], d.K, seed, device=dev)["adapter"])


def make_view(d, split, users, ends, targets, source):
    """The worker's view is rebuilt WITH the source of the original (TEST = the full train history, validation =
    inner)."""
    from .data import EvalView
    return EvalView(d, split, users, ends, targets, source=source)


def _ft_rows(args) -> tuple:
    """Fine-tune + score the view rows `rows`; returns (rows, ranks, n_support)."""
    from ppsi.fedsim.client import ClientData
    from ppsi.fedsim.finetune import FTRecipe, FTRunner
    split, users, ends, targets, source, rows = args
    d, adapter = _G["data"], _G["adapter"]
    view = make_view(d, split, users, ends, targets, source)
    rec = FTRecipe(base_digest=_G["digest"], seed=_G["seed"], **recipe.ft_recipe_kwargs(_G["recipe"]))
    runner = FTRunner(adapter, _G["theta"], rec)
    out = np.zeros(len(rows), dtype=np.int64)
    nsup = np.zeros(len(rows), dtype=np.int64)
    for j, i in enumerate(rows):
        u = int(users[i])
        cl = ClientData(d.key(u), d.client_examples(u))
        got = {}

        def hook(ad, budget, _i=int(i), _got=got):
            _got["r"] = int(rank_chunk(ad, view, np.asarray([_i]))[0])
            return _got["r"]

        res = runner.run(cl, hook, support_cutoff=int(d.lengths[u]) - 1)
        out[j] = got["r"]
        nsup[j] = res.n_support_events
    return np.asarray(rows, dtype=np.int64), out, nsup


def ft_ranks(*, dataset: str, data_root, theta_path: str, seed: int, view, workers: int, threads: int,
             rcp, say=print, device=None) -> tuple:
    """Ranks of every row of `view` after the per-user FT. Deterministic regardless of `workers`."""
    n = len(view.users)
    idx = np.arange(n, dtype=np.int64)
    chunks = [idx[w::workers] for w in range(workers)]
    t0 = time.perf_counter()
    payload = [(view.split, view.users, view.ends, view.targets, view.source, c) for c in chunks if c.size]
    if workers <= 1:
        _init_worker(dataset, str(data_root), theta_path, seed, threads, view.data.max_len, device, rcp)
        res = [_ft_rows(p) for p in payload]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(len(payload), initializer=_init_worker,
                      initargs=(dataset, str(data_root), theta_path, seed, threads, view.data.max_len, device,
                                rcp)) as pool:
            res = pool.map(_ft_rows, payload, chunksize=1)
    ranks = np.full(n, -1, dtype=np.int64)
    nsup = np.zeros(n, dtype=np.int64)
    for rows, r, s in res:
        ranks[rows] = r
        nsup[rows] = s
    say(f"[ft] {n} users in {time.perf_counter() - t0:.1f}s ({workers} workers x {threads} threads)")
    return ranks, nsup, time.perf_counter() - t0


def run_ft(*, data, data_root, arm: str, seed: int, base_run_dir: Path, run_dir: Path, run_id: str, workers: int,
           threads: int, rcp, limit_users: int | None = None, say=print, device=None) -> dict:
    """Inner-validation FT evaluation of the base's PRACTICAL_BEST and CONTROLLED_BUDGET checkpoints."""
    run_dir = Path(run_dir)
    from .common import refuse_overwrite
    refuse_overwrite(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    base_res = read_json(Path(base_run_dir) / "RESULT.json")
    want = recipe.FT_BASE_OF[arm]
    if base_res["arm"] != want:
        raise BenchRefused(f"{arm} needs a {want} base run, got {base_res['arm']}")
    if int(base_res["spec"]["cfg"]["seed"]) != int(seed):
        raise BenchRefused("the FT seed must equal the base run's seed")
    from . import accel
    dev = accel.torch_device(device)                             # the federated base must be of this device class
    dclass, cuda = accel.device_class(dev), accel.is_cuda(dev)
    if accel.device_class_of(base_res) != dclass:
        raise BenchRefused(f"the FL base is a {accel.device_class_of(base_res)} run, this FT run is {dclass}")
    devf = {"device_class": dclass, "device": accel.device_record(dev)} if cuda else {}
    view = data.validation_view()
    if limit_users:
        sel = np.arange(min(limit_users, len(view.users)))
        from .data import EvalView
        view = EvalView(data, view.split, view.users[sel], view.ends[sel], view.targets[sel])
    ftk = recipe.ft_recipe_kwargs(rcp)
    write_json(run_dir / "ARM_SPEC.json", {"arm": arm, "run_id": run_id, "base_run": str(base_run_dir),
                                           "ft_recipe": dict(ftk, passes=list(ftk["passes"])),
                                           "recipe_sha256": rcp.get("recipe_sha256"), "peak_lr": ftk["peak_lr"],
                                           "seed": int(seed), "dataset": data.dataset, "utc": utc(), **devf})
    checkpoints, val, timing = {}, {}, {}
    for rule in ("PRACTICAL_BEST", "CONTROLLED_BUDGET"):
        _theta, ck, _ = load_base(base_run_dir, rule, allow_smoke=bool(limit_users))
        ranks, nsup, wall = ft_ranks(dataset=data.dataset, data_root=data_root, theta_path=ck["abs_path"], seed=seed,
                                     view=view, workers=workers, threads=threads, rcp=rcp, say=say,
                                     device="cuda" if cuda else None)
        np.save(run_dir / f"val_{rule}.ranks_int32.npy", ranks.astype(np.int32))
        val[rule] = summarize(ranks, view, data.band)
        checkpoints[rule] = {"file": ck["abs_path"], "sha256": ck["sha256"], "mark": ck["mark"]}
        timing[rule] = {"wall_s": round(wall, 2), "n_users": len(view.users),
                        "s_per_user": wall / max(1, len(view.users)), "workers": workers, "threads": threads,
                        "mean_support_events": float(nsup.mean())}
    result = {"run_id": run_id, "arm": arm, "kind": "ft", "status": "DONE_MAIN" if not limit_users else "PARTIAL_LIMIT",
              "base_run": str(base_run_dir), "base_arm": want, "checkpoints": checkpoints, "validation": val,
              "timing": timing, "ft_recipe": dict(ftk, passes=list(ftk["passes"])),
              "spec": {"cfg": {"seed": int(seed)}, "seed": int(seed)},
              "utc": utc(), **devf}
    write_json(run_dir / "RESULT.json", result)
    write_json(run_dir / "STATUS.json", {"run_id": run_id, "status": result["status"], "utc": utc(),
                                         **({"device_class": dclass} if cuda else {})})
    return result
