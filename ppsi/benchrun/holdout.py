"""The holdout (TEST) evaluation of finished runs, after model selection on the inner validation split.

For each named run (a finished RESULT.json): a preflight first loads every selected checkpoint and scores three
inner-validation rows (first, middle and last user) through the same code path; only then is holdout.csv read, every
checkpoint (PRACTICAL_BEST, and CONTROLLED_BUDGET when the run has one) is scored on the holdout in memory, and the
results are written once per run (test/TEST_<rule>.json and test/test_<rule>.ranks_int32.npy, write-once: a second
holdout evaluation of the same run is refused). Each run is scored with its own context length. Device-arm runs
repeat the per-user fine-tuning on the holdout context. The device class must be the runs' own.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import BenchRefused, read_json, utc, write_exclusive


def _ml_of(res: dict) -> int:
    return int(res["spec"].get("max_len") or 50)


def _seed_of(res: dict) -> int:
    spec = res["spec"]
    return int(spec["cfg"]["seed"]) if "cfg" in spec else int(spec["seed"])


def score_run(res: dict, rule: str, c: dict, run_dir: Path, data, view, *, dataset: str, data_root, workers: int,
              threads: int, rcp, say=print, dev=None):
    """Ranks of `view` (TEST, or a validation subset in the preflight) under one checkpoint of one finished run."""
    from . import accel, device
    from .central import load_weights
    from .evaluate import evaluate_view
    from .fl_engine import load_pf_matrix
    from .model import build_model
    cuda = accel.is_cuda(dev)
    p = Path(c["file"]) if Path(c["file"]).is_absolute() else run_dir / c["file"]
    if res["kind"] == "ft":
        ranks, _n, _w = device.ft_ranks(dataset=dataset, data_root=data_root, theta_path=str(p), seed=_seed_of(res),
                                        view=view, workers=workers, threads=threads, rcp=rcp, say=say,
                                        **({"device": "cuda"} if cuda else {}))
        return ranks
    spec = res["spec"]
    built = build_model(spec["variant"], data.K, _seed_of(res), dropout=tuple(spec.get("dropout", (0.1,))),
                        max_len=spec.get("max_len"), **({"device": dev} if cuda else {}))
    built["adapter"].load_state_(load_weights(p)["theta"])
    P = load_pf_matrix(run_dir / c["pf_store"]["file"], data) if c.get("pf_store") else None
    return evaluate_view(built["adapter"], view, P=P)


def evaluate_holdout(runs_root, dataset: str, data, data_root, run_names, *, rcp, workers: int = 1,
                     threads: int = 16, say=print, device=None) -> dict:
    from . import accel
    from . import data as D
    from .data import EvalView
    from .evaluate import summarize
    dev = accel.torch_device(device)
    cuda = accel.is_cuda(dev)
    dclass = accel.device_class(dev)
    devs = {"device_class": dclass} if cuda else {}
    base = Path(runs_root) / dataset
    runs = []
    for name in run_names:
        if not (base / name / "RESULT.json").is_file():
            raise BenchRefused(f"{name}: no RESULT.json under {base}")
        res = read_json(base / name / "RESULT.json")
        if res.get("status") not in ("DONE_MAIN",) or res.get("kind") not in ("central", "fl", "ft"):
            raise BenchRefused(f"{name}: not a finished central / federated / device run")
        if accel.device_class_of(res) != dclass:
            raise BenchRefused(f"{name} is a {accel.device_class_of(res)} run, this evaluation runs on {dclass}")
        runs.append((name, res))
    datas, views = {data.max_len: data}, {}

    def get_data(ml):
        if ml not in datas:
            datas[ml] = D.load(dataset, data_root, max_len=ml)
        return datas[ml]

    def get_view(ml):
        if ml not in views:
            views[ml] = D.load_holdout_view(get_data(ml), data_root)
        return views[ml]

    kw = {"dataset": dataset, "data_root": data_root, "threads": threads, "rcp": rcp, **({"dev": dev} if cuda else {})}
    for name, res in runs:                                              # --- preflight (validation rows only)
        d = get_data(_ml_of(res))
        vv = d.validation_view()
        sel = np.unique(np.asarray([0, len(vv.users) // 2, len(vv.users) - 1]))
        sub = EvalView(d, "VALIDATION", vv.users[sel], vv.ends[sel], vv.targets[sel], source="inner")
        for rule, c in res["checkpoints"].items():
            accel.assert_numerics(dev)
            r = score_run(res, rule, c, base / name, d, sub, workers=1, say=lambda *_: None, **kw)
            if r.shape != (len(sel),) or (r < 0).any():
                raise BenchRefused(f"preflight: {name} / {rule} did not score its validation rows")
    scored = {}                                                         # --- score the holdout, in memory only
    for name, res in runs:
        d, v = get_data(_ml_of(res)), get_view(_ml_of(res))
        for rule, c in res["checkpoints"].items():
            accel.assert_numerics(dev)
            ranks = score_run(res, rule, c, base / name, d, v, workers=workers, say=say, **kw)
            scored[(name, rule)] = (ranks, summarize(ranks, v, d.band))
    v0 = get_view(_ml_of(runs[0][1]))
    out = {"dataset": dataset, "utc": utc(), "n_test_users": len(v0.users),
           "n_cold_holdout_items_as_misses": v0.n_cold_items, "n_holdout_rows_unknown_user": v0.n_unknown_users,
           "runs": {}, **devs}
    for name, res in runs:                                              # --- write once per run
        tdir = base / name / "test"
        tdir.mkdir(exist_ok=True)
        per = {}
        for rule, c in res["checkpoints"].items():
            ranks, summ = scored[(name, rule)]
            write_exclusive(tdir / f"TEST_{rule}.json", {"run": name, "rule": rule, "mark": c.get("mark"),
                                                         "checkpoint_sha256": c["sha256"], "summary": summ,
                                                         "utc": utc(), **devs})
            np.save(tdir / f"test_{rule}.ranks_int32.npy", ranks.astype(np.int32))
            per[rule] = summ
        out["runs"][name] = per
    return out
