"""v4bench.final — the LOCK and the ONE TEST look per dataset (BENCHMARK_PLAN_V1 §14; PREG1_AMENDMENT_07 B4).

  lock    runs_bench/<dataset>/LOCK.json (write-once, O_EXCL): every listed run must be DONE_MAIN; for each run the
          sha256 of RESULT.json and of both checkpoints (PRACTICAL_BEST, CONTROLLED_BUDGET) are recorded.
  final   refuses unless LOCK.json exists, lists the runs asked for and every recorded sha256 still matches; refuses a
          second look (TEST_LOOK.json is written O_EXCL BEFORE the first holdout value is read). Only then is the gate
          token minted that data._load_holdout_view requires; the holdout file is read nowhere else.
Nothing here modifies a run directory except writing test/ results next to it.
BM-03: the LOCK records the dataset's ONE device class (CPU-FP32 or NVIDIA_A40-FP32-DET) and refuses runs or pinned decision
files of mixed classes; verify_lock / open_test_gate are unchanged (the class is inside the self-hashed LOCK).
BM-05 (Fable §11 E1: one device class per dataset for training AND the TEST look): `final_eval(device=)` scores every locked
checkpoint (and runs the FT arms' per-user fine-tune) on that device; it refuses, before the attempt is logged and before the
holdout is read, a LOCK (or a locked run) of another device class (a LOCK without the field = CPU-FP32). device None / "cpu"
is the registered CPU path, unchanged byte for byte; on the GPU class the TEST outputs also carry the class.
"""
from __future__ import annotations

import secrets
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .common import (FL_VARIANT, BenchRefused, bench_code_sha256, canon_sha256, check_dataset, read_json, sha256_file, utc,
                     write_exclusive, write_json)

LOCK_FORMAT = "BM02_LOCK_1"
_TOKENS: dict = {}


def lock_path(runs_root, dataset: str) -> Path:
    return Path(runs_root) / check_dataset(dataset) / "LOCK.json"


def look_path(runs_root, dataset: str) -> Path:
    return Path(runs_root) / check_dataset(dataset) / "TEST_LOOK.json"


def _run_record(run_dir: Path) -> dict:
    res_p = run_dir / "RESULT.json"
    if not res_p.is_file():
        raise BenchRefused(f"{run_dir.name}: no RESULT.json (run not finished)")
    res = read_json(res_p)
    st = res.get("status")
    if st not in ("DONE_MAIN", "DONE_S_CAL"):
        raise BenchRefused(f"{run_dir.name}: status {st!r} is not DONE_MAIN / DONE_S_CAL (INCOMPLETE runs are ineligible)")
    cks = {}
    for rule, c in (res.get("checkpoints") or {}).items():
        p = Path(c["file"]) if Path(c["file"]).is_absolute() else run_dir / c["file"]
        got = sha256_file(p)
        if got != c["sha256"]:
            raise BenchRefused(f"{run_dir.name}: checkpoint {p} sha256 differs from RESULT.json")
        cks[rule] = {"file": str(c["file"]), "sha256": got, "mark": c["mark"]}
        if c.get("pf_store"):                                  # PF: the paired personal store is part of the checkpoint
            sp = run_dir / c["pf_store"]["file"]
            if sha256_file(sp) != c["pf_store"]["sha256"]:
                raise BenchRefused(f"{run_dir.name}: PF store {sp} sha256 differs from RESULT.json")
            cks[rule]["pf_store"] = dict(c["pf_store"])
    rec = {"run": run_dir.name, "arm": res["arm"], "kind": res["kind"],
           "seed": int(res["spec"]["seed"] if "seed" in res["spec"] else res["spec"]["cfg"]["seed"]),
           "result_sha256": sha256_file(res_p), "checkpoints": cks}
    if res["kind"] == "scal":
        rec["s_record_sha256"] = sha256_file(Path(res["s_record"]))
    return rec


def _label_seed(name: str) -> tuple:
    import re
    m = re.match(r"^(.*)_s(\d+)$", name)
    if not m:
        raise BenchRefused(f"run name {name!r} is not <ARM>_s<seed>")
    return m.group(1), int(m.group(2))


def _pinned_files(base: Path) -> dict:
    """M3: every selection / record file the LOCK pins (sha256): decisions, Track-B winners, S records, the manifest."""
    out = {}
    for rel in ("FA_WARM_DECISION.json", "SEED3_DECISION.json", "MANIFEST.json"):
        if (base / rel).is_file():
            out[rel] = sha256_file(base / rel)
    for sub, pat in (("trackb", "WINNER_*.json"), ("dp", "S_RECORD*.json")):
        if (base / sub).is_dir():
            for q in sorted((base / sub).glob(pat)):
                out[f"{sub}/{q.name}"] = sha256_file(q)
    return out


def make_lock(runs_root, dataset: str, run_names: Iterable[str], *, data_fingerprint: str, strict: bool = True,
              contracts_dir=None) -> dict:
    """strict (the production LOCK): pinned contracts verified, the decisions present, every Track-A label has both seeds
    (and 2028 where the R6 trigger fired), a Track-B grid cell has seed 2026 and its WINNER also seed 2027 (R5), S_CAL runs
    and their S records are listed and pinned, code-tree digest."""
    from . import recipe, selection
    from .common import CONTRACTS_DEFAULT
    names = sorted(set(run_names))
    if lock_path(runs_root, dataset).exists():
        raise BenchRefused(f"{lock_path(runs_root, dataset)} exists: the LOCK is write-once")
    if not names:
        raise BenchRefused("a LOCK needs at least one run")
    base = Path(runs_root) / check_dataset(dataset)
    recs = [_run_record(base / n) for n in names]
    pins = recipe.check_pins(contracts_dir or CONTRACTS_DEFAULT, required=strict)
    files = _pinned_files(base)
    device_class = _lock_device_class(base, names, files)       # BM-03: one device class per dataset
    for fname in ("FA_WARM_DECISION.json", "SEED3_DECISION.json"):
        if strict and fname not in files:
            raise BenchRefused(f"LOCK refused: {base / fname} is missing (R6 / B2 decisions are taken BEFORE the LOCK)")
    if strict:
        labels = {}
        for n in names:
            lab, sd = _label_seed(n)
            labels.setdefault(lab, set()).add(sd)
        need3 = set(read_json(base / "SEED3_DECISION.json")["third_seed_2028_needed_for"])
        winners = [read_json(q)["winner_run"] for q in sorted((base / "trackb").glob("WINNER_*.json"))] \
            if (base / "trackb").is_dir() else []
        for w in winners:
            if w not in names:
                raise BenchRefused(f"LOCK refused: the Track-B winner {w} is not listed")
        winner_labels = {_label_seed(w)[0] for w in winners}
        for lab, sds in labels.items():
            if lab.startswith("S_CAL"):
                want = set(selection.REQUIRED_SEEDS)              # B12: one S_CAL (and S record) per PRE seed
            elif "__B" in lab:                                   # a Track-B grid cell: seed 2026; its winner also 2027
                want = {2026} | ({2027} if lab in winner_labels else set())
            else:
                want = set(selection.REQUIRED_SEEDS) | ({2028} if lab in need3 else set())
            if not want <= sds:
                raise BenchRefused(f"LOCK refused: {lab} lacks seeds {sorted(want - sds)} (R5 / R6)")
    obj = {"format": LOCK_FORMAT, "dataset": dataset, "utc": utc(), "data_fingerprint": data_fingerprint,
           "code_sha256": bench_code_sha256(), "code_tree_digest": selection.tree_digest(), "strict": bool(strict),
           "pins": pins, "pinned_files": files, "runs": recs, "device_class": device_class}
    obj["self_sha256"] = canon_sha256({k: v for k, v in obj.items() if k != "self_sha256"})
    write_exclusive(lock_path(runs_root, dataset), obj)
    return obj


def _lock_device_class(base: Path, names, files: dict) -> str:
    """BM-03: the device class of every locked run (RESULT.json) and of every pinned decision / winner / S record (absent =
    CPU-FP32) must be the same; returns it. MANIFEST.json describes the data only and has no device class."""
    from .accel import device_class_of, one_class
    classes = {n: device_class_of(read_json(base / n / "RESULT.json")) for n in names}
    for rel in files:
        if rel.endswith(".json") and rel != "MANIFEST.json":
            classes[rel] = device_class_of(read_json(base / rel))
    return one_class(classes, "LOCK")


def verify_lock(runs_root, dataset: str, names: Optional[Iterable[str]] = None) -> dict:
    p = lock_path(runs_root, dataset)
    if not p.is_file():
        raise BenchRefused(f"TEST refused: {p} does not exist (the LOCK lists every run before the one TEST look)")
    lock = read_json(p)
    if lock.get("format") != LOCK_FORMAT or lock.get("dataset") != dataset:
        raise BenchRefused(f"{p}: not a {LOCK_FORMAT} file of {dataset}")
    if lock.get("self_sha256") != canon_sha256({k: v for k, v in lock.items() if k != "self_sha256"}):
        raise BenchRefused(f"{p}: self hash differs")
    from . import recipe, selection
    base = Path(runs_root) / dataset
    if lock.get("strict"):                                  # R9: pins and the code tree must still be the locked ones
        if recipe.check_pins(required=True) != lock["pins"]:
            raise BenchRefused("TEST refused: a pinned contract changed after the LOCK")
        if selection.tree_digest() != lock["code_tree_digest"]:
            raise BenchRefused("TEST refused: the r6 code tree differs from the LOCK's digest")
    for rel, sha in lock.get("pinned_files", {}).items():   # decisions, winners, S records, manifest
        if not (base / rel).is_file() or sha256_file(base / rel) != sha:
            raise BenchRefused(f"TEST refused: pinned file {rel} changed after the LOCK")
    listed = {r["run"]: r for r in lock["runs"]}
    want = sorted(listed) if names is None else sorted(set(names))
    missing = [n for n in want if n not in listed]
    if missing:
        raise BenchRefused(f"TEST refused: runs {missing} are not listed in {p}")
    for n in want:                                         # every recorded sha256 must still match
        now = _run_record(base / n)
        if now != listed[n]:
            raise BenchRefused(f"TEST refused: run {n} changed after the LOCK")
    return lock


def open_test_gate(runs_root, dataset: str, names: Optional[Iterable[str]] = None) -> str:
    """Verify the LOCK, refuse a second look, mint the gate token. TEST_LOOK.json is written by commit_look AFTER every
    locked run has been scored (M1), never before."""
    verify_lock(runs_root, dataset, names)
    lp = look_path(runs_root, dataset)
    if lp.exists():
        raise BenchRefused(f"TEST refused: {lp} exists - the one TEST look of {dataset} has already been taken")
    tok = secrets.token_hex(16)
    _TOKENS[tok] = dataset
    return tok


def commit_look(runs_root, dataset: str, lock: dict, devs: Optional[dict] = None) -> None:
    from .common import write_atomic_exclusive
    write_atomic_exclusive(look_path(runs_root, dataset), {"dataset": dataset, "utc": utc(),
                                                          "lock_self_sha256": lock["self_sha256"],
                                                          "runs": sorted(r["run"] for r in lock["runs"]),
                                                          **(devs or {})})


def check_gate_token(token, dataset: str) -> None:
    if not isinstance(token, str) or _TOKENS.get(token) != dataset:
        raise BenchRefused("the holdout is readable only through v4bench.final.open_test_gate (LOCK verified)")


def score_run(rec: dict, res: dict, rule: str, c: dict, run_dir: Path, data, view, *, dataset: str, data_root,
              workers: int, threads: int, say=print, dev=None):
    """Ranks of `view` (TEST, or a validation subset in the preflight) under one checkpoint of one locked run.
    BM-05: `dev` = the GPU class device, or None (CPU; the name `device` is the FT module here)."""
    from . import device
    from .central import load_weights
    from .evaluate import evaluate_view
    from .fl_engine import load_pf_matrix
    from .model import build_model
    p = Path(c["file"]) if Path(c["file"]).is_absolute() else run_dir / c["file"]
    if rec["kind"] == "ft":
        ranks, _n, _w = device.ft_ranks(dataset=dataset, data_root=data_root, theta_path=str(p), seed=rec["seed"],
                                        view=view, workers=workers, threads=threads, say=say,
                                        **({"device": "cuda"} if _is_cuda(dev) else {}))
        return ranks
    spec = res["spec"]
    built = build_model(spec.get("variant", FL_VARIANT), data.K, rec["seed"], dropout=tuple(spec.get("dropout", (0.1,))),
                        max_len=spec.get("max_len"), **({"device": dev} if _is_cuda(dev) else {}))
    built["adapter"].load_state_(load_weights(p)["theta"])
    P = load_pf_matrix(run_dir / c["pf_store"]["file"], data) if c.get("pf_store") else None
    return evaluate_view(built["adapter"], view, P=P)


def _is_cuda(device) -> bool:
    from .accel import is_cuda
    return is_cuda(device)


def check_lock_device(runs_root, dataset: str, device_class: str) -> None:
    """BM-05: the TEST look runs on the LOCK's device class only (absent = CPU-FP32); every locked run's RESULT agrees."""
    from .accel import device_class_of
    p = lock_path(runs_root, dataset)
    if not p.is_file():
        return                                              # open_test_gate refuses a missing LOCK
    lock = read_json(p)
    got = device_class_of(lock)
    if got != device_class:
        raise BenchRefused(f"TEST refused: the LOCK's device class is {got}, this look runs on {device_class} "
                           "(one device class per dataset for training and the TEST look; Fable §11 E1)")
    base = Path(runs_root) / dataset
    for r in lock.get("runs", []):
        rp = base / r["run"] / "RESULT.json"
        if rp.is_file() and device_class_of(read_json(rp)) != device_class:
            raise BenchRefused(f"TEST refused: locked run {r['run']} is not {device_class}")


def _ml_of(res: dict) -> int:
    return int(res["spec"].get("max_len") or 50)


def attempts_path(runs_root, dataset: str) -> Path:
    return Path(runs_root) / check_dataset(dataset) / "TEST_ATTEMPTS.jsonl"


def final_eval(runs_root, dataset: str, data, data_root, *, workers: int = 1, threads: int = 16, say=print,
               device=None) -> dict:
    """Every attempt (start / end / error) is appended to the append-only TEST_ATTEMPTS.jsonl; see _final_eval.
    BM-05: the device class must equal the LOCK's (refused before the attempt is logged)."""
    from . import accel
    from .common import append_jsonl
    dev = accel.torch_device(device)                         # cpu (default, untouched) or the GPU class
    cuda = accel.is_cuda(dev)
    check_lock_device(runs_root, dataset, accel.device_class(dev))
    devs = {"device_class": accel.device_class(dev)} if cuda else {}
    ap = attempts_path(runs_root, dataset)
    append_jsonl(ap, {"event": "start", "utc": utc(), "workers": workers, "threads": threads, **devs})
    try:
        out = _final_eval(runs_root, dataset, data, data_root, workers=workers, threads=threads, say=say,
                          **({"device": dev} if cuda else {}))
    except BaseException as e:                                          # noqa: BLE001 - logged, then re-raised
        append_jsonl(ap, {"event": "error", "utc": utc(), "type": type(e).__name__, "message": str(e)[:500]})
        raise
    append_jsonl(ap, {"event": "end", "utc": utc(), "n_runs": len(out["runs"]), "n_test_users": out["n_test_users"]})
    return out


def _final_eval(runs_root, dataset: str, data, data_root, *, workers: int = 1, threads: int = 16, say=print,
                device=None) -> dict:
    """The ONE TEST look. Order (M1): verify the LOCK -> PREFLIGHT (every locked checkpoint is loaded and scores validation
    rows, last user included, through the same code path) -> score every run on the holdout in memory -> only then
    TEST_LOOK.json is written (atomically) and the results are written. Each run is scored with its own max_len (M2)."""
    from . import data as D
    from .common import DATASETS
    from .data import EvalView
    from .evaluate import summarize
    from . import accel
    cuda = accel.is_cuda(device)
    devs = {"device_class": accel.device_class(device)} if cuda else {}
    dkw = {"dev": device} if cuda else {}
    tok = open_test_gate(runs_root, dataset)
    lock = read_json(lock_path(runs_root, dataset))
    base = Path(runs_root) / dataset
    datas, views = {data.max_len: data}, {}

    def get_data(ml):
        if ml not in datas:
            datas[ml] = D.load(dataset, data_root, max_len=ml)
        return datas[ml]

    def get_view(ml):
        if ml not in views:
            views[ml] = D._load_holdout_view(get_data(ml), data_root, _gate=tok)
            if dataset in DATASETS and views[ml].n_cold_items != 0:      # B10 Q5: Wave 1 has no cold holdout item
                raise BenchRefused(f"TEST refused: {views[ml].n_cold_items} cold holdout items on Wave-1 {dataset}")
        return views[ml]

    runs = [(rec, read_json(base / rec["run"] / "RESULT.json")) for rec in lock["runs"] if rec["kind"] != "scal"]
    for rec, res in runs:                                               # --- preflight (validation rows only)
        d = get_data(_ml_of(res))
        vv = d.validation_view()
        sel = np.unique(np.asarray([0, len(vv.users) // 2, len(vv.users) - 1]))
        sub = EvalView(d, "VALIDATION", vv.users[sel], vv.ends[sel], vv.targets[sel], source="inner")
        for rule, c in rec["checkpoints"].items():
            accel.assert_numerics(device)                       # BM-05 NC: TF32 / non-strict numerics refused (GPU)
            r = score_run(rec, res, rule, c, base / rec["run"], d, sub, dataset=dataset, data_root=data_root,
                          workers=1, threads=threads, say=lambda *_: None, **dkw)
            if r.shape != (len(sel),) or (r < 0).any():
                raise BenchRefused(f"preflight: {rec['run']} / {rule} did not score its validation rows")
    for ml in sorted({_ml_of(res) for _rec, res in runs}):               # the holdout is read only after the preflight
        get_view(ml)
    scored = {}                                                         # --- score the holdout, in memory only
    for rec, res in runs:
        d, v = get_data(_ml_of(res)), get_view(_ml_of(res))
        for rule, c in rec["checkpoints"].items():
            accel.assert_numerics(device)
            ranks = score_run(rec, res, rule, c, base / rec["run"], d, v, dataset=dataset, data_root=data_root,
                              workers=workers, threads=threads, say=say, **dkw)
            scored[(rec["run"], rule)] = (ranks, summarize(ranks, v, d.band))
    v0 = get_view(_ml_of(runs[0][1]))
    out = {"dataset": dataset, "utc": utc(), "n_test_users": int(len(v0.users)), "n_cold_holdout_items_as_misses":
           v0.n_cold_items, "n_holdout_rows_unknown_user": v0.n_unknown_users, "lock_self_sha256":
           lock["self_sha256"], "runs": {}, **devs}
    if cuda:
        out["device"] = accel.device_record(device)
    commit_look(runs_root, dataset, lock, devs)                         # --- the look is taken only now
    for rec, res in runs:
        run_dir = base / rec["run"]
        tdir = run_dir / "test"
        tdir.mkdir(exist_ok=True)
        per = {}
        for rule, c in rec["checkpoints"].items():
            ranks, summ = scored[(rec["run"], rule)]
            np.save(tdir / f"test_{rule}.ranks_int32.npy", ranks.astype(np.int32))
            write_exclusive(tdir / f"TEST_{rule}.json", {"run": rec["run"], "rule": rule, "mark": c["mark"],
                                                         "checkpoint_sha256": c["sha256"], "summary": summ,
                                                         "utc": utc(), **devs})
            per[rule] = summ
        out["runs"][rec["run"]] = per
    write_exclusive(base / "TEST_RESULTS.json", out)
    return out
