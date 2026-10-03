"""The federated arms on the benchmark clients (FA / FA_WARM / FA_WARM_FROZEN, FP, PF, FA_Q8 and the DP chain).

The federated machinery is ppsi.fedsim, unchanged: FLRun (planned round-LR table, round-count endpoint, drop-out,
sweep participation, server optimiser step), Server, the SGD-with-momentum client, freeze_item_tables, install_q8 and
CheckpointManager. This module only supplies
  * the clients  one benchmark user = one federated client holding exactly its own TRAIN decisions
                 (data.client_examples);
  * the spec     recipe.build_fl_spec at the recipe's values;
  * the init     theta_pre = the PRE run's endpoint_6.0 checkpoint (sha256 verified) for the warm arms;
  * evaluation   the inner-validation view every 0.5 EFE (12 marks), logged for ALL / BAND / OTHERS.
The DP chain: S_CAL runs the Poisson-sampled federation without clipping and records S = the median client update
norm (ppsi.fedsim.dp.median_clip_norm) in a write-once S record; FA_1024 (clipping at S, no noise) and DP8 (clipping
at S plus Gaussian noise z x S) read that record.
Device: `device` None / "cpu" = the CPU-FP32 path; "cuda" = the deterministic GPU class (accel.py): server and worker
models on the GPU (ppsi.fedsim is device-aware); the spec, ARM_SPEC, RESULT, STATUS and the S record carry the class.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from . import recipe
from .common import (
    BenchRefused,
    append_jsonl,
    canon_sha256,
    read_json,
    sha256_file,
    utc,
    write_json,
)
from .evaluate import evaluate_view, flat_row, summarize


def load_theta_pre(pre_run_dir: Path, dataset: str, allow_smoke: bool = False) -> tuple:
    """(state dict, init identity) of the PRE run's endpoint_6.0 weights; refuses an unfinished / foreign PRE run."""
    from ppsi.fedsim.numerics import state_digest

    from .central import load_weights
    pre_run_dir = Path(pre_run_dir)
    res = read_json(pre_run_dir / "RESULT.json")
    ok = ("DONE_MAIN", "SMOKE_DONE_NOT_EVIDENCE") if allow_smoke else ("DONE_MAIN",)
    if res.get("status") not in ok or res.get("arm") != "PRE":
        raise BenchRefused(f"{pre_run_dir} is not a finished PRE run")
    cb = res["checkpoints"]["CONTROLLED_BUDGET"]
    p = pre_run_dir / cb["file"]
    if sha256_file(p) != cb["sha256"]:
        raise BenchRefused(f"{p}: sha256 differs from RESULT.json")
    theta = load_weights(p)["theta"]
    ident = {"kind": "PRE", "dataset": dataset, "run_id": res["run_id"], "fl_init_kind": "endpoint_6.0",
             "checkpoint_sha256": cb["sha256"], "state_digest": state_digest(theta),
             "seed": res["spec"]["seed"]}
    return theta, ident


def client_keys(data) -> tuple:
    users = np.flatnonzero(np.diff(data.user_dec_offsets) >= 1)
    keys = data.keys(users)
    n_valid = {data.key(u): data.n_valid(u) for u in users}
    return users, keys, n_valid


def run_fl(*, data, arm: str, seed: int, run_dir: Path, run_id: str, rcp, pre_run_dir: Path | None = None,
           smoke_rounds: int | None = None, resume: bool = False, say: Callable = print,
           ckpt_every: int = 64, dropout=None, max_len: int | None = None, s_record: Path | None = None,
           dp_dir: Path | None = None, wall_cap_hours: float | None = None,
           provenance: dict | None = None, device=None) -> dict:
    import torch

    from ppsi.fedsim.checkpoint import CheckpointManager
    from ppsi.fedsim.client import ClientData
    from ppsi.fedsim.numerics import enforce_strict_fp32, state_digest
    from ppsi.fedsim.participation import ParticipationPlan
    from ppsi.fedsim.runtime import FLRun, RunConfig, freeze_item_tables
    from ppsi.fedsim.server import Server

    from . import accel
    from .central import load_weights  # noqa: F401
    from .model import build_model
    dev = accel.torch_device(device)                             # cpu (default, untouched) or the GPU class
    cuda = accel.is_cuda(dev)
    enforce_strict_fp32(deterministic=True, warn_only=not cuda)
    dclass = accel.device_class(dev)
    drec = accel.device_record(dev)                              # None on CPU: the CPU records stay unchanged
    devf = {"device_class": dclass, "device": drec} if cuda else {}
    devs = {"device_class": dclass} if cuda else {}
    from .common import WallCap, refuse_overwrite
    run_dir = Path(run_dir)
    refuse_overwrite(run_dir)
    cap = WallCap(wall_cap_hours)
    run_dir.mkdir(parents=True, exist_ok=True)
    _users, keys, n_valid = client_keys(data)
    n_clients = len(keys)
    n_decisions = int(sum(n_valid.values()))
    wv = recipe.warm_variant(arm)
    init_theta = init_ident = None
    if wv is not None:
        if pre_run_dir is None:
            raise BenchRefused(f"{arm}: --pre-run (the PRE run directory = theta_pre) is required")
        init_theta, init_ident = load_theta_pre(pre_run_dir, data.dataset, allow_smoke=smoke_rounds is not None)
        pre_spec = read_json(Path(pre_run_dir) / "RESULT.json")["spec"]
        if int(pre_spec["seed"]) != int(seed):               # the warm start uses the PRE of the SAME run seed
            raise BenchRefused(f"theta_pre must be the seed-{seed} PRE (got seed {pre_spec['seed']})")
        if int(pre_spec.get("budget_efe") or 6) != 6:        # the longer-budget PREs are PRE-alone rows only
            raise BenchRefused("theta_pre must be a 6-EFE PRE: 24 / 96-EFE PREs are PRE-alone rows")
        want_cell = (int(max_len) if max_len else 50, list(dropout) if dropout is not None else [0.1])
        if (int(pre_spec.get("max_len", 50)), list(pre_spec.get("dropout", [0.1]))) != (want_cell[0], want_cell[1]):
            raise BenchRefused("a warm run uses the PRE trained with the SAME (max_len, dropout)")
        pre_class = accel.device_class_of(read_json(Path(pre_run_dir) / "RESULT.json"))
        if pre_class != dclass:                                  # one device class per dataset
            raise BenchRefused(f"theta_pre is a {pre_class} run, this run is {dclass} (device classes never mix)")
    ca = recipe.base_arm(arm)
    clip = None
    srec = None
    if recipe.is_poisson_arm(arm) and ca != "S_CAL":
        if s_record is None:
            raise BenchRefused(f"{arm}: --s-record (the S_RECORD of the matching S_CAL run) is required")
        srec = load_s_record(s_record, dataset=data.dataset, variant=wv or "COLD", n_clients=n_clients, seed=int(seed),
                             init_digest=init_ident["state_digest"] if init_ident else None,
                             allow_smoke=smoke_rounds is not None)
        if accel.device_class_of(srec) != dclass:                # the S record of the same device class
            raise BenchRefused(f"{s_record}: S record is {accel.device_class_of(srec)}, this run is {dclass}")
        clip = srec["S"]
    spec = recipe.build_fl_spec(arm, seed=seed, n_clients=n_clients, recipe=rcp, init_from=init_ident,
                                clip_norm=clip)
    if srec is not None:
        spec["s_record"] = {"path": str(s_record), "S": clip, "file_sha256": sha256_file(s_record),
                            "self_sha256": srec["self_sha256"]}
    spec["provenance"] = provenance
    spec["variant"] = rcp["fl"]["model_variant"]
    spec["dropout"] = list(dropout) if dropout is not None else [0.1]
    spec["max_len"] = int(max_len) if max_len else 50
    spec["data"] = {"dataset": data.dataset, "fingerprint": data.fingerprint(), "n_clients": n_clients,
                    "n_decisions": n_decisions}
    if cuda:                                                     # the class is part of the spec (and its hash)
        spec["device_class"] = dclass
    spec_sha = canon_sha256(spec)
    cfg = RunConfig.from_dict(dict(spec["cfg"], run_id=run_id, ckpt_every=int(ckpt_every)))
    plan_cls = (_exact_q_plan_cls(rcp["dp"]["q_nominal"]) if spec["plan"]["sampling"] == "poisson"
                else ParticipationPlan)
    plan = plan_cls.build(keys, seed=int(seed), manifest_hash=data.fingerprint(),
                          group_size=spec["plan"]["group_size"], sampling=spec["plan"]["sampling"])

    mk = {"dropout": tuple(spec["dropout"]), "max_len": max_len, "device": dev}
    built = build_model(spec["variant"], data.K, seed, **mk)
    server_adapter = built["adapter"]
    worker = build_model(spec["variant"], data.K, seed, **mk)["adapter"]
    init_sha = built["init_sha256"]
    if init_theta is not None:                         # theta_pre into the server model, proven by its digest
        with torch.no_grad():
            server_adapter.load_state_(init_theta)
        got = state_digest(server_adapter.broadcast_state())
        if got != init_ident["state_digest"]:
            raise BenchRefused(f"theta_pre loaded digest {got[:12]}… != {init_ident['state_digest'][:12]}…")
        init_sha = got
    frozen = tuple(spec.get("frozen_item_tables") or ())
    freeze_rec = None
    if frozen:
        freeze_rec = freeze_item_tables(server_adapter, frozen)
        theta_pre = server_adapter.broadcast_state()
        freeze_item_tables(worker, frozen, source=theta_pre)
    server = Server(server_adapter, init_sha256=init_sha)
    from ppsi.fedsim.personal import PersonalStore
    store = PersonalStore(server_adapter.query_dim) if cfg.method == "PF" else None
    ckpt = CheckpointManager(run_dir / "ckpt", run_id, retain_weights=smoke_rounds is None)
    view = data.validation_view()
    ev_dir = run_dir / "evals"
    ev_dir.mkdir(exist_ok=True)
    holder: dict = {}

    def get_client(k: str) -> ClientData:
        return ClientData(k, data.client_examples(data.user_of_key(k)))

    def eval_fn(_server) -> float:
        run = holder["run"]
        mark = run._half_units() / 2.0
        te = time.perf_counter()
        accel.assert_numerics(dev)                               # TF32 / non-strict numerics refused (GPU)
        P = dense_personal(run.store, data, server_adapter.query_dim) if run.store is not None else None
        ranks = evaluate_view(server_adapter, view, P=P)
        summ = summarize(ranks, view, data.band)
        val = float(summ["ALL"]["ndcg@10"])
        tag = f"{mark:.1f}"
        np.save(ev_dir / f"eval_{tag}.ranks_int32.npy", ranks.astype(np.int32))
        write_json(ev_dir / f"eval_{tag}.json", {"mark": mark, "efe": run.counters.efe, "round": run.state.cursor,
                                                "select_value": val, "summary": summ,
                                                "eval_wall_s": round(time.perf_counter() - te, 2)})
        append_jsonl(run_dir / "metrics.jsonl", {"kind": "eval", "mark": mark, "efe": run.counters.efe,
                                                "round": run.state.cursor, "select_value": val, **flat_row(summ),
                                                "utc": utc()})
        if run.store is not None and smoke_rounds is None:    # PF: the store of the PRACTICAL_BEST / endpoint mark
            bm = run.state.best_metric
            if bm is None or val > bm + run.cfg.best_delta:
                save_pf_store(ckpt.dir / "pf_store_best.pt", run, mark)
            if round(mark * 2) == 12:
                save_pf_store(ckpt.dir / "pf_store_endpoint.pt", run, mark)
        return val

    kw = {"workers": [worker], "store": store, "ckpt": ckpt,
          "eval_fn": eval_fn if (smoke_rounds is None and ca != "S_CAL") else None,
          "n_valid_by_key": dict(n_valid)}
    spec_file = run_dir / "ARM_SPEC.json"
    if spec_file.is_file():                                      # a run dir never mixes device classes
        accel.check_resume(read_json(spec_file), dev)
    if resume and ((ckpt.dir / ckpt.LATEST).exists() or spec_file.exists()):
        prev = read_json(spec_file)
        if prev.get("spec_sha256") != spec_sha:
            raise BenchRefused(f"--resume: ARM_SPEC.json is spec {prev.get('spec_sha256')}, not {spec_sha}")
        run = FLRun.resume(cfg, plan, server, get_client, n_decisions, **kw)
    else:
        if (ckpt.dir / ckpt.LATEST).exists():
            raise BenchRefused(f"{ckpt.dir / ckpt.LATEST} exists: use --resume")
        run = FLRun(cfg, plan, server, get_client, n_decisions, **kw)
        write_json(spec_file, {"arm": arm, "run_id": run_id, "spec_sha256": spec_sha, "arm_spec": spec,
                               "rounds_full_run": int(spec["max_rounds"]), "utc": utc(),
                               "init_from": init_ident, "freeze": freeze_rec, "init_sha256": init_sha,
                               **({"device": drec} if cuda else {})})
    q8 = None
    if spec.get("upload_codec") is not None:
        from ppsi.fedsim.quant import install_q8
        q8 = install_q8(run, seed=cfg.seed)
    holder["run"] = run
    max_rounds = int(spec["max_rounds"]) if smoke_rounds is None else int(smoke_rounds)
    write_json(run_dir / "STATUS.json", {"run_id": run_id, "arm": arm, "status": "RUNNING", "round": run.state.cursor,
                                         "utc": utc(), **devs})
    t0 = time.perf_counter()
    walls = []
    while not run.state.done and run.state.cursor < max_rounds:
        if cap.hit() and smoke_rounds is None:               # INCOMPLETE, never eligible
            write_json(run_dir / "RESULT.json", {"run_id": run_id, "arm": arm, "kind": "fl", "spec": spec,
                                                 "status": "INCOMPLETE_WALL_CAP", "rounds_done": run.state.cursor,
                                                 "utc": utc(), **devf})
            return {"status": "INCOMPLETE_WALL_CAP", "timing": {}}
        t1 = time.perf_counter()
        out_r = run.step()
        walls.append(time.perf_counter() - t1)
        if q8 is not None:
            q8.check_round(out_r["round"], out_r["survived"])
        if run.state.cursor % 50 == 0:
            append_jsonl(run_dir / "metrics.jsonl", {"kind": "train", "round": run.state.cursor, "efe": run.counters.efe,
                                                    "lr": out_r["lr"], "mean_round_s_last50": float(np.mean(walls[-50:])),
                                                    "train_loss": out_r.get("train_loss", {}).get("mean_step_loss"),
                                                    "utc": utc()})
            write_json(run_dir / "STATUS.json", {"run_id": run_id, "arm": arm, "status": "RUNNING",
                                                 "round": run.state.cursor, "utc": utc(), **devs})
    wall = time.perf_counter() - t0
    w = np.asarray(walls)
    timing = {"rounds_this_leg": int(w.size), "mean_round_s": float(w.mean()) if w.size else None,
              "median_round_s": float(np.median(w)) if w.size else None, "wall_s_this_leg": round(wall, 2),
              "rounds_full_run": int(spec["max_rounds"]), "n_clients": n_clients, "n_decisions": n_decisions}
    if smoke_rounds is not None:
        te = time.perf_counter()
        ranks = evaluate_view(server_adapter, view,
                              P=dense_personal(run.store, data, server_adapter.query_dim) if run.store is not None else None)
        summ = summarize(ranks, view, data.band)
        timing["eval_s"] = round(time.perf_counter() - te, 2)
        write_json(run_dir / "SMOKE_SUMMARY.json", {"label": "SMOKE_NOT_EVIDENCE", "arm": arm, "run_id": run_id,
                                                    "spec_sha256": spec_sha, "timing": timing, "rounds": run.state.cursor,
                                                    "efe": run.counters.efe, "summary": summ,
                                                    "frozen": freeze_rec})
        for p in sorted(ckpt.dir.glob("*.pt")):
            p.unlink()
        if ca == "S_CAL":                                  # smoke S record (label SMOKE_NOT_EVIDENCE) for the smoke chain
            from ppsi.fedsim.dp import median_clip_norm
            nl = run.dp_norm_log
            norm = median_clip_norm(nl, rounds=range(len(nl)))
            so = {"format": S_FORMAT, "S": float(norm["S"]), "n_norms": int(norm["n_norms"]), "dataset": data.dataset,
                  "variant": wv or "COLD", "n_clients": int(n_clients), "m": spec["plan"]["group_size"],
                  "init_digest": (spec.get("init_from") or {}).get("state_digest"), "label": "SMOKE_NOT_EVIDENCE",
                  "rounds_0based": list(norm["rounds_0based"]), "rule": norm["rule"], "run_id": run_id,
                  "spec_sha256": spec_sha, "seed": spec["cfg"]["seed"], **devs}
            so["self_sha256"] = canon_sha256(so)
            write_json(run_dir / "S_RECORD_SMOKE.json", so)
        from .central import save_weights
        rec = save_weights(server_adapter, run_dir / "ckpt" / "smoke_theta.pt", run_id=run_id, mark=run.counters.efe)
        write_json(run_dir / "RESULT.json", {"run_id": run_id, "arm": arm, "kind": "fl", "spec": spec,
                                             "status": "SMOKE_DONE_NOT_EVIDENCE", "label": "SMOKE_NOT_EVIDENCE",
                                             "checkpoints": {r: dict(rec, file="ckpt/smoke_theta.pt")
                                                             for r in ("PRACTICAL_BEST", "CONTROLLED_BUDGET")},
                                             **devf})
        write_json(run_dir / "STATUS.json", {"run_id": run_id, "status": "SMOKE_DONE_NOT_EVIDENCE", "utc": utc(),
                                             **devs})
        return {"status": "SMOKE_DONE_NOT_EVIDENCE", "timing": timing}
    if ca == "S_CAL":
        return _finish_scal(run, arm, wv, data, run_dir, run_id, spec, spec_sha, server, dp_dir, n_clients, timing,
                            devf=devf)
    evals = run.state.evals
    if not evals or evals[-1][0] != 12:
        raise RuntimeError("the run ended without its 6.0 evaluation")
    best_u = run.state.best_half_efe
    ckpt.complete()
    cdir = ckpt.dir
    checkpoints = {"PRACTICAL_BEST": {"file": str((cdir / ckpt.BEST).relative_to(run_dir).as_posix()),
                                      "sha256": sha256_file(cdir / ckpt.BEST), "mark": best_u / 2.0},
                   "CONTROLLED_BUDGET": {"file": str((cdir / ckpt.ENDPOINT).relative_to(run_dir).as_posix()),
                                         "sha256": sha256_file(cdir / ckpt.ENDPOINT), "mark": 6.0}}
    if store is not None:                                  # PF: pin the paired personal stores
        for rule, fn in (("PRACTICAL_BEST", "pf_store_best.pt"), ("CONTROLLED_BUDGET", "pf_store_endpoint.pt")):
            checkpoints[rule]["pf_store"] = {"file": str((cdir / fn).relative_to(run_dir).as_posix()),
                                             "sha256": sha256_file(cdir / fn)}
    val = {rule: read_json(ev_dir / f"eval_{c['mark']:.1f}.json")["summary"] for rule, c in checkpoints.items()}
    result = {"run_id": run_id, "arm": arm, "kind": "fl", "status": "DONE_MAIN", "spec_sha256": spec_sha,
              "spec": spec, "checkpoints": checkpoints, "validation": val, "evals": evals, "timing": timing,
              "frozen": freeze_rec, "bytes": _bytes(run), "init_from": init_ident,
              "summary": {k: v for k, v in run.summary().items() if k in ("rounds", "counters", "lr_table_digest")},
              "utc": utc(), **devf}
    write_json(run_dir / "RESULT.json", result)
    write_json(run_dir / "STATUS.json", {"run_id": run_id, "status": "DONE_MAIN", "utc": utc(), **devs})
    return {"status": "DONE_MAIN", "timing": timing, "result": result}


S_FORMAT = "BENCH_S_RECORD_1"


def s_record_path(dp_dir: Path, variant: str, seed: int) -> Path:
    """One S record per run seed: S_RECORD_<VARIANT>_s<seed>.json."""
    return Path(dp_dir) / f"S_RECORD_{variant}_s{int(seed)}.json"


def load_s_record(path, *, dataset: str, variant: str, n_clients: int, init_digest: str | None,
                  allow_smoke: bool = False, seed: int | None = None) -> dict:
    obj = read_json(path)
    body = {k: v for k, v in obj.items() if k != "self_sha256"}
    if obj.get("format") != S_FORMAT or obj.get("self_sha256") != canon_sha256(body):
        raise BenchRefused(f"{path}: not a verified {S_FORMAT} file")
    if obj.get("label") != "S_CALIBRATION" and not (allow_smoke and obj.get("label") == "SMOKE_NOT_EVIDENCE"):
        raise BenchRefused(f"{path}: a smoke S record cannot feed a real DP run")
    if seed is not None and int(obj.get("seed", -1)) != int(seed):          # a run binds its OWN seed's record
        raise BenchRefused(f"{path}: S record is bound to seed {obj.get('seed')}, this run is seed {seed}")
    for k, v in (("dataset", dataset), ("variant", variant), ("n_clients", int(n_clients)), ("init_digest", init_digest)):
        if obj.get(k) != v:
            raise BenchRefused(f"{path}: S record {k} {obj.get(k)!r} != this run's {v!r}")
    return obj


def _finish_scal(run, arm, wv, data, run_dir, run_id, spec, spec_sha, server, dp_dir, n_clients, timing,
                 devf=None) -> dict:
    from ppsi.fedsim.dp import median_clip_norm
    norm = median_clip_norm(run.dp_norm_log)
    variant = wv or "COLD"
    obj = {"format": S_FORMAT, "S": float(norm["S"]), "n_norms": int(norm["n_norms"]),
           "rounds_0based": list(norm["rounds_0based"]), "rule": norm["rule"], "dataset": data.dataset,
           "variant": variant, "n_clients": int(n_clients), "m": spec["plan"]["group_size"],
           "init_digest": (spec.get("init_from") or {}).get("state_digest"), "run_id": run_id,
           "spec_sha256": spec_sha, "seed": spec["cfg"]["seed"], "label": "S_CALIBRATION",
           "disclosure": "non-private; follows the unclipped trajectory"}
    if devf:                                                     # the GPU class (absent = CPU-FP32)
        obj["device_class"] = devf["device_class"]
    obj["self_sha256"] = canon_sha256(obj)
    from .common import write_exclusive
    p = s_record_path(dp_dir if dp_dir is not None else Path(run_dir).parent / "dp", variant, spec["cfg"]["seed"])
    write_exclusive(p, obj)
    write_json(Path(run_dir) / "RESULT.json", {"run_id": run_id, "arm": arm, "kind": "scal", "status": "DONE_S_CAL",
                                               "spec": spec, "s_record": str(p), "S": obj["S"], "timing": timing,
                                               **(devf or {})})
    write_json(Path(run_dir) / "STATUS.json", {"run_id": run_id, "status": "DONE_S_CAL", "utc": utc(),
                                               **({"device_class": devf["device_class"]} if devf else {})})
    return {"status": "DONE_S_CAL", "timing": timing, "S": obj["S"]}


def dense_personal(store, data, d: int):
    """[U, d] personal matrix (zeros for a user without a state) for the PF evaluation: shared + p_u."""
    import torch
    P = torch.zeros(data.U, int(d))
    for k in store.keys():  # noqa: SIM118  (PersonalStore.keys, not a dict)
        P[data.user_of_key(k)] = store.p(k)
    return P


def save_pf_store(path, run, mark) -> None:
    from ppsi.fedsim.checkpoint import atomic_save
    atomic_save({"kind": "pf_store", "run_id": run.cfg.run_id, "mark": float(mark), "store": run.store.state_dict(),
                 "store_digest": run.store.digest(), "round": int(run.state.cursor)}, Path(path))


def load_pf_matrix(path, data):
    from ppsi.fedsim.checkpoint import load_checkpoint
    from ppsi.fedsim.personal import PersonalStore
    st = PersonalStore.from_state_dict(load_checkpoint(Path(path))["store"])
    return dense_personal(st, data, st.query_dim)


def _exact_q_plan_cls(q: float):
    from dataclasses import dataclass

    from ppsi.fedsim.numerics import derive_seed
    from ppsi.fedsim.participation import ParticipationPlan

    @dataclass(frozen=True)
    class _ExactQPlan(ParticipationPlan):
        """Every Poisson arm includes each client independently with probability EXACTLY q (1 / q an integer); the
        plan's group_size (= m = round(q N)) is only the fixed DP denominator. This subclass replaces the cohort draw
        (u_i ~ Uniform{0 .. 1/q - 1}, member iff u_i = 0) and the reported inclusion probability."""

        @property
        def inclusion_probability(self) -> float:
            return float(q)

        def round_positions(self, round_idx: int):
            if self.sampling != "poisson":
                return super().round_positions(round_idx)
            if round_idx < 0:
                raise ValueError("negative round")
            inv = round(1.0 / float(q))
            rng = np.random.Generator(np.random.PCG64(derive_seed(int(self.seed), "poisson_q", int(round_idx))))
            u = rng.integers(0, inv, size=self.n_clients, dtype=np.int64)
            return np.flatnonzero(u == 0).astype(np.int64)

        def describe(self) -> dict:
            d = super().describe()
            d["inclusion_rule"] = ("u_i ~ Uniform{0 .. 1/q - 1} (PCG64, derive_seed(seed, 'poisson_q', r)); "
                                   f"member iff u_i = 0 (q = {float(q)!r})")
            return d
    return _ExactQPlan


def _bytes(run) -> dict:
    try:
        return {"ledger": run.ledger.snapshot()}
    except Exception:  # noqa: BLE001  (monitoring only)
        return {}
