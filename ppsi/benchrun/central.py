"""C_FULL and PRE: the central recipe on the benchmark's TRAIN decisions.

One-process central training:
  data order   ppsi.central.order.pass_permutation(seed, pass, N) over the canonical decision order, consecutive
               effective batches of 256, the last batch of a pass at its own size;
  loss         ppsi.central.objective.accumulate_step (exact full-catalogue CE: sum over loss-eligible rows / n_eff);
  optimizer    AdamW(0.9, 0.999, 1e-8), weight decay 1e-5 on ndim >= 2 (adapter.param_groups), gradient clip 1.0,
               adapter.post_step() (output recentering) after every step;
  LR           peak x s(EFE at the END of the update) (ppsi.central.schedule, S0, exposure point end), the table
               planned and hashed before training; 6.0 EFE; evaluation every 0.5 EFE on the INNER VALIDATION view;
  selection    PRACTICAL_BEST = the strictly-best mark (value > best + 0.0002) of the selection metric,
               CONTROLLED_BUDGET = the 6.0 mark. C_FULL selects on ALL users' validation NDCG@10; PRE on its OWN
               population (the band users), because the server-side pretraining only has the band's data.
A longer exposure budget (24 or 96 EFE) stretches the S0 schedule to the budget's clock; with `early_stop_marks`
the run stops after that many marks without improvement (and then has no CONTROLLED_BUDGET row).
Weights: best.pt (PB) and endpoint_6.0.pt (CB) as checkpoint payloads {"kind": "weights", "theta": broadcast_state}.
Resume: latest.pt (model, optimizer, cursor, tracker, RNG) written at every mark; `resume=True` continues bitwise.
Device: `device` None / "cpu" is the CPU-FP32 path; "cuda" is the deterministic GPU class (accel.py): same data
order, same LR table, same init; spec / ARM_SPEC / RESULT / STATUS carry the device class, latest.pt also keeps the
CUDA RNG, and --resume refuses another device class.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .common import (
    PRIMARY_METRIC,
    BenchRefused,
    append_jsonl,
    canon_sha256,
    read_json,
    sha256_file,
    utc,
    write_json,
)
from .evaluate import evaluate_view, flat_row, summarize

EFFECTIVE_BATCH = 256
GRAD_CLIP = 1.0
BETAS = (0.9, 0.999)
EPS = 1e-8
WEIGHT_DECAY = 1e-5
WEIGHTS_FORMAT = "BENCH_WEIGHTS_1"


def flat_tail(history) -> int:
    """Number of consecutive final marks that did not improve (the early-stop counter)."""
    n = 0
    for h in reversed(history):
        if h["improved"]:
            break
        n += 1
    return n


def stretched_lr(peak_lr: float, exposures_before: int, n: int, N: int, budget_efe: int) -> float:
    """The S0 schedule stretched to the budget: s0(6 x EFE / budget) at the END of the update (exposure point
    'end')."""
    from ppsi.central import schedule
    return float(peak_lr) * schedule.s_main(6.0 * (exposures_before + n) / (float(budget_efe) * N))


def stretched_lr_table(peak_lr: float, N: int, B: int, budget_efe: int) -> dict:
    import hashlib
    end = budget_efe * N
    lrs, e = [], 0
    while e < end:
        n = min(B, N - e % N)
        lrs.append(stretched_lr(peak_lr, e, n, N, budget_efe))
        e += n
    arr = np.asarray(lrs, dtype="<f8")
    return {"lr": lrs, "n_updates": len(lrs), "min_lr": float(arr.min()), "last_lr": lrs[-1], "final_exposures": e,
            "sha256": hashlib.sha256(arr.tobytes()).hexdigest(), "budget_efe": budget_efe}


def save_weights(adapter, path: Path, *, run_id: str, mark: float, extra: dict | None = None) -> dict:
    from ppsi.fedsim.checkpoint import atomic_save
    payload = {"kind": "weights", "run_id": run_id, "mark": float(mark), "theta": dict(adapter.broadcast_state()),
               **(extra or {})}
    info = atomic_save(payload, Path(path))
    return {"file": Path(path).name, "sha256": sha256_file(path), "bytes": info["bytes"], "mark": float(mark)}


def load_weights(path) -> dict:
    from ppsi.fedsim.checkpoint import load_checkpoint
    return load_checkpoint(Path(path))


def run_central(*, data, arm: str, variant: str, peak_lr: float, dropout, seed: int, rows: np.ndarray,
                select_users: np.ndarray | None, run_dir: Path, run_id: str, smoke_epochs: float | None = None,
                resume: bool = False, say: Callable = print, max_len: int | None = None,
                budget_efe: int = 6, early_stop_marks: int | None = None, wall_cap_hours: float | None = None,
                provenance: dict | None = None, device=None) -> dict:
    import torch

    from ppsi.central import objective, order, schedule
    from ppsi.fedsim.numerics import enforce_strict_fp32, state_digest

    from . import accel
    from .model import build_model
    dev = accel.torch_device(device)                             # cpu (default, untouched) or the GPU class
    cuda = accel.is_cuda(dev)
    enforce_strict_fp32(deterministic=True, warn_only=not cuda)
    drec = accel.device_record(dev)                              # None on CPU: the CPU records stay unchanged
    devf = {"device_class": drec["device_class"], "device": drec} if cuda else {}
    from .common import WallCap, refuse_overwrite
    run_dir = Path(run_dir)
    refuse_overwrite(run_dir)
    cap = WallCap(wall_cap_hours)
    run_dir.mkdir(parents=True, exist_ok=True)
    N = int(rows.size)
    if N < 1:
        raise BenchRefused("no training decisions")
    B = EFFECTIVE_BATCH
    built = build_model(variant, data.K, seed, dropout=dropout, max_len=max_len, device=dev)
    adapter, module = built["adapter"], built["module"]
    shared = adapter.shared_parameters()
    opt = torch.optim.AdamW(adapter.param_groups(WEIGHT_DECAY), lr=peak_lr, betas=BETAS, eps=EPS)
    budget_efe = int(budget_efe)
    if budget_efe not in (6, 24, 96):
        raise BenchRefused("the exposure budget must be 6, 24 or 96 EFE")
    if budget_efe == 6:
        planned = schedule.planned_lr_table(peak_lr, N, B, "MAIN", "end", schedule="S0")
        marks = schedule.eval_marks("MAIN")
    else:                                                        # the S0 schedule stretched to the budget's clock
        planned = stretched_lr_table(peak_lr, N, B, budget_efe)
        marks = [round(k * budget_efe / 12, 6) for k in range(1, 13)]
    mark_exp = [-(-(k * budget_efe * N) // 12) for k in range(1, 13)]
    ep_name = "endpoint_6.0.pt" if budget_efe == 6 else f"endpoint_{budget_efe}.0.pt"
    view = data.validation_view()
    tracker = schedule.StopTracker()
    st = {"pass": 0, "pos": 0, "exposures": 0, "updates": 0, "mark_idx": 0}
    latest = run_dir / "latest.pt"
    ev_dir = run_dir / "evals"
    ev_dir.mkdir(exist_ok=True)
    spec = {"arm": arm, "variant": variant, "peak_lr": float(peak_lr), "dropout": list(dropout), "seed": int(seed),
            "N_decisions": N, "effective_batch": B, "schedule": "S0", "exposure_point": "end", "grad_clip": GRAD_CLIP,
            "betas": list(BETAS), "eps": EPS, "weight_decay": WEIGHT_DECAY, "lr_table_sha256": planned["sha256"],
            "n_updates": planned["n_updates"], "marks": marks, "init_sha256": built["init_sha256"],
            "select_population": "ALL" if select_users is None else "BAND", "max_len": max_len or 50,
            "budget_efe": budget_efe, "early_stop_marks": early_stop_marks, "provenance": provenance}
    if cuda:                                                     # the class is part of the spec (and its hash)
        spec["device_class"] = drec["device_class"]
    if (run_dir / "ARM_SPEC.json").is_file():                   # a run dir never mixes device classes
        accel.check_resume(read_json(run_dir / "ARM_SPEC.json"), dev)
    if resume and latest.exists():
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        if ck["spec_sha256"] != canon_sha256(spec):
            raise BenchRefused("--resume: the run's spec differs from the checkpoint's")
        adapter.load_state_(ck["theta"])
        opt.load_state_dict(ck["opt"])
        st = ck["state"]
        tracker = schedule.StopTracker.from_state(ck["tracker"])
        torch.set_rng_state(ck["rng"])
        if cuda:
            torch.cuda.set_rng_state(ck["cuda_rng"], dev)
    else:
        if latest.exists():
            raise BenchRefused(f"{latest} exists: use --resume")
        write_json(run_dir / "ARM_SPEC.json", dict(spec, utc=utc(), **({"device": drec} if cuda else {})))
        torch.manual_seed(int(seed))
    module.train()
    cap_exposures = None if smoke_epochs is None else round(smoke_epochs * N)
    t0 = time.perf_counter()
    perm_cache = {}
    done = st["mark_idx"] >= len(marks)
    step_wall = []
    while not done:
        if st["pass"] not in perm_cache:
            perm_cache = {st["pass"]: order.pass_permutation(seed, st["pass"], N)}
        perm = perm_cache[st["pass"]]
        a, b = st["pos"], min(N, st["pos"] + B)
        batch = data.batch(rows[perm[a:b]])
        n = b - a
        if cap.hit() and cap_exposures is None:                 # INCOMPLETE, never eligible
            write_json(run_dir / "RESULT.json", {"run_id": run_id, "arm": arm, "kind": "central", "spec": spec,
                                                 "status": "INCOMPLETE_WALL_CAP", "updates_done": st["updates"],
                                                 "mark_idx": st["mark_idx"], "utc": utc(), **devf})
            return {"status": "INCOMPLETE_WALL_CAP", "timing": {}}
        if budget_efe == 6:
            lr = schedule.lr_for_update(peak_lr, st["exposures"], n, N, "MAIN", point="end", schedule="S0")
        else:
            lr = stretched_lr(peak_lr, st["exposures"], n, N, budget_efe)
        if planned["lr"][st["updates"]] != lr:
            raise BenchRefused(f"update {st['updates']}: LR differs from the planned table")
        for g in opt.param_groups:
            g["lr"] = lr
        t1 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss_sum = objective.accumulate_step(adapter.scores, batch, n, 0, dev)
        torch.nn.utils.clip_grad_norm_(shared, GRAD_CLIP)
        opt.step()
        adapter.post_step()
        step_wall.append(time.perf_counter() - t1)
        if not bool(torch.isfinite(loss_sum)):
            raise BenchRefused(f"non-finite loss at update {st['updates']}")
        st["exposures"] += n
        st["updates"] += 1
        st["pos"] = b
        if b == N:
            st["pass"] += 1
            st["pos"] = 0
        if st["updates"] % 200 == 0:
            append_jsonl(run_dir / "metrics.jsonl", {"kind": "train", "update": st["updates"], "efe": st["exposures"] / N,
                                                    "loss": float(loss_sum) / n, "lr": lr, "step_s": step_wall[-1],
                                                    "utc": utc()})
        mark = marks[st["mark_idx"]]
        if st["exposures"] >= mark_exp[st["mark_idx"]]:
            te = time.perf_counter()
            accel.assert_numerics(dev)                           # TF32 / non-strict numerics refused (GPU)
            ranks = evaluate_view(adapter, view)
            summ = summarize(ranks, view, data.band)
            val = float(summ["ALL" if select_users is None else "BAND"][PRIMARY_METRIC])
            improved = tracker.update(mark, val, "MAIN")
            tag = f"{mark:.1f}"
            np.save(ev_dir / f"eval_{tag}.ranks_int32.npy", ranks.astype(np.int32))
            write_json(ev_dir / f"eval_{tag}.json", {"mark": mark, "efe": st["exposures"] / N, "update": st["updates"],
                                                    "select_value": val, "improved": bool(improved), "summary": summ,
                                                    "eval_wall_s": round(time.perf_counter() - te, 2)})
            append_jsonl(run_dir / "metrics.jsonl", {"kind": "eval", "mark": mark, "efe": st["exposures"] / N,
                                                    "update": st["updates"], "select_value": val,
                                                    "improved": bool(improved), **flat_row(summ), "utc": utc()})
            ck_dir = run_dir / "ckpt"
            ck_dir.mkdir(exist_ok=True)
            if improved:
                save_weights(adapter, ck_dir / "best.pt", run_id=run_id, mark=mark)
            st["mark_idx"] += 1
            done = st["mark_idx"] >= len(marks)
            if early_stop_marks and not done and flat_tail(tracker.history) >= early_stop_marks:
                done = True                                      # stop after k marks without improvement
                st["early_stopped_at"] = mark                    # no CONTROLLED_BUDGET row, PRACTICAL_BEST only
            elif done:                                           # the budget endpoint was reached
                save_weights(adapter, ck_dir / ep_name, run_id=run_id, mark=mark)
                st["cb_mark"] = mark
            if not done:
                torch.save({"spec_sha256": canon_sha256(spec), "theta": dict(adapter.broadcast_state()),
                            "opt": opt.state_dict(), "state": st, "tracker": tracker.state(),
                            "rng": torch.get_rng_state(),
                            **({"cuda_rng": torch.cuda.get_rng_state(dev)} if cuda else {})},
                           latest.with_name("latest.pt.tmp"))
                latest.with_name("latest.pt.tmp").replace(latest)
        if cap_exposures is not None and st["exposures"] >= cap_exposures and not done:
            break
    wall = time.perf_counter() - t0
    sw = np.asarray(step_wall)
    timing = {"n_steps_this_leg": int(sw.size), "mean_step_s": float(sw.mean()) if sw.size else None,
              "median_step_s": float(np.median(sw)) if sw.size else None, "wall_s_this_leg": round(wall, 2),
              "n_updates_full_run": planned["n_updates"]}
    if cap_exposures is not None and not done:                 # smoke: stop early, one evaluation
        te = time.perf_counter()
        ranks = evaluate_view(adapter, view)
        summ = summarize(ranks, view, data.band)
        timing["eval_s"] = round(time.perf_counter() - te, 2)
        write_json(run_dir / "SMOKE_SUMMARY.json", {"label": "SMOKE_NOT_EVIDENCE", "arm": arm, "spec": spec,
                                                    "timing": timing, "epochs_run": st["exposures"] / N,
                                                    "summary": summ})
        ck_dir = run_dir / "ckpt"
        ck_dir.mkdir(exist_ok=True)                      # smoke weights let the warm / FT smokes run (never evidence)
        rec = save_weights(adapter, ck_dir / "endpoint_6.0.pt", run_id=run_id, mark=st["exposures"] / N)
        write_json(run_dir / "RESULT.json", {"run_id": run_id, "arm": arm, "kind": "central", "spec": spec,
                                             "status": "SMOKE_DONE_NOT_EVIDENCE", "label": "SMOKE_NOT_EVIDENCE",
                                             "checkpoints": {"CONTROLLED_BUDGET": dict(rec, file="ckpt/endpoint_6.0.pt")},
                                             **devf})
        return {"status": "SMOKE_DONE_NOT_EVIDENCE", "timing": timing}
    latest.unlink(missing_ok=True)
    pb_mark, cb_mark = tracker.main_best_mark, st.get("cb_mark")
    ck_dir = run_dir / "ckpt"
    checkpoints = {"PRACTICAL_BEST": {"file": "ckpt/best.pt", "sha256": sha256_file(ck_dir / "best.pt"),
                                      "mark": pb_mark}}
    rules = [("PRACTICAL_BEST", pb_mark)]
    if cb_mark is not None:                                      # an early-stopped cell has no CONTROLLED_BUDGET (N/A)
        checkpoints["CONTROLLED_BUDGET"] = {"file": f"ckpt/{ep_name}", "sha256": sha256_file(ck_dir / ep_name),
                                            "mark": cb_mark}
        rules.append(("CONTROLLED_BUDGET", cb_mark))
    rows_out = {}
    for rule, mk in rules:
        rows_out[rule] = read_json(ev_dir / f"eval_{mk:.1f}.json")["summary"]
    result = {"run_id": run_id, "arm": arm, "kind": "central", "status": "DONE_MAIN", "spec": spec,
              "checkpoints": checkpoints, "validation": rows_out, "tracker": tracker.state(), "timing": timing,
              "early_stopped_at": st.get("early_stopped_at"),
              "controlled_budget": "N/A (early stopped)" if cb_mark is None else cb_mark,
              "theta_pre_state_digest_endpoint": (state_digest(load_weights(ck_dir / ep_name)["theta"])
                                                  if cb_mark is not None else None),
              "utc": utc(), **devf}
    write_json(run_dir / "RESULT.json", result)
    write_json(run_dir / "STATUS.json", {"run_id": run_id, "status": "DONE_MAIN", "utc": utc(),
                                         **({"device_class": devf["device_class"]} if cuda else {})})
    return {"status": "DONE_MAIN", "timing": timing, "result": result}
