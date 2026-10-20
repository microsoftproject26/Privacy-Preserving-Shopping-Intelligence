"""v4train.ckpt_avg — CKPT_AVG of a TUNING run (TRAIN-02 item 5; PREG1_AMENDMENT_02R1 §2-§3).

A TUNING run keeps weights-only snapshots at 5.0 / 5.5 / 6.0 EFE (fp32 state_dicts, checkpoint kinds snapshot_5.0 /
snapshot_5.5 / snapshot_6.0, each sha256-recorded in CHECKPOINTS.json by v4train.checkpoint). At the end of the run the
trainer calls `run_ckpt_avg(trainer)`:
  1. load the three snapshots through the CheckpointStore (file sha256 checked against CHECKPOINTS.json, validated);
  2. build their UNIFORM parameter average: every floating tensor = (s_5.0 + s_5.5 + s_6.0) / 3, accumulated in float64
     in that fixed order and cast back to the snapshot dtype (fp32); a non-floating tensor (the GRU class map buffer)
     must be identical in the three snapshots and is copied; no recentering or other change is applied;
  3. write the average as the weights-only checkpoint kind `ckpt_avg` (sha256 recorded);
  4. load it into the trainer's module and evaluate it through the trainer's OWN `_evaluate(6.0)` — the same frozen
     v4eval path (v4eval.frozen_config.load(), predict_adapter, evaluate) every mark uses; v4eval is not edited;
  5. write the rows, labelled HP_SEARCH (the run's label) with checkpoint_variant CKPT_AVG, into CKPT_AVG_ROWS.json
     (plus evals/ckpt_avg_6.0.result.json / .ranks_int32.npy), and the CKPT_AVG.json record (status EVALUATED: value,
     the endpoint value, the delta, every sha256, the disk peak);
  6. delete the three snapshots and the average through CheckpointStore.delete (each deletion logged with its sha256
     in RETENTION_LOG.jsonl, disk rule), restore the endpoint weights into the module, and mark the record COMPLETE.
TRAIN-02R1 (gate finding F1): a non-finite (or UNDEFINED) CKPT_AVG metric does not fail the run — 02R1 §3 excludes a
run only on a non-finite loss or ENDPOINT metric. The step records outcome CKPT_AVG_NON_FINITE with value None (no rows
are written), still deletes the snapshots and the average (sha256 logged) and the run finishes DONE_TUNING; the P2
driver keeps the valid endpoint, and a missing CKPT_AVG value is never carried.
Idempotent: a record already COMPLETE is returned as is; an EVALUATED record (a crash during the deletions) only
finishes the deletions; otherwise the whole procedure runs again (it is deterministic).
The disk peak reported is the largest total of weight files observed in the run's checkpoints directory (a file being
written counted with the files already there), tracked by the CheckpointStore and the train state since the run start.
"""
from __future__ import annotations

import math
import time
from collections import OrderedDict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from .checkpoint import read_json, write_json_atomic
from .schedule import MAIN_END, SNAPSHOT_MARKS

RECORD = "CKPT_AVG.json"
ROWS = "CKPT_AVG_ROWS.json"
AVG_KIND = "ckpt_avg"
VARIANT = "CKPT_AVG"
OUTCOME_OK = "CKPT_AVG_EVALUATED"
OUTCOME_NON_FINITE = "CKPT_AVG_NON_FINITE"


class CkptAvgError(RuntimeError):
    pass


def snapshot_kind(mark: float) -> str:
    return f"snapshot_{float(mark):.1f}"


def uniform_average(states: Sequence[Mapping[str, torch.Tensor]]) -> "OrderedDict[str, torch.Tensor]":
    """The uniform average of state_dicts with identical keys / shapes / dtypes (float64 accumulation, fixed order)."""
    if len(states) < 2:
        raise CkptAvgError("an average needs at least two snapshots")
    keys = list(states[0].keys())
    for s in states[1:]:
        if list(s.keys()) != keys:
            raise CkptAvgError("the snapshots have different state_dict keys")
    out = OrderedDict()
    for k in keys:
        ts = [s[k] for s in states]
        t0 = ts[0]
        if any(t.shape != t0.shape or t.dtype != t0.dtype for t in ts):
            raise CkptAvgError(f"{k}: shape / dtype differ across snapshots")
        if t0.is_floating_point():
            acc = torch.zeros(t0.shape, dtype=torch.float64)
            for t in ts:
                acc += t.detach().to("cpu", torch.float64)
            out[k] = (acc / len(ts)).to(t0.dtype)
        else:
            if any(not torch.equal(t, t0) for t in ts[1:]):
                raise CkptAvgError(f"{k}: a non-floating tensor differs across snapshots")
            out[k] = t0.detach().clone()
    return out


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def run_ckpt_avg(tr) -> dict:
    """CKPT_AVG of a finished TUNING run held by the CentralTrainer `tr` (see the module docstring)."""
    run_dir = Path(tr.run_dir)
    rp = run_dir / RECORD
    kinds = [snapshot_kind(m) for m in SNAPSHOT_MARKS]
    if rp.exists():
        rec = read_json(rp)
        if rec.get("status") == "COMPLETE":
            return rec
        if rec.get("status") == "EVALUATED":
            return _finish(tr, rec, kinds)
    missing = [k for k in kinds if not tr.ckpt.exists(k)]
    if missing:
        raise CkptAvgError(f"CKPT_AVG needs the snapshots {kinds}; missing {missing}")
    snap_sha = {k: tr.ckpt.index[k]["sha256"] for k in kinds}
    states = [tr.ckpt.load(k)["model"] for k in kinds]
    avg = uniform_average(states)
    endpoint_state = states[-1]
    del states
    avg_sha = tr.ckpt.save(AVG_KIND, {"meta": tr._meta(ckpt_avg_of=kinds, ckpt_avg_snapshot_sha256=snap_sha),
                                      "model": avg})
    peak = tr._disk_peak()
    tr.module.load_state_dict(avg, strict=True)
    why = None
    try:
        out = tr._evaluate(MAIN_END)
        if not _finite(out.value):
            why = f"non-finite CKPT_AVG metric {out.value!r}"
    except Exception as e:                                           # the trainer's UNDEFINED-metric refusal only
        if "primary metric UNDEFINED" not in str(e):
            raise
        out, why = None, f"UNDEFINED CKPT_AVG metric ({e})"
    finally:
        tr.module.load_state_dict(endpoint_state, strict=True)       # the endpoint weights back in the module
    end = [e for e in tr.state.evals if abs(e["mark"] - MAIN_END) < 1e-9]
    end_value = end[-1]["value"] if end else None
    if why is None:
        cid = f"{tr.cfg.run_id}:ckpt_avg({','.join(f'{m:.1f}' for m in SNAPSHOT_MARKS)})"
        rows = [dict(r, checkpoint_variant=VARIANT, checkpoint_id=cid, ckpt_avg_of=list(SNAPSHOT_MARKS))
                for r in out.rows]
        write_json_atomic(run_dir / ROWS, {"run_id": tr.cfg.run_id, "rows": rows,
                                           "note": "HP_SEARCH rows (CKPT_AVG variant); never selection evidence"})
        if out.result is not None:
            write_json_atomic(run_dir / "evals" / "ckpt_avg_6.0.result.json", out.result)
        if out.ranks is not None:
            np.save(run_dir / "evals" / "ckpt_avg_6.0.ranks_int32.npy", np.asarray(out.ranks, dtype=np.int32))
        value, secondary, rows_file, outcome = out.value, out.secondary, ROWS, OUTCOME_OK
    else:
        value, secondary, rows_file, outcome = None, {}, None, OUTCOME_NON_FINITE
    rec = {"task": "TRAIN-02 CKPT_AVG", "run_id": tr.cfg.run_id, "status": "EVALUATED", "outcome": outcome,
           "non_finite_reason": why,
           "snapshots": [{"kind": k, "mark": m, "sha256": snap_sha[k]} for k, m in zip(kinds, SNAPSHOT_MARKS)],
           "average": {"kind": AVG_KIND, "sha256": avg_sha, "rule": "uniform (5.0 + 5.5 + 6.0) / 3, float64 "
                       "accumulation, cast to fp32; non-floating tensors identical and copied"},
           "value": value, "endpoint_value": end_value,
           "delta_vs_endpoint": (value - end_value) if (value is not None and end_value is not None) else None,
           "secondary": {k: (float(v) if _finite(v) else str(v)) for k, v in secondary.items()},
           "rows_file": rows_file, "run_label": tr.run_label, "evaluator": "CentralTrainer._evaluate (frozen v4eval path)",
           "disk_peak_bytes": peak, "utc_evaluated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_json_atomic(rp, rec)
    return _finish(tr, rec, kinds)


def _finish(tr, rec: dict, kinds) -> dict:
    deleted = dict(rec.get("deleted_sha256") or {})
    for k in list(kinds) + [AVG_KIND]:
        d = tr.ckpt.delete(k, "TRAIN-02 CKPT_AVG: snapshots and average deleted after the evaluation (disk rule)")
        if d:
            deleted[k] = d
    left = [f for f in tr.ckpt.weight_files() if any(f.startswith(k) for k in list(kinds) + [AVG_KIND])]
    if left:
        raise CkptAvgError(f"CKPT_AVG weight files left behind: {left}")
    rec = dict(rec, status="COMPLETE", deleted_sha256=deleted, weight_files_left=tr.ckpt.weight_files(),
               utc_complete=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    write_json_atomic(Path(tr.run_dir) / RECORD, rec)
    return rec


__all__ = ["RECORD", "ROWS", "AVG_KIND", "VARIANT", "OUTCOME_OK", "OUTCOME_NON_FINITE", "CkptAvgError", "snapshot_kind",
           "uniform_average", "run_ckpt_avg"]
