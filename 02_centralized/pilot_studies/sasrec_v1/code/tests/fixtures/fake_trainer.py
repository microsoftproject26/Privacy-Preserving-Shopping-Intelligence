"""fake_trainer.py -- mimics train_sasrec.py's v2 CLI/exit codes/summary.json for
runtime tests (Gate P2: no pair barrier, fixed epochs, cut-rule epoch cap). No GPU,
no real data: this is the fixture the runtime tests drive through supervise_run.sh /
queue_runner.py.

v2 CLI (matches the coordinator's spec; --pair-dir/--pair-partner are GONE, exit code
7 no longer exists):
    fake_trainer.py --run-id X --seed 2026 --run-dir DIR [--resume]
                     --epoch-cap-file PATH
Fixture-only test controls (never part of the real trainer CLI):
  --epoch-seconds F        sleep per epoch (default 0.2)
  --updates-per-epoch N    small U so checkpoints/metrics stay tiny (default 50)
  --planned-epochs N       provisional epoch ceiling before the cut rule resolves,
                           and the fallback if epoch-cap-file never appears (default 6)
  --floor-start-epoch N    first epoch that gets a full-VS eval (default 3, schedule S')
  --epoch-cap-wait-seconds F  bounded poll for epoch-cap-file after epoch 2 (default 10)
  --metrics "a,b,c"        explicit per-eval-epoch primary metric sequence (else synthetic)
  --fail-mode M            none|crash|nonfinite|disk|config|gpu_busy
  --fail-at-epoch N        epoch (1-based, already "completed") at which to fail
  --fail-count N           for --fail-mode crash: crash on launches 0..N-1 (by
                           supervisor_state.json resume_count), succeed from N on

Exit codes (SASREC_INTERFACES.md sec 5, minus 7): 0 COMPLETE, 3 NONFINITE, 4 DISK,
5 CONFIG/ASSERT, 6 GPU_BUSY, other = crash (supervisor resumes).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # code/
import paths  # noqa: E402
import ckpt_io  # noqa: E402


def _read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _write_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    os.replace(tmp, path)


def _append_epoch_record(run_dir: Path, record: dict) -> None:
    p = run_dir / "epoch_records.json"
    data = _read_json(p, default=[])
    if not isinstance(data, list):
        data = []
    data.append(record)
    _write_json_atomic(p, data)


def _perm_sha256(seed: int, epoch: int) -> str:
    """Deterministic per (seed, epoch) -- NOT per run_id/architecture, simulating the
    shared data permutation two anchors of the same seed train over (Fable P2 point 2:
    "shared permutation digests")."""
    return hashlib.sha256(f"perm-{seed}-{epoch}".encode()).hexdigest()


def _deterministic_n_params(run_id: str) -> int:
    h = int(hashlib.sha256(run_id.encode()).hexdigest()[:8], 16)
    return 800_000 + (h % 400_000)


def _wait_for_epoch_cap(cap_file: Path, wait_seconds: float, fallback: int) -> int:
    if cap_file is None:
        return fallback
    t0 = time.time()
    while time.time() - t0 < wait_seconds:
        rec = _read_json(cap_file, default=None)
        if isinstance(rec, dict) and "epoch_cap" in rec:
            return int(rec["epoch_cap"])
        time.sleep(0.2)
    return fallback  # queue_runner didn't decide in time: fall back to the planned ceiling


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--epoch-cap-file", default=None)
    # fixture-only:
    ap.add_argument("--epoch-seconds", type=float, default=0.2)
    ap.add_argument("--updates-per-epoch", type=int, default=50)
    ap.add_argument("--planned-epochs", type=int, default=6)
    ap.add_argument("--floor-start-epoch", type=int, default=3)
    ap.add_argument("--epoch-cap-wait-seconds", type=float, default=10.0)
    ap.add_argument("--metrics", default=None)
    ap.add_argument("--fail-mode", default="none",
                    choices=["none", "crash", "nonfinite", "disk", "config", "gpu_busy"])
    ap.add_argument("--fail-at-epoch", type=int, default=1)
    ap.add_argument("--fail-count", type=int, default=0)
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    ckpt_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    cap_file = Path(args.epoch_cap_file) if args.epoch_cap_file else None

    sup_state = _read_json(run_dir / "supervisor_state.json", default={"resume_count": 0})
    resume_count = int(sup_state.get("resume_count", 0))

    if args.fail_mode == "crash" and resume_count < args.fail_count:
        print(f"fake_trainer: simulated crash (resume_count={resume_count} < {args.fail_count})",
              file=sys.stderr)
        return 137

    U = args.updates_per_epoch
    metrics = [float(x) for x in args.metrics.split(",")] if args.metrics else None

    epoch = 0
    update = 0
    epoch_cap = None
    if args.resume:
        try:
            obj, which = ckpt_io.load_resume(ckpt_dir)
            epoch = int(obj.get("epoch", 0))
            update = int(obj.get("update", 0))
            epoch_cap = obj.get("epoch_cap")
            print(f"fake_trainer: resumed from {which}.pt at epoch={epoch} cap={epoch_cap}",
                  file=sys.stderr)
        except FileNotFoundError:
            pass

    n_params = _deterministic_n_params(args.run_id)
    eval_epochs: list[int] = []
    target = epoch_cap if epoch_cap is not None else args.planned_epochs
    start_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    best_metric = None
    best_epoch = None

    while epoch < target:
        epoch += 1
        update += U

        time.sleep(args.epoch_seconds)

        if args.fail_mode == "nonfinite" and epoch == args.fail_at_epoch:
            _write_summary(run_dir, args, n_params, eval_epochs, "STOPPED_NONFINITE",
                           "nonfinite_loss", epoch, update, start_utc, resume_count,
                           epoch_cap, best_metric, best_epoch)
            return 3
        if args.fail_mode == "disk" and epoch == args.fail_at_epoch:
            _write_summary(run_dir, args, n_params, eval_epochs, "STOPPED_DISK",
                           "disk_floor", epoch, update, start_utc, resume_count,
                           epoch_cap, best_metric, best_epoch)
            return 4
        if args.fail_mode == "config" and epoch == args.fail_at_epoch:
            _write_summary(run_dir, args, n_params, eval_epochs, "STOPPED_CONFIG",
                           "config_assert", epoch, update, start_utc, resume_count,
                           epoch_cap, best_metric, best_epoch)
            return 5
        if args.fail_mode == "gpu_busy" and epoch == args.fail_at_epoch:
            return 6

        full_vs = None
        metric = None
        if epoch >= args.floor_start_epoch:
            idx = len(eval_epochs)
            metric = metrics[idx] if metrics and idx < len(metrics) else (0.05 + 0.01 * epoch)
            full_vs = {"macro_mrr20": metric}
            eval_epochs.append(epoch)
            if best_metric is None or metric > best_metric:
                best_metric = metric
                best_epoch = epoch

        record = {"epoch": epoch, "updates": update, "epoch_seconds": args.epoch_seconds,
                  "eval_seconds": (0.05 if full_vs is not None else None),
                  "full_vs": full_vs, "perm_sha256": _perm_sha256(args.seed, epoch)}
        _append_epoch_record(run_dir, record)

        ckpt_obj = {"epoch": epoch, "update": update, "seed": args.seed,
                    "run_id": args.run_id, "epoch_cap": epoch_cap}
        ckpt_io.save_atomic(ckpt_obj, ckpt_dir, "latest", expected_update=update,
                            required_keys=["epoch", "update"], runs_root=run_dir.parent)

        _write_json_atomic(run_dir / "status.json", {
            "state": "RUNNING", "epoch": epoch, "update": update, "pid": os.getpid(),
            "dps": U / max(args.epoch_seconds, 1e-6), "disk": ckpt_io.free_bytes(run_dir),
        })

        # Cut rule: once epoch 2 is done and no cap is known yet, wait (bounded) for
        # queue_runner.py to write epoch-cap-file, then adopt it as the new ceiling.
        if epoch >= 2 and epoch_cap is None:
            epoch_cap = _wait_for_epoch_cap(cap_file, args.epoch_cap_wait_seconds, args.planned_epochs)
            target = epoch_cap

    final_status = "COMPLETE"
    best_weights_path = ckpt_dir / "best_weights.pt"
    ckpt_io.save_weights({"epoch": best_epoch, "note": "fake weights"}, best_weights_path,
                        runs_root=run_dir.parent)
    _write_summary(run_dir, args, n_params, eval_epochs, final_status, None, epoch, update,
                  start_utc, resume_count, epoch_cap, best_metric, best_epoch)
    return 0


def _write_summary(run_dir: Path, args, n_params: int, eval_epochs: list, final_status: str,
                    stop_reason, epoch: int, update: int, start_utc: str, resume_count: int,
                    epoch_cap, best_metric, best_epoch) -> None:
    epoch_records = _read_json(run_dir / "epoch_records.json", default=[])
    summary = {
        "run_id": args.run_id, "seed": args.seed, "group": "item_only",
        "config": {"run_id": args.run_id}, "registry_sha256": None,
        "final_status": final_status, "stop_reason": stop_reason,
        "epochs_completed": epoch, "updates": update, "exposures": update * 256,
        "best_epoch": best_epoch, "best_metric": best_metric,
        "micro_recall20_at_best": (best_metric * 0.9) if best_metric is not None else None,
        "n_params": n_params,
        "planned_epochs": args.planned_epochs, "epoch_cap": epoch_cap, "eval_epochs": eval_epochs,
        "precision_training": "tf32", "precision_eval": "strict_fp32",
        "epoch_records": epoch_records,
        "checkpoints": {}, "ranks": {}, "code_sha256": {"fake_trainer.py": ckpt_io.sha256_file(__file__)},
        "precision": {"strict_fp32_eval": True, "tf32_training": True}, "start_utc": start_utc,
        "end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "resume_count": resume_count, "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "wall_seconds": None,
    }
    _write_json_atomic(run_dir / "summary.json", summary)


if __name__ == "__main__":
    sys.exit(main())
