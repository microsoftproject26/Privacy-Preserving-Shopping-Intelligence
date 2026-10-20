"""Cooldown continuation: ONE extra epoch (epoch 5) with a decaying LR (owner: Opus).

Owner request 2026-09-25 ("5 epochs"; best accuracy), designed by Fable
(review/deep/FABLE_HIGH_RECALL_RESPONSE.md, Q1): a 5th epoch at the constant floor LR is
NO-GO, so epoch 5 is a cooldown instead:
    lr(u) = 1e-6 + (1e-4 - 1e-6) * (1 - sqrt((u - 4U) / U)),   4U < u <= 5U
It continues from the source run's latest.pt (end of epoch 4, full optimizer + RNG state),
uses the epoch-5 permutation default_rng(seed + 4), evaluates full VALIDATION_SELECT at the
end (strict FP32), and keeps best = raw max over the evaluated epochs {3, 4, 5}.

Everything else is the approved v3 trainer, inherited unchanged (train_sasrec.Trainer):
data, model, objective, recentering, precision, checkpoints, exact resume, evaluation.
It writes to a NEW run dir `<source>_cool5`. The source run is never modified: this script's
checkpoint writes use a reclaim root outside $S/runs (ckpt_io.reclaim can never delete any run's
checkpoints on its behalf), and setup refuses to start unless free disk >= 10 GiB + 3 x the source
latest.pt size (Fable Gate C cooldown review, defect 1).
Disclosed as a post-hoc, owner-requested amendment (validation was seen at epochs 3-4).

CLI (supervise_run.sh compatible via SASREC_TRAINER):
  python continue_cooldown.py --source-run-dir DIR --run-id A1_RICH256 --seed 2026 --run-dir DIR_cool5 [--resume]
Refuses to run on real data unless review/GATE_C_COOLDOWN_PASS.json approves this file's sha256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import paths  # noqa: F401

import numpy as np
import torch

import train_sasrec as ts
from train_sasrec import ConfigError

COOL_EPOCH = 5
LR_END = 1e-6
COOL_PASS = paths.S / "review" / "GATE_C_COOLDOWN_PASS.json"
NO_RECLAIM_ROOT = paths.S / "runs_cooldown_no_reclaim"      # never created: reclaim() is a no-op there
SOURCE_CONFIG_KEYS = ("registry_entry", "schedule", "label_smoothing", "microbatch", "n_train", "channels",
                      "precision_training", "dataset_version_hash")


def self_sha() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def cool_lr(u: int, U: int, lr_start: float, lr_end: float = LR_END) -> float:
    """(1 - sqrt) cooldown over epoch 5; continuous with the floor at u = 4U."""
    t = (u - (COOL_EPOCH - 1) * U) / U
    if not (0.0 < t <= 1.0):
        raise ValueError(f"cooldown lr requested outside epoch {COOL_EPOCH}: u={u}")
    return lr_end + (lr_start - lr_end) * (1.0 - math.sqrt(t))


class CooldownTrainer(ts.Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.source_dir = Path(args.source_run_dir).resolve()
        if self.run_dir.name != self.source_dir.name + "_cool5" or self.run_dir.parent != self.source_dir.parent:
            raise ConfigError("cooldown run dir must be <source>_cool5 next to the source run")
        self.lr_start = self.floor_mult * self.peak_lr     # the v3 floor, 1e-4

    # ---------------------------------------------------------------- setup
    def setup(self) -> None:
        if not self.args.test_tiny:
            if not COOL_PASS.exists():
                raise ConfigError(f"cooldown firewall: {COOL_PASS} missing (Fable review required)")
            gate = json.loads(COOL_PASS.read_text(encoding="utf-8"))
            if gate.get("verdict") != "APPROVE_FOR_REAL_TRAINING" or gate.get("continue_cooldown_sha256") != self_sha():
                raise ConfigError("cooldown firewall: continue_cooldown.py not approved at this hash")
        super().setup()
        import ckpt_io
        src_latest = self.source_dir / "checkpoints" / "latest.pt"
        if not self.args.resume:
            if not src_latest.exists():
                raise ConfigError(f"source checkpoint {src_latest} missing")
            need = ckpt_io.TARGET_FREE_BYTES + 3 * src_latest.stat().st_size
            free = ckpt_io.free_bytes(self.run_dir.parent)
            if free < need:
                raise ConfigError(f"disk precheck: free {free / 2**30:.2f} GiB < {need / 2**30:.2f} GiB required")
        self.config["cooldown"] = {"epoch": COOL_EPOCH, "lr_start": self.lr_start, "lr_end": LR_END,
                                   "shape": "1 - sqrt(t)", "source_run_dir": str(self.source_dir),
                                   "continue_cooldown_sha256": self_sha(),
                                   "gate_c_cooldown_pass_sha256": ts.sha256_path(COOL_PASS),
                                   "design": "review/deep/FABLE_HIGH_RECALL_RESPONSE.md Q1",
                                   "disclosure": "post-hoc owner-requested amendment after VS epochs 3-4 were seen"}

    def effective_epochs(self, epoch_done: int) -> int:
        return COOL_EPOCH

    def eta(self):
        saved = self.planned_epochs
        self.planned_epochs = COOL_EPOCH
        try:
            return super().eta()
        finally:
            self.planned_epochs = saved

    # checkpoint writes: same atomic/validated writers, but with a reclaim root outside $S/runs
    def save_latest(self) -> None:
        import ckpt_io
        ts.precision.assert_training_precision(self.tf32)
        ckpt_io.save_atomic(self.ckpt_obj(), self.run_dir / "checkpoints", "latest",
                            expected_update=self.u, required_keys=ts.REQUIRED_CKPT_KEYS,
                            runs_root=NO_RECLAIM_ROOT, disk_log_path=paths.S / "logs" / "disk_reclaim.log")

    def save_best(self, epoch: int, metric: float) -> None:
        import ckpt_io
        path = self.run_dir / "checkpoints" / "best_weights.pt"
        ckpt_io.save_weights({"state_dict": self.model.state_dict(),
                              "meta": {"run_id": self.args.run_id, "seed": self.seed, "epoch": epoch, "update": self.u,
                                       "metric": metric, "config": self.config, "code_sha256": self.code_map,
                                       "registry_sha256": self.reg_sha}},
                             path, runs_root=NO_RECLAIM_ROOT, disk_log_path=paths.S / "logs" / "disk_reclaim.log")
        self.best_ckpt = {"path": str(path), "sha256": ts.sha256_path(path), "epoch": epoch}

    # ---------------------------------------------------------------- state
    def init_state(self) -> None:
        if self.args.resume:                       # resume THIS cooldown run's own checkpoint
            super().init_state()
            if self.best_ckpt is None:             # best still the source-run epoch (<= 4)
                src_best = self.source_dir / "checkpoints" / "best_weights.pt"
                self.best_ckpt = {"path": str(src_best), "sha256": ts.sha256_path(src_best),
                                  "epoch": self.best["epoch"], "note": "source-run best (epochs 1-4)"}
            return
        ck_dir = self.run_dir / "checkpoints"
        if ck_dir.exists() and any(ck_dir.glob("*.pt")):
            raise ConfigError(f"{ck_dir} already holds checkpoints; use --resume")
        src_summary = json.loads((self.source_dir / "summary.json").read_text(encoding="utf-8"))
        if src_summary.get("final_status") != "COMPLETE" or src_summary.get("epochs_completed") != COOL_EPOCH - 1:
            raise ConfigError("source run must be COMPLETE after exactly 4 epochs")
        obj = torch.load(self.source_dir / "checkpoints" / "latest.pt", map_location="cpu", weights_only=False)
        if obj["registry_sha256"] != self.reg_sha:
            raise ConfigError("source checkpoint was trained under a different registry")
        if ts.training_code_map(obj["code_sha256"]) != ts.training_code_map(self.code_map) and not self.args.test_tiny:
            raise ConfigError("training code drift since the source checkpoint")
        if int(obj["update"]) != (COOL_EPOCH - 1) * self.U or int(obj["epoch_idx0"]) != COOL_EPOCH - 2 \
                or int(obj["offset_in_epoch"]) != self.n or bool(obj["eval_pending"]):
            raise ConfigError("source latest.pt is not the completed end of epoch 4")
        for key in SOURCE_CONFIG_KEYS:
            a = json.dumps(obj["config"].get(key), sort_keys=True, default=str)
            b = json.dumps(self.config.get(key), sort_keys=True, default=str)
            if a != b:
                raise ConfigError(f"source run config field {key} differs from this run")
        if obj["dataset_version_hash"] != self.store.version_hash:
            raise ConfigError("source dataset hash differs")
        self.model.load_state_dict(obj["model_state_dict"])
        self.opt.load_state_dict(obj["optimizer_state_dict"])
        ts.rng_restore(obj["rng"])
        self.resume_count = 0
        self.u = (COOL_EPOCH - 1) * self.U
        self.epoch_idx0 = COOL_EPOCH - 1
        self.offset = 0
        self.cov = ts.Coverage(self.n)
        self.perm = ts.epoch_permutation(self.seed, self.epoch_idx0, self.n)
        self.perm_sha = ts.perm_digest(self.perm)
        self.epoch_records = [dict(r, inherited_from_source=True) for r in obj["epoch_records"]]
        self.best = dict(obj["best"])
        src_best = self.source_dir / "checkpoints" / "best_weights.pt"
        self.best_ckpt = {"path": str(src_best), "sha256": ts.sha256_path(src_best), "epoch": self.best["epoch"],
                          "note": "source-run best (epochs 1-4); replaced only if epoch 5 improves"}
        self.eval_pending = False
        self.cap_applied = None
        self.acc = self.fresh_acc()
        (self.run_dir / "logs").mkdir(parents=True, exist_ok=True)
        ck_dir.mkdir(parents=True, exist_ok=True)
        ts.atomic_json(self.run_dir / "config.json", self.config)
        ts.atomic_json(self.run_dir / "start.json", {"start_utc": ts.utc(), "source_latest_update": int(obj["update"])})
        self.persist_records()
        self.log(f"cooldown continuation of {self.source_dir.name}: u={self.u} epoch {COOL_EPOCH} "
                 f"lr {self.lr_start:g}->{LR_END:g} (1-sqrt), best so far epoch {self.best['epoch']} = {self.best['metric']}")
        self.metric({"type": "epoch_start", "epoch": COOL_EPOCH, "perm_sha256": self.perm_sha, "n_train": self.n,
                     "cooldown": True})

    # ---------------------------------------------------------------- update
    def train_step(self, batch_cpu, positions):
        """Parent step with the cooldown LR: the schedule module's lr_at is swapped for this
        one call only (it is read nowhere else during the step)."""
        orig = ts.sched.lr_at
        U, start = self.U, self.lr_start

        def lr_at(u, *a, **k):
            return cool_lr(u, U, start) if u > (COOL_EPOCH - 1) * U else orig(u, *a, **k)

        ts.sched.lr_at = lr_at
        try:
            return super().train_step(batch_cpu, positions)
        finally:
            ts.sched.lr_at = orig

    def finalize(self, final_status: str) -> None:
        if final_status == "COMPLETE":
            import ckpt_io
            out = self.run_dir / "checkpoints" / f"epoch{COOL_EPOCH}_weights.pt"
            ckpt_io.save_weights({"state_dict": self.model.state_dict(),
                                  "meta": {"run": self.run_dir.name, "epoch": COOL_EPOCH, "update": self.u}},
                                 out, runs_root=NO_RECLAIM_ROOT)
        super().finalize(final_status)
        p = self.run_dir / "summary.json"
        s = json.loads(p.read_text(encoding="utf-8"))
        w5 = self.run_dir / "checkpoints" / f"epoch{COOL_EPOCH}_weights.pt"
        s["epoch5_weights"] = {"path": str(w5), "sha256": ts.sha256_path(w5)} if w5.exists() else None
        s["cooldown"] = self.config["cooldown"]
        s["source_run_dir"] = str(self.source_dir)
        ts.atomic_json(p, s)


def main(argv=None) -> int:
    ap = ts.build_parser()
    ap.add_argument("--source-run-dir", required=True)
    args = ap.parse_args(argv)
    args.thread_config = ts.configure_threads()
    try:
        test_only = [args.tiny_planned_epochs is not None, args.stop_after_updates is not None,
                     args.log_every_update, args.allow_any_tiny_dir, args.device == "cpu"]
        if any(test_only) and not args.test_tiny:
            raise ConfigError("test-only options require --test-tiny")
        if args.benchmark_updates:
            raise ConfigError("no benchmark mode in the cooldown script")
        if not args.run_dir:
            raise ConfigError("--run-dir is required")
        return CooldownTrainer(args).run()
    except ConfigError as e:
        print(f"CONFIG ERROR: {e}", file=sys.stderr, flush=True)
        return ts.EXIT_CONFIG
    except ts.GpuBusy as e:
        print(f"GPU BUSY: {e}", file=sys.stderr, flush=True)
        return ts.EXIT_GPU_BUSY


if __name__ == "__main__":
    raise SystemExit(main())
