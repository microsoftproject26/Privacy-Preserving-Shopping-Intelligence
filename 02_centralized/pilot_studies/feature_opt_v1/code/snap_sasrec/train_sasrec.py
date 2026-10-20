"""SASRec-CE campaign trainer (owner: Opus). Registry v3: v2 schedule mechanics + the Fable
deep-review recipe fix (untied output table + per-step common-mode recentering, input LN,
no sqrt(d) scale, AdamW betas (0.9, 0.999), peak 1e-3, 4 fixed epochs).

Implements 03 sections 6-7 under the owner's 38 h deadline (Fable Gate P2) for ONE run
(A0_ID256 or A1_RICH256, seed 2026): full frozen TRAIN loss population, one exact
full-catalogue CE term per decision, effective batch 256 (epoch tail 128 counted),
schedule S_prime (warmup 2000, peak through epoch 1, cosine to 0.1x across epoch 2, floor
LR from epoch 3), a FIXED number of epochs (planned 6 = four full low-LR epochs, or the
one-time systems-only deadline cap, identical for both anchors), full VALIDATION_SELECT
evaluation at floor epochs only, best = raw max over evaluated epochs, exact resume.

Precision (owner-approved, parity-gated): training steps TF32 matmul; every evaluation
strict FP32 inside precision.strict_eval(). If bench/TF32_PARITY.json says the parity gate
failed, the registered fallback applies automatically: strict FP32 training, 5 epochs.

CLI:
  python train_sasrec.py --run-id A0_ID256 --seed 2026 --run-dir DIR [--resume]
      [--epoch-cap-file $S/state/epoch_cap.json] [--microbatch 256|128]
      [--ckpt-every-updates 10000] [--diag-every-updates 40000]
      [--benchmark-updates N --precision-mode fp32|tf32 [--bench-parity --bench-loss-log F]]
      [--test-tiny ...]

Exit codes: 0 COMPLETE, 3 NONFINITE, 4 DISK, 5 CONFIG/ASSERT, 6 GPU_BUSY.
Anything else is a crash (the supervisor resumes).

Firewall: production training and the real-data benchmark refuse to start unless
review/GATE_C_PASS.json approves the exact current training-code hashes (Fable Gate C).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import paths  # noqa: F401  (sys.path + thread env)

import numpy as np
import torch

import precision
import sasrec_schedule as sched
from sasrec_model import SASRecConfig, SASRecCE, parameter_inventory
from sasrec_objective import ce_sum

EXIT_OK, EXIT_NONFINITE, EXIT_DISK, EXIT_CONFIG, EXIT_GPU_BUSY = 0, 3, 4, 5, 6
EFFECTIVE_BATCH = 256
BETAS, EPS, WEIGHT_DECAY, GRAD_CLIP = (0.9, 0.999), 1e-8, 1e-5, 1.0   # v3: beta2 0.999 (Fable deep review)
EVAL_BATCH, EVAL_KS = 1024, (1, 2, 5, 10, 20)
HEARTBEAT_S = 60.0
FALLBACK_EPOCHS_FP32 = 4
GATE_C_FILE = paths.S / "review" / "GATE_C_PASS.json"
PARITY_FILE = paths.S / "bench" / "TF32_PARITY.json"
REQUIRED_CKPT_KEYS = ("model_state_dict", "optimizer_state_dict", "rng", "update", "epoch_idx0",
                      "offset_in_epoch", "coverage_packed", "perm_sha256", "best", "epoch_records",
                      "eval_pending", "cap_applied", "epoch_acc", "config", "code_sha256",
                      "registry_sha256", "precision", "dataset_version_hash", "resume_count", "saved_utc")
REVIEWED_CODE = ("train_sasrec.py", "sasrec_model.py", "sasrec_objective.py", "sasrec_schedule.py",
                 "precision.py", "paths.py", "sasrec_data.py", "ckpt_io.py", "synthetic_store.py")


class ConfigError(Exception):
    pass


class NonFiniteStop(Exception):
    pass


class GpuBusy(Exception):
    pass


# --------------------------------------------------------------------------- utils
def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_path(p: Path) -> Optional[str]:
    p = Path(p)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def atomic_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=2, allow_nan=False, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def code_sha256_map() -> dict:
    out = {}
    for p in sorted(paths.CODE.glob("*.py")):
        out[p.name] = sha256_path(p)
    for p in sorted((paths.REUSE).rglob("*.py")):
        out["reuse/" + p.relative_to(paths.REUSE).as_posix()] = sha256_path(p)
    return out


def training_code_map(code_map: dict) -> dict:
    """The code that determines training/eval semantics: reviewed files + the reused modules."""
    return {k: v for k, v in code_map.items() if k in REVIEWED_CODE or k.startswith("reuse/")}


def load_registry() -> tuple[dict, str]:
    raw = paths.REGISTRY_LOCKED.read_bytes()
    return json.loads(raw), sha256_bytes(raw)


def registry_entry(reg: dict, run_id: str) -> dict:
    for m in reg["models"]:
        if m["id"] == run_id:
            return m
    raise ConfigError(f"run id {run_id!r} is not an active run in the locked registry")


def gate_c_check(code_map: dict) -> dict:
    """Refuse real-data parameter updates unless Fable Gate C approved these exact hashes."""
    if not GATE_C_FILE.exists():
        raise ConfigError(f"Gate C firewall: {GATE_C_FILE} missing; no real-data training before code review")
    gate = json.loads(GATE_C_FILE.read_text(encoding="utf-8"))
    if gate.get("verdict") != "APPROVE_FOR_REAL_TRAINING":
        raise ConfigError(f"Gate C firewall: verdict {gate.get('verdict')!r}")
    reviewed = gate.get("code_sha256", {})
    drift = [f for f in REVIEWED_CODE if reviewed.get(f) != code_map.get(f)]
    drift += [f for f in code_map if f.startswith("reuse/") and reviewed.get(f) != code_map[f]]
    if drift:
        raise ConfigError(f"Gate C firewall: code changed since review: {drift}")
    return {"gate_c_file": str(GATE_C_FILE), "gate_c_sha256": sha256_path(GATE_C_FILE)}


def precision_plan(reg: dict, args) -> dict:
    """Training precision + planned epochs (registry v2 'precision' / 'schedule')."""
    planned = int(reg["schedule"]["planned_epochs"])
    if args.test_tiny:
        tf32 = args.precision_mode == "tf32"
        return {"tf32_training": tf32, "planned_epochs": int(args.tiny_planned_epochs or planned),
                "source": "test_tiny"}
    if args.benchmark_updates:
        if args.precision_mode not in ("fp32", "tf32"):
            raise ConfigError("benchmark needs an explicit --precision-mode fp32|tf32")
        return {"tf32_training": args.precision_mode == "tf32", "planned_epochs": planned, "source": "benchmark"}
    if args.precision_mode != "registry":
        raise ConfigError("production training takes its precision from the registry (--precision-mode registry)")
    if not PARITY_FILE.exists():
        raise ConfigError(f"TF32 parity gate result {PARITY_FILE} missing; run tf32_parity_gate.py first")
    gate = json.loads(PARITY_FILE.read_text(encoding="utf-8"))
    ok = gate.get("pass") is True
    return {"tf32_training": ok, "planned_epochs": planned if ok else FALLBACK_EPOCHS_FP32,
            "source": "registry+parity_gate", "parity_pass": ok, "parity_file_sha256": sha256_path(PARITY_FILE),
            "fallback_applied": not ok}


# ------------------------------------------------------------------------ exposure
def epoch_permutation(seed: int, epoch_idx0: int, n: int) -> np.ndarray:
    """04 'Data order': default_rng(seed + epoch_index).permutation(N), epoch_index 0-based."""
    return np.random.default_rng(int(seed) + int(epoch_idx0)).permutation(int(n))


def perm_digest(perm: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(perm, dtype=np.int64).tobytes())


class Coverage:
    """Exact once-per-epoch exposure bitmap over TRAIN positions (Gate P fix 4)."""

    def __init__(self, n: int, packed: Optional[np.ndarray] = None):
        self.n = int(n)
        self.seen = np.zeros(self.n, dtype=bool) if packed is None else \
            np.unpackbits(np.asarray(packed, dtype=np.uint8), count=self.n).astype(bool)

    def mark(self, positions: np.ndarray) -> None:
        if self.seen[positions].any():
            raise ConfigError("exposure accounting: a TRAIN decision was exposed twice in one epoch")
        if np.unique(positions).shape[0] != positions.shape[0]:
            raise ConfigError("exposure accounting: duplicate positions inside one batch")
        self.seen[positions] = True

    def count(self) -> int:
        return int(self.seen.sum())

    def complete(self) -> bool:
        return bool(self.seen.all())

    def packed(self) -> np.ndarray:
        return np.packbits(self.seen)


def best_update(best: dict, epoch: int, metric: float) -> tuple[dict, bool]:
    """Raw maximum over evaluated epochs; an exact tie keeps the earlier epoch."""
    metric = float(metric)
    if not math.isfinite(metric):
        raise ConfigError("non-finite selection metric")
    improved = best.get("metric") is None or metric > float(best["metric"])
    return ({"metric": metric, "epoch": int(epoch)} if improved else dict(best)), improved


# ---------------------------------------------------------------------------- data
def build_store(channels: str, args) -> Any:
    if args.test_tiny:
        from synthetic_store import SyntheticStore
        return SyntheticStore.build(channels=channels, n_train=args.tiny_n_train, n_vs=args.tiny_n_vs,
                                    K=args.tiny_k, seed=args.tiny_data_seed, task=args.tiny_task)
    import sasrec_data
    return sasrec_data.build_training_store(channels)


def model_inputs(batch: dict, model: SASRecCE) -> dict:
    """Allowlisted predictor channels (sasrec_data) + the store positions for the model's check."""
    import sasrec_data
    kw = sasrec_data.predictor_kwargs(batch, model)
    kw["position_ids"] = batch["position_ids"]
    return kw


def load_cohort(ds_vs, args):
    if args.test_tiny:
        return ds_vs.take(np.arange(min(len(ds_vs), 256)))
    import pyarrow.parquet as pq
    ids = pq.read_table(paths.COHORT_PARQUET, columns=["decision_id"])["decision_id"].to_numpy()
    ids = np.ascontiguousarray(ids).astype(np.int64)
    if ids.shape[0] != paths.N_COHORT:
        raise ConfigError(f"diagnostic cohort size {ids.shape[0]} != {paths.N_COHORT}")
    pos = np.searchsorted(ds_vs.decision_id, ids)
    if not bool((ds_vs.decision_id[np.clip(pos, 0, len(ds_vs) - 1)] == ids).all()):
        raise ConfigError("diagnostic cohort decision missing from VALIDATION_SELECT")
    return ds_vs.take(pos)


def launch_asserts(store, ds_train, ds_vs, args) -> None:
    if ds_train.split != "TRAIN" or ds_train.kind != "loss":
        raise ConfigError("train decision set must be TRAIN/loss")
    if ds_vs.split != "VALIDATION_SELECT" or ds_vs.kind != "eval":
        raise ConfigError("selection decision set must be VALIDATION_SELECT/eval")
    if tuple(getattr(store, "splits", ())) != ("TRAIN", "VALIDATION_SELECT"):
        raise ConfigError(f"training store must hold exactly TRAIN+VALIDATION_SELECT, got {getattr(store, 'splits', None)}")
    dec = np.asarray(ds_train.decision_id)
    if not bool(np.asarray(ds_train.loss_mask).all()):
        raise ConfigError("TRAIN set contains rows without loss_mask")
    if dec.shape[0] > 1 and not bool((np.diff(dec) > 0).all()):
        raise ConfigError("TRAIN decision ids not strictly ascending")
    if not bool((np.asarray(ds_train.target_class) >= 0).all()):
        raise ConfigError("a TRAIN loss decision has no known target (one observed target per decision required)")
    vs_ids = np.asarray(ds_vs.decision_id)
    hit = np.clip(np.searchsorted(dec, vs_ids), 0, max(0, dec.shape[0] - 1))
    if dec.shape[0] and bool((dec[hit] == vs_ids).any()):
        raise ConfigError("leakage: a VALIDATION_SELECT decision id is in TRAIN")
    if not args.test_tiny:
        if len(ds_train) != paths.N_TRAIN or len(ds_vs) != paths.N_VS:
            raise ConfigError(f"population sizes {len(ds_train)}/{len(ds_vs)} != frozen {paths.N_TRAIN}/{paths.N_VS}")
        if int(np.unique(np.asarray(ds_vs.eval_user_key)).shape[0]) != paths.N_VS_USERS:
            raise ConfigError("VALIDATION_SELECT user count differs from the frozen value")
        if store.version_hash != paths.VERSION_HASH:
            raise ConfigError("dataset version hash mismatch")


# --------------------------------------------------------------------------- model
def build_model(cfg: SASRecConfig, seed: int, store, device, args) -> SASRecCE:
    if args.test_tiny:
        model = SASRecCE(cfg, vocab_sizes=store.vocab_sizes, product_idx_of_class=store.product_idx_of_class,
                         seed=seed, impl="unpadded")
    else:
        import train as reuse_train
        model = SASRecCE(cfg, vocab_sizes=dict(reuse_train.VOCAB_SIZES),
                         product_idx_of_class=reuse_train.load_product_idx_of_class(),
                         seed=seed, impl="unpadded")
    return model.to(device)


def build_optimizer(model: SASRecCE, fused: bool) -> torch.optim.AdamW:
    """Matrices (ndim>=2, incl. every embedding) decay 1e-5; biases, LayerNorm and gate
    scalars do not. The tied item table is ONE Parameter, so it is updated once."""
    decay, no_decay, seen = [], [], set()
    for _, p in model.named_parameters():
        if id(p) in seen:
            raise ConfigError("a Parameter appears twice (tying must not duplicate)")
        seen.add(id(p))
        (decay if p.ndim >= 2 else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": WEIGHT_DECAY}, {"params": no_decay, "weight_decay": 0.0}]
    kw = dict(lr=0.0, betas=BETAS, eps=EPS)
    if fused:
        kw["fused"] = True
    else:
        kw["foreach"] = True
    return torch.optim.AdamW(groups, **kw)


def set_lr(opt, lr: float) -> None:
    for g in opt.param_groups:
        g["lr"] = lr


def rng_snapshot() -> dict:
    s = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def rng_restore(s: dict) -> None:
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if torch.cuda.is_available() and s.get("cuda") is not None:
        torch.cuda.set_rng_state_all(s["cuda"])


# ---------------------------------------------------------------------- evaluation
def evaluate_full(model, store, ds, device, run_dir: Path, epoch: int, *, save_ranks: bool) -> dict:
    """Full exact evaluation (reused audited evaluator), ALWAYS strict FP32."""
    from eval_full import aggregate, evaluate_model
    t0 = time.time()
    with precision.strict_eval():
        res = evaluate_model(model, store, ds, device, eval_batch=EVAL_BATCH, k=20, return_ranks=True)
        prec = precision.precision_record()
    ranks = res.pop("ranks")
    out = {"n": res["end_to_end"]["n"], "users": res["user_count"],
           "macro_mrr20": float(res["end_to_end"]["macro"]["mrr@20"]),
           "micro_mrr20": float(res["end_to_end"]["micro"]["mrr@20"]),
           "macro_recall20": float(res["end_to_end"]["macro"]["recall@20"]),
           "micro_recall20": float(res["end_to_end"]["micro"]["recall@20"]),
           "macro_ndcg20": float(res["end_to_end"]["macro"]["ndcg@20"]),
           "micro_ndcg20": float(res["end_to_end"]["micro"]["ndcg@20"]),
           "at_k": {}, "closed_at_k": {}, "eval_precision": prec}
    for k in EVAL_KS:
        a = res if k == 20 else aggregate(ranks, ds, k=k)
        out["at_k"][str(k)] = {"micro": {m: float(a["end_to_end"]["micro"][f"{m}@{k}"]) for m in ("recall", "mrr", "ndcg")},
                               "macro": {m: float(a["end_to_end"]["macro"][f"{m}@{k}"]) for m in ("recall", "mrr", "ndcg")}}
        out["closed_at_k"][str(k)] = {"n": a["closed"]["n"],
                                      "micro": {m: float(a["closed"]["micro"][f"{m}@{k}"]) for m in ("recall", "mrr", "ndcg")},
                                      "macro": {m: float(a["closed"]["macro"][f"{m}@{k}"]) for m in ("recall", "mrr", "ndcg")}}
    out["cohorts"] = {c: {"n": res["cohorts"][c]["end_to_end"]["n"],
                          "macro_mrr20": float(res["cohorts"][c]["end_to_end"]["macro"]["mrr@20"]),
                          "micro_recall20": float(res["cohorts"][c]["end_to_end"]["micro"]["recall@20"])}
                      for c in ("warm", "unseen")}
    out["eval_seconds"] = time.time() - t0
    out["rows_per_s"] = res.get("_rows_per_s")
    out["peak_gpu_mem_bytes"] = res.get("_peak_gpu_mem_bytes")
    if save_ranks:
        rd = run_dir / "ranks"
        rd.mkdir(parents=True, exist_ok=True)
        r32 = np.asarray(ranks)
        if r32.max(initial=-1) >= 2**31 - 1:
            raise ConfigError("rank exceeds int32")
        tmp = rd / f"epoch{epoch}.tmp{os.getpid()}.npz"
        np.savez_compressed(tmp, ranks=r32.astype(np.int32),
                            decision_id_sha256=np.frombuffer(bytes.fromhex(ds.sha256()), dtype=np.uint8))
        final = rd / f"epoch{epoch}.npz"
        os.replace(tmp, final)
        out["ranks_path"] = str(final)
        out["ranks_sha256"] = sha256_path(final)
    return out


# -------------------------------------------------------------------------- runner
class Trainer:
    def __init__(self, args):
        self.args = args
        self.run_dir = Path(args.run_dir).resolve()
        self.reg, self.reg_sha = load_registry()
        if int(self.reg.get("version", 1)) != 3:
            raise ConfigError("this trainer implements registry v3")
        self.entry = registry_entry(self.reg, args.run_id)
        self.cfg = SASRecConfig.from_registry(self.entry)
        self.seed = int(args.seed)
        self.peak_lr = float(self.entry["peak_lr"])
        self.smoothing = float(self.entry["label_smoothing"])
        sc = self.reg["schedule"]
        self.warmup = int(sc["warmup_updates"]) if not args.test_tiny else int(args.tiny_warmup)
        self.peak_epochs, self.cosine_epochs = int(sc["peak_epochs"]), int(sc["cosine_epochs"])
        self.floor_mult = float(sc["floor_multiplier"])
        self.floor_start = sched.floor_start_epoch(self.peak_epochs, self.cosine_epochs)
        if self.floor_start != int(sc["floor_start_epoch"]):
            raise ConfigError("registry floor_start_epoch inconsistent with peak/cosine epochs")
        self.min_epochs_cut = int(sc["min_epochs_under_cut"])
        self.plan = precision_plan(self.reg, args)
        self.planned_epochs = int(self.plan["planned_epochs"])
        self.tf32 = bool(self.plan["tf32_training"])
        if args.microbatch not in (256, 128):
            raise ConfigError("microbatch must be 256 or 128 (effective batch fixed at 256)")
        self.channels = "F4" if self.cfg.rich else "F0"

    # ----------------------------------------------------------------- logging
    def log(self, msg: str) -> None:
        line = f"[{utc()}] {msg}"
        print(line, flush=True)
        p = self.run_dir / "logs" / "train.log"
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(line + "\n")

    def metric(self, rec: dict) -> None:
        with (self.run_dir / "metrics.jsonl").open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(rec, allow_nan=False, default=str) + "\n")

    def status(self, **kw) -> None:
        kw.update({"run_id": self.args.run_id, "seed": self.seed, "pid": os.getpid(), "utc": utc()})
        atomic_json(self.run_dir / "status.json", kw)

    # -------------------------------------------------------------- checkpoints
    def ckpt_obj(self) -> dict:
        return {"model_state_dict": self.model.state_dict(), "optimizer_state_dict": self.opt.state_dict(),
                "rng": rng_snapshot(), "update": self.u, "epoch_idx0": self.epoch_idx0,
                "offset_in_epoch": self.offset, "coverage_packed": self.cov.packed(),
                "perm_sha256": self.perm_sha, "best": self.best, "epoch_records": self.epoch_records,
                "eval_pending": self.eval_pending, "cap_applied": self.cap_applied, "epoch_acc": self.acc,
                "config": self.config, "code_sha256": self.code_map, "registry_sha256": self.reg_sha,
                "precision": precision.precision_record(), "dataset_version_hash": self.store.version_hash,
                "resume_count": self.resume_count, "saved_utc": utc()}

    def save_latest(self) -> None:
        import ckpt_io
        precision.assert_training_precision(self.tf32)
        ckpt_io.save_atomic(self.ckpt_obj(), self.run_dir / "checkpoints", "latest",
                            expected_update=self.u, required_keys=REQUIRED_CKPT_KEYS,
                            runs_root=paths.S / "runs", disk_log_path=paths.S / "logs" / "disk_reclaim.log")

    def save_best(self, epoch: int, metric: float) -> None:
        import ckpt_io
        path = self.run_dir / "checkpoints" / "best_weights.pt"
        ckpt_io.save_weights({"state_dict": self.model.state_dict(),
                              "meta": {"run_id": self.args.run_id, "seed": self.seed, "epoch": epoch, "update": self.u,
                                       "metric": metric, "config": self.config, "code_sha256": self.code_map,
                                       "registry_sha256": self.reg_sha}},
                             path, runs_root=paths.S / "runs", disk_log_path=paths.S / "logs" / "disk_reclaim.log")
        self.best_ckpt = {"path": str(path), "sha256": sha256_path(path), "epoch": epoch}

    # ------------------------------------------------------------------ setup
    def setup(self) -> None:
        a = self.args
        self.code_map = code_sha256_map()
        if not a.test_tiny:
            self.gate = gate_c_check(self.code_map)
            if not str(self.run_dir).startswith(str(paths.S / "runs")):
                raise ConfigError("production run dirs must live under $S/runs")
        else:
            self.gate = {"test_tiny": True}
            if not str(self.run_dir).startswith(str(paths.S / "tests")) and not a.allow_any_tiny_dir:
                raise ConfigError("--test-tiny run dirs must live under $S/tests")
        precision.configure_strict_fp32(self.seed)
        canary = None
        if a.device == "cuda":
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise ConfigError("exactly one visible CUDA device required")
            self.device = torch.device("cuda:0")
            self.physical_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
            self.check_gpu_free()
            strict_c = precision.tf32_canary(self.device)
            if not strict_c["looks_strict_fp32"]:
                raise ConfigError(f"strict-FP32 canary failed: {strict_c}")
            canary = {"strict": strict_c}
        else:
            if not a.test_tiny:
                raise ConfigError("CPU device allowed only with --test-tiny")
            self.device = torch.device("cpu")
            self.physical_gpu = "cpu"
        prec_train = precision.training_precision(self.tf32)
        if canary is not None:
            tc = precision.tf32_canary(self.device)
            if self.tf32 and tc["looks_strict_fp32"]:
                raise ConfigError(f"TF32 requested but the canary still looks strict: {tc}")
            if not self.tf32 and not tc["looks_strict_fp32"]:
                raise ConfigError(f"strict FP32 training requested but TF32 detected: {tc}")
            canary["training_phase"] = tc
            with precision.strict_eval():
                ec = precision.tf32_canary(self.device)
            if not ec["looks_strict_fp32"]:
                raise ConfigError(f"strict_eval canary failed: {ec}")
            canary["eval_phase"] = ec
            precision.assert_training_precision(self.tf32)

        self.store = build_store(self.channels, a)
        self.ds_train = self.store.decisions("TRAIN", "loss")
        self.ds_vs = self.store.decisions("VALIDATION_SELECT", "eval")
        launch_asserts(self.store, self.ds_train, self.ds_vs, a)
        self.ds_cohort = load_cohort(self.ds_vs, a)
        self.n = len(self.ds_train)
        self.U = sched.updates_per_epoch(self.n, EFFECTIVE_BATCH)
        if not a.test_tiny and self.U != paths.U_PROD:
            raise ConfigError(f"U={self.U} != {paths.U_PROD}")
        if self.warmup > self.peak_epochs * self.U:
            raise ConfigError("warmup longer than the peak phase")

        self.model = build_model(self.cfg, self.seed, self.store, self.device, a)
        self.stat_rows = torch.as_tensor(np.sort(np.random.default_rng(4242).choice(
            self.model.K, min(8192, self.model.K), replace=False)), device=self.device)
        self.opt = build_optimizer(self.model, fused=(self.device.type == "cuda"))
        inv = parameter_inventory(self.model)
        self.config = {"run_id": a.run_id, "seed": self.seed, "registry_entry": self.entry,
                       "implementation": self.reg.get("implementation"),
                       "schedule": {"name": self.reg["schedule"]["name"], "warmup": self.warmup, "U": self.U,
                                    "peak_lr": self.peak_lr, "floor_mult": self.floor_mult,
                                    "peak_epochs": self.peak_epochs, "cosine_epochs": self.cosine_epochs,
                                    "floor_start_epoch": self.floor_start, "planned_epochs": self.planned_epochs,
                                    "min_epochs_under_cut": self.min_epochs_cut,
                                    "eval_epochs_planned": sched.eval_epochs(self.planned_epochs, self.floor_start)},
                       "precision_plan": self.plan, "precision_training": "tf32" if self.tf32 else "strict_fp32",
                       "precision_eval": "strict_fp32", "precision_training_record": prec_train,
                       "label_smoothing": self.smoothing, "effective_batch": EFFECTIVE_BATCH,
                       "microbatch": a.microbatch, "betas": list(BETAS), "eps": EPS,
                       "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP,
                       "optimizer": "AdamW fused" if self.device.type == "cuda" else "AdamW foreach",
                       "n_train": self.n, "n_vs": len(self.ds_vs), "n_cohort": len(self.ds_cohort),
                       "channels": self.channels, "epoch_cap_file": a.epoch_cap_file,
                       "threads": getattr(a, "thread_config", None),
                       "n_params": inv["total"], "param_inventory": inv, "canary": canary,
                       "gate_c": self.gate, "registry_sha256": self.reg_sha, "test_tiny": bool(a.test_tiny),
                       "physical_gpu": self.physical_gpu, "dataset_version_hash": self.store.version_hash,
                       "library_versions": {"torch": str(torch.__version__), "numpy": str(np.__version__),
                                            "python": sys.version.split()[0]}}
        if a.test_tiny and self.store.version_hash == paths.VERSION_HASH:
            raise ConfigError("--test-tiny must never use the real store")

    def check_gpu_free(self) -> None:
        try:
            r = subprocess.run(["nvidia-smi", "-i", str(self.physical_gpu), "--query-compute-apps=pid",
                                "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
        except Exception:
            return
        pids = [x.strip() for x in r.stdout.splitlines() if x.strip() and x.strip() != str(os.getpid())]
        if pids:
            raise GpuBusy(f"GPU {self.physical_gpu} busy: {pids}")

    def init_state(self) -> None:
        import ckpt_io
        ck_dir = self.run_dir / "checkpoints"
        self.resume_count = 0
        self.best_ckpt = None
        if self.args.resume:
            try:
                obj, name = ckpt_io.load_resume(ck_dir)
            except FileNotFoundError as e:
                raise ConfigError(f"--resume: {e}")
            missing = [k for k in REQUIRED_CKPT_KEYS if k not in obj]
            if missing:
                raise ConfigError(f"checkpoint lacks {missing}")
            if obj["registry_sha256"] != self.reg_sha:
                raise ConfigError("resume refuses: registry changed")
            if obj["dataset_version_hash"] != self.store.version_hash:
                raise ConfigError("resume refuses: dataset hash changed")
            if training_code_map(obj["code_sha256"]) != training_code_map(self.code_map) and not self.args.test_tiny:
                raise ConfigError("resume refuses: training code drift since the checkpoint")
            old = obj["config"]
            for key in ("registry_entry", "schedule", "label_smoothing", "microbatch", "n_train", "channels",
                        "precision_training"):
                if json.dumps(old.get(key), sort_keys=True, default=str) != json.dumps(self.config.get(key), sort_keys=True, default=str):
                    raise ConfigError(f"resume refuses: config field {key} changed")
            self.model.load_state_dict(obj["model_state_dict"])
            self.opt.load_state_dict(obj["optimizer_state_dict"])
            self.u = int(obj["update"])
            self.epoch_idx0 = int(obj["epoch_idx0"])
            self.offset = int(obj["offset_in_epoch"])
            self.cov = Coverage(self.n, obj["coverage_packed"])
            self.eval_pending = bool(obj["eval_pending"])
            self.cap_applied = obj["cap_applied"]
            self.acc = dict(obj["epoch_acc"])
            self.epoch_records = list(obj["epoch_records"])
            self.resume_count = int(obj["resume_count"]) + 1
            # best is recomputed from the persisted epoch records (Gate P fix 4), limited to the
            # epochs this checkpoint already contains; later records are recomputed deterministically
            ck_epochs = {r["epoch"] for r in self.epoch_records}
            persisted = [r for r in self.read_persisted_records() if r["epoch"] in ck_epochs]
            if {r["epoch"]: (r["full_vs"] or {}).get("macro_mrr20") for r in persisted} != \
                    {r["epoch"]: (r["full_vs"] or {}).get("macro_mrr20") for r in self.epoch_records}:
                raise ConfigError("resume: persisted epoch records disagree with the checkpoint")
            replay = {"metric": None, "epoch": None}
            for r in sorted(persisted, key=lambda r: r["epoch"]):
                if r["full_vs"] is not None:
                    replay, _ = best_update(replay, r["epoch"], r["full_vs"]["macro_mrr20"])
            if replay != obj["best"]:
                raise ConfigError(f"resume: best from persisted records {replay} != checkpoint {obj['best']}")
            self.best = replay
            self.perm = epoch_permutation(self.seed, self.epoch_idx0, self.n)
            self.perm_sha = perm_digest(self.perm)
            if self.perm_sha != obj["perm_sha256"]:
                raise ConfigError("resume: epoch permutation digest mismatch")
            if self.cov.count() != self.offset:
                raise ConfigError(f"resume: coverage {self.cov.count()} != cursor {self.offset}")
            if self.u != self.epoch_idx0 * self.U + -(-self.offset // EFFECTIVE_BATCH):
                raise ConfigError("resume: update counter inconsistent with epoch/cursor")
            rng_restore(obj["rng"])
            best = ck_dir / "best_weights.pt"
            if self.best["epoch"] is not None and best.exists():
                self.best_ckpt = {"path": str(best), "sha256": sha256_path(best), "epoch": self.best["epoch"]}
            self.log(f"resumed from {name}.pt u={self.u} epoch_idx0={self.epoch_idx0} offset={self.offset} "
                     f"eval_pending={self.eval_pending} resume_count={self.resume_count}")
        else:
            existing = [p.name for p in ck_dir.glob("*.pt")] if ck_dir.exists() else []
            if existing:
                raise ConfigError(f"{ck_dir} already holds {existing}; use --resume")
            (self.run_dir / "logs").mkdir(parents=True, exist_ok=True)
            ck_dir.mkdir(parents=True, exist_ok=True)
            atomic_json(self.run_dir / "config.json", self.config)
            atomic_json(self.run_dir / "start.json", {"start_utc": utc()})
            self.u, self.epoch_idx0, self.offset = 0, 0, 0
            self.cov = Coverage(self.n)
            self.perm = epoch_permutation(self.seed, 0, self.n)
            self.perm_sha = perm_digest(self.perm)
            self.best = {"metric": None, "epoch": None}
            self.epoch_records = []
            self.eval_pending = False
            self.cap_applied = None
            self.acc = self.fresh_acc()
            self.log(f"fresh run {self.args.run_id} seed={self.seed} n={self.n} U={self.U} "
                     f"params={self.config['n_params']} device={self.device} gpu={self.physical_gpu} "
                     f"training={'TF32' if self.tf32 else 'strict FP32'} planned_epochs={self.planned_epochs}")
            self.metric({"type": "epoch_start", "epoch": 1, "perm_sha256": self.perm_sha, "n_train": self.n})

    @staticmethod
    def fresh_acc() -> dict:
        return {"loss_sum": 0.0, "n_updates": 0, "phase": {"warmup": 0, "peak": 0, "cosine": 0, "floor": 0},
                "lr_first": None, "lr_last": None, "seconds": 0.0, "wait_seconds": 0.0,
                "dps": [], "t_start": time.time()}

    def read_persisted_records(self) -> list:
        p = self.run_dir / "epoch_records.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []

    def persist_records(self) -> None:
        atomic_json(self.run_dir / "epoch_records.json", self.epoch_records)

    def effective_epochs(self, epoch_done: int) -> int:
        """Planned epochs, or the one-time systems-only deadline cap (immutable once applied)."""
        f = self.args.epoch_cap_file
        if not f or not Path(f).exists():
            if self.cap_applied is not None:
                raise ConfigError("epoch cap file disappeared after being applied")
            return self.planned_epochs
        cap = int(json.loads(Path(f).read_text(encoding="utf-8"))["epoch_cap"])
        if self.cap_applied is not None and cap != self.cap_applied:
            raise ConfigError(f"epoch cap changed after being applied ({self.cap_applied} -> {cap})")
        if not (self.min_epochs_cut <= cap <= self.planned_epochs) and not self.args.test_tiny:
            raise ConfigError(f"epoch cap {cap} outside [{self.min_epochs_cut}, {self.planned_epochs}]")
        if cap < epoch_done:
            raise ConfigError(f"epoch cap {cap} is below the completed epoch {epoch_done}")
        self.cap_applied = cap
        return min(self.planned_epochs, cap)

    def eta(self) -> Optional[dict]:
        """Projected training end from REAL throughput (owner directive N); systems only."""
        dps = self.acc["dps"][-5:] or [r["decisions_per_s_median"] for r in self.epoch_records[-1:]
                                        if r.get("decisions_per_s_median")]
        if not dps:
            return None
        rate = statistics.fmean(dps)
        n_eff = self.planned_epochs if self.cap_applied is None else min(self.planned_epochs, self.cap_applied)
        rem_updates = max(0, n_eff * self.U - self.u)
        ev_secs = [r["eval_seconds"] for r in self.epoch_records if r.get("eval_seconds")]
        ev = statistics.fmean(ev_secs) if ev_secs else 1600.0
        rem_evals = len([e for e in sched.eval_epochs(n_eff, self.floor_start) if e > len(self.epoch_records)])
        secs = rem_updates * EFFECTIVE_BATCH / rate + rem_evals * ev
        return {"decisions_per_s": rate, "remaining_updates": rem_updates, "remaining_evals": rem_evals,
                "remaining_hours": secs / 3600,
                "training_end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + secs))}

    # ------------------------------------------------------------------ update
    def train_step(self, batch_cpu: dict, positions: np.ndarray) -> tuple[float, float, float]:
        b = int(positions.shape[0])
        if not bool(batch_cpu["loss_mask"].all()):
            raise ConfigError("a TRAIN batch row lacks loss_mask")
        n_total = int(((batch_cpu["loss_mask"]) & (batch_cpu["target_class"] >= 0)).sum())
        if n_total != b:
            raise ConfigError(f"one target per decision violated: {n_total} contributing of {b}")
        self.opt.zero_grad(set_to_none=True)
        loss_total = 0.0
        mb = self.args.microbatch
        for s in range(0, b, mb):
            e = min(s + mb, b)
            part = {k: v[s:e].to(self.device, non_blocking=True) for k, v in batch_cpu.items()}
            logits = self.model(**model_inputs(part, self.model))
            if logits.shape != (e - s, self.model.K):
                raise ConfigError(f"logits shape {tuple(logits.shape)} is not [B,K]")
            loss_sum, _ = ce_sum(logits, part["target_class"], part["loss_mask"], label_smoothing=self.smoothing)
            (loss_sum / n_total).backward()
            loss_total += float(loss_sum.detach())
        gn = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP))
        loss_mean = loss_total / n_total
        if not (math.isfinite(loss_mean) and math.isfinite(gn)):
            raise NonFiniteStop(f"non-finite loss {loss_mean} or grad norm {gn} at update {self.u + 1}")
        self.u += 1
        lr = sched.lr_at(self.u, self.peak_lr, self.U, self.warmup, self.floor_mult,
                         self.peak_epochs, self.cosine_epochs)
        set_lr(self.opt, lr)
        self.opt.step()
        self.model.recenter_output_()          # v3: loss-neutral common-mode removal (Fable FIX 4)
        return loss_mean, gn, lr

    # ------------------------------------------------------------- epoch end
    def epoch_end(self) -> bool:
        """Account, evaluate (floor epochs, strict FP32), checkpoint; True when training ends."""
        epoch = self.epoch_idx0 + 1
        if not self.cov.complete() or self.cov.count() != self.n:
            raise ConfigError(f"epoch {epoch}: coverage {self.cov.count()} != N {self.n}")
        if self.u != epoch * self.U:
            raise ConfigError(f"epoch {epoch}: u={self.u} != {epoch}*U={epoch * self.U}")
        acc = self.acc
        if sum(acc["phase"].values()) != self.U and not self.eval_pending:
            raise ConfigError(f"epoch {epoch}: phase counters {acc['phase']} do not sum to U")
        if epoch >= self.floor_start and acc["phase"]["floor"] != self.U:
            raise ConfigError(f"epoch {epoch}: only {acc['phase']['floor']} of {self.U} updates at the floor LR")
        if not self.eval_pending:
            self.eval_pending = True
            self.status(state="EVALUATING", epoch=epoch, update=self.u)
            self.save_latest()
        self.run_diag()
        evaluated = epoch >= self.floor_start
        vs = evaluate_full(self.model, self.store, self.ds_vs, self.device, self.run_dir, epoch,
                           save_ranks=True) if evaluated else None
        precision.assert_training_precision(self.tf32)
        improved = False
        if evaluated:
            self.metric({"type": "eval", "epoch": epoch, "update": self.u, "full_vs": vs})
            self.best, improved = best_update(self.best, epoch, vs["macro_mrr20"])
        rec = {"epoch": epoch, "update": self.u, "evaluated": evaluated, "full_vs": vs,
               "lr_first": acc["lr_first"], "lr_last": acc["lr_last"],
               "updates_by_phase": dict(acc["phase"]), "low_lr_updates": acc["phase"]["floor"],
               "train_loss_mean": acc["loss_sum"] / max(1, acc["n_updates"]),
               "epoch_seconds": acc["seconds"] + acc["wait_seconds"],
               "wall_epoch_seconds": time.time() - acc["t_start"],
               "eval_seconds": vs["eval_seconds"] if vs else None,
               "decisions_per_s_median": statistics.median(acc["dps"]) if acc["dps"] else None,
               "data_wait_fraction": acc["wait_seconds"] / max(1e-9, acc["seconds"] + acc["wait_seconds"]),
               "peak_gpu_mem_allocated_bytes": torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else None,
               "peak_gpu_mem_reserved_bytes": torch.cuda.max_memory_reserved(self.device) if self.device.type == "cuda" else None,
               "rss_bytes_max": _rss(), "coverage_ok": True, "exposures": self.n, "perm_sha256": self.perm_sha,
               "raw_improved": improved, "best_after": dict(self.best),
               "precision_training": "tf32" if self.tf32 else "strict_fp32", "precision_eval": "strict_fp32"}
        self.epoch_records = [r for r in self.epoch_records if r["epoch"] != epoch] + [rec]
        self.persist_records()
        if improved:
            self.save_best(epoch, vs["macro_mrr20"])
        n_eff = self.effective_epochs(epoch)
        stop = sched.fixed_epoch_should_stop(epoch, n_eff)
        rec["effective_epochs"] = n_eff
        rec["cap_applied"] = self.cap_applied
        self.persist_records()
        self.metric({"type": "epoch_end", "epoch": epoch, "update": self.u, "evaluated": evaluated,
                     "best": self.best, "effective_epochs": n_eff, "stop": stop})
        self.eval_pending = False
        if not stop:
            self.epoch_idx0 += 1
            self.offset = 0
            self.cov = Coverage(self.n)
            self.perm = epoch_permutation(self.seed, self.epoch_idx0, self.n)
            self.perm_sha = perm_digest(self.perm)
            self.acc = self.fresh_acc()
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            self.metric({"type": "epoch_start", "epoch": self.epoch_idx0 + 1, "perm_sha256": self.perm_sha,
                         "n_train": self.n})
        self.save_latest()
        return stop

    def run_diag(self) -> None:
        from eval_full import evaluate_model
        with precision.strict_eval():
            r = evaluate_model(self.model, self.store, self.ds_cohort, self.device, eval_batch=EVAL_BATCH, k=20)
        self.metric({"type": "diag_eval", "update": self.u, "epoch_idx0": self.epoch_idx0,
                     "cohort_macro_mrr20": float(r["end_to_end"]["macro"]["mrr@20"]),
                     "cohort_micro_recall20": float(r["end_to_end"]["micro"]["recall@20"]),
                     "note": "monitoring only (strict FP32); never read by selection"})

    # ------------------------------------------------------------------- main
    def finalize(self, final_status: str) -> None:
        best_rec = next((r for r in self.epoch_records if r["epoch"] == self.best["epoch"]), None)
        latest = self.run_dir / "checkpoints" / "latest.pt"
        evaluated = [r["epoch"] for r in self.epoch_records if r.get("evaluated")]
        summary = {
            "run_id": self.args.run_id, "seed": self.seed, "group": self.cfg.group, "config": self.config,
            "registry_sha256": self.reg_sha, "final_status": final_status,
            "stop_reason": "fixed_epochs" if final_status == "COMPLETE" else None,
            "planned_epochs": self.planned_epochs, "epoch_cap": self.cap_applied,
            "eval_epochs": evaluated, "precision_training": "tf32" if self.tf32 else "strict_fp32",
            "precision_eval": "strict_fp32", "precision_plan": self.plan,
            "epochs_completed": len(self.epoch_records), "updates": self.u,
            "exposures": sum(r["exposures"] for r in self.epoch_records),
            "low_lr_epochs_completed": len([r for r in self.epoch_records if r["updates_by_phase"]["floor"] == self.U]),
            "best_epoch": self.best["epoch"], "best_metric": self.best["metric"],
            "micro_recall20_at_best": best_rec["full_vs"]["micro_recall20"] if best_rec else None,
            "n_params": self.config["n_params"], "epoch_records": self.epoch_records,
            "perm_sha256_by_epoch": {str(r["epoch"]): r["perm_sha256"] for r in self.epoch_records},
            "checkpoints": {"best_weights": self.best_ckpt,
                            "latest": {"path": str(latest), "sha256": sha256_path(latest), "update": self.u}},
            "ranks": {str(r["epoch"]): {"path": r["full_vs"].get("ranks_path"), "sha256": r["full_vs"].get("ranks_sha256")}
                      for r in self.epoch_records if r.get("full_vs")},
            "code_sha256": self.code_map, "precision": precision.precision_record(),
            "start_utc": json.loads((self.run_dir / "start.json").read_text())["start_utc"]
            if (self.run_dir / "start.json").exists() else None,
            "end_utc": utc(), "resume_count": self.resume_count, "physical_gpu": self.physical_gpu,
            "wall_seconds": sum((r.get("wall_epoch_seconds") or 0) + (r.get("eval_seconds") or 0) for r in self.epoch_records),
        }
        atomic_json(self.run_dir / "summary.json", summary)
        self.status(state=final_status, epoch=len(self.epoch_records), update=self.u)

    def run(self) -> int:
        self.setup()
        self.init_state()
        a = self.args
        try:
            if self.eval_pending:
                self.log(f"resume: finishing the end of epoch {self.epoch_idx0 + 1}")
                if self.epoch_end():
                    self.finalize("COMPLETE")
                    return EXIT_OK
            if self.epoch_records and sched.fixed_epoch_should_stop(
                    self.epoch_records[-1]["epoch"], self.effective_epochs(self.epoch_records[-1]["epoch"])) \
                    and self.offset == 0 and self.epoch_idx0 + 1 == self.epoch_records[-1]["epoch"]:
                self.finalize("COMPLETE")
                return EXIT_OK
            ex = ThreadPoolExecutor(max_workers=1)

            def fetch(pos):
                bt = self.store.batch(self.ds_train, pos)
                if self.device.type == "cuda":
                    bt = {k: (v.pin_memory() if isinstance(v, torch.Tensor) else v) for k, v in bt.items()}
                return bt

            def next_positions():
                if self.offset >= self.n:
                    return None
                return self.perm[self.offset:min(self.offset + EFFECTIVE_BATCH, self.n)]

            pos = next_positions()
            pending = ex.submit(fetch, pos) if pos is not None else None
            last_hb = 0.0
            agg = {"loss": 0.0, "gn": 0.0, "gn_max": 0.0, "n": 0, "dec": 0, "sec": 0.0, "wait": 0.0}
            try:
                while pos is not None:
                    tw = time.perf_counter()
                    batch = pending.result()
                    wait = time.perf_counter() - tw
                    nxt_off = self.offset + pos.shape[0]
                    nxt = self.perm[nxt_off:min(nxt_off + EFFECTIVE_BATCH, self.n)] if nxt_off < self.n else None
                    pending = ex.submit(fetch, nxt) if nxt is not None else None
                    t0 = time.perf_counter()
                    loss, gn, lr = self.train_step(batch, pos)
                    dt = time.perf_counter() - t0
                    self.cov.mark(pos)
                    self.offset += int(pos.shape[0])
                    ph = sched.lr_phase(self.u, self.U, self.warmup, self.peak_epochs, self.cosine_epochs)
                    acc = self.acc
                    acc["phase"][ph] += 1
                    acc["lr_first"] = lr if acc["lr_first"] is None else acc["lr_first"]
                    acc["lr_last"] = lr
                    acc["loss_sum"] += loss
                    acc["n_updates"] += 1
                    acc["seconds"] += dt
                    acc["wait_seconds"] += wait
                    agg["loss"] += loss; agg["gn"] += gn; agg["gn_max"] = max(agg["gn_max"], gn)
                    agg["n"] += 1; agg["dec"] += int(pos.shape[0]); agg["sec"] += dt; agg["wait"] += wait
                    if a.test_tiny and a.log_every_update:
                        self.metric({"type": "update_debug", "update": self.u, "loss": loss, "grad_norm": gn, "lr": lr})
                    if self.u % 100 == 0:
                        dps = agg["dec"] / max(1e-9, agg["sec"] + agg["wait"])
                        acc["dps"].append(dps)
                        self.metric({"type": "update_agg", "update": self.u, "epoch": self.epoch_idx0 + 1,
                                     "mean_loss": agg["loss"] / agg["n"], "mean_grad_norm": agg["gn"] / agg["n"],
                                     "max_grad_norm": agg["gn_max"], "lr": lr, "phase": ph, "decisions_per_s": dps,
                                     "data_wait_fraction": agg["wait"] / max(1e-9, agg["sec"] + agg["wait"]),
                                     "gpu_mem_allocated": torch.cuda.memory_allocated(self.device) if self.device.type == "cuda" else None,
                                     "gpu_mem_reserved": torch.cuda.memory_reserved(self.device) if self.device.type == "cuda" else None,
                                     "emb": self.model.embedding_stats(self.stat_rows)})
                        agg = {"loss": 0.0, "gn": 0.0, "gn_max": 0.0, "n": 0, "dec": 0, "sec": 0.0, "wait": 0.0}
                    epoch_done = self.offset >= self.n
                    if a.ckpt_every_updates > 0 and self.u % a.ckpt_every_updates == 0 and not epoch_done:
                        self.save_latest()
                    if a.diag_every_updates > 0 and self.u % a.diag_every_updates == 0 and not epoch_done:
                        self.run_diag()
                        precision.assert_training_precision(self.tf32)
                    if time.time() - last_hb > HEARTBEAT_S:
                        self.status(state="RUNNING", epoch=self.epoch_idx0 + 1, update=self.u, offset=self.offset,
                                    lr=lr, phase=ph, loss=loss, disk_free_gib=_disk_free_gib(),
                                    epochs_completed=len(self.epoch_records), eta=self.eta())
                        last_hb = time.time()
                    if epoch_done:
                        if self.epoch_end():
                            break
                        pos = next_positions()
                        pending = ex.submit(fetch, pos) if pos is not None else None
                    else:
                        pos = nxt
                    if a.stop_after_updates and self.u >= a.stop_after_updates:
                        self.save_latest()
                        self.log(f"test hook: simulated crash after update {self.u}")
                        os._exit(99)
            finally:
                ex.shutdown(wait=False, cancel_futures=True)
        except NonFiniteStop as e:
            self.log(f"NONFINITE: {e}")
            self.finalize("STOPPED_NONFINITE")
            return EXIT_NONFINITE
        except Exception as e:
            import ckpt_io
            if isinstance(e, getattr(ckpt_io, "DiskFloor", ())):
                self.log(f"DISK FLOOR: {e}")
                self.finalize("STOPPED_DISK")
                return EXIT_DISK
            raise
        self.finalize("COMPLETE")
        return EXIT_OK


def _rss() -> Optional[int]:
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


def _disk_free_gib() -> float:
    st = os.statvfs(str(paths.S))
    return st.f_bavail * st.f_frsize / 2**30


# ----------------------------------------------------------------------- benchmark
def run_benchmark(args) -> int:
    """Isolated real-data systems benchmark / TF32 parity state (07 section 5). Fresh init,
    no run dir, no checkpoints; results to $S/bench/. Allowed only after Gate C."""
    if not args.run_dir:
        args.run_dir = str((paths.S / "tests" / "bench_scratch") if args.test_tiny else (paths.S / "bench" / "scratch"))
    t = Trainer(args)
    code_map = code_sha256_map()
    if not args.test_tiny:
        gate_c_check(code_map)
    precision.configure_strict_fp32(t.seed)
    t.device = torch.device("cuda:0") if args.device == "cuda" else torch.device("cpu")
    t.physical_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "cpu")
    t0 = time.time()
    store = build_store(t.channels, args)
    build_s = time.time() - t0
    ds = store.decisions("TRAIN", "loss")
    ds_vs = store.decisions("VALIDATION_SELECT", "eval")
    n = len(ds)
    model = build_model(t.cfg, t.seed, store, t.device, args)
    opt = build_optimizer(model, fused=(t.device.type == "cuda"))
    t.model, t.opt, t.U, t.u = model, opt, sched.updates_per_epoch(n), 0
    prec_train = precision.training_precision(t.tf32)
    model.train()
    rng = np.random.default_rng(12345)   # same batches for every benchmark/parity state
    lengths = np.asarray(ds.length)
    res = {"run_id": args.run_id, "seed": t.seed, "microbatch": args.microbatch, "store_build_seconds": build_s,
           "n_params": parameter_inventory(model)["total"], "physical_gpu": t.physical_gpu,
           "precision_training": "tf32" if t.tf32 else "strict_fp32", "precision_record": prec_train}
    loss_log = [] if args.bench_loss_log else None

    def timed(pos_fn, n_up, warm):
        ex = ThreadPoolExecutor(max_workers=1)
        fetch = lambda p: {k: (v.pin_memory() if t.device.type == "cuda" else v) for k, v in store.batch(ds, p).items()}
        p = pos_fn()
        fut = ex.submit(fetch, p)
        steps, waits = [], []
        if t.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(t.device)
        tw0 = time.perf_counter()
        for i in range(warm + n_up):
            tw = time.perf_counter()
            b = fut.result()
            w = time.perf_counter() - tw
            p2 = pos_fn()
            fut = ex.submit(fetch, p2)
            if t.device.type == "cuda":
                torch.cuda.synchronize(t.device)
            ts = time.perf_counter()
            loss, gn, lr = t.train_step(b, p)
            if loss_log is not None:
                loss_log.append({"update": t.u, "loss": loss, "grad_norm": gn, "lr": lr})
            if t.device.type == "cuda":
                torch.cuda.synchronize(t.device)
            if i >= warm:
                steps.append(time.perf_counter() - ts)
                waits.append(w)
            if i == warm - 1:
                tw0 = time.perf_counter()
            p = p2
        wall = time.perf_counter() - tw0
        ex.shutdown(wait=False, cancel_futures=True)
        return {"updates": n_up, "median_step_ms": 1000 * statistics.median(steps),
                "p90_step_ms": 1000 * float(np.percentile(steps, 90)),
                "wall_decisions_per_s": n_up * EFFECTIVE_BATCH / wall,
                "data_wait_fraction": sum(waits) / max(1e-9, sum(waits) + sum(steps)),
                "peak_alloc_gib": torch.cuda.max_memory_allocated(t.device) / 2**30 if t.device.type == "cuda" else None,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(t.device) / 2**30 if t.device.type == "cuda" else None,
                "rss_gib": (_rss() or 0) / 2**30}

    def param_sample() -> dict:
        g = np.random.default_rng(777)
        out = {}
        for name, prm in model.named_parameters():
            flat = prm.detach().reshape(-1)
            k = min(20_000, flat.numel())
            idx = torch.as_tensor(np.sort(g.choice(flat.numel(), k, replace=False)), device=flat.device)
            out[name] = flat.index_select(0, idx).double().cpu().numpy()
        return out

    warm = 0 if args.bench_parity else 50
    init_sample = param_sample() if args.bench_parity else None
    res["representative"] = timed(lambda: rng.choice(n, EFFECTIVE_BATCH, replace=False), args.benchmark_updates, warm)
    if loss_log is not None:
        extra = {}
        if args.bench_parity:
            from eval_full import evaluate_model
            cohort = load_cohort(ds_vs, args)
            with precision.strict_eval():
                cr = evaluate_model(model, store, cohort, t.device, eval_batch=EVAL_BATCH, k=20)
            extra["cohort"] = {"n": len(cohort), "macro_mrr20": float(cr["end_to_end"]["macro"]["mrr@20"]),
                               "micro_recall20": float(cr["end_to_end"]["micro"]["recall@20"]), "eval": "strict_fp32"}
            final_sample = param_sample()
            npz = Path(args.bench_loss_log).with_suffix(".params.npz")
            np.savez_compressed(npz, **{f"init::{k}": v for k, v in init_sample.items()},
                                **{f"final::{k}": v for k, v in final_sample.items()})
            extra["param_sample_file"] = str(npz)
        atomic_json(Path(args.bench_loss_log), {"run_id": args.run_id, "seed": t.seed,
                                                "precision_training": res["precision_training"], "updates": loss_log,
                                                **extra})
    if not args.bench_parity:
        for L in (1, 10, 50):
            pool = np.flatnonzero(lengths == L)
            if pool.shape[0] >= EFFECTIVE_BATCH:
                res[f"len{L}"] = timed(lambda pool=pool: rng.choice(pool, EFFECTIVE_BATCH, replace=False), 100, 10)
        from eval_full import evaluate_model
        sub = ds_vs.take(np.arange(min(len(ds_vs), 50_000 if not args.test_tiny else 512)))
        with precision.strict_eval():
            r = evaluate_model(model, store, sub, t.device, eval_batch=EVAL_BATCH, k=20)
        res["eval_rows_per_s_strict_fp32"] = r.get("_rows_per_s")
        rep = res["representative"]
        per_epoch_train_h = paths.N_TRAIN / rep["wall_decisions_per_s"] / 3600
        per_eval_h = paths.N_VS / (res["eval_rows_per_s_strict_fp32"] or 1) / 3600
        res["projection"] = {"train_hours_per_epoch": per_epoch_train_h, "eval_hours_full_vs": per_eval_h,
                             "hours_6_epochs_4_evals": 6 * per_epoch_train_h + 4 * per_eval_h,
                             "hours_5_epochs_3_evals": 5 * per_epoch_train_h + 3 * per_eval_h}
    tag = "parity" if args.bench_parity else "bench"
    out = paths.S / "bench" / (f"{tag}_{args.run_id}_s{t.seed}_{res['precision_training']}_mb{args.microbatch}"
                               f"_gpu{t.physical_gpu}_{int(time.time())}.json")
    if args.test_tiny:
        out = Path(args.bench_out) if args.bench_out else paths.S / "tests" / "bench_tiny.json"
    atomic_json(out, res)
    print(json.dumps({k: v for k, v in res.items() if k != "precision_record"}, default=str))
    return EXIT_OK


# ---------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--epoch-cap-file", default=None)
    ap.add_argument("--precision-mode", choices=["registry", "fp32", "tf32"], default="registry")
    ap.add_argument("--microbatch", type=int, default=256)
    ap.add_argument("--ckpt-every-updates", type=int, default=10_000)
    ap.add_argument("--diag-every-updates", type=int, default=40_000)
    ap.add_argument("--benchmark-updates", type=int, default=None)
    ap.add_argument("--bench-parity", action="store_true")
    ap.add_argument("--bench-loss-log", default=None)
    ap.add_argument("--bench-out", default=None)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    # test-only (require --test-tiny)
    ap.add_argument("--test-tiny", action="store_true")
    ap.add_argument("--tiny-n-train", type=int, default=4096)
    ap.add_argument("--tiny-n-vs", type=int, default=1024)
    ap.add_argument("--tiny-k", type=int, default=997)
    ap.add_argument("--tiny-data-seed", type=int, default=0)
    ap.add_argument("--tiny-task", default="mapping")
    ap.add_argument("--tiny-warmup", type=int, default=4)
    ap.add_argument("--tiny-planned-epochs", type=int, default=None)
    ap.add_argument("--stop-after-updates", type=int, default=None)
    ap.add_argument("--log-every-update", action="store_true")
    ap.add_argument("--allow-any-tiny-dir", action="store_true")
    return ap


def configure_threads() -> dict:
    """Owner directive D2: torch intra-op 3, inter-op 1 (cgroup-aware budget, not os.cpu_count)."""
    torch.set_num_threads(3)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already fixed for this process (e.g. under pytest)
    return {"torch_intra_op": torch.get_num_threads(), "torch_inter_op": torch.get_num_interop_threads(),
            "env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                   "NUMEXPR_NUM_THREADS")}}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.thread_config = configure_threads()
    try:
        test_only = [args.tiny_planned_epochs is not None, args.stop_after_updates is not None,
                     args.log_every_update, args.allow_any_tiny_dir, args.device == "cpu"]
        if any(test_only) and not args.test_tiny:
            raise ConfigError("test-only options require --test-tiny")
        if (args.bench_parity or args.bench_loss_log) and not args.benchmark_updates:
            raise ConfigError("--bench-parity/--bench-loss-log require --benchmark-updates")
        if args.benchmark_updates:
            return run_benchmark(args)
        if not args.run_dir:
            raise ConfigError("--run-dir is required")
        return Trainer(args).run()
    except ConfigError as e:
        print(f"CONFIG ERROR: {e}", file=sys.stderr, flush=True)
        return EXIT_CONFIG
    except GpuBusy as e:
        print(f"GPU BUSY: {e}", file=sys.stderr, flush=True)
        return EXIT_GPU_BUSY


if __name__ == "__main__":
    raise SystemExit(main())
