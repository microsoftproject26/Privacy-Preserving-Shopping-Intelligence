"""Full-data GRU trainer (Wave-1, owner: Sonnet B).

Implements `FULL_RUN_PROTOCOL.md` sections 4-6, 10 (gates 4/6/7), 11, 12
against the `FULL_INTERFACES.md` contract (`CompactStore`/`DecisionSet`/
`eval_full.evaluate_model`, consumed ONLY through those signatures). Reuses
the frozen pilot modules `model.py`, `objective.py`, `train.py` (helpers
only) by import -- never edits them.

CLI:
    python train_full.py --arm {F0_item_only_h256|F4_rich_side_information_h256} \
        --run-dir DIR [--resume] [--max-epochs 6] [--adamw-impl foreach|fused] \
        [--ckpt-every-updates 10000] [--diag-every-updates 20000] \
        [--benchmark-updates N] [--test-tiny]

Exit codes: 0 COMPLETE, 3 NONFINITE, 4 DISK, 5 CONFIG/ASSERT, 6 GPU_BUSY,
anything else = crash (uncaught exception).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("MKL_NUM_THREADS", "3")

WORKSPACE = Path(
    "/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/gru_v1"
)
CODE_DIR = WORKSPACE / "code"
VENDOR_DIR = CODE_DIR / "vendor"
ARTIFACTS = WORKSPACE / "artifacts"
DATASET_ROOT = Path("/home/jovyan/shared_data/rees46_nextitem_v1")
PROTOCOL_PATH = WORKSPACE / "FULL_RUN_PROTOCOL.md"
TESTS_FULL_DIR = WORKSPACE / "tests_full"
COHORT_PARQUET = ARTIFACTS / "validation_select_cohort.parquet"

for _p in (str(CODE_DIR), str(VENDOR_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

import train as pilot_train  # noqa: E402 -- frozen pilot module, helpers only
import model as model_mod  # noqa: E402
import objective as objective_mod  # noqa: E402

try:
    import pyarrow.parquet as pq  # noqa: E402
    pq.set_io_thread_count(3) if hasattr(pq, "set_io_thread_count") else None
except Exception:  # pragma: no cover -- defensive only
    pq = None

try:
    import pyarrow as pa  # noqa: E402
    pa.set_cpu_count(3)
    pa.set_io_thread_count(3)
except Exception:  # pragma: no cover -- defensive only
    pass

torch.set_num_threads(3)

# ---------------------------------------------------------------------------
# Fixed configuration (FULL_RUN_PROTOCOL.md sections 4-6, 11, pre-registered)
# ---------------------------------------------------------------------------

ARM_TO_GROUP = {
    "F0_item_only_h256": "F0_item_only",
    "F4_rich_side_information_h256": "F4_rich_side_information",
}
ARM_TO_CHANNELS = {
    "F0_item_only_h256": "F0",
    "F4_rich_side_information_h256": "F4",
}

SEED = 2026
MICROBATCH = 256
BETAS = (0.9, 0.999)
EPS = 1e-8
WEIGHT_DECAY = 1e-5
WARMUP_UPDATES = 100
LR_INITIAL = 1e-3
LR_DECAYED = 1e-4
GRAD_CLIP_NORM = 1.0
DELTA_THRESHOLD = 0.001
PATIENCE_LIMIT = 2
MIN_EPOCHS = 2
DEFAULT_MAX_EPOCHS = 6
EVAL_BATCH = 1024
EVAL_K = 20

N_TRAIN_PRODUCTION = 40_180_608
N_VS_PRODUCTION = 8_826_488
N_VS_USERS_PRODUCTION = 790_459
N_COHORT_PRODUCTION = 112_365
UPDATES_PER_EPOCH_PRODUCTION = 156_956  # ceil(40_180_608 / 256)

DISK_GUARD_BYTES = 8 * 1024 ** 3
CKPT_SIZE_SAFETY_MARGIN = 1.3

STATUS_HEARTBEAT_SECONDS = 60.0

EXIT_OK = 0
EXIT_NONFINITE = 3
EXIT_DISK = 4
EXIT_CONFIG = 5
EXIT_GPU_BUSY = 6


class ConfigError(Exception):
    """Raised for any condition that must map to exit code 5 (config/assert)."""


class NonFiniteStop(Exception):
    """Raised internally to unwind to exit code 3 after a fail-closed stop."""

    def __init__(self, message: str, status: dict):
        super().__init__(message)
        self.status = status


class DiskGuardStop(Exception):
    """Raised internally to unwind to exit code 4 after a fail-closed stop."""

    def __init__(self, message: str, status: dict):
        super().__init__(message)
        self.status = status


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_sha256_map() -> dict:
    names = ["train_full.py", "full_store.py", "eval_full.py", "model.py", "objective.py",
             "train.py", "window_cache.py", "evaluator.py", "index_utils.py",
             "vendor/rees46_eda/loaders.py", "vendor/rees46_eda/config.py"]
    out = {}
    for n in names:
        p = CODE_DIR / n
        h = sha256_file(p)
        if h is not None:
            out[n] = h
    return out


def protocol_sha256() -> Optional[str]:
    return sha256_file(PROTOCOL_PATH)


def _disk_free_bytes(path: Path) -> int:
    """Isolated so tests can monkeypatch exactly this function to simulate
    low free disk space without touching the real filesystem."""
    st = os.statvfs(str(path))
    return st.f_bavail * st.f_frsize


def estimate_checkpoint_bytes(model, optimizer) -> int:
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    total *= 3  # model weights + AdamW exp_avg + exp_avg_sq, roughly same footprint
    total += 50 * 1024 * 1024  # RNG state / config / metadata overhead
    return int(total * CKPT_SIZE_SAFETY_MARGIN)


def check_disk_guard(expected_bytes: int) -> tuple[bool, int]:
    free = _disk_free_bytes(WORKSPACE)
    ok = (free - expected_bytes) >= DISK_GUARD_BYTES
    return ok, free


def check_gpu_busy(physical_gpu: str) -> None:
    """Exit-6 precondition: `nvidia-smi --query-compute-apps` must show no
    foreign process on the physical GPU named by CUDA_VISIBLE_DEVICES."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "-i", str(physical_gpu), "--query-compute-apps=pid",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
    except Exception:
        return  # nvidia-smi unavailable -- don't block on an environment gap
    my_pid = str(os.getpid())
    pids = [ln.strip() for ln in r.stdout.splitlines() if ln.strip() and ln.strip() != my_pid]
    if pids:
        raise GpuBusy(f"GPU {physical_gpu} has existing compute processes: {pids}")


class GpuBusy(Exception):
    pass


def assert_single_visible_gpu() -> None:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    n_listed = len([x for x in cvd.split(",") if x.strip() != ""]) if cvd else 0
    if n_listed != 1:
        raise ConfigError(f"CUDA_VISIBLE_DEVICES must name exactly 1 device, got {cvd!r}")
    if not torch.cuda.is_available():
        raise ConfigError("CUDA not available")
    if torch.cuda.device_count() != 1:
        raise ConfigError(f"expected exactly 1 visible CUDA device, torch reports {torch.cuda.device_count()}")


# ---------------------------------------------------------------------------
# Selection / stopping / LR-step -- PURE function (protocol section 6)
# ---------------------------------------------------------------------------

@dataclass
class SelectionState:
    best_metric: Optional[float] = None
    best_epoch: Optional[int] = None
    patience: int = 0
    lr_decayed: bool = False
    lr_decay_epoch: Optional[int] = None
    stop: bool = False
    stop_reason: Optional[str] = None
    history: list = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "SelectionState":
        return cls(
            best_metric=d.get("best_metric"), best_epoch=d.get("best_epoch"),
            patience=d.get("patience", 0), lr_decayed=d.get("lr_decayed", False),
            lr_decay_epoch=d.get("lr_decay_epoch"), stop=d.get("stop", False),
            stop_reason=d.get("stop_reason"), history=list(d.get("history", [])),
        )


def selection_step(
    state: SelectionState, epoch: int, metric: float, *,
    max_epochs: int, min_epochs: int = MIN_EPOCHS,
    patience_limit: int = PATIENCE_LIMIT, delta_threshold: float = DELTA_THRESHOLD,
) -> tuple[SelectionState, dict]:
    """Pure function implementing protocol section 6's selection, stopping
    and (once-only) LR-step rule at one epoch-end evaluation.

    `epoch` is the 0-based index of the epoch that JUST finished (whose
    eval `metric` this call is processing). Returns a NEW `SelectionState`
    (input is never mutated) plus a plain-dict record of what happened.
    """
    new = SelectionState(
        best_metric=state.best_metric, best_epoch=state.best_epoch,
        patience=state.patience, lr_decayed=state.lr_decayed,
        lr_decay_epoch=state.lr_decay_epoch, stop=state.stop,
        stop_reason=state.stop_reason, history=list(state.history),
    )

    old_best = new.best_metric
    delta = float("inf") if old_best is None else (metric - old_best)
    raw_improved = old_best is None or metric > old_best  # strict > : ties -> earlier epoch wins
    if raw_improved:
        new.best_metric = metric
        new.best_epoch = epoch

    sub_threshold = delta < delta_threshold
    new.patience = (new.patience + 1) if sub_threshold else 0

    lr_step_now = False
    if sub_threshold and not new.lr_decayed:
        new.lr_decayed = True
        new.lr_decay_epoch = epoch
        lr_step_now = True

    stop_now = False
    stop_reason = None
    if new.patience >= patience_limit and (epoch + 1) >= min_epochs:
        stop_now, stop_reason = True, "patience"
    elif (epoch + 1) >= max_epochs:
        stop_now, stop_reason = True, "max_epochs"
    new.stop = stop_now
    new.stop_reason = stop_reason

    record = {
        "epoch": epoch, "metric": metric, "delta": delta, "raw_improved": raw_improved,
        "patience": new.patience, "lr_step_now": lr_step_now, "lr_decayed": new.lr_decayed,
        "stop_now": stop_now, "stop_reason": stop_reason, "best_metric": new.best_metric,
        "best_epoch": new.best_epoch,
    }
    new.history.append(record)
    return new, record


def _check_finite(loss_mean: float, grad_norm_val: float) -> bool:
    """Isolated so tests can monkeypatch exactly this seam to force the
    fail-closed non-finite path deterministically, without having to
    manufacture a genuine NaN through optimisation dynamics."""
    return math.isfinite(loss_mean) and math.isfinite(grad_norm_val)


def compute_lr(update_number_1based: int, lr_decayed: bool,
                warmup: int = WARMUP_UPDATES, lr_initial: float = LR_INITIAL,
                lr_decayed_value: float = LR_DECAYED) -> float:
    if update_number_1based <= warmup:
        return lr_initial * update_number_1based / warmup
    return lr_decayed_value if lr_decayed else lr_initial


# ---------------------------------------------------------------------------
# Data order (protocol section 6): epoch e permutation over canonical order
# ---------------------------------------------------------------------------

def epoch_permutation(epoch: int, n_train: int, seed_base: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed_base + epoch)
    return rng.permutation(n_train)


def epoch_permutation_sha256(perm: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(perm).astype(np.int64).tobytes()).hexdigest()


def iter_positions(start_epoch: int, start_offset: int, n_train: int, microbatch: int,
                    max_epochs: int, on_epoch_start=None, seed_base: int = SEED):
    """Yields (epoch, offset_before_batch, positions) for every remaining
    microbatch from (start_epoch, start_offset) through the end of
    `max_epochs`. Pure function of (epoch, n_train, microbatch, seed_base);
    safe to reconstruct identically on resume.

    `seed_base` defaults to the protocol seed 2026; it is only ever
    overridden by the test-only `--break-resume-perm-seed` hook (see
    `test_train_full.py`'s deliberately-broken resume xfail), which proves
    this test suite has the power to catch a "resume reconstructed the
    wrong epoch permutation" bug rather than passing vacuously.
    """
    epoch = start_epoch
    offset = start_offset
    while epoch < max_epochs:
        perm = epoch_permutation(epoch, n_train, seed_base=seed_base)
        if on_epoch_start is not None:
            on_epoch_start(epoch, offset, perm)
        while offset < n_train:
            end = min(offset + microbatch, n_train)
            yield epoch, offset, perm[offset:end]
            offset = end
        offset = 0
        epoch += 1


def updates_per_epoch(n_train: int, microbatch: int = MICROBATCH) -> int:
    return -(-n_train // microbatch)


# ---------------------------------------------------------------------------
# Store / decision-set construction
# ---------------------------------------------------------------------------

def build_store(*, channels: str, test_tiny: bool, fake_kwargs: Optional[dict] = None):
    if test_tiny:
        from fakes_for_tests import FakeStore
        kwargs = dict(
            cache_train_dir=WORKSPACE / "cache" / "train",
            cache_vs_dir=WORKSPACE / "cache" / "vs",
            cohort_parquet=COHORT_PARQUET,
            channels=channels, n_train=4096, n_vs=2048,
        )
        if fake_kwargs:
            kwargs.update(fake_kwargs)
        return FakeStore.build(**kwargs)
    from full_store import CompactStore  # owner: Sonnet A -- lazy import
    return CompactStore.build(
        DATASET_ROOT, splits=("TRAIN", "VALIDATION_SELECT"), channels=channels,
    )


def get_evaluate_fn(test_tiny: bool):
    if test_tiny:
        from fakes_for_tests import fake_evaluate_model
        return fake_evaluate_model
    from eval_full import evaluate_model  # owner: Sonnet C -- lazy import
    return evaluate_model


def load_cohort_ds(ds_vs, *, test_tiny: bool):
    """Diagnostic cohort DecisionSet: production takes it from the pilot
    10k-user cohort parquet via searchsorted into `ds_vs`; test-tiny reuses
    the (already tiny) fake VS set directly."""
    if test_tiny:
        return ds_vs
    import pyarrow.parquet as pq
    t = pq.read_table(COHORT_PARQUET, columns=["decision_id"])
    cohort_ids = np.ascontiguousarray(t["decision_id"].to_numpy(zero_copy_only=False)).astype(np.int64)
    if cohort_ids.shape[0] != N_COHORT_PRODUCTION:
        raise ConfigError(
            f"expected {N_COHORT_PRODUCTION} cohort decisions, got {cohort_ids.shape[0]}"
        )
    pos = np.searchsorted(ds_vs.decision_id, cohort_ids)
    pos_clipped = np.clip(pos, 0, len(ds_vs) - 1)
    found = ds_vs.decision_id[pos_clipped] == cohort_ids
    if not bool(found.all()):
        raise ConfigError("not every VS-cohort decision_id was found in the VS decision set")
    return ds_vs.take(pos)


def _assert_disjoint_sorted(a_sorted: np.ndarray, b: np.ndarray, msg: str) -> None:
    pos = np.searchsorted(a_sorted, b)
    pos_clipped = np.clip(pos, 0, max(0, a_sorted.shape[0] - 1))
    if a_sorted.shape[0] == 0:
        return
    hit = a_sorted[pos_clipped] == b
    if bool(hit.any()):
        raise ConfigError(msg)


def run_launch_asserts(store, ds_train, ds_vs, *, test_tiny: bool) -> None:
    """Protocol section 10 gates 2-4 (and the model class-map gate), plus
    the general shape asserts, in the strength appropriate to test-tiny."""
    if ds_train.split != "TRAIN" or ds_train.kind != "loss":
        raise ConfigError(f"ds_train split/kind wrong: {ds_train.split}/{ds_train.kind}")
    if ds_vs.split != "VALIDATION_SELECT" or ds_vs.kind != "eval":
        raise ConfigError(f"ds_vs split/kind wrong: {ds_vs.split}/{ds_vs.kind}")
    if not bool(np.asarray(ds_train.loss_mask).all()):
        raise ConfigError("ds_train.loss_mask is not all True")
    dec = np.asarray(ds_train.decision_id)
    if dec.shape[0] > 1 and not bool(np.all(np.diff(dec) > 0)):
        raise ConfigError("ds_train.decision_id is not strictly ascending")

    if not test_tiny:
        if len(ds_train) != N_TRAIN_PRODUCTION:
            raise ConfigError(f"len(ds_train)={len(ds_train)} != {N_TRAIN_PRODUCTION}")
        if len(ds_vs) != N_VS_PRODUCTION:
            raise ConfigError(f"len(ds_vs)={len(ds_vs)} != {N_VS_PRODUCTION}")
        n_users = int(np.unique(np.asarray(ds_vs.eval_user_key)).shape[0])
        if n_users != N_VS_USERS_PRODUCTION:
            raise ConfigError(f"distinct VS eval users={n_users} != {N_VS_USERS_PRODUCTION}")

    # VS-leakage gate: TRAIN decision_ids disjoint from VS decision_ids.
    _assert_disjoint_sorted(dec, np.asarray(ds_vs.decision_id),
                             "VS-leakage: a VALIDATION_SELECT decision_id was found in TRAIN")

    assert_single_visible_gpu()

    poc = pilot_train.load_product_idx_of_class()
    K = poc.shape[0]
    expected = np.arange(K, dtype=poc.dtype) + 3
    if not np.array_equal(poc, expected):
        raise ConfigError("product_idx_of_class != arange(K)+3")


# ---------------------------------------------------------------------------
# Model / optimizer
# ---------------------------------------------------------------------------

def build_model_and_optimizer(group: str, device: torch.device, adamw_impl: str):
    poc = pilot_train.load_product_idx_of_class()
    model = model_mod.build_model(group, pilot_train.VOCAB_SIZES, product_idx_of_class=poc).to(device)
    param_groups = pilot_train.build_param_groups(model, weight_decay=WEIGHT_DECAY)
    kwargs = dict(lr=LR_INITIAL, betas=BETAS, eps=EPS)
    if adamw_impl == "fused":
        kwargs["fused"] = True
    elif adamw_impl == "foreach":
        kwargs["foreach"] = True
    else:
        raise ConfigError(f"unknown adamw_impl {adamw_impl!r}")
    optimizer = torch.optim.AdamW(param_groups, **kwargs)
    return model, optimizer


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = lr


def _pin(batch: dict) -> dict:
    out = {}
    for k, v in batch.items():
        t = v if isinstance(v, torch.Tensor) else torch.as_tensor(v)
        if t.device.type == "cpu":
            try:
                t = t.pin_memory()
            except RuntimeError:
                pass
        out[k] = t
    return out


# ---------------------------------------------------------------------------
# Checkpointing (protocol section 11)
# ---------------------------------------------------------------------------

REQUIRED_CKPT_KEYS = {
    "model_state_dict", "optimizer_state_dict", "rng_state", "epoch", "offset_in_epoch",
    "update", "eval_pending", "lr_state", "patience", "selection_state", "best_metric",
    "config", "dataset_version_hash", "protocol_sha256", "code_sha256", "resume_count",
}


def build_checkpoint_obj(*, model, optimizer, epoch: int, offset_in_epoch: int, update: int,
                          eval_pending: bool, selection_state: SelectionState,
                          best_metric: Optional[float], config: dict, resume_count: int) -> dict:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": pilot_train.rng_state_snapshot(),
        "epoch": epoch,
        "offset_in_epoch": offset_in_epoch,
        "update": update,
        "eval_pending": eval_pending,
        "lr_state": {"lr_decayed": selection_state.lr_decayed,
                     "lr_decay_epoch": selection_state.lr_decay_epoch},
        "patience": selection_state.patience,
        "selection_state": dataclasses.asdict(selection_state),
        "best_metric": best_metric,
        "config": config,
        "dataset_version_hash": config.get("dataset_version_hash"),
        "protocol_sha256": protocol_sha256(),
        "code_sha256": code_sha256_map(),
        "resume_count": resume_count,
        "saved_at_utc": _utc_now_iso(),
    }


def _validate_checkpoint_file(path: Path, expected_update: int) -> None:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    missing = REQUIRED_CKPT_KEYS - set(obj.keys())
    if missing:
        raise ValueError(f"checkpoint {path} missing keys: {sorted(missing)}")
    if int(obj["update"]) != int(expected_update):
        raise ValueError(f"checkpoint {path} update {obj['update']} != expected {expected_update}")


def write_checkpoint_atomic(ckpt_dir: Path, name: str, obj: dict) -> Path:
    """Write `.tmp`, validate by reload, THEN rotate with `os.replace`.
    For name=="latest": rotates the existing latest.pt -> previous.pt first
    (so there is never a moment with zero valid checkpoints), then tmp ->
    latest.pt. For any other name (best/previous), a single atomic replace.
    Any exception here is raised BEFORE any rotation touches latest/previous
    on disk, so an interrupted write leaves prior checkpoints intact.
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tmp = ckpt_dir / f"{name}.pt.tmp"
    final = ckpt_dir / f"{name}.pt"
    try:
        torch.save(obj, tmp)
        _validate_checkpoint_file(tmp, obj["update"])
    except BaseException:
        tmp.unlink(missing_ok=True)  # never leave a stray/partial .tmp behind
        raise
    if name == "latest":
        prev = ckpt_dir / "previous.pt"
        if final.exists():
            os.replace(final, prev)
    os.replace(tmp, final)
    return final


def load_checkpoint(ckpt_dir: Path, name: str = "latest", map_location: str = "cpu") -> dict:
    path = ckpt_dir / f"{name}.pt"
    return torch.load(path, map_location=map_location, weights_only=False)


def load_resume_checkpoint(ckpt_dir: Path) -> tuple[dict, str]:
    """latest.pt if it loads and carries every required key, else previous.pt.

    `write_checkpoint_atomic` rotates latest->previous and then tmp->latest, so a
    crash between the two os.replace calls (or a latest.pt damaged on disk)
    leaves previous.pt as the newest valid state. Falling back to it is what
    keeps "never zero valid checkpoints" useful on the resume side too.
    """
    errors = []
    for name in ("latest", "previous"):
        path = ckpt_dir / f"{name}.pt"
        if not path.exists():
            errors.append(f"{name}: missing")
            continue
        try:
            obj = load_checkpoint(ckpt_dir, name)
            missing = REQUIRED_CKPT_KEYS - set(obj.keys())
            if missing:
                raise ValueError(f"missing keys {sorted(missing)}")
            return obj, name
        except Exception as exc:  # corrupt / truncated file
            errors.append(f"{name}: {exc!r}")
    raise ConfigError(f"--resume: no valid checkpoint in {ckpt_dir} ({'; '.join(errors)})")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_metric(run_dir: Path, record: dict) -> None:
    with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


def run_log(run_dir: Path, msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    p = run_dir / "logs" / "train_full.log"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def write_status(run_dir: Path, status: dict) -> None:
    status = dict(status)
    status["utc"] = _utc_now_iso()
    _atomic_write_json(run_dir / "status.json", status)


def init_run_dir(run_dir: Path, config: dict, device: torch.device) -> None:
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (run_dir / "environment.json").write_text(
        json.dumps(pilot_train.gather_environment_info(device), indent=2), encoding="utf-8")
    (run_dir / "git_state.json").write_text(
        json.dumps(pilot_train.gather_git_state(), indent=2), encoding="utf-8")
    (run_dir / "hardware.json").write_text(
        json.dumps(pilot_train.gather_hardware_info(), indent=2), encoding="utf-8")
    (run_dir / "start.json").write_text(json.dumps({"start_utc": _utc_now_iso()}), encoding="utf-8")


def write_summary(run_dir: Path, summary: dict) -> None:
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------
# Core training routine
# ---------------------------------------------------------------------------

@dataclass
class RunHandles:
    store: Any
    ds_train: Any
    ds_vs: Any
    ds_cohort: Any
    model: Any
    optimizer: Any
    device: torch.device
    evaluate_fn: Any


def _run_eval(evaluate_fn, model, store, ds, device) -> dict:
    return evaluate_fn(model, store, ds, device, eval_batch=EVAL_BATCH, k=EVAL_K)


def selection_metric_from_result(result: dict) -> float:
    return float(result["end_to_end"]["macro"]["mrr@20"])


def run_training(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    test_tiny = bool(args.test_tiny)

    if test_tiny:
        try:
            run_dir.relative_to(TESTS_FULL_DIR.resolve())
        except ValueError:
            raise ConfigError(f"--test-tiny refuses run dirs outside {TESTS_FULL_DIR}: {run_dir}")
    else:
        if getattr(args, "stop_after_updates", None) or getattr(args, "break_resume_perm_seed", False):
            raise ConfigError("--stop-after-updates/--break-resume-perm-seed require --test-tiny")

    if not torch.cuda.is_available():
        raise ConfigError("CUDA not available")
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    physical_gpu = cvd.split(",")[0].strip() if cvd else "0"
    try:
        check_gpu_busy(physical_gpu)
    except GpuBusy as e:
        run_log(run_dir, f"GPU busy at startup: {e}")
        return EXIT_GPU_BUSY

    device = torch.device("cuda:0")
    group = ARM_TO_GROUP[args.arm]
    channels = ARM_TO_CHANNELS[args.arm]

    pilot_train.set_determinism(SEED)

    store = build_store(channels=channels, test_tiny=test_tiny)
    if store.version_hash != pilot_train.VERSION_HASH:
        raise ConfigError(f"store.version_hash {store.version_hash!r} != expected {pilot_train.VERSION_HASH!r}")

    ds_train = store.decisions("TRAIN", "loss")
    ds_vs = store.decisions("VALIDATION_SELECT", "eval")
    run_launch_asserts(store, ds_train, ds_vs, test_tiny=test_tiny)
    ds_cohort = load_cohort_ds(ds_vs, test_tiny=test_tiny)

    n_train = len(ds_train)
    evaluate_fn = get_evaluate_fn(test_tiny)

    config = {
        "arm": args.arm, "group": group, "channels": channels, "adamw_impl": args.adamw_impl,
        "seed": SEED, "microbatch": MICROBATCH, "lr_initial": LR_INITIAL, "lr_decayed": LR_DECAYED,
        "betas": list(BETAS), "eps": EPS, "weight_decay": WEIGHT_DECAY,
        "warmup_updates": WARMUP_UPDATES, "grad_clip_norm": GRAD_CLIP_NORM,
        "max_epochs": args.max_epochs, "delta_threshold": DELTA_THRESHOLD,
        "patience_limit": PATIENCE_LIMIT, "min_epochs": MIN_EPOCHS,
        "n_train": n_train, "n_vs": len(ds_vs), "n_cohort": len(ds_cohort),
        "test_tiny": test_tiny, "dataset_version_hash": store.version_hash,
        "ckpt_every_updates": args.ckpt_every_updates, "diag_every_updates": args.diag_every_updates,
        # plain str(): torch.__version__ is a TorchVersion object that yaml.safe_dump rejects
        "library_versions": {"torch": str(torch.__version__), "numpy": str(np.__version__),
                             "pyarrow": str(getattr(pa, "__version__", None)) if "pa" in globals() else None,
                             "python": sys.version.split()[0]},
    }

    ckpt_dir = run_dir / "checkpoints"
    latest_path = ckpt_dir / "latest.pt"

    resume_count = 0
    if args.resume:
        ckpt, resumed_from = load_resume_checkpoint(ckpt_dir)
        if ckpt["config"].get("adamw_impl") != args.adamw_impl:
            raise ConfigError(
                f"resume refuses adamw_impl mismatch: checkpoint={ckpt['config'].get('adamw_impl')!r} "
                f"current={args.adamw_impl!r}"
            )
        if ckpt.get("dataset_version_hash") != store.version_hash:
            raise ConfigError("resume refuses dataset_version_hash mismatch")
        model, optimizer = build_model_and_optimizer(group, device, args.adamw_impl)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        pilot_train.restore_rng_state(ckpt["rng_state"])
        epoch = int(ckpt["epoch"])
        offset_in_epoch = int(ckpt["offset_in_epoch"])
        update = int(ckpt["update"])
        eval_pending = bool(ckpt["eval_pending"])
        selection_state = SelectionState.from_dict(ckpt["selection_state"])
        best_metric = ckpt.get("best_metric")
        resume_count = int(ckpt.get("resume_count", 0)) + 1
        run_log(run_dir, f"resumed from {resumed_from}.pt run_dir={run_dir} update={update} epoch={epoch} "
                          f"offset={offset_in_epoch} eval_pending={eval_pending} resume_count={resume_count}")
    else:
        existing = [n for n in ("latest.pt", "previous.pt", "best.pt") if (ckpt_dir / n).exists()]
        if existing:
            raise ConfigError(f"{ckpt_dir} holds {existing}; refusing to overwrite a non-empty run without --resume")
        init_run_dir(run_dir, config, device)
        model, optimizer = build_model_and_optimizer(group, device, args.adamw_impl)
        epoch, offset_in_epoch, update = 0, 0, 0
        eval_pending = False
        selection_state = SelectionState()
        best_metric = None
        run_log(run_dir, f"fresh run_dir={run_dir} arm={args.arm} adamw_impl={args.adamw_impl}")

    handles = RunHandles(store=store, ds_train=ds_train, ds_vs=ds_vs, ds_cohort=ds_cohort,
                         model=model, optimizer=optimizer, device=device, evaluate_fn=evaluate_fn)

    def save_ckpt(name: str, *, eval_pending_flag: bool) -> Path:
        obj = build_checkpoint_obj(
            model=model, optimizer=optimizer, epoch=epoch, offset_in_epoch=offset_in_epoch,
            update=update, eval_pending=eval_pending_flag, selection_state=selection_state,
            best_metric=best_metric, config=config, resume_count=resume_count,
        )
        expected_bytes = estimate_checkpoint_bytes(model, optimizer)
        ok, free = check_disk_guard(expected_bytes)
        if not ok:
            status = {"pid": os.getpid(), "arm": args.arm, "state": "STOPPED_DISK",
                       "epoch": epoch, "update": update, "offset_in_epoch": offset_in_epoch,
                       "disk_free_bytes": free}
            write_status(run_dir, status)
            raise DiskGuardStop(f"disk guard: free={free} expected={expected_bytes}", status)
        return write_checkpoint_atomic(ckpt_dir, name, obj)

    def run_epoch_end_eval() -> float:
        result = _run_eval(evaluate_fn, model, store, ds_vs, device)
        log_metric(run_dir, {"type": "eval", "epoch": epoch, "update": update,
                              "eval_full": result})
        return selection_metric_from_result(result)

    def run_diag_eval() -> None:
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        was_training = model.training
        result = _run_eval(evaluate_fn, model, store, ds_cohort, device)
        if was_training:
            model.train()
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        log_metric(run_dir, {"type": "diag_eval", "epoch": epoch, "update": update,
                              "cohort_client_macro_mrr@20": selection_metric_from_result(result),
                              "eval_full": result})

    def finalize(status_state: str) -> None:
        write_status(run_dir, {
            "pid": os.getpid(), "arm": args.arm, "state": status_state, "epoch": epoch,
            "update": update, "offset_in_epoch": offset_in_epoch,
            "best_metric": best_metric, "patience": selection_state.patience,
            "disk_free_bytes": _disk_free_bytes(WORKSPACE),
        })
        best_path = ckpt_dir / "best.pt"
        summary = {
            "arm": args.arm, "final_status": status_state, "epoch": epoch, "update": update,
            "best_epoch": selection_state.best_epoch, "best_metric": best_metric,
            "best_checkpoint_path": str(best_path) if best_path.exists() else None,
            "best_checkpoint_sha256": sha256_file(best_path),
            "resume_count": resume_count, "end_utc": _utc_now_iso(),
            "selection_history": selection_state.history,
        }
        write_summary(run_dir, summary)

    # -- handle a checkpoint left in "epoch complete, eval pending" state --
    if eval_pending:
        run_log(run_dir, f"resume found eval_pending at epoch={epoch}: running epoch-end eval first")
        metric = run_epoch_end_eval()
        selection_state, record = selection_step(
            selection_state, epoch, metric, max_epochs=args.max_epochs)
        log_metric(run_dir, {"type": "epoch_end", "epoch": epoch, "update": update,
                              "selection_record": record})
        if record["raw_improved"]:
            best_metric = metric
            save_ckpt("best", eval_pending_flag=False)
        eval_pending = False
        save_ckpt("latest", eval_pending_flag=False)
        if selection_state.stop:
            finalize("COMPLETE")
            return EXIT_OK

    last_status_write = 0.0
    last_metric_agg = {"loss_sum": 0.0, "n_updates": 0, "grad_norm_max": 0.0, "grad_norm_sum": 0.0,
                        "decisions": 0, "seconds": 0.0, "wait_seconds": 0.0}
    ema_loss: Optional[float] = None

    def logged_epoch_starts(epoch_i, offset_i, perm):
        sha = epoch_permutation_sha256(perm)
        log_metric(run_dir, {"type": "epoch_start", "epoch": epoch_i, "resume_offset": offset_i,
                              "perm_sha256": sha, "n_train": n_train,
                              "updates_this_epoch": updates_per_epoch(n_train, MICROBATCH)})

    perm_seed_base = SEED + 999_999 if getattr(args, "break_resume_perm_seed", False) else SEED
    positions_iter = iter_positions(epoch, offset_in_epoch, n_train, MICROBATCH,
                                     args.max_epochs, on_epoch_start=logged_epoch_starts,
                                     seed_base=perm_seed_base)

    # Test-only hook (requires --test-tiny): artificially slow down each
    # update so an external test process (e.g. the supervisor SIGKILL test)
    # can reliably observe progress and interrupt mid-epoch instead of
    # racing a tiny FakeStore run to its natural completion. Never set
    # outside a test -- the env var is not read unless test_tiny is True.
    test_update_sleep_s = 0.0
    if test_tiny:
        try:
            test_update_sleep_s = float(os.environ.get("TRAIN_FULL_TEST_UPDATE_SLEEP", "0") or 0.0)
        except ValueError:
            test_update_sleep_s = 0.0

    executor = ThreadPoolExecutor(max_workers=1)

    def fetch(positions):
        return _pin(store.batch(ds_train, positions))

    it = iter(positions_iter)
    try:
        current = next(it)
    except StopIteration:
        current = None
    pending = executor.submit(fetch, current[2]) if current is not None else None

    exit_code = EXIT_OK
    stopped_by_test_hook = False
    try:
        while current is not None:
            cur_epoch, cur_offset, positions = current
            t_wait0 = time.perf_counter()
            batch_cpu = pending.result()
            wait_s = time.perf_counter() - t_wait0

            try:
                nxt = next(it)
            except StopIteration:
                nxt = None
            pending = executor.submit(fetch, nxt[2]) if nxt is not None else None

            b_size = int(positions.shape[0])
            t0 = time.perf_counter()
            batch_gpu = {k: v.to(device, non_blocking=True) for k, v in batch_cpu.items()}
            assert bool(batch_gpu["loss_mask"].all()), (
                "VS-leakage/accounting gate: a TRAIN microbatch had loss_mask != all-True"
            )
            kw = pilot_train.build_named_kwargs(batch_gpu, model)
            target_class = batch_gpu["target_class"]
            loss_mask = batch_gpu["loss_mask"]

            optimizer.zero_grad(set_to_none=True)
            logits = model(**kw)
            loss_sum, n_contrib = objective_mod.exact_ce_loss(logits, target_class, loss_mask)
            loss_sum.backward()
            n_val = float(n_contrib.item())
            if n_val <= 0:
                raise ConfigError("update with zero loss-contributing rows on a pre-filtered TRAIN set")
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.div_(n_val)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

            loss_mean = float((loss_sum / n_val).item())
            grad_norm_val = float(grad_norm.item())

            if not _check_finite(loss_mean, grad_norm_val):
                status = {"pid": os.getpid(), "arm": args.arm, "state": "STOPPED_NONFINITE",
                           "epoch": cur_epoch, "update": update, "offset_in_epoch": cur_offset,
                           "loss": loss_mean, "grad_norm": grad_norm_val}
                write_status(run_dir, status)
                log_metric(run_dir, {"type": "nonfinite_stop", "epoch": cur_epoch, "update": update,
                                      "loss": loss_mean, "grad_norm": grad_norm_val})
                raise NonFiniteStop("non-finite loss or grad norm before optimizer.step()", status)

            update += 1
            lr = compute_lr(update, selection_state.lr_decayed)
            set_lr(optimizer, lr)
            optimizer.step()

            offset_in_epoch = cur_offset + b_size
            epoch = cur_epoch
            dt = time.perf_counter() - t0

            if test_tiny:
                # Fine-grained per-update record, test-only: production never
                # sets test_tiny, so metrics.jsonl stays at the spec'd
                # 100-update aggregation cadence in real runs. Tests (e.g.
                # resume determinism) need per-update loss visibility that
                # the 100-update aggregate cannot provide at tiny update
                # counts.
                log_metric(run_dir, {"type": "update_debug", "update": update, "epoch": epoch,
                                      "loss": loss_mean, "grad_norm": grad_norm_val, "lr": lr})

            ema_loss = loss_mean if ema_loss is None else (0.98 * ema_loss + 0.02 * loss_mean)
            last_metric_agg["loss_sum"] += loss_mean
            last_metric_agg["n_updates"] += 1
            last_metric_agg["grad_norm_sum"] += grad_norm_val
            last_metric_agg["grad_norm_max"] = max(last_metric_agg["grad_norm_max"], grad_norm_val)
            last_metric_agg["decisions"] += b_size
            last_metric_agg["seconds"] += dt
            last_metric_agg["wait_seconds"] += wait_s

            if update % 100 == 0 and last_metric_agg["n_updates"] > 0:
                agg = last_metric_agg
                dec_per_s = agg["decisions"] / agg["seconds"] if agg["seconds"] > 0 else 0.0
                wait_frac = agg["wait_seconds"] / (agg["wait_seconds"] + agg["seconds"]) if (agg["wait_seconds"] + agg["seconds"]) > 0 else 0.0
                mem_alloc = torch.cuda.memory_allocated(device) if torch.cuda.is_available() else 0
                mem_reserved = torch.cuda.memory_reserved(device) if torch.cuda.is_available() else 0
                log_metric(run_dir, {
                    "type": "update_agg", "update": update, "epoch": epoch,
                    "mean_loss": agg["loss_sum"] / agg["n_updates"], "lr": lr,
                    "mean_grad_norm": agg["grad_norm_sum"] / agg["n_updates"],
                    "max_grad_norm": agg["grad_norm_max"], "decisions_per_s": dec_per_s,
                    "data_wait_fraction": wait_frac, "gpu_mem_allocated_bytes": mem_alloc,
                    "gpu_mem_reserved_bytes": mem_reserved,
                })
                last_metric_agg = {"loss_sum": 0.0, "n_updates": 0, "grad_norm_max": 0.0,
                                    "grad_norm_sum": 0.0, "decisions": 0, "seconds": 0.0,
                                    "wait_seconds": 0.0}

            if args.ckpt_every_updates > 0 and update % args.ckpt_every_updates == 0:
                save_ckpt("latest", eval_pending_flag=False)

            if args.diag_every_updates > 0 and update % args.diag_every_updates == 0:
                run_diag_eval()

            now = time.time()
            if now - last_status_write >= STATUS_HEARTBEAT_SECONDS:
                write_status(run_dir, {
                    "pid": os.getpid(), "arm": args.arm, "state": "RUNNING", "epoch": epoch,
                    "update": update, "offset_in_epoch": offset_in_epoch,
                    "decisions_per_s": (b_size / dt) if dt > 0 else 0.0, "loss_ema": ema_loss,
                    "best_metric": best_metric, "patience": selection_state.patience, "lr": lr,
                    "disk_free_bytes": _disk_free_bytes(WORKSPACE),
                    "last_checkpoint": str(ckpt_dir / "latest.pt"),
                })
                last_status_write = now

            epoch_ended_now = offset_in_epoch >= n_train
            if epoch_ended_now:
                write_status(run_dir, {"pid": os.getpid(), "arm": args.arm, "state": "EVALUATING",
                                        "epoch": epoch, "update": update})
                save_ckpt("latest", eval_pending_flag=True)
                log_metric(run_dir, {"type": "epoch_end_accounting", "epoch": epoch,
                                      "updates_this_epoch": updates_per_epoch(n_train, MICROBATCH),
                                      "exposures_this_epoch": n_train})
                run_diag_eval()
                metric = run_epoch_end_eval()
                selection_state, record = selection_step(
                    selection_state, epoch, metric, max_epochs=args.max_epochs)
                log_metric(run_dir, {"type": "epoch_end", "epoch": epoch, "update": update,
                                      "selection_record": record})
                if record["raw_improved"]:
                    best_metric = metric
                    save_ckpt("best", eval_pending_flag=False)
                save_ckpt("latest", eval_pending_flag=False)
                if selection_state.stop:
                    break

            if test_update_sleep_s > 0:
                # Test-only hook: applied AFTER this update's own checkpoint/
                # eval bookkeeping is fully done, so an external test process
                # that kills us while we sleep still finds a just-written
                # checkpoint on disk (mirroring a real mid-epoch crash, not a
                # crash that loses the update it was sleeping "during").
                time.sleep(test_update_sleep_s)

            # Test-only hook (requires --test-tiny; see run_launch_asserts'
            # sibling check in main()): simulate a crash immediately after
            # update N has fully completed its bookkeeping (including any
            # checkpoint/eval due exactly at N), so resume-determinism tests
            # can interrupt at an exact update count instead of only at
            # epoch boundaries. Never reachable in production.
            if getattr(args, "stop_after_updates", None) and update >= args.stop_after_updates:
                stopped_by_test_hook = True
                break

            current = nxt
    except NonFiniteStop:
        exit_code = EXIT_NONFINITE
    except DiskGuardStop:
        exit_code = EXIT_DISK
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    if exit_code == EXIT_OK and stopped_by_test_hook:
        finalize("STOPPED_TEST_HOOK")
    elif exit_code == EXIT_OK:
        finalize("COMPLETE")
    elif exit_code == EXIT_NONFINITE:
        finalize("STOPPED_NONFINITE")
    elif exit_code == EXIT_DISK:
        finalize("STOPPED_DISK")
    return exit_code


# ---------------------------------------------------------------------------
# Benchmark mode: fresh init, no checkpoints/eval, writes nothing to run dirs
# ---------------------------------------------------------------------------

def run_benchmark(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise ConfigError("CUDA not available")
    device = torch.device("cuda:0")
    group = ARM_TO_GROUP[args.arm]
    channels = ARM_TO_CHANNELS[args.arm]

    pilot_train.set_determinism(SEED)
    store = build_store(channels=channels, test_tiny=bool(args.test_tiny))
    ds_train = store.decisions("TRAIN", "loss")
    n = len(ds_train)
    model, optimizer = build_model_and_optimizer(group, device, args.adamw_impl)
    model.train()

    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    ptr = 0

    def next_positions():
        nonlocal ptr, perm
        if ptr + MICROBATCH > n:
            perm = rng.permutation(n)
            ptr = 0
        pos = perm[ptr:ptr + MICROBATCH]
        ptr += MICROBATCH
        return pos

    executor = ThreadPoolExecutor(max_workers=1)

    def fetch(positions):
        return _pin(store.batch(ds_train, positions))

    torch.cuda.reset_peak_memory_stats(device)
    dec_times, wait_times = [], []
    pending = executor.submit(fetch, next_positions())
    t_loop0 = time.perf_counter()
    for i in range(args.benchmark_updates):
        t_wait0 = time.perf_counter()
        batch_cpu = pending.result()
        wait_times.append(time.perf_counter() - t_wait0)
        pending = executor.submit(fetch, next_positions())

        batch_gpu = {k: v.to(device, non_blocking=True) for k, v in batch_cpu.items()}
        kw = pilot_train.build_named_kwargs(batch_gpu, model)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits = model(**kw)
        loss_sum, n_c = objective_mod.exact_ce_loss(logits, batch_gpu["target_class"], batch_gpu["loss_mask"])
        loss_sum.backward()
        n_val = float(n_c.item())
        for p in model.parameters():
            if p.grad is not None:
                p.grad.div_(n_val)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        lr = compute_lr(i + 1, False)
        set_lr(optimizer, lr)
        optimizer.step()
        torch.cuda.synchronize(device)
        dec_times.append(time.perf_counter() - t0)
    loop_wall_s = time.perf_counter() - t_loop0
    executor.shutdown(wait=False, cancel_futures=True)

    import psutil
    dec_per_s_samples = [MICROBATCH / dt for dt in dec_times if dt > 0]
    result = {
        "arm": args.arm, "adamw_impl": args.adamw_impl, "n_updates": args.benchmark_updates,
        "median_decisions_per_s": statistics.median(dec_per_s_samples) if dec_per_s_samples else None,
        "data_wait_fraction": (sum(wait_times) / (sum(wait_times) + sum(dec_times))
                                if (sum(wait_times) + sum(dec_times)) > 0 else None),
        "median_step_seconds": statistics.median(dec_times) if dec_times else None,
        # end-to-end, including data wait and host->device copies (what an epoch actually costs)
        "wall_decisions_per_s": (args.benchmark_updates * MICROBATCH / loop_wall_s) if loop_wall_s > 0 else None,
        "loop_wall_seconds": loop_wall_s,
        "peak_gpu_mem_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_gpu_mem_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "host_rss_bytes": psutil.Process().memory_info().rss,
    }
    print(json.dumps(result))
    return EXIT_OK


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, choices=list(ARM_TO_GROUP))
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    ap.add_argument("--adamw-impl", choices=["foreach", "fused"], default="foreach")
    ap.add_argument("--ckpt-every-updates", type=int, default=10_000)
    ap.add_argument("--diag-every-updates", type=int, default=20_000)
    ap.add_argument("--benchmark-updates", type=int, default=None)
    ap.add_argument("--test-tiny", action="store_true")
    # Test-only hooks (require --test-tiny; see run_training's guard):
    ap.add_argument("--stop-after-updates", type=int, default=None,
                     help="test-only: stop right after update N finishes (simulated crash)")
    ap.add_argument("--break-resume-perm-seed", action="store_true",
                     help="test-only: deliberately reconstruct epoch permutations with the "
                          "wrong seed, for the resume-determinism xfail test")
    return ap


def main(argv: Optional[list] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        if args.benchmark_updates:
            return run_benchmark(args)
        if not args.run_dir:
            raise ConfigError("--run-dir is required outside --benchmark-updates mode")
        return run_training(args)
    except ConfigError as e:
        print(f"CONFIG ERROR: {e}", file=sys.stderr)
        return EXIT_CONFIG
    except GpuBusy as e:
        print(f"GPU BUSY: {e}", file=sys.stderr)
        return EXIT_GPU_BUSY


if __name__ == "__main__":
    raise SystemExit(main())
