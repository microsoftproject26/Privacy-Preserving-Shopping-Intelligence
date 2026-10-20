"""Training engine -- Task T-K.

Implements the training engine specified in the T-K task brief: AdamW with
two explicit param groups (weight matrices decayed, biases/LayerNorm not),
100-update linear warmup then constant LR, global grad-norm clip, exact
full-catalogue CE via `objective.exact_ce_loss` reduced through
`objective.LossAccumulator` (sum/sum, divide once -- never a per-microbatch
mean), a deterministic one-permutation-per-pass data order (seed
`2026 + pass_index`), full checkpoint/resume (model, optimizer, RNG states,
exposure counters, manifest/version fixity), structured per-update JSON
logging, and a per-pass VS-cohort evaluation hook.

Implementation only -- this module does NOT launch the pilot. `main()` is a
CLI for later, explicitly-authorised pilot runs; this task only exercises
`run_training()` directly, with tiny `target_update` values, from
`test_resume.py` (T13).

Run on Lobot only, pinned to GPU 0 (CUDA_VISIBLE_DEVICES=0).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

WORKSPACE = Path(
    "/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/gru_v1"
)
REPO_ROOT = WORKSPACE.parent.parent  # the shared git repo (Privacy-Preserving-Shopping-Intelligence)
CODE_DIR = WORKSPACE / "code"
VENDOR_DIR = CODE_DIR / "vendor"
ARTIFACTS = WORKSPACE / "artifacts"
EXPERIMENTS_DIR = WORKSPACE / "experiments" / "gru"
CACHE_TRAIN = WORKSPACE / "cache" / "train"
CACHE_VS = WORKSPACE / "cache" / "vs"
DATASET_ROOT = Path("/home/jovyan/shared_data/rees46_nextitem_v1")

for _p in (str(CODE_DIR), str(VENDOR_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

import model as model_mod  # noqa: E402
import objective as objective_mod  # noqa: E402
from evaluator import evaluate  # noqa: E402
from window_cache import CacheReader  # noqa: E402

# ---------------------------------------------------------------------------
# Fixed / measured configuration (BENCHMARK.json, FABLE_PLAN.md sec. C/F/G).
# Do NOT re-derive or change any of these -- they are pre-registered.
# ---------------------------------------------------------------------------
SEED = 2026
TRAIN_ROWS = 1_000_000
MICROBATCH = 256
ACCUMULATION = 1
LR = 1e-3
BETAS = (0.9, 0.999)
EPS = 1e-8
WEIGHT_DECAY = 1e-5
WARMUP_UPDATES = 100
GRAD_CLIP_NORM = 1.0
EVAL_K = 20
EVAL_CHUNK = 512
OOV_TOKEN = 2
PAD_TOKEN = 0

VOCAB_SIZES = {
    "item_tokens": 158489,
    "category_tokens": 663,
    "brand_tokens": 3956,
    "main_category_tokens": 16,
    "daypart_tokens": 8,
    "weekday_tokens": 10,
}
VERSION_HASH = "860e5a1c3a81b8e4bbfeff4f245fa37638a43e4f260df179585964d2d4da8dbd"
CATEGORICAL_CHANNELS = (
    "category_tokens", "brand_tokens", "main_category_tokens",
    "daypart_tokens", "weekday_tokens",
)
GROUP_CHOICES = (
    "F0_item_only", "F1_core_categorical", "F2_time_context",
    "F3_user_session_context", "F4_rich_side_information",
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_log(run_dir: Path, msg: str) -> None:
    """Print AND append to `<run_dir>/logs/train.log` (the run's human-readable
    log; `metrics.jsonl` is the machine-readable one)."""
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    log_path = run_dir / "logs" / "train.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def set_determinism(seed: int = SEED) -> None:
    """Seed 2026, deterministic (task brief): python/numpy/torch RNGs seeded,
    cuDNN deterministic, TF32 disabled (FP32 pilot phase, FABLE_PLAN.md sec.
    G Phase 1). Also opts into deterministic algorithms (warn-only, so an
    op with no deterministic kernel warns instead of hard-failing) -- needed
    because CUDA embedding-backward is otherwise nondeterministic under
    duplicate indices (very likely within a 50-token window), which would
    silently break T13's 1e-4 resume tolerance.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:  # pragma: no cover -- older torch signature
        torch.use_deterministic_algorithms(True)


def rng_state_snapshot() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


# ---------------------------------------------------------------------------
# Data order: one permutation of the manifest per pass, seed 2026+pass_index
# ---------------------------------------------------------------------------

def pass_permutation(
    pass_index: int,
    n_rows: int = TRAIN_ROWS,
    seed_base: int = SEED,
    *,
    _unseeded_for_test: bool = False,
) -> np.ndarray:
    """One permutation of manifest row indices for `pass_index`, seeded
    `seed_base + pass_index` -- a pure function of `pass_index` alone, so F0
    and F4 arms (built/run independently) see IDENTICAL order, and so a
    correct resume can reconstruct the in-progress pass's order from nothing
    but the checkpointed `pass_index` (no separate RNG state needs saving
    for this to be exact).

    `_unseeded_for_test`: TEST-ONLY escape hatch used by test_resume.py's
    deliberately-broken resume variant (T13's required xfail). Never set
    True outside a test -- it discards the `seed_base + pass_index` seed in
    favour of OS entropy, simulating a "resume forgot to restore/recompute
    the data-order RNG" bug so the row order (and hence the loss sequence)
    diverges from the uninterrupted run after the checkpoint.
    """
    rng = np.random.default_rng(None if _unseeded_for_test else (seed_base + pass_index))
    return rng.permutation(n_rows)


# ---------------------------------------------------------------------------
# Model / optimizer
# ---------------------------------------------------------------------------

def load_product_idx_of_class() -> np.ndarray:
    import pyarrow.parquet as pq
    from rees46_eda.loaders import detect_layout

    layout = detect_layout(DATASET_ROOT)
    if layout != "frozen":
        raise RuntimeError(f"expected frozen layout, got {layout!r}")
    cat = pq.read_table(DATASET_ROOT / "mappings" / "catalogue.parquet")
    return cat["product_idx"].to_numpy(zero_copy_only=False)


def build_model_for_group(group: str, poc: np.ndarray, device: torch.device):
    model = model_mod.build_model(group, VOCAB_SIZES, product_idx_of_class=poc)
    return model.to(device)


def build_param_groups(model, weight_decay: float = WEIGHT_DECAY) -> list[dict]:
    """Two explicit param groups: weight matrices (ndim>=2 -- every Linear /
    Embedding / GRU weight matrix) get `weight_decay`; biases AND LayerNorm
    params (ndim==1 in every case here, including LayerNorm's weight, which
    is a 1-D scale vector, not a matrix) get 0. Same `ndim>=2` rule already
    used by `benchmark.py:make_optimizer` -- kept consistent rather than
    reinvented.
    """
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 else no_decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_optimizer(model) -> torch.optim.AdamW:
    return torch.optim.AdamW(build_param_groups(model), lr=LR, betas=BETAS, eps=EPS)


def lr_at_update(update_number_1based: int, base_lr: float = LR,
                  warmup: int = WARMUP_UPDATES) -> float:
    """100-update linear warmup then constant. `update_number_1based` is the
    ordinal (1..) of the update that is about to run (or just ran)."""
    if update_number_1based <= warmup:
        return base_lr * update_number_1based / warmup
    return base_lr


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = lr


def build_named_kwargs(batch: dict, model) -> dict:
    """Named forward() kwargs for `model`, derived from the model's OWN
    resolved spec (`model._categorical_channels` / `.numeric_dim` /
    `.flags_dim` / `.use_user_context`) rather than a hard-coded per-group
    table -- generalises to any of F0..F4 without duplicating
    `benchmark.py`'s group-name-keyed `GROUP_CHANNELS`.
    """
    kw = dict(
        item_tokens=batch["item_tokens"], lengths=batch["lengths"],
        attention_mask=batch["attention_mask"],
    )
    for ch in CATEGORICAL_CHANNELS:
        if ch in model._categorical_channels:
            kw[ch] = batch[ch]
    if model.numeric_dim > 0:
        kw["event_numeric_features"] = batch["event_numeric_features"]
    if model.flags_dim > 0:
        kw["event_quality_flags"] = batch["event_quality_flags"]
    if model.use_user_context:
        kw["user_context"] = batch["user_context"]
        kw["user_context_masks"] = batch["user_context_masks"]
    return kw


# ---------------------------------------------------------------------------
# Run-directory scaffolding: config.yaml, environment.json, git_state.json,
# hardware.json, metrics.jsonl, summary.json, checkpoints/, logs/
# ---------------------------------------------------------------------------

def gather_environment_info(device: Optional[torch.device]) -> dict:
    info: dict[str, Any] = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if torch.cuda.is_available():
        idx = device.index if (device is not None and device.index is not None) else 0
        info["gpu_name"] = torch.cuda.get_device_name(idx)
        info["gpu_count"] = torch.cuda.device_count()
    return info


def gather_git_state() -> dict:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout
        return {"repo_root": str(REPO_ROOT), "head": head, "dirty": bool(status.strip())}
    except Exception as exc:  # pragma: no cover -- defensive only
        return {"repo_root": str(REPO_ROOT), "head": None, "dirty": None, "error": str(exc)}


def gather_hardware_info() -> dict:
    info: dict[str, Any] = {"cpu_count": os.cpu_count()}
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=30,
        )
        info["nvidia_smi_gpus"] = [ln.strip() for ln in r.stdout.strip().splitlines()]
    except Exception as exc:  # pragma: no cover -- defensive only
        info["nvidia_smi_error"] = str(exc)
    return info


def init_run_dir(run_dir: Path, config: dict, device: Optional[torch.device]) -> None:
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (run_dir / "environment.json").write_text(
        json.dumps(gather_environment_info(device), indent=2), encoding="utf-8")
    (run_dir / "git_state.json").write_text(
        json.dumps(gather_git_state(), indent=2), encoding="utf-8")
    (run_dir / "hardware.json").write_text(
        json.dumps(gather_hardware_info(), indent=2), encoding="utf-8")


def log_metric(run_dir: Path, record: dict) -> None:
    """Append one JSON object to metrics.jsonl. Only aggregate numeric
    fields / floats ever go in here -- never a raw decision_id/user_id."""
    with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


def write_summary(run_dir: Path, summary: dict) -> None:
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str),
                                           encoding="utf-8")


def load_pilot_manifest_meta() -> dict:
    return json.loads((ARTIFACTS / "PILOT_MANIFEST.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def _atomic_torch_save(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)  # atomic on POSIX -- readers never see a partial file


def save_checkpoint(
    run_dir: Path, name: str, *, model, optimizer, pass_index: int, offset_in_pass: int,
    update_number: int, consumed_decisions: int, manifest_sha256: str, version_hash: str,
    config: dict, best_metric: Optional[float],
) -> Path:
    """Save model, optimizer, RNG states (python/numpy/torch/cuda),
    consumed_decisions, pass index, manifest sha256, dataset version_hash,
    and config -- everything `load_checkpoint`+resume needs to continue the
    exact data order and exact stochastic state.
    """
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": rng_state_snapshot(),
        "pass_index": pass_index,
        "offset_in_pass": offset_in_pass,
        "update_number": update_number,
        "consumed_decisions": consumed_decisions,
        "manifest_sha256": manifest_sha256,
        "version_hash": version_hash,
        "config": config,
        "best_metric": best_metric,
        "saved_at": time.time(),
    }
    path = run_dir / "checkpoints" / f"{name}.pt"
    _atomic_torch_save(ckpt, path)
    return path


def load_checkpoint(run_dir: Path, name: str = "last", map_location: str = "cpu") -> dict:
    path = run_dir / "checkpoints" / f"{name}.pt"
    if not path.exists():
        raise FileNotFoundError(f"no checkpoint at {path}")
    return torch.load(path, map_location=map_location, weights_only=False)


# ---------------------------------------------------------------------------
# VS-cohort evaluation hook (after each pass)
# ---------------------------------------------------------------------------

def eval_user_key_array() -> np.ndarray:
    """`eval_user_key` joined from validation_select_cohort.parquet -- cache
    row order == manifest row order (window_cache.py builds cache/vs from
    exactly this manifest, in manifest row order), so a plain positional
    join is correct."""
    import pyarrow.parquet as pq
    t = pq.read_table(ARTIFACTS / "validation_select_cohort.parquet", columns=["eval_user_key"])
    return np.array(t["eval_user_key"].to_numpy(zero_copy_only=False), copy=True)


def run_vs_evaluation(model, device: torch.device, *, chunk_size: int = EVAL_CHUNK) -> dict:
    reader = CacheReader(CACHE_VS)
    n = reader.header["row_count"]
    eval_user_key = eval_user_key_array()
    if eval_user_key.shape[0] != n:
        raise ValueError(
            "validation_select_cohort.parquet row count != cache/vs row count "
            "(cache row order must equal manifest row order)"
        )
    all_rows = reader.read(np.arange(n))
    rows = {
        "target_class": all_rows["target_class"],
        "eval_mask": all_rows["eval_mask"],
        "eval_user_key": torch.from_numpy(eval_user_key),
        "client_id": all_rows["client_id"],
    }

    was_training = model.training
    model.eval()

    def scores_fn():
        with torch.no_grad():
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                idx = np.arange(start, end)
                batch = reader.read(idx)
                batch_gpu = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                kw = build_named_kwargs(batch_gpu, model)
                logits = model(**kw)
                yield logits.detach().to(torch.float32).cpu()

    result = evaluate(scores_fn, rows, k=EVAL_K, chunk_size=8192)
    if was_training:
        model.train()
    return result


def selection_metric(eval_result: dict) -> float:
    """Client-Macro MRR@20 -- macro (grouped by eval_user_key, i.e.
    per-evaluated-client) MRR@20 on the end-to-end denominator, matching
    FABLE_PLAN.md sec. F's "Micro+Macro MRR@20, end-to-end denominator"
    convention used for the GO-1/GO-2 baseline comparisons. See the T-K/L
    report's "unresolved concerns" for the ambiguity this resolves."""
    return float(eval_result["end_to_end"]["macro"]["mrr@20"])


# ---------------------------------------------------------------------------
# Core training loop
# ---------------------------------------------------------------------------

def run_training(
    *,
    group: str,
    run_dir: Path,
    target_update: int,
    device: torch.device,
    resume: bool = False,
    seed: int = SEED,
    microbatch: int = MICROBATCH,
    accumulation: int = ACCUMULATION,
    checkpoint_every: int = 200,
    train_rows: int = TRAIN_ROWS,
    run_eval: bool = True,
    _break_data_order_restore: bool = False,
) -> dict:
    """Run (or resume+continue) training up to global update `target_update`
    (1-based: after this call, `update_number == target_update`).

    Exposure/data order: `consumed_decisions` counts every row DRAWN from
    the manifest (not just loss-contributing rows), maintained exactly;
    over a full run this reaches `passes * train_rows`. Rows are drawn in
    `microbatch`-sized slices of `pass_permutation(pass_index)`, advancing
    `offset_in_pass`; crossing `train_rows` starts the next pass with its
    own permutation (seed `seed + pass_index`).

    Loss reduction: `objective.exact_ce_loss` gives `(loss_sum, n)` per
    microbatch; with accumulation==1 this is a single `LossAccumulator.add`
    per update, but the mechanics generalise (raw-sum backward, THEN divide
    every gradient by the accumulated `n` once, THEN clip) -- never a
    per-microbatch mean.

    `_break_data_order_restore` is a TEST-ONLY hook -- see
    `pass_permutation`'s docstring and test_resume.py's xfail test. It must
    never be set True from `main()`/the CLI.
    """
    if accumulation != 1:
        raise NotImplementedError(
            "only accumulation=1 is measured (BENCHMARK.json: microbatch 256, "
            "accumulation 1, effective batch 256); the sum/sum grad-accumulation "
            "path below is written to generalise but is untested for accumulation>1"
        )

    manifest_meta = load_pilot_manifest_meta()
    manifest_sha256 = manifest_meta["train_manifest"]["sha256"]
    version_hash = manifest_meta["version_hash"]
    if version_hash != VERSION_HASH:
        raise RuntimeError(
            f"PILOT_MANIFEST.json version_hash {version_hash!r} != expected {VERSION_HASH!r}"
        )

    config = {
        "group": group, "seed": seed, "microbatch": microbatch, "accumulation": accumulation,
        "lr": LR, "betas": list(BETAS), "eps": EPS, "weight_decay": WEIGHT_DECAY,
        "warmup_updates": WARMUP_UPDATES, "grad_clip_norm": GRAD_CLIP_NORM,
        "train_rows": train_rows, "manifest_sha256": manifest_sha256, "version_hash": version_hash,
        "vocab_sizes": VOCAB_SIZES,
    }

    checkpoints_last = run_dir / "checkpoints" / "last.pt"

    if resume:
        if not checkpoints_last.exists():
            raise FileNotFoundError(f"resume=True but no checkpoint at {checkpoints_last}")
        ckpt = load_checkpoint(run_dir, "last")
        if ckpt["manifest_sha256"] != manifest_sha256 or ckpt["version_hash"] != version_hash:
            raise RuntimeError(
                "checkpoint manifest_sha256/version_hash mismatch -- refusing to resume "
                "against a different dataset/manifest than the one this checkpoint was "
                f"trained on (checkpoint={ckpt['manifest_sha256']}/{ckpt['version_hash']}, "
                f"current={manifest_sha256}/{version_hash})"
            )
        set_determinism(seed)  # backend flags; RNG *state* is overwritten below
        poc = load_product_idx_of_class()
        model = build_model_for_group(group, poc, device)
        optimizer = build_optimizer(model)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        restore_rng_state(ckpt["rng_state"])
        pass_index = ckpt["pass_index"]
        offset_in_pass = ckpt["offset_in_pass"]
        update_number = ckpt["update_number"]
        consumed_decisions = ckpt["consumed_decisions"]
        best_metric = ckpt.get("best_metric")
        perm = pass_permutation(pass_index, train_rows, seed,
                                 _unseeded_for_test=_break_data_order_restore)
        run_log(run_dir, f"resumed run_dir={run_dir} update={update_number} pass={pass_index} "
                          f"offset={offset_in_pass} consumed={consumed_decisions} "
                          f"broken_data_order={_break_data_order_restore}")
    else:
        if checkpoints_last.exists():
            raise FileExistsError(
                f"{checkpoints_last} already exists; refusing to overwrite a non-empty "
                "run without resume=True"
            )
        init_run_dir(run_dir, config, device)
        set_determinism(seed)
        poc = load_product_idx_of_class()
        model = build_model_for_group(group, poc, device)
        optimizer = build_optimizer(model)
        pass_index = 0
        offset_in_pass = 0
        update_number = 0
        consumed_decisions = 0
        best_metric = None
        perm = pass_permutation(pass_index, train_rows, seed)
        run_log(run_dir, f"fresh run_dir={run_dir} group={group} target_update={target_update}")

    reader = CacheReader(CACHE_TRAIN)
    model.train()

    while update_number < target_update:
        if offset_in_pass >= train_rows:
            pass_index += 1
            offset_in_pass = 0
            perm = pass_permutation(pass_index, train_rows, seed)

        end = min(offset_in_pass + microbatch, train_rows)
        row_idx = perm[offset_in_pass:end]
        b_size = int(row_idx.shape[0])

        t0 = time.perf_counter()
        raw = reader.read(row_idx)
        assert bool(raw["loss_mask"].all()), (
            "loss_mask.all() failed on a TRAIN batch -- the manifest is pre-filtered to "
            "loss_mask==true rows; a false row here means a pipeline bug upstream, not "
            "something this trainer should paper over"
        )
        batch_gpu = {k: v.to(device, non_blocking=True) for k, v in raw.items()}
        kw = build_named_kwargs(batch_gpu, model)
        target_class = batch_gpu["target_class"]
        loss_mask = batch_gpu["loss_mask"]

        optimizer.zero_grad(set_to_none=True)
        logits = model(**kw)
        loss_sum, n = objective_mod.exact_ce_loss(logits, target_class, loss_mask)
        loss_sum.backward()  # raw-SUM gradients, unscaled

        acc = objective_mod.LossAccumulator().add(loss_sum.detach(), n.detach())
        total_n = acc.n
        total_n_val = float(total_n.item() if hasattr(total_n, "item") else total_n)
        if total_n_val <= 0:
            raise RuntimeError(
                "update with zero loss-contributing rows -- unexpected given the "
                "pre-filtered TRAIN manifest"
            )
        # Divide ONCE: d(loss_sum/n)/dw == (1/n) * d(loss_sum)/dw exactly, so
        # scaling the already-accumulated raw-sum gradient by 1/total_n here
        # is mathematically identical to (and, with accumulation==1, the same
        # computation as) backward()-ing loss_sum/n directly -- but this form
        # generalises correctly to accumulation>1 (sum-then-divide-once,
        # never a per-microbatch mean).
        for p in model.parameters():
            if p.grad is not None:
                p.grad.div_(total_n_val)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

        update_number += 1
        lr = lr_at_update(update_number)
        set_lr(optimizer, lr)
        optimizer.step()

        loss_value = float(acc.mean().item())
        offset_in_pass = end
        consumed_decisions += b_size
        dt = time.perf_counter() - t0

        log_metric(run_dir, {
            "update": update_number, "pass_index": pass_index, "loss": loss_value,
            "lr": lr, "grad_norm": float(grad_norm), "batch_size": b_size,
            "consumed_decisions": consumed_decisions, "seconds": dt,
        })

        pass_ended = offset_in_pass >= train_rows
        do_checkpoint = (
            (checkpoint_every > 0 and update_number % checkpoint_every == 0)
            or pass_ended or (update_number == target_update)
        )
        if do_checkpoint:
            save_checkpoint(
                run_dir, "last", model=model, optimizer=optimizer, pass_index=pass_index,
                offset_in_pass=offset_in_pass, update_number=update_number,
                consumed_decisions=consumed_decisions, manifest_sha256=manifest_sha256,
                version_hash=version_hash, config=config, best_metric=best_metric,
            )

        if pass_ended and run_eval:
            eval_result = run_vs_evaluation(model, device)
            metric = selection_metric(eval_result)
            run_log(run_dir, f"pass {pass_index} complete: consumed={consumed_decisions} "
                              f"client_macro_mrr@20={metric:.6f}")
            log_metric(run_dir, {
                "update": update_number, "pass_index": pass_index,
                "eval_client_macro_mrr@20": metric, "eval_full": eval_result,
            })
            if best_metric is None or metric > best_metric:
                best_metric = metric
                save_checkpoint(
                    run_dir, "best", model=model, optimizer=optimizer, pass_index=pass_index,
                    offset_in_pass=offset_in_pass, update_number=update_number,
                    consumed_decisions=consumed_decisions, manifest_sha256=manifest_sha256,
                    version_hash=version_hash, config=config, best_metric=best_metric,
                )

    summary = {
        "group": group, "update_number": update_number, "pass_index": pass_index,
        "offset_in_pass": offset_in_pass, "consumed_decisions": consumed_decisions,
        "best_metric": best_metric, "target_update": target_update,
    }
    write_summary(run_dir, summary)
    return summary


# ---------------------------------------------------------------------------
# CLI (for later, explicitly-authorised pilot runs -- NOT invoked by this task)
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="T-K training engine. Implementation only -- this CLI is not "
                     "invoked to run the F0/F4 pilot itself; that is a separate, "
                     "explicitly-authorised task."
    )
    ap.add_argument("--group", required=True, choices=GROUP_CHOICES)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--checkpoint-every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available on this worker")
    device = torch.device("cuda:0")

    n_updates_per_pass = -(-TRAIN_ROWS // MICROBATCH)  # ceil
    target_update = args.passes * n_updates_per_pass
    run_dir = EXPERIMENTS_DIR / args.run_id
    summary = run_training(
        group=args.group, run_dir=run_dir, target_update=target_update, device=device,
        resume=args.resume, seed=args.seed, checkpoint_every=args.checkpoint_every,
    )
    log(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
