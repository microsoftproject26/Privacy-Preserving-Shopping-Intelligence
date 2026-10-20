"""Resume-determinism tests -- Task T-K, blocking test T13 (FABLE_PLAN.md
sec. H / P2).

Run on Lobot, GPU 0 only:
    CUDA_VISIBLE_DEVICES=0 /home/jovyan/venvs/ppsi-gpu/bin/python \
        -m pytest code/test_resume.py -v -s

T13 / P2: run a short deterministic training (320 updates), capture the loss
at updates 301-310. Then run the same config but interrupt at update 300,
checkpoint, resume, and capture updates 301-310 again. The two sequences
must match to relative 1e-4 (FP32 deterministic, FABLE_PLAN.md sec. G
Phase 1: allow_tf32=False, cudnn.deterministic=True).

A deliberately-broken variant -- resume that does NOT restore the
data-order RNG (the pass permutation is reconstructed with an unseeded
generator instead of seed 2026+pass_index) -- is included as
`xfail(strict=True)`: it must genuinely diverge, proving this test has the
power to catch that class of bug rather than passing vacuously.

Uses GRU-F0 (F0_item_only) only: T13 tests the CHECKPOINT/RESUME MECHANICS
(data order, RNG restoration, optimizer/model state), not model quality or
any particular feature group, and F0 is by far the cheapest arm to run
twice per test.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

import train

GROUP = "F0_item_only"
TOTAL_UPDATES = 320
INTERRUPT_AT = 300
CAPTURE_UPDATES = tuple(range(301, 311))  # updates 301..310 inclusive
RTOL = 1e-4
CHECKPOINT_EVERY = 300

EXPERIMENTS = train.EXPERIMENTS_DIR


def _fresh_run_dir(name: str) -> Path:
    d = EXPERIMENTS / name
    if d.exists():
        shutil.rmtree(d)
    return d


def _read_losses(run_dir: Path, updates) -> dict:
    path = run_dir / "metrics.jsonl"
    wanted = set(updates)
    out: dict[int, float] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            u = rec.get("update")
            if u in wanted and "loss" in rec:
                out[u] = rec["loss"]
    return out


def _max_rel_diff(a: dict, b: dict, updates) -> tuple[float, dict]:
    diffs = {}
    worst = 0.0
    for u in updates:
        va, vb = a[u], b[u]
        rel = abs(va - vb) / max(abs(va), 1e-12)
        diffs[u] = {"full": va, "resumed": vb, "rel_diff": rel}
        worst = max(worst, rel)
    return worst, diffs


@pytest.fixture(scope="session", autouse=True)
def _gpu_preflight():
    """Two-machine safety policy: verify GPU 0 is free of other users'
    compute processes before any GPU work, and abort (not silently skip)
    if it is not."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available -- T13 requires Lobot GPU 0")
    try:
        r = subprocess.run(
            ["nvidia-smi", "-i", "0", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=30,
        )
    except Exception as exc:
        pytest.skip(f"nvidia-smi preflight failed, refusing to guess GPU state: {exc}")
        return
    pids = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
    if pids:
        raise RuntimeError(
            f"GPU 0 already has running compute processes ({pids}); aborting per the "
            "two-machine safety policy instead of risking another user's job."
        )


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda:0")


@pytest.fixture(scope="module")
def full_run_losses(device):
    """Uninterrupted 320-update reference run."""
    run_dir = _fresh_run_dir("test_resume_full")
    train.run_training(
        group=GROUP, run_dir=run_dir, target_update=TOTAL_UPDATES, device=device,
        resume=False, seed=train.SEED, checkpoint_every=CHECKPOINT_EVERY, run_eval=False,
    )
    losses = _read_losses(run_dir, CAPTURE_UPDATES)
    assert set(losses) == set(CAPTURE_UPDATES), (
        f"missing captured updates in {run_dir}/metrics.jsonl: have {sorted(losses)}"
    )
    return losses


def test_t13_resume_determinism(device, full_run_losses):
    run_dir = _fresh_run_dir("test_resume_interrupted")
    train.run_training(
        group=GROUP, run_dir=run_dir, target_update=INTERRUPT_AT, device=device,
        resume=False, seed=train.SEED, checkpoint_every=CHECKPOINT_EVERY, run_eval=False,
    )
    train.run_training(
        group=GROUP, run_dir=run_dir, target_update=TOTAL_UPDATES, device=device,
        resume=True, seed=train.SEED, checkpoint_every=CHECKPOINT_EVERY, run_eval=False,
    )
    resumed_losses = _read_losses(run_dir, CAPTURE_UPDATES)
    assert set(resumed_losses) == set(CAPTURE_UPDATES), (
        f"missing captured updates in {run_dir}/metrics.jsonl: have {sorted(resumed_losses)}"
    )

    max_rel, diffs = _max_rel_diff(full_run_losses, resumed_losses, CAPTURE_UPDATES)
    (EXPERIMENTS / "t13_comparison.json").write_text(
        json.dumps({"max_rel_diff": max_rel, "per_update": diffs}, indent=2), encoding="utf-8"
    )
    assert max_rel <= RTOL, (
        f"T13 resume determinism FAILED: max relative diff {max_rel} > {RTOL}: {diffs}"
    )


@pytest.mark.xfail(
    strict=True,
    reason="deliberately broken: resume reconstructs the pass permutation with an "
           "unseeded generator instead of restoring seed 2026+pass_index, so the "
           "post-resume row order diverges from the uninterrupted run and losses at "
           "updates 301-310 must NOT match to rel 1e-4.",
)
def test_t13_resume_broken_data_order_xfails(device, full_run_losses):
    run_dir = _fresh_run_dir("test_resume_broken_data_order")
    train.run_training(
        group=GROUP, run_dir=run_dir, target_update=INTERRUPT_AT, device=device,
        resume=False, seed=train.SEED, checkpoint_every=CHECKPOINT_EVERY, run_eval=False,
    )
    train.run_training(
        group=GROUP, run_dir=run_dir, target_update=TOTAL_UPDATES, device=device,
        resume=True, seed=train.SEED, checkpoint_every=CHECKPOINT_EVERY, run_eval=False,
        _break_data_order_restore=True,
    )
    broken_losses = _read_losses(run_dir, CAPTURE_UPDATES)
    assert set(broken_losses) == set(CAPTURE_UPDATES)

    max_rel, diffs = _max_rel_diff(full_run_losses, broken_losses, CAPTURE_UPDATES)
    (EXPERIMENTS / "t13_broken_comparison.json").write_text(
        json.dumps({"max_rel_diff": max_rel, "per_update": diffs}, indent=2), encoding="utf-8"
    )
    # EXPECTED to fail (hence xfail(strict=True) above): the broken variant's
    # row order diverges after the checkpoint, so losses should NOT match.
    assert max_rel <= RTOL, (
        f"broken-variant losses diverged as expected (this assertion is meant to fail): "
        f"max rel diff {max_rel}: {diffs}"
    )
