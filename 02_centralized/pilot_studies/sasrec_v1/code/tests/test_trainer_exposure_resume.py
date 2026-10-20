"""Trainer-level exposure/coverage/resume -- REWRITTEN for Gate P2 registry v2
(owner 38h deadline). `train_sasrec.py` now implements schedule S' with FIXED
epochs (no pair barrier, no --max-epochs/--min-epochs/--pair-dir/--pair-partner):
CLI is `--run-id --seed --run-dir --test-tiny --device --tiny-n-train
--tiny-n-vs --tiny-k --tiny-warmup --tiny-planned-epochs --stop-after-updates
--resume ...` (confirmed via `build_parser()` on the landed file, mtime newer
than 2026-09-23T22:40Z, docstring 'Registry v2: schedule S_prime').

Registry v2 dropped R0_ID128/C0/C1: only A0_ID256 (item_only) and A1_RICH256
(rich) remain -- we use A0_ID256, the lighter of the two, on CPU.

The `synthetic_store.py` decision_id collision (sha b8bf3a18...) is fixed;
`test_synthetic_store_decision_ids_are_globally_unique_across_splits` is kept
as a permanent regression guard.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
import pytest

CODE_DIR = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = CODE_DIR / "train_sasrec.py"
S_TESTS_B = CODE_DIR.parent / "tests" / "B"   # $S/tests/B -- our scratch namespace

needs_trainer = pytest.mark.skipif(
    not TRAIN_SCRIPT.exists(),
    reason=f"{TRAIN_SCRIPT} does not exist -- trainer-level CLI tests are ready but cannot run.",
)

REGISTRY_ID = "A0_ID256"    # registry v2: R0_ID128/C0/C1 dropped -- only A0/A1 remain
REGISTRY_ID_RICH = "A1_RICH256"
INTERRUPT_CODES = (0, 3, 4, 5, 6, 99)   # 99 = train_sasrec.py's --stop-after-updates test hook (os._exit(99))


def _fresh_run_dir(name: str) -> Path:
    d = S_TESTS_B / f"{name}-{uuid.uuid4().hex[:8]}"
    if d.exists():
        shutil.rmtree(d)
    return d


def _run_tiny(run_dir: Path, *, run_id: str = REGISTRY_ID, seed: int = 2026, planned_epochs: int = 2,
              extra=()):
    cmd = [sys.executable, str(TRAIN_SCRIPT),
           "--run-id", run_id, "--seed", str(seed), "--run-dir", str(run_dir),
           "--test-tiny", "--device", "cpu",
           "--tiny-n-train", "768", "--tiny-n-vs", "192", "--tiny-k", "53", "--tiny-warmup", "2",
           "--tiny-planned-epochs", str(planned_epochs)]
    cmd += list(extra)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=str(CODE_DIR))


def _load_summary(run_dir: Path) -> dict:
    return json.loads((run_dir / "summary.json").read_text())


# ---------------------------------------------------------------------------
# Regression guard: TRAIN and VALIDATION_SELECT decision ids must never
# collide (the bug that blocked every --test-tiny launch until it was fixed).
# ---------------------------------------------------------------------------
def test_synthetic_store_decision_ids_are_globally_unique_across_splits():
    import paths  # noqa: F401
    from synthetic_store import SyntheticStore

    store = SyntheticStore.build(channels="F0", n_train=64, n_vs=32, K=11, seed=0)
    ds_train = store.decisions("TRAIN", "loss")
    ds_vs = store.decisions("VALIDATION_SELECT", "eval")
    overlap = np.intersect1d(np.asarray(ds_train.decision_id), np.asarray(ds_vs.decision_id))
    assert overlap.shape[0] == 0, (
        f"TRAIN and VALIDATION_SELECT decision_id spaces collide on {overlap.shape[0]} ids "
        f"(e.g. {overlap[:5].tolist()}) -- this was fixed once (sha b8bf3a18...); regression."
    )


# ---------------------------------------------------------------------------
# Epoch permutation digest == np.random.default_rng(seed+epoch_idx0).permutation(N)
# (epoch_idx0 0-based); matched (same-seed) arms -- even DIFFERENT models
# (A0_ID256 vs A1_RICH256) -- must see an IDENTICAL data order, since the
# permutation is a pure function of (seed, epoch_idx0, n) only.
# ---------------------------------------------------------------------------
@needs_trainer
def test_epoch_permutation_matches_default_rng_and_matched_arms_agree():
    run_dir_a = _fresh_run_dir("perm-a")
    run_dir_b = _fresh_run_dir("perm-b")
    res_a = _run_tiny(run_dir_a, run_id=REGISTRY_ID, seed=2026, planned_epochs=2)
    res_b = _run_tiny(run_dir_b, run_id=REGISTRY_ID_RICH, seed=2026, planned_epochs=2)
    assert res_a.returncode == 0, f"stdout={res_a.stdout[-3000:]}\nstderr={res_a.stderr[-3000:]}"
    assert res_b.returncode == 0, f"stdout={res_b.stdout[-3000:]}\nstderr={res_b.stderr[-3000:]}"

    summary_a = _load_summary(run_dir_a)
    summary_b = _load_summary(run_dir_b)
    N = summary_a["config"]["n_train"]
    assert summary_a["epochs_completed"] == summary_b["epochs_completed"] == 2, \
        "matched (same-seed, same-planned-epochs) arms must complete the SAME number of epochs"
    perm_a = summary_a["perm_sha256_by_epoch"]
    perm_b = summary_b["perm_sha256_by_epoch"]
    for epoch_str, sha_a in perm_a.items():
        epoch_idx0 = int(epoch_str) - 1
        expected_perm = np.random.default_rng(2026 + epoch_idx0).permutation(N)
        expected_sha = hashlib.sha256(np.ascontiguousarray(expected_perm, dtype=np.int64).tobytes()).hexdigest()
        assert sha_a == expected_sha, f"epoch {epoch_str}: permutation digest mismatch"
        assert perm_b[epoch_str] == sha_a, \
            f"epoch {epoch_str}: A0 and A1 (same seed) must see an IDENTICAL data order"


# ---------------------------------------------------------------------------
# Coverage exactly N per epoch, including across a mid-epoch resume.
# ---------------------------------------------------------------------------
@needs_trainer
def test_coverage_exactly_N_across_a_mid_epoch_resume():
    run_dir = _fresh_run_dir("resume-cov")
    res1 = _run_tiny(run_dir, planned_epochs=2, extra=("--stop-after-updates", "2"))
    assert res1.returncode in INTERRUPT_CODES, f"stdout={res1.stdout[-3000:]}\nstderr={res1.stderr[-3000:]}"
    res2 = _run_tiny(run_dir, planned_epochs=2, extra=("--resume",))
    assert res2.returncode == 0, f"stdout={res2.stdout[-3000:]}\nstderr={res2.stderr[-3000:]}"
    summary = _load_summary(run_dir)
    for rec in summary["epoch_records"]:
        assert rec["coverage_ok"] is True, f"epoch {rec['epoch']}: coverage_ok is not True"
        assert rec["exposures"] == summary["config"]["n_train"], \
            f"epoch {rec['epoch']}: exposures {rec['exposures']} != N {summary['config']['n_train']}"


# ---------------------------------------------------------------------------
# Interrupted vs uninterrupted run: identical future losses/weights.
# ---------------------------------------------------------------------------
@needs_trainer
def test_interrupted_vs_uninterrupted_future_losses_identical():
    run_dir_clean = _fresh_run_dir("clean")
    run_dir_interrupted = _fresh_run_dir("interrupted")
    res_clean = _run_tiny(run_dir_clean, planned_epochs=2)
    assert res_clean.returncode == 0, f"stdout={res_clean.stdout[-3000:]}\nstderr={res_clean.stderr[-3000:]}"

    res1 = _run_tiny(run_dir_interrupted, planned_epochs=2, extra=("--stop-after-updates", "2"))
    assert res1.returncode in INTERRUPT_CODES, f"stdout={res1.stdout[-3000:]}\nstderr={res1.stderr[-3000:]}"
    res2 = _run_tiny(run_dir_interrupted, planned_epochs=2, extra=("--resume",))
    assert res2.returncode == 0, f"stdout={res2.stdout[-3000:]}\nstderr={res2.stderr[-3000:]}"

    summary_clean = _load_summary(run_dir_clean)
    summary_interrupted = _load_summary(run_dir_interrupted)
    for rec_c, rec_i in zip(summary_clean["epoch_records"], summary_interrupted["epoch_records"]):
        assert rec_c["train_loss_mean"] == pytest.approx(rec_i["train_loss_mean"], rel=1e-4), \
            f"epoch {rec_c['epoch']}: interrupted run's loss diverged from the uninterrupted run"
    assert (summary_clean["checkpoints"]["best_weights"] or {}).get("sha256") == \
        (summary_interrupted["checkpoints"]["best_weights"] or {}).get("sha256")


# ---------------------------------------------------------------------------
# u == epochs * U; the tail is counted (updates_by_phase sums to exactly U).
# ---------------------------------------------------------------------------
@needs_trainer
def test_u_equals_epochs_times_U_and_tail_is_counted():
    run_dir = _fresh_run_dir("u-count")
    res = _run_tiny(run_dir, planned_epochs=1)
    assert res.returncode == 0, f"stdout={res.stdout[-3000:]}\nstderr={res.stderr[-3000:]}"
    summary = _load_summary(run_dir)
    n_train = summary["config"]["n_train"]
    microbatch = summary["config"]["microbatch"]
    U_tiny = -(-int(n_train) // int(microbatch))  # ceil
    assert summary["updates"] == U_tiny * summary["epochs_completed"]
    last_epoch = summary["epoch_records"][-1]
    assert sum(last_epoch["updates_by_phase"].values()) == U_tiny, \
        "the epoch tail must be counted in some LR phase, never dropped"


# ---------------------------------------------------------------------------
# Fixed-epoch stop at exactly the planned/capped E, never earlier -- and the
# equal-epoch assertion (registry v2 pair.rule) holds for matched arms.
# ---------------------------------------------------------------------------
@needs_trainer
def test_fixed_epoch_stop_at_exactly_planned_epochs_never_earlier():
    run_dir = _fresh_run_dir("fixed-stop")
    res = _run_tiny(run_dir, planned_epochs=3)
    assert res.returncode == 0, f"stdout={res.stdout[-3000:]}\nstderr={res.stderr[-3000:]}"
    summary = _load_summary(run_dir)
    assert summary["final_status"] == "COMPLETE"
    assert summary["stop_reason"] == "fixed_epochs"
    assert summary["epochs_completed"] == 3, "must stop at exactly the planned epoch count, not earlier or later"
    assert summary["eval_epochs"] == [3], "only floor epochs (>=3) are evaluated"


@needs_trainer
def test_equal_epoch_assertion_holds_for_matched_arms():
    run_dir_a = _fresh_run_dir("equal-a")
    run_dir_b = _fresh_run_dir("equal-b")
    res_a = _run_tiny(run_dir_a, run_id=REGISTRY_ID, seed=2026, planned_epochs=3)
    res_b = _run_tiny(run_dir_b, run_id=REGISTRY_ID_RICH, seed=2026, planned_epochs=3)
    assert res_a.returncode == 0 and res_b.returncode == 0
    summary_a, summary_b = _load_summary(run_dir_a), _load_summary(run_dir_b)
    assert summary_a["epochs_completed"] == summary_b["epochs_completed"], \
        "matched arms (registry v2: no early stopping) must complete an EQUAL number of epochs"

    # Negative control: an artificially truncated summary (simulating a dead
    # arm) must fail the equal-epoch assertion a supervisor/lock would run.
    def assert_equal_epochs(*counts):
        if len(set(counts)) != 1:
            raise ValueError(f"unequal epoch counts: {counts}")
    assert_equal_epochs(summary_a["epochs_completed"], summary_b["epochs_completed"])  # must not raise
    with pytest.raises(ValueError):
        assert_equal_epochs(summary_a["epochs_completed"], summary_b["epochs_completed"] - 1)
