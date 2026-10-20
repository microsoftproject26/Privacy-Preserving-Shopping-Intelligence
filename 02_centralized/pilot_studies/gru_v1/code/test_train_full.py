"""Correctness tests for `train_full.py` (Wave-1 Sonnet B).

Run on Lobot, GPU 0 only, scratch under `$W/tests_full/B/`:
    CUDA_VISIBLE_DEVICES=0 ~/venvs/ppsi-gpu/bin/python -m pytest code/test_train_full.py -v -s

Uses `fakes_for_tests.py` (a pilot-cache-backed `FakeStore` + CPU
`fake_evaluate_model`) exclusively -- never the real `full_store.py` /
`eval_full.py` (owners: Sonnet A / C). Every test cleans up its own
checkpoints (each ~250-300MB, the real production-size model) immediately
so the scratch dir never holds more than ~1-2 runs' worth at once.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
import torch

import train_full
import train as pilot_train  # frozen pilot module
import model as model_mod  # frozen pilot module
import objective as objective_mod  # frozen pilot module
import synthetic_memorisation as synmem  # frozen pilot module (imported, not edited)
from fakes_for_tests import FakeStore

SCRATCH = train_full.TESTS_FULL_DIR / "B"
ARMS = ("F0_item_only_h256", "F4_rich_side_information_h256")


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def _gpu_preflight():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available -- these tests require Lobot GPU 0")
    try:
        train_full.check_gpu_busy("0")
    except train_full.GpuBusy as e:
        raise RuntimeError(f"refusing to start: {e}")


@pytest.fixture(scope="session", autouse=True)
def _scratch_budget_report():
    yield
    total = 0
    if SCRATCH.exists():
        for p in SCRATCH.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
    print(f"\n[test_train_full] final $W/tests_full/B size: {total / (1024**2):.1f} MiB")
    assert total <= 1 * 1024 ** 3, f"scratch dir exceeds 1 GiB at session end: {total} bytes"


def _fresh_dir(name: str) -> Path:
    d = SCRATCH / name
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cleanup(*dirs: Path) -> None:
    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)


def _read_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _update_debug_losses(run_dir: Path, updates) -> dict:
    wanted = set(updates)
    out = {}
    for rec in _read_jsonl(run_dir / "metrics.jsonl"):
        if rec.get("type") == "update_debug" and rec.get("update") in wanted:
            out[rec["update"]] = rec["loss"]
    return out


def _max_rel_diff(a: dict, b: dict, updates) -> float:
    worst = 0.0
    for u in updates:
        va, vb = a[u], b[u]
        rel = abs(va - vb) / max(abs(va), 1e-12)
        worst = max(worst, rel)
    return worst


def _run(argv: list) -> int:
    return train_full.main(argv)


# ---------------------------------------------------------------------------
# 9. Supervisor: SIGKILL mid-epoch -> auto-resume -> finishes
#
# Deliberately placed FIRST in file order (before any test that touches CUDA
# in-process): this test's own trainer runs in a SEPARATE OS subprocess, and
# its startup GPU_BUSY check (`check_gpu_busy`) correctly treats any OTHER
# process already holding a CUDA context on GPU 0 as foreign -- including
# THIS pytest process itself, once an earlier in-process test has built a
# model on the GPU. Running this test before any such in-process CUDA use
# avoids that (legitimate) self-collision; it is not a trainer bug.
# ---------------------------------------------------------------------------

def test_supervisor_sigkill_mid_epoch_resumes_and_finishes():
    arm = "F0_item_only_h256"
    run_dir = _fresh_dir("supervisor_sigkill")
    try:
        supervise_sh = str(train_full.CODE_DIR / "supervise_full.sh")
        extra = ("--test-tiny --max-epochs 1 --ckpt-every-updates 1 "
                 "--diag-every-updates 1000000")
        env = dict(os.environ)
        env["TRAIN_FULL_TEST_UPDATE_SLEEP"] = "1.0"  # slow enough for the poll loop below to catch it mid-epoch
        proc = subprocess.Popen(
            ["bash", supervise_sh, arm, "0", str(run_dir), extra],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
        )
        try:
            pid_file = run_dir / "logs" / "train_full.pid"
            deadline = time.time() + 60
            child_pid = None
            while time.time() < deadline:
                if pid_file.exists():
                    try:
                        child_pid = int(pid_file.read_text().strip())
                        break
                    except Exception:
                        pass
                time.sleep(0.5)
            assert child_pid is not None, "train_full.py child pid never appeared"

            # Wait for an actual, fully-written (atomically renamed) checkpoint
            # to exist -- not just a log line -- so the kill below is
            # guaranteed to land after update 1 has a valid, resumable
            # checkpoint on disk, never mid-write of the very first one.
            ckpt_path = run_dir / "checkpoints" / "latest.pt"
            deadline2 = time.time() + 60
            progressed = False
            while time.time() < deadline2:
                if ckpt_path.exists():
                    progressed = True
                    break
                time.sleep(0.5)
            assert progressed, "no checkpoint appeared before the kill deadline"

            os.kill(child_pid, signal.SIGKILL)
            rc = proc.wait(timeout=180)
        except Exception:
            proc.kill()
            raise
        assert rc == 0, f"supervisor did not report overall success: rc={rc}"
        assert (run_dir / "SUPERVISOR_DONE").exists()
        ckpt = torch.load(run_dir / "checkpoints" / "latest.pt", map_location="cpu", weights_only=False)
        assert ckpt["resume_count"] >= 1, "supervisor should have resumed at least once"
    finally:
        _cleanup(run_dir)


# ---------------------------------------------------------------------------
# 1. Resume determinism (launch gate 6), both AdamW impls, + broken xfail
# ---------------------------------------------------------------------------

TOTAL_UPDATES = 20
INTERRUPT_AT = 10
CAPTURE_UPDATES = (17, 18, 19, 20)  # crosses the epoch0/epoch1 boundary + its eval (n_train=4096 -> 16/epoch)
RTOL = 1e-4


@pytest.mark.parametrize("impl", ["foreach", "fused"])
def test_resume_determinism(impl):
    arm = "F0_item_only_h256"
    full_dir = _fresh_dir(f"resume_full_{impl}")
    interrupted_dir = _fresh_dir(f"resume_interrupted_{impl}")
    try:
        rc = _run(["--arm", arm, "--run-dir", str(full_dir), "--test-tiny",
                    "--adamw-impl", impl, "--ckpt-every-updates", "5",
                    "--diag-every-updates", "1000000",
                    "--stop-after-updates", str(TOTAL_UPDATES)])
        assert rc == 0
        full_losses = _update_debug_losses(full_dir, CAPTURE_UPDATES)
        assert set(full_losses) == set(CAPTURE_UPDATES)
    finally:
        _cleanup(full_dir)

    try:
        rc1 = _run(["--arm", arm, "--run-dir", str(interrupted_dir), "--test-tiny",
                     "--adamw-impl", impl, "--ckpt-every-updates", "5",
                     "--diag-every-updates", "1000000",
                     "--stop-after-updates", str(INTERRUPT_AT)])
        assert rc1 == 0
        rc2 = _run(["--arm", arm, "--run-dir", str(interrupted_dir), "--test-tiny",
                     "--resume", "--adamw-impl", impl, "--ckpt-every-updates", "5",
                     "--diag-every-updates", "1000000",
                     "--stop-after-updates", str(TOTAL_UPDATES)])
        assert rc2 == 0
        resumed_losses = _update_debug_losses(interrupted_dir, CAPTURE_UPDATES)
        assert set(resumed_losses) == set(CAPTURE_UPDATES)

        max_rel = _max_rel_diff(full_losses, resumed_losses, CAPTURE_UPDATES)
        assert max_rel <= RTOL, f"resume determinism FAILED ({impl}): max rel diff {max_rel}"
    finally:
        _cleanup(interrupted_dir)


@pytest.mark.xfail(
    strict=True,
    reason="deliberately broken: --break-resume-perm-seed reconstructs the epoch "
           "permutation with the wrong seed on resume, so post-resume losses must "
           "NOT match the uninterrupted run -- proves the determinism test has power.",
)
def test_resume_broken_perm_seed_xfails():
    arm = "F0_item_only_h256"
    impl = "foreach"
    full_dir = _fresh_dir("resume_full_broken_ref")
    broken_dir = _fresh_dir("resume_broken")
    try:
        rc = _run(["--arm", arm, "--run-dir", str(full_dir), "--test-tiny",
                    "--adamw-impl", impl, "--ckpt-every-updates", "5",
                    "--diag-every-updates", "1000000",
                    "--stop-after-updates", str(TOTAL_UPDATES)])
        assert rc == 0
        full_losses = _update_debug_losses(full_dir, CAPTURE_UPDATES)
    finally:
        _cleanup(full_dir)

    try:
        rc1 = _run(["--arm", arm, "--run-dir", str(broken_dir), "--test-tiny",
                     "--adamw-impl", impl, "--ckpt-every-updates", "5",
                     "--diag-every-updates", "1000000",
                     "--stop-after-updates", str(INTERRUPT_AT)])
        assert rc1 == 0
        rc2 = _run(["--arm", arm, "--run-dir", str(broken_dir), "--test-tiny",
                     "--resume", "--adamw-impl", impl, "--ckpt-every-updates", "5",
                     "--diag-every-updates", "1000000",
                     "--break-resume-perm-seed",
                     "--stop-after-updates", str(TOTAL_UPDATES)])
        assert rc2 == 0
        broken_losses = _update_debug_losses(broken_dir, CAPTURE_UPDATES)

        max_rel = _max_rel_diff(full_losses, broken_losses, CAPTURE_UPDATES)
        # EXPECTED to fail (hence xfail(strict=True) above): the broken
        # variant's row order diverges after the resume point.
        assert max_rel <= RTOL, f"broken-variant diverged as expected: max rel diff {max_rel}"
    finally:
        _cleanup(broken_dir)


def test_test_tiny_refuses_run_dir_outside_tests_full(tmp_path):
    outside_dir = tmp_path / "not_under_tests_full"
    rc = _run(["--arm", "F0_item_only_h256", "--run-dir", str(outside_dir), "--test-tiny",
                "--stop-after-updates", "1"])
    assert rc == train_full.EXIT_CONFIG
    assert not outside_dir.exists() or not any(outside_dir.iterdir())


def test_resume_refuses_adamw_impl_mismatch():
    arm = "F0_item_only_h256"
    run_dir = _fresh_dir("resume_adamw_mismatch")
    try:
        rc = _run(["--arm", arm, "--run-dir", str(run_dir), "--test-tiny",
                    "--adamw-impl", "foreach", "--ckpt-every-updates", "2",
                    "--diag-every-updates", "1000000", "--stop-after-updates", "2"])
        assert rc == 0
        rc2 = _run(["--arm", arm, "--run-dir", str(run_dir), "--test-tiny",
                     "--resume", "--adamw-impl", "fused",
                     "--diag-every-updates", "1000000", "--stop-after-updates", "4"])
        assert rc2 == train_full.EXIT_CONFIG
    finally:
        _cleanup(run_dir)


# ---------------------------------------------------------------------------
# 2. Checkpoint rotation safety
# ---------------------------------------------------------------------------

def _dummy_ckpt_obj(update: int) -> dict:
    return {
        "model_state_dict": {"w": torch.zeros(4)}, "optimizer_state_dict": {},
        "rng_state": pilot_train.rng_state_snapshot(), "epoch": 0, "offset_in_epoch": 0,
        "update": update, "eval_pending": False, "lr_state": {"lr_decayed": False, "lr_decay_epoch": None},
        "patience": 0, "selection_state": {}, "best_metric": None, "config": {},
        "dataset_version_hash": "x", "protocol_sha256": "y", "code_sha256": {}, "resume_count": 0,
    }


def test_checkpoint_rotation_survives_injected_failure(monkeypatch):
    ckpt_dir = _fresh_dir("ckpt_rotation") / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    try:
        train_full.write_checkpoint_atomic(ckpt_dir, "latest", _dummy_ckpt_obj(1))
        good_bytes_before = (ckpt_dir / "latest.pt").read_bytes()

        real_validate = train_full._validate_checkpoint_file
        calls = {"n": 0}

        def flaky_validate(path, expected_update):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ValueError("injected validation failure")
            return real_validate(path, expected_update)

        monkeypatch.setattr(train_full, "_validate_checkpoint_file", flaky_validate)
        with pytest.raises(ValueError):
            train_full.write_checkpoint_atomic(ckpt_dir, "latest", _dummy_ckpt_obj(2))

        # latest.pt must be untouched (byte-identical) and still loadable.
        assert (ckpt_dir / "latest.pt").read_bytes() == good_bytes_before
        assert not (ckpt_dir / "latest.pt.tmp").exists()
        loaded = torch.load(ckpt_dir / "latest.pt", map_location="cpu", weights_only=False)
        assert loaded["update"] == 1
        assert not (ckpt_dir / "previous.pt").exists()  # never rotated -- failure was pre-rotation

        # Now a successful write rotates correctly (latest -> previous, tmp -> latest).
        monkeypatch.setattr(train_full, "_validate_checkpoint_file", real_validate)
        train_full.write_checkpoint_atomic(ckpt_dir, "latest", _dummy_ckpt_obj(3))
        assert torch.load(ckpt_dir / "latest.pt", map_location="cpu", weights_only=False)["update"] == 3
        assert torch.load(ckpt_dir / "previous.pt", map_location="cpu", weights_only=False)["update"] == 1
    finally:
        _cleanup(ckpt_dir.parent)


# ---------------------------------------------------------------------------
# 3. Non-finite stop -> exit 3, no new checkpoint
# ---------------------------------------------------------------------------

def test_nonfinite_stop_exit_code_and_no_checkpoint(monkeypatch):
    arm = "F0_item_only_h256"
    run_dir = _fresh_dir("nonfinite_stop")
    try:
        calls = {"n": 0}
        real_check = train_full._check_finite

        def fake_check(loss_mean, grad_norm_val):
            calls["n"] += 1
            if calls["n"] == 2:
                return False
            return real_check(loss_mean, grad_norm_val)

        monkeypatch.setattr(train_full, "_check_finite", fake_check)
        rc = _run(["--arm", arm, "--run-dir", str(run_dir), "--test-tiny",
                    "--ckpt-every-updates", "10000", "--diag-every-updates", "1000000"])
        assert rc == train_full.EXIT_NONFINITE
        assert not (run_dir / "checkpoints" / "latest.pt").exists()
        assert not (run_dir / "checkpoints" / "best.pt").exists()
        status = json.loads((run_dir / "status.json").read_text())
        assert status["state"] == "STOPPED_NONFINITE"
    finally:
        _cleanup(run_dir)


# ---------------------------------------------------------------------------
# 4. Disk-guard stop -> exit 4, checkpoints intact
# ---------------------------------------------------------------------------

def test_disk_guard_stop_exit_code_and_checkpoints_intact(monkeypatch):
    arm = "F0_item_only_h256"
    run_dir = _fresh_dir("disk_guard_stop")
    try:
        rc0 = _run(["--arm", arm, "--run-dir", str(run_dir), "--test-tiny",
                     "--ckpt-every-updates", "2", "--diag-every-updates", "1000000",
                     "--stop-after-updates", "2"])
        assert rc0 == 0
        before_bytes = (run_dir / "checkpoints" / "latest.pt").read_bytes()

        monkeypatch.setattr(train_full, "_disk_free_bytes", lambda path: 1 * 1024 ** 3)  # 1 GiB, < 8 GiB guard
        rc1 = _run(["--arm", arm, "--run-dir", str(run_dir), "--test-tiny",
                     "--resume", "--ckpt-every-updates", "1", "--diag-every-updates", "1000000",
                     "--stop-after-updates", "3"])
        assert rc1 == train_full.EXIT_DISK
        assert (run_dir / "checkpoints" / "latest.pt").read_bytes() == before_bytes
        status = json.loads((run_dir / "status.json").read_text())
        assert status["state"] == "STOPPED_DISK"
    finally:
        _cleanup(run_dir)


# ---------------------------------------------------------------------------
# 5. selection_step on scripted metric sequences
# ---------------------------------------------------------------------------

def test_selection_step_improvement_and_ties():
    st = train_full.SelectionState()
    st, r0 = train_full.selection_step(st, 0, 0.100, max_epochs=6)
    assert r0["raw_improved"] and r0["best_metric"] == 0.100 and r0["patience"] == 0
    st, r1 = train_full.selection_step(st, 1, 0.100, max_epochs=6)  # exact tie -> not raw_improved
    assert not r1["raw_improved"] and r1["best_metric"] == 0.100 and r1["best_epoch"] == 0
    assert r1["patience"] == 1  # delta 0 < 0.001 threshold


def test_selection_step_sub_threshold_improvement_still_updates_best():
    st = train_full.SelectionState()
    st, r0 = train_full.selection_step(st, 0, 0.100, max_epochs=6)
    st, r1 = train_full.selection_step(st, 1, 0.1005, max_epochs=6)  # +0.0005 < 0.001 but > 0 -> raw improves
    assert r1["raw_improved"] is True
    assert r1["best_metric"] == 0.1005
    assert r1["patience"] == 1  # still sub-threshold for patience purposes


def test_selection_step_lr_step_once_and_patience_stop():
    st = train_full.SelectionState()
    metrics = [0.100, 0.1002, 0.1001, 0.0999]  # epoch0 best; 1,2,3 all sub-threshold deltas
    records = []
    for e, m in enumerate(metrics):
        st, r = train_full.selection_step(st, e, m, max_epochs=6)
        records.append(r)
    assert records[0]["lr_step_now"] is False  # epoch0: delta = +inf, never sub-threshold
    assert records[1]["lr_step_now"] is True  # first sub-threshold eval -> LR steps exactly here
    assert records[2]["lr_step_now"] is False  # never steps a second time
    assert records[3]["lr_step_now"] is False
    assert st.lr_decayed is True and st.lr_decay_epoch == 1
    assert records[1]["patience"] == 1
    assert records[2]["patience"] == 2
    assert records[2]["stop_now"] is True and records[2]["stop_reason"] == "patience"
    # once stopped, a further call should still reflect stop (idempotent bookkeeping upstream
    # would simply not call selection_step again; verify epoch index alone doesn't un-stop it)
    assert records[3]["stop_now"] is True  # patience never resets downward: still >= limit


def test_selection_step_max_epochs_stop_without_patience():
    st = train_full.SelectionState()
    # Strictly improving every epoch (delta always >= +0.01, never sub-threshold) but hits max_epochs.
    metrics = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
    st2 = st
    last = None
    for e, m in enumerate(metrics):
        st2, last = train_full.selection_step(st2, e, m, max_epochs=6)
    assert last["patience"] == 0
    assert last["stop_now"] is True and last["stop_reason"] == "max_epochs"
    assert st2.best_metric == 0.20 and st2.best_epoch == 5


def test_selection_step_tie_keeps_earlier_epoch():
    st = train_full.SelectionState()
    st, r0 = train_full.selection_step(st, 0, 0.15, max_epochs=6)
    st, r1 = train_full.selection_step(st, 1, 0.20, max_epochs=6)
    st, r2 = train_full.selection_step(st, 2, 0.20, max_epochs=6)  # exact tie with epoch1
    assert st.best_epoch == 1  # ties -> earlier epoch wins
    assert r2["raw_improved"] is False


# ---------------------------------------------------------------------------
# 6. VS-leakage / accounting gate
# ---------------------------------------------------------------------------

def test_vs_leakage_assert_fires_on_injected_overlap():
    store = FakeStore.build(
        cache_train_dir=train_full.WORKSPACE / "cache" / "train",
        cache_vs_dir=train_full.WORKSPACE / "cache" / "vs",
        cohort_parquet=train_full.COHORT_PARQUET, channels="F0", n_train=256, n_vs=64,
    )
    ds_train = store.decisions("TRAIN", "loss")
    ds_vs = store.decisions("VALIDATION_SELECT", "eval")

    # Sanity: unmodified fake sets must pass (disjoint by construction).
    train_full.run_launch_asserts(store, ds_train, ds_vs, test_tiny=True)

    # Inject an overlap: force one VS decision_id to equal a TRAIN decision_id.
    bad_ids = np.array(ds_vs.decision_id, copy=True)
    bad_ids[0] = ds_train.decision_id[5]
    bad_vs = ds_vs.with_decision_id(bad_ids)
    with pytest.raises(train_full.ConfigError, match="VS-leakage"):
        train_full.run_launch_asserts(store, ds_train, bad_vs, test_tiny=True)


def test_updates_per_epoch_and_last_batch_accounting():
    assert train_full.updates_per_epoch(train_full.N_TRAIN_PRODUCTION, 256) == \
        train_full.UPDATES_PER_EPOCH_PRODUCTION
    last_batch = train_full.N_TRAIN_PRODUCTION - (train_full.UPDATES_PER_EPOCH_PRODUCTION - 1) * 256
    assert last_batch == 128

    # A smaller, non-exact-multiple N: last microbatch = N % 256 (nonzero case).
    n = 1000
    assert train_full.updates_per_epoch(n, 256) == 4
    positions = list(train_full.iter_positions(0, 0, n, 256, max_epochs=1))
    sizes = [len(p[2]) for p in positions]
    assert sizes == [256, 256, 256, 232]
    assert sum(sizes) == n
    # every position appears exactly once (a permutation, not a resample)
    all_pos = np.concatenate([p[2] for p in positions])
    assert sorted(all_pos.tolist()) == list(range(n))


def test_train_batches_always_have_loss_mask_all_true():
    store = FakeStore.build(
        cache_train_dir=train_full.WORKSPACE / "cache" / "train",
        cache_vs_dir=train_full.WORKSPACE / "cache" / "vs",
        cohort_parquet=train_full.COHORT_PARQUET, channels="F0", n_train=512, n_vs=64,
    )
    ds_train = store.decisions("TRAIN", "loss")
    for _, _, positions in train_full.iter_positions(0, 0, len(ds_train), 256, max_epochs=1):
        batch = store.batch(ds_train, positions)
        assert bool(batch["loss_mask"].numpy().all())


# ---------------------------------------------------------------------------
# 7. Synthetic memorisation under the launch optimizer config
# ---------------------------------------------------------------------------

def _memorise_under_launch_config(group: str, adamw_impl: str, device, seed: int) -> dict:
    torch.manual_seed(seed)
    vocab_sizes = synmem.build_vocab_sizes(group)
    poc = np.arange(synmem.KNOWN_START, synmem.KNOWN_START + synmem.K_SYN)
    model = model_mod.build_model(group, vocab_sizes, product_idx_of_class=poc,
                                   proj_dropout=0.0, readout_dropout=0.0).to(device)
    model.train()
    param_groups = pilot_train.build_param_groups(model, weight_decay=train_full.WEIGHT_DECAY)
    kwargs = dict(lr=1e-2, betas=train_full.BETAS, eps=train_full.EPS)
    kwargs["fused" if adamw_impl == "fused" else "foreach"] = True
    opt = torch.optim.AdamW(param_groups, **kwargs)

    rng = np.random.default_rng(seed)
    batch = synmem.make_positive_batch(rng, group)
    batch = {k: v.to(device) for k, v in batch.items()}
    forward_keys = {k: v for k, v in batch.items() if k not in ("target_class", "loss_mask")}

    initial_loss = None
    for step in range(synmem.MAX_UPDATES):
        opt.zero_grad(set_to_none=True)
        logits = model(**forward_keys)
        loss_sum, n = objective_mod.exact_ce_loss(logits, batch["target_class"], batch["loss_mask"])
        loss = loss_sum / n
        assert torch.isfinite(loss).all()
        loss.backward()
        opt.step()
        if step == 0:
            initial_loss = float(loss.item())

    with torch.no_grad():
        logits = model(**forward_keys)
        loss_sum, n = objective_mod.exact_ce_loss(logits, batch["target_class"], batch["loss_mask"])
        final_loss = float((loss_sum / n).item())
        final_top1 = (logits.argmax(dim=-1) == batch["target_class"]).float().mean().item()
    return {"initial_loss": initial_loss, "final_loss": final_loss, "final_top1": final_top1}


@pytest.mark.parametrize("impl", ["foreach", "fused"])
@pytest.mark.parametrize("group", ["F0_item_only", "F4_rich_side_information"])
def test_synthetic_memorisation_under_launch_config(group, impl):
    device = torch.device("cuda:0")
    r = _memorise_under_launch_config(group, impl, device, seed=20260923)
    assert r["final_loss"] <= 0.05 * r["initial_loss"], r
    assert r["final_top1"] >= 0.95, r


# ---------------------------------------------------------------------------
# 8. GPU pinning
# ---------------------------------------------------------------------------

def test_assert_single_visible_gpu_passes_in_this_env():
    train_full.assert_single_visible_gpu()  # CUDA_VISIBLE_DEVICES=0 in this test env


def test_assert_single_visible_gpu_fails_on_multi_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(train_full.ConfigError):
        train_full.assert_single_visible_gpu()


# ---------------------------------------------------------------------------
# Benchmark pre-check (both arms, both AdamW impls) -- informational only;
# Opus runs the real benchmark later.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("impl", ["foreach", "fused"])
@pytest.mark.parametrize("arm", ARMS)
def test_benchmark_smoke(arm, impl, capsys):
    rc = _run(["--arm", arm, "--adamw-impl", impl, "--benchmark-updates", "300", "--test-tiny"])
    assert rc == 0
    out = capsys.readouterr().out.strip().splitlines()
    result = json.loads(out[-1])
    print(f"[benchmark] arm={arm} impl={impl} -> {result}")
    assert result["median_decisions_per_s"] is not None and result["median_decisions_per_s"] > 0
