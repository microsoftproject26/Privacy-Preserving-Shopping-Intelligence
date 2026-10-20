"""Monitors (loss, gradient norm, common-mode drift, data wait) without per-step syncs; the run guards.

  * no per-step host sync: during a full run, every Tensor -> Python conversion made by v4train code happens inside
    Monitor.flush (the periodic sync), at most a few per flush, never once per update;
  * the monitor log carries the loss, the pre-clip gradient norm (mean / max), the LR, the data wait and the head
    common-mode drift (SASRec: ~0 under per-step recentering); a non-finite loss is caught at the next sync, the run
    fails closed (STATUS FAILED) and the last `latest` stays a valid, finite resumable;
  * strict FP32: a run with TF32 switched on is refused before any record is written;
  * disk guard: PAUSE (free < 10 GiB) refuses a new run but allows a resume; REFUSE (free - planned peak < 8 GiB)
    refuses both;
  * roles and recipe: MAIN trains on M01 and evaluates on M02, CALIBRATION on M05 / M06 (the registry's HISTORICAL role
    map, TRAIN-01R2); a TEST manifest cannot even be
    built without the ALL_METHODS_LOCK; the C8 values, the registered peaks and seeds, the executed dropouts, the
    comparison-key fields and a registered theta0 hash are enforced.
Negative control: a monitor that reads the loss on every update (a per-step sync) is caught.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

import v4train.trainer as T
from fl.numerics import NumericsError
from trainkit import art  # noqa: E402  (TRAIN-03: sealed artifact root)
from trainkit import (K, KEY, N, SCHEMA, WIDTHS, assert_raises, cfg, disk, interrupt_after, nc, providers, read,
                      registry_keys, scripted, trainer, write_registry)
from v4eval.errors import LockGateError
from v4train.checkpoint import sha256_file
from v4train.provider import SyntheticEvalProvider, SyntheticTrainProvider
from v4train.recipe import CentralRecipe, RecipeError, RunConfig
from v4train.schedule import eval_marks
from v4train.trainer import DiskGuardError, InjectedInterruption, TrainerError, TrainingDiverged

VALS = {m: 0.1 for m in eval_marks("MAIN")}
V4TRAIN = str(Path(T.__file__).resolve().parent)
TESTS = str(Path(__file__).resolve().parent)


class SyncCounter:
    NAMES = ("item", "__float__", "__int__", "__bool__", "tolist")

    def __init__(self):
        self.calls = []
        self._orig = {}

    def __enter__(self):
        for n in self.NAMES:
            orig = getattr(torch.Tensor, n)
            self._orig[n] = orig

            def wrap(t, *a, _o=orig, _n=n, **k):
                f = sys._getframe(1)
                if str(Path(f.f_code.co_filename).resolve().parent) in (V4TRAIN, TESTS):
                    g = f                                    # attribute to the nearest v4train frame
                    while g is not None and str(Path(g.f_code.co_filename).resolve().parent) != V4TRAIN:
                        g = g.f_back
                    self.calls.append((g.f_code.co_name if g is not None else "?", _n))
                return _o(t, *a, **k)
            setattr(torch.Tensor, n, wrap)
        return self

    def __exit__(self, *exc):
        for n, o in self._orig.items():
            setattr(torch.Tensor, n, o)


def _check_no_per_step_sync(work, family="GRU"):
    tr = trainer(work / "r", cfg("sync", family, log_every=5), eval_override=scripted(VALS))
    with SyncCounter() as sc:
        s = tr.run()
    flushes = [ln for ln in (art(work / "r") / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
               if '"kind": "monitor"' in ln]
    where = {c[0] for c in sc.calls}
    # allowed: the periodic monitor flush, and the finiteness validation of a checkpoint file being written
    assert where <= {"flush", "_validate"}, f"host syncs outside the periodic flush / checkpoint check: {sorted(where)}"
    n_flush = sum(1 for c in sc.calls if c[0] == "flush")
    assert n_flush <= 4 * len(flushes) < s["updates"] * 4, "syncs must scale with flushes, not with updates"
    return s, flushes


@pytest.mark.parametrize("family", ["GRU", "SASREC"])
def test_no_per_step_sync_and_monitor_records(work, family):
    s, flushes = _check_no_per_step_sync(work, family)
    import json
    recs = [json.loads(x) for x in flushes]
    assert len(recs) < s["updates"] and all(r["loss_mean"] > 0 and r["grad_norm_max_preclip"] > 0 for r in recs)
    assert all(r["data_wait_s"] >= 0 and r["lr_last"] > 0 and not r["nonfinite"] for r in recs)
    assert sum(r["window_updates"] for r in recs) == s["updates"], "every update lands in exactly one window"
    if family == "SASREC":
        assert max(r["head_common_mode_norm"] for r in recs) < 1e-5, "per-step recentering keeps the gauge"
    else:
        assert all(r["head_common_mode_norm"] > 0 for r in recs)


class NaNAfter(SyntheticTrainProvider):
    """Serves a NaN numeric feature once `calls` exceeds a threshold (a corrupted batch)."""

    def __init__(self, *a, after: int, **k):
        super().__init__(*a, **k)
        self.after, self.calls = after, 0

    def batch(self, positions):
        b = super().batch(positions)
        self.calls += 1
        if self.calls > self.after:
            b["event_numeric_features"] = b["event_numeric_features"].clone()
            b["event_numeric_features"][0, 0, 0] = float("nan")
        return b


def test_nonfinite_fails_closed_and_keeps_a_valid_resumable(work):
    tp = NaNAfter(N, K, WIDTHS, seed=3, after=10)
    _, ep = providers()
    tr = trainer(work / "r", cfg("nan", "GRU", log_every=3, checkpoint_every=2), tp, ep, eval_override=scripted(VALS))
    with pytest.raises(TrainingDiverged):
        tr.run()
    assert read(work / "r" / "STATUS.json")["status"] == "FAILED"
    idx = read(art(work / "r") / "CHECKPOINTS.json")["latest"]
    assert idx["meta"]["updates"] <= 10 and sha256_file(art(work / "r") / idx["file"]) == idx["sha256"]
    obj = torch.load(art(work / "r") / idx["file"], weights_only=True)
    assert all(torch.isfinite(v).all() for v in obj["model"].values() if v.is_floating_point())


def test_strict_fp32_guard(work):
    tr = trainer(work / "r", cfg("tf32", "GRU"), eval_override=scripted(VALS))
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        assert_raises(NumericsError, tr.run)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = False
    assert not (work / "r" / "RUN_START.json").exists()


def test_disk_guard(work):
    assert_raises(DiskGuardError, trainer(work / "p", cfg("pause", "GRU"), disk_usage_fn=disk(9.0)).run)
    assert_raises(DiskGuardError, trainer(work / "f", cfg("refuse", "GRU"), disk_usage_fn=disk(7.5)).run)
    c = cfg("resume_pause", "GRU", checkpoint_every=4)
    with pytest.raises(InjectedInterruption):
        trainer(work / "r", c, fault=interrupt_after(9), eval_override=scripted(VALS)).run()
    s = trainer(work / "r", c, disk_usage_fn=disk(9.0), eval_override=scripted(VALS)).resume()     # PAUSE: allowed
    assert s["status"] == "DONE_MAIN"
    with pytest.raises(InjectedInterruption):
        trainer(work / "r2", cfg("resume_ref", "GRU", checkpoint_every=4), fault=interrupt_after(9),
                eval_override=scripted(VALS)).run()
    assert_raises(DiskGuardError, trainer(work / "r2", cfg("resume_ref", "GRU", checkpoint_every=4),
                                          disk_usage_fn=disk(7.5), eval_override=scripted(VALS)).resume)


def test_roles_and_test_manifest_refusal(work):
    tp, ep = providers("MAIN")
    tpc, epc = providers("CALIBRATION")
    assert_raises(TrainerError, trainer, work / "a", cfg("a", "GRU"), tp, epc)                  # MAIN on M06
    assert_raises(TrainerError, trainer, work / "b", cfg("b", "GRU", phase="CALIBRATION"), tp, epc)  # calib on M01
    val = SyntheticTrainProvider(N, K, WIDTHS, seed=3)
    val.split = "VALIDATION"
    assert_raises(TrainerError, trainer, work / "c", cfg("c", "GRU"), val, ep)
    assert_raises(LockGateError, SyntheticEvalProvider, 30, K, WIDTHS, manifest_id="M07_TEST_WARM")
    small = SyntheticEvalProvider(30, 32, WIDTHS)                                                  # K mismatch
    with pytest.raises(TrainerError, match="catalogue_K differs"):                     # registered, so K is the cause
        trainer(work / "d", cfg("d", "GRU", manifest_registry=write_registry({"MAIN": (tp, small)})), tp, small)


def test_recipe_and_key_guards(work):
    base = dict(run_id="g", family="GRU", seed=2026, key=dict(KEY), exposure_point="end", cuda_visible_devices="0",
                determinism_mode="STRICT", **registry_keys())
    for bad in (dict(effective_batch=128), dict(peak_lr=2e-3), dict(seed=2028), dict(phase="CONTINUATION"),
                dict(phase="CALIBRATION", seed=2027), dict(family="LSTM"), dict(log_every=0)):
        assert_raises(RecipeError, RunConfig(**{**base, **bad}).validate)
    assert_raises(RecipeError, RunConfig(**base, recipe=CentralRecipe(grad_clip=5.0)).validate)
    RunConfig(**base).validate()                                                                   # the C8 recipe
    RunConfig(**{**base, "seed": 2027, "peak_lr": 5e-4}).validate()
    k = dict(KEY)
    k.pop("population_hash")
    assert_raises(TrainerError, trainer, work / "k", cfg("k", "GRU", key=k))
    assert_raises(TrainerError, trainer, work / "k2", cfg("k2", "GRU", key=dict(KEY, split="RETRO_TEST")))
    assert_raises(TrainerError, trainer, work / "i", cfg("i", "GRU", expected_init_sha256="0" * 64))
    for fam in ("GRU", "SASREC"):
        tr = trainer(work / f"dr_{fam}", cfg(f"dr_{fam}", fam))
        tr._check_dropout()                                                                        # executed values
        if fam == "GRU":
            tr.module.input_dropout.p = 0.3
        else:
            object.__setattr__(tr.module.cfg, "dropout", 0.1)
        assert_raises(TrainerError, tr._check_dropout)
    tr = trainer(work / "opt", cfg("opt", "SASREC"))
    g = tr.opt.param_groups
    assert tr.opt.defaults["fused"] is True and tr.opt.defaults["betas"] == (0.9, 0.999) and tr.opt.defaults["eps"] == 1e-8
    assert [pg["weight_decay"] for pg in g] == [1e-5, 0.0] and all(p.ndim >= 2 for p in g[0]["params"])
    assert all(p.ndim < 2 for p in g[1]["params"])


@nc("a monitor that reads the loss on every update (a per-step host sync)")
def test_nc_per_step_sync(work, monkeypatch):
    orig = T.Monitor.add

    def add(self, loss_sum, *a, **k):
        float(loss_sum)                                  # MUTANT: loss.item() per step
        return orig(self, loss_sum, *a, **k)
    monkeypatch.setattr(T.Monitor, "add", add)
    _check_no_per_step_sync(work)
