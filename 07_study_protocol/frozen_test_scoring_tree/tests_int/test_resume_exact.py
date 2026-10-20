"""Resume is exact with the real providers: the real CentralTrainer (CPU, GRU at the full K, real v1 class map) run
6 updates straight equals 3 updates + a resume from `latest` with a REBUILT provider + 3 updates, bitwise (weights,
AdamW state, RNG, train state). A deliberately impure provider breaks it (negative control)."""
from __future__ import annotations

import os

import numpy as np
import pytest

from intkit import REAL_DS, nc
from v4int.launch_central import state_digests
from v4int.providers import StoreEvalProvider, StoreTrainProvider
from v4models.schema import load_schema
from v4train.recipe import RunConfig
from v4train.trainer import CentralTrainer, InjectedInterruption

Q = lambda *_: None  # noqa: E731
KEY = {"data_protocol_hash": "INT01-TEST", "feature_contract_hash": "F", "split": "VALIDATION",
       "label_policy_hash": "L", "population_hash": "P"}
GIB = 2 ** 30


class _Usage:
    total, used, free = 2000 * GIB, 1000 * GIB, 1000 * GIB


def _stop_after(u_stop):
    def f(u):
        if u == u_stop:
            raise InjectedInterruption(f"stop after update {u}")
    return f


def _trainer(run_dir, tp, ep, stop, registry_path):
    from v4int.launch_central import DETERMINISM_MODE
    extra = {"determinism_mode": DETERMINISM_MODE} if "determinism_mode" in RunConfig.__dataclass_fields__ else {}
    if "manifest_registry" in RunConfig.__dataclass_fields__:              # TRAIN-01R2: REQUIRED keys
        extra["manifest_registry"] = str(registry_path)
        extra["data_version"] = "PPSI_V4_INT01_HISTORICAL_HARNESS"
    cfg = RunConfig(run_id="INT01_RESUME", family="GRU", seed=2026, phase="CALIBRATION", peak_lr=1e-3,
                    effective_batch=16, log_every=3, checkpoint_every=3, exposure_point="end",
                    cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    key=dict(KEY, candidate_order_hash=ep.manifest.header["candidate_order_hash"]), test_overrides=True,
                    **extra)
    return CentralTrainer(cfg, load_schema(), tp, ep, run_dir, device="cpu", v1_root=REAL_DS, fault=_stop_after(stop),
                          disk_usage_fn=lambda _p: _Usage())


def _leg(tr, resume=False):
    try:
        tr.resume() if resume else tr.run()
    except InjectedInterruption:
        pass
    return state_digests(tr)


class ImpureProvider:
    """Deliberately wrong: every second call serves the requested rows in reverse order (still loss-eligible)."""

    def __init__(self, inner):
        self.inner, self.calls = inner, 0
        self.manifest_id, self.split, self.fingerprint = inner.manifest_id, inner.split, inner.fingerprint

    @property
    def n_decisions(self):
        return self.inner.n_decisions

    def decision_ids(self):
        return self.inner.decision_ids()

    def batch(self, positions):
        self.calls += 1
        p = np.asarray(positions)
        return self.inner.batch(p[::-1] if self.calls % 2 == 0 else p)


def _straight_vs_resumed(tmp_path, make_tp, ep, registry_path):
    a = _leg(_trainer(tmp_path / "A", make_tp(), ep, 6, registry_path))
    b1 = _leg(_trainer(tmp_path / "B", make_tp(), ep, 3, registry_path))
    assert b1["updates"] == 3 and (tmp_path / "B" / "checkpoints" / "latest.pt").is_file()
    b = _leg(_trainer(tmp_path / "B", make_tp(), ep, 6, registry_path), resume=True)  # a fresh trainer + rebuilt provider
    return a, b


@pytest.fixture(autouse=True)
def _launcher_env(monkeypatch):
    """What the launcher exports before CUDA starts (STRICT needs it; the T3 value from PREG1_LOCK_VALUES)."""
    from v4int.launch_central import export_environment
    env = {}
    if "CUBLAS_WORKSPACE_CONFIG" in os.environ:
        env["CUBLAS_WORKSPACE_CONFIG"] = os.environ["CUBLAS_WORKSPACE_CONFIG"]
    export_environment(0, env)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", env["CUBLAS_WORKSPACE_CONFIG"])


def test_resume_is_bitwise_with_the_real_provider(tmp_path, real_env, providers, int_registry):
    _, ep = providers
    make_tp = lambda: StoreTrainProvider("M05_CALIB_TRAIN", history_policy="SESSION_L50", paths=real_env["paths"], ds_root=REAL_DS,  # noqa: E731
                                         buckets=[0], log=Q)
    a, b = _straight_vs_resumed(tmp_path, make_tp, ep, int_registry)
    assert a["updates"] == b["updates"] == 6
    for k in ("model_state_dict", "optimizer_state_dict", "rng_torch_cpu", "train_state", "stop_state"):
        assert a[k] == b[k], k


@nc("an impure provider (the served rows depend on a call counter) makes the resumed run differ")
def test_nc_impure_provider_breaks_resume(tmp_path, real_env, providers, int_registry):
    tp0, ep = providers
    a, b = _straight_vs_resumed(tmp_path, lambda: ImpureProvider(tp0), ep, int_registry)
    assert a["model_state_dict"] == b["model_state_dict"]
