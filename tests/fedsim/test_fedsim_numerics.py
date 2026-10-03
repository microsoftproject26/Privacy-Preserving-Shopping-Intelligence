"""Strict FP32: both TF32 switches off and highest matmul precision are enforced; a visit or LO run refuses TF32.
Seed derivation is stable across processes (pinned values) and separates streams."""
from __future__ import annotations

import torch
from fedsim_testkit import assert_raises, nc, one_client, server_and_worker, solver, tiny

from ppsi.fedsim.client import client_update
from ppsi.fedsim.local_only import LORecipe, run_local_only
from ppsi.fedsim.numerics import (
    NumericsError,
    assert_strict_fp32,
    derive_seed,
    numerics_record,
    state_digest,
)


def check_strict_refused(set_flags):
    srv, w = server_and_worker()
    theta = srv.broadcast()
    c = one_client("user-a", 10)
    set_flags()
    try:
        assert_raises(NumericsError, client_update, w[0], theta, c, solver(), round_idx=0, seed=1)
        t0 = tiny(3).broadcast_state(clone=True)
        assert_raises(NumericsError, run_local_only, tiny(3), t0, c, LORecipe(state_digest(t0), 1))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")


def _tf32_matmul():
    torch.backends.cuda.matmul.allow_tf32 = True


def _tf32_cudnn():
    torch.backends.cudnn.allow_tf32 = True


def _nothing():
    pass


def test_strict_fp32_flags_enforced():
    r = numerics_record()
    assert r["cuda_matmul_allow_tf32"] is False and r["cudnn_allow_tf32"] is False
    assert r["float32_matmul_precision"] == "highest" and r["deterministic_algorithms"] is True
    assert_strict_fp32()


def test_tf32_matmul_refused():
    check_strict_refused(_tf32_matmul)


def test_tf32_cudnn_refused():
    check_strict_refused(_tf32_cudnn)


def test_derive_seed_stable_and_separating():
    assert derive_seed(2026, "visit", 0, "user-00001") == derive_seed(2026, "visit", 0, "user-00001")
    assert derive_seed(2026, "visit", 0, "user-00001") != derive_seed(2026, "visit", 1, "user-00001")
    assert derive_seed(2026, "visit", 0, "1") != derive_seed(2026, "visit", 0, 1)          # typed parts
    assert derive_seed(2026, "visit", 0, "user-00001") == 5763208357634313893            # pinned (cross-process)


@nc("the strict-FP32 guard is not triggered when nothing is switched on (the check itself is live)")
def test_nc_strict_guard_live():
    check_strict_refused(_nothing)
