"""Per-user mean losses of a model adapter (ppsi.privacy.user_losses) on the synthetic family of ppsi.fedsim."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from ppsi.fedsim.synthetic import client_examples, make_tiny_adapter
from ppsi.privacy.membership_inference import AuditError
from ppsi.privacy.user_losses import user_mean_losses

K = 24


def batches_for(n_users, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for u in range(n_users):
        ex = client_examples(int(rng.integers(1, 6)), K, 6, rng)
        ex["target_class"] = ex["target_class"].clamp(min=0)
        ex["user"] = np.full(ex["target_class"].shape[0], u)
        out.append(ex)
    return out


def test_matches_a_direct_computation_and_restores_train_mode():
    adapter = make_tiny_adapter(K=K, d=8, seed=1)
    adapter.module.train()
    batches = batches_for(7)
    got = user_mean_losses(adapter, batches, 7)
    assert adapter.module.training
    adapter.module.eval()
    with torch.no_grad():
        want = [float(F.cross_entropy(adapter.logits(adapter.query(b)), b["target_class"]).double()) for b in batches]
    assert got == pytest.approx(want, rel=1e-6)
    assert got.dtype == np.float64


def test_refuses_users_without_rows_and_missing_targets():
    adapter = make_tiny_adapter(K=K, d=8, seed=1)
    batches = batches_for(3)
    with pytest.raises(AuditError):
        user_mean_losses(adapter, batches, 4)
    bad = dict(batches[0])
    bad["target_class"] = bad["target_class"].clone()
    bad["target_class"][0] = -1
    with pytest.raises(AuditError):
        user_mean_losses(adapter, [bad], 1)
