"""Synthetic memorization, plus the gradient and common-mode drift monitors.

  * memorization: each family (default topology, K = 64, train mode with dropout on, AdamW with
    lr 1e-3, betas (0.9, 0.999), eps 1e-8, wd 1e-5 on ndim >= 2, clip 1.0, SASRec
    recentering after every step) fits 32 synthetic decisions with RANDOM targets to >= 95% train accuracy in 80
    full-batch steps; the gradient monitor says OK at every step and (SASRec) the drift monitor says OK after every
    recentered step;
  * gradient monitor: OK on a normal step; NONFINITE / ZERO / EXPLODING on injected degenerate gradients; the
    first-layer columns of six flags that never fire (always 0 in the data) get exactly-zero gradient and are named
    by zero_grad_columns, while every live column gets gradient;
  * drift monitor: with per-step recentering the SASRec head common mode stays ~0; WITHOUT it (same data, same
    steps) Adam drifts the common mode (measured here), which the monitor flags; an injected common mode on the tied
    GRU table is reported with its exact size.
Negative controls: an input-blind query cannot memorize; a finiteness-only monitor misses a zero gradient; the
un-recentered SASRec run fails the recentered-gauge check.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from seqrec_testkit import DEAD_FLAGS, FAMS, W, adapter, batch, nc, seeded

from ppsi.seqrec.monitors import (
    common_mode_report,
    drift_verdict,
    grad_report,
    table_stats,
    zero_grad_columns,
)

K = 64
STEPS = 80
DEAD = list(DEAD_FLAGS)


def _optimizer(a):
    return torch.optim.AdamW(a.param_groups(1e-5), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)


def _train(a, b, steps: int = STEPS, *, recenter: bool = True, blind: bool = False, monitor: bool = True):
    opt = _optimizer(a)
    hist = {"loss": [], "gn": [], "drift": []}
    with seeded(2026):
        for _ in range(steps):
            a.module.train()
            q = a.query(b)
            if blind:
                q = q * 0.0                                            # MUTANT: the query cannot see its input
            loss = F.cross_entropy(a.logits(q), b["target_class"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if monitor:
                rep = grad_report(a.module.named_parameters())
                assert rep["verdict"] == "OK", f"gradient monitor: {rep['verdict']} (norm {rep['global_norm']:.3g})"
                hist["gn"].append(rep["global_norm"])
            torch.nn.utils.clip_grad_norm_(a.shared_parameters(), 1.0)
            opt.step()
            if recenter:
                a.post_step()
            hist["loss"].append(float(loss.detach()))
            if a.family == "SASREC":
                hist["drift"].append(common_mode_report(a)["head"])
    a.module.eval()
    with torch.no_grad():
        acc = float((a.scores(b).argmax(1) == b["target_class"]).float().mean())
    return acc, hist


def _memorization_check(family, blind=False):
    a = adapter(family, seed=2026, K=K)
    b = batch(32, K=K, seed=11, min_len=2, max_len=8)
    acc, hist = _train(a, b, blind=blind, monitor=not blind)
    assert acc >= 0.95, f"{family} memorized only {acc:.2f} of 32 random-target decisions in {STEPS} steps"
    assert hist["loss"][-1] < 0.1 * hist["loss"][0]
    return a, hist


@pytest.mark.parametrize("family", FAMS)
def test_synthetic_memorization_with_monitors(family):
    _, hist = _memorization_check(family)
    assert all(math.isfinite(g) and g > 0 for g in hist["gn"])
    if family == "SASREC":
        assert all(drift_verdict(h) == "OK" for h in hist["drift"]), "recentered head drifted"


# ------------------------------------------------------------------------------------------------ gradient monitor
def _one_backward(a, b, scale: float = 1.0):
    a.module.train()
    a.module.zero_grad(set_to_none=True)
    with seeded(1):
        (F.cross_entropy(a.scores(b), b["target_class"]) * scale).backward()


def _monitor_detects(monitor, family="GRU"):
    a = adapter(family, K=K)
    b = batch(8, K=K, seed=3)
    _one_backward(a, b)
    assert monitor(a.module.named_parameters())["verdict"] == "OK"
    _one_backward(a, b, scale=0.0)
    assert monitor(a.module.named_parameters())["verdict"] == "ZERO", "an all-zero gradient must be flagged"
    _one_backward(a, b)
    next(iter(a.module.parameters())).grad.view(-1)[0] = float("nan")
    assert monitor(a.module.named_parameters())["verdict"] == "NONFINITE"
    _one_backward(a, b, scale=1e9)
    assert monitor(a.module.named_parameters())["verdict"] == "EXPLODING"


@pytest.mark.parametrize("family", FAMS)
def test_gradient_monitor_flags_degenerate_gradients(family):
    _monitor_detects(grad_report, family)


@pytest.mark.parametrize("family", FAMS)
def test_never_firing_flag_columns_get_zero_gradient(family):
    a = adapter(family, K=K)
    b = batch(16, K=K, seed=4, min_len=3, max_len=10, zero_flag_columns=DEAD)
    live = [c for c in range(W.flags_dim) if c not in DEAD]
    b["event_quality_flags"][0, 0, live] = 1                            # every live flag fires at least once
    _one_backward(a, b)
    m = a.module
    weight = m.numeric_flags_proj[0].weight if family == "GRU" else m.numflag_proj.weight
    names = list(W.numeric_names) + list(W.flag_names)
    dead = zero_grad_columns(weight.grad, names)
    assert sorted(dead) == sorted(W.flag_names[c] for c in DEAD), dead
    assert len(DEAD) == 6


# ------------------------------------------------------------------------------------------------ drift monitor
def _drift_run(recenter: bool):
    a = adapter("SASREC", seed=2026, K=K)
    a.post_step()                                        # theta0 in the recentered gauge, as build() does
    b = batch(32, K=K, seed=12, min_len=2, max_len=8)
    _, hist = _train(a, b, steps=30, recenter=recenter, monitor=False)
    return hist["drift"]


def test_sasrec_recentering_keeps_the_common_mode_at_zero():
    d = _drift_run(recenter=True)
    assert max(h["common_mode_norm"] for h in d) < 1e-6 and max(abs(h["b_mean"]) for h in d) < 1e-6


def test_sasrec_common_mode_drifts_without_recentering():
    d = _drift_run(recenter=False)
    assert d[-1]["common_mode_norm"] > 1e-4 and d[-1]["common_mode_norm"] > 10 * d[0]["common_mode_norm"], \
        f"expected Adam common-mode drift, got {[round(h['common_mode_norm'], 6) for h in d[::10]]}"
    assert drift_verdict(d[-1]) == "DRIFT"


def test_gru_tied_table_drift_monitor():
    a = adapter("GRU", K=K)
    base = common_mode_report(a)["head"]
    c = torch.randn(128, generator=torch.Generator().manual_seed(0))
    c = 0.5 * c / c.norm()
    with torch.no_grad():
        a.module.item_embed.weight[3:3 + K].add_(c)
    after = common_mode_report(a)["head"]
    head = a.head_weight().detach()
    assert math.isclose(after["common_mode_norm"], float(head.double().mean(0).norm()), rel_tol=1e-9)
    assert after["common_mode_norm"] > base["common_mode_norm"] and after["ratio"] > base["ratio"]
    assert table_stats(head - head.mean(0))["common_mode_norm"] < 1e-6


# ------------------------------------------------------------------------------------------------ negative controls
@nc("an input-blind query (q * 0) cannot memorize random targets")
def test_nc_blind_query_cannot_memorize():
    _memorization_check("GRU", blind=True)


@nc("a monitor that only checks finiteness misses an all-zero gradient")
def test_nc_finiteness_only_monitor():
    def naive(named):
        ok = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for _, p in named)
        return {"verdict": "OK" if ok else "NONFINITE"}
    _monitor_detects(naive)


@nc("without per-step recentering the SASRec head leaves the recentered gauge")
def test_nc_no_recentering_fails_gauge_check():
    d = _drift_run(recenter=False)
    assert max(h["common_mode_norm"] for h in d) < 1e-6
