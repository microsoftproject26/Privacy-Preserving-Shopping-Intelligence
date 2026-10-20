"""One full-batch local SGD step per client, n_consumed-weighted, equals the pooled SGD step (synthetic only).

Algebra: theta - lr * sum_u n_u g_u / sum_u n_u = theta - lr * g_pooled when each g_u is the gradient of the client's
MEAN CE over its n_u valid decisions and there is no dropout / clipping / weight decay. So this checks the weight
(n_consumed), the decision-weighted mean normalization and the valid-row filter together.
Tolerance: 2e-6 absolute on parameters of scale ~0.5 — the per-client means and the pooled mean sum in a different
FP32 order. Measured on this CPU: 1.2e-7 (one ulp) with the step moving parameters by up to 7.8e-2; the negative
controls are off by 1.0e-1 (uniform weights) and 1.7 (summed CE).
Does NOT prove Adam / multi-step / multi-pass equivalence (which does not hold).
"""
from __future__ import annotations

from collections import OrderedDict

import torch
from fedsim_testkit import clients, nc, server_and_worker, solver

import ppsi.fedsim.client as flc
from ppsi.fedsim.aggregate import aggregate_uploads
from ppsi.fedsim.client import client_update, valid_rows
from ppsi.fedsim.numerics import max_abs_diff
from ppsi.fedsim.server import run_round

LR = 0.5
TOL = 2e-6


def _setup():
    srv, workers = server_and_worker(seed=11, dropout=0.0)
    cs = clients(5, seed=21, sizes=[3, 7, 12, 30, 1], invalid_frac=0.25)
    sgd = solver(lr=LR, optimizer="sgd", passes=1, batch_size=10_000, clip=None, weight_decay=0.0)
    return srv, workers, cs, sgd


def pooled_step(adapter, theta, cs):
    adapter.load_state_(theta)
    rows = {k: torch.cat([c.examples[k].index_select(0, valid_rows(c.examples)) for c in cs]) for k in cs[0].examples}
    adapter.module.train()
    for p in adapter.shared_parameters():
        p.grad = None
    logits = adapter.scores(rows)
    loss = torch.nn.functional.cross_entropy(logits, rows["target_class"], reduction="mean")
    loss.backward()
    with torch.no_grad():
        return OrderedDict((k, p - LR * p.grad) for k, p in zip(adapter.manifest.shared_keys,
                                                                  adapter.shared_parameters()))


def fa_round(srv, workers, cs, sgd):
    run_round(srv, workers, cs, sgd, round_idx=0, seed=5)
    return srv.adapter.extract_shared(clone=True)


def uniform_weight_round(srv, workers, cs, sgd):
    theta = srv.broadcast()
    ups = []
    for c in cs:
        r = client_update(workers[0], theta, c, sgd, round_idx=0, seed=5, clone_upload=True)
        ups.append((c.key, r.upload, 1 if r.n_consumed else 0))
    return OrderedDict((k, v) for k, v in aggregate_uploads(srv.manifest, theta, ups).items()
                       if k in srv.manifest.shared_keys)


def check_pooled(round_fn):
    srv, workers, cs, sgd = _setup()
    theta = srv.broadcast()
    ref = pooled_step(workers[0], theta, cs)
    got = round_fn(srv, workers, cs, sgd)
    d = max_abs_diff(got, ref)
    moved = max_abs_diff(OrderedDict((k, theta[k]) for k in ref), ref)
    assert moved > 1e-3, "the step must move the parameters for the check to mean anything"
    assert d < TOL, f"FA(1 full-batch SGD step) vs pooled step: max |diff| = {d:.3g} (tol {TOL})"


def test_fa_one_step_equals_pooled_step():
    check_pooled(fa_round)


@nc("uniform client weights instead of n_consumed")
def test_nc_uniform_weights():
    check_pooled(uniform_weight_round)


@nc("summed (not mean) per-batch CE, i.e. n_u^2 effective weighting")
def test_nc_sum_ce(monkeypatch):
    monkeypatch.setattr(flc, "batch_ce",
                        lambda logits, t: torch.nn.functional.cross_entropy(logits, t, reduction="sum"))
    check_pooled(fa_round)
