"""T17 — one loss contribution per decision, tail normalization, loss sum and count accumulation.

Both V2 families at the Clean V2 widths (eval mode: dropout off, so microbatch splits do not change any draw), with the
executed objectives: GRU reuse `objective.exact_ce_loss` + `LossAccumulator`, SASRec `sasrec_objective.ce_sum`
(both SUM per call, count returned, divided exactly once), and the FL local step `fl.client.batch_ce`.
  * the model emits ONE [B, K] logit row per decision (never per position); the count is the number of rows with
    loss_mask & target >= 0; the summed loss equals the sum of per-decision losses computed one decision at a time;
    excluded rows get exactly-zero logit gradient and included rows get softmax - onehot;
  * tail normalization: 37 decisions in microbatches 16 / 16 / 5 (with excluded rows inside them): sum-then-divide
    reproduces the full-batch mean loss AND gradient; the FL tail batch is normalized by its own size.
Negative controls: the mean of microbatch means, and dividing by B instead of the contributing count, both fail.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import objective as gru_objective              # snapshot copy: snap_v1/reuse/objective.py
import sasrec_objective                        # snapshot copy: snap_v1/sasrec_objective.py
from fl.client import batch_ce
from models_testkit import FAMS, K_SMALL, batch, forward, is_gru, maxdiff, nc, v2_model
from v4models.synthetic import take

N, MB = 37, 16


def _loss_fn(m):
    return gru_objective.exact_ce_loss if is_gru(m) else sasrec_objective.ce_sum


def _data(seed=0):
    b = batch(N, seed=seed, min_len=1, max_len=12)
    b["loss_mask"][[3, 17, 33]] = False                      # excluded by loss_mask
    b["target_class"][[5, 20, 36]] = -1                      # no rankable target
    return b


def _included(b):
    return b["loss_mask"] & (b["target_class"] >= 0)


@pytest.mark.parametrize("family", FAMS)
def test_t17_one_contribution_per_decision(family):
    m = v2_model(family, 5).eval()
    b = _data()
    logits = forward(m, b)
    assert logits.shape == (N, K_SMALL), "one [B, K] logit row per decision"
    loss_sum, n = _loss_fn(m)(logits, b["target_class"], b["loss_mask"])
    inc = _included(b)
    assert int(n) == int(inc.sum()) == N - 6
    with torch.no_grad():
        per = []
        for i in torch.nonzero(inc).flatten().tolist():
            li = forward(m, take(b, [i]))
            per.append(F.cross_entropy(li, b["target_class"][i:i + 1], reduction="sum"))
        ref = torch.stack(per).sum()
    assert torch.allclose(loss_sum.detach(), ref, rtol=1e-6, atol=1e-5), \
        f"summed loss != sum of per-decision losses ({float(loss_sum)} vs {float(ref)})"
    ref2, n2 = sasrec_objective.ce_sum_reference(logits.detach(), b["target_class"], b["loss_mask"])
    assert int(n2) == int(n) and abs(float(ref2) - float(loss_sum)) < 1e-4
    lg = logits.detach().requires_grad_(True)
    ls, _ = _loss_fn(m)(lg, b["target_class"], b["loss_mask"])
    ls.backward()
    assert float(lg.grad[~inc].abs().sum()) == 0.0, "an excluded decision contributed gradient"
    expect = torch.softmax(lg.detach()[inc], 1)
    expect[torch.arange(int(inc.sum())), b["target_class"][inc]] -= 1.0
    assert torch.allclose(lg.grad[inc], expect, atol=1e-6)


def _accumulate(m, b, divide: str = "once"):
    """Microbatches of 16 with the tail at its own size; returns (loss, grads)."""
    m.zero_grad(set_to_none=True)
    fn = _loss_fn(m)
    acc = gru_objective.LossAccumulator()
    means = []
    for s in range(0, N, MB):
        mb = take(b, list(range(s, min(N, s + MB))))
        ls, n = fn(forward(m, mb), mb["target_class"], mb["loss_mask"])
        acc.add(ls, n)
        means.append(ls / (n if divide != "batch_size" else mb["target_class"].shape[0]))
    if divide == "once":
        loss = acc.mean()
    elif divide == "mean_of_means":
        loss = torch.stack(means).mean()
    else:                                                             # "batch_size": divide by B, not the count
        loss = acc.loss_sum / N
    loss.backward()
    return loss.detach(), {k: p.grad.clone() for k, p in m.named_parameters() if p.grad is not None}


def _full(m, b):
    m.zero_grad(set_to_none=True)
    ls, n = _loss_fn(m)(forward(m, b), b["target_class"], b["loss_mask"])
    loss = ls / n
    loss.backward()
    return loss.detach(), {k: p.grad.clone() for k, p in m.named_parameters() if p.grad is not None}


def _tail_check(family, divide):
    m = v2_model(family, 6).eval()
    b = _data(seed=1)
    l_full, g_full = _full(m, b)
    l_acc, g_acc = _accumulate(m, b, divide)
    assert abs(float(l_acc) - float(l_full)) < 1e-6, f"loss {float(l_acc)} != full-batch mean {float(l_full)}"
    worst = max(maxdiff(g_acc[k], g_full[k]) / (float(g_full[k].abs().max()) + 1e-12) for k in g_full)
    assert worst < 1e-4, f"accumulated gradient differs from the full-batch gradient (relative {worst:.3g})"


@pytest.mark.parametrize("family", FAMS)
def test_t17_tail_normalization_sum_then_divide_once(family):
    _tail_check(family, "once")


def test_t17_accumulator_counts_and_refuses_empty():
    acc = gru_objective.LossAccumulator()
    acc.add(torch.tensor(6.0), torch.tensor(2)).add(torch.tensor(3.0), torch.tensor(1))
    assert float(acc.mean()) == 3.0 and int(acc.n) == 3
    with pytest.raises(ZeroDivisionError):
        gru_objective.LossAccumulator().mean()
    z = torch.zeros(4, K_SMALL, requires_grad=True)
    ls, n = sasrec_objective.ce_sum(z, torch.full((4,), -1), torch.ones(4, dtype=torch.bool))
    assert int(n) == 0 and float(ls) == 0.0 and ls.requires_grad          # zero contributions, graph kept


@pytest.mark.parametrize("family", FAMS)
def test_t17_fl_tail_batch_normalized_by_its_own_size(family):
    m = v2_model(family, 7).eval()
    b = batch(5, seed=9)                                      # an FL tail batch of 5 valid decisions
    with torch.no_grad():
        logits = forward(m, b)
        ls, n = sasrec_objective.ce_sum(logits, b["target_class"], b["loss_mask"])
        assert int(n) == 5
        assert torch.allclose(batch_ce(logits, b["target_class"]), ls / 5, atol=1e-6)


# ------------------------------------------------------------------------------------------------ negative controls
@nc("mean of microbatch means (the tail microbatch is over-weighted)")
@pytest.mark.parametrize("family", FAMS)
def test_nc_t17_mean_of_means(family):
    _tail_check(family, "mean_of_means")


@nc("dividing by the row count B instead of the contributing-decision count")
def test_nc_t17_divide_by_batch_size():
    _tail_check("GRU", "batch_size")
