"""T17 — loss accumulation in the trainer: one contribution per decision, tail normalization, sum/count accumulation.

  * the trainer's sync-free objective `ce_sum_count` equals the executed objectives (reuse objective.exact_ce_loss and
    sasrec_objective.ce_sum): same sum, same count, zero logit gradient on excluded rows;
  * exact microbatch accumulation (package 10 §2): one effective batch split into microbatches of 16 or 7 (uneven,
    with a remainder) gives the full-batch loss sum and gradient (eval mode, both V2 families; FP32 summation-order
    tolerance), and a tail batch of 8 is normalized by its own count;
  * in a run, every pass consumes exactly N exposures (6 x 32 + a tail of 8 for N = 200), 6.0 EFE = 6N exposures, and a
    run in microbatch mode (dropout off) follows the single-pass trajectory within FP32 tolerance.
Negative control: dividing each microbatch by its own count (mean of microbatch means) changes the gradient.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import objective as gru_objective          # snapshot copies (via ppsi_env)
import sasrec_objective
from trainkit import B, K, N, WIDTHS, cfg, nc, scripted, trainer
from v4models.build import build
from v4models.synthetic import synthetic_examples_v2
from v4train.objective import accumulate_step, ce_sum_count, host_contributing
from v4train.schedule import eval_marks

FAMS = ["GRU", "SASREC"]


def _batch(n, seed=0):
    b = synthetic_examples_v2(n, K, WIDTHS, torch.Generator().manual_seed(seed), max_len=8, window=8)
    return b


def test_ce_sum_count_equals_the_executed_objectives():
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(12, K, generator=g, requires_grad=True)
    tgt = torch.randint(0, K, (12,), generator=g)
    tgt[[2, 7]] = -1
    lm = torch.ones(12, dtype=torch.bool)
    lm[[4]] = False
    s, n = ce_sum_count(logits, tgt, lm)
    s1, n1 = gru_objective.exact_ce_loss(logits, tgt, lm)
    s2, n2 = sasrec_objective.ce_sum(logits, tgt, lm)
    assert int(n) == int(n1) == int(n2) == 9
    assert torch.allclose(s, s1, rtol=1e-6) and torch.allclose(s, s2, rtol=1e-6)
    s.backward()
    assert float(logits.grad[[2, 4, 7]].abs().sum()) == 0.0, "excluded rows contribute nothing"
    assert host_contributing({"target_class": tgt, "loss_mask": lm}) == 9


def _grads(module):
    return {k: p.grad.detach().clone() for k, p in module.named_parameters() if p.grad is not None}


def _accum_check(family, micro, divide_once=True):
    bm = build(family, WIDTHS, 7, K=K)
    a = bm.adapter
    a.module.eval()                                    # dropout off: microbatch splits cannot change any draw
    b = _batch(37, seed=2)
    n = host_contributing(b)
    a.module.zero_grad(set_to_none=True)
    full = accumulate_step(a.scores, b, n, 0, torch.device("cpu"))
    g_full = _grads(a.module)
    a.module.zero_grad(set_to_none=True)
    if divide_once:
        part = accumulate_step(a.scores, b, n, micro, torch.device("cpu"))
    else:                                              # MUTANT: mean of microbatch means
        part = None
        for s in range(0, 37, micro):
            mb = {k: (v[s:s + micro] if torch.is_tensor(v) else v) for k, v in b.items()}
            ls, c = ce_sum_count(a.scores(mb), mb["target_class"], mb["loss_mask"])
            (ls / c / -(-37 // micro)).backward()
            part = ls.detach() if part is None else part + ls.detach()
    g_part = _grads(a.module)
    assert torch.allclose(full, part, rtol=1e-6), "the loss sum must be preserved"
    worst = max(float((g_full[k] - g_part[k]).abs().max()) / (float(g_full[k].abs().max()) + 1e-12) for k in g_full)
    assert worst < 1e-4, f"accumulated gradient != full-batch gradient (relative {worst:.3g})"


@pytest.mark.parametrize("family", FAMS)
@pytest.mark.parametrize("micro", [16, 7])
def test_t17_exact_microbatch_accumulation(family, micro):
    _accum_check(family, micro)


@pytest.mark.parametrize("family", FAMS)
def test_t17_tail_batch_normalized_by_its_own_count(family):
    bm = build(family, WIDTHS, 7, K=K)
    a = bm.adapter
    a.module.eval()
    tail = _batch(8, seed=3)
    a.module.zero_grad(set_to_none=True)
    s = accumulate_step(a.scores, tail, host_contributing(tail), 0, torch.device("cpu"))
    g = _grads(a.module)
    a.module.zero_grad(set_to_none=True)
    F.cross_entropy(a.scores(tail), tail["target_class"]).backward()       # the mean over the 8 tail decisions
    g2 = _grads(a.module)
    assert host_contributing(tail) == 8 and float(s) > 0
    assert all(torch.allclose(g[k], g2[k], rtol=1e-5, atol=1e-8) for k in g)


def test_t17_pass_exposure_accounting_and_microbatch_trajectory(work):
    vals = {m: 0.1 for m in eval_marks("MAIN")}
    runs = {}
    for name, micro in (("single", 0), ("micro", 10)):
        tr = trainer(work / name, cfg(f"acc_{name}", "GRU", microbatch=micro), eval_override=scripted(vals))
        tr.module.input_dropout.p = 0.0                 # dropout off so the two runs are comparable (test only)
        tr.module.readout_dropout.p = 0.0
        s = tr.run()
        runs[name] = (s, {k: v.detach().clone() for k, v in tr.module.state_dict().items()}, tr.train.access_log)
    s1, w1, log1 = runs["single"]
    s2, w2, log2 = runs["micro"]
    per_pass = -(-N // B)
    assert s1["exposures"] == s2["exposures"] == 6 * N and s1["updates"] == s2["updates"] == 6 * per_pass
    assert len(log1) == 6 * N and log1 == log2, "same order, every decision once per pass"
    for p in range(6):
        assert sorted(log1[p * N:(p + 1) * N]) == sorted(set(log1[p * N:(p + 1) * N])), "one exposure per decision"
    worst = max(float((w1[k].double() - w2[k].double()).abs().max()) for k in w1 if w1[k].is_floating_point())
    assert worst < 1e-4, f"microbatch trajectory drifted by {worst:.3g}"


@nc("mean of microbatch means (each microbatch divided by its own count)")
def test_nc_t17_mean_of_means():
    _accum_check("GRU", 16, divide_once=False)
