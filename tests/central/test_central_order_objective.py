"""The data-order policy and the exact objective.

  * pass_permutation is a pure function of (seed, pass, N), a permutation of N, distinct per pass, and two seeds
    never share a pass permutation; its sha256 identifies it;
  * ce_sum_count sums the per-row CE of the loss-eligible rows (loss_mask AND target_class >= 0) and counts them;
    excluded rows get no gradient;
  * accumulate_step: the microbatch-accumulated gradient equals the full-batch gradient (each microbatch divides by the
    effective batch's count, known before the forward) and the loss sum is preserved; a tail batch is normalised by
    its own count.
Negative controls: the default_rng(seed + pass) order collides across seeds; a mean of microbatch means.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from ppsi.central.objective import accumulate_step, ce_sum_count, host_contributing, split_rows
from ppsi.central.order import naive_permutation, pass_permutation, permutation_sha256
from ppsi.fedsim.synthetic import client_examples, make_tiny_adapter

K = 24


def nc(reason: str):
    return pytest.mark.xfail(strict=True, raises=AssertionError, reason="NEGATIVE CONTROL: " + reason)


# ------------------------------------------------------------------------------------------------ order
def _check_seeds_independent(perm_fn):
    a = perm_fn(2026, 1, 1000)
    b = perm_fn(2027, 0, 1000)
    assert not np.array_equal(a, b), "two seeds must never share a pass permutation"
    for s in (2026, 2027):
        ps = [perm_fn(s, p, 1000) for p in range(8)]
        assert len({permutation_sha256(p) for p in ps}) == 8


def test_pass_permutation_is_pure_and_distinct():
    p1, p2 = pass_permutation(2026, 3, 5000), pass_permutation(2026, 3, 5000)
    assert np.array_equal(p1, p2) and np.array_equal(np.sort(p1), np.arange(5000))
    assert permutation_sha256(p1) == permutation_sha256(p2) and p1.dtype == np.int64
    _check_seeds_independent(pass_permutation)
    with pytest.raises(ValueError):
        pass_permutation(2026, -1, 10)
    with pytest.raises(ValueError):
        pass_permutation(2026, 0, 0)


@nc("the default_rng(seed + pass) order rule collides across seeds 2026 / 2027")
def test_nc_naive_order_collides_across_seeds():
    _check_seeds_independent(naive_permutation)


# ------------------------------------------------------------------------------------------------ objective
def test_ce_sum_count_sums_the_eligible_rows():
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(12, K, generator=g, requires_grad=True)
    tgt = torch.randint(0, K, (12,), generator=g)
    tgt[[2, 7]] = -1
    lm = torch.ones(12, dtype=torch.bool)
    lm[[4]] = False
    s, n = ce_sum_count(logits, tgt, lm)
    keep = torch.tensor([i for i in range(12) if i not in (2, 4, 7)])
    ref = F.cross_entropy(logits[keep], tgt[keep], reduction="sum")
    assert int(n) == 9 and torch.allclose(s, ref, rtol=1e-6)
    s.backward()
    assert float(logits.grad[[2, 4, 7]].abs().sum()) == 0.0, "excluded rows contribute nothing"
    assert host_contributing({"target_class": tgt, "loss_mask": lm}) == 9
    with pytest.raises(ValueError):
        ce_sum_count(logits[0], tgt, lm)


def _batch(n, seed):
    b = client_examples(n, K, 8, np.random.default_rng(seed))
    b["loss_mask"] = torch.ones(n, dtype=torch.bool)
    return b


def _grads(module):
    return {k: p.grad.detach().clone() for k, p in module.named_parameters() if p.grad is not None}


def _accum_check(micro, divide_once=True):
    a = make_tiny_adapter(K=K, d=8, tied=False, dropout=0.0, seed=7)
    a.module.eval()
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
            mb = split_rows(b, s, min(37, s + micro))
            ls, c = ce_sum_count(a.scores(mb), mb["target_class"], mb["loss_mask"])
            (ls / c / -(-37 // micro)).backward()
            part = ls.detach() if part is None else part + ls.detach()
    g_part = _grads(a.module)
    assert torch.allclose(full, part, rtol=1e-6), "the loss sum must be preserved"
    worst = max(float((g_full[k] - g_part[k]).abs().max()) / (float(g_full[k].abs().max()) + 1e-12) for k in g_full)
    assert worst < 1e-4, f"accumulated gradient != full-batch gradient (relative {worst:.3g})"


@pytest.mark.parametrize("micro", [16, 7])
def test_exact_microbatch_accumulation(micro):
    _accum_check(micro)


def test_tail_batch_normalized_by_its_own_count():
    a = make_tiny_adapter(K=K, d=8, tied=False, dropout=0.0, seed=7)
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
    with pytest.raises(ValueError):
        accumulate_step(a.scores, tail, 0, 0, torch.device("cpu"))


@nc("mean of microbatch means (each microbatch divided by its own count)")
def test_nc_mean_of_means():
    _accum_check(16, divide_once=False)
