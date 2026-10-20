"""Tests for the training objective (Task T-B), per the brief's "Tests"
section (T7 + sampled-CE toy test).

Run with the LOCAL python from `.local_ai\\gru_v1`:
    <local python> -m pytest code\\test_objective.py -v
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from objective import LossAccumulator, exact_ce_loss, sampled_ce_loss

# ---------------------------------------------------------------------------
# T7 fixture: 8 rows, K=6.
#
# "Masked out" rows exercise BOTH exclusion pathways:
#   - loss_mask == False  -> indices 2, 5
#   - target_class == -1  -> index 7   (loss_mask True, but no target)
# Contributing rows: 0, 1, 3, 4, 6  (5 of 8).
# ---------------------------------------------------------------------------

K = 6
B = 8

torch.manual_seed(2026)
_LOGITS = torch.randn(B, K, dtype=torch.float32)

_TARGET = torch.tensor([1, 0, 3, 4, 2, 5, 3, -1], dtype=torch.long)
_LOSS_MASK = torch.tensor(
    [True, True, False, True, True, False, True, True], dtype=torch.bool
)
_CONTRIBUTING = torch.tensor([0, 1, 3, 4, 6], dtype=torch.long)


def _reference_mean() -> torch.Tensor:
    return F.cross_entropy(
        _LOGITS[_CONTRIBUTING], _TARGET[_CONTRIBUTING], reduction="mean"
    )


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------

def test_t7_masking_and_denominator():
    """3 of 8 rows masked out -> accumulated mean equals the mean over the 5
    contributing rows only (hand-verified against a direct F.cross_entropy
    mean restricted to those rows)."""
    loss_sum, n = exact_ce_loss(_LOGITS, _TARGET, _LOSS_MASK)
    assert int(n.item()) == 5

    acc = LossAccumulator()
    acc.add(loss_sum, n)
    got = acc.mean()

    expected = _reference_mean()
    assert torch.allclose(got, expected, atol=1e-6)


def test_t7_accumulation_equivalence_uneven_microbatches():
    """Splitting the 8 rows into two UNEVEN microbatches (5 + 3) and
    accumulating sum/count separately gives a result equal to the
    single-batch computation within 1e-6. This is the test that catches
    per-microbatch averaging: microbatch 1 (rows 0-4) has 4 contributing
    rows, microbatch 2 (rows 5-7) has only 1 -- an uneven split that a naive
    "average the two microbatch means" bug would mis-weight."""
    mb1_logits, mb1_target, mb1_mask = _LOGITS[:5], _TARGET[:5], _LOSS_MASK[:5]
    mb2_logits, mb2_target, mb2_mask = _LOGITS[5:], _TARGET[5:], _LOSS_MASK[5:]

    ls1, n1 = exact_ce_loss(mb1_logits, mb1_target, mb1_mask)
    ls2, n2 = exact_ce_loss(mb2_logits, mb2_target, mb2_mask)
    assert int(n1.item()) == 4
    assert int(n2.item()) == 1

    acc = LossAccumulator()
    acc.add(ls1, n1).add(ls2, n2)
    accumulated_mean = acc.mean()

    single_loss_sum, single_n = exact_ce_loss(_LOGITS, _TARGET, _LOSS_MASK)
    single_mean = single_loss_sum / single_n

    assert torch.allclose(accumulated_mean, single_mean, atol=1e-6)
    assert torch.allclose(accumulated_mean, _reference_mean(), atol=1e-6)


def test_zero_contributing_rows_is_finite_and_autograd_connected():
    logits = torch.randn(4, K, dtype=torch.float32, requires_grad=True)
    target = torch.full((4,), -1, dtype=torch.long)
    loss_mask = torch.tensor([True, True, True, True], dtype=torch.bool)

    loss_sum, n = exact_ce_loss(logits, target, loss_mask)
    assert int(n.item()) == 0
    assert torch.isfinite(loss_sum)
    assert float(loss_sum.item()) == 0.0

    loss_sum.backward()
    assert logits.grad is not None
    assert torch.all(logits.grad == 0)


# ---------------------------------------------------------------------------
# Broken variants -- must genuinely xfail (strict).
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="dividing by B instead of the contributing count")
def test_broken_divide_by_batch_size():
    loss_sum, _n = exact_ce_loss(_LOGITS, _TARGET, _LOSS_MASK)
    broken_mean = loss_sum / _LOGITS.shape[0]  # divide by B=8, not n_contributing=5
    expected = _reference_mean()
    assert torch.allclose(broken_mean, expected, atol=1e-6)


@pytest.mark.xfail(strict=True, reason="averaging two microbatch means instead of sum/sum")
def test_broken_average_of_microbatch_means():
    mb1_logits, mb1_target, mb1_mask = _LOGITS[:5], _TARGET[:5], _LOSS_MASK[:5]
    mb2_logits, mb2_target, mb2_mask = _LOGITS[5:], _TARGET[5:], _LOSS_MASK[5:]

    ls1, n1 = exact_ce_loss(mb1_logits, mb1_target, mb1_mask)
    ls2, n2 = exact_ce_loss(mb2_logits, mb2_target, mb2_mask)
    mean1 = ls1 / n1
    mean2 = ls2 / n2
    broken_avg = (mean1 + mean2) / 2  # ignores that n1=4 != n2=1

    expected = _reference_mean()
    assert torch.allclose(broken_avg, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Sampled-CE: default-OFF gate + toy correctness test (K=5, N=4).
# ---------------------------------------------------------------------------

def test_sampled_ce_disabled_by_default_raises():
    logits = torch.randn(2, 5)
    target = torch.tensor([0, 1], dtype=torch.long)
    loss_mask = torch.tensor([True, True])
    log_q = torch.log(torch.full((5,), 0.2))
    with pytest.raises(RuntimeError):
        sampled_ce_loss(logits, target, loss_mask, log_q)  # enabled defaults False
    with pytest.raises(RuntimeError):
        sampled_ce_loss(logits, target, loss_mask, log_q, enabled=False)


def test_sampled_ce_toy_logq_correction_and_accidental_hit_masking():
    """K=5, N=4. Proposal q is heavily concentrated on class 2 so that, with
    overwhelming probability, class 2 is drawn among the 4 shared negatives.
    Row 0's target is also class 2, so this is an "accidental hit" that must
    be masked to -inf before softmax. Rows 1-3 have different targets, so
    class 2 appearing among their negatives is an ordinary negative -- not
    masked for them.

    We reproduce the exact negatives draw with an independently-seeded
    generator (same seed -> same torch.multinomial stream) and hand-compute
    the expected sampled-softmax loss (logQ subtracted from every column,
    accidental hits masked, CE with target index 0), then compare against
    sampled_ce_loss's returned (loss_sum, n_contributing) exactly.
    """
    K, N, Bn = 5, 4, 4
    q = torch.tensor([0.001, 0.001, 0.995, 0.0015, 0.0015], dtype=torch.float32)
    assert abs(float(q.sum().item()) - 1.0) < 1e-6
    log_q = torch.log(q)

    torch.manual_seed(2026)
    logits = torch.randn(Bn, K, dtype=torch.float32)
    target_class = torch.tensor([2, 0, 1, 3], dtype=torch.long)  # row 0 target == 2
    loss_mask = torch.tensor([True, True, True, True], dtype=torch.bool)

    seed = 12345
    gen_for_call = torch.Generator().manual_seed(seed)
    loss_sum, n_contrib = sampled_ce_loss(
        logits, target_class, loss_mask, log_q,
        n_negatives=N, enabled=True, generator=gen_for_call,
    )
    assert int(n_contrib.item()) == 4

    # Reproduce the identical negatives draw with a freshly-seeded generator.
    gen_reference = torch.Generator().manual_seed(seed)
    negatives = torch.multinomial(q, N, replacement=True, generator=gen_reference)

    # Overwhelmingly likely given q[2] = 0.995; assert it explicitly so the
    # "accidental hit masked" behaviour is actually exercised by this run.
    assert bool((negatives == 2).any()), (
        "expected class 2 among the sampled negatives (q[2]=0.995); "
        "rerun with a different seed if this ever fails"
    )

    candidates = torch.cat(
        [target_class.unsqueeze(1), negatives.unsqueeze(0).expand(Bn, -1)], dim=1
    )  # [Bn, N+1]
    cand_logits = torch.gather(logits, 1, candidates)
    log_nq = math.log(N) + log_q[candidates]
    cand_logits = cand_logits - log_nq

    neg_cols = candidates[:, 1:]
    hit = neg_cols == target_class.unsqueeze(1)
    # logQ correction (sec. C) applies to every column, including the
    # positive, BEFORE the accidental-hit mask is applied to negatives only.
    neg_logits = cand_logits[:, 1:].masked_fill(hit, float("-inf"))
    cand_logits = torch.cat([cand_logits[:, :1], neg_logits], dim=1)

    target_idx = torch.zeros(Bn, dtype=torch.long)
    expected_loss_sum = F.cross_entropy(cand_logits, target_idx, reduction="sum")

    assert torch.allclose(loss_sum, expected_loss_sum, atol=1e-5)
    # Row 0 really did have its own target among the negatives -> masked.
    assert bool(hit[0].any())
