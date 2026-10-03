"""Full-K dense output-gradient coverage; a negative control shows dropping non-interacted updates changes the
result. Derived from basic softmax calculus.

Hand derivation: full-K softmax cross-entropy gives every class k a strictly positive probability
softmax_k = exp(logit_k) / sum_j exp(logit_j) > 0 (softmax is never exactly zero for finite logits). The
per-example gradient of the head weight row k is (softmax_k - 1{k=target}) * q. Summed over a batch where
class k is NEVER a target, this becomes (sum_i softmax_{i,k} * q_i), which is a sum of terms that are each
generically nonzero (softmax_{i,k} > 0 and q_i is a generic nonzero vector) -- so EVERY row of the K x d head
table receives a nonzero gradient contribution from full-K CE, even from classes absent from the batch's
targets. A sparse ("only touch classes seen as targets") update would zero out the other rows and reach a
different result.
"""
from __future__ import annotations

import numpy as np
import torch
from fedsim_crosscheck_kit import fresh_adapter

from ppsi.fedsim.client import batch_ce


def test_every_head_row_gets_a_nonzero_gradient_even_for_absent_classes():
    K, d = 6, 4
    adapter = fresh_adapter(K=K, d=d, tied=False, seed=3)   # untied table: one row per class, no alias sharing
    rng = np.random.default_rng(0)
    from ppsi.fedsim.synthetic import client_examples
    batch = client_examples(5, K, 4, rng)
    # force targets to hit only classes {0, 1}: classes 2..5 are never targets in this batch
    batch["target_class"] = torch.tensor([0, 1, 0, 1, 0], dtype=torch.long)
    absent_classes = [2, 3, 4, 5]

    head = adapter.head_weight()   # [K, d], requires_grad (it IS output_embed, a leaf Parameter)
    logits = adapter.logits(adapter.query(batch))
    loss = batch_ce(logits, batch["target_class"])
    head.grad = None
    loss.backward()

    assert head.grad is not None
    row_norms = head.grad.abs().sum(dim=1)
    for k in absent_classes:
        assert float(row_norms[k]) > 0.0, (
            f"full-K CE must give class {k} (absent from this batch's targets) a nonzero gradient row")
    # a sanity floor: softmax is strictly positive everywhere, so this is not a fluke of scale
    assert float(row_norms.min()) > 1e-12


def test_negative_control_dropping_non_interacted_rows_changes_the_optimizer_result():
    # Compare one AdamW step using the REAL dense gradient vs a hand-simulated "sparse" bug that zeroes the
    # gradient rows for classes absent from the batch before the optimizer step. The two must diverge.
    K, d = 6, 4
    adapter_dense = fresh_adapter(K=K, d=d, tied=False, seed=3)
    adapter_sparse_bug = fresh_adapter(K=K, d=d, tied=False, seed=3)
    rng = np.random.default_rng(0)
    from ppsi.fedsim.synthetic import client_examples
    batch = client_examples(5, K, 4, rng)
    batch["target_class"] = torch.tensor([0, 1, 0, 1, 0], dtype=torch.long)
    absent_classes = [2, 3, 4, 5]

    def one_adamw_step(adapter, zero_absent_rows: bool):
        head = adapter.head_weight()
        opt = torch.optim.AdamW(adapter.shared_parameters(), lr=1e-2)
        opt.zero_grad(set_to_none=True)
        logits = adapter.logits(adapter.query(batch))
        loss = batch_ce(logits, batch["target_class"])
        loss.backward()
        if zero_absent_rows:                       # simulate a broken "only touch interacted classes" update
            with torch.no_grad():
                head.grad[absent_classes] = 0.0
        opt.step()
        return head.detach().clone()

    dense_result = one_adamw_step(adapter_dense, zero_absent_rows=False)
    sparse_bug_result = one_adamw_step(adapter_sparse_bug, zero_absent_rows=True)
    for k in absent_classes:
        assert not torch.allclose(dense_result[k], sparse_bug_result[k]), (
            f"negative control: dropping the update for absent class {k} must change the result")
