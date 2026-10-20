"""T32 on the real families at the full K (closes the QA-03 gap: T32 was covered only on TinyRec, K = 6).

Proves: on the Clean V2 GRU (tied head: item_embed rows 3..3+K) and SASRec (untied W_out), ONE full-K CE step on 3
synthetic decisions gives a non-zero gradient to EVERY one of the 158,486 head rows (softmax mass is never exactly
zero in FP32 here), so a dense upload is required; and dropping the non-interacted rows' gradient changes the
AdamW update (the negative control shows the check detects it). Does not prove: anything about sparse protocols.
"""
from __future__ import annotations

import pytest
import torch

from fl.client import LocalSolver, make_optimizer
from fl.families import build_v2, synthetic_examples_v2
from fl_testkit import nc

K = 158_486


def _head_grad(adapter):
    m = adapter.module
    return m.item_embed.weight.grad[3:3 + K] if hasattr(m, "readout_proj") else m.output_embed.grad


def _step(family, mask_uninteracted: bool):
    torch.set_num_threads(4)
    a = build_v2(family, seed=2026, K=K).adapter
    b = synthetic_examples_v2(3, K, torch.Generator().manual_seed(5))
    opt = make_optimizer(a, LocalSolver(lr=1e-3))
    a.module.train()
    torch.manual_seed(0)
    loss = torch.nn.functional.cross_entropy(a.scores(b), b["target_class"])
    opt.zero_grad(set_to_none=True)
    loss.backward()
    g = _head_grad(a)
    if mask_uninteracted:                                     # WRONG: keep only the targets' rows
        keep = torch.zeros(K, dtype=torch.bool)
        keep[b["target_class"]] = True
        g[~keep] = 0.0
    zero_rows = int((g.abs().sum(1) == 0).sum())
    before = [p.detach().clone() for p in a.shared_parameters()]
    opt.step()
    moved = int(sum(int((p.detach() != q).any(1).sum()) if p.ndim == 2 and p.shape[0] >= K else 0
                    for p, q in zip(a.shared_parameters(), before)))
    return zero_rows, moved


def check_dense(family, mask):
    zero_rows, moved = _step(family, mask)
    assert zero_rows == 0, f"{family}: {zero_rows} of {K} head rows got no gradient (dense upload required)"
    assert moved >= K, f"{family}: only {moved} head rows moved"


@pytest.mark.parametrize("family", ["GRU", "SASREC"])
def test_t32_every_head_row_gets_gradient_full_k(family):
    check_dense(family, mask=False)


@nc("drop the non-interacted rows' gradient (sparse-update assumption)")
def test_nc_t32_sparse_rows():
    check_dense("SASREC", mask=True)
