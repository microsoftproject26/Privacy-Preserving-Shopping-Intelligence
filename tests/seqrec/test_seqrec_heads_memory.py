"""The GRU tied alias is counted once; the SASRec input and output tables are distinct; no [B, L, K] tensor.

  * GRU: the head IS the item table (a view of item_embed rows [3, 3+K), same storage); there is exactly one item
    Parameter and no head Parameter besides output_bias; nothing is registered twice; manifest shared numel =
    parameter inventory total = sum of unique parameters; visit bytes = 4 x that; the head gradient reaches the item
    table rows of classes never seen in the input;
  * SASRec: output_embed [K, d] is its own Parameter with no storage overlap with item_embed; both are SHARED in the
    manifest; the dense CE gradient reaches every output row while the input table gets gradient only on rows of
    tokens in the batch; post_step recentering moves only W_out / b;
  * no [B, L, K]: every tensor produced by every aten op of a full forward + backward (recorded with a dispatch mode)
    that has a K-sized dimension is a bias [K], a table [K, d] (or its gradient / transpose) or a [B, K] row block.
Negative controls: an untied GRU head copy, a tied SASRec head, and dense all-position logits (both families) fail.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from seqrec_testkit import (
    FAMS,
    KShapeRecorder,
    adapter,
    batch,
    kwargs_for,
    nc,
    seeded,
    train_step_under,
)
from torch import nn

from ppsi.fedsim.adapter import Role
from ppsi.fedsim.comm import visit_bytes
from ppsi.seqrec.build import inventory

KB = 997           # a K distinct from every other dimension used here (B, d, T, N, B*T, K + 3)
B = 3


# ------------------------------------------------------------------------------------------------ GRU tied head
def _gru_tied_check(a):
    m = a.module
    W = a.head_weight()
    E = m.item_embed.weight
    assert W.untyped_storage().data_ptr() == E.untyped_storage().data_ptr(), "head must share the item table storage"
    assert W.data_ptr() == E[3].data_ptr() and W.shape == (m.K, 128)
    table_shapes = {(m.K, 128), (m.item_vocab_size, 128)}
    item_like = [n for n, p in m.named_parameters() if tuple(p.shape) in table_shapes]
    assert item_like == ["item_embed.weight"], f"exactly one item-table Parameter expected, got {item_like}"


def test_gru_tied_alias_counted_once():
    a = adapter("GRU", K=64)
    m = a.module
    _gru_tied_check(a)
    head_params = [n for n, _ in m.named_parameters() if n.startswith("output")]
    assert head_params == ["output_bias"]
    assert len(list(m.named_parameters(remove_duplicate=False))) == len(list(m.named_parameters()))
    unique = sum(p.numel() for p in {id(p): p for p in m.parameters()}.values())
    inv = inventory(m)
    assert a.manifest.shared_numel == unique == inv["total"] and a.manifest.alias_keys == ()
    assert inv["item_embed"] == (64 + 3) * 128 and inv["output_bias"] == 64
    assert visit_bytes(a.manifest)["download_bytes"] == 4 * unique
    b = batch(4, seed=1)
    b["item_tokens"] = torch.where(b["attention_mask"], torch.full_like(b["item_tokens"], 3), b["item_tokens"])
    m.zero_grad(set_to_none=True)
    F.cross_entropy(a.scores(b), b["target_class"]).backward()
    g = m.item_embed.weight.grad
    assert bool((g[4:3 + 64].abs().sum(dim=1) > 0).all()), "head gradient must reach every class row of the tied table"
    assert float(g[0].abs().sum()) == 0.0, "PAD row (padding_idx 0) gets no gradient"


# ------------------------------------------------------------------------------------------------ SASRec untied head
def _sasrec_distinct_check(a):
    m = a.module
    W, E = a.head_weight(), m.item_embed.weight
    assert W is m.output_embed and isinstance(W, nn.Parameter) and W.shape == (m.K, m.d)
    assert W.untyped_storage().data_ptr() != E.untyped_storage().data_ptr(), "input and output tables share storage"
    assert a.manifest.entries["output_embed"].role == Role.SHARED
    assert a.manifest.entries["item_embed.weight"].role == Role.SHARED and a.manifest.alias_keys == ()


def test_sasrec_input_and_output_tables_distinct():
    a = adapter("SASREC", K=64)
    m = a.module
    _sasrec_distinct_check(a)
    b = batch(4, seed=2, min_len=2, max_len=5)
    m.train()
    m.zero_grad(set_to_none=True)
    with seeded(3):
        F.cross_entropy(a.scores(b), b["target_class"]).backward()
    assert bool((m.output_embed.grad.abs().sum(dim=1) > 0).all()), "dense full-K CE must reach every output row"
    used = torch.unique(b["item_tokens"][b["attention_mask"]])
    unused = torch.ones(m.item_vocab, dtype=torch.bool)
    unused[used] = False
    assert float(m.item_embed.weight.grad[unused].abs().sum()) == 0.0, \
        "the input table must get gradient only through the tokens in the window (it is not the output head)"
    E0 = m.item_embed.weight.detach().clone()
    with torch.no_grad():
        m.output_embed.add_(0.3)
        m.output_bias.add_(1.0)
    a.post_step()
    assert torch.equal(E0, m.item_embed.weight.detach()), "recentering must not touch the input table"
    assert float(m.output_embed.detach().mean(0).abs().max()) < 1e-6 and abs(float(m.output_bias.mean())) < 1e-6


# ------------------------------------------------------------------------------------------------ no [B, L, K]
def _no_blk_check(family: str, dense: bool = False):
    a = adapter(family, K=KB)
    m = a.module
    m.train()
    b = batch(B, K=KB, seed=4, min_len=5, max_len=7)
    rec = KShapeRecorder()

    def step():
        with seeded(5):
            if not dense:
                return F.cross_entropy(a.scores(b), b["target_class"])
            if family == "SASREC":        # MUTANT: SASRec-style all-position logits [B, T, K]
                seq = m.hidden_states(**kwargs_for(m, b))
                return F.cross_entropy((m.final_norm(seq) @ m.output_embed.t()).flatten(0, 1),
                                       torch.randint(0, KB, (seq.shape[0] * seq.shape[1],)))
            kw = kwargs_for(m, b)         # MUTANT: GRU per-position logits [B, L, K]
            x = m._embed_and_project(**{k: v for k, v in kw.items()
                                        if k not in ("lengths", "attention_mask", "user_context", "user_context_masks")})
            out, _ = m.gru(x)
            logits = out[..., :128] @ a.head_weight().t()
            return F.cross_entropy(logits.flatten(0, 1), torch.randint(0, KB, (logits.shape[0] * logits.shape[1],)))

    train_step_under(rec, step)
    kt = rec.k_tensors(KB)
    assert any(s == (KB, a.query_dim) and "mm" in f for f, s in kt), "sanity: the backward head-gradient op was recorded"
    bad = rec.violations(KB, allowed_rows=(1, B, a.query_dim))
    assert not bad, f"a [B, L, K]-like tensor was materialized: {bad[:4]} (peak rows {rec.peak_k_rows(KB)})"
    assert rec.peak_k_rows(KB) == a.query_dim                    # the largest K-tensor is the head table itself


@pytest.mark.parametrize("family", FAMS)
def test_no_BLK_tensor_in_forward_and_backward(family):
    _no_blk_check(family)


# ------------------------------------------------------------------------------------------------ negative controls
@nc("GRU head is an untied COPY of the item rows (a second parameter, double-counted)")
def test_nc_gru_untied_copy():
    a = adapter("GRU", K=64)
    m = a.module
    m.head_copy = nn.Parameter(m.item_embed.weight[3:3 + m.K].detach().clone())
    a.head_weight = lambda: m.head_copy
    _gru_tied_check(a)


@nc("SASRec head tied back to the input table (the configuration that drifted under dense-softmax Adam)")
def test_nc_sasrec_tied_head():
    a = adapter("SASREC", K=64)
    m = a.module
    a.head_weight = lambda: m.item_embed.weight[3:3 + m.K]
    _sasrec_distinct_check(a)


@nc("dense all-position logits materialize a [B, L, K] tensor")
@pytest.mark.parametrize("family", FAMS)
def test_nc_dense_all_position_logits(family):
    _no_blk_check(family, dense=True)
