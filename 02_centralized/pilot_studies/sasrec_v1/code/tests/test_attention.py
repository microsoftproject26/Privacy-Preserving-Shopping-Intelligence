"""Attention correctness (05_TESTS_AND_PRETRAIN_GATE.md sec.B, items 10-16;
Fable #1 PAD/layout equivalence, #2 prefix consistency/causality).

Every mandatory check here has a negative control that applies an in-memory
mutant (tests/mutants/model_mutants.py) and asserts the SAME check now fires.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from conftest import make_batch, make_config, make_model, forward_kwargs, TINY_K
from mutants import model_mutants as mm

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Independent (hand-written, not copy-pasted from sasrec_model.py) reference
# implementation of the per-block math documented in 03_ARCHITECTURE_CONTRACT
# sec.3-4 and sasrec_model.py's module docstring. Runs in whatever dtype the
# model's parameters are in (float64 for the FP64 hand test).
# ---------------------------------------------------------------------------
def manual_hidden_sequence(model, item_tokens, lengths, attention_mask):
    d = model.d
    B, T = item_tokens.shape
    dtype = model.item_embed.weight.dtype
    ar = torch.arange(T)
    valid = ar.unsqueeze(0) < lengths.unsqueeze(1)
    vmask = valid.unsqueeze(-1).to(dtype)
    pos = ar.unsqueeze(0).expand(B, T)
    # registry v3 input: LayerNorm_in(E[item] + P[pos]), no sqrt(d) scale
    x = F.layer_norm(model.item_embed(item_tokens) + model.pos_embed(pos), (d,),
                     model.input_norm.weight, model.input_norm.bias, model.input_norm.eps)
    x = x * vmask
    causal = model.causal[:T, :T]
    for blk in model.blocks:
        ln1 = F.layer_norm(x, (d,), blk.ln1.weight, blk.ln1.bias, blk.ln1.eps)
        qkv = ln1 @ blk.qkv.weight.t() + blk.qkv.bias
        q, k, v = qkv[..., :d], qkv[..., d:2 * d], qkv[..., 2 * d:3 * d]
        h, dh = blk.h, blk.dh

        def split(t):
            return t.view(B, T, h, dh).transpose(1, 2)  # [B,h,T,dh]

        qh, kh, vh = split(q), split(k), split(v)
        s = torch.matmul(qh, kh.transpose(-1, -2)) * (1.0 / math.sqrt(dh))
        s = s.masked_fill(~causal, float("-inf"))
        a = torch.softmax(s, dim=-1)
        o = torch.matmul(a, vh).transpose(1, 2).reshape(B, T, d)
        o = o @ blk.out.weight.t() + blk.out.bias
        u = x + o
        ln2 = F.layer_norm(u, (d,), blk.ln2.weight, blk.ln2.bias, blk.ln2.eps)
        f1 = F.gelu(ln2 @ blk.ff1.weight.t() + blk.ff1.bias)
        f2 = f1 @ blk.ff2.weight.t() + blk.ff2.bias
        x = (u + f2) * vmask
    return x


# ---------------------------------------------------------------------------
# 05 item 10: FP64 hand-computable causal attention vs the model's own math.
# ---------------------------------------------------------------------------
def test_fp64_hand_computed_matches_padded_reference():
    cfg = make_config(group="item_only", d_model=6, blocks=1, heads=2, dropout=0.0, max_len=8)
    model = make_model(cfg, K=11, seed=3, impl="padded_reference").double().eval()

    lengths = torch.tensor([1, 3, 5])
    batch = make_batch(lengths, T=5, K=11, seed=1)
    item_tokens, attention_mask = batch["item_tokens"], batch["attention_mask"]

    got = model.hidden_states(item_tokens=item_tokens, lengths=lengths, attention_mask=attention_mask)
    expected = manual_hidden_sequence(model, item_tokens, lengths, attention_mask)
    assert torch.allclose(got, expected, atol=1e-10, rtol=1e-10)

    # Negative control: disable causal masking -> hand math and model math must diverge.
    with mm.disable_causal_mask(model):
        broken = model.hidden_states(item_tokens=item_tokens, lengths=lengths, attention_mask=attention_mask)
    assert not torch.allclose(broken, expected, atol=1e-8)


# ---------------------------------------------------------------------------
# 05 items 11/12, Fable #2 causality half: future mutation cannot change an
# earlier position's representation or its gradient; a positive control
# (changing an allowed historical token) DOES change the later output.
# ---------------------------------------------------------------------------
def test_future_mutation_does_not_leak_backward_or_forward():
    cfg = make_config(group="item_only", d_model=12, blocks=2, heads=2, dropout=0.0, max_len=10)
    model = make_model(cfg, K=TINY_K, seed=5, impl="unpadded")

    T, p0 = 6, 2
    lengths = torch.tensor([T])
    # Distinct ids per position so each position's embedding row is unique --
    # lets us attribute a nonzero/zero gradient to a specific position.
    ids = torch.tensor(np.random.default_rng(7).choice(np.arange(3, 3 + TINY_K), size=T, replace=False))
    item_tokens = ids.unsqueeze(0).clone()
    attention_mask = torch.ones(1, T, dtype=torch.bool)

    seq = model.hidden_states(item_tokens=item_tokens, lengths=lengths, attention_mask=attention_mask)
    y = seq[:, p0, :].sum()
    model.zero_grad()
    y.backward()
    grad = model.item_embed.weight.grad
    future_rows = ids[p0 + 1:]
    past_present_rows = ids[:p0 + 1]
    assert torch.count_nonzero(grad[future_rows]) == 0, "future tokens must get exactly-zero gradient"
    assert torch.count_nonzero(grad[past_present_rows]) > 0, "sanity: earlier rows DO get gradient (not dead net)"

    # Forward-side check: mutate a FUTURE token, hidden states at <= p0 unchanged.
    mutated_future = item_tokens.clone()
    mutated_future[0, p0 + 1] = int(np.random.default_rng(9).choice(
        [t for t in range(3, 3 + TINY_K) if t not in ids.tolist()]))
    with torch.no_grad():
        seq_future = model.hidden_states(item_tokens=mutated_future, lengths=lengths, attention_mask=attention_mask)
    assert torch.allclose(seq[:, :p0 + 1, :], seq_future[:, :p0 + 1, :], atol=1e-6), \
        "mutating a FUTURE token must not change earlier positions"

    # Positive control: mutate an ALLOWED historical token (<=p0) -> output at p0 DOES change.
    mutated_past = item_tokens.clone()
    mutated_past[0, 0] = int(np.random.default_rng(11).choice(
        [t for t in range(3, 3 + TINY_K) if t not in ids.tolist()]))
    with torch.no_grad():
        seq_past = model.hidden_states(item_tokens=mutated_past, lengths=lengths, attention_mask=attention_mask)
    assert not torch.allclose(seq[:, p0, :], seq_past[:, p0, :], atol=1e-6), \
        "positive control failed: changing an allowed historical token must change a later output"

    # Negative control: with causal masking disabled, the future-mutation
    # invariant must fail (future DOES leak backward into an earlier output).
    with mm.disable_causal_mask(model):
        model.zero_grad()
        seq2 = model.hidden_states(item_tokens=item_tokens, lengths=lengths, attention_mask=attention_mask)
        seq2[:, p0, :].sum().backward()
        grad2 = model.item_embed.weight.grad
    assert torch.count_nonzero(grad2[future_rows]) > 0, \
        "negative control failed: disabling the causal mask should leak future gradient"


# ---------------------------------------------------------------------------
# 05 item 13: mask direction/diagonal/prefix1/right PAD/no NaN rows.
# ---------------------------------------------------------------------------
def test_causal_mask_direction_diagonal_and_no_nan():
    cfg = make_config(group="item_only", d_model=8, blocks=2, heads=2, dropout=0.0, max_len=6)
    model = make_model(cfg, K=TINY_K, seed=1)
    causal = model.causal
    n = causal.shape[0]
    assert bool(causal.diagonal().all()), "self-attention (diagonal) must be allowed"
    for i in range(1, n):
        assert not bool(causal[i - 1, i]), "earlier query must not attend a later key"
        assert bool(causal[i, i - 1]), "later query may attend an earlier key"

    # prefix1: a single-token window must not crash and must be NaN-free.
    lengths = torch.tensor([1])
    batch = make_batch(lengths, T=1, K=TINY_K, seed=2)
    out = model(**forward_kwargs(batch))
    assert torch.isfinite(out).all()

    # right PAD, mixed lengths incl. one full-length row: no NaN anywhere,
    # including in the PAD query rows of the padded-reference impl.
    lengths2 = torch.tensor([1, 3, 6])
    batch2 = make_batch(lengths2, T=6, K=TINY_K, seed=4)
    model_padded = make_model(cfg, K=TINY_K, seed=1, impl="padded_reference")
    seq = model_padded.hidden_states(**forward_kwargs(batch2))
    assert torch.isfinite(seq).all(), "no NaN rows allowed anywhere, including PAD query rows"
    pad_positions = ~batch2["attention_mask"]
    assert torch.count_nonzero(seq[pad_positions]) == 0, "PAD rows of hidden_states must be exactly zero"


# ---------------------------------------------------------------------------
# 05 item 14: eval dropout exactly zero; repeated evals bitwise identical.
# ---------------------------------------------------------------------------
def test_eval_dropout_is_zero_and_repeated_evals_are_bitwise_identical():
    cfg = make_config(group="item_only", d_model=16, blocks=2, heads=4, dropout=0.5, max_len=10)
    model = make_model(cfg, K=TINY_K, seed=2)
    model.eval()
    lengths = torch.tensor([4, 10])
    batch = make_batch(lengths, T=10, K=TINY_K, seed=6)
    with torch.no_grad():
        out1 = model(**forward_kwargs(batch))
        out2 = model(**forward_kwargs(batch))
    assert torch.equal(out1, out2), "eval-mode forward must be bitwise deterministic (dropout==0)"

    # Negative control: leaving the model in train() mode with dropout>0
    # must make repeated forwards differ -- proves the eval-mode check has
    # teeth (it isn't trivially true regardless of dropout state).
    model.train()
    torch.manual_seed(1)
    with torch.no_grad():
        out3 = model(**forward_kwargs(batch))
    torch.manual_seed(2)
    with torch.no_grad():
        out4 = model(**forward_kwargs(batch))
    assert not torch.equal(out3, out4), \
        "negative control failed: train-mode dropout>0 should make repeated forwards differ"


# ---------------------------------------------------------------------------
# 05 item 15: unpadded vs padded_reference agree (outputs AND gradients),
# tolerance 1e-5, on a length histogram including 1 and 50.
# ---------------------------------------------------------------------------
def test_unpadded_vs_padded_reference_outputs_and_gradients_agree():
    cfg = make_config(group="item_only", d_model=32, blocks=2, heads=4, dropout=0.0, max_len=50)
    seed = 42
    m_unpadded = make_model(cfg, K=TINY_K, seed=seed, impl="unpadded")
    m_padded = make_model(cfg, K=TINY_K, seed=seed, impl="padded_reference")

    rng = np.random.default_rng(0)
    lengths = np.concatenate([[1, 50], rng.integers(1, 51, size=14)]).astype(np.int64)
    batch = make_batch(lengths, T=50, K=TINY_K, seed=8)
    kw = forward_kwargs(batch)

    out_u = m_unpadded(**kw)
    out_p = m_padded(**kw)
    assert torch.allclose(out_u, out_p, rtol=1e-5, atol=1e-6), "valid outputs must agree within 1e-5 rel"

    loss_u = out_u.sum()
    loss_p = out_p.sum()
    m_unpadded.zero_grad()
    m_padded.zero_grad()
    loss_u.backward()
    loss_p.backward()
    for (name_u, p_u), (name_p, p_p) in zip(m_unpadded.named_parameters(), m_padded.named_parameters()):
        assert name_u == name_p
        if p_u.grad is None and p_p.grad is None:
            continue
        assert p_u.grad is not None and p_p.grad is not None, f"{name_u}: one impl produced no grad"
        assert torch.allclose(p_u.grad, p_p.grad, rtol=1e-5, atol=1e-6), f"{name_u}: gradients disagree"

    # Negative control: corrupt the unpadded PAD-scatter buffer by disabling
    # the causal mask on ONE impl only -> the agreement must break.
    with mm.disable_causal_mask(m_unpadded):
        out_u_broken = m_unpadded(**kw)
    assert not torch.allclose(out_u_broken, out_p, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# 05 item 16 + Fable #1: final query is the last REAL item; randomising
# PAD-column tokens (and, for rich, side/numeric/flag values) leaves real
# outputs and loss unchanged.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("impl", ["unpadded", "padded_reference"])
def test_pad_column_randomisation_leaves_output_unchanged(impl):
    cfg = make_config(group="item_only", d_model=16, blocks=2, heads=2, dropout=0.0, max_len=12)
    model = make_model(cfg, K=TINY_K, seed=13, impl=impl)

    lengths = torch.tensor([1, 5, 12, 8])
    batch_a = make_batch(lengths, T=12, K=TINY_K, seed=20, pad_item_value=0)
    batch_b = make_batch(lengths, T=12, K=TINY_K, seed=20, pad_item_value=None)
    # Overwrite PAD columns of batch_b with high-entropy, in-vocab, non-zero ids.
    rng = np.random.default_rng(99)
    item_b = batch_b["item_tokens"].clone()
    for b in range(item_b.shape[0]):
        L = int(lengths[b])
        if L < 12:
            item_b[b, L:] = torch.from_numpy(rng.integers(3, 3 + TINY_K, size=12 - L))
    batch_b["item_tokens"] = item_b

    out_a = model(**forward_kwargs(batch_a))
    out_b = model(**forward_kwargs(batch_b))
    assert torch.allclose(out_a, out_b, atol=1e-6), \
        "randomising PAD-column tokens must not change real outputs (Fable #1)"

    # Negative control: force every column to be treated as 'valid' -> the
    # randomised PAD tokens now DO leak (compare the full hidden_states
    # sequence, which is robust to the h_last-gather index becoming
    # meaningless once the compacted-length invariant is broken by the mutant).
    with mm.force_all_positions_valid(model):
        seq_a_mut = model.hidden_states(**forward_kwargs(batch_a))
        seq_b_mut = model.hidden_states(**forward_kwargs(batch_b))
    assert not torch.allclose(seq_a_mut, seq_b_mut, atol=1e-6), \
        "negative control failed: force_all_positions_valid should let PAD tokens leak"


def test_final_query_is_last_real_item_not_column_zero():
    cfg = make_config(group="item_only", d_model=16, blocks=2, heads=2, dropout=0.0, max_len=10)
    model = make_model(cfg, K=TINY_K, seed=17)

    lengths = torch.tensor([1, 4, 10])
    batch = make_batch(lengths, T=10, K=TINY_K, seed=21)
    seq = model.hidden_states(**forward_kwargs(batch))
    q = model.query(**forward_kwargs(batch))
    expected_last = torch.stack([seq[b, int(lengths[b]) - 1, :] for b in range(len(lengths))])
    expected_q = model.final_norm(expected_last)
    assert torch.allclose(q, expected_q, atol=1e-6)
    # For the length-1 row, column-1 (index 1) is PAD; reading it would be wrong.
    assert not torch.allclose(q[0], model.final_norm(seq[0, 1, :]), atol=1e-6)

    with mm.blind_column_zero_readout(model):
        q_blind = model.query(**forward_kwargs(batch))
    assert not torch.allclose(q_blind, q, atol=1e-6), \
        "negative control failed: blind-column-0 readout must diverge from the real readout"


# ---------------------------------------------------------------------------
# Fable #2: hidden state at position p of a length-L window equals the
# readout of the window truncated to p+1; target_class can never influence
# forward output (it is not even a forward argument).
# ---------------------------------------------------------------------------
def test_prefix_consistency_truncated_window_matches():
    cfg = make_config(group="item_only", d_model=16, blocks=2, heads=2, dropout=0.0, max_len=16)
    model = make_model(cfg, K=TINY_K, seed=23).eval()

    L = 10
    lengths_full = torch.tensor([L])
    batch_full = make_batch(lengths_full, T=L, K=TINY_K, seed=31)
    seq_full = model.hidden_states(**forward_kwargs(batch_full))

    for p in (0, 3, L - 1):
        trunc_len = p + 1
        item_trunc = batch_full["item_tokens"][:, :trunc_len].clone()
        mask_trunc = batch_full["attention_mask"][:, :trunc_len].clone()
        pos_trunc = batch_full["position_ids"][:, :trunc_len].clone()
        lengths_trunc = torch.tensor([trunc_len])
        seq_trunc = model.hidden_states(item_tokens=item_trunc, lengths=lengths_trunc,
                                         attention_mask=mask_trunc, position_ids=pos_trunc)
        assert torch.allclose(seq_full[:, p, :], seq_trunc[:, p, :], atol=1e-6), \
            f"hidden state at position {p} must match the readout of the window truncated to p+1"

    # Negative control: disabling the causal mask breaks prefix consistency
    # (a later, now-visible-to-nobody-truncated token would otherwise change p).
    with mm.disable_causal_mask(model):
        seq_full_mut = model.hidden_states(**forward_kwargs(batch_full))
        item_trunc = batch_full["item_tokens"][:, :4].clone()
        mask_trunc = batch_full["attention_mask"][:, :4].clone()
        pos_trunc = batch_full["position_ids"][:, :4].clone()
        seq_trunc_mut = model.hidden_states(item_tokens=item_trunc, lengths=torch.tensor([4]),
                                             attention_mask=mask_trunc, position_ids=pos_trunc)
    assert not torch.allclose(seq_full_mut[:, 3, :], seq_trunc_mut[:, 3, :], atol=1e-6)


def test_target_class_is_not_a_forward_argument():
    cfg = make_config(group="item_only", d_model=8, blocks=1, heads=2, dropout=0.0, max_len=6)
    model = make_model(cfg, K=TINY_K, seed=27)
    lengths = torch.tensor([4])
    batch = make_batch(lengths, T=6, K=TINY_K, seed=33)
    kw = forward_kwargs(batch)
    kw_with_target = dict(kw, target_class=batch["target_class"])
    with pytest.raises(TypeError):
        model.forward(**kw_with_target)
