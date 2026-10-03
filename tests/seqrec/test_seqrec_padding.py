"""Padding and readout invariance under right padding; no all-padding row; masked-SDPA conventions.

Both families (SASRec in both implementations), eval mode:
  * width invariance: the same decisions stored with a narrower (or, for the GRU, wider) right-padded window give a
    bitwise-identical query;
  * PAD-content invariance: arbitrary values in every PAD cell (tokens, numeric, flags) leave the query bitwise
    unchanged, and the query's gradient w.r.t. PAD-cell inputs is exactly 0;
  * batch-composition invariance: a decision's query alone equals its query next to longer neighbours;
  * readout = the LAST REAL event (lengths - 1), never the last column;
  * no all-padding row: a length-0 decision is refused; SASRec also refuses a non-contiguous or inconsistent mask;
    the padded reference is finite everywhere with exactly-zero PAD rows; every causal row keeps its diagonal;
  * masked SDPA conventions: the explicit attention equals torch SDPA with a BOOLEAN mask where True = attend (the
    SDPA convention), with is_causal=True and with the additive 0/-inf float mask; for right-padded rows an extra
    key-padding mask changes nothing on real query rows (PAD keys are always in the future).
Negative controls: a last-column readout (both families) and the inverted boolean SDPA mask fail these checks.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from seqrec_testkit import FAMS, assert_raises, batch, is_gru, kwargs_for, maxdiff, model, nc, query

from ppsi.seqrec.synthetic import crop, pad_to, take

CASES = [("GRU", "unpadded"), ("SASREC", "unpadded"), ("SASREC", "padded_reference")]
VOC = {"item_tokens": 67, "category_tokens": 663, "brand_tokens": 3956, "main_category_tokens": 16,
       "daypart_tokens": 8, "weekday_tokens": 10}


def _m(family, impl="unpadded"):
    return model(family, 3, impl=impl).eval()


def _q(m, b):
    with torch.no_grad():
        return query(m, b)


def _garbage_in_pad(b: dict, seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    pad = ~b["attention_mask"]
    out = dict(b)
    for ch, hi in VOC.items():
        out[ch] = torch.where(pad, torch.randint(3, hi, pad.shape, generator=g), b[ch])
    num = b["event_numeric_features"]
    out["event_numeric_features"] = torch.where(pad[..., None], torch.randn(num.shape, generator=g) * 5, num)
    out["event_quality_flags"] = torch.where(pad[..., None], torch.ones_like(b["event_quality_flags"]),
                                             b["event_quality_flags"])
    return out


@pytest.mark.parametrize("family,impl", CASES)
def test_right_padding_width_invariance(family, impl):
    m = _m(family, impl)
    b = batch(6, seed=1, min_len=2, max_len=12)
    q50 = _q(m, b)
    assert torch.equal(q50, _q(m, crop(b, 12))), "a narrower right-padded window changed the query"
    assert torch.equal(q50, _q(m, crop(b, 20)))
    if is_gru(m):
        assert torch.equal(q50, _q(m, pad_to(b, 80))), "a wider right-padded window changed the GRU query"


@pytest.mark.parametrize("family,impl", CASES)
def test_pad_content_invariance_and_zero_pad_gradient(family, impl):
    m = _m(family, impl)
    b = batch(6, seed=2, min_len=1, max_len=30)
    assert torch.equal(_q(m, b), _q(m, _garbage_in_pad(b, 5))), "PAD-cell content changed the query"
    num = b["event_numeric_features"].clone().requires_grad_(True)
    kw = kwargs_for(m, b)
    kw["event_numeric_features"] = num
    m.zero_grad(set_to_none=True)
    q = m.query(**kw)
    w = torch.randn(q.shape, generator=torch.Generator().manual_seed(1))   # sum(q) is degenerate after LayerNorm
    (q * w).sum().backward()
    pad = ~b["attention_mask"]
    assert float(num.grad[pad].abs().sum()) == 0.0, "the query depends on PAD-cell inputs"
    assert float(num.grad[b["attention_mask"]].abs().sum()) > 0.0


@pytest.mark.parametrize("family,impl", CASES)
def test_batch_composition_invariance(family, impl):
    m = _m(family, impl)
    b = batch(6, seed=4, min_len=1, max_len=40)
    qb = _q(m, b)
    worst = 0.0
    for i in range(6):
        one = crop(take(b, [i]), int(b["lengths"][i]))
        worst = max(worst, maxdiff(_q(m, one)[0], qb[i]))
    assert worst < 2e-6, f"a decision's query depends on its batch neighbours: max |d| {worst:.3g}"


def _gru_ref_query(m, b, last_column: bool = False):
    """Reference GRU readout computed by hand: per decision, the final hidden state of the UNPADDED sequence (or,
    for the mutant, the output at the last stored column of the padded sequence)."""
    kw = kwargs_for(m, b)
    x = m._embed_and_project(**{k: v for k, v in kw.items() if k not in ("lengths", "attention_mask",
                                                                           "user_context", "user_context_masks")})
    rows = []
    for i in range(x.shape[0]):
        n = int(b["lengths"][i])
        xi = x[i:i + 1] if last_column else x[i:i + 1, :n]
        out, _ = m.gru(xi)
        rows.append(out[:, -1])
    h = torch.cat(rows, 0)
    uc = m.user_ctx_proj(torch.cat([b["user_context"], b["user_context_masks"]], dim=-1))
    return m.readout_dropout(m.readout_proj(torch.cat([h, uc], dim=-1)))


def _sasrec_ref_query(m, b, last_column: bool = False):
    kw = kwargs_for(m, b)
    seq = m.hidden_states(**kw)
    idx = (torch.full_like(b["lengths"], seq.shape[1]) if last_column else b["lengths"]) - 1
    h = seq[torch.arange(seq.shape[0]), idx]
    side = {k: v for k, v in kw.items() if k not in ("item_tokens", "lengths", "attention_mask", "position_ids")}
    return m._readout(h, side)


def _readout_check(family, impl, last_column: bool):
    m = _m(family, impl)
    b = batch(6, seed=6, min_len=2, max_len=15)
    with torch.no_grad():
        ref = _gru_ref_query(m, b, last_column) if is_gru(m) else _sasrec_ref_query(m, b, last_column)
        q = query(m, b)
    assert torch.allclose(q, ref, atol=2e-6, rtol=0), f"readout is not the last real event: max |d| {maxdiff(q, ref):.3g}"


@pytest.mark.parametrize("family,impl", CASES)
def test_readout_is_the_last_real_event(family, impl):
    _readout_check(family, impl, last_column=False)


@pytest.mark.parametrize("family", FAMS)
def test_no_all_padding_row(family):
    m = _m(family)
    b = batch(3, seed=7, min_len=2, max_len=6)
    z = dict(b)
    z["lengths"] = b["lengths"].clone()
    z["lengths"][1] = 0
    z["attention_mask"] = torch.arange(b["attention_mask"].shape[1])[None, :] < z["lengths"][:, None]
    z["position_ids"] = torch.arange(1, b["attention_mask"].shape[1] + 1)[None, :] * z["attention_mask"]
    assert_raises((ValueError, RuntimeError), query, m, z)          # a length-0 (all-padding) decision is refused
    if not is_gru(m):
        hole = dict(b)
        hole["attention_mask"] = b["attention_mask"].clone()
        hole["attention_mask"][0, 0] = False                        # non-contiguous prefix
        assert_raises(ValueError, query, m, hole)
        wrongpos = dict(b)
        wrongpos["position_ids"] = b["position_ids"] + b["attention_mask"].long()
        assert_raises(ValueError, query, m, wrongpos)
        longer = dict(b)
        longer["lengths"] = b["lengths"] + 1                        # lengths disagree with the mask
        assert_raises(ValueError, query, m, longer)


def test_padded_reference_finite_and_pad_rows_zero():
    m = _m("SASREC", "padded_reference")
    b = batch(5, seed=8, min_len=1, max_len=9)
    with torch.no_grad():
        seq = m.hidden_states(**kwargs_for(m, b))
    T = seq.shape[1]
    assert torch.isfinite(seq).all()
    assert torch.count_nonzero(seq[~b["attention_mask"][:, :T]]) == 0, "PAD rows must be exactly zero"
    c = m.causal
    assert bool(c.diagonal().all()) and bool(c.any(dim=1).all()), "every query row must keep at least its diagonal"
    assert not bool(c.triu(1).any()), "no query may see a later key"


# ------------------------------------------------------------------------------------------------ SDPA conventions
def _attention_inputs(m, B=3, T=9, seed=0):
    blk = m.blocks[0]
    g = torch.Generator().manual_seed(seed)
    qkv = torch.randn(B, T, 3 * blk.d, generator=g)
    q, k, v = qkv.view(B, T, 3, blk.h, blk.dh).permute(2, 0, 3, 1, 4)
    return blk, qkv, q, k, v


def _sdpa_check(invert: bool = False):
    m = _m("SASREC")
    blk, qkv, q, k, v = _attention_inputs(m)
    B, T = qkv.shape[:2]
    with torch.no_grad():
        ours = blk._attend(qkv, m.causal)
        mask = m.causal[:T, :T]
        sd = F.scaled_dot_product_attention(q, k, v, attn_mask=(~mask if invert else mask), dropout_p=0.0)
        sd = sd.transpose(1, 2).reshape(B, T, blk.d)
    assert torch.allclose(ours, sd, atol=1e-5, rtol=0), f"SDPA(bool mask, True = attend) mismatch: {maxdiff(ours, sd)}"
    return m, blk, qkv, q, k, v, ours


def test_masked_sdpa_conventions():
    m, blk, qkv, q, k, v, ours = _sdpa_check()
    B, T = qkv.shape[:2]
    with torch.no_grad():
        causal = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, blk.d)
        add = torch.zeros(T, T).masked_fill(~m.causal[:T, :T], float("-inf"))
        additive = F.scaled_dot_product_attention(q, k, v, attn_mask=add).transpose(1, 2).reshape(B, T, blk.d)
    assert torch.allclose(ours, causal, atol=1e-5, rtol=0)
    assert torch.allclose(ours, additive, atol=1e-5, rtol=0)
    assert blk.dh == 64 and math.isclose(1.0 / math.sqrt(blk.dh), 0.125)       # SDPA default scale 1/sqrt(dh)
    # right padding: masking PAD keys as well changes nothing on the real query rows
    lengths = torch.tensor([9, 4, 1])
    keyvalid = torch.arange(T)[None, :] < lengths[:, None]                       # [B, T]
    full = (m.causal[:T, :T][None] & keyvalid[:, None, :])[:, None]              # [B, 1, T, T]
    with torch.no_grad():
        kp = F.scaled_dot_product_attention(q, k, v, attn_mask=full).transpose(1, 2).reshape(B, T, blk.d)
    real = keyvalid
    assert torch.allclose(ours[real], kp[real], atol=1e-5, rtol=0), "PAD keys influenced a real query row"


# ------------------------------------------------------------------------------------------------ negative controls
@nc("readout taken at the last stored column instead of the last real event")
@pytest.mark.parametrize("family", FAMS)
def test_nc_last_column_readout(family):
    _readout_check(family, "unpadded", last_column=True)


@nc("SDPA called with the inverted boolean convention (True = masked, the nn.MultiheadAttention key_padding style)")
def test_nc_inverted_sdpa_mask():
    _sdpa_check(invert=True)
