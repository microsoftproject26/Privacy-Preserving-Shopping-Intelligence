"""SCHEMA-01R1: the extension channels of DECISION_LOG 47 / 48 in v4models (both families).

  * embedding shapes: session_offset_tokens (vocab 9, dim 4), session_mode_main_category_tokens (G13; vocab 17, dim 8);
    padding_idx 0; row 0 is exactly zero at init (SASRec re-zeroes it after the seeded normal init) and gets exactly
    zero gradient;
  * PAD handling: changing the extension tokens on PAD positions changes nothing (bitwise; SASRec unpadded and padded
    reference); changing them on a real position changes the output; a batch without a kept key is refused;
  * the G13 decision-level values and their mask are appended as the last user_context / mask columns by the adapter's
    projection (values 2 columns, mask 1 column; the mask is an input like the Clean V2 masks, not a gate) — checked
    against a hand-built batch; a value kept without its mask is refused by the loader (tests_fs);
  * the L0 / L3 guard is unchanged: a batch that ALSO carries the new keys gives bitwise the same output as the batch
    without them, and the L0 build still equals the MODEL-01 build (test_schema01_derived);
  * parameter counts at the full K (exact): L0S = L0 + 1,900 (GRU) / + 3,436 (SASRec); L3S = L3 + the same:
      GRU    +36 (9 x 4) +136 (17 x 8) embeddings, +1,536 input_proj (12 more inputs x 128), +192 user_ctx_proj (3 x 64)
      SASRec +36 +136 embeddings, +3,072 side_proj (12 more inputs x 256), +192 user_mlp (3 x 64).
Negative control (strict xfail): a projection that appends the G13 values without their mask.
MEASURED note: padding_idx is defence in depth only; both families already give PAD cells zero gradient
(GRU packs the sequence, SASRec's unpadded path never embeds PAD and the padded reference masks it).
"""
from __future__ import annotations

import gc

import pytest
import torch

from models_testkit import FAMS, K_SMALL, SCHEMA, assert_raises, nc
from v4fs import schemas as S
from v4models.build import build, module_forward
from v4models.synthetic import synthetic_examples_v2

SEED = 2026
L0, L3 = S.load_registered("L0_CLEAN_V2"), S.load_registered("L3_NEAR_DUPLICATE")
L0S, L3S = S.load_registered("L0S_CLEAN_V2_SESSION"), S.load_registered("L3S_NEAR_DUPLICATE_SESSION")
EXT_TOKENS = {"session_offset_tokens": (9, 4), "session_mode_main_category_tokens": (17, 8)}
G13_FLOATS = ("current_matches_session_mode", "session_mode_share")
G13_MASK = "current_matches_session_mode_available"
FULL_K_COUNTS = {("L0S_CLEAN_V2_SESSION", "GRU"): 20_965_706, ("L0S_CLEAN_V2_SESSION", "SASREC"): 83_097_548,
                 ("L3S_NEAR_DUPLICATE_SESSION", "GRU"): 20_963_210, ("L3S_NEAR_DUPLICATE_SESSION", "SASREC"): 83_093_516}


def batch_s(n=7, seed=1, K=K_SMALL):
    return synthetic_examples_v2(n, K, SCHEMA.widths, torch.Generator().manual_seed(seed), with_session_intent=True,
                                 with_names=True)


def emb(module, ch):
    if hasattr(module, "side_embeds"):
        return module.side_embeds[ch] if module.side_embeds is not None and ch in module.side_embeds else None
    return getattr(module, {"session_offset_tokens": "session_offset_embed",
                            "session_mode_main_category_tokens": "session_mode_embed"}[ch], None)


@pytest.mark.parametrize("family", FAMS)
def test_extension_embeddings_shape_and_pad_row(family):
    bm = build(family, L3S, SEED, "cpu", K=K_SMALL)
    m = bm.module
    for ch, (vocab, dim) in EXT_TOKENS.items():
        e = emb(m, ch)
        assert e is not None and tuple(e.weight.shape) == (vocab, dim) and e.padding_idx == 0, ch
        assert bool((e.weight[0] == 0).all()) and bool((e.weight[1:] != 0).any()), ch
    m.train()
    b = batch_s()
    torch.manual_seed(3)
    y = bm.adapter.scores(b)
    y.float().pow(2).mean().backward()
    for ch in EXT_TOKENS:
        g = emb(m, ch).weight.grad
        assert g is not None and bool((g[0] == 0).all()) and bool((g[1:] != 0).any()), ch
    for s0 in (L0, L3):
        mm = build(family, s0, SEED, "cpu", K=K_SMALL).module
        assert all(emb(mm, ch) is None for ch in EXT_TOKENS)
        assert not any("session" in k for k in mm.state_dict())


@pytest.mark.parametrize("family,impl", [("GRU", "unpadded"), ("SASREC", "unpadded"), ("SASREC", "padded_reference")])
def test_extension_tokens_pad_invariance_and_real_positions(family, impl):
    bm = build(family, L3S, SEED, "cpu", K=K_SMALL, impl=impl)
    bm.module.eval()
    b = batch_s(n=9, seed=4)
    am = b["attention_mask"].bool()
    with torch.no_grad():
        y = bm.adapter.scores(b)
        for ch, (vocab, _) in EXT_TOKENS.items():
            pad = b[ch].clone()
            pad[~am] = vocab - 1                                           # PAD cells: any value
            assert torch.equal(bm.adapter.scores(dict(b, **{ch: pad})), y), f"{ch} on PAD changed the output"
            real = b[ch].clone()
            last = b["lengths"] - 1
            real[torch.arange(real.shape[0]), last] = torch.where(real[torch.arange(real.shape[0]), last] == 2, 3, 2)
            assert not torch.equal(bm.adapter.scores(dict(b, **{ch: real})), y), f"{ch} on real events is ignored"
    for k in list(EXT_TOKENS) + list(G13_FLOATS) + [G13_MASK]:
        bad = {x: v for x, v in b.items() if x != k}
        assert_raises(Exception, bm.adapter.scores, bad)


def test_g13_values_and_mask_are_appended_columns():
    b = batch_s(n=5, seed=2)
    b["current_matches_session_mode"] = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0])
    b["session_mode_share"] = torch.tensor([0.5, 0.25, 1.0, 0.0, 0.75])
    b[G13_MASK] = torch.tensor([1.0, 1.0, 1.0, 0.0, 1.0])
    p = L3S.projection.apply(b)
    base_uc = [SCHEMA.widths.user_context_names.index(n) for n in L3.widths.user_context_names]
    assert p["user_context"].shape == (5, 10) and p["user_context_masks"].shape == (5, 5)
    assert torch.equal(p["user_context"][:, :8], b["user_context"][:, base_uc])
    assert torch.equal(p["user_context"][:, 8], b["current_matches_session_mode"])
    assert torch.equal(p["user_context"][:, 9], b["session_mode_share"])
    assert torch.equal(p["user_context_masks"][:, :4], b["user_context_masks"])
    assert torch.equal(p["user_context_masks"][:, 4], b[G13_MASK])
    assert p["user_context_names"][-2:] == list(G13_FLOATS) and p["user_context_mask_names"][-1] == G13_MASK
    pl = L3.projection.apply(b)                                            # L3 drops every extension key
    assert not any(k in pl for k in list(EXT_TOKENS) + list(G13_FLOATS) + [G13_MASK])
    assert L0.projection.apply(b) is b                                     # L0: the batch object itself
    bad = dict(b, session_mode_share=torch.zeros(4))
    assert_raises(Exception, L3S.projection.apply, bad)                    # not [B]


@pytest.mark.parametrize("family", FAMS)
def test_l0_l3_outputs_ignore_the_new_keys(family):
    b = batch_s(n=6, seed=5)
    plain = {k: v for k, v in b.items() if k not in list(EXT_TOKENS) + list(G13_FLOATS) + [G13_MASK]}
    for s0 in (SCHEMA, L0, L3):
        bm = build(family, s0, SEED, "cpu", K=K_SMALL)
        bm.module.eval()
        with torch.no_grad():
            assert torch.equal(bm.adapter.scores(b), bm.adapter.scores(plain))
            assert torch.equal(module_forward(bm.adapter, b), module_forward(bm.adapter, plain))


@pytest.mark.parametrize("sid", ["L0S_CLEAN_V2_SESSION", "L3S_NEAR_DUPLICATE_SESSION"])
def test_session_schema_parameter_counts_full_k(sid):
    torch.set_num_threads(4)
    ref = {"L0S_CLEAN_V2_SESSION": L0, "L3S_NEAR_DUPLICATE_SESSION": L3}[sid]
    ds = S.load_registered(sid)
    for fam in FAMS:
        a = build(fam, ref, SEED, "cpu", K=158_486)
        b = build(fam, ds, SEED, "cpu", K=158_486)
        try:
            na, nb = a.manifest.shared_numel, b.manifest.shared_numel
            assert nb == FULL_K_COUNTS[(sid, fam)], (sid, fam, nb)
            emb_ = 9 * 4 + 17 * 8
            first = 12 * (128 if fam == "GRU" else 256) + 3 * 64
            assert nb - na == emb_ + first == {"GRU": 1_900, "SASREC": 3_436}[fam]
            m = b.module
            if fam == "GRU":
                assert m.input_proj.in_features == a.module.input_proj.in_features + 12
                assert m.user_ctx_proj[0].in_features == a.module.user_ctx_proj[0].in_features + 3
            else:
                assert m.side_proj.in_features == a.module.side_proj.in_features + 12
                assert m.user_mlp.in_features == a.module.user_mlp.in_features + 3
        finally:
            del a, b
            gc.collect()


@nc("a projection that appends the G13 values but forgets their availability mask")
def test_nc_projection_without_the_g13_mask():
    import dataclasses
    mutant = dataclasses.replace(L3S.projection, ext_columns={"user_context": G13_FLOATS, "user_context_masks": ()})
    b = batch_s(n=5, seed=2)
    p = mutant.apply(b)
    assert p["user_context_masks"].shape[-1] == 5 and torch.equal(p["user_context_masks"][:, -1], b[G13_MASK]),         "the G13 mask column is missing"
