"""Parity with the byte snapshot (catches an accidental change during the copy).

With the widths set to the executed v1 values (19 / 15 / 12 / 6, schema family_sizing.SASRec.constants_v1):
  * the same seed builds a bitwise-identical state (parameter creation and RNG consumption order unchanged);
  * with the snapshot's weights loaded strictly, the V2 module reproduces the snapshot logits bitwise in eval mode, and
    in train mode (same dropout RNG) the logits AND every parameter gradient are bitwise equal;
  * at the Clean V2 widths the state differs from the v1-width state ONLY in the two width-dependent matrices.
Proves: the copy changed nothing but the widths. Does not prove: anything about training quality at V2 widths.
"""
from __future__ import annotations

import types

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from fl.numerics import states_equal
from models_testkit import FAMS, K_SMALL, V1, V2, batch, forward, maxdiff, nc, seeded, snap_model, v2_model
from v4models import gru_v2, sasrec_v2

SEED = 2026
CASES = [("GRU", "unpadded"), ("SASREC", "unpadded"), ("SASREC", "padded_reference")]


def _pair(family: str, impl: str = "unpadded"):
    ref = snap_model(family, SEED, impl=impl)
    new = v2_model(family, SEED + 17, widths=V1, impl=impl)     # another seed: its weights must come from the load
    new.load_state_dict(ref.state_dict(), strict=True)
    return ref, new


def _assert_eval_parity(ref, new, b) -> None:
    ref.eval()
    new.eval()
    with torch.no_grad():
        ya, yb = forward(ref, b), forward(new, b)
    assert ya.shape == yb.shape == (b["item_tokens"].shape[0], K_SMALL)
    assert torch.equal(ya, yb), f"V2 (v1 widths) logits differ from the snapshot: max |d| = {maxdiff(ya, yb):.3g}"


@pytest.mark.parametrize("family", FAMS)
def test_parity_same_seed_builds_identical_state(family):
    a, b = snap_model(family, SEED), v2_model(family, SEED, widths=V1)
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb), "state_dict keys / order changed"
    assert states_equal(sa, sb), "same seed must give the snapshot's init bitwise (creation / RNG order unchanged)"
    assert [n for n, _ in a.named_buffers()] == [n for n, _ in b.named_buffers()]


@pytest.mark.parametrize("family,impl", CASES)
def test_parity_logits_eval_bitwise(family, impl):
    ref, new = _pair(family, impl)
    _assert_eval_parity(ref, new, batch(9, widths=V1, seed=3))
    _assert_eval_parity(ref, new, batch(5, widths=V1, seed=5, max_len=1))       # single-event windows


@pytest.mark.parametrize("family,impl", CASES)
def test_parity_train_mode_logits_and_gradients_bitwise(family, impl):
    ref, new = _pair(family, impl)
    b = batch(9, widths=V1, seed=4)
    outs = []
    for m in (ref, new):
        m.train()
        m.zero_grad(set_to_none=True)
        with seeded(99):
            y = forward(m, b)
            loss = F.cross_entropy(y, b["target_class"])
            loss.backward()
        outs.append((y.detach(), {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}))
    (ya, ga), (yb, gb) = outs
    assert torch.equal(ya, yb), f"train-mode logits differ: {maxdiff(ya, yb):.3g}"
    assert ga.keys() == gb.keys() and len(ga) > 10
    bad = [k for k in ga if not torch.equal(ga[k], gb[k])]
    assert not bad, f"gradients differ for {bad}"


@pytest.mark.parametrize("family", FAMS)
def test_v2_widths_change_only_the_two_width_matrices(family):
    a, b = v2_model(family, SEED, widths=V1), v2_model(family, SEED, widths=V2)
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb)
    changed = {k: (tuple(sa[k].shape), tuple(sb[k].shape)) for k in sa if sa[k].shape != sb[k].shape}
    expect = {"GRU": {"numeric_flags_proj.0.weight": ((32, 34), (32, 30)),
                      "user_ctx_proj.0.weight": ((64, 18), (64, 16))},
              "SASREC": {"numflag_proj.weight": ((32, 34), (32, 30)),
                         "user_mlp.weight": ((64, 18), (64, 16))}}[family]
    assert changed == expect


def test_v2_layer_sizes_match_the_contract():
    """DATA_CONTRACT_CLEAN_V2 section 7 sizing, read off the built V2 modules."""
    g = v2_model("GRU", SEED)
    assert isinstance(g, gru_v2.ContextGRU)
    assert (g.numeric_dim, g.flags_dim, g.user_context_dim, g.user_mask_dim) == (15, 15, 12, 4)
    assert (g.numeric_flags_proj[0].in_features, g.numeric_flags_proj[0].out_features) == (30, 32)
    assert isinstance(g.numeric_flags_proj[1], nn.GELU)
    assert g.concat_width == 240 and (g.input_proj.in_features, g.input_proj.out_features) == (240, 128)
    assert (g.gru.input_size, g.gru.hidden_size, g.gru.num_layers) == (128, 256, 1)
    assert (g.user_ctx_proj[0].in_features, g.user_ctx_proj[0].out_features) == (16, 64)
    assert (g.readout_proj.in_features, g.readout_proj.out_features) == (320, 128)
    assert g.item_embed.embedding_dim == 128
    s = v2_model("SASREC", SEED)
    assert isinstance(s, sasrec_v2.SASRecCE)
    assert (s.numeric_dim, s.flags_dim, s.user_context_dim, s.user_mask_dim) == (15, 15, 12, 4)
    assert (s.d, len(s.blocks), s.blocks[0].h, s.blocks[0].ff1.out_features) == (256, 2, 4, 1024)
    assert (s.numflag_proj.in_features, s.numflag_proj.out_features) == (30, 32)
    assert (s.side_proj.in_features, s.side_proj.out_features) == (112, 256)
    assert (s.user_mlp.in_features, s.user_mlp.out_features) == (16, 64)
    assert (s.user_proj.in_features, s.user_proj.out_features) == (64, 256)
    assert s.cfg.dropout == 0.2 and s.output_embed.shape == (K_SMALL, 256)


# ------------------------------------------------------------------------------------------------ negative controls
@nc("copy error: GRU numeric+flag activation GELU -> ReLU (same weights)")
def test_nc_parity_gru_activation_mutant():
    ref, new = _pair("GRU")
    new.numeric_flags_proj[1] = nn.ReLU()
    _assert_eval_parity(ref, new, batch(9, widths=V1, seed=3))


@nc("copy error: SASRec side residual concatenates flags before numerics (same weights)")
def test_nc_parity_sasrec_concat_order_mutant():
    ref, new = _pair("SASREC")

    def wrong(self, tok_sel, num_sel, flag_sel):
        parts = [self.side_embeds[ch](tok_sel[ch]) for ch in sasrec_v2.SIDE_CHANNELS]
        nf = torch.cat([flag_sel.to(torch.float32), num_sel.to(torch.float32)], dim=-1)
        parts.append(F.gelu(self.numflag_proj(nf)))
        return torch.sigmoid(self.side_gate) * self.side_norm(self.side_proj(torch.cat(parts, dim=-1)))

    new._side_residual = types.MethodType(wrong, new)
    _assert_eval_parity(ref, new, batch(9, widths=V1, seed=3))


@nc("a V2 build whose init consumes the RNG in another order (different seed) is not the snapshot init")
def test_nc_parity_init_order_mutant():
    a, b = snap_model("GRU", SEED), v2_model("GRU", SEED + 1, widths=V1)
    assert states_equal(a.state_dict(), b.state_dict())
