"""Rich variant (03 sec.5, 05 item 24, Fable #7): zero-gate equivalence,
finite/nonzero active-branch gradients, side-tensor sensitivity, masked-field
invariance, and bit-identical common-parameter init vs the ID arm.
"""
from __future__ import annotations

import numpy as np
import torch

from conftest import make_batch, make_config, make_model, forward_kwargs, TINY_K, SIDE_VOCAB
from mutants import model_mutants as mm
from sasrec_model import SIDE_CHANNELS


COMMON_PARAM_PREFIXES = ("item_embed", "pos_embed", "blocks", "final_norm", "output_bias",
                         "input_norm", "output_embed")


def _common_named_params(model):
    return {name: p for name, p in model.named_parameters()
            if name.split(".")[0] in COMMON_PARAM_PREFIXES}


# ---------------------------------------------------------------------------
# Common-parameter init is bit-identical between ID and rich, same seed.
# ---------------------------------------------------------------------------
def test_common_param_init_bit_identical_between_id_and_rich_same_seed():
    id_cfg = make_config(group="item_only", d_model=16, blocks=2, heads=2, dropout=0.0)
    rich_cfg = make_config(group="rich_frozen_channels", d_model=16, blocks=2, heads=2, dropout=0.0)
    id_model = make_model(id_cfg, K=TINY_K, seed=555)
    rich_model = make_model(rich_cfg, K=TINY_K, seed=555)

    id_common = _common_named_params(id_model)
    rich_common = _common_named_params(rich_model)
    assert set(id_common) == set(rich_common)
    for name in id_common:
        assert torch.equal(id_common[name], rich_common[name]), f"{name}: common init differs (must be bit-identical)"

    # Negative control: a DIFFERENT seed must NOT be bit-identical (sanity
    # that the check isn't trivially true regardless of seed).
    rich_model_diff_seed = make_model(rich_cfg, K=TINY_K, seed=556)
    rich_common_diff = _common_named_params(rich_model_diff_seed)
    assert not torch.equal(id_common["item_embed.weight"], rich_common_diff["item_embed.weight"])


# ---------------------------------------------------------------------------
# side_enabled=False (+ identical common weights, guaranteed by same-seed
# construction above) recovers the ID network exactly.
# ---------------------------------------------------------------------------
def test_side_disabled_rich_model_equals_id_model():
    d, blocks, heads = 16, 2, 2
    id_cfg = make_config(group="item_only", d_model=d, blocks=blocks, heads=heads, dropout=0.0, max_len=12)
    rich_cfg = make_config(group="rich_frozen_channels", d_model=d, blocks=blocks, heads=heads, dropout=0.0, max_len=12)
    seed = 71
    id_model = make_model(id_cfg, K=TINY_K, seed=seed).eval()
    rich_model = make_model(rich_cfg, K=TINY_K, seed=seed).eval()

    lengths = torch.tensor([1, 5, 12, 8])
    batch_id = make_batch(lengths, T=12, K=TINY_K, rich=False, seed=3)
    batch_rich = make_batch(lengths, T=12, K=TINY_K, rich=True, seed=3)
    # item_tokens/attention_mask/position_ids must match between the two batches.
    batch_rich["item_tokens"] = batch_id["item_tokens"].clone()
    batch_rich["attention_mask"] = batch_id["attention_mask"].clone()
    batch_rich["position_ids"] = batch_id["position_ids"].clone()

    id_logits = id_model(**forward_kwargs(batch_id))

    rich_model.side_enabled = False
    rich_logits_gate_off = rich_model(**forward_kwargs(batch_rich))
    assert torch.allclose(id_logits, rich_logits_gate_off, atol=1e-5, rtol=1e-5), \
        "side_enabled=False + identical common weights must recover the ID network"

    # Negative control: leave side_enabled=True (default) with the SAME side
    # inputs -- the gate at init (sigmoid(-3) != 0) means the rich output
    # must NOT trivially equal the ID output; otherwise the equivalence test
    # above would be passing for a dead/ignored-input reason, not because
    # side_enabled actually gates the residual.
    rich_model.side_enabled = True
    rich_logits_gate_on = rich_model(**forward_kwargs(batch_rich))
    assert not torch.allclose(id_logits, rich_logits_gate_on, atol=1e-5, rtol=1e-5), \
        "negative control failed: with side_enabled=True the rich output should differ from the ID output"


# ---------------------------------------------------------------------------
# Each active side/user branch has finite, nonzero gradients.
# ---------------------------------------------------------------------------
def test_active_side_and_user_branches_have_finite_nonzero_gradients():
    cfg = make_config(group="rich_frozen_channels", d_model=16, blocks=2, heads=2, dropout=0.0, max_len=10)
    model = make_model(cfg, K=TINY_K, seed=8)
    lengths = torch.tensor([3, 10])
    batch = make_batch(lengths, T=10, K=TINY_K, rich=True, seed=4)

    logits = model(**forward_kwargs(batch))
    model.zero_grad()
    logits.sum().backward()

    checked = ["side_proj.weight", "side_gate", "user_proj.weight", "user_gate", "user_mlp.weight",
               "numflag_proj.weight"] + [f"side_embeds.{ch}.weight" for ch in SIDE_CHANNELS]
    for name, p in model.named_parameters():
        if name in checked:
            assert p.grad is not None, f"{name}: no gradient at all"
            assert torch.isfinite(p.grad).all(), f"{name}: non-finite gradient"
            assert p.grad.abs().sum() > 0, f"{name}: all-zero gradient (branch looks inactive)"


# ---------------------------------------------------------------------------
# Independently permuting a side tensor changes the output.
# ---------------------------------------------------------------------------
def test_permuting_a_side_tensor_changes_output():
    cfg = make_config(group="rich_frozen_channels", d_model=16, blocks=2, heads=2, dropout=0.0, max_len=10)
    model = make_model(cfg, K=TINY_K, seed=12).eval()
    lengths = torch.tensor([4, 10])
    batch = make_batch(lengths, T=10, K=TINY_K, rich=True, seed=6)
    baseline = model(**forward_kwargs(batch))

    mutated = dict(batch)
    rng = np.random.default_rng(50)
    brand = batch["brand_tokens"].clone()
    for b in range(brand.shape[0]):
        L = int(lengths[b])
        brand[b, :L] = torch.from_numpy(rng.integers(1, SIDE_VOCAB["brand_tokens"], size=L))
    mutated["brand_tokens"] = brand
    out_mutated = model(**forward_kwargs(mutated))
    assert not torch.allclose(baseline, out_mutated, atol=1e-6), \
        "independently permuting brand_tokens (real positions) must change the output"


# ---------------------------------------------------------------------------
# Masked-field value change (PAD positions) -> no output change.
# ---------------------------------------------------------------------------
def test_pad_position_side_field_change_does_not_affect_output():
    cfg = make_config(group="rich_frozen_channels", d_model=16, blocks=2, heads=2, dropout=0.0, max_len=10)
    model = make_model(cfg, K=TINY_K, seed=15).eval()
    lengths = torch.tensor([1, 5, 10])
    batch_a = make_batch(lengths, T=10, K=TINY_K, rich=True, seed=9, pad_side_value=0)
    batch_b = make_batch(lengths, T=10, K=TINY_K, rich=True, seed=9, pad_side_value=0)
    batch_b["item_tokens"] = batch_a["item_tokens"].clone()
    batch_b["attention_mask"] = batch_a["attention_mask"].clone()
    batch_b["position_ids"] = batch_a["position_ids"].clone()

    rng = np.random.default_rng(77)
    for ch in SIDE_CHANNELS:
        vocab = SIDE_VOCAB[ch]
        tok = batch_a[ch].clone()
        for b in range(tok.shape[0]):
            L = int(lengths[b])
            if L < 10:
                tok[b, L:] = torch.from_numpy(rng.integers(1, vocab, size=10 - L))
        batch_b[ch] = tok
    numf = batch_a["event_numeric_features"].clone()
    flags = batch_a["event_quality_flags"].clone()
    for b in range(numf.shape[0]):
        L = int(lengths[b])
        if L < 10:
            numf[b, L:] = torch.from_numpy(rng.normal(size=(10 - L, 19)).astype(np.float32))
            flags[b, L:] = torch.from_numpy(rng.integers(0, 2, size=(10 - L, 15)).astype(np.int8))
    batch_b["event_numeric_features"] = numf
    batch_b["event_quality_flags"] = flags

    out_a = model(**forward_kwargs(batch_a))
    out_b = model(**forward_kwargs(batch_b))
    assert torch.allclose(out_a, out_b, atol=1e-6), \
        "changing side/numeric/flag VALUES only at PAD positions must not affect the output"

    # Negative control: force every position 'valid' -> PAD-side leakage appears.
    with mm.force_all_positions_valid(model):
        seq_a = model.hidden_states(**forward_kwargs(batch_a))
        seq_b = model.hidden_states(**forward_kwargs(batch_b))
    assert not torch.allclose(seq_a, seq_b, atol=1e-6), \
        "negative control failed: force_all_positions_valid should let masked side fields leak"
