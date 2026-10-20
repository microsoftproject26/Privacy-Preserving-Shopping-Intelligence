"""Head / objective / optimisation (05 sec.C items 17-20; Fable #3 tying,
#6 objective). Every mandatory check has a negative control.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import paths  # noqa: F401
import train as reuse_train  # audited/frozen helper: build_param_groups

from conftest import make_batch, make_config, make_model, forward_kwargs, TINY_K
from mutants import model_mutants as mm
from sasrec_objective import ce_sum, ce_sum_reference


# ---------------------------------------------------------------------------
# Registry v3 (Fable deep review FIX 1): the output table is a SEPARATE Parameter
# (untied from the input item table), appearing exactly once in the optimizer groups.
# ---------------------------------------------------------------------------
def _is_untied(model) -> bool:
    ow = model.output_weight()
    with torch.no_grad():
        probe = ow[2, 0].item()
        model.item_embed.weight[5, 0] += 1.2345
        changed = abs(model.output_weight()[2, 0].item() - probe) > 1e-9
        model.item_embed.weight[5, 0] -= 1.2345
    return (ow is model.output_embed) and not changed and ow.data_ptr() != model.item_embed.weight.data_ptr() \
        and tuple(ow.shape) == (model.K, model.d)


def test_output_head_is_untied_separate_parameter_once_in_optimizer_groups():
    cfg = make_config(group="item_only", d_model=12, blocks=1, heads=2, dropout=0.0)
    model = make_model(cfg, K=TINY_K, seed=4)
    assert _is_untied(model), "v3 head must be the separate output_embed Parameter, not a view of item_embed"
    groups = reuse_train.build_param_groups(model)
    all_params = [p for g in groups for p in g["params"]]
    assert sum(1 for p in all_params if p is model.output_embed) == 1
    assert sum(1 for p in all_params if p is model.item_embed.weight) == 1
    ids = [id(p) for p in all_params]
    assert len(ids) == len(set(ids)), "optimizer groups must not contain a duplicated Parameter"
    # the head really uses output_embed: a gradient reaches it through the logits
    lengths = torch.tensor([3, 1])
    batch = make_batch(lengths, T=3, K=TINY_K, seed=2)
    logits = model(**forward_kwargs(batch))
    logits[:, 0].sum().backward()
    assert model.output_embed.grad is not None and torch.count_nonzero(model.output_embed.grad[0]) > 0
    # Negative control: a re-tied head (the v2 bug) must be detected.
    with mm.retie_head_weight(model):
        assert not _is_untied(model), "negative control failed: a re-tied head was not detected"


def test_recenter_output_is_loss_neutral_and_zeroes_common_mode():
    cfg = make_config(group="item_only", d_model=12, blocks=1, heads=2, dropout=0.0)
    model = make_model(cfg, K=TINY_K, seed=9).eval()
    with torch.no_grad():   # give the table/bias a large common mode first
        model.output_embed.add_(torch.randn(1, model.d) * 3.0)
        model.output_bias.add_(5.0)
    lengths = torch.tensor([4, 2, 1])
    batch = make_batch(lengths, T=4, K=TINY_K, seed=5)
    kw = forward_kwargs(batch)
    with torch.no_grad():
        lp_before = torch.log_softmax(model(**kw).double(), dim=1)
        model.recenter_output_()
        lp_after = torch.log_softmax(model(**kw).double(), dim=1)
    assert torch.allclose(lp_before, lp_after, atol=1e-5), "recentering must not change log-probabilities"
    assert float(model.output_embed.mean(0).norm()) < 1e-5 and abs(float(model.output_bias.mean())) < 1e-5
    # Negative control: a non-constant (per-class) shift is NOT loss-neutral.
    with mm.recenter_by_row_(model):
        with torch.no_grad():
            model.recenter_output_()
            lp_bad = torch.log_softmax(model(**kw).double(), dim=1)
    assert not torch.allclose(lp_after, lp_bad, atol=1e-5), "negative control failed: per-class shift went undetected"


def test_known_class_map_gate_rejects_noncontiguous_map():
    cfg = make_config(group="item_only", d_model=8, blocks=1, heads=2)
    vocab_sizes = {"item_tokens": 3 + TINY_K}
    good_poc = np.arange(TINY_K) + 3
    make_model(cfg, K=TINY_K, seed=1, vocab_sizes=vocab_sizes, poc=good_poc)  # must not raise

    bad_poc = good_poc.copy()
    bad_poc[-1] = bad_poc[-1] + 5  # break contiguity: leaves a gap
    with pytest.raises(ValueError):
        make_model(cfg, K=TINY_K, seed=1, vocab_sizes=vocab_sizes, poc=bad_poc)

    wrong_offset = np.arange(TINY_K) + 2  # contiguous but wrong starting offset (would collide with OOV=2)
    with pytest.raises(ValueError):
        make_model(cfg, K=TINY_K, seed=1, vocab_sizes=vocab_sizes, poc=wrong_offset)


# ---------------------------------------------------------------------------
# 100 synthetic AdamW updates: PAD row stays exactly zero.
# ---------------------------------------------------------------------------
def test_pad_row_stays_zero_after_100_updates():
    cfg = make_config(group="item_only", d_model=16, blocks=2, heads=2, dropout=0.0)
    model = make_model(cfg, K=TINY_K, seed=9)
    groups = reuse_train.build_param_groups(model)
    opt = torch.optim.AdamW(groups, lr=1e-3, betas=(0.9, 0.98), eps=1e-8)

    rng = np.random.default_rng(0)
    for _ in range(100):
        lengths = torch.from_numpy(rng.integers(1, 11, size=8).astype(np.int64))
        batch = make_batch(lengths, T=10, K=TINY_K, seed=int(rng.integers(0, 1_000_000)))
        opt.zero_grad()
        logits = model(**forward_kwargs(batch))
        loss_sum, n = ce_sum(logits, batch["target_class"], batch["loss_mask"])
        (loss_sum / n.clamp(min=1)).backward()
        opt.step()
    assert torch.count_nonzero(model.item_embed.weight[0]) == 0, "PAD row (index 0) must stay exactly zero"

    # Negative control: inject the bug this test is meant to catch.
    mm.corrupt_pad_row(model)
    assert torch.count_nonzero(model.item_embed.weight[0]) > 0, \
        "negative control failed: corrupt_pad_row should have produced a nonzero PAD row"


# ---------------------------------------------------------------------------
# 05 item 18/19: logits exactly [B,K], never [B,T,K].
# ---------------------------------------------------------------------------
def test_logits_are_BK_and_no_BTK_tensor_is_ever_created():
    cfg = make_config(group="item_only", d_model=16, blocks=2, heads=2, dropout=0.0, max_len=12)
    model = make_model(cfg, K=TINY_K, seed=14)
    lengths = torch.tensor([3, 12])
    batch = make_batch(lengths, T=12, K=TINY_K, seed=15)

    shapes = []
    hooks = []
    for name, mod in model.named_modules():
        def make_hook(nm):
            def hook(module, inp, out):
                outs = out if isinstance(out, (tuple, list)) else (out,)
                for o in outs:
                    if torch.is_tensor(o):
                        shapes.append((nm, tuple(o.shape)))
            return hook
        hooks.append(mod.register_forward_hook(make_hook(name)))
    try:
        logits = model(**forward_kwargs(batch))
    finally:
        for h in hooks:
            h.remove()

    assert logits.ndim == 2 and logits.shape == (2, model.K), f"expected [B,K]={(2, model.K)}, got {tuple(logits.shape)}"
    K = model.K
    offending = [(nm, shp) for nm, shp in shapes if len(shp) == 3 and shp[-1] == K]
    assert offending == [], f"found a [*, *, K] tensor during forward (candidate [B,T,K] leak): {offending}"


# ---------------------------------------------------------------------------
# 05 item 19/20, Fable #6: ce_sum vs ce_sum_reference, plus an independent
# (hand-written, NOT re-using ce_sum_reference) full-K smoothing formula.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("smoothing", [0.0, 0.05])
def test_ce_sum_matches_reference_and_independent_full_K_formula(smoothing):
    torch.manual_seed(0)
    K = 11
    logits = torch.randn(9, K) * 3.0
    target = torch.randint(0, K, (9,))
    loss_mask = torch.tensor([True, True, False, True, True, True, False, True, True])
    target = target.clone()
    target[2] = -1  # loss_mask False AND target<0 -- excluded either way

    loss_sum, n = ce_sum(logits, target, loss_mask, label_smoothing=smoothing)
    loss_sum_ref, n_ref = ce_sum_reference(logits, target, loss_mask, label_smoothing=smoothing)
    assert int(n) == int(n_ref)
    assert abs(float(loss_sum) - float(loss_sum_ref)) < 1e-4

    # Independent, from-scratch full-K smoothing formula (numpy, no torch
    # softmax/cross_entropy reuse at all): loss_i = (1-eps)*(-log p[y_i]) +
    # eps * mean_j(-log p[j]).
    sel = (loss_mask.numpy() & (target.numpy() >= 0))
    logits_np = logits.numpy().astype(np.float64)[sel]
    target_np = target.numpy()[sel]
    m = logits_np.max(axis=1, keepdims=True)
    logp = logits_np - m - np.log(np.exp(logits_np - m).sum(axis=1, keepdims=True))
    nll = -logp[np.arange(logp.shape[0]), target_np]
    uniform = -logp.mean(axis=1)
    manual = ((1.0 - smoothing) * nll + smoothing * uniform).sum()
    assert abs(float(loss_sum) - manual) < 1e-4, "hand-written full-K smoothing formula disagrees with ce_sum"

    # Negative control: label CORRUPTION (not the registered smoothing
    # formula) must NOT match the full-K formula except at eps=0.
    if smoothing > 0:
        corrupted_target = target.clone()
        valid_idx = (target >= 0).nonzero(as_tuple=True)[0]
        corrupted_target[valid_idx] = (target[valid_idx] + 1) % K
        loss_sum_corrupt, _ = ce_sum(logits, corrupted_target, loss_mask, label_smoothing=0.0)
        assert abs(float(loss_sum_corrupt) - float(loss_sum)) > 1e-3


# ---------------------------------------------------------------------------
# 05 item 20: sum/count across UNEVEN microbatches (2x128 vs 256) gives
# equal gradients (<=1e-6) before clipping, dropout off.
# ---------------------------------------------------------------------------
def test_uneven_microbatch_accumulation_matches_full_batch_gradient():
    cfg = make_config(group="item_only", d_model=16, blocks=2, heads=2, dropout=0.0, max_len=20)
    K = 23
    model_full = make_model(cfg, K=K, seed=77)
    model_split = make_model(cfg, K=K, seed=77)  # identical init

    rng = np.random.default_rng(3)
    B = 256
    lengths = rng.integers(1, 21, size=B).astype(np.int64)
    batch = make_batch(lengths, T=20, K=K, seed=5)
    # Make it genuinely UNEVEN: knock out loss_mask on an uneven split of rows.
    loss_mask = batch["loss_mask"].clone()
    loss_mask[rng.choice(B, size=17, replace=False)] = False
    batch["loss_mask"] = loss_mask

    kw_full = forward_kwargs(batch)
    logits_full = model_full(**kw_full)
    loss_sum_full, n_full = ce_sum(logits_full, batch["target_class"], batch["loss_mask"])
    model_full.zero_grad()
    (loss_sum_full / n_full).backward()
    full_grads = {name: p.grad.clone() for name, p in model_full.named_parameters() if p.grad is not None}

    split_a = slice(0, 128)
    split_b = slice(128, 256)
    sums, counts = [], []
    model_split.zero_grad()
    for sl in (split_a, split_b):
        sub_kw = {k: v[sl] for k, v in kw_full.items()}
        logits_sub = model_split(**sub_kw)
        loss_sum_sub, n_sub = ce_sum(logits_sub, batch["target_class"][sl], batch["loss_mask"][sl])
        sums.append(float(loss_sum_sub.detach()))
        counts.append(int(n_sub))
    total_n = sum(counts)
    assert total_n == int(n_full)
    # Re-run the two backward passes scaled by the POOLED count (this is the
    # gradient-accumulation pattern the trainer must use).
    for sl in (split_a, split_b):
        sub_kw = {k: v[sl] for k, v in kw_full.items()}
        logits_sub = model_split(**sub_kw)
        loss_sum_sub, n_sub = ce_sum(logits_sub, batch["target_class"][sl], batch["loss_mask"][sl])
        (loss_sum_sub / total_n).backward()
    split_grads = {name: p.grad.clone() for name, p in model_split.named_parameters() if p.grad is not None}

    assert set(full_grads) == set(split_grads)
    for name in full_grads:
        assert torch.allclose(full_grads[name], split_grads[name], atol=1e-6, rtol=1e-5), \
            f"{name}: accumulated-microbatch gradient disagrees with the full-batch gradient"

    # Negative control: the WRONG aggregation (mean-of-microbatch-means)
    # must disagree with the correct sum/count total whenever the two
    # microbatches contribute different counts.
    from mutants.objective_mutants import mean_of_microbatch_means
    assert counts[0] != counts[1], "test setup must be genuinely uneven"
    wrong_loss = mean_of_microbatch_means(sums, counts)
    correct_loss = sum(sums) / total_n
    assert abs(wrong_loss - correct_loss) > 1e-6


# ---------------------------------------------------------------------------
# init loss ~ ln K.
# ---------------------------------------------------------------------------
def test_init_loss_is_approximately_ln_K():
    K = 41
    cfg = make_config(group="item_only", d_model=32, blocks=2, heads=4, dropout=0.0, max_len=20)
    model = make_model(cfg, K=K, seed=100)
    model.eval()
    lengths = torch.from_numpy(np.random.default_rng(1).integers(1, 21, size=64).astype(np.int64))
    batch = make_batch(lengths, T=20, K=K, seed=2)
    with torch.no_grad():
        logits = model(**forward_kwargs(batch))
        loss_sum, n = ce_sum(logits, batch["target_class"], batch["loss_mask"])
        mean_loss = float(loss_sum / n)
    assert abs(mean_loss - math.log(K)) < 0.25, f"init loss {mean_loss:.4f} far from ln(K)={math.log(K):.4f}"
