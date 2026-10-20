"""SASRec causal attention (rich model, both implementations).

  * perturbing every channel (item + 5 side tokens + numeric + flags) of the events AFTER position p leaves the hidden
    states at positions <= p bitwise unchanged; a positive control (perturbing an earlier event) changes them;
  * the query of the window truncated after p equals the readout of the full window's hidden state at p (the query
    never looks past its own last event);
  * gradient leakage: d h_p / d(input at positions > p) is exactly 0 for the item-table rows of the future items and
    for the future numeric inputs, and non-zero for every position <= p; in eval and in train mode (dropout on).
Negative controls: with the causal mask disabled, the future perturbation and the gradient test both fail.
"""
from __future__ import annotations

import pytest
import torch
from seqrec_testkit import batch, kwargs_for, maxdiff, model, nc, seeded

IMPLS = ["unpadded", "padded_reference"]
P = 5
VOC = {"item_tokens": 67, "category_tokens": 663, "brand_tokens": 3956, "main_category_tokens": 16,
       "daypart_tokens": 8, "weekday_tokens": 10}


def _model(impl: str, broken_causal: bool = False):
    m = model("SASREC", 7, impl=impl).eval()
    if broken_causal:
        m.causal.fill_(True)                     # MUTANT: every query may attend every key
    return m


def _perturb(b: dict, positions: torch.Tensor, gen: torch.Generator) -> dict:
    """Replace every channel at the [B, L] cells in `positions` (a bool mask of real cells)."""
    out = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}
    for ch, hi in VOC.items():
        out[ch] = torch.where(positions, torch.randint(3, hi, positions.shape, generator=gen), out[ch])
    num = out["event_numeric_features"]
    out["event_numeric_features"] = torch.where(positions[..., None], torch.randn(num.shape, generator=gen) * 3, num)
    flg = out["event_quality_flags"]
    out["event_quality_flags"] = torch.where(positions[..., None], (1 - flg).to(torch.int8), flg)
    return out


def _hidden(m, b):
    with torch.no_grad():
        return m.hidden_states(**kwargs_for(m, b))


def _future_check(m, seed: int = 0):
    b = batch(4, seed=seed, min_len=9, max_len=14)
    am = b["attention_mask"]
    fut = am.clone()
    fut[:, :P + 1] = False
    h0 = _hidden(m, b)
    h1 = _hidden(m, _perturb(b, fut, torch.Generator().manual_seed(seed + 1)))
    assert torch.equal(h0[:, :P + 1], h1[:, :P + 1]), \
        f"a future event changed an earlier hidden state (max |d| {maxdiff(h0[:, :P + 1], h1[:, :P + 1]):.3g})"
    return b, h0, h1


@pytest.mark.parametrize("impl", IMPLS)
def test_future_perturbation_leaves_prefix_unchanged(impl):
    m = _model(impl)
    b, h0, h1 = _future_check(m)
    assert not torch.equal(h0[:, P + 1:], h1[:, P + 1:]), "sanity: the perturbation reached the future positions"
    past = b["attention_mask"].clone()
    past[:, 1:] = False                                                   # positive control: perturb event 0
    h2 = _hidden(m, _perturb(b, past, torch.Generator().manual_seed(9)))
    assert (h0[:, P] - h2[:, P]).abs().amax(dim=-1).min() > 1e-4, "an earlier event must influence position p"


@pytest.mark.parametrize("impl", IMPLS)
def test_query_of_prefix_equals_readout_at_p(impl):
    m = _model(impl)
    b = batch(4, seed=3, min_len=9, max_len=14)
    h = _hidden(m, b)
    side = {k: v for k, v in kwargs_for(m, b).items()
            if k not in ("item_tokens", "lengths", "attention_mask", "position_ids")}
    keep = torch.arange(b["attention_mask"].shape[1])[None, :] < P + 1
    pre = {}
    for k, v in b.items():
        if torch.is_tensor(v) and v.ndim >= 2 and k not in ("user_context", "user_context_masks"):
            mask = keep if v.ndim == 2 else keep[..., None]
            pre[k] = (v * mask).to(v.dtype)
        else:
            pre[k] = v
    pre["lengths"] = torch.full_like(b["lengths"], P + 1)
    pre["attention_mask"] = b["attention_mask"] & keep
    with torch.no_grad():
        q_pre = m.query(**kwargs_for(m, pre))
        q_ref = m._readout(h[:, P], side)
    assert torch.allclose(q_pre, q_ref, atol=2e-6, rtol=0), f"max |d| {maxdiff(q_pre, q_ref):.3g}"


def _grad_check(m, train: bool):
    b = batch(1, seed=5, min_len=10, max_len=10)
    ids = torch.randperm(64, generator=torch.Generator().manual_seed(1))[:10] + 3
    b["item_tokens"][0, :10] = ids                                         # distinct rows per position
    num = b["event_numeric_features"].clone().requires_grad_(True)
    kw = kwargs_for(m, b)
    kw["event_numeric_features"] = num
    m.train(train)
    m.zero_grad(set_to_none=True)
    with seeded(11):
        h = m.hidden_states(**kw)
        h[:, P].sum().backward()
    g = m.item_embed.weight.grad
    assert torch.count_nonzero(g[ids[P + 1:]]) == 0, "future item rows received gradient (leakage)"
    assert float(num.grad[0, P + 1:].abs().sum()) == 0.0, "future numeric inputs received gradient (leakage)"
    assert bool((g[ids[:P + 1]].abs().sum(dim=1) > 0).all()), "sanity: every past item row gets gradient"
    assert bool((num.grad[0, :P + 1].abs().sum(dim=1) > 0).all()), "sanity: every past numeric input gets gradient"


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("train", [False, True], ids=["eval", "train"])
def test_gradient_leakage(impl, train):
    _grad_check(_model(impl), train)


# ------------------------------------------------------------------------------------------------ negative controls
@nc("causal mask disabled: a future event changes an earlier hidden state")
@pytest.mark.parametrize("impl", IMPLS)
def test_nc_future_perturbation_without_causal_mask(impl):
    _future_check(_model(impl, broken_causal=True))


@nc("causal mask disabled: gradient leaks from h_p into future inputs")
def test_nc_gradient_leakage_without_causal_mask():
    _grad_check(_model("unpadded", broken_causal=True), train=False)
