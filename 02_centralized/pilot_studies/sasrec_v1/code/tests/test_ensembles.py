"""Unit tests for the pre-registered separate systems (ensembles.py) and the v3b task sharding."""
from __future__ import annotations

import torch

from conftest import make_batch, make_config, make_model, forward_kwargs, TINY_K
import ensembles


def _rich_kwargs(batch):
    return forward_kwargs(batch, with_position_ids=False)


def test_ensemble_of_identical_members_ranks_like_one_member():
    cfg = make_config(group="rich_frozen_channels", d_model=16, blocks=1, heads=2, dropout=0.0)
    m1 = make_model(cfg, K=TINY_K, seed=3).eval()
    m2 = make_model(cfg, K=TINY_K, seed=3).eval()
    ens = ensembles.LogProbEnsemble([m1, m2]).eval()
    batch = make_batch(torch.tensor([3, 1, 5]), T=5, K=TINY_K, rich=True, seed=1)
    kw = _rich_kwargs(batch)
    with torch.no_grad():
        lp_one = torch.log_softmax(m1(**kw), dim=1)
        lp_ens = ens(**kw)
    assert torch.allclose(lp_one, lp_ens, atol=1e-6)
    # negative control: two DIFFERENT members change the scores
    m3 = make_model(cfg, K=TINY_K, seed=4).eval()
    with torch.no_grad():
        other = ensembles.LogProbEnsemble([m1, m3])(**kw)
    assert not torch.allclose(lp_one, other, atol=1e-6)


def test_item_only_member_never_receives_side_channels():
    id_cfg = make_config(group="item_only", d_model=16, blocks=1, heads=2, dropout=0.0)
    rich_cfg = make_config(group="rich_frozen_channels", d_model=16, blocks=1, heads=2, dropout=0.0)
    a0, a1 = make_model(id_cfg, K=TINY_K, seed=1).eval(), make_model(rich_cfg, K=TINY_K, seed=1).eval()
    ens = ensembles.LogProbEnsemble([a0, a1]).eval()
    assert ens.use_user_context and ens.numeric_dim > 0 and "category_tokens" in ens._categorical_channels
    batch = make_batch(torch.tensor([2, 4]), T=4, K=TINY_K, rich=True, seed=2)
    kw = _rich_kwargs(batch)
    with torch.no_grad():
        out = ens(**kw)                      # a0 would raise if it received side channels
        expect = (torch.log_softmax(a0(item_tokens=kw["item_tokens"], lengths=kw["lengths"],
                                       attention_mask=kw["attention_mask"]), 1)
                  + torch.log_softmax(a1(**kw), 1)) / 2
    assert torch.allclose(out, expect, atol=1e-6)


def test_average_state_dicts_identity_and_mean(tmp_path):
    cfg = make_config(group="item_only", d_model=16, blocks=1, heads=2, dropout=0.0)
    a, b = make_model(cfg, K=TINY_K, seed=5), make_model(cfg, K=TINY_K, seed=6)
    pa, pb = tmp_path / "a.pt", tmp_path / "b.pt"
    torch.save({"state_dict": a.state_dict()}, pa)
    torch.save({"state_dict": b.state_dict()}, pb)
    same = ensembles.average_state_dicts([pa, pa])
    assert all(torch.equal(same[k], a.state_dict()[k]) for k in same)
    avg = ensembles.average_state_dicts([pa, pb])
    k = "output_embed"
    assert torch.allclose(avg[k], (a.state_dict()[k] + b.state_dict()[k]) / 2, atol=1e-7)
    m = make_model(cfg, K=TINY_K, seed=0)
    m.load_state_dict(avg)                   # loads cleanly into the architecture


def test_v3b_task_sharding_is_disjoint_and_complete():
    import lock_retro_v3b as lr
    entries = {n: {"kind": "single"} for n in ("A0_ID256_s2026", "A1_RICH256_s2026", "A1_RICH256_s2027", "S4a", "S4b")}
    entries.update({"S1": {"kind": "ensemble_logprob_mean", "members": [1, 2]},
                    "S2": {"kind": "ensemble_logprob_mean", "members": [1, 2]},
                    "S3": {"kind": "ensemble_logprob_mean", "members": [1, 2, 3]}})
    t = lr.tasks({"entries": entries})
    keys = [(sp, n) for sp, n, _ in t]
    assert len(keys) == len(set(keys)) == 2 * len(entries)
    assert {w for *_, w in t} == {0, 1}
