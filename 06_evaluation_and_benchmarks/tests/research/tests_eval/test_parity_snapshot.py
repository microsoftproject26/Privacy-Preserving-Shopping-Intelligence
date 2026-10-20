"""Parity with the snapshot evaluator where the semantics agree, and the one prediction path for every method.

Proves: on finite scores the V4 ranks equal eval_full.rank_targets and evaluator._ranks_chunked exactly (ties
included, any class_chunk); the V4 E2E / RANKABLE micro and macro equal the snapshot aggregate()'s end_to_end /
closed blocks within 1e-6 (the snapshot accumulates in float32) with equal user counts, for every k in {1, 5, 10, 20};
predict_adapter on the FL synthetic family (tied GRU-like and untied SASRec-like heads, with and without a PF p_u)
equals the snapshot ranks of adapter.scores() exactly when one score block covers K, stays within float tolerance for
smaller blocks, restores train mode and every RNG, and scores only RANKABLE rows.
Does not prove: parity where the semantics deliberately differ (NaN, empty populations, the client_idx warm cut —
see the negative controls in the other files).
"""
from __future__ import annotations

import random

import numpy as np
import pytest
import torch

import eval_full
import evaluator as legacy_evaluator
from evaltestkit import artifact_from_scores, frozen, manifest_from_rows, random_manifest, scores_for
from fl.synthetic import client_examples, make_tiny_adapter
from evaltestkit import evaluate_frozen as evaluate
from v4eval.predict import _research_predict_adapter, predict_adapter
from v4eval.ranking import rank_from_scores

KS = (1, 5, 10, 20)


def test_rank_parity_with_snapshot():
    g = torch.Generator().manual_seed(17)
    s = torch.round(torch.randn(120, 333, generator=g) * 3) / 3
    t = torch.randint(0, 333, (120,), generator=g)
    for c in (1, 13, 333, 32768):
        mine = rank_from_scores(s, t, class_chunk=c).ranks
        assert torch.equal(mine, eval_full.rank_targets(s, t, class_chunk=c))
        assert torch.equal(mine, legacy_evaluator._ranks_chunked(s, t, c))
    si = (s * 3).to(torch.long)                                            # int64 baseline scores
    assert torch.equal(rank_from_scores(si, t).ranks, eval_full.rank_targets(si, t))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_metric_parity_with_snapshot_aggregate(seed):
    m = random_manifest(n_users=70, seed=seed, oov=0.15, cens=0.1)
    art = artifact_from_scores(m, scores_for(m, seed + 50))
    mine = evaluate(m, art)
    ds = {"target_class": m.target_class, "eval_mask": m.eval_mask, "eval_user_key": m.eval_user_key,
          "client_idx": np.full(m.n, 5, dtype=np.int64), "prior_view_count": np.zeros(m.n, dtype=np.int64),
          "prefix_position": np.ones(m.n, dtype=np.int64)}
    for k in KS:
        leg = eval_full.aggregate(art.ranks, ds, k=k)
        for mine_den, leg_den in (("E2E", "end_to_end"), ("RANKABLE", "closed")):
            assert mine[mine_den]["n_decisions"] == leg[leg_den]["n"]
            assert mine[mine_den]["n_users"] == leg[leg_den]["macro"]["user_count"]
            for fam, leg_fam in (("mrr", "mrr"), ("hr", "recall"), ("ndcg", "ndcg")):
                for agg in ("micro", "macro"):
                    a = mine[mine_den][agg][f"{fam}@{k}"]
                    b = leg[leg_den][agg][f"{leg_fam}@{k}"]
                    assert a == pytest.approx(b, abs=1e-6), (k, mine_den, fam, agg)


def _adapter_case(tied: bool, seed: int, n: int = 64, K: int = 24, L: int = 6):
    a = make_tiny_adapter(K=K, d=8, tied=tied, dropout=0.1, seed=seed)
    ex = client_examples(n, K, L, np.random.default_rng(seed), invalid_frac=0.2)
    tgt = ex["target_class"].numpy()
    users = np.arange(n) // 4
    rows = [(int(u), "R" if t >= 0 else "O", int(t)) for u, t in zip(users, tgt)]     # invalid target = OOV row
    m = manifest_from_rows(rows, K_=K)
    ex["decision_id"] = torch.as_tensor(m.decision_id)
    ex["eval_user_key"] = torch.as_tensor(m.eval_user_key)
    return a, ex, m


def _batches(ex, size=None):
    size = frozen().eval_batch if size is None else size
    n = ex["decision_id"].shape[0]
    return [{k: v[s:s + size] for k, v in ex.items()} for s in range(0, n, size)]


@pytest.mark.parametrize("tied", [True, False])
@pytest.mark.parametrize("personal", [False, True])
def test_adapter_path_equals_snapshot_ranks(tied, personal):
    a, ex, m = _adapter_case(tied, seed=3 if tied else 4)
    P = torch.randn(int(m.eval_user_key.max()) + 1, 8, generator=torch.Generator().manual_seed(9)) * 0.3
    pfn = (lambda b: P[b["eval_user_key"]]) if personal else None
    a.module.train()
    random.seed(1)
    torch.manual_seed(1)
    before = (random.random(), torch.rand(1))
    random.seed(1)
    torch.manual_seed(1)
    art = predict_adapter(a, _batches(ex), m, config=frozen(), model_sha256="tiny", personal_fn=pfn)
    assert a.module.training                                               # train mode restored
    assert (random.random(), torch.rand(1)) == before                     # RNG streams untouched
    a.module.eval()
    with torch.no_grad():
        scores = a.scores(ex, P[ex["eval_user_key"]] if personal else None)
    rk = m.rankable
    ref = eval_full.rank_targets(scores[torch.as_tensor(rk)], torch.as_tensor(m.target_class[rk]))
    assert np.array_equal(art.ranks[rk], ref.numpy())
    assert np.all(art.ranks[~rk] == -1)
    for blk in (1, 5, 7):                                                  # smaller blocks: float tolerance only
        art_b = _research_predict_adapter(a, _batches(ex, 16), m, model_sha256="tiny", personal_fn=pfn,
                                          score_block=blk, class_chunk=frozen().class_chunk)
        s_rk = scores[torch.as_tensor(rk)]
        s_t = s_rk.gather(1, torch.as_tensor(m.target_class[rk]).view(-1, 1))
        close = ((s_rk - s_t).abs() <= 1e-5).sum(1) - 1
        assert np.all(np.abs(art_b.ranks[rk] - ref.numpy()) <= close.numpy())
    res = evaluate(m, art)
    assert res["E2E"]["n_decisions"] == int(m.eval_mask.sum())
