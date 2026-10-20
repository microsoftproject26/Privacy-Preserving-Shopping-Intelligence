"""BASE-02: KNN_EMB — checkpoint state-dict extraction (GRU tied table, SASRec untied table), a hand-computed
cosine fixture, the empty/degenerate-history popularity fallback, the last-`history_window` truncation, and
causality (padding never leaks into the mean embedding even when poisoned to look like a valid recent item).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math

import pytest
import torch

from basekit import nc
from v4base import knn_emb

K = 4
# class-ordered table: e0, e1 orthonormal-ish, e2 = -e0, e3 = zero-norm (degenerate row)
TABLE = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0]])
FALLBACK = torch.tensor([5.0, 2.0, 1.0, 0.0])


def _batch(item_tokens, lengths):
    return {"item_tokens": torch.as_tensor(item_tokens, dtype=torch.int64),
           "lengths": torch.as_tensor(lengths, dtype=torch.int64)}


# --------------------------------------------------------------------------------------------- checkpoint loading

def test_gru_extraction_contiguous():
    W = torch.zeros(7, 2)
    W[3:7] = TABLE
    sd = {knn_emb.GRU_TABLE_KEY: W, knn_emb.GRU_CLASS_MAP_KEY: torch.tensor([3, 4, 5, 6])}
    table = knn_emb.item_table_from_state_dict(sd, "GRU", K)
    assert torch.equal(table, TABLE)


def test_gru_extraction_non_contiguous_class_map():
    W = torch.zeros(7, 2)
    W[3:7] = TABLE                                   # W[3]=e0 W[4]=e1 W[5]=e2 W[6]=e3
    sd = {knn_emb.GRU_TABLE_KEY: W, knn_emb.GRU_CLASS_MAP_KEY: torch.tensor([6, 5, 4, 3])}
    table = knn_emb.item_table_from_state_dict(sd, "gru", K)   # lower-case family also accepted
    assert torch.equal(table, TABLE[[3, 2, 1, 0]])    # class0->W[6]=e3, class1->W[5]=e2, class2->W[4]=e1, class3->W[3]=e0


def test_sasrec_extraction_is_direct_no_offset():
    sd = {knn_emb.SASREC_TABLE_KEY: TABLE.clone()}
    table = knn_emb.item_table_from_state_dict(sd, "SASREC", K)
    assert torch.equal(table, TABLE)


def test_gru_missing_class_map_raises_key_error():
    sd = {knn_emb.GRU_TABLE_KEY: torch.zeros(7, 2)}
    with pytest.raises(KeyError):
        knn_emb.item_table_from_state_dict(sd, "GRU", K)


def test_sasrec_wrong_shape_raises_value_error():
    sd = {knn_emb.SASREC_TABLE_KEY: torch.zeros(K + 1, 2)}
    with pytest.raises(ValueError):
        knn_emb.item_table_from_state_dict(sd, "SASREC", K)


def test_unknown_family_raises_value_error():
    with pytest.raises(ValueError):
        knn_emb.item_table_from_state_dict({}, "TRANSFORMER", K)


@pytest.mark.parametrize("wrapper", [lambda sd: sd, lambda sd: {"model": sd}, lambda sd: {"theta": sd},
                                     lambda sd: {"state_dict": sd}, lambda sd: {"model_state_dict": sd}])
def test_load_checkpoint_state_dict_unwraps_every_documented_container(tmp_path, wrapper):
    sd = {knn_emb.SASREC_TABLE_KEY: TABLE.clone()}
    p = tmp_path / "ckpt.pt"
    torch.save(wrapper(sd), p)
    table = knn_emb.load_item_table(p, "SASREC", K)
    assert torch.equal(table, TABLE)


# --------------------------------------------------------------------------------------------- cosine + fallback

def test_hand_computed_cosine_and_zero_norm_row():
    scorer = knn_emb.KnnEmbScorer(TABLE, FALLBACK, history_window=10)
    batch = _batch([[3, 4]], [2])                     # history classes [0, 1]
    out = scorer.score_batch(batch)[0]
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    assert out[0].item() == pytest.approx(inv_sqrt2, abs=1e-6)
    assert out[1].item() == pytest.approx(inv_sqrt2, abs=1e-6)
    assert out[2].item() == pytest.approx(-inv_sqrt2, abs=1e-6)
    assert out[3].item() == 0.0                       # zero-norm catalogue row -> defined 0.0, never NaN


def test_empty_history_falls_back_to_the_popularity_row():
    scorer = knn_emb.KnnEmbScorer(TABLE, FALLBACK, history_window=10)
    batch = _batch([[0, 0]], [0])                     # length 0: no valid history at all
    out = scorer.score_batch(batch)[0]
    assert torch.equal(out, FALLBACK)


def test_degenerate_zero_mean_history_falls_back_too():
    scorer = knn_emb.KnnEmbScorer(TABLE, FALLBACK, history_window=10)
    batch = _batch([[3, 5]], [2])                      # classes [0, 2] = e0, e2 = -e0: mean is the zero vector
    out = scorer.score_batch(batch)[0]
    assert torch.equal(out, FALLBACK)


def test_history_window_keeps_only_the_most_recent_items():
    scorer = knn_emb.KnnEmbScorer(TABLE, FALLBACK, history_window=1)
    batch = _batch([[3, 4, 5]], [3])                   # classes [0, 1, 2]; window=1 keeps only the LAST (class2=e2)
    out = scorer.score_batch(batch)[0]
    assert out[0].item() == pytest.approx(-1.0, abs=1e-6)   # cosine(e2, e0) = -1
    assert out[1].item() == pytest.approx(0.0, abs=1e-6)
    assert out[2].item() == pytest.approx(1.0, abs=1e-6)


def test_history_window_counts_valid_items_not_columns_skipping_an_oov_gap():
    # tokens: class0 (valid), OOV (token 2, invalid -- no catalogue embedding), class1 (valid). With
    # history_window=2, the last TWO VALID items are class0 and class1 (the OOV gap is skipped, reaching one
    # column further back) -- NOT "the last 2 columns" (which would wrongly drop class0 and keep only class1).
    scorer = knn_emb.KnnEmbScorer(TABLE, FALLBACK, history_window=2)
    batch = _batch([[3, 2, 4]], [3])
    out = scorer.score_batch(batch)[0]
    mean_emb = TABLE[[0, 1]].mean(dim=0)                     # mean(e0, e1) = [0.5, 0.5]
    mnorm = float(mean_emb.norm())
    expected0 = float((TABLE[0] @ mean_emb) / (TABLE[0].norm() * mnorm))
    assert out[0].item() == pytest.approx(expected0, abs=1e-6)
    assert out[3].item() == 0.0                              # zero-norm row unaffected


def test_padding_never_leaks_into_the_mean_even_when_poisoned():
    scorer = knn_emb.KnnEmbScorer(TABLE, FALLBACK, history_window=10)
    # length=1 (only class0 real); the padded column is poisoned to look like a real, very different item (class2)
    batch = _batch([[3, 5]], [1])
    out = scorer.score_batch(batch)[0]
    assert out[0].item() == pytest.approx(1.0, abs=1e-6)     # cosine(e0, e0) = 1, NOT influenced by the poisoned class2
    assert out[2].item() == pytest.approx(-1.0, abs=1e-6)


@nc("without the mean-norm guard, a degenerate (norm-zero) mean history embedding would divide by zero (NaN), not "
   "fall back to the popularity row")
def test_nc_zero_norm_mean_would_produce_nan_without_the_guard():
    mean_emb = TABLE[[0, 2]].mean(dim=0)               # [0, 0], exactly the degenerate case above
    unguarded = (TABLE @ mean_emb) / (TABLE.norm(dim=1) * float(mean_emb.norm()))  # 0/0 -> nan by construction
    produced_nan = bool(torch.isnan(unguarded).any())
    assert not produced_nan
