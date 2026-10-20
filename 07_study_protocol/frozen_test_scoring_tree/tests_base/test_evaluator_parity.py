"""BASE-01: evaluator parity — a hand-computed score matrix and target set, with expected ranks computed by hand
from the QUOTED tie rule (v4eval/ranking.py: rank = 1 + #{s_j > s_t} + #{j < t : s_j == s_t}), compared bit-for-bit
against what `v4eval.predict.predict_scores` + `v4base.scoring` actually produce through the SAME frozen T3 path a
trained model's rows go through.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch

from basekit import MINIMAL_KEY_BASE
from v4base import scoring as base_scoring
from v4eval import frozen_config as fc
from v4eval.manifest import build_eval_manifest
from v4eval.predict import predict_scores

K = 5
DECISION_ID = [100, 200, 300]
TARGET_CLASS = [2, 0, 4]
# hand-chosen scores, including one exact tie (row 0: classes 1 and 2 both score 2.0) to exercise the ascending-
# class-id tie rule.
SCORES = torch.tensor([[1.0, 2.0, 2.0, 0.5, 3.0],
                       [5.0, 1.0, 1.0, 1.0, 1.0],
                       [0.0, 0.0, 0.0, 0.0, 10.0]], dtype=torch.float32)
# hand-computed: row0 target=2 (score 2.0): gt={class4=3.0} -> 1; eq_before={class1=2.0, idx1<2} -> 1; rank=1+1+1=3
# row1 target=0 (score 5.0): gt=0, eq_before=0 (no j<0); rank=1
# row2 target=4 (score 10.0): gt=0, eq_before=0; rank=1
EXPECTED_RANKS = [3, 1, 1]


def _manifest():
    header = {"manifest_id": "PARITY_FIXTURE", "catalogue_K": K, "candidate_order_hash": "PARITY-CANDIDATES"}
    cols = {"decision_id": DECISION_ID, "eval_user_key": [0, 1, 2], "eval_mask": [True, True, True],
           "target_class": TARGET_CLASS, "target_oov": [False, False, False],
           "censor_class": ["NONE", "NONE", "NONE"]}
    import numpy as np
    cols = {k: np.asarray(v) for k, v in cols.items()}
    return build_eval_manifest(header, cols)


class _FixedScorer:
    def score_batch(self, batch: dict) -> torch.Tensor:
        return SCORES


def test_predict_scores_ranks_match_the_hand_computed_ranks():
    manifest = _manifest()
    cfg = fc.load()
    fc.apply_runtime_flags(cfg)
    batches = [{"decision_id": torch.tensor(DECISION_ID, dtype=torch.int64)}]
    art = predict_scores(base_scoring.score_batches(_FixedScorer(), batches), manifest, model_sha256="PARITY",
                         config=cfg)
    assert art.ranks.tolist() == EXPECTED_RANKS


def test_full_baseline_evaluation_agrees_and_produces_baseline_rows():
    manifest = _manifest()
    cfg = fc.load()
    result = base_scoring.run_baseline_evaluation(_FixedScorer(), manifest,
                                                  [{"decision_id": torch.tensor(DECISION_ID, dtype=torch.int64)}],
                                                  model_sha256="PARITY", config=cfg)
    # macro MRR@20, E2E, over the 3 hand-computed ranks: mean(1/rank) = mean(1/3, 1/1, 1/1)
    expected_mrr20 = (1 / 3 + 1 / 1 + 1 / 1) / 3
    assert result["E2E"]["macro"]["mrr@20"] == pytest.approx(expected_mrr20, rel=1e-9)
    rows = base_scoring.build_rows(result, MINIMAL_KEY_BASE, method="POP")
    assert rows, "no rows produced"
    assert all(r["run_label"] == "BASELINE" for r in rows)
    assert all(r["method"] == "POP" for r in rows)
