"""Real-data protocol tests (server only; skipped where the release or the probes repo is absent).

Run with PYTHONPATH=<probes repo>/src:/home/jovyan/venvs/bench-probes-site (see the BM-02 handoff).
  * the inner-validation items equal eSASRec's rule on the real train.csv (ML-1M: 2,483 users whose validation row is not
    the last ascending row);
  * the TEST population / catalogue equal the published denominators (BM-00 §2.3-2.4);
  * PARITY ON REAL DATA: the probes' own MC model (capacity_probes.models.mc.build_mc, evidence rows of the last train item)
    is scored through THIS evaluator (TEST view, full-history seen filter, ascending-id ties) and must equal the NDCG@10 /
    Recall@10 that results_cpu/<ds>/mc.json holds from the official RecTools path (1e-9).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from v4bench import common, data as D, final, metrics as M
from test_data import esasrec_val_mask

ROOT = common.DATA_ROOT_DEFAULT
REPO = ROOT.parents[1]
needs_data = pytest.mark.skipif(not (ROOT / "s3_beauty" / "statistics.csv").exists(),
                                reason="the real release is on the server only")
RESULTS = {"s3_beauty": "beauty", "s3_sports": "sports", "s3_toys": "toys", "ml_1m": "ml1m"}
HOLDOUT_USERS = {"s3_beauty": 22311, "s3_sports": 35539, "s3_toys": 19365, "ml_1m": 6038}
TRAIN_CATALOGUE = {"s3_beauty": 12092, "s3_sports": 18347, "s3_toys": 11917, "ml_1m": 3704}
VAL_NOT_LAST = {"s3_beauty": 0, "s3_sports": 0, "s3_toys": 0, "ml_1m": 2483}      # BM-00 §2.1 (MEASURED there)


@pytest.fixture(scope="module", params=common.DATASETS)
def real(request):
    name = request.param
    if not (ROOT / name / "leave_one_out" / "train.csv").exists():
        pytest.skip("real release absent")
    d = D.load(name, ROOT)
    final._TOKENS["r" * 32] = name                      # TEST-only shortcut for the gate (read-only evaluation)
    tv = D._load_holdout_view(d, ROOT, _gate="r" * 32)
    return name, d, tv


@needs_data
def test_inner_validation_items_equal_esasrecs(real):
    name, d, _ = real
    fr = pd.read_csv(ROOT / name / "leave_one_out" / "train.csv", parse_dates=["datetime"])
    mask = esasrec_val_mask(fr)
    want = fr[mask].set_index("user_id")["item_id"].sort_index()
    got = pd.Series(d.catalog[d.val_item], index=d.user_ids).sort_index()
    assert len(want) == d.U and (want.index.to_numpy() == got.index.to_numpy()).all()
    assert (want.to_numpy() == got.to_numpy()).all()
    assert d.stats["n_val_row_not_last"] == VAL_NOT_LAST[name]


@needs_data
def test_test_population_and_catalogue(real):
    name, d, tv = real
    assert len(tv.users) == HOLDOUT_USERS[name]
    assert d.K == TRAIN_CATALOGUE[name]
    assert (tv.ends == d.lengths[tv.users]).all() and tv.source == "full"
    rows, items = tv.seen_pairs(np.arange(min(500, len(tv.users))))
    for j in range(min(500, len(tv.users))):             # whole train history filtered, validation row included
        u = tv.users[j]
        full = d.full_items[d.full_offsets[u]:d.full_offsets[u + 1]]
        assert set(items[rows == j].tolist()) == set(full.tolist())
        assert tv.targets[j] not in set(full.tolist())


@needs_data
def test_parity_with_probes_mc_on_real_data(real, tmp_path):
    name, d, tv = real
    pytest.importorskip("capacity_probes")
    pytest.importorskip("rectools")
    from capacity_probes.io import read_interactions
    from capacity_probes.models.mc import build_mc
    from capacity_probes.sequences import SequenceStore
    out = json.loads((REPO / "results_cpu" / RESULTS[name] / "mc.json").read_text())
    store = SequenceStore.from_frame(read_interactions(ROOT / name / "leave_one_out" / "train.csv"))
    assert np.array_equal(store.catalog, d.catalog)
    model = build_mc(store, tmp_path / "cache")
    ev = model.evidence.tocsr()
    last = d.full_items[d.full_offsets[tv.users + 1] - 1]            # the store's last (datetime, file-order) item
    ranks = np.zeros(len(tv.users), dtype=np.int64)
    for s in range(0, len(tv.users), 512):
        sel = np.arange(s, min(len(tv.users), s + 512))
        scores = torch.from_numpy(ev[last[sel]].toarray().astype(np.float32))
        rows, items = tv.seen_pairs(sel)
        ranks[sel] = M.ranks_from_scores(scores, tv.targets[sel], rows, items)
    mine = M.metrics_from_ranks(ranks)
    assert mine["n_users"] == out["custom_metrics"]["users"] == HOLDOUT_USERS[name]
    assert mine["ndcg@10"] == pytest.approx(out["official_metrics"]["ndcg@10"], abs=1e-9)
    assert mine["hr@10"] == pytest.approx(out["official_metrics"]["recall@10"], abs=1e-9)


VAL_POP = {"s3_beauty": 22308, "s3_sports": 35475, "s3_toys": 19312, "ml_1m": 6038}      # BM-00 §2.2 with the RecTools tail (51 / 201)


@needs_data
def test_inner_validation_population_equals_rectools(real):
    name, d, _ = real
    assert len(d.validation_users()) == VAL_POP[name]
