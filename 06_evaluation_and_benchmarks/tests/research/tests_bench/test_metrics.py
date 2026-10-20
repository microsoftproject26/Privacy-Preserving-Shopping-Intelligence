"""Metric tests: exact parity with the official capacity_probes code (pinned fixture copy of metrics.py at @705710f7)
and brute-force checks of the rank rule."""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from v4bench import metrics as M
import ppsi_paths

OFFICIAL = Path(__file__).resolve().parent / "official" / "capacity_probes_metrics_705710f7.py"


def _official():
    """Import the official metrics module; RecTools is stubbed when absent (gains_for / update_totals need numpy only)."""
    try:
        import rectools  # noqa: F401
    except Exception:
        rt = types.ModuleType("rectools")
        rt.Columns = types.SimpleNamespace(User="user_id", Item="item_id", Score="score", Rank="rank",
                                           Datetime="datetime")
        rm = types.ModuleType("rectools.metrics")
        rm.NDCG = rm.Recall = rm.calc_metrics = None
        sys.modules.setdefault("rectools", rt)
        sys.modules.setdefault("rectools.metrics", rm)
    spec = importlib.util.spec_from_file_location("official_metrics", OFFICIAL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["official_metrics"] = mod
    spec.loader.exec_module(mod)
    return mod


def _ranks_bruteforce(scores, target, seen):
    B, K = scores.shape
    out = []
    for b in range(B):
        t = int(target[b])
        if t in seen[b]:
            out.append(0)
            continue
        cand = [j for j in range(K) if j not in seen[b]]
        order = sorted(cand, key=lambda j: (-float(scores[b, j]), j))
        out.append(order.index(t) + 1)
    return np.asarray(out)


def _fixture(seed=0, B=60, K=40):
    g = torch.Generator().manual_seed(seed)
    scores = torch.randn(B, K, generator=g)
    scores[:5, :6] = 1.0                                 # ties: smaller class id first
    rng = np.random.default_rng(seed)
    target = rng.integers(0, K, size=B)
    seen = [set(rng.choice(K, size=int(rng.integers(0, 12)), replace=False).tolist()) for _ in range(B)]
    rows = np.repeat(np.arange(B), [len(s) for s in seen])
    items = np.concatenate([sorted(s) for s in seen]) if rows.size else np.zeros(0, dtype=np.int64)
    return scores, target, seen, rows, items


def test_ranks_match_bruteforce_with_ties_and_seen_filter():
    scores, target, seen, rows, items = _fixture()
    got = M.ranks_from_scores(scores.clone(), target, rows, items)
    assert np.array_equal(got, _ranks_bruteforce(scores.numpy(), target, seen))


def test_unreachable_target_is_rank_zero():
    s = torch.randn(3, 10)
    got = M.ranks_from_scores(s.clone(), [2, 4, 5], [0, 1], [2, 4])
    assert got[0] == 0 and got[1] == 0 and got[2] >= 1


def test_metric_parity_with_official_update_totals():
    off = _official()
    scores, target, seen, rows, items = _fixture(seed=3, B=80, K=60)
    ranks = M.ranks_from_scores(scores.clone(), target, rows, items)
    mine = M.metrics_from_ranks(ranks)
    for k in (10, 20):
        tot = np.zeros(4)
        for b in range(scores.shape[0]):
            cand = [j for j in range(scores.shape[1]) if j not in seen[b]]
            order = sorted(cand, key=lambda j: (-float(scores[b, j]), j))[:k]    # official topk: score desc, id asc
            tgt = np.asarray([target[b]], dtype=np.int32)
            off.update_totals(tot, off.gains_for(np.asarray(order, dtype=np.int64), tgt), tgt, k)
        n = scores.shape[0]
        assert mine[f"ndcg@{k}"] == pytest.approx(tot[0] / n, abs=1e-12)
        assert mine[f"hr@{k}"] == pytest.approx(tot[1] / n, abs=1e-12)
        assert tot[1] == tot[2]                           # one target: recall == hit rate


def test_mrr_at_20_definition():
    r = np.asarray([1, 2, 20, 21, 0, 5])
    m = M.metrics_from_ranks(r)
    assert m["mrr@20"] == pytest.approx((1 + 0.5 + 0.05 + 0 + 0 + 0.2) / 6)
    assert m["n_unreachable"] == 1 and m["hr@10"] == pytest.approx(3 / 6)


def test_grouped_metrics_partition():
    r = np.asarray([1, 3, 0, 12, 2, 40])
    users = np.arange(6)
    band = np.asarray([True, False, True, False, False, True])
    g = M.grouped_metrics(r, users, band)
    assert g["BAND"]["n_users"] == 3 and g["OTHERS"]["n_users"] == 3 and g["ALL"]["n_users"] == 6
    assert g["ALL"]["ndcg@10"] == pytest.approx((3 * g["BAND"]["ndcg@10"] + 3 * g["OTHERS"]["ndcg@10"]) / 6)


def test_nc_no_seen_filter_changes_the_result():
    """Negative control: dropping the seen-item filter must change the metric (a filter that does nothing is caught)."""
    scores, target, seen, rows, items = _fixture(seed=5)
    scores[np.arange(len(seen)), [max(s) if s else 0 for s in seen]] += 5.0     # a seen item scored highest
    with_f = M.metrics_from_ranks(M.ranks_from_scores(scores.clone(), target, rows, items))
    no_f = M.metrics_from_ranks(M.ranks_from_scores(scores.clone(), target, np.zeros(0, dtype=np.int64),
                                                    np.zeros(0, dtype=np.int64)))
    assert with_f["ndcg@10"] != no_f["ndcg@10"]


def test_nc_wrong_tie_break_is_detected():
    s = torch.zeros(1, 5)
    assert M.ranks_from_scores(s.clone(), [3], [], [])[0] == 4      # ties: smaller ids first -> id 3 is 4th
    assert M.ranks_from_scores(s.clone(), [0], [], [])[0] == 1


def test_nonfinite_scores_refused():
    s = torch.zeros(2, 4)
    s[0, 1] = float("nan")
    with pytest.raises(Exception):
        M.ranks_from_scores(s, [0, 1], [], [])


def test_fixture_is_the_pinned_official_file():
    txt = OFFICIAL.read_text(encoding="utf-8")
    assert "def update_totals" in txt and "def gains_for" in txt and "def official_metrics" in txt
    server = (ppsi_paths.PROJECT_ROOT / "bench_raw" / "spotify-sequential-capacity-probes" / "src" / "capacity_probes"
              / "metrics.py")
    if server.exists():
        assert server.read_text(encoding="utf-8").strip().replace("\r", "") == txt.strip().replace("\r", "")


def test_rectools_parity_when_available():
    """On the server (PYTHONPATH incl. the RecTools site dir): the official NDCG@10 / Recall@10 equal ours."""
    import pandas as pd
    off = _official()
    if getattr(sys.modules.get("rectools.metrics"), "NDCG", None) is None:
        pytest.skip("RecTools is not installed here (server test)")
    scores, target, seen, rows, items = _fixture(seed=11, B=50, K=50)
    ranks = M.ranks_from_scores(scores.clone(), target, rows, items)
    mine = M.metrics_from_ranks(ranks)
    recs = []
    for b in range(50):
        cand = [j for j in range(50) if j not in seen[b]]
        order = sorted(cand, key=lambda j: (-float(scores[b, j]), j))[:10]
        recs += [(b, int(j) + 100, float(scores[b, j]), i + 1) for i, j in enumerate(order)]
    reco = pd.DataFrame(recs, columns=["user_id", "item_id", "score", "rank"])
    hold = pd.DataFrame({"user_id": np.arange(50), "item_id": target + 100, "datetime": pd.Timestamp("2000-01-01"),
                         "weight": 1.0})
    prev = pd.DataFrame({"user_id": [0], "item_id": [100], "datetime": pd.Timestamp("1999-01-01"), "weight": 1.0})
    res = off.official_metrics(reco, prev, hold, k=10)
    assert res["ndcg@10"] == pytest.approx(mine["ndcg@10"], abs=1e-9)
    assert res["recall@10"] == pytest.approx(mine["hr@10"], abs=1e-9)
