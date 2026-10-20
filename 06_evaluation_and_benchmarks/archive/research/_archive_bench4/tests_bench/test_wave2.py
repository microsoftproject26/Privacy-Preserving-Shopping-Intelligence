"""Wave-2 plumbing: generic dataset names / roots, the rating column, cold holdout items counted as MISSES."""
from __future__ import annotations

import numpy as np
import pytest

from conftest import make_release
from v4bench import common, data as D, final
from v4bench.common import BenchRefused
from v4bench.evaluate import evaluate_view, summarize


def test_dataset_names_and_default_roots():
    for n in common.WAVE2_DATASETS:
        assert common.check_dataset(n) == n and common.default_data_root(n) == common.WAVE2_ROOT_DEFAULT
    assert common.default_data_root("ml_1m") == common.DATA_ROOT_DEFAULT
    with pytest.raises(BenchRefused):
        common.check_dataset("mbht_tmall")                         # dropped by the owner


def test_cold_holdout_items_are_counted_as_misses(tmp_path):
    name = "amazon2023_video_games"
    rel = make_release(tmp_path / "p", name=name, n_users=40, n_items=30, drop_holdout=0, rating=True, cold_holdout=3)
    d = D.load(name, rel["root"])                                  # the rating column is ignored
    assert D.statistics_check(name, rel["root"], d)["ok"]
    final._TOKENS["w" * 32] = name
    tv = D._load_holdout_view(d, rel["root"], _gate="w" * 32)
    assert len(tv.users) == 40 and tv.n_cold_items == 3 and int((tv.targets < 0).sum()) == 3   # kept, not dropped
    ranks = evaluate_view(d_adapter(d), tv)
    assert (ranks[tv.targets < 0] == 0).all()                      # misses
    summ = summarize(ranks, tv, d.band)
    assert summ["ALL"]["n_users"] == 40 and summ["ALL"]["n_unreachable"] >= 3


def d_adapter(d):
    from v4bench.model import build_model
    return build_model("SASREC_D64_B2", d.K, 2026)["adapter"]


def test_unknown_holdout_user_is_counted_not_scored(tmp_path):
    name = "s3_beauty"
    rel = make_release(tmp_path / "p", name=name, n_users=20, n_items=30, drop_holdout=0)
    hp = rel["root"] / name / "leave_one_out" / "holdout.csv"
    with open(hp, "a") as f:
        f.write("99999,5,1970-01-01 00:00:00.000000099,1.0\n")
    d = D.load(name, rel["root"])
    final._TOKENS["v" * 32] = name
    tv = D._load_holdout_view(d, rel["root"], _gate="v" * 32)
    assert tv.n_unknown_users == 1 and len(tv.users) == 20
