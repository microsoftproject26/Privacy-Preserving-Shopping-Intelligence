"""Decisions between runs and the LOCK: Track-B tie rules, FA_WARM* (N2), the third-seed trigger, strict LOCK."""
from __future__ import annotations

import json

import pytest

from v4bench import final, selection
from v4bench.common import BenchRefused

DS = "s3_beauty"


def fake_run(runs, name, arm, seed, value, *, budget=6, max_len=50, dropout=0.1, kind="fl", pop="ALL"):
    d = runs / DS / name
    d.mkdir(parents=True)
    spec = {"seed": seed, "cfg": {"seed": seed}, "budget_efe": budget, "max_len": max_len, "dropout": [dropout],
            "select_population": pop}
    summ = {g: {"ndcg@10": value} for g in ("ALL", "BAND", "OTHERS")}
    (d / "RESULT.json").write_text(json.dumps({"status": "DONE_MAIN", "arm": arm, "kind": kind, "spec": spec,
                                               "validation": {"PRACTICAL_BEST": summ}, "checkpoints": {}}))


def test_trackb_tie_rule_order():
    cells = {"a": (6, 50, 0.1), "b": (24, 50, 0.1), "c": (24, 200, 0.1), "d": (24, 50, 0.5), "e": (96, 50, 0.2)}
    # all within 0.0005 of the best: the Track-A cell wins
    assert selection.trackb_pick(cells, {"a": 0.100, "b": 0.1004, "c": 0.1003, "d": 0.1001, "e": 0.1002}) == "a"
    # Track-A cell clearly worse: smaller budget, then smaller max_len, then larger dropout
    assert selection.trackb_pick(cells, {"a": 0.09, "b": 0.1004, "c": 0.1003, "d": 0.1001, "e": 0.1002}) == "d"
    assert selection.trackb_pick(cells, {"a": 0.09, "b": 0.1004, "c": 0.1003, "d": 0.0990, "e": 0.1002}) == "b"
    assert selection.trackb_pick(cells, {"a": 0.09, "b": 0.0990, "c": 0.1000, "d": 0.0990, "e": 0.1}) == "c"
    # a clear winner beyond the tie band is taken as is
    assert selection.trackb_pick(cells, {"a": 0.1, "b": 0.1, "c": 0.1, "d": 0.1, "e": 0.11}) == "e"


def test_trackb_select_writes_once_and_needs_track_a_cell(tmp_path):
    fake_run(tmp_path, "FA_s2026", "FA", 2026, 0.10)
    fake_run(tmp_path, "FA__B6_L50_D0.5_s2026", "FA", 2026, 0.12, dropout=0.5)
    obj = selection.trackb_select(tmp_path, DS, "FA", ["FA_s2026", "FA__B6_L50_D0.5_s2026"])
    assert obj["winner_run"] == "FA__B6_L50_D0.5_s2026"
    with pytest.raises(BenchRefused):
        selection.trackb_select(tmp_path, DS, "FA", ["FA_s2026", "FA__B6_L50_D0.5_s2026"])      # write-once
    fake_run(tmp_path, "FB__B24_L50_D0.5_s2026", "FB", 2026, 0.12, budget=24, dropout=0.5)
    with pytest.raises(BenchRefused, match="Track-A cell"):
        selection.trackb_select(tmp_path, DS, "FB", ["FB__B24_L50_D0.5_s2026"])


def test_decide_warm_rule(tmp_path):
    for sd, v7, v8 in ((2026, 0.1010, 0.1000), (2027, 0.1006, 0.1004)):
        fake_run(tmp_path, f"FA_WARM_s{sd}", "FA_WARM", sd, v7)
        fake_run(tmp_path, f"FA_WARM_FROZEN_s{sd}", "FA_WARM_FROZEN", sd, v8)
    obj = selection.decide_warm(tmp_path, DS, ["FA_WARM_s2026", "FA_WARM_s2027"],
                                ["FA_WARM_FROZEN_s2026", "FA_WARM_FROZEN_s2027"])
    assert obj["FA_WARM_STAR"] == "FA_WARM"                  # means .1008 vs .1002: gap .0006 > .0005 -> S7
    t2 = tmp_path / "t2"
    for sd, v7, v8 in ((2026, 0.1004, 0.1000), (2027, 0.1004, 0.1004)):
        fake_run(t2, f"FA_WARM_s{sd}", "FA_WARM", sd, v7)
        fake_run(t2, f"FA_WARM_FROZEN_s{sd}", "FA_WARM_FROZEN", sd, v8)
    obj = selection.decide_warm(t2, DS, ["FA_WARM_s2026", "FA_WARM_s2027"],
                                ["FA_WARM_FROZEN_s2026", "FA_WARM_FROZEN_s2027"])
    assert obj["FA_WARM_STAR"] == "FA_WARM_FROZEN"           # gap .0002 within the 0.0005 band -> S8


def test_seed3_trigger(tmp_path):
    for sd, vc, vw in ((2026, 0.100, 0.100), (2027, 0.1021, 0.1019)):
        fake_run(tmp_path, f"C_FULL_s{sd}", "C_FULL", sd, vc, kind="central")
        fake_run(tmp_path, f"FA_WARM_FROZEN_s{sd}", "FA_WARM_FROZEN", sd, vw)
    (tmp_path / DS / "FA_WARM_DECISION.json").write_text(json.dumps({"FA_WARM_STAR": "FA_WARM_FROZEN"}))
    obj = selection.seed3_check(tmp_path, DS, {"C_FULL": ["C_FULL_s2026", "C_FULL_s2027"],
                                               "FA_WARM_FROZEN": ["FA_WARM_FROZEN_s2026", "FA_WARM_FROZEN_s2027"]})
    assert obj["third_seed_2028_needed_for"] == ["C_FULL"]                 # gap .0021 > .002; the warm gap .0019 is not
    with pytest.raises(BenchRefused):
        selection.seed3_check(tmp_path, DS, {"C_FULL": ["C_FULL_s2026", "C_FULL_s2027"], "FA_WARM": []})


def test_tree_digest_is_stable():
    assert selection.tree_digest() == selection.tree_digest() and len(selection.tree_digest()) == 64


def test_label_seed_parsing():
    assert final._label_seed("FA__B6_L50_D0.5_s2027") == ("FA__B6_L50_D0.5", 2027)
    with pytest.raises(BenchRefused):
        final._label_seed("nonsense")
